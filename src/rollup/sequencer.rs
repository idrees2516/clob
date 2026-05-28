//! Based Sequencer Implementation
//! 
//! Implements a based rollup sequencer that:
//! - Collects orders from mempool
//! - Creates deterministic batches
//! - Compresses batch data for DA efficiency
//! - Synchronizes with L1 blocks

use crate::orderbook::types::{Order, OrderId};
use crate::rollup::{
    types::*, 
    compression::CompressionEngine, 
    RollupError
};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, info, warn, error};

/// Based sequencer for order batching and L1 synchronization
pub struct BasedSequencer {
    config: SequencerConfig,
    mempool: Arc<RwLock<OrderMempool>>,
    batch_builder: Arc<Mutex<BatchBuilder>>,
    l1_client: Arc<dyn L1Client>,
    compression_engine: CompressionEngine,
    current_batch_id: Arc<Mutex<BatchId>>,
    pending_batches: Arc<RwLock<HashMap<BatchId, OrderBatch>>>,
    l1_sync_state: Arc<RwLock<L1SyncState>>,
}

/// Order mempool for collecting pending orders
#[derive(Debug)]
pub struct OrderMempool {
    orders: HashMap<OrderId, Order>,
    order_queue: VecDeque<OrderId>,
    total_size: usize,
    max_size: usize,
}

/// Batch builder for creating deterministic batches
#[derive(Debug)]
pub struct BatchBuilder {
    current_orders: Vec<Order>,
    batch_start_time: Instant,
    last_l1_block: L1BlockNumber,
    state_root: StateRoot,
}

/// L1 synchronization state
#[derive(Debug, Clone)]
pub struct L1SyncState {
    pub latest_block_number: L1BlockNumber,
    pub latest_block_hash: [u8; 32],
    pub latest_block_timestamp: Timestamp,
    pub last_sync_time: Instant,
}

/// L1 client trait for blockchain interaction
#[async_trait::async_trait]
pub trait L1Client: Send + Sync {
    async fn get_latest_block(&self) -> Result<L1Block, RollupError>;
    async fn get_block_by_number(&self, number: L1BlockNumber) -> Result<L1Block, RollupError>;
    async fn submit_transaction(&self, tx_data: Vec<u8>) -> Result<TxHash, RollupError>;
    async fn get_transaction_receipt(&self, tx_hash: TxHash) -> Result<Option<TransactionReceipt>, RollupError>;
}

/// L1 block information
#[derive(Debug, Clone)]
pub struct L1Block {
    pub number: L1BlockNumber,
    pub hash: [u8; 32],
    pub timestamp: Timestamp,
    pub parent_hash: [u8; 32],
    pub gas_limit: u64,
    pub gas_used: u64,
}

/// Transaction receipt
#[derive(Debug, Clone)]
pub struct TransactionReceipt {
    pub tx_hash: TxHash,
    pub block_number: L1BlockNumber,
    pub gas_used: u64,
    pub status: bool,
    pub logs: Vec<EventLog>,
}

/// Event log from L1 transaction
#[derive(Debug, Clone)]
pub struct EventLog {
    pub address: String,
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

impl BasedSequencer {
    /// Create a new based sequencer
    pub fn new(
        config: SequencerConfig,
        l1_client: Arc<dyn L1Client>,
        compression_engine: CompressionEngine,
    ) -> Self {
        let mempool = Arc::new(RwLock::new(OrderMempool::new(config.mempool_size_limit)));
        let batch_builder = Arc::new(Mutex::new(BatchBuilder::new()));
        let current_batch_id = Arc::new(Mutex::new(1));
        let pending_batches = Arc::new(RwLock::new(HashMap::new()));
        let l1_sync_state = Arc::new(RwLock::new(L1SyncState {
            latest_block_number: 0,
            latest_block_hash: [0u8; 32],
            latest_block_timestamp: 0,
            last_sync_time: Instant::now(),
        }));

        Self {
            config,
            mempool,
            batch_builder,
            l1_client,
            compression_engine,
            current_batch_id,
            pending_batches,
            l1_sync_state,
        }
    }

    /// Start the sequencer with background tasks
    pub async fn start(&self) -> Result<(), RollupError> {
        info!("Starting based sequencer");

        // Start L1 synchronization task
        let l1_sync_task = self.start_l1_sync_task();
        
        // Start batch building task
        let batch_task = self.start_batch_building_task();

        // Wait for both tasks (they run indefinitely)
        tokio::select! {
            result = l1_sync_task => {
                error!("L1 sync task terminated: {:?}", result);
                result
            }
            result = batch_task => {
                error!("Batch building task terminated: {:?}", result);
                result
            }
        }
    }

    /// Add an order to the mempool
    pub async fn add_order(&self, order: Order) -> Result<(), RollupError> {
        let mut mempool = self.mempool.write();
        mempool.add_order(order)?;
        debug!("Added order to mempool, current size: {}", mempool.size());
        Ok(())
    }

    /// Remove an order from the mempool
    pub async fn remove_order(&self, order_id: OrderId) -> Result<Option<Order>, RollupError> {
        let mut mempool = self.mempool.write();
        Ok(mempool.remove_order(order_id))
    }

    /// Get current mempool statistics
    pub fn get_mempool_stats(&self) -> MempoolStats {
        let mempool = self.mempool.read();
        MempoolStats {
            order_count: mempool.order_count(),
            total_size: mempool.size(),
            max_size: mempool.max_size(),
            utilization: mempool.utilization(),
        }
    }

    /// Build a batch from current mempool orders
    pub async fn build_batch(&self) -> Result<OrderBatch, RollupError> {
        let l1_state = self.l1_sync_state.read().clone();
        let mut batch_builder = self.batch_builder.lock();
        let mut mempool = self.mempool.write();

        // Collect orders for the batch
        let orders = mempool.collect_orders_for_batch(self.config.max_batch_size);
        
        if orders.is_empty() {
            return Err(RollupError::SequencerError("No orders available for batching".to_string()));
        }

        // Sort orders deterministically
        let sorted_orders = if self.config.deterministic_sorting {
            self.sort_orders_deterministically(orders)
        } else {
            orders
        };

        // Create batch
        let batch_id = {
            let mut current_id = self.current_batch_id.lock();
            let id = *current_id;
            *current_id += 1;
            id
        };

        let mut batch = OrderBatch::new(
            batch_id,
            sorted_orders,
            l1_state.latest_block_number,
            batch_builder.state_root,
        );

        // Update batch builder state
        batch_builder.last_l1_block = l1_state.latest_block_number;
        batch_builder.batch_start_time = Instant::now();

        info!(
            "Built batch {} with {} orders at L1 block {}",
            batch.batch_id,
            batch.order_count(),
            batch.l1_block_number
        );

        Ok(batch)
    }

    /// Submit batch to L1 with compressed data
    pub async fn submit_batch(&self, mut batch: OrderBatch) -> Result<TxHash, RollupError> {
        // Compress batch data if enabled
        let compressed_data = if self.config.enable_compression {
            self.compression_engine.compress_batch(&batch)?
        } else {
            bincode::serialize(&batch)?
        };

        // Create settlement transaction
        let settlement_tx = SettlementTransaction {
            batch_id: batch.batch_id,
            state_root: batch.state_root_after,
            blob_hash: self.compute_blob_hash(&compressed_data),
            proof_data: vec![], // Will be filled by zkVM proof
            gas_limit: 500000,
            gas_price: 20_000_000_000, // 20 gwei
        };

        // Submit to L1
        let tx_data = bincode::serialize(&settlement_tx)?;
        let tx_hash = self.l1_client.submit_transaction(tx_data).await?;

        // Store pending batch
        {
            let mut pending = self.pending_batches.write();
            pending.insert(batch.batch_id, batch);
        }

        info!("Submitted batch {} to L1 with tx hash: {:?}", settlement_tx.batch_id, tx_hash);
        Ok(tx_hash)
    }

    /// Start L1 synchronization background task
    async fn start_l1_sync_task(&self) -> Result<(), RollupError> {
        let mut interval = interval(Duration::from_millis(self.config.l1_sync_interval_ms));
        let l1_client = Arc::clone(&self.l1_client);
        let sync_state = Arc::clone(&self.l1_sync_state);

        loop {
            interval.tick().await;

            match l1_client.get_latest_block().await {
                Ok(block) => {
                    let mut state = sync_state.write();
                    if block.number > state.latest_block_number {
                        debug!("L1 sync: new block {} (was {})", block.number, state.latest_block_number);
                        state.latest_block_number = block.number;
                        state.latest_block_hash = block.hash;
                        state.latest_block_timestamp = block.timestamp;
                        state.last_sync_time = Instant::now();
                    }
                }
                Err(e) => {
                    warn!("Failed to sync with L1: {}", e);
                }
            }
        }
    }

    /// Start batch building background task
    async fn start_batch_building_task(&self) -> Result<(), RollupError> {
        let mut interval = interval(Duration::from_millis(self.config.batch_timeout_ms));

        loop {
            interval.tick().await;

            // Check if we should build a batch
            let should_build = {
                let mempool = self.mempool.read();
                let batch_builder = self.batch_builder.lock();
                
                // Build batch if:
                // 1. Mempool has enough orders, OR
                // 2. Timeout has elapsed since last batch
                mempool.order_count() >= self.config.max_batch_size ||
                (mempool.order_count() > 0 && 
                 batch_builder.batch_start_time.elapsed().as_millis() >= self.config.batch_timeout_ms as u128)
            };

            if should_build {
                match self.build_batch().await {
                    Ok(batch) => {
                        debug!("Auto-built batch {} with {} orders", batch.batch_id, batch.order_count());
                        // Note: In a real implementation, you'd want to execute the batch
                        // and then submit it. For now, we just log it.
                    }
                    Err(e) => {
                        debug!("Failed to auto-build batch: {}", e);
                    }
                }
            }
        }
    }

    /// Sort orders deterministically for consistent execution
    fn sort_orders_deterministically(&self, mut orders: Vec<Order>) -> Vec<Order> {
        orders.sort_by(|a, b| {
            // Primary sort: timestamp
            a.timestamp.cmp(&b.timestamp)
                // Secondary sort: price (higher prices first for buys, lower for sells)
                .then_with(|| {
                    match (a.side, b.side) {
                        (crate::orderbook::types::Side::Buy, crate::orderbook::types::Side::Buy) => {
                            b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (crate::orderbook::types::Side::Sell, crate::orderbook::types::Side::Sell) => {
                            a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        (crate::orderbook::types::Side::Buy, crate::orderbook::types::Side::Sell) => {
                            std::cmp::Ordering::Less // Buys before sells
                        }
                        (crate::orderbook::types::Side::Sell, crate::orderbook::types::Side::Buy) => {
                            std::cmp::Ordering::Greater // Sells after buys
                        }
                    }
                })
                // Tertiary sort: order ID for complete determinism
                .then_with(|| a.id.cmp(&b.id))
        });
        orders
    }

    /// Compute blob hash for data availability
    fn compute_blob_hash(&self, data: &[u8]) -> BlobHash {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }
}

impl OrderMempool {
    /// Create a new order mempool
    pub fn new(max_size: usize) -> Self {
        Self {
            orders: HashMap::new(),
            order_queue: VecDeque::new(),
            total_size: 0,
            max_size,
        }
    }

    /// Add an order to the mempool
    pub fn add_order(&mut self, order: Order) -> Result<(), RollupError> {
        if self.total_size >= self.max_size {
            return Err(RollupError::SequencerError("Mempool is full".to_string()));
        }

        if self.orders.contains_key(&order.id) {
            return Err(RollupError::SequencerError("Order already exists in mempool".to_string()));
        }

        self.orders.insert(order.id, order);
        self.order_queue.push_back(order.id);
        self.total_size += 1;

        Ok(())
    }

    /// Remove an order from the mempool
    pub fn remove_order(&mut self, order_id: OrderId) -> Option<Order> {
        if let Some(order) = self.orders.remove(&order_id) {
            // Remove from queue (O(n) operation, but mempool should be relatively small)
            if let Some(pos) = self.order_queue.iter().position(|&id| id == order_id) {
                self.order_queue.remove(pos);
            }
            self.total_size -= 1;
            Some(order)
        } else {
            None
        }
    }

    /// Collect orders for batch creation
    pub fn collect_orders_for_batch(&mut self, max_count: usize) -> Vec<Order> {
        let count = std::cmp::min(max_count, self.order_queue.len());
        let mut orders = Vec::with_capacity(count);

        for _ in 0..count {
            if let Some(order_id) = self.order_queue.pop_front() {
                if let Some(order) = self.orders.remove(&order_id) {
                    orders.push(order);
                    self.total_size -= 1;
                }
            }
        }

        orders
    }

    /// Get number of orders in mempool
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Get total size of mempool
    pub fn size(&self) -> usize {
        self.total_size
    }

    /// Get maximum size of mempool
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get mempool utilization as a percentage
    pub fn utilization(&self) -> f64 {
        if self.max_size == 0 {
            0.0
        } else {
            (self.total_size as f64 / self.max_size as f64) * 100.0
        }
    }
}

impl BatchBuilder {
    /// Create a new batch builder
    pub fn new() -> Self {
        Self {
            current_orders: Vec::new(),
            batch_start_time: Instant::now(),
            last_l1_block: 0,
            state_root: [0u8; 32],
        }
    }

    /// Update the state root
    pub fn update_state_root(&mut self, new_root: StateRoot) {
        self.state_root = new_root;
    }
}

/// Mempool statistics
#[derive(Debug, Clone)]
pub struct MempoolStats {
    pub order_count: usize,
    pub total_size: usize,
    pub max_size: usize,
    pub utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, Side, OrderType};

    /// Mock L1 client for testing
    struct MockL1Client {
        current_block: Arc<Mutex<L1Block>>,
    }

    impl MockL1Client {
        fn new() -> Self {
            Self {
                current_block: Arc::new(Mutex::new(L1Block {
                    number: 1,
                    hash: [1u8; 32],
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    parent_hash: [0u8; 32],
                    gas_limit: 30_000_000,
                    gas_used: 15_000_000,
                })),
            }
        }
    }

    #[async_trait::async_trait]
    impl L1Client for MockL1Client {
        async fn get_latest_block(&self) -> Result<L1Block, RollupError> {
            Ok(self.current_block.lock().clone())
        }

        async fn get_block_by_number(&self, _number: L1BlockNumber) -> Result<L1Block, RollupError> {
            Ok(self.current_block.lock().clone())
        }

        async fn submit_transaction(&self, _tx_data: Vec<u8>) -> Result<TxHash, RollupError> {
            Ok([42u8; 32])
        }

        async fn get_transaction_receipt(&self, _tx_hash: TxHash) -> Result<Option<TransactionReceipt>, RollupError> {
            Ok(None)
        }
    }

    fn create_test_order(id: u64, price: f64, quantity: f64, side: Side) -> Order {
        Order {
            id,
            symbol: "BTC-USD".to_string(),
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            user_id: format!("user_{}", id),
        }
    }

    #[tokio::test]
    async fn test_mempool_operations() {
        let mut mempool = OrderMempool::new(10);
        
        let order1 = create_test_order(1, 100.0, 1.0, Side::Buy);
        let order2 = create_test_order(2, 101.0, 2.0, Side::Sell);

        // Test adding orders
        assert!(mempool.add_order(order1.clone()).is_ok());
        assert!(mempool.add_order(order2.clone()).is_ok());
        assert_eq!(mempool.order_count(), 2);

        // Test removing orders
        let removed = mempool.remove_order(1);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, 1);
        assert_eq!(mempool.order_count(), 1);

        // Test collecting orders for batch
        let batch_orders = mempool.collect_orders_for_batch(5);
        assert_eq!(batch_orders.len(), 1);
        assert_eq!(batch_orders[0].id, 2);
        assert_eq!(mempool.order_count(), 0);
    }

    #[tokio::test]
    async fn test_batch_building() {
        let config = SequencerConfig::default();
        let l1_client = Arc::new(MockL1Client::new());
        let compression_engine = CompressionEngine::new(Default::default());
        
        let sequencer = BasedSequencer::new(config, l1_client, compression_engine);

        // Add some test orders
        let order1 = create_test_order(1, 100.0, 1.0, Side::Buy);
        let order2 = create_test_order(2, 101.0, 2.0, Side::Sell);
        
        sequencer.add_order(order1).await.unwrap();
        sequencer.add_order(order2).await.unwrap();

        // Build a batch
        let batch = sequencer.build_batch().await.unwrap();
        assert_eq!(batch.order_count(), 2);
        assert!(batch.batch_id > 0);
    }

    #[test]
    fn test_deterministic_sorting() {
        let config = SequencerConfig::default();
        let l1_client = Arc::new(MockL1Client::new());
        let compression_engine = CompressionEngine::new(Default::default());
        
        let sequencer = BasedSequencer::new(config, l1_client, compression_engine);

        let mut orders = vec![
            create_test_order(3, 99.0, 1.0, Side::Sell),
            create_test_order(1, 100.0, 1.0, Side::Buy),
            create_test_order(2, 101.0, 2.0, Side::Buy),
        ];

        // Sort multiple times to ensure determinism
        let sorted1 = sequencer.sort_orders_deterministically(orders.clone());
        let sorted2 = sequencer.sort_orders_deterministically(orders.clone());
        
        assert_eq!(sorted1.len(), sorted2.len());
        for (a, b) in sorted1.iter().zip(sorted2.iter()) {
            assert_eq!(a.id, b.id);
        }
    }
}