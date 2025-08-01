//! Integration tests for the rollup system

use hf_quoting_liquidity_clob::rollup::*;
use hf_quoting_liquidity_clob::orderbook::types::{Order, Trade, Side, OrderType};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio;

#[tokio::test]
async fn test_rollup_integration() {
    // Create test components
    let sequencer_config = SequencerConfig::default();
    let compression_engine = CompressionEngine::new(CompressionConfig::default());
    
    // Create mock L1 client
    let l1_client = Arc::new(MockL1Client::new());
    
    // Create sequencer
    let sequencer = BasedSequencer::new(
        sequencer_config,
        l1_client,
        compression_engine,
    );

    // Create test order
    let order = Order {
        id: 1,
        symbol: "BTC-USD".to_string(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        price: 50000.0,
        quantity: 1.0,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        user_id: "test_user".to_string(),
    };

    // Add order to mempool
    sequencer.add_order(order).await.expect("Failed to add order");

    // Check mempool stats
    let stats = sequencer.get_mempool_stats();
    assert_eq!(stats.order_count, 1);

    // Build batch
    let batch = sequencer.build_batch().await.expect("Failed to build batch");
    assert_eq!(batch.order_count(), 1);
    assert_eq!(batch.orders[0].id, 1);
}

/// Mock L1 client for testing
struct MockL1Client {
    current_block: parking_lot::Mutex<L1Block>,
}

impl MockL1Client {
    fn new() -> Self {
        Self {
            current_block: parking_lot::Mutex::new(L1Block {
                number: 1,
                hash: [1u8; 32],
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                parent_hash: [0u8; 32],
                gas_limit: 30_000_000,
                gas_used: 15_000_000,
            }),
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

#[tokio::test]
async fn test_data_availability_integration() {
    // Create test components
    let config = DataAvailabilityConfig::default();
    let blob_storage = Arc::new(MockBlobStorage::new());
    let compression_engine = CompressionEngine::new(CompressionConfig::default());
    
    let mut da_client = DataAvailabilityClient::new(
        config,
        blob_storage,
        None,
        compression_engine,
    );

    // Create test data
    let orders = vec![
        Order {
            id: 1,
            symbol: "BTC-USD".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 50000.0,
            quantity: 1.0,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            user_id: "user_1".to_string(),
        }
    ];

    let batch = OrderBatch::new(1, orders, 100, [0u8; 32]);
    let trades = vec![
        Trade {
            id: 1,
            symbol: "BTC-USD".to_string(),
            price: 50000.0,
            quantity: 0.5,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            buyer_id: "buyer_1".to_string(),
            seller_id: "seller_1".to_string(),
            buy_order_id: 1,
            sell_order_id: 2,
        }
    ];

    // Store data
    let commitment = da_client.store_trading_data(&batch, &trades).await
        .expect("Failed to store trading data");

    // Verify commitment
    assert!(!commitment.blob_hash.iter().all(|&b| b == 0));
    assert_eq!(commitment.backend, "Mock");

    // Retrieve data
    let (retrieved_batch, retrieved_trades) = da_client
        .retrieve_trading_data(commitment.blob_hash)
        .await
        .expect("Failed to retrieve trading data");

    // Verify retrieved data
    assert_eq!(retrieved_batch.orders.len(), 1);
    assert_eq!(retrieved_trades.len(), 1);
    assert_eq!(retrieved_batch.orders[0].id, 1);
    assert_eq!(retrieved_trades[0].id, 1);
}

/// Mock blob storage for testing
struct MockBlobStorage {
    blobs: Arc<tokio::sync::RwLock<std::collections::HashMap<BlobHash, DataBlob>>>,
}

impl MockBlobStorage {
    fn new() -> Self {
        Self {
            blobs: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
        }
    }
}

#[async_trait::async_trait]
impl BlobStorage for MockBlobStorage {
    async fn store_blob(&self, blob: &DataBlob) -> Result<BlobCommitment, RollupError> {
        let mut blobs = self.blobs.write().await;
        blobs.insert(blob.blob_id, blob.clone());
        
        Ok(BlobCommitment {
            blob_hash: blob.blob_id,
            storage_proof: vec![42u8; 32],
            timestamp: blob.timestamp,
            backend: "Mock".to_string(),
            ipfs_backup: None,
        })
    }

    async fn retrieve_blob(&self, blob_hash: BlobHash) -> Result<DataBlob, RollupError> {
        let blobs = self.blobs.read().await;
        blobs.get(&blob_hash)
            .cloned()
            .ok_or_else(|| RollupError::DataAvailabilityError("Blob not found".to_string()))
    }

    async fn blob_exists(&self, blob_hash: BlobHash) -> Result<bool, RollupError> {
        let blobs = self.blobs.read().await;
        Ok(blobs.contains_key(&blob_hash))
    }

    async fn get_blob_metadata(&self, blob_hash: BlobHash) -> Result<BlobMetadata, RollupError> {
        let blobs = self.blobs.read().await;
        if let Some(blob) = blobs.get(&blob_hash) {
            Ok(BlobMetadata {
                blob_hash,
                size: blob.compressed_data.len(),
                compression_ratio: blob.compression_ratio,
                timestamp: blob.timestamp,
                storage_backend: "Mock".to_string(),
                ipfs_hash: None,
            })
        } else {
            Err(RollupError::DataAvailabilityError("Blob not found".to_string()))
        }
    }

    async fn delete_blob(&self, blob_hash: BlobHash) -> Result<(), RollupError> {
        let mut blobs = self.blobs.write().await;
        blobs.remove(&blob_hash);
        Ok(())
    }

    async fn list_blobs(&self, limit: Option<usize>) -> Result<Vec<BlobHash>, RollupError> {
        let blobs = self.blobs.read().await;
        let mut hashes: Vec<BlobHash> = blobs.keys().cloned().collect();
        if let Some(limit) = limit {
            hashes.truncate(limit);
        }
        Ok(hashes)
    }
}