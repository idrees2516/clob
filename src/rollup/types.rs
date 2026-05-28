//! Common types for rollup operations

use crate::orderbook::types::{Order, Trade};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for batches
pub type BatchId = u64;

/// L1 block number
pub type L1BlockNumber = u64;

/// Transaction hash on L1
pub type TxHash = [u8; 32];

/// State root hash
pub type StateRoot = [u8; 32];

/// Blob hash for data availability
pub type BlobHash = [u8; 32];

/// Timestamp in seconds since Unix epoch
pub type Timestamp = u64;

/// Order batch for rollup processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBatch {
    pub batch_id: BatchId,
    pub orders: Vec<Order>,
    pub timestamp: Timestamp,
    pub l1_block_number: L1BlockNumber,
    pub state_root_before: StateRoot,
    pub state_root_after: StateRoot,
    pub sequencer_signature: Option<Vec<u8>>,
}

/// Batch execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExecutionResult {
    pub batch_id: BatchId,
    pub trades_generated: Vec<Trade>,
    pub state_transitions: Vec<StateTransition>,
    pub gas_used: u64,
    pub execution_time_ms: u64,
    pub final_state_root: StateRoot,
}

/// State transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub transition_type: StateTransitionType,
    pub order_id: Option<u64>,
    pub before_state: Vec<u8>,
    pub after_state: Vec<u8>,
    pub gas_cost: u64,
}

/// Types of state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateTransitionType {
    OrderPlacement,
    OrderCancellation,
    TradeExecution,
    InventoryUpdate,
    PriceUpdate,
}

/// L1 settlement transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementTransaction {
    pub batch_id: BatchId,
    pub state_root: StateRoot,
    pub blob_hash: BlobHash,
    pub proof_data: Vec<u8>,
    pub gas_limit: u64,
    pub gas_price: u64,
}

/// Data availability blob
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBlob {
    pub blob_id: BlobHash,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
    pub compression_ratio: f64,
    pub merkle_root: [u8; 32],
    pub timestamp: Timestamp,
}

/// Batch commitment for L1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCommitment {
    pub batch_id: BatchId,
    pub state_root: StateRoot,
    pub blob_hash: BlobHash,
    pub l1_block_number: L1BlockNumber,
    pub commitment_hash: [u8; 32],
}

/// Sequencer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencerConfig {
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub l1_sync_interval_ms: u64,
    pub mempool_size_limit: usize,
    pub enable_compression: bool,
    pub deterministic_sorting: bool,
}

/// Settlement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementConfig {
    pub l1_rpc_url: String,
    pub contract_address: String,
    pub private_key: Option<String>,
    pub gas_limit: u64,
    pub gas_price_multiplier: f64,
    pub confirmation_blocks: u64,
}

/// Data availability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAvailabilityConfig {
    pub blob_storage_type: BlobStorageType,
    pub max_blob_size: usize,
    pub compression_level: u8,
    pub enable_ipfs_backup: bool,
    pub ipfs_gateway: Option<String>,
    pub retention_period_days: u32,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub level: u8,
    pub enable_delta_encoding: bool,
    pub enable_dictionary_compression: bool,
    pub dictionary_size: usize,
}

/// Blob storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlobStorageType {
    EIP4844,
    IPFS,
    Celestia,
    Avail,
    Custom(String),
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Zstd,
    Lz4,
    Brotli,
    Custom(String),
}

impl Default for SequencerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            batch_timeout_ms: 5000, // 5 seconds
            l1_sync_interval_ms: 12000, // 12 seconds (Ethereum block time)
            mempool_size_limit: 10000,
            enable_compression: true,
            deterministic_sorting: true,
        }
    }
}

impl Default for SettlementConfig {
    fn default() -> Self {
        Self {
            l1_rpc_url: "http://localhost:8545".to_string(),
            contract_address: "0x0000000000000000000000000000000000000000".to_string(),
            private_key: None,
            gas_limit: 500000,
            gas_price_multiplier: 1.1,
            confirmation_blocks: 3,
        }
    }
}

impl Default for DataAvailabilityConfig {
    fn default() -> Self {
        Self {
            blob_storage_type: BlobStorageType::EIP4844,
            max_blob_size: 128 * 1024, // 128KB
            compression_level: 6,
            enable_ipfs_backup: false,
            ipfs_gateway: None,
            retention_period_days: 30,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Zstd,
            level: 6,
            enable_delta_encoding: true,
            enable_dictionary_compression: true,
            dictionary_size: 64 * 1024, // 64KB
        }
    }
}

impl OrderBatch {
    /// Create a new order batch
    pub fn new(
        batch_id: BatchId,
        orders: Vec<Order>,
        l1_block_number: L1BlockNumber,
        state_root_before: StateRoot,
    ) -> Self {
        Self {
            batch_id,
            orders,
            timestamp: chrono::Utc::now().timestamp() as u64,
            l1_block_number,
            state_root_before,
            state_root_after: [0u8; 32], // Will be computed after execution
            sequencer_signature: None,
        }
    }

    /// Get the number of orders in the batch
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Get total volume of all orders
    pub fn total_volume(&self) -> f64 {
        self.orders.iter().map(|order| order.quantity).sum()
    }

    /// Sign the batch with sequencer key
    pub fn sign(&mut self, signature: Vec<u8>) {
        self.sequencer_signature = Some(signature);
    }

    /// Verify batch signature
    pub fn verify_signature(&self, public_key: &[u8]) -> bool {
        // Implementation would verify the signature
        // For now, just check if signature exists
        self.sequencer_signature.is_some()
    }
}

impl BatchExecutionResult {
    /// Create a new batch execution result
    pub fn new(
        batch_id: BatchId,
        trades_generated: Vec<Trade>,
        state_transitions: Vec<StateTransition>,
        gas_used: u64,
        execution_time_ms: u64,
        final_state_root: StateRoot,
    ) -> Self {
        Self {
            batch_id,
            trades_generated,
            state_transitions,
            gas_used,
            execution_time_ms,
            final_state_root,
        }
    }

    /// Get the number of trades generated
    pub fn trade_count(&self) -> usize {
        self.trades_generated.len()
    }

    /// Get total traded volume
    pub fn total_traded_volume(&self) -> f64 {
        self.trades_generated.iter().map(|trade| trade.quantity).sum()
    }

    /// Calculate average gas per order
    pub fn average_gas_per_order(&self, order_count: usize) -> f64 {
        if order_count == 0 {
            0.0
        } else {
            self.gas_used as f64 / order_count as f64
        }
    }
}