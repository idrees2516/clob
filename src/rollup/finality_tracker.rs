//! L1 Transaction Finality Tracker
//! 
//! This module implements transaction finality tracking for ethrex L1 integration,
//! monitoring transaction confirmations and handling finality events.

use crate::rollup::{
    ethrex_client::*,
    types::*,
    RollupError,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
use tracing::{debug, info, warn, error};

/// Finality tracker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityTrackerConfig {
    /// Polling interval for checking transaction status
    pub polling_interval_ms: u64,
    /// Number of confirmations required for finality
    pub finality_confirmations: u64,
    /// Maximum time to wait for transaction confirmation
    pub confirmation_timeout_seconds: u64,
    /// Maximum number of transactions to track simultaneously
    pub max_tracked_transactions: usize,
    /// Enable deep reorganization detection
    pub enable_deep_reorg_detection: bool,
    /// Maximum depth to check for reorganizations
    pub max_reorg_depth: u64,
}

impl Default for FinalityTrackerConfig {
    fn default() -> Self {
        Self {
            polling_interval_ms: 5000, // 5 seconds
            finality_confirmations: 12, // 12 confirmations for finality
            confirmation_timeout_seconds: 3600, // 1 hour timeout
            max_tracked_transactions: 10000,
            enable_deep_reorg_detection: true,
            max_reorg_depth: 64, // Check up to 64 blocks for reorgs
        }
    }
}

/// Transaction finality tracker
pub struct FinalityTracker {
    config: FinalityTrackerConfig,
    rpc_client: Arc<dyn EthereumRpcClient>,
    tracked_transactions: Arc<RwLock<HashMap<TxHash, TrackedTransaction>>>,
    finalized_transactions: Arc<RwLock<HashMap<TxHash, FinalizedTransactionInfo>>>,
    finality_events_tx: mpsc::UnboundedSender<FinalityEvent>,
    finality_events_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<FinalityEvent>>>>,
    block_history: Arc<RwLock<VecDeque<BlockInfo>>>,
    last_processed_block: Arc<RwLock<Option<L1BlockNumber>>>,
}

/// Information about a tracked transaction
#[derive(Debug, Clone)]
pub struct TrackedTransaction {
    /// Transaction hash
    pub tx_hash: TxHash,
    /// Transaction type for context
    pub transaction_type: TransactionType,
    /// Block number where transaction was included (if known)
    pub inclusion_block: Option<L1BlockNumber>,
    /// Current number of confirmations
    pub confirmations: u64,
    /// Time when tracking started
    pub tracking_start_time: Instant,
    /// Whether transaction has been confirmed
    pub is_confirmed: bool,
    /// Whether transaction has reached finality
    pub is_finalized: bool,
    /// Last status check time
    pub last_check_time: Instant,
    /// Associated state root (for proof commitments)
    pub associated_state_root: Option<StateRoot>,
}

/// Information about a finalized transaction
#[derive(Debug, Clone)]
pub struct FinalizedTransactionInfo {
    /// Transaction hash
    pub tx_hash: TxHash,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Block where transaction was finalized
    pub finalization_block: L1BlockNumber,
    /// Time when finality was reached
    pub finalization_time: Instant,
    /// Total confirmations at finalization
    pub final_confirmations: u64,
    /// Associated state root
    pub associated_state_root: Option<StateRoot>,
}

/// Block information for reorganization detection
#[derive(Debug, Clone)]
pub struct BlockInfo {
    /// Block number
    pub number: L1BlockNumber,
    /// Block hash
    pub hash: [u8; 32],
    /// Parent block hash
    pub parent_hash: [u8; 32],
    /// Block timestamp
    pub timestamp: u64,
    /// Transactions in this block
    pub transactions: Vec<TxHash>,
}

/// Finality events emitted by the tracker
#[derive(Debug, Clone)]
pub enum FinalityEvent {
    /// Transaction was confirmed (reached required confirmations)
    TransactionConfirmed {
        tx_hash: TxHash,
        transaction_type: TransactionType,
        block_number: L1BlockNumber,
        confirmations: u64,
        associated_state_root: Option<StateRoot>,
    },
    /// Transaction reached finality
    TransactionFinalized {
        tx_hash: TxHash,
        transaction_type: TransactionType,
        finalization_block: L1BlockNumber,
        final_confirmations: u64,
        associated_state_root: Option<StateRoot>,
    },
    /// Transaction failed or was dropped
    TransactionFailed {
        tx_hash: TxHash,
        transaction_type: TransactionType,
        reason: String,
    },
    /// Reorganization detected
    ReorganizationDetected {
        reorg_block: L1BlockNumber,
        original_hash: [u8; 32],
        new_hash: [u8; 32],
        affected_transactions: Vec<TxHash>,
    },
    /// Deep reorganization detected (beyond normal confirmation depth)
    DeepReorganizationDetected {
        reorg_depth: u64,
        reorg_block: L1BlockNumber,
        affected_transactions: Vec<TxHash>,
    },
}

/// Finality tracker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalityStats {
    /// Number of currently tracked transactions
    pub tracked_count: usize,
    /// Number of finalized transactions
    pub finalized_count: usize,
    /// Number of failed transactions
    pub failed_count: usize,
    /// Average confirmation time
    pub average_confirmation_time_seconds: f64,
    /// Average finalization time
    pub average_finalization_time_seconds: f64,
    /// Number of reorganizations detected
    pub reorganizations_detected: usize,
    /// Current block height
    pub current_block_height: Option<L1BlockNumber>,
}

impl FinalityTracker {
    /// Create a new finality tracker
    pub fn new(
        config: FinalityTrackerConfig,
        rpc_client: Arc<dyn EthereumRpcClient>,
    ) -> Self {
        let (finality_events_tx, finality_events_rx) = mpsc::unbounded_channel();

        Self {
            config,
            rpc_client,
            tracked_transactions: Arc::new(RwLock::new(HashMap::new())),
            finalized_transactions: Arc::new(RwLock::new(HashMap::new())),
            finality_events_tx,
            finality_events_rx: Arc::new(RwLock::new(Some(finality_events_rx))),
            block_history: Arc::new(RwLock::new(VecDeque::new())),
            last_processed_block: Arc::new(RwLock::new(None)),
        }
    }

    /// Start tracking a transaction for finality
    pub async fn track_transaction(
        &self,
        tx_hash: TxHash,
        transaction_type: TransactionType,
        associated_state_root: Option<StateRoot>,
    ) -> Result<(), RollupError> {
        info!("Starting to track transaction: {:?}", tx_hash);

        let tracked_tx = TrackedTransaction {
            tx_hash,
            transaction_type: transaction_type.clone(),
            inclusion_block: None,
            confirmations: 0,
            tracking_start_time: Instant::now(),
            is_confirmed: false,
            is_finalized: false,
            last_check_time: Instant::now(),
            associated_state_root,
        };

        {
            let mut tracked = self.tracked_transactions.write().await;
            
            // Check if we're at capacity
            if tracked.len() >= self.config.max_tracked_transactions {
                return Err(RollupError::L1SyncError(
                    "Maximum tracked transactions limit reached".to_string()
                ));
            }
            
            tracked.insert(tx_hash, tracked_tx);
        }

        debug!("Added transaction {:?} to tracking", tx_hash);
        Ok(())
    }

    /// Stop tracking a transaction
    pub async fn stop_tracking(&self, tx_hash: TxHash) -> Result<(), RollupError> {
        let mut tracked = self.tracked_transactions.write().await;
        if tracked.remove(&tx_hash).is_some() {
            info!("Stopped tracking transaction: {:?}", tx_hash);
        }
        Ok(())
    }

    /// Get finality event receiver
    pub async fn get_finality_events(&self) -> Option<mpsc::UnboundedReceiver<FinalityEvent>> {
        let mut rx_option = self.finality_events_rx.write().await;
        rx_option.take()
    }

    /// Start the finality tracking loop
    pub async fn start_tracking_loop(&self) -> Result<(), RollupError> {
        info!("Starting finality tracking loop");

        let mut interval = interval(Duration::from_millis(self.config.polling_interval_ms));

        loop {
            interval.tick().await;

            if let Err(e) = self.process_tracking_cycle().await {
                error!("Error in tracking cycle: {}", e);
                // Continue processing despite errors
            }
        }
    }

    /// Process one cycle of transaction tracking
    async fn process_tracking_cycle(&self) -> Result<(), RollupError> {
        // Get current block number
        let current_block = self.rpc_client.get_block_number().await?;
        
        // Update block history for reorganization detection
        self.update_block_history(current_block).await?;
        
        // Check for reorganizations
        if self.config.enable_deep_reorg_detection {
            self.detect_reorganizations().await?;
        }

        // Process tracked transactions
        self.process_tracked_transactions(current_block).await?;

        // Clean up old finalized transactions
        self.cleanup_old_transactions().await?;

        // Update last processed block
        {
            let mut last_block = self.last_processed_block.write().await;
            *last_block = Some(current_block);
        }

        Ok(())
    }

    /// Update block history for reorganization detection
    async fn update_block_history(&self, current_block: L1BlockNumber) -> Result<(), RollupError> {
        let block = self.rpc_client.get_block(current_block).await?;
        
        let block_info = BlockInfo {
            number: block.number,
            hash: block.hash,
            parent_hash: block.parent_hash,
            timestamp: block.timestamp,
            transactions: block.transactions,
        };

        {
            let mut history = self.block_history.write().await;
            history.push_back(block_info);

            // Keep only recent blocks for reorganization detection
            while history.len() > self.config.max_reorg_depth as usize {
                history.pop_front();
            }
        }

        Ok(())
    }

    /// Detect reorganizations by checking block history
    async fn detect_reorganizations(&self) -> Result<(), RollupError> {
        let history = self.block_history.read().await;
        
        if history.len() < 2 {
            return Ok(());
        }

        // Check if the chain is consistent
        for i in 1..history.len() {
            let current_block = &history[i];
            let previous_block = &history[i - 1];

            // Check if current block's parent hash matches previous block's hash
            if current_block.parent_hash != previous_block.hash {
                // Reorganization detected
                warn!(
                    "Reorganization detected at block {}: expected parent {:?}, got {:?}",
                    current_block.number, previous_block.hash, current_block.parent_hash
                );

                let affected_transactions = self.find_affected_transactions(current_block.number).await;

                let reorg_event = FinalityEvent::ReorganizationDetected {
                    reorg_block: current_block.number,
                    original_hash: previous_block.hash,
                    new_hash: current_block.hash,
                    affected_transactions: affected_transactions.clone(),
                };

                // Emit reorganization event
                if let Err(e) = self.finality_events_tx.send(reorg_event) {
                    error!("Failed to send reorganization event: {}", e);
                }

                // Update affected transactions
                self.handle_reorganization_for_transactions(affected_transactions).await?;

                break; // Only handle one reorg per cycle
            }
        }

        Ok(())
    }

    /// Find transactions affected by reorganization
    async fn find_affected_transactions(&self, reorg_block: L1BlockNumber) -> Vec<TxHash> {
        let tracked = self.tracked_transactions.read().await;
        
        tracked.values()
            .filter(|tx| {
                if let Some(inclusion_block) = tx.inclusion_block {
                    inclusion_block >= reorg_block
                } else {
                    false
                }
            })
            .map(|tx| tx.tx_hash)
            .collect()
    }

    /// Handle reorganization for affected transactions
    async fn handle_reorganization_for_transactions(&self, affected_txs: Vec<TxHash>) -> Result<(), RollupError> {
        let mut tracked = self.tracked_transactions.write().await;

        for tx_hash in affected_txs {
            if let Some(tx) = tracked.get_mut(&tx_hash) {
                // Reset transaction status
                tx.inclusion_block = None;
                tx.confirmations = 0;
                tx.is_confirmed = false;
                tx.is_finalized = false;
                tx.last_check_time = Instant::now();

                info!("Reset transaction {:?} due to reorganization", tx_hash);
            }
        }

        Ok(())
    }

    /// Process all tracked transactions
    async fn process_tracked_transactions(&self, current_block: L1BlockNumber) -> Result<(), RollupError> {
        let tracked_txs: Vec<(TxHash, TrackedTransaction)> = {
            let tracked = self.tracked_transactions.read().await;
            tracked.iter().map(|(k, v)| (*k, v.clone())).collect()
        };

        for (tx_hash, mut tracked_tx) in tracked_txs {
            // Check for timeout
            if tracked_tx.tracking_start_time.elapsed().as_secs() > self.config.confirmation_timeout_seconds {
                self.handle_transaction_timeout(tx_hash, tracked_tx).await?;
                continue;
            }

            // Get transaction receipt
            if let Some(receipt) = self.rpc_client.get_transaction_receipt(tx_hash).await? {
                let confirmations = current_block.saturating_sub(receipt.block_number);
                
                // Update transaction info
                tracked_tx.inclusion_block = Some(receipt.block_number);
                tracked_tx.confirmations = confirmations;
                tracked_tx.last_check_time = Instant::now();

                // Check if transaction failed
                if matches!(receipt.status, TransactionStatus::Failed) {
                    self.handle_transaction_failure(tx_hash, tracked_tx, "Transaction failed on L1".to_string()).await?;
                    continue;
                }

                // Check for confirmation
                if !tracked_tx.is_confirmed && confirmations >= self.config.finality_confirmations {
                    tracked_tx.is_confirmed = true;
                    
                    let confirmation_event = FinalityEvent::TransactionConfirmed {
                        tx_hash,
                        transaction_type: tracked_tx.transaction_type.clone(),
                        block_number: receipt.block_number,
                        confirmations,
                        associated_state_root: tracked_tx.associated_state_root,
                    };

                    if let Err(e) = self.finality_events_tx.send(confirmation_event) {
                        error!("Failed to send confirmation event: {}", e);
                    }

                    info!("Transaction {:?} confirmed with {} confirmations", tx_hash, confirmations);
                }

                // Check for finality
                if !tracked_tx.is_finalized && confirmations >= self.config.finality_confirmations {
                    tracked_tx.is_finalized = true;
                    
                    // Move to finalized transactions
                    let finalized_info = FinalizedTransactionInfo {
                        tx_hash,
                        transaction_type: tracked_tx.transaction_type.clone(),
                        finalization_block: receipt.block_number,
                        finalization_time: Instant::now(),
                        final_confirmations: confirmations,
                        associated_state_root: tracked_tx.associated_state_root,
                    };

                    {
                        let mut finalized = self.finalized_transactions.write().await;
                        finalized.insert(tx_hash, finalized_info);
                    }

                    let finality_event = FinalityEvent::TransactionFinalized {
                        tx_hash,
                        transaction_type: tracked_tx.transaction_type.clone(),
                        finalization_block: receipt.block_number,
                        final_confirmations: confirmations,
                        associated_state_root: tracked_tx.associated_state_root,
                    };

                    if let Err(e) = self.finality_events_tx.send(finality_event) {
                        error!("Failed to send finality event: {}", e);
                    }

                    // Remove from tracked transactions
                    {
                        let mut tracked = self.tracked_transactions.write().await;
                        tracked.remove(&tx_hash);
                    }

                    info!("Transaction {:?} finalized at block {}", tx_hash, receipt.block_number);
                    continue;
                }

                // Update tracked transaction
                {
                    let mut tracked = self.tracked_transactions.write().await;
                    if let Some(tx) = tracked.get_mut(&tx_hash) {
                        *tx = tracked_tx;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle transaction timeout
    async fn handle_transaction_timeout(&self, tx_hash: TxHash, tracked_tx: TrackedTransaction) -> Result<(), RollupError> {
        warn!("Transaction {:?} timed out after {} seconds", tx_hash, self.config.confirmation_timeout_seconds);

        let failure_event = FinalityEvent::TransactionFailed {
            tx_hash,
            transaction_type: tracked_tx.transaction_type,
            reason: "Transaction confirmation timeout".to_string(),
        };

        if let Err(e) = self.finality_events_tx.send(failure_event) {
            error!("Failed to send timeout event: {}", e);
        }

        // Remove from tracked transactions
        {
            let mut tracked = self.tracked_transactions.write().await;
            tracked.remove(&tx_hash);
        }

        Ok(())
    }

    /// Handle transaction failure
    async fn handle_transaction_failure(&self, tx_hash: TxHash, tracked_tx: TrackedTransaction, reason: String) -> Result<(), RollupError> {
        warn!("Transaction {:?} failed: {}", tx_hash, reason);

        let failure_event = FinalityEvent::TransactionFailed {
            tx_hash,
            transaction_type: tracked_tx.transaction_type,
            reason,
        };

        if let Err(e) = self.finality_events_tx.send(failure_event) {
            error!("Failed to send failure event: {}", e);
        }

        // Remove from tracked transactions
        {
            let mut tracked = self.tracked_transactions.write().await;
            tracked.remove(&tx_hash);
        }

        Ok(())
    }

    /// Clean up old finalized transactions
    async fn cleanup_old_transactions(&self) -> Result<(), RollupError> {
        // This is a simple cleanup - in practice you might want more sophisticated retention policies
        let mut finalized = self.finalized_transactions.write().await;
        
        // Keep only recent finalized transactions (e.g., last 1000)
        if finalized.len() > 1000 {
            let mut sorted_txs: Vec<_> = finalized.iter().collect();
            sorted_txs.sort_by_key(|(_, info)| info.finalization_time);
            
            // Remove oldest transactions
            let to_remove = sorted_txs.len() - 1000;
            for (tx_hash, _) in sorted_txs.into_iter().take(to_remove) {
                finalized.remove(tx_hash);
            }
        }

        Ok(())
    }

    /// Get finality statistics
    pub async fn get_finality_stats(&self) -> FinalityStats {
        let tracked = self.tracked_transactions.read().await;
        let finalized = self.finalized_transactions.read().await;
        let last_block = self.last_processed_block.read().await;

        // Calculate average times
        let mut total_confirmation_time = 0.0;
        let mut total_finalization_time = 0.0;
        let mut confirmation_count = 0;
        let mut finalization_count = 0;

        for tx in tracked.values() {
            if tx.is_confirmed {
                total_confirmation_time += tx.last_check_time.duration_since(tx.tracking_start_time).as_secs_f64();
                confirmation_count += 1;
            }
        }

        for info in finalized.values() {
            total_finalization_time += info.finalization_time.duration_since(info.finalization_time).as_secs_f64();
            finalization_count += 1;
        }

        FinalityStats {
            tracked_count: tracked.len(),
            finalized_count: finalized.len(),
            failed_count: 0, // Would need to track this separately
            average_confirmation_time_seconds: if confirmation_count > 0 {
                total_confirmation_time / confirmation_count as f64
            } else {
                0.0
            },
            average_finalization_time_seconds: if finalization_count > 0 {
                total_finalization_time / finalization_count as f64
            } else {
                0.0
            },
            reorganizations_detected: 0, // Would need to track this separately
            current_block_height: *last_block,
        }
    }

    /// Get information about a specific tracked transaction
    pub async fn get_transaction_info(&self, tx_hash: TxHash) -> Option<TrackedTransaction> {
        let tracked = self.tracked_transactions.read().await;
        tracked.get(&tx_hash).cloned()
    }

    /// Get information about a finalized transaction
    pub async fn get_finalized_transaction_info(&self, tx_hash: TxHash) -> Option<FinalizedTransactionInfo> {
        let finalized = self.finalized_transactions.read().await;
        finalized.get(&tx_hash).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollup::ethrex_client::MockEthereumRpcClient;
    use std::time::Duration;

    #[tokio::test]
    async fn test_finality_tracker_creation() {
        let config = FinalityTrackerConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        
        let tracker = FinalityTracker::new(config, rpc_client);
        
        let stats = tracker.get_finality_stats().await;
        assert_eq!(stats.tracked_count, 0);
        assert_eq!(stats.finalized_count, 0);
    }

    #[tokio::test]
    async fn test_transaction_tracking() {
        let config = FinalityTrackerConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        let tracker = FinalityTracker::new(config, rpc_client);

        let tx_hash = [123u8; 32];
        let tx_type = TransactionType::ProofCommitment { batch_id: 1 };

        tracker.track_transaction(tx_hash, tx_type, Some([1u8; 32])).await.unwrap();

        let stats = tracker.get_finality_stats().await;
        assert_eq!(stats.tracked_count, 1);

        let tx_info = tracker.get_transaction_info(tx_hash).await;
        assert!(tx_info.is_some());
        assert_eq!(tx_info.unwrap().tx_hash, tx_hash);
    }

    #[tokio::test]
    async fn test_stop_tracking() {
        let config = FinalityTrackerConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        let tracker = FinalityTracker::new(config, rpc_client);

        let tx_hash = [123u8; 32];
        let tx_type = TransactionType::ProofCommitment { batch_id: 1 };

        tracker.track_transaction(tx_hash, tx_type, None).await.unwrap();
        assert_eq!(tracker.get_finality_stats().await.tracked_count, 1);

        tracker.stop_tracking(tx_hash).await.unwrap();
        assert_eq!(tracker.get_finality_stats().await.tracked_count, 0);
    }
}