//! ethrex L1 Integration Manager
//! 
//! This module provides a high-level interface for ethrex L1 integration,
//! coordinating proof commitment submission, state anchoring, transaction
//! confirmation, and finality tracking.

use crate::rollup::{
    ethrex_client::*,
    ethrex_rpc_client::*,
    finality_tracker::*,
    types::*,
    RollupError,
};
use crate::rollup::proof_anchoring::L1VerificationClient;
use crate::zkvm::ZkProof;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn, error};

/// ethrex L1 integration manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthrexIntegrationConfig {
    /// ethrex client configuration
    pub client_config: EthrexClientConfig,
    /// Finality tracker configuration
    pub finality_config: FinalityTrackerConfig,
    /// Enable automatic state synchronization
    pub enable_auto_sync: bool,
    /// State synchronization interval in milliseconds
    pub sync_interval_ms: u64,
    /// Enable automatic finality tracking
    pub enable_finality_tracking: bool,
    /// Maximum concurrent proof submissions
    pub max_concurrent_submissions: usize,
}

impl Default for EthrexIntegrationConfig {
    fn default() -> Self {
        Self {
            client_config: EthrexClientConfig::default(),
            finality_config: FinalityTrackerConfig::default(),
            enable_auto_sync: true,
            sync_interval_ms: 12000, // 12 seconds (Ethereum block time)
            enable_finality_tracking: true,
            max_concurrent_submissions: 10,
        }
    }
}

/// ethrex L1 integration manager
pub struct EthrexIntegrationManager {
    config: EthrexIntegrationConfig,
    l1_client: Arc<EthrexL1Client>,
    finality_tracker: Arc<FinalityTracker>,
    finality_events_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<FinalityEvent>>>>,
    sync_task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    finality_task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    submission_semaphore: Arc<tokio::sync::Semaphore>,
    integration_stats: Arc<RwLock<IntegrationStats>>,
}

/// Integration statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Total proof commitments submitted
    pub total_proof_commitments: u64,
    /// Total state root anchors submitted
    pub total_state_anchors: u64,
    /// Successful submissions
    pub successful_submissions: u64,
    /// Failed submissions
    pub failed_submissions: u64,
    /// Average submission time in milliseconds
    pub average_submission_time_ms: f64,
    /// Current pending transactions
    pub pending_transactions: usize,
    /// Finalized transactions
    pub finalized_transactions: usize,
    /// Reorganizations handled
    pub reorganizations_handled: u64,
    /// Last sync time
    pub last_sync_time: Option<u64>,
    /// Integration uptime in seconds
    pub uptime_seconds: u64,
}

impl Default for IntegrationStats {
    fn default() -> Self {
        Self {
            total_proof_commitments: 0,
            total_state_anchors: 0,
            successful_submissions: 0,
            failed_submissions: 0,
            average_submission_time_ms: 0.0,
            pending_transactions: 0,
            finalized_transactions: 0,
            reorganizations_handled: 0,
            last_sync_time: None,
            uptime_seconds: 0,
        }
    }
}

/// Proof commitment request
#[derive(Debug, Clone)]
pub struct ProofCommitmentRequest {
    /// Batch ID being committed
    pub batch_id: BatchId,
    /// ZK proof to commit
    pub proof: ZkProof,
    /// State root before batch execution
    pub state_root_before: StateRoot,
    /// State root after batch execution
    pub state_root_after: StateRoot,
    /// Merkle root of order book state
    pub order_book_merkle_root: [u8; 32],
    /// Priority level for submission
    pub priority: SubmissionPriority,
}

/// State anchor request
#[derive(Debug, Clone)]
pub struct StateAnchorRequest {
    /// State root to anchor
    pub state_root: StateRoot,
    /// ZK proof for state validity
    pub proof: ZkProof,
    /// Priority level for submission
    pub priority: SubmissionPriority,
}

/// Submission priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SubmissionPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Submission result
#[derive(Debug, Clone)]
pub struct SubmissionResult {
    /// Transaction hash on L1
    pub tx_hash: TxHash,
    /// Submission timestamp
    pub submission_time: Instant,
    /// Associated state root
    pub state_root: StateRoot,
    /// Batch ID (for proof commitments)
    pub batch_id: Option<BatchId>,
}

impl EthrexIntegrationManager {
    /// Create a new ethrex integration manager
    pub async fn new(config: EthrexIntegrationConfig) -> Result<Self, RollupError> {
        info!("Initializing ethrex L1 integration manager");

        // Create RPC client
        let rpc_client = EthrexRpcClientFactory::create_client(&config.client_config)?;

        // Create L1 client
        let l1_client = Arc::new(EthrexL1Client::new(
            config.client_config.clone(),
            rpc_client.clone(),
        ));

        // Create finality tracker
        let finality_tracker = Arc::new(FinalityTracker::new(
            config.finality_config.clone(),
            rpc_client,
        ));

        // Get finality events receiver
        let finality_events_rx = finality_tracker.get_finality_events().await;

        let manager = Self {
            config: config.clone(),
            l1_client,
            finality_tracker,
            finality_events_rx: Arc::new(RwLock::new(finality_events_rx)),
            sync_task_handle: Arc::new(RwLock::new(None)),
            finality_task_handle: Arc::new(RwLock::new(None)),
            submission_semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_submissions)),
            integration_stats: Arc::new(RwLock::new(IntegrationStats::default())),
        };

        info!("ethrex L1 integration manager initialized successfully");
        Ok(manager)
    }

    /// Start the integration manager
    pub async fn start(&self) -> Result<(), RollupError> {
        info!("Starting ethrex L1 integration manager");

        // Start finality tracking if enabled
        if self.config.enable_finality_tracking {
            let finality_tracker = Arc::clone(&self.finality_tracker);
            let finality_handle = tokio::spawn(async move {
                if let Err(e) = finality_tracker.start_tracking_loop().await {
                    error!("Finality tracking loop failed: {}", e);
                }
            });

            {
                let mut handle = self.finality_task_handle.write().await;
                *handle = Some(finality_handle);
            }

            // Start finality event processing
            self.start_finality_event_processing().await?;
        }

        // Start automatic synchronization if enabled
        if self.config.enable_auto_sync {
            let l1_client = Arc::clone(&self.l1_client);
            let sync_interval = Duration::from_millis(self.config.sync_interval_ms);
            let stats = Arc::clone(&self.integration_stats);

            let sync_handle = tokio::spawn(async move {
                let mut interval = tokio::time::interval(sync_interval);
                
                loop {
                    interval.tick().await;
                    
                    if let Err(e) = l1_client.update_transaction_confirmations().await {
                        error!("Failed to update transaction confirmations: {}", e);
                    } else {
                        // Update sync time in stats
                        let mut stats_guard = stats.write().await;
                        stats_guard.last_sync_time = Some(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs()
                        );
                    }
                }
            });

            {
                let mut handle = self.sync_task_handle.write().await;
                *handle = Some(sync_handle);
            }
        }

        info!("ethrex L1 integration manager started successfully");
        Ok(())
    }

    /// Stop the integration manager
    pub async fn stop(&self) -> Result<(), RollupError> {
        info!("Stopping ethrex L1 integration manager");

        // Stop sync task
        if let Some(handle) = {
            let mut sync_handle = self.sync_task_handle.write().await;
            sync_handle.take()
        } {
            handle.abort();
        }

        // Stop finality task
        if let Some(handle) = {
            let mut finality_handle = self.finality_task_handle.write().await;
            finality_handle.take()
        } {
            handle.abort();
        }

        info!("ethrex L1 integration manager stopped");
        Ok(())
    }

    /// Submit proof commitment to L1 (Requirement 2.1, 2.2)
    pub async fn submit_proof_commitment(
        &self,
        request: ProofCommitmentRequest,
    ) -> Result<SubmissionResult, RollupError> {
        info!("Submitting proof commitment for batch {}", request.batch_id);

        // Acquire submission semaphore to limit concurrent submissions
        let _permit = self.submission_semaphore.acquire().await.unwrap();

        let start_time = Instant::now();

        // Submit to L1
        let tx_hash = self.l1_client.submit_proof_commitment(
            request.batch_id,
            &request.proof,
            request.state_root_before,
            request.state_root_after,
            request.order_book_merkle_root,
        ).await?;

        // Track transaction for finality if enabled
        if self.config.enable_finality_tracking {
            self.finality_tracker.track_transaction(
                tx_hash,
                TransactionType::ProofCommitment { batch_id: request.batch_id },
                Some(request.state_root_after),
            ).await?;
        }

        let submission_time = start_time.elapsed();

        // Update statistics
        {
            let mut stats = self.integration_stats.write().await;
            stats.total_proof_commitments += 1;
            stats.successful_submissions += 1;
            stats.average_submission_time_ms = 
                (stats.average_submission_time_ms * (stats.successful_submissions - 1) as f64 + 
                 submission_time.as_millis() as f64) / stats.successful_submissions as f64;
        }

        let result = SubmissionResult {
            tx_hash,
            submission_time: start_time,
            state_root: request.state_root_after,
            batch_id: Some(request.batch_id),
        };

        info!(
            "Successfully submitted proof commitment for batch {} with tx hash: {:?} (took {:?})",
            request.batch_id, tx_hash, submission_time
        );

        Ok(result)
    }

    /// Submit state root anchor to L1
    pub async fn submit_state_anchor(
        &self,
        request: StateAnchorRequest,
    ) -> Result<SubmissionResult, RollupError> {
        info!("Submitting state root anchor: {:?}", request.state_root);

        let _permit = self.submission_semaphore.acquire().await.unwrap();
        let start_time = Instant::now();

        let tx_hash = self.l1_client.submit_state_root_anchor(
            request.state_root,
            &request.proof,
        ).await?;

        // Track transaction for finality if enabled
        if self.config.enable_finality_tracking {
            self.finality_tracker.track_transaction(
                tx_hash,
                TransactionType::StateRootAnchor { state_root: request.state_root },
                Some(request.state_root),
            ).await?;
        }

        let submission_time = start_time.elapsed();

        // Update statistics
        {
            let mut stats = self.integration_stats.write().await;
            stats.total_state_anchors += 1;
            stats.successful_submissions += 1;
            stats.average_submission_time_ms = 
                (stats.average_submission_time_ms * (stats.successful_submissions - 1) as f64 + 
                 submission_time.as_millis() as f64) / stats.successful_submissions as f64;
        }

        let result = SubmissionResult {
            tx_hash,
            submission_time: start_time,
            state_root: request.state_root,
            batch_id: None,
        };

        info!(
            "Successfully submitted state root anchor with tx hash: {:?} (took {:?})",
            tx_hash, submission_time
        );

        Ok(result)
    }

    /// Get L1 commitment record for a state root (Requirement 2.6)
    pub async fn get_commitment_record(&self, state_root: StateRoot) -> Option<L1CommitmentRecord> {
        self.l1_client.get_commitment_record(state_root).await
    }

    /// Get L1 transaction hash for historical state commitment (Requirement 2.7)
    pub async fn get_l1_transaction_hash(&self, state_root: StateRoot) -> Option<TxHash> {
        self.l1_client.get_l1_transaction_hash(state_root).await
    }

    /// Get all commitment records for audit purposes
    pub async fn get_all_commitment_records(&self) -> HashMap<StateRoot, L1CommitmentRecord> {
        self.l1_client.get_all_commitment_records().await
    }

    /// Get integration statistics
    pub async fn get_integration_stats(&self) -> IntegrationStats {
        let mut stats = self.integration_stats.read().await.clone();
        
        // Update finality statistics
        let finality_stats = self.finality_tracker.get_finality_stats().await;
        stats.pending_transactions = finality_stats.tracked_count;
        stats.finalized_transactions = finality_stats.finalized_count;

        stats
    }

    /// Get finality statistics
    pub async fn get_finality_stats(&self) -> FinalityStats {
        self.finality_tracker.get_finality_stats().await
    }

    /// Handle reorganization event (Requirement 2.8)
    pub async fn handle_reorganization(&self, reorg_event: ReorgEvent) -> Result<(), RollupError> {
        info!("Handling reorganization event at block {}", reorg_event.reorg_block);

        // Delegate to L1 client
        self.l1_client.handle_reorganization(reorg_event).await?;

        // Update statistics
        {
            let mut stats = self.integration_stats.write().await;
            stats.reorganizations_handled += 1;
        }

        Ok(())
    }

    /// Start processing finality events
    async fn start_finality_event_processing(&self) -> Result<(), RollupError> {
        let finality_events_rx = {
            let mut rx_option = self.finality_events_rx.write().await;
            rx_option.take()
        };

        if let Some(mut rx) = finality_events_rx {
            let stats = Arc::clone(&self.integration_stats);
            
            tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        FinalityEvent::TransactionConfirmed { tx_hash, transaction_type, .. } => {
                            info!("Transaction {:?} ({:?}) confirmed", tx_hash, transaction_type);
                        }
                        FinalityEvent::TransactionFinalized { tx_hash, transaction_type, .. } => {
                            info!("Transaction {:?} ({:?}) finalized", tx_hash, transaction_type);
                        }
                        FinalityEvent::TransactionFailed { tx_hash, transaction_type, reason } => {
                            warn!("Transaction {:?} ({:?}) failed: {}", tx_hash, transaction_type, reason);
                            
                            // Update failure statistics
                            let mut stats_guard = stats.write().await;
                            stats_guard.failed_submissions += 1;
                        }
                        FinalityEvent::ReorganizationDetected { reorg_block, affected_transactions, .. } => {
                            warn!(
                                "Reorganization detected at block {} affecting {} transactions",
                                reorg_block, affected_transactions.len()
                            );
                        }
                        FinalityEvent::DeepReorganizationDetected { reorg_depth, reorg_block, affected_transactions } => {
                            error!(
                                "Deep reorganization detected (depth: {}) at block {} affecting {} transactions",
                                reorg_depth, reorg_block, affected_transactions.len()
                            );
                        }
                    }
                }
            });
        }

        Ok(())
    }

    /// Check if the integration is healthy
    pub async fn health_check(&self) -> Result<bool, RollupError> {
        // Check if we can get the current block number
        let current_block = self.l1_client.rpc_client.get_block_number().await?;
        
        // Check if finality tracker is working
        let finality_stats = self.finality_tracker.get_finality_stats().await;
        
        // Basic health checks
        let is_healthy = current_block > 0 && 
                        finality_stats.current_block_height.is_some();

        if is_healthy {
            debug!("ethrex integration health check passed");
        } else {
            warn!("ethrex integration health check failed");
        }

        Ok(is_healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::{ZkVMBackend, ProofMetadata};

    fn create_test_proof() -> ZkProof {
        ZkProof {
            backend: ZkVMBackend::SP1Local,
            proof_data: vec![1, 2, 3, 4],
            public_inputs: vec![5, 6, 7, 8],
            verification_key_hash: [9u8; 32],
            proof_metadata: ProofMetadata {
                proof_id: "test_proof".to_string(),
                generation_time: 1234567890,
                proof_size: 4,
                security_level: 128,
                circuit_size: 1000000,
            },
        }
    }

    #[tokio::test]
    async fn test_integration_manager_creation() {
        let config = EthrexIntegrationConfig::default();
        let manager = EthrexIntegrationManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_proof_commitment_request() {
        let config = EthrexIntegrationConfig::default();
        let manager = EthrexIntegrationManager::new(config).await.unwrap();

        let request = ProofCommitmentRequest {
            batch_id: 1,
            proof: create_test_proof(),
            state_root_before: [1u8; 32],
            state_root_after: [2u8; 32],
            order_book_merkle_root: [3u8; 32],
            priority: SubmissionPriority::Normal,
        };

        // This would fail in a real test without a running ethrex node
        // but demonstrates the interface
        let result = manager.submit_proof_commitment(request).await;
        // In a mock environment, this might succeed or fail depending on the mock setup
    }

    #[tokio::test]
    async fn test_integration_stats() {
        let config = EthrexIntegrationConfig::default();
        let manager = EthrexIntegrationManager::new(config).await.unwrap();

        let stats = manager.get_integration_stats().await;
        assert_eq!(stats.total_proof_commitments, 0);
        assert_eq!(stats.total_state_anchors, 0);
        assert_eq!(stats.successful_submissions, 0);
        assert_eq!(stats.failed_submissions, 0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = EthrexIntegrationConfig::default();
        let manager = EthrexIntegrationManager::new(config).await.unwrap();

        // This would require a mock that returns valid responses
        let health = manager.health_check().await;
        // The result depends on the mock implementation
    }
}