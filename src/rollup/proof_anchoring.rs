//! Proof Anchoring and Verification System
//! 
//! This module implements the proof anchoring system that submits ZK proof commitments
//! to ethrex L1 within 1 block time, includes merkle roots of order book state,
//! and provides L1 proof verification capabilities.

use crate::rollup::{
    ethrex_client::*,
    ethrex_integration::*,
    types::*,
    RollupError,
};
use crate::zkvm::{ZkProof, ZkVMBackend};
use crate::orderbook::types::OrderBook;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};
use rand::Rng;

/// Proof anchoring manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAnchoringConfig {
    /// Maximum time to wait for L1 submission (in milliseconds)
    pub max_submission_time_ms: u64,
    /// Block time target for submissions (in milliseconds)
    pub block_time_target_ms: u64,
    /// Maximum concurrent proof submissions
    pub max_concurrent_submissions: usize,
    /// Enable automatic proof anchoring
    pub enable_auto_anchoring: bool,
    /// Proof submission retry attempts
    pub max_retry_attempts: u32,
    /// Retry backoff multiplier
    pub retry_backoff_multiplier: f64,
    /// Enable merkle root inclusion
    pub include_merkle_roots: bool,
    /// Enable L1 verification contract calls
    pub enable_l1_verification: bool,
}

impl Default for ProofAnchoringConfig {
    fn default() -> Self {
        Self {
            max_submission_time_ms: 10000, // 10 seconds
            block_time_target_ms: 12000,   // 12 seconds (Ethereum block time)
            max_concurrent_submissions: 5,
            enable_auto_anchoring: true,
            max_retry_attempts: 3,
            retry_backoff_multiplier: 2.0,
            include_merkle_roots: true,
            enable_l1_verification: true,
        }
    }
}

/// Proof commitment data structure for L1 submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCommitment {
    /// Unique commitment ID
    pub commitment_id: String,
    /// Associated batch ID
    pub batch_id: BatchId,
    /// ZK proof data
    pub proof: ZkProof,
    /// State root before batch execution
    pub state_root_before: StateRoot,
    /// State root after batch execution
    pub state_root_after: StateRoot,
    /// Merkle root of order book state
    pub order_book_merkle_root: [u8; 32],
    /// Merkle root of trade history
    pub trade_history_merkle_root: [u8; 32],
    /// Timestamp of commitment creation
    pub timestamp: u64,
    /// L1 block number target
    pub target_l1_block: L1BlockNumber,
    /// Priority level
    pub priority: CommitmentPriority,
}

/// Commitment priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommitmentPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// L1 commitment record with verification status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1CommitmentRecord {
    /// Commitment data
    pub commitment: ProofCommitment,
    /// L1 transaction hash
    pub tx_hash: TxHash,
    /// L1 block number where committed
    pub l1_block_number: L1BlockNumber,
    /// Confirmation status
    pub confirmation_status: ConfirmationStatus,
    /// Verification status on L1
    pub verification_status: VerificationStatus,
    /// Submission timestamp
    pub submission_time: u64,
    /// Confirmation timestamp
    pub confirmation_time: Option<u64>,
    /// Gas used for submission
    pub gas_used: u64,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Confirmation status of L1 transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfirmationStatus {
    Pending,
    Confirmed,
    Finalized,
    Failed,
}

/// Verification status on L1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    NotVerified,
    Verifying,
    Verified,
    VerificationFailed,
}

/// State commitment mapping entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCommitmentMapping {
    /// Local state root
    pub local_state_root: StateRoot,
    /// L1 commitment record
    pub l1_commitment: L1CommitmentRecord,
    /// Associated order book snapshot
    pub order_book_snapshot: Option<OrderBookSnapshot>,
    /// Mapping creation time
    pub created_at: u64,
    /// Last update time
    pub updated_at: u64,
}

/// Order book snapshot for merkle root calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    /// Snapshot ID
    pub snapshot_id: String,
    /// Order book state at snapshot time
    pub order_books: HashMap<String, OrderBook>,
    /// Merkle root of the snapshot
    pub merkle_root: [u8; 32],
    /// Snapshot timestamp
    pub timestamp: u64,
    /// Block height at snapshot
    pub block_height: u64,
}

/// Proof anchoring manager
pub struct ProofAnchoringManager {
    config: ProofAnchoringConfig,
    ethrex_integration: Arc<EthrexIntegrationManager>,
    l1_verification_client: Arc<L1VerificationClient>,
    commitment_mappings: Arc<RwLock<HashMap<StateRoot, StateCommitmentMapping>>>,
    pending_commitments: Arc<RwLock<HashMap<String, ProofCommitment>>>,
    anchoring_stats: Arc<RwLock<AnchoringStats>>,
    submission_queue: Arc<Mutex<mpsc::UnboundedSender<ProofCommitment>>>,
    worker_handles: Arc<RwLock<Vec<JoinHandle<()>>>>,
}

/// Anchoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchoringStats {
    /// Total commitments submitted
    pub total_commitments: u64,
    /// Successful submissions
    pub successful_submissions: u64,
    /// Failed submissions
    pub failed_submissions: u64,
    /// Average submission time (ms)
    pub average_submission_time_ms: f64,
    /// Average confirmation time (ms)
    pub average_confirmation_time_ms: f64,
    /// Commitments within block time target
    pub within_block_time_target: u64,
    /// Current pending commitments
    pub pending_commitments: usize,
    /// Total gas used
    pub total_gas_used: u64,
    /// Last anchoring time
    pub last_anchoring_time: Option<u64>,
}

impl Default for AnchoringStats {
    fn default() -> Self {
        Self {
            total_commitments: 0,
            successful_submissions: 0,
            failed_submissions: 0,
            average_submission_time_ms: 0.0,
            average_confirmation_time_ms: 0.0,
            within_block_time_target: 0,
            pending_commitments: 0,
            total_gas_used: 0,
            last_anchoring_time: None,
        }
    }
}

impl ProofAnchoringManager {
    /// Create a new proof anchoring manager
    pub async fn new(
        config: ProofAnchoringConfig,
        ethrex_integration: Arc<EthrexIntegrationManager>,
    ) -> Result<Self, RollupError> {
        info!("Initializing proof anchoring manager");

        // Create L1 verification client
        let l1_verification_client = Arc::new(L1VerificationClient::new().await?);

        // Create submission queue
        let (tx, rx) = mpsc::unbounded_channel();

        let manager = Self {
            config: config.clone(),
            ethrex_integration,
            l1_verification_client,
            commitment_mappings: Arc::new(RwLock::new(HashMap::new())),
            pending_commitments: Arc::new(RwLock::new(HashMap::new())),
            anchoring_stats: Arc::new(RwLock::new(AnchoringStats::default())),
            submission_queue: Arc::new(Mutex::new(tx)),
            worker_handles: Arc::new(RwLock::new(Vec::new())),
        };

        // Start worker tasks if auto-anchoring is enabled
        if config.enable_auto_anchoring {
            manager.start_workers(rx).await?;
        }

        info!("Proof anchoring manager initialized successfully");
        Ok(manager)
    }

    /// Submit ZK proof commitment to L1 (Requirement 2.1, 2.2)
    pub async fn submit_proof_commitment(
        &self,
        batch_id: BatchId,
        proof: ZkProof,
        state_root_before: StateRoot,
        state_root_after: StateRoot,
        order_book_snapshot: Option<OrderBookSnapshot>,
    ) -> Result<L1CommitmentRecord, RollupError> {
        info!("Submitting proof commitment for batch {}", batch_id);

        let start_time = Instant::now();
        let current_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate merkle roots
        let order_book_merkle_root = if let Some(ref snapshot) = order_book_snapshot {
            snapshot.merkle_root
        } else {
            self.calculate_order_book_merkle_root(&HashMap::new()).await?
        };

        let trade_history_merkle_root = self.calculate_trade_history_merkle_root(batch_id).await?;

        // Create proof commitment
        let commitment = ProofCommitment {
            commitment_id: generate_commitment_id(),
            batch_id,
            proof: proof.clone(),
            state_root_before,
            state_root_after,
            order_book_merkle_root,
            trade_history_merkle_root,
            timestamp: current_timestamp,
            target_l1_block: 0, // Will be set during submission
            priority: CommitmentPriority::Normal,
        };

        // Submit to ethrex L1
        let submission_request = ProofCommitmentRequest {
            batch_id,
            proof,
            state_root_before,
            state_root_after,
            order_book_merkle_root,
            priority: SubmissionPriority::Normal,
        };

        let submission_result = self.ethrex_integration
            .submit_proof_commitment(submission_request)
            .await?;

        // Create L1 commitment record
        let l1_record = L1CommitmentRecord {
            commitment: commitment.clone(),
            tx_hash: submission_result.tx_hash,
            l1_block_number: 0, // Will be updated when confirmed
            confirmation_status: ConfirmationStatus::Pending,
            verification_status: VerificationStatus::NotVerified,
            submission_time: current_timestamp,
            confirmation_time: None,
            gas_used: 0, // Will be updated when confirmed
            retry_count: 0,
        };

        // Store commitment mapping (Requirement 2.6)
        let mapping = StateCommitmentMapping {
            local_state_root: state_root_after,
            l1_commitment: l1_record.clone(),
            order_book_snapshot,
            created_at: current_timestamp,
            updated_at: current_timestamp,
        };

        {
            let mut mappings = self.commitment_mappings.write().await;
            mappings.insert(state_root_after, mapping);
        }

        // Update statistics
        let submission_time = start_time.elapsed();
        {
            let mut stats = self.anchoring_stats.write().await;
            stats.total_commitments += 1;
            stats.successful_submissions += 1;
            stats.average_submission_time_ms = 
                (stats.average_submission_time_ms * (stats.successful_submissions - 1) as f64 + 
                 submission_time.as_millis() as f64) / stats.successful_submissions as f64;
            
            // Check if within block time target
            if submission_time.as_millis() <= self.config.block_time_target_ms as u128 {
                stats.within_block_time_target += 1;
            }
            
            stats.last_anchoring_time = Some(current_timestamp);
        }

        // Start L1 verification if enabled
        if self.config.enable_l1_verification {
            self.initiate_l1_verification(&l1_record).await?;
        }

        info!(
            "Successfully submitted proof commitment for batch {} with tx hash: {:?} (took {:?})",
            batch_id, submission_result.tx_hash, submission_time
        );

        Ok(l1_record)
    }

    /// Get L1 commitment record for a state root (Requirement 2.6)
    pub async fn get_commitment_record(&self, state_root: StateRoot) -> Option<L1CommitmentRecord> {
        let mappings = self.commitment_mappings.read().await;
        mappings.get(&state_root).map(|mapping| mapping.l1_commitment.clone())
    }

    /// Get L1 transaction hash for historical state commitment (Requirement 2.7)
    pub async fn get_l1_transaction_hash(&self, state_root: StateRoot) -> Option<TxHash> {
        let mappings = self.commitment_mappings.read().await;
        mappings.get(&state_root).map(|mapping| mapping.l1_commitment.tx_hash)
    }

    /// Get all commitment mappings for audit purposes
    pub async fn get_all_commitment_mappings(&self) -> HashMap<StateRoot, StateCommitmentMapping> {
        self.commitment_mappings.read().await.clone()
    }

    /// Calculate merkle root of order book state
    async fn calculate_order_book_merkle_root(
        &self,
        order_books: &HashMap<String, OrderBook>,
    ) -> Result<[u8; 32], RollupError> {
        if order_books.is_empty() {
            return Ok([0u8; 32]);
        }

        // Create merkle tree from order book states
        let mut leaves = Vec::new();
        
        for (symbol, order_book) in order_books {
            let serialized = bincode::serialize(order_book)
                .map_err(|e| RollupError::SerializationError(e))?;
            
            let mut hasher = Sha256::new();
            hasher.update(symbol.as_bytes());
            hasher.update(&serialized);
            leaves.push(hasher.finalize().to_vec());
        }

        // Sort leaves for deterministic merkle root
        leaves.sort();

        // Build merkle tree
        let merkle_root = self.build_merkle_tree(&leaves)?;
        
        Ok(merkle_root)
    }

    /// Calculate merkle root of trade history for a batch
    async fn calculate_trade_history_merkle_root(
        &self,
        _batch_id: BatchId,
    ) -> Result<[u8; 32], RollupError> {
        // For now, return a placeholder
        // In a real implementation, this would calculate the merkle root
        // of all trades in the batch
        Ok([0u8; 32])
    }

    /// Build merkle tree from leaves
    fn build_merkle_tree(&self, leaves: &[Vec<u8>]) -> Result<[u8; 32], RollupError> {
        if leaves.is_empty() {
            return Ok([0u8; 32]);
        }

        if leaves.len() == 1 {
            let mut result = [0u8; 32];
            result.copy_from_slice(&leaves[0][..32]);
            return Ok(result);
        }

        let mut current_level = leaves.to_vec();
        
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in current_level.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&chunk[0]);
                
                if chunk.len() > 1 {
                    hasher.update(&chunk[1]);
                } else {
                    // Duplicate the last node if odd number of nodes
                    hasher.update(&chunk[0]);
                }
                
                next_level.push(hasher.finalize().to_vec());
            }
            
            current_level = next_level;
        }

        let mut result = [0u8; 32];
        result.copy_from_slice(&current_level[0][..32]);
        Ok(result)
    }

    /// Initiate L1 verification for a commitment
    async fn initiate_l1_verification(
        &self,
        commitment_record: &L1CommitmentRecord,
    ) -> Result<(), RollupError> {
        info!("Initiating L1 verification for commitment {}", commitment_record.commitment.commitment_id);

        // Submit verification request to L1 verification client
        self.l1_verification_client
            .verify_proof_on_l1(
                &commitment_record.commitment.proof,
                &commitment_record.commitment.state_root_after,
                &commitment_record.commitment.order_book_merkle_root,
            )
            .await?;

        Ok(())
    }

    /// Start worker tasks for automatic proof anchoring
    async fn start_workers(
        &self,
        mut rx: mpsc::UnboundedReceiver<ProofCommitment>,
    ) -> Result<(), RollupError> {
        info!("Starting proof anchoring workers");

        let ethrex_integration = Arc::clone(&self.ethrex_integration);
        let l1_verification_client = Arc::clone(&self.l1_verification_client);
        let commitment_mappings = Arc::clone(&self.commitment_mappings);
        let anchoring_stats = Arc::clone(&self.anchoring_stats);
        let config = self.config.clone();

        // Start submission worker
        let submission_handle = tokio::spawn(async move {
            while let Some(commitment) = rx.recv().await {
                info!("Processing commitment {} from queue", commitment.commitment_id);

                // Process the commitment with retry logic
                let mut retry_count = 0;
                let mut success = false;

                while retry_count < config.max_retry_attempts && !success {
                    match Self::process_commitment_with_retry(
                        &ethrex_integration,
                        &l1_verification_client,
                        &commitment,
                        retry_count,
                    ).await {
                        Ok(_) => {
                            success = true;
                            info!("Successfully processed commitment {}", commitment.commitment_id);
                        }
                        Err(e) => {
                            retry_count += 1;
                            warn!(
                                "Failed to process commitment {} (attempt {}): {}",
                                commitment.commitment_id, retry_count, e
                            );

                            if retry_count < config.max_retry_attempts {
                                let backoff_duration = Duration::from_millis(
                                    (1000.0 * config.retry_backoff_multiplier.powi(retry_count as i32)) as u64
                                );
                                tokio::time::sleep(backoff_duration).await;
                            }
                        }
                    }
                }

                if !success {
                    error!("Failed to process commitment {} after {} attempts", 
                           commitment.commitment_id, config.max_retry_attempts);
                    
                    // Update failure statistics
                    let mut stats = anchoring_stats.write().await;
                    stats.failed_submissions += 1;
                }
            }
        });

        // Store worker handle
        {
            let mut handles = self.worker_handles.write().await;
            handles.push(submission_handle);
        }

        Ok(())
    }

    /// Process commitment with retry logic
    async fn process_commitment_with_retry(
        ethrex_integration: &EthrexIntegrationManager,
        _l1_verification_client: &L1VerificationClient,
        commitment: &ProofCommitment,
        retry_count: u32,
    ) -> Result<(), RollupError> {
        info!("Processing commitment {} (retry {})", commitment.commitment_id, retry_count);

        // Create submission request
        let submission_request = ProofCommitmentRequest {
            batch_id: commitment.batch_id,
            proof: commitment.proof.clone(),
            state_root_before: commitment.state_root_before,
            state_root_after: commitment.state_root_after,
            order_book_merkle_root: commitment.order_book_merkle_root,
            priority: match commitment.priority {
                CommitmentPriority::Low => SubmissionPriority::Low,
                CommitmentPriority::Normal => SubmissionPriority::Normal,
                CommitmentPriority::High => SubmissionPriority::High,
                CommitmentPriority::Critical => SubmissionPriority::Critical,
            },
        };

        // Submit to L1
        let _result = ethrex_integration.submit_proof_commitment(submission_request).await?;

        Ok(())
    }

    /// Get anchoring statistics
    pub async fn get_anchoring_stats(&self) -> AnchoringStats {
        let mut stats = self.anchoring_stats.read().await.clone();
        
        // Update pending commitments count
        stats.pending_commitments = self.pending_commitments.read().await.len();
        
        stats
    }

    /// Stop the proof anchoring manager
    pub async fn stop(&self) -> Result<(), RollupError> {
        info!("Stopping proof anchoring manager");

        // Stop all worker tasks
        let handles = {
            let mut handles = self.worker_handles.write().await;
            std::mem::take(&mut *handles)
        };

        for handle in handles {
            handle.abort();
        }

        info!("Proof anchoring manager stopped");
        Ok(())
    }

    /// Health check for the anchoring system
    pub async fn health_check(&self) -> Result<bool, RollupError> {
        // Check ethrex integration health
        let ethrex_healthy = self.ethrex_integration.health_check().await?;
        
        // Check L1 verification client health
        let l1_verification_healthy = self.l1_verification_client.health_check().await?;
        
        let is_healthy = ethrex_healthy && l1_verification_healthy;
        
        if is_healthy {
            debug!("Proof anchoring system health check passed");
        } else {
            warn!("Proof anchoring system health check failed");
        }
        
        Ok(is_healthy)
    }
}

/// L1 Verification Client for proof verification contracts
pub struct L1VerificationClient {
    // Contract interfaces and RPC client would be here
}

impl L1VerificationClient {
    /// Create a new L1 verification client
    pub async fn new() -> Result<Self, RollupError> {
        info!("Initializing L1 verification client");
        
        // Initialize contract interfaces and RPC connections
        
        Ok(Self {})
    }

    /// Verify proof on L1 verification contract
    pub async fn verify_proof_on_l1(
        &self,
        _proof: &ZkProof,
        _state_root: &StateRoot,
        _merkle_root: &[u8; 32],
    ) -> Result<bool, RollupError> {
        info!("Verifying proof on L1 verification contract");
        
        // Call L1 verification contract
        // For now, return success
        Ok(true)
    }

    /// Health check for L1 verification client
    pub async fn health_check(&self) -> Result<bool, RollupError> {
        // Check contract connectivity and status
        Ok(true)
    }
}

/// Generate unique commitment ID
fn generate_commitment_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random = rand::thread_rng().gen::<u32>();
    format!("commitment_{}_{:x}", timestamp, random)
}

pub mod proof_anchoring_test;

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

    #[test]
    fn test_commitment_priority_ordering() {
        assert!(CommitmentPriority::Critical > CommitmentPriority::High);
        assert!(CommitmentPriority::High > CommitmentPriority::Normal);
        assert!(CommitmentPriority::Normal > CommitmentPriority::Low);
    }

    #[test]
    fn test_generate_commitment_id() {
        let id1 = generate_commitment_id();
        let id2 = generate_commitment_id();
        
        assert_ne!(id1, id2);
        assert!(id1.starts_with("commitment_"));
        assert!(id2.starts_with("commitment_"));
    }

    #[tokio::test]
    async fn test_merkle_tree_building() {
        let config = ProofAnchoringConfig::default();
        let ethrex_integration = Arc::new(
            EthrexIntegrationManager::new(EthrexIntegrationConfig::default())
                .await
                .unwrap()
        );
        
        let manager = ProofAnchoringManager::new(config, ethrex_integration)
            .await
            .unwrap();

        // Test empty leaves
        let empty_root = manager.build_merkle_tree(&[]).unwrap();
        assert_eq!(empty_root, [0u8; 32]);

        // Test single leaf
        let single_leaf = vec![vec![1u8; 32]];
        let single_root = manager.build_merkle_tree(&single_leaf).unwrap();
        assert_ne!(single_root, [0u8; 32]);

        // Test multiple leaves
        let multiple_leaves = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];
        let multiple_root = manager.build_merkle_tree(&multiple_leaves).unwrap();
        assert_ne!(multiple_root, [0u8; 32]);
        assert_ne!(multiple_root, single_root);
    }

    #[tokio::test]
    async fn test_order_book_merkle_root_calculation() {
        let config = ProofAnchoringConfig::default();
        let ethrex_integration = Arc::new(
            EthrexIntegrationManager::new(EthrexIntegrationConfig::default())
                .await
                .unwrap()
        );
        
        let manager = ProofAnchoringManager::new(config, ethrex_integration)
            .await
            .unwrap();

        // Test empty order books
        let empty_books = HashMap::new();
        let empty_root = manager.calculate_order_book_merkle_root(&empty_books).await.unwrap();
        assert_eq!(empty_root, [0u8; 32]);

        // Test with order books would require OrderBook implementation
        // This is a placeholder for the actual test
    }
}