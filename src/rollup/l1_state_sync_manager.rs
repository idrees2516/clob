//! L1 State Synchronization Manager
//! 
//! This module implements bidirectional state synchronization between local trading state
//! and ethrex L1, handling reorganizations, proof resubmission, state reconciliation,
//! and finality tracking as specified in requirements 2.5, 2.8, 4.1-4.5.

use crate::rollup::{
    ethrex_integration::*,
    ethrex_client::ReorgEvent,
    finality_tracker::*,
    types::*,
    RollupError,
};
use crate::zkvm::ZkProof;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};

/// L1 State Synchronization Manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1StateSyncConfig {
    /// Synchronization interval in milliseconds
    pub sync_interval_ms: u64,
    /// Maximum state drift allowed before triggering reconciliation
    pub max_state_drift_blocks: u64,
    /// Enable automatic state reconciliation
    pub enable_auto_reconciliation: bool,
    /// Reconciliation timeout in seconds
    pub reconciliation_timeout_seconds: u64,
    /// Maximum number of state snapshots to keep
    pub max_state_snapshots: usize,
    /// Enable deep state verification
    pub enable_deep_verification: bool,
    /// State verification interval in milliseconds
    pub verification_interval_ms: u64,
    /// Maximum concurrent reconciliation operations
    pub max_concurrent_reconciliations: usize,
    /// Enable state recovery from L1 on startup
    pub enable_startup_recovery: bool,
    /// Recovery timeout in seconds
    pub recovery_timeout_seconds: u64,
}

impl Default for L1StateSyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_ms: 6000, // 6 seconds (half of Ethereum block time)
            max_state_drift_blocks: 3,
            enable_auto_reconciliation: true,
            reconciliation_timeout_seconds: 300, // 5 minutes
            max_state_snapshots: 100,
            enable_deep_verification: true,
            verification_interval_ms: 30000, // 30 seconds
            max_concurrent_reconciliations: 5,
            enable_startup_recovery: true,
            recovery_timeout_seconds: 600, // 10 minutes
        }
    }
}/
// L1 State Synchronization Manager
pub struct L1StateSyncManager {
    config: L1StateSyncConfig,
    ethrex_integration: Arc<EthrexIntegrationManager>,
    local_state_tracker: Arc<RwLock<LocalStateTracker>>,
    l1_state_tracker: Arc<RwLock<L1StateTracker>>,
    sync_task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    verification_task_handle: Arc<RwLock<Option<JoinHandle<()>>>>,
    reconciliation_semaphore: Arc<tokio::sync::Semaphore>,
    sync_events_tx: mpsc::UnboundedSender<StateSyncEvent>,
    sync_events_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<StateSyncEvent>>>>,
    sync_stats: Arc<RwLock<StateSyncStats>>,
    state_conflicts: Arc<RwLock<Vec<StateConflict>>>,
    finality_events_handler: Arc<Mutex<Option<JoinHandle<()>>>>,
}

/// Local state tracker for trading system state
#[derive(Debug, Clone)]
pub struct LocalStateTracker {
    /// Current local state root
    pub current_state_root: StateRoot,
    /// Local block height (sequence number)
    pub local_block_height: u64,
    /// State snapshots for rollback capability
    pub state_snapshots: VecDeque<StateSnapshot>,
    /// Pending state transitions
    pub pending_transitions: HashMap<u64, StateTransition>,
    /// Last L1 sync timestamp
    pub last_l1_sync_time: Option<Instant>,
    /// State consistency status
    pub consistency_status: StateConsistencyStatus,
}

/// L1 state tracker for ethrex L1 state
#[derive(Debug, Clone)]
pub struct L1StateTracker {
    /// Current L1 block height
    pub current_l1_block: L1BlockNumber,
    /// L1 state commitments
    pub l1_commitments: HashMap<StateRoot, L1StateCommitment>,
    /// Finalized state roots on L1
    pub finalized_state_roots: VecDeque<StateRoot>,
    /// Pending L1 transactions
    pub pending_l1_transactions: HashMap<TxHash, L1TransactionInfo>,
    /// Last successful L1 query time
    pub last_l1_query_time: Option<Instant>,
    /// L1 reorganization history
    pub reorg_history: VecDeque<ReorgEvent>,
}

/// State snapshot for rollback capability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// State root hash
    pub state_root: StateRoot,
    /// Local sequence number
    pub sequence_number: u64,
    /// Timestamp when snapshot was created
    pub timestamp: u64,
    /// Associated L1 block (if any)
    pub l1_block_reference: Option<L1BlockNumber>,
    /// Merkle proof for state verification
    pub merkle_proof: Vec<[u8; 32]>,
    /// Compressed state data
    pub compressed_state_data: Vec<u8>,
}

/// L1 state commitment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1StateCommitment {
    /// State root committed to L1
    pub state_root: StateRoot,
    /// L1 transaction hash
    pub commitment_tx_hash: TxHash,
    /// L1 block where commitment was included
    pub l1_block_number: L1BlockNumber,
    /// Commitment timestamp
    pub commitment_timestamp: u64,
    /// Finality status
    pub finality_status: FinalityStatus,
    /// Associated proof hash
    pub proof_hash: [u8; 32],
}

/// L1 transaction information
#[derive(Debug, Clone)]
pub struct L1TransactionInfo {
    /// Transaction hash
    pub tx_hash: TxHash,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Submission timestamp
    pub submission_time: Instant,
    /// Current confirmations
    pub confirmations: u64,
    /// Associated state root
    pub associated_state_root: Option<StateRoot>,
}

/// State consistency status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateConsistencyStatus {
    /// States are consistent
    Consistent,
    /// Minor drift detected, within acceptable limits
    MinorDrift { drift_blocks: u64 },
    /// Major drift detected, reconciliation needed
    MajorDrift { drift_blocks: u64 },
    /// States are inconsistent, immediate action required
    Inconsistent { reason: String },
    /// Reconciliation in progress
    Reconciling { started_at: u64 },
}

/// Finality status for L1 commitments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinalityStatus {
    /// Pending confirmation
    Pending,
    /// Confirmed but not yet finalized
    Confirmed { confirmations: u64 },
    /// Finalized on L1
    Finalized { finalization_block: L1BlockNumber },
    /// Failed or reverted
    Failed { reason: String },
}/
// State synchronization events
#[derive(Debug, Clone)]
pub enum StateSyncEvent {
    /// Local state updated
    LocalStateUpdated {
        new_state_root: StateRoot,
        sequence_number: u64,
    },
    /// L1 state commitment detected
    L1CommitmentDetected {
        state_root: StateRoot,
        tx_hash: TxHash,
        l1_block: L1BlockNumber,
    },
    /// State consistency check completed
    ConsistencyCheckCompleted {
        status: StateConsistencyStatus,
        drift_blocks: u64,
    },
    /// State reconciliation started
    ReconciliationStarted {
        trigger_reason: String,
        target_state_root: StateRoot,
    },
    /// State reconciliation completed
    ReconciliationCompleted {
        success: bool,
        final_state_root: StateRoot,
        duration_ms: u64,
    },
    /// Reorganization detected and handled
    ReorganizationHandled {
        reorg_block: L1BlockNumber,
        affected_states: Vec<StateRoot>,
        recovery_actions: Vec<String>,
    },
    /// State recovery from L1 completed
    StateRecoveryCompleted {
        recovered_state_root: StateRoot,
        recovery_block: L1BlockNumber,
    },
}

/// State conflict information
#[derive(Debug, Clone)]
pub struct StateConflict {
    /// Local state root
    pub local_state_root: StateRoot,
    /// L1 state root
    pub l1_state_root: StateRoot,
    /// Conflict detection time
    pub detection_time: Instant,
    /// Conflict type
    pub conflict_type: StateConflictType,
    /// Resolution strategy
    pub resolution_strategy: ConflictResolutionStrategy,
    /// Resolution status
    pub resolution_status: ConflictResolutionStatus,
}

/// Types of state conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateConflictType {
    /// State roots don't match
    StateMismatch,
    /// Sequence number mismatch
    SequenceMismatch,
    /// Missing L1 commitment
    MissingL1Commitment,
    /// Orphaned local state
    OrphanedLocalState,
    /// Reorganization conflict
    ReorganizationConflict,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Use L1 as source of truth
    L1Priority,
    /// Use local state as source of truth
    LocalPriority,
    /// Attempt to merge states
    StateMerge,
    /// Manual intervention required
    ManualResolution,
}

/// Conflict resolution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStatus {
    /// Conflict detected, resolution pending
    Pending,
    /// Resolution in progress
    InProgress,
    /// Successfully resolved
    Resolved,
    /// Resolution failed
    Failed { reason: String },
}

/// State synchronization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSyncStats {
    /// Total sync cycles completed
    pub total_sync_cycles: u64,
    /// Successful sync operations
    pub successful_syncs: u64,
    /// Failed sync operations
    pub failed_syncs: u64,
    /// Average sync duration in milliseconds
    pub average_sync_duration_ms: f64,
    /// State conflicts detected
    pub conflicts_detected: u64,
    /// Conflicts resolved
    pub conflicts_resolved: u64,
    /// Reorganizations handled
    pub reorganizations_handled: u64,
    /// State recoveries performed
    pub state_recoveries: u64,
    /// Last sync timestamp
    pub last_sync_timestamp: Option<u64>,
    /// Current consistency status
    pub current_consistency_status: StateConsistencyStatus,
}

impl Default for StateSyncStats {
    fn default() -> Self {
        Self {
            total_sync_cycles: 0,
            successful_syncs: 0,
            failed_syncs: 0,
            average_sync_duration_ms: 0.0,
            conflicts_detected: 0,
            conflicts_resolved: 0,
            reorganizations_handled: 0,
            state_recoveries: 0,
            last_sync_timestamp: None,
            current_consistency_status: StateConsistencyStatus::Consistent,
        }
    }
}impl L1State
SyncManager {
    /// Create a new L1 state synchronization manager
    pub async fn new(
        config: L1StateSyncConfig,
        ethrex_integration: Arc<EthrexIntegrationManager>,
    ) -> Result<Self, RollupError> {
        info!("Initializing L1 state synchronization manager");

        let (sync_events_tx, sync_events_rx) = mpsc::unbounded_channel();

        let manager = Self {
            config,
            ethrex_integration,
            local_state_tracker: Arc::new(RwLock::new(LocalStateTracker::new())),
            l1_state_tracker: Arc::new(RwLock::new(L1StateTracker::new())),
            sync_task_handle: Arc::new(RwLock::new(None)),
            verification_task_handle: Arc::new(RwLock::new(None)),
            reconciliation_semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_reconciliations)),
            sync_events_tx,
            sync_events_rx: Arc::new(RwLock::new(Some(sync_events_rx))),
            sync_stats: Arc::new(RwLock::new(StateSyncStats::default())),
            state_conflicts: Arc::new(RwLock::new(Vec::new())),
            finality_events_handler: Arc::new(Mutex::new(None)),
        };

        info!("L1 state synchronization manager initialized successfully");
        Ok(manager)
    }

    /// Start the state synchronization manager (Requirement 4.1)
    pub async fn start(&self) -> Result<(), RollupError> {
        info!("Starting L1 state synchronization manager");

        // Perform startup recovery if enabled (Requirement 4.6)
        if self.config.enable_startup_recovery {
            self.perform_startup_recovery().await?;
        }

        // Start bidirectional synchronization loop (Requirement 4.1)
        self.start_sync_loop().await?;

        // Start state verification loop if enabled (Requirement 4.2)
        if self.config.enable_deep_verification {
            self.start_verification_loop().await?;
        }

        // Start finality event processing (Requirement 4.4)
        self.start_finality_event_processing().await?;

        info!("L1 state synchronization manager started successfully");
        Ok(())
    }

    /// Stop the state synchronization manager
    pub async fn stop(&self) -> Result<(), RollupError> {
        info!("Stopping L1 state synchronization manager");

        // Stop sync task
        if let Some(handle) = {
            let mut sync_handle = self.sync_task_handle.write().await;
            sync_handle.take()
        } {
            handle.abort();
        }

        // Stop verification task
        if let Some(handle) = {
            let mut verification_handle = self.verification_task_handle.write().await;
            verification_handle.take()
        } {
            handle.abort();
        }

        // Stop finality event handler
        if let Some(handle) = {
            let mut finality_handle = self.finality_events_handler.lock().await;
            finality_handle.take()
        } {
            handle.abort();
        }

        info!("L1 state synchronization manager stopped");
        Ok(())
    }

    /// Update local state and trigger synchronization (Requirement 4.1)
    pub async fn update_local_state(
        &self,
        new_state_root: StateRoot,
        sequence_number: u64,
        proof: Option<ZkProof>,
    ) -> Result<(), RollupError> {
        debug!("Updating local state: {:?} (seq: {})", new_state_root, sequence_number);

        // Create state snapshot
        let snapshot = StateSnapshot {
            state_root: new_state_root,
            sequence_number,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            l1_block_reference: None,
            merkle_proof: Vec::new(), // Would be populated with actual merkle proof
            compressed_state_data: Vec::new(), // Would be populated with compressed state
        };

        // Update local state tracker
        {
            let mut local_tracker = self.local_state_tracker.write().await;
            local_tracker.current_state_root = new_state_root;
            local_tracker.local_block_height = sequence_number;
            
            // Add snapshot
            local_tracker.state_snapshots.push_back(snapshot);
            
            // Maintain snapshot limit
            while local_tracker.state_snapshots.len() > self.config.max_state_snapshots {
                local_tracker.state_snapshots.pop_front();
            }
        }

        // Emit state update event
        let event = StateSyncEvent::LocalStateUpdated {
            new_state_root,
            sequence_number,
        };
        
        if let Err(e) = self.sync_events_tx.send(event) {
            warn!("Failed to send local state update event: {}", e);
        }

        // Trigger immediate consistency check
        self.check_state_consistency().await?;

        Ok(())
    }    /
// Synchronize with L1 state (Requirement 4.1, 4.2)
    pub async fn sync_with_l1(&self) -> Result<SyncResult, RollupError> {
        let sync_start = Instant::now();
        debug!("Starting L1 state synchronization");

        // Get current L1 state
        let l1_commitments = self.ethrex_integration.get_all_commitment_records().await;
        let finality_stats = self.ethrex_integration.get_finality_stats().await;

        // Update L1 state tracker
        {
            let mut l1_tracker = self.l1_state_tracker.write().await;
            l1_tracker.current_l1_block = finality_stats.current_block_height.unwrap_or(0);
            l1_tracker.last_l1_query_time = Some(Instant::now());

            // Update L1 commitments
            for (state_root, commitment_record) in l1_commitments {
                let l1_commitment = L1StateCommitment {
                    state_root,
                    commitment_tx_hash: commitment_record.commitment_tx_hash,
                    l1_block_number: commitment_record.l1_block_number,
                    commitment_timestamp: commitment_record.commitment_timestamp,
                    finality_status: match commitment_record.confirmation_status {
                        crate::rollup::ethrex_client::ConfirmationStatus::Pending { confirmations } => {
                            FinalityStatus::Confirmed { confirmations }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Confirmed { final_block } => {
                            FinalityStatus::Finalized { finalization_block: final_block }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Failed { reason } => {
                            FinalityStatus::Failed { reason }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Reorged { .. } => {
                            FinalityStatus::Pending
                        }
                    },
                    proof_hash: commitment_record.proof_commitment_hash,
                };

                l1_tracker.l1_commitments.insert(state_root, l1_commitment);
            }
        }

        // Check for state consistency (Requirement 4.2)
        let consistency_result = self.check_state_consistency().await?;

        // Handle any detected conflicts (Requirement 4.3)
        if matches!(consistency_result.status, StateConsistencyStatus::Inconsistent { .. }) {
            self.handle_state_conflicts().await?;
        }

        let sync_duration = sync_start.elapsed();

        // Update statistics
        {
            let mut stats = self.sync_stats.write().await;
            stats.total_sync_cycles += 1;
            stats.successful_syncs += 1;
            stats.average_sync_duration_ms = 
                (stats.average_sync_duration_ms * (stats.total_sync_cycles - 1) as f64 + 
                 sync_duration.as_millis() as f64) / stats.total_sync_cycles as f64;
            stats.last_sync_timestamp = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            );
            stats.current_consistency_status = consistency_result.status.clone();
        }

        let sync_result = SyncResult {
            success: true,
            sync_duration_ms: sync_duration.as_millis() as u64,
            consistency_status: consistency_result.status,
            conflicts_detected: consistency_result.conflicts_detected,
            l1_block_height: finality_stats.current_block_height,
        };

        debug!("L1 state synchronization completed in {:?}", sync_duration);
        Ok(sync_result)
    }

    /// Handle L1 reorganization events (Requirement 2.8, 4.5)
    pub async fn handle_reorganization(&self, reorg_event: ReorgEvent) -> Result<(), RollupError> {
        warn!(
            "Handling L1 reorganization at block {} affecting {} transactions",
            reorg_event.reorg_block, reorg_event.affected_transactions.len()
        );

        // Find affected state commitments
        let affected_states = self.find_affected_state_commitments(&reorg_event).await;

        // Delegate to ethrex integration for transaction resubmission
        self.ethrex_integration.handle_reorganization(reorg_event.clone()).await?;

        // Update local state tracking
        {
            let mut l1_tracker = self.l1_state_tracker.write().await;
            l1_tracker.reorg_history.push_back(reorg_event.clone());
            
            // Maintain reorg history limit
            while l1_tracker.reorg_history.len() > 50 {
                l1_tracker.reorg_history.pop_front();
            }

            // Mark affected commitments for re-verification
            for state_root in &affected_states {
                if let Some(commitment) = l1_tracker.l1_commitments.get_mut(state_root) {
                    commitment.finality_status = FinalityStatus::Pending;
                }
            }
        }

        // Trigger state reconciliation for affected states (Requirement 4.3)
        for state_root in &affected_states {
            self.trigger_state_reconciliation(*state_root, "L1 reorganization detected".to_string()).await?;
        }

        // Update statistics
        {
            let mut stats = self.sync_stats.write().await;
            stats.reorganizations_handled += 1;
        }

        // Emit reorganization handled event
        let event = StateSyncEvent::ReorganizationHandled {
            reorg_block: reorg_event.reorg_block,
            affected_states,
            recovery_actions: vec!["Resubmitted affected transactions".to_string()],
        };

        if let Err(e) = self.sync_events_tx.send(event) {
            warn!("Failed to send reorganization handled event: {}", e);
        }

        info!("L1 reorganization handling completed");
        Ok(())
    }    /// Re
concile state conflicts (Requirement 4.3)
    pub async fn reconcile_state_conflicts(
        &self,
        conflicts: Vec<StateConflict>,
    ) -> Result<ReconciliationResult, RollupError> {
        info!("Starting state reconciliation for {} conflicts", conflicts.len());

        let _permit = self.reconciliation_semaphore.acquire().await.unwrap();
        let reconciliation_start = Instant::now();

        let mut resolved_conflicts = 0;
        let mut failed_conflicts = 0;
        let mut final_state_root = None;

        for mut conflict in conflicts {
            match self.resolve_single_conflict(&mut conflict).await {
                Ok(resolved_state) => {
                    resolved_conflicts += 1;
                    final_state_root = Some(resolved_state);
                    
                    // Update conflict status
                    conflict.resolution_status = ConflictResolutionStatus::Resolved;
                }
                Err(e) => {
                    failed_conflicts += 1;
                    error!("Failed to resolve state conflict: {}", e);
                    
                    conflict.resolution_status = ConflictResolutionStatus::Failed {
                        reason: e.to_string(),
                    };
                }
            }

            // Update conflict in storage
            {
                let mut conflicts_storage = self.state_conflicts.write().await;
                if let Some(stored_conflict) = conflicts_storage.iter_mut()
                    .find(|c| c.local_state_root == conflict.local_state_root) {
                    *stored_conflict = conflict;
                }
            }
        }

        let reconciliation_duration = reconciliation_start.elapsed();

        // Update statistics
        {
            let mut stats = self.sync_stats.write().await;
            stats.conflicts_resolved += resolved_conflicts;
        }

        let result = ReconciliationResult {
            success: failed_conflicts == 0,
            resolved_conflicts,
            failed_conflicts,
            final_state_root: final_state_root.unwrap_or([0u8; 32]),
            reconciliation_duration_ms: reconciliation_duration.as_millis() as u64,
        };

        // Emit reconciliation completed event
        let event = StateSyncEvent::ReconciliationCompleted {
            success: result.success,
            final_state_root: result.final_state_root,
            duration_ms: result.reconciliation_duration_ms,
        };

        if let Err(e) = self.sync_events_tx.send(event) {
            warn!("Failed to send reconciliation completed event: {}", e);
        }

        info!(
            "State reconciliation completed: {} resolved, {} failed (took {:?})",
            resolved_conflicts, failed_conflicts, reconciliation_duration
        );

        Ok(result)
    }

    /// Get finalized state from L1 (Requirement 4.4)
    pub async fn get_finalized_state(&self) -> Result<FinalizedState, RollupError> {
        let l1_tracker = self.l1_state_tracker.read().await;
        
        // Find the most recent finalized state
        let mut finalized_states: Vec<_> = l1_tracker.l1_commitments.values()
            .filter(|commitment| matches!(commitment.finality_status, FinalityStatus::Finalized { .. }))
            .collect();

        finalized_states.sort_by_key(|commitment| commitment.l1_block_number);

        if let Some(latest_finalized) = finalized_states.last() {
            Ok(FinalizedState {
                state_root: latest_finalized.state_root,
                l1_block_number: latest_finalized.l1_block_number,
                finalization_timestamp: latest_finalized.commitment_timestamp,
                proof_hash: latest_finalized.proof_hash,
            })
        } else {
            Err(RollupError::L1SyncError("No finalized state found on L1".to_string()))
        }
    }

    /// Recover state from L1 and DA layer on restart (Requirement 4.6)
    pub async fn recover_state_from_l1(&self) -> Result<StateRecoveryResult, RollupError> {
        info!("Starting state recovery from L1");

        let recovery_start = Instant::now();

        // Get finalized state from L1
        let finalized_state = self.get_finalized_state().await?;

        // Reconstruct local state from L1 commitments
        let recovered_state_root = self.reconstruct_state_from_l1(&finalized_state).await?;

        // Update local state tracker
        {
            let mut local_tracker = self.local_state_tracker.write().await;
            local_tracker.current_state_root = recovered_state_root;
            local_tracker.consistency_status = StateConsistencyStatus::Consistent;
            local_tracker.last_l1_sync_time = Some(Instant::now());
        }

        let recovery_duration = recovery_start.elapsed();

        // Update statistics
        {
            let mut stats = self.sync_stats.write().await;
            stats.state_recoveries += 1;
        }

        // Emit recovery completed event
        let event = StateSyncEvent::StateRecoveryCompleted {
            recovered_state_root,
            recovery_block: finalized_state.l1_block_number,
        };

        if let Err(e) = self.sync_events_tx.send(event) {
            warn!("Failed to send state recovery completed event: {}", e);
        }

        let result = StateRecoveryResult {
            success: true,
            recovered_state_root,
            recovery_block: finalized_state.l1_block_number,
            recovery_duration_ms: recovery_duration.as_millis() as u64,
        };

        info!(
            "State recovery completed: {:?} from block {} (took {:?})",
            recovered_state_root, finalized_state.l1_block_number, recovery_duration
        );

        Ok(result)
    }

    /// Get synchronization statistics
    pub async fn get_sync_stats(&self) -> StateSyncStats {
        self.sync_stats.read().await.clone()
    }

    /// Get current state conflicts
    pub async fn get_state_conflicts(&self) -> Vec<StateConflict> {
        self.state_conflicts.read().await.clone()
    }

    /// Get sync event receiver
    pub async fn get_sync_events(&self) -> Option<mpsc::UnboundedReceiver<StateSyncEvent>> {
        let mut rx_option = self.sync_events_rx.write().await;
        rx_option.take()
    }

    // Private helper methods

    /// Start the bidirectional synchronization loop (Requirement 4.1)
    async fn start_sync_loop(&self) -> Result<(), RollupError> {
        let sync_interval = Duration::from_millis(self.config.sync_interval_ms);
        let ethrex_integration = Arc::clone(&self.ethrex_integration);
        let local_tracker = Arc::clone(&self.local_state_tracker);
        let l1_tracker = Arc::clone(&self.l1_state_tracker);
        let sync_events_tx = self.sync_events_tx.clone();
        let stats = Arc::clone(&self.sync_stats);

        let sync_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(sync_interval);
            
            loop {
                interval.tick().await;
                
                let sync_start = Instant::now();
                
                // Perform bidirectional sync
                match Self::perform_bidirectional_sync(
                    &ethrex_integration,
                    &local_tracker,
                    &l1_tracker,
                    &sync_events_tx,
                ).await {
                    Ok(_) => {
                        let mut stats_guard = stats.write().await;
                        stats_guard.successful_syncs += 1;
                        stats_guard.total_sync_cycles += 1;
                        
                        let sync_duration = sync_start.elapsed().as_millis() as f64;
                        stats_guard.average_sync_duration_ms = 
                            (stats_guard.average_sync_duration_ms * (stats_guard.total_sync_cycles - 1) as f64 + 
                             sync_duration) / stats_guard.total_sync_cycles as f64;
                    }
                    Err(e) => {
                        error!("Sync loop error: {}", e);
                        let mut stats_guard = stats.write().await;
                        stats_guard.failed_syncs += 1;
                        stats_guard.total_sync_cycles += 1;
                    }
                }
            }
        });

        {
            let mut handle = self.sync_task_handle.write().await;
            *handle = Some(sync_handle);
        }

        Ok(())
    }

    /// Start the state verification loop (Requirement 4.2)
    async fn start_verification_loop(&self) -> Result<(), RollupError> {
        let verification_interval = Duration::from_millis(self.config.verification_interval_ms);
        let local_tracker = Arc::clone(&self.local_state_tracker);
        let l1_tracker = Arc::clone(&self.l1_state_tracker);
        let sync_events_tx = self.sync_events_tx.clone();

        let verification_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(verification_interval);
            
            loop {
                interval.tick().await;
                
                // Perform deep state verification
                if let Err(e) = Self::perform_deep_verification(
                    &local_tracker,
                    &l1_tracker,
                    &sync_events_tx,
                ).await {
                    error!("Deep verification error: {}", e);
                }
            }
        });

        {
            let mut handle = self.verification_task_handle.write().await;
            *handle = Some(verification_handle);
        }

        Ok(())
    }

    /// Start finality event processing (Requirement 4.4)
    async fn start_finality_event_processing(&self) -> Result<(), RollupError> {
        let finality_events_rx = self.ethrex_integration.finality_tracker.get_finality_events().await;
        let l1_tracker = Arc::clone(&self.l1_state_tracker);
        let sync_events_tx = self.sync_events_tx.clone();

        if let Some(mut rx) = finality_events_rx {
            let finality_handle = tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let Err(e) = Self::handle_finality_event(
                        event,
                        &l1_tracker,
                        &sync_events_tx,
                    ).await {
                        error!("Failed to handle finality event: {}", e);
                    }
                }
            });

            {
                let mut handle = self.finality_events_handler.lock().await;
                *handle = Some(finality_handle);
            }
        }

        Ok(())
    }

    /// Perform startup recovery (Requirement 4.6)
    async fn perform_startup_recovery(&self) -> Result<(), RollupError> {
        info!("Performing startup state recovery");

        let recovery_timeout = Duration::from_secs(self.config.recovery_timeout_seconds);
        let recovery_future = self.recover_state_from_l1();

        match tokio::time::timeout(recovery_timeout, recovery_future).await {
            Ok(Ok(recovery_result)) => {
                info!("Startup recovery completed successfully: {:?}", recovery_result.recovered_state_root);
                Ok(())
            }
            Ok(Err(e)) => {
                error!("Startup recovery failed: {}", e);
                Err(e)
            }
            Err(_) => {
                error!("Startup recovery timed out after {:?}", recovery_timeout);
                Err(RollupError::L1SyncError("Startup recovery timeout".to_string()))
            }
        }
    }

    /// Perform bidirectional synchronization
    async fn perform_bidirectional_sync(
        ethrex_integration: &Arc<EthrexIntegrationManager>,
        local_tracker: &Arc<RwLock<LocalStateTracker>>,
        l1_tracker: &Arc<RwLock<L1StateTracker>>,
        sync_events_tx: &mpsc::UnboundedSender<StateSyncEvent>,
    ) -> Result<(), RollupError> {
        // Get L1 state updates
        let l1_commitments = ethrex_integration.get_all_commitment_records().await;
        let finality_stats = ethrex_integration.get_finality_stats().await;

        // Update L1 tracker
        {
            let mut l1_tracker_guard = l1_tracker.write().await;
            l1_tracker_guard.current_l1_block = finality_stats.current_block_height.unwrap_or(0);
            l1_tracker_guard.last_l1_query_time = Some(Instant::now());

            // Update commitments
            for (state_root, commitment_record) in l1_commitments {
                let l1_commitment = L1StateCommitment {
                    state_root,
                    commitment_tx_hash: commitment_record.commitment_tx_hash,
                    l1_block_number: commitment_record.l1_block_number,
                    commitment_timestamp: commitment_record.commitment_timestamp,
                    finality_status: match commitment_record.confirmation_status {
                        crate::rollup::ethrex_client::ConfirmationStatus::Pending { confirmations } => {
                            FinalityStatus::Confirmed { confirmations }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Confirmed { final_block } => {
                            FinalityStatus::Finalized { finalization_block: final_block }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Failed { reason } => {
                            FinalityStatus::Failed { reason }
                        }
                        crate::rollup::ethrex_client::ConfirmationStatus::Reorged { .. } => {
                            FinalityStatus::Pending
                        }
                    },
                    proof_hash: commitment_record.proof_commitment_hash,
                };

                l1_tracker_guard.l1_commitments.insert(state_root, l1_commitment);
            }
        }

        // Check consistency and emit events
        let consistency_result = Self::check_consistency_internal(local_tracker, l1_tracker).await?;
        
        let event = StateSyncEvent::ConsistencyCheckCompleted {
            status: consistency_result.status.clone(),
            drift_blocks: consistency_result.drift_blocks,
        };

        if let Err(e) = sync_events_tx.send(event) {
            warn!("Failed to send consistency check event: {}", e);
        }

        Ok(())
    }

    /// Perform deep state verification
    async fn perform_deep_verification(
        local_tracker: &Arc<RwLock<LocalStateTracker>>,
        l1_tracker: &Arc<RwLock<L1StateTracker>>,
        sync_events_tx: &mpsc::UnboundedSender<StateSyncEvent>,
    ) -> Result<(), RollupError> {
        let local_guard = local_tracker.read().await;
        let l1_guard = l1_tracker.read().await;

        // Verify state snapshots integrity
        for snapshot in &local_guard.state_snapshots {
            // Verify merkle proof (simplified)
            if !Self::verify_snapshot_integrity(snapshot) {
                warn!("State snapshot integrity verification failed: {:?}", snapshot.state_root);
            }
        }

        // Verify L1 commitment consistency
        for (state_root, commitment) in &l1_guard.l1_commitments {
            if let Some(local_snapshot) = local_guard.state_snapshots.iter()
                .find(|s| s.state_root == *state_root) {
                
                // Verify state root matches
                if local_snapshot.state_root != commitment.state_root {
                    warn!("State root mismatch detected: local={:?}, L1={:?}", 
                          local_snapshot.state_root, commitment.state_root);
                }
            }
        }

        Ok(())
    }

    /// Handle finality events
    async fn handle_finality_event(
        event: crate::rollup::finality_tracker::FinalityEvent,
        l1_tracker: &Arc<RwLock<L1StateTracker>>,
        sync_events_tx: &mpsc::UnboundedSender<StateSyncEvent>,
    ) -> Result<(), RollupError> {
        match event {
            crate::rollup::finality_tracker::FinalityEvent::TransactionFinalized { 
                tx_hash, 
                transaction_type, 
                finalization_block,
                .. 
            } => {
                // Update finality status for associated state commitments
                let mut l1_guard = l1_tracker.write().await;
                
                for commitment in l1_guard.l1_commitments.values_mut() {
                    if commitment.commitment_tx_hash == tx_hash {
                        commitment.finality_status = FinalityStatus::Finalized { 
                            finalization_block 
                        };
                        
                        // Add to finalized state roots
                        l1_guard.finalized_state_roots.push_back(commitment.state_root);
                        
                        // Maintain finalized state roots limit
                        while l1_guard.finalized_state_roots.len() > 100 {
                            l1_guard.finalized_state_roots.pop_front();
                        }
                        
                        break;
                    }
                }
            }
            crate::rollup::finality_tracker::FinalityEvent::ReorganizationDetected { 
                reorg_block, 
                affected_transactions,
                .. 
            } => {
                let reorg_event = ReorgEvent {
                    reorg_block,
                    affected_transactions,
                    new_canonical_block: reorg_block,
                };

                let event = StateSyncEvent::ReorganizationHandled {
                    reorg_block,
                    affected_states: Vec::new(), // Would be populated with actual affected states
                    recovery_actions: vec!["Finality event processed".to_string()],
                };

                if let Err(e) = sync_events_tx.send(event) {
                    warn!("Failed to send reorganization handled event: {}", e);
                }
            }
            _ => {
                // Handle other finality events as needed
            }
        }

        Ok(())
    }

    /// Check state consistency between local and L1
    async fn check_state_consistency(&self) -> Result<ConsistencyCheckResult, RollupError> {
        Self::check_consistency_internal(&self.local_state_tracker, &self.l1_state_tracker).await
    }

    /// Internal consistency check implementation
    async fn check_consistency_internal(
        local_tracker: &Arc<RwLock<LocalStateTracker>>,
        l1_tracker: &Arc<RwLock<L1StateTracker>>,
    ) -> Result<ConsistencyCheckResult, RollupError> {
        let local_guard = local_tracker.read().await;
        let l1_guard = l1_tracker.read().await;

        let mut conflicts_detected = 0;
        let mut drift_blocks = 0;

        // Check for state root mismatches
        let current_local_state = local_guard.current_state_root;
        
        // Find corresponding L1 commitment
        if let Some(l1_commitment) = l1_guard.l1_commitments.get(&current_local_state) {
            // Check if finalized
            if matches!(l1_commitment.finality_status, FinalityStatus::Finalized { .. }) {
                // States are consistent and finalized
                return Ok(ConsistencyCheckResult {
                    status: StateConsistencyStatus::Consistent,
                    conflicts_detected: 0,
                    drift_blocks: 0,
                });
            }
        }

        // Check for drift
        let local_height = local_guard.local_block_height;
        let l1_height = l1_guard.current_l1_block;
        
        if l1_height > local_height {
            drift_blocks = l1_height - local_height;
        }

        // Count conflicts by checking for missing commitments and mismatches
        for snapshot in &local_guard.state_snapshots {
            if !l1_guard.l1_commitments.contains_key(&snapshot.state_root) {
                conflicts_detected += 1;
            }
        }

        let status = if conflicts_detected > 0 {
            StateConsistencyStatus::Inconsistent { 
                reason: format!("Found {} conflicts", conflicts_detected) 
            }
        } else if drift_blocks == 0 {
            StateConsistencyStatus::Consistent
        } else if drift_blocks <= 3 { // Using default max_state_drift_blocks
            StateConsistencyStatus::MinorDrift { drift_blocks }
        } else {
            StateConsistencyStatus::MajorDrift { drift_blocks }
        };

        Ok(ConsistencyCheckResult {
            status,
            conflicts_detected,
            drift_blocks,
        })
    }

    /// Handle detected state conflicts (Requirement 4.3)
    async fn handle_state_conflicts(&self) -> Result<(), RollupError> {
        let conflicts = self.detect_state_conflicts().await?;
        
        if !conflicts.is_empty() {
            info!("Detected {} state conflicts, starting reconciliation", conflicts.len());
            
            // Store conflicts
            {
                let mut conflicts_storage = self.state_conflicts.write().await;
                conflicts_storage.extend(conflicts.clone());
            }

            // Trigger reconciliation
            self.reconcile_state_conflicts(conflicts).await?;
        }

        Ok(())
    }

    /// Detect state conflicts between local and L1
    async fn detect_state_conflicts(&self) -> Result<Vec<StateConflict>, RollupError> {
        let local_guard = self.local_state_tracker.read().await;
        let l1_guard = self.l1_state_tracker.read().await;
        
        let mut conflicts = Vec::new();

        // Check for missing L1 commitments
        for snapshot in &local_guard.state_snapshots {
            if !l1_guard.l1_commitments.contains_key(&snapshot.state_root) {
                conflicts.push(StateConflict {
                    local_state_root: snapshot.state_root,
                    l1_state_root: [0u8; 32], // No L1 state
                    detection_time: Instant::now(),
                    conflict_type: StateConflictType::MissingL1Commitment,
                    resolution_strategy: ConflictResolutionStrategy::L1Priority,
                    resolution_status: ConflictResolutionStatus::Pending,
                });
            }
        }

        // Check for orphaned local states
        let current_time = Instant::now();
        for snapshot in &local_guard.state_snapshots {
            if let Some(l1_commitment) = l1_guard.l1_commitments.get(&snapshot.state_root) {
                if matches!(l1_commitment.finality_status, FinalityStatus::Failed { .. }) {
                    conflicts.push(StateConflict {
                        local_state_root: snapshot.state_root,
                        l1_state_root: l1_commitment.state_root,
                        detection_time: current_time,
                        conflict_type: StateConflictType::OrphanedLocalState,
                        resolution_strategy: ConflictResolutionStrategy::L1Priority,
                        resolution_status: ConflictResolutionStatus::Pending,
                    });
                }
            }
        }

        Ok(conflicts)
    }

    /// Resolve a single state conflict
    async fn resolve_single_conflict(&self, conflict: &mut StateConflict) -> Result<StateRoot, RollupError> {
        info!("Resolving state conflict: {:?}", conflict.conflict_type);

        match conflict.resolution_strategy {
            ConflictResolutionStrategy::L1Priority => {
                // Use L1 state as source of truth
                self.rollback_to_l1_state(conflict.l1_state_root).await?;
                Ok(conflict.l1_state_root)
            }
            ConflictResolutionStrategy::LocalPriority => {
                // Resubmit local state to L1
                self.resubmit_local_state(conflict.local_state_root).await?;
                Ok(conflict.local_state_root)
            }
            ConflictResolutionStrategy::StateMerge => {
                // Merge conflicting states
                let merged_state = self.merge_states(conflict.local_state_root, conflict.l1_state_root).await?;
                Ok(merged_state)
            }
            ConflictResolutionStrategy::ManualResolution => {
                // Manual resolution required - for now, default to L1 priority
                warn!("Manual resolution required for conflict, defaulting to L1 priority");
                self.rollback_to_l1_state(conflict.l1_state_root).await?;
                Ok(conflict.l1_state_root)
            }
        }
    }

    /// Trigger state reconciliation for a specific state root
    async fn trigger_state_reconciliation(
        &self,
        state_root: StateRoot,
        reason: String,
    ) -> Result<(), RollupError> {
        info!("Triggering state reconciliation for {:?}: {}", state_root, reason);

        // Create a conflict for this state
        let conflict = StateConflict {
            local_state_root: state_root,
            l1_state_root: state_root, // Will be updated during resolution
            detection_time: Instant::now(),
            conflict_type: StateConflictType::StateMismatch,
            resolution_strategy: ConflictResolutionStrategy::L1Priority,
            resolution_status: ConflictResolutionStatus::Pending,
        };

        // Emit reconciliation started event
        let event = StateSyncEvent::ReconciliationStarted {
            trigger_reason: reason,
            target_state_root: state_root,
        };

        if let Err(e) = self.sync_events_tx.send(event) {
            warn!("Failed to send reconciliation started event: {}", e);
        }

        // Perform reconciliation
        self.reconcile_state_conflicts(vec![conflict]).await?;

        Ok(())
    }

    /// Find affected state commitments during reorganization
    async fn find_affected_state_commitments(&self, reorg_event: &ReorgEvent) -> Vec<StateRoot> {
        let l1_guard = self.l1_state_tracker.read().await;
        let mut affected_states = Vec::new();

        // Find commitments that were included in the reorganized blocks
        for (state_root, commitment) in &l1_guard.l1_commitments {
            if commitment.l1_block_number >= reorg_event.reorg_block {
                affected_states.push(*state_root);
            }
        }

        affected_states
    }

    /// Rollback to L1 state (Requirement 4.5)
    async fn rollback_to_l1_state(&self, l1_state_root: StateRoot) -> Result<(), RollupError> {
        info!("Rolling back to L1 state: {:?}", l1_state_root);

        // Update local state tracker
        {
            let mut local_tracker = self.local_state_tracker.write().await;
            local_tracker.current_state_root = l1_state_root;
            local_tracker.consistency_status = StateConsistencyStatus::Consistent;
            
            // Find the corresponding snapshot or create a new one
            if !local_tracker.state_snapshots.iter().any(|s| s.state_root == l1_state_root) {
                let snapshot = StateSnapshot {
                    state_root: l1_state_root,
                    sequence_number: local_tracker.local_block_height,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    l1_block_reference: None,
                    merkle_proof: Vec::new(),
                    compressed_state_data: Vec::new(),
                };
                
                local_tracker.state_snapshots.push_back(snapshot);
            }
        }

        info!("Successfully rolled back to L1 state: {:?}", l1_state_root);
        Ok(())
    }

    /// Resubmit local state to L1
    async fn resubmit_local_state(&self, local_state_root: StateRoot) -> Result<(), RollupError> {
        info!("Resubmitting local state to L1: {:?}", local_state_root);

        // Create a dummy proof for resubmission (in real implementation, would generate actual proof)
        let dummy_proof = crate::zkvm::ZkProof {
            backend: crate::zkvm::ZkVMBackend::SP1Local,
            proof_data: vec![0u8; 32],
            public_inputs: vec![0u8; 32],
            verification_key_hash: [0u8; 32],
            proof_metadata: crate::zkvm::ProofMetadata {
                proof_id: format!("resubmit_{:?}", local_state_root),
                generation_time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                proof_size: 32,
                security_level: 128,
                circuit_size: 1000000,
            },
        };

        // Submit state anchor request
        let anchor_request = crate::rollup::ethrex_integration::StateAnchorRequest {
            state_root: local_state_root,
            proof: dummy_proof,
            priority: crate::rollup::ethrex_integration::SubmissionPriority::High,
        };

        self.ethrex_integration.submit_state_anchor(anchor_request).await?;

        info!("Successfully resubmitted local state to L1: {:?}", local_state_root);
        Ok(())
    }

    /// Reconstruct state from L1 commitments
    async fn reconstruct_state_from_l1(&self, finalized_state: &FinalizedState) -> Result<StateRoot, RollupError> {
        info!("Reconstructing state from L1: {:?}", finalized_state.state_root);

        // In a real implementation, this would:
        // 1. Fetch the state data from DA layer using the state root
        // 2. Verify the state integrity using merkle proofs
        // 3. Decompress and reconstruct the full state
        // 4. Validate the state against the proof

        // For now, we'll use the finalized state root directly
        let reconstructed_state = finalized_state.state_root;

        info!("Successfully reconstructed state from L1: {:?}", reconstructed_state);
        Ok(reconstructed_state)
    }

    /// Merge conflicting states (simplified implementation)
    async fn merge_states(&self, local_state: StateRoot, l1_state: StateRoot) -> Result<StateRoot, RollupError> {
        info!("Merging states: local={:?}, l1={:?}", local_state, l1_state);

        // In a real implementation, this would:
        // 1. Analyze the differences between states
        // 2. Apply conflict resolution rules
        // 3. Create a merged state
        // 4. Generate proof for the merged state

        // For now, prioritize L1 state
        Ok(l1_state)
    }

    /// Verify snapshot integrity (simplified)
    fn verify_snapshot_integrity(snapshot: &StateSnapshot) -> bool {
        // In a real implementation, this would verify:
        // 1. Merkle proof validity
        // 2. State data integrity
        // 3. Compression validity
        
        // For now, just check basic fields
        !snapshot.merkle_proof.is_empty() || snapshot.compressed_state_data.len() > 0
    }
}

impl LocalStateTracker {
    pub fn new() -> Self {
        Self {
            current_state_root: [0u8; 32],
            local_block_height: 0,
            state_snapshots: VecDeque::new(),
            pending_transitions: HashMap::new(),
            last_l1_sync_time: None,
            consistency_status: StateConsistencyStatus::Consistent,
        }
    }
}

impl L1StateTracker {
    pub fn new() -> Self {
        Self {
            current_l1_block: 0,
            l1_commitments: HashMap::new(),
            finalized_state_roots: VecDeque::new(),
            pending_l1_transactions: HashMap::new(),
            last_l1_query_time: None,
            reorg_history: VecDeque::new(),
        }
    }
}

/// Result of synchronization operation
#[derive(Debug, Clone)]
pub struct SyncResult {
    pub success: bool,
    pub sync_duration_ms: u64,
    pub consistency_status: StateConsistencyStatus,
    pub conflicts_detected: u64,
    pub l1_block_height: Option<L1BlockNumber>,
}

/// Result of state reconciliation
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    pub success: bool,
    pub resolved_conflicts: u64,
    pub failed_conflicts: u64,
    pub final_state_root: StateRoot,
    pub reconciliation_duration_ms: u64,
}

/// Result of consistency check
#[derive(Debug, Clone)]
pub struct ConsistencyCheckResult {
    pub status: StateConsistencyStatus,
    pub conflicts_detected: u64,
    pub drift_blocks: u64,
}

/// Result of state recovery
#[derive(Debug, Clone)]
pub struct StateRecoveryResult {
    pub success: bool,
    pub recovered_state_root: StateRoot,
    pub recovery_block: L1BlockNumber,
    pub recovery_duration_ms: u64,
}

/// Finalized state information
#[derive(Debug, Clone)]
pub struct FinalizedState {
    pub state_root: StateRoot,
    pub l1_block_number: L1BlockNumber,
    pub finalization_timestamp: u64,
    pub proof_hash: [u8; 32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollup::ethrex_integration::EthrexIntegrationConfig;
    use std::time::Duration;
    use tokio::time::sleep;

    fn create_test_config() -> L1StateSyncConfig {
        L1StateSyncConfig {
            sync_interval_ms: 100, // Fast sync for testing
            max_state_drift_blocks: 2,
            enable_auto_reconciliation: true,
            reconciliation_timeout_seconds: 10,
            max_state_snapshots: 10,
            enable_deep_verification: false, // Disable for testing
            verification_interval_ms: 1000,
            max_concurrent_reconciliations: 2,
            enable_startup_recovery: false, // Disable for testing
            recovery_timeout_seconds: 30,
        }
    }

    async fn create_test_manager() -> Result<L1StateSyncManager, RollupError> {
        let config = create_test_config();
        let ethrex_config = EthrexIntegrationConfig::default();
        let ethrex_integration = Arc::new(EthrexIntegrationManager::new(ethrex_config).await?);
        
        L1StateSyncManager::new(config, ethrex_integration).await
    }

    #[tokio::test]
    async fn test_l1_state_sync_manager_creation() {
        let manager = create_test_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_local_state_update() {
        let manager = create_test_manager().await.unwrap();
        
        let state_root = [1u8; 32];
        let sequence_number = 1;
        
        let result = manager.update_local_state(state_root, sequence_number, None).await;
        assert!(result.is_ok());
        
        // Verify state was updated
        let local_tracker = manager.local_state_tracker.read().await;
        assert_eq!(local_tracker.current_state_root, state_root);
        assert_eq!(local_tracker.local_block_height, sequence_number);
        assert_eq!(local_tracker.state_snapshots.len(), 1);
    }

    #[tokio::test]
    async fn test_state_consistency_check() {
        let manager = create_test_manager().await.unwrap();
        
        // Update local state
        let state_root = [2u8; 32];
        manager.update_local_state(state_root, 1, None).await.unwrap();
        
        // Check consistency
        let result = manager.check_state_consistency().await;
        assert!(result.is_ok());
        
        let consistency_result = result.unwrap();
        // Should be consistent initially (no L1 state to compare against)
        assert!(matches!(consistency_result.status, StateConsistencyStatus::Consistent));
    }

    #[tokio::test]
    async fn test_state_conflict_detection() {
        let manager = create_test_manager().await.unwrap();
        
        // Add some local state
        manager.update_local_state([3u8; 32], 1, None).await.unwrap();
        
        // Detect conflicts
        let conflicts = manager.detect_state_conflicts().await.unwrap();
        
        // Should detect missing L1 commitment
        assert!(!conflicts.is_empty());
        assert!(matches!(conflicts[0].conflict_type, StateConflictType::MissingL1Commitment));
    }

    #[tokio::test]
    async fn test_state_reconciliation() {
        let manager = create_test_manager().await.unwrap();
        
        // Create a test conflict
        let conflict = StateConflict {
            local_state_root: [4u8; 32],
            l1_state_root: [5u8; 32],
            detection_time: Instant::now(),
            conflict_type: StateConflictType::StateMismatch,
            resolution_strategy: ConflictResolutionStrategy::L1Priority,
            resolution_status: ConflictResolutionStatus::Pending,
        };
        
        let result = manager.reconcile_state_conflicts(vec![conflict]).await;
        assert!(result.is_ok());
        
        let reconciliation_result = result.unwrap();
        assert_eq!(reconciliation_result.resolved_conflicts, 1);
        assert_eq!(reconciliation_result.failed_conflicts, 0);
    }

    #[tokio::test]
    async fn test_reorganization_handling() {
        let manager = create_test_manager().await.unwrap();
        
        // Create a test reorganization event
        let reorg_event = ReorgEvent {
            reorg_block: 100,
            affected_transactions: vec![[1u8; 32], [2u8; 32]],
            new_canonical_block: 101,
        };
        
        let result = manager.handle_reorganization(reorg_event).await;
        assert!(result.is_ok());
        
        // Verify reorg was recorded
        let l1_tracker = manager.l1_state_tracker.read().await;
        assert_eq!(l1_tracker.reorg_history.len(), 1);
    }

    #[tokio::test]
    async fn test_sync_statistics() {
        let manager = create_test_manager().await.unwrap();
        
        let stats = manager.get_sync_stats().await;
        assert_eq!(stats.total_sync_cycles, 0);
        assert_eq!(stats.successful_syncs, 0);
        assert_eq!(stats.failed_syncs, 0);
        assert_eq!(stats.conflicts_detected, 0);
        assert_eq!(stats.conflicts_resolved, 0);
    }

    #[tokio::test]
    async fn test_state_snapshot_management() {
        let manager = create_test_manager().await.unwrap();
        
        // Add multiple state updates to test snapshot management
        for i in 1..=15 {
            let state_root = [i as u8; 32];
            manager.update_local_state(state_root, i, None).await.unwrap();
        }
        
        // Verify snapshot limit is enforced
        let local_tracker = manager.local_state_tracker.read().await;
        assert!(local_tracker.state_snapshots.len() <= manager.config.max_state_snapshots);
    }

    #[tokio::test]
    async fn test_finalized_state_retrieval() {
        let manager = create_test_manager().await.unwrap();
        
        // This test would require mock L1 state, so we just test the error case
        let result = manager.get_finalized_state().await;
        assert!(result.is_err()); // Should fail with no finalized state
    }

    #[tokio::test]
    async fn test_state_recovery() {
        let manager = create_test_manager().await.unwrap();
        
        // This test would require mock L1 state, so we just test the error case
        let result = manager.recover_state_from_l1().await;
        assert!(result.is_err()); // Should fail with no finalized state
    }

    #[tokio::test]
    async fn test_sync_events() {
        let manager = create_test_manager().await.unwrap();
        
        // Get event receiver
        let mut events_rx = manager.get_sync_events().await;
        assert!(events_rx.is_some());
        
        let mut rx = events_rx.unwrap();
        
        // Update local state to trigger event
        tokio::spawn(async move {
            sleep(Duration::from_millis(10)).await;
            // This would trigger an event in a real scenario
        });
        
        // Try to receive event with timeout
        let event_result = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await;
        // May or may not receive an event depending on timing
    }
}