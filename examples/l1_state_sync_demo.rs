//! L1 State Synchronization Manager Demo
//! 
//! This example demonstrates the L1 State Synchronization Manager functionality,
//! including bidirectional state sync, reorganization handling, state reconciliation,
//! and finality tracking as specified in requirements 2.5, 2.8, 4.1-4.5.

use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};

use zk_provable_orderbook::rollup::{
    L1StateSyncManager, L1StateSyncConfig,
    EthrexIntegrationManager, EthrexIntegrationConfig,
    StateRoot, StateSyncEvent, StateConsistencyStatus,
    RollupError,
};
use zk_provable_orderbook::zkvm::{ZkProof, ZkVMBackend, ProofMetadata};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting L1 State Synchronization Manager Demo");

    // Create ethrex integration manager
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config).await
            .map_err(|e| format!("Failed to create ethrex integration: {}", e))?
    );

    // Create L1 state sync manager
    let sync_config = L1StateSyncConfig {
        sync_interval_ms: 2000, // 2 seconds for demo
        max_state_drift_blocks: 2,
        enable_auto_reconciliation: true,
        reconciliation_timeout_seconds: 30,
        max_state_snapshots: 10,
        enable_deep_verification: true,
        verification_interval_ms: 5000, // 5 seconds for demo
        max_concurrent_reconciliations: 3,
        enable_startup_recovery: false, // Disable for demo
        recovery_timeout_seconds: 60,
    };

    let sync_manager = L1StateSyncManager::new(sync_config, ethrex_integration).await
        .map_err(|e| format!("Failed to create L1 state sync manager: {}", e))?;

    info!("L1 State Sync Manager created successfully");

    // Demonstrate bidirectional state synchronization (Requirement 4.1)
    demo_bidirectional_sync(&sync_manager).await?;

    // Demonstrate state consistency checking (Requirement 4.2)
    demo_state_consistency_check(&sync_manager).await?;

    // Demonstrate state reconciliation (Requirement 4.3)
    demo_state_reconciliation(&sync_manager).await?;

    // Demonstrate finality tracking (Requirement 4.4)
    demo_finality_tracking(&sync_manager).await?;

    // Demonstrate reorganization handling (Requirement 2.8, 4.5)
    demo_reorganization_handling(&sync_manager).await?;

    // Start the sync manager
    info!("Starting L1 State Sync Manager...");
    sync_manager.start().await
        .map_err(|e| format!("Failed to start sync manager: {}", e))?;

    // Get sync event receiver to monitor events
    let mut sync_events = sync_manager.get_sync_events().await;

    // Monitor sync events for a short period
    if let Some(mut rx) = sync_events {
        info!("Monitoring sync events for 10 seconds...");
        
        let monitor_task = tokio::spawn(async move {
            let mut event_count = 0;
            let start_time = std::time::Instant::now();
            
            while start_time.elapsed() < Duration::from_secs(10) {
                tokio::select! {
                    event = rx.recv() => {
                        if let Some(event) = event {
                            event_count += 1;
                            match event {
                                StateSyncEvent::LocalStateUpdated { new_state_root, sequence_number } => {
                                    info!("Event {}: Local state updated - root: {:?}, seq: {}", 
                                          event_count, new_state_root, sequence_number);
                                }
                                StateSyncEvent::ConsistencyCheckCompleted { status, drift_blocks } => {
                                    info!("Event {}: Consistency check completed - status: {:?}, drift: {}", 
                                          event_count, status, drift_blocks);
                                }
                                StateSyncEvent::ReconciliationStarted { trigger_reason, target_state_root } => {
                                    info!("Event {}: Reconciliation started - reason: {}, target: {:?}", 
                                          event_count, trigger_reason, target_state_root);
                                }
                                StateSyncEvent::ReconciliationCompleted { success, final_state_root, duration_ms } => {
                                    info!("Event {}: Reconciliation completed - success: {}, root: {:?}, duration: {}ms", 
                                          event_count, success, final_state_root, duration_ms);
                                }
                                StateSyncEvent::ReorganizationHandled { reorg_block, affected_states, recovery_actions } => {
                                    info!("Event {}: Reorganization handled - block: {}, affected: {}, actions: {:?}", 
                                          event_count, reorg_block, affected_states.len(), recovery_actions);
                                }
                                StateSyncEvent::StateRecoveryCompleted { recovered_state_root, recovery_block } => {
                                    info!("Event {}: State recovery completed - root: {:?}, block: {}", 
                                          event_count, recovered_state_root, recovery_block);
                                }
                                _ => {
                                    info!("Event {}: Other sync event received", event_count);
                                }
                            }
                        }
                    }
                    _ = sleep(Duration::from_millis(100)) => {
                        // Continue monitoring
                    }
                }
            }
            
            info!("Monitored {} sync events", event_count);
        });

        monitor_task.await?;
    }

    // Display final statistics
    let stats = sync_manager.get_sync_stats().await;
    info!("Final sync statistics:");
    info!("  Total sync cycles: {}", stats.total_sync_cycles);
    info!("  Successful syncs: {}", stats.successful_syncs);
    info!("  Failed syncs: {}", stats.failed_syncs);
    info!("  Average sync duration: {:.2}ms", stats.average_sync_duration_ms);
    info!("  Conflicts detected: {}", stats.conflicts_detected);
    info!("  Conflicts resolved: {}", stats.conflicts_resolved);
    info!("  Reorganizations handled: {}", stats.reorganizations_handled);
    info!("  State recoveries: {}", stats.state_recoveries);
    info!("  Current consistency status: {:?}", stats.current_consistency_status);

    // Stop the sync manager
    info!("Stopping L1 State Sync Manager...");
    sync_manager.stop().await
        .map_err(|e| format!("Failed to stop sync manager: {}", e))?;

    info!("L1 State Synchronization Manager Demo completed successfully!");
    Ok(())
}

/// Demonstrate bidirectional state synchronization (Requirement 4.1)
async fn demo_bidirectional_sync(sync_manager: &L1StateSyncManager) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Bidirectional State Synchronization ===");

    // Simulate local state updates
    let test_states = vec![
        ([1u8; 32], 1),
        ([2u8; 32], 2),
        ([3u8; 32], 3),
    ];

    for (state_root, sequence) in test_states {
        info!("Updating local state: {:?} (seq: {})", state_root, sequence);
        
        // Create a test proof
        let test_proof = create_test_proof();
        
        sync_manager.update_local_state(state_root, sequence, Some(test_proof)).await
            .map_err(|e| format!("Failed to update local state: {}", e))?;
        
        sleep(Duration::from_millis(500)).await;
    }

    // Perform L1 synchronization
    info!("Performing L1 synchronization...");
    let sync_result = sync_manager.sync_with_l1().await
        .map_err(|e| format!("Failed to sync with L1: {}", e))?;

    info!("L1 sync result:");
    info!("  Success: {}", sync_result.success);
    info!("  Duration: {}ms", sync_result.sync_duration_ms);
    info!("  Consistency status: {:?}", sync_result.consistency_status);
    info!("  Conflicts detected: {}", sync_result.conflicts_detected);
    info!("  L1 block height: {:?}", sync_result.l1_block_height);

    Ok(())
}

/// Demonstrate state consistency checking (Requirement 4.2)
async fn demo_state_consistency_check(sync_manager: &L1StateSyncManager) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating State Consistency Checking ===");

    // Update local state to create potential inconsistency
    let inconsistent_state = [99u8; 32];
    sync_manager.update_local_state(inconsistent_state, 10, None).await
        .map_err(|e| format!("Failed to update local state: {}", e))?;

    // Perform sync to check consistency
    let sync_result = sync_manager.sync_with_l1().await
        .map_err(|e| format!("Failed to sync with L1: {}", e))?;

    info!("Consistency check result:");
    match sync_result.consistency_status {
        StateConsistencyStatus::Consistent => {
            info!("  Status: Consistent - local and L1 states match");
        }
        StateConsistencyStatus::MinorDrift { drift_blocks } => {
            info!("  Status: Minor drift detected - {} blocks behind", drift_blocks);
        }
        StateConsistencyStatus::MajorDrift { drift_blocks } => {
            warn!("  Status: Major drift detected - {} blocks behind", drift_blocks);
        }
        StateConsistencyStatus::Inconsistent { reason } => {
            warn!("  Status: Inconsistent - reason: {}", reason);
        }
        StateConsistencyStatus::Reconciling { started_at } => {
            info!("  Status: Reconciliation in progress - started at: {}", started_at);
        }
    }

    Ok(())
}

/// Demonstrate state reconciliation (Requirement 4.3)
async fn demo_state_reconciliation(sync_manager: &L1StateSyncManager) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating State Reconciliation ===");

    // Get current state conflicts
    let conflicts = sync_manager.get_state_conflicts().await;
    info!("Current state conflicts: {}", conflicts.len());

    if !conflicts.is_empty() {
        info!("Reconciling {} state conflicts...", conflicts.len());
        
        let reconciliation_result = sync_manager.reconcile_state_conflicts(conflicts).await
            .map_err(|e| format!("Failed to reconcile state conflicts: {}", e))?;

        info!("Reconciliation result:");
        info!("  Success: {}", reconciliation_result.success);
        info!("  Resolved conflicts: {}", reconciliation_result.resolved_conflicts);
        info!("  Failed conflicts: {}", reconciliation_result.failed_conflicts);
        info!("  Final state root: {:?}", reconciliation_result.final_state_root);
        info!("  Duration: {}ms", reconciliation_result.reconciliation_duration_ms);
    } else {
        info!("No state conflicts to reconcile");
    }

    Ok(())
}

/// Demonstrate finality tracking (Requirement 4.4)
async fn demo_finality_tracking(sync_manager: &L1StateSyncManager) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Finality Tracking ===");

    // Try to get finalized state from L1
    match sync_manager.get_finalized_state().await {
        Ok(finalized_state) => {
            info!("Finalized state found:");
            info!("  State root: {:?}", finalized_state.state_root);
            info!("  L1 block number: {}", finalized_state.l1_block_number);
            info!("  Finalization timestamp: {}", finalized_state.finalization_timestamp);
            info!("  Proof hash: {:?}", finalized_state.proof_hash);
        }
        Err(RollupError::L1SyncError(msg)) if msg.contains("No finalized state found") => {
            info!("No finalized state found on L1 (expected in demo environment)");
        }
        Err(e) => {
            warn!("Error getting finalized state: {}", e);
        }
    }

    Ok(())
}

/// Demonstrate reorganization handling (Requirement 2.8, 4.5)
async fn demo_reorganization_handling(sync_manager: &L1StateSyncManager) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Reorganization Handling ===");

    // Create a simulated reorganization event
    let reorg_event = zk_provable_orderbook::rollup::ReorgEvent {
        reorg_block: 100,
        affected_transactions: vec![[1u8; 32], [2u8; 32]],
        new_canonical_block: 101,
    };

    info!("Simulating reorganization at block {}", reorg_event.reorg_block);
    info!("Affected transactions: {}", reorg_event.affected_transactions.len());

    // Handle the reorganization
    sync_manager.handle_reorganization(reorg_event).await
        .map_err(|e| format!("Failed to handle reorganization: {}", e))?;

    info!("Reorganization handled successfully");

    Ok(())
}

/// Create a test ZK proof for demonstration
fn create_test_proof() -> ZkProof {
    ZkProof {
        backend: ZkVMBackend::SP1Local,
        proof_data: vec![1, 2, 3, 4, 5, 6, 7, 8],
        public_inputs: vec![9, 10, 11, 12],
        verification_key_hash: [42u8; 32],
        proof_metadata: ProofMetadata {
            proof_id: format!("demo_proof_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()),
            generation_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            proof_size: 8,
            security_level: 128,
            circuit_size: 1000,
        },
    }
}