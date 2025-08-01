//! ethrex L1 Integration Demo
//! 
//! This example demonstrates how to use the ethrex L1 client integration
//! for proof commitment submission, state anchoring, and transaction finality tracking.

use hf_quoting_liquidity_clob::rollup::{
    EthrexIntegrationManager, EthrexIntegrationConfig, EthrexClientConfig,
    FinalityTrackerConfig, ProofCommitmentRequest, StateAnchorRequest,
    SubmissionPriority, FinalityEvent,
};
use hf_quoting_liquidity_clob::zkvm::{ZkProof, ZkVMBackend, ProofMetadata};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("Starting ethrex L1 integration demo");

    // Create configuration
    let config = EthrexIntegrationConfig {
        client_config: EthrexClientConfig {
            rpc_url: "http://localhost:8545".to_string(), // Local ethrex node
            private_key: Some("0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".to_string()),
            proof_commitment_contract: "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6".to_string(),
            gas_limit: 500_000,
            gas_price_multiplier: 1.2,
            max_retry_attempts: 5,
            retry_backoff_base_ms: 1000,
            confirmation_blocks: 3,
            transaction_timeout_seconds: 300,
        },
        finality_config: FinalityTrackerConfig {
            polling_interval_ms: 5000,
            finality_confirmations: 12,
            confirmation_timeout_seconds: 3600,
            max_tracked_transactions: 1000,
            enable_deep_reorg_detection: true,
            max_reorg_depth: 64,
        },
        enable_auto_sync: true,
        sync_interval_ms: 12000,
        enable_finality_tracking: true,
        max_concurrent_submissions: 10,
    };

    // Create integration manager
    let integration_manager = match EthrexIntegrationManager::new(config).await {
        Ok(manager) => manager,
        Err(e) => {
            error!("Failed to create integration manager: {}", e);
            return Err(e.into());
        }
    };

    // Start the integration manager
    if let Err(e) = integration_manager.start().await {
        error!("Failed to start integration manager: {}", e);
        return Err(e.into());
    }

    info!("ethrex L1 integration manager started successfully");

    // Demonstrate proof commitment submission
    demo_proof_commitment_submission(&integration_manager).await?;

    // Demonstrate state root anchoring
    demo_state_root_anchoring(&integration_manager).await?;

    // Demonstrate commitment record retrieval
    demo_commitment_record_retrieval(&integration_manager).await?;

    // Monitor finality events for a while
    demo_finality_monitoring(&integration_manager).await?;

    // Show integration statistics
    demo_integration_statistics(&integration_manager).await?;

    // Perform health check
    demo_health_check(&integration_manager).await?;

    // Stop the integration manager
    if let Err(e) = integration_manager.stop().await {
        error!("Failed to stop integration manager: {}", e);
    }

    info!("ethrex L1 integration demo completed");
    Ok(())
}

/// Demonstrate proof commitment submission (Requirements 2.1, 2.2)
async fn demo_proof_commitment_submission(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Proof Commitment Submission ===");

    // Create a mock ZK proof
    let proof = ZkProof {
        backend: ZkVMBackend::SP1Local,
        proof_data: vec![1, 2, 3, 4, 5, 6, 7, 8], // Mock proof data
        public_inputs: vec![9, 10, 11, 12], // Mock public inputs
        verification_key_hash: [42u8; 32],
        proof_metadata: ProofMetadata {
            proof_id: "demo_proof_001".to_string(),
            generation_time: chrono::Utc::now().timestamp() as u64,
            proof_size: 8,
            security_level: 128,
            circuit_size: 1_000_000,
        },
    };

    // Create proof commitment request
    let commitment_request = ProofCommitmentRequest {
        batch_id: 1,
        proof,
        state_root_before: [1u8; 32],
        state_root_after: [2u8; 32],
        order_book_merkle_root: [3u8; 32], // Requirement 2.2: Include merkle root
        priority: SubmissionPriority::High,
    };

    // Submit proof commitment
    match integration_manager.submit_proof_commitment(commitment_request).await {
        Ok(result) => {
            info!(
                "✅ Proof commitment submitted successfully!"
            );
            info!("   Transaction hash: {:?}", result.tx_hash);
            info!("   State root: {:?}", result.state_root);
            info!("   Batch ID: {:?}", result.batch_id);
            info!("   Submission time: {:?}", result.submission_time.elapsed());
        }
        Err(e) => {
            warn!("❌ Failed to submit proof commitment: {}", e);
            // In a real scenario, this might fail if ethrex node is not running
            info!("   This is expected if no ethrex node is running locally");
        }
    }

    Ok(())
}

/// Demonstrate state root anchoring
async fn demo_state_root_anchoring(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating State Root Anchoring ===");

    let proof = ZkProof {
        backend: ZkVMBackend::ZisK,
        proof_data: vec![10, 20, 30, 40],
        public_inputs: vec![50, 60],
        verification_key_hash: [84u8; 32],
        proof_metadata: ProofMetadata {
            proof_id: "state_anchor_proof_001".to_string(),
            generation_time: chrono::Utc::now().timestamp() as u64,
            proof_size: 4,
            security_level: 128,
            circuit_size: 500_000,
        },
    };

    let anchor_request = StateAnchorRequest {
        state_root: [100u8; 32],
        proof,
        priority: SubmissionPriority::Normal,
    };

    match integration_manager.submit_state_anchor(anchor_request).await {
        Ok(result) => {
            info!("✅ State root anchor submitted successfully!");
            info!("   Transaction hash: {:?}", result.tx_hash);
            info!("   State root: {:?}", result.state_root);
            info!("   Submission time: {:?}", result.submission_time.elapsed());
        }
        Err(e) => {
            warn!("❌ Failed to submit state root anchor: {}", e);
            info!("   This is expected if no ethrex node is running locally");
        }
    }

    Ok(())
}

/// Demonstrate commitment record retrieval (Requirement 2.6)
async fn demo_commitment_record_retrieval(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Commitment Record Retrieval ===");

    let test_state_root = [2u8; 32]; // From the proof commitment example

    // Get commitment record for state root (Requirement 2.6)
    if let Some(record) = integration_manager.get_commitment_record(test_state_root).await {
        info!("✅ Found commitment record for state root!");
        info!("   State root: {:?}", record.state_root);
        info!("   L1 transaction hash: {:?}", record.commitment_tx_hash);
        info!("   L1 block number: {}", record.l1_block_number);
        info!("   Order book merkle root: {:?}", record.order_book_merkle_root);
        info!("   Confirmation status: {:?}", record.confirmation_status);
    } else {
        info!("ℹ️  No commitment record found for test state root");
        info!("   This is expected if the proof commitment wasn't successfully submitted");
    }

    // Get L1 transaction hash for historical state commitment (Requirement 2.7)
    if let Some(tx_hash) = integration_manager.get_l1_transaction_hash(test_state_root).await {
        info!("✅ Found L1 transaction hash for state root: {:?}", tx_hash);
    } else {
        info!("ℹ️  No L1 transaction hash found for test state root");
    }

    // Get all commitment records for audit
    let all_records = integration_manager.get_all_commitment_records().await;
    info!("📊 Total commitment records: {}", all_records.len());

    for (state_root, record) in all_records.iter().take(3) {
        info!("   State root: {:?} -> Tx: {:?}", state_root, record.commitment_tx_hash);
    }

    Ok(())
}

/// Demonstrate finality monitoring
async fn demo_finality_monitoring(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Finality Monitoring ===");

    // Monitor for a short period to show finality events
    info!("Monitoring finality events for 10 seconds...");
    
    let start_time = std::time::Instant::now();
    let monitor_duration = Duration::from_secs(10);

    while start_time.elapsed() < monitor_duration {
        let finality_stats = integration_manager.get_finality_stats().await;
        
        info!("📈 Finality Stats:");
        info!("   Tracked transactions: {}", finality_stats.tracked_count);
        info!("   Finalized transactions: {}", finality_stats.finalized_count);
        info!("   Current block height: {:?}", finality_stats.current_block_height);
        info!("   Average confirmation time: {:.2}s", finality_stats.average_confirmation_time_seconds);
        info!("   Average finalization time: {:.2}s", finality_stats.average_finalization_time_seconds);

        sleep(Duration::from_secs(2)).await;
    }

    Ok(())
}

/// Demonstrate integration statistics
async fn demo_integration_statistics(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Integration Statistics ===");

    let stats = integration_manager.get_integration_stats().await;

    info!("📊 Integration Statistics:");
    info!("   Total proof commitments: {}", stats.total_proof_commitments);
    info!("   Total state anchors: {}", stats.total_state_anchors);
    info!("   Successful submissions: {}", stats.successful_submissions);
    info!("   Failed submissions: {}", stats.failed_submissions);
    info!("   Average submission time: {:.2}ms", stats.average_submission_time_ms);
    info!("   Pending transactions: {}", stats.pending_transactions);
    info!("   Finalized transactions: {}", stats.finalized_transactions);
    info!("   Reorganizations handled: {}", stats.reorganizations_handled);
    
    if let Some(last_sync) = stats.last_sync_time {
        let sync_time = std::time::UNIX_EPOCH + Duration::from_secs(last_sync);
        info!("   Last sync time: {:?}", sync_time);
    }

    Ok(())
}

/// Demonstrate health check
async fn demo_health_check(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Health Check ===");

    match integration_manager.health_check().await {
        Ok(is_healthy) => {
            if is_healthy {
                info!("✅ ethrex L1 integration is healthy");
            } else {
                warn!("⚠️  ethrex L1 integration health check failed");
            }
        }
        Err(e) => {
            error!("❌ Health check error: {}", e);
            info!("   This is expected if no ethrex node is running locally");
        }
    }

    Ok(())
}

/// Helper function to create a mock reorganization event for demonstration
#[allow(dead_code)]
fn create_mock_reorg_event() -> hf_quoting_liquidity_clob::rollup::ReorgEvent {
    use hf_quoting_liquidity_clob::rollup::ReorgEvent;
    
    ReorgEvent {
        reorg_block: 1000,
        original_block_hash: [1u8; 32],
        new_block_hash: [2u8; 32],
        affected_transactions: vec![[123u8; 32], [124u8; 32]],
        detection_time: std::time::Instant::now(),
    }
}

/// Helper function to demonstrate reorganization handling (Requirement 2.8)
#[allow(dead_code)]
async fn demo_reorganization_handling(
    integration_manager: &EthrexIntegrationManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demonstrating Reorganization Handling ===");

    let reorg_event = create_mock_reorg_event();
    
    match integration_manager.handle_reorganization(reorg_event).await {
        Ok(()) => {
            info!("✅ Reorganization handled successfully");
        }
        Err(e) => {
            error!("❌ Failed to handle reorganization: {}", e);
        }
    }

    Ok(())
}