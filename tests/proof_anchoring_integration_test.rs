//! Integration tests for the Proof Anchoring and Verification System
//! 
//! This test suite verifies that the proof anchoring system correctly:
//! - Submits ZK proof commitments to ethrex L1 within 1 block time
//! - Includes merkle roots of order book state in commitments
//! - Creates L1 proof verification contracts
//! - Adds mapping between local state and L1 commitments

use hft_limit_order_book::rollup::{
    proof_anchoring::*,
    l1_verification_contracts::*,
    ethrex_integration::*,
    types::*,
    RollupError,
};
use hft_limit_order_book::zkvm::{ZkProof, ZkVMBackend, ProofMetadata};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Test helper to create sample ZK proofs
fn create_test_proof(backend: ZkVMBackend, proof_id: &str) -> ZkProof {
    ZkProof {
        backend,
        proof_data: format!("proof_data_{}", proof_id).into_bytes(),
        public_inputs: format!("public_inputs_{}", proof_id).into_bytes(),
        verification_key_hash: {
            let mut hash = [0u8; 32];
            let id_bytes = proof_id.as_bytes();
            let len = std::cmp::min(id_bytes.len(), 32);
            hash[..len].copy_from_slice(&id_bytes[..len]);
            hash
        },
        proof_metadata: ProofMetadata {
            proof_id: proof_id.to_string(),
            generation_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            proof_size: format!("proof_data_{}", proof_id).len(),
            security_level: 128,
            circuit_size: 1000000,
        },
    }
}

/// Test helper to create sample order book snapshot
fn create_test_order_book_snapshot(snapshot_id: &str) -> OrderBookSnapshot {
    OrderBookSnapshot {
        snapshot_id: snapshot_id.to_string(),
        order_books: HashMap::new(), // Empty for testing
        merkle_root: {
            let mut root = [0u8; 32];
            let id_bytes = snapshot_id.as_bytes();
            let len = std::cmp::min(id_bytes.len(), 32);
            root[..len].copy_from_slice(&id_bytes[..len]);
            root
        },
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        block_height: 12345,
    }
}

/// Test Requirement 2.1: Submit ZK proof commitments to ethrex L1 within 1 block time
#[tokio::test]
async fn test_proof_commitment_submission_within_block_time() {
    // Create ethrex integration manager
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    // Create proof anchoring manager with block time target
    let mut anchoring_config = ProofAnchoringConfig::default();
    anchoring_config.block_time_target_ms = 12000; // 12 seconds (Ethereum block time)
    anchoring_config.max_submission_time_ms = 10000; // 10 seconds max

    let anchoring_manager = ProofAnchoringManager::new(anchoring_config, ethrex_integration)
        .await
        .expect("Failed to create proof anchoring manager");

    // Create test proof and snapshot
    let proof = create_test_proof(ZkVMBackend::ZisK, "test_proof_1");
    let snapshot = create_test_order_book_snapshot("test_snapshot_1");

    // Measure submission time
    let start_time = Instant::now();
    
    let result = timeout(
        Duration::from_millis(15000), // 15 second timeout
        anchoring_manager.submit_proof_commitment(
            1, // batch_id
            proof,
            [1u8; 32], // state_root_before
            [2u8; 32], // state_root_after
            Some(snapshot),
        )
    ).await;

    assert!(result.is_ok(), "Proof commitment submission timed out");
    
    let commitment_record = result.unwrap().expect("Failed to submit proof commitment");
    let submission_time = start_time.elapsed();

    // Verify submission was within block time target
    assert!(
        submission_time.as_millis() <= 12000,
        "Proof commitment submission took {}ms, exceeding 12s block time target",
        submission_time.as_millis()
    );

    // Verify commitment record structure
    assert_eq!(commitment_record.commitment.batch_id, 1);
    assert_eq!(commitment_record.commitment.state_root_before, [1u8; 32]);
    assert_eq!(commitment_record.commitment.state_root_after, [2u8; 32]);
    assert_ne!(commitment_record.tx_hash, [0u8; 32]);
}

/// Test Requirement 2.2: Include merkle roots of order book state in commitments
#[tokio::test]
async fn test_merkle_root_inclusion_in_commitments() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_config = ProofAnchoringConfig {
        include_merkle_roots: true,
        ..Default::default()
    };

    let anchoring_manager = ProofAnchoringManager::new(anchoring_config, ethrex_integration)
        .await
        .expect("Failed to create proof anchoring manager");

    let proof = create_test_proof(ZkVMBackend::SP1Local, "test_proof_2");
    let snapshot = create_test_order_book_snapshot("test_snapshot_2");
    let expected_merkle_root = snapshot.merkle_root;

    let commitment_record = anchoring_manager.submit_proof_commitment(
        2, // batch_id
        proof,
        [2u8; 32], // state_root_before
        [3u8; 32], // state_root_after
        Some(snapshot),
    ).await.expect("Failed to submit proof commitment");

    // Verify merkle root is included in commitment
    assert_eq!(
        commitment_record.commitment.order_book_merkle_root,
        expected_merkle_root,
        "Order book merkle root not correctly included in commitment"
    );

    // Verify trade history merkle root is also included
    assert_ne!(
        commitment_record.commitment.trade_history_merkle_root,
        [0u8; 32],
        "Trade history merkle root should be calculated and included"
    );
}

/// Test Requirement 2.6: Add mapping between local state and L1 commitments
#[tokio::test]
async fn test_state_commitment_mapping() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    let proof = create_test_proof(ZkVMBackend::ZisK, "test_proof_3");
    let snapshot = create_test_order_book_snapshot("test_snapshot_3");
    let state_root_after = [4u8; 32];

    // Submit proof commitment
    let commitment_record = anchoring_manager.submit_proof_commitment(
        3, // batch_id
        proof,
        [3u8; 32], // state_root_before
        state_root_after,
        Some(snapshot.clone()),
    ).await.expect("Failed to submit proof commitment");

    // Test mapping retrieval by state root
    let retrieved_record = anchoring_manager.get_commitment_record(state_root_after).await;
    assert!(retrieved_record.is_some(), "Commitment record not found in mapping");

    let retrieved_record = retrieved_record.unwrap();
    assert_eq!(retrieved_record.commitment.batch_id, 3);
    assert_eq!(retrieved_record.commitment.state_root_after, state_root_after);
    assert_eq!(retrieved_record.tx_hash, commitment_record.tx_hash);

    // Test L1 transaction hash retrieval
    let tx_hash = anchoring_manager.get_l1_transaction_hash(state_root_after).await;
    assert!(tx_hash.is_some(), "L1 transaction hash not found in mapping");
    assert_eq!(tx_hash.unwrap(), commitment_record.tx_hash);

    // Test retrieval of all mappings
    let all_mappings = anchoring_manager.get_all_commitment_mappings().await;
    assert!(all_mappings.contains_key(&state_root_after), "State root not found in all mappings");
    
    let mapping = &all_mappings[&state_root_after];
    assert_eq!(mapping.local_state_root, state_root_after);
    assert!(mapping.order_book_snapshot.is_some());
    assert_eq!(mapping.order_book_snapshot.as_ref().unwrap().snapshot_id, snapshot.snapshot_id);
}

/// Test Requirement 2.7: Get L1 transaction hash for historical state commitment
#[tokio::test]
async fn test_historical_l1_transaction_hash_retrieval() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    // Submit multiple proof commitments to create history
    let mut state_roots = Vec::new();
    let mut expected_tx_hashes = Vec::new();

    for i in 0..5 {
        let proof = create_test_proof(ZkVMBackend::SP1Local, &format!("historical_proof_{}", i));
        let state_root = [i as u8; 32];
        state_roots.push(state_root);

        let commitment_record = anchoring_manager.submit_proof_commitment(
            i as u64 + 10, // batch_id
            proof,
            [(i as u8).wrapping_sub(1); 32], // state_root_before
            state_root,
            None,
        ).await.expect("Failed to submit proof commitment");

        expected_tx_hashes.push(commitment_record.tx_hash);
    }

    // Verify we can retrieve L1 transaction hashes for all historical commitments
    for (i, &state_root) in state_roots.iter().enumerate() {
        let tx_hash = anchoring_manager.get_l1_transaction_hash(state_root).await;
        assert!(tx_hash.is_some(), "L1 transaction hash not found for historical state root {}", i);
        assert_eq!(tx_hash.unwrap(), expected_tx_hashes[i], "L1 transaction hash mismatch for historical state root {}", i);
    }

    // Test retrieval of non-existent state root
    let non_existent_state_root = [255u8; 32];
    let tx_hash = anchoring_manager.get_l1_transaction_hash(non_existent_state_root).await;
    assert!(tx_hash.is_none(), "Should not find transaction hash for non-existent state root");
}

/// Test L1 proof verification contracts functionality
#[tokio::test]
async fn test_l1_proof_verification_contracts() {
    let verification_config = L1VerificationConfig::default();
    let verification_manager = L1VerificationContractManager::new(verification_config);

    // Test ZisK proof verification
    let zisk_proof = create_test_proof(ZkVMBackend::ZisK, "zisk_verification_test");
    let public_inputs = vec![1, 2, 3, 4];
    let state_root = [10u8; 32];

    let zisk_result = verification_manager.verify_proof(&zisk_proof, &public_inputs, &state_root)
        .await
        .expect("Failed to verify ZisK proof");

    assert!(zisk_result.is_valid, "ZisK proof verification should succeed");
    assert!(zisk_result.verification_tx_hash.is_some(), "ZisK verification should have transaction hash");
    assert!(zisk_result.gas_used > 0, "ZisK verification should report gas usage");

    // Test SP1 proof verification
    let sp1_proof = create_test_proof(ZkVMBackend::SP1Local, "sp1_verification_test");
    let sp1_result = verification_manager.verify_proof(&sp1_proof, &public_inputs, &state_root)
        .await
        .expect("Failed to verify SP1 proof");

    assert!(sp1_result.is_valid, "SP1 proof verification should succeed");
    assert!(sp1_result.verification_tx_hash.is_some(), "SP1 verification should have transaction hash");
    assert!(sp1_result.gas_used > 0, "SP1 verification should report gas usage");

    // Test state commitment verification
    let order_book_merkle_root = [20u8; 32];
    let merkle_proof = vec![vec![21u8; 32], vec![22u8; 32], vec![23u8; 32]];

    let state_result = verification_manager.verify_state_commitment(
        &state_root,
        &order_book_merkle_root,
        &merkle_proof,
    ).await.expect("Failed to verify state commitment");

    assert!(state_result.is_valid, "State commitment verification should succeed");
    assert!(state_result.verification_tx_hash.is_some(), "State verification should have transaction hash");

    // Verify statistics are updated
    let stats = verification_manager.get_verification_stats().await;
    assert_eq!(stats.total_requests, 3, "Should have processed 3 verification requests");
    assert_eq!(stats.successful_verifications, 3, "All verifications should be successful");
    assert_eq!(stats.failed_verifications, 0, "No verifications should fail");
    assert!(stats.average_verification_time_ms > 0.0, "Average verification time should be positive");
    assert!(stats.total_gas_used > 0, "Total gas used should be positive");
}

/// Test merkle tree building functionality
#[tokio::test]
async fn test_merkle_tree_building() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    // Test empty merkle tree
    let empty_leaves: Vec<Vec<u8>> = vec![];
    let empty_root = anchoring_manager.build_merkle_tree(&empty_leaves)
        .expect("Failed to build empty merkle tree");
    assert_eq!(empty_root, [0u8; 32], "Empty merkle tree should have zero root");

    // Test single leaf merkle tree
    let single_leaf = vec![vec![1u8; 32]];
    let single_root = anchoring_manager.build_merkle_tree(&single_leaf)
        .expect("Failed to build single leaf merkle tree");
    assert_ne!(single_root, [0u8; 32], "Single leaf merkle tree should have non-zero root");

    // Test multiple leaves merkle tree
    let multiple_leaves = vec![
        vec![1u8; 32],
        vec![2u8; 32],
        vec![3u8; 32],
        vec![4u8; 32],
    ];
    let multiple_root = anchoring_manager.build_merkle_tree(&multiple_leaves)
        .expect("Failed to build multiple leaves merkle tree");
    assert_ne!(multiple_root, [0u8; 32], "Multiple leaves merkle tree should have non-zero root");
    assert_ne!(multiple_root, single_root, "Different leaf sets should produce different roots");

    // Test deterministic property - same leaves should produce same root
    let duplicate_leaves = vec![
        vec![1u8; 32],
        vec![2u8; 32],
        vec![3u8; 32],
        vec![4u8; 32],
    ];
    let duplicate_root = anchoring_manager.build_merkle_tree(&duplicate_leaves)
        .expect("Failed to build duplicate merkle tree");
    assert_eq!(multiple_root, duplicate_root, "Same leaf sets should produce same merkle root");

    // Test odd number of leaves
    let odd_leaves = vec![
        vec![1u8; 32],
        vec![2u8; 32],
        vec![3u8; 32],
    ];
    let odd_root = anchoring_manager.build_merkle_tree(&odd_leaves)
        .expect("Failed to build odd leaves merkle tree");
    assert_ne!(odd_root, [0u8; 32], "Odd leaves merkle tree should have non-zero root");
}

/// Test system statistics and monitoring
#[tokio::test]
async fn test_system_statistics() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    let verification_manager = L1VerificationContractManager::new(L1VerificationConfig::default());

    // Initial statistics should be zero
    let initial_anchoring_stats = anchoring_manager.get_anchoring_stats().await;
    assert_eq!(initial_anchoring_stats.total_commitments, 0);
    assert_eq!(initial_anchoring_stats.successful_submissions, 0);
    assert_eq!(initial_anchoring_stats.failed_submissions, 0);

    let initial_verification_stats = verification_manager.get_verification_stats().await;
    assert_eq!(initial_verification_stats.total_requests, 0);
    assert_eq!(initial_verification_stats.successful_verifications, 0);

    // Submit some proof commitments
    for i in 0..3 {
        let proof = create_test_proof(ZkVMBackend::ZisK, &format!("stats_test_{}", i));
        let _commitment = anchoring_manager.submit_proof_commitment(
            i as u64 + 100,
            proof.clone(),
            [i as u8; 32],
            [(i + 1) as u8; 32],
            None,
        ).await.expect("Failed to submit proof commitment");

        // Also verify the proof
        let _verification = verification_manager.verify_proof(
            &proof,
            &[i as u8, (i + 1) as u8],
            &[(i + 1) as u8; 32],
        ).await.expect("Failed to verify proof");
    }

    // Check updated statistics
    let final_anchoring_stats = anchoring_manager.get_anchoring_stats().await;
    assert_eq!(final_anchoring_stats.total_commitments, 3);
    assert_eq!(final_anchoring_stats.successful_submissions, 3);
    assert_eq!(final_anchoring_stats.failed_submissions, 0);
    assert!(final_anchoring_stats.average_submission_time_ms > 0.0);
    assert!(final_anchoring_stats.total_gas_used > 0);

    let final_verification_stats = verification_manager.get_verification_stats().await;
    assert_eq!(final_verification_stats.total_requests, 3);
    assert_eq!(final_verification_stats.successful_verifications, 3);
    assert_eq!(final_verification_stats.failed_verifications, 0);
    assert!(final_verification_stats.average_verification_time_ms > 0.0);
}

/// Test health check functionality
#[tokio::test]
async fn test_health_checks() {
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    // Test health check
    let health_result = anchoring_manager.health_check().await;
    assert!(health_result.is_ok(), "Health check should not return error");
    
    // The actual health status depends on the mock implementation
    // In a real environment, this would check actual connectivity
    let is_healthy = health_result.unwrap();
    // We can't assert the specific value since it depends on the mock setup
    println!("System health status: {}", is_healthy);
}

/// Integration test combining all requirements
#[tokio::test]
async fn test_complete_proof_anchoring_workflow() {
    // Setup
    let ethrex_config = EthrexIntegrationConfig::default();
    let ethrex_integration = Arc::new(
        EthrexIntegrationManager::new(ethrex_config)
            .await
            .expect("Failed to create ethrex integration manager")
    );

    let anchoring_manager = ProofAnchoringManager::new(
        ProofAnchoringConfig::default(),
        ethrex_integration
    ).await.expect("Failed to create proof anchoring manager");

    let verification_manager = L1VerificationContractManager::new(L1VerificationConfig::default());

    // Step 1: Create proof and snapshot
    let proof = create_test_proof(ZkVMBackend::SP1Local, "complete_workflow_test");
    let snapshot = create_test_order_book_snapshot("complete_workflow_snapshot");
    let state_root_after = [100u8; 32];

    // Step 2: Submit proof commitment (Requirements 2.1, 2.2)
    let start_time = Instant::now();
    let commitment_record = anchoring_manager.submit_proof_commitment(
        999, // batch_id
        proof.clone(),
        [99u8; 32], // state_root_before
        state_root_after,
        Some(snapshot.clone()),
    ).await.expect("Failed to submit proof commitment");

    let submission_time = start_time.elapsed();
    assert!(submission_time.as_millis() <= 12000, "Submission should be within block time");

    // Step 3: Verify mapping exists (Requirement 2.6)
    let retrieved_record = anchoring_manager.get_commitment_record(state_root_after).await;
    assert!(retrieved_record.is_some(), "Commitment mapping should exist");

    // Step 4: Verify L1 transaction hash retrieval (Requirement 2.7)
    let tx_hash = anchoring_manager.get_l1_transaction_hash(state_root_after).await;
    assert!(tx_hash.is_some(), "L1 transaction hash should be retrievable");
    assert_eq!(tx_hash.unwrap(), commitment_record.tx_hash);

    // Step 5: Verify proof on L1 contract
    let verification_result = verification_manager.verify_proof(
        &proof,
        &[1, 2, 3, 4],
        &state_root_after,
    ).await.expect("Failed to verify proof on L1");

    assert!(verification_result.is_valid, "Proof should be valid on L1");

    // Step 6: Verify state commitment
    let state_verification = verification_manager.verify_state_commitment(
        &state_root_after,
        &snapshot.merkle_root,
        &vec![vec![1u8; 32], vec![2u8; 32]],
    ).await.expect("Failed to verify state commitment");

    assert!(state_verification.is_valid, "State commitment should be valid");

    // Step 7: Check final statistics
    let anchoring_stats = anchoring_manager.get_anchoring_stats().await;
    assert!(anchoring_stats.total_commitments > 0);
    assert!(anchoring_stats.successful_submissions > 0);

    let verification_stats = verification_manager.get_verification_stats().await;
    assert!(verification_stats.total_requests > 0);
    assert!(verification_stats.successful_verifications > 0);

    println!("Complete workflow test passed successfully!");
    println!("Anchoring stats: {:?}", anchoring_stats);
    println!("Verification stats: {:?}", verification_stats);
}