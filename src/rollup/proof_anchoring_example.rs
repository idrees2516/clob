//! Proof Anchoring System Integration Example
//! 
//! This example demonstrates how to use the proof anchoring and verification system
//! to submit ZK proof commitments to ethrex L1 and verify them on-chain.

use crate::rollup::{
    proof_anchoring::*,
    l1_verification_contracts::*,
    ethrex_integration::*,
    types::*,
    RollupError,
};
use crate::zkvm::{ZkProof, ZkVMBackend, ProofMetadata};
use crate::orderbook::types::OrderBook;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn, error};

/// Example configuration for proof anchoring
pub struct ProofAnchoringExample {
    anchoring_manager: Arc<ProofAnchoringManager>,
    verification_manager: Arc<L1VerificationContractManager>,
}

impl ProofAnchoringExample {
    /// Create a new proof anchoring example
    pub async fn new() -> Result<Self, RollupError> {
        info!("Setting up proof anchoring example");

        // Create ethrex integration manager
        let ethrex_config = EthrexIntegrationConfig::default();
        let ethrex_integration = Arc::new(EthrexIntegrationManager::new(ethrex_config).await?);

        // Create proof anchoring manager
        let anchoring_config = ProofAnchoringConfig::default();
        let anchoring_manager = Arc::new(
            ProofAnchoringManager::new(anchoring_config, ethrex_integration).await?
        );

        // Create L1 verification contract manager
        let verification_config = L1VerificationConfig::default();
        let verification_manager = Arc::new(
            L1VerificationContractManager::new(verification_config)
        );

        Ok(Self {
            anchoring_manager,
            verification_manager,
        })
    }

    /// Run the complete proof anchoring example
    pub async fn run_example(&self) -> Result<(), RollupError> {
        info!("Running proof anchoring example");

        // Step 1: Create sample ZK proofs for different backends
        let zisk_proof = self.create_sample_zisk_proof();
        let sp1_proof = self.create_sample_sp1_proof();

        // Step 2: Create sample order book snapshot
        let order_book_snapshot = self.create_sample_order_book_snapshot().await?;

        // Step 3: Submit ZisK proof commitment to L1
        info!("Submitting ZisK proof commitment to L1");
        let zisk_commitment = self.anchoring_manager.submit_proof_commitment(
            1, // batch_id
            zisk_proof.clone(),
            [1u8; 32], // state_root_before
            [2u8; 32], // state_root_after
            Some(order_book_snapshot.clone()),
        ).await?;

        info!("ZisK proof commitment submitted: {:?}", zisk_commitment.tx_hash);

        // Step 4: Submit SP1 proof commitment to L1
        info!("Submitting SP1 proof commitment to L1");
        let sp1_commitment = self.anchoring_manager.submit_proof_commitment(
            2, // batch_id
            sp1_proof.clone(),
            [2u8; 32], // state_root_before
            [3u8; 32], // state_root_after
            Some(order_book_snapshot.clone()),
        ).await?;

        info!("SP1 proof commitment submitted: {:?}", sp1_commitment.tx_hash);

        // Step 5: Verify proofs on L1 verification contracts
        info!("Verifying ZisK proof on L1 contract");
        let zisk_verification = self.verification_manager.verify_proof(
            &zisk_proof,
            &[1, 2, 3, 4], // public_inputs
            &[2u8; 32], // state_root
        ).await?;

        info!("ZisK proof verification result: valid={}, gas_used={}", 
              zisk_verification.is_valid, zisk_verification.gas_used);

        info!("Verifying SP1 proof on L1 contract");
        let sp1_verification = self.verification_manager.verify_proof(
            &sp1_proof,
            &[5, 6, 7, 8], // public_inputs
            &[3u8; 32], // state_root
        ).await?;

        info!("SP1 proof verification result: valid={}, gas_used={}", 
              sp1_verification.is_valid, sp1_verification.gas_used);

        // Step 6: Verify state commitment with merkle proofs
        info!("Verifying state commitment with merkle proofs");
        let merkle_proof = vec![
            vec![1u8; 32],
            vec![2u8; 32],
            vec![3u8; 32],
        ];

        let state_verification = self.verification_manager.verify_state_commitment(
            &[3u8; 32], // state_root
            &order_book_snapshot.merkle_root, // order_book_merkle_root
            &merkle_proof,
        ).await?;

        info!("State commitment verification result: valid={}, gas_used={}", 
              state_verification.is_valid, state_verification.gas_used);

        // Step 7: Demonstrate commitment mapping retrieval
        info!("Retrieving commitment records from mapping");
        
        let zisk_record = self.anchoring_manager.get_commitment_record([2u8; 32]).await;
        if let Some(record) = zisk_record {
            info!("Retrieved ZisK commitment record: batch_id={}, tx_hash={:?}", 
                  record.commitment.batch_id, record.tx_hash);
        }

        let sp1_record = self.anchoring_manager.get_commitment_record([3u8; 32]).await;
        if let Some(record) = sp1_record {
            info!("Retrieved SP1 commitment record: batch_id={}, tx_hash={:?}", 
                  record.commitment.batch_id, record.tx_hash);
        }

        // Step 8: Demonstrate L1 transaction hash retrieval
        info!("Retrieving L1 transaction hashes");
        
        let zisk_tx_hash = self.anchoring_manager.get_l1_transaction_hash([2u8; 32]).await;
        if let Some(tx_hash) = zisk_tx_hash {
            info!("ZisK L1 transaction hash: {:?}", tx_hash);
        }

        let sp1_tx_hash = self.anchoring_manager.get_l1_transaction_hash([3u8; 32]).await;
        if let Some(tx_hash) = sp1_tx_hash {
            info!("SP1 L1 transaction hash: {:?}", tx_hash);
        }

        // Step 9: Display statistics
        self.display_statistics().await?;

        info!("Proof anchoring example completed successfully");
        Ok(())
    }

    /// Create a sample ZisK proof
    fn create_sample_zisk_proof(&self) -> ZkProof {
        ZkProof {
            backend: ZkVMBackend::ZisK,
            proof_data: vec![0x01, 0x02, 0x03, 0x04, 0x05], // Sample proof data
            public_inputs: vec![0x10, 0x20, 0x30, 0x40], // Sample public inputs
            verification_key_hash: [0x11u8; 32], // Sample verification key hash
            proof_metadata: ProofMetadata {
                proof_id: "zisk_proof_example".to_string(),
                generation_time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                proof_size: 5,
                security_level: 128,
                circuit_size: 1000000,
            },
        }
    }

    /// Create a sample SP1 proof
    fn create_sample_sp1_proof(&self) -> ZkProof {
        ZkProof {
            backend: ZkVMBackend::SP1Local,
            proof_data: vec![0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B], // Sample proof data
            public_inputs: vec![0x50, 0x60, 0x70, 0x80], // Sample public inputs
            verification_key_hash: [0x22u8; 32], // Sample verification key hash
            proof_metadata: ProofMetadata {
                proof_id: "sp1_proof_example".to_string(),
                generation_time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                proof_size: 6,
                security_level: 256,
                circuit_size: 2000000,
            },
        }
    }

    /// Create a sample order book snapshot
    async fn create_sample_order_book_snapshot(&self) -> Result<OrderBookSnapshot, RollupError> {
        let mut order_books = HashMap::new();
        
        // Create sample order books for different symbols
        // Note: This would use actual OrderBook instances in a real implementation
        // For now, we'll create a placeholder structure
        
        let snapshot = OrderBookSnapshot {
            snapshot_id: "snapshot_example_001".to_string(),
            order_books, // Empty for this example
            merkle_root: [0x33u8; 32], // Sample merkle root
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            block_height: 12345,
        };

        Ok(snapshot)
    }

    /// Display system statistics
    async fn display_statistics(&self) -> Result<(), RollupError> {
        info!("=== Proof Anchoring Statistics ===");
        
        let anchoring_stats = self.anchoring_manager.get_anchoring_stats().await;
        info!("Total commitments: {}", anchoring_stats.total_commitments);
        info!("Successful submissions: {}", anchoring_stats.successful_submissions);
        info!("Failed submissions: {}", anchoring_stats.failed_submissions);
        info!("Average submission time: {:.2}ms", anchoring_stats.average_submission_time_ms);
        info!("Within block time target: {}", anchoring_stats.within_block_time_target);
        info!("Total gas used: {}", anchoring_stats.total_gas_used);

        info!("=== L1 Verification Statistics ===");
        
        let verification_stats = self.verification_manager.get_verification_stats().await;
        info!("Total verification requests: {}", verification_stats.total_requests);
        info!("Successful verifications: {}", verification_stats.successful_verifications);
        info!("Failed verifications: {}", verification_stats.failed_verifications);
        info!("Average verification time: {:.2}ms", verification_stats.average_verification_time_ms);
        info!("Total verification gas used: {}", verification_stats.total_gas_used);

        info!("Verifications by backend:");
        for (backend, count) in &verification_stats.verifications_by_backend {
            info!("  {}: {}", backend, count);
        }

        info!("=== Contract Addresses ===");
        let contract_addresses = self.verification_manager.get_contract_addresses();
        for (name, address) in &contract_addresses {
            info!("  {}: {}", name, address);
        }

        Ok(())
    }

    /// Demonstrate error handling and retry mechanisms
    pub async fn demonstrate_error_handling(&self) -> Result<(), RollupError> {
        info!("Demonstrating error handling and retry mechanisms");

        // This would simulate various error conditions and show how the system handles them
        // For example:
        // - Network failures during L1 submission
        // - Gas estimation failures
        // - Proof verification failures
        // - State inconsistencies

        warn!("Error handling demonstration would be implemented here");
        Ok(())
    }

    /// Demonstrate performance under load
    pub async fn demonstrate_performance_test(&self) -> Result<(), RollupError> {
        info!("Demonstrating performance under load");

        let start_time = std::time::Instant::now();
        let num_proofs = 10;

        for i in 0..num_proofs {
            let proof = if i % 2 == 0 {
                self.create_sample_zisk_proof()
            } else {
                self.create_sample_sp1_proof()
            };

            let state_root_before = [i as u8; 32];
            let state_root_after = [(i + 1) as u8; 32];

            match self.anchoring_manager.submit_proof_commitment(
                i as u64,
                proof,
                state_root_before,
                state_root_after,
                None,
            ).await {
                Ok(commitment) => {
                    info!("Submitted proof {} successfully: {:?}", i, commitment.tx_hash);
                }
                Err(e) => {
                    error!("Failed to submit proof {}: {}", i, e);
                }
            }
        }

        let total_time = start_time.elapsed();
        info!("Submitted {} proofs in {:?} (avg: {:?} per proof)", 
              num_proofs, total_time, total_time / num_proofs);

        Ok(())
    }

    /// Health check for the entire system
    pub async fn health_check(&self) -> Result<bool, RollupError> {
        info!("Performing system health check");

        let anchoring_healthy = self.anchoring_manager.health_check().await?;
        
        // The verification manager doesn't have a health check method,
        // but we can check if it's functioning by trying a simple operation
        let verification_healthy = true; // Placeholder

        let overall_healthy = anchoring_healthy && verification_healthy;

        if overall_healthy {
            info!("System health check passed");
        } else {
            warn!("System health check failed - anchoring: {}, verification: {}", 
                  anchoring_healthy, verification_healthy);
        }

        Ok(overall_healthy)
    }
}

/// Run the proof anchoring example
pub async fn run_proof_anchoring_example() -> Result<(), RollupError> {
    info!("Starting proof anchoring system example");

    let example = ProofAnchoringExample::new().await?;
    
    // Run the main example
    example.run_example().await?;
    
    // Demonstrate error handling
    example.demonstrate_error_handling().await?;
    
    // Demonstrate performance testing
    example.demonstrate_performance_test().await?;
    
    // Perform health check
    example.health_check().await?;

    info!("Proof anchoring system example completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proof_anchoring_example_creation() {
        let example = ProofAnchoringExample::new().await;
        assert!(example.is_ok());
    }

    #[tokio::test]
    async fn test_sample_proof_creation() {
        let example = ProofAnchoringExample::new().await.unwrap();
        
        let zisk_proof = example.create_sample_zisk_proof();
        assert_eq!(zisk_proof.backend, ZkVMBackend::ZisK);
        assert!(!zisk_proof.proof_data.is_empty());
        
        let sp1_proof = example.create_sample_sp1_proof();
        assert_eq!(sp1_proof.backend, ZkVMBackend::SP1Local);
        assert!(!sp1_proof.proof_data.is_empty());
    }

    #[tokio::test]
    async fn test_order_book_snapshot_creation() {
        let example = ProofAnchoringExample::new().await.unwrap();
        
        let snapshot = example.create_sample_order_book_snapshot().await;
        assert!(snapshot.is_ok());
        
        let snapshot = snapshot.unwrap();
        assert!(!snapshot.snapshot_id.is_empty());
        assert_ne!(snapshot.merkle_root, [0u8; 32]);
    }

    #[tokio::test]
    async fn test_health_check() {
        let example = ProofAnchoringExample::new().await.unwrap();
        
        let health = example.health_check().await;
        assert!(health.is_ok());
    }
}