//! Basic tests for proof anchoring functionality

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollup::{
        proof_anchoring::*,
        l1_verification_contracts::*,
        ethrex_integration::*,
    };
    use crate::zkvm::{ZkProof, ZkVMBackend, ProofMetadata};

    fn create_test_proof() -> ZkProof {
        ZkProof {
            backend: ZkVMBackend::ZisK,
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
    fn test_proof_anchoring_config() {
        let config = ProofAnchoringConfig::default();
        assert_eq!(config.max_submission_time_ms, 10000);
        assert_eq!(config.block_time_target_ms, 12000);
        assert!(config.enable_auto_anchoring);
        assert!(config.include_merkle_roots);
    }

    #[test]
    fn test_l1_verification_config() {
        let config = L1VerificationConfig::default();
        assert!(!config.zisk_verifier_address.is_empty());
        assert!(!config.sp1_verifier_address.is_empty());
        assert!(!config.state_verifier_address.is_empty());
        assert_eq!(config.verification_gas_limit, 1_000_000);
    }

    #[test]
    fn test_proof_commitment_creation() {
        let proof = create_test_proof();
        let commitment = ProofCommitment {
            commitment_id: "test_commitment".to_string(),
            batch_id: 1,
            proof: proof.clone(),
            state_root_before: [1u8; 32],
            state_root_after: [2u8; 32],
            order_book_merkle_root: [3u8; 32],
            trade_history_merkle_root: [4u8; 32],
            timestamp: 1234567890,
            target_l1_block: 12345,
            priority: CommitmentPriority::Normal,
        };

        assert_eq!(commitment.batch_id, 1);
        assert_eq!(commitment.proof.backend, ZkVMBackend::ZisK);
        assert_eq!(commitment.priority, CommitmentPriority::Normal);
    }

    #[test]
    fn test_commitment_priority_ordering() {
        assert!(CommitmentPriority::Critical > CommitmentPriority::High);
        assert!(CommitmentPriority::High > CommitmentPriority::Normal);
        assert!(CommitmentPriority::Normal > CommitmentPriority::Low);
    }

    #[test]
    fn test_verification_priority_ordering() {
        assert!(VerificationPriority::Critical > VerificationPriority::High);
        assert!(VerificationPriority::High > VerificationPriority::Normal);
        assert!(VerificationPriority::Normal > VerificationPriority::Low);
    }

    #[test]
    fn test_anchoring_stats_default() {
        let stats = AnchoringStats::default();
        assert_eq!(stats.total_commitments, 0);
        assert_eq!(stats.successful_submissions, 0);
        assert_eq!(stats.failed_submissions, 0);
        assert_eq!(stats.average_submission_time_ms, 0.0);
    }

    #[test]
    fn test_verification_stats_default() {
        let stats = VerificationStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_verifications, 0);
        assert_eq!(stats.failed_verifications, 0);
        assert_eq!(stats.average_verification_time_ms, 0.0);
    }

    #[tokio::test]
    async fn test_l1_verification_contract_manager_creation() {
        let config = L1VerificationConfig::default();
        let manager = L1VerificationContractManager::new(config);
        
        let addresses = manager.get_contract_addresses();
        assert!(addresses.contains_key("zisk_verifier"));
        assert!(addresses.contains_key("sp1_verifier"));
        assert!(addresses.contains_key("state_verifier"));
    }

    #[test]
    fn test_order_book_snapshot_creation() {
        let snapshot = OrderBookSnapshot {
            snapshot_id: "test_snapshot".to_string(),
            order_books: std::collections::HashMap::new(),
            merkle_root: [1u8; 32],
            timestamp: 1234567890,
            block_height: 12345,
        };

        assert_eq!(snapshot.snapshot_id, "test_snapshot");
        assert_eq!(snapshot.merkle_root, [1u8; 32]);
        assert_eq!(snapshot.block_height, 12345);
    }
}