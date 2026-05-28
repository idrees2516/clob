//! L1 Proof Verification Contracts
//! 
//! This module provides interfaces for L1 proof verification contracts
//! that verify ZK proofs and state commitments on the Ethereum L1.

use crate::rollup::{types::*, RollupError};
use crate::zkvm::{ZkProof, ZkVMBackend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};
use rand::Rng;

/// L1 verification contract configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1VerificationConfig {
    /// Contract address for ZisK proof verification
    pub zisk_verifier_address: String,
    /// Contract address for SP1 proof verification
    pub sp1_verifier_address: String,
    /// Contract address for state commitment verification
    pub state_verifier_address: String,
    /// Gas limit for verification transactions
    pub verification_gas_limit: u64,
    /// Gas price multiplier for verification transactions
    pub gas_price_multiplier: f64,
    /// Maximum verification attempts
    pub max_verification_attempts: u32,
    /// Verification timeout in seconds
    pub verification_timeout_seconds: u64,
}

impl Default for L1VerificationConfig {
    fn default() -> Self {
        Self {
            zisk_verifier_address: "0x0000000000000000000000000000000000000001".to_string(),
            sp1_verifier_address: "0x0000000000000000000000000000000000000002".to_string(),
            state_verifier_address: "0x0000000000000000000000000000000000000003".to_string(),
            verification_gas_limit: 1_000_000,
            gas_price_multiplier: 1.2,
            max_verification_attempts: 3,
            verification_timeout_seconds: 300,
        }
    }
}

/// L1 verification request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1VerificationRequest {
    /// Request ID
    pub request_id: String,
    /// ZK proof to verify
    pub proof: ZkProof,
    /// Public inputs for verification
    pub public_inputs: Vec<u8>,
    /// State root to verify
    pub state_root: StateRoot,
    /// Order book merkle root
    pub order_book_merkle_root: [u8; 32],
    /// Batch ID associated with the proof
    pub batch_id: BatchId,
    /// Priority level
    pub priority: VerificationPriority,
    /// Request timestamp
    pub timestamp: u64,
}

/// Verification priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerificationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// L1 verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1VerificationResult {
    /// Request ID
    pub request_id: String,
    /// Verification success status
    pub is_valid: bool,
    /// L1 transaction hash for verification
    pub verification_tx_hash: Option<TxHash>,
    /// L1 block number where verified
    pub verification_block: Option<L1BlockNumber>,
    /// Gas used for verification
    pub gas_used: u64,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
    /// Error message if verification failed
    pub error_message: Option<String>,
    /// Verification timestamp
    pub timestamp: u64,
}

/// Contract call data for proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCallData {
    /// Function selector
    pub function_selector: [u8; 4],
    /// Encoded proof data
    pub proof_data: Vec<u8>,
    /// Encoded public inputs
    pub public_inputs: Vec<u8>,
    /// Additional verification parameters
    pub verification_params: Vec<u8>,
}

/// L1 verification contract interface
#[async_trait::async_trait]
pub trait L1VerificationContract: Send + Sync {
    /// Verify a ZK proof on L1
    async fn verify_proof(
        &self,
        proof: &ZkProof,
        public_inputs: &[u8],
        state_root: &StateRoot,
    ) -> Result<L1VerificationResult, RollupError>;

    /// Get verification status for a request
    async fn get_verification_status(
        &self,
        request_id: &str,
    ) -> Result<Option<L1VerificationResult>, RollupError>;

    /// Get contract address
    fn contract_address(&self) -> &str;

    /// Get supported zkVM backend
    fn supported_backend(&self) -> ZkVMBackend;
}

/// ZisK proof verification contract
pub struct ZiskVerificationContract {
    config: L1VerificationConfig,
    contract_address: String,
    verification_results: Arc<RwLock<HashMap<String, L1VerificationResult>>>,
}

impl ZiskVerificationContract {
    /// Create a new ZisK verification contract interface
    pub fn new(config: L1VerificationConfig) -> Self {
        Self {
            contract_address: config.zisk_verifier_address.clone(),
            config,
            verification_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Encode proof data for contract call
    fn encode_proof_data(&self, proof: &ZkProof) -> Result<VerificationCallData, RollupError> {
        // Function selector for verifyProof(bytes,bytes,bytes32)
        let function_selector = [0x12, 0x34, 0x56, 0x78]; // Placeholder

        // Encode proof data (would use actual ABI encoding)
        let proof_data = proof.proof_data.clone();
        let public_inputs = proof.public_inputs.clone();
        let verification_params = vec![]; // Additional parameters if needed

        Ok(VerificationCallData {
            function_selector,
            proof_data,
            public_inputs,
            verification_params,
        })
    }
}

#[async_trait::async_trait]
impl L1VerificationContract for ZiskVerificationContract {
    async fn verify_proof(
        &self,
        proof: &ZkProof,
        public_inputs: &[u8],
        state_root: &StateRoot,
    ) -> Result<L1VerificationResult, RollupError> {
        info!("Verifying ZisK proof on L1 contract at {}", self.contract_address);

        let request_id = generate_verification_request_id();
        let start_time = std::time::Instant::now();

        // Encode call data
        let _call_data = self.encode_proof_data(proof)?;

        // Simulate contract call (in real implementation, this would call the actual contract)
        let verification_result = L1VerificationResult {
            request_id: request_id.clone(),
            is_valid: true, // Placeholder - would be actual verification result
            verification_tx_hash: Some([1u8; 32]), // Placeholder transaction hash
            verification_block: Some(12345), // Placeholder block number
            gas_used: 500000,
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Store result
        {
            let mut results = self.verification_results.write().await;
            results.insert(request_id.clone(), verification_result.clone());
        }

        info!("ZisK proof verification completed for request {}", request_id);
        Ok(verification_result)
    }

    async fn get_verification_status(
        &self,
        request_id: &str,
    ) -> Result<Option<L1VerificationResult>, RollupError> {
        let results = self.verification_results.read().await;
        Ok(results.get(request_id).cloned())
    }

    fn contract_address(&self) -> &str {
        &self.contract_address
    }

    fn supported_backend(&self) -> ZkVMBackend {
        ZkVMBackend::ZisK
    }
}

/// SP1 proof verification contract
pub struct SP1VerificationContract {
    config: L1VerificationConfig,
    contract_address: String,
    verification_results: Arc<RwLock<HashMap<String, L1VerificationResult>>>,
}

impl SP1VerificationContract {
    /// Create a new SP1 verification contract interface
    pub fn new(config: L1VerificationConfig) -> Self {
        Self {
            contract_address: config.sp1_verifier_address.clone(),
            config,
            verification_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Encode proof data for SP1 contract call
    fn encode_sp1_proof_data(&self, proof: &ZkProof) -> Result<VerificationCallData, RollupError> {
        // Function selector for verifySP1Proof(bytes,bytes,bytes32)
        let function_selector = [0x87, 0x65, 0x43, 0x21]; // Placeholder

        // SP1-specific encoding
        let proof_data = proof.proof_data.clone();
        let public_inputs = proof.public_inputs.clone();
        let verification_params = vec![]; // SP1-specific parameters

        Ok(VerificationCallData {
            function_selector,
            proof_data,
            public_inputs,
            verification_params,
        })
    }
}

#[async_trait::async_trait]
impl L1VerificationContract for SP1VerificationContract {
    async fn verify_proof(
        &self,
        proof: &ZkProof,
        public_inputs: &[u8],
        state_root: &StateRoot,
    ) -> Result<L1VerificationResult, RollupError> {
        info!("Verifying SP1 proof on L1 contract at {}", self.contract_address);

        let request_id = generate_verification_request_id();
        let start_time = std::time::Instant::now();

        // Encode SP1-specific call data
        let _call_data = self.encode_sp1_proof_data(proof)?;

        // Simulate contract call
        let verification_result = L1VerificationResult {
            request_id: request_id.clone(),
            is_valid: true, // Placeholder
            verification_tx_hash: Some([2u8; 32]), // Placeholder
            verification_block: Some(12346), // Placeholder
            gas_used: 750000, // SP1 proofs might use more gas
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Store result
        {
            let mut results = self.verification_results.write().await;
            results.insert(request_id.clone(), verification_result.clone());
        }

        info!("SP1 proof verification completed for request {}", request_id);
        Ok(verification_result)
    }

    async fn get_verification_status(
        &self,
        request_id: &str,
    ) -> Result<Option<L1VerificationResult>, RollupError> {
        let results = self.verification_results.read().await;
        Ok(results.get(request_id).cloned())
    }

    fn contract_address(&self) -> &str {
        &self.contract_address
    }

    fn supported_backend(&self) -> ZkVMBackend {
        ZkVMBackend::SP1Local // Also supports SP1Network
    }
}

/// State commitment verification contract
pub struct StateCommitmentVerificationContract {
    config: L1VerificationConfig,
    contract_address: String,
    verification_results: Arc<RwLock<HashMap<String, L1VerificationResult>>>,
}

impl StateCommitmentVerificationContract {
    /// Create a new state commitment verification contract interface
    pub fn new(config: L1VerificationConfig) -> Self {
        Self {
            contract_address: config.state_verifier_address.clone(),
            config,
            verification_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Verify state commitment with merkle proofs
    pub async fn verify_state_commitment(
        &self,
        state_root: &StateRoot,
        order_book_merkle_root: &[u8; 32],
        merkle_proof: &[Vec<u8>],
    ) -> Result<L1VerificationResult, RollupError> {
        info!("Verifying state commitment on L1 contract at {}", self.contract_address);

        let request_id = generate_verification_request_id();
        let start_time = std::time::Instant::now();

        // Encode state commitment verification call data
        let _call_data = self.encode_state_commitment_data(
            state_root,
            order_book_merkle_root,
            merkle_proof,
        )?;

        // Simulate contract call
        let verification_result = L1VerificationResult {
            request_id: request_id.clone(),
            is_valid: true, // Placeholder
            verification_tx_hash: Some([3u8; 32]), // Placeholder
            verification_block: Some(12347), // Placeholder
            gas_used: 300000, // State verification might use less gas
            verification_time_ms: start_time.elapsed().as_millis() as u64,
            error_message: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        // Store result
        {
            let mut results = self.verification_results.write().await;
            results.insert(request_id.clone(), verification_result.clone());
        }

        info!("State commitment verification completed for request {}", request_id);
        Ok(verification_result)
    }

    /// Encode state commitment data for contract call
    fn encode_state_commitment_data(
        &self,
        state_root: &StateRoot,
        order_book_merkle_root: &[u8; 32],
        merkle_proof: &[Vec<u8>],
    ) -> Result<VerificationCallData, RollupError> {
        // Function selector for verifyStateCommitment(bytes32,bytes32,bytes[])
        let function_selector = [0xAB, 0xCD, 0xEF, 0x12]; // Placeholder

        // Encode state commitment data
        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(state_root);
        proof_data.extend_from_slice(order_book_merkle_root);

        // Encode merkle proof
        let mut public_inputs = Vec::new();
        for proof_element in merkle_proof {
            public_inputs.extend_from_slice(proof_element);
        }

        Ok(VerificationCallData {
            function_selector,
            proof_data,
            public_inputs,
            verification_params: vec![],
        })
    }
}

/// L1 verification contract manager
pub struct L1VerificationContractManager {
    config: L1VerificationConfig,
    zisk_contract: Arc<ZiskVerificationContract>,
    sp1_contract: Arc<SP1VerificationContract>,
    state_contract: Arc<StateCommitmentVerificationContract>,
    verification_stats: Arc<RwLock<VerificationStats>>,
}

/// Verification statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStats {
    /// Total verification requests
    pub total_requests: u64,
    /// Successful verifications
    pub successful_verifications: u64,
    /// Failed verifications
    pub failed_verifications: u64,
    /// Average verification time (ms)
    pub average_verification_time_ms: f64,
    /// Total gas used for verifications
    pub total_gas_used: u64,
    /// Verifications by backend
    pub verifications_by_backend: HashMap<String, u64>,
}

impl Default for VerificationStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_verifications: 0,
            failed_verifications: 0,
            average_verification_time_ms: 0.0,
            total_gas_used: 0,
            verifications_by_backend: HashMap::new(),
        }
    }
}

impl L1VerificationContractManager {
    /// Create a new L1 verification contract manager
    pub fn new(config: L1VerificationConfig) -> Self {
        let zisk_contract = Arc::new(ZiskVerificationContract::new(config.clone()));
        let sp1_contract = Arc::new(SP1VerificationContract::new(config.clone()));
        let state_contract = Arc::new(StateCommitmentVerificationContract::new(config.clone()));

        Self {
            config,
            zisk_contract,
            sp1_contract,
            state_contract,
            verification_stats: Arc::new(RwLock::new(VerificationStats::default())),
        }
    }

    /// Verify proof using appropriate contract based on backend
    pub async fn verify_proof(
        &self,
        proof: &ZkProof,
        public_inputs: &[u8],
        state_root: &StateRoot,
    ) -> Result<L1VerificationResult, RollupError> {
        let start_time = std::time::Instant::now();

        let result = match proof.backend {
            ZkVMBackend::ZisK => {
                self.zisk_contract.verify_proof(proof, public_inputs, state_root).await
            }
            ZkVMBackend::SP1Local | ZkVMBackend::SP1Network => {
                self.sp1_contract.verify_proof(proof, public_inputs, state_root).await
            }
        };

        // Update statistics
        {
            let mut stats = self.verification_stats.write().await;
            stats.total_requests += 1;

            match &result {
                Ok(verification_result) => {
                    if verification_result.is_valid {
                        stats.successful_verifications += 1;
                    } else {
                        stats.failed_verifications += 1;
                    }
                    stats.total_gas_used += verification_result.gas_used;
                    
                    // Update backend-specific stats
                    let backend_name = format!("{:?}", proof.backend);
                    *stats.verifications_by_backend.entry(backend_name).or_insert(0) += 1;
                }
                Err(_) => {
                    stats.failed_verifications += 1;
                }
            }

            // Update average verification time
            let verification_time = start_time.elapsed().as_millis() as f64;
            stats.average_verification_time_ms = 
                (stats.average_verification_time_ms * (stats.total_requests - 1) as f64 + 
                 verification_time) / stats.total_requests as f64;
        }

        result
    }

    /// Verify state commitment
    pub async fn verify_state_commitment(
        &self,
        state_root: &StateRoot,
        order_book_merkle_root: &[u8; 32],
        merkle_proof: &[Vec<u8>],
    ) -> Result<L1VerificationResult, RollupError> {
        self.state_contract
            .verify_state_commitment(state_root, order_book_merkle_root, merkle_proof)
            .await
    }

    /// Get verification statistics
    pub async fn get_verification_stats(&self) -> VerificationStats {
        self.verification_stats.read().await.clone()
    }

    /// Get contract addresses
    pub fn get_contract_addresses(&self) -> HashMap<String, String> {
        let mut addresses = HashMap::new();
        addresses.insert("zisk_verifier".to_string(), self.zisk_contract.contract_address().to_string());
        addresses.insert("sp1_verifier".to_string(), self.sp1_contract.contract_address().to_string());
        addresses.insert("state_verifier".to_string(), self.state_contract.contract_address.clone());
        addresses
    }
}

/// Generate unique verification request ID
fn generate_verification_request_id() -> String {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random = rand::thread_rng().gen::<u32>();
    format!("verification_{}_{:x}", timestamp, random)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::{ZkVMBackend, ProofMetadata};

    fn create_test_proof(backend: ZkVMBackend) -> ZkProof {
        ZkProof {
            backend,
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
    fn test_verification_priority_ordering() {
        assert!(VerificationPriority::Critical > VerificationPriority::High);
        assert!(VerificationPriority::High > VerificationPriority::Normal);
        assert!(VerificationPriority::Normal > VerificationPriority::Low);
    }

    #[test]
    fn test_generate_verification_request_id() {
        let id1 = generate_verification_request_id();
        let id2 = generate_verification_request_id();
        
        assert_ne!(id1, id2);
        assert!(id1.starts_with("verification_"));
        assert!(id2.starts_with("verification_"));
    }

    #[tokio::test]
    async fn test_zisk_verification_contract() {
        let config = L1VerificationConfig::default();
        let contract = ZiskVerificationContract::new(config);
        
        assert_eq!(contract.supported_backend(), ZkVMBackend::ZisK);
        assert_eq!(contract.contract_address(), "0x0000000000000000000000000000000000000001");

        let proof = create_test_proof(ZkVMBackend::ZisK);
        let public_inputs = vec![1, 2, 3, 4];
        let state_root = [5u8; 32];

        let result = contract.verify_proof(&proof, &public_inputs, &state_root).await;
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        assert!(verification_result.is_valid);
        assert!(verification_result.verification_tx_hash.is_some());
    }

    #[tokio::test]
    async fn test_sp1_verification_contract() {
        let config = L1VerificationConfig::default();
        let contract = SP1VerificationContract::new(config);
        
        assert_eq!(contract.supported_backend(), ZkVMBackend::SP1Local);
        assert_eq!(contract.contract_address(), "0x0000000000000000000000000000000000000002");

        let proof = create_test_proof(ZkVMBackend::SP1Local);
        let public_inputs = vec![1, 2, 3, 4];
        let state_root = [5u8; 32];

        let result = contract.verify_proof(&proof, &public_inputs, &state_root).await;
        assert!(result.is_ok());
        
        let verification_result = result.unwrap();
        assert!(verification_result.is_valid);
        assert!(verification_result.verification_tx_hash.is_some());
    }

    #[tokio::test]
    async fn test_l1_verification_contract_manager() {
        let config = L1VerificationConfig::default();
        let manager = L1VerificationContractManager::new(config);

        // Test ZisK proof verification
        let zisk_proof = create_test_proof(ZkVMBackend::ZisK);
        let public_inputs = vec![1, 2, 3, 4];
        let state_root = [5u8; 32];

        let result = manager.verify_proof(&zisk_proof, &public_inputs, &state_root).await;
        assert!(result.is_ok());

        // Test SP1 proof verification
        let sp1_proof = create_test_proof(ZkVMBackend::SP1Local);
        let result = manager.verify_proof(&sp1_proof, &public_inputs, &state_root).await;
        assert!(result.is_ok());

        // Check statistics
        let stats = manager.get_verification_stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.successful_verifications, 2);
        assert_eq!(stats.failed_verifications, 0);
    }

    #[tokio::test]
    async fn test_state_commitment_verification() {
        let config = L1VerificationConfig::default();
        let contract = StateCommitmentVerificationContract::new(config);

        let state_root = [1u8; 32];
        let order_book_merkle_root = [2u8; 32];
        let merkle_proof = vec![vec![3u8; 32], vec![4u8; 32]];

        let result = contract.verify_state_commitment(
            &state_root,
            &order_book_merkle_root,
            &merkle_proof,
        ).await;

        assert!(result.is_ok());
        let verification_result = result.unwrap();
        assert!(verification_result.is_valid);
    }
}