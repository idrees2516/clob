//! zkVM proof utilities
//! 
//! This module provides common proof generation and verification utilities
//! for zkVM proofs across different backends.

use crate::zkvm::{ZkVMError, traits::*};
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};
use rand::Rng;

/// Proof builder for creating zkVM proofs
pub struct ProofBuilder {
    backend: ZkVMBackend,
    proof_data: Vec<u8>,
    public_inputs: Vec<u8>,
    verification_key_hash: Option<[u8; 32]>,
    security_level: u8,
}

impl ProofBuilder {
    pub fn new(backend: ZkVMBackend) -> Self {
        Self {
            backend,
            proof_data: Vec::new(),
            public_inputs: Vec::new(),
            verification_key_hash: None,
            security_level: 128,
        }
    }

    pub fn proof_data(mut self, data: Vec<u8>) -> Self {
        self.proof_data = data;
        self
    }

    pub fn public_inputs(mut self, inputs: Vec<u8>) -> Self {
        self.public_inputs = inputs;
        self
    }

    pub fn verification_key_hash(mut self, hash: [u8; 32]) -> Self {
        self.verification_key_hash = Some(hash);
        self
    }

    pub fn security_level(mut self, level: u8) -> Self {
        self.security_level = level;
        self
    }

    pub fn build(self) -> Result<ZkProof, ZkVMError> {
        if self.proof_data.is_empty() {
            return Err(ZkVMError::ProofGenerationFailed(
                "Proof data cannot be empty".to_string()
            ));
        }

        let vk_hash = self.verification_key_hash.unwrap_or([0u8; 32]);
        
        Ok(ZkProof {
            backend: self.backend,
            proof_data: self.proof_data.clone(),
            public_inputs: self.public_inputs,
            verification_key_hash: vk_hash,
            proof_metadata: ProofMetadata {
                proof_id: generate_proof_id(),
                generation_time: current_timestamp(),
                proof_size: self.proof_data.len(),
                security_level: self.security_level,
                circuit_size: 0, // Will be set by specific backend
            },
        })
    }
}

/// Generate unique proof ID
pub fn generate_proof_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random = rand::thread_rng().gen::<u32>();
    format!("proof_{}_{:x}", timestamp, random)
}

/// Get current timestamp in milliseconds
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Validate proof structure
pub fn validate_proof_structure(proof: &ZkProof) -> Result<(), ZkVMError> {
    if proof.proof_data.is_empty() {
        return Err(ZkVMError::ProofVerificationFailed(
            "Proof data is empty".to_string()
        ));
    }

    if proof.verification_key_hash == [0u8; 32] {
        return Err(ZkVMError::ProofVerificationFailed(
            "Invalid verification key hash".to_string()
        ));
    }

    if proof.proof_metadata.proof_size != proof.proof_data.len() {
        return Err(ZkVMError::ProofVerificationFailed(
            "Proof size mismatch".to_string()
        ));
    }

    Ok(())
}

/// Compute proof hash for integrity checking
pub fn compute_proof_hash(proof: &ZkProof) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&proof.proof_data);
    hasher.update(&proof.public_inputs);
    hasher.update(&proof.verification_key_hash);
    hasher.update(&proof.proof_metadata.proof_id.as_bytes());
    hasher.finalize().into()
}

/// Proof verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub verification_time_ms: u64,
    pub error_message: Option<String>,
}

impl VerificationResult {
    pub fn valid(verification_time_ms: u64) -> Self {
        Self {
            is_valid: true,
            verification_time_ms,
            error_message: None,
        }
    }

    pub fn invalid(verification_time_ms: u64, error: String) -> Self {
        Self {
            is_valid: false,
            verification_time_ms,
            error_message: Some(error),
        }
    }
}

/// Batch proof verification
pub async fn batch_verify_proofs(
    proofs: &[ZkProof],
    public_inputs: &[&[u8]],
    verification_keys: &[VerificationKey],
    verifier: &dyn ZkVMInstance,
) -> Result<Vec<VerificationResult>, ZkVMError> {
    if proofs.len() != public_inputs.len() || proofs.len() != verification_keys.len() {
        return Err(ZkVMError::ProofVerificationFailed(
            "Mismatched array lengths for batch verification".to_string()
        ));
    }

    let mut results = Vec::new();
    
    for (i, proof) in proofs.iter().enumerate() {
        let start_time = std::time::Instant::now();
        
        match verifier.verify_proof(proof, public_inputs[i], &verification_keys[i]).await {
            Ok(is_valid) => {
                let verification_time = start_time.elapsed().as_millis() as u64;
                if is_valid {
                    results.push(VerificationResult::valid(verification_time));
                } else {
                    results.push(VerificationResult::invalid(
                        verification_time,
                        "Proof verification failed".to_string()
                    ));
                }
            }
            Err(e) => {
                let verification_time = start_time.elapsed().as_millis() as u64;
                results.push(VerificationResult::invalid(
                    verification_time,
                    e.to_string()
                ));
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_builder() {
        let proof = ProofBuilder::new(ZkVMBackend::ZisK)
            .proof_data(vec![1, 2, 3, 4])
            .public_inputs(vec![5, 6, 7, 8])
            .verification_key_hash([1u8; 32])
            .security_level(256)
            .build();

        assert!(proof.is_ok());
        let proof = proof.unwrap();
        assert_eq!(proof.backend, ZkVMBackend::ZisK);
        assert_eq!(proof.proof_data, vec![1, 2, 3, 4]);
        assert_eq!(proof.public_inputs, vec![5, 6, 7, 8]);
        assert_eq!(proof.verification_key_hash, [1u8; 32]);
        assert_eq!(proof.proof_metadata.security_level, 256);
    }

    #[test]
    fn test_generate_proof_id() {
        let id1 = generate_proof_id();
        let id2 = generate_proof_id();
        
        assert_ne!(id1, id2);
        assert!(id1.starts_with("proof_"));
        assert!(id2.starts_with("proof_"));
    }

    #[test]
    fn test_validate_proof_structure() {
        let valid_proof = ZkProof {
            backend: ZkVMBackend::ZisK,
            proof_data: vec![1, 2, 3, 4],
            public_inputs: vec![5, 6, 7, 8],
            verification_key_hash: [1u8; 32],
            proof_metadata: ProofMetadata {
                proof_id: "test_proof".to_string(),
                generation_time: current_timestamp(),
                proof_size: 4,
                security_level: 128,
                circuit_size: 1000,
            },
        };

        assert!(validate_proof_structure(&valid_proof).is_ok());

        let invalid_proof = ZkProof {
            backend: ZkVMBackend::ZisK,
            proof_data: vec![],
            public_inputs: vec![5, 6, 7, 8],
            verification_key_hash: [0u8; 32],
            proof_metadata: ProofMetadata {
                proof_id: "test_proof".to_string(),
                generation_time: current_timestamp(),
                proof_size: 0,
                security_level: 128,
                circuit_size: 1000,
            },
        };

        assert!(validate_proof_structure(&invalid_proof).is_err());
    }

    #[test]
    fn test_compute_proof_hash() {
        let proof = ZkProof {
            backend: ZkVMBackend::ZisK,
            proof_data: vec![1, 2, 3, 4],
            public_inputs: vec![5, 6, 7, 8],
            verification_key_hash: [1u8; 32],
            proof_metadata: ProofMetadata {
                proof_id: "test_proof".to_string(),
                generation_time: current_timestamp(),
                proof_size: 4,
                security_level: 128,
                circuit_size: 1000,
            },
        };

        let hash1 = compute_proof_hash(&proof);
        let hash2 = compute_proof_hash(&proof);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, [0u8; 32]);
    }
}