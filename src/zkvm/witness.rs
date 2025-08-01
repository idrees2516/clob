//! zkVM witness utilities
//! 
//! This module provides common witness generation and validation utilities
//! for zkVM execution witnesses across different backends.

use crate::zkvm::{ZkVMError, traits::*};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use sha2::{Sha256, Digest};

/// Witness builder for creating execution witnesses
pub struct WitnessBuilder {
    execution_id: String,
    witness_data: Vec<u8>,
    public_inputs: Vec<u8>,
    private_inputs: Vec<u8>,
    auxiliary_data: HashMap<String, Vec<u8>>,
}

impl WitnessBuilder {
    pub fn new(execution_id: String) -> Self {
        Self {
            execution_id,
            witness_data: Vec::new(),
            public_inputs: Vec::new(),
            private_inputs: Vec::new(),
            auxiliary_data: HashMap::new(),
        }
    }

    pub fn witness_data(mut self, data: Vec<u8>) -> Self {
        self.witness_data = data;
        self
    }

    pub fn public_inputs(mut self, inputs: Vec<u8>) -> Self {
        self.public_inputs = inputs;
        self
    }

    pub fn private_inputs(mut self, inputs: Vec<u8>) -> Self {
        self.private_inputs = inputs;
        self
    }

    pub fn auxiliary_data(mut self, key: String, data: Vec<u8>) -> Self {
        self.auxiliary_data.insert(key, data);
        self
    }

    pub fn build(self) -> Result<ExecutionWitness, ZkVMError> {
        if self.witness_data.is_empty() {
            return Err(ZkVMError::WitnessGenerationFailed(
                "Witness data cannot be empty".to_string()
            ));
        }

        Ok(ExecutionWitness {
            execution_id: self.execution_id,
            witness_data: self.witness_data,
            public_inputs: self.public_inputs,
            private_inputs: self.private_inputs,
            auxiliary_data: self.auxiliary_data,
        })
    }
}

/// Validate witness structure and consistency
pub fn validate_witness(witness: &ExecutionWitness) -> Result<(), ZkVMError> {
    if witness.execution_id.is_empty() {
        return Err(ZkVMError::WitnessGenerationFailed(
            "Execution ID cannot be empty".to_string()
        ));
    }

    if witness.witness_data.is_empty() {
        return Err(ZkVMError::WitnessGenerationFailed(
            "Witness data cannot be empty".to_string()
        ));
    }

    // Validate witness data structure based on common patterns
    if witness.witness_data.len() < 16 {
        return Err(ZkVMError::WitnessGenerationFailed(
            "Witness data too small to be valid".to_string()
        ));
    }

    Ok(())
}

/// Compute witness hash for integrity checking
pub fn compute_witness_hash(witness: &ExecutionWitness) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(&witness.execution_id.as_bytes());
    hasher.update(&witness.witness_data);
    hasher.update(&witness.public_inputs);
    hasher.update(&witness.private_inputs);
    
    // Include auxiliary data in hash
    let mut aux_keys: Vec<_> = witness.auxiliary_data.keys().collect();
    aux_keys.sort();
    for key in aux_keys {
        hasher.update(key.as_bytes());
        hasher.update(&witness.auxiliary_data[key]);
    }
    
    hasher.finalize().into()
}

/// Witness compression utilities
pub mod compression {
    use super::*;
    use std::io::{Read, Write};

    /// Compress witness data using zstd
    pub fn compress_witness(witness: &ExecutionWitness) -> Result<Vec<u8>, ZkVMError> {
        let serialized = bincode::serialize(witness)
            .map_err(|e| ZkVMError::WitnessGenerationFailed(format!("Serialization failed: {}", e)))?;
        
        let compressed = zstd::encode_all(&serialized[..], 6)
            .map_err(|e| ZkVMError::WitnessGenerationFailed(format!("Compression failed: {}", e)))?;
        
        Ok(compressed)
    }

    /// Decompress witness data
    pub fn decompress_witness(compressed_data: &[u8]) -> Result<ExecutionWitness, ZkVMError> {
        let decompressed = zstd::decode_all(compressed_data)
            .map_err(|e| ZkVMError::WitnessGenerationFailed(format!("Decompression failed: {}", e)))?;
        
        let witness = bincode::deserialize(&decompressed)
            .map_err(|e| ZkVMError::WitnessGenerationFailed(format!("Deserialization failed: {}", e)))?;
        
        Ok(witness)
    }
}

/// Witness statistics
#[derive(Debug, Clone)]
pub struct WitnessStats {
    pub witness_size: usize,
    pub public_inputs_size: usize,
    pub private_inputs_size: usize,
    pub auxiliary_data_size: usize,
    pub compression_ratio: Option<f64>,
}

impl WitnessStats {
    pub fn from_witness(witness: &ExecutionWitness) -> Self {
        let auxiliary_size = witness.auxiliary_data.values()
            .map(|v| v.len())
            .sum();

        Self {
            witness_size: witness.witness_data.len(),
            public_inputs_size: witness.public_inputs.len(),
            private_inputs_size: witness.private_inputs.len(),
            auxiliary_data_size: auxiliary_size,
            compression_ratio: None,
        }
    }

    pub fn with_compression_ratio(mut self, original_size: usize, compressed_size: usize) -> Self {
        if compressed_size > 0 {
            self.compression_ratio = Some(original_size as f64 / compressed_size as f64);
        }
        self
    }

    pub fn total_size(&self) -> usize {
        self.witness_size + self.public_inputs_size + self.private_inputs_size + self.auxiliary_data_size
    }
}

/// Witness verification utilities
pub mod verification {
    use super::*;

    /// Verify witness against execution result
    pub fn verify_witness_consistency(
        witness: &ExecutionWitness,
        execution: &ExecutionResult,
    ) -> Result<bool, ZkVMError> {
        // Check execution ID matches
        if witness.execution_id != execution.execution_id {
            return Ok(false);
        }

        // Check public inputs consistency
        if witness.public_inputs != execution.public_outputs {
            return Ok(false);
        }

        // Verify witness structure
        validate_witness(witness)?;

        Ok(true)
    }

    /// Verify witness integrity using hash
    pub fn verify_witness_integrity(
        witness: &ExecutionWitness,
        expected_hash: &[u8; 32],
    ) -> bool {
        let actual_hash = compute_witness_hash(witness);
        actual_hash == *expected_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_builder() {
        let witness = WitnessBuilder::new("test_execution".to_string())
            .witness_data(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .public_inputs(vec![1, 2, 3])
            .private_inputs(vec![4, 5, 6])
            .auxiliary_data("trace".to_string(), vec![7, 8, 9])
            .build();

        assert!(witness.is_ok());
        let witness = witness.unwrap();
        assert_eq!(witness.execution_id, "test_execution");
        assert_eq!(witness.witness_data.len(), 16);
        assert_eq!(witness.public_inputs, vec![1, 2, 3]);
        assert_eq!(witness.private_inputs, vec![4, 5, 6]);
        assert_eq!(witness.auxiliary_data.get("trace"), Some(&vec![7, 8, 9]));
    }

    #[test]
    fn test_validate_witness() {
        let valid_witness = ExecutionWitness {
            execution_id: "test".to_string(),
            witness_data: vec![0; 32], // Sufficient size
            public_inputs: vec![1, 2, 3],
            private_inputs: vec![4, 5, 6],
            auxiliary_data: HashMap::new(),
        };

        assert!(validate_witness(&valid_witness).is_ok());

        let invalid_witness = ExecutionWitness {
            execution_id: "".to_string(), // Empty ID
            witness_data: vec![0; 32],
            public_inputs: vec![1, 2, 3],
            private_inputs: vec![4, 5, 6],
            auxiliary_data: HashMap::new(),
        };

        assert!(validate_witness(&invalid_witness).is_err());
    }

    #[test]
    fn test_compute_witness_hash() {
        let witness = ExecutionWitness {
            execution_id: "test".to_string(),
            witness_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            public_inputs: vec![1, 2, 3],
            private_inputs: vec![4, 5, 6],
            auxiliary_data: HashMap::new(),
        };

        let hash1 = compute_witness_hash(&witness);
        let hash2 = compute_witness_hash(&witness);
        
        assert_eq!(hash1, hash2);
        assert_ne!(hash1, [0u8; 32]);
    }

    #[test]
    fn test_witness_stats() {
        let mut aux_data = HashMap::new();
        aux_data.insert("trace".to_string(), vec![1, 2, 3, 4, 5]);
        
        let witness = ExecutionWitness {
            execution_id: "test".to_string(),
            witness_data: vec![0; 100],
            public_inputs: vec![0; 20],
            private_inputs: vec![0; 30],
            auxiliary_data: aux_data,
        };

        let stats = WitnessStats::from_witness(&witness);
        assert_eq!(stats.witness_size, 100);
        assert_eq!(stats.public_inputs_size, 20);
        assert_eq!(stats.private_inputs_size, 30);
        assert_eq!(stats.auxiliary_data_size, 5);
        assert_eq!(stats.total_size(), 155);
    }

    #[test]
    fn test_compression() {
        let witness = ExecutionWitness {
            execution_id: "test".to_string(),
            witness_data: vec![0; 1000], // Compressible data
            public_inputs: vec![1, 2, 3],
            private_inputs: vec![4, 5, 6],
            auxiliary_data: HashMap::new(),
        };

        let compressed = compression::compress_witness(&witness).unwrap();
        let decompressed = compression::decompress_witness(&compressed).unwrap();

        assert_eq!(witness.execution_id, decompressed.execution_id);
        assert_eq!(witness.witness_data, decompressed.witness_data);
        assert_eq!(witness.public_inputs, decompressed.public_inputs);
        assert_eq!(witness.private_inputs, decompressed.private_inputs);
    }
}