//! zkVM Integration Layer
//! 
//! This module provides a unified interface for both ZisK and SP1 zkVMs,
//! enabling the CLOB system to generate zero-knowledge proofs for order
//! execution and state transitions.

pub mod traits;
pub mod zisk;
pub mod sp1;
pub mod sp1_types;
pub mod sp1_proof;
pub mod succinct_network;
pub mod execution;
pub mod proof;
pub mod witness;
pub mod router;
pub mod trading_integration;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod trading_integration_test;

pub use traits::*;
pub use execution::*;
pub use proof::*;
pub use witness::*;
pub use router::*;
pub use trading_integration::*;
pub use trading_integration::*;
pub use trading_integration::*;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// zkVM execution errors
#[derive(Error, Debug)]
pub enum ZkVMError {
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),
    
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),
    
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),
    
    #[error("Witness generation failed: {0}")]
    WitnessGenerationFailed(String),
    
    #[error("Circuit compilation failed: {0}")]
    CircuitCompilationFailed(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// zkVM backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkVMBackend {
    ZisK,
    SP1Local,
    SP1Network,
}

/// zkVM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkVMConfig {
    pub backend: ZkVMBackend,
    pub max_cycles: u64,
    pub memory_limit: usize,
    pub timeout_seconds: u64,
    pub network_config: Option<NetworkConfig>,
    pub zisk_config: Option<ZiskConfig>,
    pub sp1_config: Option<SP1Config>,
}

/// Network configuration for Succinct Prover Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub endpoint: String,
    pub api_key: String,
    pub max_retries: u32,
    pub timeout_seconds: u64,
}

/// ZisK-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZiskConfig {
    pub optimization_level: u8,
    pub enable_gpu: bool,
    pub memory_pool_size: usize,
}

/// SP1-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SP1Config {
    pub prover_mode: SP1ProverMode,
    pub enable_cuda: bool,
    pub shard_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SP1ProverMode {
    Mock,
    Local,
    Network,
}

impl Default for ZkVMConfig {
    fn default() -> Self {
        Self {
            backend: ZkVMBackend::SP1Local,
            max_cycles: 1_000_000,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            timeout_seconds: 300, // 5 minutes
            network_config: None,
            zisk_config: Some(ZiskConfig {
                optimization_level: 2,
                enable_gpu: false,
                memory_pool_size: 512 * 1024 * 1024, // 512MB
            }),
            sp1_config: Some(SP1Config {
                prover_mode: SP1ProverMode::Local,
                enable_cuda: false,
                shard_size: 1 << 22, // 4M cycles per shard
            }),
        }
    }
}

/// zkVM factory for creating instances
pub struct ZkVMFactory;

impl ZkVMFactory {
    /// Create a zkVM instance based on configuration
    pub fn create(config: ZkVMConfig) -> Result<Box<dyn ZkVMInstance>, ZkVMError> {
        match config.backend {
            ZkVMBackend::ZisK => {
                #[cfg(feature = "zisk")]
                {
                    Ok(Box::new(zisk::ZiskVM::new(config)?))
                }
                #[cfg(not(feature = "zisk"))]
                {
                    Err(ZkVMError::ExecutionFailed(
                        "ZisK feature not enabled".to_string()
                    ))
                }
            }
            ZkVMBackend::SP1Local => {
                #[cfg(feature = "sp1")]
                {
                    Ok(Box::new(sp1::SP1VM::new_local(config)?))
                }
                #[cfg(not(feature = "sp1"))]
                {
                    Err(ZkVMError::ExecutionFailed(
                        "SP1 feature not enabled".to_string()
                    ))
                }
            }
            ZkVMBackend::SP1Network => {
                #[cfg(feature = "succinct-network")]
                {
                    Ok(Box::new(succinct_network::SuccinctNetworkVM::new(config)?))
                }
                #[cfg(not(feature = "succinct-network"))]
                {
                    Err(ZkVMError::ExecutionFailed(
                        "Succinct Network feature not enabled".to_string()
                    ))
                }
            }
        }
    }
}