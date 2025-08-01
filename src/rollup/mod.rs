//! Based Rollup Integration
//! 
//! This module implements a based rollup system for the CLOB, providing:
//! - Order batching and sequencing
//! - L1 settlement integration
//! - Data availability layer with blob storage
//! - State compression and verification

pub mod sequencer;
pub mod settlement;
pub mod data_availability;
pub mod advanced_da;
pub mod compression;
pub mod types;
pub mod integration_example;
pub mod ethrex_client;
pub mod ethrex_rpc_client;
pub mod finality_tracker;
pub mod ethrex_integration;
pub mod l1_state_sync_manager;
pub mod proof_anchoring;
pub mod l1_verification_contracts;
pub mod proof_anchoring_example;
pub mod lattice_fold_plus;
pub mod lattice_commitments;
pub mod lattice_sampling;
pub mod lattice_challenges;
pub mod recursive_folding;
pub mod recursive_folding_engine;
pub mod amortized_folding;

pub use sequencer::*;
pub use settlement::*;
pub use data_availability::*;
pub use advanced_da::*;
pub use compression::*;
pub use types::*;
pub use integration_example::*;
pub use ethrex_client::*;
pub use ethrex_rpc_client::*;
pub use finality_tracker::*;
pub use ethrex_integration::*;
pub use l1_state_sync_manager::*;
pub use proof_anchoring::*;
pub use l1_verification_contracts::*;
pub use proof_anchoring_example::*;
pub use lattice_fold_plus::*;
pub use lattice_commitments::*;
pub use lattice_sampling::*;
pub use lattice_challenges::*;
pub use recursive_folding::*;
pub use recursive_folding_engine::*;
pub use amortized_folding::*;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Rollup-specific errors
#[derive(Error, Debug)]
pub enum RollupError {
    #[error("Sequencer error: {0}")]
    SequencerError(String),
    
    #[error("Settlement error: {0}")]
    SettlementError(String),
    
    #[error("Data availability error: {0}")]
    DataAvailabilityError(String),
    
    #[error("Compression error: {0}")]
    CompressionError(String),
    
    #[error("State transition error: {0}")]
    StateTransitionError(String),
    
    #[error("L1 synchronization error: {0}")]
    L1SyncError(String),
    
    #[error("Proof generation error: {0}")]
    ProofError(String),
    
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Rollup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupConfig {
    pub sequencer_config: SequencerConfig,
    pub settlement_config: SettlementConfig,
    pub da_config: DataAvailabilityConfig,
    pub compression_config: CompressionConfig,
}

impl Default for RollupConfig {
    fn default() -> Self {
        Self {
            sequencer_config: SequencerConfig::default(),
            settlement_config: SettlementConfig::default(),
            da_config: DataAvailabilityConfig::default(),
            compression_config: CompressionConfig::default(),
        }
    }
}