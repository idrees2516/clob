//! Machine Learning Integration Module
//! 
//! This module provides comprehensive machine learning capabilities for the high-frequency
//! trading system, including regime detection, order flow prediction, and adaptive learning.

pub mod regime_detection;
pub mod order_flow_prediction;
pub mod adaptive_learning;
pub mod feature_engineering;
pub mod model_monitoring;

pub use regime_detection::*;
pub use order_flow_prediction::*;
pub use adaptive_learning::*;
pub use feature_engineering::*;
pub use model_monitoring::*;

use crate::error::TradingSystemError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common types used across ML modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: f64,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub training_metrics: MLMetrics,
    pub validation_metrics: MLMetrics,
    pub test_metrics: Option<MLMetrics>,
    pub training_time: std::time::Duration,
    pub inference_time_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub feature_name: String,
    pub importance_score: f64,
    pub rank: usize,
}

/// Configuration for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub regime_detection: RegimeDetectionConfig,
    pub order_flow_prediction: OrderFlowPredictionConfig,
    pub adaptive_learning: AdaptiveLearningConfig,
    pub feature_engineering: FeatureEngineeringConfig,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            regime_detection: RegimeDetectionConfig::default(),
            order_flow_prediction: OrderFlowPredictionConfig::default(),
            adaptive_learning: AdaptiveLearningConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
        }
    }
}

/// Result type for ML operations
pub type MLResult<T> = Result<T, MLError>;

#[derive(Debug, thiserror::Error)]
pub enum MLError {
    #[error("Model training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Model prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Feature engineering failed: {0}")]
    FeatureEngineeringFailed(String),
    
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Data preprocessing failed: {0}")]
    PreprocessingFailed(String),
    
    #[error("Model serialization failed: {0}")]
    SerializationFailed(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

impl From<MLError> for TradingSystemError {
    fn from(err: MLError) -> Self {
        TradingSystemError::MachineLearningError(err.to_string())
    }
}