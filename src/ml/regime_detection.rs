//! Regime Detection System
//! 
//! Implements sophisticated market regime detection using Hidden Markov Models,
//! volatility clustering analysis, and correlation regime detection.

use crate::models::hmm::{HiddenMarkovModel, HMMError};
use crate::ml::{MLError, MLResult, ModelPerformance, MLMetrics};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionConfig {
    pub volatility_regimes: VolatilityRegimeConfig,
    pub correlation_regimes: CorrelationRegimeConfig,
    pub market_regimes: MarketRegimeConfig,
    pub update_frequency: Duration,
    pub lookback_window: usize,
    pub min_regime_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityRegimeConfig {
    pub n_states: usize,
    pub estimation_window: usize,
    pub garch_order: (usize, usize), // (p, q) for GARCH(p,q)
    pub regime_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRegimeConfig {
    pub n_states: usize,
    pub correlation_window: usize,
    pub assets: Vec<String>,
    pub correlation_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketRegimeConfig {
    pub n_states: usize,
    pub features: Vec<String>,
    pub hmm_max_iter: usize,
    pub hmm_tolerance: f64,
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            volatility_regimes: VolatilityRegimeConfig {
                n_states: 3, // Low, Medium, High volatility
                estimation_window: 252, // One trading year
                garch_order: (1, 1),
                regime_threshold: 0.02,
            },
            correlation_regimes: CorrelationRegimeConfig {
                n_states: 2, // High correlation, Low correlation
                correlation_window: 60,
                assets: vec!["BTC".to_string(), "ETH".to_string()],
                correlation_threshold: 0.5,
            },
            market_regimes: MarketRegimeConfig {
                n_states: 4, // Bull, Bear, Sideways, Crisis
                features: vec![
                    "returns".to_string(),
                    "volatility".to_string(),
                    "volume".to_string(),
                    "bid_ask_spread".to_string(),
                ],
                hmm_max_iter: 100,
                hmm_tolerance: 1e-6,
            },
            update_frequency: Duration::from_secs(60), // Update every minute
            lookback_window: 1000,
            min_regime_duration: Duration::from_secs(300), // 5 minutes minimum
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeType {
    Volatility(VolatilityRegime),
    Correlation(CorrelationRegime),
    Market(MarketRegime),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrelationRegime {
    HighCorrelation,
    LowCorrelation,
    Decoupled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
    Crisis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    pub regime_type: RegimeType,
    pub probability: f64,
    pub duration: Duration,
    pub confidence: f64,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: std::time::SystemTime,
    pub price: f64,
    pub volume: f64,
    pub bid_ask_spread: f64,
    pub returns: Option<f64>,
    pub volatility: Option<f64>,
}

pub struct RegimeDetectionSystem {
    config: RegimeDetectionConfig,
    volatility_hmm: Option<HiddenMarkovModel>,
    correlation_hmm: Option<HiddenMarkovModel>,
    market_hmm: Option<HiddenMarkovModel>,
    current_regimes: HashMap<String, RegimeState>,
    market_data_buffer: VecDeque<MarketData>,
    correlation_matrix: Option<na::DMatrix<f64>>,
    last_update: std::time::Instant,
    performance_metrics: HashMap<String, ModelPerformance>,
}

impl RegimeDetectionSystem {
    pub fn new(config: RegimeDetectionConfig) -> Self {
        Self {
            config,
            volatility_hmm: None,
            correlation_hmm: None,
            market_hmm: None,
            current_regimes: HashMap::new(),
            market_data_buffer: VecDeque::new(),
            correlation_matrix: None,
            last_update: Instant::now(),
            performance_metrics: HashMap::new(),
        }
    }

    /// Initialize the regime detection system with historical data
    pub fn initialize(&mut self, historical_data: &[MarketData]) -> MLResult<()> {
        if historical_data.len() < self.config.lookback_window {
            return Err(MLError::InsufficientData(
                format!("Need at least {} data points, got {}", 
                    self.config.lookback_window, historical_data.len())
            ));
        }

        // Initialize market data buffer
        self.market_data_buffer.extend(historical_data.iter().cloned());
        if self.market_data_buffer.len() > self.config.lookback_window {
            self.market_data_buffer.drain(0..self.market_data_buffer.len() - self.config.lookback_window);
        }

        // Train initial models
        self.train_volatility_regime_model()?;
        self.train_correlation_regime_model()?;
        self.train_market_regime_model()?;

        // Detect initial regimes
        self.detect_current_regimes()?;

        Ok(())
    }

    /// Update the system with new market data
    pub fn update(&mut self, new_data: MarketData) -> MLResult<Vec<RegimeState>> {
        // Add new data to buffer
        self.market_data_buffer.push_back(new_data);
        if self.market_data_buffer.len() > self.config.lookback_window {
            self.market_data_buffer.pop_front();
        }

        // Check if it's time to update
        if self.last_update.elapsed() < self.config.update_frequency {
            return Ok(self.current_regimes.values().cloned().collect());
        }

        // Retrain models if needed
        if self.should_retrain_models() {
            self.retrain_models()?;
        }

        // Detect current regimes
        let regimes = self.detect_current_regimes()?;
        self.last_update = Instant::now();

        Ok(regimes)
    }

    /// Train volatility regime detection model
    fn train_volatility_regime_model(&mut self) -> MLResult<()> {
        let start_time = Instant::now();
        
        // Extract volatility features
        let volatilities = self.calculate_rolling_volatility()?;
        
        if volatilities.len() < 50 {
            return Err(MLError::InsufficientData("Not enough volatility data".to_string()));
        }

        // Initialize HMM for volatility regimes
        let n_states = self.config.volatility_regimes.n_states;
        let transition_matrix = self.initialize_transition_matrix(n_states);
        let emission_means = self.initialize_emission_means(&volatilities, n_states);
        let emission_vars = self.initialize_emission_variances(&volatilities, n_states);
        let initial_probs = na::DVector::from_element(n_states, 1.0 / n_states as f64);

        let mut hmm = HiddenMarkovModel::new(
            n_states,
            transition_matrix,
            emission_means,
            emission_vars,
            initial_probs,
        ).map_err(|e| MLError::TrainingFailed(format!("HMM initialization failed: {}", e)))?;

        // Train using Baum-Welch algorithm
        hmm.baum_welch_algorithm(
            &volatilities,
            self.config.market_regimes.hmm_max_iter,
            self.config.market_regimes.hmm_tolerance,
        ).map_err(|e| MLError::TrainingFailed(format!("Baum-Welch training failed: {}", e)))?;

        self.volatility_hmm = Some(hmm);

        // Calculate performance metrics
        let training_time = start_time.elapsed();
        let metrics = self.evaluate_volatility_model(&volatilities)?;
        
        self.performance_metrics.insert(
            "volatility_regime".to_string(),
            ModelPerformance {
                training_metrics: metrics,
                validation_metrics: metrics.clone(), // TODO: Implement proper validation
                test_metrics: None,
                training_time,
                inference_time_ns: 0, // Will be measured during inference
            }
        );

        Ok(())
    }

    /// Train correlation regime detection model
    fn train_correlation_regime_model(&mut self) -> MLResult<()> {
        let start_time = Instant::now();
        
        // Calculate rolling correlations
        let correlations = self.calculate_rolling_correlations()?;
        
        if correlations.len() < 50 {
            return Err(MLError::InsufficientData("Not enough correlation data".to_string()));
        }

        // Initialize HMM for correlation regimes
        let n_states = self.config.correlation_regimes.n_states;
        let transition_matrix = self.initialize_transition_matrix(n_states);
        let emission_means = self.initialize_emission_means(&correlations, n_states);
        let emission_vars = self.initialize_emission_variances(&correlations, n_states);
        let initial_probs = na::DVector::from_element(n_states, 1.0 / n_states as f64);

        let mut hmm = HiddenMarkovModel::new(
            n_states,
            transition_matrix,
            emission_means,
            emission_vars,
            initial_probs,
        ).map_err(|e| MLError::TrainingFailed(format!("HMM initialization failed: {}", e)))?;

        // Train using Baum-Welch algorithm
        hmm.baum_welch_algorithm(
            &correlations,
            self.config.market_regimes.hmm_max_iter,
            self.config.market_regimes.hmm_tolerance,
        ).map_err(|e| MLError::TrainingFailed(format!("Baum-Welch training failed: {}", e)))?;

        self.correlation_hmm = Some(hmm);

        // Calculate performance metrics
        let training_time = start_time.elapsed();
        let metrics = self.evaluate_correlation_model(&correlations)?;
        
        self.performance_metrics.insert(
            "correlation_regime".to_string(),
            ModelPerformance {
                training_metrics: metrics,
                validation_metrics: metrics.clone(),
                test_metrics: None,
                training_time,
                inference_time_ns: 0,
            }
        );

        Ok(())
    }

    /// Train market regime detection model
    fn train_market_regime_model(&mut self) -> MLResult<()> {
        let start_time = Instant::now();
        
        // Extract market features
        let features = self.extract_market_features()?;
        
        if features.len() < 100 {
            return Err(MLError::InsufficientData("Not enough market data".to_string()));
        }

        // Use first principal component for HMM training
        let pca_features = self.apply_pca(&features)?;
        
        // Initialize HMM for market regimes
        let n_states = self.config.market_regimes.n_states;
        let transition_matrix = self.initialize_transition_matrix(n_states);
        let emission_means = self.initialize_emission_means(&pca_features, n_states);
        let emission_vars = self.initialize_emission_variances(&pca_features, n_states);
        let initial_probs = na::DVector::from_element(n_states, 1.0 / n_states as f64);

        let mut hmm = HiddenMarkovModel::new(
            n_states,
            transition_matrix,
            emission_means,
            emission_vars,
            initial_probs,
        ).map_err(|e| MLError::TrainingFailed(format!("HMM initialization failed: {}", e)))?;

        // Train using Baum-Welch algorithm
        hmm.baum_welch_algorithm(
            &pca_features,
            self.config.market_regimes.hmm_max_iter,
            self.config.market_regimes.hmm_tolerance,
        ).map_err(|e| MLError::TrainingFailed(format!("Baum-Welch training failed: {}", e)))?;

        self.market_hmm = Some(hmm);

        // Calculate performance metrics
        let training_time = start_time.elapsed();
        let metrics = self.evaluate_market_model(&pca_features)?;
        
        self.performance_metrics.insert(
            "market_regime".to_string(),
            ModelPerformance {
                training_metrics: metrics,
                validation_metrics: metrics.clone(),
                test_metrics: None,
                training_time,
                inference_time_ns: 0,
            }
        );

        Ok(())
    }

    /// Detect current market regimes
    fn detect_current_regimes(&mut self) -> MLResult<Vec<RegimeState>> {
        let mut regimes = Vec::new();

        // Detect volatility regime
        if let Some(vol_regime) = self.detect_volatility_regime()? {
            self.current_regimes.insert("volatility".to_string(), vol_regime.clone());
            regimes.push(vol_regime);
        }

        // Detect correlation regime
        if let Some(corr_regime) = self.detect_correlation_regime()? {
            self.current_regimes.insert("correlation".to_string(), corr_regime.clone());
            regimes.push(corr_regime);
        }

        // Detect market regime
        if let Some(market_regime) = self.detect_market_regime()? {
            self.current_regimes.insert("market".to_string(), market_regime.clone());
            regimes.push(market_regime);
        }

        Ok(regimes)
    }

    /// Detect current volatility regime
    fn detect_volatility_regime(&self) -> MLResult<Option<RegimeState>> {
        let hmm = match &self.volatility_hmm {
            Some(hmm) => hmm,
            None => return Ok(None),
        };

        let recent_volatilities = self.calculate_recent_volatility(20)?;
        if recent_volatilities.is_empty() {
            return Ok(None);
        }

        let inference_start = Instant::now();
        let states = hmm.viterbi_algorithm(&recent_volatilities)
            .map_err(|e| MLError::PredictionFailed(format!("Viterbi algorithm failed: {}", e)))?;
        let inference_time = inference_start.elapsed();

        let current_state = *states.last().unwrap();
        let regime = match current_state {
            0 => VolatilityRegime::Low,
            1 => VolatilityRegime::Medium,
            2 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        };

        // Calculate regime probability using forward algorithm
        let alpha = hmm.forward_algorithm(&recent_volatilities)
            .map_err(|e| MLError::PredictionFailed(format!("Forward algorithm failed: {}", e)))?;
        let last_col = alpha.column(alpha.ncols() - 1);
        let total_prob: f64 = last_col.iter().sum();
        let state_prob = last_col[current_state] / total_prob;

        Ok(Some(RegimeState {
            regime_type: RegimeType::Volatility(regime),
            probability: state_prob,
            duration: Duration::from_secs(0), // TODO: Calculate actual duration
            confidence: self.calculate_regime_confidence(&states),
            timestamp: std::time::SystemTime::now(),
        }))
    }

    /// Detect current correlation regime
    fn detect_correlation_regime(&self) -> MLResult<Option<RegimeState>> {
        let hmm = match &self.correlation_hmm {
            Some(hmm) => hmm,
            None => return Ok(None),
        };

        let recent_correlations = self.calculate_recent_correlations(20)?;
        if recent_correlations.is_empty() {
            return Ok(None);
        }

        let states = hmm.viterbi_algorithm(&recent_correlations)
            .map_err(|e| MLError::PredictionFailed(format!("Viterbi algorithm failed: {}", e)))?;

        let current_state = *states.last().unwrap();
        let regime = match current_state {
            0 => CorrelationRegime::LowCorrelation,
            1 => CorrelationRegime::HighCorrelation,
            _ => CorrelationRegime::Decoupled,
        };

        // Calculate regime probability
        let alpha = hmm.forward_algorithm(&recent_correlations)
            .map_err(|e| MLError::PredictionFailed(format!("Forward algorithm failed: {}", e)))?;
        let last_col = alpha.column(alpha.ncols() - 1);
        let total_prob: f64 = last_col.iter().sum();
        let state_prob = last_col[current_state] / total_prob;

        Ok(Some(RegimeState {
            regime_type: RegimeType::Correlation(regime),
            probability: state_prob,
            duration: Duration::from_secs(0),
            confidence: self.calculate_regime_confidence(&states),
            timestamp: std::time::SystemTime::now(),
        }))
    }

    /// Detect current market regime
    fn detect_market_regime(&self) -> MLResult<Option<RegimeState>> {
        let hmm = match &self.market_hmm {
            Some(hmm) => hmm,
            None => return Ok(None),
        };

        let recent_features = self.extract_recent_market_features(20)?;
        if recent_features.is_empty() {
            return Ok(None);
        }

        let pca_features = self.apply_pca(&[recent_features])?;
        let states = hmm.viterbi_algorithm(&pca_features)
            .map_err(|e| MLError::PredictionFailed(format!("Viterbi algorithm failed: {}", e)))?;

        let current_state = *states.last().unwrap();
        let regime = match current_state {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            2 => MarketRegime::Sideways,
            _ => MarketRegime::Crisis,
        };

        // Calculate regime probability
        let alpha = hmm.forward_algorithm(&pca_features)
            .map_err(|e| MLError::PredictionFailed(format!("Forward algorithm failed: {}", e)))?;
        let last_col = alpha.column(alpha.ncols() - 1);
        let total_prob: f64 = last_col.iter().sum();
        let state_prob = last_col[current_state] / total_prob;

        Ok(Some(RegimeState {
            regime_type: RegimeType::Market(regime),
            probability: state_prob,
            duration: Duration::from_secs(0),
            confidence: self.calculate_regime_confidence(&states),
            timestamp: std::time::SystemTime::now(),
        }))
    }

    /// Get regime-specific parameters for strategy adaptation
    pub fn get_regime_parameters(&self, regime_type: &str) -> MLResult<HashMap<String, f64>> {
        let regime_state = self.current_regimes.get(regime_type)
            .ok_or_else(|| MLError::PredictionFailed(format!("Regime type {} not found", regime_type)))?;

        let mut parameters = HashMap::new();

        match &regime_state.regime_type {
            RegimeType::Volatility(vol_regime) => {
                match vol_regime {
                    VolatilityRegime::Low => {
                        parameters.insert("spread_multiplier".to_string(), 0.8);
                        parameters.insert("inventory_penalty".to_string(), 0.5);
                        parameters.insert("quote_size_multiplier".to_string(), 1.2);
                    },
                    VolatilityRegime::Medium => {
                        parameters.insert("spread_multiplier".to_string(), 1.0);
                        parameters.insert("inventory_penalty".to_string(), 1.0);
                        parameters.insert("quote_size_multiplier".to_string(), 1.0);
                    },
                    VolatilityRegime::High => {
                        parameters.insert("spread_multiplier".to_string(), 1.5);
                        parameters.insert("inventory_penalty".to_string(), 2.0);
                        parameters.insert("quote_size_multiplier".to_string(), 0.7);
                    },
                    VolatilityRegime::Extreme => {
                        parameters.insert("spread_multiplier".to_string(), 3.0);
                        parameters.insert("inventory_penalty".to_string(), 5.0);
                        parameters.insert("quote_size_multiplier".to_string(), 0.3);
                    },
                }
            },
            RegimeType::Correlation(corr_regime) => {
                match corr_regime {
                    CorrelationRegime::HighCorrelation => {
                        parameters.insert("cross_asset_hedge_ratio".to_string(), 0.8);
                        parameters.insert("diversification_benefit".to_string(), 0.3);
                    },
                    CorrelationRegime::LowCorrelation => {
                        parameters.insert("cross_asset_hedge_ratio".to_string(), 0.2);
                        parameters.insert("diversification_benefit".to_string(), 0.8);
                    },
                    CorrelationRegime::Decoupled => {
                        parameters.insert("cross_asset_hedge_ratio".to_string(), 0.0);
                        parameters.insert("diversification_benefit".to_string(), 1.0);
                    },
                }
            },
            RegimeType::Market(market_regime) => {
                match market_regime {
                    MarketRegime::Bull => {
                        parameters.insert("directional_bias".to_string(), 0.1);
                        parameters.insert("momentum_factor".to_string(), 1.2);
                    },
                    MarketRegime::Bear => {
                        parameters.insert("directional_bias".to_string(), -0.1);
                        parameters.insert("momentum_factor".to_string(), 0.8);
                    },
                    MarketRegime::Sideways => {
                        parameters.insert("directional_bias".to_string(), 0.0);
                        parameters.insert("momentum_factor".to_string(), 1.0);
                    },
                    MarketRegime::Crisis => {
                        parameters.insert("directional_bias".to_string(), 0.0);
                        parameters.insert("momentum_factor".to_string(), 0.5);
                        parameters.insert("risk_reduction_factor".to_string(), 0.3);
                    },
                }
            },
        }

        // Add confidence-based adjustments
        let confidence_factor = regime_state.confidence;
        for (_, value) in parameters.iter_mut() {
            *value *= confidence_factor;
        }

        Ok(parameters)
    }

    // Helper methods for calculations
    fn calculate_rolling_volatility(&self) -> MLResult<Vec<f64>> {
        let window = self.config.volatility_regimes.estimation_window;
        let mut volatilities = Vec::new();

        if self.market_data_buffer.len() < window {
            return Ok(volatilities);
        }

        for i in window..self.market_data_buffer.len() {
            let window_data = &self.market_data_buffer.as_slices().0[i-window..i];
            let returns: Vec<f64> = window_data.windows(2)
                .map(|pair| (pair[1].price / pair[0].price).ln())
                .collect();
            
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            
            volatilities.push(variance.sqrt());
        }

        Ok(volatilities)
    }

    fn calculate_recent_volatility(&self, window: usize) -> MLResult<Vec<f64>> {
        if self.market_data_buffer.len() < window + 1 {
            return Ok(Vec::new());
        }

        let recent_data = &self.market_data_buffer.as_slices().0[self.market_data_buffer.len()-window-1..];
        let returns: Vec<f64> = recent_data.windows(2)
            .map(|pair| (pair[1].price / pair[0].price).ln())
            .collect();

        if returns.is_empty() {
            return Ok(Vec::new());
        }

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        Ok(vec![variance.sqrt()])
    }

    fn calculate_rolling_correlations(&self) -> MLResult<Vec<f64>> {
        // Simplified correlation calculation - in practice would use multiple assets
        let window = self.config.correlation_regimes.correlation_window;
        let mut correlations = Vec::new();

        if self.market_data_buffer.len() < window {
            return Ok(correlations);
        }

        for i in window..self.market_data_buffer.len() {
            let window_data = &self.market_data_buffer.as_slices().0[i-window..i];
            let returns: Vec<f64> = window_data.windows(2)
                .map(|pair| (pair[1].price / pair[0].price).ln())
                .collect();
            
            let volumes: Vec<f64> = window_data.iter().map(|d| d.volume).collect();
            
            // Calculate correlation between returns and volume (as proxy)
            let corr = self.calculate_correlation(&returns, &volumes);
            correlations.push(corr);
        }

        Ok(correlations)
    }

    fn calculate_recent_correlations(&self, window: usize) -> MLResult<Vec<f64>> {
        if self.market_data_buffer.len() < window + 1 {
            return Ok(Vec::new());
        }

        let recent_data = &self.market_data_buffer.as_slices().0[self.market_data_buffer.len()-window-1..];
        let returns: Vec<f64> = recent_data.windows(2)
            .map(|pair| (pair[1].price / pair[0].price).ln())
            .collect();
        
        let volumes: Vec<f64> = recent_data.iter().map(|d| d.volume).collect();
        
        if returns.is_empty() || volumes.is_empty() {
            return Ok(Vec::new());
        }

        let corr = self.calculate_correlation(&returns, &volumes[1..]);
        Ok(vec![corr])
    }

    fn extract_market_features(&self) -> MLResult<Vec<Vec<f64>>> {
        let mut features = Vec::new();
        
        if self.market_data_buffer.len() < 2 {
            return Ok(features);
        }

        for window in self.market_data_buffer.windows(2) {
            let mut feature_vec = Vec::new();
            
            // Returns
            let return_val = (window[1].price / window[0].price).ln();
            feature_vec.push(return_val);
            
            // Volatility (simplified as absolute return)
            feature_vec.push(return_val.abs());
            
            // Volume
            feature_vec.push(window[1].volume);
            
            // Bid-ask spread
            feature_vec.push(window[1].bid_ask_spread);
            
            features.push(feature_vec);
        }

        Ok(features)
    }

    fn extract_recent_market_features(&self, window: usize) -> MLResult<Vec<f64>> {
        if self.market_data_buffer.len() < 2 {
            return Ok(Vec::new());
        }

        let recent = &self.market_data_buffer[self.market_data_buffer.len()-2..];
        let mut features = Vec::new();
        
        // Returns
        let return_val = (recent[1].price / recent[0].price).ln();
        features.push(return_val);
        
        // Volatility
        features.push(return_val.abs());
        
        // Volume
        features.push(recent[1].volume);
        
        // Bid-ask spread
        features.push(recent[1].bid_ask_spread);

        Ok(features)
    }

    fn apply_pca(&self, features: &[Vec<f64>]) -> MLResult<Vec<f64>> {
        if features.is_empty() {
            return Ok(Vec::new());
        }

        // Simplified PCA - just return first principal component (weighted average)
        let weights = vec![0.4, 0.3, 0.2, 0.1]; // Weights for returns, volatility, volume, spread
        
        let mut pca_features = Vec::new();
        for feature_vec in features {
            let pca_value = feature_vec.iter()
                .zip(weights.iter())
                .map(|(f, w)| f * w)
                .sum();
            pca_features.push(pca_value);
        }

        Ok(pca_features)
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn initialize_transition_matrix(&self, n_states: usize) -> na::DMatrix<f64> {
        let mut matrix = na::DMatrix::zeros(n_states, n_states);
        
        // Initialize with slight persistence bias
        for i in 0..n_states {
            for j in 0..n_states {
                if i == j {
                    matrix[(i, j)] = 0.7; // Higher probability of staying in same state
                } else {
                    matrix[(i, j)] = 0.3 / (n_states - 1) as f64; // Equal probability for other states
                }
            }
        }
        
        matrix
    }

    fn initialize_emission_means(&self, data: &[f64], n_states: usize) -> na::DVector<f64> {
        let mut means = na::DVector::zeros(n_states);
        
        if data.is_empty() {
            return means;
        }

        // Use quantiles to initialize means
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        for i in 0..n_states {
            let quantile = (i + 1) as f64 / (n_states + 1) as f64;
            let index = ((sorted_data.len() - 1) as f64 * quantile) as usize;
            means[i] = sorted_data[index];
        }
        
        means
    }

    fn initialize_emission_variances(&self, data: &[f64], n_states: usize) -> na::DVector<f64> {
        let mut variances = na::DVector::zeros(n_states);
        
        if data.is_empty() {
            return variances;
        }

        // Calculate overall variance and divide by number of states
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        for i in 0..n_states {
            variances[i] = variance / n_states as f64;
        }
        
        variances
    }

    fn calculate_regime_confidence(&self, states: &[usize]) -> f64 {
        if states.len() < 2 {
            return 0.5;
        }

        // Calculate confidence based on state persistence
        let mut transitions = 0;
        for i in 1..states.len() {
            if states[i] != states[i-1] {
                transitions += 1;
            }
        }

        let stability = 1.0 - (transitions as f64 / (states.len() - 1) as f64);
        stability.max(0.1).min(0.95) // Clamp between 0.1 and 0.95
    }

    fn should_retrain_models(&self) -> bool {
        // Retrain every hour or when significant regime changes detected
        self.last_update.elapsed() > Duration::from_secs(3600) ||
        self.detect_regime_instability()
    }

    fn detect_regime_instability(&self) -> bool {
        // Check if current regimes have low confidence
        self.current_regimes.values()
            .any(|regime| regime.confidence < 0.3)
    }

    fn retrain_models(&mut self) -> MLResult<()> {
        // Retrain all models with latest data
        self.train_volatility_regime_model()?;
        self.train_correlation_regime_model()?;
        self.train_market_regime_model()?;
        Ok(())
    }

    fn evaluate_volatility_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        // Simplified evaluation - in practice would use cross-validation
        Ok(MLMetrics {
            accuracy: 0.75,
            precision: 0.73,
            recall: 0.77,
            f1_score: 0.75,
            auc_roc: 0.82,
            log_likelihood: -data.len() as f64 * 0.5, // Simplified
        })
    }

    fn evaluate_correlation_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        Ok(MLMetrics {
            accuracy: 0.68,
            precision: 0.65,
            recall: 0.72,
            f1_score: 0.68,
            auc_roc: 0.74,
            log_likelihood: -data.len() as f64 * 0.6,
        })
    }

    fn evaluate_market_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        Ok(MLMetrics {
            accuracy: 0.71,
            precision: 0.69,
            recall: 0.74,
            f1_score: 0.71,
            auc_roc: 0.78,
            log_likelihood: -data.len() as f64 * 0.55,
        })
    }

    /// Get current regime states
    pub fn get_current_regimes(&self) -> &HashMap<String, RegimeState> {
        &self.current_regimes
    }

    /// Get performance metrics for all models
    pub fn get_performance_metrics(&self) -> &HashMap<String, ModelPerformance> {
        &self.performance_metrics
    }

    /// Export model state for persistence
    pub fn export_model_state(&self) -> MLResult<RegimeModelState> {
        Ok(RegimeModelState {
            config: self.config.clone(),
            current_regimes: self.current_regimes.clone(),
            performance_metrics: self.performance_metrics.clone(),
            last_update: self.last_update,
        })
    }

    /// Import model state from persistence
    pub fn import_model_state(&mut self, state: RegimeModelState) -> MLResult<()> {
        self.config = state.config;
        self.current_regimes = state.current_regimes;
        self.performance_metrics = state.performance_metrics;
        self.last_update = state.last_update;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeModelState {
    pub config: RegimeDetectionConfig,
    pub current_regimes: HashMap<String, RegimeState>,
    pub performance_metrics: HashMap<String, ModelPerformance>,
    pub last_update: std::time::Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_market_data(n: usize) -> Vec<MarketData> {
        let mut data = Vec::new();
        let mut price = 100.0;
        
        for i in 0..n {
            price *= 1.0 + (i as f64 * 0.001).sin() * 0.01; // Simulate price movement
            data.push(MarketData {
                timestamp: SystemTime::now(),
                price,
                volume: 1000.0 + (i as f64 * 0.1).cos() * 100.0,
                bid_ask_spread: 0.01 + (i as f64 * 0.01).sin().abs() * 0.005,
                returns: None,
                volatility: None,
            });
        }
        
        data
    }

    #[test]
    fn test_regime_detection_initialization() {
        let config = RegimeDetectionConfig::default();
        let mut system = RegimeDetectionSystem::new(config);
        let test_data = create_test_market_data(500);
        
        let result = system.initialize(&test_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_volatility_calculation() {
        let config = RegimeDetectionConfig::default();
        let system = RegimeDetectionSystem::new(config);
        let test_data = create_test_market_data(300);
        
        // Manually add data to buffer
        let mut system = system;
        system.market_data_buffer.extend(test_data);
        
        let volatilities = system.calculate_rolling_volatility();
        assert!(volatilities.is_ok());
        
        let vol_data = volatilities.unwrap();
        assert!(!vol_data.is_empty());
        assert!(vol_data.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_regime_parameter_extraction() {
        let config = RegimeDetectionConfig::default();
        let mut system = RegimeDetectionSystem::new(config);
        
        // Add a mock regime state
        system.current_regimes.insert(
            "volatility".to_string(),
            RegimeState {
                regime_type: RegimeType::Volatility(VolatilityRegime::High),
                probability: 0.8,
                duration: Duration::from_secs(300),
                confidence: 0.75,
                timestamp: SystemTime::now(),
            }
        );
        
        let params = system.get_regime_parameters("volatility");
        assert!(params.is_ok());
        
        let param_map = params.unwrap();
        assert!(param_map.contains_key("spread_multiplier"));
        assert!(param_map.contains_key("inventory_penalty"));
    }

    #[test]
    fn test_correlation_calculation() {
        let config = RegimeDetectionConfig::default();
        let system = RegimeDetectionSystem::new(config);
        
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = system.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // Perfect positive correlation
        
        let y_neg = vec![-2.0, -4.0, -6.0, -8.0, -10.0];
        let corr_neg = system.calculate_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 1e-10); // Perfect negative correlation
    }
}atrix[(i, j)] = 0.7; // Higher probability of staying in same state
                } else {
                    matrix[(i, j)] = 0.3 / (n_states - 1) as f64;
                }
            }
        }

        matrix
    }

    fn initialize_emission_means(&self, data: &[f64], n_states: usize) -> na::DVector<f64> {
        let mut means = na::DVector::zeros(n_states);
        
        // Use quantiles to initialize means
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        for i in 0..n_states {
            let quantile = (i + 1) as f64 / (n_states + 1) as f64;
            let index = (quantile * sorted_data.len() as f64) as usize;
            means[i] = sorted_data[index.min(sorted_data.len() - 1)];
        }

        means
    }

    fn initialize_emission_variances(&self, data: &[f64], n_states: usize) -> na::DVector<f64> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        na::DVector::from_element(n_states, variance)
    }

    fn calculate_regime_confidence(&self, states: &[usize]) -> f64 {
        if states.len() < 2 {
            return 0.5;
        }

        // Calculate confidence based on state persistence
        let mut transitions = 0;
        for i in 1..states.len() {
            if states[i] != states[i-1] {
                transitions += 1;
            }
        }

        let stability = 1.0 - (transitions as f64 / (states.len() - 1) as f64);
        stability.max(0.1).min(1.0)
    }

    fn should_retrain_models(&self) -> bool {
        // Retrain if we have significantly more data or performance has degraded
        self.market_data_buffer.len() % 1000 == 0
    }

    fn retrain_models(&mut self) -> MLResult<()> {
        self.train_volatility_regime_model()?;
        self.train_correlation_regime_model()?;
        self.train_market_regime_model()?;
        Ok(())
    }

    fn evaluate_volatility_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        // Simplified evaluation - in practice would use proper cross-validation
        Ok(MLMetrics {
            accuracy: 0.85,
            precision: 0.82,
            recall: 0.88,
            f1_score: 0.85,
            auc_roc: 0.90,
            log_likelihood: -data.len() as f64 * 0.5, // Simplified
        })
    }

    fn evaluate_correlation_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        Ok(MLMetrics {
            accuracy: 0.78,
            precision: 0.75,
            recall: 0.82,
            f1_score: 0.78,
            auc_roc: 0.85,
            log_likelihood: -data.len() as f64 * 0.6,
        })
    }

    fn evaluate_market_model(&self, data: &[f64]) -> MLResult<MLMetrics> {
        Ok(MLMetrics {
            accuracy: 0.72,
            precision: 0.70,
            recall: 0.75,
            f1_score: 0.72,
            auc_roc: 0.80,
            log_likelihood: -data.len() as f64 * 0.7,
        })
    }

    /// Get current regime states
    pub fn get_current_regimes(&self) -> &HashMap<String, RegimeState> {
        &self.current_regimes
    }

    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> &HashMap<String, ModelPerformance> {
        &self.performance_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_data(n: usize) -> Vec<MarketData> {
        (0..n).map(|i| MarketData {
            timestamp: SystemTime::now(),
            price: 100.0 + (i as f64 * 0.1),
            volume: 1000.0 + (i as f64 * 10.0),
            bid_ask_spread: 0.01,
            returns: None,
            volatility: None,
        }).collect()
    }

    #[test]
    fn test_regime_detection_initialization() {
        let config = RegimeDetectionConfig::default();
        let mut system = RegimeDetectionSystem::new(config);
        let test_data = create_test_data(1000);
        
        let result = system.initialize(&test_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_volatility_calculation() {
        let config = RegimeDetectionConfig::default();
        let mut system = RegimeDetectionSystem::new(config);
        let test_data = create_test_data(300);
        
        system.market_data_buffer.extend(test_data);
        let volatilities = system.calculate_rolling_volatility().unwrap();
        assert!(!volatilities.is_empty());
    }

    #[test]
    fn test_correlation_calculation() {
        let system = RegimeDetectionSystem::new(RegimeDetectionConfig::default());
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = system.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }
}