//! Cartea-Jaimungal Jump-Diffusion Market Making Model
//!
//! This module implements the Cartea-Jaimungal jump-diffusion market making model
//! with advanced jump detection, parameter estimation, and risk premium calculation.

use crate::math::fixed_point::FixedPoint;
use crate::math::jump_diffusion::{
    JumpEvent, JumpDiffusionError, BiPowerVariationJumpDetector, JumpClusteringDetector,
    JumpSizeDistribution, GBMJumpParams
};
use crate::math::hawkes_process::MultivariateHawkesSimulator;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CarteaJaimungalError {
    #[error("Jump diffusion error: {0}")]
    JumpDiffusionError(#[from] JumpDiffusionError),
    #[error("Parameter estimation error: {0}")]
    ParameterEstimationError(String),
    #[error("Jump detection error: {0}")]
    JumpDetectionError(String),
    #[error("Risk calculation error: {0}")]
    RiskCalculationError(String),
}

/// Market regime states for jump-diffusion modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegimeState {
    Normal,
    HighVolatility,
    Crisis,
    Recovery,
}

/// Jump parameters for double exponential distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpParameters {
    pub lambda: FixedPoint,          // Jump intensity
    pub eta_plus: FixedPoint,        // Upward jump decay parameter
    pub eta_minus: FixedPoint,       // Downward jump decay parameter
    pub p: FixedPoint,               // Probability of upward jump
    pub regime_dependent: bool,
    pub regime_parameters: HashMap<RegimeState, JumpParameters>,
}

impl JumpParameters {
    pub fn new(
        lambda: FixedPoint,
        eta_plus: FixedPoint,
        eta_minus: FixedPoint,
        p: FixedPoint,
    ) -> Result<Self, CarteaJaimungalError> {
        if lambda.to_float() < 0.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump intensity must be non-negative".to_string()
            ));
        }
        
        if eta_plus.to_float() <= 0.0 || eta_minus.to_float() <= 0.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump decay parameters must be positive".to_string()
            ));
        }
        
        if p.to_float() < 0.0 || p.to_float() > 1.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump probability must be in [0,1]".to_string()
            ));
        }

        Ok(Self {
            lambda,
            eta_plus,
            eta_minus,
            p,
            regime_dependent: false,
            regime_parameters: HashMap::new(),
        })
    }

    /// Calculate expected jump size: E[J] = p/η⁺ - (1-p)/η⁻
    pub fn expected_jump_size(&self) -> FixedPoint {
        self.p / self.eta_plus - (FixedPoint::one() - self.p) / self.eta_minus
    }

    /// Calculate jump size variance
    pub fn jump_size_variance(&self) -> FixedPoint {
        let mean = self.expected_jump_size();
        self.p / (self.eta_plus * self.eta_plus) + 
        (FixedPoint::one() - self.p) / (self.eta_minus * self.eta_minus) - 
        mean * mean
    }

    /// Get parameters for specific regime
    pub fn get_regime_parameters(&self, regime: RegimeState) -> &JumpParameters {
        if self.regime_dependent {
            self.regime_parameters.get(&regime).unwrap_or(self)
        } else {
            self
        }
    }
}

/// Advanced jump detection with bi-power variation and statistical significance
pub struct AdvancedJumpDetector {
    pub bipower_detector: BiPowerVariationJumpDetector,
    pub clustering_detector: JumpClusteringDetector,
    pub significance_level: FixedPoint,
    pub adaptive_threshold_multiplier: FixedPoint,
    pub regime_dependent_thresholds: HashMap<RegimeState, FixedPoint>,
    pub jump_history: VecDeque<JumpEvent>,
    pub max_history: usize,
}

impl AdvancedJumpDetector {
    pub fn new(
        significance_level: FixedPoint,
        window_size: usize,
        max_history: usize,
    ) -> Self {
        let bipower_detector = BiPowerVariationJumpDetector::new(significance_level, window_size);
        
        let clustering_detector = JumpClusteringDetector::new(
            FixedPoint::from_float(0.1),  // baseline intensity
            FixedPoint::from_float(0.5),  // self-excitation
            FixedPoint::from_float(2.0),  // decay rate
        );

        let mut regime_thresholds = HashMap::new();
        regime_thresholds.insert(RegimeState::Normal, FixedPoint::from_float(3.0));
        regime_thresholds.insert(RegimeState::HighVolatility, FixedPoint::from_float(2.5));
        regime_thresholds.insert(RegimeState::Crisis, FixedPoint::from_float(2.0));
        regime_thresholds.insert(RegimeState::Recovery, FixedPoint::from_float(2.8));

        Self {
            bipower_detector,
            clustering_detector,
            significance_level,
            adaptive_threshold_multiplier: FixedPoint::from_float(1.0),
            regime_dependent_thresholds: regime_thresholds,
            jump_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Implement bi-power variation jump test with statistical significance
    pub fn detect_jumps_bipower_variation(
        &self,
        returns: &[FixedPoint],
        regime: RegimeState,
    ) -> Result<Vec<usize>, CarteaJaimungalError> {
        if returns.len() < self.bipower_detector.window_size {
            return Err(CarteaJaimungalError::JumpDetectionError(
                "Insufficient data for bi-power variation test".to_string()
            ));
        }

        let mut jump_indices = Vec::new();
        let regime_threshold = self.regime_dependent_thresholds
            .get(&regime)
            .unwrap_or(&FixedPoint::from_float(3.0));

        for i in self.bipower_detector.window_size..returns.len() {
            let window_start = i.saturating_sub(self.bipower_detector.window_size);
            let window_returns = &returns[window_start..i];

            // Calculate realized variance (quadratic variation)
            let rv = window_returns.iter()
                .map(|r| *r * *r)
                .fold(FixedPoint::zero(), |acc, x| acc + x);

            // Calculate bi-power variation: BV = (π/2) * Σ|r_{i-1}||r_i|
            let mut bv = FixedPoint::zero();
            for j in 1..window_returns.len() {
                bv = bv + window_returns[j-1].abs() * window_returns[j].abs();
            }
            bv = bv * FixedPoint::from_float(std::f64::consts::PI / 2.0);

            // Jump component: J = RV - BV
            let jump_component = rv - bv;
            
            // Theoretical variance of the test statistic
            // Var(J) ≈ (π²/4 + π - 5) * ∫σ⁴dt ≈ 2.61 * BV for high-frequency data
            let test_variance = bv * FixedPoint::from_float(2.61);
            
            if test_variance > FixedPoint::zero() {
                // Standardized test statistic: Z = J / √Var(J)
                let test_statistic = jump_component / test_variance.sqrt();
                
                // Adaptive threshold based on regime and recent volatility
                let adaptive_threshold = *regime_threshold * self.adaptive_threshold_multiplier;
                
                // Apply statistical significance test
                if test_statistic.abs() > adaptive_threshold {
                    jump_indices.push(i);
                }
            }
        }

        Ok(jump_indices)
    }

    /// Create threshold-based jump identification with adaptive thresholds
    pub fn detect_jumps_threshold_based(
        &mut self,
        returns: &[FixedPoint],
        prices: &[FixedPoint],
        regime: RegimeState,
    ) -> Result<Vec<JumpEvent>, CarteaJaimungalError> {
        if returns.len() != prices.len() - 1 {
            return Err(CarteaJaimungalError::JumpDetectionError(
                "Returns and prices length mismatch".to_string()
            ));
        }

        let mut jump_events = Vec::new();
        
        // Calculate rolling volatility for adaptive threshold
        let window_size = 20.min(returns.len() / 4);
        
        for i in window_size..returns.len() {
            let window_start = i.saturating_sub(window_size);
            let window_returns = &returns[window_start..i];
            
            // Calculate local volatility estimate
            let mean_return = window_returns.iter()
                .fold(FixedPoint::zero(), |acc, x| acc + *x) / 
                FixedPoint::from_float(window_returns.len() as f64);
            
            let variance = window_returns.iter()
                .map(|r| (*r - mean_return) * (*r - mean_return))
                .fold(FixedPoint::zero(), |acc, x| acc + x) / 
                FixedPoint::from_float((window_returns.len() - 1) as f64);
            
            let local_volatility = variance.sqrt();
            
            // Adaptive threshold based on regime and local volatility
            let base_threshold = self.regime_dependent_thresholds
                .get(&regime)
                .unwrap_or(&FixedPoint::from_float(3.0));
            
            let adaptive_threshold = *base_threshold * local_volatility * self.adaptive_threshold_multiplier;
            
            // Check if current return exceeds threshold
            if returns[i].abs() > adaptive_threshold {
                let jump_event = JumpEvent {
                    time: FixedPoint::from_float(i as f64), // Simplified time indexing
                    size: returns[i],
                    price_before: prices[i],
                    price_after: prices[i + 1],
                };
                
                jump_events.push(jump_event.clone());
                
                // Add to history
                if self.jump_history.len() >= self.max_history {
                    self.jump_history.pop_front();
                }
                self.jump_history.push_back(jump_event);
            }
        }

        Ok(jump_events)
    }

    /// Add jump clustering detection using Hawkes process framework
    pub fn detect_jump_clustering(
        &self,
        jump_events: &[JumpEvent],
        threshold_multiplier: FixedPoint,
    ) -> Result<Vec<(FixedPoint, FixedPoint)>, CarteaJaimungalError> {
        if jump_events.is_empty() {
            return Ok(Vec::new());
        }

        let clustering_periods = self.clustering_detector.detect_clustering_periods(
            jump_events,
            threshold_multiplier,
        );

        Ok(clustering_periods)
    }

    /// Update adaptive threshold based on recent market conditions
    pub fn update_adaptive_threshold(&mut self, recent_volatility: FixedPoint, regime: RegimeState) {
        // Adjust threshold multiplier based on recent volatility
        let volatility_adjustment = if recent_volatility > FixedPoint::from_float(0.3) {
            FixedPoint::from_float(0.8) // Lower threshold in high volatility
        } else if recent_volatility < FixedPoint::from_float(0.1) {
            FixedPoint::from_float(1.2) // Higher threshold in low volatility
        } else {
            FixedPoint::one()
        };

        // Regime-specific adjustments
        let regime_adjustment = match regime {
            RegimeState::Crisis => FixedPoint::from_float(0.7),
            RegimeState::HighVolatility => FixedPoint::from_float(0.85),
            RegimeState::Recovery => FixedPoint::from_float(0.9),
            RegimeState::Normal => FixedPoint::one(),
        };

        self.adaptive_threshold_multiplier = volatility_adjustment * regime_adjustment;
    }

    /// Get jump intensity estimate from recent history
    pub fn estimate_current_jump_intensity(&self, lookback_time: FixedPoint) -> FixedPoint {
        if self.jump_history.is_empty() {
            return FixedPoint::zero();
        }

        let current_time = self.jump_history.back().unwrap().time;
        let cutoff_time = current_time - lookback_time;

        let recent_jumps = self.jump_history
            .iter()
            .filter(|jump| jump.time >= cutoff_time)
            .count();

        FixedPoint::from_float(recent_jumps as f64) / lookback_time
    }
}

/// Maximum likelihood estimator for jump parameters
pub struct JumpParameterEstimator {
    pub convergence_tolerance: FixedPoint,
    pub max_iterations: usize,
    pub regularization_lambda: FixedPoint,
}

impl JumpParameterEstimator {
    pub fn new(convergence_tolerance: FixedPoint, max_iterations: usize) -> Self {
        Self {
            convergence_tolerance,
            max_iterations,
            regularization_lambda: FixedPoint::from_float(1e-6),
        }
    }

    /// Create maximum likelihood estimation for double exponential jumps
    pub fn estimate_double_exponential_parameters(
        &self,
        jump_sizes: &[FixedPoint],
    ) -> Result<JumpParameters, CarteaJaimungalError> {
        if jump_sizes.is_empty() {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "No jump sizes provided for estimation".to_string()
            ));
        }

        // Separate positive and negative jumps
        let positive_jumps: Vec<FixedPoint> = jump_sizes.iter()
            .filter(|&&x| x.to_float() > 0.0)
            .cloned()
            .collect();
        
        let negative_jumps: Vec<FixedPoint> = jump_sizes.iter()
            .filter(|&&x| x.to_float() < 0.0)
            .map(|&x| -x) // Make positive for exponential fitting
            .collect();

        if positive_jumps.is_empty() && negative_jumps.is_empty() {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "No valid jump sizes for estimation".to_string()
            ));
        }

        // Estimate probability of upward jump
        let p = if jump_sizes.len() > 0 {
            FixedPoint::from_float(positive_jumps.len() as f64 / jump_sizes.len() as f64)
        } else {
            FixedPoint::from_float(0.5)
        };

        // Estimate eta_plus (upward jump decay parameter)
        let eta_plus = if !positive_jumps.is_empty() {
            self.estimate_exponential_parameter(&positive_jumps)?
        } else {
            FixedPoint::from_float(10.0) // Default value
        };

        // Estimate eta_minus (downward jump decay parameter)  
        let eta_minus = if !negative_jumps.is_empty() {
            self.estimate_exponential_parameter(&negative_jumps)?
        } else {
            FixedPoint::from_float(10.0) // Default value
        };

        // Estimate jump intensity (would typically use time series data)
        let lambda = FixedPoint::from_float(jump_sizes.len() as f64); // Simplified

        JumpParameters::new(lambda, eta_plus, eta_minus, p)
    }

    /// Implement separate estimation for upward (η⁺) and downward (η⁻) jump parameters
    fn estimate_exponential_parameter(
        &self,
        samples: &[FixedPoint],
    ) -> Result<FixedPoint, CarteaJaimungalError> {
        if samples.is_empty() {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "No samples provided for exponential parameter estimation".to_string()
            ));
        }

        // Maximum likelihood estimator for exponential distribution: λ = 1/mean
        let sum = samples.iter().fold(FixedPoint::zero(), |acc, &x| acc + x);
        let mean = sum / FixedPoint::from_float(samples.len() as f64);
        
        if mean <= FixedPoint::zero() {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Invalid mean for exponential parameter estimation".to_string()
            ));
        }

        Ok(FixedPoint::one() / mean)
    }

    /// Add time-varying jump intensity modeling
    pub fn estimate_time_varying_intensity(
        &self,
        jump_times: &[FixedPoint],
        observation_period: FixedPoint,
        window_size: usize,
    ) -> Result<Vec<(FixedPoint, FixedPoint)>, CarteaJaimungalError> {
        if jump_times.is_empty() {
            return Ok(Vec::new());
        }

        let mut intensity_estimates = Vec::new();
        let window_duration = observation_period / FixedPoint::from_float(window_size as f64);

        for i in 0..window_size {
            let window_start = FixedPoint::from_float(i as f64) * window_duration;
            let window_end = window_start + window_duration;

            // Count jumps in this window
            let jumps_in_window = jump_times.iter()
                .filter(|&&t| t >= window_start && t < window_end)
                .count();

            // Estimate intensity as jumps per unit time
            let intensity = FixedPoint::from_float(jumps_in_window as f64) / window_duration;
            
            intensity_estimates.push((window_start + window_duration / FixedPoint::from_float(2.0), intensity));
        }

        Ok(intensity_estimates)
    }

    /// Create parameter validation and stability checks
    pub fn validate_parameters(&self, params: &JumpParameters) -> Result<(), CarteaJaimungalError> {
        // Check parameter bounds
        if params.lambda.to_float() < 0.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump intensity must be non-negative".to_string()
            ));
        }

        if params.eta_plus.to_float() <= 0.0 || params.eta_minus.to_float() <= 0.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump decay parameters must be positive".to_string()
            ));
        }

        if params.p.to_float() < 0.0 || params.p.to_float() > 1.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump probability must be in [0,1]".to_string()
            ));
        }

        // Check stability conditions
        let expected_jump_size = params.expected_jump_size();
        if expected_jump_size.abs().to_float() > 0.5 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Expected jump size too large for stability".to_string()
            ));
        }

        // Check that parameters are not extreme
        if params.eta_plus.to_float() > 1000.0 || params.eta_minus.to_float() > 1000.0 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump decay parameters too large".to_string()
            ));
        }

        if params.eta_plus.to_float() < 0.1 || params.eta_minus.to_float() < 0.1 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Jump decay parameters too small".to_string()
            ));
        }

        Ok(())
    }

    /// Estimate parameters with confidence intervals
    pub fn estimate_with_confidence_intervals(
        &self,
        jump_sizes: &[FixedPoint],
        confidence_level: FixedPoint,
    ) -> Result<(JumpParameters, JumpParameterConfidenceIntervals), CarteaJaimungalError> {
        let params = self.estimate_double_exponential_parameters(jump_sizes)?;
        
        // Bootstrap confidence intervals
        let n_bootstrap = 1000;
        let mut bootstrap_params = Vec::new();
        let mut rng_state = 12345u64; // Simple LCG seed

        for _ in 0..n_bootstrap {
            // Bootstrap sample using simple LCG
            let mut bootstrap_sample = Vec::new();
            for _ in 0..jump_sizes.len() {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let idx = (rng_state as usize) % jump_sizes.len();
                bootstrap_sample.push(jump_sizes[idx]);
            }

            if let Ok(bootstrap_param) = self.estimate_double_exponential_parameters(&bootstrap_sample) {
                bootstrap_params.push(bootstrap_param);
            }
        }

        let confidence_intervals = self.calculate_confidence_intervals(
            &bootstrap_params,
            confidence_level,
        )?;

        Ok((params, confidence_intervals))
    }

    fn calculate_confidence_intervals(
        &self,
        bootstrap_params: &[JumpParameters],
        confidence_level: FixedPoint,
    ) -> Result<JumpParameterConfidenceIntervals, CarteaJaimungalError> {
        if bootstrap_params.is_empty() {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "No bootstrap parameters for confidence intervals".to_string()
            ));
        }

        let alpha = FixedPoint::one() - confidence_level;
        let lower_percentile = alpha / FixedPoint::from_float(2.0);
        let upper_percentile = FixedPoint::one() - lower_percentile;

        // Extract parameter vectors
        let mut lambdas: Vec<f64> = bootstrap_params.iter().map(|p| p.lambda.to_float()).collect();
        let mut eta_plus_vec: Vec<f64> = bootstrap_params.iter().map(|p| p.eta_plus.to_float()).collect();
        let mut eta_minus_vec: Vec<f64> = bootstrap_params.iter().map(|p| p.eta_minus.to_float()).collect();
        let mut p_vec: Vec<f64> = bootstrap_params.iter().map(|p| p.p.to_float()).collect();

        // Sort for percentile calculation
        lambdas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eta_plus_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eta_minus_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        p_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = bootstrap_params.len();
        let lower_idx = ((lower_percentile.to_float() * n as f64) as usize).min(n - 1);
        let upper_idx = ((upper_percentile.to_float() * n as f64) as usize).min(n - 1);

        Ok(JumpParameterConfidenceIntervals {
            lambda_ci: (FixedPoint::from_float(lambdas[lower_idx]), FixedPoint::from_float(lambdas[upper_idx])),
            eta_plus_ci: (FixedPoint::from_float(eta_plus_vec[lower_idx]), FixedPoint::from_float(eta_plus_vec[upper_idx])),
            eta_minus_ci: (FixedPoint::from_float(eta_minus_vec[lower_idx]), FixedPoint::from_float(eta_minus_vec[upper_idx])),
            p_ci: (FixedPoint::from_float(p_vec[lower_idx]), FixedPoint::from_float(p_vec[upper_idx])),
            confidence_level,
        })
    }
}

/// Confidence intervals for jump parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpParameterConfidenceIntervals {
    pub lambda_ci: (FixedPoint, FixedPoint),
    pub eta_plus_ci: (FixedPoint, FixedPoint),
    pub eta_minus_ci: (FixedPoint, FixedPoint),
    pub p_ci: (FixedPoint, FixedPoint),
    pub confidence_level: FixedPoint,
}

/// Time-varying jump intensity model
pub struct TimeVaryingJumpIntensity {
    pub baseline_intensity: FixedPoint,
    pub volatility_coefficient: FixedPoint,
    pub momentum_coefficient: FixedPoint,
    pub mean_reversion_speed: FixedPoint,
    pub intensity_history: VecDeque<(FixedPoint, FixedPoint)>,
    pub max_history: usize,
}

impl TimeVaryingJumpIntensity {
    pub fn new(
        baseline_intensity: FixedPoint,
        volatility_coefficient: FixedPoint,
        momentum_coefficient: FixedPoint,
        mean_reversion_speed: FixedPoint,
        max_history: usize,
    ) -> Self {
        Self {
            baseline_intensity,
            volatility_coefficient,
            momentum_coefficient,
            mean_reversion_speed,
            intensity_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Calculate time-varying intensity: λ(t) = λ₀ + α*σ(t) + β*|r(t-1)| - γ*(λ(t-1) - λ₀)
    pub fn calculate_intensity(
        &mut self,
        current_time: FixedPoint,
        current_volatility: FixedPoint,
        previous_return: FixedPoint,
    ) -> FixedPoint {
        let previous_intensity = self.intensity_history
            .back()
            .map(|(_, intensity)| *intensity)
            .unwrap_or(self.baseline_intensity);

        let volatility_component = self.volatility_coefficient * current_volatility;
        let momentum_component = self.momentum_coefficient * previous_return.abs();
        let mean_reversion_component = self.mean_reversion_speed * (previous_intensity - self.baseline_intensity);

        let new_intensity = self.baseline_intensity + volatility_component + momentum_component - mean_reversion_component;
        
        // Ensure intensity is non-negative
        let new_intensity = new_intensity.max(FixedPoint::zero());

        // Update history
        if self.intensity_history.len() >= self.max_history {
            self.intensity_history.pop_front();
        }
        self.intensity_history.push_back((current_time, new_intensity));

        new_intensity
    }

    /// Update model parameters based on recent data
    pub fn update_parameters(
        &mut self,
        jump_events: &[JumpEvent],
        volatility_series: &[(FixedPoint, FixedPoint)],
        return_series: &[(FixedPoint, FixedPoint)],
    ) -> Result<(), CarteaJaimungalError> {
        if jump_events.len() < 10 || volatility_series.len() < 10 || return_series.len() < 10 {
            return Err(CarteaJaimungalError::ParameterEstimationError(
                "Insufficient data for parameter update".to_string()
            ));
        }

        // Simple regression-based parameter update (simplified implementation)
        // In practice, would use maximum likelihood or Kalman filtering

        // Update baseline intensity
        let total_time = jump_events.last().unwrap().time - jump_events.first().unwrap().time;
        self.baseline_intensity = FixedPoint::from_float(jump_events.len() as f64) / total_time;

        // Update volatility coefficient (simplified)
        let avg_volatility = volatility_series.iter()
            .map(|(_, vol)| *vol)
            .fold(FixedPoint::zero(), |acc, x| acc + x) / 
            FixedPoint::from_float(volatility_series.len() as f64);

        if avg_volatility > FixedPoint::zero() {
            self.volatility_coefficient = self.baseline_intensity / avg_volatility * FixedPoint::from_float(0.5);
        }

        Ok(())
    }
}

/// Regime detector for jump-diffusion modeling
pub struct RegimeDetector {
    pub volatility_thresholds: HashMap<RegimeState, (FixedPoint, FixedPoint)>,
    pub jump_intensity_thresholds: HashMap<RegimeState, FixedPoint>,
    pub transition_probabilities: HashMap<(RegimeState, RegimeState), FixedPoint>,
    pub current_regime: RegimeState,
    pub regime_history: VecDeque<(FixedPoint, RegimeState)>,
    pub max_history: usize,
}

impl RegimeDetector {
    pub fn new(max_history: usize) -> Self {
        let mut volatility_thresholds = HashMap::new();
        volatility_thresholds.insert(RegimeState::Normal, (FixedPoint::from_float(0.05), FixedPoint::from_float(0.25)));
        volatility_thresholds.insert(RegimeState::HighVolatility, (FixedPoint::from_float(0.25), FixedPoint::from_float(0.5)));
        volatility_thresholds.insert(RegimeState::Crisis, (FixedPoint::from_float(0.5), FixedPoint::from_float(1.0)));
        volatility_thresholds.insert(RegimeState::Recovery, (FixedPoint::from_float(0.15), FixedPoint::from_float(0.35)));

        let mut jump_intensity_thresholds = HashMap::new();
        jump_intensity_thresholds.insert(RegimeState::Normal, FixedPoint::from_float(0.1));
        jump_intensity_thresholds.insert(RegimeState::HighVolatility, FixedPoint::from_float(0.3));
        jump_intensity_thresholds.insert(RegimeState::Crisis, FixedPoint::from_float(0.8));
        jump_intensity_thresholds.insert(RegimeState::Recovery, FixedPoint::from_float(0.2));

        // Simplified transition probabilities (would be estimated from data in practice)
        let mut transition_probabilities = HashMap::new();
        transition_probabilities.insert((RegimeState::Normal, RegimeState::HighVolatility), FixedPoint::from_float(0.1));
        transition_probabilities.insert((RegimeState::HighVolatility, RegimeState::Crisis), FixedPoint::from_float(0.2));
        transition_probabilities.insert((RegimeState::Crisis, RegimeState::Recovery), FixedPoint::from_float(0.3));
        transition_probabilities.insert((RegimeState::Recovery, RegimeState::Normal), FixedPoint::from_float(0.4));

        Self {
            volatility_thresholds,
            jump_intensity_thresholds,
            transition_probabilities,
            current_regime: RegimeState::Normal,
            regime_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Detect current regime based on volatility and jump intensity
    pub fn detect_regime(
        &mut self,
        current_time: FixedPoint,
        volatility: FixedPoint,
        jump_intensity: FixedPoint,
    ) -> RegimeState {
        let mut regime_scores = HashMap::new();

        // Score each regime based on current conditions
        for (&regime, &(vol_min, vol_max)) in &self.volatility_thresholds {
            let vol_score = if volatility >= vol_min && volatility <= vol_max {
                FixedPoint::one()
            } else {
                FixedPoint::zero()
            };

            let jump_threshold = self.jump_intensity_thresholds[&regime];
            let jump_score = if jump_intensity <= jump_threshold * FixedPoint::from_float(1.5) {
                FixedPoint::one() - (jump_intensity - jump_threshold).abs() / jump_threshold
            } else {
                FixedPoint::zero()
            };

            regime_scores.insert(regime, vol_score * FixedPoint::from_float(0.6) + jump_score * FixedPoint::from_float(0.4));
        }

        // Find regime with highest score
        let new_regime = regime_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(&regime, _)| regime)
            .unwrap_or(RegimeState::Normal);

        // Update regime if changed
        if new_regime != self.current_regime {
            self.current_regime = new_regime;
            
            if self.regime_history.len() >= self.max_history {
                self.regime_history.pop_front();
            }
            self.regime_history.push_back((current_time, new_regime));
        }

        self.current_regime
    }

    /// Get regime-dependent jump detection parameters
    pub fn get_regime_jump_parameters(&self, regime: RegimeState) -> (FixedPoint, FixedPoint) {
        match regime {
            RegimeState::Normal => (FixedPoint::from_float(3.0), FixedPoint::from_float(0.05)),
            RegimeState::HighVolatility => (FixedPoint::from_float(2.5), FixedPoint::from_float(0.1)),
            RegimeState::Crisis => (FixedPoint::from_float(2.0), FixedPoint::from_float(0.2)),
            RegimeState::Recovery => (FixedPoint::from_float(2.8), FixedPoint::from_float(0.08)),
        }
    }
}

/// Jump risk premium calculator
pub struct JumpRiskPremiumCalculator {
    pub risk_aversion: FixedPoint,
    pub inventory_sensitivity: FixedPoint,
    pub clustering_penalty: FixedPoint,
    pub regime_adjustments: HashMap<RegimeState, FixedPoint>,
}

impl JumpRiskPremiumCalculator {
    pub fn new(risk_aversion: FixedPoint, inventory_sensitivity: FixedPoint) -> Self {
        let mut regime_adjustments = HashMap::new();
        regime_adjustments.insert(RegimeState::Normal, FixedPoint::one());
        regime_adjustments.insert(RegimeState::HighVolatility, FixedPoint::from_float(1.5));
        regime_adjustments.insert(RegimeState::Crisis, FixedPoint::from_float(2.5));
        regime_adjustments.insert(RegimeState::Recovery, FixedPoint::from_float(1.2));

        Self {
            risk_aversion,
            inventory_sensitivity,
            clustering_penalty: FixedPoint::from_float(0.1),
            regime_adjustments,
        }
    }

    /// Implement expected jump size calculation: E[J] = p/η⁺ - (1-p)/η⁻
    pub fn calculate_expected_jump_size(&self, params: &JumpParameters) -> FixedPoint {
        params.expected_jump_size()
    }

    /// Calculate basic jump risk premium
    pub fn calculate_basic_jump_risk_premium(
        &self,
        params: &JumpParameters,
        time_horizon: FixedPoint,
    ) -> FixedPoint {
        let expected_jump_size = self.calculate_expected_jump_size(params);
        let jump_variance = params.jump_size_variance();
        
        // Risk premium = λ * E[|J|] * T + (γ/2) * λ * Var[J] * T
        let expected_absolute_jump = self.calculate_expected_absolute_jump_size(params);
        let basic_premium = params.lambda * expected_absolute_jump * time_horizon;
        let variance_premium = self.risk_aversion / FixedPoint::from_float(2.0) * 
                              params.lambda * jump_variance * time_horizon;
        
        basic_premium + variance_premium
    }

    /// Create asymmetric jump risk adjustment based on inventory position
    pub fn calculate_asymmetric_jump_adjustment(
        &self,
        params: &JumpParameters,
        inventory: i64,
        time_horizon: FixedPoint,
    ) -> FixedPoint {
        if inventory == 0 {
            return FixedPoint::zero();
        }

        let inventory_fp = FixedPoint::from_float(inventory as f64);
        let expected_jump_size = self.calculate_expected_jump_size(params);
        
        // Asymmetric adjustment based on inventory direction
        let directional_risk = if inventory > 0 {
            // Long position: more sensitive to negative jumps
            let negative_jump_expectation = (FixedPoint::one() - params.p) / params.eta_minus;
            -negative_jump_expectation * self.inventory_sensitivity
        } else {
            // Short position: more sensitive to positive jumps
            let positive_jump_expectation = params.p / params.eta_plus;
            positive_jump_expectation * self.inventory_sensitivity
        };

        // Scale by inventory size and time horizon
        inventory_fp.abs() * directional_risk * params.lambda * time_horizon
    }

    /// Add regime-dependent jump risk premiums
    pub fn calculate_regime_dependent_premium(
        &self,
        base_premium: FixedPoint,
        regime: RegimeState,
        regime_params: Option<&JumpParameters>,
    ) -> FixedPoint {
        let regime_multiplier = self.regime_adjustments
            .get(&regime)
            .unwrap_or(&FixedPoint::one());

        let mut adjusted_premium = base_premium * *regime_multiplier;

        // Additional regime-specific adjustments
        if let Some(regime_jump_params) = regime_params {
            let regime_intensity_ratio = regime_jump_params.lambda / 
                                       (regime_jump_params.lambda + FixedPoint::from_float(0.1)); // Avoid division by zero
            adjusted_premium = adjusted_premium * (FixedPoint::one() + regime_intensity_ratio);
        }

        adjusted_premium
    }

    /// Implement jump clustering adjustment for spread widening
    pub fn calculate_clustering_adjustment(
        &self,
        clustering_periods: &[(FixedPoint, FixedPoint)],
        current_time: FixedPoint,
        lookback_time: FixedPoint,
    ) -> FixedPoint {
        if clustering_periods.is_empty() {
            return FixedPoint::zero();
        }

        let cutoff_time = current_time - lookback_time;
        
        // Count recent clustering periods
        let recent_clusters = clustering_periods.iter()
            .filter(|(start, end)| *end >= cutoff_time)
            .count();

        if recent_clusters == 0 {
            return FixedPoint::zero();
        }

        // Calculate clustering intensity
        let clustering_intensity = FixedPoint::from_float(recent_clusters as f64) / lookback_time;
        
        // Clustering penalty increases with intensity
        self.clustering_penalty * clustering_intensity * lookback_time
    }

    /// Calculate comprehensive jump risk premium
    pub fn calculate_comprehensive_jump_risk_premium(
        &self,
        params: &JumpParameters,
        inventory: i64,
        time_horizon: FixedPoint,
        regime: RegimeState,
        clustering_periods: &[(FixedPoint, FixedPoint)],
        current_time: FixedPoint,
    ) -> Result<JumpRiskPremiumComponents, CarteaJaimungalError> {
        // Basic jump risk premium
        let basic_premium = self.calculate_basic_jump_risk_premium(params, time_horizon);

        // Asymmetric adjustment for inventory
        let asymmetric_adjustment = self.calculate_asymmetric_jump_adjustment(
            params, 
            inventory, 
            time_horizon
        );

        // Regime-dependent adjustment
        let regime_params = params.get_regime_parameters(regime);
        let regime_premium = self.calculate_regime_dependent_premium(
            basic_premium,
            regime,
            Some(regime_params),
        );

        // Clustering adjustment
        let clustering_adjustment = self.calculate_clustering_adjustment(
            clustering_periods,
            current_time,
            time_horizon,
        );

        // Total premium
        let total_premium = regime_premium + asymmetric_adjustment + clustering_adjustment;

        Ok(JumpRiskPremiumComponents {
            basic_premium,
            asymmetric_adjustment,
            regime_adjustment: regime_premium - basic_premium,
            clustering_adjustment,
            total_premium,
            regime,
            inventory,
            time_horizon,
        })
    }

    /// Calculate expected absolute jump size for risk premium calculation
    fn calculate_expected_absolute_jump_size(&self, params: &JumpParameters) -> FixedPoint {
        // E[|J|] = p * E[J⁺] + (1-p) * E[J⁻] where J⁺,J⁻ are positive/negative jumps
        let positive_expectation = params.p / params.eta_plus;
        let negative_expectation = (FixedPoint::one() - params.p) / params.eta_minus;
        
        positive_expectation + negative_expectation
    }

    /// Update risk aversion based on market conditions
    pub fn update_risk_aversion(&mut self, new_risk_aversion: FixedPoint, regime: RegimeState) {
        self.risk_aversion = new_risk_aversion;
        
        // Adjust regime multipliers based on new risk aversion
        let base_multiplier = match regime {
            RegimeState::Crisis => FixedPoint::from_float(1.2),
            RegimeState::HighVolatility => FixedPoint::from_float(1.1),
            _ => FixedPoint::one(),
        };
        
        let risk_adjustment = (new_risk_aversion / FixedPoint::from_float(0.1)).min(FixedPoint::from_float(3.0));
        
        for (regime_state, multiplier) in self.regime_adjustments.iter_mut() {
            *multiplier = base_multiplier * risk_adjustment;
        }
    }
}

/// Components of jump risk premium calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpRiskPremiumComponents {
    pub basic_premium: FixedPoint,
    pub asymmetric_adjustment: FixedPoint,
    pub regime_adjustment: FixedPoint,
    pub clustering_adjustment: FixedPoint,
    pub total_premium: FixedPoint,
    pub regime: RegimeState,
    pub inventory: i64,
    pub time_horizon: FixedPoint,
}

impl JumpRiskPremiumComponents {
    /// Get premium as basis points
    pub fn total_premium_bps(&self) -> FixedPoint {
        self.total_premium * FixedPoint::from_float(10000.0)
    }

    /// Get breakdown of premium components as percentages
    pub fn get_component_breakdown(&self) -> HashMap<String, FixedPoint> {
        let mut breakdown = HashMap::new();
        
        if self.total_premium != FixedPoint::zero() {
            let total = self.total_premium;
            breakdown.insert("basic".to_string(), self.basic_premium / total * FixedPoint::from_float(100.0));
            breakdown.insert("asymmetric".to_string(), self.asymmetric_adjustment / total * FixedPoint::from_float(100.0));
            breakdown.insert("regime".to_string(), self.regime_adjustment / total * FixedPoint::from_float(100.0));
            breakdown.insert("clustering".to_string(), self.clustering_adjustment / total * FixedPoint::from_float(100.0));
        }
        
        breakdown
    }
}

/// Cartea-Jaimungal Jump-Diffusion Market Making Engine
pub struct CarteaJaimungalEngine {
    // Jump-diffusion parameters
    pub jump_intensity_estimator: JumpParameterEstimator,
    pub jump_size_estimator: JumpParameterEstimator,
    pub diffusion_estimator: JumpParameterEstimator,
    
    // Jump detection
    pub jump_detector: AdvancedJumpDetector,
    pub regime_detector: RegimeDetector,
    pub clustering_detector: JumpClusteringDetector,
    
    // Model components
    pub jump_diffusion_solver: crate::math::jump_diffusion::JumpDiffusionSimulator,
    pub hawkes_jump_intensity: TimeVaryingJumpIntensity,
    pub risk_premium_calculator: JumpRiskPremiumCalculator,
    
    // Current state
    pub current_jump_parameters: JumpParameters,
    pub current_regime: RegimeState,
    pub jump_history: VecDeque<JumpEvent>,
}

impl CarteaJaimungalEngine {
    pub fn new(
        risk_aversion: FixedPoint,
        inventory_sensitivity: FixedPoint,
        max_history: usize,
    ) -> Result<Self, CarteaJaimungalError> {
        let jump_intensity_estimator = JumpParameterEstimator::new(
            FixedPoint::from_float(1e-6),
            100,
        );
        
        let jump_detector = AdvancedJumpDetector::new(
            FixedPoint::from_float(0.05),
            20,
            max_history,
        );
        
        let regime_detector = RegimeDetector::new(max_history);
        
        let clustering_detector = JumpClusteringDetector::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(2.0),
        );
        
        let jump_diffusion_solver = crate::math::jump_diffusion::JumpDiffusionSimulator::new(
            42, // seed
            max_history,
        );
        
        let hawkes_jump_intensity = TimeVaryingJumpIntensity::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.2),
            max_history,
        );
        
        let risk_premium_calculator = JumpRiskPremiumCalculator::new(
            risk_aversion,
            inventory_sensitivity,
        );
        
        let current_jump_parameters = JumpParameters::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.5),
        )?;

        Ok(Self {
            jump_intensity_estimator,
            jump_size_estimator: jump_intensity_estimator.clone(),
            diffusion_estimator: jump_intensity_estimator.clone(),
            jump_detector,
            regime_detector,
            clustering_detector,
            jump_diffusion_solver,
            hawkes_jump_intensity,
            risk_premium_calculator,
            current_jump_parameters,
            current_regime: RegimeState::Normal,
            jump_history: VecDeque::with_capacity(max_history),
        })
    }

    /// Integrate jump-diffusion solver with market making engine
    pub fn calculate_jump_adjusted_quotes(
        &mut self,
        base_quotes: &OptimalQuotes,
        mid_price: FixedPoint,
        inventory: i64,
        time_horizon: FixedPoint,
        market_volatility: FixedPoint,
    ) -> Result<JumpAdjustedQuotes, CarteaJaimungalError> {
        // Update regime based on current market conditions
        let jump_intensity = self.hawkes_jump_intensity.calculate_intensity(
            FixedPoint::from_float(0.0), // current time (simplified)
            market_volatility,
            FixedPoint::zero(), // previous return (simplified)
        );
        
        self.current_regime = self.regime_detector.detect_regime(
            FixedPoint::from_float(0.0),
            market_volatility,
            jump_intensity,
        );

        // Get regime-specific jump parameters
        let regime_params = self.current_jump_parameters.get_regime_parameters(self.current_regime);

        // Calculate clustering periods
        let clustering_periods = self.clustering_detector.detect_clustering_periods(
            &self.jump_history.iter().cloned().collect::<Vec<_>>(),
            FixedPoint::from_float(2.0),
        );

        // Calculate comprehensive jump risk premium
        let premium_components = self.risk_premium_calculator.calculate_comprehensive_jump_risk_premium(
            regime_params,
            inventory,
            time_horizon,
            self.current_regime,
            &clustering_periods,
            FixedPoint::from_float(0.0), // current time (simplified)
        )?;

        Ok(JumpAdjustedQuotes {
            base_quotes: base_quotes.clone(),
            jump_risk_premium: premium_components.basic_premium,
            asymmetric_adjustment: premium_components.asymmetric_adjustment,
            regime_adjustment: premium_components.regime_adjustment,
            clustering_adjustment: premium_components.clustering_adjustment,
        })
    }

    /// Create jump-adjusted reservation price calculation
    pub fn calculate_jump_adjusted_reservation_price(
        &self,
        base_reservation_price: FixedPoint,
        inventory: i64,
        jump_parameters: &JumpParameters,
        time_horizon: FixedPoint,
    ) -> FixedPoint {
        let inventory_fp = FixedPoint::from_float(inventory as f64);
        let expected_jump_size = jump_parameters.expected_jump_size();
        
        // Jump adjustment to reservation price: r_adj = r_base - q * λ * E[J] * T
        let jump_adjustment = inventory_fp * jump_parameters.lambda * expected_jump_size * time_horizon;
        
        base_reservation_price - jump_adjustment
    }

    /// Add jump risk to optimal spread formula
    pub fn calculate_jump_adjusted_optimal_spread(
        &self,
        base_spread: FixedPoint,
        jump_risk_premium: FixedPoint,
        regime_multiplier: FixedPoint,
    ) -> FixedPoint {
        // Adjusted spread: δ_adj = δ_base + JRP * regime_multiplier
        base_spread + jump_risk_premium * regime_multiplier
    }

    /// Detect jumps from price series and update parameters
    pub fn detect_jumps(&mut self, price_series: &[FixedPoint]) -> Result<Vec<JumpEvent>, CarteaJaimungalError> {
        if price_series.len() < 2 {
            return Ok(Vec::new());
        }

        // Calculate returns
        let returns: Vec<FixedPoint> = price_series.windows(2)
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();

        // Detect jumps using threshold-based method
        let jump_events = self.jump_detector.detect_jumps_threshold_based(
            &returns,
            price_series,
            self.current_regime,
        )?;

        // Update jump history
        for event in &jump_events {
            if self.jump_history.len() >= self.jump_history.capacity() {
                self.jump_history.pop_front();
            }
            self.jump_history.push_back(event.clone());
        }

        Ok(jump_events)
    }

    /// Estimate jump parameters from detected jumps
    pub fn estimate_jump_parameters(&mut self, jump_events: &[JumpEvent]) -> Result<JumpParameters, CarteaJaimungalError> {
        if jump_events.is_empty() {
            return Ok(self.current_jump_parameters.clone());
        }

        let jump_sizes: Vec<FixedPoint> = jump_events.iter()
            .map(|event| event.size)
            .collect();

        let estimated_params = self.jump_intensity_estimator.estimate_double_exponential_parameters(&jump_sizes)?;
        
        // Validate parameters before updating
        self.jump_intensity_estimator.validate_parameters(&estimated_params)?;
        
        self.current_jump_parameters = estimated_params.clone();
        
        Ok(estimated_params)
    }

    /// Implement numerical validation and stability testing
    pub fn validate_numerical_stability(
        &self,
        jump_parameters: &JumpParameters,
        market_conditions: &MarketConditions,
    ) -> Result<ValidationResults, CarteaJaimungalError> {
        let mut validation_results = ValidationResults::new();

        // Check parameter bounds
        if jump_parameters.lambda.to_float() > 10.0 {
            validation_results.add_warning("Jump intensity very high - may cause numerical instability".to_string());
        }

        if jump_parameters.eta_plus.to_float() < 1.0 || jump_parameters.eta_minus.to_float() < 1.0 {
            validation_results.add_warning("Jump decay parameters very low - may cause large jumps".to_string());
        }

        // Check expected jump size
        let expected_jump = jump_parameters.expected_jump_size();
        if expected_jump.abs().to_float() > 0.2 {
            validation_results.add_error("Expected jump size too large for stability".to_string());
        }

        // Check market conditions compatibility
        if market_conditions.volatility.to_float() > 0.5 && jump_parameters.lambda.to_float() > 1.0 {
            validation_results.add_warning("High volatility and high jump intensity may cause instability".to_string());
        }

        // Stability condition: ensure finite variance
        let jump_variance = jump_parameters.jump_size_variance();
        if jump_variance.to_float() > 1.0 {
            validation_results.add_error("Jump variance too large - may cause numerical overflow".to_string());
        }

        Ok(validation_results)
    }

    /// Update all model components with new market data
    pub fn update_with_market_data(
        &mut self,
        price_series: &[FixedPoint],
        volatility: FixedPoint,
        current_time: FixedPoint,
    ) -> Result<(), CarteaJaimungalError> {
        // Detect new jumps
        let new_jumps = self.detect_jumps(price_series)?;
        
        // Update jump parameters if we have enough data
        if !new_jumps.is_empty() {
            self.estimate_jump_parameters(&new_jumps)?;
        }

        // Update regime
        let jump_intensity = self.hawkes_jump_intensity.calculate_intensity(
            current_time,
            volatility,
            FixedPoint::zero(), // simplified
        );
        
        self.current_regime = self.regime_detector.detect_regime(
            current_time,
            volatility,
            jump_intensity,
        );

        // Update adaptive thresholds
        self.jump_detector.update_adaptive_threshold(volatility, self.current_regime);

        Ok(())
    }
}

/// Jump-adjusted quotes structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpAdjustedQuotes {
    pub base_quotes: OptimalQuotes,
    pub jump_risk_premium: FixedPoint,
    pub asymmetric_adjustment: FixedPoint,
    pub regime_adjustment: FixedPoint,
    pub clustering_adjustment: FixedPoint,
}

impl JumpAdjustedQuotes {
    /// Get final bid price with all jump adjustments
    pub fn get_adjusted_bid_price(&self) -> FixedPoint {
        let total_adjustment = self.jump_risk_premium + self.asymmetric_adjustment + 
                              self.regime_adjustment + self.clustering_adjustment;
        self.base_quotes.bid_price - total_adjustment / FixedPoint::from_float(2.0)
    }

    /// Get final ask price with all jump adjustments
    pub fn get_adjusted_ask_price(&self) -> FixedPoint {
        let total_adjustment = self.jump_risk_premium + self.asymmetric_adjustment + 
                              self.regime_adjustment + self.clustering_adjustment;
        self.base_quotes.ask_price + total_adjustment / FixedPoint::from_float(2.0)
    }

    /// Get total spread adjustment
    pub fn get_total_spread_adjustment(&self) -> FixedPoint {
        self.jump_risk_premium + self.asymmetric_adjustment + 
        self.regime_adjustment + self.clustering_adjustment
    }
}

/// Market conditions for validation
#[derive(Debug, Clone)]
pub struct MarketConditions {
    pub volatility: FixedPoint,
    pub liquidity: FixedPoint,
    pub bid_ask_spread: FixedPoint,
    pub order_flow_imbalance: FixedPoint,
}

/// Validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub is_valid: bool,
}

impl ValidationResults {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            is_valid: true,
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Optimal quotes structure (simplified for this implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalQuotes {
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_size: FixedPoint,
    pub ask_size: FixedPoint,
    pub reservation_price: FixedPoint,
    pub optimal_spread: FixedPoint,
    pub timestamp: u64,
}

impl Default for OptimalQuotes {
    fn default() -> Self {
        Self {
            bid_price: FixedPoint::from_float(99.5),
            ask_price: FixedPoint::from_float(100.5),
            bid_size: FixedPoint::from_float(100.0),
            ask_size: FixedPoint::from_float(100.0),
            reservation_price: FixedPoint::from_float(100.0),
            optimal_spread: FixedPoint::from_float(1.0),
            timestamp: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jump_parameters_creation() {
        let params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        assert_eq!(params.lambda.to_float(), 0.5);
        assert_eq!(params.p.to_float(), 0.4);

        let expected_jump_size = 0.4 / 10.0 - 0.6 / 15.0;
        assert!((params.expected_jump_size().to_float() - expected_jump_size).abs() < 1e-10);
    }

    #[test]
    fn test_invalid_jump_parameters() {
        // Test negative intensity
        assert!(JumpParameters::new(
            FixedPoint::from_float(-0.1),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).is_err());

        // Test invalid probability
        assert!(JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(1.5),
        ).is_err());
    }

    #[test]
    fn test_advanced_jump_detector_creation() {
        let detector = AdvancedJumpDetector::new(
            FixedPoint::from_float(0.05),
            20,
            100,
        );

        assert_eq!(detector.significance_level.to_float(), 0.05);
        assert_eq!(detector.max_history, 100);
        assert!(detector.regime_dependent_thresholds.contains_key(&RegimeState::Normal));
    }

    #[test]
    fn test_regime_detector() {
        let mut detector = RegimeDetector::new(50);
        
        // Test normal regime detection
        let regime = detector.detect_regime(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.15),
            FixedPoint::from_float(0.05),
        );
        
        assert_eq!(regime, RegimeState::Normal);

        // Test crisis regime detection
        let regime = detector.detect_regime(
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.6),
            FixedPoint::from_float(0.9),
        );
        
        assert_eq!(regime, RegimeState::Crisis);
    }

    #[test]
    fn test_bipower_variation_jump_detection() {
        let detector = AdvancedJumpDetector::new(
            FixedPoint::from_float(0.05),
            10,
            50,
        );

        // Create synthetic returns with a jump
        let mut returns = Vec::new();
        for i in 0..50 {
            if i == 25 {
                returns.push(FixedPoint::from_float(0.1)); // Large jump
            } else {
                returns.push(FixedPoint::from_float(0.001 * (i as f64).sin()));
            }
        }

        let jump_indices = detector.detect_jumps_bipower_variation(&returns, RegimeState::Normal).unwrap();
        
        // Should detect the jump around index 25
        assert!(!jump_indices.is_empty());
    }

    #[test]
    fn test_threshold_based_jump_detection() {
        let mut detector = AdvancedJumpDetector::new(
            FixedPoint::from_float(0.05),
            10,
            50,
        );

        // Create synthetic data
        let mut returns = Vec::new();
        let mut prices = vec![FixedPoint::from_float(100.0)];
        
        for i in 0..50 {
            let return_val = if i == 25 {
                FixedPoint::from_float(0.05) // 5% jump
            } else {
                FixedPoint::from_float(0.001 * (i as f64).sin())
            };
            
            returns.push(return_val);
            let new_price = prices.last().unwrap() * (FixedPoint::one() + return_val);
            prices.push(new_price);
        }

        let jump_events = detector.detect_jumps_threshold_based(
            &returns, 
            &prices, 
            RegimeState::Normal
        ).unwrap();

        assert!(!jump_events.is_empty());
        assert!(jump_events.iter().any(|event| (event.time.to_float() - 25.0).abs() < 1.0));
    }

    #[test]
    fn test_jump_parameter_estimation() {
        let estimator = JumpParameterEstimator::new(
            FixedPoint::from_float(1e-6),
            100,
        );

        // Create synthetic jump sizes
        let jump_sizes = vec![
            FixedPoint::from_float(0.05),   // Positive jump
            FixedPoint::from_float(-0.03),  // Negative jump
            FixedPoint::from_float(0.02),   // Positive jump
            FixedPoint::from_float(-0.04),  // Negative jump
            FixedPoint::from_float(0.01),   // Positive jump
        ];

        let params = estimator.estimate_double_exponential_parameters(&jump_sizes).unwrap();

        // Should have reasonable parameters
        assert!(params.lambda.to_float() > 0.0);
        assert!(params.eta_plus.to_float() > 0.0);
        assert!(params.eta_minus.to_float() > 0.0);
        assert!(params.p.to_float() >= 0.0 && params.p.to_float() <= 1.0);

        // Should be 3/5 = 0.6 probability of positive jumps
        assert!((params.p.to_float() - 0.6).abs() < 0.1);
    }

    #[test]
    fn test_parameter_validation() {
        let estimator = JumpParameterEstimator::new(
            FixedPoint::from_float(1e-6),
            100,
        );

        // Valid parameters
        let valid_params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        assert!(estimator.validate_parameters(&valid_params).is_ok());

        // Invalid parameters (negative intensity)
        let mut invalid_params = valid_params.clone();
        invalid_params.lambda = FixedPoint::from_float(-0.1);
        assert!(estimator.validate_parameters(&invalid_params).is_err());
    }

    #[test]
    fn test_time_varying_intensity() {
        let mut intensity_model = TimeVaryingJumpIntensity::new(
            FixedPoint::from_float(0.1),  // baseline
            FixedPoint::from_float(0.5),  // volatility coefficient
            FixedPoint::from_float(0.3),  // momentum coefficient
            FixedPoint::from_float(0.2),  // mean reversion
            50,
        );

        let intensity = intensity_model.calculate_intensity(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.2),  // volatility
            FixedPoint::from_float(0.01), // previous return
        );

        assert!(intensity.to_float() > 0.0);
        assert_eq!(intensity_model.intensity_history.len(), 1);
    }

    #[test]
    fn test_exponential_parameter_estimation() {
        let estimator = JumpParameterEstimator::new(
            FixedPoint::from_float(1e-6),
            100,
        );

        // Exponential samples with known parameter
        let true_lambda = 5.0;
        let samples = vec![
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.05),
            FixedPoint::from_float(0.4),
            FixedPoint::from_float(0.2),
        ];

        let estimated_lambda = estimator.estimate_exponential_parameter(&samples).unwrap();
        
        // Should be close to 1/mean
        let sample_mean = samples.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                         FixedPoint::from_float(samples.len() as f64);
        let expected_lambda = FixedPoint::one() / sample_mean;
        
        assert!((estimated_lambda.to_float() - expected_lambda.to_float()).abs() < 1e-10);
    }

    #[test]
    fn test_jump_risk_premium_calculator() {
        let calculator = JumpRiskPremiumCalculator::new(
            FixedPoint::from_float(0.1), // risk aversion
            FixedPoint::from_float(0.5), // inventory sensitivity
        );

        let params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        let basic_premium = calculator.calculate_basic_jump_risk_premium(
            &params,
            FixedPoint::from_float(0.1), // time horizon
        );

        assert!(basic_premium.to_float() > 0.0);

        // Test asymmetric adjustment
        let asymmetric_adjustment = calculator.calculate_asymmetric_jump_adjustment(
            &params,
            100, // long inventory
            FixedPoint::from_float(0.1),
        );

        // Should be negative for long inventory (more sensitive to negative jumps)
        assert!(asymmetric_adjustment.to_float() < 0.0);
    }

    #[test]
    fn test_regime_dependent_premium() {
        let calculator = JumpRiskPremiumCalculator::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
        );

        let base_premium = FixedPoint::from_float(0.01);

        let normal_premium = calculator.calculate_regime_dependent_premium(
            base_premium,
            RegimeState::Normal,
            None,
        );

        let crisis_premium = calculator.calculate_regime_dependent_premium(
            base_premium,
            RegimeState::Crisis,
            None,
        );

        // Crisis premium should be higher than normal
        assert!(crisis_premium.to_float() > normal_premium.to_float());
    }

    #[test]
    fn test_clustering_adjustment() {
        let calculator = JumpRiskPremiumCalculator::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
        );

        let clustering_periods = vec![
            (FixedPoint::from_float(1.0), FixedPoint::from_float(1.5)),
            (FixedPoint::from_float(2.0), FixedPoint::from_float(2.3)),
        ];

        let adjustment = calculator.calculate_clustering_adjustment(
            &clustering_periods,
            FixedPoint::from_float(3.0), // current time
            FixedPoint::from_float(2.5), // lookback time
        );

        assert!(adjustment.to_float() > 0.0);
    }

    #[test]
    fn test_comprehensive_jump_risk_premium() {
        let calculator = JumpRiskPremiumCalculator::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
        );

        let params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        let clustering_periods = vec![
            (FixedPoint::from_float(1.0), FixedPoint::from_float(1.5)),
        ];

        let components = calculator.calculate_comprehensive_jump_risk_premium(
            &params,
            50, // inventory
            FixedPoint::from_float(0.1), // time horizon
            RegimeState::HighVolatility,
            &clustering_periods,
            FixedPoint::from_float(2.0), // current time
        ).unwrap();

        assert!(components.total_premium.to_float() > 0.0);
        assert!(components.basic_premium.to_float() > 0.0);
        assert_eq!(components.regime, RegimeState::HighVolatility);
        assert_eq!(components.inventory, 50);

        // Test breakdown
        let breakdown = components.get_component_breakdown();
        assert!(breakdown.contains_key("basic"));
        assert!(breakdown.contains_key("asymmetric"));
    }

    #[test]
    fn test_expected_absolute_jump_size() {
        let calculator = JumpRiskPremiumCalculator::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
        );

        let params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        let expected_abs_jump = calculator.calculate_expected_absolute_jump_size(&params);
        
        // E[|J|] = p/η⁺ + (1-p)/η⁻ = 0.4/10 + 0.6/15 = 0.04 + 0.04 = 0.08
        let expected_value = 0.4 / 10.0 + 0.6 / 15.0;
        assert!((expected_abs_jump.to_float() - expected_value).abs() < 1e-10);
    }

    #[test]
    fn test_cartea_jaimungal_engine_creation() {
        let engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1), // risk aversion
            FixedPoint::from_float(0.5), // inventory sensitivity
            100, // max history
        ).unwrap();

        assert_eq!(engine.current_regime, RegimeState::Normal);
        assert!(engine.current_jump_parameters.lambda.to_float() > 0.0);
    }

    #[test]
    fn test_jump_adjusted_quotes_calculation() {
        let mut engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            100,
        ).unwrap();

        let base_quotes = OptimalQuotes::default();
        
        let jump_adjusted_quotes = engine.calculate_jump_adjusted_quotes(
            &base_quotes,
            FixedPoint::from_float(100.0), // mid price
            50, // inventory
            FixedPoint::from_float(0.1), // time horizon
            FixedPoint::from_float(0.2), // market volatility
        ).unwrap();

        assert!(jump_adjusted_quotes.jump_risk_premium.to_float() >= 0.0);
        
        // Test adjusted prices
        let adjusted_bid = jump_adjusted_quotes.get_adjusted_bid_price();
        let adjusted_ask = jump_adjusted_quotes.get_adjusted_ask_price();
        
        assert!(adjusted_bid.to_float() <= base_quotes.bid_price.to_float());
        assert!(adjusted_ask.to_float() >= base_quotes.ask_price.to_float());
    }

    #[test]
    fn test_jump_adjusted_reservation_price() {
        let engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            100,
        ).unwrap();

        let base_reservation = FixedPoint::from_float(100.0);
        let jump_params = JumpParameters::new(
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.4),
        ).unwrap();

        let adjusted_reservation = engine.calculate_jump_adjusted_reservation_price(
            base_reservation,
            100, // long inventory
            &jump_params,
            FixedPoint::from_float(0.1),
        );

        // Should be different from base reservation due to jump adjustment
        assert!(adjusted_reservation != base_reservation);
    }

    #[test]
    fn test_jump_detection_integration() {
        let mut engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            100,
        ).unwrap();

        // Create price series with a jump
        let mut prices = Vec::new();
        for i in 0..50 {
            if i == 25 {
                prices.push(FixedPoint::from_float(105.0)); // Jump up
            } else if i < 25 {
                prices.push(FixedPoint::from_float(100.0 + 0.1 * (i as f64).sin()));
            } else {
                prices.push(FixedPoint::from_float(105.0 + 0.1 * (i as f64).sin()));
            }
        }

        let detected_jumps = engine.detect_jumps(&prices).unwrap();
        
        // Should detect at least one jump
        assert!(!detected_jumps.is_empty());
        
        // Update parameters based on detected jumps
        let updated_params = engine.estimate_jump_parameters(&detected_jumps).unwrap();
        assert!(updated_params.lambda.to_float() > 0.0);
    }

    #[test]
    fn test_numerical_stability_validation() {
        let engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            100,
        ).unwrap();

        let market_conditions = MarketConditions {
            volatility: FixedPoint::from_float(0.2),
            liquidity: FixedPoint::from_float(1000.0),
            bid_ask_spread: FixedPoint::from_float(0.01),
            order_flow_imbalance: FixedPoint::zero(),
        };

        // Test with stable parameters
        let stable_params = JumpParameters::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(10.0),
            FixedPoint::from_float(15.0),
            FixedPoint::from_float(0.5),
        ).unwrap();

        let validation = engine.validate_numerical_stability(&stable_params, &market_conditions).unwrap();
        assert!(validation.is_valid);

        // Test with unstable parameters
        let unstable_params = JumpParameters::new(
            FixedPoint::from_float(15.0), // Very high intensity
            FixedPoint::from_float(0.5),  // Very low decay
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(0.5),
        ).unwrap();

        let validation = engine.validate_numerical_stability(&unstable_params, &market_conditions).unwrap();
        assert!(!validation.is_valid);
        assert!(!validation.errors.is_empty());
    }

    #[test]
    fn test_market_data_update() {
        let mut engine = CarteaJaimungalEngine::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.5),
            100,
        ).unwrap();

        let prices = vec![
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(100.1),
            FixedPoint::from_float(100.05),
            FixedPoint::from_float(102.0), // Jump
            FixedPoint::from_float(101.9),
        ];

        let result = engine.update_with_market_data(
            &prices,
            FixedPoint::from_float(0.3), // High volatility
            FixedPoint::from_float(1.0),
        );

        assert!(result.is_ok());
        
        // Should have detected regime change due to high volatility
        assert_ne!(engine.current_regime, RegimeState::Normal);
    }

    #[test]
    fn test_jump_adjusted_quotes_components() {
        let base_quotes = OptimalQuotes::default();
        
        let jump_adjusted = JumpAdjustedQuotes {
            base_quotes: base_quotes.clone(),
            jump_risk_premium: FixedPoint::from_float(0.01),
            asymmetric_adjustment: FixedPoint::from_float(0.005),
            regime_adjustment: FixedPoint::from_float(0.002),
            clustering_adjustment: FixedPoint::from_float(0.003),
        };

        let total_adjustment = jump_adjusted.get_total_spread_adjustment();
        assert_eq!(total_adjustment.to_float(), 0.02);

        let adjusted_bid = jump_adjusted.get_adjusted_bid_price();
        let adjusted_ask = jump_adjusted.get_adjusted_ask_price();

        // Spread should be wider due to jump risk
        assert!(adjusted_ask.to_float() - adjusted_bid.to_float() > 
                base_quotes.ask_price.to_float() - base_quotes.bid_price.to_float());
    }
}