//! Jump-Diffusion Process Implementation
//!
//! This module implements jump-diffusion processes for financial modeling,
//! including geometric Brownian motion with jumps, jump detection algorithms,
//! and various jump size distributions.

use crate::math::fixed_point::{FixedPoint, DeterministicRng, BoxMullerGenerator, PoissonJumpGenerator};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum JumpDiffusionError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Jump detection error: {0}")]
    JumpDetectionError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

/// Jump size distribution types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JumpSizeDistribution {
    /// Normal distribution N(μ, σ²)
    Normal { mean: FixedPoint, std_dev: FixedPoint },
    /// Double exponential (Laplace) distribution
    DoubleExponential { eta_plus: FixedPoint, eta_minus: FixedPoint, p: FixedPoint },
    /// Kou double exponential distribution
    Kou { eta_plus: FixedPoint, eta_minus: FixedPoint, p: FixedPoint },
}

/// Geometric Brownian Motion with jumps state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMJumpState {
    pub price: FixedPoint,
    pub log_price: FixedPoint,
    pub volatility: FixedPoint,
    pub time: FixedPoint,
    pub jump_count: u32,
    pub total_jump_size: FixedPoint,
}

impl GBMJumpState {
    pub fn new(initial_price: FixedPoint, volatility: FixedPoint) -> Self {
        Self {
            price: initial_price,
            log_price: initial_price.ln(),
            volatility,
            time: FixedPoint::zero(),
            jump_count: 0,
            total_jump_size: FixedPoint::zero(),
        }
    }
}

/// GBM with jumps parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMJumpParams {
    pub drift: FixedPoint,                      // μ - drift coefficient
    pub volatility: FixedPoint,                 // σ - volatility coefficient
    pub jump_intensity: FixedPoint,             // λ - jump arrival rate
    pub jump_distribution: JumpSizeDistribution, // Jump size distribution
    pub risk_free_rate: FixedPoint,             // r - risk-free rate
}

impl GBMJumpParams {
    pub fn validate(&self) -> Result<(), JumpDiffusionError> {
        if self.volatility.to_float() <= 0.0 {
            return Err(JumpDiffusionError::InvalidParameters(
                "Volatility must be positive".to_string()
            ));
        }
        
        if self.jump_intensity.to_float() < 0.0 {
            return Err(JumpDiffusionError::InvalidParameters(
                "Jump intensity must be non-negative".to_string()
            ));
        }
        
        match self.jump_distribution {
            JumpSizeDistribution::Normal { std_dev, .. } => {
                if std_dev.to_float() <= 0.0 {
                    return Err(JumpDiffusionError::InvalidParameters(
                        "Jump standard deviation must be positive".to_string()
                    ));
                }
            }
            JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } |
            JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                if eta_plus.to_float() <= 0.0 || eta_minus.to_float() <= 0.0 {
                    return Err(JumpDiffusionError::InvalidParameters(
                        "Jump decay parameters must be positive".to_string()
                    ));
                }
                if p.to_float() < 0.0 || p.to_float() > 1.0 {
                    return Err(JumpDiffusionError::InvalidParameters(
                        "Jump probability must be in [0,1]".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// Calculate expected jump size
    pub fn expected_jump_size(&self) -> FixedPoint {
        match self.jump_distribution {
            JumpSizeDistribution::Normal { mean, .. } => mean,
            JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } |
            JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                p / eta_plus - (FixedPoint::one() - p) / eta_minus
            }
        }
    }

    /// Calculate jump size variance
    pub fn jump_size_variance(&self) -> FixedPoint {
        match self.jump_distribution {
            JumpSizeDistribution::Normal { std_dev, .. } => std_dev * std_dev,
            JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } |
            JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                let mean = self.expected_jump_size();
                p / (eta_plus * eta_plus) + (FixedPoint::one() - p) / (eta_minus * eta_minus) - mean * mean
            }
        }
    }
}

/// Jump event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JumpEvent {
    pub time: FixedPoint,
    pub size: FixedPoint,
    pub price_before: FixedPoint,
    pub price_after: FixedPoint,
}

/// Jump-diffusion process simulator
pub struct JumpDiffusionSimulator {
    pub normal_generator: BoxMullerGenerator,
    pub poisson_generator: PoissonJumpGenerator,
    pub jump_history: VecDeque<JumpEvent>,
    pub max_history: usize,
}

impl JumpDiffusionSimulator {
    pub fn new(seed: u64, max_history: usize) -> Self {
        Self {
            normal_generator: BoxMullerGenerator::new(seed),
            poisson_generator: PoissonJumpGenerator::new(seed + 1),
            jump_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Generate jump size according to specified distribution
    pub fn generate_jump_size(
        &mut self,
        distribution: &JumpSizeDistribution,
    ) -> FixedPoint {
        match distribution {
            JumpSizeDistribution::Normal { mean, std_dev } => {
                self.normal_generator.next_normal(*mean, *std_dev)
            }
            JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } => {
                let u = FixedPoint::from_float(
                    self.normal_generator.rng.next_fixed().to_float()
                );
                
                if u <= *p {
                    // Positive jump: exponential with rate eta_plus
                    let exp_sample = self.normal_generator.rng.next_exponential_fixed(*eta_plus);
                    exp_sample
                } else {
                    // Negative jump: exponential with rate eta_minus
                    let exp_sample = self.normal_generator.rng.next_exponential_fixed(*eta_minus);
                    -exp_sample
                }
            }
            JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                // Kou model: similar to double exponential but with different interpretation
                self.generate_jump_size(&JumpSizeDistribution::DoubleExponential {
                    eta_plus: *eta_plus,
                    eta_minus: *eta_minus,
                    p: *p,
                })
            }
        }
    }

    /// Simulate one time step of jump-diffusion process
    pub fn simulate_step(
        &mut self,
        dt: FixedPoint,
        state: &GBMJumpState,
        params: &GBMJumpParams,
    ) -> Result<GBMJumpState, JumpDiffusionError> {
        params.validate()?;

        // Generate Brownian increment
        let sqrt_dt = dt.sqrt();
        let dw = self.normal_generator.next_standard_normal() * sqrt_dt;

        // Generate jump component
        let jump_count = self.poisson_generator.next_poisson_count(params.jump_intensity, dt);
        let mut total_jump_size = FixedPoint::zero();
        let mut new_jump_count = state.jump_count;

        for _ in 0..jump_count {
            let jump_size = self.generate_jump_size(&params.jump_distribution);
            total_jump_size = total_jump_size + jump_size;
            new_jump_count += 1;

            // Record jump event
            let jump_event = JumpEvent {
                time: state.time + dt,
                size: jump_size,
                price_before: state.price,
                price_after: state.price * (jump_size + FixedPoint::one()).exp(),
            };

            if self.jump_history.len() >= self.max_history {
                self.jump_history.pop_front();
            }
            self.jump_history.push_back(jump_event);
        }

        // Update log-price using SDE: dlog(S) = (μ - σ²/2)dt + σdW + ΣJ
        let drift_term = (params.drift - params.volatility * params.volatility / FixedPoint::from_float(2.0)) * dt;
        let diffusion_term = params.volatility * dw;
        
        let new_log_price = state.log_price + drift_term + diffusion_term + total_jump_size;
        let new_price = new_log_price.exp();

        Ok(GBMJumpState {
            price: new_price,
            log_price: new_log_price,
            volatility: state.volatility,
            time: state.time + dt,
            jump_count: new_jump_count,
            total_jump_size: state.total_jump_size + total_jump_size,
        })
    }

    /// Simulate entire path
    pub fn simulate_path(
        &mut self,
        initial_state: GBMJumpState,
        params: &GBMJumpParams,
        time_horizon: FixedPoint,
        n_steps: usize,
    ) -> Result<Vec<GBMJumpState>, JumpDiffusionError> {
        let dt = time_horizon / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial_state;

        path.push(current_state.clone());

        for _ in 0..n_steps {
            current_state = self.simulate_step(dt, &current_state, params)?;
            path.push(current_state.clone());
        }

        Ok(path)
    }

    /// Get recent jump events
    pub fn get_recent_jumps(&self, lookback_time: FixedPoint) -> Vec<&JumpEvent> {
        if self.jump_history.is_empty() {
            return Vec::new();
        }

        let current_time = self.jump_history.back().unwrap().time;
        let cutoff_time = current_time - lookback_time;

        self.jump_history
            .iter()
            .filter(|jump| jump.time >= cutoff_time)
            .collect()
    }
}

/// Bi-power variation jump detection
pub struct BiPowerVariationJumpDetector {
    pub significance_level: FixedPoint,
    pub window_size: usize,
}

impl BiPowerVariationJumpDetector {
    pub fn new(significance_level: FixedPoint, window_size: usize) -> Self {
        Self {
            significance_level,
            window_size,
        }
    }

    /// Detect jumps using bi-power variation test
    pub fn detect_jumps(&self, returns: &[FixedPoint]) -> Result<Vec<usize>, JumpDiffusionError> {
        if returns.len() < self.window_size {
            return Err(JumpDiffusionError::JumpDetectionError(
                "Insufficient data for jump detection".to_string()
            ));
        }

        let mut jump_indices = Vec::new();

        for i in self.window_size..returns.len() {
            let window_start = i.saturating_sub(self.window_size);
            let window_returns = &returns[window_start..i];

            // Calculate realized variance (quadratic variation)
            let rv = window_returns.iter()
                .map(|r| *r * *r)
                .fold(FixedPoint::zero(), |acc, x| acc + x);

            // Calculate bi-power variation
            let mut bv = FixedPoint::zero();
            for j in 1..window_returns.len() {
                bv = bv + window_returns[j-1].abs() * window_returns[j].abs();
            }
            bv = bv * FixedPoint::from_float(std::f64::consts::PI / 2.0);

            // Jump test statistic
            let jump_component = rv - bv;
            
            // Theoretical variance of the test statistic (simplified)
            let test_variance = bv * FixedPoint::from_float(2.0);
            
            if test_variance > FixedPoint::zero() {
                let test_statistic = jump_component / test_variance.sqrt();
                
                // Critical value for normal distribution (approximate)
                let critical_value = FixedPoint::from_float(1.96); // 5% significance level
                
                if test_statistic.abs() > critical_value {
                    jump_indices.push(i);
                }
            }
        }

        Ok(jump_indices)
    }

    /// Estimate jump intensity from detected jumps
    pub fn estimate_jump_intensity(
        &self,
        jump_indices: &[usize],
        total_time: FixedPoint,
        n_observations: usize,
    ) -> FixedPoint {
        let jump_count = FixedPoint::from_float(jump_indices.len() as f64);
        jump_count / total_time
    }

    /// Estimate jump sizes from detected jumps
    pub fn estimate_jump_sizes(
        &self,
        returns: &[FixedPoint],
        jump_indices: &[usize],
    ) -> Vec<FixedPoint> {
        jump_indices.iter()
            .filter_map(|&i| {
                if i < returns.len() {
                    Some(returns[i])
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Jump clustering detector using Hawkes process framework
pub struct JumpClusteringDetector {
    pub baseline_intensity: FixedPoint,
    pub self_excitation: FixedPoint,
    pub decay_rate: FixedPoint,
}

impl JumpClusteringDetector {
    pub fn new(
        baseline_intensity: FixedPoint,
        self_excitation: FixedPoint,
        decay_rate: FixedPoint,
    ) -> Self {
        Self {
            baseline_intensity,
            self_excitation,
            decay_rate,
        }
    }

    /// Calculate current jump intensity given jump history
    pub fn calculate_intensity(
        &self,
        current_time: FixedPoint,
        jump_times: &[FixedPoint],
    ) -> FixedPoint {
        let mut intensity = self.baseline_intensity;

        for &jump_time in jump_times {
            if jump_time < current_time {
                let time_diff = current_time - jump_time;
                let decay = (-self.decay_rate * time_diff).exp();
                intensity = intensity + self.self_excitation * decay;
            }
        }

        intensity
    }

    /// Detect clustering periods
    pub fn detect_clustering_periods(
        &self,
        jump_events: &[JumpEvent],
        threshold_multiplier: FixedPoint,
    ) -> Vec<(FixedPoint, FixedPoint)> {
        let mut clustering_periods = Vec::new();
        let threshold = self.baseline_intensity * threshold_multiplier;

        let mut in_cluster = false;
        let mut cluster_start = FixedPoint::zero();

        for (i, event) in jump_events.iter().enumerate() {
            let jump_times: Vec<FixedPoint> = jump_events[..i]
                .iter()
                .map(|e| e.time)
                .collect();

            let current_intensity = self.calculate_intensity(event.time, &jump_times);

            if !in_cluster && current_intensity > threshold {
                in_cluster = true;
                cluster_start = event.time;
            } else if in_cluster && current_intensity <= threshold {
                in_cluster = false;
                clustering_periods.push((cluster_start, event.time));
            }
        }

        // Close any open cluster
        if in_cluster && !jump_events.is_empty() {
            clustering_periods.push((cluster_start, jump_events.last().unwrap().time));
        }

        clustering_periods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gbm_jump_params_validation() {
        let valid_params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.1),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };

        assert!(valid_params.validate().is_ok());

        let invalid_params = GBMJumpParams {
            volatility: FixedPoint::from_float(-0.1), // Invalid negative volatility
            ..valid_params
        };

        assert!(invalid_params.validate().is_err());
    }

    #[test]
    fn test_jump_diffusion_simulation() {
        let mut simulator = JumpDiffusionSimulator::new(42, 100);
        
        let initial_state = GBMJumpState::new(
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(0.2),
        );

        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.5),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.1),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };

        let path = simulator.simulate_path(
            initial_state,
            &params,
            FixedPoint::one(),
            100,
        ).unwrap();

        assert_eq!(path.len(), 101);
        assert!(path.last().unwrap().price.to_float() > 0.0);
    }

    #[test]
    fn test_double_exponential_jumps() {
        let mut simulator = JumpDiffusionSimulator::new(123, 50);
        
        let distribution = JumpSizeDistribution::DoubleExponential {
            eta_plus: FixedPoint::from_float(10.0),
            eta_minus: FixedPoint::from_float(15.0),
            p: FixedPoint::from_float(0.4),
        };

        let mut positive_jumps = 0;
        let mut negative_jumps = 0;

        for _ in 0..1000 {
            let jump_size = simulator.generate_jump_size(&distribution);
            if jump_size.to_float() > 0.0 {
                positive_jumps += 1;
            } else {
                negative_jumps += 1;
            }
        }

        // Should be roughly 40% positive, 60% negative
        let positive_ratio = positive_jumps as f64 / 1000.0;
        assert!((positive_ratio - 0.4).abs() < 0.1);
    }

    #[test]
    fn test_bipower_variation_jump_detection() {
        let detector = BiPowerVariationJumpDetector::new(
            FixedPoint::from_float(0.05),
            20,
        );

        // Create synthetic returns with a jump
        let mut returns = Vec::new();
        for i in 0..100 {
            if i == 50 {
                // Insert a large jump
                returns.push(FixedPoint::from_float(0.1));
            } else {
                // Normal small returns
                returns.push(FixedPoint::from_float(0.001 * (i as f64).sin()));
            }
        }

        let jump_indices = detector.detect_jumps(&returns).unwrap();
        
        // Should detect the jump around index 50
        assert!(!jump_indices.is_empty());
        assert!(jump_indices.iter().any(|&i| (i as i32 - 50).abs() < 5));
    }

    #[test]
    fn test_jump_clustering_detection() {
        let detector = JumpClusteringDetector::new(
            FixedPoint::from_float(0.1),  // baseline
            FixedPoint::from_float(0.5),  // self-excitation
            FixedPoint::from_float(2.0),  // decay
        );

        // Create jump events with clustering
        let jump_events = vec![
            JumpEvent {
                time: FixedPoint::from_float(1.0),
                size: FixedPoint::from_float(0.05),
                price_before: FixedPoint::from_float(100.0),
                price_after: FixedPoint::from_float(105.0),
            },
            JumpEvent {
                time: FixedPoint::from_float(1.1),
                size: FixedPoint::from_float(0.03),
                price_before: FixedPoint::from_float(105.0),
                price_after: FixedPoint::from_float(108.0),
            },
            JumpEvent {
                time: FixedPoint::from_float(1.2),
                size: FixedPoint::from_float(0.04),
                price_before: FixedPoint::from_float(108.0),
                price_after: FixedPoint::from_float(112.0),
            },
        ];

        let clustering_periods = detector.detect_clustering_periods(
            &jump_events,
            FixedPoint::from_float(2.0),
        );

        // Should detect clustering period
        assert!(!clustering_periods.is_empty());
    }
}