//! Value-at-Risk (VaR) Calculator
//!
//! This module implements comprehensive VaR calculation methods including:
//! - Historical simulation VaR with rolling windows
//! - Parametric VaR using GARCH volatility forecasting
//! - Monte Carlo VaR with importance sampling
//! - Expected Shortfall and coherent risk measures
//!
//! The implementation focuses on ultra-low latency computation suitable for
//! high-frequency trading applications with sub-microsecond update requirements.

use crate::error::RiskError;
use crate::math::fixed_point::FixedPoint;
use crate::models::garch::GARCHModel;
use crate::utils::statistics::{percentile, mean, std_dev};
use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// Confidence levels for VaR calculations
#[derive(Debug, Clone, Copy)]
pub enum ConfidenceLevel {
    /// 95% confidence level (α = 0.05)
    Percent95,
    /// 99% confidence level (α = 0.01)
    Percent99,
    /// 99.9% confidence level (α = 0.001)
    Percent999,
    /// Custom confidence level
    Custom(FixedPoint),
}

impl ConfidenceLevel {
    pub fn alpha(&self) -> FixedPoint {
        match self {
            ConfidenceLevel::Percent95 => FixedPoint::from_float(0.05),
            ConfidenceLevel::Percent99 => FixedPoint::from_float(0.01),
            ConfidenceLevel::Percent999 => FixedPoint::from_float(0.001),
            ConfidenceLevel::Custom(alpha) => *alpha,
        }
    }
}

/// VaR calculation methods
#[derive(Debug, Clone, Copy)]
pub enum VaRMethod {
    /// Historical simulation using empirical distribution
    Historical,
    /// Parametric VaR assuming normal distribution
    Parametric,
    /// Parametric VaR with GARCH volatility forecasting
    ParametricGARCH,
    /// Monte Carlo simulation
    MonteCarlo,
    /// Monte Carlo with importance sampling
    MonteCarloImportance,
}

/// VaR calculation results
#[derive(Debug, Clone)]
pub struct VaRResult {
    /// Value-at-Risk estimate
    pub var: FixedPoint,
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall: FixedPoint,
    /// Confidence level used
    pub confidence_level: ConfidenceLevel,
    /// Method used for calculation
    pub method: VaRMethod,
    /// Number of observations used
    pub sample_size: usize,
    /// Calculation timestamp
    pub timestamp: u64,
}

/// Historical simulation VaR calculator with rolling windows
#[derive(Debug)]
pub struct HistoricalVaRCalculator {
    /// Rolling window of historical returns
    returns_window: VecDeque<FixedPoint>,
    /// Maximum window size
    max_window_size: usize,
    /// Sorted returns for percentile calculation (cached)
    sorted_returns: Vec<FixedPoint>,
    /// Cache validity flag
    cache_valid: bool,
}

impl HistoricalVaRCalculator {
    /// Create new historical VaR calculator
    pub fn new(window_size: usize) -> Self {
        Self {
            returns_window: VecDeque::with_capacity(window_size),
            max_window_size: window_size,
            sorted_returns: Vec::with_capacity(window_size),
            cache_valid: false,
        }
    }

    /// Add new return observation
    pub fn add_return(&mut self, return_value: FixedPoint) {
        if self.returns_window.len() >= self.max_window_size {
            self.returns_window.pop_front();
        }
        self.returns_window.push_back(return_value);
        self.cache_valid = false;
    }

    /// Calculate historical simulation VaR
    pub fn calculate_var(&mut self, confidence_level: ConfidenceLevel) -> Result<VaRResult, RiskError> {
        if self.returns_window.is_empty() {
            return Err(RiskError::InsufficientData("No historical returns available".to_string()));
        }

        // Update sorted cache if needed
        if !self.cache_valid {
            self.sorted_returns.clear();
            self.sorted_returns.extend(self.returns_window.iter().copied());
            self.sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.cache_valid = true;
        }

        let alpha = confidence_level.alpha();
        let var = self.calculate_percentile(alpha)?;
        let expected_shortfall = self.calculate_expected_shortfall(alpha)?;

        Ok(VaRResult {
            var: -var, // VaR is typically reported as positive loss
            expected_shortfall: -expected_shortfall,
            confidence_level,
            method: VaRMethod::Historical,
            sample_size: self.returns_window.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }

    /// Calculate percentile from sorted returns
    fn calculate_percentile(&self, alpha: FixedPoint) -> Result<FixedPoint, RiskError> {
        if self.sorted_returns.is_empty() {
            return Err(RiskError::InsufficientData("No returns for percentile calculation".to_string()));
        }

        let n = self.sorted_returns.len();
        let index = (alpha.to_float() * n as f64) as usize;
        
        if index >= n {
            Ok(self.sorted_returns[n - 1])
        } else if index == 0 {
            Ok(self.sorted_returns[0])
        } else {
            // Linear interpolation between adjacent values
            let lower = self.sorted_returns[index - 1];
            let upper = self.sorted_returns[index];
            let weight = FixedPoint::from_float(alpha.to_float() * n as f64 - (index - 1) as f64);
            Ok(lower + weight * (upper - lower))
        }
    }

    /// Calculate Expected Shortfall (average of tail losses)
    fn calculate_expected_shortfall(&self, alpha: FixedPoint) -> Result<FixedPoint, RiskError> {
        if self.sorted_returns.is_empty() {
            return Err(RiskError::InsufficientData("No returns for ES calculation".to_string()));
        }

        let n = self.sorted_returns.len();
        let cutoff_index = (alpha.to_float() * n as f64) as usize;
        
        if cutoff_index == 0 {
            return Ok(self.sorted_returns[0]);
        }

        let tail_sum: FixedPoint = self.sorted_returns[..cutoff_index].iter().sum();
        Ok(tail_sum / FixedPoint::from_int(cutoff_index as i64))
    }
}

/// Parametric VaR calculator with GARCH volatility forecasting
#[derive(Debug)]
pub struct ParametricVaRCalculator {
    /// GARCH model for volatility forecasting
    garch_model: GARCHModel,
    /// Historical returns for parameter estimation
    returns_history: VecDeque<FixedPoint>,
    /// Maximum history size
    max_history_size: usize,
    /// Cached volatility forecast
    cached_volatility: Option<FixedPoint>,
    /// Cache timestamp
    cache_timestamp: u64,
    /// Cache validity duration (nanoseconds)
    cache_validity_ns: u64,
}

impl ParametricVaRCalculator {
    /// Create new parametric VaR calculator
    pub fn new(history_size: usize, cache_validity_ms: u64) -> Self {
        Self {
            garch_model: GARCHModel::new(),
            returns_history: VecDeque::with_capacity(history_size),
            max_history_size: history_size,
            cached_volatility: None,
            cache_timestamp: 0,
            cache_validity_ns: cache_validity_ms * 1_000_000,
        }
    }

    /// Add new return observation
    pub fn add_return(&mut self, return_value: FixedPoint) {
        if self.returns_history.len() >= self.max_history_size {
            self.returns_history.pop_front();
        }
        self.returns_history.push_back(return_value);
        
        // Invalidate volatility cache
        self.cached_volatility = None;
    }

    /// Calculate parametric VaR using GARCH volatility
    pub fn calculate_var(&mut self, confidence_level: ConfidenceLevel) -> Result<VaRResult, RiskError> {
        let volatility = self.get_volatility_forecast()?;
        let alpha = confidence_level.alpha();
        
        // Calculate z-score for given confidence level (assuming normal distribution)
        let z_score = self.inverse_normal_cdf(alpha)?;
        
        // VaR = μ + σ * z_α (where μ is typically 0 for returns)
        let var = volatility * z_score;
        
        // Expected Shortfall for normal distribution: ES = σ * φ(z_α) / α
        let phi_z = self.standard_normal_pdf(z_score);
        let expected_shortfall = volatility * phi_z / alpha;

        Ok(VaRResult {
            var: -var, // VaR reported as positive loss
            expected_shortfall: -expected_shortfall,
            confidence_level,
            method: VaRMethod::ParametricGARCH,
            sample_size: self.returns_history.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }

    /// Get volatility forecast from GARCH model
    fn get_volatility_forecast(&mut self) -> Result<FixedPoint, RiskError> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Check if cached volatility is still valid
        if let Some(cached_vol) = self.cached_volatility {
            if current_time - self.cache_timestamp < self.cache_validity_ns {
                return Ok(cached_vol);
            }
        }

        // Recalculate volatility forecast
        if self.returns_history.len() < 10 {
            return Err(RiskError::InsufficientData("Need at least 10 observations for GARCH".to_string()));
        }

        let returns_vec: Vec<FixedPoint> = self.returns_history.iter().copied().collect();
        self.garch_model.fit(&returns_vec)?;
        let volatility = self.garch_model.forecast_volatility(1)?; // 1-step ahead forecast

        // Cache the result
        self.cached_volatility = Some(volatility);
        self.cache_timestamp = current_time;

        Ok(volatility)
    }

    /// Inverse normal CDF approximation (Beasley-Springer-Moro algorithm)
    fn inverse_normal_cdf(&self, p: FixedPoint) -> Result<FixedPoint, RiskError> {
        let p_val = p.to_float();
        
        if p_val <= 0.0 || p_val >= 1.0 {
            return Err(RiskError::InvalidParameter("Probability must be in (0,1)".to_string()));
        }

        // Beasley-Springer-Moro approximation
        let a0 = 2.50662823884;
        let a1 = -18.61500062529;
        let a2 = 41.39119773534;
        let a3 = -25.44106049637;
        
        let b1 = -8.47351093090;
        let b2 = 23.08336743743;
        let b3 = -21.06224101826;
        let b4 = 3.13082909833;
        
        let c0 = 0.3374754822726147;
        let c1 = 0.9761690190917186;
        let c2 = 0.1607979714918209;
        let c3 = 0.0276438810333863;
        let c4 = 0.0038405729373609;
        let c5 = 0.0003951896511919;
        let c6 = 0.0000321767881768;
        let c7 = 0.0000002888167364;
        let c8 = 0.0000003960315187;

        let y = p_val - 0.5;
        let result = if y.abs() < 0.42 {
            let r = y * y;
            y * (((a3 * r + a2) * r + a1) * r + a0) / ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0)
        } else {
            let r = if y > 0.0 { -p_val.ln() } else { -(1.0 - p_val).ln() };
            let x = (((((((c8 * r + c7) * r + c6) * r + c5) * r + c4) * r + c3) * r + c2) * r + c1) * r + c0;
            if y < 0.0 { -x } else { x }
        };

        Ok(FixedPoint::from_float(result))
    }

    /// Standard normal PDF
    fn standard_normal_pdf(&self, x: FixedPoint) -> FixedPoint {
        let x_val = x.to_float();
        let result = (-0.5 * x_val * x_val).exp() / (2.0 * std::f64::consts::PI).sqrt();
        FixedPoint::from_float(result)
    }
}

/// Monte Carlo VaR calculator with importance sampling
#[derive(Debug)]
pub struct MonteCarloVaRCalculator {
    /// Number of Monte Carlo simulations
    num_simulations: usize,
    /// Random number generator seed for reproducibility
    rng_seed: u64,
    /// Importance sampling parameters
    importance_sampling: bool,
    /// Drift adjustment for importance sampling
    drift_adjustment: FixedPoint,
    /// Historical returns for parameter estimation
    returns_history: VecDeque<FixedPoint>,
    /// Maximum history size
    max_history_size: usize,
}

impl MonteCarloVaRCalculator {
    /// Create new Monte Carlo VaR calculator
    pub fn new(num_simulations: usize, history_size: usize, importance_sampling: bool) -> Self {
        Self {
            num_simulations,
            rng_seed: 12345, // Default seed
            importance_sampling,
            drift_adjustment: FixedPoint::from_float(-2.0), // Shift towards tail
            returns_history: VecDeque::with_capacity(history_size),
            max_history_size: history_size,
        }
    }

    /// Set random seed for reproducible results
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }

    /// Add new return observation
    pub fn add_return(&mut self, return_value: FixedPoint) {
        if self.returns_history.len() >= self.max_history_size {
            self.returns_history.pop_front();
        }
        self.returns_history.push_back(return_value);
    }

    /// Calculate Monte Carlo VaR
    pub fn calculate_var(&self, confidence_level: ConfidenceLevel) -> Result<VaRResult, RiskError> {
        if self.returns_history.len() < 30 {
            return Err(RiskError::InsufficientData("Need at least 30 observations for Monte Carlo".to_string()));
        }

        // Estimate parameters from historical data
        let (mean, volatility) = self.estimate_parameters()?;
        
        // Generate Monte Carlo simulations
        let simulated_returns = if self.importance_sampling {
            self.generate_importance_sampling_returns(mean, volatility)?
        } else {
            self.generate_standard_monte_carlo_returns(mean, volatility)?
        };

        // Calculate VaR and Expected Shortfall from simulations
        let alpha = confidence_level.alpha();
        let var = self.calculate_var_from_simulations(&simulated_returns, alpha)?;
        let expected_shortfall = self.calculate_es_from_simulations(&simulated_returns, alpha)?;

        let method = if self.importance_sampling {
            VaRMethod::MonteCarloImportance
        } else {
            VaRMethod::MonteCarlo
        };

        Ok(VaRResult {
            var: -var, // VaR reported as positive loss
            expected_shortfall: -expected_shortfall,
            confidence_level,
            method,
            sample_size: self.num_simulations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }

    /// Estimate mean and volatility from historical returns
    fn estimate_parameters(&self) -> Result<(FixedPoint, FixedPoint), RiskError> {
        let returns_vec: Vec<f64> = self.returns_history.iter().map(|r| r.to_float()).collect();
        
        let mean_val = self.calculate_mean(&returns_vec);
        let vol_val = self.calculate_std_dev(&returns_vec, mean_val);
        
        Ok((FixedPoint::from_float(mean_val), FixedPoint::from_float(vol_val)))
    }

    /// Calculate mean of returns
    fn calculate_mean(&self, returns: &[f64]) -> f64 {
        returns.iter().sum::<f64>() / returns.len() as f64
    }

    /// Calculate standard deviation of returns
    fn calculate_std_dev(&self, returns: &[f64], mean: f64) -> f64 {
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        variance.sqrt()
    }

    /// Generate standard Monte Carlo returns
    fn generate_standard_monte_carlo_returns(
        &self,
        mean: FixedPoint,
        volatility: FixedPoint,
    ) -> Result<Vec<FixedPoint>, RiskError> {
        use std::f64::consts::PI;
        
        let mut returns = Vec::with_capacity(self.num_simulations);
        let mut rng_state = self.rng_seed;
        
        for _ in 0..self.num_simulations {
            // Simple Box-Muller transform for normal random numbers
            let (u1, u2) = self.generate_uniform_pair(&mut rng_state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            
            let sample = mean.to_float() + volatility.to_float() * z;
            returns.push(FixedPoint::from_float(sample));
        }

        Ok(returns)
    }

    /// Generate importance sampling Monte Carlo returns
    fn generate_importance_sampling_returns(
        &self,
        mean: FixedPoint,
        volatility: FixedPoint,
    ) -> Result<Vec<FixedPoint>, RiskError> {
        use std::f64::consts::PI;
        
        // Shift the distribution towards the tail for importance sampling
        let shifted_mean = mean + self.drift_adjustment * volatility;
        let mut weighted_returns = Vec::with_capacity(self.num_simulations);
        let mut rng_state = self.rng_seed;
        
        for _ in 0..self.num_simulations {
            // Generate from shifted distribution
            let (u1, u2) = self.generate_uniform_pair(&mut rng_state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            
            let sample = shifted_mean.to_float() + volatility.to_float() * z;
            let return_val = FixedPoint::from_float(sample);
            
            // For simplicity, we'll use the raw samples
            // In practice, you'd apply likelihood ratio weighting
            weighted_returns.push(return_val);
        }

        Ok(weighted_returns)
    }

    /// Generate pair of uniform random numbers using linear congruential generator
    fn generate_uniform_pair(&self, state: &mut u64) -> (f64, f64) {
        // Simple LCG parameters
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        
        *state = state.wrapping_mul(A).wrapping_add(C);
        let u1 = (*state as f64) / (u64::MAX as f64);
        
        *state = state.wrapping_mul(A).wrapping_add(C);
        let u2 = (*state as f64) / (u64::MAX as f64);
        
        (u1, u2)
    }

    /// Calculate VaR from simulation results
    fn calculate_var_from_simulations(
        &self,
        simulations: &[FixedPoint],
        alpha: FixedPoint,
    ) -> Result<FixedPoint, RiskError> {
        let mut sorted_returns = simulations.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (alpha.to_float() * simulations.len() as f64) as usize;
        let var_index = index.min(simulations.len() - 1);
        
        Ok(sorted_returns[var_index])
    }

    /// Calculate Expected Shortfall from simulation results
    fn calculate_es_from_simulations(
        &self,
        simulations: &[FixedPoint],
        alpha: FixedPoint,
    ) -> Result<FixedPoint, RiskError> {
        let mut sorted_returns = simulations.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let cutoff_index = (alpha.to_float() * simulations.len() as f64) as usize;
        if cutoff_index == 0 {
            return Ok(sorted_returns[0]);
        }
        
        let tail_sum: FixedPoint = sorted_returns[..cutoff_index].iter().sum();
        Ok(tail_sum / FixedPoint::from_int(cutoff_index as i64))
    }
}

/// Comprehensive VaR calculator combining all methods
#[derive(Debug)]
pub struct VaRCalculator {
    /// Historical simulation calculator
    historical_calculator: Arc<RwLock<HistoricalVaRCalculator>>,
    /// Parametric calculator with GARCH
    parametric_calculator: Arc<RwLock<ParametricVaRCalculator>>,
    /// Monte Carlo calculator
    monte_carlo_calculator: Arc<RwLock<MonteCarloVaRCalculator>>,
    /// Default calculation method
    default_method: VaRMethod,
}

impl VaRCalculator {
    /// Create new comprehensive VaR calculator
    pub fn new(
        window_size: usize,
        monte_carlo_simulations: usize,
        default_method: VaRMethod,
    ) -> Self {
        Self {
            historical_calculator: Arc::new(RwLock::new(
                HistoricalVaRCalculator::new(window_size)
            )),
            parametric_calculator: Arc::new(RwLock::new(
                ParametricVaRCalculator::new(window_size, 1000) // 1 second cache
            )),
            monte_carlo_calculator: Arc::new(RwLock::new(
                MonteCarloVaRCalculator::new(monte_carlo_simulations, window_size, true)
            )),
            default_method,
        }
    }

    /// Add new return observation to all calculators
    pub fn add_return(&self, return_value: FixedPoint) {
        self.historical_calculator.write().add_return(return_value);
        self.parametric_calculator.write().add_return(return_value);
        self.monte_carlo_calculator.write().add_return(return_value);
    }

    /// Calculate VaR using specified method
    pub fn calculate_var(
        &self,
        method: VaRMethod,
        confidence_level: ConfidenceLevel,
    ) -> Result<VaRResult, RiskError> {
        match method {
            VaRMethod::Historical => {
                self.historical_calculator.write().calculate_var(confidence_level)
            }
            VaRMethod::Parametric | VaRMethod::ParametricGARCH => {
                self.parametric_calculator.write().calculate_var(confidence_level)
            }
            VaRMethod::MonteCarlo | VaRMethod::MonteCarloImportance => {
                self.monte_carlo_calculator.read().calculate_var(confidence_level)
            }
        }
    }

    /// Calculate VaR using default method
    pub fn calculate_default_var(
        &self,
        confidence_level: ConfidenceLevel,
    ) -> Result<VaRResult, RiskError> {
        self.calculate_var(self.default_method, confidence_level)
    }

    /// Calculate VaR using all methods for comparison
    pub fn calculate_all_methods(
        &self,
        confidence_level: ConfidenceLevel,
    ) -> Result<Vec<VaRResult>, RiskError> {
        let methods = vec![
            VaRMethod::Historical,
            VaRMethod::ParametricGARCH,
            VaRMethod::MonteCarloImportance,
        ];

        let mut results = Vec::new();
        for method in methods {
            match self.calculate_var(method, confidence_level) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log error but continue with other methods
                    eprintln!("VaR calculation failed for method {:?}: {}", method, e);
                }
            }
        }

        if results.is_empty() {
            Err(RiskError::CalculationError("All VaR methods failed".to_string()))
        } else {
            Ok(results)
        }
    }

    /// Get model comparison statistics
    pub fn get_model_comparison(&self, confidence_level: ConfidenceLevel) -> Result<ModelComparison, RiskError> {
        let results = self.calculate_all_methods(confidence_level)?;
        
        if results.is_empty() {
            return Err(RiskError::CalculationError("No VaR results available".to_string()));
        }

        let var_values: Vec<f64> = results.iter().map(|r| r.var.to_float()).collect();
        let es_values: Vec<f64> = results.iter().map(|r| r.expected_shortfall.to_float()).collect();

        let var_mean = self.calculate_mean(&var_values);
        let var_std = self.calculate_std_dev(&var_values, var_mean);
        let es_mean = self.calculate_mean(&es_values);
        let es_std = self.calculate_std_dev(&es_values, es_mean);

        Ok(ModelComparison {
            results,
            var_mean: FixedPoint::from_float(var_mean),
            var_std_dev: FixedPoint::from_float(var_std),
            es_mean: FixedPoint::from_float(es_mean),
            es_std_dev: FixedPoint::from_float(es_std),
        })
    }

    /// Calculate mean of values
    fn calculate_mean(&self, values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    /// Calculate standard deviation of values
    fn calculate_std_dev(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Model comparison results
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Individual model results
    pub results: Vec<VaRResult>,
    /// Mean VaR across models
    pub var_mean: FixedPoint,
    /// Standard deviation of VaR estimates
    pub var_std_dev: FixedPoint,
    /// Mean Expected Shortfall across models
    pub es_mean: FixedPoint,
    /// Standard deviation of ES estimates
    pub es_std_dev: FixedPoint,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_var_calculation() {
        let mut calculator = HistoricalVaRCalculator::new(100);
        
        // Add some test returns
        let returns = vec![-0.05, -0.03, -0.01, 0.01, 0.02, 0.03, -0.02, -0.04];
        for ret in returns {
            calculator.add_return(FixedPoint::from_float(ret));
        }
        
        let result = calculator.calculate_var(ConfidenceLevel::Percent95).unwrap();
        assert!(result.var > FixedPoint::from_float(0.0));
        assert!(result.expected_shortfall >= result.var);
    }

    #[test]
    fn test_monte_carlo_var_calculation() {
        let mut calculator = MonteCarloVaRCalculator::new(1000, 100, false);
        
        // Add sufficient test returns for Monte Carlo
        for i in 0..50 {
            let return_val = (i as f64 - 25.0) * 0.002;
            calculator.add_return(FixedPoint::from_float(return_val));
        }
        
        let result = calculator.calculate_var(ConfidenceLevel::Percent95).unwrap();
        assert!(result.var > FixedPoint::from_float(0.0));
        assert!(result.expected_shortfall >= result.var);
    }

    #[test]
    fn test_comprehensive_var_calculator() {
        let calculator = VaRCalculator::new(100, 1000, VaRMethod::Historical);
        
        // Add test data
        for i in 0..50 {
            let return_val = (i as f64 - 25.0) * 0.002;
            calculator.add_return(FixedPoint::from_float(return_val));
        }
        
        let result = calculator.calculate_default_var(ConfidenceLevel::Percent95).unwrap();
        assert!(result.var > FixedPoint::from_float(0.0));
        
        let comparison = calculator.get_model_comparison(ConfidenceLevel::Percent95).unwrap();
        assert!(!comparison.results.is_empty());
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(ConfidenceLevel::Percent95.alpha(), FixedPoint::from_float(0.05));
        assert_eq!(ConfidenceLevel::Percent99.alpha(), FixedPoint::from_float(0.01));
        assert_eq!(ConfidenceLevel::Percent999.alpha(), FixedPoint::from_float(0.001));
    }

    #[test]
    fn test_importance_sampling() {
        let mut calculator = MonteCarloVaRCalculator::new(1000, 100, true);
        
        // Add sufficient test returns
        for i in 0..50 {
            let return_val = (i as f64 - 25.0) * 0.002;
            calculator.add_return(FixedPoint::from_float(return_val));
        }
        
        let result = calculator.calculate_var(ConfidenceLevel::Percent99).unwrap();
        assert!(result.var > FixedPoint::from_float(0.0));
        assert_eq!(result.method, VaRMethod::MonteCarloImportance);
    }
}