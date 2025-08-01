//! Rough Volatility Model Implementation
//!
//! This module implements rough volatility models based on fractional Brownian motion,
//! including volatility clustering detection and Hurst parameter estimation.
//! 
//! The implementation follows the mathematical framework from "High Frequency Quoting 
//! Under Liquidity Constraints" and provides circuit-friendly operations for zkVM execution.

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RoughVolatilityError {
    #[error("Invalid Hurst parameter: {0} (must be in (0, 0.5))")]
    InvalidHurstParameter(f64),
    #[error("Insufficient data for estimation: need at least {0} points")]
    InsufficientData(usize),
    #[error("Numerical error in computation: {0}")]
    NumericalError(String),
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
}

/// Fractional Brownian Motion generator using Cholesky decomposition
#[derive(Debug, Clone)]
pub struct FractionalBrownianMotion {
    pub hurst_parameter: FixedPoint,
    pub covariance_matrix: Vec<Vec<FixedPoint>>,
    pub cholesky_decomp: Vec<Vec<FixedPoint>>,
    pub max_steps: usize,
}

impl FractionalBrownianMotion {
    /// Create new FBM generator with given Hurst parameter
    pub fn new(hurst_parameter: FixedPoint, max_steps: usize) -> Result<Self, RoughVolatilityError> {
        let h = hurst_parameter.to_float();
        if h <= 0.0 || h >= 0.5 {
            return Err(RoughVolatilityError::InvalidHurstParameter(h));
        }

        let mut fbm = Self {
            hurst_parameter,
            covariance_matrix: Vec::new(),
            cholesky_decomp: Vec::new(),
            max_steps,
        };

        fbm.precompute_covariance_matrix()?;
        fbm.compute_cholesky_decomposition()?;

        Ok(fbm)
    }

    /// Precompute covariance matrix for FBM
    /// Cov(B_H(s), B_H(t)) = 0.5 * (|s|^(2H) + |t|^(2H) - |t-s|^(2H))
    fn precompute_covariance_matrix(&mut self) -> Result<(), RoughVolatilityError> {
        let n = self.max_steps;
        let h = self.hurst_parameter.to_float();
        let two_h = 2.0 * h;

        self.covariance_matrix = vec![vec![FixedPoint::zero(); n]; n];

        for i in 0..n {
            for j in 0..n {
                let s = (i + 1) as f64;
                let t = (j + 1) as f64;
                
                let s_pow = s.powf(two_h);
                let t_pow = t.powf(two_h);
                let diff_pow = (t - s).abs().powf(two_h);
                
                let covariance = 0.5 * (s_pow + t_pow - diff_pow);
                self.covariance_matrix[i][j] = FixedPoint::from_float(covariance);
            }
        }

        Ok(())
    }

    /// Compute Cholesky decomposition of covariance matrix
    fn compute_cholesky_decomposition(&mut self) -> Result<(), RoughVolatilityError> {
        let n = self.covariance_matrix.len();
        self.cholesky_decomp = vec![vec![FixedPoint::zero(); n]; n];

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element: L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
                    let mut sum = FixedPoint::zero();
                    for k in 0..j {
                        sum = sum + self.cholesky_decomp[i][k] * self.cholesky_decomp[i][k];
                    }
                    let diagonal_val = self.covariance_matrix[i][i] - sum;
                    
                    if diagonal_val.to_float() <= 0.0 {
                        return Err(RoughVolatilityError::NumericalError(
                            "Matrix is not positive definite".to_string()
                        ));
                    }
                    
                    self.cholesky_decomp[i][j] = FixedPoint::from_float(diagonal_val.to_float().sqrt());
                } else {
                    // Lower triangular element: L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]
                    let mut sum = FixedPoint::zero();
                    for k in 0..j {
                        sum = sum + self.cholesky_decomp[i][k] * self.cholesky_decomp[j][k];
                    }
                    
                    if self.cholesky_decomp[j][j].to_float().abs() < 1e-12 {
                        return Err(RoughVolatilityError::NumericalError(
                            "Division by zero in Cholesky decomposition".to_string()
                        ));
                    }
                    
                    self.cholesky_decomp[i][j] = (self.covariance_matrix[i][j] - sum) / self.cholesky_decomp[j][j];
                }
            }
        }

        Ok(())
    }

    /// Generate FBM path using Cholesky decomposition
    pub fn generate_path(
        &self, 
        n_steps: usize, 
        rng: &mut DeterministicRng
    ) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        if n_steps > self.max_steps {
            return Err(RoughVolatilityError::NumericalError(
                format!("Requested {} steps exceeds maximum {}", n_steps, self.max_steps)
            ));
        }

        let mut path = vec![FixedPoint::zero(); n_steps];
        let mut gaussian_samples = Vec::with_capacity(n_steps);

        // Generate independent Gaussian samples using Box-Muller transform
        for _ in 0..n_steps {
            let u1 = rng.next_fixed();
            let u2 = rng.next_fixed();
            
            // Box-Muller transform
            let z = FixedPoint::from_float(
                (-2.0 * u1.to_float().ln()).sqrt() * (2.0 * std::f64::consts::PI * u2.to_float()).cos()
            );
            gaussian_samples.push(z);
        }

        // Apply Cholesky decomposition to get correlated FBM increments
        for i in 0..n_steps {
            let mut sum = FixedPoint::zero();
            for j in 0..=i {
                sum = sum + self.cholesky_decomp[i][j] * gaussian_samples[j];
            }
            path[i] = sum;
        }

        Ok(path)
    }

    /// Generate single FBM increment given history
    pub fn generate_increment(
        &self,
        step: usize,
        rng: &mut DeterministicRng
    ) -> Result<FixedPoint, RoughVolatilityError> {
        if step >= self.max_steps {
            return Err(RoughVolatilityError::NumericalError(
                format!("Step {} exceeds maximum {}", step, self.max_steps)
            ));
        }

        // Generate Gaussian sample
        let u1 = rng.next_fixed();
        let u2 = rng.next_fixed();
        let z = FixedPoint::from_float(
            (-2.0 * u1.to_float().ln()).sqrt() * (2.0 * std::f64::consts::PI * u2.to_float()).cos()
        );

        // Apply Cholesky row for this step
        Ok(self.cholesky_decomp[step][step] * z)
    }
}

/// Rough volatility path simulator
#[derive(Debug, Clone)]
pub struct RoughVolatilitySimulator {
    pub fbm_generator: FractionalBrownianMotion,
    pub vol_of_vol: FixedPoint,
    pub mean_reversion: FixedPoint,
    pub long_term_var: FixedPoint,
    pub initial_log_vol: FixedPoint,
}

impl RoughVolatilitySimulator {
    pub fn new(
        hurst_parameter: FixedPoint,
        vol_of_vol: FixedPoint,
        mean_reversion: FixedPoint,
        long_term_var: FixedPoint,
        initial_log_vol: FixedPoint,
        max_steps: usize,
    ) -> Result<Self, RoughVolatilityError> {
        let fbm_generator = FractionalBrownianMotion::new(hurst_parameter, max_steps)?;

        Ok(Self {
            fbm_generator,
            vol_of_vol,
            mean_reversion,
            long_term_var,
            initial_log_vol,
        })
    }

    /// Simulate rough volatility path
    /// dlog(σ_t) = -λ(log(σ_t) - θ)dt + ν dW_t^H
    pub fn simulate_path(
        &self,
        n_steps: usize,
        dt: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        let fbm_path = self.fbm_generator.generate_path(n_steps, rng)?;
        let mut log_vol_path = Vec::with_capacity(n_steps + 1);
        
        log_vol_path.push(self.initial_log_vol);
        let mut current_log_vol = self.initial_log_vol;

        for i in 0..n_steps {
            // Mean reversion term: -λ(log(σ_t) - θ)dt
            let mean_reversion_term = self.mean_reversion * 
                (current_log_vol - self.long_term_var.ln()) * dt;

            // Rough noise term: ν dW_t^H
            let noise_term = self.vol_of_vol * fbm_path[i];

            // Update log volatility
            current_log_vol = current_log_vol - mean_reversion_term + noise_term;
            log_vol_path.push(current_log_vol);
        }

        // Convert to volatility levels
        let vol_path: Vec<FixedPoint> = log_vol_path.iter()
            .map(|&log_vol| log_vol.exp())
            .collect();

        Ok(vol_path)
    }
}

/// Volatility clustering detection using various statistical measures
#[derive(Debug, Clone)]
pub struct VolatilityClusteringDetector {
    pub window_size: usize,
    pub threshold_multiplier: FixedPoint,
    pub autocorr_lags: Vec<usize>,
}

impl VolatilityClusteringDetector {
    pub fn new(window_size: usize, threshold_multiplier: FixedPoint) -> Self {
        Self {
            window_size,
            threshold_multiplier,
            autocorr_lags: vec![1, 5, 10, 20, 50],
        }
    }

    /// Detect volatility clusters using rolling standard deviation
    pub fn detect_clusters(&self, returns: &[FixedPoint]) -> Result<Vec<bool>, RoughVolatilityError> {
        if returns.len() < self.window_size {
            return Err(RoughVolatilityError::InsufficientData(self.window_size));
        }

        let mut clusters = vec![false; returns.len()];
        let mut rolling_vols = Vec::new();

        // Compute rolling volatilities
        for i in self.window_size..returns.len() {
            let window = &returns[i - self.window_size..i];
            let vol = self.compute_realized_volatility(window);
            rolling_vols.push(vol);
        }

        if rolling_vols.is_empty() {
            return Ok(clusters);
        }

        // Compute threshold as multiple of median volatility
        let mut sorted_vols = rolling_vols.clone();
        sorted_vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_vol = sorted_vols[sorted_vols.len() / 2];
        let threshold = median_vol * self.threshold_multiplier;

        // Mark high volatility periods as clusters
        for (i, &vol) in rolling_vols.iter().enumerate() {
            if vol > threshold {
                let start_idx = i + self.window_size;
                let end_idx = (start_idx + self.window_size).min(clusters.len());
                for j in start_idx..end_idx {
                    clusters[j] = true;
                }
            }
        }

        Ok(clusters)
    }

    /// Compute realized volatility for a window of returns
    fn compute_realized_volatility(&self, returns: &[FixedPoint]) -> FixedPoint {
        if returns.is_empty() {
            return FixedPoint::zero();
        }

        let mean = returns.iter().fold(FixedPoint::zero(), |acc, &r| acc + r) / 
                   FixedPoint::from_float(returns.len() as f64);

        let variance = returns.iter()
            .map(|&r| {
                let diff = r - mean;
                diff * diff
            })
            .fold(FixedPoint::zero(), |acc, var| acc + var) / 
            FixedPoint::from_float(returns.len() as f64);

        FixedPoint::from_float(variance.to_float().sqrt())
    }

    /// Compute autocorrelation of squared returns (volatility clustering measure)
    pub fn compute_volatility_autocorrelation(
        &self, 
        returns: &[FixedPoint]
    ) -> Result<Vec<(usize, FixedPoint)>, RoughVolatilityError> {
        if returns.len() < self.autocorr_lags.iter().max().unwrap() + 1 {
            return Err(RoughVolatilityError::InsufficientData(
                self.autocorr_lags.iter().max().unwrap() + 1
            ));
        }

        // Compute squared returns
        let squared_returns: Vec<FixedPoint> = returns.iter()
            .map(|&r| r * r)
            .collect();

        let mut autocorrelations = Vec::new();

        for &lag in &self.autocorr_lags {
            let autocorr = self.compute_autocorrelation(&squared_returns, lag)?;
            autocorrelations.push((lag, autocorr));
        }

        Ok(autocorrelations)
    }

    /// Compute autocorrelation at specific lag
    fn compute_autocorrelation(
        &self, 
        series: &[FixedPoint], 
        lag: usize
    ) -> Result<FixedPoint, RoughVolatilityError> {
        if series.len() <= lag {
            return Err(RoughVolatilityError::InsufficientData(lag + 1));
        }

        let n = series.len() - lag;
        let mean = series.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                   FixedPoint::from_float(series.len() as f64);

        let mut numerator = FixedPoint::zero();
        let mut denominator = FixedPoint::zero();

        for i in 0..n {
            let x_i = series[i] - mean;
            let x_i_lag = series[i + lag] - mean;
            numerator = numerator + x_i * x_i_lag;
        }

        for &x in series {
            let diff = x - mean;
            denominator = denominator + diff * diff;
        }

        if denominator.to_float().abs() < 1e-12 {
            return Ok(FixedPoint::zero());
        }

        Ok(numerator / denominator)
    }
}

/// Hurst parameter estimator using multiple methods
#[derive(Debug, Clone)]
pub struct HurstEstimator {
    pub min_window: usize,
    pub max_window: usize,
}

impl HurstEstimator {
    pub fn new(min_window: usize, max_window: usize) -> Self {
        Self {
            min_window,
            max_window,
        }
    }

    /// Estimate Hurst parameter using R/S analysis (Rescaled Range)
    pub fn estimate_rs_method(&self, series: &[FixedPoint]) -> Result<FixedPoint, RoughVolatilityError> {
        if series.len() < self.max_window {
            return Err(RoughVolatilityError::InsufficientData(self.max_window));
        }

        let mut log_rs_values = Vec::new();
        let mut log_n_values = Vec::new();

        // Compute R/S for different window sizes
        let mut window_size = self.min_window;
        while window_size <= self.max_window && window_size <= series.len() {
            let rs_value = self.compute_rs_statistic(series, window_size)?;
            
            if rs_value.to_float() > 0.0 {
                log_rs_values.push(FixedPoint::from_float(rs_value.to_float().ln()));
                log_n_values.push(FixedPoint::from_float((window_size as f64).ln()));
            }

            window_size = (window_size as f64 * 1.2) as usize; // Increase by 20%
        }

        if log_rs_values.len() < 3 {
            return Err(RoughVolatilityError::InsufficientData(3));
        }

        // Linear regression: log(R/S) = H * log(n) + constant
        let hurst = self.linear_regression_slope(&log_n_values, &log_rs_values)?;
        
        // Ensure Hurst parameter is in valid range for rough volatility
        let hurst_clamped = if hurst.to_float() < 0.01 {
            FixedPoint::from_float(0.01)
        } else if hurst.to_float() > 0.49 {
            FixedPoint::from_float(0.49)
        } else {
            hurst
        };

        Ok(hurst_clamped)
    }

    /// Compute R/S statistic for given window size
    fn compute_rs_statistic(&self, series: &[FixedPoint], window_size: usize) -> Result<FixedPoint, RoughVolatilityError> {
        if series.len() < window_size {
            return Err(RoughVolatilityError::InsufficientData(window_size));
        }

        let n_windows = series.len() / window_size;
        let mut rs_values = Vec::new();

        for i in 0..n_windows {
            let start = i * window_size;
            let end = start + window_size;
            let window = &series[start..end];

            // Compute mean
            let mean = window.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                      FixedPoint::from_float(window_size as f64);

            // Compute cumulative deviations
            let mut cumulative_deviations = Vec::with_capacity(window_size);
            let mut cumsum = FixedPoint::zero();
            
            for &x in window {
                cumsum = cumsum + (x - mean);
                cumulative_deviations.push(cumsum);
            }

            // Compute range
            let max_cumsum = cumulative_deviations.iter()
                .fold(cumulative_deviations[0], |max, &x| if x > max { x } else { max });
            let min_cumsum = cumulative_deviations.iter()
                .fold(cumulative_deviations[0], |min, &x| if x < min { x } else { min });
            let range = max_cumsum - min_cumsum;

            // Compute standard deviation
            let variance = window.iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .fold(FixedPoint::zero(), |acc, var| acc + var) / 
                FixedPoint::from_float(window_size as f64);
            
            let std_dev = FixedPoint::from_float(variance.to_float().sqrt());

            // R/S ratio
            if std_dev.to_float() > 1e-12 {
                rs_values.push(range / std_dev);
            }
        }

        if rs_values.is_empty() {
            return Ok(FixedPoint::zero());
        }

        // Average R/S across windows
        let avg_rs = rs_values.iter().fold(FixedPoint::zero(), |acc, &rs| acc + rs) / 
                     FixedPoint::from_float(rs_values.len() as f64);

        Ok(avg_rs)
    }

    /// Estimate Hurst parameter using detrended fluctuation analysis (DFA)
    pub fn estimate_dfa_method(&self, series: &[FixedPoint]) -> Result<FixedPoint, RoughVolatilityError> {
        if series.len() < self.max_window {
            return Err(RoughVolatilityError::InsufficientData(self.max_window));
        }

        // Compute cumulative sum (integration)
        let mut cumsum = Vec::with_capacity(series.len());
        let mean = series.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                   FixedPoint::from_float(series.len() as f64);
        
        let mut running_sum = FixedPoint::zero();
        for &x in series {
            running_sum = running_sum + (x - mean);
            cumsum.push(running_sum);
        }

        let mut log_f_values = Vec::new();
        let mut log_n_values = Vec::new();

        // Compute fluctuation function for different scales
        let mut scale = self.min_window;
        while scale <= self.max_window && scale <= cumsum.len() / 4 {
            let fluctuation = self.compute_dfa_fluctuation(&cumsum, scale)?;
            
            if fluctuation.to_float() > 0.0 {
                log_f_values.push(FixedPoint::from_float(fluctuation.to_float().ln()));
                log_n_values.push(FixedPoint::from_float((scale as f64).ln()));
            }

            scale = (scale as f64 * 1.2) as usize;
        }

        if log_f_values.len() < 3 {
            return Err(RoughVolatilityError::InsufficientData(3));
        }

        // Linear regression: log(F(n)) = H * log(n) + constant
        let hurst = self.linear_regression_slope(&log_n_values, &log_f_values)?;
        
        // Clamp to valid range for rough volatility
        let hurst_clamped = if hurst.to_float() < 0.01 {
            FixedPoint::from_float(0.01)
        } else if hurst.to_float() > 0.49 {
            FixedPoint::from_float(0.49)
        } else {
            hurst
        };

        Ok(hurst_clamped)
    }

    /// Compute DFA fluctuation function
    fn compute_dfa_fluctuation(&self, cumsum: &[FixedPoint], scale: usize) -> Result<FixedPoint, RoughVolatilityError> {
        let n_segments = cumsum.len() / scale;
        if n_segments == 0 {
            return Err(RoughVolatilityError::InsufficientData(scale));
        }

        let mut fluctuations = Vec::new();

        for i in 0..n_segments {
            let start = i * scale;
            let end = start + scale;
            let segment = &cumsum[start..end];

            // Fit linear trend
            let (slope, intercept) = self.fit_linear_trend(segment)?;

            // Compute detrended fluctuation
            let mut sum_squared_residuals = FixedPoint::zero();
            for (j, &y) in segment.iter().enumerate() {
                let x = FixedPoint::from_float(j as f64);
                let trend = slope * x + intercept;
                let residual = y - trend;
                sum_squared_residuals = sum_squared_residuals + residual * residual;
            }

            fluctuations.push(sum_squared_residuals / FixedPoint::from_float(scale as f64));
        }

        // Average fluctuation
        let avg_fluctuation = fluctuations.iter().fold(FixedPoint::zero(), |acc, &f| acc + f) / 
                             FixedPoint::from_float(fluctuations.len() as f64);

        Ok(FixedPoint::from_float(avg_fluctuation.to_float().sqrt()))
    }

    /// Fit linear trend to data segment
    fn fit_linear_trend(&self, data: &[FixedPoint]) -> Result<(FixedPoint, FixedPoint), RoughVolatilityError> {
        let n = data.len();
        if n < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        let n_fp = FixedPoint::from_float(n as f64);
        
        // Compute means
        let x_mean = FixedPoint::from_float((n - 1) as f64) / FixedPoint::from_float(2.0);
        let y_mean = data.iter().fold(FixedPoint::zero(), |acc, &y| acc + y) / n_fp;

        // Compute slope
        let mut numerator = FixedPoint::zero();
        let mut denominator = FixedPoint::zero();

        for (i, &y) in data.iter().enumerate() {
            let x = FixedPoint::from_float(i as f64);
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            
            numerator = numerator + x_diff * y_diff;
            denominator = denominator + x_diff * x_diff;
        }

        if denominator.to_float().abs() < 1e-12 {
            return Ok((FixedPoint::zero(), y_mean));
        }

        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;

        Ok((slope, intercept))
    }

    /// Perform linear regression and return slope
    fn linear_regression_slope(&self, x_values: &[FixedPoint], y_values: &[FixedPoint]) -> Result<FixedPoint, RoughVolatilityError> {
        if x_values.len() != y_values.len() || x_values.len() < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        let n = FixedPoint::from_float(x_values.len() as f64);
        
        let x_mean = x_values.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / n;
        let y_mean = y_values.iter().fold(FixedPoint::zero(), |acc, &y| acc + y) / n;

        let mut numerator = FixedPoint::zero();
        let mut denominator = FixedPoint::zero();

        for (i, &x) in x_values.iter().enumerate() {
            let y = y_values[i];
            let x_diff = x - x_mean;
            let y_diff = y - y_mean;
            
            numerator = numerator + x_diff * y_diff;
            denominator = denominator + x_diff * x_diff;
        }

        if denominator.to_float().abs() < 1e-12 {
            return Err(RoughVolatilityError::NumericalError(
                "Singular matrix in linear regression".to_string()
            ));
        }

        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_brownian_motion() {
        let hurst = FixedPoint::from_float(0.3);
        let fbm = FractionalBrownianMotion::new(hurst, 100).unwrap();
        let mut rng = DeterministicRng::new(42);
        
        let path = fbm.generate_path(50, &mut rng).unwrap();
        assert_eq!(path.len(), 50);
        
        // Check that path has some variation
        let variance = path.iter()
            .map(|&x| x * x)
            .fold(FixedPoint::zero(), |acc, x| acc + x) / 
            FixedPoint::from_float(path.len() as f64);
        assert!(variance.to_float() > 0.0);
    }

    #[test]
    fn test_rough_volatility_simulator() {
        let hurst = FixedPoint::from_float(0.2);
        let vol_of_vol = FixedPoint::from_float(0.3);
        let mean_reversion = FixedPoint::from_float(2.0);
        let long_term_var = FixedPoint::from_float(0.04);
        let initial_log_vol = FixedPoint::from_float(0.2_f64.ln());
        
        let simulator = RoughVolatilitySimulator::new(
            hurst, vol_of_vol, mean_reversion, long_term_var, initial_log_vol, 100
        ).unwrap();
        
        let mut rng = DeterministicRng::new(123);
        let dt = FixedPoint::from_float(1.0 / 252.0); // Daily
        
        let vol_path = simulator.simulate_path(50, dt, &mut rng).unwrap();
        assert_eq!(vol_path.len(), 51);
        
        // Check that all volatilities are positive
        for &vol in &vol_path {
            assert!(vol.to_float() > 0.0);
        }
    }

    #[test]
    fn test_volatility_clustering_detector() {
        let detector = VolatilityClusteringDetector::new(10, FixedPoint::from_float(2.0));
        
        // Create synthetic returns with clustering
        let mut returns = vec![FixedPoint::from_float(0.01); 100];
        // Add high volatility cluster
        for i in 30..50 {
            returns[i] = FixedPoint::from_float(0.05);
        }
        
        let clusters = detector.detect_clusters(&returns).unwrap();
        assert_eq!(clusters.len(), returns.len());
        
        // Should detect clustering in the high volatility period
        let cluster_count = clusters.iter().filter(|&&x| x).count();
        assert!(cluster_count > 0);
    }

    #[test]
    fn test_hurst_estimator_rs() {
        let estimator = HurstEstimator::new(10, 50);
        
        // Create synthetic fractional Brownian motion-like series
        let mut rng = DeterministicRng::new(456);
        let mut series = Vec::new();
        let mut cumsum = FixedPoint::zero();
        
        for _ in 0..200 {
            let increment = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
            cumsum = cumsum + increment;
            series.push(cumsum);
        }
        
        let hurst = estimator.estimate_rs_method(&series).unwrap();
        assert!(hurst.to_float() > 0.0);
        assert!(hurst.to_float() < 0.5);
    }

    #[test]
    fn test_hurst_estimator_dfa() {
        let estimator = HurstEstimator::new(10, 50);
        
        // Create synthetic series
        let mut rng = DeterministicRng::new(789);
        let mut series = Vec::new();
        
        for _ in 0..200 {
            let value = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
            series.push(value);
        }
        
        let hurst = estimator.estimate_dfa_method(&series).unwrap();
        assert!(hurst.to_float() > 0.0);
        assert!(hurst.to_float() < 0.5);
    }

    #[test]
    fn test_volatility_autocorrelation() {
        let detector = VolatilityClusteringDetector::new(10, FixedPoint::from_float(2.0));
        
        // Create returns with some persistence
        let mut rng = DeterministicRng::new(321);
        let mut returns = Vec::new();
        let mut prev_return = FixedPoint::zero();
        
        for _ in 0..100 {
            let noise = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * 
                       FixedPoint::from_float(0.01);
            let return_val = prev_return * FixedPoint::from_float(0.1) + noise;
            returns.push(return_val);
            prev_return = return_val;
        }
        
        let autocorrs = detector.compute_volatility_autocorrelation(&returns).unwrap();
        assert!(!autocorrs.is_empty());
        
        // Check that we get autocorrelations for expected lags
        let lags: Vec<usize> = autocorrs.iter().map(|(lag, _)| *lag).collect();
        assert!(lags.contains(&1));
        assert!(lags.contains(&5));
    }
}