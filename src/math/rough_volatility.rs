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

/// Generation method for FBM
#[derive(Debug, Clone, Copy)]
pub enum FBMGenerationMethod {
    /// Exact Cholesky decomposition (O(n^3) memory, O(n^2) per step)
    ExactCholesky,
    /// Approximate circulant embedding (O(n log n) per step)
    CirculantEmbedding,
    /// Hybrid method: exact for small n, approximate for large n
    Hybrid { threshold: usize },
}

/// Fractional Brownian Motion generator using multiple methods
#[derive(Debug, Clone)]
pub struct FractionalBrownianMotion {
    pub hurst_parameter: FixedPoint,
    pub covariance_matrix: Vec<Vec<FixedPoint>>,
    pub cholesky_decomp: Vec<Vec<FixedPoint>>,
    pub max_steps: usize,
    pub generation_method: FBMGenerationMethod,
    pub circulant_eigenvalues: Vec<FixedPoint>,
    pub fft_workspace: Vec<FixedPoint>,
}

impl FractionalBrownianMotion {
    /// Create new FBM generator with given Hurst parameter and method
    pub fn new(hurst_parameter: FixedPoint, max_steps: usize) -> Result<Self, RoughVolatilityError> {
        Self::new_with_method(hurst_parameter, max_steps, FBMGenerationMethod::Hybrid { threshold: 100 })
    }

    /// Create new FBM generator with specified generation method
    pub fn new_with_method(
        hurst_parameter: FixedPoint, 
        max_steps: usize,
        method: FBMGenerationMethod
    ) -> Result<Self, RoughVolatilityError> {
        let h = hurst_parameter.to_float();
        if h <= 0.0 || h >= 0.5 {
            return Err(RoughVolatilityError::InvalidHurstParameter(h));
        }

        let mut fbm = Self {
            hurst_parameter,
            covariance_matrix: Vec::new(),
            cholesky_decomp: Vec::new(),
            max_steps,
            generation_method: method,
            circulant_eigenvalues: Vec::new(),
            fft_workspace: Vec::new(),
        };

        // Initialize based on method
        match method {
            FBMGenerationMethod::ExactCholesky => {
                fbm.precompute_covariance_matrix()?;
                fbm.compute_cholesky_decomposition()?;
            },
            FBMGenerationMethod::CirculantEmbedding => {
                fbm.precompute_circulant_eigenvalues()?;
                fbm.fft_workspace = vec![FixedPoint::zero(); 2 * max_steps];
            },
            FBMGenerationMethod::Hybrid { threshold } => {
                if max_steps <= threshold {
                    fbm.precompute_covariance_matrix()?;
                    fbm.compute_cholesky_decomposition()?;
                } else {
                    fbm.precompute_circulant_eigenvalues()?;
                    fbm.fft_workspace = vec![FixedPoint::zero(); 2 * max_steps];
                }
            }
        }

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

    /// Generate FBM path using the configured method
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

        match self.generation_method {
            FBMGenerationMethod::ExactCholesky => {
                self.generate_path_cholesky(n_steps, rng)
            },
            FBMGenerationMethod::CirculantEmbedding => {
                self.generate_path_circulant(n_steps, rng)
            },
            FBMGenerationMethod::Hybrid { threshold } => {
                if n_steps <= threshold {
                    self.generate_path_cholesky(n_steps, rng)
                } else {
                    self.generate_path_circulant(n_steps, rng)
                }
            }
        }
    }

    /// Generate FBM path using Cholesky decomposition (exact method)
    fn generate_path_cholesky(
        &self, 
        n_steps: usize, 
        rng: &mut DeterministicRng
    ) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
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

    /// Generate FBM path using circulant embedding (approximate method)
    fn generate_path_circulant(
        &self, 
        n_steps: usize, 
        rng: &mut DeterministicRng
    ) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        let n = 2 * n_steps; // Circulant matrix size
        let mut complex_gaussian = Vec::with_capacity(n);

        // Generate complex Gaussian random variables
        for i in 0..n {
            let u1 = rng.next_fixed();
            let u2 = rng.next_fixed();
            
            if i < n / 2 {
                // Real part using Box-Muller
                let real = FixedPoint::from_float(
                    (-2.0 * u1.to_float().ln()).sqrt() * (2.0 * std::f64::consts::PI * u2.to_float()).cos()
                );
                complex_gaussian.push(real);
            } else {
                // Imaginary part using Box-Muller
                let imag = FixedPoint::from_float(
                    (-2.0 * u1.to_float().ln()).sqrt() * (2.0 * std::f64::consts::PI * u2.to_float()).sin()
                );
                complex_gaussian.push(imag);
            }
        }

        // Multiply by square root of eigenvalues (element-wise)
        for i in 0..n {
            if i < self.circulant_eigenvalues.len() {
                let sqrt_eigenval = FixedPoint::from_float(self.circulant_eigenvalues[i].to_float().sqrt());
                complex_gaussian[i] = complex_gaussian[i] * sqrt_eigenval;
            }
        }

        // Apply inverse FFT (simplified version)
        let fft_result = self.inverse_fft(&complex_gaussian)?;

        // Extract the first n_steps real parts
        let mut path = Vec::with_capacity(n_steps);
        for i in 0..n_steps {
            path.push(fft_result[i]);
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

        // Apply Cholesky row for this step (if available)
        if !self.cholesky_decomp.is_empty() && step < self.cholesky_decomp.len() {
            Ok(self.cholesky_decomp[step][step] * z)
        } else {
            // Fallback to simple scaling for circulant method
            let h = self.hurst_parameter.to_float();
            let scaling = FixedPoint::from_float((step + 1) as f64).pow(FixedPoint::from_float(h));
            Ok(z * scaling)
        }
    }

    /// Precompute eigenvalues for circulant embedding
    fn precompute_circulant_eigenvalues(&mut self) -> Result<(), RoughVolatilityError> {
        let n = 2 * self.max_steps;
        let h = self.hurst_parameter.to_float();
        let two_h = 2.0 * h;

        self.circulant_eigenvalues = Vec::with_capacity(n);

        // First row of circulant matrix contains covariances
        let mut first_row = Vec::with_capacity(n);
        
        // Covariances for positive lags
        for k in 0..self.max_steps {
            let t = (k + 1) as f64;
            let covariance = if k == 0 {
                1.0 // Variance at lag 0
            } else {
                0.5 * (2.0_f64.powf(two_h) + (k as f64 - 1.0).abs().powf(two_h) - (k as f64 + 1.0).abs().powf(two_h))
            };
            first_row.push(FixedPoint::from_float(covariance));
        }

        // Covariances for negative lags (symmetric)
        for k in 1..self.max_steps {
            let idx = self.max_steps - k;
            if idx < first_row.len() {
                first_row.push(first_row[idx]);
            }
        }

        // Compute eigenvalues using DFT of first row
        self.circulant_eigenvalues = self.compute_dft_eigenvalues(&first_row)?;

        // Ensure all eigenvalues are non-negative (numerical stability)
        for eigenval in &mut self.circulant_eigenvalues {
            if eigenval.to_float() < 0.0 {
                *eigenval = FixedPoint::zero();
            }
        }

        Ok(())
    }

    /// Compute DFT eigenvalues for circulant matrix
    fn compute_dft_eigenvalues(&self, first_row: &[FixedPoint]) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        let n = first_row.len();
        let mut eigenvalues = Vec::with_capacity(n);

        // Simple DFT computation (O(n^2) - could be optimized with FFT)
        for k in 0..n {
            let mut real_sum = FixedPoint::zero();
            let mut imag_sum = FixedPoint::zero();

            for j in 0..n {
                let angle = -2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let cos_val = FixedPoint::from_float(angle.cos());
                let sin_val = FixedPoint::from_float(angle.sin());

                real_sum = real_sum + first_row[j] * cos_val;
                imag_sum = imag_sum + first_row[j] * sin_val;
            }

            // Eigenvalue magnitude
            let magnitude = FixedPoint::from_float(
                (real_sum.to_float().powi(2) + imag_sum.to_float().powi(2)).sqrt()
            );
            eigenvalues.push(magnitude);
        }

        Ok(eigenvalues)
    }

    /// Simplified inverse FFT implementation
    fn inverse_fft(&self, complex_data: &[FixedPoint]) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        let n = complex_data.len();
        let mut result = vec![FixedPoint::zero(); n];

        // Simple inverse DFT (O(n^2) - could be optimized)
        for k in 0..n {
            let mut real_sum = FixedPoint::zero();

            for j in 0..n {
                let angle = 2.0 * std::f64::consts::PI * (k * j) as f64 / n as f64;
                let cos_val = FixedPoint::from_float(angle.cos());

                // Only use real part for simplicity
                real_sum = real_sum + complex_data[j] * cos_val;
            }

            result[k] = real_sum / FixedPoint::from_float(n as f64);
        }

        Ok(result)
    }

    /// Get memory usage estimate for current configuration
    pub fn get_memory_usage(&self) -> usize {
        let mut usage = 0;
        
        // Covariance matrix
        usage += self.covariance_matrix.len() * self.covariance_matrix.get(0).map_or(0, |row| row.len()) * 8;
        
        // Cholesky decomposition
        usage += self.cholesky_decomp.len() * self.cholesky_decomp.get(0).map_or(0, |row| row.len()) * 8;
        
        // Circulant eigenvalues
        usage += self.circulant_eigenvalues.len() * 8;
        
        // FFT workspace
        usage += self.fft_workspace.len() * 8;
        
        usage
    }

    /// Get computational complexity estimate for n steps
    pub fn get_complexity_estimate(&self, n_steps: usize) -> (usize, usize) {
        match self.generation_method {
            FBMGenerationMethod::ExactCholesky => {
                // O(n^2) time, O(n^2) space
                (n_steps * n_steps, n_steps * n_steps)
            },
            FBMGenerationMethod::CirculantEmbedding => {
                // O(n log n) time, O(n) space
                (n_steps * (n_steps as f64).log2() as usize, n_steps)
            },
            FBMGenerationMethod::Hybrid { threshold } => {
                if n_steps <= threshold {
                    (n_steps * n_steps, n_steps * n_steps)
                } else {
                    (n_steps * (n_steps as f64).log2() as usize, n_steps)
                }
            }
        }
    }
}

/// Rough volatility state for SDE simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughVolatilityState {
    pub log_volatility: FixedPoint,
    pub volatility: FixedPoint,
    pub time: FixedPoint,
    pub fbm_history: VecDeque<FixedPoint>,
}

impl RoughVolatilityState {
    pub fn new(initial_log_vol: FixedPoint, time: FixedPoint) -> Self {
        Self {
            log_volatility: initial_log_vol,
            volatility: initial_log_vol.exp(),
            time,
            fbm_history: VecDeque::new(),
        }
    }

    pub fn update(&mut self, new_log_vol: FixedPoint, new_time: FixedPoint, fbm_increment: FixedPoint) {
        self.log_volatility = new_log_vol;
        self.volatility = new_log_vol.exp();
        self.time = new_time;
        self.fbm_history.push_back(fbm_increment);
        
        // Limit history size for memory management
        if self.fbm_history.len() > 1000 {
            self.fbm_history.pop_front();
        }
    }
}

/// Rough volatility parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughVolatilityParams {
    pub hurst_parameter: FixedPoint,
    pub vol_of_vol: FixedPoint,
    pub mean_reversion: FixedPoint,
    pub long_term_variance: FixedPoint,
    pub correlation: FixedPoint, // Correlation with price process
}

impl RoughVolatilityParams {
    pub fn new(
        hurst_parameter: FixedPoint,
        vol_of_vol: FixedPoint,
        mean_reversion: FixedPoint,
        long_term_variance: FixedPoint,
        correlation: FixedPoint,
    ) -> Result<Self, RoughVolatilityError> {
        // Validate parameters
        let h = hurst_parameter.to_float();
        if h <= 0.0 || h >= 0.5 {
            return Err(RoughVolatilityError::InvalidHurstParameter(h));
        }

        if vol_of_vol.to_float() <= 0.0 {
            return Err(RoughVolatilityError::NumericalError(
                "Vol of vol must be positive".to_string()
            ));
        }

        if mean_reversion.to_float() <= 0.0 {
            return Err(RoughVolatilityError::NumericalError(
                "Mean reversion must be positive".to_string()
            ));
        }

        if long_term_variance.to_float() <= 0.0 {
            return Err(RoughVolatilityError::NumericalError(
                "Long term variance must be positive".to_string()
            ));
        }

        let corr = correlation.to_float();
        if corr < -1.0 || corr > 1.0 {
            return Err(RoughVolatilityError::NumericalError(
                "Correlation must be between -1 and 1".to_string()
            ));
        }

        Ok(Self {
            hurst_parameter,
            vol_of_vol,
            mean_reversion,
            long_term_variance,
            correlation,
        })
    }
}

/// Rough volatility path simulator
#[derive(Debug, Clone)]
pub struct RoughVolatilitySimulator {
    pub fbm_generator: FractionalBrownianMotion,
    pub params: RoughVolatilityParams,
    pub initial_log_vol: FixedPoint,
}

impl RoughVolatilitySimulator {
    pub fn new(
        params: RoughVolatilityParams,
        initial_log_vol: FixedPoint,
        max_steps: usize,
    ) -> Result<Self, RoughVolatilityError> {
        let fbm_generator = FractionalBrownianMotion::new(params.hurst_parameter, max_steps)?;

        Ok(Self {
            fbm_generator,
            params,
            initial_log_vol,
        })
    }

    pub fn new_with_method(
        params: RoughVolatilityParams,
        initial_log_vol: FixedPoint,
        max_steps: usize,
        method: FBMGenerationMethod,
    ) -> Result<Self, RoughVolatilityError> {
        let fbm_generator = FractionalBrownianMotion::new_with_method(
            params.hurst_parameter, 
            max_steps, 
            method
        )?;

        Ok(Self {
            fbm_generator,
            params,
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
            let mean_reversion_term = self.params.mean_reversion * 
                (current_log_vol - self.params.long_term_variance.ln()) * dt;

            // Rough noise term: ν dW_t^H
            let noise_term = self.params.vol_of_vol * fbm_path[i];

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

    /// Simulate single step of rough volatility process
    pub fn simulate_step(
        &self,
        current_state: &RoughVolatilityState,
        dt: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<RoughVolatilityState, RoughVolatilityError> {
        // Generate FBM increment
        let step = current_state.fbm_history.len();
        let fbm_increment = self.fbm_generator.generate_increment(step, rng)?;

        // Mean reversion term
        let mean_reversion_term = self.params.mean_reversion * 
            (current_state.log_volatility - self.params.long_term_variance.ln()) * dt;

        // Rough noise term
        let noise_term = self.params.vol_of_vol * fbm_increment;

        // Update log volatility
        let new_log_vol = current_state.log_volatility - mean_reversion_term + noise_term;
        let new_time = current_state.time + dt;

        let mut new_state = RoughVolatilityState::new(new_log_vol, new_time);
        new_state.fbm_history = current_state.fbm_history.clone();
        new_state.update(new_log_vol, new_time, fbm_increment);

        Ok(new_state)
    }

    /// Calibrate parameters from historical volatility data
    pub fn calibrate_parameters(
        &mut self,
        historical_volatilities: &[FixedPoint],
        dt: FixedPoint,
    ) -> Result<RoughVolatilityParams, RoughVolatilityError> {
        if historical_volatilities.len() < 100 {
            return Err(RoughVolatilityError::InsufficientData(100));
        }

        // Convert to log volatilities
        let log_vols: Vec<FixedPoint> = historical_volatilities.iter()
            .map(|&vol| {
                if vol.to_float() > 0.0 {
                    vol.ln()
                } else {
                    FixedPoint::from_float(-5.0) // Floor for numerical stability
                }
            })
            .collect();

        // Estimate Hurst parameter using existing estimator
        let hurst_estimator = HurstEstimator::new(10, 50);
        let hurst = hurst_estimator.estimate_dfa_method(&log_vols)?;

        // Estimate mean reversion and long-term variance using AR(1) model
        let (mean_reversion, long_term_var) = self.estimate_ar1_parameters(&log_vols, dt)?;

        // Estimate vol of vol from residuals
        let vol_of_vol = self.estimate_vol_of_vol(&log_vols, mean_reversion, long_term_var, dt)?;

        // Create new parameters
        let new_params = RoughVolatilityParams::new(
            hurst,
            vol_of_vol,
            mean_reversion,
            long_term_var,
            FixedPoint::zero(), // Default correlation
        )?;

        self.params = new_params.clone();
        Ok(new_params)
    }

    /// Estimate AR(1) parameters for mean reversion
    fn estimate_ar1_parameters(
        &self,
        log_vols: &[FixedPoint],
        dt: FixedPoint,
    ) -> Result<(FixedPoint, FixedPoint), RoughVolatilityError> {
        if log_vols.len() < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        let n = log_vols.len() - 1;
        let mut sum_x = FixedPoint::zero();
        let mut sum_y = FixedPoint::zero();
        let mut sum_xy = FixedPoint::zero();
        let mut sum_x2 = FixedPoint::zero();

        // Linear regression: y_t = α + β * y_{t-1} + ε_t
        for i in 0..n {
            let x = log_vols[i];     // y_{t-1}
            let y = log_vols[i + 1]; // y_t

            sum_x = sum_x + x;
            sum_y = sum_y + y;
            sum_xy = sum_xy + x * y;
            sum_x2 = sum_x2 + x * x;
        }

        let n_fp = FixedPoint::from_float(n as f64);
        let x_mean = sum_x / n_fp;
        let y_mean = sum_y / n_fp;

        let numerator = sum_xy - n_fp * x_mean * y_mean;
        let denominator = sum_x2 - n_fp * x_mean * x_mean;

        if denominator.to_float().abs() < 1e-12 {
            return Err(RoughVolatilityError::NumericalError(
                "Singular matrix in AR(1) estimation".to_string()
            ));
        }

        let beta = numerator / denominator;
        let alpha = y_mean - beta * x_mean;

        // Convert to continuous-time parameters
        let mean_reversion = -beta.ln() / dt;
        let long_term_var = alpha / (FixedPoint::one() - beta);

        Ok((mean_reversion, long_term_var.exp()))
    }

    /// Estimate vol of vol from AR(1) residuals
    fn estimate_vol_of_vol(
        &self,
        log_vols: &[FixedPoint],
        mean_reversion: FixedPoint,
        long_term_var: FixedPoint,
        dt: FixedPoint,
    ) -> Result<FixedPoint, RoughVolatilityError> {
        if log_vols.len() < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        let mut residual_variance = FixedPoint::zero();
        let n = log_vols.len() - 1;

        for i in 0..n {
            // Expected change based on mean reversion
            let expected_change = -mean_reversion * 
                (log_vols[i] - long_term_var.ln()) * dt;
            
            // Actual change
            let actual_change = log_vols[i + 1] - log_vols[i];
            
            // Residual
            let residual = actual_change - expected_change;
            residual_variance = residual_variance + residual * residual;
        }

        residual_variance = residual_variance / FixedPoint::from_float(n as f64);
        
        // Convert to vol of vol (accounting for dt scaling)
        let vol_of_vol = FixedPoint::from_float(
            (residual_variance.to_float() / dt.to_float()).sqrt()
        );

        Ok(vol_of_vol)
    }
}

/// Realized volatility estimator for high-frequency data
#[derive(Debug, Clone)]
pub struct RealizedVolatilityEstimator {
    pub sampling_frequency: FixedPoint, // In seconds
    pub microstructure_adjustment: bool,
    pub jump_robust: bool,
}

impl RealizedVolatilityEstimator {
    pub fn new(sampling_frequency: FixedPoint) -> Self {
        Self {
            sampling_frequency,
            microstructure_adjustment: true,
            jump_robust: true,
        }
    }

    /// Compute realized volatility from high-frequency returns
    pub fn compute_realized_volatility(
        &self,
        returns: &[FixedPoint],
        timestamps: &[FixedPoint],
    ) -> Result<FixedPoint, RoughVolatilityError> {
        if returns.len() != timestamps.len() || returns.len() < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        if self.jump_robust {
            self.compute_bipower_variation(returns)
        } else {
            self.compute_simple_realized_volatility(returns)
        }
    }

    /// Simple realized volatility: RV = Σ r_i^2
    fn compute_simple_realized_volatility(&self, returns: &[FixedPoint]) -> Result<FixedPoint, RoughVolatilityError> {
        let sum_squared: FixedPoint = returns.iter()
            .map(|&r| r * r)
            .fold(FixedPoint::zero(), |acc, x| acc + x);

        Ok(sum_squared.sqrt())
    }

    /// Jump-robust bipower variation: BV = Σ |r_i| * |r_{i-1}|
    fn compute_bipower_variation(&self, returns: &[FixedPoint]) -> Result<FixedPoint, RoughVolatilityError> {
        if returns.len() < 2 {
            return Err(RoughVolatilityError::InsufficientData(2));
        }

        let mut bipower_sum = FixedPoint::zero();
        for i in 1..returns.len() {
            bipower_sum = bipower_sum + returns[i].abs() * returns[i-1].abs();
        }

        // Scale by π/2 for consistency with quadratic variation
        let pi_half = FixedPoint::from_float(std::f64::consts::PI / 2.0);
        Ok((bipower_sum * pi_half).sqrt())
    }

    /// Compute realized volatility with microstructure noise adjustment
    pub fn compute_noise_adjusted_rv(
        &self,
        returns: &[FixedPoint],
        noise_variance: FixedPoint,
    ) -> Result<FixedPoint, RoughVolatilityError> {
        let raw_rv = self.compute_simple_realized_volatility(returns)?;
        let raw_variance = raw_rv * raw_rv;
        
        // Adjust for microstructure noise: RV_adjusted = RV_raw - 2 * noise_variance
        let adjusted_variance = raw_variance - FixedPoint::from_float(2.0) * noise_variance;
        
        if adjusted_variance.to_float() <= 0.0 {
            Ok(FixedPoint::from_float(0.001)) // Floor for numerical stability
        } else {
            Ok(adjusted_variance.sqrt())
        }
    }

    /// Compute intraday volatility pattern
    pub fn compute_intraday_pattern(
        &self,
        returns_by_time: &[(FixedPoint, FixedPoint)], // (time_of_day, return)
        n_buckets: usize,
    ) -> Result<Vec<FixedPoint>, RoughVolatilityError> {
        if returns_by_time.len() < n_buckets {
            return Err(RoughVolatilityError::InsufficientData(n_buckets));
        }

        let mut buckets = vec![Vec::new(); n_buckets];
        let trading_hours = FixedPoint::from_float(6.5 * 3600.0); // 6.5 hours in seconds

        // Assign returns to time buckets
        for &(time_of_day, return_val) in returns_by_time {
            let bucket_idx = ((time_of_day / trading_hours).to_float() * n_buckets as f64) as usize;
            let bucket_idx = bucket_idx.min(n_buckets - 1);
            buckets[bucket_idx].push(return_val);
        }

        // Compute volatility for each bucket
        let mut pattern = Vec::with_capacity(n_buckets);
        for bucket in buckets {
            if bucket.is_empty() {
                pattern.push(FixedPoint::zero());
            } else {
                let bucket_rv = self.compute_simple_realized_volatility(&bucket)?;
                pattern.push(bucket_rv);
            }
        }

        Ok(pattern)
    }
}

/// GARCH model with rough volatility extensions
#[derive(Debug, Clone)]
pub struct RoughGARCHModel {
    pub alpha: FixedPoint,      // ARCH coefficient
    pub beta: FixedPoint,       // GARCH coefficient
    pub omega: FixedPoint,      // Constant term
    pub hurst: FixedPoint,      // Hurst parameter for rough extension
    pub rough_weight: FixedPoint, // Weight of rough component
    pub fbm_generator: FractionalBrownianMotion,
}

impl RoughGARCHModel {
    pub fn new(
        alpha: FixedPoint,
        beta: FixedPoint,
        omega: FixedPoint,
        hurst: FixedPoint,
        rough_weight: FixedPoint,
    ) -> Result<Self, RoughVolatilityError> {
        // Validate GARCH parameters
        if (alpha + beta).to_float() >= 1.0 {
            return Err(RoughVolatilityError::NumericalError(
                "GARCH parameters must satisfy alpha + beta < 1".to_string()
            ));
        }

        let fbm_generator = FractionalBrownianMotion::new(hurst, 1000)?;

        Ok(Self {
            alpha,
            beta,
            omega,
            hurst,
            rough_weight,
            fbm_generator,
        })
    }

    /// Forecast volatility using rough GARCH model
    /// σ²_{t+1} = ω + α*r²_t + β*σ²_t + γ*W^H_t
    pub fn forecast_volatility(
        &self,
        current_variance: FixedPoint,
        last_return: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<FixedPoint, RoughVolatilityError> {
        // Standard GARCH component
        let garch_component = self.omega + 
            self.alpha * last_return * last_return + 
            self.beta * current_variance;

        // Rough volatility component
        let rough_increment = self.fbm_generator.generate_increment(0, rng)?;
        let rough_component = self.rough_weight * rough_increment;

        let forecasted_variance = garch_component + rough_component;
        
        // Ensure positive variance
        if forecasted_variance.to_float() <= 0.0 {
            Ok(FixedPoint::from_float(0.0001))
        } else {
            Ok(forecasted_variance.sqrt())
        }
    }

    /// Estimate GARCH parameters from return series
    pub fn estimate_parameters(
        &mut self,
        returns: &[FixedPoint],
    ) -> Result<(), RoughVolatilityError> {
        if returns.len() < 100 {
            return Err(RoughVolatilityError::InsufficientData(100));
        }

        // Simple method of moments estimation
        let mean_return = returns.iter().fold(FixedPoint::zero(), |acc, &r| acc + r) / 
                         FixedPoint::from_float(returns.len() as f64);

        let mut sum_squared_deviations = FixedPoint::zero();
        let mut sum_fourth_moments = FixedPoint::zero();

        for &r in returns {
            let deviation = r - mean_return;
            let squared_dev = deviation * deviation;
            sum_squared_deviations = sum_squared_deviations + squared_dev;
            sum_fourth_moments = sum_fourth_moments + squared_dev * squared_dev;
        }

        let variance = sum_squared_deviations / FixedPoint::from_float(returns.len() as f64);
        let kurtosis = sum_fourth_moments / FixedPoint::from_float(returns.len() as f64) / 
                      (variance * variance);

        // Method of moments estimates
        self.omega = variance * FixedPoint::from_float(0.1);
        self.alpha = FixedPoint::from_float(0.1);
        self.beta = FixedPoint::from_float(0.8);

        // Adjust based on excess kurtosis
        if kurtosis.to_float() > 3.5 {
            self.alpha = self.alpha * FixedPoint::from_float(1.2);
            self.beta = self.beta * FixedPoint::from_float(0.9);
        }

        Ok(())
    }
}

/// Regime-dependent volatility model with smooth transitions
#[derive(Debug, Clone)]
pub struct RegimeDependentVolatilityModel {
    pub regimes: Vec<VolatilityRegime>,
    pub transition_matrix: Vec<Vec<FixedPoint>>,
    pub current_regime_probabilities: Vec<FixedPoint>,
    pub smoothing_parameter: FixedPoint,
}

#[derive(Debug, Clone)]
pub struct VolatilityRegime {
    pub id: usize,
    pub mean_volatility: FixedPoint,
    pub volatility_of_volatility: FixedPoint,
    pub persistence: FixedPoint,
    pub rough_params: Option<RoughVolatilityParams>,
}

impl RegimeDependentVolatilityModel {
    pub fn new(regimes: Vec<VolatilityRegime>, smoothing_parameter: FixedPoint) -> Self {
        let n_regimes = regimes.len();
        let uniform_prob = FixedPoint::one() / FixedPoint::from_float(n_regimes as f64);
        
        Self {
            regimes,
            transition_matrix: vec![vec![uniform_prob; n_regimes]; n_regimes],
            current_regime_probabilities: vec![uniform_prob; n_regimes],
            smoothing_parameter,
        }
    }

    /// Update regime probabilities based on observed volatility
    pub fn update_regime_probabilities(
        &mut self,
        observed_volatility: FixedPoint,
    ) -> Result<(), RoughVolatilityError> {
        let mut new_probabilities = Vec::with_capacity(self.regimes.len());

        // Compute likelihood of observed volatility under each regime
        let mut total_likelihood = FixedPoint::zero();
        for regime in &self.regimes {
            let likelihood = self.compute_regime_likelihood(observed_volatility, regime);
            new_probabilities.push(likelihood);
            total_likelihood = total_likelihood + likelihood;
        }

        // Normalize probabilities
        if total_likelihood.to_float() > 1e-12 {
            for prob in &mut new_probabilities {
                *prob = *prob / total_likelihood;
            }
        } else {
            // Fallback to uniform distribution
            let uniform = FixedPoint::one() / FixedPoint::from_float(self.regimes.len() as f64);
            new_probabilities = vec![uniform; self.regimes.len()];
        }

        // Apply smoothing
        for i in 0..self.current_regime_probabilities.len() {
            self.current_regime_probabilities[i] = 
                self.smoothing_parameter * new_probabilities[i] + 
                (FixedPoint::one() - self.smoothing_parameter) * self.current_regime_probabilities[i];
        }

        Ok(())
    }

    /// Compute likelihood of volatility under a specific regime
    fn compute_regime_likelihood(&self, volatility: FixedPoint, regime: &VolatilityRegime) -> FixedPoint {
        // Simple Gaussian likelihood
        let diff = volatility - regime.mean_volatility;
        let variance = regime.volatility_of_volatility * regime.volatility_of_volatility;
        
        let exponent = -(diff * diff) / (FixedPoint::from_float(2.0) * variance);
        let normalization = FixedPoint::one() / (variance * FixedPoint::from_float(2.0 * std::f64::consts::PI)).sqrt();
        
        normalization * exponent.exp()
    }

    /// Forecast volatility using regime-weighted average
    pub fn forecast_volatility(
        &self,
        current_volatility: FixedPoint,
        dt: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<FixedPoint, RoughVolatilityError> {
        let mut weighted_forecast = FixedPoint::zero();

        for (i, regime) in self.regimes.iter().enumerate() {
            let regime_prob = self.current_regime_probabilities[i];
            
            // Simple mean reversion forecast for this regime
            let mean_reversion_rate = FixedPoint::one() - regime.persistence;
            let regime_forecast = current_volatility * regime.persistence + 
                                regime.mean_volatility * mean_reversion_rate;

            // Add rough volatility component if available
            let final_regime_forecast = if let Some(ref rough_params) = regime.rough_params {
                let rough_simulator = RoughVolatilitySimulator::new(
                    rough_params.clone(),
                    regime_forecast.ln(),
                    100,
                )?;
                
                let rough_state = RoughVolatilityState::new(regime_forecast.ln(), FixedPoint::zero());
                let new_state = rough_simulator.simulate_step(&rough_state, dt, rng)?;
                new_state.volatility
            } else {
                regime_forecast
            };

            weighted_forecast = weighted_forecast + regime_prob * final_regime_forecast;
        }

        Ok(weighted_forecast)
    }

    /// Get the most likely current regime
    pub fn get_most_likely_regime(&self) -> usize {
        let mut max_prob = FixedPoint::zero();
        let mut max_idx = 0;

        for (i, &prob) in self.current_regime_probabilities.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_idx = i;
            }
        }

        max_idx
    }
}

/// Volatility surface construction and interpolation
#[derive(Debug, Clone)]
pub struct VolatilitySurface {
    pub strikes: Vec<FixedPoint>,
    pub maturities: Vec<FixedPoint>,
    pub volatilities: Vec<Vec<FixedPoint>>, // volatilities[maturity][strike]
    pub spot_price: FixedPoint,
}

impl VolatilitySurface {
    pub fn new(
        strikes: Vec<FixedPoint>,
        maturities: Vec<FixedPoint>,
        volatilities: Vec<Vec<FixedPoint>>,
        spot_price: FixedPoint,
    ) -> Result<Self, RoughVolatilityError> {
        // Validate dimensions
        if volatilities.len() != maturities.len() {
            return Err(RoughVolatilityError::NumericalError(
                "Volatility matrix dimensions don't match maturities".to_string()
            ));
        }

        for vol_row in &volatilities {
            if vol_row.len() != strikes.len() {
                return Err(RoughVolatilityError::NumericalError(
                    "Volatility matrix dimensions don't match strikes".to_string()
                ));
            }
        }

        Ok(Self {
            strikes,
            maturities,
            volatilities,
            spot_price,
        })
    }

    /// Interpolate volatility for given strike and maturity
    pub fn interpolate_volatility(
        &self,
        strike: FixedPoint,
        maturity: FixedPoint,
    ) -> Result<FixedPoint, RoughVolatilityError> {
        // Find surrounding points
        let (t_idx, t_weight) = self.find_interpolation_indices(&self.maturities, maturity)?;
        let (k_idx, k_weight) = self.find_interpolation_indices(&self.strikes, strike)?;

        // Bilinear interpolation
        let vol_00 = self.volatilities[t_idx][k_idx];
        let vol_01 = if k_idx + 1 < self.strikes.len() {
            self.volatilities[t_idx][k_idx + 1]
        } else {
            vol_00
        };
        let vol_10 = if t_idx + 1 < self.maturities.len() {
            self.volatilities[t_idx + 1][k_idx]
        } else {
            vol_00
        };
        let vol_11 = if t_idx + 1 < self.maturities.len() && k_idx + 1 < self.strikes.len() {
            self.volatilities[t_idx + 1][k_idx + 1]
        } else {
            vol_00
        };

        // Interpolate in strike direction
        let vol_0 = vol_00 * (FixedPoint::one() - k_weight) + vol_01 * k_weight;
        let vol_1 = vol_10 * (FixedPoint::one() - k_weight) + vol_11 * k_weight;

        // Interpolate in time direction
        let interpolated_vol = vol_0 * (FixedPoint::one() - t_weight) + vol_1 * t_weight;

        Ok(interpolated_vol)
    }

    /// Find interpolation indices and weights
    fn find_interpolation_indices(
        &self,
        grid: &[FixedPoint],
        target: FixedPoint,
    ) -> Result<(usize, FixedPoint), RoughVolatilityError> {
        if grid.is_empty() {
            return Err(RoughVolatilityError::InsufficientData(1));
        }

        // Handle boundary cases
        if target <= grid[0] {
            return Ok((0, FixedPoint::zero()));
        }
        if target >= grid[grid.len() - 1] {
            return Ok((grid.len() - 1, FixedPoint::zero()));
        }

        // Find surrounding indices
        for i in 0..grid.len() - 1 {
            if target >= grid[i] && target <= grid[i + 1] {
                let weight = (target - grid[i]) / (grid[i + 1] - grid[i]);
                return Ok((i, weight));
            }
        }

        Ok((0, FixedPoint::zero()))
    }

    /// Construct volatility surface from market data using rough volatility model
    pub fn construct_from_rough_model(
        strikes: Vec<FixedPoint>,
        maturities: Vec<FixedPoint>,
        spot_price: FixedPoint,
        rough_params: &RoughVolatilityParams,
        initial_volatility: FixedPoint,
    ) -> Result<Self, RoughVolatilityError> {
        let mut volatilities = Vec::with_capacity(maturities.len());

        for &maturity in &maturities {
            let mut vol_row = Vec::with_capacity(strikes.len());
            
            for &strike in &strikes {
                // Simple rough volatility surface model
                let moneyness = (strike / spot_price).ln();
                let time_scaling = maturity.pow(rough_params.hurst_parameter);
                
                // Rough volatility adjustment based on moneyness and time
                let rough_adjustment = rough_params.vol_of_vol * 
                    moneyness.abs() * time_scaling;
                
                let vol = initial_volatility + rough_adjustment;
                vol_row.push(vol.max(FixedPoint::from_float(0.01))); // Floor at 1%
            }
            
            volatilities.push(vol_row);
        }

        Self::new(strikes, maturities, volatilities, spot_price)
    }

    /// Get at-the-money volatility term structure
    pub fn get_atm_term_structure(&self) -> Vec<(FixedPoint, FixedPoint)> {
        let mut term_structure = Vec::new();

        for (i, &maturity) in self.maturities.iter().enumerate() {
            // Find closest strike to spot
            let mut closest_idx = 0;
            let mut min_diff = (self.strikes[0] - self.spot_price).abs();

            for (j, &strike) in self.strikes.iter().enumerate() {
                let diff = (strike - self.spot_price).abs();
                if diff < min_diff {
                    min_diff = diff;
                    closest_idx = j;
                }
            }

            let atm_vol = self.volatilities[i][closest_idx];
            term_structure.push((maturity, atm_vol));
        }

        term_structure
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
        let params = RoughVolatilityParams::new(
            FixedPoint::from_float(0.2),  // hurst
            FixedPoint::from_float(0.3),  // vol_of_vol
            FixedPoint::from_float(2.0),  // mean_reversion
            FixedPoint::from_float(0.04), // long_term_variance
            FixedPoint::zero(),           // correlation
        ).unwrap();
        
        let initial_log_vol = FixedPoint::from_float(0.2_f64.ln());
        let simulator = RoughVolatilitySimulator::new(params, initial_log_vol, 100).unwrap();
        
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
    fn test_fbm_generation_methods() {
        let hurst = FixedPoint::from_float(0.3);
        
        // Test exact Cholesky method
        let fbm_exact = FractionalBrownianMotion::new_with_method(
            hurst, 50, FBMGenerationMethod::ExactCholesky
        ).unwrap();
        
        let mut rng = DeterministicRng::new(42);
        let path_exact = fbm_exact.generate_path(30, &mut rng).unwrap();
        assert_eq!(path_exact.len(), 30);
        
        // Test circulant embedding method
        let fbm_circulant = FractionalBrownianMotion::new_with_method(
            hurst, 50, FBMGenerationMethod::CirculantEmbedding
        ).unwrap();
        
        let mut rng2 = DeterministicRng::new(42);
        let path_circulant = fbm_circulant.generate_path(30, &mut rng2).unwrap();
        assert_eq!(path_circulant.len(), 30);
        
        // Test hybrid method
        let fbm_hybrid = FractionalBrownianMotion::new_with_method(
            hurst, 200, FBMGenerationMethod::Hybrid { threshold: 100 }
        ).unwrap();
        
        let mut rng3 = DeterministicRng::new(42);
        let path_hybrid_small = fbm_hybrid.generate_path(50, &mut rng3).unwrap(); // Should use exact
        assert_eq!(path_hybrid_small.len(), 50);
        
        let mut rng4 = DeterministicRng::new(42);
        let path_hybrid_large = fbm_hybrid.generate_path(150, &mut rng4).unwrap(); // Should use circulant
        assert_eq!(path_hybrid_large.len(), 150);
    }

    #[test]
    fn test_rough_volatility_state() {
        let initial_log_vol = FixedPoint::from_float(0.2_f64.ln());
        let mut state = RoughVolatilityState::new(initial_log_vol, FixedPoint::zero());
        
        assert_eq!(state.log_volatility, initial_log_vol);
        assert!((state.volatility.to_float() - 0.2).abs() < 1e-6);
        
        // Update state
        let new_log_vol = FixedPoint::from_float(0.25_f64.ln());
        let new_time = FixedPoint::from_float(1.0);
        let fbm_increment = FixedPoint::from_float(0.01);
        
        state.update(new_log_vol, new_time, fbm_increment);
        
        assert_eq!(state.log_volatility, new_log_vol);
        assert_eq!(state.time, new_time);
        assert_eq!(state.fbm_history.len(), 1);
        assert_eq!(state.fbm_history[0], fbm_increment);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let hurst = FixedPoint::from_float(0.3);
        
        let fbm_small = FractionalBrownianMotion::new_with_method(
            hurst, 10, FBMGenerationMethod::ExactCholesky
        ).unwrap();
        
        let fbm_large = FractionalBrownianMotion::new_with_method(
            hurst, 100, FBMGenerationMethod::CirculantEmbedding
        ).unwrap();
        
        let usage_small = fbm_small.get_memory_usage();
        let usage_large = fbm_large.get_memory_usage();
        
        // Circulant should use less memory for large n
        assert!(usage_small > 0);
        assert!(usage_large > 0);
        
        // Test complexity estimates
        let (time_small, space_small) = fbm_small.get_complexity_estimate(10);
        let (time_large, space_large) = fbm_large.get_complexity_estimate(100);
        
        assert!(time_small > 0);
        assert!(space_small > 0);
        assert!(time_large > 0);
        assert!(space_large > 0);
    }

    #[test]
    fn test_parameter_validation() {
        // Test invalid Hurst parameter
        let result = RoughVolatilityParams::new(
            FixedPoint::from_float(0.6), // Invalid: > 0.5
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.04),
            FixedPoint::zero(),
        );
        assert!(result.is_err());
        
        // Test invalid vol of vol
        let result = RoughVolatilityParams::new(
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(-0.1), // Invalid: negative
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.04),
            FixedPoint::zero(),
        );
        assert!(result.is_err());
        
        // Test invalid correlation
        let result = RoughVolatilityParams::new(
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.04),
            FixedPoint::from_float(1.5), // Invalid: > 1
        );
        assert!(result.is_err());
        
        // Test valid parameters
        let result = RoughVolatilityParams::new(
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.04),
            FixedPoint::from_float(0.5),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_realized_volatility_estimator() {
        let estimator = RealizedVolatilityEstimator::new(FixedPoint::from_float(1.0));
        
        // Create synthetic high-frequency returns
        let returns = vec![
            FixedPoint::from_float(0.001),
            FixedPoint::from_float(-0.002),
            FixedPoint::from_float(0.0015),
            FixedPoint::from_float(-0.001),
            FixedPoint::from_float(0.0005),
        ];
        
        let timestamps = vec![
            FixedPoint::from_float(0.0),
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(3.0),
            FixedPoint::from_float(4.0),
        ];
        
        let rv = estimator.compute_realized_volatility(&returns, &timestamps).unwrap();
        assert!(rv.to_float() > 0.0);
        
        // Test bipower variation (should be different from simple RV)
        let bv = estimator.compute_bipower_variation(&returns).unwrap();
        assert!(bv.to_float() > 0.0);
        assert_ne!(rv.to_float(), bv.to_float());
    }

    #[test]
    fn test_rough_garch_model() {
        let mut model = RoughGARCHModel::new(
            FixedPoint::from_float(0.1),  // alpha
            FixedPoint::from_float(0.8),  // beta
            FixedPoint::from_float(0.01), // omega
            FixedPoint::from_float(0.3),  // hurst
            FixedPoint::from_float(0.1),  // rough_weight
        ).unwrap();
        
        let mut rng = DeterministicRng::new(123);
        let current_variance = FixedPoint::from_float(0.04);
        let last_return = FixedPoint::from_float(0.02);
        
        let forecasted_vol = model.forecast_volatility(current_variance, last_return, &mut rng).unwrap();
        assert!(forecasted_vol.to_float() > 0.0);
        
        // Test parameter estimation
        let returns = vec![FixedPoint::from_float(0.01); 100];
        model.estimate_parameters(&returns).unwrap();
        
        // Parameters should be updated
        assert!(model.alpha.to_float() > 0.0);
        assert!(model.beta.to_float() > 0.0);
        assert!(model.omega.to_float() > 0.0);
    }

    #[test]
    fn test_regime_dependent_volatility() {
        let regime1 = VolatilityRegime {
            id: 0,
            mean_volatility: FixedPoint::from_float(0.15),
            volatility_of_volatility: FixedPoint::from_float(0.3),
            persistence: FixedPoint::from_float(0.9),
            rough_params: None,
        };
        
        let regime2 = VolatilityRegime {
            id: 1,
            mean_volatility: FixedPoint::from_float(0.35),
            volatility_of_volatility: FixedPoint::from_float(0.5),
            persistence: FixedPoint::from_float(0.7),
            rough_params: None,
        };
        
        let mut model = RegimeDependentVolatilityModel::new(
            vec![regime1, regime2],
            FixedPoint::from_float(0.1),
        );
        
        // Test regime probability updates
        model.update_regime_probabilities(FixedPoint::from_float(0.2)).unwrap();
        assert_eq!(model.current_regime_probabilities.len(), 2);
        
        // Test volatility forecasting
        let mut rng = DeterministicRng::new(456);
        let forecast = model.forecast_volatility(
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(1.0 / 252.0),
            &mut rng,
        ).unwrap();
        assert!(forecast.to_float() > 0.0);
        
        // Test most likely regime
        let regime_idx = model.get_most_likely_regime();
        assert!(regime_idx < 2);
    }

    #[test]
    fn test_volatility_surface() {
        let strikes = vec![
            FixedPoint::from_float(90.0),
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(110.0),
        ];
        
        let maturities = vec![
            FixedPoint::from_float(0.25),
            FixedPoint::from_float(0.5),
        ];
        
        let volatilities = vec![
            vec![
                FixedPoint::from_float(0.25),
                FixedPoint::from_float(0.2),
                FixedPoint::from_float(0.25),
            ],
            vec![
                FixedPoint::from_float(0.22),
                FixedPoint::from_float(0.18),
                FixedPoint::from_float(0.22),
            ],
        ];
        
        let spot_price = FixedPoint::from_float(100.0);
        let surface = VolatilitySurface::new(strikes, maturities, volatilities, spot_price).unwrap();
        
        // Test interpolation
        let interpolated_vol = surface.interpolate_volatility(
            FixedPoint::from_float(105.0), // Strike between 100 and 110
            FixedPoint::from_float(0.375), // Maturity between 0.25 and 0.5
        ).unwrap();
        
        assert!(interpolated_vol.to_float() > 0.0);
        assert!(interpolated_vol.to_float() < 0.3);
        
        // Test ATM term structure
        let atm_structure = surface.get_atm_term_structure();
        assert_eq!(atm_structure.len(), 2);
        assert_eq!(atm_structure[0].0, FixedPoint::from_float(0.25));
        assert_eq!(atm_structure[1].0, FixedPoint::from_float(0.5));
    }

    #[test]
    fn test_volatility_surface_construction() {
        let strikes = vec![
            FixedPoint::from_float(90.0),
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(110.0),
        ];
        
        let maturities = vec![
            FixedPoint::from_float(0.25),
            FixedPoint::from_float(0.5),
        ];
        
        let spot_price = FixedPoint::from_float(100.0);
        let rough_params = RoughVolatilityParams::new(
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.04),
            FixedPoint::zero(),
        ).unwrap();
        
        let initial_vol = FixedPoint::from_float(0.2);
        
        let surface = VolatilitySurface::construct_from_rough_model(
            strikes,
            maturities,
            spot_price,
            &rough_params,
            initial_vol,
        ).unwrap();
        
        assert_eq!(surface.strikes.len(), 3);
        assert_eq!(surface.maturities.len(), 2);
        assert_eq!(surface.volatilities.len(), 2);
        assert_eq!(surface.volatilities[0].len(), 3);
        
        // All volatilities should be positive
        for vol_row in &surface.volatilities {
            for &vol in vol_row {
                assert!(vol.to_float() > 0.0);
            }
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