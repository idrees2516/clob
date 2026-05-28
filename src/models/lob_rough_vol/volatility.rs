use std::f64::consts::PI;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_rand::rand_distr::{Distribution, Normal};
use ndarray_rand::RandomExt;
use num_complex::Complex;
use num_traits::{Float, FromPrimitive, Zero};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rust_decimal::Decimal;
use rustfft::{Fft, FftPlanner};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{LOBError, LOBResult};

/// Error type for volatility modeling
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("FFT error: {0}")]
    FftError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

impl From<VolatilityError> for LOBError {
    fn from(err: VolatilityError) -> Self {
        LOBError::VolatilityError(err.to_string())
    }
}

/// Parameters for the rough volatility model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughVolatilityParams {
    /// Hurst parameter (0 < H < 0.5)
    pub hurst: f64,
    /// Volatility of volatility
    pub nu: f64,
    /// Mean reversion speed
    pub kappa: f64,
    /// Long-term mean of volatility
    pub theta: f64,
    /// Initial volatility
    pub v0: f64,
    /// Correlation between price and volatility
    pub rho: f64,
    /// Number of time steps per day
    pub time_steps_per_day: usize,
    /// Number of days to simulate
    pub days: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for RoughVolatilityParams {
    fn default() -> Self {
        Self {
            hurst: 0.1,
            nu: 0.3,
            kappa: 1.0,
            theta: 0.1,
            v0: 0.1,
            rho: -0.7,
            time_steps_per_day: 390, // Typical trading minutes
            days: 1,
            seed: None,
        }
    }
}

impl RoughVolatilityParams {
    /// Validate the parameters
    pub fn validate(&self) -> Result<(), VolatilityError> {
        if self.hurst <= 0.0 || self.hurst >= 0.5 {
            return Err(VolatilityError::InvalidParameter(
                "Hurst parameter must be in (0, 0.5)".to_string(),
            ));
        }
        if self.nu <= 0.0 {
            return Err(VolatilityError::InvalidParameter(
                "Volatility of volatility must be positive".to_string(),
            ));
        }
        if self.kappa <= 0.0 {
            return Err(VolatilityError::InvalidParameter(
                "Mean reversion speed must be positive".to_string(),
            ));
        }
        if self.theta <= 0.0 {
            return Err(VolatilityError::InvalidParameter(
                "Long-term mean of volatility must be positive".to_string(),
            ));
        }
        if self.v0 <= 0.0 {
            return Err(VolatilityError::InvalidParameter(
                "Initial volatility must be positive".to_string(),
            ));
        }
        if self.rho < -1.0 || self.rho > 1.0 {
            return Err(VolatilityError::InvalidParameter(
                "Correlation must be in [-1, 1]".to_string(),
            ));
        }
        if self.time_steps_per_day == 0 {
            return Err(VolatilityError::InvalidParameter(
                "Number of time steps per day must be positive".to_string(),
            ));
        }
        if self.days == 0 {
            return Err(VolatilityError::InvalidParameter(
                "Number of days must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Implementation of the rough Bergomi model for rough volatility
pub struct RoughVolatilityModel {
    params: RoughVolatilityParams,
    rng: StdRng,
    fft_planner: FftPlanner<f64>,
}

impl RoughVolatilityModel {
    /// Create a new RoughVolatilityModel with the given parameters
    pub fn new(params: RoughVolatilityParams) -> LOBResult<Self> {
        params.validate()?;
        
        let seed = params.seed.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });
        
        Ok(Self {
            params,
            rng: StdRng::seed_from_u64(seed),
            fft_planner: FftPlanner::new(),
        })
    }
    
    /// Simulate a volatility path using the rough Bergomi model
    pub fn simulate_volatility_path(&mut self) -> LOBResult<Array1<f64>> {
        let n = self.params.time_steps_per_day * self.params.days;
        let dt = 1.0 / (self.params.time_steps_per_day as f64);
        
        // Generate fractional Gaussian noise using FFT
        let fgn = self.fractional_gaussian_noise(n, self.params.hurst)?;
        
        // Initialize volatility path
        let mut v = Array1::zeros(n);
        v[0] = self.params.v0;
        
        // Simulate volatility path
        for i in 1..n {
            let dw_v = fgn[i] * dt.sqrt();
            v[i] = v[i-1] + self.params.kappa * (self.params.theta - v[i-1]) * dt 
                 + self.params.nu * v[i-1].sqrt() * dw_v;
            
            // Ensure non-negativity using reflection
            v[i] = v[i].max(1e-10);
        }
        
        Ok(v)
    }
    
    /// Simulate a price path using the rough volatility model
    pub fn simulate_price_path(&mut self, s0: f64) -> LOBResult<Array1<f64>> {
        let n = self.params.time_steps_per_day * self.params.days;
        let dt = 1.0 / (self.params.time_steps_per_day as f64);
        
        // Generate volatility path
        let v = self.simulate_volatility_path()?;
        
        // Generate correlated Brownian motions
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            VolatilityError::SimulationError(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let mut z = Array1::zeros(n);
        for i in 0..n {
            z[i] = normal.sample(&mut self.rng);
        }
        
        // Initialize price path
        let mut s = Array1::zeros(n);
        s[0] = s0;
        
        // Simulate price path
        for i in 1..n {
            let dw_s = self.params.rho * v[i-1].sqrt() * dt.sqrt() 
                     + (1.0 - self.params.rho.powi(2)).sqrt() * z[i] * dt.sqrt();
            
            s[i] = s[i-1] * (1.0 + v[i-1] * dt + v[i-1].sqrt() * dw_s);
            
            // Ensure non-negativity
            s[i] = s[i].max(1e-10);
        }
        
        Ok(s)
    }
    
    /// Generate fractional Gaussian noise using FFT
    fn fractional_gaussian_noise(&mut self, n: usize, hurst: f64) -> LOBResult<Array1<f64>> {
        // Generate covariance matrix
        let cov = self.fractional_gaussian_covariance(n, hurst)?;
        
        // Compute Cholesky decomposition
        let l = cholesky(&cov).map_err(|e| {
            VolatilityError::SimulationError(format!("Cholesky decomposition failed: {}", e))
        })?;
        
        // Generate standard normal random variables
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            VolatilityError::SimulationError(format!("Failed to create normal distribution: {}", e))
        })?;
        
        let z: Array1<f64> = (0..n).map(|_| normal.sample(&mut self.rng)).collect();
        
        // Transform to correlated normal variables
        Ok(l.dot(&z))
    }
    
    /// Compute the covariance matrix for fractional Gaussian noise
    fn fractional_gaussian_covariance(&self, n: usize, hurst: f64) -> LOBResult<Array2<f64>> {
        let mut cov = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..=i {
                let k = (i as isize - j as isize).abs() as usize;
                cov[[i, j]] = 0.5 * ((k + 1).powf(2.0 * hurst) - 2.0 * (k as f64).powf(2.0 * hurst) 
                                   + (k as isize - 1).max(0) as f64).powf(2.0 * hurst));
                cov[[j, i]] = cov[[i, j]];
            }
        }
        
        // Add small diagonal term for numerical stability
        for i in 0..n {
            cov[[i, i]] += 1e-10;
        }
        
        Ok(cov)
    }
}

/// Compute the Cholesky decomposition of a matrix
fn cholesky(mat: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = mat.nrows();
    let mut l = Array2::zeros((n, n));
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            
            if i == j {
                for k in 0..j {
                    sum += l[[j, k]].powi(2);
                }
                
                let diag = mat[[j, j]] - sum;
                if diag <= 0.0 {
                    return Err("Matrix is not positive definite".to_string());
                }
                
                l[[j, j]] = diag.sqrt();
            } else {
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }
                
                l[[i, j]] = (mat[[i, j]] - sum) / l[[j, j]];
            }
        }
    }
    
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_rough_volatility_params_validation() {
        let mut params = RoughVolatilityParams::default();
        
        // Test valid parameters
        assert!(params.validate().is_ok());
        
        // Test invalid Hurst parameter
        let old_hurst = params.hurst;
        params.hurst = 0.6;
        assert!(params.validate().is_err());
        params.hurst = old_hurst;
        
        // Test invalid nu
        let old_nu = params.nu;
        params.nu = -0.1;
        assert!(params.validate().is_err());
        params.nu = old_nu;
        
        // Test invalid kappa
        let old_kappa = params.kappa;
        params.kappa = -0.1;
        assert!(params.validate().is_err());
        params.kappa = old_kappa;
    }
    
    #[test]
    fn test_rough_volatility_simulation() {
        let params = RoughVolatilityParams {
            hurst: 0.1,
            nu: 0.3,
            kappa: 1.0,
            theta: 0.1,
            v0: 0.1,
            rho: -0.7,
            time_steps_per_day: 10,
            days: 1,
            seed: Some(42),
        };
        
        let mut model = RoughVolatilityModel::new(params).unwrap();
        
        // Test volatility simulation
        let v = model.simulate_volatility_path().unwrap();
        assert_eq!(v.len(), 10);
        assert!(v.iter().all(|&x| x >= 0.0));
        
        // Test price simulation
        let s = model.simulate_price_path(100.0).unwrap();
        assert_eq!(s.len(), 10);
        assert!(s.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_fractional_gaussian_noise() {
        let params = RoughVolatilityParams::default();
        let mut model = RoughVolatilityModel::new(params).unwrap();
        
        let n = 100;
        let hurst = 0.3;
        let fgn = model.fractional_gaussian_noise(n, hurst).unwrap();
        
        assert_eq!(fgn.len(), n);
        
        // Check that the sample mean is close to zero
        let mean = fgn.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 0.5);
        
        // Check that the sample variance is close to 1
        let var = fgn.iter().map(|&x| x * x).sum::<f64>() / (n as f64);
        assert_relative_eq!(var, 1.0, epsilon = 0.5);
    }
}
