use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, StudentT};
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use statrs::function::gamma::gamma;
use std::f64::consts::PI;
use neldermead::{NelderMead, SDTolerance, MaxIterations};

#[derive(Debug, Error)]
pub enum RoughVolatilityError {
    #[error("Rough volatility error: {0}")]
    RoughVolatilityError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

/// Represents the rough volatility model parameters
pub struct RoughVolatilityParams {
    pub hurst_index: f64,      // Hurst parameter H ∈ (0,1/2)
    pub vol_of_vol: f64,       // Volatility of volatility σ
    pub mean_reversion: f64,   // Mean reversion speed λ
    pub theta: f64,            // Long-term mean level θ
    pub correlation: f64,      // Correlation ρ between price and volatility
    pub time_scale: f64,       // Time scale parameter τ
}

/// Represents the rough kernel model parameters
#[derive(Debug, Clone)]
pub struct RoughKernelParams {
    pub gamma: f64,
    pub nu: f64,
    pub rho: f64,
    pub theta: f64,
    pub xi: f64,
    pub eta: Vec<f64>,
}

/// Implements the rough volatility model for high-frequency trading
pub struct RoughVolatilityModel {
    params: RoughVolatilityParams,
    grid_points: usize,
    dt: f64,
    rng: ThreadRng,
    initial_vol: f64,
    initial_price: f64,
    risk_free_rate: f64,
    time_horizon: f64,
}

impl RoughVolatilityModel {
    pub fn new(
        params: RoughVolatilityParams,
        grid_points: usize,
        time_horizon: f64,
        initial_vol: f64,
        initial_price: f64,
        risk_free_rate: f64,
    ) -> Result<Self, RoughVolatilityError> {
        // Validate parameters
        if params.hurst_index >= 0.5 || params.hurst_index <= 0.0 {
            return Err(RoughVolatilityError::ParameterError(
                "Hurst index must be in (0,1/2)".to_string(),
            ));
        }

        if params.vol_of_vol <= 0.0 || params.mean_reversion <= 0.0 || params.time_scale <= 0.0 {
            return Err(RoughVolatilityError::ParameterError(
                "Volatility parameters must be positive".to_string(),
            ));
        }

        if params.correlation.abs() > 1.0 {
            return Err(RoughVolatilityError::ParameterError(
                "Correlation must be in [-1,1]".to_string(),
            ));
        }

        Ok(Self {
            params,
            grid_points,
            dt: time_horizon / grid_points as f64,
            rng: thread_rng(),
            initial_vol,
            initial_price,
            risk_free_rate,
            time_horizon,
        })
    }

    /// Simulates the rough volatility process using Volterra representation
    pub fn simulate_volatility(&mut self) -> Result<Vec<f64>, RoughVolatilityError> {
        let n = self.grid_points;
        let mut volatility = vec![0.0; n];
        
        // Generate standard Brownian increments
        let normal = Normal::new(0.0, self.dt.sqrt())
            .map_err(|e| RoughVolatilityError::RoughVolatilityError(e.to_string()))?;
        
        let dw: Vec<f64> = (0..n)
            .map(|_| self.rng.sample(normal))
            .collect();

        // Compute Volterra kernel
        let kernel = self.compute_volterra_kernel()?;

        // Simulate rough volatility using convolution
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum += kernel[i - j] * dw[j];
            }
            
            // Apply Ornstein-Uhlenbeck dynamics
            let mean_rev = -self.params.mean_reversion * (volatility[i] - self.params.theta);
            volatility[i] = mean_rev * self.dt + self.params.vol_of_vol * sum;
        }

        Ok(volatility)
    }

    /// Computes the Volterra kernel for rough volatility
    fn compute_volterra_kernel(&self) -> Result<Vec<f64>, RoughVolatilityError> {
        let n = self.grid_points;
        let h = self.params.hurst_index;
        let mut kernel = vec![0.0; n];

        // Compute K_H(t) = √(2H) * t^(H-1/2)
        for i in 1..n {
            let t = i as f64 * self.dt;
            kernel[i] = (2.0 * h).sqrt() * t.powf(h - 0.5);
        }

        // Handle singularity at t=0
        kernel[0] = if h > 0.25 {
            (2.0 * h).sqrt() * self.dt.powf(h - 0.5)
        } else {
            0.0
        };

        Ok(kernel)
    }

    /// Simulates the price process with rough volatility
    pub fn simulate_price_process(
        &mut self,
        initial_price: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), RoughVolatilityError> {
        let n = self.grid_points;
        let mut prices = vec![initial_price; n];
        let volatility = self.simulate_volatility()?;

        // Generate correlated Brownian increments
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| RoughVolatilityError::RoughVolatilityError(e.to_string()))?;

        let z1: Vec<f64> = (0..n)
            .map(|_| self.rng.sample(normal))
            .collect();

        let z2: Vec<f64> = (0..n)
            .map(|_| self.rng.sample(normal))
            .collect();

        // Construct correlated price increments
        let rho = self.params.correlation;
        let sqrt_dt = self.dt.sqrt();

        for i in 1..n {
            let dw_p = sqrt_dt * (rho * z1[i] + (1.0 - rho * rho).sqrt() * z2[i]);
            prices[i] = prices[i-1] * (1.0 + volatility[i-1].sqrt() * dw_p);
        }

        Ok((prices, volatility))
    }

    /// Computes the fractional roughness index
    pub fn compute_roughness_index(&self, volatility: &[f64]) -> Result<f64, RoughVolatilityError> {
        let n = volatility.len();
        if n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Insufficient data for roughness computation".to_string(),
            ));
        }

        // Compute quadratic variations at different scales
        let max_scale = (n as f64).log2().floor() as usize;
        let mut scales = Vec::with_capacity(max_scale);
        let mut variations = Vec::with_capacity(max_scale);

        for k in 1..=max_scale {
            let scale = 2usize.pow(k as u32);
            let mut qv = 0.0;
            
            for i in 0..(n/scale) {
                let increment = volatility[i * scale + scale - 1] - volatility[i * scale];
                qv += increment.powi(2);
            }

            scales.push((scale as f64).ln());
            variations.push((qv / (n/scale) as f64).ln());
        }

        // Estimate Hurst index through linear regression
        let (slope, _) = self.linear_regression(&scales, &variations)?;
        Ok(slope / 2.0)
    }

    /// Helper function for linear regression
    fn linear_regression(
        &self,
        x: &[f64],
        y: &[f64],
    ) -> Result<(f64, f64), RoughVolatilityError> {
        let n = x.len();
        if n != y.len() || n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Invalid data for regression".to_string(),
            ));
        }

        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            numerator += dx * (y[i] - mean_y);
            denominator += dx * dx;
        }

        let slope = numerator / denominator;
        let intercept = mean_y - slope * mean_x;

        Ok((slope, intercept))
    }

    /// Calibrates the rough volatility model to market data
    pub fn calibrate(
        &mut self,
        market_prices: &[f64],
        market_volumes: &[f64],
    ) -> Result<RoughVolatilityParams, RoughVolatilityError> {
        // Compute realized volatility
        let realized_vol = self.compute_realized_volatility(market_prices)?;
        
        // Estimate Hurst index
        let hurst = self.compute_roughness_index(&realized_vol)?;
        
        // Estimate volatility of volatility
        let vol_of_vol = realized_vol.iter()
            .zip(realized_vol.iter().skip(1))
            .map(|(&v1, &v2)| (v2 - v1).powi(2))
            .sum::<f64>()
            .sqrt() / (realized_vol.len() - 1) as f64;

        // Estimate mean reversion parameters
        let (mean_rev, theta) = self.estimate_ou_parameters(&realized_vol)?;

        // Estimate correlation
        let correlation = self.estimate_price_vol_correlation(market_prices, &realized_vol)?;

        // Estimate time scale from market volumes
        let time_scale = self.estimate_time_scale(market_volumes)?;

        Ok(RoughVolatilityParams {
            hurst_index: hurst,
            vol_of_vol,
            mean_reversion: mean_rev,
            theta,
            correlation,
            time_scale,
        })
    }

    /// Computes realized volatility from price data
    fn compute_realized_volatility(&self, prices: &[f64]) -> Result<Vec<f64>, RoughVolatilityError> {
        let n = prices.len();
        if n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Insufficient price data".to_string(),
            ));
        }

        let window_size = (n as f64).sqrt().floor() as usize;
        let mut realized_vol = Vec::with_capacity(n - window_size + 1);

        for i in 0..(n - window_size + 1) {
            let mut sum_squared_returns = 0.0;
            for j in 0..(window_size-1) {
                let ret = (prices[i + j + 1] / prices[i + j]).ln();
                sum_squared_returns += ret * ret;
            }
            realized_vol.push((sum_squared_returns / window_size as f64).sqrt());
        }

        Ok(realized_vol)
    }

    /// Estimates Ornstein-Uhlenbeck parameters
    fn estimate_ou_parameters(
        &self,
        volatility: &[f64],
    ) -> Result<(f64, f64), RoughVolatilityError> {
        let n = volatility.len();
        if n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Insufficient volatility data".to_string(),
            ));
        }

        let mean = volatility.iter().sum::<f64>() / n as f64;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for i in 0..(n-1) {
            sum_xy += (volatility[i] - mean) * (volatility[i+1] - volatility[i]);
            sum_xx += (volatility[i] - mean).powi(2);
        }

        let mean_rev = -sum_xy / (self.dt * sum_xx);
        let theta = mean;

        Ok((mean_rev, theta))
    }

    /// Estimates correlation between price and volatility
    fn estimate_price_vol_correlation(
        &self,
        prices: &[f64],
        volatility: &[f64],
    ) -> Result<f64, RoughVolatilityError> {
        let n = prices.len().min(volatility.len());
        if n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Insufficient data for correlation estimation".to_string(),
            ));
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let vol_changes: Vec<f64> = volatility.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let mean_vol = vol_changes.iter().sum::<f64>() / vol_changes.len() as f64;

        let mut covariance = 0.0;
        let mut var_ret = 0.0;
        let mut var_vol = 0.0;

        for i in 0..returns.len() {
            let dr = returns[i] - mean_ret;
            let dv = vol_changes[i] - mean_vol;
            covariance += dr * dv;
            var_ret += dr * dr;
            var_vol += dv * dv;
        }

        Ok(covariance / (var_ret * var_vol).sqrt())
    }

    /// Estimates characteristic time scale from volume data
    fn estimate_time_scale(&self, volumes: &[f64]) -> Result<f64, RoughVolatilityError> {
        let n = volumes.len();
        if n < 2 {
            return Err(RoughVolatilityError::ParameterError(
                "Insufficient volume data".to_string(),
            ));
        }

        // Compute volume autocorrelation function
        let mean_vol = volumes.iter().sum::<f64>() / n as f64;
        let mut acf = Vec::with_capacity(n/2);

        for lag in 1..=(n/2) {
            let mut sum = 0.0;
            for i in 0..(n-lag) {
                sum += (volumes[i] - mean_vol) * (volumes[i+lag] - mean_vol);
            }
            acf.push(sum / (n - lag) as f64);
        }

        // Find first zero crossing
        let mut time_scale = self.dt;
        for (i, &ac) in acf.iter().enumerate() {
            if ac <= 0.0 {
                time_scale = (i + 1) as f64 * self.dt;
                break;
            }
        }

        Ok(time_scale)
    }

    /// Computes the rough kernel for rough volatility
    pub fn compute_rough_kernel(
        &self,
        params: &RoughKernelParams,
        t: f64,
        s: f64,
    ) -> Result<f64, RoughVolatilityError> {
        let h = 0.5 + params.gamma;
        let dt = t - s;
        
        if dt <= 0.0 {
            return Ok(0.0);
        }

        let power_term = dt.powf(h - 1.0);
        let exp_term = (-params.nu * dt).exp();
        let rho_term = params.rho * (params.theta - dt).exp();
        
        let eta_term: f64 = params.eta.iter().enumerate()
            .map(|(i, &eta_i)| {
                let i_float = i as f64;
                eta_i * (-params.xi * i_float * dt).exp()
            })
            .sum();

        Ok(power_term * exp_term * (1.0 + rho_term + eta_term))
    }

    /// Simulates the rough Bergomi model
    pub fn simulate_rough_bergomi(
        &self,
        n_paths: usize,
        dt: f64,
        params: &RoughKernelParams,
    ) -> Result<(Vec<f64>, Vec<f64>), RoughVolatilityError> {
        let n_steps = (self.time_horizon / dt).ceil() as usize;
        let h = 0.5 + params.gamma;
        
        // Generate correlated Brownian motions
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| RoughVolatilityError::SimulationError(e.to_string()))?;

        let mut w1: Vec<f64> = Vec::with_capacity(n_steps);
        let mut w2: Vec<f64> = Vec::with_capacity(n_steps);
        
        for _ in 0..n_steps {
            let z1: f64 = normal.sample(&mut rng);
            let z2: f64 = normal.sample(&mut rng);
            let w1_step = z1;
            let w2_step = params.rho * z1 + (1.0 - params.rho * params.rho).sqrt() * z2;
            w1.push(w1_step);
            w2.push(w2_step);
        }

        // Compute volatility process
        let mut vol = vec![self.initial_vol; n_steps];
        let mut price = vec![self.initial_price; n_steps];
        
        // Pre-compute kernel values
        let mut kernel_matrix = vec![vec![0.0; n_steps]; n_steps];
        for i in 0..n_steps {
            for j in 0..=i {
                kernel_matrix[i][j] = self.compute_rough_kernel(
                    params,
                    i as f64 * dt,
                    j as f64 * dt,
                )?;
            }
        }

        // Simulate paths using parallel processing
        let chunk_size = n_paths / rayon::current_num_threads().max(1);
        let results: Vec<_> = (0..n_paths)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut path_vols = Vec::with_capacity(n_steps);
                let mut path_prices = Vec::with_capacity(n_steps);
                
                for _ in chunk {
                    let mut vol_path = vec![self.initial_vol; n_steps];
                    let mut price_path = vec![self.initial_price; n_steps];
                    
                    for i in 1..n_steps {
                        let mut vol_integral = 0.0;
                        for j in 0..i {
                            vol_integral += kernel_matrix[i][j] * w1[j] * dt;
                        }
                        
                        vol_path[i] = self.initial_vol * (vol_integral).exp();
                        
                        let drift = (self.risk_free_rate - 0.5 * vol_path[i].powi(2)) * dt;
                        let diffusion = vol_path[i] * w2[i] * dt.sqrt();
                        price_path[i] = price_path[i-1] * (drift + diffusion).exp();
                    }
                    
                    path_vols.push(vol_path);
                    path_prices.push(price_path);
                }
                
                (path_vols, path_prices)
            })
            .collect();

        // Aggregate results
        let mut final_vols = vec![0.0; n_steps];
        let mut final_prices = vec![0.0; n_steps];
        
        for (path_vols, path_prices) in results {
            for (i, path) in path_vols.iter().enumerate() {
                for j in 0..n_steps {
                    final_vols[j] += path[j] / n_paths as f64;
                }
            }
            for (i, path) in path_prices.iter().enumerate() {
                for j in 0..n_steps {
                    final_prices[j] += path[j] / n_paths as f64;
                }
            }
        }

        Ok((final_vols, final_prices))
    }

    /// Calibrates the rough kernel model to market data
    pub fn calibrate_rough_kernel(
        &self,
        market_data: &[(f64, f64)],
        initial_params: RoughKernelParams,
    ) -> Result<RoughKernelParams, RoughVolatilityError> {
        let objective = |params: &[f64]| -> f64 {
            let kernel_params = RoughKernelParams {
                gamma: params[0],
                nu: params[1],
                rho: params[2].clamp(-1.0, 1.0),
                theta: params[3],
                xi: params[4],
                eta: params[5..].to_vec(),
            };
            
            let mut error = 0.0;
            for &(t, market_vol) in market_data {
                if let Ok((model_vols, _)) = self.simulate_rough_bergomi(
                    100,
                    0.01,
                    &kernel_params,
                ) {
                    let model_vol = model_vols.last().unwrap_or(&0.0);
                    error += (market_vol - model_vol).powi(2);
                }
            }
            error
        };

        let initial_x = vec![
            initial_params.gamma,
            initial_params.nu,
            initial_params.rho,
            initial_params.theta,
            initial_params.xi,
        ];
        initial_x.extend(&initial_params.eta);

        let mut optimizer = NelderMead::new(initial_x)
            .with_sd_tolerance(1e-8)
            .with_max_iterations(1000);

        let result = optimizer
            .minimize(&objective)
            .map_err(|e| RoughVolatilityError::OptimizationError(e.to_string()))?;

        Ok(RoughKernelParams {
            gamma: result[0],
            nu: result[1],
            rho: result[2].clamp(-1.0, 1.0),
            theta: result[3],
            xi: result[4],
            eta: result[5..].to_vec(),
        })
    }
}
