use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::f64;

#[derive(Debug, Error)]
pub enum GarchError {
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

#[derive(Debug, Clone)]
pub struct GarchModel {
    omega: f64,      // Constant term
    alpha: f64,      // ARCH parameter
    beta: f64,       // GARCH parameter
    returns: Vec<f64>,
    volatilities: Vec<f64>,
}

impl GarchModel {
    pub fn new() -> Self {
        Self {
            omega: 0.0,
            alpha: 0.0,
            beta: 0.0,
            returns: Vec::new(),
            volatilities: Vec::new(),
        }
    }

    pub fn with_params(omega: f64, alpha: f64, beta: f64) -> Result<Self, GarchError> {
        // Validate parameters
        if omega <= 0.0 {
            return Err(GarchError::InvalidParameters(
                "omega must be positive".to_string(),
            ));
        }
        if alpha < 0.0 || beta < 0.0 {
            return Err(GarchError::InvalidParameters(
                "alpha and beta must be non-negative".to_string(),
            ));
        }
        if alpha + beta >= 1.0 {
            return Err(GarchError::InvalidParameters(
                "alpha + beta must be less than 1 for stationarity".to_string(),
            ));
        }

        Ok(Self {
            omega,
            alpha,
            beta,
            returns: Vec::new(),
            volatilities: Vec::new(),
        })
    }

    pub fn fit(&mut self, returns: Vec<f64>, max_iter: usize, tolerance: f64) -> Result<(), GarchError> {
        if returns.len() < 2 {
            return Err(GarchError::InsufficientData(
                "Need at least 2 observations".to_string(),
            ));
        }

        self.returns = returns;
        let n = self.returns.len();
        
        // Initial parameter guesses if not already set
        if self.omega == 0.0 && self.alpha == 0.0 && self.beta == 0.0 {
            let variance = self.returns.iter()
                .map(|&x| x * x)
                .sum::<f64>() / n as f64;
            self.omega = variance * 0.1;
            self.alpha = 0.1;
            self.beta = 0.8;
        }

        let mut prev_likelihood = f64::NEG_INFINITY;
        
        for _ in 0..max_iter {
            // E-step: Compute volatilities
            self.update_volatilities()?;
            
            // M-step: Update parameters
            let (new_omega, new_alpha, new_beta) = self.optimize_parameters()?;
            
            // Calculate likelihood
            let likelihood = self.log_likelihood()?;
            
            // Check convergence
            if (likelihood - prev_likelihood).abs() < tolerance {
                break;
            }
            
            self.omega = new_omega;
            self.alpha = new_alpha;
            self.beta = new_beta;
            prev_likelihood = likelihood;
        }

        Ok(())
    }

    fn update_volatilities(&mut self) -> Result<(), GarchError> {
        let n = self.returns.len();
        self.volatilities = vec![0.0; n];
        
        // Initialize with unconditional variance
        let unconditional_var = self.omega / (1.0 - self.alpha - self.beta);
        self.volatilities[0] = unconditional_var;
        
        // Update volatilities recursively
        for t in 1..n {
            self.volatilities[t] = self.omega 
                + self.alpha * self.returns[t-1].powi(2)
                + self.beta * self.volatilities[t-1];
        }
        
        Ok(())
    }

    fn optimize_parameters(&self) -> Result<(f64, f64, f64), GarchError> {
        // Implement numerical optimization (e.g., BFGS or Nelder-Mead)
        // Here we use a simple grid search for demonstration
        let grid_size = 20;
        let mut best_params = (self.omega, self.alpha, self.beta);
        let mut best_likelihood = f64::NEG_INFINITY;
        
        let omega_range = (0.0001..0.01).step_by(grid_size);
        let alpha_range = (0.0..0.3).step_by(grid_size);
        let beta_range = (0.6..0.99).step_by(grid_size);
        
        for omega in omega_range {
            for alpha in alpha_range.clone() {
                for beta in beta_range.clone() {
                    if alpha + beta >= 1.0 {
                        continue;
                    }
                    
                    let mut temp_model = Self::new(omega, alpha, beta)?;
                    temp_model.returns = self.returns.clone();
                    temp_model.update_volatilities()?;
                    
                    let likelihood = temp_model.log_likelihood()?;
                    if likelihood > best_likelihood {
                        best_likelihood = likelihood;
                        best_params = (omega, alpha, beta);
                    }
                }
            }
        }
        
        Ok(best_params)
    }

    fn log_likelihood(&self) -> Result<f64, GarchError> {
        let n = self.returns.len();
        let log_likelihood: f64 = (0..n)
            .into_par_iter()
            .map(|t| {
                let sigma2 = self.volatilities[t];
                -0.5 * (f64::consts::PI * 2.0).ln() 
                    - 0.5 * sigma2.ln() 
                    - 0.5 * self.returns[t].powi(2) / sigma2
            })
            .sum();
            
        Ok(log_likelihood)
    }

    pub fn forecast(&self, horizon: usize) -> Result<Vec<f64>, GarchError> {
        if self.returns.is_empty() {
            return Err(GarchError::InsufficientData(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        let mut forecasts = Vec::with_capacity(horizon);
        let mut last_variance = *self.volatilities.last().unwrap();
        
        for _ in 0..horizon {
            let forecast = self.omega + self.alpha * self.returns.last().unwrap().powi(2) 
                + self.beta * last_variance;
            forecasts.push(forecast);
            last_variance = forecast;
        }
        
        Ok(forecasts)
    }

    pub fn forecast_volatility(&self, horizon: usize) -> Result<crate::math::fixed_point::FixedPoint, GarchError> {
        if self.returns.is_empty() {
            return Err(GarchError::InsufficientData(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        let mut last_variance = *self.volatilities.last().unwrap();
        
        for _ in 0..horizon {
            last_variance = self.omega + self.alpha * self.returns.last().unwrap().powi(2) 
                + self.beta * last_variance;
        }
        
        // Return volatility (square root of variance)
        Ok(crate::math::fixed_point::FixedPoint::from_float(last_variance.sqrt()))
    }

    pub fn fit(&mut self, returns: &[crate::math::fixed_point::FixedPoint]) -> Result<(), crate::error::RiskError> {
        if returns.len() < 10 {
            return Err(crate::error::RiskError::InsufficientData(
                "Need at least 10 observations for GARCH fitting".to_string(),
            ));
        }

        // Convert FixedPoint returns to f64 for internal processing
        let f64_returns: Vec<f64> = returns.iter().map(|r| r.to_float()).collect();
        
        // Use existing fit method with default parameters
        self.fit(f64_returns, 100, 1e-6)
            .map_err(|e| crate::error::RiskError::ModelError(e.to_string()))
    }

    pub fn get_volatilities(&self) -> &[f64] {
        &self.volatilities
    }

    pub fn get_parameters(&self) -> (f64, f64, f64) {
        (self.omega, self.alpha, self.beta)
    }

    pub fn persistence(&self) -> f64 {
        self.alpha + self.beta
    }

    pub fn unconditional_variance(&self) -> Result<f64, GarchError> {
        if self.alpha + self.beta >= 1.0 {
            return Err(GarchError::InvalidParameters(
                "Model is not stationary".to_string(),
            ));
        }
        Ok(self.omega / (1.0 - self.alpha - self.beta))
    }

    pub fn half_life(&self) -> Result<f64, GarchError> {
        let persistence = self.persistence();
        if persistence >= 1.0 {
            return Err(GarchError::InvalidParameters(
                "Model is not stationary".to_string(),
            ));
        }
        Ok(f64::ln(0.5) / f64::ln(persistence))
    }
}
