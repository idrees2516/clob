use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Poisson, Normal};
use thiserror::Error;
use std::f64;

#[derive(Debug, Error)]
pub enum PINError {
    #[error("Estimation error: {0}")]
    EstimationError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
}

#[derive(Debug, Clone)]
pub struct PINParameters {
    pub alpha: f64,   // Probability of information event
    pub delta: f64,   // Probability of bad news
    pub mu: f64,      // Arrival rate of informed traders
    pub epsilon_b: f64, // Arrival rate of uninformed buyers
    pub epsilon_s: f64, // Arrival rate of uninformed sellers
}

pub struct PINModel {
    max_iterations: usize,
    tolerance: f64,
    num_integration_points: usize,
}

impl PINModel {
    pub fn new(
        max_iterations: usize,
        tolerance: f64,
        num_integration_points: usize,
    ) -> Result<Self, PINError> {
        if tolerance <= 0.0 {
            return Err(PINError::InvalidParameters(
                "Tolerance must be positive".to_string(),
            ));
        }

        Ok(Self {
            max_iterations,
            tolerance,
            num_integration_points,
        })
    }

    pub fn estimate(
        &self,
        buys: &[usize],
        sells: &[usize],
        initial_params: Option<PINParameters>,
    ) -> Result<(PINParameters, f64), PINError> {
        if buys.len() != sells.len() {
            return Err(PINError::InvalidParameters(
                "Buy and sell vectors must have same length".to_string(),
            ));
        }

        let params = initial_params.unwrap_or_else(|| self.initialize_parameters(buys, sells));
        self.maximize_likelihood(buys, sells, params)
    }

    fn initialize_parameters(&self, buys: &[usize], sells: &[usize]) -> PINParameters {
        let mean_buys = buys.iter().sum::<usize>() as f64 / buys.len() as f64;
        let mean_sells = sells.iter().sum::<usize>() as f64 / sells.len() as f64;
        
        PINParameters {
            alpha: 0.3,
            delta: 0.5,
            mu: (mean_buys.max(mean_sells) - mean_buys.min(mean_sells)) / 2.0,
            epsilon_b: mean_buys / 2.0,
            epsilon_s: mean_sells / 2.0,
        }
    }

    fn maximize_likelihood(
        &self,
        buys: &[usize],
        sells: &[usize],
        initial_params: PINParameters,
    ) -> Result<(PINParameters, f64), PINError> {
        let mut current_params = initial_params;
        let mut current_likelihood = f64::NEG_INFINITY;
        
        for _ in 0..self.max_iterations {
            let gradients = self.compute_gradients(buys, sells, &current_params)?;
            let hessian = self.compute_hessian(buys, sells, &current_params)?;
            
            // Newton-Raphson update
            let delta = match hessian.try_inverse() {
                Some(h_inv) => h_inv * gradients,
                None => return Err(PINError::OptimizationError(
                    "Singular Hessian matrix".to_string(),
                )),
            };

            let new_params = self.update_parameters(&current_params, &delta)?;
            let new_likelihood = self.compute_likelihood(buys, sells, &new_params)?;

            if (new_likelihood - current_likelihood).abs() < self.tolerance {
                return Ok((new_params, new_likelihood));
            }

            current_params = new_params;
            current_likelihood = new_likelihood;
        }

        Err(PINError::OptimizationError(
            "Maximum iterations reached without convergence".to_string(),
        ))
    }

    fn compute_likelihood(
        &self,
        buys: &[usize],
        sells: &[usize],
        params: &PINParameters,
    ) -> Result<f64, PINError> {
        let mut total_log_likelihood = 0.0;

        for (&b, &s) in buys.iter().zip(sells.iter()) {
            let day_likelihood = self.compute_day_likelihood(b, s, params)?;
            total_log_likelihood += day_likelihood.ln();
        }

        Ok(total_log_likelihood)
    }

    fn compute_day_likelihood(
        &self,
        buys: usize,
        sells: usize,
        params: &PINParameters,
    ) -> Result<f64, PINError> {
        let PINParameters { alpha, delta, mu, epsilon_b, epsilon_s } = *params;

        // No news
        let l1 = (1.0 - alpha) * self.compute_poisson_probability(buys, epsilon_b)?
            * self.compute_poisson_probability(sells, epsilon_s)?;

        // Bad news
        let l2 = alpha * delta * self.compute_poisson_probability(buys, epsilon_b)?
            * self.compute_poisson_probability(sells, epsilon_s + mu)?;

        // Good news
        let l3 = alpha * (1.0 - delta) * self.compute_poisson_probability(buys, epsilon_b + mu)?
            * self.compute_poisson_probability(sells, epsilon_s)?;

        Ok(l1 + l2 + l3)
    }

    fn compute_poisson_probability(
        &self,
        k: usize,
        lambda: f64,
    ) -> Result<f64, PINError> {
        if lambda <= 0.0 {
            return Err(PINError::InvalidParameters(
                "Poisson rate must be positive".to_string(),
            ));
        }

        let poisson = Poisson::new(lambda)
            .map_err(|e| PINError::EstimationError(e.to_string()))?;
        
        Ok(poisson.pmf(k as f64))
    }

    fn compute_gradients(
        &self,
        buys: &[usize],
        sells: &[usize],
        params: &PINParameters,
    ) -> Result<na::DVector<f64>, PINError> {
        let h = 1e-5;
        let mut gradients = na::DVector::zeros(5);
        let base_likelihood = self.compute_likelihood(buys, sells, params)?;

        // Numerical gradients using central differences
        let param_values = [
            params.alpha, params.delta, params.mu,
            params.epsilon_b, params.epsilon_s
        ];

        for i in 0..5 {
            let mut params_plus = params.clone();
            let mut params_minus = params.clone();
            
            match i {
                0 => {
                    params_plus.alpha += h;
                    params_minus.alpha -= h;
                },
                1 => {
                    params_plus.delta += h;
                    params_minus.delta -= h;
                },
                2 => {
                    params_plus.mu += h;
                    params_minus.mu -= h;
                },
                3 => {
                    params_plus.epsilon_b += h;
                    params_minus.epsilon_b -= h;
                },
                4 => {
                    params_plus.epsilon_s += h;
                    params_minus.epsilon_s -= h;
                },
                _ => unreachable!(),
            }

            let likelihood_plus = self.compute_likelihood(buys, sells, &params_plus)?;
            let likelihood_minus = self.compute_likelihood(buys, sells, &params_minus)?;
            
            gradients[i] = (likelihood_plus - likelihood_minus) / (2.0 * h);
        }

        Ok(gradients)
    }

    fn compute_hessian(
        &self,
        buys: &[usize],
        sells: &[usize],
        params: &PINParameters,
    ) -> Result<na::DMatrix<f64>, PINError> {
        let h = 1e-5;
        let mut hessian = na::DMatrix::zeros(5, 5);
        let base_gradients = self.compute_gradients(buys, sells, params)?;

        for i in 0..5 {
            let mut params_plus = params.clone();
            
            match i {
                0 => params_plus.alpha += h,
                1 => params_plus.delta += h,
                2 => params_plus.mu += h,
                3 => params_plus.epsilon_b += h,
                4 => params_plus.epsilon_s += h,
                _ => unreachable!(),
            }

            let gradients_plus = self.compute_gradients(buys, sells, &params_plus)?;
            
            for j in 0..5 {
                hessian[(i, j)] = (gradients_plus[j] - base_gradients[j]) / h;
            }
        }

        // Symmetrize the Hessian
        for i in 0..5 {
            for j in (i+1)..5 {
                hessian[(j, i)] = hessian[(i, j)];
            }
        }

        Ok(hessian)
    }

    fn update_parameters(
        &self,
        params: &PINParameters,
        delta: &na::DVector<f64>,
    ) -> Result<PINParameters, PINError> {
        let step_size = self.compute_step_size(delta);
        
        let new_params = PINParameters {
            alpha: (params.alpha - step_size * delta[0]).clamp(0.001, 0.999),
            delta: (params.delta - step_size * delta[1]).clamp(0.001, 0.999),
            mu: (params.mu - step_size * delta[2]).max(0.001),
            epsilon_b: (params.epsilon_b - step_size * delta[3]).max(0.001),
            epsilon_s: (params.epsilon_s - step_size * delta[4]).max(0.001),
        };

        Ok(new_params)
    }

    fn compute_step_size(&self, delta: &na::DVector<f64>) -> f64 {
        let norm = delta.norm();
        if norm > 1.0 {
            0.5 / norm
        } else {
            0.5
        }
    }

    pub fn compute_pin(&self, params: &PINParameters) -> f64 {
        let PINParameters { alpha, mu, epsilon_b, epsilon_s, .. } = *params;
        let expected_informed = alpha * mu;
        let expected_uninformed = epsilon_b + epsilon_s;
        
        expected_informed / (expected_informed + expected_uninformed)
    }

    pub fn bootstrap_confidence_intervals(
        &self,
        buys: &[usize],
        sells: &[usize],
        params: &PINParameters,
        n_bootstrap: usize,
        confidence_level: f64,
    ) -> Result<(f64, f64), PINError> {
        let mut pin_estimates = Vec::with_capacity(n_bootstrap);
        let n_days = buys.len();
        
        for _ in 0..n_bootstrap {
            let mut bootstrap_buys = Vec::with_capacity(n_days);
            let mut bootstrap_sells = Vec::with_capacity(n_days);
            
            // Resample with replacement
            for _ in 0..n_days {
                let idx = rand::random::<usize>() % n_days;
                bootstrap_buys.push(buys[idx]);
                bootstrap_sells.push(sells[idx]);
            }
            
            if let Ok((bootstrap_params, _)) = self.estimate(&bootstrap_buys, &bootstrap_sells, Some(params.clone())) {
                pin_estimates.push(self.compute_pin(&bootstrap_params));
            }
        }
        
        if pin_estimates.is_empty() {
            return Err(PINError::EstimationError(
                "Bootstrap failed to produce valid estimates".to_string(),
            ));
        }
        
        pin_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = ((1.0 - confidence_level) / 2.0 * n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 + confidence_level) / 2.0 * n_bootstrap as f64) as usize;
        
        Ok((pin_estimates[lower_idx], pin_estimates[upper_idx]))
    }
}
