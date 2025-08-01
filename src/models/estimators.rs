use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Normal, StudentsT, ChiSquare, ContinuousCDF};
use thiserror::Error;
use std::collections::HashMap;
use crate::models::state_space::{StateSpaceModel, FilteringResult};

#[derive(Debug, Error)]
pub enum EstimatorError {
    #[error("Estimation error: {0}")]
    EstimationError(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Data error: {0}")]
    DataError(String),
}

pub struct SpreadEstimator {
    max_lags: usize,
    significance_level: f64,
    use_robust_estimation: bool,
    gmm_iterations: usize,
}

impl SpreadEstimator {
    pub fn new(
        max_lags: usize,
        significance_level: f64,
        use_robust_estimation: bool,
        gmm_iterations: usize,
    ) -> Result<Self, EstimatorError> {
        if significance_level <= 0.0 || significance_level >= 1.0 {
            return Err(EstimatorError::EstimationError(
                "Significance level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            max_lags,
            significance_level,
            use_robust_estimation,
            gmm_iterations,
        })
    }

    pub fn estimate_roll_spread(
        &self,
        returns: &[f64],
    ) -> Result<RollEstimate, EstimatorError> {
        let n = returns.len();
        if n < 2 {
            return Err(EstimatorError::DataError(
                "Insufficient data for estimation".to_string(),
            ));
        }

        // Compute first-order serial covariance
        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let centered_returns: Vec<f64> = returns.iter()
            .map(|&x| x - mean_return)
            .collect();

        let serial_cov = centered_returns.windows(2)
            .map(|w| w[0] * w[1])
            .sum::<f64>() / (n - 1) as f64;

        // Compute Roll estimator
        let spread_estimate = if serial_cov < 0.0 {
            2.0 * (-serial_cov).sqrt()
        } else {
            0.0
        };

        // Compute standard error using GMM
        let std_error = self.compute_gmm_standard_error(returns, spread_estimate)?;

        // Compute confidence intervals
        let t_dist = StudentsT::new((n - 1) as f64)
            .map_err(|e| EstimatorError::NumericalError(e.to_string()))?;
        
        let t_value = t_dist.inverse_cdf(1.0 - (1.0 - self.significance_level) / 2.0);
        let margin = t_value * std_error;

        Ok(RollEstimate {
            spread: spread_estimate,
            std_error,
            confidence_interval: (spread_estimate - margin, spread_estimate + margin),
            serial_covariance: serial_cov,
        })
    }

    pub fn estimate_css_spread(
        &self,
        returns: &[f64],
        volumes: &[f64],
    ) -> Result<CSSEstimate, EstimatorError> {
        let n = returns.len();
        if n != volumes.len() {
            return Err(EstimatorError::DataError(
                "Returns and volumes must have same length".to_string(),
            ));
        }

        // Compute volume-weighted returns
        let total_volume: f64 = volumes.iter().sum();
        let weighted_returns: Vec<f64> = returns.iter()
            .zip(volumes.iter())
            .map(|(&r, &v)| r * v / total_volume)
            .collect();

        // Estimate parameters using GMM
        let params = self.estimate_css_parameters(&weighted_returns)?;
        
        let spread_estimate = params.spread;
        let ar_coefficient = params.ar_coefficient;

        // Compute standard errors
        let (spread_se, ar_se) = self.compute_css_standard_errors(
            &weighted_returns,
            spread_estimate,
            ar_coefficient,
        )?;

        // Compute confidence intervals
        let t_dist = StudentsT::new((n - 2) as f64)
            .map_err(|e| EstimatorError::NumericalError(e.to_string()))?;
        
        let t_value = t_dist.inverse_cdf(1.0 - (1.0 - self.significance_level) / 2.0);
        
        let spread_margin = t_value * spread_se;
        let ar_margin = t_value * ar_se;

        Ok(CSSEstimate {
            spread: spread_estimate,
            ar_coefficient,
            spread_std_error: spread_se,
            ar_std_error: ar_se,
            spread_confidence_interval: (
                spread_estimate - spread_margin,
                spread_estimate + spread_margin,
            ),
            ar_confidence_interval: (
                ar_coefficient - ar_margin,
                ar_coefficient + ar_margin,
            ),
        })
    }

    pub fn estimate_modified_spread(
        &self,
        returns: &[f64],
        volumes: &[f64],
        trade_signs: &[i8],
    ) -> Result<ModifiedEstimate, EstimatorError> {
        let n = returns.len();
        if n != volumes.len() || n != trade_signs.len() {
            return Err(EstimatorError::DataError(
                "Input vectors must have same length".to_string(),
            ));
        }

        // Create state space model
        let model = StateSpaceModel::new(
            returns,
            volumes,
            trade_signs,
            self.max_lags,
        )?;

        // Run Kalman filter
        let filter_result = model.run_filter()?;

        // Extract estimates
        let spread_estimate = filter_result.parameter_estimates[0];
        let serial_params: Vec<f64> = filter_result.parameter_estimates[1..].to_vec();

        // Compute standard errors
        let std_errors = self.compute_modified_standard_errors(
            &filter_result,
            returns,
            volumes,
            trade_signs,
        )?;

        // Compute confidence intervals
        let t_dist = StudentsT::new((n - self.max_lags - 1) as f64)
            .map_err(|e| EstimatorError::NumericalError(e.to_string()))?;
        
        let t_value = t_dist.inverse_cdf(1.0 - (1.0 - self.significance_level) / 2.0);

        let spread_margin = t_value * std_errors[0];
        let serial_margins: Vec<f64> = std_errors[1..].iter()
            .map(|&se| t_value * se)
            .collect();

        let serial_confidence_intervals: Vec<(f64, f64)> = serial_params.iter()
            .zip(serial_margins.iter())
            .map(|(&param, &margin)| (param - margin, param + margin))
            .collect();

        Ok(ModifiedEstimate {
            spread: spread_estimate,
            serial_parameters: serial_params,
            spread_std_error: std_errors[0],
            serial_std_errors: std_errors[1..].to_vec(),
            spread_confidence_interval: (
                spread_estimate - spread_margin,
                spread_estimate + spread_margin,
            ),
            serial_confidence_intervals,
            filtered_states: filter_result.filtered_states,
            smoothed_states: filter_result.smoothed_states,
        })
    }

    fn compute_gmm_standard_error(
        &self,
        returns: &[f64],
        spread: f64,
    ) -> Result<f64, EstimatorError> {
        let n = returns.len();
        
        // Compute moment conditions
        let moment_fn = |ret: &[f64], s: f64| {
            let mut moments = Vec::with_capacity(2);
            
            // First moment: E[r_t] = 0
            moments.push(ret.iter().sum::<f64>() / n as f64);
            
            // Second moment: E[r_t * r_{t-1}] = -s^2/4
            let serial_prod: f64 = ret.windows(2)
                .map(|w| w[0] * w[1])
                .sum::<f64>() / (n - 1) as f64;
            moments.push(serial_prod + s * s / 4.0);
            
            moments
        };

        // Compute Jacobian
        let epsilon = 1e-6;
        let base_moments = moment_fn(returns, spread);
        let perturbed_moments = moment_fn(returns, spread + epsilon);
        
        let jacobian: Vec<f64> = base_moments.iter()
            .zip(perturbed_moments.iter())
            .map(|(&b, &p)| (p - b) / epsilon)
            .collect();

        // Compute covariance matrix of moments
        let mut cov_matrix = na::DMatrix::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for t in 0..(n-1) {
                    let moment_i = if i == 0 { returns[t] } else { returns[t] * returns[t+1] };
                    let moment_j = if j == 0 { returns[t] } else { returns[t] * returns[t+1] };
                    sum += moment_i * moment_j;
                }
                cov_matrix[(i, j)] = sum / n as f64;
            }
        }

        // Compute standard error
        let jacobian_matrix = na::DMatrix::from_vec(2, 1, jacobian);
        let variance = (jacobian_matrix.transpose() * cov_matrix * jacobian_matrix)[(0, 0)];
        
        Ok(variance.sqrt() / n as f64)
    }

    fn estimate_css_parameters(
        &self,
        returns: &[f64],
    ) -> Result<CSSParameters, EstimatorError> {
        let n = returns.len();
        
        // Initial parameter guesses
        let mut spread = self.estimate_roll_spread(returns)?.spread;
        let mut ar_coef = 0.0;

        // GMM estimation
        for _ in 0..self.gmm_iterations {
            // Compute moment conditions
            let moments = |s: f64, rho: f64| {
                let mut m = Vec::with_capacity(3);
                
                // E[r_t] = 0
                m.push(returns.iter().sum::<f64>() / n as f64);
                
                // E[r_t * r_{t-1}] = rho * var(r_t) - s^2/4
                let serial_prod: f64 = returns.windows(2)
                    .map(|w| w[0] * w[1])
                    .sum::<f64>() / (n - 1) as f64;
                let variance = returns.iter()
                    .map(|&x| x * x)
                    .sum::<f64>() / n as f64;
                m.push(serial_prod - (rho * variance - s * s / 4.0));
                
                // E[r_t * r_{t-2}] = rho^2 * var(r_t)
                let second_order: f64 = returns.windows(3)
                    .map(|w| w[0] * w[2])
                    .sum::<f64>() / (n - 2) as f64;
                m.push(second_order - rho * rho * variance);
                
                m
            };

            // Compute objective function
            let objective = |s: f64, rho: f64| {
                let m = moments(s, rho);
                m.iter().map(|&x| x * x).sum::<f64>()
            };

            // Numerical optimization using grid search
            let mut min_obj = f64::INFINITY;
            let mut best_s = spread;
            let mut best_rho = ar_coef;

            for s_mult in [-0.1, -0.05, 0.0, 0.05, 0.1].iter() {
                for rho_mult in [-0.1, -0.05, 0.0, 0.05, 0.1].iter() {
                    let s = spread * (1.0 + s_mult);
                    let rho = ar_coef + rho_mult;
                    
                    let obj = objective(s, rho);
                    if obj < min_obj {
                        min_obj = obj;
                        best_s = s;
                        best_rho = rho;
                    }
                }
            }

            // Update parameters
            spread = best_s;
            ar_coef = best_rho;
        }

        Ok(CSSParameters {
            spread,
            ar_coefficient: ar_coef,
        })
    }

    fn compute_css_standard_errors(
        &self,
        returns: &[f64],
        spread: f64,
        ar_coef: f64,
    ) -> Result<(f64, f64), EstimatorError> {
        let n = returns.len();
        
        // Compute score matrix
        let epsilon = 1e-6;
        let score_fn = |ret: &[f64], s: f64, rho: f64| {
            let var = ret.iter().map(|&x| x * x).sum::<f64>() / n as f64;
            let serial_cov = ret.windows(2).map(|w| w[0] * w[1]).sum::<f64>() / (n - 1) as f64;
            
            let score_s = -s / 2.0 * (serial_cov + s * s / 4.0);
            let score_rho = var * (serial_cov - rho * var);
            
            vec![score_s, score_rho]
        };

        let base_score = score_fn(returns, spread, ar_coef);
        let score_s = score_fn(returns, spread + epsilon, ar_coef);
        let score_rho = score_fn(returns, spread, ar_coef + epsilon);

        let mut information_matrix = na::DMatrix::zeros(2, 2);
        for i in 0..2 {
            information_matrix[(i, 0)] = (score_s[i] - base_score[i]) / epsilon;
            information_matrix[(i, 1)] = (score_rho[i] - base_score[i]) / epsilon;
        }

        // Compute standard errors
        match information_matrix.try_inverse() {
            Some(inv_info) => {
                let spread_se = (inv_info[(0, 0)] / n as f64).sqrt();
                let ar_se = (inv_info[(1, 1)] / n as f64).sqrt();
                Ok((spread_se, ar_se))
            },
            None => Err(EstimatorError::NumericalError(
                "Singular information matrix".to_string(),
            )),
        }
    }

    fn compute_modified_standard_errors(
        &self,
        filter_result: &FilteringResult,
        returns: &[f64],
        volumes: &[f64],
        trade_signs: &[i8],
    ) -> Result<Vec<f64>, EstimatorError> {
        let n = returns.len();
        let n_params = filter_result.parameter_estimates.len();
        
        // Compute score matrix
        let epsilon = 1e-6;
        let mut score_matrix = na::DMatrix::zeros(n_params, n_params);

        for i in 0..n_params {
            let mut perturbed_params = filter_result.parameter_estimates.clone();
            perturbed_params[i] += epsilon;

            let perturbed_model = StateSpaceModel::new(
                returns,
                volumes,
                trade_signs,
                self.max_lags,
            )?;

            let perturbed_result = perturbed_model.run_filter()?;
            
            for j in 0..n_params {
                score_matrix[(j, i)] = (perturbed_result.log_likelihood
                    - filter_result.log_likelihood) / epsilon;
            }
        }

        // Compute information matrix
        let information_matrix = score_matrix.transpose() * score_matrix / n as f64;

        // Compute standard errors
        match information_matrix.try_inverse() {
            Some(inv_info) => {
                let std_errors: Vec<f64> = (0..n_params)
                    .map(|i| (inv_info[(i, i)] / n as f64).sqrt())
                    .collect();
                Ok(std_errors)
            },
            None => Err(EstimatorError::NumericalError(
                "Singular information matrix".to_string(),
            )),
        }
    }
}

#[derive(Debug)]
pub struct RollEstimate {
    pub spread: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub serial_covariance: f64,
}

#[derive(Debug)]
pub struct CSSParameters {
    pub spread: f64,
    pub ar_coefficient: f64,
}

#[derive(Debug)]
pub struct CSSEstimate {
    pub spread: f64,
    pub ar_coefficient: f64,
    pub spread_std_error: f64,
    pub ar_std_error: f64,
    pub spread_confidence_interval: (f64, f64),
    pub ar_confidence_interval: (f64, f64),
}

#[derive(Debug)]
pub struct ModifiedEstimate {
    pub spread: f64,
    pub serial_parameters: Vec<f64>,
    pub spread_std_error: f64,
    pub serial_std_errors: Vec<f64>,
    pub spread_confidence_interval: (f64, f64),
    pub serial_confidence_intervals: Vec<(f64, f64)>,
    pub filtered_states: Vec<Vec<f64>>,
    pub smoothed_states: Vec<Vec<f64>>,
}
