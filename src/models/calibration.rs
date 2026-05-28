use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order},
    optimal_trading::{OptimalController, MarketImpact},
    toxicity::{ToxicityMetrics, ToxicityState},
    risk::{RiskManager, RiskMetrics},
};

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("GMM error: {0}")]
    GMMError(String),
    #[error("Maximum likelihood error: {0}")]
    MLError(String),
    #[error("Cross-validation error: {0}")]
    CVError(String),
}

/// Market impact parameter estimation using GMM
pub struct ImpactCalibration {
    pub window_size: usize,
    pub num_moments: usize,
    pub weighting_matrix: na::DMatrix<f64>,
    pub optimization_params: OptimizationParams,
    pub m: na::DVector<f64>,
    pub v: na::DVector<f64>,
    pub t: usize,
}

impl ImpactCalibration {
    pub fn new(
        window_size: usize,
        num_moments: usize,
        optimization_params: OptimizationParams,
    ) -> Result<Self, CalibrationError> {
        let weighting_matrix = na::DMatrix::identity(num_moments, num_moments);
        
        Ok(Self {
            window_size,
            num_moments,
            weighting_matrix,
            optimization_params,
            m: na::DVector::zeros(4),
            v: na::DVector::zeros(4),
            t: 0,
        })
    }

    pub fn estimate_impact_params(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> Result<ImpactParameters, CalibrationError> {
        // Compute empirical moments
        let empirical_moments = self.compute_empirical_moments(trades, prices)?;
        
        // Initialize parameters
        let mut params = self.initialize_parameters();
        
        // Iterative GMM estimation
        for iteration in 0..self.optimization_params.max_iterations {
            // Compute model moments
            let model_moments = self.compute_model_moments(&params, trades, prices)?;
            
            // Compute moment differences
            let moment_diff = &empirical_moments - &model_moments;
            
            // Compute GMM objective
            let obj_value = self.compute_gmm_objective(
                &moment_diff,
                &self.weighting_matrix,
            );
            
            // Check convergence
            if obj_value < self.optimization_params.tolerance {
                break;
            }
            
            // Update parameters
            params = self.update_parameters(
                &params,
                &moment_diff,
                &self.weighting_matrix,
            )?;
            
            // Update weighting matrix if needed
            if iteration == 0 {
                self.update_weighting_matrix(&moment_diff)?;
            }
        }
        
        Ok(params)
    }

    fn compute_empirical_moments(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> Result<na::DVector<f64>, CalibrationError> {
        let mut moments = na::DVector::zeros(self.num_moments);
        
        // Compute various moment conditions
        for window in trades.windows(self.window_size) {
            // Price impact moments
            moments[0] += self.compute_price_impact_moment(window, prices);
            
            // Volume-volatility relationship
            moments[1] += self.compute_volume_volatility_moment(window, prices);
            
            // Autocorrelation of price changes
            moments[2] += self.compute_autocorrelation_moment(window, prices);
            
            // Higher moments for non-linear effects
            moments[3] += self.compute_nonlinear_moment(window, prices);
        }
        
        // Normalize moments
        let n_windows = trades.len() - self.window_size + 1;
        moments.scale_mut(1.0 / n_windows as f64);
        
        Ok(moments)
    }

    fn compute_model_moments(
        &self,
        params: &ImpactParameters,
        trades: &[Trade],
        prices: &[f64],
    ) -> Result<na::DVector<f64>, CalibrationError> {
        let mut moments = na::DVector::zeros(self.num_moments);
        
        // Simulate model with given parameters
        let simulated_prices = self.simulate_price_impact(params, trades)?;
        
        // Compute model-implied moments
        for window in trades.windows(self.window_size) {
            moments[0] += self.compute_price_impact_moment(window, &simulated_prices);
            moments[1] += self.compute_volume_volatility_moment(window, &simulated_prices);
            moments[2] += self.compute_autocorrelation_moment(window, &simulated_prices);
            moments[3] += self.compute_nonlinear_moment(window, &simulated_prices);
        }
        
        let n_windows = trades.len() - self.window_size + 1;
        moments.scale_mut(1.0 / n_windows as f64);
        
        Ok(moments)
    }

    fn compute_price_impact_moment(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> f64 {
        let mut impact = 0.0;
        
        for i in 1..trades.len() {
            let volume = trades[i].volume;
            let direction = trades[i].direction as f64;
            let price_change = (prices[i] - prices[i-1]) / prices[i-1];
            
            // Non-linear price impact model
            let signed_volume = direction * volume.powf(0.6);
            let expected_impact = 0.1 * signed_volume; // Baseline impact
            
            // Deviation from expected impact
            impact += (price_change - expected_impact).powi(2);
        }
        
        impact / (trades.len() - 1) as f64
    }

    fn compute_volume_volatility_moment(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> f64 {
        let mut volume_volatility = 0.0;
        let window = 10;
        
        for i in window..trades.len() {
            // Compute realized volatility
            let returns: Vec<f64> = prices[i-window..i].windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
                
            let vol = returns.iter()
                .map(|r| r.powi(2))
                .sum::<f64>()
                .sqrt() * (252.0 * window as f64).sqrt();
                
            // Compute volume profile
            let volume = trades[i-window..i].iter()
                .map(|t| t.volume)
                .sum::<f64>();
                
            // Volume-volatility relationship
            let expected_vol = 0.2 * volume.powf(0.3); // Power law relationship
            volume_volatility += (vol - expected_vol).powi(2);
        }
        
        volume_volatility / (trades.len() - window) as f64
    }

    fn compute_autocorrelation_moment(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> f64 {
        let mut autocorr = 0.0;
        let lags = 5;
        
        // Compute returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
            
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let var_return = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
            
        // Compute autocorrelation at multiple lags
        for lag in 1..=lags {
            let mut lag_sum = 0.0;
            for i in lag..returns.len() {
                lag_sum += (returns[i] - mean_return) * (returns[i-lag] - mean_return);
            }
            
            let lag_autocorr = lag_sum / ((returns.len() - lag) as f64 * var_return);
            autocorr += lag_autocorr.powi(2);
        }
        
        autocorr / lags as f64
    }

    fn compute_nonlinear_moment(
        &self,
        trades: &[Trade],
        prices: &[f64],
    ) -> f64 {
        let mut nonlinear = 0.0;
        let window = 20;
        
        for i in window..trades.len() {
            // Compute signed volume
            let signed_volume = trades[i].volume * trades[i].direction as f64;
            
            // Compute price impact
            let price_change = (prices[i] - prices[i-1]) / prices[i-1];
            
            // Historical volume profile
            let hist_volume = trades[i-window..i].iter()
                .map(|t| t.volume)
                .sum::<f64>() / window as f64;
                
            // Volume ratio
            let volume_ratio = trades[i].volume / hist_volume;
            
            // Non-linear effects
            let expected_impact = if volume_ratio > 2.0 {
                // Large trades have higher impact
                0.2 * signed_volume.powf(0.7)
            } else {
                // Normal trades
                0.1 * signed_volume.powf(0.6)
            };
            
            nonlinear += (price_change - expected_impact).powi(2);
        }
        
        nonlinear / (trades.len() - window) as f64
    }

    fn update_parameters(
        &mut self,
        current_params: &ImpactParameters,
        moment_diff: &na::DVector<f64>,
        weighting_matrix: &na::DMatrix<f64>,
    ) -> Result<ImpactParameters, CalibrationError> {
        // Compute gradient using finite differences
        let epsilon = 1e-6;
        let mut gradient = na::DVector::zeros(4);
        
        // Permanent impact parameter
        let mut params_p = current_params.clone();
        params_p.permanent_impact += epsilon;
        let obj_p = self.compute_gmm_objective(moment_diff, weighting_matrix);
        gradient[0] = (obj_p - self.compute_gmm_objective(moment_diff, weighting_matrix)) / epsilon;
        
        // Temporary impact parameter
        let mut params_t = current_params.clone();
        params_t.temporary_impact += epsilon;
        let obj_t = self.compute_gmm_objective(moment_diff, weighting_matrix);
        gradient[1] = (obj_t - self.compute_gmm_objective(moment_diff, weighting_matrix)) / epsilon;
        
        // Decay rate parameter
        let mut params_d = current_params.clone();
        params_d.decay_rate += epsilon;
        let obj_d = self.compute_gmm_objective(moment_diff, weighting_matrix);
        gradient[2] = (obj_d - self.compute_gmm_objective(moment_diff, weighting_matrix)) / epsilon;
        
        // Non-linear factor parameter
        let mut params_n = current_params.clone();
        params_n.nonlinear_factor += epsilon;
        let obj_n = self.compute_gmm_objective(moment_diff, weighting_matrix);
        gradient[3] = (obj_n - self.compute_gmm_objective(moment_diff, weighting_matrix)) / epsilon;
        
        // Update parameters using Adam optimizer
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon_adam = 1e-8;
        
        // First moment estimate
        self.m = beta1 * self.m + (1.0 - beta1) * gradient;
        
        // Second moment estimate
        self.v = beta2 * self.v + (1.0 - beta2) * gradient.component_mul(&gradient);
        
        // Bias correction
        let m_hat = self.m / (1.0 - beta1.powi(self.t as i32));
        let v_hat = self.v / (1.0 - beta2.powi(self.t as i32));
        
        // Update parameters
        let step_size = self.optimization_params.learning_rate;
        let mut new_params = current_params.clone();
        
        new_params.permanent_impact -= step_size * m_hat[0] / (v_hat[0].sqrt() + epsilon_adam);
        new_params.temporary_impact -= step_size * m_hat[1] / (v_hat[1].sqrt() + epsilon_adam);
        new_params.decay_rate -= step_size * m_hat[2] / (v_hat[2].sqrt() + epsilon_adam);
        new_params.nonlinear_factor -= step_size * m_hat[3] / (v_hat[3].sqrt() + epsilon_adam);
        
        // Apply constraints
        new_params.permanent_impact = new_params.permanent_impact.max(0.0);
        new_params.temporary_impact = new_params.temporary_impact.max(0.0);
        new_params.decay_rate = new_params.decay_rate.clamp(0.0, 1.0);
        new_params.nonlinear_factor = new_params.nonlinear_factor.max(0.0);
        
        self.t += 1;
        
        Ok(new_params)
    }

    fn simulate_price_impact(
        &self,
        params: &ImpactParameters,
        trades: &[Trade],
    ) -> Result<Vec<f64>, CalibrationError> {
        let mut prices = vec![100.0]; // Initial price
        let mut temporary_impact = 0.0;
        
        for (i, trade) in trades.iter().enumerate() {
            let signed_volume = trade.volume * trade.direction as f64;
            
            // Permanent impact
            let permanent = params.permanent_impact * signed_volume.powf(params.nonlinear_factor);
            
            // Temporary impact with decay
            temporary_impact *= (-params.decay_rate * trade.timestamp).exp();
            temporary_impact += params.temporary_impact * signed_volume.powf(params.nonlinear_factor);
            
            // Total price impact
            let total_impact = permanent + temporary_impact;
            let new_price = prices[i] * (1.0 + total_impact);
            prices.push(new_price);
        }
        
        Ok(prices)
    }
}

impl RoughVolatilityCalibration {
    fn compute_volatility_factors(
        &self,
        returns: &[f64],
        params: &RoughVolatilityParams,
    ) -> Result<Vec<Vec<f64>>, CalibrationError> {
        let mut factors = vec![vec![0.0; returns.len()]; self.num_factors];
        
        // Initialize factors using principal component analysis
        let returns_matrix = na::DMatrix::from_row_slice(
            returns.len(),
            1,
            returns,
        );
        
        let svd = returns_matrix.svd(true, true);
        let singular_values = svd.singular_values;
        let left_singular_vectors = svd.u.unwrap();
        
        // Extract principal components
        for i in 0..self.num_factors.min(singular_values.len()) {
            let factor = left_singular_vectors.column(i).iter()
                .map(|&x| x * singular_values[i])
                .collect::<Vec<f64>>();
                
            factors[i] = factor;
        }
        
        // Apply rough volatility transformation
        for factor in factors.iter_mut() {
            let mut rough_factor = vec![0.0; factor.len()];
            
            for i in 1..factor.len() {
                let dt = 1.0 / 252.0; // Daily data
                let h = params.hurst_parameter;
                
                // Fractional Brownian motion increment
                let increment = factor[i] - factor[i-1];
                rough_factor[i] = rough_factor[i-1] + 
                    increment * dt.powf(h);
            }
            
            *factor = rough_factor;
        }
        
        Ok(factors)
    }

    fn compute_window_likelihood(
        &self,
        returns: &[f64],
        factors: &[Vec<f64>],
        params: &RoughVolatilityParams,
    ) -> Result<f64, CalibrationError> {
        let mut log_likelihood = 0.0;
        
        // Compute volatility process
        let mut volatility = vec![params.base_volatility; returns.len()];
        
        for t in 1..returns.len() {
            let mut factor_sum = 0.0;
            
            // Combine factors
            for (i, factor) in factors.iter().enumerate() {
                factor_sum += params.factor_loadings[i] * factor[t];
            }
            
            // Update volatility with mean reversion
            volatility[t] = params.base_volatility + 
                params.mean_reversion * (volatility[t-1] - params.base_volatility) +
                factor_sum;
        }
        
        // Compute likelihood
        for t in 0..returns.len() {
            let standardized_return = returns[t] / volatility[t];
            
            // Student-t likelihood for fat tails
            let dof = params.degrees_freedom;
            let log_density = (dof + 1.0) / 2.0 * 
                ((1.0 + standardized_return.powi(2) / dof).ln()) -
                0.5 * (2.0 * std::f64::consts::PI * volatility[t].powi(2)).ln();
                
            log_likelihood += log_density;
        }
        
        Ok(log_likelihood)
    }

    fn update_volatility_params(
        &self,
        current_params: &RoughVolatilityParams,
        returns: &[f64],
        likelihood: f64,
    ) -> Result<RoughVolatilityParams, CalibrationError> {
        // Compute gradient using automatic differentiation
        let epsilon = 1e-6;
        let mut gradient = na::DVector::zeros(4 + self.num_factors);
        
        // Base volatility
        let mut params_v = current_params.clone();
        params_v.base_volatility += epsilon;
        let like_v = self.compute_likelihood(returns, &params_v)?;
        gradient[0] = (like_v - likelihood) / epsilon;
        
        // Hurst parameter
        let mut params_h = current_params.clone();
        params_h.hurst_parameter += epsilon;
        let like_h = self.compute_likelihood(returns, &params_h)?;
        gradient[1] = (like_h - likelihood) / epsilon;
        
        // Mean reversion
        let mut params_m = current_params.clone();
        params_m.mean_reversion += epsilon;
        let like_m = self.compute_likelihood(returns, &params_m)?;
        gradient[2] = (like_m - likelihood) / epsilon;
        
        // Degrees of freedom
        let mut params_d = current_params.clone();
        params_d.degrees_freedom += epsilon;
        let like_d = self.compute_likelihood(returns, &params_d)?;
        gradient[3] = (like_d - likelihood) / epsilon;
        
        // Factor loadings
        for i in 0..self.num_factors {
            let mut params_f = current_params.clone();
            params_f.factor_loadings[i] += epsilon;
            let like_f = self.compute_likelihood(returns, &params_f)?;
            gradient[4 + i] = (like_f - likelihood) / epsilon;
        }
        
        // Update parameters using natural gradient
        let fisher_info = self.compute_fisher_information(
            returns,
            current_params,
            &gradient,
        )?;
        
        let natural_gradient = fisher_info.try_inverse()
            .ok_or(CalibrationError::MLError(
                "Failed to compute Fisher information inverse".to_string(),
            ))? * gradient;
            
        // Update parameters with constraints
        let step_size = self.optimization_params.learning_rate;
        let mut new_params = current_params.clone();
        
        new_params.base_volatility = (current_params.base_volatility + 
            step_size * natural_gradient[0]).max(0.0);
            
        new_params.hurst_parameter = (current_params.hurst_parameter +
            step_size * natural_gradient[1]).clamp(0.0, 0.5);
            
        new_params.mean_reversion = (current_params.mean_reversion +
            step_size * natural_gradient[2]).clamp(0.0, 1.0);
            
        new_params.degrees_freedom = (current_params.degrees_freedom +
            step_size * natural_gradient[3]).max(2.1);
            
        for i in 0..self.num_factors {
            new_params.factor_loadings[i] = current_params.factor_loadings[i] +
                step_size * natural_gradient[4 + i];
        }
        
        Ok(new_params)
    }

    fn compute_fisher_information(
        &self,
        returns: &[f64],
        params: &RoughVolatilityParams,
        gradient: &na::DVector<f64>,
    ) -> Result<na::DMatrix<f64>, CalibrationError> {
        let n = 4 + self.num_factors;
        let mut fisher = na::DMatrix::zeros(n, n);
        
        // Compute outer product of score function
        for t in 0..returns.len() {
            let score = self.compute_score(returns[t], params)?;
            fisher += &(score * score.transpose());
        }
        
        fisher *= 1.0 / returns.len() as f64;
        
        // Add regularization for numerical stability
        for i in 0..n {
            fisher[(i, i)] += 1e-6;
        }
        
        Ok(fisher)
    }

    fn compute_score(
        &self,
        return_value: f64,
        params: &RoughVolatilityParams,
    ) -> Result<na::DVector<f64>, CalibrationError> {
        let n = 4 + self.num_factors;
        let mut score = na::DVector::zeros(n);
        
        // Compute score function components
        let vol = params.base_volatility;
        let std_return = return_value / vol;
        
        // Score for base volatility
        score[0] = -1.0 / vol + std_return.powi(2) / vol;
        
        // Score for Hurst parameter
        score[1] = std_return.powi(2) * params.hurst_parameter.ln();
        
        // Score for mean reversion
        score[2] = std_return * (1.0 - params.mean_reversion);
        
        // Score for degrees of freedom
        let dof = params.degrees_freedom;
        score[3] = 0.5 * (dof.digamma() - (1.0 + std_return.powi(2) / dof).ln());
        
        // Score for factor loadings
        for i in 0..self.num_factors {
            score[4 + i] = std_return * params.factor_loadings[i];
        }
        
        Ok(score)
    }
}

/// Cross-validation framework
pub struct CrossValidation {
    pub num_folds: usize,
    pub validation_metrics: Vec<Box<dyn ValidationMetric>>,
}

impl CrossValidation {
    pub fn new(num_folds: usize) -> Self {
        Self {
            num_folds,
            validation_metrics: Vec::new(),
        }
    }

    pub fn add_metric(&mut self, metric: Box<dyn ValidationMetric>) {
        self.validation_metrics.push(metric);
    }

    pub fn cross_validate<T>(
        &self,
        data: &[T],
        model: &mut dyn Model<T>,
    ) -> Result<ValidationResults, CalibrationError> {
        let fold_size = data.len() / self.num_folds;
        let mut results = ValidationResults::new(self.validation_metrics.len());
        
        for fold in 0..self.num_folds {
            // Split data into training and validation sets
            let val_start = fold * fold_size;
            let val_end = val_start + fold_size;
            
            let validation_data = &data[val_start..val_end];
            let mut training_data = Vec::new();
            training_data.extend_from_slice(&data[0..val_start]);
            training_data.extend_from_slice(&data[val_end..]);
            
            // Train model
            model.train(&training_data)?;
            
            // Compute validation metrics
            for (i, metric) in self.validation_metrics.iter().enumerate() {
                let score = metric.compute(model, validation_data)?;
                results.add_score(i, score);
            }
        }
        
        Ok(results)
    }
}

pub trait Model<T> {
    fn train(&mut self, data: &[T]) -> Result<(), CalibrationError>;
    fn predict(&self, input: &T) -> Result<f64, CalibrationError>;
}

pub trait ValidationMetric {
    fn compute<T>(
        &self,
        model: &dyn Model<T>,
        data: &[T],
    ) -> Result<f64, CalibrationError>;
}

#[derive(Debug)]
pub struct ValidationResults {
    pub scores: Vec<Vec<f64>>,
}

impl ValidationResults {
    pub fn new(num_metrics: usize) -> Self {
        Self {
            scores: vec![Vec::new(); num_metrics],
        }
    }

    pub fn add_score(&mut self, metric_idx: usize, score: f64) {
        self.scores[metric_idx].push(score);
    }

    pub fn mean_scores(&self) -> Vec<f64> {
        self.scores.iter()
            .map(|metric_scores| {
                metric_scores.iter().sum::<f64>() / metric_scores.len() as f64
            })
            .collect()
    }

    pub fn std_scores(&self) -> Vec<f64> {
        self.scores.iter()
            .map(|metric_scores| {
                let mean = metric_scores.iter().sum::<f64>() / metric_scores.len() as f64;
                let variance = metric_scores.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / metric_scores.len() as f64;
                variance.sqrt()
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationParams {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub learning_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ImpactParameters {
    pub permanent_impact: f64,
    pub temporary_impact: f64,
    pub decay_rate: f64,
    pub nonlinear_factor: f64,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub volume: f64,
    pub direction: i8,
    pub timestamp: f64,
}
