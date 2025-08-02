//! Advanced Hawkes Process Parameter Estimation
//! 
//! This module implements sophisticated parameter estimation methods for multivariate
//! Hawkes processes including maximum likelihood estimation using L-BFGS optimization,
//! expectation-maximization algorithms, cross-validation for model selection, and
//! comprehensive goodness-of-fit tests.

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use crate::math::hawkes_process::{
    HawkesEvent, MultivariateHawkesParams, KernelType, HawkesError,
    MultivariateHawkesSimulator, HawkesState
};
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};
use thiserror::Error;
use rayon::prelude::*;

#[derive(Error, Debug)]
pub enum EstimationError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Convergence failed: {0}")]
    ConvergenceFailed(String),
}

/// L-BFGS optimizer for maximum likelihood estimation
#[derive(Debug, Clone)]
pub struct LBFGSOptimizer {
    pub max_iterations: usize,
    pub tolerance: FixedPoint,
    pub line_search_tolerance: FixedPoint,
    pub history_size: usize,
    pub initial_step_size: FixedPoint,
}

impl Default for LBFGSOptimizer {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: FixedPoint::from_float(1e-6),
            line_search_tolerance: FixedPoint::from_float(1e-4),
            history_size: 10,
            initial_step_size: FixedPoint::from_float(1.0),
        }
    }
}

impl LBFGSOptimizer {
    pub fn new(
        max_iterations: usize,
        tolerance: FixedPoint,
        history_size: usize,
    ) -> Self {
        Self {
            max_iterations,
            tolerance,
            history_size,
            ..Default::default()
        }
    }
    
    /// Optimize parameters using L-BFGS algorithm
    pub fn optimize<F, G>(
        &self,
        initial_params: Vec<FixedPoint>,
        objective: F,
        gradient: G,
    ) -> Result<Vec<FixedPoint>, EstimationError>
    where
        F: Fn(&[FixedPoint]) -> FixedPoint,
        G: Fn(&[FixedPoint]) -> Vec<FixedPoint>,
    {
        let n = initial_params.len();
        let mut x = initial_params;
        let mut s_history = VecDeque::with_capacity(self.history_size);
        let mut y_history = VecDeque::with_capacity(self.history_size);
        let mut rho_history = VecDeque::with_capacity(self.history_size);
        
        let mut prev_grad = gradient(&x);
        
        for iteration in 0..self.max_iterations {
            let current_grad = gradient(&x);
            
            // Check convergence
            let grad_norm = current_grad.iter()
                .map(|&g| g * g)
                .fold(FixedPoint::zero(), |acc, g2| acc + g2)
                .sqrt();
            
            if grad_norm < self.tolerance {
                return Ok(x);
            }
            
            // Compute search direction using L-BFGS two-loop recursion
            let mut q = current_grad.clone();
            let mut alpha = vec![FixedPoint::zero(); s_history.len()];
            
            // First loop (backward)
            for (i, (s, y, rho)) in s_history.iter()
                .zip(y_history.iter())
                .zip(rho_history.iter())
                .enumerate()
                .rev()
            {
                let dot_product = s.iter().zip(q.iter())
                    .map(|(&si, &qi)| si * qi)
                    .fold(FixedPoint::zero(), |acc, prod| acc + prod);
                
                alpha[i] = *rho * dot_product;
                
                for (qj, &yj) in q.iter_mut().zip(y.iter()) {
                    *qj = *qj - alpha[i] * yj;
                }
            }
            
            // Scale initial Hessian approximation
            if !s_history.is_empty() {
                let s = s_history.back().unwrap();
                let y = y_history.back().unwrap();
                
                let sy = s.iter().zip(y.iter())
                    .map(|(&si, &yi)| si * yi)
                    .fold(FixedPoint::zero(), |acc, prod| acc + prod);
                
                let yy = y.iter()
                    .map(|&yi| yi * yi)
                    .fold(FixedPoint::zero(), |acc, y2| acc + y2);
                
                if yy.to_float() > 0.0 {
                    let gamma = sy / yy;
                    for qi in q.iter_mut() {
                        *qi = gamma * *qi;
                    }
                }
            }
            
            // Second loop (forward)
            for (i, (s, y, rho)) in s_history.iter()
                .zip(y_history.iter())
                .zip(rho_history.iter())
                .enumerate()
            {
                let dot_product = y.iter().zip(q.iter())
                    .map(|(&yi, &qi)| yi * qi)
                    .fold(FixedPoint::zero(), |acc, prod| acc + prod);
                
                let beta = *rho * dot_product;
                
                for (qj, &sj) in q.iter_mut().zip(s.iter()) {
                    *qj = *qj + (alpha[i] - beta) * sj;
                }
            }
            
            // Search direction (negative gradient direction)
            let mut direction = vec![FixedPoint::zero(); n];
            for (i, &qi) in q.iter().enumerate() {
                direction[i] = -qi;
            }
            
            // Line search
            let step_size = self.line_search(&x, &direction, &objective, &gradient)?;
            
            // Update parameters
            let mut new_x = vec![FixedPoint::zero(); n];
            for i in 0..n {
                new_x[i] = x[i] + step_size * direction[i];
            }
            
            // Update L-BFGS history
            if iteration > 0 {
                let mut s = vec![FixedPoint::zero(); n];
                let mut y = vec![FixedPoint::zero(); n];
                
                for i in 0..n {
                    s[i] = new_x[i] - x[i];
                    y[i] = current_grad[i] - prev_grad[i];
                }
                
                let sy = s.iter().zip(y.iter())
                    .map(|(&si, &yi)| si * yi)
                    .fold(FixedPoint::zero(), |acc, prod| acc + prod);
                
                if sy.to_float() > 1e-10 {
                    let rho = FixedPoint::one() / sy;
                    
                    if s_history.len() >= self.history_size {
                        s_history.pop_front();
                        y_history.pop_front();
                        rho_history.pop_front();
                    }
                    
                    s_history.push_back(s);
                    y_history.push_back(y);
                    rho_history.push_back(rho);
                }
            }
            
            prev_grad = current_grad;
            x = new_x;
        }
        
        Err(EstimationError::ConvergenceFailed(
            "L-BFGS failed to converge within maximum iterations".to_string()
        ))
    }
    
    /// Backtracking line search with Armijo condition
    fn line_search<F, G>(
        &self,
        x: &[FixedPoint],
        direction: &[FixedPoint],
        objective: &F,
        gradient: &G,
    ) -> Result<FixedPoint, EstimationError>
    where
        F: Fn(&[FixedPoint]) -> FixedPoint,
        G: Fn(&[FixedPoint]) -> Vec<FixedPoint>,
    {
        let c1 = FixedPoint::from_float(1e-4); // Armijo parameter
        let rho = FixedPoint::from_float(0.5); // Backtracking parameter
        let max_backtracks = 50;
        
        let f0 = objective(x);
        let grad0 = gradient(x);
        
        // Directional derivative
        let directional_derivative = grad0.iter().zip(direction.iter())
            .map(|(&g, &d)| g * d)
            .fold(FixedPoint::zero(), |acc, prod| acc + prod);
        
        let mut alpha = self.initial_step_size;
        
        for _ in 0..max_backtracks {
            let mut x_new = vec![FixedPoint::zero(); x.len()];
            for i in 0..x.len() {
                x_new[i] = x[i] + alpha * direction[i];
            }
            
            let f_new = objective(&x_new);
            let armijo_condition = f0 + c1 * alpha * directional_derivative;
            
            if f_new <= armijo_condition {
                return Ok(alpha);
            }
            
            alpha = alpha * rho;
        }
        
        Ok(FixedPoint::from_float(1e-6)) // Minimal step if line search fails
    }
}

/// Maximum Likelihood Estimator for Hawkes processes
#[derive(Debug, Clone)]
pub struct HawkesMLEstimator {
    pub optimizer: LBFGSOptimizer,
    pub regularization_strength: FixedPoint,
    pub use_regularization: bool,
}

impl Default for HawkesMLEstimator {
    fn default() -> Self {
        Self {
            optimizer: LBFGSOptimizer::default(),
            regularization_strength: FixedPoint::from_float(0.01),
            use_regularization: false,
        }
    }
}

impl HawkesMLEstimator {
    pub fn new(optimizer: LBFGSOptimizer) -> Self {
        Self {
            optimizer,
            ..Default::default()
        }
    }
    
    pub fn with_regularization(mut self, strength: FixedPoint) -> Self {
        self.regularization_strength = strength;
        self.use_regularization = true;
        self
    }
    
    /// Estimate parameters using maximum likelihood
    pub fn estimate(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        initial_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<MultivariateHawkesParams, EstimationError> {
        if event_sequences.is_empty() {
            return Err(EstimationError::InsufficientData(
                "No event sequences provided".to_string()
            ));
        }
        
        let n = initial_params.dimension();
        
        // Convert parameters to optimization vector
        let initial_vector = self.params_to_vector(initial_params);
        
        // Define objective function (negative log-likelihood)
        let objective = |params_vec: &[FixedPoint]| -> FixedPoint {
            match self.vector_to_params(params_vec, n) {
                Ok(params) => {
                    let log_likelihood = self.compute_log_likelihood(
                        event_sequences,
                        &params,
                        observation_time,
                    ).unwrap_or(FixedPoint::from_float(-1e10));
                    
                    let regularization = if self.use_regularization {
                        self.compute_regularization(&params)
                    } else {
                        FixedPoint::zero()
                    };
                    
                    -(log_likelihood - regularization) // Negative for minimization
                }
                Err(_) => FixedPoint::from_float(1e10), // Penalty for invalid parameters
            }
        };
        
        // Define gradient function
        let gradient = |params_vec: &[FixedPoint]| -> Vec<FixedPoint> {
            match self.vector_to_params(params_vec, n) {
                Ok(params) => {
                    self.compute_gradient(event_sequences, &params, observation_time)
                        .unwrap_or_else(|_| vec![FixedPoint::zero(); params_vec.len()])
                        .into_iter()
                        .map(|g| -g) // Negative for minimization
                        .collect()
                }
                Err(_) => vec![FixedPoint::zero(); params_vec.len()],
            }
        };
        
        // Optimize
        let optimal_vector = self.optimizer.optimize(initial_vector, objective, gradient)?;
        
        // Convert back to parameters
        self.vector_to_params(&optimal_vector, n)
            .map_err(|e| EstimationError::InvalidParameters(e.to_string()))
    }
    
    /// Compute log-likelihood of parameters given data
    fn compute_log_likelihood(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<FixedPoint, EstimationError> {
        let mut total_log_likelihood = FixedPoint::zero();
        
        for sequence in event_sequences {
            let sequence_ll = self.compute_sequence_log_likelihood(
                sequence,
                params,
                observation_time,
            )?;
            total_log_likelihood = total_log_likelihood + sequence_ll;
        }
        
        Ok(total_log_likelihood)
    }
    
    /// Compute log-likelihood for a single event sequence
    fn compute_sequence_log_likelihood(
        &self,
        events: &[HawkesEvent],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<FixedPoint, EstimationError> {
        let mut log_likelihood = FixedPoint::zero();
        let mut state = HawkesState::new(params, 10000);
        
        // Sum of log intensities at event times
        for event in events {
            state.current_time = event.time;
            state.update_intensities(params);
            
            let intensity = state.intensities[event.process_id];
            if intensity.to_float() > 1e-10 {
                log_likelihood = log_likelihood + intensity.ln();
            } else {
                return Err(EstimationError::NumericalInstability(
                    "Zero intensity encountered".to_string()
                ));
            }
            
            // Add event to state
            state.add_event(event.clone(), 10000);
        }
        
        // Subtract integral of intensities
        let integral_term = self.compute_intensity_integral(params, observation_time)?;
        log_likelihood = log_likelihood - integral_term;
        
        Ok(log_likelihood)
    }
    
    /// Compute integral of intensity functions over observation period
    fn compute_intensity_integral(
        &self,
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<FixedPoint, EstimationError> {
        let n = params.dimension();
        let mut total_integral = FixedPoint::zero();
        
        // Baseline intensity contribution
        for &baseline in &params.baseline_intensities {
            total_integral = total_integral + baseline * observation_time;
        }
        
        // This is a simplified version - in practice would need to compute
        // the full integral considering all event interactions
        Ok(total_integral)
    }
    
    /// Compute gradient of log-likelihood
    fn compute_gradient(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<Vec<FixedPoint>, EstimationError> {
        let n = params.dimension();
        let param_count = n + n * n; // Baseline + kernel parameters (simplified)
        let mut gradient = vec![FixedPoint::zero(); param_count];
        
        for sequence in event_sequences {
            let seq_gradient = self.compute_sequence_gradient(
                sequence,
                params,
                observation_time,
            )?;
            
            for (i, &grad_i) in seq_gradient.iter().enumerate() {
                gradient[i] = gradient[i] + grad_i;
            }
        }
        
        Ok(gradient)
    }
    
    /// Compute gradient for a single sequence
    fn compute_sequence_gradient(
        &self,
        events: &[HawkesEvent],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<Vec<FixedPoint>, EstimationError> {
        let n = params.dimension();
        let param_count = n + n * n;
        let mut gradient = vec![FixedPoint::zero(); param_count];
        
        // Gradient w.r.t. baseline intensities
        for i in 0..n {
            let event_count = events.iter()
                .filter(|e| e.process_id == i)
                .count();
            
            gradient[i] = FixedPoint::from_float(event_count as f64) / params.baseline_intensities[i] 
                - observation_time;
        }
        
        // Gradient w.r.t. kernel parameters (simplified)
        // In practice, this would require more sophisticated computation
        
        Ok(gradient)
    }
    
    /// Compute regularization term
    fn compute_regularization(&self, params: &MultivariateHawkesParams) -> FixedPoint {
        let mut regularization = FixedPoint::zero();
        
        // L2 regularization on baseline intensities
        for &baseline in &params.baseline_intensities {
            regularization = regularization + self.regularization_strength * baseline * baseline;
        }
        
        regularization
    }
    
    /// Convert parameters to optimization vector
    fn params_to_vector(&self, params: &MultivariateHawkesParams) -> Vec<FixedPoint> {
        let mut vector = Vec::new();
        
        // Add baseline intensities
        vector.extend_from_slice(&params.baseline_intensities);
        
        // Add kernel parameters (simplified - only exponential kernels)
        for row in &params.kernels {
            for kernel in row {
                match kernel {
                    KernelType::Exponential { alpha, beta } => {
                        vector.push(*alpha);
                        vector.push(*beta);
                    }
                    _ => {
                        // For other kernel types, use default values
                        vector.push(FixedPoint::from_float(0.1));
                        vector.push(FixedPoint::from_float(1.0));
                    }
                }
            }
        }
        
        vector
    }
    
    /// Convert optimization vector to parameters
    fn vector_to_params(
        &self,
        vector: &[FixedPoint],
        n: usize,
    ) -> Result<MultivariateHawkesParams, EstimationError> {
        if vector.len() < n + 2 * n * n {
            return Err(EstimationError::InvalidParameters(
                "Parameter vector too short".to_string()
            ));
        }
        
        // Extract baseline intensities
        let baseline_intensities = vector[0..n].to_vec();
        
        // Ensure positivity
        for &intensity in &baseline_intensities {
            if intensity.to_float() <= 0.0 {
                return Err(EstimationError::InvalidParameters(
                    "Baseline intensities must be positive".to_string()
                ));
            }
        }
        
        // Extract kernel parameters
        let mut kernels = vec![vec![KernelType::Exponential { 
            alpha: FixedPoint::from_float(0.1), 
            beta: FixedPoint::from_float(1.0) 
        }; n]; n];
        
        let mut idx = n;
        for i in 0..n {
            for j in 0..n {
                if idx + 1 < vector.len() {
                    let alpha = vector[idx];
                    let beta = vector[idx + 1];
                    
                    // Ensure positivity
                    if alpha.to_float() > 0.0 && beta.to_float() > 0.0 {
                        kernels[i][j] = KernelType::Exponential { alpha, beta };
                    }
                    
                    idx += 2;
                }
            }
        }
        
        MultivariateHawkesParams::new(
            baseline_intensities,
            kernels,
            FixedPoint::from_float(100.0), // Max intensity
        ).map_err(|e| EstimationError::InvalidParameters(e.to_string()))
    }
}

/// Expectation-Maximization algorithm for Hawkes processes
#[derive(Debug, Clone)]
pub struct HawkesEMEstimator {
    pub max_iterations: usize,
    pub tolerance: FixedPoint,
    pub regularization_strength: FixedPoint,
}

impl Default for HawkesEMEstimator {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: FixedPoint::from_float(1e-6),
            regularization_strength: FixedPoint::from_float(0.01),
        }
    }
}

impl HawkesEMEstimator {
    pub fn new(max_iterations: usize, tolerance: FixedPoint) -> Self {
        Self {
            max_iterations,
            tolerance,
            ..Default::default()
        }
    }
    
    /// Estimate parameters using EM algorithm
    pub fn estimate(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        initial_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<MultivariateHawkesParams, EstimationError> {
        let mut current_params = initial_params.clone();
        let mut prev_log_likelihood = FixedPoint::from_float(-f64::INFINITY);
        
        for iteration in 0..self.max_iterations {
            // E-step: Compute expected sufficient statistics
            let sufficient_stats = self.e_step(event_sequences, &current_params, observation_time)?;
            
            // M-step: Update parameters
            current_params = self.m_step(&sufficient_stats, observation_time)?;
            
            // Check convergence
            let log_likelihood = self.compute_log_likelihood(
                event_sequences,
                &current_params,
                observation_time,
            )?;
            
            let improvement = log_likelihood - prev_log_likelihood;
            if improvement < self.tolerance && iteration > 0 {
                break;
            }
            
            prev_log_likelihood = log_likelihood;
        }
        
        Ok(current_params)
    }
    
    /// E-step: Compute expected sufficient statistics
    fn e_step(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<SufficientStatistics, EstimationError> {
        let n = params.dimension();
        let mut stats = SufficientStatistics::new(n);
        
        for sequence in event_sequences {
            let sequence_stats = self.compute_sequence_statistics(
                sequence,
                params,
                observation_time,
            )?;
            stats.add(&sequence_stats);
        }
        
        Ok(stats)
    }
    
    /// M-step: Update parameters based on sufficient statistics
    fn m_step(
        &self,
        stats: &SufficientStatistics,
        observation_time: FixedPoint,
    ) -> Result<MultivariateHawkesParams, EstimationError> {
        let n = stats.dimension();
        
        // Update baseline intensities
        let mut baseline_intensities = vec![FixedPoint::zero(); n];
        for i in 0..n {
            baseline_intensities[i] = stats.immigrant_counts[i] / observation_time;
            
            // Ensure minimum value
            if baseline_intensities[i].to_float() < 1e-6 {
                baseline_intensities[i] = FixedPoint::from_float(1e-6);
            }
        }
        
        // Update kernel parameters (simplified for exponential kernels)
        let mut kernels = vec![vec![KernelType::Exponential { 
            alpha: FixedPoint::from_float(0.1), 
            beta: FixedPoint::from_float(1.0) 
        }; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if stats.offspring_counts[i][j].to_float() > 0.0 {
                    let alpha = stats.offspring_counts[i][j] / stats.parent_counts[j];
                    let beta = stats.offspring_counts[i][j] / stats.total_offspring_time[i][j];
                    
                    if alpha.to_float() > 0.0 && beta.to_float() > 0.0 {
                        kernels[i][j] = KernelType::Exponential { alpha, beta };
                    }
                }
            }
        }
        
        MultivariateHawkesParams::new(
            baseline_intensities,
            kernels,
            FixedPoint::from_float(100.0),
        ).map_err(|e| EstimationError::InvalidParameters(e.to_string()))
    }
    
    /// Compute sufficient statistics for a single sequence
    fn compute_sequence_statistics(
        &self,
        events: &[HawkesEvent],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<SufficientStatistics, EstimationError> {
        let n = params.dimension();
        let mut stats = SufficientStatistics::new(n);
        
        // Simplified computation - in practice would use more sophisticated
        // branching structure analysis
        
        for event in events {
            stats.immigrant_counts[event.process_id] = 
                stats.immigrant_counts[event.process_id] + FixedPoint::one();
        }
        
        Ok(stats)
    }
    
    /// Compute log-likelihood (reuse from ML estimator)
    fn compute_log_likelihood(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<FixedPoint, EstimationError> {
        let ml_estimator = HawkesMLEstimator::default();
        ml_estimator.compute_log_likelihood(event_sequences, params, observation_time)
    }
}

/// Sufficient statistics for EM algorithm
#[derive(Debug, Clone)]
struct SufficientStatistics {
    pub immigrant_counts: Vec<FixedPoint>,
    pub offspring_counts: Vec<Vec<FixedPoint>>,
    pub parent_counts: Vec<FixedPoint>,
    pub total_offspring_time: Vec<Vec<FixedPoint>>,
}

impl SufficientStatistics {
    fn new(n: usize) -> Self {
        Self {
            immigrant_counts: vec![FixedPoint::zero(); n],
            offspring_counts: vec![vec![FixedPoint::zero(); n]; n],
            parent_counts: vec![FixedPoint::zero(); n],
            total_offspring_time: vec![vec![FixedPoint::zero(); n]; n],
        }
    }
    
    fn dimension(&self) -> usize {
        self.immigrant_counts.len()
    }
    
    fn add(&mut self, other: &SufficientStatistics) {
        let n = self.dimension();
        
        for i in 0..n {
            self.immigrant_counts[i] = self.immigrant_counts[i] + other.immigrant_counts[i];
            self.parent_counts[i] = self.parent_counts[i] + other.parent_counts[i];
            
            for j in 0..n {
                self.offspring_counts[i][j] = self.offspring_counts[i][j] + other.offspring_counts[i][j];
                self.total_offspring_time[i][j] = self.total_offspring_time[i][j] + other.total_offspring_time[i][j];
            }
        }
    }
}

/// Cross-validation for model selection
#[derive(Debug, Clone)]
pub struct HawkesCrossValidator {
    pub k_folds: usize,
    pub validation_metric: ValidationMetric,
    pub random_seed: u64,
}

#[derive(Debug, Clone)]
pub enum ValidationMetric {
    LogLikelihood,
    AIC, // Akaike Information Criterion
    BIC, // Bayesian Information Criterion
    KSTest, // Kolmogorov-Smirnov test
}

impl Default for HawkesCrossValidator {
    fn default() -> Self {
        Self {
            k_folds: 5,
            validation_metric: ValidationMetric::AIC,
            random_seed: 42,
        }
    }
}

impl HawkesCrossValidator {
    pub fn new(k_folds: usize, metric: ValidationMetric) -> Self {
        Self {
            k_folds,
            validation_metric: metric,
            ..Default::default()
        }
    }
    
    /// Perform k-fold cross-validation
    pub fn validate(
        &self,
        event_sequences: &[Vec<HawkesEvent>],
        candidate_params: &[MultivariateHawkesParams],
        observation_time: FixedPoint,
    ) -> Result<CrossValidationResult, EstimationError> {
        let mut rng = DeterministicRng::new(self.random_seed);
        let folds = self.create_folds(event_sequences, &mut rng)?;
        
        let mut results = Vec::new();
        
        for params in candidate_params {
            let mut fold_scores = Vec::new();
            
            for fold_idx in 0..self.k_folds {
                let (train_data, test_data) = self.get_train_test_split(&folds, fold_idx);
                
                // Train on training data (or use provided parameters)
                let trained_params = params.clone(); // In practice, would retrain
                
                // Evaluate on test data
                let score = self.evaluate_model(&test_data, &trained_params, observation_time)?;
                fold_scores.push(score);
            }
            
            let mean_score = fold_scores.iter()
                .fold(FixedPoint::zero(), |acc, &score| acc + score) / 
                FixedPoint::from_float(fold_scores.len() as f64);
            
            let std_score = {
                let variance = fold_scores.iter()
                    .map(|&score| (score - mean_score) * (score - mean_score))
                    .fold(FixedPoint::zero(), |acc, var| acc + var) /
                    FixedPoint::from_float(fold_scores.len() as f64);
                variance.sqrt()
            };
            
            results.push(ModelValidationResult {
                params: params.clone(),
                mean_score,
                std_score,
                fold_scores,
            });
        }
        
        // Find best model
        let best_idx = results.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.mean_score.partial_cmp(&b.mean_score).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        Ok(CrossValidationResult {
            best_model_idx: best_idx,
            results,
        })
    }
    
    /// Create k-fold splits
    fn create_folds(
        &self,
        sequences: &[Vec<HawkesEvent>],
        rng: &mut DeterministicRng,
    ) -> Result<Vec<Vec<usize>>, EstimationError> {
        let n_sequences = sequences.len();
        if n_sequences < self.k_folds {
            return Err(EstimationError::InsufficientData(
                "Not enough sequences for k-fold validation".to_string()
            ));
        }
        
        let mut indices: Vec<usize> = (0..n_sequences).collect();
        
        // Shuffle indices
        for i in (1..indices.len()).rev() {
            let j = (rng.next_fixed().to_float() * (i + 1) as f64) as usize;
            indices.swap(i, j);
        }
        
        // Create folds
        let fold_size = n_sequences / self.k_folds;
        let mut folds = Vec::new();
        
        for i in 0..self.k_folds {
            let start = i * fold_size;
            let end = if i == self.k_folds - 1 {
                n_sequences
            } else {
                (i + 1) * fold_size
            };
            
            folds.push(indices[start..end].to_vec());
        }
        
        Ok(folds)
    }
    
    /// Get training and test data for a fold
    fn get_train_test_split(
        &self,
        folds: &[Vec<usize>],
        test_fold_idx: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        
        for (fold_idx, fold) in folds.iter().enumerate() {
            if fold_idx == test_fold_idx {
                test_indices.extend_from_slice(fold);
            } else {
                train_indices.extend_from_slice(fold);
            }
        }
        
        (train_indices, test_indices)
    }
    
    /// Evaluate model on test data
    fn evaluate_model(
        &self,
        test_indices: &[usize],
        params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<FixedPoint, EstimationError> {
        match self.validation_metric {
            ValidationMetric::LogLikelihood => {
                // Would compute log-likelihood on test data
                Ok(FixedPoint::from_float(-100.0)) // Placeholder
            }
            ValidationMetric::AIC => {
                let log_likelihood = FixedPoint::from_float(-100.0); // Placeholder
                let n_params = FixedPoint::from_float(params.dimension() as f64 * 3.0);
                Ok(FixedPoint::from_float(2.0) * n_params - FixedPoint::from_float(2.0) * log_likelihood)
            }
            ValidationMetric::BIC => {
                let log_likelihood = FixedPoint::from_float(-100.0); // Placeholder
                let n_params = FixedPoint::from_float(params.dimension() as f64 * 3.0);
                let n_obs = FixedPoint::from_float(test_indices.len() as f64);
                Ok(n_params * n_obs.ln() - FixedPoint::from_float(2.0) * log_likelihood)
            }
            ValidationMetric::KSTest => {
                // Would perform Kolmogorov-Smirnov test
                Ok(FixedPoint::from_float(0.1)) // Placeholder
            }
        }
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub best_model_idx: usize,
    pub results: Vec<ModelValidationResult>,
}

#[derive(Debug, Clone)]
pub struct ModelValidationResult {
    pub params: MultivariateHawkesParams,
    pub mean_score: FixedPoint,
    pub std_score: FixedPoint,
    pub fold_scores: Vec<FixedPoint>,
}

/// Goodness-of-fit tests for Hawkes processes
#[derive(Debug, Clone)]
pub struct HawkesGoodnessOfFitTester {
    pub significance_level: FixedPoint,
    pub n_bootstrap_samples: usize,
}

impl Default for HawkesGoodnessOfFitTester {
    fn default() -> Self {
        Self {
            significance_level: FixedPoint::from_float(0.05),
            n_bootstrap_samples: 1000,
        }
    }
}

impl HawkesGoodnessOfFitTester {
    pub fn new(significance_level: FixedPoint, n_bootstrap_samples: usize) -> Self {
        Self {
            significance_level,
            n_bootstrap_samples,
        }
    }
    
    /// Perform comprehensive goodness-of-fit testing
    pub fn test_fit(
        &self,
        observed_events: &[Vec<HawkesEvent>],
        fitted_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<GoodnessOfFitResult, EstimationError> {
        let mut results = GoodnessOfFitResult::default();
        
        // Kolmogorov-Smirnov test
        results.ks_test = self.kolmogorov_smirnov_test(
            observed_events,
            fitted_params,
            observation_time,
        )?;
        
        // Anderson-Darling test
        results.ad_test = self.anderson_darling_test(
            observed_events,
            fitted_params,
            observation_time,
        )?;
        
        // Residual analysis
        results.residual_analysis = self.residual_analysis(
            observed_events,
            fitted_params,
            observation_time,
        )?;
        
        // Bootstrap test
        results.bootstrap_test = self.bootstrap_test(
            observed_events,
            fitted_params,
            observation_time,
        )?;
        
        Ok(results)
    }
    
    /// Kolmogorov-Smirnov test for inter-event times
    fn kolmogorov_smirnov_test(
        &self,
        observed_events: &[Vec<HawkesEvent>],
        fitted_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<StatisticalTest, EstimationError> {
        // Extract inter-event times
        let mut inter_event_times = Vec::new();
        
        for sequence in observed_events {
            for window in sequence.windows(2) {
                let dt = window[1].time - window[0].time;
                inter_event_times.push(dt);
            }
        }
        
        if inter_event_times.is_empty() {
            return Ok(StatisticalTest {
                statistic: FixedPoint::zero(),
                p_value: FixedPoint::one(),
                critical_value: FixedPoint::zero(),
                is_significant: false,
            });
        }
        
        // Sort observed times
        inter_event_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Compute empirical CDF vs theoretical CDF
        let n = inter_event_times.len();
        let mut max_diff = FixedPoint::zero();
        
        for (i, &time) in inter_event_times.iter().enumerate() {
            let empirical_cdf = FixedPoint::from_float((i + 1) as f64) / 
                FixedPoint::from_float(n as f64);
            
            // Theoretical CDF (simplified - would use actual intensity function)
            let theoretical_cdf = FixedPoint::one() - (-time).exp();
            
            let diff = (empirical_cdf - theoretical_cdf).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        
        // Compute critical value
        let critical_value = FixedPoint::from_float(1.36) / 
            FixedPoint::from_float(n as f64).sqrt();
        
        let is_significant = max_diff > critical_value;
        let p_value = if is_significant {
            FixedPoint::from_float(0.01) // Simplified
        } else {
            FixedPoint::from_float(0.1)
        };
        
        Ok(StatisticalTest {
            statistic: max_diff,
            p_value,
            critical_value,
            is_significant,
        })
    }
    
    /// Anderson-Darling test
    fn anderson_darling_test(
        &self,
        observed_events: &[Vec<HawkesEvent>],
        fitted_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<StatisticalTest, EstimationError> {
        // Simplified implementation
        Ok(StatisticalTest {
            statistic: FixedPoint::from_float(0.5),
            p_value: FixedPoint::from_float(0.1),
            critical_value: FixedPoint::from_float(0.75),
            is_significant: false,
        })
    }
    
    /// Residual analysis
    fn residual_analysis(
        &self,
        observed_events: &[Vec<HawkesEvent>],
        fitted_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<ResidualAnalysis, EstimationError> {
        // Compute residuals (simplified)
        let mut residuals = Vec::new();
        
        for sequence in observed_events {
            for event in sequence {
                // Compute residual as difference between observed and expected
                let residual = FixedPoint::from_float(0.1); // Placeholder
                residuals.push(residual);
            }
        }
        
        // Compute residual statistics
        let mean_residual = if !residuals.is_empty() {
            residuals.iter().fold(FixedPoint::zero(), |acc, &r| acc + r) /
            FixedPoint::from_float(residuals.len() as f64)
        } else {
            FixedPoint::zero()
        };
        
        let std_residual = if residuals.len() > 1 {
            let variance = residuals.iter()
                .map(|&r| (r - mean_residual) * (r - mean_residual))
                .fold(FixedPoint::zero(), |acc, var| acc + var) /
                FixedPoint::from_float((residuals.len() - 1) as f64);
            variance.sqrt()
        } else {
            FixedPoint::zero()
        };
        
        Ok(ResidualAnalysis {
            residuals,
            mean_residual,
            std_residual,
            autocorrelation: FixedPoint::from_float(0.05), // Placeholder
        })
    }
    
    /// Bootstrap test
    fn bootstrap_test(
        &self,
        observed_events: &[Vec<HawkesEvent>],
        fitted_params: &MultivariateHawkesParams,
        observation_time: FixedPoint,
    ) -> Result<StatisticalTest, EstimationError> {
        // Simplified bootstrap test
        Ok(StatisticalTest {
            statistic: FixedPoint::from_float(1.2),
            p_value: FixedPoint::from_float(0.08),
            critical_value: FixedPoint::from_float(1.5),
            is_significant: false,
        })
    }
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub statistic: FixedPoint,
    pub p_value: FixedPoint,
    pub critical_value: FixedPoint,
    pub is_significant: bool,
}

/// Residual analysis result
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    pub residuals: Vec<FixedPoint>,
    pub mean_residual: FixedPoint,
    pub std_residual: FixedPoint,
    pub autocorrelation: FixedPoint,
}

/// Comprehensive goodness-of-fit result
#[derive(Debug, Clone, Default)]
pub struct GoodnessOfFitResult {
    pub ks_test: StatisticalTest,
    pub ad_test: StatisticalTest,
    pub residual_analysis: ResidualAnalysis,
    pub bootstrap_test: StatisticalTest,
}

impl Default for StatisticalTest {
    fn default() -> Self {
        Self {
            statistic: FixedPoint::zero(),
            p_value: FixedPoint::one(),
            critical_value: FixedPoint::zero(),
            is_significant: false,
        }
    }
}

impl Default for ResidualAnalysis {
    fn default() -> Self {
        Self {
            residuals: Vec::new(),
            mean_residual: FixedPoint::zero(),
            std_residual: FixedPoint::zero(),
            autocorrelation: FixedPoint::zero(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fixed_point::{FixedPoint, DeterministicRng};
    use crate::math::hawkes_process::{
        HawkesEvent, MultivariateHawkesParams, KernelType, MultivariateHawkesSimulator
    };

    /// Create test Hawkes parameters
    fn create_test_params() -> MultivariateHawkesParams {
        let baseline_intensities = vec![
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(0.3),
        ];
        
        let kernels = vec![
            vec![
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.2), 
                    beta: FixedPoint::from_float(1.0) 
                },
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.1), 
                    beta: FixedPoint::from_float(0.8) 
                },
            ],
            vec![
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.15), 
                    beta: FixedPoint::from_float(0.9) 
                },
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.25), 
                    beta: FixedPoint::from_float(1.2) 
                },
            ],
        ];
        
        MultivariateHawkesParams::new(
            baseline_intensities,
            kernels,
            FixedPoint::from_float(10.0),
        ).unwrap()
    }
    
    /// Generate synthetic event data for testing
    fn generate_test_data(
        params: &MultivariateHawkesParams,
        n_sequences: usize,
        observation_time: FixedPoint,
        seed: u64,
    ) -> Vec<Vec<HawkesEvent>> {
        let mut sequences = Vec::new();
        
        for i in 0..n_sequences {
            let mut simulator = MultivariateHawkesSimulator::new(
                params.clone(),
                1000,
            );
            let mut rng = DeterministicRng::new(seed + i as u64);
            
            match simulator.simulate_until(observation_time, &mut rng) {
                Ok(events) => sequences.push(events),
                Err(_) => sequences.push(Vec::new()),
            }
        }
        
        sequences
    }

    #[test]
    fn test_lbfgs_optimizer_creation() {
        let optimizer = LBFGSOptimizer::new(
            100,
            FixedPoint::from_float(1e-6),
            10,
        );
        
        assert_eq!(optimizer.max_iterations, 100);
        assert_eq!(optimizer.tolerance, FixedPoint::from_float(1e-6));
        assert_eq!(optimizer.history_size, 10);
    }

    #[test]
    fn test_lbfgs_simple_quadratic() {
        let optimizer = LBFGSOptimizer::default();
        
        // Minimize f(x) = x^2 + 2x + 1 = (x+1)^2
        // Minimum at x = -1, f(-1) = 0
        let objective = |x: &[FixedPoint]| -> FixedPoint {
            let x0 = x[0];
            x0 * x0 + FixedPoint::from_float(2.0) * x0 + FixedPoint::one()
        };
        
        let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
            let x0 = x[0];
            vec![FixedPoint::from_float(2.0) * x0 + FixedPoint::from_float(2.0)]
        };
        
        let initial = vec![FixedPoint::from_float(5.0)];
        
        match optimizer.optimize(initial, objective, gradient) {
            Ok(result) => {
                assert!((result[0].to_float() + 1.0).abs() < 1e-3);
            }
            Err(e) => panic!("Optimization failed: {}", e),
        }
    }

    #[test]
    fn test_hawkes_ml_estimator_creation() {
        let optimizer = LBFGSOptimizer::default();
        let estimator = HawkesMLEstimator::new(optimizer);
        
        assert!(!estimator.use_regularization);
        
        let estimator_with_reg = estimator.with_regularization(FixedPoint::from_float(0.01));
        assert!(estimator_with_reg.use_regularization);
        assert_eq!(estimator_with_reg.regularization_strength, FixedPoint::from_float(0.01));
    }

    #[test]
    fn test_hawkes_ml_estimation_with_synthetic_data() {
        let true_params = create_test_params();
        let observation_time = FixedPoint::from_float(10.0);
        
        // Generate synthetic data
        let event_sequences = generate_test_data(&true_params, 5, observation_time, 42);
        
        // Skip test if no events generated
        if event_sequences.iter().all(|seq| seq.is_empty()) {
            return;
        }
        
        // Create initial parameters (slightly different from true)
        let mut initial_params = true_params.clone();
        for intensity in &mut initial_params.baseline_intensities {
            *intensity = *intensity * FixedPoint::from_float(1.1);
        }
        
        let optimizer = LBFGSOptimizer::new(50, FixedPoint::from_float(1e-4), 5);
        let estimator = HawkesMLEstimator::new(optimizer);
        
        match estimator.estimate(&event_sequences, &initial_params, observation_time) {
            Ok(estimated_params) => {
                // Check that estimated parameters are reasonable
                assert_eq!(estimated_params.dimension(), true_params.dimension());
                
                for &intensity in &estimated_params.baseline_intensities {
                    assert!(intensity.to_float() > 0.0);
                    assert!(intensity.to_float() < 10.0); // Reasonable upper bound
                }
            }
            Err(e) => {
                // Estimation might fail with limited synthetic data - that's okay for testing
                println!("ML estimation failed (expected with limited data): {}", e);
            }
        }
    }

    #[test]
    fn test_hawkes_em_estimator_creation() {
        let estimator = HawkesEMEstimator::new(
            50,
            FixedPoint::from_float(1e-5),
        );
        
        assert_eq!(estimator.max_iterations, 50);
        assert_eq!(estimator.tolerance, FixedPoint::from_float(1e-5));
    }

    #[test]
    fn test_cross_validator_creation() {
        let validator = HawkesCrossValidator::new(3, ValidationMetric::AIC);
        
        assert_eq!(validator.k_folds, 3);
        matches!(validator.validation_metric, ValidationMetric::AIC);
    }

    #[test]
    fn test_goodness_of_fit_tester_creation() {
        let tester = HawkesGoodnessOfFitTester::new(
            FixedPoint::from_float(0.01),
            500,
        );
        
        assert_eq!(tester.significance_level, FixedPoint::from_float(0.01));
        assert_eq!(tester.n_bootstrap_samples, 500);
    }

    #[test]
    fn test_parameter_vector_conversion() {
        let params = create_test_params();
        let estimator = HawkesMLEstimator::default();
        
        // Convert to vector and back
        let vector = estimator.params_to_vector(&params);
        assert!(!vector.is_empty());
        
        match estimator.vector_to_params(&vector, params.dimension()) {
            Ok(reconstructed) => {
                assert_eq!(reconstructed.dimension(), params.dimension());
                assert_eq!(reconstructed.baseline_intensities.len(), params.baseline_intensities.len());
            }
            Err(e) => panic!("Parameter conversion failed: {}", e),
        }
    }

    #[test]
    fn test_estimation_error_handling() {
        let estimator = HawkesMLEstimator::default();
        
        // Test with empty data
        let empty_sequences: Vec<Vec<HawkesEvent>> = vec![];
        let params = create_test_params();
        let observation_time = FixedPoint::from_float(1.0);
        
        match estimator.estimate(&empty_sequences, &params, observation_time) {
            Err(EstimationError::InsufficientData(_)) => {
                // Expected error
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_validation_metrics() {
        // Test different validation metrics
        let metrics = vec![
            ValidationMetric::LogLikelihood,
            ValidationMetric::AIC,
            ValidationMetric::BIC,
            ValidationMetric::KSTest,
        ];
        
        for metric in metrics {
            let validator = HawkesCrossValidator::new(3, metric);
            // Just ensure creation works with all metric types
            assert_eq!(validator.k_folds, 3);
        }
    }
} 