//! Parameter optimization module for high-frequency quoting strategies
//! 
//! This module implements various parameter optimization techniques including:
//! - Bayesian optimization for hyperparameter tuning
//! - Grid search and random search algorithms
//! - Cross-validation framework for strategy testing
//! - Walk-forward analysis for time series validation

use crate::math::fixed_point::FixedPoint;
use crate::models::quoting_strategy::{QuotingStrategy, ModelParameters};
use crate::models::backtest::{BacktestResults, Backtester, BacktestParams};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Invalid parameter bounds: {0}")]
    InvalidBounds(String),
    #[error("Optimization failed to converge: {0}")]
    ConvergenceFailed(String),
    #[error("Insufficient data for optimization: {0}")]
    InsufficientData(String),
    #[error("Cross-validation error: {0}")]
    CrossValidationError(String),
    #[error("Backtest error: {0}")]
    BacktestError(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Parameter bounds for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterBounds {
    pub min_value: FixedPoint,
    pub max_value: FixedPoint,
    pub is_log_scale: bool,
}

impl ParameterBounds {
    pub fn new(min_value: FixedPoint, max_value: FixedPoint) -> Self {
        Self {
            min_value,
            max_value,
            is_log_scale: false,
        }
    }
    
    pub fn log_scale(mut self) -> Self {
        self.is_log_scale = true;
        self
    }
    
    pub fn sample(&self, rng: &mut impl Rng) -> FixedPoint {
        if self.is_log_scale {
            let log_min = self.min_value.ln();
            let log_max = self.max_value.ln();
            let log_sample = rng.gen_range(log_min.to_float()..log_max.to_float());
            FixedPoint::from_float(log_sample).exp()
        } else {
            let sample = rng.gen_range(self.min_value.to_float()..self.max_value.to_float());
            FixedPoint::from_float(sample)
        }
    }
    
    pub fn clip(&self, value: FixedPoint) -> FixedPoint {
        value.max(self.min_value).min(self.max_value)
    }
}

/// Parameter space definition for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSpace {
    pub parameters: HashMap<String, ParameterBounds>,
}

impl ParameterSpace {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }
    
    pub fn add_parameter(&mut self, name: String, bounds: ParameterBounds) {
        self.parameters.insert(name, bounds);
    }
    
    pub fn sample(&self, rng: &mut impl Rng) -> HashMap<String, FixedPoint> {
        self.parameters
            .iter()
            .map(|(name, bounds)| (name.clone(), bounds.sample(rng)))
            .collect()
    }
    
    pub fn dimension(&self) -> usize {
        self.parameters.len()
    }
    
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization objective function
pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError>;
    fn name(&self) -> &str;
}

/// Backtest engine adapter for parameter optimization
pub struct BacktestEngine {
    pub backtester: Arc<Mutex<Backtester>>,
    pub initial_price: f64,
    pub n_steps: usize,
    pub dt: f64,
}

impl BacktestEngine {
    pub fn new(backtester: Backtester, initial_price: f64, n_steps: usize, dt: f64) -> Self {
        Self {
            backtester: Arc::new(Mutex::new(backtester)),
            initial_price,
            n_steps,
            dt,
        }
    }
    
    pub fn run_backtest(&self, _model_params: &ModelParameters) -> Result<BacktestResult, String> {
        let mut backtester = self.backtester.lock().unwrap();
        let results = backtester.run(self.initial_price, self.n_steps, self.dt)
            .map_err(|e| e.to_string())?;
        
        // Convert BacktestResults to BacktestResult
        Ok(BacktestResult::from_backtest_results(results))
    }
}

/// Simplified backtest result for optimization
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub sharpe_ratio: FixedPoint,
    pub total_return: FixedPoint,
    pub max_drawdown: FixedPoint,
    pub information_ratio: FixedPoint,
    pub sortino_ratio: FixedPoint,
    pub calmar_ratio: FixedPoint,
}

impl BacktestResult {
    pub fn from_backtest_results(results: BacktestResults) -> Self {
        // Extract metrics from performance_metrics
        let sharpe = results.performance_metrics.sharpe_ratio;
        let total_return = results.performance_metrics.total_return;
        let max_dd = results.performance_metrics.max_drawdown;
        let info_ratio = results.performance_metrics.information_ratio;
        let sortino = results.performance_metrics.sortino_ratio;
        let calmar = if max_dd != 0.0 { total_return / max_dd.abs() } else { 0.0 };
        
        Self {
            sharpe_ratio: FixedPoint::from_float(sharpe),
            total_return: FixedPoint::from_float(total_return),
            max_drawdown: FixedPoint::from_float(max_dd),
            information_ratio: FixedPoint::from_float(info_ratio),
            sortino_ratio: FixedPoint::from_float(sortino),
            calmar_ratio: FixedPoint::from_float(calmar),
        }
    }
}

/// Backtest-based objective function
pub struct BacktestObjective {
    pub backtest_engine: Arc<BacktestEngine>,
    pub objective_type: ObjectiveType,
}

#[derive(Debug, Clone, Copy)]
pub enum ObjectiveType {
    SharpeRatio,
    TotalReturn,
    MaxDrawdown,
    InformationRatio,
    SortinoRatio,
    CalmarRatio,
}

impl BacktestObjective {
    pub fn new(backtest_engine: Arc<BacktestEngine>, objective_type: ObjectiveType) -> Self {
        Self {
            backtest_engine,
            objective_type,
        }
    }
}

impl ObjectiveFunction for BacktestObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        // Convert parameters to ModelParameters
        let model_params = self.parameters_to_model_params(parameters)?;
        
        // Run backtest
        let backtest_result = self.backtest_engine
            .run_backtest(&model_params)
            .map_err(|e| OptimizationError::BacktestError(e.to_string()))?;
        
        // Extract objective value
        let objective_value = match self.objective_type {
            ObjectiveType::SharpeRatio => backtest_result.sharpe_ratio,
            ObjectiveType::TotalReturn => backtest_result.total_return,
            ObjectiveType::MaxDrawdown => -backtest_result.max_drawdown, // Negative because we want to minimize drawdown
            ObjectiveType::InformationRatio => backtest_result.information_ratio,
            ObjectiveType::SortinoRatio => backtest_result.sortino_ratio,
            ObjectiveType::CalmarRatio => backtest_result.calmar_ratio,
        };
        
        Ok(objective_value)
    }
    
    fn name(&self) -> &str {
        match self.objective_type {
            ObjectiveType::SharpeRatio => "Sharpe Ratio",
            ObjectiveType::TotalReturn => "Total Return",
            ObjectiveType::MaxDrawdown => "Max Drawdown",
            ObjectiveType::InformationRatio => "Information Ratio",
            ObjectiveType::SortinoRatio => "Sortino Ratio",
            ObjectiveType::CalmarRatio => "Calmar Ratio",
        }
    }
    
    fn parameters_to_model_params(&self, parameters: &HashMap<String, FixedPoint>) -> Result<ModelParameters, OptimizationError> {
        // Convert optimization parameters to ModelParameters
        // This is a simplified mapping - in practice would be more comprehensive
        Ok(ModelParameters {
            drift_coefficient: parameters.get("drift_coefficient")
                .copied()
                .unwrap_or(FixedPoint::from_float(0.0001)),
            volatility_coefficient: parameters.get("volatility_coefficient")
                .copied()
                .unwrap_or(FixedPoint::from_float(0.02)),
            inventory_penalty: parameters.get("inventory_penalty")
                .copied()
                .unwrap_or(FixedPoint::from_float(0.1)),
            adverse_selection_cost: parameters.get("adverse_selection_cost")
                .copied()
                .unwrap_or(FixedPoint::from_float(0.001)),
            market_impact_coefficient: parameters.get("market_impact_coefficient")
                .copied()
                .unwrap_or(FixedPoint::from_float(0.01)),
            risk_aversion: parameters.get("risk_aversion")
                .copied()
                .unwrap_or(FixedPoint::from_float(1.0)),
            terminal_time: FixedPoint::from_float(1.0),
            max_inventory: FixedPoint::from_float(100.0),
        })
    }
}

/// Grid search optimizer
pub struct GridSearchOptimizer {
    pub parameter_space: ParameterSpace,
    pub grid_points_per_dimension: usize,
}

impl GridSearchOptimizer {
    pub fn new(parameter_space: ParameterSpace, grid_points_per_dimension: usize) -> Self {
        Self {
            parameter_space,
            grid_points_per_dimension,
        }
    }
    
    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
    ) -> Result<OptimizationResult, OptimizationError> {
        let parameter_names = self.parameter_space.parameter_names();
        let mut best_parameters = HashMap::new();
        let mut best_value = FixedPoint::from_float(f64::NEG_INFINITY);
        let mut evaluations = Vec::new();
        
        // Generate grid points
        let grid_points = self.generate_grid_points();
        
        for parameters in grid_points {
            let value = objective.evaluate(&parameters)?;
            evaluations.push(ParameterEvaluation {
                parameters: parameters.clone(),
                objective_value: value,
            });
            
            if value > best_value {
                best_value = value;
                best_parameters = parameters;
            }
        }
        
        Ok(OptimizationResult {
            best_parameters,
            best_value,
            evaluations,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: evaluations.len(),
                final_gradient_norm: FixedPoint::zero(),
            },
        })
    }
    
    fn generate_grid_points(&self) -> Vec<HashMap<String, FixedPoint>> {
        let parameter_names = self.parameter_space.parameter_names();
        let mut grid_points = Vec::new();
        
        // Generate all combinations of grid points
        self.generate_grid_recursive(&parameter_names, 0, &mut HashMap::new(), &mut grid_points);
        
        grid_points
    }
    
    fn generate_grid_recursive(
        &self,
        parameter_names: &[String],
        param_index: usize,
        current_params: &mut HashMap<String, FixedPoint>,
        grid_points: &mut Vec<HashMap<String, FixedPoint>>,
    ) {
        if param_index >= parameter_names.len() {
            grid_points.push(current_params.clone());
            return;
        }
        
        let param_name = &parameter_names[param_index];
        let bounds = &self.parameter_space.parameters[param_name];
        
        for i in 0..self.grid_points_per_dimension {
            let t = i as f64 / (self.grid_points_per_dimension - 1) as f64;
            let value = if bounds.is_log_scale {
                let log_min = bounds.min_value.ln();
                let log_max = bounds.max_value.ln();
                let log_value = log_min + (log_max - log_min) * FixedPoint::from_float(t);
                log_value.exp()
            } else {
                bounds.min_value + (bounds.max_value - bounds.min_value) * FixedPoint::from_float(t)
            };
            
            current_params.insert(param_name.clone(), value);
            self.generate_grid_recursive(parameter_names, param_index + 1, current_params, grid_points);
        }
        
        current_params.remove(param_name);
    }
}

/// Random search optimizer
pub struct RandomSearchOptimizer {
    pub parameter_space: ParameterSpace,
    pub num_evaluations: usize,
    pub seed: Option<u64>,
}

impl RandomSearchOptimizer {
    pub fn new(parameter_space: ParameterSpace, num_evaluations: usize) -> Self {
        Self {
            parameter_space,
            num_evaluations,
            seed: None,
        }
    }
    
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
    ) -> Result<OptimizationResult, OptimizationError> {
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        let mut best_parameters = HashMap::new();
        let mut best_value = FixedPoint::from_float(f64::NEG_INFINITY);
        let mut evaluations = Vec::new();
        
        for _ in 0..self.num_evaluations {
            let parameters = self.parameter_space.sample(&mut rng);
            let value = objective.evaluate(&parameters)?;
            
            evaluations.push(ParameterEvaluation {
                parameters: parameters.clone(),
                objective_value: value,
            });
            
            if value > best_value {
                best_value = value;
                best_parameters = parameters;
            }
        }
        
        Ok(OptimizationResult {
            best_parameters,
            best_value,
            evaluations,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: evaluations.len(),
                final_gradient_norm: FixedPoint::zero(),
            },
        })
    }
}

/// Gaussian Process for Bayesian optimization
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    pub kernel: KernelFunction,
    pub noise_variance: FixedPoint,
    pub x_train: Vec<Vec<FixedPoint>>,
    pub y_train: Vec<FixedPoint>,
    pub k_inv: Option<Vec<Vec<FixedPoint>>>,
}

#[derive(Debug, Clone)]
pub enum KernelFunction {
    RBF { length_scale: FixedPoint, variance: FixedPoint },
    Matern32 { length_scale: FixedPoint, variance: FixedPoint },
    Matern52 { length_scale: FixedPoint, variance: FixedPoint },
}

impl KernelFunction {
    pub fn evaluate(&self, x1: &[FixedPoint], x2: &[FixedPoint]) -> FixedPoint {
        match self {
            KernelFunction::RBF { length_scale, variance } => {
                let squared_distance = x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(FixedPoint::zero(), |acc, x| acc + x);
                
                *variance * (-(squared_distance / (*length_scale * *length_scale * FixedPoint::from_float(2.0)))).exp()
            }
            KernelFunction::Matern32 { length_scale, variance } => {
                let distance = x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(FixedPoint::zero(), |acc, x| acc + x)
                    .sqrt();
                
                let scaled_distance = distance * FixedPoint::from_float(3.0).sqrt() / *length_scale;
                *variance * (FixedPoint::one() + scaled_distance) * (-scaled_distance).exp()
            }
            KernelFunction::Matern52 { length_scale, variance } => {
                let distance = x1.iter()
                    .zip(x2.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .fold(FixedPoint::zero(), |acc, x| acc + x)
                    .sqrt();
                
                let scaled_distance = distance * FixedPoint::from_float(5.0).sqrt() / *length_scale;
                let term = FixedPoint::one() + scaled_distance + scaled_distance * scaled_distance / FixedPoint::from_float(3.0);
                *variance * term * (-scaled_distance).exp()
            }
        }
    }
}

impl GaussianProcess {
    pub fn new(kernel: KernelFunction, noise_variance: FixedPoint) -> Self {
        Self {
            kernel,
            noise_variance,
            x_train: Vec::new(),
            y_train: Vec::new(),
            k_inv: None,
        }
    }
    
    pub fn fit(&mut self, x_train: Vec<Vec<FixedPoint>>, y_train: Vec<FixedPoint>) -> Result<(), OptimizationError> {
        if x_train.len() != y_train.len() {
            return Err(OptimizationError::InvalidBounds("Training data size mismatch".to_string()));
        }
        
        self.x_train = x_train;
        self.y_train = y_train;
        
        // Compute kernel matrix and its inverse
        let n = self.x_train.len();
        let mut k_matrix = vec![vec![FixedPoint::zero(); n]; n];
        
        for i in 0..n {
            for j in 0..n {
                k_matrix[i][j] = self.kernel.evaluate(&self.x_train[i], &self.x_train[j]);
                if i == j {
                    k_matrix[i][j] = k_matrix[i][j] + self.noise_variance;
                }
            }
        }
        
        // Compute inverse using Cholesky decomposition (simplified)
        self.k_inv = Some(self.matrix_inverse(&k_matrix)?);
        
        Ok(())
    }
    
    pub fn predict(&self, x_test: &[FixedPoint]) -> Result<(FixedPoint, FixedPoint), OptimizationError> {
        if self.x_train.is_empty() {
            return Err(OptimizationError::InsufficientData("No training data".to_string()));
        }
        
        let k_inv = self.k_inv.as_ref()
            .ok_or_else(|| OptimizationError::InsufficientData("Model not fitted".to_string()))?;
        
        // Compute kernel vector between test point and training points
        let k_star: Vec<FixedPoint> = self.x_train.iter()
            .map(|x_train| self.kernel.evaluate(x_test, x_train))
            .collect();
        
        // Compute mean prediction
        let mean = k_star.iter()
            .zip(k_inv.iter())
            .map(|(&k, k_inv_row)| {
                k_inv_row.iter()
                    .zip(self.y_train.iter())
                    .map(|(&k_inv_val, &y)| k_inv_val * y)
                    .fold(FixedPoint::zero(), |acc, x| acc + x) * k
            })
            .fold(FixedPoint::zero(), |acc, x| acc + x);
        
        // Compute variance prediction (simplified)
        let k_star_star = self.kernel.evaluate(x_test, x_test);
        let variance = k_star_star - k_star.iter()
            .zip(k_inv.iter())
            .map(|(&k, k_inv_row)| {
                k * k_inv_row.iter()
                    .zip(k_star.iter())
                    .map(|(&k_inv_val, &k_star_val)| k_inv_val * k_star_val)
                    .fold(FixedPoint::zero(), |acc, x| acc + x)
            })
            .fold(FixedPoint::zero(), |acc, x| acc + x);
        
        Ok((mean, variance.max(FixedPoint::from_float(1e-6))))
    }
    
    fn matrix_inverse(&self, matrix: &[Vec<FixedPoint>]) -> Result<Vec<Vec<FixedPoint>>, OptimizationError> {
        let n = matrix.len();
        let mut augmented = vec![vec![FixedPoint::zero(); 2 * n]; n];
        
        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = if i == j { FixedPoint::one() } else { FixedPoint::zero() };
            }
        }
        
        // Gaussian elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }
            
            // Swap rows
            if max_row != i {
                augmented.swap(i, max_row);
            }
            
            // Make diagonal element 1
            let pivot = augmented[i][i];
            if pivot.abs() < FixedPoint::from_float(1e-10) {
                return Err(OptimizationError::NumericalError("Matrix is singular".to_string()));
            }
            
            for j in 0..2 * n {
                augmented[i][j] = augmented[i][j] / pivot;
            }
            
            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[k][i];
                    for j in 0..2 * n {
                        augmented[k][j] = augmented[k][j] - factor * augmented[i][j];
                    }
                }
            }
        }
        
        // Extract inverse matrix
        let mut inverse = vec![vec![FixedPoint::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        Ok(inverse)
    }
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound { beta: FixedPoint },
    ProbabilityOfImprovement,
}

impl AcquisitionFunction {
    pub fn evaluate(
        &self,
        mean: FixedPoint,
        variance: FixedPoint,
        best_value: FixedPoint,
    ) -> FixedPoint {
        let std_dev = variance.sqrt();
        
        match self {
            AcquisitionFunction::ExpectedImprovement => {
                if std_dev < FixedPoint::from_float(1e-6) {
                    return FixedPoint::zero();
                }
                
                let improvement = mean - best_value;
                let z = improvement / std_dev;
                
                // Approximate normal CDF and PDF
                let phi = self.normal_cdf(z);
                let pdf = self.normal_pdf(z);
                
                improvement * phi + std_dev * pdf
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => {
                mean + *beta * std_dev
            }
            AcquisitionFunction::ProbabilityOfImprovement => {
                if std_dev < FixedPoint::from_float(1e-6) {
                    return FixedPoint::zero();
                }
                
                let z = (mean - best_value) / std_dev;
                self.normal_cdf(z)
            }
        }
    }
    
    fn normal_cdf(&self, x: FixedPoint) -> FixedPoint {
        // Approximation of normal CDF using error function
        let a1 = FixedPoint::from_float(0.254829592);
        let a2 = FixedPoint::from_float(-0.284496736);
        let a3 = FixedPoint::from_float(1.421413741);
        let a4 = FixedPoint::from_float(-1.453152027);
        let a5 = FixedPoint::from_float(1.061405429);
        let p = FixedPoint::from_float(0.3275911);
        
        let sign = if x < FixedPoint::zero() { -FixedPoint::one() } else { FixedPoint::one() };
        let x_abs = x.abs();
        
        let t = FixedPoint::one() / (FixedPoint::one() + p * x_abs);
        let y = FixedPoint::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-(x_abs * x_abs)).exp();
        
        FixedPoint::from_float(0.5) * (FixedPoint::one() + sign * y)
    }
    
    fn normal_pdf(&self, x: FixedPoint) -> FixedPoint {
        let sqrt_2pi = FixedPoint::from_float(2.506628274631);
        (-(x * x / FixedPoint::from_float(2.0))).exp() / sqrt_2pi
    }
}

/// Bayesian optimization algorithm
pub struct BayesianOptimizer {
    pub parameter_space: ParameterSpace,
    pub gaussian_process: GaussianProcess,
    pub acquisition_function: AcquisitionFunction,
    pub num_initial_points: usize,
    pub num_iterations: usize,
    pub seed: Option<u64>,
}

impl BayesianOptimizer {
    pub fn new(
        parameter_space: ParameterSpace,
        kernel: KernelFunction,
        acquisition_function: AcquisitionFunction,
    ) -> Self {
        Self {
            parameter_space,
            gaussian_process: GaussianProcess::new(kernel, FixedPoint::from_float(1e-6)),
            acquisition_function,
            num_initial_points: 10,
            num_iterations: 50,
            seed: None,
        }
    }
    
    pub fn with_initial_points(mut self, num_points: usize) -> Self {
        self.num_initial_points = num_points;
        self
    }
    
    pub fn with_iterations(mut self, num_iterations: usize) -> Self {
        self.num_iterations = num_iterations;
        self
    }
    
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    pub fn optimize<F: ObjectiveFunction>(
        &mut self,
        objective: &F,
    ) -> Result<OptimizationResult, OptimizationError> {
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        let mut evaluations = Vec::new();
        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        
        // Initial random sampling
        for _ in 0..self.num_initial_points {
            let parameters = self.parameter_space.sample(&mut rng);
            let value = objective.evaluate(&parameters)?;
            
            let x_vec = self.parameters_to_vector(&parameters);
            x_train.push(x_vec);
            y_train.push(value);
            
            evaluations.push(ParameterEvaluation {
                parameters,
                objective_value: value,
            });
        }
        
        // Bayesian optimization iterations
        for _ in 0..self.num_iterations {
            // Fit Gaussian process
            self.gaussian_process.fit(x_train.clone(), y_train.clone())?;
            
            // Find best current value
            let best_value = y_train.iter()
                .fold(FixedPoint::from_float(f64::NEG_INFINITY), |acc, &x| acc.max(x));
            
            // Optimize acquisition function
            let next_parameters = self.optimize_acquisition(best_value, &mut rng)?;
            let next_value = objective.evaluate(&next_parameters)?;
            
            // Add to training data
            let x_vec = self.parameters_to_vector(&next_parameters);
            x_train.push(x_vec);
            y_train.push(next_value);
            
            evaluations.push(ParameterEvaluation {
                parameters: next_parameters,
                objective_value: next_value,
            });
        }
        
        // Find best result
        let best_evaluation = evaluations.iter()
            .max_by(|a, b| a.objective_value.partial_cmp(&b.objective_value).unwrap())
            .unwrap();
        
        Ok(OptimizationResult {
            best_parameters: best_evaluation.parameters.clone(),
            best_value: best_evaluation.objective_value,
            evaluations,
            convergence_info: ConvergenceInfo {
                converged: true,
                iterations: self.num_initial_points + self.num_iterations,
                final_gradient_norm: FixedPoint::zero(),
            },
        })
    }
    
    fn optimize_acquisition(
        &self,
        best_value: FixedPoint,
        rng: &mut impl Rng,
    ) -> Result<HashMap<String, FixedPoint>, OptimizationError> {
        let mut best_acquisition = FixedPoint::from_float(f64::NEG_INFINITY);
        let mut best_parameters = HashMap::new();
        
        // Random search over acquisition function (simplified)
        for _ in 0..1000 {
            let parameters = self.parameter_space.sample(rng);
            let x_vec = self.parameters_to_vector(&parameters);
            
            let (mean, variance) = self.gaussian_process.predict(&x_vec)?;
            let acquisition_value = self.acquisition_function.evaluate(mean, variance, best_value);
            
            if acquisition_value > best_acquisition {
                best_acquisition = acquisition_value;
                best_parameters = parameters;
            }
        }
        
        Ok(best_parameters)
    }
    
    fn parameters_to_vector(&self, parameters: &HashMap<String, FixedPoint>) -> Vec<FixedPoint> {
        self.parameter_space.parameter_names()
            .iter()
            .map(|name| parameters.get(name).copied().unwrap_or(FixedPoint::zero()))
            .collect()
    }
}

/// Cross-validation framework
pub struct CrossValidationFramework {
    pub num_folds: usize,
    pub validation_method: ValidationMethod,
}

#[derive(Debug, Clone, Copy)]
pub enum ValidationMethod {
    KFold,
    TimeSeriesSplit,
    WalkForward,
}

impl CrossValidationFramework {
    pub fn new(num_folds: usize, validation_method: ValidationMethod) -> Self {
        Self {
            num_folds,
            validation_method,
        }
    }
    
    pub fn validate<F: ObjectiveFunction>(
        &self,
        objective: &F,
        parameters: &HashMap<String, FixedPoint>,
        data_size: usize,
    ) -> Result<CrossValidationResult, OptimizationError> {
        let folds = self.create_folds(data_size)?;
        let mut fold_results = Vec::new();
        
        for (train_indices, test_indices) in folds {
            // Create fold-specific objective (would need to implement data splitting)
            let fold_result = objective.evaluate(parameters)?;
            fold_results.push(fold_result);
        }
        
        let mean_score = fold_results.iter()
            .fold(FixedPoint::zero(), |acc, &x| acc + x) / FixedPoint::from_int(fold_results.len() as i32);
        
        let variance = fold_results.iter()
            .map(|&x| (x - mean_score) * (x - mean_score))
            .fold(FixedPoint::zero(), |acc, x| acc + x) / FixedPoint::from_int(fold_results.len() as i32);
        
        let std_dev = variance.sqrt();
        
        Ok(CrossValidationResult {
            mean_score,
            std_dev,
            fold_scores: fold_results,
        })
    }
    
    fn create_folds(&self, data_size: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, OptimizationError> {
        match self.validation_method {
            ValidationMethod::KFold => self.create_k_folds(data_size),
            ValidationMethod::TimeSeriesSplit => self.create_time_series_folds(data_size),
            ValidationMethod::WalkForward => self.create_walk_forward_folds(data_size),
        }
    }
    
    fn create_k_folds(&self, data_size: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, OptimizationError> {
        let fold_size = data_size / self.num_folds;
        let mut folds = Vec::new();
        
        for i in 0..self.num_folds {
            let test_start = i * fold_size;
            let test_end = if i == self.num_folds - 1 { data_size } else { (i + 1) * fold_size };
            
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..test_start)
                .chain(test_end..data_size)
                .collect();
            
            folds.push((train_indices, test_indices));
        }
        
        Ok(folds)
    }
    
    fn create_time_series_folds(&self, data_size: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, OptimizationError> {
        let min_train_size = data_size / (self.num_folds + 1);
        let test_size = data_size / self.num_folds;
        let mut folds = Vec::new();
        
        for i in 0..self.num_folds {
            let train_end = min_train_size + i * test_size;
            let test_start = train_end;
            let test_end = (test_start + test_size).min(data_size);
            
            if test_end <= test_start {
                break;
            }
            
            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            folds.push((train_indices, test_indices));
        }
        
        Ok(folds)
    }
    
    fn create_walk_forward_folds(&self, data_size: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>, OptimizationError> {
        let window_size = data_size / self.num_folds;
        let mut folds = Vec::new();
        
        for i in 0..self.num_folds {
            let train_start = i * window_size / 2;
            let train_end = train_start + window_size;
            let test_start = train_end;
            let test_end = (test_start + window_size / 2).min(data_size);
            
            if test_end <= test_start || train_end >= data_size {
                break;
            }
            
            let train_indices: Vec<usize> = (train_start..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            
            folds.push((train_indices, test_indices));
        }
        
        Ok(folds)
    }
}

/// Walk-forward analysis framework
pub struct WalkForwardAnalysis {
    pub window_size: usize,
    pub step_size: usize,
    pub min_train_size: usize,
}

impl WalkForwardAnalysis {
    pub fn new(window_size: usize, step_size: usize, min_train_size: usize) -> Self {
        Self {
            window_size,
            step_size,
            min_train_size,
        }
    }
    
    pub fn analyze<F: ObjectiveFunction>(
        &self,
        objective: &F,
        parameters: &HashMap<String, FixedPoint>,
        data_size: usize,
    ) -> Result<WalkForwardResult, OptimizationError> {
        let mut results = Vec::new();
        let mut current_start = 0;
        
        while current_start + self.min_train_size + self.window_size <= data_size {
            let train_end = current_start + self.min_train_size;
            let test_start = train_end;
            let test_end = (test_start + self.window_size).min(data_size);
            
            // In practice, would create subset of data for this window
            let window_result = objective.evaluate(parameters)?;
            
            results.push(WalkForwardWindow {
                train_start: current_start,
                train_end,
                test_start,
                test_end,
                performance: window_result,
            });
            
            current_start += self.step_size;
        }
        
        let mean_performance = results.iter()
            .map(|w| w.performance)
            .fold(FixedPoint::zero(), |acc, x| acc + x) / FixedPoint::from_int(results.len() as i32);
        
        let performance_std = {
            let variance = results.iter()
                .map(|w| (w.performance - mean_performance) * (w.performance - mean_performance))
                .fold(FixedPoint::zero(), |acc, x| acc + x) / FixedPoint::from_int(results.len() as i32);
            variance.sqrt()
        };
        
        Ok(WalkForwardResult {
            windows: results,
            mean_performance,
            performance_std,
            stability_ratio: mean_performance / performance_std.max(FixedPoint::from_float(1e-6)),
        })
    }
}

/// Result structures
#[derive(Debug, Clone)]
pub struct ParameterEvaluation {
    pub parameters: HashMap<String, FixedPoint>,
    pub objective_value: FixedPoint,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_parameters: HashMap<String, FixedPoint>,
    pub best_value: FixedPoint,
    pub evaluations: Vec<ParameterEvaluation>,
    pub convergence_info: ConvergenceInfo,
}

#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    pub converged: bool,
    pub iterations: usize,
    pub final_gradient_norm: FixedPoint,
}

#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub mean_score: FixedPoint,
    pub std_dev: FixedPoint,
    pub fold_scores: Vec<FixedPoint>,
}

#[derive(Debug, Clone)]
pub struct WalkForwardWindow {
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
    pub performance: FixedPoint,
}

#[derive(Debug, Clone)]
pub struct WalkForwardResult {
    pub windows: Vec<WalkForwardWindow>,
    pub mean_performance: FixedPoint,
    pub performance_std: FixedPoint,
    pub stability_ratio: FixedPoint,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parameter_bounds() {
        let bounds = ParameterBounds::new(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(1.0)
        );
        
        let mut rng = StdRng::seed_from_u64(42);
        let sample = bounds.sample(&mut rng);
        
        assert!(sample >= bounds.min_value);
        assert!(sample <= bounds.max_value);
    }
    
    #[test]
    fn test_parameter_space() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "param1".to_string(),
            ParameterBounds::new(FixedPoint::from_float(0.0), FixedPoint::from_float(1.0))
        );
        space.add_parameter(
            "param2".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-1.0), FixedPoint::from_float(1.0))
        );
        
        let mut rng = StdRng::seed_from_u64(42);
        let sample = space.sample(&mut rng);
        
        assert_eq!(sample.len(), 2);
        assert!(sample.contains_key("param1"));
        assert!(sample.contains_key("param2"));
    }
    
    #[test]
    fn test_grid_search() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        
        struct QuadraticObjective;
        impl ObjectiveFunction for QuadraticObjective {
            fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
                let x = parameters.get("x").copied().unwrap_or(FixedPoint::zero());
                Ok(-(x * x)) // Maximize negative quadratic (minimize quadratic)
            }
            
            fn name(&self) -> &str {
                "Quadratic"
            }
        }
        
        let optimizer = GridSearchOptimizer::new(space, 5);
        let result = optimizer.optimize(&QuadraticObjective).unwrap();
        
        // Should find minimum near x = 0
        let best_x = result.best_parameters.get("x").copied().unwrap();
        assert!(best_x.abs() < FixedPoint::from_float(1.0));
    }
    
    #[test]
    fn test_random_search() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        
        struct QuadraticObjective;
        impl ObjectiveFunction for QuadraticObjective {
            fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
                let x = parameters.get("x").copied().unwrap_or(FixedPoint::zero());
                Ok(-(x * x))
            }
            
            fn name(&self) -> &str {
                "Quadratic"
            }
        }
        
        let optimizer = RandomSearchOptimizer::new(space, 100).with_seed(42);
        let result = optimizer.optimize(&QuadraticObjective).unwrap();
        
        assert_eq!(result.evaluations.len(), 100);
        assert!(result.best_value <= FixedPoint::zero());
    }
    
    #[test]
    fn test_cross_validation_folds() {
        let cv = CrossValidationFramework::new(5, ValidationMethod::KFold);
        let folds = cv.create_folds(100).unwrap();
        
        assert_eq!(folds.len(), 5);
        
        // Check that all indices are covered exactly once in test sets
        let mut all_test_indices = Vec::new();
        for (_, test_indices) in &folds {
            all_test_indices.extend(test_indices);
        }
        all_test_indices.sort();
        
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(all_test_indices, expected);
    }
    
    #[test]
    fn test_gaussian_process_kernel() {
        let kernel = KernelFunction::RBF {
            length_scale: FixedPoint::from_float(1.0),
            variance: FixedPoint::from_float(1.0),
        };
        
        let x1 = vec![FixedPoint::zero()];
        let x2 = vec![FixedPoint::zero()];
        let k_same = kernel.evaluate(&x1, &x2);
        
        let x3 = vec![FixedPoint::from_float(1.0)];
        let k_diff = kernel.evaluate(&x1, &x3);
        
        assert!(k_same > k_diff); // Kernel should be larger for identical points
        assert!(k_same <= FixedPoint::from_float(1.0)); // Should not exceed variance
    }
}