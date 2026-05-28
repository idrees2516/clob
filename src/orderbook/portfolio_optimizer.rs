//! Multi-Asset Portfolio Optimization
//!
//! This module implements modern portfolio theory optimization techniques including
//! mean-variance optimization, risk parity, and correlation-aware allocation strategies.

use std::collections::HashMap;
use nalgebra as na;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::Symbol;
use super::multi_asset::Position;

/// Portfolio optimization errors
#[derive(Error, Debug, Clone)]
pub enum OptimizationError {
    #[error("Insufficient data for optimization: {0}")]
    InsufficientData(String),
    #[error("Matrix inversion failed: {0}")]
    MatrixInversionError(String),
    #[error("Optimization convergence failed: {0}")]
    ConvergenceError(String),
    #[error("Invalid constraints: {0}")]
    InvalidConstraints(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Portfolio allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAllocation {
    /// Target weights for each symbol (sum to 1.0)
    pub weights: HashMap<Symbol, f64>,
    
    /// Expected portfolio return
    pub expected_return: f64,
    
    /// Expected portfolio volatility
    pub expected_volatility: f64,
    
    /// Sharpe ratio
    pub sharpe_ratio: f64,
    
    /// Maximum individual weight
    pub max_weight: f64,
    
    /// Portfolio diversification ratio
    pub diversification_ratio: f64,
    
    /// Optimization method used
    pub method: OptimizationMethod,
    
    /// Timestamp of calculation
    pub timestamp: u64,
}

/// Available optimization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationMethod {
    MeanVariance,
    MinimumVariance,
    RiskParity,
    MaximumDiversification,
    EqualWeight,
}

/// Optimization constraints
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Minimum weight per asset
    pub min_weight: f64,
    
    /// Maximum weight per asset
    pub max_weight: f64,
    
    /// Target return (for mean-variance optimization)
    pub target_return: Option<f64>,
    
    /// Maximum portfolio volatility
    pub max_volatility: Option<f64>,
    
    /// Minimum portfolio return
    pub min_return: Option<f64>,
    
    /// Long-only constraint
    pub long_only: bool,
    
    /// Sector/group constraints
    pub group_constraints: HashMap<String, (f64, f64)>, // (min_weight, max_weight)
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            min_weight: 0.0,
            max_weight: 1.0,
            target_return: None,
            max_volatility: None,
            min_return: None,
            long_only: true,
            group_constraints: HashMap::new(),
        }
    }
}

/// Multi-asset portfolio optimizer
#[derive(Debug)]
pub struct MultiAssetPortfolioOptimizer {
    /// Expected returns for each symbol
    expected_returns: HashMap<Symbol, f64>,
    
    /// Return covariance matrix
    covariance_matrix: Option<na::DMatrix<f64>>,
    
    /// Symbol ordering for matrix operations
    symbol_order: Vec<Symbol>,
    
    /// Risk-free rate for Sharpe ratio calculation
    risk_free_rate: f64,
    
    /// Optimization constraints
    constraints: OptimizationConstraints,
    
    /// Maximum iterations for optimization
    max_iterations: usize,
    
    /// Convergence tolerance
    tolerance: f64,
}

impl MultiAssetPortfolioOptimizer {
    /// Create a new portfolio optimizer
    pub fn new() -> Self {
        Self {
            expected_returns: HashMap::new(),
            covariance_matrix: None,
            symbol_order: Vec::new(),
            risk_free_rate: 0.02, // 2% annual risk-free rate
            constraints: OptimizationConstraints::default(),
            max_iterations: 1000,
            tolerance: 1e-8,
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(
        risk_free_rate: f64,
        constraints: OptimizationConstraints,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            expected_returns: HashMap::new(),
            covariance_matrix: None,
            symbol_order: Vec::new(),
            risk_free_rate,
            constraints,
            max_iterations,
            tolerance,
        }
    }
    
    /// Update expected returns for symbols
    pub fn update_expected_returns(&mut self, returns: HashMap<Symbol, f64>) {
        self.expected_returns = returns;
        self.update_symbol_order();
    }
    
    /// Update covariance matrix
    pub fn update_covariance_matrix(&mut self, correlation_matrix: na::DMatrix<f64>) -> Result<(), OptimizationError> {
        if correlation_matrix.nrows() != correlation_matrix.ncols() {
            return Err(OptimizationError::InvalidConstraints(
                "Correlation matrix must be square".to_string()
            ));
        }
        
        if correlation_matrix.nrows() != self.symbol_order.len() {
            return Err(OptimizationError::InvalidConstraints(
                "Correlation matrix size doesn't match number of symbols".to_string()
            ));
        }
        
        // Convert correlation to covariance using expected returns as volatility proxy
   
       // Convert correlation to covariance using volatilities
        let mut covariance = na::DMatrix::zeros(correlation_matrix.nrows(), correlation_matrix.ncols());
        
        for i in 0..self.symbol_order.len() {
            for j in 0..self.symbol_order.len() {
                let vol_i = self.expected_returns.get(&self.symbol_order[i])
                    .copied()
                    .unwrap_or(0.1)
                    .abs(); // Use absolute return as volatility proxy
                let vol_j = self.expected_returns.get(&self.symbol_order[j])
                    .copied()
                    .unwrap_or(0.1)
                    .abs();
                
                covariance[(i, j)] = correlation_matrix[(i, j)] * vol_i * vol_j;
            }
        }
        
        self.covariance_matrix = Some(covariance);
        Ok(())
    }
    
    /// Update constraints
    pub fn update_constraints(&mut self, constraints: OptimizationConstraints) {
        self.constraints = constraints;
    }
    
    /// Optimize portfolio using specified method
    pub fn optimize(
        &mut self,
        current_positions: HashMap<Symbol, Position>,
        correlation_matrix: na::DMatrix<f64>,
        target_return: f64,
    ) -> Result<PortfolioAllocation, String> {
        // Update covariance matrix
        self.update_covariance_matrix(correlation_matrix)
            .map_err(|e| format!("Failed to update covariance matrix: {}", e))?;
        
        // Set target return constraint
        let mut constraints = self.constraints.clone();
        constraints.target_return = Some(target_return);
        
        // Perform mean-variance optimization
        self.mean_variance_optimization(&constraints)
    }
    
    /// Mean-variance optimization (Markowitz)
    pub fn mean_variance_optimization(
        &self,
        constraints: &OptimizationConstraints,
    ) -> Result<PortfolioAllocation, String> {
        if self.symbol_order.is_empty() {
            return Err("No symbols available for optimization".to_string());
        }
        
        let n = self.symbol_order.len();
        
        if n == 1 {
            // Single asset case
            let symbol = &self.symbol_order[0];
            let mut weights = HashMap::new();
            weights.insert(symbol.clone(), 1.0);
            
            let expected_return = self.expected_returns.get(symbol).copied().unwrap_or(0.0);
            let expected_volatility = self.covariance_matrix
                .as_ref()
                .map(|cov| cov[(0, 0)].sqrt())
                .unwrap_or(0.0);
            
            return Ok(PortfolioAllocation {
                weights,
                expected_return,
                expected_volatility,
                sharpe_ratio: if expected_volatility > 0.0 {
                    (expected_return - self.risk_free_rate) / expected_volatility
                } else {
                    0.0
                },
                max_weight: 1.0,
                diversification_ratio: 1.0,
                method: OptimizationMethod::MeanVariance,
                timestamp: current_timestamp(),
            });
        }
        
        // Multi-asset optimization
        let covariance = self.covariance_matrix.as_ref()
            .ok_or("Covariance matrix not available")?;
        
        // Build expected returns vector
        let mut mu = na::DVector::zeros(n);
        for (i, symbol) in self.symbol_order.iter().enumerate() {
            mu[i] = self.expected_returns.get(symbol).copied().unwrap_or(0.0);
        }
        
        // Solve quadratic optimization problem
        let weights = if let Some(target_ret) = constraints.target_return {
            self.solve_mean_variance_with_target_return(covariance, &mu, target_ret, constraints)?
        } else {
            self.solve_minimum_variance(covariance, constraints)?
        };
        
        // Calculate portfolio metrics
        let expected_return = mu.dot(&weights);
        let expected_variance = weights.transpose() * covariance * &weights;
        let expected_volatility = expected_variance.sqrt();
        
        let sharpe_ratio = if expected_volatility > 0.0 {
            (expected_return - self.risk_free_rate) / expected_volatility
        } else {
            0.0
        };
        
        // Convert weights to HashMap
        let mut weight_map = HashMap::new();
        let mut max_weight = 0.0;
        
        for (i, symbol) in self.symbol_order.iter().enumerate() {
            let weight = weights[i];
            weight_map.insert(symbol.clone(), weight);
            max_weight = max_weight.max(weight.abs());
        }
        
        let diversification_ratio = self.calculate_diversification_ratio(&weights, covariance);
        
        Ok(PortfolioAllocation {
            weights: weight_map,
            expected_return,
            expected_volatility,
            sharpe_ratio,
            max_weight,
            diversification_ratio,
            method: OptimizationMethod::MeanVariance,
            timestamp: current_timestamp(),
        })
    }
    
    /// Solve mean-variance optimization with target return
    fn solve_mean_variance_with_target_return(
        &self,
        covariance: &na::DMatrix<f64>,
        mu: &na::DVector<f64>,
        target_return: f64,
        constraints: &OptimizationConstraints,
    ) -> Result<na::DVector<f64>, String> {
        let n = covariance.nrows();
        
        // Try to invert covariance matrix
        let cov_inv = covariance.try_inverse()
            .ok_or("Failed to invert covariance matrix")?;
        
        // Create constraint matrices
        // Constraint 1: sum of weights = 1
        // Constraint 2: expected return = target_return
        let ones = na::DVector::from_element(n, 1.0);
        
        // A = [mu^T; 1^T], b = [target_return; 1]
        let mut a_matrix = na::DMatrix::zeros(2, n);
        a_matrix.set_row(0, &mu.transpose());
        a_matrix.set_row(1, &ones.transpose());
        
        let b_vector = na::DVector::from_vec(vec![target_return, 1.0]);
        
        // Solve: w = Σ^(-1) * A^T * (A * Σ^(-1) * A^T)^(-1) * b
        let sigma_inv_at = &cov_inv * a_matrix.transpose();
        let middle_matrix = &a_matrix * &sigma_inv_at;
        
        let middle_inv = middle_matrix.try_inverse()
            .ok_or("Failed to solve optimization system")?;
        
        let weights = sigma_inv_at * middle_inv * b_vector;
        
        // Apply box constraints if needed
        self.apply_box_constraints(weights, constraints)
    }
    
    /// Solve minimum variance optimization
    fn solve_minimum_variance(
        &self,
        covariance: &na::DMatrix<f64>,
        constraints: &OptimizationConstraints,
    ) -> Result<na::DVector<f64>, String> {
        let n = covariance.nrows();
        
        let cov_inv = covariance.try_inverse()
            .ok_or("Failed to invert covariance matrix")?;
        
        let ones = na::DVector::from_element(n, 1.0);
        let numerator = &cov_inv * &ones;
        let denominator = ones.dot(&numerator);
        
        if denominator.abs() < 1e-12 {
            return Err("Singular optimization problem".to_string());
        }
        
        let weights = numerator / denominator;
        
        self.apply_box_constraints(weights, constraints)
    }
    
    /// Apply box constraints to weights
    fn apply_box_constraints(
        &self,
        mut weights: na::DVector<f64>,
        constraints: &OptimizationConstraints,
    ) -> Result<na::DVector<f64>, String> {
        // Simple projection onto box constraints
        for i in 0..weights.len() {
            weights[i] = weights[i].max(constraints.min_weight).min(constraints.max_weight);
        }
        
        // Renormalize to sum to 1
        let sum = weights.sum();
        if sum.abs() > 1e-12 {
            weights /= sum;
        } else {
            // Equal weights fallback
            weights.fill(1.0 / weights.len() as f64);
        }
        
        Ok(weights)
    }
    
    /// Risk parity optimization
    pub fn risk_parity_optimization(&self) -> Result<PortfolioAllocation, String> {
        let n = self.symbol_order.len();
        
        if n == 0 {
            return Err("No symbols available".to_string());
        }
        
        let covariance = self.covariance_matrix.as_ref()
            .ok_or("Covariance matrix not available")?;
        
        // Initialize with equal weights
        let mut weights = na::DVector::from_element(n, 1.0 / n as f64);
        
        // Iterative risk parity algorithm
        for _ in 0..self.max_iterations {
            let old_weights = weights.clone();
            
            // Calculate risk contributions
            let portfolio_vol = (weights.transpose() * covariance * &weights).sqrt();
            let marginal_risk = covariance * &weights / portfolio_vol;
            let risk_contributions = weights.component_mul(&marginal_risk);
            
            // Update weights to equalize risk contributions
            let target_risk = risk_contributions.sum() / n as f64;
            
            for i in 0..n {
                if risk_contributions[i] > 1e-12 {
                    weights[i] *= (target_risk / risk_contributions[i]).sqrt();
                }
            }
            
            // Normalize weights
            let sum = weights.sum();
            if sum > 1e-12 {
                weights /= sum;
            }
            
            // Check convergence
            let diff = (&weights - &old_weights).norm();
            if diff < self.tolerance {
                break;
            }
        }
        
        // Calculate portfolio metrics
        let expected_return = if !self.expected_returns.is_empty() {
            self.symbol_order.iter().enumerate()
                .map(|(i, symbol)| weights[i] * self.expected_returns.get(symbol).copied().unwrap_or(0.0))
                .sum()
        } else {
            0.0
        };
        
        let expected_variance = weights.transpose() * covariance * &weights;
        let expected_volatility = expected_variance.sqrt();
        
        let sharpe_ratio = if expected_volatility > 0.0 {
            (expected_return - self.risk_free_rate) / expected_volatility
        } else {
            0.0
        };
        
        // Convert to HashMap
        let mut weight_map = HashMap::new();
        let mut max_weight = 0.0;
        
        for (i, symbol) in self.symbol_order.iter().enumerate() {
            let weight = weights[i];
            weight_map.insert(symbol.clone(), weight);
            max_weight = max_weight.max(weight);
        }
        
        let diversification_ratio = self.calculate_diversification_ratio(&weights, covariance);
        
        Ok(PortfolioAllocation {
            weights: weight_map,
            expected_return,
            expected_volatility,
            sharpe_ratio,
            max_weight,
            diversification_ratio,
            method: OptimizationMethod::RiskParity,
            timestamp: current_timestamp(),
        })
    }
    
    /// Equal weight portfolio
    pub fn equal_weight_portfolio(&self) -> Result<PortfolioAllocation, String> {
        let n = self.symbol_order.len();
        
        if n == 0 {
            return Err("No symbols available".to_string());
        }
        
        let weight = 1.0 / n as f64;
        let mut weight_map = HashMap::new();
        
        for symbol in &self.symbol_order {
            weight_map.insert(symbol.clone(), weight);
        }
        
        // Calculate portfolio metrics if data is available
        let expected_return = if !self.expected_returns.is_empty() {
            self.expected_returns.values().sum::<f64>() / n as f64
        } else {
            0.0
        };
        
        let expected_volatility = if let Some(covariance) = &self.covariance_matrix {
            let weights = na::DVector::from_element(n, weight);
            (weights.transpose() * covariance * &weights).sqrt()
        } else {
            0.0
        };
        
        let sharpe_ratio = if expected_volatility > 0.0 {
            (expected_return - self.risk_free_rate) / expected_volatility
        } else {
            0.0
        };
        
        let diversification_ratio = if let Some(covariance) = &self.covariance_matrix {
            let weights = na::DVector::from_element(n, weight);
            self.calculate_diversification_ratio(&weights, covariance)
        } else {
            1.0
        };
        
        Ok(PortfolioAllocation {
            weights: weight_map,
            expected_return,
            expected_volatility,
            sharpe_ratio,
            max_weight: weight,
            diversification_ratio,
            method: OptimizationMethod::EqualWeight,
            timestamp: current_timestamp(),
        })
    }
    
    /// Calculate diversification ratio
    fn calculate_diversification_ratio(
        &self,
        weights: &na::DVector<f64>,
        covariance: &na::DMatrix<f64>,
    ) -> f64 {
        // Diversification ratio = (weighted average volatility) / (portfolio volatility)
        let portfolio_variance = weights.transpose() * covariance * weights;
        let portfolio_volatility = portfolio_variance.sqrt();
        
        if portfolio_volatility < 1e-12 {
            return 1.0;
        }
        
        let weighted_avg_volatility: f64 = weights.iter().enumerate()
            .map(|(i, &w)| w * covariance[(i, i)].sqrt())
            .sum();
        
        weighted_avg_volatility / portfolio_volatility
    }
    
    /// Update symbol ordering
    fn update_symbol_order(&mut self) {
        self.symbol_order = self.expected_returns.keys().cloned().collect();
        self.symbol_order.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    }
    
    /// Get current symbol order
    pub fn get_symbol_order(&self) -> &[Symbol] {
        &self.symbol_order
    }
}

/// Get current timestamp in nanoseconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::Symbol;
    
    #[test]
    fn test_optimizer_creation() {
        let optimizer = MultiAssetPortfolioOptimizer::new();
        assert_eq!(optimizer.risk_free_rate, 0.02);
        assert_eq!(optimizer.max_iterations, 1000);
    }
    
    #[test]
    fn test_equal_weight_portfolio() {
        let mut optimizer = MultiAssetPortfolioOptimizer::new();
        
        let mut returns = HashMap::new();
        returns.insert(Symbol::new("BTCUSD").unwrap(), 0.15);
        returns.insert(Symbol::new("ETHUSD").unwrap(), 0.12);
        returns.insert(Symbol::new("ADAUSD").unwrap(), 0.10);
        
        optimizer.update_expected_returns(returns);
        
        let allocation = optimizer.equal_weight_portfolio().unwrap();
        
        assert_eq!(allocation.weights.len(), 3);
        for weight in allocation.weights.values() {
            assert!((weight - 1.0/3.0).abs() < 1e-10);
        }
        
        assert_eq!(allocation.method, OptimizationMethod::EqualWeight);
    }
    
    #[test]
    fn test_minimum_variance_optimization() {
        let mut optimizer = MultiAssetPortfolioOptimizer::new();
        
        let mut returns = HashMap::new();
        returns.insert(Symbol::new("BTCUSD").unwrap(), 0.15);
        returns.insert(Symbol::new("ETHUSD").unwrap(), 0.12);
        
        optimizer.update_expected_returns(returns);
        
        // Create a simple correlation matrix
        let correlation = na::DMatrix::from_vec(2, 2, vec![1.0, 0.5, 0.5, 1.0]);
        optimizer.update_covariance_matrix(correlation).unwrap();
        
        let constraints = OptimizationConstraints::default();
        let allocation = optimizer.solve_minimum_variance(
            optimizer.covariance_matrix.as_ref().unwrap(),
            &constraints
        ).unwrap();
        
        // Weights should sum to 1
        let sum: f64 = allocation.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_diversification_ratio() {
        let optimizer = MultiAssetPortfolioOptimizer::new();
        
        // Perfect correlation case - diversification ratio should be 1
        let weights = na::DVector::from_vec(vec![0.5, 0.5]);
        let covariance = na::DMatrix::from_vec(2, 2, vec![0.04, 0.04, 0.04, 0.04]);
        
        let div_ratio = optimizer.calculate_diversification_ratio(&weights, &covariance);
        assert!((div_ratio - 1.0).abs() < 1e-10);
        
        // Zero correlation case - diversification ratio should be > 1
        let covariance_uncorr = na::DMatrix::from_vec(2, 2, vec![0.04, 0.0, 0.0, 0.04]);
        let div_ratio_uncorr = optimizer.calculate_diversification_ratio(&weights, &covariance_uncorr);
        assert!(div_ratio_uncorr > 1.0);
    }
}