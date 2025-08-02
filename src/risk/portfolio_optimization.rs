//! Portfolio Optimization Integration
//!
//! This module implements real-time portfolio optimization including:
//! - Real-time portfolio optimization using quadratic programming
//! - Diversification constraints and concentration limits
//! - Stress testing with historical and Monte Carlo scenarios
//! - Performance monitoring and risk-adjusted return calculation

use super::{Portfolio, MarketData, RiskError, AssetId, Position};
use crate::math::fixed_point::FixedPoint;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Portfolio optimization objective
#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    /// Maximize Sharpe ratio
    MaxSharpe,
    /// Minimize variance for target return
    MinVariance,
    /// Maximize utility (return - risk penalty)
    MaxUtility,
    /// Risk parity (equal risk contribution)
    RiskParity,
}

/// Portfolio constraints
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    /// Maximum weight per asset
    pub max_weight_per_asset: FixedPoint,
    
    /// Minimum weight per asset (can be negative for short positions)
    pub min_weight_per_asset: FixedPoint,
    
    /// Maximum total leverage
    pub max_leverage: FixedPoint,
    
    /// Sector concentration limits
    pub sector_limits: HashMap<String, FixedPoint>,
    
    /// Geographic concentration limits
    pub geographic_limits: HashMap<String, FixedPoint>,
    
    /// Turnover constraint (maximum change from current portfolio)
    pub max_turnover: FixedPoint,
    
    /// Minimum number of assets
    pub min_assets: usize,
    
    /// Maximum number of assets
    pub max_assets: usize,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            max_weight_per_asset: FixedPoint::from_float(0.2), // 20% max per asset
            min_weight_per_asset: FixedPoint::from_float(-0.1), // 10% max short
            max_leverage: FixedPoint::from_float(1.0), // No leverage by default
            sector_limits: HashMap::new(),
            geographic_limits: HashMap::new(),
            max_turnover: FixedPoint::from_float(0.5), // 50% max turnover
            min_assets: 5,
            max_assets: 50,
        }
    }
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimal weights by asset
    pub weights: HashMap<AssetId, FixedPoint>,
    
    /// Expected portfolio return
    pub expected_return: FixedPoint,
    
    /// Portfolio volatility
    pub portfolio_volatility: FixedPoint,
    
    /// Sharpe ratio
    pub sharpe_ratio: FixedPoint,
    
    /// Maximum drawdown estimate
    pub max_drawdown_estimate: FixedPoint,
    
    /// Diversification ratio
    pub diversification_ratio: FixedPoint,
    
    /// Total turnover required
    pub turnover: FixedPoint,
    
    /// Optimization objective value
    pub objective_value: FixedPoint,
    
    /// Constraint violations (if any)
    pub constraint_violations: Vec<String>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Stress test scenario
#[derive(Debug, Clone)]
pub struct StressScenario {
    /// Scenario name
    pub name: String,
    
    /// Asset return shocks
    pub return_shocks: HashMap<AssetId, FixedPoint>,
    
    /// Volatility multipliers
    pub volatility_multipliers: HashMap<AssetId, FixedPoint>,
    
    /// Correlation adjustments
    pub correlation_adjustments: HashMap<(AssetId, AssetId), FixedPoint>,
    
    /// Scenario probability
    pub probability: FixedPoint,
}

/// Stress test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario name
    pub scenario_name: String,
    
    /// Portfolio loss in scenario
    pub portfolio_loss: FixedPoint,
    
    /// Loss as percentage of portfolio value
    pub loss_percentage: FixedPoint,
    
    /// Worst performing asset
    pub worst_asset: AssetId,
    
    /// Best performing asset
    pub best_asset: AssetId,
    
    /// Correlation breakdown indicator
    pub correlation_breakdown: bool,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return
    pub total_return: FixedPoint,
    
    /// Annualized return
    pub annualized_return: FixedPoint,
    
    /// Volatility
    pub volatility: FixedPoint,
    
    /// Sharpe ratio
    pub sharpe_ratio: FixedPoint,
    
    /// Sortino ratio
    pub sortino_ratio: FixedPoint,
    
    /// Calmar ratio
    pub calmar_ratio: FixedPoint,
    
    /// Maximum drawdown
    pub max_drawdown: FixedPoint,
    
    /// Win rate
    pub win_rate: FixedPoint,
    
    /// Average win/loss ratio
    pub win_loss_ratio: FixedPoint,
    
    /// Information ratio
    pub information_ratio: FixedPoint,
}

/// Real-time Portfolio Optimizer
pub struct PortfolioOptimizer {
    /// Optimization objective
    objective: OptimizationObjective,
    
    /// Portfolio constraints
    constraints: PortfolioConstraints,
    
    /// Risk aversion parameter (for utility maximization)
    risk_aversion: FixedPoint,
    
    /// Target return (for minimum variance optimization)
    target_return: FixedPoint,
    
    /// Historical performance data
    performance_history: Vec<(u64, FixedPoint)>,
    
    /// Stress test scenarios
    stress_scenarios: Vec<StressScenario>,
    
    /// Transaction cost model
    transaction_costs: HashMap<AssetId, FixedPoint>,
}

impl PortfolioOptimizer {
    /// Create a new portfolio optimizer
    pub fn new(
        objective: OptimizationObjective,
        constraints: PortfolioConstraints,
        risk_aversion: FixedPoint,
    ) -> Self {
        Self {
            objective,
            constraints,
            risk_aversion,
            target_return: FixedPoint::from_float(0.08), // 8% default target
            performance_history: Vec::new(),
            stress_scenarios: Self::create_default_stress_scenarios(),
            transaction_costs: HashMap::new(),
        }
    }
    
    /// Optimize portfolio using quadratic programming
    pub fn optimize_portfolio(
        &self,
        current_portfolio: &Portfolio,
        market_data: &MarketData,
        available_assets: &[AssetId],
    ) -> Result<OptimizationResult, RiskError> {
        let n_assets = available_assets.len();
        if n_assets == 0 {
            return Err(RiskError::InvalidParameters("No assets provided".to_string()));
        }
        
        // Build expected return vector
        let expected_returns = self.build_expected_return_vector(available_assets, market_data)?;
        
        // Build covariance matrix
        let covariance_matrix = self.build_covariance_matrix(available_assets, market_data)?;
        
        // Get current weights
        let current_weights = self.get_current_weights(current_portfolio, available_assets);
        
        // Solve optimization problem
        let optimal_weights = match self.objective {
            OptimizationObjective::MaxSharpe => {
                self.solve_max_sharpe(&expected_returns, &covariance_matrix, &current_weights)?
            },
            OptimizationObjective::MinVariance => {
                self.solve_min_variance(&expected_returns, &covariance_matrix, &current_weights)?
            },
            OptimizationObjective::MaxUtility => {
                self.solve_max_utility(&expected_returns, &covariance_matrix, &current_weights)?
            },
            OptimizationObjective::RiskParity => {
                self.solve_risk_parity(&covariance_matrix, &current_weights)?
            },
        };
        
        // Convert to HashMap and apply constraints
        let mut weights_map = HashMap::new();
        for (i, asset_id) in available_assets.iter().enumerate() {
            weights_map.insert(asset_id.clone(), FixedPoint::from_float(optimal_weights[i]));
        }
        
        // Apply constraints
        let constrained_weights = self.apply_constraints(&weights_map, &current_weights)?;
        
        // Calculate portfolio metrics
        let portfolio_metrics = self.calculate_portfolio_metrics(
            &constrained_weights,
            available_assets,
            market_data,
        )?;
        
        // Calculate turnover
        let turnover = self.calculate_turnover(&constrained_weights, &current_weights);
        
        // Check constraint violations
        let violations = self.check_constraint_violations(&constrained_weights, &current_weights);
        
        Ok(OptimizationResult {
            weights: constrained_weights,
            expected_return: portfolio_metrics.0,
            portfolio_volatility: portfolio_metrics.1,
            sharpe_ratio: portfolio_metrics.2,
            max_drawdown_estimate: portfolio_metrics.3,
            diversification_ratio: portfolio_metrics.4,
            turnover,
            objective_value: portfolio_metrics.5,
            constraint_violations: violations,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }
    
    /// Perform stress testing on portfolio
    pub fn stress_test_portfolio(
        &self,
        portfolio_weights: &HashMap<AssetId, FixedPoint>,
        market_data: &MarketData,
    ) -> Result<Vec<StressTestResult>, RiskError> {
        let mut results = Vec::new();
        
        for scenario in &self.stress_scenarios {
            let result = self.run_stress_scenario(portfolio_weights, market_data, scenario)?;
            results.push(result);
        }
        
        // Add Monte Carlo stress tests
        let monte_carlo_results = self.run_monte_carlo_stress_test(portfolio_weights, market_data)?;
        results.extend(monte_carlo_results);
        
        Ok(results)
    }
    
    /// Calculate performance metrics
    pub fn calculate_performance_metrics(
        &self,
        returns: &[FixedPoint],
        benchmark_returns: Option<&[FixedPoint]>,
        risk_free_rate: FixedPoint,
    ) -> Result<PerformanceMetrics, RiskError> {
        if returns.is_empty() {
            return Err(RiskError::InsufficientData("No return data provided".to_string()));
        }
        
        let total_return = returns.iter().fold(FixedPoint::one(), |acc, &r| acc * (FixedPoint::one() + r)) - FixedPoint::one();
        let n_periods = returns.len() as f64;
        let annualized_return = (FixedPoint::one() + total_return).pow((252.0 / n_periods) as u32) - FixedPoint::one();
        
        // Calculate volatility
        let mean_return: FixedPoint = returns.iter().sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32);
        let variance: FixedPoint = returns.iter()
            .map(|&r| (r - mean_return) * (r - mean_return))
            .sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32 - 1);
        let volatility = variance.sqrt() * FixedPoint::from_float(252.0_f64.sqrt()); // Annualized
        
        // Calculate Sharpe ratio
        let sharpe_ratio = if volatility.is_zero() {
            FixedPoint::zero()
        } else {
            (annualized_return - risk_free_rate) / volatility
        };
        
        // Calculate Sortino ratio (downside deviation)
        let downside_returns: Vec<FixedPoint> = returns.iter()
            .filter(|&&r| r < FixedPoint::zero())
            .copied()
            .collect();
        
        let downside_deviation = if downside_returns.is_empty() {
            FixedPoint::zero()
        } else {
            let downside_variance: FixedPoint = downside_returns.iter()
                .map(|&r| r * r)
                .sum::<FixedPoint>() / FixedPoint::from_int(downside_returns.len() as i32);
            downside_variance.sqrt() * FixedPoint::from_float(252.0_f64.sqrt())
        };
        
        let sortino_ratio = if downside_deviation.is_zero() {
            FixedPoint::zero()
        } else {
            (annualized_return - risk_free_rate) / downside_deviation
        };
        
        // Calculate maximum drawdown
        let max_drawdown = self.calculate_max_drawdown(returns);
        
        // Calculate Calmar ratio
        let calmar_ratio = if max_drawdown.is_zero() {
            FixedPoint::zero()
        } else {
            annualized_return / max_drawdown
        };
        
        // Calculate win rate
        let winning_periods = returns.iter().filter(|&&r| r > FixedPoint::zero()).count();
        let win_rate = FixedPoint::from_int(winning_periods as i32) / FixedPoint::from_int(returns.len() as i32);
        
        // Calculate win/loss ratio
        let avg_win: FixedPoint = returns.iter()
            .filter(|&&r| r > FixedPoint::zero())
            .sum::<FixedPoint>() / FixedPoint::from_int(winning_periods.max(1) as i32);
        
        let losing_periods = returns.len() - winning_periods;
        let avg_loss: FixedPoint = returns.iter()
            .filter(|&&r| r < FixedPoint::zero())
            .sum::<FixedPoint>() / FixedPoint::from_int(losing_periods.max(1) as i32);
        
        let win_loss_ratio = if avg_loss.is_zero() {
            FixedPoint::zero()
        } else {
            avg_win / (-avg_loss)
        };
        
        // Calculate Information Ratio (vs benchmark)
        let information_ratio = if let Some(benchmark) = benchmark_returns {
            if benchmark.len() == returns.len() {
                let excess_returns: Vec<FixedPoint> = returns.iter()
                    .zip(benchmark.iter())
                    .map(|(&r, &b)| r - b)
                    .collect();
                
                let tracking_error = self.calculate_tracking_error(&excess_returns);
                let excess_return: FixedPoint = excess_returns.iter().sum::<FixedPoint>() / 
                    FixedPoint::from_int(excess_returns.len() as i32);
                
                if tracking_error.is_zero() {
                    FixedPoint::zero()
                } else {
                    excess_return / tracking_error
                }
            } else {
                FixedPoint::zero()
            }
        } else {
            FixedPoint::zero()
        };
        
        Ok(PerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            sortino_ratio,
            calmar_ratio,
            max_drawdown,
            win_rate,
            win_loss_ratio,
            information_ratio,
        })
    }
    
    /// Solve maximum Sharpe ratio optimization
    fn solve_max_sharpe(
        &self,
        expected_returns: &DVector<f64>,
        covariance_matrix: &DMatrix<f64>,
        _current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Result<DVector<f64>, RiskError> {
        // For maximum Sharpe ratio: w* = Σ⁻¹(μ-r) / 1'Σ⁻¹(μ-r)
        let risk_free_vec = DVector::from_element(expected_returns.len(), 0.02); // 2% risk-free rate
        let excess_returns = expected_returns - risk_free_vec;
        
        let inv_cov = covariance_matrix.try_inverse()
            .ok_or_else(|| RiskError::MatrixError("Covariance matrix is not invertible".to_string()))?;
        
        let numerator = &inv_cov * &excess_returns;
        let denominator = DVector::from_element(expected_returns.len(), 1.0).dot(&numerator);
        
        if denominator.abs() < 1e-10 {
            return Err(RiskError::ComputationError("Denominator too small in Sharpe optimization".to_string()));
        }
        
        Ok(numerator / denominator)
    }
    
    /// Solve minimum variance optimization
    fn solve_min_variance(
        &self,
        _expected_returns: &DVector<f64>,
        covariance_matrix: &DMatrix<f64>,
        _current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Result<DVector<f64>, RiskError> {
        // For minimum variance: w* = Σ⁻¹1 / 1'Σ⁻¹1
        let inv_cov = covariance_matrix.try_inverse()
            .ok_or_else(|| RiskError::MatrixError("Covariance matrix is not invertible".to_string()))?;
        
        let ones = DVector::from_element(covariance_matrix.nrows(), 1.0);
        let numerator = &inv_cov * &ones;
        let denominator = ones.dot(&numerator);
        
        if denominator.abs() < 1e-10 {
            return Err(RiskError::ComputationError("Denominator too small in minimum variance optimization".to_string()));
        }
        
        Ok(numerator / denominator)
    }
    
    /// Solve maximum utility optimization
    fn solve_max_utility(
        &self,
        expected_returns: &DVector<f64>,
        covariance_matrix: &DMatrix<f64>,
        _current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Result<DVector<f64>, RiskError> {
        // For maximum utility: w* = (1/γ)Σ⁻¹μ
        let inv_cov = covariance_matrix.try_inverse()
            .ok_or_else(|| RiskError::MatrixError("Covariance matrix is not invertible".to_string()))?;
        
        let weights = &inv_cov * expected_returns / self.risk_aversion.to_float();
        
        // Normalize to sum to 1
        let weight_sum = weights.sum();
        if weight_sum.abs() < 1e-10 {
            return Err(RiskError::ComputationError("Weight sum too small in utility optimization".to_string()));
        }
        
        Ok(weights / weight_sum)
    }
    
    /// Solve risk parity optimization
    fn solve_risk_parity(
        &self,
        covariance_matrix: &DMatrix<f64>,
        _current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Result<DVector<f64>, RiskError> {
        let n = covariance_matrix.nrows();
        
        // Start with equal weights
        let mut weights = DVector::from_element(n, 1.0 / n as f64);
        
        // Iterative algorithm for risk parity
        for _ in 0..100 { // Maximum iterations
            let portfolio_vol = (weights.transpose() * covariance_matrix * &weights).sqrt();
            let marginal_risk = covariance_matrix * &weights / portfolio_vol;
            let risk_contributions = weights.component_mul(&marginal_risk);
            
            // Target risk contribution is 1/n for each asset
            let target_risk = portfolio_vol / n as f64;
            
            // Update weights based on risk contribution deviation
            for i in 0..n {
                let adjustment = target_risk / risk_contributions[i];
                weights[i] *= adjustment.sqrt();
            }
            
            // Normalize weights
            let weight_sum = weights.sum();
            weights /= weight_sum;
            
            // Check convergence
            let max_deviation = risk_contributions.iter()
                .map(|&rc| (rc - target_risk).abs())
                .fold(0.0, f64::max);
            
            if max_deviation < 1e-6 {
                break;
            }
        }
        
        Ok(weights)
    }
    
    /// Apply portfolio constraints
    fn apply_constraints(
        &self,
        weights: &HashMap<AssetId, FixedPoint>,
        _current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Result<HashMap<AssetId, FixedPoint>, RiskError> {
        let mut constrained_weights = weights.clone();
        
        // Apply individual asset weight constraints
        for (_, weight) in constrained_weights.iter_mut() {
            *weight = weight.max(self.constraints.min_weight_per_asset);
            *weight = weight.min(self.constraints.max_weight_per_asset);
        }
        
        // Apply leverage constraint
        let total_abs_weight: FixedPoint = constrained_weights.values()
            .map(|w| w.abs())
            .sum();
        
        if total_abs_weight > self.constraints.max_leverage {
            let scale_factor = self.constraints.max_leverage / total_abs_weight;
            for weight in constrained_weights.values_mut() {
                *weight = *weight * scale_factor;
            }
        }
        
        // Normalize to sum to 1 (for long-only portfolios)
        let weight_sum: FixedPoint = constrained_weights.values().sum();
        if !weight_sum.is_zero() && weight_sum != FixedPoint::one() {
            for weight in constrained_weights.values_mut() {
                *weight = *weight / weight_sum;
            }
        }
        
        Ok(constrained_weights)
    }
    
    /// Build expected return vector
    fn build_expected_return_vector(
        &self,
        assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<DVector<f64>, RiskError> {
        let mut returns = Vec::with_capacity(assets.len());
        
        for asset_id in assets {
            let asset_returns = market_data.returns.get(asset_id)
                .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", asset_id)))?;
            
            if asset_returns.is_empty() {
                returns.push(0.0);
            } else {
                let mean_return: FixedPoint = asset_returns.iter().sum::<FixedPoint>() / 
                    FixedPoint::from_int(asset_returns.len() as i32);
                returns.push(mean_return.to_float());
            }
        }
        
        Ok(DVector::from_vec(returns))
    }
    
    /// Build covariance matrix
    fn build_covariance_matrix(
        &self,
        assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<DMatrix<f64>, RiskError> {
        let n = assets.len();
        let mut covariance_matrix = DMatrix::zeros(n, n);
        
        for (i, asset_i) in assets.iter().enumerate() {
            for (j, asset_j) in assets.iter().enumerate() {
                let vol_i = market_data.volatilities.get(asset_i)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_i)))?;
                let vol_j = market_data.volatilities.get(asset_j)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_j)))?;
                
                let correlation = if i == j {
                    FixedPoint::one()
                } else {
                    market_data.correlations.get(&(asset_i.clone(), asset_j.clone()))
                        .copied()
                        .unwrap_or(FixedPoint::zero())
                };
                
                let covariance = correlation * vol_i * vol_j;
                covariance_matrix[(i, j)] = covariance.to_float();
            }
        }
        
        Ok(covariance_matrix)
    }
    
    /// Get current portfolio weights
    fn get_current_weights(&self, portfolio: &Portfolio, assets: &[AssetId]) -> HashMap<AssetId, FixedPoint> {
        let mut weights = HashMap::new();
        let total_value = portfolio.total_market_value();
        
        for asset_id in assets {
            let weight = if let Some(position) = portfolio.positions.get(asset_id) {
                if total_value.is_zero() {
                    FixedPoint::zero()
                } else {
                    position.market_value() / total_value
                }
            } else {
                FixedPoint::zero()
            };
            weights.insert(asset_id.clone(), weight);
        }
        
        weights
    }
    
    /// Calculate portfolio metrics
    fn calculate_portfolio_metrics(
        &self,
        weights: &HashMap<AssetId, FixedPoint>,
        assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint, FixedPoint, FixedPoint, FixedPoint, FixedPoint), RiskError> {
        let mut portfolio_return = FixedPoint::zero();
        let mut portfolio_variance = FixedPoint::zero();
        let mut individual_vol_sum = FixedPoint::zero();
        
        // Calculate expected portfolio return
        for (asset_id, &weight) in weights {
            let asset_returns = market_data.returns.get(asset_id)
                .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", asset_id)))?;
            
            let expected_return = if asset_returns.is_empty() {
                FixedPoint::zero()
            } else {
                asset_returns.iter().sum::<FixedPoint>() / FixedPoint::from_int(asset_returns.len() as i32)
            };
            
            portfolio_return = portfolio_return + weight * expected_return;
            
            let volatility = market_data.volatilities.get(asset_id)
                .copied()
                .unwrap_or(FixedPoint::zero());
            individual_vol_sum = individual_vol_sum + weight.abs() * volatility;
        }
        
        // Calculate portfolio variance
        for (asset_i, &weight_i) in weights {
            for (asset_j, &weight_j) in weights {
                let vol_i = market_data.volatilities.get(asset_i)
                    .copied()
                    .unwrap_or(FixedPoint::zero());
                let vol_j = market_data.volatilities.get(asset_j)
                    .copied()
                    .unwrap_or(FixedPoint::zero());
                
                let correlation = if asset_i == asset_j {
                    FixedPoint::one()
                } else {
                    market_data.correlations.get(&(asset_i.clone(), asset_j.clone()))
                        .copied()
                        .unwrap_or(FixedPoint::zero())
                };
                
                let covariance = correlation * vol_i * vol_j;
                portfolio_variance = portfolio_variance + weight_i * weight_j * covariance;
            }
        }
        
        let portfolio_volatility = portfolio_variance.sqrt();
        let risk_free_rate = FixedPoint::from_float(0.02);
        
        let sharpe_ratio = if portfolio_volatility.is_zero() {
            FixedPoint::zero()
        } else {
            (portfolio_return - risk_free_rate) / portfolio_volatility
        };
        
        // Estimate maximum drawdown (simplified)
        let max_drawdown_estimate = portfolio_volatility * FixedPoint::from_float(2.5); // Rough estimate
        
        // Diversification benefit
        let diversification_ratio = if portfolio_volatility.is_zero() {
            FixedPoint::one()
        } else {
            individual_vol_sum / portfolio_volatility
        };
        
        // Objective value depends on optimization type
        let objective_value = match self.objective {
            OptimizationObjective::MaxSharpe => sharpe_ratio,
            OptimizationObjective::MinVariance => -portfolio_variance,
            OptimizationObjective::MaxUtility => portfolio_return - self.risk_aversion * portfolio_variance / FixedPoint::from_float(2.0),
            OptimizationObjective::RiskParity => -portfolio_variance, // Minimize variance for risk parity
        };
        
        Ok((portfolio_return, portfolio_volatility, sharpe_ratio, max_drawdown_estimate, diversification_ratio, objective_value))
    }
    
    /// Calculate turnover
    fn calculate_turnover(
        &self,
        new_weights: &HashMap<AssetId, FixedPoint>,
        current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> FixedPoint {
        let mut turnover = FixedPoint::zero();
        
        // Get all assets
        let mut all_assets = std::collections::HashSet::new();
        all_assets.extend(new_weights.keys());
        all_assets.extend(current_weights.keys());
        
        for asset_id in all_assets {
            let new_weight = new_weights.get(asset_id).copied().unwrap_or(FixedPoint::zero());
            let current_weight = current_weights.get(asset_id).copied().unwrap_or(FixedPoint::zero());
            turnover = turnover + (new_weight - current_weight).abs();
        }
        
        turnover / FixedPoint::from_float(2.0) // Divide by 2 for one-way turnover
    }
    
    /// Check constraint violations
    fn check_constraint_violations(
        &self,
        weights: &HashMap<AssetId, FixedPoint>,
        current_weights: &HashMap<AssetId, FixedPoint>,
    ) -> Vec<String> {
        let mut violations = Vec::new();
        
        // Check individual weight constraints
        for (asset_id, &weight) in weights {
            if weight > self.constraints.max_weight_per_asset {
                violations.push(format!("Asset {} exceeds maximum weight: {:.2}%", 
                    asset_id, weight.to_float() * 100.0));
            }
            if weight < self.constraints.min_weight_per_asset {
                violations.push(format!("Asset {} below minimum weight: {:.2}%", 
                    asset_id, weight.to_float() * 100.0));
            }
        }
        
        // Check leverage constraint
        let total_leverage: FixedPoint = weights.values().map(|w| w.abs()).sum();
        if total_leverage > self.constraints.max_leverage {
            violations.push(format!("Total leverage exceeds limit: {:.2}x", total_leverage.to_float()));
        }
        
        // Check turnover constraint
        let turnover = self.calculate_turnover(weights, current_weights);
        if turnover > self.constraints.max_turnover {
            violations.push(format!("Turnover exceeds limit: {:.2}%", turnover.to_float() * 100.0));
        }
        
        violations
    }
    
    /// Run stress scenario
    fn run_stress_scenario(
        &self,
        portfolio_weights: &HashMap<AssetId, FixedPoint>,
        market_data: &MarketData,
        scenario: &StressScenario,
    ) -> Result<StressTestResult, RiskError> {
        let mut portfolio_loss = FixedPoint::zero();
        let mut worst_loss = FixedPoint::zero();
        let mut best_gain = FixedPoint::zero();
        let mut worst_asset = String::new();
        let mut best_asset = String::new();
        
        for (asset_id, &weight) in portfolio_weights {
            let base_return = market_data.returns.get(asset_id)
                .and_then(|returns| returns.last())
                .copied()
                .unwrap_or(FixedPoint::zero());
            
            let shock = scenario.return_shocks.get(asset_id)
                .copied()
                .unwrap_or(FixedPoint::zero());
            
            let stressed_return = base_return + shock;
            let asset_loss = weight * stressed_return;
            portfolio_loss = portfolio_loss + asset_loss;
            
            if asset_loss < worst_loss {
                worst_loss = asset_loss;
                worst_asset = asset_id.clone();
            }
            
            if asset_loss > best_gain {
                best_gain = asset_loss;
                best_asset = asset_id.clone();
            }
        }
        
        let portfolio_value = portfolio_weights.values().sum::<FixedPoint>();
        let loss_percentage = if portfolio_value.is_zero() {
            FixedPoint::zero()
        } else {
            portfolio_loss / portfolio_value
        };
        
        // Check for correlation breakdown (simplified)
        let correlation_breakdown = scenario.correlation_adjustments.values()
            .any(|&adj| adj.abs() > FixedPoint::from_float(0.3));
        
        Ok(StressTestResult {
            scenario_name: scenario.name.clone(),
            portfolio_loss: -portfolio_loss, // Convert to positive loss
            loss_percentage: -loss_percentage,
            worst_asset,
            best_asset,
            correlation_breakdown,
        })
    }
    
    /// Run Monte Carlo stress test
    fn run_monte_carlo_stress_test(
        &self,
        portfolio_weights: &HashMap<AssetId, FixedPoint>,
        market_data: &MarketData,
    ) -> Result<Vec<StressTestResult>, RiskError> {
        let mut rng = StdRng::from_entropy();
        let mut results = Vec::new();
        
        let n_simulations = 1000;
        let mut losses = Vec::with_capacity(n_simulations);
        
        for _ in 0..n_simulations {
            let mut portfolio_loss = FixedPoint::zero();
            
            for (asset_id, &weight) in portfolio_weights {
                let volatility = market_data.volatilities.get(asset_id)
                    .copied()
                    .unwrap_or(FixedPoint::from_float(0.2));
                
                let normal = Normal::new(0.0, volatility.to_float()).unwrap();
                let random_return = FixedPoint::from_float(normal.sample(&mut rng));
                
                portfolio_loss = portfolio_loss + weight * random_return;
            }
            
            losses.push(portfolio_loss);
        }
        
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Create results for different percentiles
        let percentiles = [0.01, 0.05, 0.1]; // 1%, 5%, 10% worst cases
        
        for &percentile in &percentiles {
            let index = (percentile * n_simulations as f64) as usize;
            let loss = -losses[index]; // Convert to positive loss
            
            results.push(StressTestResult {
                scenario_name: format!("Monte Carlo {}% VaR", (1.0 - percentile) * 100.0),
                portfolio_loss: loss,
                loss_percentage: loss, // Assuming normalized weights
                worst_asset: "N/A".to_string(),
                best_asset: "N/A".to_string(),
                correlation_breakdown: false,
            });
        }
        
        Ok(results)
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[FixedPoint]) -> FixedPoint {
        if returns.is_empty() {
            return FixedPoint::zero();
        }
        
        let mut cumulative_return = FixedPoint::one();
        let mut peak = FixedPoint::one();
        let mut max_drawdown = FixedPoint::zero();
        
        for &return_val in returns {
            cumulative_return = cumulative_return * (FixedPoint::one() + return_val);
            
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            
            let drawdown = (peak - cumulative_return) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
    
    /// Calculate tracking error
    fn calculate_tracking_error(&self, excess_returns: &[FixedPoint]) -> FixedPoint {
        if excess_returns.is_empty() {
            return FixedPoint::zero();
        }
        
        let mean_excess: FixedPoint = excess_returns.iter().sum::<FixedPoint>() / 
            FixedPoint::from_int(excess_returns.len() as i32);
        
        let variance: FixedPoint = excess_returns.iter()
            .map(|&r| (r - mean_excess) * (r - mean_excess))
            .sum::<FixedPoint>() / FixedPoint::from_int(excess_returns.len() as i32 - 1);
        
        variance.sqrt()
    }
    
    /// Create default stress scenarios
    fn create_default_stress_scenarios() -> Vec<StressScenario> {
        vec![
            StressScenario {
                name: "2008 Financial Crisis".to_string(),
                return_shocks: [
                    ("SPY".to_string(), FixedPoint::from_float(-0.37)),
                    ("QQQ".to_string(), FixedPoint::from_float(-0.42)),
                    ("IWM".to_string(), FixedPoint::from_float(-0.34)),
                ].iter().cloned().collect(),
                volatility_multipliers: HashMap::new(),
                correlation_adjustments: HashMap::new(),
                probability: FixedPoint::from_float(0.01),
            },
            StressScenario {
                name: "COVID-19 Crash".to_string(),
                return_shocks: [
                    ("SPY".to_string(), FixedPoint::from_float(-0.34)),
                    ("QQQ".to_string(), FixedPoint::from_float(-0.30)),
                    ("IWM".to_string(), FixedPoint::from_float(-0.42)),
                ].iter().cloned().collect(),
                volatility_multipliers: HashMap::new(),
                correlation_adjustments: HashMap::new(),
                probability: FixedPoint::from_float(0.005),
            },
            StressScenario {
                name: "Interest Rate Shock".to_string(),
                return_shocks: [
                    ("TLT".to_string(), FixedPoint::from_float(-0.25)),
                    ("REIT".to_string(), FixedPoint::from_float(-0.20)),
                    ("UTIL".to_string(), FixedPoint::from_float(-0.15)),
                ].iter().cloned().collect(),
                volatility_multipliers: HashMap::new(),
                correlation_adjustments: HashMap::new(),
                probability: FixedPoint::from_float(0.02),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::{Portfolio, Position, MarketData};
    
    fn create_test_portfolio() -> Portfolio {
        let mut portfolio = Portfolio::new(FixedPoint::from_float(10000.0));
        
        portfolio.add_position(Position::new(
            "AAPL".to_string(),
            100,
            FixedPoint::from_float(150.0),
            FixedPoint::from_float(155.0),
        ));
        
        portfolio.add_position(Position::new(
            "GOOGL".to_string(),
            50,
            FixedPoint::from_float(2800.0),
            FixedPoint::from_float(2850.0),
        ));
        
        portfolio
    }
    
    fn create_test_market_data() -> MarketData {
        let mut market_data = MarketData::new();
        
        let aapl_returns = vec![
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(-0.02),
            FixedPoint::from_float(0.015),
            FixedPoint::from_float(-0.01),
            FixedPoint::from_float(0.005),
        ];
        
        let googl_returns = vec![
            FixedPoint::from_float(0.012),
            FixedPoint::from_float(-0.018),
            FixedPoint::from_float(0.02),
            FixedPoint::from_float(-0.008),
            FixedPoint::from_float(0.003),
        ];
        
        market_data.add_returns("AAPL".to_string(), aapl_returns);
        market_data.add_returns("GOOGL".to_string(), googl_returns);
        
        market_data.add_volatility("AAPL".to_string(), FixedPoint::from_float(0.25));
        market_data.add_volatility("GOOGL".to_string(), FixedPoint::from_float(0.30));
        
        market_data.add_correlation("AAPL".to_string(), "GOOGL".to_string(), FixedPoint::from_float(0.6));
        
        market_data
    }
    
    #[test]
    fn test_portfolio_optimization() {
        let optimizer = PortfolioOptimizer::new(
            OptimizationObjective::MaxSharpe,
            PortfolioConstraints::default(),
            FixedPoint::from_float(2.0),
        );
        
        let portfolio = create_test_portfolio();
        let market_data = create_test_market_data();
        let assets = vec!["AAPL".to_string(), "GOOGL".to_string()];
        
        let result = optimizer.optimize_portfolio(&portfolio, &market_data, &assets).unwrap();
        
        assert_eq!(result.weights.len(), 2);
        assert!(result.sharpe_ratio >= FixedPoint::zero());
        assert!(result.portfolio_volatility >= FixedPoint::zero());
    }
    
    #[test]
    fn test_stress_testing() {
        let optimizer = PortfolioOptimizer::new(
            OptimizationObjective::MaxSharpe,
            PortfolioConstraints::default(),
            FixedPoint::from_float(2.0),
        );
        
        let weights = [
            ("AAPL".to_string(), FixedPoint::from_float(0.6)),
            ("GOOGL".to_string(), FixedPoint::from_float(0.4)),
        ].iter().cloned().collect();
        
        let market_data = create_test_market_data();
        
        let stress_results = optimizer.stress_test_portfolio(&weights, &market_data).unwrap();
        
        assert!(!stress_results.is_empty());
        for result in stress_results {
            assert!(result.portfolio_loss >= FixedPoint::zero());
        }
    }
    
    #[test]
    fn test_performance_metrics() {
        let optimizer = PortfolioOptimizer::new(
            OptimizationObjective::MaxSharpe,
            PortfolioConstraints::default(),
            FixedPoint::from_float(2.0),
        );
        
        let returns = vec![
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(-0.02),
            FixedPoint::from_float(0.015),
            FixedPoint::from_float(-0.01),
            FixedPoint::from_float(0.005),
        ];
        
        let metrics = optimizer.calculate_performance_metrics(
            &returns,
            None,
            FixedPoint::from_float(0.02),
        ).unwrap();
        
        assert!(metrics.volatility >= FixedPoint::zero());
        assert!(metrics.max_drawdown >= FixedPoint::zero());
        assert!(metrics.win_rate >= FixedPoint::zero());
        assert!(metrics.win_rate <= FixedPoint::one());
    }
}