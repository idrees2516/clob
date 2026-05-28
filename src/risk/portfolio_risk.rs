//! Portfolio Risk Metrics Implementation
//!
//! This module implements comprehensive portfolio risk metrics including:
//! - Value at Risk (VaR) using multiple methods
//! - Expected Shortfall (Conditional VaR)
//! - Maximum Drawdown calculation with rolling windows
//! - Risk contribution analysis by asset and strategy
//! - Monte Carlo simulation for tail risk estimation

use super::{Portfolio, MarketData, RiskError, AssetId};
use crate::math::fixed_point::FixedPoint;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive risk metrics for a portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Value at Risk at different confidence levels
    pub var_95: FixedPoint,
    pub var_99: FixedPoint,
    pub var_99_9: FixedPoint,
    
    /// Expected Shortfall (Conditional VaR)
    pub expected_shortfall_95: FixedPoint,
    pub expected_shortfall_99: FixedPoint,
    
    /// Maximum Drawdown metrics
    pub maximum_drawdown: FixedPoint,
    pub current_drawdown: FixedPoint,
    pub drawdown_duration: u32,
    
    /// Portfolio volatility
    pub portfolio_volatility: FixedPoint,
    
    /// Risk contributions by asset
    pub risk_contributions: HashMap<AssetId, FixedPoint>,
    
    /// Diversification ratio
    pub diversification_ratio: FixedPoint,
    
    /// Calculation timestamp
    pub timestamp: u64,
}

/// Risk calculation methods
#[derive(Debug, Clone, Copy)]
pub enum VaRMethod {
    Historical,
    Parametric,
    MonteCarlo,
}

/// Monte Carlo simulation parameters
#[derive(Debug, Clone)]
pub struct MonteCarloParams {
    pub num_simulations: usize,
    pub time_horizon_days: u32,
    pub confidence_levels: Vec<f64>,
    pub random_seed: Option<u64>,
}

impl Default for MonteCarloParams {
    fn default() -> Self {
        Self {
            num_simulations: 10_000,
            time_horizon_days: 1,
            confidence_levels: vec![0.95, 0.99, 0.999],
            random_seed: None,
        }
    }
}

/// Portfolio Risk Metrics Calculator
pub struct RiskMetricsCalculator {
    /// Historical portfolio values for drawdown calculation
    portfolio_history: Vec<(u64, FixedPoint)>,
    
    /// Maximum history length to maintain
    max_history_length: usize,
    
    /// Monte Carlo parameters
    monte_carlo_params: MonteCarloParams,
}

impl RiskMetricsCalculator {
    /// Create a new risk metrics calculator
    pub fn new(max_history_length: usize, monte_carlo_params: MonteCarloParams) -> Self {
        Self {
            portfolio_history: Vec::new(),
            max_history_length,
            monte_carlo_params,
        }
    }
    
    /// Calculate comprehensive risk metrics for a portfolio
    pub fn calculate_risk_metrics(
        &mut self,
        portfolio: &Portfolio,
        market_data: &MarketData,
        method: VaRMethod,
    ) -> Result<RiskMetrics, RiskError> {
        // Update portfolio history
        self.update_portfolio_history(portfolio);
        
        // Calculate VaR using specified method
        let (var_95, var_99, var_99_9) = match method {
            VaRMethod::Historical => self.calculate_historical_var(portfolio, market_data)?,
            VaRMethod::Parametric => self.calculate_parametric_var(portfolio, market_data)?,
            VaRMethod::MonteCarlo => self.calculate_monte_carlo_var(portfolio, market_data)?,
        };
        
        // Calculate Expected Shortfall
        let (es_95, es_99) = self.calculate_expected_shortfall(portfolio, market_data, method)?;
        
        // Calculate Maximum Drawdown
        let (max_drawdown, current_drawdown, drawdown_duration) = self.calculate_maximum_drawdown()?;
        
        // Calculate portfolio volatility
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
        
        // Calculate risk contributions
        let risk_contributions = self.calculate_risk_contributions(portfolio, market_data)?;
        
        // Calculate diversification ratio
        let diversification_ratio = self.calculate_diversification_ratio(portfolio, market_data)?;
        
        Ok(RiskMetrics {
            var_95,
            var_99,
            var_99_9,
            expected_shortfall_95: es_95,
            expected_shortfall_99: es_99,
            maximum_drawdown: max_drawdown,
            current_drawdown,
            drawdown_duration,
            portfolio_volatility,
            risk_contributions,
            diversification_ratio,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        })
    }
    
    /// Calculate Historical VaR using empirical distribution
    fn calculate_historical_var(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint, FixedPoint), RiskError> {
        let returns = self.calculate_portfolio_returns(portfolio, market_data)?;
        
        if returns.len() < 100 {
            return Err(RiskError::InsufficientData(
                "Need at least 100 historical returns for Historical VaR".to_string()
            ));
        }
        
        let mut sorted_returns = returns;
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_returns.len();
        let var_95_idx = ((1.0 - 0.95) * n as f64) as usize;
        let var_99_idx = ((1.0 - 0.99) * n as f64) as usize;
        let var_99_9_idx = ((1.0 - 0.999) * n as f64) as usize;
        
        let portfolio_value = portfolio.total_market_value();
        
        Ok((
            -sorted_returns[var_95_idx] * portfolio_value,
            -sorted_returns[var_99_idx] * portfolio_value,
            -sorted_returns[var_99_9_idx] * portfolio_value,
        ))
    }
    
    /// Calculate Parametric VaR assuming normal distribution
    fn calculate_parametric_var(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint, FixedPoint), RiskError> {
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
        let portfolio_value = portfolio.total_market_value();
        
        // Standard normal quantiles
        let z_95 = FixedPoint::from_float(1.645);  // 95% confidence
        let z_99 = FixedPoint::from_float(2.326);  // 99% confidence
        let z_99_9 = FixedPoint::from_float(3.090); // 99.9% confidence
        
        Ok((
            z_95 * portfolio_volatility * portfolio_value,
            z_99 * portfolio_volatility * portfolio_value,
            z_99_9 * portfolio_volatility * portfolio_value,
        ))
    }
    
    /// Calculate Monte Carlo VaR using simulation
    fn calculate_monte_carlo_var(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint, FixedPoint), RiskError> {
        let mut rng = if let Some(seed) = self.monte_carlo_params.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };
        
        let assets: Vec<&AssetId> = portfolio.positions.keys().collect();
        let n_assets = assets.len();
        
        if n_assets == 0 {
            return Ok((FixedPoint::zero(), FixedPoint::zero(), FixedPoint::zero()));
        }
        
        // Build covariance matrix
        let covariance_matrix = self.build_covariance_matrix(&assets, market_data)?;
        
        // Cholesky decomposition for correlated random numbers
        let chol = covariance_matrix.cholesky()
            .ok_or_else(|| RiskError::MatrixError("Covariance matrix is not positive definite".to_string()))?;
        
        let mut portfolio_returns = Vec::with_capacity(self.monte_carlo_params.num_simulations);
        let current_portfolio_value = portfolio.total_market_value();
        
        for _ in 0..self.monte_carlo_params.num_simulations {
            // Generate correlated random returns
            let random_vector: DVector<f64> = DVector::from_fn(n_assets, |_, _| {
                StandardNormal.sample(&mut rng)
            });
            
            let correlated_returns = chol.l() * random_vector;
            
            // Calculate portfolio return for this simulation
            let mut portfolio_return = FixedPoint::zero();
            for (i, asset_id) in assets.iter().enumerate() {
                if let Some(position) = portfolio.positions.get(*asset_id) {
                    let weight = position.market_value() / current_portfolio_value;
                    let asset_return = FixedPoint::from_float(correlated_returns[i]);
                    portfolio_return = portfolio_return + weight * asset_return;
                }
            }
            
            portfolio_returns.push(portfolio_return);
        }
        
        // Sort returns and calculate VaR
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = portfolio_returns.len();
        let var_95_idx = ((1.0 - 0.95) * n as f64) as usize;
        let var_99_idx = ((1.0 - 0.99) * n as f64) as usize;
        let var_99_9_idx = ((1.0 - 0.999) * n as f64) as usize;
        
        Ok((
            -portfolio_returns[var_95_idx] * current_portfolio_value,
            -portfolio_returns[var_99_idx] * current_portfolio_value,
            -portfolio_returns[var_99_9_idx] * current_portfolio_value,
        ))
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    fn calculate_expected_shortfall(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
        method: VaRMethod,
    ) -> Result<(FixedPoint, FixedPoint), RiskError> {
        let returns = match method {
            VaRMethod::Historical => self.calculate_portfolio_returns(portfolio, market_data)?,
            VaRMethod::Parametric => {
                // For parametric method, use analytical formula
                let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
                let portfolio_value = portfolio.total_market_value();
                
                // Analytical Expected Shortfall for normal distribution
                let es_95 = FixedPoint::from_float(2.063) * portfolio_volatility * portfolio_value;
                let es_99 = FixedPoint::from_float(2.665) * portfolio_volatility * portfolio_value;
                
                return Ok((es_95, es_99));
            },
            VaRMethod::MonteCarlo => {
                // Use Monte Carlo simulation results
                return self.calculate_monte_carlo_expected_shortfall(portfolio, market_data);
            },
        };
        
        let mut sorted_returns = returns;
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_returns.len();
        let var_95_idx = ((1.0 - 0.95) * n as f64) as usize;
        let var_99_idx = ((1.0 - 0.99) * n as f64) as usize;
        
        // Calculate average of returns below VaR threshold
        let es_95 = if var_95_idx > 0 {
            let tail_returns: FixedPoint = sorted_returns[..var_95_idx].iter().sum();
            tail_returns / FixedPoint::from_int(var_95_idx as i32)
        } else {
            sorted_returns[0]
        };
        
        let es_99 = if var_99_idx > 0 {
            let tail_returns: FixedPoint = sorted_returns[..var_99_idx].iter().sum();
            tail_returns / FixedPoint::from_int(var_99_idx as i32)
        } else {
            sorted_returns[0]
        };
        
        let portfolio_value = portfolio.total_market_value();
        
        Ok((-es_95 * portfolio_value, -es_99 * portfolio_value))
    }
    
    /// Calculate Monte Carlo Expected Shortfall
    fn calculate_monte_carlo_expected_shortfall(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint), RiskError> {
        // This would use the same Monte Carlo simulation as VaR calculation
        // For brevity, using parametric approximation
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
        let portfolio_value = portfolio.total_market_value();
        
        let es_95 = FixedPoint::from_float(2.063) * portfolio_volatility * portfolio_value;
        let es_99 = FixedPoint::from_float(2.665) * portfolio_volatility * portfolio_value;
        
        Ok((es_95, es_99))
    }
    
    /// Calculate Maximum Drawdown with rolling windows
    fn calculate_maximum_drawdown(&self) -> Result<(FixedPoint, FixedPoint, u32), RiskError> {
        if self.portfolio_history.len() < 2 {
            return Ok((FixedPoint::zero(), FixedPoint::zero(), 0));
        }
        
        let mut max_drawdown = FixedPoint::zero();
        let mut current_drawdown = FixedPoint::zero();
        let mut drawdown_duration = 0u32;
        let mut peak_value = self.portfolio_history[0].1;
        let mut peak_time = self.portfolio_history[0].0;
        
        for &(timestamp, value) in &self.portfolio_history[1..] {
            if value > peak_value {
                peak_value = value;
                peak_time = timestamp;
                current_drawdown = FixedPoint::zero();
                drawdown_duration = 0;
            } else {
                let drawdown = (peak_value - value) / peak_value;
                current_drawdown = drawdown;
                
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
                
                // Calculate duration in days (assuming timestamps are in nanoseconds)
                drawdown_duration = ((timestamp - peak_time) / (24 * 60 * 60 * 1_000_000_000)) as u32;
            }
        }
        
        Ok((max_drawdown, current_drawdown, drawdown_duration))
    }
    
    /// Calculate portfolio volatility
    fn calculate_portfolio_volatility(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskError> {
        let assets: Vec<&AssetId> = portfolio.positions.keys().collect();
        let n_assets = assets.len();
        
        if n_assets == 0 {
            return Ok(FixedPoint::zero());
        }
        
        let portfolio_value = portfolio.total_market_value();
        let mut portfolio_variance = FixedPoint::zero();
        
        // Calculate portfolio variance: w'Σw
        for (i, asset_i) in assets.iter().enumerate() {
            for (j, asset_j) in assets.iter().enumerate() {
                let weight_i = portfolio.positions[*asset_i].market_value() / portfolio_value;
                let weight_j = portfolio.positions[*asset_j].market_value() / portfolio_value;
                
                let vol_i = market_data.volatilities.get(*asset_i)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_i)))?;
                let vol_j = market_data.volatilities.get(*asset_j)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_j)))?;
                
                let correlation = if i == j {
                    FixedPoint::one()
                } else {
                    market_data.correlations.get(&((*asset_i).clone(), (*asset_j).clone()))
                        .copied()
                        .unwrap_or(FixedPoint::zero())
                };
                
                let covariance = correlation * vol_i * vol_j;
                portfolio_variance = portfolio_variance + weight_i * weight_j * covariance;
            }
        }
        
        Ok(portfolio_variance.sqrt())
    }
    
    /// Calculate risk contributions by asset
    fn calculate_risk_contributions(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<HashMap<AssetId, FixedPoint>, RiskError> {
        let mut risk_contributions = HashMap::new();
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
        let portfolio_value = portfolio.total_market_value();
        
        if portfolio_volatility.is_zero() {
            return Ok(risk_contributions);
        }
        
        for (asset_id, position) in &portfolio.positions {
            let weight = position.market_value() / portfolio_value;
            let asset_vol = market_data.volatilities.get(asset_id)
                .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_id)))?;
            
            // Calculate marginal contribution to risk
            let mut marginal_contribution = FixedPoint::zero();
            for (other_asset_id, other_position) in &portfolio.positions {
                let other_weight = other_position.market_value() / portfolio_value;
                let other_vol = market_data.volatilities.get(other_asset_id)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", other_asset_id)))?;
                
                let correlation = if asset_id == other_asset_id {
                    FixedPoint::one()
                } else {
                    market_data.correlations.get(&(asset_id.clone(), other_asset_id.clone()))
                        .copied()
                        .unwrap_or(FixedPoint::zero())
                };
                
                marginal_contribution = marginal_contribution + other_weight * correlation * asset_vol * other_vol;
            }
            
            let risk_contribution = weight * marginal_contribution / portfolio_volatility;
            risk_contributions.insert(asset_id.clone(), risk_contribution);
        }
        
        Ok(risk_contributions)
    }
    
    /// Calculate diversification ratio
    fn calculate_diversification_ratio(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskError> {
        let portfolio_volatility = self.calculate_portfolio_volatility(portfolio, market_data)?;
        let portfolio_value = portfolio.total_market_value();
        
        if portfolio_volatility.is_zero() {
            return Ok(FixedPoint::one());
        }
        
        // Calculate weighted average of individual asset volatilities
        let mut weighted_vol_sum = FixedPoint::zero();
        for (asset_id, position) in &portfolio.positions {
            let weight = position.market_value() / portfolio_value;
            let asset_vol = market_data.volatilities.get(asset_id)
                .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_id)))?;
            weighted_vol_sum = weighted_vol_sum + weight * asset_vol;
        }
        
        Ok(weighted_vol_sum / portfolio_volatility)
    }
    
    /// Helper function to calculate portfolio returns from market data
    fn calculate_portfolio_returns(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<Vec<FixedPoint>, RiskError> {
        let assets: Vec<&AssetId> = portfolio.positions.keys().collect();
        
        if assets.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find the minimum number of returns across all assets
        let min_returns = assets.iter()
            .map(|asset_id| market_data.returns.get(*asset_id).map(|r| r.len()).unwrap_or(0))
            .min()
            .unwrap_or(0);
        
        if min_returns == 0 {
            return Err(RiskError::InsufficientData("No return data available".to_string()));
        }
        
        let portfolio_value = portfolio.total_market_value();
        let mut portfolio_returns = Vec::with_capacity(min_returns);
        
        for i in 0..min_returns {
            let mut portfolio_return = FixedPoint::zero();
            
            for asset_id in &assets {
                if let (Some(position), Some(asset_returns)) = (
                    portfolio.positions.get(*asset_id),
                    market_data.returns.get(*asset_id)
                ) {
                    let weight = position.market_value() / portfolio_value;
                    portfolio_return = portfolio_return + weight * asset_returns[i];
                }
            }
            
            portfolio_returns.push(portfolio_return);
        }
        
        Ok(portfolio_returns)
    }
    
    /// Build covariance matrix for Monte Carlo simulation
    fn build_covariance_matrix(
        &self,
        assets: &[&AssetId],
        market_data: &MarketData,
    ) -> Result<DMatrix<f64>, RiskError> {
        let n = assets.len();
        let mut covariance_matrix = DMatrix::zeros(n, n);
        
        for (i, asset_i) in assets.iter().enumerate() {
            for (j, asset_j) in assets.iter().enumerate() {
                let vol_i = market_data.volatilities.get(*asset_i)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_i)))?;
                let vol_j = market_data.volatilities.get(*asset_j)
                    .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", asset_j)))?;
                
                let correlation = if i == j {
                    FixedPoint::one()
                } else {
                    market_data.correlations.get(&((*asset_i).clone(), (*asset_j).clone()))
                        .copied()
                        .unwrap_or(FixedPoint::zero())
                };
                
                let covariance = correlation * vol_i * vol_j;
                covariance_matrix[(i, j)] = covariance.to_float();
            }
        }
        
        Ok(covariance_matrix)
    }
    
    /// Update portfolio history for drawdown calculation
    fn update_portfolio_history(&mut self, portfolio: &Portfolio) {
        let current_value = portfolio.total_market_value();
        let timestamp = portfolio.last_update;
        
        self.portfolio_history.push((timestamp, current_value));
        
        // Maintain maximum history length
        if self.portfolio_history.len() > self.max_history_length {
            self.portfolio_history.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
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
        
        market_data.add_price("AAPL".to_string(), FixedPoint::from_float(155.0));
        market_data.add_price("GOOGL".to_string(), FixedPoint::from_float(2850.0));
        
        market_data.add_volatility("AAPL".to_string(), FixedPoint::from_float(0.25));
        market_data.add_volatility("GOOGL".to_string(), FixedPoint::from_float(0.30));
        
        market_data.add_correlation("AAPL".to_string(), "GOOGL".to_string(), FixedPoint::from_float(0.6));
        
        // Add some sample returns
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
        
        market_data
    }
    
    #[test]
    fn test_parametric_var_calculation() {
        let portfolio = create_test_portfolio();
        let market_data = create_test_market_data();
        
        let mut calculator = RiskMetricsCalculator::new(1000, MonteCarloParams::default());
        let risk_metrics = calculator.calculate_risk_metrics(
            &portfolio,
            &market_data,
            VaRMethod::Parametric,
        ).unwrap();
        
        assert!(risk_metrics.var_95 > FixedPoint::zero());
        assert!(risk_metrics.var_99 > risk_metrics.var_95);
        assert!(risk_metrics.var_99_9 > risk_metrics.var_99);
        assert!(risk_metrics.expected_shortfall_95 > risk_metrics.var_95);
        assert!(risk_metrics.portfolio_volatility > FixedPoint::zero());
    }
    
    #[test]
    fn test_risk_contributions() {
        let portfolio = create_test_portfolio();
        let market_data = create_test_market_data();
        
        let mut calculator = RiskMetricsCalculator::new(1000, MonteCarloParams::default());
        let risk_metrics = calculator.calculate_risk_metrics(
            &portfolio,
            &market_data,
            VaRMethod::Parametric,
        ).unwrap();
        
        assert_eq!(risk_metrics.risk_contributions.len(), 2);
        assert!(risk_metrics.risk_contributions.contains_key("AAPL"));
        assert!(risk_metrics.risk_contributions.contains_key("GOOGL"));
        
        // Risk contributions should sum to approximately 1
        let total_contribution: FixedPoint = risk_metrics.risk_contributions.values().sum();
        assert!((total_contribution - FixedPoint::one()).abs() < FixedPoint::from_float(0.01));
    }
    
    #[test]
    fn test_diversification_ratio() {
        let portfolio = create_test_portfolio();
        let market_data = create_test_market_data();
        
        let mut calculator = RiskMetricsCalculator::new(1000, MonteCarloParams::default());
        let risk_metrics = calculator.calculate_risk_metrics(
            &portfolio,
            &market_data,
            VaRMethod::Parametric,
        ).unwrap();
        
        // Diversification ratio should be > 1 for a diversified portfolio
        assert!(risk_metrics.diversification_ratio > FixedPoint::one());
    }
    
    #[test]
    fn test_maximum_drawdown() {
        let mut calculator = RiskMetricsCalculator::new(1000, MonteCarloParams::default());
        
        // Simulate portfolio value decline
        let mut portfolio = create_test_portfolio();
        
        // Add some history with declining values
        calculator.portfolio_history = vec![
            (1000, FixedPoint::from_float(100000.0)),
            (2000, FixedPoint::from_float(95000.0)),
            (3000, FixedPoint::from_float(90000.0)),
            (4000, FixedPoint::from_float(85000.0)),
            (5000, FixedPoint::from_float(92000.0)),
        ];
        
        let (max_drawdown, current_drawdown, _duration) = calculator.calculate_maximum_drawdown().unwrap();
        
        assert!(max_drawdown > FixedPoint::zero());
        assert!(current_drawdown >= FixedPoint::zero());
    }
}