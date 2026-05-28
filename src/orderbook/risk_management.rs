//! Correlation-Aware Risk Management
//!
//! This module provides sophisticated risk management capabilities that take into account
//! cross-asset correlations, position concentrations, and portfolio-level risk metrics.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::{Symbol, Order, Side};
use super::multi_asset::{Position, MultiAssetError};

/// Risk management errors
#[derive(Error, Debug, Clone)]
pub enum RiskError {
    #[error("Position limit exceeded for {symbol}: current {current}, limit {limit}")]
    PositionLimitExceeded { symbol: String, current: f64, limit: f64 },
    #[error("Notional limit exceeded for {symbol}: current {current}, limit {limit}")]
    NotionalLimitExceeded { symbol: String, current: f64, limit: f64 },
    #[error("Portfolio VaR limit exceeded: current {current}, limit {limit}")]
    VaRLimitExceeded { current: f64, limit: f64 },
    #[error("Concentration limit exceeded for {symbol}: weight {weight}%, limit {limit}%")]
    ConcentrationLimitExceeded { symbol: String, weight: f64, limit: f64 },
    #[error("Correlation risk limit exceeded: portfolio correlation {correlation}")]
    CorrelationRiskExceeded { correlation: f64 },
    #[error("Insufficient margin: required {required}, available {available}")]
    InsufficientMargin { required: f64, available: f64 },
    #[error("Risk calculation error: {0}")]
    CalculationError(String),
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum position size per symbol (in base units)
    pub max_position_size: HashMap<Symbol, f64>,
    
    /// Maximum notional exposure per symbol
    pub max_notional_exposure: HashMap<Symbol, f64>,
    
    /// Maximum portfolio Value-at-Risk (VaR)
    pub max_portfolio_var: f64,
    
    /// Maximum concentration per symbol (as percentage of total portfolio)
    pub max_concentration_pct: f64,
    
    /// Maximum correlation exposure (portfolio correlation with market)
    pub max_correlation_exposure: f64,
    
    /// Minimum margin requirement
    pub min_margin_ratio: f64,
    
    /// Default position limit for new symbols
    pub default_position_limit: f64,
    
    /// Default notional limit for new symbols
    pub default_notional_limit: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_position_size: HashMap::new(),
            max_notional_exposure: HashMap::new(),
            max_portfolio_var: 0.05,        // 5% daily VaR
            max_concentration_pct: 25.0,    // 25% max concentration
            max_correlation_exposure: 0.8,  // 80% max correlation
            min_margin_ratio: 0.1,          // 10% margin requirement
            default_position_limit: 1000000.0,
            default_notional_limit: 10000000.0,
        }
    }
}

/// Portfolio risk metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub portfolio_var: f64,
    pub portfolio_volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub concentration_risk: f64,
    pub correlation_risk: f64,
    pub margin_utilization: f64,
    pub timestamp: u64,
}

/// Correlation-aware risk manager
#[derive(Debug)]
pub struct CorrelationAwareRiskManager {
    /// Risk limits configuration
    limits: RiskLimits,
    
    /// Current positions
    positions: HashMap<Symbol, Position>,
    
    /// Historical returns for VaR calculation
    returns_history: HashMap<Symbol, Vec<f64>>,
    
    /// Maximum history length for calculations
    max_history_length: usize,
    
    /// VaR confidence level (e.g., 0.95 for 95% VaR)
    var_confidence_level: f64,
    
    /// Available margin/capital
    available_margin: f64,
    
    /// Risk metrics cache
    cached_metrics: Option<CachedRiskMetrics>,
    
    /// Cache validity duration (nanoseconds)
    cache_duration: u64,
}

/// Cached risk metrics with timestamp
#[derive(Debug, Clone)]
struct CachedRiskMetrics {
    metrics: RiskMetrics,
    timestamp: u64,
}

impl CorrelationAwareRiskManager {
    /// Create a new risk manager with default limits
    pub fn new() -> Self {
        Self {
            limits: RiskLimits::default(),
            positions: HashMap::new(),
            returns_history: HashMap::new(),
            max_history_length: 252, // ~1 year of daily returns
            var_confidence_level: 0.95,
            available_margin: 1000000.0, // Default $1M margin
            cached_metrics: None,
            cache_duration: 300_000_000_000, // 5 minutes cache
        }
    }
    
    /// Create with custom limits and parameters
    pub fn with_config(
        limits: RiskLimits,
        var_confidence_level: f64,
        available_margin: f64,
    ) -> Self {
        Self {
            limits,
            positions: HashMap::new(),
            returns_history: HashMap::new(),
            max_history_length: 252,
            var_confidence_level,
            available_margin,
            cached_metrics: None,
            cache_duration: 300_000_000_000,
        }
    }
    
    /// Update risk limits
    pub fn update_limits(&mut self, limits: RiskLimits) {
        self.limits = limits;
        self.cached_metrics = None; // Invalidate cache
    }
    
    /// Update positions from external source
    pub fn update_positions(&mut self, positions: HashMap<Symbol, Position>) {
        self.positions = positions;
        self.cached_metrics = None; // Invalidate cache
    }
    
    /// Add return observation for a symbol
    pub fn add_return_observation(&mut self, symbol: &Symbol, return_value: f64) {
        let history = self.returns_history.entry(symbol.clone()).or_insert_with(Vec::new);
        history.push(return_value);
        
        // Maintain maximum history length
        while history.len() > self.max_history_length {
            history.remove(0);
        }
        
        self.cached_metrics = None; // Invalidate cache
    }
    
    /// Check if an order violates risk limits
    pub fn check_order_risk(&mut self, order: &Order) -> Result<(), MultiAssetError> {
        // Check position limits
        self.check_position_limits(order)?;
        
        // Check notional limits
        self.check_notional_limits(order)?;
        
        // Check portfolio-level limits
        self.check_portfolio_limits(order)?;
        
        // Check margin requirements
        self.check_margin_requirements(order)?;
        
        Ok(())
    }
    
    /// Check position size limits
    fn check_position_limits(&self, order: &Order) -> Result<(), MultiAssetError> {
        let current_position = self.positions.get(&order.symbol)
            .map(|p| p.quantity as f64)
            .unwrap_or(0.0);
        
        let order_size = if order.side == Side::Buy {
            order.size as f64
        } else {
            -(order.size as f64)
        };
        
        let new_position = current_position + order_size;
        let position_limit = self.limits.max_position_size
            .get(&order.symbol)
            .copied()
            .unwrap_or(self.limits.default_position_limit);
        
        if new_position.abs() > position_limit {
            return Err(MultiAssetError::RiskLimitExceeded {
                symbol: order.symbol.to_string(),
                limit: position_limit,
                current: new_position.abs(),
            });
        }
        
        Ok(())
    }
    
    /// Check notional exposure limits
    fn check_notional_limits(&self, order: &Order) -> Result<(), MultiAssetError> {
        let current_notional = self.positions.get(&order.symbol)
            .map(|p| p.notional.abs())
            .unwrap_or(0.0);
        
        let order_notional = (order.price * order.size) as f64;
        let new_notional = current_notional + order_notional;
        
        let notional_limit = self.limits.max_notional_exposure
            .get(&order.symbol)
            .copied()
            .unwrap_or(self.limits.default_notional_limit);
        
        if new_notional > notional_limit {
            return Err(MultiAssetError::RiskLimitExceeded {
                symbol: order.symbol.to_string(),
                limit: notional_limit,
                current: new_notional,
            });
        }
        
        Ok(())
    }
    
    /// Check portfolio-level risk limits
    fn check_portfolio_limits(&mut self, order: &Order) -> Result<(), MultiAssetError> {
        // Calculate portfolio VaR with the new order
        let simulated_positions = self.simulate_order_impact(order);
        let portfolio_var = self.calculate_portfolio_var(&simulated_positions)
            .map_err(|e| MultiAssetError::RiskLimitExceeded {
                symbol: "PORTFOLIO".to_string(),
                limit: self.limits.max_portfolio_var,
                current: 0.0,
            })?;
        
        if portfolio_var > self.limits.max_portfolio_var {
            return Err(MultiAssetError::RiskLimitExceeded {
                symbol: "PORTFOLIO_VAR".to_string(),
                limit: self.limits.max_portfolio_var,
                current: portfolio_var,
            });
        }
        
        // Check concentration limits
        let concentration = self.calculate_concentration_risk(&simulated_positions);
        if concentration > self.limits.max_concentration_pct {
            return Err(MultiAssetError::RiskLimitExceeded {
                symbol: order.symbol.to_string(),
                limit: self.limits.max_concentration_pct,
                current: concentration,
            });
        }
        
        Ok(())
    }
    
    /// Check margin requirements
    fn check_margin_requirements(&self, order: &Order) -> Result<(), MultiAssetError> {
        let order_notional = (order.price * order.size) as f64;
        let required_margin = order_notional * self.limits.min_margin_ratio;
        
        if required_margin > self.available_margin {
            return Err(MultiAssetError::RiskLimitExceeded {
                symbol: "MARGIN".to_string(),
                limit: self.available_margin,
                current: required_margin,
            });
        }
        
        Ok(())
    }
    
    /// Simulate the impact of an order on positions
    fn simulate_order_impact(&self, order: &Order) -> HashMap<Symbol, Position> {
        let mut simulated_positions = self.positions.clone();
        
        let position = simulated_positions.entry(order.symbol.clone())
            .or_insert_with(Position::default);
        
        let size_change = if order.side == Side::Buy {
            order.size as i64
        } else {
            -(order.size as i64)
        };
        
        position.quantity += size_change;
        position.notional += (order.price * order.size) as f64;
        
        if position.quantity != 0 {
            position.average_price = position.notional / position.quantity.abs() as f64;
        }
        
        simulated_positions
    }
    
    /// Calculate portfolio Value-at-Risk
    fn calculate_portfolio_var(&self, positions: &HashMap<Symbol, Position>) -> Result<f64, String> {
        if positions.is_empty() {
            return Ok(0.0);
        }
        
        // Get symbols with positions
        let symbols: Vec<Symbol> = positions.keys().cloned().collect();
        let n = symbols.len();
        
        if n == 1 {
            // Single asset VaR
            let symbol = &symbols[0];
            let position = &positions[symbol];
            let returns = self.returns_history.get(symbol)
                .ok_or_else(|| "No return history for symbol".to_string())?;
            
            if returns.len() < 30 {
                return Err("Insufficient return history for VaR calculation".to_string());
            }
            
            let var_quantile = self.calculate_var_quantile(returns)?;
            return Ok(position.notional.abs() * var_quantile.abs());
        }
        
        // Multi-asset portfolio VaR using correlation matrix
        let mut portfolio_returns = Vec::new();
        let mut weights = Vec::new();
        let total_notional: f64 = positions.values().map(|p| p.notional.abs()).sum();
        
        if total_notional == 0.0 {
            return Ok(0.0);
        }
        
        // Calculate portfolio returns
        let max_history = symbols.iter()
            .filter_map(|s| self.returns_history.get(s))
            .map(|h| h.len())
            .min()
            .unwrap_or(0);
        
        if max_history < 30 {
            return Err("Insufficient return history for portfolio VaR".to_string());
        }
        
        for i in 0..max_history {
            let mut portfolio_return = 0.0;
            
            for symbol in &symbols {
                let position = &positions[symbol];
                let returns = &self.returns_history[symbol];
                let weight = position.notional.abs() / total_notional;
                
                portfolio_return += weight * returns[i];
            }
            
            portfolio_returns.push(portfolio_return);
        }
        
        let var_quantile = self.calculate_var_quantile(&portfolio_returns)?;
        Ok(total_notional * var_quantile.abs())
    }
    
    /// Calculate VaR quantile from return series
    fn calculate_var_quantile(&self, returns: &[f64]) -> Result<f64, String> {
        if returns.is_empty() {
            return Err("Empty return series".to_string());
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - self.var_confidence_level) * sorted_returns.len() as f64) as usize;
        let index = index.min(sorted_returns.len() - 1);
        
        Ok(sorted_returns[index])
    }
    
    /// Calculate concentration risk
    fn calculate_concentration_risk(&self, positions: &HashMap<Symbol, Position>) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }
        
        let total_notional: f64 = positions.values().map(|p| p.notional.abs()).sum();
        
        if total_notional == 0.0 {
            return 0.0;
        }
        
        // Find maximum concentration
        positions.values()
            .map(|p| (p.notional.abs() / total_notional) * 100.0)
            .fold(0.0, f64::max)
    }
    
    /// Calculate comprehensive risk metrics
    pub fn calculate_risk_metrics(&mut self) -> Result<RiskMetrics, RiskError> {
        let current_time = current_timestamp();
        
        // Check cache
        if let Some(ref cached) = self.cached_metrics {
            if current_time.saturating_sub(cached.timestamp) < self.cache_duration {
                return Ok(cached.metrics.clone());
            }
        }
        
        let portfolio_var = self.calculate_portfolio_var(&self.positions)
            .map_err(|e| RiskError::CalculationError(e))?;
        
        let portfolio_volatility = self.calculate_portfolio_volatility()
            .map_err(|e| RiskError::CalculationError(e))?;
        
        let concentration_risk = self.calculate_concentration_risk(&self.positions);
        
        let correlation_risk = self.calculate_correlation_risk()
            .map_err(|e| RiskError::CalculationError(e))?;
        
        let total_notional: f64 = self.positions.values().map(|p| p.notional.abs()).sum();
        let margin_utilization = if self.available_margin > 0.0 {
            (total_notional * self.limits.min_margin_ratio) / self.available_margin
        } else {
            1.0
        };
        
        let metrics = RiskMetrics {
            portfolio_var,
            portfolio_volatility,
            max_drawdown: 0.0, // Would need P&L history to calculate
            sharpe_ratio: 0.0,  // Would need return history to calculate
            concentration_risk,
            correlation_risk,
            margin_utilization,
            timestamp: current_time,
        };
        
        // Cache the result
        self.cached_metrics = Some(CachedRiskMetrics {
            metrics: metrics.clone(),
            timestamp: current_time,
        });
        
        Ok(metrics)
    }
    
    /// Calculate portfolio volatility
    fn calculate_portfolio_volatility(&self) -> Result<f64, String> {
        if self.positions.is_empty() {
            return Ok(0.0);
        }
        
        let symbols: Vec<Symbol> = self.positions.keys().cloned().collect();
        let total_notional: f64 = self.positions.values().map(|p| p.notional.abs()).sum();
        
        if total_notional == 0.0 {
            return Ok(0.0);
        }
        
        let mut portfolio_variance = 0.0;
        
        for i in 0..symbols.len() {
            let symbol_i = &symbols[i];
            let weight_i = self.positions[symbol_i].notional.abs() / total_notional;
            let returns_i = self.returns_history.get(symbol_i)
                .ok_or_else(|| format!("No return history for {}", symbol_i))?;
            
            if returns_i.len() < 30 {
                continue;
            }
            
            let vol_i = calculate_volatility(returns_i);
            
            for j in 0..symbols.len() {
                let symbol_j = &symbols[j];
                let weight_j = self.positions[symbol_j].notional.abs() / total_notional;
                let returns_j = self.returns_history.get(symbol_j)
                    .ok_or_else(|| format!("No return history for {}", symbol_j))?;
                
                if returns_j.len() < 30 {
                    continue;
                }
                
                let vol_j = calculate_volatility(returns_j);
                
                let correlation = if i == j {
                    1.0
                } else {
                    calculate_correlation(returns_i, returns_j).unwrap_or(0.0)
                };
                
                portfolio_variance += weight_i * weight_j * vol_i * vol_j * correlation;
            }
        }
        
        Ok(portfolio_variance.sqrt())
    }
    
    /// Calculate correlation risk (average correlation with market)
    fn calculate_correlation_risk(&self) -> Result<f64, String> {
        if self.positions.len() < 2 {
            return Ok(0.0);
        }
        
        let symbols: Vec<Symbol> = self.positions.keys().cloned().collect();
        let mut correlations = Vec::new();
        
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let returns_i = self.returns_history.get(&symbols[i])
                    .ok_or_else(|| format!("No return history for {}", symbols[i]))?;
                let returns_j = self.returns_history.get(&symbols[j])
                    .ok_or_else(|| format!("No return history for {}", symbols[j]))?;
                
                if returns_i.len() >= 30 && returns_j.len() >= 30 {
                    if let Ok(corr) = calculate_correlation(returns_i, returns_j) {
                        correlations.push(corr.abs());
                    }
                }
            }
        }
        
        if correlations.is_empty() {
            return Ok(0.0);
        }
        
        Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
    }
    
    /// Update available margin
    pub fn update_margin(&mut self, margin: f64) {
        self.available_margin = margin;
        self.cached_metrics = None;
    }
    
    /// Get current risk limits
    pub fn get_limits(&self) -> &RiskLimits {
        &self.limits
    }
}

/// Calculate volatility (standard deviation) of returns
fn calculate_volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    
    variance.sqrt()
}

/// Calculate correlation between two return series
fn calculate_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Invalid input series".to_string());
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        return Ok(0.0);
    }
    
    Ok(numerator / denominator)
}

/// Get current timestamp in nanoseconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{OrderId, OrderType, TimeInForce};
    
    #[test]
    fn test_risk_manager_creation() {
        let manager = CorrelationAwareRiskManager::new();
        assert_eq!(manager.limits.max_portfolio_var, 0.05);
        assert_eq!(manager.var_confidence_level, 0.95);
    }
    
    #[test]
    fn test_position_limit_check() {
        let mut manager = CorrelationAwareRiskManager::new();
        let symbol = Symbol::new("BTCUSD").unwrap();
        
        // Set a small position limit
        manager.limits.max_position_size.insert(symbol.clone(), 100.0);
        
        let order = Order {
            id: OrderId::new(1),
            symbol: symbol.clone(),
            side: Side::Buy,
            price: 50000_000000,
            size: 200_000000, // Exceeds limit
            timestamp: current_timestamp(),
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
        };
        
        let result = manager.check_order_risk(&order);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_volatility_calculation() {
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let vol = calculate_volatility(&returns);
        assert!(vol > 0.0);
    }
    
    #[test]
    fn test_correlation_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = calculate_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }
}