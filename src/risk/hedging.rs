//! Dynamic Hedging Framework
//!
//! This module implements sophisticated hedging strategies including:
//! - Minimum variance hedge ratio calculation
//! - Multi-asset hedging optimization using quadratic programming
//! - Dynamic hedge adjustment based on correlation changes
//! - Hedge effectiveness monitoring and rebalancing

use super::{Portfolio, MarketData, RiskError, AssetId, Position};
use crate::math::fixed_point::FixedPoint;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Hedge recommendation for a specific asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgeRecommendation {
    /// Asset to be hedged
    pub target_asset: AssetId,
    
    /// Hedging instrument
    pub hedge_asset: AssetId,
    
    /// Optimal hedge ratio
    pub hedge_ratio: FixedPoint,
    
    /// Recommended hedge quantity
    pub hedge_quantity: i64,
    
    /// Expected hedge effectiveness (R²)
    pub effectiveness: FixedPoint,
    
    /// Confidence interval for hedge ratio
    pub confidence_interval: (FixedPoint, FixedPoint),
    
    /// Transaction cost estimate
    pub transaction_cost: FixedPoint,
    
    /// Timestamp of recommendation
    pub timestamp: u64,
}

/// Hedge performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgePerformance {
    /// Hedge effectiveness (R²)
    pub effectiveness: FixedPoint,
    
    /// Variance reduction achieved
    pub variance_reduction: FixedPoint,
    
    /// Tracking error
    pub tracking_error: FixedPoint,
    
    /// Basis risk
    pub basis_risk: FixedPoint,
    
    /// Cost of hedging
    pub hedging_cost: FixedPoint,
    
    /// Time since last rebalance
    pub time_since_rebalance: u64,
}

/// Hedging strategy parameters
#[derive(Debug, Clone)]
pub struct HedgingParams {
    /// Minimum hedge effectiveness threshold
    pub min_effectiveness: FixedPoint,
    
    /// Maximum transaction cost tolerance
    pub max_transaction_cost: FixedPoint,
    
    /// Rebalancing frequency (in nanoseconds)
    pub rebalance_frequency: u64,
    
    /// Correlation threshold for hedge adjustment
    pub correlation_threshold: FixedPoint,
    
    /// Confidence level for hedge ratio estimation
    pub confidence_level: FixedPoint,
    
    /// Lookback window for correlation estimation (number of periods)
    pub lookback_window: usize,
}

impl Default for HedgingParams {
    fn default() -> Self {
        Self {
            min_effectiveness: FixedPoint::from_float(0.7),
            max_transaction_cost: FixedPoint::from_float(0.001),
            rebalance_frequency: 24 * 60 * 60 * 1_000_000_000, // 1 day in nanoseconds
            correlation_threshold: FixedPoint::from_float(0.1),
            confidence_level: FixedPoint::from_float(0.95),
            lookback_window: 252, // 1 year of daily data
        }
    }
}

/// Dynamic Hedging Framework
pub struct DynamicHedgingFramework {
    /// Hedging parameters
    params: HedgingParams,
    
    /// Current hedge positions
    hedge_positions: HashMap<AssetId, HedgeRecommendation>,
    
    /// Historical hedge performance
    performance_history: Vec<(u64, HedgePerformance)>,
    
    /// Available hedging instruments
    hedging_instruments: HashMap<AssetId, Vec<AssetId>>,
    
    /// Transaction cost model
    transaction_costs: HashMap<AssetId, FixedPoint>,
}

impl DynamicHedgingFramework {
    /// Create a new dynamic hedging framework
    pub fn new(params: HedgingParams) -> Self {
        Self {
            params,
            hedge_positions: HashMap::new(),
            performance_history: Vec::new(),
            hedging_instruments: HashMap::new(),
            transaction_costs: HashMap::new(),
        }
    }
    
    /// Add hedging instrument mapping
    pub fn add_hedging_instrument(&mut self, target_asset: AssetId, hedge_assets: Vec<AssetId>) {
        self.hedging_instruments.insert(target_asset, hedge_assets);
    }
    
    /// Set transaction cost for an asset
    pub fn set_transaction_cost(&mut self, asset_id: AssetId, cost: FixedPoint) {
        self.transaction_costs.insert(asset_id, cost);
    }
    
    /// Calculate optimal hedge recommendations for a portfolio
    pub fn calculate_hedge_recommendations(
        &mut self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<Vec<HedgeRecommendation>, RiskError> {
        let mut recommendations = Vec::new();
        
        for (asset_id, position) in &portfolio.positions {
            if let Some(hedge_instruments) = self.hedging_instruments.get(asset_id) {
                let recommendation = self.calculate_optimal_hedge(
                    asset_id,
                    position,
                    hedge_instruments,
                    market_data,
                )?;
                
                if recommendation.effectiveness >= self.params.min_effectiveness {
                    recommendations.push(recommendation);
                }
            }
        }
        
        Ok(recommendations)
    }
    
    /// Calculate minimum variance hedge ratio
    pub fn calculate_minimum_variance_hedge_ratio(
        &self,
        target_asset: &AssetId,
        hedge_asset: &AssetId,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskError> {
        let target_returns = market_data.returns.get(target_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", target_asset)))?;
        
        let hedge_returns = market_data.returns.get(hedge_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", hedge_asset)))?;
        
        if target_returns.len() != hedge_returns.len() || target_returns.len() < 30 {
            return Err(RiskError::InsufficientData(
                "Need at least 30 matching return observations".to_string()
            ));
        }
        
        // Calculate covariance and variance
        let target_mean: FixedPoint = target_returns.iter().sum::<FixedPoint>() / 
            FixedPoint::from_int(target_returns.len() as i32);
        let hedge_mean: FixedPoint = hedge_returns.iter().sum::<FixedPoint>() / 
            FixedPoint::from_int(hedge_returns.len() as i32);
        
        let mut covariance = FixedPoint::zero();
        let mut hedge_variance = FixedPoint::zero();
        
        for i in 0..target_returns.len() {
            let target_dev = target_returns[i] - target_mean;
            let hedge_dev = hedge_returns[i] - hedge_mean;
            
            covariance = covariance + target_dev * hedge_dev;
            hedge_variance = hedge_variance + hedge_dev * hedge_dev;
        }
        
        let n = FixedPoint::from_int(target_returns.len() as i32 - 1);
        covariance = covariance / n;
        hedge_variance = hedge_variance / n;
        
        if hedge_variance.is_zero() {
            return Err(RiskError::ComputationError("Hedge asset has zero variance".to_string()));
        }
        
        Ok(covariance / hedge_variance)
    }
    
    /// Calculate multi-asset hedge using quadratic programming
    pub fn calculate_multi_asset_hedge(
        &self,
        target_asset: &AssetId,
        hedge_assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<HashMap<AssetId, FixedPoint>, RiskError> {
        if hedge_assets.is_empty() {
            return Ok(HashMap::new());
        }
        
        let target_returns = market_data.returns.get(target_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", target_asset)))?;
        
        let n_obs = target_returns.len();
        let n_hedges = hedge_assets.len();
        
        // Build hedge returns matrix
        let mut hedge_returns_matrix = Vec::with_capacity(n_hedges);
        for hedge_asset in hedge_assets {
            let returns = market_data.returns.get(hedge_asset)
                .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", hedge_asset)))?;
            
            if returns.len() != n_obs {
                return Err(RiskError::InsufficientData("Mismatched return series lengths".to_string()));
            }
            
            hedge_returns_matrix.push(returns.clone());
        }
        
        // Convert to nalgebra matrices
        let mut h_matrix = DMatrix::zeros(n_obs, n_hedges);
        for (j, hedge_returns) in hedge_returns_matrix.iter().enumerate() {
            for (i, &return_val) in hedge_returns.iter().enumerate() {
                h_matrix[(i, j)] = return_val.to_float();
            }
        }
        
        let target_vector = DVector::from_vec(
            target_returns.iter().map(|r| r.to_float()).collect()
        );
        
        // Solve normal equations: (H'H)β = H'r
        let hth = h_matrix.transpose() * &h_matrix;
        let htr = h_matrix.transpose() * target_vector;
        
        // Add regularization for numerical stability
        let regularization = 1e-8;
        let regularized_hth = hth + DMatrix::identity(n_hedges, n_hedges) * regularization;
        
        let decomp = regularized_hth.lu();
        let beta = decomp.solve(&htr)
            .ok_or_else(|| RiskError::MatrixError("Failed to solve hedge optimization".to_string()))?;
        
        // Convert back to HashMap
        let mut hedge_ratios = HashMap::new();
        for (i, hedge_asset) in hedge_assets.iter().enumerate() {
            hedge_ratios.insert(hedge_asset.clone(), FixedPoint::from_float(beta[i]));
        }
        
        Ok(hedge_ratios)
    }
    
    /// Calculate hedge effectiveness (R²)
    pub fn calculate_hedge_effectiveness(
        &self,
        target_asset: &AssetId,
        hedge_asset: &AssetId,
        hedge_ratio: FixedPoint,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskError> {
        let target_returns = market_data.returns.get(target_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", target_asset)))?;
        
        let hedge_returns = market_data.returns.get(hedge_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", hedge_asset)))?;
        
        if target_returns.len() != hedge_returns.len() {
            return Err(RiskError::InsufficientData("Mismatched return series lengths".to_string()));
        }
        
        // Calculate hedged returns
        let mut hedged_returns = Vec::with_capacity(target_returns.len());
        for i in 0..target_returns.len() {
            let hedged_return = target_returns[i] - hedge_ratio * hedge_returns[i];
            hedged_returns.push(hedged_return);
        }
        
        // Calculate variances
        let target_variance = self.calculate_variance(target_returns);
        let hedged_variance = self.calculate_variance(&hedged_returns);
        
        if target_variance.is_zero() {
            return Ok(FixedPoint::zero());
        }
        
        let variance_reduction = (target_variance - hedged_variance) / target_variance;
        Ok(variance_reduction.max(FixedPoint::zero()))
    }
    
    /// Monitor hedge performance and suggest rebalancing
    pub fn monitor_hedge_performance(
        &mut self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<Vec<AssetId>, RiskError> {
        let mut rebalance_needed = Vec::new();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        for (asset_id, hedge_rec) in &self.hedge_positions {
            // Check if rebalancing is needed based on time
            if current_time - hedge_rec.timestamp > self.params.rebalance_frequency {
                rebalance_needed.push(asset_id.clone());
                continue;
            }
            
            // Check if correlation has changed significantly
            let current_correlation = market_data.correlations
                .get(&(asset_id.clone(), hedge_rec.hedge_asset.clone()))
                .copied()
                .unwrap_or(FixedPoint::zero());
            
            let expected_correlation = self.estimate_correlation_from_hedge_ratio(
                hedge_rec.hedge_ratio,
                asset_id,
                &hedge_rec.hedge_asset,
                market_data,
            )?;
            
            if (current_correlation - expected_correlation).abs() > self.params.correlation_threshold {
                rebalance_needed.push(asset_id.clone());
            }
            
            // Check hedge effectiveness
            let current_effectiveness = self.calculate_hedge_effectiveness(
                asset_id,
                &hedge_rec.hedge_asset,
                hedge_rec.hedge_ratio,
                market_data,
            )?;
            
            if current_effectiveness < self.params.min_effectiveness {
                rebalance_needed.push(asset_id.clone());
            }
        }
        
        Ok(rebalance_needed)
    }
    
    /// Update hedge positions
    pub fn update_hedge_positions(&mut self, recommendations: Vec<HedgeRecommendation>) {
        for recommendation in recommendations {
            self.hedge_positions.insert(recommendation.target_asset.clone(), recommendation);
        }
    }
    
    /// Calculate optimal hedge for a single position
    fn calculate_optimal_hedge(
        &self,
        target_asset: &AssetId,
        position: &Position,
        hedge_instruments: &[AssetId],
        market_data: &MarketData,
    ) -> Result<HedgeRecommendation, RiskError> {
        let mut best_hedge = None;
        let mut best_effectiveness = FixedPoint::zero();
        
        // Try each hedging instrument
        for hedge_asset in hedge_instruments {
            let hedge_ratio = self.calculate_minimum_variance_hedge_ratio(
                target_asset,
                hedge_asset,
                market_data,
            )?;
            
            let effectiveness = self.calculate_hedge_effectiveness(
                target_asset,
                hedge_asset,
                hedge_ratio,
                market_data,
            )?;
            
            if effectiveness > best_effectiveness {
                let hedge_quantity = (FixedPoint::from_int(position.quantity) * hedge_ratio).to_int();
                let transaction_cost = self.calculate_transaction_cost(hedge_asset, hedge_quantity);
                let confidence_interval = self.calculate_hedge_ratio_confidence_interval(
                    target_asset,
                    hedge_asset,
                    hedge_ratio,
                    market_data,
                )?;
                
                best_hedge = Some(HedgeRecommendation {
                    target_asset: target_asset.clone(),
                    hedge_asset: hedge_asset.clone(),
                    hedge_ratio,
                    hedge_quantity,
                    effectiveness,
                    confidence_interval,
                    transaction_cost,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64,
                });
                
                best_effectiveness = effectiveness;
            }
        }
        
        best_hedge.ok_or_else(|| RiskError::ComputationError("No suitable hedge found".to_string()))
    }
    
    /// Calculate confidence interval for hedge ratio
    fn calculate_hedge_ratio_confidence_interval(
        &self,
        target_asset: &AssetId,
        hedge_asset: &AssetId,
        hedge_ratio: FixedPoint,
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint), RiskError> {
        let target_returns = market_data.returns.get(target_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", target_asset)))?;
        
        let hedge_returns = market_data.returns.get(hedge_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", hedge_asset)))?;
        
        let n = target_returns.len() as f64;
        if n < 30.0 {
            return Err(RiskError::InsufficientData("Need at least 30 observations for confidence interval".to_string()));
        }
        
        // Calculate standard error of hedge ratio
        let hedge_variance = self.calculate_variance(hedge_returns);
        let residual_variance = self.calculate_residual_variance(target_returns, hedge_returns, hedge_ratio);
        
        let standard_error = (residual_variance / (hedge_variance * FixedPoint::from_float(n - 2.0))).sqrt();
        
        // Use t-distribution critical value (approximated with normal for large n)
        let alpha = FixedPoint::one() - self.params.confidence_level;
        let t_critical = FixedPoint::from_float(1.96); // 95% confidence level approximation
        
        let margin = t_critical * standard_error;
        
        Ok((hedge_ratio - margin, hedge_ratio + margin))
    }
    
    /// Calculate transaction cost for a hedge trade
    fn calculate_transaction_cost(&self, asset_id: &AssetId, quantity: i64) -> FixedPoint {
        let cost_rate = self.transaction_costs.get(asset_id)
            .copied()
            .unwrap_or(FixedPoint::from_float(0.001)); // Default 0.1% transaction cost
        
        cost_rate * FixedPoint::from_int(quantity.abs())
    }
    
    /// Estimate correlation from hedge ratio
    fn estimate_correlation_from_hedge_ratio(
        &self,
        hedge_ratio: FixedPoint,
        target_asset: &AssetId,
        hedge_asset: &AssetId,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskError> {
        let target_vol = market_data.volatilities.get(target_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", target_asset)))?;
        
        let hedge_vol = market_data.volatilities.get(hedge_asset)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing volatility for {}", hedge_asset)))?;
        
        if hedge_vol.is_zero() {
            return Ok(FixedPoint::zero());
        }
        
        // Correlation = hedge_ratio * hedge_vol / target_vol
        Ok(hedge_ratio * hedge_vol / target_vol)
    }
    
    /// Calculate variance of a return series
    fn calculate_variance(&self, returns: &[FixedPoint]) -> FixedPoint {
        if returns.is_empty() {
            return FixedPoint::zero();
        }
        
        let mean: FixedPoint = returns.iter().sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32);
        let variance: FixedPoint = returns.iter()
            .map(|&r| (r - mean) * (r - mean))
            .sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32 - 1);
        
        variance
    }
    
    /// Calculate residual variance for hedge effectiveness
    fn calculate_residual_variance(
        &self,
        target_returns: &[FixedPoint],
        hedge_returns: &[FixedPoint],
        hedge_ratio: FixedPoint,
    ) -> FixedPoint {
        let mut residuals = Vec::with_capacity(target_returns.len());
        
        for i in 0..target_returns.len() {
            let residual = target_returns[i] - hedge_ratio * hedge_returns[i];
            residuals.push(residual);
        }
        
        self.calculate_variance(&residuals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_market_data() -> MarketData {
        let mut market_data = MarketData::new();
        
        // Add correlated returns for testing
        let target_returns = vec![
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(-0.02),
            FixedPoint::from_float(0.015),
            FixedPoint::from_float(-0.01),
            FixedPoint::from_float(0.005),
            FixedPoint::from_float(0.02),
            FixedPoint::from_float(-0.015),
            FixedPoint::from_float(0.008),
            FixedPoint::from_float(-0.005),
            FixedPoint::from_float(0.012),
        ];
        
        // Create hedge returns with 0.8 correlation
        let hedge_returns: Vec<FixedPoint> = target_returns.iter()
            .map(|&r| r * FixedPoint::from_float(0.8) + FixedPoint::from_float(0.001))
            .collect();
        
        market_data.add_returns("AAPL".to_string(), target_returns);
        market_data.add_returns("SPY".to_string(), hedge_returns);
        
        market_data.add_volatility("AAPL".to_string(), FixedPoint::from_float(0.25));
        market_data.add_volatility("SPY".to_string(), FixedPoint::from_float(0.15));
        
        market_data.add_correlation("AAPL".to_string(), "SPY".to_string(), FixedPoint::from_float(0.8));
        
        market_data
    }
    
    #[test]
    fn test_minimum_variance_hedge_ratio() {
        let market_data = create_test_market_data();
        let framework = DynamicHedgingFramework::new(HedgingParams::default());
        
        let hedge_ratio = framework.calculate_minimum_variance_hedge_ratio(
            &"AAPL".to_string(),
            &"SPY".to_string(),
            &market_data,
        ).unwrap();
        
        assert!(hedge_ratio > FixedPoint::zero());
        assert!(hedge_ratio < FixedPoint::from_float(2.0));
    }
    
    #[test]
    fn test_hedge_effectiveness() {
        let market_data = create_test_market_data();
        let framework = DynamicHedgingFramework::new(HedgingParams::default());
        
        let hedge_ratio = framework.calculate_minimum_variance_hedge_ratio(
            &"AAPL".to_string(),
            &"SPY".to_string(),
            &market_data,
        ).unwrap();
        
        let effectiveness = framework.calculate_hedge_effectiveness(
            &"AAPL".to_string(),
            &"SPY".to_string(),
            hedge_ratio,
            &market_data,
        ).unwrap();
        
        assert!(effectiveness > FixedPoint::zero());
        assert!(effectiveness <= FixedPoint::one());
    }
    
    #[test]
    fn test_multi_asset_hedge() {
        let market_data = create_test_market_data();
        let framework = DynamicHedgingFramework::new(HedgingParams::default());
        
        let hedge_assets = vec!["SPY".to_string()];
        let hedge_ratios = framework.calculate_multi_asset_hedge(
            &"AAPL".to_string(),
            &hedge_assets,
            &market_data,
        ).unwrap();
        
        assert_eq!(hedge_ratios.len(), 1);
        assert!(hedge_ratios.contains_key("SPY"));
    }
    
    #[test]
    fn test_hedge_recommendations() {
        let mut framework = DynamicHedgingFramework::new(HedgingParams::default());
        framework.add_hedging_instrument("AAPL".to_string(), vec!["SPY".to_string()]);
        
        let mut portfolio = Portfolio::new(FixedPoint::from_float(10000.0));
        portfolio.add_position(Position::new(
            "AAPL".to_string(),
            100,
            FixedPoint::from_float(150.0),
            FixedPoint::from_float(155.0),
        ));
        
        let market_data = create_test_market_data();
        
        let recommendations = framework.calculate_hedge_recommendations(
            &portfolio,
            &market_data,
        ).unwrap();
        
        assert!(!recommendations.is_empty());
        assert_eq!(recommendations[0].target_asset, "AAPL");
        assert_eq!(recommendations[0].hedge_asset, "SPY");
    }
}