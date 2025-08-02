//! Kelly Criterion Position Sizing
//!
//! This module implements sophisticated position sizing strategies including:
//! - Kelly criterion: f* = (μ-r)/(γ*σ²)
//! - Multi-asset Kelly optimization with correlation matrix
//! - Risk overlay with maximum position limits
//! - Fractional Kelly for risk management

use super::{Portfolio, MarketData, RiskError, AssetId};
use crate::math::fixed_point::FixedPoint;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position sizing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizeRecommendation {
    /// Asset identifier
    pub asset_id: AssetId,
    
    /// Recommended position size (as fraction of capital)
    pub position_fraction: FixedPoint,
    
    /// Recommended absolute position size
    pub position_size: FixedPoint,
    
    /// Kelly fraction before risk overlay
    pub raw_kelly_fraction: FixedPoint,
    
    /// Expected return used in calculation
    pub expected_return: FixedPoint,
    
    /// Risk (volatility) used in calculation
    pub risk: FixedPoint,
    
    /// Confidence in the recommendation
    pub confidence: FixedPoint,
    
    /// Risk-adjusted return (Sharpe ratio)
    pub sharpe_ratio: FixedPoint,
    
    /// Timestamp of recommendation
    pub timestamp: u64,
}

/// Kelly criterion parameters
#[derive(Debug, Clone)]
pub struct KellyParams {
    /// Risk-free rate
    pub risk_free_rate: FixedPoint,
    
    /// Risk aversion parameter (for utility maximization)
    pub risk_aversion: FixedPoint,
    
    /// Maximum position fraction (risk overlay)
    pub max_position_fraction: FixedPoint,
    
    /// Fractional Kelly multiplier (for conservative sizing)
    pub fractional_kelly: FixedPoint,
    
    /// Minimum expected return threshold
    pub min_expected_return: FixedPoint,
    
    /// Maximum leverage allowed
    pub max_leverage: FixedPoint,
    
    /// Lookback period for return estimation (number of periods)
    pub lookback_period: usize,
    
    /// Confidence threshold for position sizing
    pub confidence_threshold: FixedPoint,
}

impl Default for KellyParams {
    fn default() -> Self {
        Self {
            risk_free_rate: FixedPoint::from_float(0.02), // 2% annual
            risk_aversion: FixedPoint::from_float(2.0),
            max_position_fraction: FixedPoint::from_float(0.25), // 25% max per asset
            fractional_kelly: FixedPoint::from_float(0.5), // Half Kelly for safety
            min_expected_return: FixedPoint::from_float(0.05), // 5% minimum annual return
            max_leverage: FixedPoint::from_float(2.0), // 2x maximum leverage
            lookback_period: 252, // 1 year of daily data
            confidence_threshold: FixedPoint::from_float(0.6), // 60% confidence minimum
        }
    }
}

/// Multi-asset Kelly optimization result
#[derive(Debug, Clone)]
pub struct MultiAssetKellyResult {
    /// Optimal position fractions by asset
    pub position_fractions: HashMap<AssetId, FixedPoint>,
    
    /// Expected portfolio return
    pub expected_portfolio_return: FixedPoint,
    
    /// Portfolio volatility
    pub portfolio_volatility: FixedPoint,
    
    /// Portfolio Sharpe ratio
    pub portfolio_sharpe_ratio: FixedPoint,
    
    /// Total leverage used
    pub total_leverage: FixedPoint,
    
    /// Diversification benefit
    pub diversification_benefit: FixedPoint,
}

/// Kelly Criterion Position Sizing Engine
pub struct KellyCriterionEngine {
    /// Kelly parameters
    params: KellyParams,
    
    /// Historical performance tracking
    performance_history: Vec<(u64, FixedPoint)>,
    
    /// Expected return estimates
    expected_returns: HashMap<AssetId, FixedPoint>,
    
    /// Return confidence estimates
    return_confidence: HashMap<AssetId, FixedPoint>,
}

impl KellyCriterionEngine {
    /// Create a new Kelly criterion engine
    pub fn new(params: KellyParams) -> Self {
        Self {
            params,
            performance_history: Vec::new(),
            expected_returns: HashMap::new(),
            return_confidence: HashMap::new(),
        }
    }
    
    /// Calculate single-asset Kelly fraction
    pub fn calculate_kelly_fraction(
        &self,
        expected_return: FixedPoint,
        volatility: FixedPoint,
        risk_free_rate: FixedPoint,
    ) -> Result<FixedPoint, RiskError> {
        if volatility.is_zero() {
            return Err(RiskError::ComputationError("Volatility cannot be zero".to_string()));
        }
        
        let excess_return = expected_return - risk_free_rate;
        let variance = volatility * volatility;
        
        // Kelly fraction: f* = (μ-r) / (γ*σ²)
        let kelly_fraction = excess_return / (self.params.risk_aversion * variance);
        
        Ok(kelly_fraction)
    }
    
    /// Calculate position size recommendations for portfolio
    pub fn calculate_position_recommendations(
        &mut self,
        portfolio: &Portfolio,
        market_data: &MarketData,
        available_capital: FixedPoint,
    ) -> Result<Vec<PositionSizeRecommendation>, RiskError> {
        let mut recommendations = Vec::new();
        
        // Update expected returns from market data
        self.update_expected_returns(market_data)?;
        
        for (asset_id, _position) in &portfolio.positions {
            if let Some(recommendation) = self.calculate_single_asset_recommendation(
                asset_id,
                market_data,
                available_capital,
            )? {
                recommendations.push(recommendation);
            }
        }
        
        // Also consider assets not currently in portfolio but with good opportunities
        for asset_id in market_data.returns.keys() {
            if !portfolio.positions.contains_key(asset_id) {
                if let Some(recommendation) = self.calculate_single_asset_recommendation(
                    asset_id,
                    market_data,
                    available_capital,
                )? {
                    if recommendation.expected_return > self.params.min_expected_return {
                        recommendations.push(recommendation);
                    }
                }
            }
        }
        
        Ok(recommendations)
    }
    
    /// Calculate multi-asset Kelly optimization
    pub fn calculate_multi_asset_kelly(
        &self,
        assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<MultiAssetKellyResult, RiskError> {
        let n_assets = assets.len();
        if n_assets == 0 {
            return Err(RiskError::InvalidParameters("No assets provided".to_string()));
        }
        
        // Build expected return vector
        let mut expected_returns = DVector::zeros(n_assets);
        for (i, asset_id) in assets.iter().enumerate() {
            let expected_return = self.expected_returns.get(asset_id)
                .copied()
                .unwrap_or_else(|| self.estimate_expected_return(asset_id, market_data).unwrap_or(FixedPoint::zero()));
            
            expected_returns[i] = (expected_return - self.params.risk_free_rate).to_float();
        }
        
        // Build covariance matrix
        let covariance_matrix = self.build_covariance_matrix(assets, market_data)?;
        
        // Solve for optimal weights: w* = Σ⁻¹(μ-r) / γ
        let inv_cov = covariance_matrix.try_inverse()
            .ok_or_else(|| RiskError::MatrixError("Covariance matrix is not invertible".to_string()))?;
        
        let optimal_weights = inv_cov * expected_returns / self.params.risk_aversion.to_float();
        
        // Apply constraints and fractional Kelly
        let mut position_fractions = HashMap::new();
        let mut total_weight = 0.0;
        
        for (i, asset_id) in assets.iter().enumerate() {
            let mut weight = optimal_weights[i] * self.params.fractional_kelly.to_float();
            
            // Apply maximum position constraint
            weight = weight.min(self.params.max_position_fraction.to_float());
            weight = weight.max(-self.params.max_position_fraction.to_float());
            
            position_fractions.insert(asset_id.clone(), FixedPoint::from_float(weight));
            total_weight += weight.abs();
        }
        
        // Apply leverage constraint
        if total_weight > self.params.max_leverage.to_float() {
            let scale_factor = self.params.max_leverage.to_float() / total_weight;
            for (_, fraction) in position_fractions.iter_mut() {
                *fraction = *fraction * FixedPoint::from_float(scale_factor);
            }
            total_weight = self.params.max_leverage.to_float();
        }
        
        // Calculate portfolio metrics
        let portfolio_metrics = self.calculate_portfolio_metrics(
            &position_fractions,
            assets,
            market_data,
        )?;
        
        Ok(MultiAssetKellyResult {
            position_fractions,
            expected_portfolio_return: portfolio_metrics.0,
            portfolio_volatility: portfolio_metrics.1,
            portfolio_sharpe_ratio: portfolio_metrics.2,
            total_leverage: FixedPoint::from_float(total_weight),
            diversification_benefit: portfolio_metrics.3,
        })
    }
    
    /// Calculate fractional Kelly with risk management overlay
    pub fn calculate_fractional_kelly(
        &self,
        kelly_fraction: FixedPoint,
        confidence: FixedPoint,
        current_drawdown: FixedPoint,
    ) -> FixedPoint {
        let mut adjusted_fraction = kelly_fraction * self.params.fractional_kelly;
        
        // Reduce position size based on confidence
        if confidence < self.params.confidence_threshold {
            let confidence_adjustment = confidence / self.params.confidence_threshold;
            adjusted_fraction = adjusted_fraction * confidence_adjustment;
        }
        
        // Reduce position size during drawdowns
        if current_drawdown > FixedPoint::zero() {
            let drawdown_adjustment = FixedPoint::one() - current_drawdown;
            adjusted_fraction = adjusted_fraction * drawdown_adjustment;
        }
        
        // Apply maximum position constraint
        adjusted_fraction = adjusted_fraction.min(self.params.max_position_fraction);
        adjusted_fraction = adjusted_fraction.max(-self.params.max_position_fraction);
        
        adjusted_fraction
    }
    
    /// Update expected returns from market data
    fn update_expected_returns(&mut self, market_data: &MarketData) -> Result<(), RiskError> {
        for (asset_id, returns) in &market_data.returns {
            if returns.len() >= self.params.lookback_period {
                let expected_return = self.estimate_expected_return(asset_id, market_data)?;
                let confidence = self.estimate_return_confidence(asset_id, market_data)?;
                
                self.expected_returns.insert(asset_id.clone(), expected_return);
                self.return_confidence.insert(asset_id.clone(), confidence);
            }
        }
        
        Ok(())
    }
    
    /// Estimate expected return for an asset
    fn estimate_expected_return(&self, asset_id: &AssetId, market_data: &MarketData) -> Result<FixedPoint, RiskError> {
        let returns = market_data.returns.get(asset_id)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", asset_id)))?;
        
        if returns.is_empty() {
            return Ok(FixedPoint::zero());
        }
        
        // Use recent returns with exponential weighting
        let mut weighted_sum = FixedPoint::zero();
        let mut weight_sum = FixedPoint::zero();
        let decay_factor = FixedPoint::from_float(0.94); // Daily decay factor
        
        for (i, &return_val) in returns.iter().rev().enumerate() {
            let weight = decay_factor.pow(i as u32);
            weighted_sum = weighted_sum + weight * return_val;
            weight_sum = weight_sum + weight;
        }
        
        if weight_sum.is_zero() {
            return Ok(FixedPoint::zero());
        }
        
        // Annualize the return (assuming daily returns)
        let daily_return = weighted_sum / weight_sum;
        Ok(daily_return * FixedPoint::from_float(252.0)) // 252 trading days per year
    }
    
    /// Estimate confidence in return estimate
    fn estimate_return_confidence(&self, asset_id: &AssetId, market_data: &MarketData) -> Result<FixedPoint, RiskError> {
        let returns = market_data.returns.get(asset_id)
            .ok_or_else(|| RiskError::InsufficientData(format!("Missing returns for {}", asset_id)))?;
        
        if returns.len() < 30 {
            return Ok(FixedPoint::from_float(0.1)); // Low confidence for insufficient data
        }
        
        // Calculate confidence based on return consistency and sample size
        let mean_return: FixedPoint = returns.iter().sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32);
        let variance: FixedPoint = returns.iter()
            .map(|&r| (r - mean_return) * (r - mean_return))
            .sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i32 - 1);
        
        let standard_error = variance.sqrt() / FixedPoint::from_int(returns.len() as i32).sqrt();
        
        // Higher confidence for lower standard error and larger sample size
        let sample_size_factor = (FixedPoint::from_int(returns.len() as i32) / FixedPoint::from_float(252.0)).min(FixedPoint::one());
        let precision_factor = FixedPoint::one() / (FixedPoint::one() + standard_error * FixedPoint::from_float(100.0));
        
        Ok(sample_size_factor * precision_factor)
    }
    
    /// Calculate single asset recommendation
    fn calculate_single_asset_recommendation(
        &self,
        asset_id: &AssetId,
        market_data: &MarketData,
        available_capital: FixedPoint,
    ) -> Result<Option<PositionSizeRecommendation>, RiskError> {
        let expected_return = self.expected_returns.get(asset_id)
            .copied()
            .unwrap_or_else(|| self.estimate_expected_return(asset_id, market_data).unwrap_or(FixedPoint::zero()));
        
        let volatility = market_data.volatilities.get(asset_id)
            .copied()
            .unwrap_or(FixedPoint::zero());
        
        let confidence = self.return_confidence.get(asset_id)
            .copied()
            .unwrap_or(FixedPoint::from_float(0.5));
        
        if volatility.is_zero() || expected_return <= self.params.risk_free_rate {
            return Ok(None);
        }
        
        // Calculate raw Kelly fraction
        let raw_kelly = self.calculate_kelly_fraction(
            expected_return,
            volatility,
            self.params.risk_free_rate,
        )?;
        
        // Apply fractional Kelly and risk overlay
        let adjusted_kelly = self.calculate_fractional_kelly(
            raw_kelly,
            confidence,
            FixedPoint::zero(), // TODO: Get current drawdown from portfolio history
        );
        
        let position_size = adjusted_kelly * available_capital;
        let sharpe_ratio = (expected_return - self.params.risk_free_rate) / volatility;
        
        Ok(Some(PositionSizeRecommendation {
            asset_id: asset_id.clone(),
            position_fraction: adjusted_kelly,
            position_size,
            raw_kelly_fraction: raw_kelly,
            expected_return,
            risk: volatility,
            confidence,
            sharpe_ratio,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }))
    }
    
    /// Build covariance matrix for multi-asset optimization
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
    
    /// Calculate portfolio metrics
    fn calculate_portfolio_metrics(
        &self,
        position_fractions: &HashMap<AssetId, FixedPoint>,
        assets: &[AssetId],
        market_data: &MarketData,
    ) -> Result<(FixedPoint, FixedPoint, FixedPoint, FixedPoint), RiskError> {
        let mut portfolio_return = FixedPoint::zero();
        let mut portfolio_variance = FixedPoint::zero();
        let mut individual_vol_sum = FixedPoint::zero();
        
        // Calculate expected portfolio return
        for (asset_id, &weight) in position_fractions {
            let expected_return = self.expected_returns.get(asset_id)
                .copied()
                .unwrap_or(FixedPoint::zero());
            portfolio_return = portfolio_return + weight * expected_return;
            
            let volatility = market_data.volatilities.get(asset_id)
                .copied()
                .unwrap_or(FixedPoint::zero());
            individual_vol_sum = individual_vol_sum + weight.abs() * volatility;
        }
        
        // Calculate portfolio variance
        for (asset_i, &weight_i) in position_fractions {
            for (asset_j, &weight_j) in position_fractions {
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
        let portfolio_sharpe = if portfolio_volatility.is_zero() {
            FixedPoint::zero()
        } else {
            (portfolio_return - self.params.risk_free_rate) / portfolio_volatility
        };
        
        // Diversification benefit = weighted average vol / portfolio vol
        let diversification_benefit = if portfolio_volatility.is_zero() {
            FixedPoint::one()
        } else {
            individual_vol_sum / portfolio_volatility
        };
        
        Ok((portfolio_return, portfolio_volatility, portfolio_sharpe, diversification_benefit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::{Portfolio, Position, MarketData};
    
    fn create_test_market_data() -> MarketData {
        let mut market_data = MarketData::new();
        
        // Add sample returns
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
    fn test_kelly_fraction_calculation() {
        let engine = KellyCriterionEngine::new(KellyParams::default());
        
        let kelly_fraction = engine.calculate_kelly_fraction(
            FixedPoint::from_float(0.12), // 12% expected return
            FixedPoint::from_float(0.20), // 20% volatility
            FixedPoint::from_float(0.02), // 2% risk-free rate
        ).unwrap();
        
        assert!(kelly_fraction > FixedPoint::zero());
        assert!(kelly_fraction < FixedPoint::one());
    }
    
    #[test]
    fn test_position_recommendations() {
        let mut engine = KellyCriterionEngine::new(KellyParams::default());
        
        let mut portfolio = Portfolio::new(FixedPoint::from_float(100000.0));
        portfolio.add_position(Position::new(
            "AAPL".to_string(),
            100,
            FixedPoint::from_float(150.0),
            FixedPoint::from_float(155.0),
        ));
        
        let market_data = create_test_market_data();
        
        let recommendations = engine.calculate_position_recommendations(
            &portfolio,
            &market_data,
            FixedPoint::from_float(100000.0),
        ).unwrap();
        
        assert!(!recommendations.is_empty());
        assert!(recommendations[0].position_fraction >= FixedPoint::zero());
    }
    
    #[test]
    fn test_multi_asset_kelly() {
        let engine = KellyCriterionEngine::new(KellyParams::default());
        let market_data = create_test_market_data();
        
        let assets = vec!["AAPL".to_string(), "GOOGL".to_string()];
        let result = engine.calculate_multi_asset_kelly(&assets, &market_data).unwrap();
        
        assert_eq!(result.position_fractions.len(), 2);
        assert!(result.total_leverage <= FixedPoint::from_float(2.0)); // Max leverage constraint
        assert!(result.portfolio_volatility >= FixedPoint::zero());
    }
    
    #[test]
    fn test_fractional_kelly() {
        let engine = KellyCriterionEngine::new(KellyParams::default());
        
        let raw_kelly = FixedPoint::from_float(0.4);
        let confidence = FixedPoint::from_float(0.8);
        let drawdown = FixedPoint::from_float(0.1);
        
        let fractional_kelly = engine.calculate_fractional_kelly(raw_kelly, confidence, drawdown);
        
        assert!(fractional_kelly < raw_kelly); // Should be reduced
        assert!(fractional_kelly > FixedPoint::zero());
    }
}