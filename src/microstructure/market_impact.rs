//! Market Impact Modeling
//! 
//! Implements sophisticated market impact measurement and modeling including:
//! - Square-root law for market impact estimation
//! - Temporary and permanent impact decomposition
//! - Cross-impact modeling for related assets
//! - Impact decay function estimation

use super::{MarketData, Trade, TradeDirection, MicrostructureError, MicrostructureResult};
use crate::math::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};

/// Market impact metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactMetrics {
    pub timestamp: u64,
    pub temporary_impact: FixedPoint,
    pub permanent_impact: FixedPoint,
    pub total_impact: FixedPoint,
    pub impact_decay_rate: FixedPoint,
    pub participation_rate: FixedPoint,
    pub volume_impact_coefficient: FixedPoint,
    pub cross_impact_correlation: FixedPoint,
}

/// Impact model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactModelParams {
    pub temporary_impact_coefficient: FixedPoint,  // η in I_temp = η * v^α
    pub temporary_impact_exponent: FixedPoint,     // α in I_temp = η * v^α
    pub permanent_impact_coefficient: FixedPoint,  // λ in I_perm = λ * x
    pub decay_rate: FixedPoint,                    // δ in decay function
    pub participation_rate_threshold: FixedPoint,  // Threshold for nonlinear effects
}

/// Trade impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeImpactAnalysis {
    pub trade_id: u64,
    pub timestamp: u64,
    pub pre_trade_price: FixedPoint,
    pub trade_price: FixedPoint,
    pub post_trade_price: FixedPoint,
    pub immediate_impact: FixedPoint,
    pub temporary_impact: FixedPoint,
    pub permanent_impact: FixedPoint,
    pub volume: FixedPoint,
    pub participation_rate: FixedPoint,
    pub direction: TradeDirection,
}

/// Cross-asset impact relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossImpactRelationship {
    pub primary_asset: String,
    pub related_asset: String,
    pub correlation_coefficient: FixedPoint,
    pub impact_transmission_rate: FixedPoint,
    pub lag_milliseconds: u64,
}

/// Market impact analyzer
pub struct MarketImpactAnalyzer {
    market_data_history: VecDeque<MarketData>,
    trade_history: VecDeque<Trade>,
    impact_analyses: VecDeque<TradeImpactAnalysis>,
    model_params: ImpactModelParams,
    cross_impact_relationships: HashMap<String, Vec<CrossImpactRelationship>>,
    max_history_size: usize,
    impact_measurement_window_ms: u64,
    decay_measurement_window_ms: u64,
}

impl MarketImpactAnalyzer {
    /// Create new market impact analyzer
    pub fn new(
        max_history_size: usize,
        impact_measurement_window_ms: u64,
        decay_measurement_window_ms: u64,
    ) -> Self {
        Self {
            market_data_history: VecDeque::with_capacity(max_history_size),
            trade_history: VecDeque::with_capacity(max_history_size),
            impact_analyses: VecDeque::with_capacity(max_history_size),
            model_params: ImpactModelParams::default(),
            cross_impact_relationships: HashMap::new(),
            max_history_size,
            impact_measurement_window_ms,
            decay_measurement_window_ms,
        }
    }

    /// Update with new market data
    pub fn update_market_data(&mut self, market_data: MarketData) {
        self.market_data_history.push_back(market_data);
        if self.market_data_history.len() > self.max_history_size {
            self.market_data_history.pop_front();
        }
    }

    /// Analyze trade impact
    pub fn analyze_trade_impact(&mut self, trade: Trade) -> MicrostructureResult<TradeImpactAnalysis> {
        // Find pre-trade price (most recent market data before trade)
        let pre_trade_price = self.find_price_before_trade(trade.timestamp)
            .ok_or_else(|| MicrostructureError::InsufficientData(
                "No market data available before trade".to_string()
            ))?;

        // Find post-trade price (market data after impact measurement window)
        let post_trade_timestamp = trade.timestamp + self.impact_measurement_window_ms;
        let post_trade_price = self.find_price_after_timestamp(post_trade_timestamp)
            .unwrap_or(trade.price); // Use trade price if no later data

        // Calculate participation rate
        let participation_rate = self.calculate_participation_rate(&trade)?;

        // Calculate different types of impact
        let immediate_impact = self.calculate_immediate_impact(pre_trade_price, trade.price, trade.direction);
        let temporary_impact = self.calculate_temporary_impact(&trade, participation_rate)?;
        let permanent_impact = self.calculate_permanent_impact(pre_trade_price, post_trade_price, trade.direction);

        let analysis = TradeImpactAnalysis {
            trade_id: trade.trade_id,
            timestamp: trade.timestamp,
            pre_trade_price,
            trade_price: trade.price,
            post_trade_price,
            immediate_impact,
            temporary_impact,
            permanent_impact,
            volume: trade.volume,
            participation_rate,
            direction: trade.direction,
        };

        // Store analysis
        self.impact_analyses.push_back(analysis.clone());
        if self.impact_analyses.len() > self.max_history_size {
            self.impact_analyses.pop_front();
        }

        // Store trade
        self.trade_history.push_back(trade);
        if self.trade_history.len() > self.max_history_size {
            self.trade_history.pop_front();
        }

        Ok(analysis)
    }

    /// Calculate immediate impact (trade price vs pre-trade price)
    fn calculate_immediate_impact(
        &self,
        pre_trade_price: FixedPoint,
        trade_price: FixedPoint,
        direction: TradeDirection,
    ) -> FixedPoint {
        let price_change = trade_price - pre_trade_price;
        match direction {
            TradeDirection::Buy => price_change,
            TradeDirection::Sell => -price_change,
            TradeDirection::Unknown => FixedPoint::ZERO,
        }
    }

    /// Calculate temporary impact using square-root law
    fn calculate_temporary_impact(
        &self,
        trade: &Trade,
        participation_rate: FixedPoint,
    ) -> MicrostructureResult<FixedPoint> {
        // Square-root law: I_temp = η * (v/V)^α
        // where v is trade volume, V is average volume, η and α are parameters
        
        let normalized_volume = participation_rate;
        let impact = self.model_params.temporary_impact_coefficient * 
                    normalized_volume.powf(self.model_params.temporary_impact_exponent.to_float());
        
        Ok(impact)
    }

    /// Calculate permanent impact (linear in volume)
    fn calculate_permanent_impact(
        &self,
        pre_trade_price: FixedPoint,
        post_trade_price: FixedPoint,
        direction: TradeDirection,
    ) -> FixedPoint {
        let price_change = post_trade_price - pre_trade_price;
        match direction {
            TradeDirection::Buy => price_change,
            TradeDirection::Sell => -price_change,
            TradeDirection::Unknown => FixedPoint::ZERO,
        }
    }

    /// Calculate participation rate for a trade
    fn calculate_participation_rate(&self, trade: &Trade) -> MicrostructureResult<FixedPoint> {
        // Find recent volume to calculate participation rate
        let lookback_window = 300000; // 5 minutes
        let cutoff_time = trade.timestamp.saturating_sub(lookback_window);
        
        let recent_volume: FixedPoint = self.trade_history.iter()
            .filter(|t| t.timestamp >= cutoff_time && t.timestamp < trade.timestamp)
            .map(|t| t.volume)
            .sum();

        if recent_volume > FixedPoint::ZERO {
            Ok(trade.volume / recent_volume)
        } else {
            Ok(FixedPoint::from_float(0.1)) // Default assumption
        }
    }

    /// Estimate model parameters using historical data
    pub fn estimate_model_parameters(&mut self) -> MicrostructureResult<ImpactModelParams> {
        if self.impact_analyses.len() < 50 {
            return Err(MicrostructureError::InsufficientData(
                "Need at least 50 trade analyses for parameter estimation".to_string()
            ));
        }

        // Estimate temporary impact parameters using regression
        let (temp_coeff, temp_exp) = self.estimate_temporary_impact_params()?;
        
        // Estimate permanent impact coefficient
        let perm_coeff = self.estimate_permanent_impact_coefficient()?;
        
        // Estimate decay rate
        let decay_rate = self.estimate_decay_rate()?;

        let params = ImpactModelParams {
            temporary_impact_coefficient: temp_coeff,
            temporary_impact_exponent: temp_exp,
            permanent_impact_coefficient: perm_coeff,
            decay_rate,
            participation_rate_threshold: FixedPoint::from_float(0.1),
        };

        self.model_params = params.clone();
        Ok(params)
    }

    /// Estimate temporary impact parameters using power law regression
    fn estimate_temporary_impact_params(&self) -> MicrostructureResult<(FixedPoint, FixedPoint)> {
        let analyses: Vec<_> = self.impact_analyses.iter()
            .filter(|a| a.participation_rate > FixedPoint::ZERO && a.temporary_impact > FixedPoint::ZERO)
            .collect();

        if analyses.len() < 20 {
            return Ok((FixedPoint::from_float(0.1), FixedPoint::from_float(0.5))); // Default values
        }

        // Log-linear regression: log(impact) = log(η) + α * log(participation_rate)
        let mut sum_log_participation = 0.0;
        let mut sum_log_impact = 0.0;
        let mut sum_log_participation_sq = 0.0;
        let mut sum_log_participation_impact = 0.0;
        let n = analyses.len() as f64;

        for analysis in &analyses {
            let log_participation = analysis.participation_rate.to_float().ln();
            let log_impact = analysis.temporary_impact.to_float().ln();
            
            sum_log_participation += log_participation;
            sum_log_impact += log_impact;
            sum_log_participation_sq += log_participation * log_participation;
            sum_log_participation_impact += log_participation * log_impact;
        }

        // Calculate regression coefficients
        let denominator = n * sum_log_participation_sq - sum_log_participation * sum_log_participation;
        
        if denominator.abs() < 1e-10 {
            return Ok((FixedPoint::from_float(0.1), FixedPoint::from_float(0.5)));
        }

        let alpha = (n * sum_log_participation_impact - sum_log_participation * sum_log_impact) / denominator;
        let log_eta = (sum_log_impact - alpha * sum_log_participation) / n;
        let eta = log_eta.exp();

        Ok((FixedPoint::from_float(eta), FixedPoint::from_float(alpha)))
    }

    /// Estimate permanent impact coefficient
    fn estimate_permanent_impact_coefficient(&self) -> MicrostructureResult<FixedPoint> {
        let analyses: Vec<_> = self.impact_analyses.iter()
            .filter(|a| a.volume > FixedPoint::ZERO)
            .collect();

        if analyses.is_empty() {
            return Ok(FixedPoint::from_float(0.01)); // Default value
        }

        // Simple linear regression: permanent_impact = λ * volume
        let mut sum_volume = 0.0;
        let mut sum_impact = 0.0;
        let mut sum_volume_sq = 0.0;
        let mut sum_volume_impact = 0.0;
        let n = analyses.len() as f64;

        for analysis in &analyses {
            let volume = analysis.volume.to_float();
            let impact = analysis.permanent_impact.to_float();
            
            sum_volume += volume;
            sum_impact += impact;
            sum_volume_sq += volume * volume;
            sum_volume_impact += volume * impact;
        }

        let denominator = n * sum_volume_sq - sum_volume * sum_volume;
        
        if denominator.abs() < 1e-10 {
            return Ok(FixedPoint::from_float(0.01));
        }

        let lambda = (n * sum_volume_impact - sum_volume * sum_impact) / denominator;
        Ok(FixedPoint::from_float(lambda))
    }

    /// Estimate impact decay rate
    fn estimate_decay_rate(&self) -> MicrostructureResult<FixedPoint> {
        // Analyze how temporary impact decays over time
        let mut decay_observations = Vec::new();
        
        for analysis in &self.impact_analyses {
            if let Some(decay_rate) = self.measure_impact_decay(analysis)? {
                decay_observations.push(decay_rate);
            }
        }

        if decay_observations.is_empty() {
            return Ok(FixedPoint::from_float(0.1)); // Default decay rate
        }

        // Calculate average decay rate
        let sum: FixedPoint = decay_observations.iter().sum();
        let average_decay = sum / FixedPoint::from_int(decay_observations.len() as i64);
        
        Ok(average_decay)
    }

    /// Measure impact decay for a specific trade
    fn measure_impact_decay(&self, analysis: &TradeImpactAnalysis) -> MicrostructureResult<Option<FixedPoint>> {
        // Find price evolution after the trade
        let decay_window_end = analysis.timestamp + self.decay_measurement_window_ms;
        
        let price_evolution: Vec<_> = self.market_data_history.iter()
            .filter(|md| md.timestamp > analysis.timestamp && md.timestamp <= decay_window_end)
            .collect();

        if price_evolution.len() < 5 {
            return Ok(None); // Not enough data points
        }

        // Fit exponential decay: impact(t) = impact_0 * exp(-δ * t)
        let initial_impact = analysis.immediate_impact;
        if initial_impact.abs() < FixedPoint::from_float(1e-6) {
            return Ok(None);
        }

        // Simple approximation: measure half-life
        let mid_price_initial = (analysis.pre_trade_price + analysis.trade_price) / FixedPoint::from_float(2.0);
        let target_impact = initial_impact / FixedPoint::from_float(2.0);
        
        for (i, market_data) in price_evolution.iter().enumerate() {
            let current_mid = (market_data.bid_price + market_data.ask_price) / FixedPoint::from_float(2.0);
            let current_impact = match analysis.direction {
                TradeDirection::Buy => current_mid - mid_price_initial,
                TradeDirection::Sell => mid_price_initial - current_mid,
                TradeDirection::Unknown => FixedPoint::ZERO,
            };

            if current_impact.abs() <= target_impact.abs() {
                let time_elapsed = market_data.timestamp - analysis.timestamp;
                if time_elapsed > 0 {
                    // δ = ln(2) / half_life
                    let decay_rate = FixedPoint::from_float(0.693147) / 
                                   FixedPoint::from_int(time_elapsed as i64);
                    return Ok(Some(decay_rate));
                }
            }
        }

        Ok(None)
    }

    /// Calculate comprehensive market impact metrics
    pub fn calculate_metrics(&self) -> MicrostructureResult<MarketImpactMetrics> {
        let timestamp = self.impact_analyses.back()
            .map(|a| a.timestamp)
            .unwrap_or(0);

        let recent_analyses: Vec<_> = self.impact_analyses.iter()
            .rev()
            .take(100)
            .collect();

        if recent_analyses.is_empty() {
            return Err(MicrostructureError::InsufficientData(
                "No impact analyses available".to_string()
            ));
        }

        // Calculate average impacts
        let temporary_impact = recent_analyses.iter()
            .map(|a| a.temporary_impact)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_analyses.len() as i64);

        let permanent_impact = recent_analyses.iter()
            .map(|a| a.permanent_impact)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_analyses.len() as i64);

        let total_impact = temporary_impact + permanent_impact;

        // Calculate average participation rate
        let participation_rate = recent_analyses.iter()
            .map(|a| a.participation_rate)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_analyses.len() as i64);

        // Calculate volume impact coefficient (impact per unit volume)
        let volume_impact_coefficient = if participation_rate > FixedPoint::ZERO {
            total_impact / participation_rate
        } else {
            FixedPoint::ZERO
        };

        Ok(MarketImpactMetrics {
            timestamp,
            temporary_impact,
            permanent_impact,
            total_impact,
            impact_decay_rate: self.model_params.decay_rate,
            participation_rate,
            volume_impact_coefficient,
            cross_impact_correlation: self.calculate_cross_impact_correlation()?,
        })
    }

    /// Calculate cross-impact correlation
    fn calculate_cross_impact_correlation(&self) -> MicrostructureResult<FixedPoint> {
        // This is a simplified implementation
        // In practice, you would analyze correlations between impacts across related assets
        
        if self.cross_impact_relationships.is_empty() {
            return Ok(FixedPoint::ZERO);
        }

        let total_correlation: FixedPoint = self.cross_impact_relationships.values()
            .flatten()
            .map(|rel| rel.correlation_coefficient)
            .sum();

        let count = self.cross_impact_relationships.values()
            .map(|rels| rels.len())
            .sum::<usize>();

        if count > 0 {
            Ok(total_correlation / FixedPoint::from_int(count as i64))
        } else {
            Ok(FixedPoint::ZERO)
        }
    }

    /// Add cross-impact relationship
    pub fn add_cross_impact_relationship(&mut self, relationship: CrossImpactRelationship) {
        self.cross_impact_relationships
            .entry(relationship.primary_asset.clone())
            .or_insert_with(Vec::new)
            .push(relationship);
    }

    /// Predict impact for a hypothetical trade
    pub fn predict_impact(
        &self,
        volume: FixedPoint,
        current_price: FixedPoint,
        direction: TradeDirection,
    ) -> MicrostructureResult<FixedPoint> {
        // Calculate expected participation rate
        let recent_volume: FixedPoint = self.trade_history.iter()
            .rev()
            .take(20)
            .map(|t| t.volume)
            .sum();

        let avg_recent_volume = if !self.trade_history.is_empty() {
            recent_volume / FixedPoint::from_int(self.trade_history.len().min(20) as i64)
        } else {
            FixedPoint::from_float(1000.0) // Default assumption
        };

        let participation_rate = volume / avg_recent_volume.max(FixedPoint::from_float(1.0));

        // Predict temporary impact using model parameters
        let temporary_impact = self.model_params.temporary_impact_coefficient * 
                              participation_rate.powf(self.model_params.temporary_impact_exponent.to_float());

        // Predict permanent impact
        let permanent_impact = self.model_params.permanent_impact_coefficient * volume;

        let total_predicted_impact = temporary_impact + permanent_impact;

        // Apply direction
        match direction {
            TradeDirection::Buy => Ok(total_predicted_impact),
            TradeDirection::Sell => Ok(-total_predicted_impact),
            TradeDirection::Unknown => Ok(FixedPoint::ZERO),
        }
    }

    /// Find price before trade timestamp
    fn find_price_before_trade(&self, trade_timestamp: u64) -> Option<FixedPoint> {
        self.market_data_history.iter()
            .rev()
            .find(|md| md.timestamp <= trade_timestamp)
            .map(|md| (md.bid_price + md.ask_price) / FixedPoint::from_float(2.0))
    }

    /// Find price after timestamp
    fn find_price_after_timestamp(&self, timestamp: u64) -> Option<FixedPoint> {
        self.market_data_history.iter()
            .find(|md| md.timestamp >= timestamp)
            .map(|md| (md.bid_price + md.ask_price) / FixedPoint::from_float(2.0))
    }

    /// Get impact summary
    pub fn get_impact_summary(&self) -> MicrostructureResult<ImpactSummary> {
        let metrics = self.calculate_metrics()?;
        
        Ok(ImpactSummary {
            timestamp: metrics.timestamp,
            temporary_impact_bps: (metrics.temporary_impact * FixedPoint::from_float(10000.0)).to_float() as i32,
            permanent_impact_bps: (metrics.permanent_impact * FixedPoint::from_float(10000.0)).to_float() as i32,
            total_impact_bps: (metrics.total_impact * FixedPoint::from_float(10000.0)).to_float() as i32,
            participation_rate_pct: (metrics.participation_rate * FixedPoint::from_float(100.0)).to_float() as u16,
            decay_rate_per_sec: (metrics.impact_decay_rate * FixedPoint::from_float(1000.0)).to_float() as u32,
        })
    }
}

impl Default for ImpactModelParams {
    fn default() -> Self {
        Self {
            temporary_impact_coefficient: FixedPoint::from_float(0.1),
            temporary_impact_exponent: FixedPoint::from_float(0.5),
            permanent_impact_coefficient: FixedPoint::from_float(0.01),
            decay_rate: FixedPoint::from_float(0.1),
            participation_rate_threshold: FixedPoint::from_float(0.1),
        }
    }
}

/// Simplified impact summary for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactSummary {
    pub timestamp: u64,
    pub temporary_impact_bps: i32,    // Temporary impact in basis points
    pub permanent_impact_bps: i32,    // Permanent impact in basis points
    pub total_impact_bps: i32,        // Total impact in basis points
    pub participation_rate_pct: u16,  // Participation rate as percentage
    pub decay_rate_per_sec: u32,      // Decay rate per second (scaled by 1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_market_data(timestamp: u64, bid: f64, ask: f64) -> MarketData {
        MarketData {
            timestamp,
            bid_price: FixedPoint::from_float(bid),
            ask_price: FixedPoint::from_float(ask),
            bid_volume: FixedPoint::from_float(1000.0),
            ask_volume: FixedPoint::from_float(1000.0),
            last_trade_price: FixedPoint::from_float((bid + ask) / 2.0),
            last_trade_volume: FixedPoint::from_float(100.0),
            trade_direction: Some(TradeDirection::Buy),
        }
    }

    fn create_test_trade(timestamp: u64, price: f64, volume: f64, direction: TradeDirection, trade_id: u64) -> Trade {
        Trade {
            timestamp,
            price: FixedPoint::from_float(price),
            volume: FixedPoint::from_float(volume),
            direction,
            trade_id,
        }
    }

    #[test]
    fn test_immediate_impact_calculation() {
        let analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
        
        let pre_trade_price = FixedPoint::from_float(100.0);
        let trade_price = FixedPoint::from_float(100.05);
        
        let buy_impact = analyzer.calculate_immediate_impact(
            pre_trade_price, 
            trade_price, 
            TradeDirection::Buy
        );
        
        assert_eq!(buy_impact, FixedPoint::from_float(0.05));
        
        let sell_impact = analyzer.calculate_immediate_impact(
            pre_trade_price, 
            trade_price, 
            TradeDirection::Sell
        );
        
        assert_eq!(sell_impact, FixedPoint::from_float(-0.05));
    }

    #[test]
    fn test_trade_impact_analysis() {
        let mut analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
        
        // Add market data before trade
        let pre_market_data = create_test_market_data(1000, 99.95, 100.05);
        analyzer.update_market_data(pre_market_data);
        
        // Add market data after trade
        let post_market_data = create_test_market_data(7000, 100.00, 100.10);
        analyzer.update_market_data(post_market_data);
        
        // Analyze trade
        let trade = create_test_trade(2000, 100.03, 1000.0, TradeDirection::Buy, 1);
        let analysis = analyzer.analyze_trade_impact(trade).unwrap();
        
        assert_eq!(analysis.trade_id, 1);
        assert_eq!(analysis.direction, TradeDirection::Buy);
        assert!(analysis.immediate_impact > FixedPoint::ZERO);
    }

    #[test]
    fn test_participation_rate_calculation() {
        let mut analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
        
        // Add some historical trades
        for i in 0..10 {
            let trade = create_test_trade(1000 + i * 1000, 100.0, 500.0, TradeDirection::Buy, i + 1);
            analyzer.trade_history.push_back(trade);
        }
        
        // Test trade
        let test_trade = create_test_trade(12000, 100.0, 1000.0, TradeDirection::Buy, 11);
        let participation_rate = analyzer.calculate_participation_rate(&test_trade).unwrap();
        
        // Should be 1000 / (10 * 500) = 0.2
        assert!((participation_rate.to_float() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_impact_prediction() {
        let analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
        
        let predicted_impact = analyzer.predict_impact(
            FixedPoint::from_float(1000.0),
            FixedPoint::from_float(100.0),
            TradeDirection::Buy,
        ).unwrap();
        
        // Should predict positive impact for buy order
        assert!(predicted_impact >= FixedPoint::ZERO);
    }

    #[test]
    fn test_model_parameter_defaults() {
        let params = ImpactModelParams::default();
        
        assert!(params.temporary_impact_coefficient > FixedPoint::ZERO);
        assert!(params.temporary_impact_exponent > FixedPoint::ZERO);
        assert!(params.permanent_impact_coefficient > FixedPoint::ZERO);
        assert!(params.decay_rate > FixedPoint::ZERO);
    }

    #[test]
    fn test_cross_impact_relationship() {
        let mut analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
        
        let relationship = CrossImpactRelationship {
            primary_asset: "AAPL".to_string(),
            related_asset: "MSFT".to_string(),
            correlation_coefficient: FixedPoint::from_float(0.7),
            impact_transmission_rate: FixedPoint::from_float(0.3),
            lag_milliseconds: 100,
        };
        
        analyzer.add_cross_impact_relationship(relationship);
        
        let correlation = analyzer.calculate_cross_impact_correlation().unwrap();
        assert_eq!(correlation, FixedPoint::from_float(0.7));
    }
}