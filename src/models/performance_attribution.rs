//! Performance Attribution Analysis Module
//! 
//! This module provides detailed performance attribution analysis for high-frequency
//! quoting strategies, breaking down returns into various contributing factors.

use crate::models::{
    performance_metrics::{ComprehensivePerformanceMetrics, PerformanceCalculator, PerformanceData, TradeData, MarketData},
    backtesting_framework::{SimulationResults, AttributionResults, PerformanceAnalysisConfig, AttributionDepth},
};
use nalgebra as na;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AttributionError {
    #[error("Attribution calculation error: {0}")]
    CalculationError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Detailed performance attribution analyzer
pub struct PerformanceAttributionAnalyzer {
    config: PerformanceAnalysisConfig,
}

impl PerformanceAttributionAnalyzer {
    pub fn new(config: &PerformanceAnalysisConfig) -> Result<Self, AttributionError> {
        Ok(Self {
            config: config.clone(),
        })
    }

    /// Analyze performance with detailed attribution
    pub async fn analyze_performance(
        &self,
        results: &SimulationResults,
    ) -> Result<AttributionResults, AttributionError> {
        let mut attribution_breakdown = HashMap::new();

        // Calculate total P&L
        let total_pnl = results.pnl_series.last().unwrap_or(&0.0) - results.pnl_series.first().unwrap_or(&0.0);

        // Perform attribution analysis based on configured depth
        match self.config.attribution_depth {
            AttributionDepth::Basic => {
                self.basic_attribution(results, &mut attribution_breakdown).await?;
            },
            AttributionDepth::Detailed => {
                self.detailed_attribution(results, &mut attribution_breakdown).await?;
            },
            AttributionDepth::Comprehensive => {
                self.comprehensive_attribution(results, &mut attribution_breakdown).await?;
            },
        }

        // Risk decomposition if enabled
        if self.config.risk_decomposition {
            self.risk_decomposition(results, &mut attribution_breakdown).await?;
        }

        // Transaction cost analysis if enabled
        if self.config.transaction_cost_analysis {
            self.transaction_cost_attribution(results, &mut attribution_breakdown).await?;
        }

        // Market making specific metrics if enabled
        if self.config.market_making_metrics {
            self.market_making_attribution(results, &mut attribution_breakdown).await?;
        }

        Ok(AttributionResults {
            total_pnl,
            attribution_breakdown,
        })
    }

    /// Basic attribution analysis
    async fn basic_attribution(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Trading P&L
        let trading_pnl: f64 = results.trades.iter()
            .map(|trade| {
                // Simplified P&L calculation
                match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => -trade.price * trade.quantity,
                    crate::models::performance_metrics::TradeSide::Sell => trade.price * trade.quantity,
                }
            })
            .sum();

        attribution.insert("Trading P&L".to_string(), trading_pnl);

        // Transaction costs
        let transaction_costs: f64 = results.trades.iter()
            .map(|trade| trade.transaction_cost)
            .sum();

        attribution.insert("Transaction Costs".to_string(), -transaction_costs);

        // Market impact costs
        let market_impact_costs: f64 = results.trades.iter()
            .map(|trade| trade.market_impact)
            .sum();

        attribution.insert("Market Impact".to_string(), -market_impact_costs);

        Ok(())
    }

    /// Detailed attribution analysis
    async fn detailed_attribution(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Start with basic attribution
        self.basic_attribution(results, attribution).await?;

        // Spread capture analysis
        let spread_capture = self.calculate_spread_capture(results)?;
        attribution.insert("Spread Capture".to_string(), spread_capture);

        // Inventory carrying costs
        let inventory_costs = self.calculate_inventory_costs(results)?;
        attribution.insert("Inventory Costs".to_string(), inventory_costs);

        // Adverse selection costs
        let adverse_selection = self.calculate_adverse_selection(results)?;
        attribution.insert("Adverse Selection".to_string(), adverse_selection);

        // Timing alpha
        let timing_alpha = self.calculate_timing_alpha(results)?;
        attribution.insert("Timing Alpha".to_string(), timing_alpha);

        Ok(())
    }

    /// Comprehensive attribution analysis
    async fn comprehensive_attribution(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Start with detailed attribution
        self.detailed_attribution(results, attribution).await?;

        // Volatility timing
        let volatility_timing = self.calculate_volatility_timing(results)?;
        attribution.insert("Volatility Timing".to_string(), volatility_timing);

        // Liquidity provision premium
        let liquidity_premium = self.calculate_liquidity_premium(results)?;
        attribution.insert("Liquidity Premium".to_string(), liquidity_premium);

        // Order flow toxicity costs
        let toxicity_costs = self.calculate_toxicity_costs(results)?;
        attribution.insert("Order Flow Toxicity".to_string(), toxicity_costs);

        // Regime-specific performance
        let regime_attribution = self.calculate_regime_attribution(results)?;
        for (regime, pnl) in regime_attribution {
            attribution.insert(format!("Regime: {}", regime), pnl);
        }

        // Intraday patterns
        let intraday_attribution = self.calculate_intraday_attribution(results)?;
        for (period, pnl) in intraday_attribution {
            attribution.insert(format!("Intraday: {}", period), pnl);
        }

        Ok(())
    }

    /// Risk decomposition analysis
    async fn risk_decomposition(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Market risk contribution
        let market_risk = self.calculate_market_risk_contribution(results)?;
        attribution.insert("Market Risk".to_string(), market_risk);

        // Idiosyncratic risk contribution
        let idiosyncratic_risk = self.calculate_idiosyncratic_risk(results)?;
        attribution.insert("Idiosyncratic Risk".to_string(), idiosyncratic_risk);

        // Liquidity risk contribution
        let liquidity_risk = self.calculate_liquidity_risk_contribution(results)?;
        attribution.insert("Liquidity Risk".to_string(), liquidity_risk);

        Ok(())
    }

    /// Transaction cost attribution
    async fn transaction_cost_attribution(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Explicit transaction costs
        let explicit_costs: f64 = results.trades.iter()
            .map(|trade| trade.transaction_cost)
            .sum();
        attribution.insert("Explicit Transaction Costs".to_string(), -explicit_costs);

        // Implicit costs (market impact)
        let implicit_costs: f64 = results.trades.iter()
            .map(|trade| trade.market_impact)
            .sum();
        attribution.insert("Implicit Transaction Costs".to_string(), -implicit_costs);

        // Timing costs
        let timing_costs = self.calculate_timing_costs(results)?;
        attribution.insert("Timing Costs".to_string(), timing_costs);

        // Opportunity costs
        let opportunity_costs = self.calculate_opportunity_costs(results)?;
        attribution.insert("Opportunity Costs".to_string(), opportunity_costs);

        Ok(())
    }

    /// Market making specific attribution
    async fn market_making_attribution(
        &self,
        results: &SimulationResults,
        attribution: &mut HashMap<String, f64>,
    ) -> Result<(), AttributionError> {
        // Bid-ask spread capture
        let spread_capture = self.calculate_spread_capture(results)?;
        attribution.insert("Bid-Ask Spread Capture".to_string(), spread_capture);

        // Inventory risk premium
        let inventory_premium = self.calculate_inventory_risk_premium(results)?;
        attribution.insert("Inventory Risk Premium".to_string(), inventory_premium);

        // Order flow payment
        let order_flow_payment = self.calculate_order_flow_payment(results)?;
        attribution.insert("Order Flow Payment".to_string(), order_flow_payment);

        // Quote competitiveness alpha
        let competitiveness_alpha = self.calculate_competitiveness_alpha(results)?;
        attribution.insert("Quote Competitiveness Alpha".to_string(), competitiveness_alpha);

        Ok(())
    }

    // Helper methods for specific calculations

    fn calculate_spread_capture(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut total_spread_capture = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                let spread_capture = match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => {
                        (market_tick.mid_price - trade.price).max(0.0) * trade.quantity.abs()
                    },
                    crate::models::performance_metrics::TradeSide::Sell => {
                        (trade.price - market_tick.mid_price).max(0.0) * trade.quantity.abs()
                    },
                };
                total_spread_capture += spread_capture;
            }
        }

        Ok(total_spread_capture)
    }

    fn calculate_inventory_costs(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut inventory_costs = 0.0;
        let risk_penalty = 0.01; // 1% annual inventory risk penalty

        for (i, &position) in results.positions.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Calculate inventory carrying cost
                let time_delta = if i > 0 { 1.0 / (365.0 * 24.0 * 60.0 * 60.0) } else { 0.0 }; // 1 second
                let inventory_cost = position.abs() * market_tick.mid_price * risk_penalty * time_delta;
                inventory_costs -= inventory_cost;
            }
        }

        Ok(inventory_costs)
    }

    fn calculate_adverse_selection(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut adverse_selection = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            // Look at price movement after trade (simplified)
            if let (Some(current_tick), Some(future_tick)) = (
                results.market_data.ticks.get(i),
                results.market_data.ticks.get(i + 10) // 10 ticks later
            ) {
                let price_change = future_tick.mid_price - current_tick.mid_price;
                let adverse_cost = match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => {
                        if price_change < 0.0 { price_change * trade.quantity.abs() } else { 0.0 }
                    },
                    crate::models::performance_metrics::TradeSide::Sell => {
                        if price_change > 0.0 { -price_change * trade.quantity.abs() } else { 0.0 }
                    },
                };
                adverse_selection += adverse_cost;
            }
        }

        Ok(adverse_selection)
    }

    fn calculate_timing_alpha(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified timing alpha calculation
        let mut timing_alpha = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let (Some(prev_tick), Some(current_tick)) = (
                results.market_data.ticks.get(i.saturating_sub(1)),
                results.market_data.ticks.get(i)
            ) {
                let momentum = current_tick.mid_price - prev_tick.mid_price;
                let timing_contribution = match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => {
                        if momentum > 0.0 { momentum * trade.quantity.abs() } else { 0.0 }
                    },
                    crate::models::performance_metrics::TradeSide::Sell => {
                        if momentum < 0.0 { -momentum * trade.quantity.abs() } else { 0.0 }
                    },
                };
                timing_alpha += timing_contribution;
            }
        }

        Ok(timing_alpha)
    }

    fn calculate_volatility_timing(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Calculate returns from volatility timing
        let mut vol_timing_pnl = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Higher volatility should lead to wider spreads and higher profits
                let vol_premium = market_tick.volatility * trade.quantity.abs() * 0.1; // 10% of volatility
                vol_timing_pnl += vol_premium;
            }
        }

        Ok(vol_timing_pnl)
    }

    fn calculate_liquidity_premium(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut liquidity_premium = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Lower liquidity (higher spread) should provide higher premium
                let spread_ratio = market_tick.spread / market_tick.mid_price;
                let premium = spread_ratio * trade.quantity.abs() * market_tick.mid_price * 0.5;
                liquidity_premium += premium;
            }
        }

        Ok(liquidity_premium)
    }

    fn calculate_toxicity_costs(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified order flow toxicity calculation
        let mut toxicity_costs = 0.0;
        let mut recent_trades = std::collections::VecDeque::new();
        
        for (i, trade) in results.trades.iter().enumerate() {
            // Keep only recent trades (last 100)
            recent_trades.push_back(trade);
            if recent_trades.len() > 100 {
                recent_trades.pop_front();
            }
            
            // Calculate trade imbalance
            let buy_volume: f64 = recent_trades.iter()
                .filter(|t| matches!(t.side, crate::models::performance_metrics::TradeSide::Buy))
                .map(|t| t.quantity.abs())
                .sum();
            let sell_volume: f64 = recent_trades.iter()
                .filter(|t| matches!(t.side, crate::models::performance_metrics::TradeSide::Sell))
                .map(|t| t.quantity.abs())
                .sum();
            
            let imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-8);
            let toxicity_cost = imbalance.abs() * trade.quantity.abs() * 0.001; // 0.1% cost
            toxicity_costs -= toxicity_cost;
        }

        Ok(toxicity_costs)
    }

    fn calculate_regime_attribution(&self, results: &SimulationResults) -> Result<HashMap<String, f64>, AttributionError> {
        let mut regime_attribution = HashMap::new();
        
        // Simplified regime detection based on volatility
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                let regime = if market_tick.volatility > 0.3 {
                    "High Volatility"
                } else if market_tick.volatility > 0.15 {
                    "Medium Volatility"
                } else {
                    "Low Volatility"
                };
                
                let trade_pnl = match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => -trade.price * trade.quantity,
                    crate::models::performance_metrics::TradeSide::Sell => trade.price * trade.quantity,
                };
                
                *regime_attribution.entry(regime.to_string()).or_insert(0.0) += trade_pnl;
            }
        }

        Ok(regime_attribution)
    }

    fn calculate_intraday_attribution(&self, results: &SimulationResults) -> Result<HashMap<String, f64>, AttributionError> {
        let mut intraday_attribution = HashMap::new();
        
        for trade in &results.trades {
            // Convert timestamp to hour of day (simplified)
            let hour = ((trade.timestamp as u64) % (24 * 3600)) / 3600;
            let period = match hour {
                0..=5 => "Overnight",
                6..=9 => "Morning",
                10..=15 => "Midday",
                16..=20 => "Afternoon",
                _ => "Evening",
            };
            
            let trade_pnl = match trade.side {
                crate::models::performance_metrics::TradeSide::Buy => -trade.price * trade.quantity,
                crate::models::performance_metrics::TradeSide::Sell => trade.price * trade.quantity,
            };
            
            *intraday_attribution.entry(period.to_string()).or_insert(0.0) += trade_pnl;
        }

        Ok(intraday_attribution)
    }

    fn calculate_market_risk_contribution(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified market risk calculation using position exposure
        let mut market_risk_pnl = 0.0;
        
        for (i, &position) in results.positions.iter().enumerate() {
            if let (Some(current_tick), Some(prev_tick)) = (
                results.market_data.ticks.get(i),
                results.market_data.ticks.get(i.saturating_sub(1))
            ) {
                let market_return = (current_tick.mid_price - prev_tick.mid_price) / prev_tick.mid_price;
                market_risk_pnl += position * prev_tick.mid_price * market_return;
            }
        }

        Ok(market_risk_pnl)
    }

    fn calculate_idiosyncratic_risk(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified idiosyncratic risk (residual after market risk)
        let total_pnl = results.pnl_series.last().unwrap_or(&0.0) - results.pnl_series.first().unwrap_or(&0.0);
        let market_risk = self.calculate_market_risk_contribution(results)?;
        Ok(total_pnl - market_risk)
    }

    fn calculate_liquidity_risk_contribution(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut liquidity_risk_pnl = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Liquidity risk based on spread widening
                let liquidity_cost = market_tick.spread * trade.quantity.abs() * 0.5;
                liquidity_risk_pnl -= liquidity_cost;
            }
        }

        Ok(liquidity_risk_pnl)
    }

    fn calculate_timing_costs(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified timing cost calculation
        let mut timing_costs = 0.0;
        
        for trade in &results.trades {
            // Assume 0.1 bps timing cost per trade
            timing_costs -= trade.quantity.abs() * trade.price * 0.000001;
        }

        Ok(timing_costs)
    }

    fn calculate_opportunity_costs(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Simplified opportunity cost calculation
        let mut opportunity_costs = 0.0;
        
        // Calculate missed opportunities (simplified)
        for (i, &position) in results.positions.iter().enumerate() {
            if position == 0.0 && i > 0 {
                if let (Some(current_tick), Some(prev_tick)) = (
                    results.market_data.ticks.get(i),
                    results.market_data.ticks.get(i - 1)
                ) {
                    let price_move = (current_tick.mid_price - prev_tick.mid_price).abs();
                    opportunity_costs -= price_move * 100.0; // Assume 100 shares missed opportunity
                }
            }
        }

        Ok(opportunity_costs)
    }

    fn calculate_inventory_risk_premium(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        let mut risk_premium = 0.0;
        
        for (i, &position) in results.positions.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Risk premium for holding inventory
                let premium = position.abs() * market_tick.volatility * 0.01; // 1% of volatility
                risk_premium += premium;
            }
        }

        Ok(risk_premium)
    }

    fn calculate_order_flow_payment(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Payment for providing liquidity
        let mut order_flow_payment = 0.0;
        
        for trade in &results.trades {
            // Assume 0.1 bps payment for providing liquidity
            order_flow_payment += trade.quantity.abs() * trade.price * 0.000001;
        }

        Ok(order_flow_payment)
    }

    fn calculate_competitiveness_alpha(&self, results: &SimulationResults) -> Result<f64, AttributionError> {
        // Alpha from competitive quoting
        let mut competitiveness_alpha = 0.0;
        
        for (i, trade) in results.trades.iter().enumerate() {
            if let Some(market_tick) = results.market_data.ticks.get(i) {
                // Better execution than mid-price
                let execution_improvement = match trade.side {
                    crate::models::performance_metrics::TradeSide::Buy => {
                        (market_tick.mid_price - trade.price).max(0.0)
                    },
                    crate::models::performance_metrics::TradeSide::Sell => {
                        (trade.price - market_tick.mid_price).max(0.0)
                    },
                };
                competitiveness_alpha += execution_improvement * trade.quantity.abs();
            }
        }

        Ok(competitiveness_alpha)
    }
}