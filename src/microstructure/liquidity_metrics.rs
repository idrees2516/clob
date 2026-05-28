//! Real-Time Liquidity Metrics
//! 
//! Implements comprehensive liquidity measurement including:
//! - Bid-ask spread calculation (absolute and relative)
//! - Effective spread and realized spread measurement
//! - Depth-at-best and order book slope calculation
//! - Price impact measurement for trade classification

use super::{MarketData, OrderBookSnapshot, Trade, TradeDirection, MicrostructureError, MicrostructureResult};
use crate::math::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Comprehensive liquidity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    pub timestamp: u64,
    pub bid_ask_spread_absolute: FixedPoint,
    pub bid_ask_spread_relative: FixedPoint,
    pub effective_spread: FixedPoint,
    pub realized_spread: FixedPoint,
    pub depth_at_best_bid: FixedPoint,
    pub depth_at_best_ask: FixedPoint,
    pub order_book_slope_bid: FixedPoint,
    pub order_book_slope_ask: FixedPoint,
    pub price_impact: FixedPoint,
    pub market_quality_score: FixedPoint,
}

/// Real-time liquidity metrics calculator
pub struct LiquidityMetricsCalculator {
    market_data_history: VecDeque<MarketData>,
    trade_history: VecDeque<Trade>,
    orderbook_history: VecDeque<OrderBookSnapshot>,
    max_history_size: usize,
    lookback_window_ms: u64,
}

impl LiquidityMetricsCalculator {
    /// Create new liquidity metrics calculator
    pub fn new(max_history_size: usize, lookback_window_ms: u64) -> Self {
        Self {
            market_data_history: VecDeque::with_capacity(max_history_size),
            trade_history: VecDeque::with_capacity(max_history_size),
            orderbook_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            lookback_window_ms,
        }
    }

    /// Update with new market data
    pub fn update_market_data(&mut self, market_data: MarketData) {
        self.market_data_history.push_back(market_data);
        if self.market_data_history.len() > self.max_history_size {
            self.market_data_history.pop_front();
        }
    }

    /// Update with new trade
    pub fn update_trade(&mut self, trade: Trade) {
        self.trade_history.push_back(trade);
        if self.trade_history.len() > self.max_history_size {
            self.trade_history.pop_front();
        }
    }

    /// Update with new order book snapshot
    pub fn update_orderbook(&mut self, orderbook: OrderBookSnapshot) {
        self.orderbook_history.push_back(orderbook);
        if self.orderbook_history.len() > self.max_history_size {
            self.orderbook_history.pop_front();
        }
    }

    /// Calculate comprehensive liquidity metrics
    pub fn calculate_metrics(&self) -> MicrostructureResult<LiquidityMetrics> {
        let current_market_data = self.market_data_history.back()
            .ok_or_else(|| MicrostructureError::InsufficientData("No market data available".to_string()))?;

        let current_orderbook = self.orderbook_history.back()
            .ok_or_else(|| MicrostructureError::InsufficientData("No order book data available".to_string()))?;

        let timestamp = current_market_data.timestamp;
        
        // Calculate bid-ask spreads
        let (bid_ask_spread_absolute, bid_ask_spread_relative) = 
            self.calculate_bid_ask_spreads(current_market_data)?;

        // Calculate effective and realized spreads
        let effective_spread = self.calculate_effective_spread()?;
        let realized_spread = self.calculate_realized_spread()?;

        // Calculate depth at best
        let (depth_at_best_bid, depth_at_best_ask) = 
            self.calculate_depth_at_best(current_orderbook)?;

        // Calculate order book slopes
        let (order_book_slope_bid, order_book_slope_ask) = 
            self.calculate_order_book_slopes(current_orderbook)?;

        // Calculate price impact
        let price_impact = self.calculate_price_impact()?;

        // Calculate overall market quality score
        let market_quality_score = self.calculate_market_quality_score(
            bid_ask_spread_relative,
            effective_spread,
            depth_at_best_bid + depth_at_best_ask,
            price_impact,
        )?;

        Ok(LiquidityMetrics {
            timestamp,
            bid_ask_spread_absolute,
            bid_ask_spread_relative,
            effective_spread,
            realized_spread,
            depth_at_best_bid,
            depth_at_best_ask,
            order_book_slope_bid,
            order_book_slope_ask,
            price_impact,
            market_quality_score,
        })
    }

    /// Calculate absolute and relative bid-ask spreads
    fn calculate_bid_ask_spreads(&self, market_data: &MarketData) -> MicrostructureResult<(FixedPoint, FixedPoint)> {
        if market_data.ask_price <= market_data.bid_price {
            return Err(MicrostructureError::DataQuality(
                "Ask price must be greater than bid price".to_string()
            ));
        }

        let absolute_spread = market_data.ask_price - market_data.bid_price;
        let mid_price = (market_data.bid_price + market_data.ask_price) / FixedPoint::from_float(2.0);
        
        if mid_price == FixedPoint::ZERO {
            return Err(MicrostructureError::CalculationError(
                "Mid price cannot be zero".to_string()
            ));
        }

        let relative_spread = absolute_spread / mid_price;

        Ok((absolute_spread, relative_spread))
    }

    /// Calculate effective spread using recent trades
    fn calculate_effective_spread(&self) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.market_data_history.back()
            .map(|md| md.timestamp.saturating_sub(self.lookback_window_ms))
            .unwrap_or(0);

        let recent_trades: Vec<_> = self.trade_history.iter()
            .filter(|trade| trade.timestamp >= cutoff_time)
            .collect();

        if recent_trades.is_empty() {
            return Ok(FixedPoint::ZERO);
        }

        let mut total_effective_spread = FixedPoint::ZERO;
        let mut valid_trades = 0;

        for trade in recent_trades {
            if let Some(market_data) = self.find_market_data_at_time(trade.timestamp) {
                let mid_price = (market_data.bid_price + market_data.ask_price) / FixedPoint::from_float(2.0);
                
                if mid_price > FixedPoint::ZERO {
                    let effective_spread = match trade.direction {
                        TradeDirection::Buy => (trade.price - mid_price) * FixedPoint::from_float(2.0),
                        TradeDirection::Sell => (mid_price - trade.price) * FixedPoint::from_float(2.0),
                        TradeDirection::Unknown => FixedPoint::ZERO,
                    };
                    
                    total_effective_spread += effective_spread;
                    valid_trades += 1;
                }
            }
        }

        if valid_trades == 0 {
            Ok(FixedPoint::ZERO)
        } else {
            Ok(total_effective_spread / FixedPoint::from_int(valid_trades as i64))
        }
    }

    /// Calculate realized spread (effective spread minus price impact)
    fn calculate_realized_spread(&self) -> MicrostructureResult<FixedPoint> {
        let effective_spread = self.calculate_effective_spread()?;
        let price_impact = self.calculate_price_impact()?;
        
        Ok(effective_spread - price_impact)
    }

    /// Calculate depth at best bid and ask
    fn calculate_depth_at_best(&self, orderbook: &OrderBookSnapshot) -> MicrostructureResult<(FixedPoint, FixedPoint)> {
        let best_bid_depth = orderbook.bids.first()
            .map(|level| level.volume)
            .unwrap_or(FixedPoint::ZERO);

        let best_ask_depth = orderbook.asks.first()
            .map(|level| level.volume)
            .unwrap_or(FixedPoint::ZERO);

        Ok((best_bid_depth, best_ask_depth))
    }

    /// Calculate order book slopes (price sensitivity to volume)
    fn calculate_order_book_slopes(&self, orderbook: &OrderBookSnapshot) -> MicrostructureResult<(FixedPoint, FixedPoint)> {
        let bid_slope = self.calculate_slope(&orderbook.bids, false)?;
        let ask_slope = self.calculate_slope(&orderbook.asks, true)?;
        
        Ok((bid_slope, ask_slope))
    }

    /// Calculate slope for one side of the order book
    fn calculate_slope(&self, levels: &[super::OrderBookLevel], is_ask_side: bool) -> MicrostructureResult<FixedPoint> {
        if levels.len() < 2 {
            return Ok(FixedPoint::ZERO);
        }

        let mut cumulative_volume = FixedPoint::ZERO;
        let mut weighted_price_sum = FixedPoint::ZERO;
        let mut total_weight = FixedPoint::ZERO;

        for (i, level) in levels.iter().enumerate().take(5) { // Use top 5 levels
            cumulative_volume += level.volume;
            
            if i > 0 {
                let price_diff = if is_ask_side {
                    level.price - levels[0].price
                } else {
                    levels[0].price - level.price
                };
                
                if cumulative_volume > FixedPoint::ZERO {
                    let weight = level.volume;
                    weighted_price_sum += price_diff * weight;
                    total_weight += weight;
                }
            }
        }

        if total_weight > FixedPoint::ZERO && cumulative_volume > FixedPoint::ZERO {
            Ok(weighted_price_sum / (total_weight * cumulative_volume))
        } else {
            Ok(FixedPoint::ZERO)
        }
    }

    /// Calculate price impact from recent trades
    fn calculate_price_impact(&self) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.market_data_history.back()
            .map(|md| md.timestamp.saturating_sub(self.lookback_window_ms))
            .unwrap_or(0);

        let recent_trades: Vec<_> = self.trade_history.iter()
            .filter(|trade| trade.timestamp >= cutoff_time)
            .collect();

        if recent_trades.len() < 2 {
            return Ok(FixedPoint::ZERO);
        }

        let mut total_impact = FixedPoint::ZERO;
        let mut valid_impacts = 0;

        for window in recent_trades.windows(2) {
            let trade1 = window[0];
            let trade2 = window[1];
            
            if trade1.direction != TradeDirection::Unknown && trade2.direction != TradeDirection::Unknown {
                let price_change = trade2.price - trade1.price;
                let direction_factor = match trade1.direction {
                    TradeDirection::Buy => FixedPoint::from_float(1.0),
                    TradeDirection::Sell => FixedPoint::from_float(-1.0),
                    TradeDirection::Unknown => FixedPoint::ZERO,
                };
                
                let impact = price_change * direction_factor;
                total_impact += impact;
                valid_impacts += 1;
            }
        }

        if valid_impacts == 0 {
            Ok(FixedPoint::ZERO)
        } else {
            Ok(total_impact / FixedPoint::from_int(valid_impacts as i64))
        }
    }

    /// Calculate overall market quality score (0-1, higher is better)
    fn calculate_market_quality_score(
        &self,
        relative_spread: FixedPoint,
        effective_spread: FixedPoint,
        total_depth: FixedPoint,
        price_impact: FixedPoint,
    ) -> MicrostructureResult<FixedPoint> {
        // Normalize components (lower spreads and impact = better quality)
        let spread_score = FixedPoint::from_float(1.0) / (FixedPoint::from_float(1.0) + relative_spread * FixedPoint::from_float(1000.0));
        let effective_spread_score = FixedPoint::from_float(1.0) / (FixedPoint::from_float(1.0) + effective_spread * FixedPoint::from_float(1000.0));
        let depth_score = total_depth / (total_depth + FixedPoint::from_float(1000.0)); // Normalize depth
        let impact_score = FixedPoint::from_float(1.0) / (FixedPoint::from_float(1.0) + price_impact.abs() * FixedPoint::from_float(1000.0));

        // Weighted average (equal weights for simplicity)
        let quality_score = (spread_score + effective_spread_score + depth_score + impact_score) / FixedPoint::from_float(4.0);
        
        Ok(quality_score)
    }

    /// Find market data closest to given timestamp
    fn find_market_data_at_time(&self, timestamp: u64) -> Option<&MarketData> {
        self.market_data_history.iter()
            .min_by_key(|md| (md.timestamp as i64 - timestamp as i64).abs())
    }

    /// Get current liquidity summary
    pub fn get_liquidity_summary(&self) -> MicrostructureResult<LiquiditySummary> {
        let metrics = self.calculate_metrics()?;
        
        Ok(LiquiditySummary {
            timestamp: metrics.timestamp,
            spread_bps: (metrics.bid_ask_spread_relative * FixedPoint::from_float(10000.0)).to_float() as u32,
            depth_usd: ((metrics.depth_at_best_bid + metrics.depth_at_best_ask) * 
                       self.market_data_history.back().unwrap().bid_price).to_float() as u64,
            quality_score: (metrics.market_quality_score * FixedPoint::from_float(100.0)).to_float() as u8,
            impact_bps: (metrics.price_impact.abs() * FixedPoint::from_float(10000.0)).to_float() as u32,
        })
    }
}

/// Simplified liquidity summary for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquiditySummary {
    pub timestamp: u64,
    pub spread_bps: u32,      // Spread in basis points
    pub depth_usd: u64,       // Total depth in USD
    pub quality_score: u8,    // Quality score 0-100
    pub impact_bps: u32,      // Price impact in basis points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_market_data(timestamp: u64, bid: f64, ask: f64, bid_vol: f64, ask_vol: f64) -> MarketData {
        MarketData {
            timestamp,
            bid_price: FixedPoint::from_float(bid),
            ask_price: FixedPoint::from_float(ask),
            bid_volume: FixedPoint::from_float(bid_vol),
            ask_volume: FixedPoint::from_float(ask_vol),
            last_trade_price: FixedPoint::from_float((bid + ask) / 2.0),
            last_trade_volume: FixedPoint::from_float(100.0),
            trade_direction: Some(TradeDirection::Buy),
        }
    }

    fn create_test_orderbook(timestamp: u64) -> OrderBookSnapshot {
        OrderBookSnapshot {
            timestamp,
            bids: vec![
                super::OrderBookLevel {
                    price: FixedPoint::from_float(99.95),
                    volume: FixedPoint::from_float(1000.0),
                    order_count: 5,
                },
                super::OrderBookLevel {
                    price: FixedPoint::from_float(99.90),
                    volume: FixedPoint::from_float(2000.0),
                    order_count: 8,
                },
            ],
            asks: vec![
                super::OrderBookLevel {
                    price: FixedPoint::from_float(100.05),
                    volume: FixedPoint::from_float(1500.0),
                    order_count: 6,
                },
                super::OrderBookLevel {
                    price: FixedPoint::from_float(100.10),
                    volume: FixedPoint::from_float(2500.0),
                    order_count: 10,
                },
            ],
        }
    }

    #[test]
    fn test_bid_ask_spread_calculation() {
        let mut calculator = LiquidityMetricsCalculator::new(100, 1000);
        let market_data = create_test_market_data(1000, 99.95, 100.05, 1000.0, 1500.0);
        
        calculator.update_market_data(market_data.clone());
        calculator.update_orderbook(create_test_orderbook(1000));

        let metrics = calculator.calculate_metrics().unwrap();
        
        // Absolute spread should be 0.10
        assert!((metrics.bid_ask_spread_absolute.to_float() - 0.10).abs() < 1e-6);
        
        // Relative spread should be approximately 0.10/100.00 = 0.001
        assert!((metrics.bid_ask_spread_relative.to_float() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_depth_calculation() {
        let mut calculator = LiquidityMetricsCalculator::new(100, 1000);
        let market_data = create_test_market_data(1000, 99.95, 100.05, 1000.0, 1500.0);
        
        calculator.update_market_data(market_data);
        calculator.update_orderbook(create_test_orderbook(1000));

        let metrics = calculator.calculate_metrics().unwrap();
        
        // Best bid depth should be 1000.0
        assert!((metrics.depth_at_best_bid.to_float() - 1000.0).abs() < 1e-6);
        
        // Best ask depth should be 1500.0
        assert!((metrics.depth_at_best_ask.to_float() - 1500.0).abs() < 1e-6);
    }

    #[test]
    fn test_effective_spread_calculation() {
        let mut calculator = LiquidityMetricsCalculator::new(100, 1000);
        let market_data = create_test_market_data(1000, 99.95, 100.05, 1000.0, 1500.0);
        
        calculator.update_market_data(market_data);
        calculator.update_orderbook(create_test_orderbook(1000));
        
        // Add a buy trade at ask price
        let trade = Trade {
            timestamp: 1000,
            price: FixedPoint::from_float(100.05),
            volume: FixedPoint::from_float(100.0),
            direction: TradeDirection::Buy,
            trade_id: 1,
        };
        calculator.update_trade(trade);

        let metrics = calculator.calculate_metrics().unwrap();
        
        // Effective spread should be positive for buy trade above mid
        assert!(metrics.effective_spread.to_float() > 0.0);
    }

    #[test]
    fn test_market_quality_score() {
        let mut calculator = LiquidityMetricsCalculator::new(100, 1000);
        let market_data = create_test_market_data(1000, 99.95, 100.05, 1000.0, 1500.0);
        
        calculator.update_market_data(market_data);
        calculator.update_orderbook(create_test_orderbook(1000));

        let metrics = calculator.calculate_metrics().unwrap();
        
        // Quality score should be between 0 and 1
        assert!(metrics.market_quality_score.to_float() >= 0.0);
        assert!(metrics.market_quality_score.to_float() <= 1.0);
    }

    #[test]
    fn test_liquidity_summary() {
        let mut calculator = LiquidityMetricsCalculator::new(100, 1000);
        let market_data = create_test_market_data(1000, 99.95, 100.05, 1000.0, 1500.0);
        
        calculator.update_market_data(market_data);
        calculator.update_orderbook(create_test_orderbook(1000));

        let summary = calculator.get_liquidity_summary().unwrap();
        
        assert_eq!(summary.timestamp, 1000);
        assert!(summary.spread_bps > 0);
        assert!(summary.depth_usd > 0);
        assert!(summary.quality_score <= 100);
    }
}