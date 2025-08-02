//! Order Flow Analysis
//! 
//! Implements sophisticated order flow analysis including:
//! - Lee-Ready algorithm for trade classification
//! - Order flow imbalance calculation
//! - VPIN (Volume-Synchronized Probability of Informed Trading) metric
//! - Bulk volume classification for institutional flow detection

use super::{MarketData, OrderBookSnapshot, Trade, TradeDirection, MicrostructureError, MicrostructureResult};
use crate::math::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};

/// Order flow analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowMetrics {
    pub timestamp: u64,
    pub order_flow_imbalance: FixedPoint,
    pub vpin: FixedPoint,
    pub toxic_flow_probability: FixedPoint,
    pub institutional_flow_ratio: FixedPoint,
    pub buy_volume_ratio: FixedPoint,
    pub sell_volume_ratio: FixedPoint,
    pub trade_classification_accuracy: FixedPoint,
}

/// Trade classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeClassification {
    pub trade_id: u64,
    pub timestamp: u64,
    pub price: FixedPoint,
    pub volume: FixedPoint,
    pub direction: TradeDirection,
    pub confidence: FixedPoint,
    pub method: ClassificationMethod,
}

/// Classification method used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassificationMethod {
    LeeReady,
    Quote,
    Tick,
    Depth,
    Hybrid,
}

/// VPIN bucket for volume synchronization
#[derive(Debug, Clone)]
struct VPINBucket {
    timestamp: u64,
    buy_volume: FixedPoint,
    sell_volume: FixedPoint,
    total_volume: FixedPoint,
    imbalance: FixedPoint,
}

/// Bulk volume classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkVolumeClassification {
    pub timestamp: u64,
    pub is_institutional: bool,
    pub volume_percentile: FixedPoint,
    pub size_category: VolumeCategory,
    pub clustering_score: FixedPoint,
}

/// Volume size categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolumeCategory {
    Small,      // < 25th percentile
    Medium,     // 25th - 75th percentile  
    Large,      // 75th - 95th percentile
    Block,      // > 95th percentile
}

/// Order flow analyzer
pub struct OrderFlowAnalyzer {
    market_data_history: VecDeque<MarketData>,
    trade_history: VecDeque<Trade>,
    orderbook_history: VecDeque<OrderBookSnapshot>,
    classified_trades: VecDeque<TradeClassification>,
    vpin_buckets: VecDeque<VPINBucket>,
    volume_statistics: VolumeStatistics,
    max_history_size: usize,
    vpin_bucket_size: FixedPoint,
    lookback_window_ms: u64,
}

/// Volume statistics for classification
#[derive(Debug, Clone)]
struct VolumeStatistics {
    volume_percentiles: [FixedPoint; 101], // 0th to 100th percentile
    average_volume: FixedPoint,
    volume_std: FixedPoint,
    last_update: u64,
}

impl OrderFlowAnalyzer {
    /// Create new order flow analyzer
    pub fn new(
        max_history_size: usize, 
        vpin_bucket_size: FixedPoint,
        lookback_window_ms: u64
    ) -> Self {
        Self {
            market_data_history: VecDeque::with_capacity(max_history_size),
            trade_history: VecDeque::with_capacity(max_history_size),
            orderbook_history: VecDeque::with_capacity(max_history_size),
            classified_trades: VecDeque::with_capacity(max_history_size),
            vpin_buckets: VecDeque::with_capacity(max_history_size / 10),
            volume_statistics: VolumeStatistics {
                volume_percentiles: [FixedPoint::ZERO; 101],
                average_volume: FixedPoint::ZERO,
                volume_std: FixedPoint::ZERO,
                last_update: 0,
            },
            max_history_size,
            vpin_bucket_size,
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

    /// Update with new trade and classify it
    pub fn update_trade(&mut self, trade: Trade) -> MicrostructureResult<TradeClassification> {
        // Classify the trade using Lee-Ready algorithm
        let classification = self.classify_trade_lee_ready(&trade)?;
        
        // Store classified trade
        self.classified_trades.push_back(classification.clone());
        if self.classified_trades.len() > self.max_history_size {
            self.classified_trades.pop_front();
        }

        // Store original trade
        self.trade_history.push_back(trade);
        if self.trade_history.len() > self.max_history_size {
            self.trade_history.pop_front();
        }

        // Update VPIN buckets
        self.update_vpin_buckets(&classification)?;

        // Update volume statistics periodically
        if self.trade_history.len() % 100 == 0 {
            self.update_volume_statistics()?;
        }

        Ok(classification)
    }

    /// Update with new order book snapshot
    pub fn update_orderbook(&mut self, orderbook: OrderBookSnapshot) {
        self.orderbook_history.push_back(orderbook);
        if self.orderbook_history.len() > self.max_history_size {
            self.orderbook_history.pop_front();
        }
    }

    /// Classify trade using Lee-Ready algorithm
    fn classify_trade_lee_ready(&self, trade: &Trade) -> MicrostructureResult<TradeClassification> {
        // Find the most recent market data before the trade
        let market_data = self.find_market_data_before_trade(trade.timestamp)
            .ok_or_else(|| MicrostructureError::InsufficientData(
                "No market data available for trade classification".to_string()
            ))?;

        let mid_price = (market_data.bid_price + market_data.ask_price) / FixedPoint::from_float(2.0);
        
        let (direction, confidence) = if trade.price > mid_price {
            // Trade above mid-price is likely a buy
            (TradeDirection::Buy, self.calculate_confidence(trade.price, market_data.ask_price, mid_price))
        } else if trade.price < mid_price {
            // Trade below mid-price is likely a sell
            (TradeDirection::Sell, self.calculate_confidence(mid_price, trade.price, market_data.bid_price))
        } else {
            // Trade at mid-price - use tick rule
            self.apply_tick_rule(trade)?
        };

        Ok(TradeClassification {
            trade_id: trade.trade_id,
            timestamp: trade.timestamp,
            price: trade.price,
            volume: trade.volume,
            direction,
            confidence,
            method: ClassificationMethod::LeeReady,
        })
    }

    /// Apply tick rule for trades at mid-price
    fn apply_tick_rule(&self, trade: &Trade) -> MicrostructureResult<(TradeDirection, FixedPoint)> {
        // Find previous trade to compare prices
        let previous_trade = self.trade_history.iter()
            .rev()
            .find(|t| t.timestamp < trade.timestamp && t.price != trade.price);

        match previous_trade {
            Some(prev) => {
                if trade.price > prev.price {
                    Ok((TradeDirection::Buy, FixedPoint::from_float(0.7)))
                } else if trade.price < prev.price {
                    Ok((TradeDirection::Sell, FixedPoint::from_float(0.7)))
                } else {
                    // Same price - use previous classification if available
                    let prev_classification = self.classified_trades.iter()
                        .rev()
                        .find(|c| c.timestamp < trade.timestamp);
                    
                    match prev_classification {
                        Some(prev_class) => Ok((prev_class.direction, FixedPoint::from_float(0.5))),
                        None => Ok((TradeDirection::Unknown, FixedPoint::ZERO)),
                    }
                }
            }
            None => Ok((TradeDirection::Unknown, FixedPoint::ZERO)),
        }
    }

    /// Calculate classification confidence based on price position
    fn calculate_confidence(&self, trade_price: FixedPoint, quote_price: FixedPoint, mid_price: FixedPoint) -> FixedPoint {
        let distance_from_mid = (trade_price - mid_price).abs();
        let spread = (quote_price - mid_price).abs();
        
        if spread > FixedPoint::ZERO {
            let confidence = distance_from_mid / spread;
            confidence.min(FixedPoint::from_float(1.0))
        } else {
            FixedPoint::from_float(0.5)
        }
    }

    /// Update VPIN buckets with new classified trade
    fn update_vpin_buckets(&mut self, classification: &TradeClassification) -> MicrostructureResult<()> {
        // Get or create current bucket
        let current_bucket = if let Some(bucket) = self.vpin_buckets.back_mut() {
            if bucket.total_volume < self.vpin_bucket_size {
                bucket
            } else {
                // Create new bucket
                self.vpin_buckets.push_back(VPINBucket {
                    timestamp: classification.timestamp,
                    buy_volume: FixedPoint::ZERO,
                    sell_volume: FixedPoint::ZERO,
                    total_volume: FixedPoint::ZERO,
                    imbalance: FixedPoint::ZERO,
                });
                self.vpin_buckets.back_mut().unwrap()
            }
        } else {
            // Create first bucket
            self.vpin_buckets.push_back(VPINBucket {
                timestamp: classification.timestamp,
                buy_volume: FixedPoint::ZERO,
                sell_volume: FixedPoint::ZERO,
                total_volume: FixedPoint::ZERO,
                imbalance: FixedPoint::ZERO,
            });
            self.vpin_buckets.back_mut().unwrap()
        };

        // Add volume to appropriate side
        match classification.direction {
            TradeDirection::Buy => current_bucket.buy_volume += classification.volume,
            TradeDirection::Sell => current_bucket.sell_volume += classification.volume,
            TradeDirection::Unknown => {
                // Split unknown trades equally
                let half_volume = classification.volume / FixedPoint::from_float(2.0);
                current_bucket.buy_volume += half_volume;
                current_bucket.sell_volume += half_volume;
            }
        }

        // Update totals
        current_bucket.total_volume = current_bucket.buy_volume + current_bucket.sell_volume;
        current_bucket.imbalance = (current_bucket.buy_volume - current_bucket.sell_volume).abs() 
            / current_bucket.total_volume.max(FixedPoint::from_float(1.0));

        // Limit bucket history
        while self.vpin_buckets.len() > self.max_history_size / 10 {
            self.vpin_buckets.pop_front();
        }

        Ok(())
    }

    /// Calculate comprehensive order flow metrics
    pub fn calculate_metrics(&self) -> MicrostructureResult<OrderFlowMetrics> {
        let timestamp = self.trade_history.back()
            .map(|t| t.timestamp)
            .unwrap_or(0);

        let order_flow_imbalance = self.calculate_order_flow_imbalance()?;
        let vpin = self.calculate_vpin()?;
        let toxic_flow_probability = self.calculate_toxic_flow_probability()?;
        let institutional_flow_ratio = self.calculate_institutional_flow_ratio()?;
        let (buy_volume_ratio, sell_volume_ratio) = self.calculate_volume_ratios()?;
        let trade_classification_accuracy = self.estimate_classification_accuracy()?;

        Ok(OrderFlowMetrics {
            timestamp,
            order_flow_imbalance,
            vpin,
            toxic_flow_probability,
            institutional_flow_ratio,
            buy_volume_ratio,
            sell_volume_ratio,
            trade_classification_accuracy,
        })
    }

    /// Calculate order flow imbalance
    fn calculate_order_flow_imbalance(&self) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.classified_trades.back()
            .map(|t| t.timestamp.saturating_sub(self.lookback_window_ms))
            .unwrap_or(0);

        let recent_trades: Vec<_> = self.classified_trades.iter()
            .filter(|t| t.timestamp >= cutoff_time)
            .collect();

        if recent_trades.is_empty() {
            return Ok(FixedPoint::ZERO);
        }

        let mut buy_volume = FixedPoint::ZERO;
        let mut sell_volume = FixedPoint::ZERO;

        for trade in recent_trades {
            match trade.direction {
                TradeDirection::Buy => buy_volume += trade.volume,
                TradeDirection::Sell => sell_volume += trade.volume,
                TradeDirection::Unknown => {
                    // Split unknown volume
                    let half_volume = trade.volume / FixedPoint::from_float(2.0);
                    buy_volume += half_volume;
                    sell_volume += half_volume;
                }
            }
        }

        let total_volume = buy_volume + sell_volume;
        if total_volume > FixedPoint::ZERO {
            Ok((buy_volume - sell_volume) / total_volume)
        } else {
            Ok(FixedPoint::ZERO)
        }
    }

    /// Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
    fn calculate_vpin(&self) -> MicrostructureResult<FixedPoint> {
        if self.vpin_buckets.len() < 50 {
            return Ok(FixedPoint::ZERO);
        }

        // Use last 50 buckets for VPIN calculation
        let recent_buckets: Vec<_> = self.vpin_buckets.iter()
            .rev()
            .take(50)
            .collect();

        let total_imbalance: FixedPoint = recent_buckets.iter()
            .map(|bucket| bucket.imbalance)
            .sum();

        let vpin = total_imbalance / FixedPoint::from_int(recent_buckets.len() as i64);
        Ok(vpin)
    }

    /// Calculate toxic flow probability
    fn calculate_toxic_flow_probability(&self) -> MicrostructureResult<FixedPoint> {
        let vpin = self.calculate_vpin()?;
        let order_flow_imbalance = self.calculate_order_flow_imbalance()?.abs();
        
        // Combine VPIN and order flow imbalance for toxic flow estimation
        let toxic_probability = (vpin + order_flow_imbalance) / FixedPoint::from_float(2.0);
        
        // Apply sigmoid transformation to bound between 0 and 1
        let sigmoid_input = (toxic_probability - FixedPoint::from_float(0.5)) * FixedPoint::from_float(10.0);
        let exp_neg_x = (-sigmoid_input.to_float()).exp();
        let sigmoid = 1.0 / (1.0 + exp_neg_x);
        
        Ok(FixedPoint::from_float(sigmoid))
    }

    /// Calculate institutional flow ratio
    fn calculate_institutional_flow_ratio(&self) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.classified_trades.back()
            .map(|t| t.timestamp.saturating_sub(self.lookback_window_ms))
            .unwrap_or(0);

        let recent_trades: Vec<_> = self.classified_trades.iter()
            .filter(|t| t.timestamp >= cutoff_time)
            .collect();

        if recent_trades.is_empty() {
            return Ok(FixedPoint::ZERO);
        }

        let mut institutional_volume = FixedPoint::ZERO;
        let mut total_volume = FixedPoint::ZERO;

        for trade in recent_trades {
            total_volume += trade.volume;
            
            // Classify as institutional if volume is in top 5% (block trades)
            let volume_percentile = self.get_volume_percentile(trade.volume);
            if volume_percentile >= FixedPoint::from_float(95.0) {
                institutional_volume += trade.volume;
            }
        }

        if total_volume > FixedPoint::ZERO {
            Ok(institutional_volume / total_volume)
        } else {
            Ok(FixedPoint::ZERO)
        }
    }

    /// Calculate buy and sell volume ratios
    fn calculate_volume_ratios(&self) -> MicrostructureResult<(FixedPoint, FixedPoint)> {
        let cutoff_time = self.classified_trades.back()
            .map(|t| t.timestamp.saturating_sub(self.lookback_window_ms))
            .unwrap_or(0);

        let recent_trades: Vec<_> = self.classified_trades.iter()
            .filter(|t| t.timestamp >= cutoff_time)
            .collect();

        if recent_trades.is_empty() {
            return Ok((FixedPoint::ZERO, FixedPoint::ZERO));
        }

        let mut buy_volume = FixedPoint::ZERO;
        let mut sell_volume = FixedPoint::ZERO;

        for trade in recent_trades {
            match trade.direction {
                TradeDirection::Buy => buy_volume += trade.volume,
                TradeDirection::Sell => sell_volume += trade.volume,
                TradeDirection::Unknown => {
                    let half_volume = trade.volume / FixedPoint::from_float(2.0);
                    buy_volume += half_volume;
                    sell_volume += half_volume;
                }
            }
        }

        let total_volume = buy_volume + sell_volume;
        if total_volume > FixedPoint::ZERO {
            Ok((buy_volume / total_volume, sell_volume / total_volume))
        } else {
            Ok((FixedPoint::ZERO, FixedPoint::ZERO))
        }
    }

    /// Estimate trade classification accuracy
    fn estimate_classification_accuracy(&self) -> MicrostructureResult<FixedPoint> {
        // This is a simplified estimation based on confidence scores
        let recent_classifications: Vec<_> = self.classified_trades.iter()
            .rev()
            .take(100)
            .collect();

        if recent_classifications.is_empty() {
            return Ok(FixedPoint::ZERO);
        }

        let average_confidence: FixedPoint = recent_classifications.iter()
            .map(|c| c.confidence)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_classifications.len() as i64);

        Ok(average_confidence)
    }

    /// Classify bulk volume
    pub fn classify_bulk_volume(&self, trade: &Trade) -> MicrostructureResult<BulkVolumeClassification> {
        let volume_percentile = self.get_volume_percentile(trade.volume);
        
        let size_category = if volume_percentile >= FixedPoint::from_float(95.0) {
            VolumeCategory::Block
        } else if volume_percentile >= FixedPoint::from_float(75.0) {
            VolumeCategory::Large
        } else if volume_percentile >= FixedPoint::from_float(25.0) {
            VolumeCategory::Medium
        } else {
            VolumeCategory::Small
        };

        let is_institutional = matches!(size_category, VolumeCategory::Block | VolumeCategory::Large);
        
        let clustering_score = self.calculate_clustering_score(trade)?;

        Ok(BulkVolumeClassification {
            timestamp: trade.timestamp,
            is_institutional,
            volume_percentile,
            size_category,
            clustering_score,
        })
    }

    /// Get volume percentile for a given volume
    fn get_volume_percentile(&self, volume: FixedPoint) -> FixedPoint {
        // Binary search in percentiles array
        let mut left = 0;
        let mut right = 100;
        
        while left < right {
            let mid = (left + right) / 2;
            if self.volume_statistics.volume_percentiles[mid] < volume {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        FixedPoint::from_int(left as i64)
    }

    /// Calculate clustering score for institutional detection
    fn calculate_clustering_score(&self, trade: &Trade) -> MicrostructureResult<FixedPoint> {
        let time_window = 60000; // 1 minute
        let volume_threshold = trade.volume * FixedPoint::from_float(0.5);
        
        let similar_trades = self.trade_history.iter()
            .filter(|t| {
                (t.timestamp as i64 - trade.timestamp as i64).abs() < time_window &&
                t.volume >= volume_threshold &&
                t.trade_id != trade.trade_id
            })
            .count();

        // Normalize clustering score
        let max_expected_trades = 10;
        let clustering_score = FixedPoint::from_int(similar_trades as i64) / 
                              FixedPoint::from_int(max_expected_trades);
        
        Ok(clustering_score.min(FixedPoint::from_float(1.0)))
    }

    /// Update volume statistics for percentile calculations
    fn update_volume_statistics(&mut self) -> MicrostructureResult<()> {
        if self.trade_history.is_empty() {
            return Ok(());
        }

        let mut volumes: Vec<f64> = self.trade_history.iter()
            .map(|t| t.volume.to_float())
            .collect();
        
        volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate percentiles
        for i in 0..=100 {
            let index = (i * (volumes.len() - 1)) / 100;
            self.volume_statistics.volume_percentiles[i] = FixedPoint::from_float(volumes[index]);
        }

        // Calculate average and standard deviation
        let sum: f64 = volumes.iter().sum();
        self.volume_statistics.average_volume = FixedPoint::from_float(sum / volumes.len() as f64);
        
        let variance: f64 = volumes.iter()
            .map(|v| (v - self.volume_statistics.average_volume.to_float()).powi(2))
            .sum::<f64>() / volumes.len() as f64;
        
        self.volume_statistics.volume_std = FixedPoint::from_float(variance.sqrt());
        self.volume_statistics.last_update = self.trade_history.back().unwrap().timestamp;

        Ok(())
    }

    /// Find market data before a specific trade timestamp
    fn find_market_data_before_trade(&self, trade_timestamp: u64) -> Option<&MarketData> {
        self.market_data_history.iter()
            .rev()
            .find(|md| md.timestamp <= trade_timestamp)
    }

    /// Get order flow summary
    pub fn get_order_flow_summary(&self) -> MicrostructureResult<OrderFlowSummary> {
        let metrics = self.calculate_metrics()?;
        
        Ok(OrderFlowSummary {
            timestamp: metrics.timestamp,
            imbalance_pct: (metrics.order_flow_imbalance * FixedPoint::from_float(100.0)).to_float() as i8,
            vpin_pct: (metrics.vpin * FixedPoint::from_float(100.0)).to_float() as u8,
            toxic_flow_pct: (metrics.toxic_flow_probability * FixedPoint::from_float(100.0)).to_float() as u8,
            institutional_pct: (metrics.institutional_flow_ratio * FixedPoint::from_float(100.0)).to_float() as u8,
            classification_accuracy_pct: (metrics.trade_classification_accuracy * FixedPoint::from_float(100.0)).to_float() as u8,
        })
    }
}

/// Simplified order flow summary for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowSummary {
    pub timestamp: u64,
    pub imbalance_pct: i8,           // Order flow imbalance -100 to +100
    pub vpin_pct: u8,                // VPIN 0-100
    pub toxic_flow_pct: u8,          // Toxic flow probability 0-100
    pub institutional_pct: u8,       // Institutional flow ratio 0-100
    pub classification_accuracy_pct: u8, // Classification accuracy 0-100
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

    fn create_test_trade(timestamp: u64, price: f64, volume: f64, trade_id: u64) -> Trade {
        Trade {
            timestamp,
            price: FixedPoint::from_float(price),
            volume: FixedPoint::from_float(volume),
            direction: TradeDirection::Unknown,
            trade_id,
        }
    }

    #[test]
    fn test_lee_ready_classification() {
        let mut analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(10000.0), 60000);
        
        // Add market data
        let market_data = create_test_market_data(1000, 99.95, 100.05);
        analyzer.update_market_data(market_data);
        
        // Test buy trade (above mid-price)
        let buy_trade = create_test_trade(1001, 100.03, 500.0, 1);
        let classification = analyzer.update_trade(buy_trade).unwrap();
        
        assert_eq!(classification.direction, TradeDirection::Buy);
        assert!(classification.confidence > FixedPoint::ZERO);
        
        // Test sell trade (below mid-price)
        let sell_trade = create_test_trade(1002, 99.97, 300.0, 2);
        let classification = analyzer.update_trade(sell_trade).unwrap();
        
        assert_eq!(classification.direction, TradeDirection::Sell);
        assert!(classification.confidence > FixedPoint::ZERO);
    }

    #[test]
    fn test_order_flow_imbalance() {
        let mut analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(10000.0), 60000);
        
        // Add market data
        let market_data = create_test_market_data(1000, 99.95, 100.05);
        analyzer.update_market_data(market_data);
        
        // Add more buy trades than sell trades
        for i in 0..5 {
            let buy_trade = create_test_trade(1001 + i, 100.03, 1000.0, i + 1);
            analyzer.update_trade(buy_trade).unwrap();
        }
        
        for i in 0..2 {
            let sell_trade = create_test_trade(1006 + i, 99.97, 1000.0, i + 6);
            analyzer.update_trade(sell_trade).unwrap();
        }
        
        let metrics = analyzer.calculate_metrics().unwrap();
        
        // Should have positive imbalance (more buy volume)
        assert!(metrics.order_flow_imbalance > FixedPoint::ZERO);
    }

    #[test]
    fn test_bulk_volume_classification() {
        let mut analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(10000.0), 60000);
        
        // Add some trades to build volume statistics
        for i in 0..100 {
            let volume = 100.0 + (i as f64 * 10.0); // Increasing volumes
            let trade = create_test_trade(1000 + i, 100.0, volume, i + 1);
            analyzer.trade_history.push_back(trade);
        }
        
        // Update volume statistics
        analyzer.update_volume_statistics().unwrap();
        
        // Test large volume trade
        let large_trade = create_test_trade(2000, 100.0, 5000.0, 101);
        let classification = analyzer.classify_bulk_volume(&large_trade).unwrap();
        
        assert!(classification.is_institutional);
        assert!(matches!(classification.size_category, VolumeCategory::Block));
    }

    #[test]
    fn test_vpin_calculation() {
        let mut analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(1000.0), 60000);
        
        // Add market data
        let market_data = create_test_market_data(1000, 99.95, 100.05);
        analyzer.update_market_data(market_data);
        
        // Add enough trades to create multiple VPIN buckets
        for i in 0..100 {
            let price = if i % 2 == 0 { 100.03 } else { 99.97 }; // Alternate buy/sell
            let trade = create_test_trade(1001 + i, price, 200.0, i + 1);
            analyzer.update_trade(trade).unwrap();
        }
        
        let metrics = analyzer.calculate_metrics().unwrap();
        
        // VPIN should be calculated
        assert!(metrics.vpin >= FixedPoint::ZERO);
        assert!(metrics.vpin <= FixedPoint::from_float(1.0));
    }

    #[test]
    fn test_toxic_flow_probability() {
        let mut analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(1000.0), 60000);
        
        // Add market data
        let market_data = create_test_market_data(1000, 99.95, 100.05);
        analyzer.update_market_data(market_data);
        
        // Add imbalanced trades (more buys)
        for i in 0..20 {
            let trade = create_test_trade(1001 + i, 100.03, 500.0, i + 1);
            analyzer.update_trade(trade).unwrap();
        }
        
        let metrics = analyzer.calculate_metrics().unwrap();
        
        // Should detect some toxic flow probability
        assert!(metrics.toxic_flow_probability >= FixedPoint::ZERO);
        assert!(metrics.toxic_flow_probability <= FixedPoint::from_float(1.0));
    }
}