//! Market data generation for the CLOB
//!
//! Implements real-time market depth calculation, best bid/ask price tracking,
//! trade tick generation with volume-weighted prices, and market statistics
//! including OHLCV and volume profiles.

use super::types::{
    CentralLimitOrderBook, Trade, Symbol, Side, MarketDepth, OrderBookError
};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Real-time market data generator for CLOB
#[derive(Debug, Clone)]
pub struct MarketDataGenerator {
    /// Symbol this generator is tracking
    pub symbol: Symbol,
    
    /// Current best bid/ask tracker
    pub best_bid_ask: BestBidAskTracker,
    
    /// Trade tick generator
    pub trade_tick_generator: TradeTickGenerator,
    
    /// Market statistics calculator
    pub market_stats: MarketStatistics,
    
    /// Volume profile tracker
    pub volume_profile: VolumeProfile,
}

/// Best bid/ask price tracker with real-time updates
#[derive(Debug, Clone)]
pub struct BestBidAskTracker {
    /// Current best bid price
    pub best_bid: Option<u64>,
    
    /// Current best ask price
    pub best_ask: Option<u64>,
    
    /// Current spread
    pub spread: Option<u64>,
    
    /// Mid price
    pub mid_price: Option<u64>,
    
    /// Last update timestamp
    pub last_update: u64,
    
    /// Sequence number for updates
    pub sequence: u64,
}

/// Trade tick with volume-weighted price information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeTick {
    /// Trade ID
    pub trade_id: u64,
    
    /// Symbol
    pub symbol: Symbol,
    
    /// Execution price
    pub price: u64,
    
    /// Trade size
    pub size: u64,
    
    /// Trade timestamp
    pub timestamp: u64,
    
    /// Trade side (from taker perspective)
    pub side: Side,
    
    /// Volume-weighted average price over recent window
    pub vwap: u64,
    
    /// Sequence number
    pub sequence: u64,
    
    /// Is this trade from a maker order
    pub is_maker: bool,
}

/// Trade tick generator with VWAP calculation
#[derive(Debug, Clone)]
pub struct TradeTickGenerator {
    /// Recent trades for VWAP calculation (sliding window)
    pub recent_trades: VecDeque<Trade>,
    
    /// VWAP calculation window in nanoseconds (default: 1 minute)
    pub vwap_window_ns: u64,
    
    /// Current VWAP
    pub current_vwap: Option<u64>,
    
    /// Total volume in VWAP window
    pub vwap_volume: u64,
    
    /// Total notional in VWAP window
    pub vwap_notional: u64,
    
    /// Next tick sequence number
    pub next_sequence: u64,
}

/// OHLCV (Open, High, Low, Close, Volume) data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OHLCV {
    /// Opening price for the period
    pub open: u64,
    
    /// Highest price for the period
    pub high: u64,
    
    /// Lowest price for the period
    pub low: u64,
    
    /// Closing price for the period
    pub close: u64,
    
    /// Total volume for the period
    pub volume: u64,
    
    /// Period start timestamp
    pub timestamp: u64,
    
    /// Period duration in nanoseconds
    pub period_ns: u64,
    
    /// Number of trades in this period
    pub trade_count: u32,
    
    /// Volume-weighted average price for the period
    pub vwap: u64,
}

/// Market statistics calculator
#[derive(Debug, Clone)]
pub struct MarketStatistics {
    /// Current OHLCV data
    pub current_ohlcv: Option<OHLCV>,
    
    /// OHLCV period in nanoseconds (default: 1 minute)
    pub ohlcv_period_ns: u64,
    
    /// Historical OHLCV data (limited buffer)
    pub ohlcv_history: VecDeque<OHLCV>,
    
    /// Maximum history to keep
    pub max_history: usize,
    
    /// Total trades processed
    pub total_trades: u64,
    
    /// Total volume processed
    pub total_volume: u64,
    
    /// Last trade price
    pub last_price: Option<u64>,
    
    /// 24h statistics
    pub daily_stats: DailyStatistics,
}

/// 24-hour rolling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStatistics {
    /// 24h high
    pub high_24h: Option<u64>,
    
    /// 24h low
    pub low_24h: Option<u64>,
    
    /// 24h volume
    pub volume_24h: u64,
    
    /// 24h trade count
    pub trade_count_24h: u32,
    
    /// Price change from 24h ago
    pub price_change_24h: Option<i64>,
    
    /// Percentage change from 24h ago
    pub price_change_pct_24h: Option<f64>,
    
    /// Last update timestamp
    pub last_update: u64,
}

/// Volume profile tracking price levels and their volumes
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    /// Volume at each price level
    pub price_volumes: HashMap<u64, u64>,
    
    /// Price level with highest volume (Point of Control)
    pub poc: Option<u64>,
    
    /// Value Area High (70% of volume above this price)
    pub vah: Option<u64>,
    
    /// Value Area Low (70% of volume below this price)
    pub val: Option<u64>,
    
    /// Total volume tracked
    pub total_volume: u64,
    
    /// Time period for this profile
    pub period_start: u64,
    pub period_end: u64,
}

impl MarketDataGenerator {
    /// Create a new market data generator for a symbol
    pub fn new(symbol: Symbol) -> Self {
        Self {
            symbol: symbol.clone(),
            best_bid_ask: BestBidAskTracker::new(),
            trade_tick_generator: TradeTickGenerator::new(),
            market_stats: MarketStatistics::new(),
            volume_profile: VolumeProfile::new(),
        }
    }
    
    /// Update market data from order book state
    pub fn update_from_order_book(&mut self, order_book: &CentralLimitOrderBook) -> Result<(), OrderBookError> {
        // Update best bid/ask
        self.best_bid_ask.update_from_order_book(order_book)?;
        
        Ok(())
    }
    
    /// Process a new trade and generate trade tick
    pub fn process_trade(&mut self, trade: &Trade) -> Result<TradeTick, OrderBookError> {
        // Generate trade tick with VWAP
        let tick = self.trade_tick_generator.generate_tick(trade, &self.symbol)?;
        
        // Update market statistics
        self.market_stats.process_trade(trade)?;
        
        // Update volume profile
        self.volume_profile.add_trade(trade);
        
        Ok(tick)
    }
    
    /// Get current market depth
    pub fn get_market_depth(&self, order_book: &CentralLimitOrderBook, levels: usize) -> MarketDepth {
        order_book.get_market_depth(levels)
    }
    
    /// Get current best bid and ask
    pub fn get_best_bid_ask(&self) -> (Option<u64>, Option<u64>) {
        (self.best_bid_ask.best_bid, self.best_bid_ask.best_ask)
    }
    
    /// Get current spread
    pub fn get_spread(&self) -> Option<u64> {
        self.best_bid_ask.spread
    }
    
    /// Get current mid price
    pub fn get_mid_price(&self) -> Option<u64> {
        self.best_bid_ask.mid_price
    }
    
    /// Get current VWAP
    pub fn get_current_vwap(&self) -> Option<u64> {
        self.trade_tick_generator.current_vwap
    }
    
    /// Get current OHLCV
    pub fn get_current_ohlcv(&self) -> Option<&OHLCV> {
        self.market_stats.current_ohlcv.as_ref()
    }
    
    /// Get volume profile
    pub fn get_volume_profile(&self) -> &VolumeProfile {
        &self.volume_profile
    }
    
    /// Get daily statistics
    pub fn get_daily_stats(&self) -> &DailyStatistics {
        &self.market_stats.daily_stats
    }
}

impl BestBidAskTracker {
    /// Create a new best bid/ask tracker
    pub fn new() -> Self {
        Self {
            best_bid: None,
            best_ask: None,
            spread: None,
            mid_price: None,
            last_update: 0,
            sequence: 0,
        }
    }
    
    /// Update from order book state
    pub fn update_from_order_book(&mut self, order_book: &CentralLimitOrderBook) -> Result<(), OrderBookError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Get current best prices
        let new_best_bid = order_book.get_best_bid();
        let new_best_ask = order_book.get_best_ask();
        
        // Check if anything changed
        if new_best_bid != self.best_bid || new_best_ask != self.best_ask {
            self.best_bid = new_best_bid;
            self.best_ask = new_best_ask;
            
            // Calculate spread and mid price
            self.spread = match (new_best_ask, new_best_bid) {
                (Some(ask), Some(bid)) => Some(ask.saturating_sub(bid)),
                _ => None,
            };
            
            self.mid_price = match (new_best_ask, new_best_bid) {
                (Some(ask), Some(bid)) => Some((ask + bid) / 2),
                _ => None,
            };
            
            self.last_update = timestamp;
            self.sequence += 1;
        }
        
        Ok(())
    }
}

impl TradeTickGenerator {
    /// Create a new trade tick generator
    pub fn new() -> Self {
        Self {
            recent_trades: VecDeque::new(),
            vwap_window_ns: 60_000_000_000, // 1 minute in nanoseconds
            current_vwap: None,
            vwap_volume: 0,
            vwap_notional: 0,
            next_sequence: 1,
        }
    }
    
    /// Generate a trade tick with VWAP calculation
    pub fn generate_tick(&mut self, trade: &Trade, symbol: &Symbol) -> Result<TradeTick, OrderBookError> {
        // Add trade to recent trades
        self.recent_trades.push_back(trade.clone());
        
        // Clean old trades outside VWAP window
        self.clean_old_trades(trade.timestamp);
        
        // Recalculate VWAP
        self.calculate_vwap();
        
        // Determine trade side from taker perspective
        let side = if trade.is_buyer_maker {
            Side::Sell // Buyer was maker, so seller was taker
        } else {
            Side::Buy // Seller was maker, so buyer was taker
        };
        
        let tick = TradeTick {
            trade_id: trade.id,
            symbol: symbol.clone(),
            price: trade.price,
            size: trade.size,
            timestamp: trade.timestamp,
            side,
            vwap: self.current_vwap.unwrap_or(trade.price),
            sequence: self.next_sequence,
            is_maker: trade.is_buyer_maker,
        };
        
        self.next_sequence += 1;
        
        Ok(tick)
    }
    
    /// Clean trades outside the VWAP window
    fn clean_old_trades(&mut self, current_timestamp: u64) {
        let cutoff_time = current_timestamp.saturating_sub(self.vwap_window_ns);
        
        while let Some(front_trade) = self.recent_trades.front() {
            if front_trade.timestamp < cutoff_time {
                self.recent_trades.pop_front();
            } else {
                break;
            }
        }
    }
    
    /// Calculate volume-weighted average price
    fn calculate_vwap(&mut self) {
        if self.recent_trades.is_empty() {
            self.current_vwap = None;
            self.vwap_volume = 0;
            self.vwap_notional = 0;
            return;
        }
        
        let mut total_notional = 0u64;
        let mut total_volume = 0u64;
        
        for trade in &self.recent_trades {
            let notional = trade.price.saturating_mul(trade.size);
            total_notional = total_notional.saturating_add(notional);
            total_volume = total_volume.saturating_add(trade.size);
        }
        
        self.vwap_volume = total_volume;
        self.vwap_notional = total_notional;
        
        if total_volume > 0 {
            self.current_vwap = Some(total_notional / total_volume);
        } else {
            self.current_vwap = None;
        }
    }
}

impl MarketStatistics {
    /// Create a new market statistics calculator
    pub fn new() -> Self {
        Self {
            current_ohlcv: None,
            ohlcv_period_ns: 60_000_000_000, // 1 minute
            ohlcv_history: VecDeque::new(),
            max_history: 1440, // 24 hours of 1-minute bars
            total_trades: 0,
            total_volume: 0,
            last_price: None,
            daily_stats: DailyStatistics::new(),
        }
    }
    
    /// Process a new trade and update statistics
    pub fn process_trade(&mut self, trade: &Trade) -> Result<(), OrderBookError> {
        // Update totals
        self.total_trades += 1;
        self.total_volume = self.total_volume.saturating_add(trade.size);
        self.last_price = Some(trade.price);
        
        // Update or create OHLCV bar
        self.update_ohlcv(trade)?;
        
        // Update daily statistics
        self.daily_stats.update_with_trade(trade);
        
        Ok(())
    }
    
    /// Update OHLCV data with new trade
    fn update_ohlcv(&mut self, trade: &Trade) -> Result<(), OrderBookError> {
        let period_start = (trade.timestamp / self.ohlcv_period_ns) * self.ohlcv_period_ns;
        
        match &mut self.current_ohlcv {
            Some(ohlcv) if ohlcv.timestamp == period_start => {
                // Update existing OHLCV bar
                ohlcv.high = ohlcv.high.max(trade.price);
                ohlcv.low = ohlcv.low.min(trade.price);
                ohlcv.close = trade.price;
                ohlcv.volume = ohlcv.volume.saturating_add(trade.size);
                ohlcv.trade_count += 1;
                
                // Recalculate VWAP
                let total_notional = ohlcv.vwap.saturating_mul(ohlcv.volume.saturating_sub(trade.size))
                    .saturating_add(trade.price.saturating_mul(trade.size));
                ohlcv.vwap = if ohlcv.volume > 0 {
                    total_notional / ohlcv.volume
                } else {
                    trade.price
                };
            }
            _ => {
                // Finalize previous OHLCV if exists
                if let Some(completed_ohlcv) = self.current_ohlcv.take() {
                    self.ohlcv_history.push_back(completed_ohlcv);
                    
                    // Limit history size
                    if self.ohlcv_history.len() > self.max_history {
                        self.ohlcv_history.pop_front();
                    }
                }
                
                // Create new OHLCV bar
                self.current_ohlcv = Some(OHLCV {
                    open: trade.price,
                    high: trade.price,
                    low: trade.price,
                    close: trade.price,
                    volume: trade.size,
                    timestamp: period_start,
                    period_ns: self.ohlcv_period_ns,
                    trade_count: 1,
                    vwap: trade.price,
                });
            }
        }
        
        Ok(())
    }
}

impl DailyStatistics {
    /// Create new daily statistics
    pub fn new() -> Self {
        Self {
            high_24h: None,
            low_24h: None,
            volume_24h: 0,
            trade_count_24h: 0,
            price_change_24h: None,
            price_change_pct_24h: None,
            last_update: 0,
        }
    }
    
    /// Update with new trade
    pub fn update_with_trade(&mut self, trade: &Trade) {
        // Update 24h high/low
        self.high_24h = Some(self.high_24h.map_or(trade.price, |h| h.max(trade.price)));
        self.low_24h = Some(self.low_24h.map_or(trade.price, |l| l.min(trade.price)));
        
        // Update 24h volume and trade count
        self.volume_24h = self.volume_24h.saturating_add(trade.size);
        self.trade_count_24h += 1;
        
        self.last_update = trade.timestamp;
        
        // Note: Price change calculations would require historical price data
        // This is a simplified implementation
    }
}

impl VolumeProfile {
    /// Create a new volume profile
    pub fn new() -> Self {
        Self {
            price_volumes: HashMap::new(),
            poc: None,
            vah: None,
            val: None,
            total_volume: 0,
            period_start: 0,
            period_end: 0,
        }
    }
    
    /// Add a trade to the volume profile
    pub fn add_trade(&mut self, trade: &Trade) {
        // Add volume to price level
        *self.price_volumes.entry(trade.price).or_insert(0) += trade.size;
        self.total_volume = self.total_volume.saturating_add(trade.size);
        
        // Update time period
        if self.period_start == 0 {
            self.period_start = trade.timestamp;
        }
        self.period_end = trade.timestamp;
        
        // Recalculate key levels
        self.calculate_key_levels();
    }
    
    /// Calculate Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL)
    fn calculate_key_levels(&mut self) {
        if self.price_volumes.is_empty() {
            return;
        }
        
        // Find Point of Control (price with highest volume)
        let mut max_volume = 0u64;
        let mut poc_price = 0u64;
        
        for (&price, &volume) in &self.price_volumes {
            if volume > max_volume {
                max_volume = volume;
                poc_price = price;
            }
        }
        
        self.poc = Some(poc_price);
        
        // Calculate Value Area (70% of volume)
        let target_volume = (self.total_volume * 70) / 100;
        let mut sorted_prices: Vec<_> = self.price_volumes.iter().collect();
        sorted_prices.sort_by_key(|(_, &volume)| std::cmp::Reverse(volume));
        
        let mut accumulated_volume = 0u64;
        let mut value_area_prices = Vec::new();
        
        for (&price, &volume) in sorted_prices {
            accumulated_volume += volume;
            value_area_prices.push(price);
            
            if accumulated_volume >= target_volume {
                break;
            }
        }
        
        if !value_area_prices.is_empty() {
            value_area_prices.sort_unstable();
            self.val = value_area_prices.first().copied();
            self.vah = value_area_prices.last().copied();
        }
    }
}

/// Utility functions for market data calculations
pub mod utils {
    use super::*;
    
    /// Calculate volume-weighted average price for a set of trades
    pub fn calculate_vwap(trades: &[Trade]) -> Option<u64> {
        if trades.is_empty() {
            return None;
        }
        
        let mut total_notional = 0u64;
        let mut total_volume = 0u64;
        
        for trade in trades {
            total_notional = total_notional.saturating_add(trade.price.saturating_mul(trade.size));
            total_volume = total_volume.saturating_add(trade.size);
        }
        
        if total_volume > 0 {
            Some(total_notional / total_volume)
        } else {
            None
        }
    }
    
    /// Calculate time-weighted average price for a set of trades
    pub fn calculate_twap(trades: &[Trade]) -> Option<u64> {
        if trades.is_empty() {
            return None;
        }
        
        let total_price: u64 = trades.iter().map(|t| t.price).sum();
        Some(total_price / trades.len() as u64)
    }
    
    /// Calculate price volatility (standard deviation of returns)
    pub fn calculate_volatility(prices: &[u64]) -> Option<f64> {
        if prices.len() < 2 {
            return None;
        }
        
        // Calculate returns
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            if prices[i-1] > 0 {
                let return_val = (prices[i] as f64 / prices[i-1] as f64).ln();
                returns.push(return_val);
            }
        }
        
        if returns.is_empty() {
            return None;
        }
        
        // Calculate mean return
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        
        // Calculate variance
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Some(variance.sqrt())
    }
} 