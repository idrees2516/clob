//! Central Limit Order Book (CLOB) engine module
//!
//! This module provides the core data structures and logic for a production-grade CLOB engine
//! with price-time priority, deterministic execution, and circuit compatibility.

pub mod types;
pub mod matching_engine;
pub mod compressed_state;
pub mod management;
pub mod market_data;
pub mod venue_adapter;
pub mod venue_implementations;
pub mod cross_venue_manager;
pub mod fx_support;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod market_data_tests;

#[cfg(test)]
mod integration_test;

// Re-export core types
pub use types::{
    Order, OrderId, Symbol, Side, PriceLevel, Trade, CentralLimitOrderBook, 
    OrderResult, OrderStatus, OrderType, TimeInForce, MarketDepth, 
    OrderBookStatistics, OrderBookError, PRICE_SCALE, VOLUME_SCALE, MAX_ORDERS_PER_LEVEL
};

// Type alias for backward compatibility
pub type OrderBook = CentralLimitOrderBook;
pub use matching_engine::{MatchingEngine, MatchingError};
pub use compressed_state::{
    CompressedOrderBook, CompressedPriceLevel, CompressedPriceTree,
    StateTransition, OrderTransition, StateBatch, TransitionType,
    StateError,
};
pub use market_data::{
    MarketDataGenerator, BestBidAskTracker, TradeTickGenerator, TradeTick,
    MarketStatistics, OHLCV, DailyStatistics, VolumeProfile
};
pub use management::{
    OrderManager, OrderManagementError, OrderState, CancellationReason,
    OrderModification, OrderEvent, OrderEventType, RiskLimits, Position,
    PerformanceMetrics, OrderSubmissionResult, OrderSubmissionStatus,
    OrderCancellationResult, OrderModificationResult
};
pub use venue_adapter::{
    VenueAdapter, VenueId, VenueOrder, VenueTrade, VenueCapabilities, VenueOrderBook,
    VenueFees, RateLimits, VenueOrderStatus, VenueError, VenueResult,
    ArbitrageDetector, SmartOrderRouter, VenueRiskManager, ArbitrageOpportunity,
    RoutingDecision, VenueRiskLimits, VenueExposure
};
pub use venue_implementations::{
    MockVenueAdapter, RestApiVenueAdapter, VenueAdapterFactory
};
pub use cross_venue_manager::{
    CrossVenueManager, CrossVenueConfig, ArbitragePosition, ArbitrageStatus,
    CrossVenuePerformance, VenuePerformance
};
pub use fx_support::{
    Currency, CurrencyPair, ExchangeRate, MultiCurrencyPosition, FxRateProvider,
    MockFxRateProvider, HedgingStrategy, FxHedgingManager, HedgeOrder, HedgeOrderStatus,
    CurrencyRiskManager, CurrencyRiskLimits, FxError, FxResult
};

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Comprehensive error types for CLOB engine operations
#[derive(Error, Debug)]
pub enum CLOBError {
    #[error("Order book error: {0}")]
    OrderBookError(#[from] OrderBookError),
    #[error("Matching engine error: {0}")]
    MatchingError(#[from] MatchingError),
    #[error("Order management error: {0}")]
    OrderManagementError(#[from] OrderManagementError),
    #[error("State error: {0}")]
    StateError(#[from] StateError),
    #[error("Market data error: {0}")]
    MarketDataError(String),
    #[error("Integration error: {0}")]
    IntegrationError(String),
}

/// Result type for CLOB operations
pub type CLOBResult<T> = Result<T, CLOBError>;

/// Integrated Central Limit Order Book Engine
/// 
/// This is the main entry point for the CLOB system, integrating all components:
/// - Order book with price-time priority matching
/// - Deterministic matching engine
/// - Order management with lifecycle tracking
/// - Real-time market data generation
/// - Compressed state for zkVM compatibility
pub struct CLOBEngine {
    /// Core order manager with integrated order book
    pub order_manager: OrderManager,
    
    /// Deterministic matching engine
    pub matching_engine: MatchingEngine,
    
    /// Market data generator
    pub market_data_generator: MarketDataGenerator,
    
    /// Compressed state representation
    pub compressed_state: CompressedOrderBook,
    
    /// Symbol this engine handles
    pub symbol: Symbol,
    
    /// Engine creation timestamp
    pub created_at: u64,
    
    /// Last update timestamp
    pub last_update: u64,
    
    /// Total operations processed
    pub total_operations: u64,
}

impl CLOBEngine {
    /// Create a new CLOB engine for a specific symbol
    pub fn new(symbol: Symbol, risk_limits: Option<RiskLimits>) -> CLOBResult<Self> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let order_manager = OrderManager::new(symbol.clone(), risk_limits);
        let matching_engine = MatchingEngine::new();
        let market_data_generator = MarketDataGenerator::new(symbol.clone());
        
        // Create compressed state representation
        let symbol_id = Self::symbol_to_id(&symbol);
        let compressed_state = CompressedOrderBook::new(symbol_id);
        
        Ok(Self {
            order_manager,
            matching_engine,
            market_data_generator,
            compressed_state,
            symbol,
            created_at: timestamp,
            last_update: timestamp,
            total_operations: 0,
        })
    }
    
    /// Submit a new order to the CLOB engine
    pub fn submit_order(&mut self, order: Order) -> CLOBResult<OrderSubmissionResult> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Update matching engine timestamp for deterministic execution
        self.matching_engine.set_current_time(timestamp);
        
        // Submit order through order manager
        let result = self.order_manager.submit_order(order)?;
        
        // Update market data if there were trades
        if !result.trades.is_empty() {
            for trade in &result.trades {
                let _tick = self.market_data_generator.process_trade(trade)?;
            }
        }
        
        // Update market data from order book state
        self.market_data_generator.update_from_order_book(&self.order_manager.order_book)?;
        
        // Update compressed state
        self.update_compressed_state()?;
        
        self.total_operations += 1;
        self.last_update = timestamp;
        
        Ok(result)
    }
    
    /// Cancel an order
    pub fn cancel_order(&mut self, order_id: OrderId, reason: CancellationReason) -> CLOBResult<OrderCancellationResult> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Cancel through order manager
        let result = self.order_manager.cancel_order(order_id, reason)?;
        
        // Update market data from order book state
        self.market_data_generator.update_from_order_book(&self.order_manager.order_book)?;
        
        // Update compressed state
        self.update_compressed_state()?;
        
        self.total_operations += 1;
        self.last_update = timestamp;
        
        Ok(result)
    }
    
    /// Modify an order
    pub fn modify_order(&mut self, modification: OrderModification) -> CLOBResult<OrderModificationResult> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Modify through order manager
        let result = self.order_manager.modify_order(modification)?;
        
        // Update market data if there were trades in the new order
        if !result.submission_result.trades.is_empty() {
            for trade in &result.submission_result.trades {
                let _tick = self.market_data_generator.process_trade(trade)?;
            }
        }
        
        // Update market data from order book state
        self.market_data_generator.update_from_order_book(&self.order_manager.order_book)?;
        
        // Update compressed state
        self.update_compressed_state()?;
        
        self.total_operations += 1;
        self.last_update = timestamp;
        
        Ok(result)
    }
    
    /// Get current market depth
    pub fn get_market_depth(&self, levels: usize) -> MarketDepth {
        self.market_data_generator.get_market_depth(&self.order_manager.order_book, levels)
    }
    
    /// Get best bid and ask prices
    pub fn get_best_bid_ask(&self) -> (Option<u64>, Option<u64>) {
        self.market_data_generator.get_best_bid_ask()
    }
    
    /// Get current spread
    pub fn get_spread(&self) -> Option<u64> {
        self.market_data_generator.get_spread()
    }
    
    /// Get current mid price
    pub fn get_mid_price(&self) -> Option<u64> {
        self.market_data_generator.get_mid_price()
    }
    
    /// Get current VWAP
    pub fn get_current_vwap(&self) -> Option<u64> {
        self.market_data_generator.get_current_vwap()
    }
    
    /// Get current OHLCV data
    pub fn get_current_ohlcv(&self) -> Option<&OHLCV> {
        self.market_data_generator.get_current_ohlcv()
    }
    
    /// Get volume profile
    pub fn get_volume_profile(&self) -> &VolumeProfile {
        self.market_data_generator.get_volume_profile()
    }
    
    /// Get daily statistics
    pub fn get_daily_stats(&self) -> &DailyStatistics {
        self.market_data_generator.get_daily_stats()
    }
    
    /// Get order state
    pub fn get_order_state(&self, order_id: OrderId) -> Option<&OrderState> {
        self.order_manager.get_order_state(order_id)
    }
    
    /// Get order events for a specific order
    pub fn get_order_events(&self, order_id: OrderId) -> Vec<&OrderEvent> {
        self.order_manager.get_order_events(order_id)
    }
    
    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<(OrderId, &Order)> {
        self.order_manager.get_active_orders()
    }
    
    /// Get current position
    pub fn get_position(&self) -> &Position {
        &self.order_manager.position
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.order_manager.performance_metrics
    }
    
    /// Get compressed state for zkVM execution
    pub fn get_compressed_state(&self) -> &CompressedOrderBook {
        &self.compressed_state
    }
    
    /// Cancel all orders (emergency function)
    pub fn cancel_all_orders(&mut self, reason: CancellationReason) -> CLOBResult<Vec<OrderCancellationResult>> {
        let results = self.order_manager.cancel_all_orders(reason)?;
        
        // Update market data and compressed state
        self.market_data_generator.update_from_order_book(&self.order_manager.order_book)?;
        self.update_compressed_state()?;
        
        self.total_operations += results.len() as u64;
        self.last_update = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Ok(results)
    }
    
    /// Update compressed state representation
    fn update_compressed_state(&mut self) -> CLOBResult<()> {
        let symbol_id = Self::symbol_to_id(&self.symbol);
        self.compressed_state = CompressedOrderBook::from_order_book(
            &self.order_manager.order_book,
            symbol_id
        )?;
        Ok(())
    }
    
    /// Convert symbol to numeric ID for compression
    fn symbol_to_id(symbol: &Symbol) -> u32 {
        // Simple hash-based ID generation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        symbol.as_str().hash(&mut hasher);
        (hasher.finish() as u32) % 1_000_000 // Limit to reasonable range
    }
    
    /// Get engine statistics
    pub fn get_engine_stats(&self) -> EngineStatistics {
        EngineStatistics {
            symbol: self.symbol.clone(),
            total_operations: self.total_operations,
            active_orders: self.order_manager.active_order_count,
            total_orders_processed: self.order_manager.total_orders_processed.load(std::sync::atomic::Ordering::SeqCst),
            best_bid: self.get_best_bid_ask().0,
            best_ask: self.get_best_bid_ask().1,
            spread: self.get_spread(),
            mid_price: self.get_mid_price(),
            last_trade_price: self.order_manager.order_book.last_trade_price,
            last_trade_time: self.order_manager.order_book.last_trade_time,
            created_at: self.created_at,
            last_update: self.last_update,
            compressed_state_size: self.compressed_state.compressed_size(),
        }
    }
    
    /// Verify engine state integrity
    pub fn verify_integrity(&self) -> CLOBResult<bool> {
        // Verify compressed state integrity
        let state_valid = self.compressed_state.verify_state()?;
        
        if !state_valid {
            return Ok(false);
        }
        
        // Verify order book consistency
        let (best_bid, best_ask) = self.get_best_bid_ask();
        let book_best_bid = self.order_manager.order_book.get_best_bid();
        let book_best_ask = self.order_manager.order_book.get_best_ask();
        
        if best_bid != book_best_bid || best_ask != book_best_ask {
            return Ok(false);
        }
        
        Ok(true)
    }
}

/// Engine statistics for monitoring and debugging
#[derive(Debug, Clone)]
pub struct EngineStatistics {
    pub symbol: Symbol,
    pub total_operations: u64,
    pub active_orders: u32,
    pub total_orders_processed: u64,
    pub best_bid: Option<u64>,
    pub best_ask: Option<u64>,
    pub spread: Option<u64>,
    pub mid_price: Option<u64>,
    pub last_trade_price: Option<u64>,
    pub last_trade_time: u64,
    pub created_at: u64,
    pub last_update: u64,
    pub compressed_state_size: usize,
} 