//! Multi-Asset Order Book Management
//!
//! This module provides concurrent order book management for multiple symbols,
//! cross-asset position tracking, correlation-aware risk management, and
//! multi-asset portfolio optimization capabilities.

use std::collections::{HashMap, BTreeMap};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::broadcast;
use nalgebra as na;

use super::types::{Symbol, OrderId, Order, Trade, OrderBook, OrderBookError, Side};
use super::matching_engine::{MatchingEngine, MatchingError};

/// Errors specific to multi-asset operations
#[derive(Error, Debug, Clone)]
pub enum MultiAssetError {
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("Correlation calculation failed: {0}")]
    CorrelationError(String),
    #[error("Portfolio optimization failed: {0}")]
    OptimizationError(String),
    #[error("Position tracking error: {0}")]
    PositionError(String),
    #[error("Risk limit exceeded: {symbol}, limit: {limit}, current: {current}")]
    RiskLimitExceeded { symbol: String, limit: f64, current: f64 },
    #[error("Order book error: {0}")]
    OrderBookError(#[from] OrderBookError),
    #[error("Matching engine error: {0}")]
    MatchingEngineError(#[from] MatchingError),
}

/// Multi-symbol order book manager
#[derive(Debug)]
pub struct MultiSymbolOrderBookManager {
    /// Map of symbol to order book
    order_books: Arc<RwLock<HashMap<Symbol, Arc<Mutex<OrderBook>>>>>,
    
    /// Map of symbol to matching engine
    matching_engines: Arc<RwLock<HashMap<Symbol, Arc<Mutex<MatchingEngine>>>>>,
    
    /// Cross-asset position tracker
    position_tracker: Arc<Mutex<CrossAssetPositionTracker>>,
    
    /// Correlation analyzer
    correlation_analyzer: Arc<Mutex<CorrelationAnalyzer>>,
    
    /// Risk manager
    risk_manager: Arc<Mutex<CorrelationAwareRiskManager>>,
    
    /// Portfolio optimizer
    portfolio_optimizer: Arc<Mutex<MultiAssetPortfolioOptimizer>>,
    
    /// Event broadcaster for market data updates
    market_data_sender: broadcast::Sender<MarketDataUpdate>,
    
    /// Event broadcaster for trade updates
    trade_sender: broadcast::Sender<TradeUpdate>,
}

impl MultiSymbolOrderBookManager {
    /// Create a new multi-symbol order book manager
    pub fn new() -> Self {
        let (market_data_sender, _) = broadcast::channel(10000);
        let (trade_sender, _) = broadcast::channel(10000);
        
        Self {
            order_books: Arc::new(RwLock::new(HashMap::new())),
            matching_engines: Arc::new(RwLock::new(HashMap::new())),
            position_tracker: Arc::new(Mutex::new(CrossAssetPositionTracker::new())),
            correlation_analyzer: Arc::new(Mutex::new(CorrelationAnalyzer::new())),
            risk_manager: Arc::new(Mutex::new(CorrelationAwareRiskManager::new())),
            portfolio_optimizer: Arc::new(Mutex::new(MultiAssetPortfolioOptimizer::new())),
            market_data_sender,
            trade_sender,
        }
    }
    
    /// Add a new symbol to the manager
    pub async fn add_symbol(&self, symbol: Symbol) -> Result<(), MultiAssetError> {
        let order_book = Arc::new(Mutex::new(OrderBook::new(symbol.to_string())));
        let matching_engine = Arc::new(Mutex::new(MatchingEngine::new(symbol.clone())));
        
        // Add to collections
        {
            let mut books = self.order_books.write().unwrap();
            books.insert(symbol.clone(), order_book);
        }
        
        {
            let mut engines = self.matching_engines.write().unwrap();
            engines.insert(symbol.clone(), matching_engine);
        }
        
        // Initialize position tracking for this symbol
        {
            let mut tracker = self.position_tracker.lock().unwrap();
            tracker.add_symbol(symbol.clone());
        }
        
        // Initialize correlation tracking
        {
            let mut analyzer = self.correlation_analyzer.lock().unwrap();
            analyzer.add_symbol(symbol.clone());
        }
        
        Ok(())
    }
    
    /// Process an order across the multi-symbol system
    pub async fn process_order(&self, order: Order) -> Result<OrderResult, MultiAssetError> {
        // Get the matching engine for this symbol
        let engine = {
            let engines = self.matching_engines.read().unwrap();
            engines.get(&order.symbol)
                .ok_or_else(|| MultiAssetError::SymbolNotFound(order.symbol.to_string()))?
                .clone()
        };
        
        // Check risk limits before processing
        {
            let mut risk_manager = self.risk_manager.lock().unwrap();
            risk_manager.check_order_risk(&order)?;
        }
        
        // Process the order
        let result = {
            let mut engine = engine.lock().unwrap();
            engine.process_order(order.clone())?
        };
        
        // Update position tracking
        if !result.trades.is_empty() {
            let mut tracker = self.position_tracker.lock().unwrap();
            for trade in &result.trades {
                tracker.update_position_from_trade(&order.symbol, trade, &order)?;
            }
        }
        
        // Update correlation data
        if let Some(last_price) = result.trades.last().map(|t| t.price) {
            let mut analyzer = self.correlation_analyzer.lock().unwrap();
            analyzer.add_price_observation(&order.symbol, last_price, current_timestamp())?;
        }
        
        // Broadcast updates
        let _ = self.trade_sender.send(TradeUpdate {
            symbol: order.symbol.clone(),
            trades: result.trades.clone(),
            timestamp: current_timestamp(),
        });
        
        let _ = self.market_data_sender.send(MarketDataUpdate {
            symbol: order.symbol.clone(),
            best_bid: self.get_best_bid(&order.symbol).await?,
            best_ask: self.get_best_ask(&order.symbol).await?,
            timestamp: current_timestamp(),
        });
        
        Ok(OrderResult {
            order_id: result.order_id,
            status: result.status,
            filled_size: result.filled_size,
            remaining_size: result.remaining_size,
            trades: result.trades,
        })
    }
    
    /// Get best bid for a symbol
    pub async fn get_best_bid(&self, symbol: &Symbol) -> Result<Option<u64>, MultiAssetError> {
        let books = self.order_books.read().unwrap();
        let book = books.get(symbol)
            .ok_or_else(|| MultiAssetError::SymbolNotFound(symbol.to_string()))?;
        
        let book = book.lock().unwrap();
        Ok(book.get_best_bid())
    }
    
    /// Get best ask for a symbol
    pub async fn get_best_ask(&self, symbol: &Symbol) -> Result<Option<u64>, MultiAssetError> {
        let books = self.order_books.read().unwrap();
        let book = books.get(symbol)
            .ok_or_else(|| MultiAssetError::SymbolNotFound(symbol.to_string()))?;
        
        let book = book.lock().unwrap();
        Ok(book.get_best_ask())
    }
    
    /// Get current positions across all symbols
    pub async fn get_positions(&self) -> Result<HashMap<Symbol, Position>, MultiAssetError> {
        let tracker = self.position_tracker.lock().unwrap();
        Ok(tracker.get_all_positions())
    }
    
    /// Get correlation matrix for all symbols
    pub async fn get_correlation_matrix(&self) -> Result<na::DMatrix<f64>, MultiAssetError> {
        let analyzer = self.correlation_analyzer.lock().unwrap();
        analyzer.calculate_correlation_matrix()
            .map_err(|e| MultiAssetError::CorrelationError(e))
    }
    
    /// Optimize portfolio allocation
    pub async fn optimize_portfolio(&self, target_return: f64) -> Result<PortfolioAllocation, MultiAssetError> {
        let positions = self.get_positions().await?;
        let correlation_matrix = self.get_correlation_matrix().await?;
        
        let mut optimizer = self.portfolio_optimizer.lock().unwrap();
        optimizer.optimize(positions, correlation_matrix, target_return)
            .map_err(|e| MultiAssetError::OptimizationError(e))
    }
    
    /// Subscribe to market data updates
    pub fn subscribe_market_data(&self) -> broadcast::Receiver<MarketDataUpdate> {
        self.market_data_sender.subscribe()
    }
    
    /// Subscribe to trade updates
    pub fn subscribe_trades(&self) -> broadcast::Receiver<TradeUpdate> {
        self.trade_sender.subscribe()
    }
}

/// Cross-asset position tracker
#[derive(Debug)]
pub struct CrossAssetPositionTracker {
    positions: HashMap<Symbol, Position>,
    trade_history: Vec<PositionUpdate>,
}

impl CrossAssetPositionTracker {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            trade_history: Vec::new(),
        }
    }
    
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.positions.insert(symbol, Position::default());
    }
    
    pub fn update_position_from_trade(
        &mut self, 
        symbol: &Symbol, 
        trade: &Trade, 
        order: &Order
    ) -> Result<(), MultiAssetError> {
        let position = self.positions.get_mut(symbol)
            .ok_or_else(|| MultiAssetError::SymbolNotFound(symbol.to_string()))?;
        
        let size_change = if order.side == Side::Buy {
            trade.size as i64
        } else {
            -(trade.size as i64)
        };
        
        position.quantity += size_change;
        position.notional += (trade.price * trade.size) as f64;
        position.average_price = if position.quantity != 0 {
            position.notional / position.quantity.abs() as f64
        } else {
            0.0
        };
        position.last_update = current_timestamp();
        
        // Record the update
        self.trade_history.push(PositionUpdate {
            symbol: symbol.clone(),
            trade_id: trade.id,
            size_change,
            price: trade.price,
            timestamp: trade.timestamp,
        });
        
        Ok(())
    }
    
    pub fn get_position(&self, symbol: &Symbol) -> Option<&Position> {
        self.positions.get(symbol)
    }
    
    pub fn get_all_positions(&self) -> HashMap<Symbol, Position> {
        self.positions.clone()
    }
    
    pub fn get_total_notional(&self) -> f64 {
        self.positions.values().map(|p| p.notional.abs()).sum()
    }
}

/// Position information for a single symbol
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Position {
    pub quantity: i64,        // Net position (positive = long, negative = short)
    pub notional: f64,        // Total notional value
    pub average_price: f64,   // Volume-weighted average price
    pub unrealized_pnl: f64,  // Unrealized P&L
    pub realized_pnl: f64,    // Realized P&L
    pub last_update: u64,     // Timestamp of last update
}

/// Position update record
#[derive(Debug, Clone)]
pub struct PositionUpdate {
    pub symbol: Symbol,
    pub trade_id: u64,
    pub size_change: i64,
    pub price: u64,
    pub timestamp: u64,
}

/// Order processing result
#[derive(Debug, Clone)]
pub struct OrderResult {
    pub order_id: u64,
    pub status: OrderStatus,
    pub filled_size: u64,
    pub remaining_size: u64,
    pub trades: Vec<Trade>,
}

/// Order status enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    FullyFilled,
    PartiallyFilled,
    Added,
    Rejected,
}

/// Market data update event
#[derive(Debug, Clone)]
pub struct MarketDataUpdate {
    pub symbol: Symbol,
    pub best_bid: Option<u64>,
    pub best_ask: Option<u64>,
    pub timestamp: u64,
}

/// Trade update event
#[derive(Debug, Clone)]
pub struct TradeUpdate {
    pub symbol: Symbol,
    pub trades: Vec<Trade>,
    pub timestamp: u64,
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
    use crate::orderbook::types::{OrderId, Side};
    
    #[tokio::test]
    async fn test_multi_symbol_manager_creation() {
        let manager = MultiSymbolOrderBookManager::new();
        
        let btc_symbol = Symbol::new("BTCUSD").unwrap();
        let eth_symbol = Symbol::new("ETHUSD").unwrap();
        
        manager.add_symbol(btc_symbol.clone()).await.unwrap();
        manager.add_symbol(eth_symbol.clone()).await.unwrap();
        
        // Verify symbols were added
        assert!(manager.get_best_bid(&btc_symbol).await.unwrap().is_none());
        assert!(manager.get_best_ask(&eth_symbol).await.unwrap().is_none());
    }
    
    #[tokio::test]
    async fn test_position_tracking() {
        let mut tracker = CrossAssetPositionTracker::new();
        let symbol = Symbol::new("BTCUSD").unwrap();
        
        tracker.add_symbol(symbol.clone());
        
        let trade = Trade {
            id: 1,
            price: 50000_000000, // $50,000 in fixed point
            size: 1_000000,      // 1 BTC in fixed point
            timestamp: current_timestamp(),
            buyer_order_id: 1,
            seller_order_id: 2,
            is_buyer_maker: false,
        };
        
        let order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000_000000,
            1_000000,
            current_timestamp(),
        );
        
        tracker.update_position_from_trade(&symbol, &trade, &order).unwrap();
        
        let position = tracker.get_position(&symbol).unwrap();
        assert_eq!(position.quantity, 1_000000);
        assert_eq!(position.average_price, 50000_000000.0);
    }
}