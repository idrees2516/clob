use std::collections::{BTreeMap, VecDeque, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Fixed-point scaling factors for deterministic arithmetic
pub const PRICE_SCALE: u64 = 1_000_000;  // 6 decimal places for price precision
pub const VOLUME_SCALE: u64 = 1_000_000; // 6 decimal places for volume precision
pub const MAX_ORDERS_PER_LEVEL: u32 = 1000; // Maximum orders per price level

/// Comprehensive error types for order book operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OrderBookError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    #[error("Order not found: {0}")]
    OrderNotFound(u64),
    #[error("Invalid price: {0}")]
    InvalidPrice(u64),
    #[error("Invalid size: {0}")]
    InvalidSize(u64),
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(u64),
    #[error("Symbol mismatch: expected {expected}, got {actual}")]
    SymbolMismatch { expected: String, actual: String },
    #[error("Insufficient liquidity for order {order_id}")]
    InsufficientLiquidity { order_id: u64 },
    #[error("Price level not found: {0}")]
    PriceLevelNotFound(u64),
    #[error("Order book is locked")]
    OrderBookLocked,
    #[error("Maximum orders per level exceeded: {0}")]
    MaxOrdersExceeded(u32),
}

/// Unique identifier for orders across the system
/// 
/// OrderId is designed to be:
/// - Globally unique across all symbols and time
/// - Monotonically increasing for deterministic ordering
/// - Efficiently comparable and hashable
/// - Compatible with zkVM circuits (64-bit integer)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OrderId(pub u64);

impl OrderId {
    /// Create a new OrderId from a u64
    pub fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw u64 value
    pub fn as_u64(&self) -> u64 {
        self.0
    }
    
    /// Generate the next OrderId in sequence
    pub fn next(&self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

impl From<u64> for OrderId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<OrderId> for u64 {
    fn from(id: OrderId) -> u64 {
        id.0
    }
}

/// Trading symbol identifier
/// 
/// Represents a trading pair or instrument symbol with validation
/// and normalization capabilities for consistent handling across
/// the system.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(String);

impl Symbol {
    /// Create a new Symbol with validation
    /// 
    /// # Arguments
    /// * `symbol` - The symbol string (e.g., "BTCUSD", "ETHUSD")
    /// 
    /// # Returns
    /// * `Result<Symbol, OrderBookError>` - Valid symbol or error
    /// 
    /// # Examples
    /// ```
    /// use hf_quoting_liquidity_clob::orderbook::Symbol;
    /// 
    /// let symbol = Symbol::new("BTCUSD").unwrap();
    /// assert_eq!(symbol.as_str(), "BTCUSD");
    /// ```
    pub fn new(symbol: &str) -> Result<Self, OrderBookError> {
        let normalized = symbol.trim().to_uppercase();
        
        if normalized.is_empty() {
            return Err(OrderBookError::InvalidOrder(
                "Symbol cannot be empty".to_string()
            ));
        }
        
        if normalized.len() > 20 {
            return Err(OrderBookError::InvalidOrder(
                "Symbol too long (max 20 characters)".to_string()
            ));
        }
        
        // Basic validation for trading pair format
        if !normalized.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(OrderBookError::InvalidOrder(
                "Symbol contains invalid characters".to_string()
            ));
        }
        
        Ok(Self(normalized))
    }
    
    /// Get the symbol as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Get the symbol as an owned String
    pub fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Self(s.trim().to_uppercase())
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Self(s.trim().to_uppercase())
    }
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Represents an order in the system
/// 
/// Orders are the fundamental unit of trading activity. Each order
/// contains all necessary information for matching, tracking, and
/// audit purposes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Order {
    /// Unique identifier for this order
    pub id: OrderId,
    
    /// Trading symbol for this order
    pub symbol: Symbol,
    
    /// Order side (Buy or Sell)
    pub side: Side,
    
    /// Price in fixed-point representation (scaled by PRICE_SCALE)
    /// For market orders, this may be 0 or ignored
    pub price: u64,
    
    /// Order size/quantity in fixed-point representation (scaled by VOLUME_SCALE)
    pub size: u64,
    
    /// Timestamp when order was created (nanoseconds since epoch)
    pub timestamp: u64,
    
    /// Order type (Market or Limit)
    pub order_type: OrderType,
    
    /// Optional time-in-force specification
    pub time_in_force: TimeInForce,
    
    /// Client order ID for tracking (optional)
    pub client_order_id: Option<String>,
}

/// Time-in-force specifications for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good Till Cancelled - remains active until explicitly cancelled
    GTC,
    /// Immediate Or Cancel - execute immediately, cancel remainder
    IOC,
    /// Fill Or Kill - execute completely or cancel entirely
    FOK,
    /// Good Till Date - remains active until specified timestamp
    GTD(u64),
}

impl Default for TimeInForce {
    fn default() -> Self {
        TimeInForce::GTC
    }
}

impl Order {
    /// Create a new limit order
    pub fn new_limit(
        id: OrderId,
        symbol: Symbol,
        side: Side,
        price: u64,
        size: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            price,
            size,
            timestamp,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
        }
    }
    
    /// Create a new market order
    pub fn new_market(
        id: OrderId,
        symbol: Symbol,
        side: Side,
        size: u64,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            symbol,
            side,
            price: 0, // Market orders don't have a limit price
            size,
            timestamp,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC, // Market orders are typically IOC
            client_order_id: None,
        }
    }
    
    /// Check if this is a buy order
    pub fn is_buy(&self) -> bool {
        self.side == Side::Buy
    }
    
    /// Check if this is a sell order
    pub fn is_sell(&self) -> bool {
        self.side == Side::Sell
    }
    
    /// Check if this is a market order
    pub fn is_market_order(&self) -> bool {
        self.order_type == OrderType::Market
    }
    
    /// Check if this is a limit order
    pub fn is_limit_order(&self) -> bool {
        self.order_type == OrderType::Limit
    }
    
    /// Validate order parameters
    pub fn validate(&self) -> Result<(), OrderBookError> {
        // Size must be positive
        if self.size == 0 {
            return Err(OrderBookError::InvalidSize(self.size));
        }
        
        // Limit orders must have positive price
        if self.order_type == OrderType::Limit && self.price == 0 {
            return Err(OrderBookError::InvalidPrice(self.price));
        }
        
        // Timestamp should be reasonable (not zero, not too far in future)
        if self.timestamp == 0 {
            return Err(OrderBookError::InvalidTimestamp(self.timestamp));
        }
        
        Ok(())
    }
}

/// Represents a price level with FIFO order queue
/// 
/// PriceLevel maintains all orders at a specific price point with
/// strict FIFO (First-In-First-Out) ordering for price-time priority.
/// This is the core data structure for maintaining order precedence
/// within each price level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    /// Price for this level (fixed-point representation)
    pub price: u64,
    
    /// Total size/volume at this price level
    pub total_size: u64,
    
    /// Number of orders at this price level
    pub order_count: u32,
    
    /// FIFO queue of orders at this price level
    pub orders: VecDeque<OrderId>,
    
    /// Timestamp of first order at this level (for level priority)
    pub timestamp: u64,
    
    /// Last update timestamp for this level
    pub last_update: u64,
}

impl PriceLevel {
    /// Create a new empty price level
    pub fn new(price: u64) -> Self {
        Self {
            price,
            total_size: 0,
            order_count: 0,
            orders: VecDeque::new(),
            timestamp: 0,
            last_update: 0,
        }
    }
    
    /// Create a new price level with an initial order
    pub fn with_order(order: &Order) -> Result<Self, OrderBookError> {
        if order.price != order.price {
            return Err(OrderBookError::InvalidPrice(order.price));
        }
        
        let mut level = Self::new(order.price);
        level.add_order_id(order.id, order.size, order.timestamp)?;
        Ok(level)
    }
    
    /// Add an order ID to this price level
    pub fn add_order_id(&mut self, order_id: OrderId, size: u64, timestamp: u64) -> Result<(), OrderBookError> {
        if self.order_count >= MAX_ORDERS_PER_LEVEL {
            return Err(OrderBookError::MaxOrdersExceeded(self.order_count));
        }
        
        if self.orders.is_empty() {
            self.timestamp = timestamp;
        }
        
        self.total_size = self.total_size.saturating_add(size);
        self.order_count += 1;
        self.last_update = timestamp;
        self.orders.push_back(order_id);
        
        Ok(())
    }
    
    /// Remove an order ID from this price level
    pub fn remove_order_id(&mut self, order_id: OrderId, size: u64, timestamp: u64) -> Result<bool, OrderBookError> {
        if let Some(pos) = self.orders.iter().position(|&id| id == order_id) {
            self.orders.remove(pos);
            self.total_size = self.total_size.saturating_sub(size);
            self.order_count = self.order_count.saturating_sub(1);
            self.last_update = timestamp;
            
            Ok(self.orders.is_empty())
        } else {
            Err(OrderBookError::OrderNotFound(order_id.as_u64()))
        }
    }
    
    /// Get the first (oldest) order ID in the FIFO queue
    pub fn front_order_id(&self) -> Option<OrderId> {
        self.orders.front().copied()
    }
    
    /// Remove and return the first order ID from the FIFO queue
    pub fn pop_front_order_id(&mut self, size: u64, timestamp: u64) -> Option<OrderId> {
        if let Some(order_id) = self.orders.pop_front() {
            self.total_size = self.total_size.saturating_sub(size);
            self.order_count = self.order_count.saturating_sub(1);
            self.last_update = timestamp;
            Some(order_id)
        } else {
            None
        }
    }
    
    /// Check if this price level is empty
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
    
    /// Get the number of orders at this level
    pub fn len(&self) -> usize {
        self.orders.len()
    }
    
    /// Validate the internal consistency of this price level
    pub fn validate(&self) -> Result<(), OrderBookError> {
        if self.orders.len() != self.order_count as usize {
            return Err(OrderBookError::InvalidOrder(
                "Order count mismatch in price level".to_string()
            ));
        }
        
        if self.orders.is_empty() && self.total_size != 0 {
            return Err(OrderBookError::InvalidOrder(
                "Empty price level with non-zero size".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Represents a trade execution with comprehensive details
/// 
/// Trade contains all information about a completed transaction
/// between two orders, including execution details, participant
/// information, and audit trail data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier
    pub id: u64,
    
    /// Trading symbol for this trade
    pub symbol: Symbol,
    
    /// Execution price (fixed-point representation)
    pub price: u64,
    
    /// Executed size/quantity (fixed-point representation)
    pub size: u64,
    
    /// Trade execution timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    
    /// Order ID of the buyer
    pub buyer_order_id: OrderId,
    
    /// Order ID of the seller
    pub seller_order_id: OrderId,
    
    /// True if buyer was the maker (passive side)
    pub is_buyer_maker: bool,
    
    /// Sequence number for this trade
    pub sequence: u64,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        id: u64,
        symbol: Symbol,
        price: u64,
        size: u64,
        timestamp: u64,
        buyer_order_id: OrderId,
        seller_order_id: OrderId,
        is_buyer_maker: bool,
        sequence: u64,
    ) -> Self {
        Self {
            id,
            symbol,
            price,
            size,
            timestamp,
            buyer_order_id,
            seller_order_id,
            is_buyer_maker,
            sequence,
        }
    }
    
    /// Get the maker order ID
    pub fn maker_order_id(&self) -> OrderId {
        if self.is_buyer_maker {
            self.buyer_order_id
        } else {
            self.seller_order_id
        }
    }
    
    /// Get the taker order ID
    pub fn taker_order_id(&self) -> OrderId {
        if self.is_buyer_maker {
            self.seller_order_id
        } else {
            self.buyer_order_id
        }
    }
    
    /// Calculate the notional value of this trade
    pub fn notional_value(&self) -> u64 {
        // Using fixed-point arithmetic: (price * size) / SCALE
        (self.price.saturating_mul(self.size)) / PRICE_SCALE
    }
}

/// Order execution result
#[derive(Debug, Clone)]
pub struct OrderResult {
    pub order_id: u64,
    pub status: OrderStatus,
    pub filled_size: u64,
    pub remaining_size: u64,
    pub trades: Vec<Trade>,
}

/// Order status after processing
#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    FullyFilled,
    PartiallyFilled,
    Added, // Added to book without any fills
    Rejected,
}

/// Order type
#[derive(Debug, Clone, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
}

/// Market depth information for a specific number of levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDepth {
    /// Bid levels (price, size) sorted by price descending
    pub bids: Vec<(u64, u64)>,
    
    /// Ask levels (price, size) sorted by price ascending  
    pub asks: Vec<(u64, u64)>,
    
    /// Timestamp when this depth snapshot was taken
    pub timestamp: u64,
    
    /// Sequence number for this depth update
    pub sequence: u64,
}

impl MarketDepth {
    /// Create a new empty market depth
    pub fn new(timestamp: u64, sequence: u64) -> Self {
        Self {
            bids: Vec::new(),
            asks: Vec::new(),
            timestamp,
            sequence,
        }
    }
    
    /// Get the best bid price
    pub fn best_bid(&self) -> Option<u64> {
        self.bids.first().map(|(price, _)| *price)
    }
    
    /// Get the best ask price
    pub fn best_ask(&self) -> Option<u64> {
        self.asks.first().map(|(price, _)| *price)
    }
    
    /// Get the spread between best bid and ask
    pub fn spread(&self) -> Option<u64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some(ask.saturating_sub(bid)),
            _ => None,
        }
    }
    
    /// Get the mid price
    pub fn mid_price(&self) -> Option<u64> {
        match (self.best_ask(), self.best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2),
            _ => None,
        }
    }
}

/// Central Limit Order Book data structure with comprehensive functionality
/// 
/// This is the core data structure that maintains all orders for a specific
/// trading symbol with price-time priority matching. It provides efficient
/// order management, matching, and market data generation capabilities.
#[derive(Debug)]
pub struct CentralLimitOrderBook {
    /// Trading symbol for this order book
    pub symbol: Symbol,
    
    /// Bid side price levels (sorted descending by price for efficient best bid access)
    pub bids: BTreeMap<u64, PriceLevel>,
    
    /// Ask side price levels (sorted ascending by price for efficient best ask access)
    pub asks: BTreeMap<u64, PriceLevel>,
    
    /// Fast lookup map from OrderId to Order
    pub orders: HashMap<OrderId, Order>,
    
    /// Monotonically increasing sequence number for all operations
    pub sequence_number: AtomicU64,
    
    /// Price of the last executed trade
    pub last_trade_price: Option<u64>,
    
    /// Timestamp of the last executed trade
    pub last_trade_time: u64,
    
    /// Next trade ID to be assigned
    pub next_trade_id: AtomicU64,
    
    /// Total number of orders currently in the book
    pub total_orders: u32,
    
    /// Total volume on bid side
    pub total_bid_volume: u64,
    
    /// Total volume on ask side
    pub total_ask_volume: u64,
    
    /// Creation timestamp of this order book
    pub created_at: u64,
    
    /// Last update timestamp
    pub last_update: u64,
}

impl CentralLimitOrderBook {
    /// Create a new empty order book for the given symbol
    pub fn new(symbol: Symbol) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
            
        Self {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            sequence_number: AtomicU64::new(0),
            last_trade_price: None,
            last_trade_time: 0,
            next_trade_id: AtomicU64::new(1),
            total_orders: 0,
            total_bid_volume: 0,
            total_ask_volume: 0,
            created_at: now,
            last_update: now,
        }
    }
    
    /// Add an order to the book and return any resulting trades
    pub fn add_order(&mut self, order: Order) -> Result<Vec<Trade>, OrderBookError> {
        // Validate the order first
        order.validate()?;
        
        // Check symbol matches
        if order.symbol != self.symbol {
            return Err(OrderBookError::SymbolMismatch {
                expected: self.symbol.to_string(),
                actual: order.symbol.to_string(),
            });
        }
        
        let mut trades = Vec::new();
        let mut remaining_order = order.clone();
        
        // Handle market orders and aggressive limit orders by matching first
        if remaining_order.is_market_order() || self.can_match_immediately(&remaining_order) {
            trades = self.match_order(&mut remaining_order)?;
        }
        
        // Add remaining quantity to the book if any
        if remaining_order.size > 0 {
            self.add_order_to_book(remaining_order)?;
        }
        
        // Update sequence number and timestamp
        self.sequence_number.fetch_add(1, Ordering::SeqCst);
        self.last_update = remaining_order.timestamp;
        
        Ok(trades)
    }
    
    /// Cancel an order from the book
    pub fn cancel_order(&mut self, order_id: OrderId) -> Result<Order, OrderBookError> {
        // Find and remove the order
        let order = self.orders.remove(&order_id)
            .ok_or(OrderBookError::OrderNotFound(order_id.as_u64()))?;
        
        // Remove from the appropriate price level
        let price_levels = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        
        if let Some(level) = price_levels.get_mut(&order.price) {
            let is_empty = level.remove_order_id(order_id, order.size, order.timestamp)?;
            
            // Remove empty price level
            if is_empty {
                price_levels.remove(&order.price);
            }
            
            // Update volume tracking
            match order.side {
                Side::Buy => self.total_bid_volume = self.total_bid_volume.saturating_sub(order.size),
                Side::Sell => self.total_ask_volume = self.total_ask_volume.saturating_sub(order.size),
            }
        }
        
        self.total_orders = self.total_orders.saturating_sub(1);
        self.sequence_number.fetch_add(1, Ordering::SeqCst);
        self.last_update = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Ok(order)
    }
    
    /// Get the best bid price
    pub fn get_best_bid(&self) -> Option<u64> {
        self.bids.keys().next_back().copied()
    }
    
    /// Get the best ask price
    pub fn get_best_ask(&self) -> Option<u64> {
        self.asks.keys().next().copied()
    }
    
    /// Get the current spread
    pub fn get_spread(&self) -> Option<u64> {
        match (self.get_best_ask(), self.get_best_bid()) {
            (Some(ask), Some(bid)) => Some(ask.saturating_sub(bid)),
            _ => None,
        }
    }
    
    /// Get the mid price
    pub fn get_mid_price(&self) -> Option<u64> {
        match (self.get_best_ask(), self.get_best_bid()) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2),
            _ => None,
        }
    }
    
    /// Get market depth for specified number of levels
    pub fn get_market_depth(&self, levels: usize) -> MarketDepth {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let sequence = self.sequence_number.load(Ordering::SeqCst);
        
        let mut depth = MarketDepth::new(timestamp, sequence);
        
        // Collect bid levels (highest price first)
        depth.bids = self.bids.iter()
            .rev()
            .take(levels)
            .map(|(&price, level)| (price, level.total_size))
            .collect();
        
        // Collect ask levels (lowest price first)
        depth.asks = self.asks.iter()
            .take(levels)
            .map(|(&price, level)| (price, level.total_size))
            .collect();
        
        depth
    }
    
    /// Get total volume at or better than specified price
    pub fn get_volume_at_or_better(&self, side: Side, price: u64) -> u64 {
        match side {
            Side::Buy => {
                // For buys, sum all ask levels at or below the price
                self.asks.range(..=price)
                    .map(|(_, level)| level.total_size)
                    .sum()
            }
            Side::Sell => {
                // For sells, sum all bid levels at or above the price
                self.bids.range(price..)
                    .map(|(_, level)| level.total_size)
                    .sum()
            }
        }
    }
    
    /// Check if an order can be matched immediately
    fn can_match_immediately(&self, order: &Order) -> bool {
        match order.side {
            Side::Buy => {
                if let Some(best_ask) = self.get_best_ask() {
                    order.price >= best_ask
                } else {
                    false
                }
            }
            Side::Sell => {
                if let Some(best_bid) = self.get_best_bid() {
                    order.price <= best_bid
                } else {
                    false
                }
            }
        }
    }
    
    /// Match an order against the book and return resulting trades
    fn match_order(&mut self, order: &mut Order) -> Result<Vec<Trade>, OrderBookError> {
        let mut trades = Vec::new();
        
        match order.side {
            Side::Buy => {
                // Match against asks (lowest price first)
                while order.size > 0 {
                    if let Some((&ask_price, _)) = self.asks.first_key_value() {
                        if order.is_market_order() || order.price >= ask_price {
                            let trade = self.match_against_ask(ask_price, order)?;
                            if let Some(t) = trade {
                                trades.push(t);
                            } else {
                                break; // No more liquidity at this level
                            }
                        } else {
                            break; // Price doesn't cross
                        }
                    } else {
                        break; // No asks available
                    }
                }
            }
            Side::Sell => {
                // Match against bids (highest price first)
                while order.size > 0 {
                    if let Some((&bid_price, _)) = self.bids.last_key_value() {
                        if order.is_market_order() || order.price <= bid_price {
                            let trade = self.match_against_bid(bid_price, order)?;
                            if let Some(t) = trade {
                                trades.push(t);
                            } else {
                                break; // No more liquidity at this level
                            }
                        } else {
                            break; // Price doesn't cross
                        }
                    } else {
                        break; // No bids available
                    }
                }
            }
        }
        
        Ok(trades)
    }
    
    /// Match against the best ask level
    fn match_against_ask(&mut self, ask_price: u64, buy_order: &mut Order) -> Result<Option<Trade>, OrderBookError> {
        if let Some(ask_level) = self.asks.get_mut(&ask_price) {
            if let Some(maker_order_id) = ask_level.front_order_id() {
                if let Some(maker_order) = self.orders.get(&maker_order_id).cloned() {
                    let trade_size = std::cmp::min(buy_order.size, maker_order.size);
                    let trade_id = self.next_trade_id.fetch_add(1, Ordering::SeqCst);
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    
                    // Create the trade
                    let trade = Trade::new(
                        trade_id,
                        self.symbol.clone(),
                        ask_price,
                        trade_size,
                        timestamp,
                        buy_order.id,
                        maker_order.id,
                        false, // Seller is the maker
                        self.sequence_number.load(Ordering::SeqCst),
                    );
                    
                    // Update orders
                    buy_order.size = buy_order.size.saturating_sub(trade_size);
                    
                    // Update or remove maker order
                    if maker_order.size == trade_size {
                        // Fully filled - remove from book
                        ask_level.pop_front_order_id(trade_size, timestamp);
                        self.orders.remove(&maker_order_id);
                        self.total_orders = self.total_orders.saturating_sub(1);
                        
                        // Remove empty price level
                        if ask_level.is_empty() {
                            self.asks.remove(&ask_price);
                        }
                    } else {
                        // Partially filled - update size
                        if let Some(mut_maker_order) = self.orders.get_mut(&maker_order_id) {
                            mut_maker_order.size = mut_maker_order.size.saturating_sub(trade_size);
                        }
                        ask_level.total_size = ask_level.total_size.saturating_sub(trade_size);
                    }
                    
                    // Update book state
                    self.total_ask_volume = self.total_ask_volume.saturating_sub(trade_size);
                    self.last_trade_price = Some(ask_price);
                    self.last_trade_time = timestamp;
                    
                    return Ok(Some(trade));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Match against the best bid level
    fn match_against_bid(&mut self, bid_price: u64, sell_order: &mut Order) -> Result<Option<Trade>, OrderBookError> {
        if let Some(bid_level) = self.bids.get_mut(&bid_price) {
            if let Some(maker_order_id) = bid_level.front_order_id() {
                if let Some(maker_order) = self.orders.get(&maker_order_id).cloned() {
                    let trade_size = std::cmp::min(sell_order.size, maker_order.size);
                    let trade_id = self.next_trade_id.fetch_add(1, Ordering::SeqCst);
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64;
                    
                    // Create the trade
                    let trade = Trade::new(
                        trade_id,
                        self.symbol.clone(),
                        bid_price,
                        trade_size,
                        timestamp,
                        maker_order.id,
                        sell_order.id,
                        true, // Buyer is the maker
                        self.sequence_number.load(Ordering::SeqCst),
                    );
                    
                    // Update orders
                    sell_order.size = sell_order.size.saturating_sub(trade_size);
                    
                    // Update or remove maker order
                    if maker_order.size == trade_size {
                        // Fully filled - remove from book
                        bid_level.pop_front_order_id(trade_size, timestamp);
                        self.orders.remove(&maker_order_id);
                        self.total_orders = self.total_orders.saturating_sub(1);
                        
                        // Remove empty price level
                        if bid_level.is_empty() {
                            self.bids.remove(&bid_price);
                        }
                    } else {
                        // Partially filled - update size
                        if let Some(mut_maker_order) = self.orders.get_mut(&maker_order_id) {
                            mut_maker_order.size = mut_maker_order.size.saturating_sub(trade_size);
                        }
                        bid_level.total_size = bid_level.total_size.saturating_sub(trade_size);
                    }
                    
                    // Update book state
                    self.total_bid_volume = self.total_bid_volume.saturating_sub(trade_size);
                    self.last_trade_price = Some(bid_price);
                    self.last_trade_time = timestamp;
                    
                    return Ok(Some(trade));
                }
            }
        }
        
        Ok(None)
    }
    
    /// Add an order to the appropriate side of the book
    fn add_order_to_book(&mut self, order: Order) -> Result<(), OrderBookError> {
        let price_levels = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        
        // Get or create price level
        let level = price_levels.entry(order.price).or_insert_with(|| PriceLevel::new(order.price));
        
        // Add order to level
        level.add_order_id(order.id, order.size, order.timestamp)?;
        
        // Update volume tracking
        match order.side {
            Side::Buy => self.total_bid_volume = self.total_bid_volume.saturating_add(order.size),
            Side::Sell => self.total_ask_volume = self.total_ask_volume.saturating_add(order.size),
        }
        
        // Store order for fast lookup
        self.orders.insert(order.id, order);
        self.total_orders += 1;
        
        Ok(())
    }
    
    /// Validate the internal consistency of the order book
    pub fn validate(&self) -> Result<(), OrderBookError> {
        // Check that all orders in price levels exist in the orders map
        for (_, level) in &self.bids {
            level.validate()?;
            for &order_id in &level.orders {
                if !self.orders.contains_key(&order_id) {
                    return Err(OrderBookError::InvalidOrder(
                        format!("Order {} in bid level but not in orders map", order_id)
                    ));
                }
            }
        }
        
        for (_, level) in &self.asks {
            level.validate()?;
            for &order_id in &level.orders {
                if !self.orders.contains_key(&order_id) {
                    return Err(OrderBookError::InvalidOrder(
                        format!("Order {} in ask level but not in orders map", order_id)
                    ));
                }
            }
        }
        
        // Check total order count
        let expected_orders = self.bids.values().map(|l| l.order_count).sum::<u32>() +
                             self.asks.values().map(|l| l.order_count).sum::<u32>();
        
        if expected_orders != self.total_orders {
            return Err(OrderBookError::InvalidOrder(
                format!("Order count mismatch: expected {}, got {}", expected_orders, self.total_orders)
            ));
        }
        
        Ok(())
    }
    
    /// Get statistics about the order book
    pub fn get_statistics(&self) -> OrderBookStatistics {
        OrderBookStatistics {
            symbol: self.symbol.clone(),
            total_orders: self.total_orders,
            bid_levels: self.bids.len() as u32,
            ask_levels: self.asks.len() as u32,
            total_bid_volume: self.total_bid_volume,
            total_ask_volume: self.total_ask_volume,
            best_bid: self.get_best_bid(),
            best_ask: self.get_best_ask(),
            spread: self.get_spread(),
            mid_price: self.get_mid_price(),
            last_trade_price: self.last_trade_price,
            last_trade_time: self.last_trade_time,
            sequence_number: self.sequence_number.load(Ordering::SeqCst),
            created_at: self.created_at,
            last_update: self.last_update,
        }
    }
}

/// Order book statistics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookStatistics {
    pub symbol: Symbol,
    pub total_orders: u32,
    pub bid_levels: u32,
    pub ask_levels: u32,
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub best_bid: Option<u64>,
    pub best_ask: Option<u64>,
    pub spread: Option<u64>,
    pub mid_price: Option<u64>,
    pub last_trade_price: Option<u64>,
    pub last_trade_time: u64,
    pub sequence_number: u64,
    pub created_at: u64,
    pub last_update: u64,
} 