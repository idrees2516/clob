use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use rust_decimal::Decimal;
use thiserror::Error;

use crate::models::lob_rough_vol::{
    LOBError, LOBResult, Order, OrderId, OrderStatus, OrderType, Price, PriceLevel, Side, 
    TimeInForce, Trade, generate_order_id, current_timestamp
};

/// Errors specific to the matching engine
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum MatchingError {
    #[error("Order {0} not found")]
    OrderNotFound(OrderId),
    
    #[error("Insufficient liquidity for order {0}")]
    InsufficientLiquidity(OrderId),
    
    #[error("Order {0} would lock or cross the market")]
    WouldLockOrCross(OrderId),
    
    #[error("Order {0} is not active")]
    OrderNotActive(OrderId),
}

impl From<MatchingError> for LOBError {
    fn from(err: MatchingError) -> Self {
        LOBError::MatchingError(err.to_string())
    }
}

/// The matching engine handles order matching and trade execution
pub struct MatchingEngine {
    /// Queue of market data events to process
    market_data_events: VecDeque<MarketDataEvent>,
    /// Queue of order events to process
    order_events: VecDeque<OrderEvent>,
    /// Current timestamp for event processing
    current_time: u64,
    /// Core order book state
    bids: BTreeMap<Decimal, Vec<Order>>, // price -> FIFO queue
    asks: BTreeMap<Decimal, Vec<Order>>,
    orders: HashMap<OrderId, Order>, // All active orders
    /// For concurrency
    lock: Arc<Mutex<()>>,
    /// For persistence/event logging (stub)
    event_log: Vec<String>,
    /// For risk checks (stub)
    risk_engine: Option<Arc<dyn Fn(&Order) -> bool + Send + Sync>>,
    // Add stop/stop-limit order queues to MatchingEngine
    stop_orders: Vec<(Order, Decimal)>, // (order, stop_price)
    stop_limit_orders: Vec<(Order, Decimal, Decimal)>, // (order, stop_price, limit_price)
}

/// Represents a market data event
#[derive(Debug, Clone)]
pub enum MarketDataEvent {
    /// Price update with timestamp and new price
    PriceUpdate { timestamp: u64, price: Decimal },
    /// Volatility update with new value
    VolatilityUpdate { timestamp: u64, volatility: f64 },
    /// Market open/close
    MarketStateChange { timestamp: u64, is_open: bool },
}

/// Represents an order-related event
#[derive(Debug, Clone)]
pub enum OrderEvent {
    /// New order to be added to the book
    NewOrder(Order),
    /// Cancel an existing order
    CancelOrder { order_id: OrderId, timestamp: u64 },
    /// Modify an existing order
    ModifyOrder { 
        order_id: OrderId, 
        new_quantity: Option<u64>,
        new_price: Option<Decimal>,
        timestamp: u64 
    },
}

impl MatchingEngine {
    /// Create a new matching engine
    pub fn new() -> Self {
        Self {
            market_data_events: VecDeque::new(),
            order_events: VecDeque::new(),
            current_time: current_timestamp(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
            lock: Arc::new(Mutex::new(())),
            event_log: Vec::new(),
            risk_engine: None,
            stop_orders: Vec::new(),
            stop_limit_orders: Vec::new(),
        }
    }
    
    /// Process all pending events
    pub fn process_events(&mut self) -> LOBResult<Vec<Trade>> {
        let mut trades = Vec::new();
        
        // Process market data events first
        while let Some(event) = self.market_data_events.pop_front() {
            self.current_time = match event {
                MarketDataEvent::PriceUpdate { timestamp, price: _ } => timestamp,
                MarketDataEvent::VolatilityUpdate { timestamp, volatility: _ } => timestamp,
                MarketDataEvent::MarketStateChange { timestamp, is_open: _ } => timestamp,
            };
            // Process market data update
            // This would typically update internal state or trigger callbacks
        }
        
        // Process order events
        while let Some(event) = self.order_events.pop_front() {
            let trade = match event {
                OrderEvent::NewOrder(order) => self.process_new_order(order)?,
                OrderEvent::CancelOrder { order_id, timestamp } => {
                    self.current_time = timestamp;
                    self.process_cancel_order(order_id)?;
                    None
                },
                OrderEvent::ModifyOrder { order_id, new_quantity, new_price, timestamp } => {
                    self.current_time = timestamp;
                    self.process_modify_order(order_id, new_quantity, new_price)?
                },
            };
            
            if let Some(t) = trade {
                trades.push(t);
            }
        }
        
        Ok(trades)
    }
    
    /// Add a market data event to the queue
    pub fn add_market_data_event(&mut self, event: MarketDataEvent) {
        self.market_data_events.push_back(event);
    }
    
    /// Add an order event to the queue
    pub fn add_order_event(&mut self, event: OrderEvent) {
        self.order_events.push_back(event);
    }
    
    /// Process a new order (extended for stop/stop-limit)
    fn process_new_order(&mut self, mut order: Order) -> LOBResult<Option<Trade>> {
        let _guard = self.lock.lock().unwrap();
        if order.timestamp == 0 {
            order.timestamp = self.current_time;
        }
        // Risk check
        if let Some(ref risk) = self.risk_engine {
            if !risk(&order) {
                self.event_log.push(format!("Order {} failed risk check", order.id));
                return Err(LOBError::from(MatchingError::OrderNotActive(order.id)));
            }
        }
        // Self-trade prevention
        if let Some(user_id) = &order.user_id {
            if self.detect_self_trade(user_id, &order) {
                self.event_log.push(format!("Order {} would self-trade", order.id));
                return Err(LOBError::from(MatchingError::WouldLockOrCross(order.id)));
            }
        }
        // Order type dispatch
        match order.order_type {
            OrderType::Market => self.match_market_order(order),
            OrderType::Limit => self.match_limit_order(order),
            OrderType::Stop(stop_price) => self.handle_stop_order(order, stop_price),
            OrderType::StopLimit { stop_price, limit_price } => self.handle_stop_limit_order(order, stop_price, limit_price),
            _ => Err(LOBError::from(MatchingError::OrderNotActive(order.id))),
        }
    }

    /// Handle stop order: add to stop order queue, trigger when price crosses
    fn handle_stop_order(&mut self, order: Order, stop_price: Decimal) -> LOBResult<Option<Trade>> {
        // TODO: Add to stop order queue, monitor price updates in event loop
        self.stop_orders.push((order.clone(), stop_price));
        self.event_log.push(format!("Stop order {} registered at stop {}", order.id, stop_price));
        Ok(None)
    }

    /// Handle stop-limit order: add to stop-limit queue, trigger when price crosses
    fn handle_stop_limit_order(&mut self, order: Order, stop_price: Decimal, limit_price: Decimal) -> LOBResult<Option<Trade>> {
        // TODO: Add to stop-limit order queue, monitor price updates in event loop
        self.stop_limit_orders.push((order.clone(), stop_price, limit_price));
        self.event_log.push(format!("Stop-limit order {} registered at stop {} limit {}", order.id, stop_price, limit_price));
        Ok(None)
    }

    /// In the event loop, trigger stop/stop-limit orders when price crosses
    pub fn trigger_stop_orders(&mut self, last_trade_price: Decimal) {
        // Trigger stop orders
        let mut triggered = Vec::new();
        self.stop_orders.retain(|(order, stop_price)| {
            let should_trigger = match order.side {
                Side::Bid => last_trade_price >= *stop_price,
                Side::Ask => last_trade_price <= *stop_price,
            };
            if should_trigger {
                self.event_log.push(format!("Stop order {} triggered at price {}", order.id, last_trade_price));
                let mut market_order = order.clone();
                market_order.order_type = OrderType::Market;
                let _ = self.process_new_order(market_order);
                triggered.push(order.id);
                false // Remove from queue
            } else {
                true // Keep in queue
            }
        });
        // Trigger stop-limit orders
        self.stop_limit_orders.retain(|(order, stop_price, limit_price)| {
            let should_trigger = match order.side {
                Side::Bid => last_trade_price >= *stop_price,
                Side::Ask => last_trade_price <= *stop_price,
            };
            if should_trigger {
                self.event_log.push(format!("Stop-limit order {} triggered at price {}", order.id, last_trade_price));
                let mut limit_order = order.clone();
                limit_order.order_type = OrderType::Limit;
                limit_order.price = *limit_price;
                let _ = self.process_new_order(limit_order);
                triggered.push(order.id);
                false // Remove from queue
            } else {
                true // Keep in queue
            }
        });
    }
    
    /// Execute a market order immediately
    fn match_market_order(&mut self, mut order: Order) -> LOBResult<Option<Trade>> {
        let book = if order.side == Side::Bid { &mut self.asks } else { &mut self.bids };
        let mut remaining_qty = order.quantity;
        let mut trades = Vec::new();
        let mut best_prices: Vec<Decimal> = book.keys().cloned().collect();
        if order.side == Side::Bid {
            best_prices.sort(); // Ascending for asks
        } else {
            best_prices.sort_by(|a, b| b.cmp(a)); // Descending for bids
        }
        for price in best_prices {
            let queue = book.get_mut(&price).unwrap();
            let mut i = 0;
            while i < queue.len() && remaining_qty > 0.0 {
                let counter_order = &mut queue[i];
                let fill_qty = remaining_qty.min(counter_order.quantity - counter_order.filled_quantity);
                if fill_qty > 0.0 {
                    // Partial fill
                    counter_order.filled_quantity += fill_qty;
                    remaining_qty -= fill_qty;
                    let trade = Trade {
                        taker_order_id: order.id,
                        maker_order_id: counter_order.id,
                        price,
                        quantity: fill_qty,
                        timestamp: self.current_time,
                        is_buyer_maker: order.side == Side::Ask,
                    };
                    trades.push(trade);
                    self.event_log.push(format!("Trade: {} @ {} qty {}", order.id, price, fill_qty));
                }
                if counter_order.filled_quantity >= counter_order.quantity {
                    queue.remove(i);
                } else {
                    i += 1;
                }
            }
            if queue.is_empty() {
                book.remove(&price);
            }
            if remaining_qty <= 0.0 {
                break;
            }
        }
        if remaining_qty > 0.0 {
            // Unfilled portion is cancelled for market orders
            self.event_log.push(format!("Order {} partially filled, {} unfilled", order.id, remaining_qty));
        }
        Ok(trades.into_iter().next()) // Return first trade for now (batch support below)
    }
    
    /// Execute or add a limit order to the book
    fn match_limit_order(&mut self, mut order: Order) -> LOBResult<Option<Trade>> {
        let book = if order.side == Side::Bid { &mut self.asks } else { &mut self.bids };
        let mut remaining_qty = order.quantity;
        let mut trades = Vec::new();
        let mut best_prices: Vec<Decimal> = book.keys().cloned().collect();
        if order.side == Side::Bid {
            best_prices.sort(); // Ascending for asks
        } else {
            best_prices.sort_by(|a, b| b.cmp(a)); // Descending for bids
        }
        for price in best_prices {
            let is_cross = if order.side == Side::Bid {
                order.price >= price
            } else {
                order.price <= price
            };
            if !is_cross {
                break;
            }
            let queue = book.get_mut(&price).unwrap();
            let mut i = 0;
            while i < queue.len() && remaining_qty > 0.0 {
                let counter_order = &mut queue[i];
                let fill_qty = remaining_qty.min(counter_order.quantity - counter_order.filled_quantity);
                if fill_qty > 0.0 {
                    counter_order.filled_quantity += fill_qty;
                    remaining_qty -= fill_qty;
                    let trade = Trade {
                        taker_order_id: order.id,
                        maker_order_id: counter_order.id,
                        price,
                        quantity: fill_qty,
                        timestamp: self.current_time,
                        is_buyer_maker: order.side == Side::Ask,
                    };
                    trades.push(trade);
                    self.event_log.push(format!("Trade: {} @ {} qty {}", order.id, price, fill_qty));
                }
                if counter_order.filled_quantity >= counter_order.quantity {
                    queue.remove(i);
                } else {
                    i += 1;
                }
            }
            if queue.is_empty() {
                book.remove(&price);
            }
            if remaining_qty <= 0.0 {
                break;
            }
        }
        // After matching, handle TIF
        match order.time_in_force {
            TimeInForce::IOC => {
                // Immediate or Cancel: do not add to book if not fully filled
                if remaining_qty > 0.0 {
                    self.event_log.push(format!("IOC order {} unfilled portion cancelled", order.id));
                }
            }
            TimeInForce::FOK => {
                // Fill or Kill: only execute if fully filled
                if remaining_qty > 0.0 {
                    self.event_log.push(format!("FOK order {} not fully filled, cancelled", order.id));
                    // Rollback any partial fills (not shown here for brevity)
                    return Ok(None);
                }
            }
            TimeInForce::GTT(expiry) => {
                // Good Till Time: add to book, but must expire at expiry timestamp
                // (Expiration logic to be handled in event loop or background task)
                if remaining_qty > 0.0 {
                    let book_side = if order.side == Side::Bid { &mut self.bids } else { &mut self.asks };
                    book_side.entry(order.price).or_insert_with(Vec::new).push(order.clone());
                    self.orders.insert(order.id, order.clone());
                    self.event_log.push(format!("GTT order {} added to book at {} qty {} expires at {}", order.id, order.price, remaining_qty, expiry));
                }
            }
            TimeInForce::GTC | _ => {
                // Good Till Cancelled: add to book as usual
                if remaining_qty > 0.0 {
                    let book_side = if order.side == Side::Bid { &mut self.bids } else { &mut self.asks };
                    book_side.entry(order.price).or_insert_with(Vec::new).push(order.clone());
                    self.orders.insert(order.id, order.clone());
                    self.event_log.push(format!("Order {} added to book at {} qty {}", order.id, order.price, remaining_qty));
                }
            }
        }
        Ok(trades.into_iter().next())
    }

    fn detect_self_trade(&self, user_id: &str, order: &Order) -> bool {
        let book = if order.side == Side::Bid { &self.asks } else { &self.bids };
        for queue in book.values() {
            for counter_order in queue {
                if let Some(counter_user) = &counter_order.user_id {
                    if counter_user == user_id {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    /// Process an order cancellation
    fn process_cancel_order(&mut self, order_id: OrderId) -> LOBResult<()> {
        let _guard = self.lock.lock().unwrap();
        if let Some(order) = self.orders.remove(&order_id) {
            let book_side = if order.side == Side::Bid { &mut self.bids } else { &mut self.asks };
            if let Some(queue) = book_side.get_mut(&order.price) {
                queue.retain(|o| o.id != order_id);
                if queue.is_empty() {
                    book_side.remove(&order.price);
                }
            }
            self.event_log.push(format!("Order {} cancelled", order_id));
            Ok(())
        } else {
            self.event_log.push(format!("Cancel failed: order {} not found", order_id));
            Err(LOBError::from(MatchingError::OrderNotFound(order_id)))
        }
    }
    
    /// Process an order modification
    fn process_modify_order(
        &mut self,
        order_id: OrderId,
        new_quantity: Option<u64>,
        new_price: Option<Decimal>,
    ) -> LOBResult<Option<Trade>> {
        let _guard = self.lock.lock().unwrap();
        if let Some(mut order) = self.orders.remove(&order_id) {
            let book_side = if order.side == Side::Bid { &mut self.bids } else { &mut self.asks };
            if let Some(queue) = book_side.get_mut(&order.price) {
                queue.retain(|o| o.id != order_id);
                if queue.is_empty() {
                    book_side.remove(&order.price);
                }
            }
            if let Some(qty) = new_quantity {
                order.quantity = qty as f64;
            }
            if let Some(price) = new_price {
                order.price = price;
            }
            self.event_log.push(format!("Order {} modified: qty {:?}, price {:?}", order_id, new_quantity, new_price));
            // Re-insert as new order (price-time priority)
            self.process_new_order(order)
        } else {
            self.event_log.push(format!("Modify failed: order {} not found", order_id));
            Err(LOBError::from(MatchingError::OrderNotFound(order_id)))
        }
    }

    /// Process a batch of orders atomically (all-or-nothing)
    pub fn process_order_batch(&mut self, orders: Vec<Order>) -> LOBResult<Vec<Trade>> {
        let _guard = self.lock.lock().unwrap();
        let mut trades = Vec::new();
        let mut rollback_state = (self.bids.clone(), self.asks.clone(), self.orders.clone());
        let mut event_log_snapshot = self.event_log.clone();
        for order in &orders {
            match self.process_new_order(order.clone()) {
                Ok(Some(trade)) => trades.push(trade),
                Ok(None) => {},
                Err(e) => {
                    // Rollback all changes
                    self.bids = rollback_state.0;
                    self.asks = rollback_state.1;
                    self.orders = rollback_state.2;
                    self.event_log = event_log_snapshot;
                    self.event_log.push(format!("Batch failed, rolled back: {}", e));
                    return Err(e);
                }
            }
        }
        self.event_log.push(format!("Batch processed: {} orders, {} trades", orders.len(), trades.len()));
        Ok(trades)
    }

    /// Persist current state to durable storage (snapshot)
    pub fn snapshot(&self) {
        // TODO: Serialize bids, asks, orders, event_log to disk or database
        // Example: serde_json, bincode, or custom format
        // This is a stub for now
        self.event_log.push("Snapshot taken".to_string());
    }

    /// Flush event log to persistent storage
    pub fn flush_event_log(&self) {
        // TODO: Write event_log to disk, database, or external event bus
        // Example: Kafka, NATS, or file
        // This is a stub for now
    }

    /// Publish event to external event bus
    pub fn publish_event(&self, event: &str) {
        // TODO: Integrate with Kafka, NATS, or other pub/sub system
        // This is a stub for now
    }

    /// Prepare for high-throughput, multi-core operation (sharding/partitioning stub)
    pub fn enable_sharding(&mut self, _num_shards: usize) {
        // TODO: Partition order book by price range or instrument for lock-free scaling
        // This is a stub for now
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_market_order_execution() {
        let mut engine = MatchingEngine::new();
        
        // Create a market buy order
        let order = Order::market(
            Side::Bid,
            100, // quantity
            None,
            None,
            None,
        ).unwrap();
        
        // Process the order
        let result = engine.process_new_order(order).unwrap();
        
        // In a real implementation, we would check if the order was executed
        // and trades were generated
        assert!(result.is_none());
    }
    
    #[test]
    fn test_limit_order_placement() {
        let mut engine = MatchingEngine::new();
        
        // Create a limit sell order
        let order = Order::limit(
            Side::Ask,
            dec!(100.0), // price
            100,         // quantity
            None,
            None,
            None,
        ).unwrap();
        
        // Process the order
        let result = engine.process_new_order(order).unwrap();
        
        // The order should be added to the book, no trades yet
        assert!(result.is_none());
    }
}
