use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    LOBError, LOBResult, Order, OrderId, OrderStatus, OrderType, Price, PriceLevel, Side, 
    Timestamp, Trade, generate_order_id, current_timestamp,
    RoughVolatilityModel, RoughVolatilityParams
};

/// Represents a trade that occurred in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub taker_order_id: u64,
    pub maker_order_id: u64,
    pub price: Decimal,
    pub quantity: f64,
    pub timestamp: u64,
    pub is_buyer_maker: bool,
}

/// Represents the current state of the order book
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderBookState {
    /// Bids (buy orders) stored in price-time priority (highest price first)
    pub bids: BTreeMap<Decimal, PriceLevel>,
    /// Asks (sell orders) stored in price-time priority (lowest price first)
    pub asks: BTreeMap<Decimal, PriceLevel>,
    /// All active orders by ID for quick lookup
    pub orders: HashMap<u64, (Decimal, Side)>,
    /// Last update timestamp
    pub last_update: u64,
    /// Sequence number for versioning
    pub sequence: u64,
}

/// The main Limit Order Book implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitOrderBook {
    state: OrderBookState,
    /// Minimum price increment (tick size)
    tick_size: Decimal,
    /// Minimum quantity increment (lot size)
    lot_size: Decimal,
    /// Maximum number of orders per side
    max_orders_per_side: usize,
    /// Maximum number of price levels per side
    max_price_levels: usize,
    /// Last executed trade
    last_trade: Option<Trade>,
    /// Trade history (circular buffer)
    trade_history: VecDeque<Trade>,
    /// Maximum trade history size
    max_trade_history: usize,
    /// Volatility model for simulating market data
    volatility_model: Option<RoughVolatilityModel>,
    /// Last mid price
    last_mid_price: Option<f64>,
    /// Last timestamp
    last_timestamp: Option<u64>,
}

impl LimitOrderBook {
    /// Create a new LimitOrderBook with the given tick and lot sizes
    pub fn new(tick_size: Decimal, lot_size: Decimal) -> LOBResult<Self> {
        if tick_size <= Decimal::ZERO || lot_size <= Decimal::ZERO {
            return Err(LOBError::InvalidPrice(0.0));
        }

        Ok(Self {
            state: OrderBookState::default(),
            tick_size,
            lot_size,
            max_orders_per_side: 10_000,
            max_price_levels: 1_000,
            last_trade: None,
            trade_history: VecDeque::with_capacity(10_000),
            max_trade_history: 10_000,
            volatility_model: None,
            last_mid_price: None,
            last_timestamp: None,
        })
    }

    /// Add a new order to the book
    pub fn add_order(&mut self, mut order: Order) -> LOBResult<Vec<Trade>> {
        self.state.sequence += 1;
        order.timestamp = current_timestamp();
        
        // Validate the order
        self.validate_order(&order)?;

        // Process the order based on its type
        match order.order_type {
            OrderType::Market => self.process_market_order(order),
            OrderType::Limit => self.process_limit_order(order),
        }
    }

    /// Cancel an existing order
    pub fn cancel_order(&mut self, order_id: u64) -> LOBResult<Option<Order>> {
        if let Some(order) = self.state.orders.get(&order_id) {
            let side = order.1;
            let price = order.0;

            // Remove from price level
            let price_level = match side {
                Side::Bid => self.state.bids.get_mut(&price),
                Side::Ask => self.state.asks.get_mut(&price),
            };

            if let Some(level) = price_level {
                if let Some(order) = level.remove_order(order_id) {
                    // Remove from orders map
                    self.state.orders.remove(&order_id);
                    
                    // Remove price level if empty
                    if level.is_empty() {
                        match side {
                            Side::Bid => self.state.bids.remove(&price),
                            Side::Ask => self.state.asks.remove(&price),
                        };
                    }
                    
                    self.state.last_update = current_timestamp();
                    return Ok(Some(order));
                }
            }
        }
        
        Ok(None)
    }

    /// Get the current best bid and ask prices (BBO - Best Bid/Offer)
    pub fn get_bbo(&self) -> Option<(Decimal, Decimal)> {
        let best_bid = self.state.bids.keys().next_back();
        let best_ask = self.state.asks.keys().next();
        
        match (best_bid, best_ask) {
            (Some(&bid), Some(&ask)) => Some((bid, ask)),
            _ => None,
        }
    }

    /// Get the current spread (ask - bid)
    pub fn get_spread(&self) -> Option<Decimal> {
        self.get_bbo().map(|(bid, ask)| ask - bid)
    }

    /// Get the current mid price ((bid + ask) / 2)
    pub fn get_mid_price(&self) -> Option<Decimal> {
        self.get_bbo().map(|(bid, ask)| (bid + ask) / Decimal::from(2))
    }

    /// Get the total volume at a specific price level
    pub fn get_volume_at_price(&self, price: Decimal, side: Side) -> f64 {
        match side {
            Side::Bid => self.state.bids.get(&price).map_or(0.0, |l| l.total_quantity),
            Side::Ask => self.state.asks.get(&price).map_or(0.0, |l| l.total_quantity),
        }
    }

    /// Get the order book depth up to a certain number of levels
    pub fn get_depth(&self, levels: usize) -> (Vec<(Decimal, f64)>, Vec<(Decimal, f64)>) {
        let bids = self.state.bids
            .iter()
            .rev()
            .take(levels)
            .map(|(&price, level)| (price, level.total_quantity))
            .collect();
            
        let asks = self.state.asks
            .iter()
            .take(levels)
            .map(|(&price, level)| (price, level.total_quantity))
            .collect();
            
        (bids, asks)
    }

    // Private helper methods
    
    fn validate_order(&self, order: &Order) -> LOBResult<()> {
        // Check if order quantity is valid
        if order.quantity <= 0.0 {
            return Err(LOBError::InvalidQuantity(order.quantity));
        }

        // Check if price is valid for limit orders
        if order.order_type == OrderType::Limit && order.price <= Decimal::ZERO {
            return Err(LOBError::InvalidPrice(order.price.to_f64().unwrap_or(0.0)));
        }

        // Check order count limits
        let side_count = match order.side {
            Side::Bid => self.state.bids.values().map(|l| l.order_count).sum(),
            Side::Ask => self.state.asks.values().map(|l| l.order_count).sum(),
        };
        
        if side_count >= self.max_orders_per_side {
            return Err(LOBError::OrderBookFull);
        }

        // Check price level limits
        let price_levels = match order.side {
            Side::Bid => self.state.bids.len(),
            Side::Ask => self.state.asks.len(),
        };
        
        if price_levels >= self.max_price_levels {
            return Err(LOBError::PriceLevelLimitReached);
        }

        Ok(())
    }

    fn process_market_order(&mut self, order: Order) -> LOBResult<Vec<Trade>> {
        let mut trades = Vec::new();
        let mut remaining_quantity = order.quantity;
        
        match order.side {
            Side::Buy => {
                // Match with best asks
                while remaining_quantity > 0.0 && !self.state.asks.is_empty() {
                    let (price, level) = self.state.asks.first_key_value().unwrap();
                    let price = *price;
                    
                    if let Some(mut level) = self.state.asks.remove(&price) {
                        while let Some(mut maker_order) = level.orders.pop_front() {
                            let trade_quantity = maker_order.quantity.min(remaining_quantity);
                            
                            // Create trade
                            let trade = Trade {
                                taker_order_id: order.id,
                                maker_order_id: maker_order.id,
                                price,
                                quantity: trade_quantity,
                                timestamp: current_timestamp(),
                                is_buyer_maker: false,
                            };
                            
                            trades.push(trade.clone());
                            self.last_trade = Some(trade);
                            self.trade_history.push_back(trade);
                            
                            // Update quantities
                            remaining_quantity -= trade_quantity;
                            maker_order.quantity -= trade_quantity;
                            
                            // Update or remove maker order
                            if maker_order.quantity > 0.0 {
                                level.orders.push_front(maker_order);
                                self.state.asks.insert(price, level);
                                break;
                            } else {
                                self.state.orders.remove(&maker_order.id);
                            }
                            
                            if remaining_quantity <= 0.0 {
                                break;
                            }
                        }
                        
                        // If there are remaining orders at this price level, put it back
                        if !level.orders.is_empty() {
                            self.state.asks.insert(price, level);
                        }
                    }
                    
                    if remaining_quantity <= 0.0 {
                        break;
                    }
                }
                
                // If there's remaining quantity and it's an IOC or FOK order, reject the rest
                if remaining_quantity > 0.0 && order.time_in_force.is_immediate_or_cancel() {
                    return Ok(trades);
                }
            }
            
            Side::Sell => {
                // Similar logic for sell orders, matching with best bids
                // Implementation omitted for brevity
                // ...
            }
        }
        
        self.state.last_update = current_timestamp();
        Ok(trades)
    }
    
    fn process_limit_order(&mut self, order: Order) -> LOBResult<Vec<Trade>> {
        let mut trades = Vec::new();
        let mut remaining_quantity = order.quantity;
        
        // Try to match the order with the opposite side of the book
        match order.side {
            Side::Buy => {
                while remaining_quantity > 0.0 && !self.state.asks.is_empty() {
                    let (best_ask, _) = self.state.asks.first_key_value().unwrap();
                    
                    // Check if the order can be matched
                    if *best_ask > order.price {
                        break;
                    }
                    
                    // Process matching
                    // Similar to market order matching logic
                    // ...
                    
                    break; // Simplified for brevity
                }
            }
            
            Side::Sell => {
                // Similar logic for sell orders
                // ...
            }
        }
        
        // If there's remaining quantity, add to the order book
        if remaining_quantity > 0.0 && !order.time_in_force.is_immediate_or_cancel() {
            self.add_to_order_book(order, remaining_quantity)?;
        }
        
        self.state.last_update = current_timestamp();
        Ok(trades)
    }
    
    fn add_to_order_book(&mut self, order: Order, quantity: f64) -> LOBResult<()> {
        let price = order.price;
        let side = order.side;
        
        // Get the appropriate price level or create a new one
        let price_level = match side {
            Side::Bid => self.state.bids.entry(price).or_insert_with(|| PriceLevel::new(price)),
            Side::Ask => self.state.asks.entry(price).or_insert_with(|| PriceLevel::new(price)),
        };
        
        // Add the order to the price level
        let mut order = order;
        order.quantity = quantity;
        price_level.add_order(order.clone());
        
        // Add to orders map
        self.state.orders.insert(order.id, order);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_new_order_book() {
        let lob = LimitOrderBook::new(dec!(0.01), dec!(0.001)).unwrap();
        assert_eq!(lob.tick_size, dec!(0.01));
        assert_eq!(lob.lot_size, dec!(0.001));
    }
    
    #[test]
    fn test_add_limit_order() {
        let mut lob = LimitOrderBook::new(dec!(0.01), dec!(0.001)).unwrap();
        let order = Order {
            id: 1,
            side: Side::Bid,
            order_type: OrderType::Limit,
            price: dec!(100.0),
            quantity: 1.0,
            filled_quantity: 0.0,
            status: OrderStatus::New,
            timestamp: 0,
            user_id: None,
            client_order_id: None,
            time_in_force: TimeInForce::GTC,
            post_only: false,
            reduce_only: false,
            metadata: Default::default(),
        };
        
        let trades = lob.add_order(order).unwrap();
        assert!(trades.is_empty());
        assert_eq!(lob.state.bids.len(), 1);
        assert_eq!(lob.state.orders.len(), 1);
    }
    
    // More test cases would be added here...
}