//! Deterministic matching engine for the CLOB
//!
//! Implements price-time priority matching algorithm with:
//! - Partial fill handling with remaining quantity tracking
//! - Market order execution against best available prices  
//! - Limit order placement with immediate matching check
//! - Deterministic execution for zkVM compatibility
//!
//! This engine ensures consistent, reproducible matching behavior
//! across different execution environments.

use super::types::{
    Order, PriceLevel, Trade, OrderBook, OrderResult, OrderStatus, OrderType,
    OrderId, Symbol, Side, TimeInForce, OrderBookError
};
use std::cmp;

/// Error types for matching engine operations
#[derive(Debug, Clone, PartialEq)]
pub enum MatchingError {
    InvalidOrder(String),
    InvalidPrice,
    InvalidSize,
    OrderNotFound(u64),
    InsufficientLiquidity,
}

/// Deterministic matching engine for the CLOB
pub struct MatchingEngine {
    /// Current system timestamp for deterministic execution
    pub current_time: u64,
}

impl MatchingEngine {
    pub fn new() -> Self {
        Self {
            current_time: 0,
        }
    }

    /// Set current timestamp for deterministic execution
    pub fn set_current_time(&mut self, timestamp: u64) {
        self.current_time = timestamp;
    }

    /// Process a new order with deterministic matching
    /// Returns the order result with any trades generated
    pub fn process_order(&mut self, order_book: &mut OrderBook, mut order: Order) -> Result<OrderResult, MatchingError> {
        // Validate order
        order.validate().map_err(|e| MatchingError::InvalidOrder(e.to_string()))?;
        
        // Set timestamp if not provided
        if order.timestamp == 0 {
            order.timestamp = self.current_time;
        }

        let mut trades = Vec::new();
        let mut filled_size = 0u64;
        let original_size = order.size;

        match order.order_type {
            OrderType::Market => {
                // Market orders execute immediately against best available prices
                filled_size = self.execute_market_order(order_book, &mut order, &mut trades)?;
            }
            OrderType::Limit => {
                // Limit orders first try to match, then add remainder to book
                filled_size = self.execute_limit_order(order_book, &mut order, &mut trades)?;
            }
        }

        // Determine final status
        let status = if filled_size == 0 {
            if order.order_type == OrderType::Market {
                OrderStatus::Rejected // Market order with no fills is rejected
            } else {
                OrderStatus::Added // Limit order added to book
            }
        } else if filled_size == original_size {
            OrderStatus::FullyFilled
        } else {
            OrderStatus::PartiallyFilled
        };

        // Update sequence number for deterministic ordering
        order_book.sequence_number += 1;

        Ok(OrderResult {
            order_id: order.id.as_u64(),
            status,
            filled_size,
            remaining_size: original_size.saturating_sub(filled_size),
            trades,
        })
    }

    /// Execute market order against best available prices
    fn execute_market_order(&mut self, order_book: &mut OrderBook, order: &mut Order, trades: &mut Vec<Trade>) -> Result<u64, MatchingError> {
        let mut total_filled = 0u64;

        if order.side == Side::Buy {
            // Buy market order matches against asks (ascending price order)
            let ask_prices: Vec<u64> = order_book.asks.keys().copied().collect();
            
            for ask_price in ask_prices {
                if order.size == 0 {
                    break;
                }

                let filled = self.match_against_price_level(
                    order_book,
                    order,
                    ask_price,
                    false, // asks
                    trades,
                )?;
                
                total_filled += filled;
            }
        } else {
            // Sell market order matches against bids (descending price order)
            let bid_prices: Vec<u64> = order_book.bids.keys().rev().copied().collect();
            
            for bid_price in bid_prices {
                if order.size == 0 {
                    break;
                }

                let filled = self.match_against_price_level(
                    order_book,
                    order,
                    bid_price,
                    true, // bids
                    trades,
                )?;
                
                total_filled += filled;
            }
        }

        Ok(total_filled)
    }

    /// Execute limit order with immediate matching check, then add remainder to book
    fn execute_limit_order(&mut self, order_book: &mut OrderBook, order: &mut Order, trades: &mut Vec<Trade>) -> Result<u64, MatchingError> {
        let mut total_filled = 0u64;

        // First, try to match against existing orders
        if order.side == Side::Buy {
            // Buy limit order matches against asks at or below limit price
            let ask_prices: Vec<u64> = order_book.asks.keys()
                .filter(|&&price| price <= order.price)
                .copied()
                .collect();
            
            for ask_price in ask_prices {
                if order.size == 0 {
                    break;
                }

                let filled = self.match_against_price_level(
                    order_book,
                    order,
                    ask_price,
                    false, // asks
                    trades,
                )?;
                
                total_filled += filled;
            }
        } else {
            // Sell limit order matches against bids at or above limit price
            let bid_prices: Vec<u64> = order_book.bids.keys()
                .filter(|&&price| price >= order.price)
                .rev()
                .copied()
                .collect();
            
            for bid_price in bid_prices {
                if order.size == 0 {
                    break;
                }

                let filled = self.match_against_price_level(
                    order_book,
                    order,
                    bid_price,
                    true, // bids
                    trades,
                )?;
                
                total_filled += filled;
            }
        }

        // Add remaining quantity to book if any
        if order.size > 0 {
            self.add_order_to_book(order_book, order.clone())?;
        }

        Ok(total_filled)
    }

    /// Match order against a specific price level with price-time priority
    fn match_against_price_level(
        &mut self,
        order_book: &mut OrderBook,
        incoming_order: &mut Order,
        price: u64,
        is_bid_side: bool,
        trades: &mut Vec<Trade>,
    ) -> Result<u64, MatchingError> {
        let mut total_matched = 0u64;

        // Get the price level
        let price_level = if is_bid_side {
            order_book.bids.get_mut(&price)
        } else {
            order_book.asks.get_mut(&price)
        };

        let Some(level) = price_level else {
            return Ok(0);
        };

        // Process orders in FIFO order (price-time priority)
        while !level.orders.is_empty() && incoming_order.size > 0 {
            let resting_order = level.orders.front().unwrap().clone();
            
            // Calculate match size
            let match_size = cmp::min(incoming_order.size, resting_order.size);
            
            // Create trade
            let trade = Trade {
                id: order_book.next_trade_id,
                price,
                size: match_size,
                timestamp: self.current_time,
                buyer_order_id: if incoming_order.side == Side::Buy { incoming_order.id.as_u64() } else { resting_order.id.as_u64() },
                seller_order_id: if incoming_order.side == Side::Buy { resting_order.id.as_u64() } else { incoming_order.id.as_u64() },
                is_buyer_maker: incoming_order.side == Side::Sell, // Resting order is maker
            };

            trades.push(trade);
            order_book.next_trade_id += 1;
            
            // Update order sizes
            incoming_order.size -= match_size;
            total_matched += match_size;
            
            // Update resting order
            if resting_order.size == match_size {
                // Fully filled - remove from book
                level.orders.pop_front();
                level.total_size -= match_size;
                level.order_count -= 1;
                order_book.orders.remove(&resting_order.id);
            } else {
                // Partially filled - update size
                if let Some(front_order) = level.orders.front_mut() {
                    front_order.size -= match_size;
                    level.total_size -= match_size;
                    
                    // Update in orders map
                    if let Some(stored_order) = order_book.orders.get_mut(&resting_order.id) {
                        stored_order.size -= match_size;
                    }
                }
            }

            // Update last trade info
            order_book.last_trade_price = Some(price);
            order_book.last_trade_time = self.current_time;
        }

        // Remove empty price level
        if level.is_empty() {
            if is_bid_side {
                order_book.bids.remove(&price);
            } else {
                order_book.asks.remove(&price);
            }
        }

        Ok(total_matched)
    }

    /// Add order to the appropriate side of the book
    fn add_order_to_book(&self, order_book: &mut OrderBook, order: Order) -> Result<(), MatchingError> {
        // Add to orders map
        order_book.orders.insert(order.id, order.clone());

        // Add to appropriate price level
        if order.side == Side::Buy {
            let level = order_book.bids.entry(order.price).or_insert_with(|| PriceLevel::new(order.price));
            level.add_order(order);
        } else {
            let level = order_book.asks.entry(order.price).or_insert_with(|| PriceLevel::new(order.price));
            level.add_order(order);
        }

        Ok(())
    }

    /// Cancel an existing order from the book
    pub fn cancel_order(&mut self, order_book: &mut OrderBook, order_id: OrderId) -> Result<Order, MatchingError> {
        // Find and remove from orders map
        let order = order_book.orders.remove(&order_id)
            .ok_or(MatchingError::OrderNotFound(order_id.as_u64()))?;

        // Remove from appropriate price level
        let removed = if order.side == Side::Buy {
            if let Some(level) = order_book.bids.get_mut(&order.price) {
                let removed = level.remove_order(order_id.as_u64());
                if level.is_empty() {
                    order_book.bids.remove(&order.price);
                }
                removed
            } else {
                None
            }
        } else {
            if let Some(level) = order_book.asks.get_mut(&order.price) {
                let removed = level.remove_order(order_id.as_u64());
                if level.is_empty() {
                    order_book.asks.remove(&order.price);
                }
                removed
            } else {
                None
            }
        };

        if removed.is_none() {
            return Err(MatchingError::OrderNotFound(order_id.as_u64()));
        }

        // Update sequence number
        order_book.sequence_number += 1;

        Ok(order)
    }

    /// Additional validation for matching engine specific requirements
    fn validate_order_for_matching(&self, order: &Order) -> Result<(), MatchingError> {
        // Market orders can have price 0, but limit orders must have positive price
        if order.order_type == OrderType::Limit && order.price == 0 {
            return Err(MatchingError::InvalidPrice);
        }
        
        if order.size == 0 {
            return Err(MatchingError::InvalidSize);
        }

        Ok(())
    }
}

impl Default for MatchingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_order_matching() {
        let mut engine = MatchingEngine::new();
        engine.set_current_time(1000);
        
        let mut order_book = OrderBook::new("BTCUSD".to_string());

        // Add a sell order to the book
        let sell_order = Order::new_limit(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50000,
            100,
            1000,
        );

        let result = engine.process_order(&mut order_book, sell_order).unwrap();
        assert_eq!(result.status, OrderStatus::Added);
        assert_eq!(result.filled_size, 0);

        // Add a buy order that should match
        let buy_order = Order::new_limit(
            OrderId::new(2),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            50000,
            50,
            1001,
        );

        let result = engine.process_order(&mut order_book, buy_order).unwrap();
        assert_eq!(result.status, OrderStatus::FullyFilled);
        assert_eq!(result.filled_size, 50);
        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].price, 50000);
        assert_eq!(result.trades[0].size, 50);
    }

    #[test]
    fn test_market_order_execution() {
        let mut engine = MatchingEngine::new();
        engine.set_current_time(1000);
        
        let mut order_book = OrderBook::new("BTCUSD".to_string());

        // Add multiple sell orders at different prices
        let sell_order1 = Order::new_limit(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50000,
            50,
            1000,
        );
        let sell_order2 = Order::new_limit(
            OrderId::new(2),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50100,
            75,
            1001,
        );

        engine.process_order(&mut order_book, sell_order1).unwrap();
        engine.process_order(&mut order_book, sell_order2).unwrap();

        // Market buy order should match against best prices
        let market_buy = Order::new_market(
            OrderId::new(3),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            100,
            1002,
        );

        let result = engine.process_order(&mut order_book, market_buy).unwrap();
        assert_eq!(result.status, OrderStatus::FullyFilled);
        assert_eq!(result.filled_size, 100);
        assert_eq!(result.trades.len(), 2);
        
        // Should match at 50000 first (better price)
        assert_eq!(result.trades[0].price, 50000);
        assert_eq!(result.trades[0].size, 50);
        
        // Then at 50100
        assert_eq!(result.trades[1].price, 50100);
        assert_eq!(result.trades[1].size, 50);
    }

    #[test]
    fn test_partial_fill_handling() {
        let mut engine = MatchingEngine::new();
        engine.set_current_time(1000);
        
        let mut order_book = OrderBook::new("BTCUSD".to_string());

        // Add a small sell order
        let sell_order = Order::new_limit(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50000,
            30,
            1000,
        );

        engine.process_order(&mut order_book, sell_order).unwrap();

        // Large buy order should be partially filled
        let buy_order = Order::new_limit(
            OrderId::new(2),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            50000,
            100,
            1001,
        );

        let result = engine.process_order(&mut order_book, buy_order).unwrap();
        assert_eq!(result.status, OrderStatus::PartiallyFilled);
        assert_eq!(result.filled_size, 30);
        assert_eq!(result.remaining_size, 70);
        assert_eq!(result.trades.len(), 1);
        
        // Remaining 70 should be added to book
        assert_eq!(order_book.get_best_bid(), Some(50000));
    }

    #[test]
    fn test_price_time_priority() {
        let mut engine = MatchingEngine::new();
        
        let mut order_book = OrderBook::new("BTCUSD".to_string());

        // Add two sell orders at same price, different times
        engine.set_current_time(1000);
        let sell_order1 = Order::new_limit(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50000,
            50,
            1000,
        );
        engine.process_order(&mut order_book, sell_order1).unwrap();

        engine.set_current_time(1001);
        let sell_order2 = Order::new_limit(
            OrderId::new(2),
            Symbol::new("BTCUSD").unwrap(),
            Side::Sell,
            50000,
            50,
            1001,
        );
        engine.process_order(&mut order_book, sell_order2).unwrap();

        // Buy order should match first order first (time priority)
        engine.set_current_time(1002);
        let buy_order = Order::new_limit(
            OrderId::new(3),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            50000,
            50,
            1002,
        );

        let result = engine.process_order(&mut order_book, buy_order).unwrap();
        assert_eq!(result.trades[0].seller_order_id, 1); // First order matched first
    }

    #[test]
    fn test_order_cancellation() {
        let mut engine = MatchingEngine::new();
        engine.set_current_time(1000);
        
        let mut order_book = OrderBook::new("BTCUSD".to_string());

        // Add an order
        let order = Order::new_limit(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            50000,
            100,
            1000,
        );

        engine.process_order(&mut order_book, order.clone()).unwrap();
        assert_eq!(order_book.get_best_bid(), Some(50000));

        // Cancel the order
        let cancelled = engine.cancel_order(&mut order_book, OrderId::new(1)).unwrap();
        assert_eq!(cancelled.id, OrderId::new(1));
        assert_eq!(order_book.get_best_bid(), None);
    }
} 