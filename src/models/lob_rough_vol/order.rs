use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use super::{
    generate_order_id, current_timestamp, LOBResult, LOBError, OrderId, Price, Quantity, Side, 
    OrderStatus, OrderType, Timestamp
};

/// Represents a single order in the limit order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Price,
    pub quantity: Quantity,
    pub filled_quantity: Quantity,
    pub status: OrderStatus,
    pub timestamp: Timestamp,
    pub user_id: Option<String>,
    pub client_order_id: Option<String>,
    pub time_in_force: TimeInForce,
    pub post_only: bool,
    pub reduce_only: bool,
    pub metadata: HashMap<String, String>,
}

/// Time in force for orders
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good Till Cancelled
    GTC,
    /// Immediate or Cancel
    IOC,
    /// Fill or Kill
    FOK,
    /// Good Till Time (milliseconds)
    GTT(u64),
}

impl Default for TimeInForce {
    fn default() -> Self {
        Self::GTC
    }
}

impl Order {
    /// Create a new order
    pub fn new(
        side: Side,
        order_type: OrderType,
        price: Price,
        quantity: Quantity,
        user_id: Option<String>,
        client_order_id: Option<String>,
        time_in_force: Option<TimeInForce>,
        post_only: bool,
        reduce_only: bool,
    ) -> LOBResult<Self> {
        if quantity <= 0.0 {
            return Err(LOBError::InvalidQuantity(quantity));
        }

        if order_type == OrderType::Limit && price <= Decimal::ZERO {
            return Err(LOBError::InvalidPrice(price.to_f64().unwrap_or(0.0)));
        }

        Ok(Self {
            id: generate_order_id(),
            side,
            order_type,
            price,
            quantity,
            filled_quantity: 0.0,
            status: OrderStatus::New,
            timestamp: current_timestamp(),
            user_id,
            client_order_id,
            time_in_force: time_in_force.unwrap_or_default(),
            post_only,
            reduce_only,
            metadata: HashMap::new(),
        })
    }

    /// Check if the order is fully filled
    pub fn is_fully_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }

    /// Check if the order is active (can be matched)
    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::New | OrderStatus::PartiallyFilled)
    }

    /// Get the remaining quantity to be filled
    pub fn remaining_quantity(&self) -> Quantity {
        self.quantity - self.filled_quantity
    }

    /// Update the filled quantity
    pub fn fill(&mut self, quantity: Quantity) -> LOBResult<()> {
        if quantity <= 0.0 || quantity > self.remaining_quantity() {
            return Err(LOBError::InvalidQuantity(quantity));
        }

        self.filled_quantity += quantity;
        
        if (self.filled_quantity - self.quantity).abs() < f64::EPSILON {
            self.status = OrderStatus::Filled;
        } else if self.filled_quantity > 0.0 {
            self.status = OrderStatus::PartiallyFilled;
        }

        Ok(())
    }

    /// Cancel the order
    pub fn cancel(&mut self) -> LOBResult<()> {
        if !self.is_active() {
            return Err(LOBError::OrderNotFound(self.id));
        }
        self.status = OrderStatus::Cancelled;
        Ok(())
    }

    /// Reject the order
    pub fn reject(&mut self) {
        self.status = OrderStatus::Rejected;
    }

    /// Check if the order has expired
    pub fn is_expired(&self, current_time: Option<Timestamp>) -> bool {
        match self.time_in_force {
            TimeInForce::GTT(ttl) => {
                let current = current_time.unwrap_or_else(current_timestamp);
                current > self.timestamp + (ttl as u128) * 1_000_000 // Convert ms to ns
            }
            _ => false,
        }
    }
}

/// Builder for creating orders with a fluent interface
#[derive(Debug, Clone)]
pub struct OrderBuilder {
    side: Side,
    order_type: OrderType,
    price: Decimal,
    quantity: Quantity,
    user_id: Option<String>,
    client_order_id: Option<String>,
    time_in_force: Option<TimeInForce>,
    post_only: bool,
    reduce_only: bool,
}

impl OrderBuilder {
    /// Create a new order builder
    pub fn new(side: Side, order_type: OrderType, price: Decimal, quantity: Quantity) -> Self {
        Self {
            side,
            order_type,
            price,
            quantity,
            user_id: None,
            client_order_id: None,
            time_in_force: None,
            post_only: false,
            reduce_only: false,
        }
    }

    /// Set the user ID
    pub fn user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set the client order ID
    pub fn client_order_id(mut self, client_order_id: impl Into<String>) -> Self {
        self.client_order_id = Some(client_order_id.into());
        self
    }

    /// Set the time in force
    pub fn time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = Some(tif);
        self
    }

    /// Set post-only flag
    pub fn post_only(mut self, post_only: bool) -> Self {
        self.post_only = post_only;
        self
    }

    /// Set reduce-only flag
    pub fn reduce_only(mut self, reduce_only: bool) -> Self {
        self.reduce_only = reduce_only;
        self
    }

    /// Build the order
    pub fn build(self) -> LOBResult<Order> {
        Order::new(
            self.side,
            self.order_type,
            self.price,
            self.quantity,
            self.user_id,
            self.client_order_id,
            self.time_in_force,
            self.post_only,
            self.reduce_only,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            Side::Bid,
            OrderType::Limit,
            dec!(100.0),
            1.0,
            Some("user1".to_string()),
            Some("client1".to_string()),
            Some(TimeInForce::GTC),
            false,
            false,
        ).unwrap();

        assert_eq!(order.side, Side::Bid);
        assert_eq!(order.order_type, OrderType::Limit);
        assert_eq!(order.price, dec!(100.0));
        assert_eq!(order.quantity, 1.0);
        assert_eq!(order.filled_quantity, 0.0);
        assert_eq!(order.status, OrderStatus::New);
        assert!(order.user_id.is_some());
        assert!(order.client_order_id.is_some());
    }

    #[test]
    fn test_order_filling() {
        let mut order = Order::new(
            Side::Bid,
            OrderType::Limit,
            dec!(100.0),
            2.0,
            None,
            None,
            None,
            false,
            false,
        ).unwrap();

        assert!(order.fill(1.0).is_ok());
        assert_eq!(order.filled_quantity, 1.0);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        assert!(order.fill(1.0).is_ok());
        assert_eq!(order.filled_quantity, 2.0);
        assert_eq!(order.status, OrderStatus::Filled);
        assert!(order.is_fully_filled());

        // Test overfill
        assert!(order.fill(0.1).is_err());
    }

    #[test]
    fn test_order_builder() {
        let order = OrderBuilder::new(Side::Ask, OrderType::Limit, dec!(200.0), 1.5)
            .user_id("user2")
            .client_order_id("client2")
            .time_in_force(TimeInForce::IOC)
            .post_only(true)
            .reduce_only(false)
            .build()
            .unwrap();

        assert_eq!(order.side, Side::Ask);
        assert_eq!(order.price, dec!(200.0));
        assert_eq!(order.quantity, 1.5);
        assert_eq!(order.time_in_force, TimeInForce::IOC);
        assert!(order.post_only);
        assert!(!order.reduce_only);
    }
}