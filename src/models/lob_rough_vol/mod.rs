//! Limit Order Book (LOB) model with rough volatility
//! Implementation based on "A Limit Order Book Model for High Frequency trading with rough volatility"

#![allow(clippy::module_inception)]
#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

pub use book::LimitOrderBook;
pub use matching_engine::{MatchingEngine, MarketDataEvent, OrderEvent, MatchingError};
pub use optimization::{RateLimitConfig, RateLimiter, CircuitBreaker, OrderBookLevels, OrderBatch, OptimizationError};
pub use order::{Order, OrderType, Side, TimeInForce, OrderStatus};
pub use price::{Price, PriceLevel, TickSizeTable};
pub use volatility::{RoughVolatilityModel, RoughVolatilityParams, VolatilityError};
pub mod book;
pub mod matching_engine;
pub mod optimization;
pub mod order;
pub mod price;
pub mod volatility;
pub mod simulation;
pub mod metrics;

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// Custom error type for LOB operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum LOBError {
    #[error("Order not found: {0}")]
    OrderNotFound(u64),
    #[error("Insufficient quantity: available {available}, requested {requested}")]
    InsufficientQuantity { available: f64, requested: f64 },
    #[error("Invalid price: {0}")]
    InvalidPrice(f64),
    #[error("Invalid quantity: {0}")]
    InvalidQuantity(f64),
    #[error("Order book is empty")]
    EmptyBook,
    #[error("Spread too wide: {0}")]
    SpreadTooWide(f64),
    #[error("Volatility error: {0}")]
    VolatilityError(String),
}

/// Result type for LOB operations
pub type LOBResult<T> = Result<T, LOBError>;

/// Order side (Bid/Ask)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    pub fn opposite(&self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

/// Order type (Limit/Market)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    Limit,
    Market,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Unique order identifier
type OrderId = u64;

/// Price type (using Decimal for precise decimal arithmetic)
type Price = Decimal;

/// Quantity type
type Quantity = f64;

/// Timestamp in nanoseconds since UNIX epoch
type Timestamp = u128;

/// Generate a unique order ID
fn generate_order_id() -> OrderId {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Get current timestamp in nanoseconds
fn current_timestamp() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_nanos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_side_opposite() {
        assert_eq!(Side::Bid.opposite(), Side::Ask);
        assert_eq!(Side::Ask.opposite(), Side::Bid);
    }

    #[test]
    fn test_order_id_generation() {
        let id1 = generate_order_id();
        let id2 = generate_order_id();
        assert_ne!(id1, id2);
    }
}
