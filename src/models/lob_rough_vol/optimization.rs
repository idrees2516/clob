use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use rust_decimal::Decimal;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use thiserror::Error;

use crate::models::lob_rough_vol::{
    LOBError, LOBResult, Order, OrderId, Price, Side, Trade
};

/// Error type for optimization and security related issues
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum OptimizationError {
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    
    #[error("Order size exceeds limit: {0} > {1}")]
    OrderSizeLimitExceeded(u64, u64),
    
    #[error("Price deviation too large: {0}% > {1}%")]
    PriceDeviationTooLarge(f64, f64),
    
    #[error("Circuit breaker triggered: {0}")]
    CircuitBreaker(String),
}

impl From<OptimizationError> for LOBError {
    fn from(err: OptimizationError) -> Self {
        LOBError::OptimizationError(err.to_string())
    }
}

/// Configuration for rate limiting
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of orders per second
    pub orders_per_second: u32,
    /// Maximum number of orders per price level
    pub orders_per_price_level: usize,
    /// Maximum order size in base currency units
    pub max_order_size: u64,
    /// Maximum price deviation percentage
    pub max_price_deviation_pct: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            orders_per_second: 1000,
            orders_per_price_level: 100,
            max_order_size: 10_000,
            max_price_deviation_pct: 5.0, // 5% deviation from mid price
        }
    }
}

/// Circuit breaker state
#[derive(Debug)]
pub struct CircuitBreaker {
    is_triggered: AtomicBool,
    last_trigger_time: RwLock<Option<Instant>>,
    cooldown_period: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given cooldown period
    pub fn new(cooldown_period: Duration) -> Self {
        Self {
            is_triggered: AtomicBool::new(false),
            last_trigger_time: RwLock::new(None),
            cooldown_period,
        }
    }
    
    /// Check if the circuit breaker is triggered
    pub fn is_triggered(&self) -> bool {
        self.is_triggered.load(Ordering::SeqCst)
    }
    
    /// Trigger the circuit breaker
    pub fn trigger(&self) {
        if !self.is_triggered.swap(true, Ordering::SeqCst) {
            *self.last_trigger_time.write() = Some(Instant::now());
        }
    }
    
    /// Reset the circuit breaker if the cooldown period has passed
    pub fn try_reset(&self) -> bool {
        if !self.is_triggered() {
            return true;
        }
        
        if let Some(trigger_time) = *self.last_trigger_time.read() {
            if trigger_time.elapsed() >= self.cooldown_period {
                self.is_triggered.store(false, Ordering::SeqCst);
                *self.last_trigger_time.write() = None;
                return true;
            }
        }
        
        false
    }
}

/// Rate limiter for order operations
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    circuit_breaker: CircuitBreaker,
    last_order_times: RwLock<Vec<Instant>>,
    order_counters: parking_lot::RwLock<lru::LruCache<OrderId, u32>>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            circuit_breaker: CircuitBreaker::new(Duration::from_secs(60)), // 1 minute cooldown
            last_order_times: RwLock::new(Vec::with_capacity(1000)),
            order_counters: parking_lot::RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(10_000).unwrap()
            )),
        }
    }
    
    /// Advanced: check order against all risk, liquidity, and microstructure constraints
    pub fn check_order(&self, order: &Order, mid_price: Option<Decimal>, liquidity_risk: f64) -> LOBResult<()> {
        if self.circuit_breaker.is_triggered() {
            return Err(OptimizationError::CircuitBreaker(
                "Trading is temporarily suspended due to high volatility".to_string()
            ).into());
        }
        if order.quantity > self.config.max_order_size {
            return Err(OptimizationError::OrderSizeLimitExceeded(
                order.quantity,
                self.config.max_order_size
            ).into());
        }
        if liquidity_risk > 0.8 {
            return Err(OptimizationError::OrderSizeLimitExceeded(
                order.quantity,
                self.config.max_order_size
            ).into());
        }
        if let (Some(price), Some(mid)) = (order.price, mid_price) {
            let deviation_pct = (price - mid).abs() / mid * Decimal::from(100);
            if deviation_pct > Decimal::from_f64(self.config.max_price_deviation_pct).unwrap() {
                return Err(OptimizationError::PriceDeviationTooLarge(
                    deviation_pct.to_f64().unwrap_or(f64::MAX),
                    self.config.max_price_deviation_pct
                ).into());
            }
        }
        self.check_rate_limits()?;
        Ok(())
    }
    
    /// Update the rate limiter with a new order
    pub fn update_order(&self, order_id: OrderId) {
        let now = Instant::now();
        
        // Update order times
        {
            let mut times = self.last_order_times.write();
            times.push(now);
            
            // Remove timestamps older than 1 second
            let one_second_ago = now - Duration::from_secs(1);
            times.retain(|&t| t >= one_second_ago);
        }
        
        // Update order counter
        {
            let mut counters = self.order_counters.write();
            let count = counters.get_mut(&order_id).map(|c| *c).unwrap_or(0);
            counters.put(order_id, count + 1);
        }
    }
    
    /// Check if rate limits are being exceeded
    fn check_rate_limits(&self) -> LOBResult<()> {
        // Check orders per second
        let one_second_ago = Instant::now() - Duration::from_secs(1);
        let recent_orders = self.last_order_times
            .read()
            .iter()
            .filter(|&&t| t >= one_second_ago)
            .count();
            
        if recent_orders >= self.config.orders_per_second as usize {
            return Err(OptimizationError::RateLimitExceeded(
                format!("Exceeded {} orders per second", self.config.orders_per_second)
            ).into());
        }
        
        // Check for excessive order cancellation
        // This is a simplified check - in practice, you'd want more sophisticated detection
        let recent_cancellations = self.order_counters
            .read()
            .iter()
            .filter(|(_, &count)| count > 10) // More than 10 modifications
            .count();
            
        if recent_cancellations > 50 { // More than 50 orders with >10 modifications
            self.circuit_breaker.trigger();
            return Err(OptimizationError::CircuitBreaker(
                "Excessive order modifications detected".to_string()
            ).into());
        }
        
        Ok(())
    }
}

/// Performance optimization: Pre-allocated order book levels
#[derive(Debug)]
pub struct OrderBookLevels {
    pub bids: Vec<(Price, u64)>, // (price, total_quantity)
    pub asks: Vec<(Price, u64)>,
    pub last_update: Instant,
}

impl OrderBookLevels {
    /// Create a new pre-allocated order book
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bids: Vec::with_capacity(capacity),
            asks: Vec::with_capacity(capacity),
            last_update: Instant::now(),
        }
    }
    
    /// Update the order book levels
    pub fn update(&mut self, bids: Vec<(Price, u64)>, asks: Vec<(Price, u64)>) {
        self.bids = bids;
        self.asks = asks;
        self.last_update = Instant::now();
    }
}

/// Performance optimization: Batch order processing
pub struct OrderBatch {
    orders: Vec<Order>,
    max_batch_size: usize,
}

impl OrderBatch {
    /// Create a new order batch with the given maximum size
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            orders: Vec::with_capacity(max_batch_size),
            max_batch_size,
        }
    }
    
    /// Add an order to the batch
    pub fn add_order(&mut self, order: Order) -> LOBResult<()> {
        if self.orders.len() >= self.max_batch_size {
            return Err(LOBError::OptimizationError(
                format!("Batch size limit of {} orders exceeded", self.max_batch_size)
            ));
        }
        
        self.orders.push(order);
        Ok(())
    }
    
    /// Process all orders in the batch
    pub fn process<F>(self, mut processor: F) -> LOBResult<Vec<Trade>>
    where
        F: FnMut(Order) -> LOBResult<Option<Trade>>,
    {
        let mut trades = Vec::with_capacity(self.orders.len());
        
        for order in self.orders {
            if let Some(trade) = processor(order)? {
                trades.push(trade);
            }
        }
        
        Ok(trades)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_rate_limiter() {
        let config = RateLimitConfig {
            orders_per_second: 2,
            orders_per_price_level: 10,
            max_order_size: 1000,
            max_price_deviation_pct: 10.0,
        };
        
        let limiter = RateLimiter::new(config);
        let order = Order::limit(Side::Bid, dec!(100.0), 100, None, None, None).unwrap();
        
        // First two orders should be allowed
        assert!(limiter.check_order(&order, Some(dec!(100.0)), 0.0).is_ok());
        limiter.update_order(1);
        
        assert!(limiter.check_order(&order, Some(dec!(100.0)), 0.0).is_ok());
        limiter.update_order(2);
        
        // Third order in the same second should be rejected
        assert!(matches!(
            limiter.check_order(&order, Some(dec!(100.0)), 0.0),
            Err(LOBError::OptimizationError(OptimizationError::RateLimitExceeded(_)))
        ));
    }
    
    #[test]
    fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(Duration::from_millis(100));
        assert!(!breaker.is_triggered());
        
        breaker.trigger();
        assert!(breaker.is_triggered());
        
        // Should still be triggered before cooldown
        std::thread::sleep(Duration::from_millis(50));
        assert!(breaker.is_triggered());
        
        // Should reset after cooldown
        std::thread::sleep(Duration::from_millis(60));
        assert!(!breaker.is_triggered());
    }
}
