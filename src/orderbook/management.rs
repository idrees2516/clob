//! Order Management System
//!
//! This module provides comprehensive order lifecycle management including:
//! - Order cancellation with book cleanup
//! - Order modification (cancel-replace) functionality  
//! - Order status tracking and lifecycle management
//! - Order validation and risk checks

use crate::orderbook::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Comprehensive error types for order management operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum OrderManagementError {
    #[error("Order book error: {0}")]
    OrderBookError(#[from] OrderBookError),
    #[error("Order not found: {0}")]
    OrderNotFound(OrderId),
    #[error("Invalid order modification: {0}")]
    InvalidModification(String),
    #[error("Order already cancelled: {0}")]
    OrderAlreadyCancelled(OrderId),
    #[error("Order already filled: {0}")]
    OrderAlreadyFilled(OrderId),
    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),
    #[error("Order validation failed: {0}")]
    ValidationFailed(String),
    #[error("Insufficient permissions for order {0}")]
    InsufficientPermissions(OrderId),
    #[error("Order modification not allowed in current state: {0:?}")]
    ModificationNotAllowed(OrderState),
}

/// Comprehensive order state tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderState {
    /// Order is pending validation
    Pending,
    /// Order is active in the order book
    Active,
    /// Order is partially filled
    PartiallyFilled { filled_size: u64, remaining_size: u64 },
    /// Order is completely filled
    Filled { filled_size: u64 },
    /// Order has been cancelled
    Cancelled { reason: CancellationReason },
    /// Order was rejected
    Rejected { reason: String },
    /// Order has expired
    Expired,
    /// Order is being modified
    Modifying,
}

/// Reasons for order cancellation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CancellationReason {
    /// User requested cancellation
    UserRequested,
    /// System cancelled due to risk limits
    RiskLimit,
    /// Order expired (GTD orders)
    Expired,
    /// Insufficient funds
    InsufficientFunds,
    /// Market closed
    MarketClosed,
    /// Emergency halt
    EmergencyHalt,
    /// Order modification (cancel-replace)
    Modification,
}

/// Order modification request
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderModification {
    /// Order ID to modify
    pub order_id: OrderId,
    /// New price (None to keep existing)
    pub new_price: Option<u64>,
    /// New size (None to keep existing)
    pub new_size: Option<u64>,
    /// New time-in-force (None to keep existing)
    pub new_time_in_force: Option<TimeInForce>,
    /// Timestamp of modification request
    pub timestamp: u64,
    /// Client modification ID for tracking
    pub modification_id: Option<String>,
}

/// Order submission result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderSubmissionResult {
    /// Order ID
    pub order_id: OrderId,
    /// Submission status
    pub status: OrderSubmissionStatus,
    /// Optional message
    pub message: Option<String>,
    /// Trades generated from this order
    pub trades: Vec<Trade>,
}

/// Order submission status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderSubmissionStatus {
    /// Order accepted and added to book
    Accepted,
    /// Order partially filled
    PartiallyFilled,
    /// Order completely filled
    Filled,
    /// Order rejected due to validation
    Rejected,
    /// Order rejected due to risk limits
    RejectedRisk,
}

/// Order cancellation result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderCancellationResult {
    /// Order ID that was cancelled
    pub order_id: OrderId,
    /// The cancelled order
    pub cancelled_order: Order,
    /// Reason for cancellation
    pub reason: CancellationReason,
    /// Cancellation timestamp
    pub timestamp: u64,
}

/// Order modification result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderModificationResult {
    /// Original order ID
    pub original_order_id: OrderId,
    /// New order ID (from replacement)
    pub new_order_id: OrderId,
    /// New order details
    pub new_order: Order,
    /// Cancellation result for original order
    pub cancellation_result: OrderCancellationResult,
    /// Submission result for new order
    pub submission_result: OrderSubmissionResult,
    /// Client modification ID
    pub modification_id: Option<String>,
}

/// Order lifecycle event for audit trail
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderEvent {
    /// Event ID
    pub id: u64,
    /// Order ID this event relates to
    pub order_id: OrderId,
    /// Type of event
    pub event_type: OrderEventType,
    /// Timestamp of event
    pub timestamp: u64,
    /// Additional event data
    pub data: OrderEventData,
    /// Sequence number for ordering
    pub sequence: u64,
}

/// Types of order events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderEventType {
    /// Order was created
    Created,
    /// Order was validated
    Validated,
    /// Order was added to book
    AddedToBook,
    /// Order was partially filled
    PartiallyFilled,
    /// Order was completely filled
    Filled,
    /// Order was cancelled
    Cancelled,
    /// Order was rejected
    Rejected,
    /// Order was modified
    Modified,
    /// Order expired
    Expired,
    /// Risk check performed
    RiskCheck,
}

/// Additional data for order events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderEventData {
    /// Order creation data
    Created { order: Order },
    /// Validation result
    Validated { success: bool, message: Option<String> },
    /// Trade execution data
    Trade { trade: Trade },
    /// Cancellation data
    Cancelled { reason: CancellationReason },
    /// Modification data
    Modified { old_order: Order, new_order: Order },
    /// Risk check data
    RiskCheck { passed: bool, limits_checked: Vec<String> },
    /// Generic message
    Message { text: String },
}

/// Risk limits for order validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskLimits {
    /// Maximum order size
    pub max_order_size: u64,
    /// Maximum order value (price * size)
    pub max_order_value: u64,
    /// Maximum number of orders per symbol
    pub max_orders_per_symbol: u32,
    /// Maximum total exposure
    pub max_total_exposure: u64,
    /// Minimum price increment
    pub min_price_increment: u64,
    /// Minimum order size
    pub min_order_size: u64,
    /// Maximum price deviation from mid (basis points)
    pub max_price_deviation_bps: u32,
    /// Maximum position size (long)
    pub max_long_position: i64,
    /// Maximum position size (short)
    pub max_short_position: i64,
    /// Maximum drawdown before halt (basis points)
    pub max_drawdown_bps: u32,
    /// Maximum orders per second
    pub max_orders_per_second: u32,
    /// Maximum daily loss limit
    pub max_daily_loss: u64,
    /// Minimum time between orders (microseconds)
    pub min_order_interval_us: u64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_order_size: 1000 * VOLUME_SCALE,      // 1000 units
            max_order_value: 1_000_000 * PRICE_SCALE, // $1M
            max_orders_per_symbol: 100,
            max_total_exposure: 10_000_000 * PRICE_SCALE, // $10M
            min_price_increment: PRICE_SCALE / 100,    // $0.01
            min_order_size: VOLUME_SCALE / 1000,       // 0.001 units
            max_price_deviation_bps: 1000,             // 10%
            max_long_position: 10000 * VOLUME_SCALE as i64,  // 10,000 units long
            max_short_position: -10000 * VOLUME_SCALE as i64, // 10,000 units short
            max_drawdown_bps: 500,                     // 5% max drawdown
            max_orders_per_second: 100,                // 100 orders/sec
            max_daily_loss: 100_000 * PRICE_SCALE,    // $100k daily loss limit
            min_order_interval_us: 1000,               // 1ms between orders
        }
    }
}

/// Position tracking for risk management
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Position {
    /// Current position size (positive = long, negative = short)
    pub size: i64,
    /// Average entry price
    pub avg_price: u64,
    /// Realized PnL
    pub realized_pnl: i64,
    /// Unrealized PnL (based on current mid price)
    pub unrealized_pnl: i64,
    /// Total volume traded
    pub total_volume: u64,
    /// Number of trades
    pub trade_count: u32,
    /// Last update timestamp
    pub last_update: u64,
}

/// Performance metrics for order management
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average order processing latency (nanoseconds)
    pub avg_processing_latency_ns: u64,
    /// Maximum processing latency (nanoseconds)
    pub max_processing_latency_ns: u64,
    /// Orders processed per second
    pub orders_per_second: f64,
    /// Fill rate (percentage of orders that get filled)
    pub fill_rate: f64,
    /// Average time to fill (nanoseconds)
    pub avg_time_to_fill_ns: u64,
    /// Risk check failures
    pub risk_check_failures: u32,
    /// Validation failures
    pub validation_failures: u32,
    /// Last performance update
    pub last_update: u64,
}

/// Order expiration manager
#[derive(Debug, Clone)]
pub struct OrderExpirationManager {
    /// Orders with expiration times
    pub expiring_orders: HashMap<OrderId, u64>,
    /// Last expiration check timestamp
    pub last_check: u64,
}

impl OrderExpirationManager {
    pub fn new() -> Self {
        Self {
            expiring_orders: HashMap::new(),
            last_check: 0,
        }
    }
    
    /// Add an order for expiration tracking
    pub fn add_order(&mut self, order_id: OrderId, expiration_time: u64) {
        self.expiring_orders.insert(order_id, expiration_time);
    }
    
    /// Remove an order from expiration tracking
    pub fn remove_order(&mut self, order_id: OrderId) {
        self.expiring_orders.remove(&order_id);
    }
    
    /// Get expired orders
    pub fn get_expired_orders(&mut self, current_time: u64) -> Vec<OrderId> {
        self.last_check = current_time;
        
        let expired: Vec<OrderId> = self.expiring_orders
            .iter()
            .filter(|(_, &expiration)| current_time >= expiration)
            .map(|(&order_id, _)| order_id)
            .collect();
        
        // Remove expired orders from tracking
        for &order_id in &expired {
            self.expiring_orders.remove(&order_id);
        }
        
        expired
    }
}

/// Rate limiter for order submission
#[derive(Debug, Clone)]
pub struct OrderRateLimiter {
    /// Order timestamps for rate limiting
    pub order_timestamps: VecDeque<u64>,
    /// Maximum orders per window
    pub max_orders: u32,
    /// Time window in nanoseconds
    pub window_ns: u64,
}

impl OrderRateLimiter {
    pub fn new(max_orders: u32, window_ns: u64) -> Self {
        Self {
            order_timestamps: VecDeque::new(),
            max_orders,
            window_ns,
        }
    }
    
    /// Check if order can be submitted
    pub fn can_submit_order(&mut self, current_time: u64) -> bool {
        // Remove old timestamps outside the window
        while let Some(&front_time) = self.order_timestamps.front() {
            if current_time - front_time > self.window_ns {
                self.order_timestamps.pop_front();
            } else {
                break;
            }
        }
        
        self.order_timestamps.len() < self.max_orders as usize
    }
    
    /// Record order submission
    pub fn record_order(&mut self, timestamp: u64) {
        self.order_timestamps.push_back(timestamp);
    }
}

/// Comprehensive order management system
pub struct OrderManager {
    /// Order book reference
    pub order_book: CentralLimitOrderBook,
    /// Order state tracking
    pub order_states: HashMap<OrderId, OrderState>,
    /// Order event history
    pub order_events: Vec<OrderEvent>,
    /// Risk limits
    pub risk_limits: RiskLimits,
    /// Next event ID
    pub next_event_id: AtomicU64,
    /// Event sequence number
    pub event_sequence: AtomicU64,
    /// Total orders processed
    pub total_orders_processed: AtomicU64,
    /// Active order count
    pub active_order_count: u32,
    /// Current position
    pub position: Position,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Order expiration manager
    pub expiration_manager: OrderExpirationManager,
    /// Rate limiter
    pub rate_limiter: OrderRateLimiter,
    /// Last order timestamp for interval checking
    pub last_order_timestamp: u64,
}

impl OrderManager {
    /// Create a new order manager
    pub fn new(symbol: Symbol, risk_limits: Option<RiskLimits>) -> Self {
        let limits = risk_limits.unwrap_or_default();
        Self {
            order_book: CentralLimitOrderBook::new(symbol),
            order_states: HashMap::new(),
            order_events: Vec::new(),
            risk_limits: limits.clone(),
            next_event_id: AtomicU64::new(1),
            event_sequence: AtomicU64::new(0),
            total_orders_processed: AtomicU64::new(0),
            active_order_count: 0,
            position: Position::default(),
            performance_metrics: PerformanceMetrics::default(),
            expiration_manager: OrderExpirationManager::new(),
            rate_limiter: OrderRateLimiter::new(
                limits.max_orders_per_second,
                1_000_000_000, // 1 second in nanoseconds
            ),
            last_order_timestamp: 0,
        }
    }
    
    /// Submit a new order with comprehensive validation and risk checks
    pub fn submit_order(&mut self, order: Order) -> Result<OrderSubmissionResult, OrderManagementError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Log order creation event
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: order.id,
            event_type: OrderEventType::Created,
            timestamp,
            data: OrderEventData::Created { order: order.clone() },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        // Set initial state
        self.order_states.insert(order.id, OrderState::Pending);
        
        // Perform comprehensive validation
        let validation_result = self.validate_order(&order);
        
        // Log validation event
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: order.id,
            event_type: OrderEventType::Validated,
            timestamp,
            data: OrderEventData::Validated {
                success: validation_result.is_ok(),
                message: validation_result.as_ref().err().map(|e| e.to_string()),
            },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        if let Err(e) = validation_result {
            // Mark as rejected
            self.order_states.insert(order.id, OrderState::Rejected { 
                reason: e.to_string() 
            });
            
            self.log_event(OrderEvent {
                id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
                order_id: order.id,
                event_type: OrderEventType::Rejected,
                timestamp,
                data: OrderEventData::Message { text: e.to_string() },
                sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
            });
            
            return Ok(OrderSubmissionResult {
                order_id: order.id,
                status: OrderSubmissionStatus::Rejected,
                message: Some(e.to_string()),
                trades: Vec::new(),
            });
        }
        
        // Perform risk checks
        let risk_check_result = self.perform_risk_checks(&order);
        
        // Log risk check event
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: order.id,
            event_type: OrderEventType::RiskCheck,
            timestamp,
            data: OrderEventData::RiskCheck {
                passed: risk_check_result.is_ok(),
                limits_checked: vec![
                    "max_order_size".to_string(),
                    "max_order_value".to_string(),
                    "price_deviation".to_string(),
                ],
            },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        if let Err(e) = risk_check_result {
            // Mark as rejected due to risk
            self.order_states.insert(order.id, OrderState::Rejected { 
                reason: format!("Risk check failed: {}", e) 
            });
            
            return Ok(OrderSubmissionResult {
                order_id: order.id,
                status: OrderSubmissionStatus::RejectedRisk,
                message: Some(e.to_string()),
                trades: Vec::new(),
            });
        }
        
        // Add order to book
        let trades = self.order_book.add_order(order.clone())
            .map_err(OrderManagementError::OrderBookError)?;
        
        // Update order state based on execution results
        let new_state = if trades.is_empty() {
            // Order added to book without fills
            self.active_order_count += 1;
            OrderState::Active
        } else {
            // Order had some fills
            let total_filled: u64 = trades.iter().map(|t| t.size).sum();
            if total_filled >= order.size {
                // Completely filled
                OrderState::Filled { filled_size: total_filled }
            } else {
                // Partially filled
                self.active_order_count += 1;
                OrderState::PartiallyFilled {
                    filled_size: total_filled,
                    remaining_size: order.size - total_filled,
                }
            }
        };
        
        self.order_states.insert(order.id, new_state.clone());
        
        // Log appropriate events
        if trades.is_empty() {
            self.log_event(OrderEvent {
                id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
                order_id: order.id,
                event_type: OrderEventType::AddedToBook,
                timestamp,
                data: OrderEventData::Message { text: "Order added to book".to_string() },
                sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
            });
        } else {
            // Log trade events
            for trade in &trades {
                let event_type = if matches!(new_state, OrderState::Filled { .. }) {
                    OrderEventType::Filled
                } else {
                    OrderEventType::PartiallyFilled
                };
                
                self.log_event(OrderEvent {
                    id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
                    order_id: order.id,
                    event_type,
                    timestamp,
                    data: OrderEventData::Trade { trade: trade.clone() },
                    sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
                });
            }
        }
        
        self.total_orders_processed.fetch_add(1, Ordering::SeqCst);
        
        let status = match new_state {
            OrderState::Active => OrderSubmissionStatus::Accepted,
            OrderState::PartiallyFilled { .. } => OrderSubmissionStatus::PartiallyFilled,
            OrderState::Filled { .. } => OrderSubmissionStatus::Filled,
            _ => OrderSubmissionStatus::Accepted,
        };
        
        Ok(OrderSubmissionResult {
            order_id: order.id,
            status,
            message: None,
            trades,
        })
    }
    
    /// Cancel an order with comprehensive cleanup
    pub fn cancel_order(&mut self, order_id: OrderId, reason: CancellationReason) -> Result<OrderCancellationResult, OrderManagementError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Check if order exists and can be cancelled
        let current_state = self.order_states.get(&order_id)
            .ok_or(OrderManagementError::OrderNotFound(order_id))?;
        
        match current_state {
            OrderState::Cancelled { .. } => {
                return Err(OrderManagementError::OrderAlreadyCancelled(order_id));
            }
            OrderState::Filled { .. } => {
                return Err(OrderManagementError::OrderAlreadyFilled(order_id));
            }
            OrderState::Rejected { .. } => {
                return Err(OrderManagementError::OrderNotFound(order_id));
            }
            _ => {} // Can be cancelled
        }
        
        // Remove from order book
        let cancelled_order = self.order_book.cancel_order(order_id)
            .map_err(OrderManagementError::OrderBookError)?;
        
        // Update state
        self.order_states.insert(order_id, OrderState::Cancelled { reason: reason.clone() });
        
        // Update active order count
        if matches!(current_state, OrderState::Active | OrderState::PartiallyFilled { .. }) {
            self.active_order_count = self.active_order_count.saturating_sub(1);
        }
        
        // Log cancellation event
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id,
            event_type: OrderEventType::Cancelled,
            timestamp,
            data: OrderEventData::Cancelled { reason: reason.clone() },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        Ok(OrderCancellationResult {
            order_id,
            cancelled_order,
            reason,
            timestamp,
        })
    }
    
    /// Modify an order using cancel-replace logic
    pub fn modify_order(&mut self, modification: OrderModification) -> Result<OrderModificationResult, OrderManagementError> {
        let timestamp = modification.timestamp;
        
        // Check if order exists and can be modified
        let current_state = self.order_states.get(&modification.order_id)
            .ok_or(OrderManagementError::OrderNotFound(modification.order_id))?;
        
        match current_state {
            OrderState::Cancelled { .. } => {
                return Err(OrderManagementError::OrderAlreadyCancelled(modification.order_id));
            }
            OrderState::Filled { .. } => {
                return Err(OrderManagementError::OrderAlreadyFilled(modification.order_id));
            }
            OrderState::Rejected { .. } => {
                return Err(OrderManagementError::OrderNotFound(modification.order_id));
            }
            OrderState::Modifying => {
                return Err(OrderManagementError::ModificationNotAllowed(current_state.clone()));
            }
            _ => {} // Can be modified
        }
        
        // Set modifying state
        self.order_states.insert(modification.order_id, OrderState::Modifying);
        
        // Get the current order
        let current_order = self.order_book.orders.get(&modification.order_id)
            .ok_or(OrderManagementError::OrderNotFound(modification.order_id))?
            .clone();
        
        // Create new order with modifications
        let mut new_order = current_order.clone();
        
        if let Some(new_price) = modification.new_price {
            new_order.price = new_price;
        }
        
        if let Some(new_size) = modification.new_size {
            new_order.size = new_size;
        }
        
        if let Some(new_tif) = modification.new_time_in_force {
            new_order.time_in_force = new_tif;
        }
        
        new_order.timestamp = timestamp;
        
        // Validate the modified order
        let validation_result = self.validate_order(&new_order);
        if let Err(e) = validation_result {
            // Restore original state
            self.order_states.insert(modification.order_id, current_state.clone());
            return Err(OrderManagementError::ValidationFailed(e.to_string()));
        }
        
        // Cancel the original order
        let cancellation_result = self.cancel_order(modification.order_id, CancellationReason::Modification)?;
        
        // Generate new order ID for the replacement
        let new_order_id = OrderId::new(
            self.total_orders_processed.load(Ordering::SeqCst) + 1000000 // Ensure uniqueness
        );
        new_order.id = new_order_id;
        
        // Submit the new order
        let submission_result = self.submit_order(new_order.clone())?;
        
        // Log modification event
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: modification.order_id,
            event_type: OrderEventType::Modified,
            timestamp,
            data: OrderEventData::Modified {
                old_order: current_order,
                new_order: new_order.clone(),
            },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        Ok(OrderModificationResult {
            original_order_id: modification.order_id,
            new_order_id,
            new_order,
            cancellation_result,
            submission_result,
            modification_id: modification.modification_id,
        })
    }
    
    /// Get current order state
    pub fn get_order_state(&self, order_id: OrderId) -> Option<&OrderState> {
        self.order_states.get(&order_id)
    }
    
    /// Get order events for a specific order
    pub fn get_order_events(&self, order_id: OrderId) -> Vec<&OrderEvent> {
        self.order_events.iter()
            .filter(|event| event.order_id == order_id)
            .collect()
    }
    
    /// Get all active orders
    pub fn get_active_orders(&self) -> Vec<(OrderId, &Order)> {
        self.order_book.orders.iter()
            .filter(|(order_id, _)| {
                matches!(
                    self.order_states.get(order_id),
                    Some(OrderState::Active) | Some(OrderState::PartiallyFilled { .. })
                )
            })
            .map(|(id, order)| (*id, order))
            .collect()
    }
    
    /// Cancel all orders for emergency situations
    pub fn cancel_all_orders(&mut self, reason: CancellationReason) -> Result<Vec<OrderCancellationResult>, OrderManagementError> {
        let active_orders: Vec<OrderId> = self.get_active_orders()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        
        let mut results = Vec::new();
        
        for order_id in active_orders {
            match self.cancel_order(order_id, reason.clone()) {
                Ok(result) => results.push(result),
                Err(e) => {
                    // Log error but continue with other cancellations
                    eprintln!("Failed to cancel order {}: {}", order_id, e);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Validate an order against business rules and risk limits
    fn validate_order(&self, order: &Order) -> Result<(), OrderManagementError> {
        // Basic order validation
        order.validate()
            .map_err(|e| OrderManagementError::ValidationFailed(e.to_string()))?;
        
        // Size limits
        if order.size > self.risk_limits.max_order_size {
            return Err(OrderManagementError::ValidationFailed(
                format!("Order size {} exceeds maximum {}", 
                        order.size, self.risk_limits.max_order_size)
            ));
        }
        
        if order.size < self.risk_limits.min_order_size {
            return Err(OrderManagementError::ValidationFailed(
                format!("Order size {} below minimum {}", 
                        order.size, self.risk_limits.min_order_size)
            ));
        }
        
        // Value limits
        if order.order_type == OrderType::Limit {
            let order_value = order.price.saturating_mul(order.size) / PRICE_SCALE;
            if order_value > self.risk_limits.max_order_value {
                return Err(OrderManagementError::ValidationFailed(
                    format!("Order value {} exceeds maximum {}", 
                            order_value, self.risk_limits.max_order_value)
                ));
            }
        }
        
        // Price increment validation
        if order.order_type == OrderType::Limit {
            if order.price % self.risk_limits.min_price_increment != 0 {
                return Err(OrderManagementError::ValidationFailed(
                    format!("Price {} not aligned to minimum increment {}", 
                            order.price, self.risk_limits.min_price_increment)
                ));
            }
        }
        
        // Price deviation check (if we have market data)
        if order.order_type == OrderType::Limit {
            if let Some(mid_price) = self.order_book.get_mid_price() {
                let max_deviation = (mid_price * self.risk_limits.max_price_deviation_bps as u64) / 10000;
                let price_diff = if order.price > mid_price {
                    order.price - mid_price
                } else {
                    mid_price - order.price
                };
                
                if price_diff > max_deviation {
                    return Err(OrderManagementError::ValidationFailed(
                        format!("Price {} deviates too much from mid price {}", 
                                order.price, mid_price)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform comprehensive risk checks
    fn perform_risk_checks(&self, order: &Order) -> Result<(), OrderManagementError> {
        // Check maximum orders per symbol
        if self.active_order_count >= self.risk_limits.max_orders_per_symbol {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Maximum orders per symbol ({}) exceeded", 
                        self.risk_limits.max_orders_per_symbol)
            ));
        }
        
        // Calculate total exposure
        let current_bid_exposure = self.order_book.total_bid_volume;
        let current_ask_exposure = self.order_book.total_ask_volume;
        let total_exposure = current_bid_exposure + current_ask_exposure;
        
        let new_exposure = match order.side {
            Side::Buy => total_exposure + order.size,
            Side::Sell => total_exposure + order.size,
        };
        
        if new_exposure > self.risk_limits.max_total_exposure {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Total exposure {} would exceed maximum {}", 
                        new_exposure, self.risk_limits.max_total_exposure)
            ));
        }
        
        Ok(())
    }
    
    /// Log an order event
    fn log_event(&mut self, event: OrderEvent) {
        self.order_events.push(event);
        
        // Optionally limit event history size
        if self.order_events.len() > 10000 {
            self.order_events.drain(0..1000); // Remove oldest 1000 events
        }
    }
    
    /// Get comprehensive statistics
    pub fn get_statistics(&self) -> OrderManagerStatistics {
        let order_book_stats = self.order_book.get_statistics();
        
        let state_counts = self.order_states.values().fold(
            OrderStateCount::default(),
            |mut acc, state| {
                match state {
                    OrderState::Pending => acc.pending += 1,
                    OrderState::Active => acc.active += 1,
                    OrderState::PartiallyFilled { .. } => acc.partially_filled += 1,
                    OrderState::Filled { .. } => acc.filled += 1,
                    OrderState::Cancelled { .. } => acc.cancelled += 1,
                    OrderState::Rejected { .. } => acc.rejected += 1,
                    OrderState::Expired => acc.expired += 1,
                    OrderState::Modifying => acc.modifying += 1,
                }
                acc
            }
        );
        
        OrderManagerStatistics {
            order_book_stats,
            total_orders_processed: self.total_orders_processed.load(Ordering::SeqCst),
            active_order_count: self.active_order_count,
            total_events: self.order_events.len() as u64,
            state_counts,
            risk_limits: self.risk_limits.clone(),
        }
    }
}

/// Result of order submission
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderSubmissionResult {
    pub order_id: OrderId,
    pub status: OrderSubmissionStatus,
    pub message: Option<String>,
    pub trades: Vec<Trade>,
}

/// Order submission status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderSubmissionStatus {
    Accepted,
    PartiallyFilled,
    Filled,
    Rejected,
    RejectedRisk,
}

/// Result of order cancellation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderCancellationResult {
    pub order_id: OrderId,
    pub cancelled_order: Order,
    pub reason: CancellationReason,
    pub timestamp: u64,
}

/// Result of order modification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderModificationResult {
    pub original_order_id: OrderId,
    pub new_order_id: OrderId,
    pub new_order: Order,
    pub cancellation_result: OrderCancellationResult,
    pub submission_result: OrderSubmissionResult,
    pub modification_id: Option<String>,
}

/// Order state counts for statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderStateCount {
    pub pending: u32,
    pub active: u32,
    pub partially_filled: u32,
    pub filled: u32,
    pub cancelled: u32,
    pub rejected: u32,
    pub expired: u32,
    pub modifying: u32,
}

/// Comprehensive order manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerStatistics {
    pub order_book_stats: OrderBookStatistics,
    pub total_orders_processed: u64,
    pub active_order_count: u32,
    pub total_events: u64,
    pub state_counts: OrderStateCount,
    pub risk_limits: RiskLimits,
}  
 if self.active_order_count >= self.risk_limits.max_orders_per_symbol {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Maximum orders per symbol ({}) exceeded", 
                        self.risk_limits.max_orders_per_symbol)
            ));
        }
        
        // Rate limiting check
        if !self.rate_limiter.can_submit_order(order.timestamp) {
            return Err(OrderManagementError::RiskCheckFailed(
                "Order rate limit exceeded".to_string()
            ));
        }
        
        // Minimum interval check
        if self.last_order_timestamp > 0 {
            let interval = order.timestamp.saturating_sub(self.last_order_timestamp);
            let min_interval_ns = self.risk_limits.min_order_interval_us * 1000;
            if interval < min_interval_ns {
                return Err(OrderManagementError::RiskCheckFailed(
                    format!("Order interval {} ns below minimum {} ns", 
                            interval, min_interval_ns)
                ));
            }
        }
        
        // Position limit checks
        let position_change = match order.side {
            Side::Buy => order.size as i64,
            Side::Sell => -(order.size as i64),
        };
        
        let new_position = self.position.size + position_change;
        
        if new_position > self.risk_limits.max_long_position {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Position {} would exceed maximum long position {}", 
                        new_position, self.risk_limits.max_long_position)
            ));
        }
        
        if new_position < self.risk_limits.max_short_position {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Position {} would exceed maximum short position {}", 
                        new_position, self.risk_limits.max_short_position)
            ));
        }
        
        // Calculate total exposure
        let current_bid_exposure = self.order_book.total_bid_volume;
        let current_ask_exposure = self.order_book.total_ask_volume;
        let total_exposure = current_bid_exposure + current_ask_exposure;
        
        let new_exposure = match order.side {
            Side::Buy => total_exposure + order.size,
            Side::Sell => total_exposure + order.size,
        };
        
        if new_exposure > self.risk_limits.max_total_exposure {
            return Err(OrderManagementError::RiskCheckFailed(
                format!("Total exposure {} would exceed maximum {}", 
                        new_exposure, self.risk_limits.max_total_exposure)
            ));
        }
        
        Ok(())
    }
    
    /// Update position tracking with new trade
    pub fn update_position(&mut self, trade: &Trade) {
        let timestamp = trade.timestamp;
        
        // Determine if this trade affects our position
        let (trade_side, trade_size) = if trade.buyer_order_id == trade.seller_order_id {
            // This shouldn't happen, but handle gracefully
            return;
        } else {
            // Check if we're the buyer or seller
            let our_orders: Vec<OrderId> = self.order_book.orders.keys().copied().collect();
            
            if our_orders.contains(&trade.buyer_order_id) {
                (Side::Buy, trade.size as i64)
            } else if our_orders.contains(&trade.seller_order_id) {
                (Side::Sell, -(trade.size as i64))
            } else {
                // Trade doesn't involve our orders
                return;
            }
        };
        
        // Update position
        let old_position = self.position.size;
        let new_position = old_position + trade_size;
        
        // Calculate realized PnL if position is reducing
        if (old_position > 0 && trade_side == Side::Sell) || (old_position < 0 && trade_side == Side::Buy) {
            let closing_size = std::cmp::min(old_position.abs(), trade_size.abs()) as u64;
            let pnl_per_unit = match trade_side {
                Side::Buy => self.position.avg_price as i64 - trade.price as i64,
                Side::Sell => trade.price as i64 - self.position.avg_price as i64,
            };
            let realized_pnl = (pnl_per_unit * closing_size as i64) / PRICE_SCALE as i64;
            self.position.realized_pnl += realized_pnl;
        }
        
        // Update average price for new position
        if new_position != 0 {
            if (old_position >= 0 && trade_side == Side::Buy) || (old_position <= 0 && trade_side == Side::Sell) {
                // Adding to position
                let old_notional = (old_position.abs() as u64 * self.position.avg_price) / PRICE_SCALE;
                let new_notional = (trade.size * trade.price) / PRICE_SCALE;
                let total_notional = old_notional + new_notional;
                let total_size = old_position.abs() as u64 + trade.size;
                
                if total_size > 0 {
                    self.position.avg_price = (total_notional * PRICE_SCALE) / total_size;
                }
            }
        } else {
            // Position closed
            self.position.avg_price = 0;
        }
        
        self.position.size = new_position;
        self.position.total_volume += trade.size;
        self.position.trade_count += 1;
        self.position.last_update = timestamp;
        
        // Update unrealized PnL based on current mid price
        if let Some(mid_price) = self.order_book.get_mid_price() {
            self.update_unrealized_pnl(mid_price);
        }
    }
    
    /// Update unrealized PnL based on current market price
    pub fn update_unrealized_pnl(&mut self, current_price: u64) {
        if self.position.size != 0 && self.position.avg_price > 0 {
            let pnl_per_unit = if self.position.size > 0 {
                current_price as i64 - self.position.avg_price as i64
            } else {
                self.position.avg_price as i64 - current_price as i64
            };
            
            self.position.unrealized_pnl = (pnl_per_unit * self.position.size.abs()) / PRICE_SCALE as i64;
        } else {
            self.position.unrealized_pnl = 0;
        }
    }
    
    /// Check and handle expired orders
    pub fn handle_expired_orders(&mut self) -> Result<Vec<OrderCancellationResult>, OrderManagementError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let expired_orders = self.expiration_manager.get_expired_orders(current_time);
        let mut results = Vec::new();
        
        for order_id in expired_orders {
            // Mark as expired first
            self.order_states.insert(order_id, OrderState::Expired);
            
            // Log expiration event
            self.log_event(OrderEvent {
                id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
                order_id,
                event_type: OrderEventType::Expired,
                timestamp: current_time,
                data: OrderEventData::Message { text: "Order expired".to_string() },
                sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
            });
            
            // Cancel the expired order
            match self.cancel_order(order_id, CancellationReason::Expired) {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to cancel expired order {}: {}", order_id, e);
                }
            }
        }
        
        Ok(results)
    }
    
    /// Update performance metrics
    pub fn update_performance_metrics(&mut self, processing_latency_ns: u64) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Update latency metrics
        if self.performance_metrics.avg_processing_latency_ns == 0 {
            self.performance_metrics.avg_processing_latency_ns = processing_latency_ns;
        } else {
            // Exponential moving average
            self.performance_metrics.avg_processing_latency_ns = 
                (self.performance_metrics.avg_processing_latency_ns * 9 + processing_latency_ns) / 10;
        }
        
        self.performance_metrics.max_processing_latency_ns = 
            self.performance_metrics.max_processing_latency_ns.max(processing_latency_ns);
        
        // Calculate orders per second
        let time_window = 1_000_000_000; // 1 second in nanoseconds
        if current_time > self.performance_metrics.last_update + time_window {
            let orders_in_window = self.total_orders_processed.load(Ordering::SeqCst);
            let time_elapsed = current_time - self.performance_metrics.last_update;
            
            if time_elapsed > 0 {
                self.performance_metrics.orders_per_second = 
                    (orders_in_window as f64 * 1_000_000_000.0) / time_elapsed as f64;
            }
            
            self.performance_metrics.last_update = current_time;
        }
        
        // Calculate fill rate
        let total_orders = self.total_orders_processed.load(Ordering::SeqCst);
        if total_orders > 0 {
            let filled_orders = self.order_states.values()
                .filter(|state| matches!(state, OrderState::Filled { .. } | OrderState::PartiallyFilled { .. }))
                .count() as u64;
            
            self.performance_metrics.fill_rate = (filled_orders as f64 / total_orders as f64) * 100.0;
        }
    }
    
    /// Add order to expiration tracking if needed
    fn track_order_expiration(&mut self, order: &Order) {
        if let TimeInForce::GTD(expiration_time) = order.time_in_force {
            self.expiration_manager.add_order(order.id, expiration_time);
        }
    }
    
    /// Log an order event
    fn log_event(&mut self, event: OrderEvent) {
        self.order_events.push(event);
        
        // Optionally limit event history size
        if self.order_events.len() > 10000 {
            self.order_events.drain(0..1000); // Remove oldest 1000 events
        }
    }
    
    /// Get comprehensive statistics
    pub fn get_statistics(&self) -> OrderManagerStatistics {
        let order_book_stats = self.order_book.get_statistics();
        
        let state_counts = self.order_states.values().fold(
            OrderStateCount::default(),
            |mut acc, state| {
                match state {
                    OrderState::Pending => acc.pending += 1,
                    OrderState::Active => acc.active += 1,
                    OrderState::PartiallyFilled { .. } => acc.partially_filled += 1,
                    OrderState::Filled { .. } => acc.filled += 1,
                    OrderState::Cancelled { .. } => acc.cancelled += 1,
                    OrderState::Rejected { .. } => acc.rejected += 1,
                    OrderState::Expired => acc.expired += 1,
                    OrderState::Modifying => acc.modifying += 1,
                }
                acc
            }
        );
        
        OrderManagerStatistics {
            order_book_stats,
            total_orders_processed: self.total_orders_processed.load(Ordering::SeqCst),
            active_order_count: self.active_order_count,
            total_events: self.order_events.len() as u64,
            state_counts,
            risk_limits: self.risk_limits.clone(),
            position: self.position.clone(),
            performance_metrics: self.performance_metrics.clone(),
        }
    }
    
    /// Emergency halt - cancel all orders and stop accepting new ones
    pub fn emergency_halt(&mut self) -> Result<Vec<OrderCancellationResult>, OrderManagementError> {
        // Cancel all active orders
        let results = self.cancel_all_orders(CancellationReason::EmergencyHalt)?;
        
        // Set extremely restrictive risk limits to prevent new orders
        self.risk_limits.max_orders_per_symbol = 0;
        self.risk_limits.max_order_size = 0;
        self.risk_limits.max_total_exposure = 0;
        
        // Log emergency halt event
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: OrderId::new(0), // System event
            event_type: OrderEventType::Cancelled,
            timestamp,
            data: OrderEventData::Message { text: "Emergency halt activated".to_string() },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
        
        Ok(results)
    }
    
    /// Resume normal operations after emergency halt
    pub fn resume_operations(&mut self, new_risk_limits: Option<RiskLimits>) {
        if let Some(limits) = new_risk_limits {
            self.risk_limits = limits;
        } else {
            self.risk_limits = RiskLimits::default();
        }
        
        // Log resume event
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        self.log_event(OrderEvent {
            id: self.next_event_id.fetch_add(1, Ordering::SeqCst),
            order_id: OrderId::new(0), // System event
            event_type: OrderEventType::Created,
            timestamp,
            data: OrderEventData::Message { text: "Operations resumed".to_string() },
            sequence: self.event_sequence.fetch_add(1, Ordering::SeqCst),
        });
    }
}

/// Result of order submission
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderSubmissionResult {
    pub order_id: OrderId,
    pub status: OrderSubmissionStatus,
    pub message: Option<String>,
    pub trades: Vec<Trade>,
}

/// Order submission status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrderSubmissionStatus {
    Accepted,
    PartiallyFilled,
    Filled,
    Rejected,
    RejectedRisk,
}

/// Result of order cancellation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderCancellationResult {
    pub order_id: OrderId,
    pub cancelled_order: Order,
    pub reason: CancellationReason,
    pub timestamp: u64,
}

/// Result of order modification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrderModificationResult {
    pub original_order_id: OrderId,
    pub new_order_id: OrderId,
    pub new_order: Order,
    pub cancellation_result: OrderCancellationResult,
    pub submission_result: OrderSubmissionResult,
    pub modification_id: Option<String>,
}

/// Order state counts for statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrderStateCount {
    pub pending: u32,
    pub active: u32,
    pub partially_filled: u32,
    pub filled: u32,
    pub cancelled: u32,
    pub rejected: u32,
    pub expired: u32,
    pub modifying: u32,
}

/// Comprehensive order manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagerStatistics {
    pub order_book_stats: OrderBookStatistics,
    pub total_orders_processed: u64,
    pub active_order_count: u32,
    pub total_events: u64,
    pub state_counts: OrderStateCount,
    pub risk_limits: RiskLimits,
    pub position: Position,
    pub performance_metrics: PerformanceMetrics,
}