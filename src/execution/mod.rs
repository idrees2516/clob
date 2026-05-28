//! Execution algorithms for optimal order execution
//! 
//! This module implements sophisticated execution strategies including:
//! - TWAP (Time-Weighted Average Price) execution with adaptive scheduling
//! - VWAP (Volume-Weighted Average Price) execution
//! - Implementation Shortfall optimization
//! - Market impact modeling and cost optimization

pub mod twap;
pub mod volume_forecasting;
pub mod market_impact;
pub mod execution_control;

use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Execution-related errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("Invalid execution parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Insufficient market data: {0}")]
    InsufficientData(String),
    
    #[error("Market impact calculation failed: {0}")]
    MarketImpactError(String),
    
    #[error("Volume forecasting error: {0}")]
    VolumeForecasting(String),
    
    #[error("Execution control error: {0}")]
    ExecutionControl(String),
    
    #[error("Risk limit breach: {0}")]
    RiskLimitBreach(String),
}

/// Order side enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

/// Order structure for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: u64,
    pub price: Option<FixedPoint>,
    pub timestamp: u64,
    pub time_in_force: TimeInForce,
}

/// Time in force options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForce {
    Day,
    GoodTillCanceled,
    ImmediateOrCancel,
    FillOrKill,
}

/// Market state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub symbol: String,
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub last_price: FixedPoint,
    pub last_volume: u64,
    pub timestamp: u64,
    pub volatility: FixedPoint,
    pub average_daily_volume: u64,
}

impl MarketState {
    /// Calculate mid price
    pub fn mid_price(&self) -> FixedPoint {
        (self.bid_price + self.ask_price) / FixedPoint::from_integer(2)
    }
    
    /// Calculate bid-ask spread
    pub fn spread(&self) -> FixedPoint {
        self.ask_price - self.bid_price
    }
    
    /// Calculate relative spread
    pub fn relative_spread(&self) -> FixedPoint {
        self.spread() / self.mid_price()
    }
}

/// Market conditions for execution adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility_regime: VolatilityRegime,
    pub liquidity_level: LiquidityLevel,
    pub trend_direction: TrendDirection,
    pub market_hours: MarketHours,
    pub news_impact: NewsImpact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiquidityLevel {
    Low,
    Normal,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Strong_Up,
    Weak_Up,
    Sideways,
    Weak_Down,
    Strong_Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketHours {
    PreMarket,
    Open,
    Regular,
    Close,
    AfterHours,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NewsImpact {
    None,
    Low,
    Medium,
    High,
    Extreme,
}

/// Execution performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    pub total_executed: u64,
    pub average_price: FixedPoint,
    pub execution_time: u64,
    pub implementation_shortfall: FixedPoint,
    pub tracking_error: FixedPoint,
    pub market_impact: FixedPoint,
    pub timing_cost: FixedPoint,
    pub opportunity_cost: FixedPoint,
    pub slippage: FixedPoint,
}

/// Volume patterns for intraday analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePatterns {
    pub hourly_patterns: HashMap<u8, FixedPoint>, // Hour -> normalized volume
    pub daily_patterns: HashMap<u8, FixedPoint>,  // Day of week -> normalized volume
    pub seasonal_adjustments: HashMap<String, FixedPoint>, // Season/event -> adjustment factor
}

impl VolumePatterns {
    /// Get expected volume multiplier for given hour
    pub fn get_hourly_multiplier(&self, hour: u8) -> FixedPoint {
        self.hourly_patterns.get(&hour).copied()
            .unwrap_or(FixedPoint::from_integer(1))
    }
    
    /// Get expected volume multiplier for given day of week
    pub fn get_daily_multiplier(&self, day_of_week: u8) -> FixedPoint {
        self.daily_patterns.get(&day_of_week).copied()
            .unwrap_or(FixedPoint::from_integer(1))
    }
}