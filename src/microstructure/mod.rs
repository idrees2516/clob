//! High-Frequency Market Microstructure Analytics
//! 
//! This module provides comprehensive market microstructure analysis tools including:
//! - Real-time liquidity metrics calculation
//! - Order flow analysis with toxic flow detection  
//! - Market impact measurement and modeling
//! - High-frequency volatility estimation

pub mod liquidity_metrics;
pub mod order_flow;
pub mod market_impact;
pub mod volatility;

pub use liquidity_metrics::*;
pub use order_flow::*;
pub use market_impact::*;
pub use volatility::*;

use crate::math::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Core market data structure for microstructure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub timestamp: u64,
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_volume: FixedPoint,
    pub ask_volume: FixedPoint,
    pub last_trade_price: FixedPoint,
    pub last_trade_volume: FixedPoint,
    pub trade_direction: Option<TradeDirection>,
}

/// Trade direction classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TradeDirection {
    Buy = 1,
    Sell = -1,
    Unknown = 0,
}

/// Order book level data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: FixedPoint,
    pub volume: FixedPoint,
    pub order_count: u32,
}

/// Complete order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub timestamp: u64,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

/// Trade record for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: u64,
    pub price: FixedPoint,
    pub volume: FixedPoint,
    pub direction: TradeDirection,
    pub trade_id: u64,
}

/// Market microstructure error types
#[derive(Debug, thiserror::Error)]
pub enum MicrostructureError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Calculation error: {0}")]
    CalculationError(String),
    
    #[error("Data quality issue: {0}")]
    DataQuality(String),
}

/// Result type for microstructure operations
pub type MicrostructureResult<T> = Result<T, MicrostructureError>;