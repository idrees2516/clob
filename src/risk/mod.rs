//! Risk Management Module
//!
//! This module provides comprehensive risk management capabilities including:
//! - Portfolio risk metrics (VaR, Expected Shortfall, Maximum Drawdown)
//! - Dynamic hedging strategies
//! - Kelly criterion position sizing
//! - Real-time portfolio optimization

pub mod portfolio_risk;
pub mod hedging;
pub mod position_sizing;
pub mod portfolio_optimization;
pub mod real_time_monitor;

pub use portfolio_risk::*;
pub use hedging::*;
pub use position_sizing::*;
pub use portfolio_optimization::*;
pub use real_time_monitor::*;

use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Risk management errors
#[derive(Debug, Error)]
pub enum RiskError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
    
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),
    
    #[error("Optimization failed: {0}")]
    OptimizationError(String),
}

/// Asset identifier
pub type AssetId = String;

/// Position in a single asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub asset_id: AssetId,
    pub quantity: i64,
    pub average_price: FixedPoint,
    pub current_price: FixedPoint,
    pub unrealized_pnl: FixedPoint,
    pub realized_pnl: FixedPoint,
    pub last_update: u64,
}

impl Position {
    pub fn new(asset_id: AssetId, quantity: i64, average_price: FixedPoint, current_price: FixedPoint) -> Self {
        let unrealized_pnl = FixedPoint::from_int(quantity) * (current_price - average_price);
        
        Self {
            asset_id,
            quantity,
            average_price,
            current_price,
            unrealized_pnl,
            realized_pnl: FixedPoint::zero(),
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    pub fn update_price(&mut self, new_price: FixedPoint) {
        self.current_price = new_price;
        self.unrealized_pnl = FixedPoint::from_int(self.quantity) * (new_price - self.average_price);
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
    
    pub fn market_value(&self) -> FixedPoint {
        FixedPoint::from_int(self.quantity) * self.current_price
    }
}

/// Portfolio containing multiple positions
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub positions: HashMap<AssetId, Position>,
    pub cash: FixedPoint,
    pub last_update: u64,
}

impl Portfolio {
    pub fn new(cash: FixedPoint) -> Self {
        Self {
            positions: HashMap::new(),
            cash,
            last_update: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    pub fn add_position(&mut self, position: Position) {
        self.positions.insert(position.asset_id.clone(), position);
        self.update_timestamp();
    }
    
    pub fn update_position_price(&mut self, asset_id: &AssetId, new_price: FixedPoint) {
        if let Some(position) = self.positions.get_mut(asset_id) {
            position.update_price(new_price);
        }
        self.update_timestamp();
    }
    
    pub fn total_market_value(&self) -> FixedPoint {
        let positions_value: FixedPoint = self.positions
            .values()
            .map(|p| p.market_value())
            .sum();
        positions_value + self.cash
    }
    
    pub fn total_unrealized_pnl(&self) -> FixedPoint {
        self.positions
            .values()
            .map(|p| p.unrealized_pnl)
            .sum()
    }
    
    pub fn total_realized_pnl(&self) -> FixedPoint {
        self.positions
            .values()
            .map(|p| p.realized_pnl)
            .sum()
    }
    
    fn update_timestamp(&mut self) {
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
}

/// Market data for risk calculations
#[derive(Debug, Clone)]
pub struct MarketData {
    pub prices: HashMap<AssetId, FixedPoint>,
    pub returns: HashMap<AssetId, Vec<FixedPoint>>,
    pub correlations: HashMap<(AssetId, AssetId), FixedPoint>,
    pub volatilities: HashMap<AssetId, FixedPoint>,
    pub timestamp: u64,
}

impl MarketData {
    pub fn new() -> Self {
        Self {
            prices: HashMap::new(),
            returns: HashMap::new(),
            correlations: HashMap::new(),
            volatilities: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    pub fn add_price(&mut self, asset_id: AssetId, price: FixedPoint) {
        self.prices.insert(asset_id, price);
        self.update_timestamp();
    }
    
    pub fn add_returns(&mut self, asset_id: AssetId, returns: Vec<FixedPoint>) {
        self.returns.insert(asset_id, returns);
        self.update_timestamp();
    }
    
    pub fn add_correlation(&mut self, asset1: AssetId, asset2: AssetId, correlation: FixedPoint) {
        self.correlations.insert((asset1.clone(), asset2.clone()), correlation);
        self.correlations.insert((asset2, asset1), correlation);
        self.update_timestamp();
    }
    
    pub fn add_volatility(&mut self, asset_id: AssetId, volatility: FixedPoint) {
        self.volatilities.insert(asset_id, volatility);
        self.update_timestamp();
    }
    
    fn update_timestamp(&mut self) {
        self.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
    }
}