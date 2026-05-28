//! Real-Time Risk Monitor
//!
//! This module provides ultra-low latency risk monitoring with sub-microsecond updates,
//! automatic position flattening, and comprehensive limit enforcement.

use crate::math::fixed_point::FixedPoint;
use crate::performance::lock_free::hash_map::LockFreeHashMap;
use crate::performance::lock_free::atomic_operations::AtomicFixedPoint;
use crate::risk::{Portfolio, Position, AssetId, RiskError, MarketData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Real-time risk monitoring errors
#[derive(Debug, Error)]
pub enum RiskMonitorError {
    #[error("Risk limit breach: {0}")]
    LimitBreach(String),
    
    #[error("Position flattening failed: {0}")]
    FlatteningFailed(String),
    
    #[error("Cache update failed: {0}")]
    CacheUpdateFailed(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Risk calculation error: {0}")]
    CalculationError(#[from] RiskError),
}

/// Risk limit types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LimitType {
    PositionLimit,
    LossLimit,
    LeverageLimit,
    ConcentrationLimit,
    VaRLimit,
    DrawdownLimit,
}

/// Risk limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimit {
    pub limit_type: LimitType,
    pub asset_id: Option<AssetId>,
    pub threshold: FixedPoint,
    pub warning_threshold: FixedPoint,
    pub enabled: bool,
    pub auto_flatten: bool,
}

impl RiskLimit {
    pub fn new(
        limit_type: LimitType,
        asset_id: Option<AssetId>,
        threshold: FixedPoint,
        warning_threshold: FixedPoint,
    ) -> Self {
        Self {
            limit_type,
            asset_id,
            threshold,
            warning_threshold,
            enabled: true,
            auto_flatten: true,
        }
    }
}

/// Risk limit breach information
#[derive(Debug, Clone)]
pub struct LimitBreach {
    pub limit_type: LimitType,
    pub asset_id: Option<AssetId>,
    pub current_value: FixedPoint,
    pub threshold: FixedPoint,
    pub severity: BreachSeverity,
    pub timestamp: u64,
    pub auto_flatten_required: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BreachSeverity {
    Warning,
    Critical,
    Emergency,
}

/// Cached risk metrics for ultra-low latency access
#[derive(Debug, Clone)]
pub struct CachedRiskMetrics {
    pub portfolio_value: AtomicFixedPoint,
    pub total_pnl: AtomicFixedPoint,
    pub leverage_ratio: AtomicFixedPoint,
    pub max_position_concentration: AtomicFixedPoint,
    pub var_estimate: AtomicFixedPoint,
    pub current_drawdown: AtomicFixedPoint,
    pub last_update: AtomicU64,
}

impl CachedRiskMetrics {
    pub fn new() -> Self {
        Self {
            portfolio_value: AtomicFixedPoint::new(FixedPoint::zero()),
            total_pnl: AtomicFixedPoint::new(FixedPoint::zero()),
            leverage_ratio: AtomicFixedPoint::new(FixedPoint::zero()),
            max_position_concentration: AtomicFixedPoint::new(FixedPoint::zero()),
            var_estimate: AtomicFixedPoint::new(FixedPoint::zero()),
            current_drawdown: AtomicFixedPoint::new(FixedPoint::zero()),
            last_update: AtomicU64::new(0),
        }
    }
    
    pub fn update_all(
        &self,
        portfolio_value: FixedPoint,
        total_pnl: FixedPoint,
        leverage_ratio: FixedPoint,
        max_concentration: FixedPoint,
        var_estimate: FixedPoint,
        current_drawdown: FixedPoint,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
            
        self.portfolio_value.store(portfolio_value, Ordering::Release);
        self.total_pnl.store(total_pnl, Ordering::Release);
        self.leverage_ratio.store(leverage_ratio, Ordering::Release);
        self.max_position_concentration.store(max_concentration, Ordering::Release);
        self.var_estimate.store(var_estimate, Ordering::Release);
        self.current_drawdown.store(current_drawdown, Ordering::Release);
        self.last_update.store(timestamp, Ordering::Release);
    }
}

/// Position flattening action
#[derive(Debug, Clone)]
pub struct FlatteningAction {
    pub asset_id: AssetId,
    pub current_quantity: i64,
    pub target_quantity: i64,
    pub flatten_percentage: FixedPoint,
    pub urgency: FlatteningUrgency,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlatteningUrgency {
    Low,
    Medium,
    High,
    Emergency,
}

/// Real-time risk monitor with sub-microsecond updates
pub struct RealTimeRiskMonitor {
    // Risk limits configuration
    limits: Vec<RiskLimit>,
    
    // Cached risk metrics for ultra-low latency access
    cached_metrics: Arc<CachedRiskMetrics>,
    
    // Position tracking
    position_cache: LockFreeHashMap<AssetId, AtomicFixedPoint>,
    pnl_cache: LockFreeHashMap<AssetId, AtomicFixedPoint>,
    
    // Limit breach tracking
    active_breaches: LockFreeHashMap<String, LimitBreach>,
    breach_count: AtomicU64,
    
    // Performance tracking
    update_count: AtomicU64,
    last_update_duration: AtomicU64,
    max_update_duration: AtomicU64,
    
    // Configuration
    update_frequency_ns: u64,
    cache_warmup_enabled: AtomicBool,
    auto_flattening_enabled: AtomicBool,
    
    // Historical tracking for drawdown calculation
    peak_portfolio_value: AtomicFixedPoint,
    drawdown_start_time: AtomicU64,
}

impl RealTimeRiskMonitor {
    /// Create a new real-time risk monitor
    pub fn new(limits: Vec<RiskLimit>) -> Result<Self, RiskMonitorError> {
        // Validate limits configuration
        for limit in &limits {
            if limit.threshold <= FixedPoint::zero() {
                return Err(RiskMonitorError::InvalidConfiguration(
                    format!("Invalid threshold for {:?}", limit.limit_type)
                ));
            }
            if limit.warning_threshold >= limit.threshold {
                return Err(RiskMonitorError::InvalidConfiguration(
                    "Warning threshold must be less than limit threshold".to_string()
                ));
            }
        }
        
        Ok(Self {
            limits,
            cached_metrics: Arc::new(CachedRiskMetrics::new()),
            position_cache: LockFreeHashMap::new(),
            pnl_cache: LockFreeHashMap::new(),
            active_breaches: LockFreeHashMap::new(),
            breach_count: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
            last_update_duration: AtomicU64::new(0),
            max_update_duration: AtomicU64::new(0),
            update_frequency_ns: 100, // 100ns target update frequency
            cache_warmup_enabled: AtomicBool::new(true),
            auto_flattening_enabled: AtomicBool::new(true),
            peak_portfolio_value: AtomicFixedPoint::new(FixedPoint::zero()),
            drawdown_start_time: AtomicU64::new(0),
        })
    }
    
    /// Update risk metrics with sub-microsecond latency
    pub fn update_risk_metrics(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<Vec<LimitBreach>, RiskMonitorError> {
        let start_time = Instant::now();
        
        // Calculate core risk metrics
        let portfolio_value = self.calculate_portfolio_value(portfolio, market_data)?;
        let total_pnl = portfolio.total_unrealized_pnl() + portfolio.total_realized_pnl();
        let leverage_ratio = self.calculate_leverage_ratio(portfolio, market_data)?;
        let max_concentration = self.calculate_max_concentration(portfolio)?;
        let var_estimate = self.calculate_var_estimate(portfolio, market_data)?;
        let current_drawdown = self.calculate_current_drawdown(portfolio_value);
        
        // Update cached metrics atomically
        self.cached_metrics.update_all(
            portfolio_value,
            total_pnl,
            leverage_ratio,
            max_concentration,
            var_estimate,
            current_drawdown,
        );
        
        // Update position and PnL caches
        self.update_position_caches(portfolio)?;
        
        // Check all limits and identify breaches
        let breaches = self.check_all_limits(
            portfolio_value,
            total_pnl,
            leverage_ratio,
            max_concentration,
            var_estimate,
            current_drawdown,
            portfolio,
        )?;
        
        // Update performance metrics
        let update_duration = start_time.elapsed().as_nanos() as u64;
        self.last_update_duration.store(update_duration, Ordering::Relaxed);
        self.max_update_duration.fetch_max(update_duration, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(breaches)
    }
    
    /// Get cached risk metrics with nanosecond access time
    pub fn get_cached_metrics(&self) -> (FixedPoint, FixedPoint, FixedPoint, FixedPoint, FixedPoint, FixedPoint, u64) {
        (
            self.cached_metrics.portfolio_value.load(Ordering::Acquire),
            self.cached_metrics.total_pnl.load(Ordering::Acquire),
            self.cached_metrics.leverage_ratio.load(Ordering::Acquire),
            self.cached_metrics.max_position_concentration.load(Ordering::Acquire),
            self.cached_metrics.var_estimate.load(Ordering::Acquire),
            self.cached_metrics.current_drawdown.load(Ordering::Acquire),
            self.cached_metrics.last_update.load(Ordering::Acquire),
        )
    }
    
    /// Generate position flattening actions for limit breaches
    pub fn generate_flattening_actions(
        &self,
        breaches: &[LimitBreach],
        portfolio: &Portfolio,
    ) -> Result<Vec<FlatteningAction>, RiskMonitorError> {
        let mut actions = Vec::new();
        
        for breach in breaches {
            if !breach.auto_flatten_required {
                continue;
            }
            
            match breach.limit_type {
                LimitType::PositionLimit => {
                    if let Some(asset_id) = &breach.asset_id {
                        if let Some(position) = portfolio.positions.get(asset_id) {
                            let flatten_percentage = self.calculate_flatten_percentage(breach);
                            let target_quantity = (FixedPoint::from_int(position.quantity) * 
                                (FixedPoint::one() - flatten_percentage)).to_int();
                            
                            actions.push(FlatteningAction {
                                asset_id: asset_id.clone(),
                                current_quantity: position.quantity,
                                target_quantity,
                                flatten_percentage,
                                urgency: self.determine_urgency(breach),
                                reason: format!("Position limit breach: {} > {}", 
                                    breach.current_value, breach.threshold),
                            });
                        }
                    }
                }
                LimitType::LossLimit | LimitType::DrawdownLimit => {
                    // Flatten all positions proportionally
                    let flatten_percentage = self.calculate_flatten_percentage(breach);
                    
                    for (asset_id, position) in &portfolio.positions {
                        if position.quantity != 0 {
                            let target_quantity = (FixedPoint::from_int(position.quantity) * 
                                (FixedPoint::one() - flatten_percentage)).to_int();
                            
                            actions.push(FlatteningAction {
                                asset_id: asset_id.clone(),
                                current_quantity: position.quantity,
                                target_quantity,
                                flatten_percentage,
                                urgency: self.determine_urgency(breach),
                                reason: format!("{:?} breach: {} > {}", 
                                    breach.limit_type, breach.current_value, breach.threshold),
                            });
                        }
                    }
                }
                LimitType::ConcentrationLimit => {
                    // Flatten the most concentrated positions
                    if let Some(asset_id) = &breach.asset_id {
                        if let Some(position) = portfolio.positions.get(asset_id) {
                            let flatten_percentage = self.calculate_flatten_percentage(breach);
                            let target_quantity = (FixedPoint::from_int(position.quantity) * 
                                (FixedPoint::one() - flatten_percentage)).to_int();
                            
                            actions.push(FlatteningAction {
                                asset_id: asset_id.clone(),
                                current_quantity: position.quantity,
                                target_quantity,
                                flatten_percentage,
                                urgency: self.determine_urgency(breach),
                                reason: format!("Concentration limit breach: {} > {}", 
                                    breach.current_value, breach.threshold),
                            });
                        }
                    }
                }
                _ => {
                    // For other limit types, implement proportional flattening
                    let flatten_percentage = self.calculate_flatten_percentage(breach);
                    
                    for (asset_id, position) in &portfolio.positions {
                        if position.quantity != 0 {
                            let target_quantity = (FixedPoint::from_int(position.quantity) * 
                                (FixedPoint::one() - flatten_percentage)).to_int();
                            
                            actions.push(FlatteningAction {
                                asset_id: asset_id.clone(),
                                current_quantity: position.quantity,
                                target_quantity,
                                flatten_percentage,
                                urgency: self.determine_urgency(breach),
                                reason: format!("{:?} breach requires position reduction", breach.limit_type),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(actions)
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> (u64, u64, u64, u64) {
        (
            self.update_count.load(Ordering::Relaxed),
            self.last_update_duration.load(Ordering::Relaxed),
            self.max_update_duration.load(Ordering::Relaxed),
            self.breach_count.load(Ordering::Relaxed),
        )
    }
    
    /// Enable or disable automatic position flattening
    pub fn set_auto_flattening(&self, enabled: bool) {
        self.auto_flattening_enabled.store(enabled, Ordering::Relaxed);
    }
    
    // Private helper methods
    
    fn calculate_portfolio_value(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskMonitorError> {
        let mut total_value = portfolio.cash;
        
        for (asset_id, position) in &portfolio.positions {
            if let Some(price) = market_data.prices.get(asset_id) {
                total_value += FixedPoint::from_int(position.quantity) * *price;
            } else {
                // Use last known price from position
                total_value += FixedPoint::from_int(position.quantity) * position.current_price;
            }
        }
        
        Ok(total_value)
    }
    
    fn calculate_leverage_ratio(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskMonitorError> {
        let portfolio_value = self.calculate_portfolio_value(portfolio, market_data)?;
        
        if portfolio_value <= FixedPoint::zero() {
            return Ok(FixedPoint::zero());
        }
        
        let mut gross_exposure = FixedPoint::zero();
        
        for (asset_id, position) in &portfolio.positions {
            if let Some(price) = market_data.prices.get(asset_id) {
                gross_exposure += (FixedPoint::from_int(position.quantity) * *price).abs();
            } else {
                gross_exposure += (FixedPoint::from_int(position.quantity) * position.current_price).abs();
            }
        }
        
        Ok(gross_exposure / portfolio_value)
    }
    
    fn calculate_max_concentration(
        &self,
        portfolio: &Portfolio,
    ) -> Result<FixedPoint, RiskMonitorError> {
        let total_value = portfolio.total_market_value();
        
        if total_value <= FixedPoint::zero() {
            return Ok(FixedPoint::zero());
        }
        
        let mut max_concentration = FixedPoint::zero();
        
        for position in portfolio.positions.values() {
            let position_value = position.market_value().abs();
            let concentration = position_value / total_value;
            
            if concentration > max_concentration {
                max_concentration = concentration;
            }
        }
        
        Ok(max_concentration)
    }
    
    fn calculate_var_estimate(
        &self,
        portfolio: &Portfolio,
        market_data: &MarketData,
    ) -> Result<FixedPoint, RiskMonitorError> {
        // Simple VaR estimate using portfolio volatility
        // This is a placeholder - full VaR implementation is in task 11.2
        
        let portfolio_value = self.calculate_portfolio_value(portfolio, market_data)?;
        
        // Estimate portfolio volatility from individual asset volatilities
        let mut portfolio_variance = FixedPoint::zero();
        
        for (asset_id, position) in &portfolio.positions {
            if let Some(volatility) = market_data.volatilities.get(asset_id) {
                let weight = position.market_value() / portfolio_value;
                portfolio_variance += weight * weight * volatility * volatility;
            }
        }
        
        let portfolio_volatility = portfolio_variance.sqrt();
        
        // 95% VaR estimate (1.645 standard deviations)
        let var_multiplier = FixedPoint::from_float(1.645);
        Ok(portfolio_value * portfolio_volatility * var_multiplier)
    }
    
    fn calculate_current_drawdown(&self, current_value: FixedPoint) -> FixedPoint {
        let peak_value = self.peak_portfolio_value.load(Ordering::Acquire);
        
        if current_value > peak_value {
            // New peak - update and reset drawdown
            self.peak_portfolio_value.store(current_value, Ordering::Release);
            self.drawdown_start_time.store(
                SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                Ordering::Release
            );
            FixedPoint::zero()
        } else if peak_value > FixedPoint::zero() {
            // Calculate drawdown as percentage
            (peak_value - current_value) / peak_value
        } else {
            FixedPoint::zero()
        }
    }
    
    fn update_position_caches(&self, portfolio: &Portfolio) -> Result<(), RiskMonitorError> {
        for (asset_id, position) in &portfolio.positions {
            let position_value = AtomicFixedPoint::new(position.market_value());
            let pnl_value = AtomicFixedPoint::new(position.unrealized_pnl + position.realized_pnl);
            
            self.position_cache.insert(asset_id.clone(), position_value);
            self.pnl_cache.insert(asset_id.clone(), pnl_value);
        }
        
        Ok(())
    }
    
    fn check_all_limits(
        &self,
        portfolio_value: FixedPoint,
        total_pnl: FixedPoint,
        leverage_ratio: FixedPoint,
        max_concentration: FixedPoint,
        var_estimate: FixedPoint,
        current_drawdown: FixedPoint,
        portfolio: &Portfolio,
    ) -> Result<Vec<LimitBreach>, RiskMonitorError> {
        let mut breaches = Vec::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        for limit in &self.limits {
            if !limit.enabled {
                continue;
            }
            
            let (current_value, breach_asset_id) = match limit.limit_type {
                LimitType::PositionLimit => {
                    if let Some(asset_id) = &limit.asset_id {
                        if let Some(position) = portfolio.positions.get(asset_id) {
                            (position.market_value().abs(), Some(asset_id.clone()))
                        } else {
                            continue;
                        }
                    } else {
                        // Portfolio-wide position limit
                        (portfolio_value, None)
                    }
                }
                LimitType::LossLimit => {
                    (-total_pnl, None) // Negative PnL is a loss
                }
                LimitType::LeverageLimit => {
                    (leverage_ratio, None)
                }
                LimitType::ConcentrationLimit => {
                    if let Some(asset_id) = &limit.asset_id {
                        if let Some(position) = portfolio.positions.get(asset_id) {
                            let concentration = position.market_value().abs() / portfolio_value;
                            (concentration, Some(asset_id.clone()))
                        } else {
                            continue;
                        }
                    } else {
                        (max_concentration, None)
                    }
                }
                LimitType::VaRLimit => {
                    (var_estimate, None)
                }
                LimitType::DrawdownLimit => {
                    (current_drawdown, None)
                }
            };
            
            let severity = if current_value >= limit.threshold {
                BreachSeverity::Critical
            } else if current_value >= limit.warning_threshold {
                BreachSeverity::Warning
            } else {
                continue; // No breach
            };
            
            let breach = LimitBreach {
                limit_type: limit.limit_type.clone(),
                asset_id: breach_asset_id,
                current_value,
                threshold: if severity == BreachSeverity::Warning {
                    limit.warning_threshold
                } else {
                    limit.threshold
                },
                severity,
                timestamp,
                auto_flatten_required: limit.auto_flatten && severity == BreachSeverity::Critical,
            };
            
            // Store breach for tracking
            let breach_key = format!("{:?}_{}", limit.limit_type, 
                limit.asset_id.as_ref().unwrap_or(&"portfolio".to_string()));
            self.active_breaches.insert(breach_key, breach.clone());
            
            breaches.push(breach);
        }
        
        if !breaches.is_empty() {
            self.breach_count.fetch_add(breaches.len() as u64, Ordering::Relaxed);
        }
        
        Ok(breaches)
    }
    
    fn calculate_flatten_percentage(&self, breach: &LimitBreach) -> FixedPoint {
        match breach.severity {
            BreachSeverity::Warning => FixedPoint::from_float(0.1), // 10%
            BreachSeverity::Critical => {
                // Calculate based on how much over the limit we are
                let excess_ratio = (breach.current_value - breach.threshold) / breach.threshold;
                let base_percentage = FixedPoint::from_float(0.25); // 25% base
                let additional = excess_ratio * FixedPoint::from_float(0.5); // Up to 50% more
                (base_percentage + additional).min(FixedPoint::from_float(0.75)) // Max 75%
            }
            BreachSeverity::Emergency => FixedPoint::from_float(1.0), // 100% - full liquidation
        }
    }
    
    fn determine_urgency(&self, breach: &LimitBreach) -> FlatteningUrgency {
        match breach.severity {
            BreachSeverity::Warning => FlatteningUrgency::Low,
            BreachSeverity::Critical => {
                let excess_ratio = (breach.current_value - breach.threshold) / breach.threshold;
                if excess_ratio > FixedPoint::from_float(0.5) {
                    FlatteningUrgency::High
                } else {
                    FlatteningUrgency::Medium
                }
            }
            BreachSeverity::Emergency => FlatteningUrgency::Emergency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_risk_monitor_creation() {
        let limits = vec![
            RiskLimit::new(
                LimitType::LossLimit,
                None,
                FixedPoint::from_float(10000.0),
                FixedPoint::from_float(8000.0),
            ),
            RiskLimit::new(
                LimitType::LeverageLimit,
                None,
                FixedPoint::from_float(3.0),
                FixedPoint::from_float(2.5),
            ),
        ];
        
        let monitor = RealTimeRiskMonitor::new(limits).unwrap();
        assert_eq!(monitor.limits.len(), 2);
    }
    
    #[test]
    fn test_cached_metrics_update() {
        let metrics = CachedRiskMetrics::new();
        
        metrics.update_all(
            FixedPoint::from_float(100000.0),
            FixedPoint::from_float(5000.0),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(2000.0),
            FixedPoint::from_float(0.05),
        );
        
        let (portfolio_value, total_pnl, leverage, concentration, var, drawdown, _) = 
            (metrics.portfolio_value.load(Ordering::Acquire),
             metrics.total_pnl.load(Ordering::Acquire),
             metrics.leverage_ratio.load(Ordering::Acquire),
             metrics.max_position_concentration.load(Ordering::Acquire),
             metrics.var_estimate.load(Ordering::Acquire),
             metrics.current_drawdown.load(Ordering::Acquire),
             metrics.last_update.load(Ordering::Acquire));
        
        assert_eq!(portfolio_value, FixedPoint::from_float(100000.0));
        assert_eq!(total_pnl, FixedPoint::from_float(5000.0));
        assert_eq!(leverage, FixedPoint::from_float(2.0));
        assert_eq!(concentration, FixedPoint::from_float(0.3));
        assert_eq!(var, FixedPoint::from_float(2000.0));
        assert_eq!(drawdown, FixedPoint::from_float(0.05));
    }
    
    #[test]
    fn test_limit_breach_detection() {
        let limits = vec![
            RiskLimit::new(
                LimitType::LossLimit,
                None,
                FixedPoint::from_float(1000.0),
                FixedPoint::from_float(800.0),
            ),
        ];
        
        let monitor = RealTimeRiskMonitor::new(limits).unwrap();
        let mut portfolio = Portfolio::new(FixedPoint::from_float(10000.0));
        let mut market_data = MarketData::new();
        
        // Add a losing position
        let mut position = Position::new(
            "AAPL".to_string(),
            100,
            FixedPoint::from_float(150.0),
            FixedPoint::from_float(140.0), // $10 loss per share
        );
        position.unrealized_pnl = FixedPoint::from_float(-1500.0); // $1500 loss
        portfolio.add_position(position);
        
        market_data.add_price("AAPL".to_string(), FixedPoint::from_float(140.0));
        market_data.add_volatility("AAPL".to_string(), FixedPoint::from_float(0.2));
        
        let breaches = monitor.update_risk_metrics(&portfolio, &market_data).unwrap();
        assert!(!breaches.is_empty());
        assert_eq!(breaches[0].limit_type, LimitType::LossLimit);
        assert_eq!(breaches[0].severity, BreachSeverity::Critical);
    }
    
    #[test]
    fn test_flattening_action_generation() {
        let limits = vec![
            RiskLimit::new(
                LimitType::PositionLimit,
                Some("AAPL".to_string()),
                FixedPoint::from_float(10000.0),
                FixedPoint::from_float(8000.0),
            ),
        ];
        
        let monitor = RealTimeRiskMonitor::new(limits).unwrap();
        let mut portfolio = Portfolio::new(FixedPoint::from_float(50000.0));
        
        // Add a large position that exceeds the limit
        let position = Position::new(
            "AAPL".to_string(),
            100,
            FixedPoint::from_float(150.0),
            FixedPoint::from_float(150.0),
        );
        portfolio.add_position(position);
        
        let breach = LimitBreach {
            limit_type: LimitType::PositionLimit,
            asset_id: Some("AAPL".to_string()),
            current_value: FixedPoint::from_float(15000.0),
            threshold: FixedPoint::from_float(10000.0),
            severity: BreachSeverity::Critical,
            timestamp: 0,
            auto_flatten_required: true,
        };
        
        let actions = monitor.generate_flattening_actions(&[breach], &portfolio).unwrap();
        assert!(!actions.is_empty());
        assert_eq!(actions[0].asset_id, "AAPL");
        assert!(actions[0].target_quantity < actions[0].current_quantity);
    }
}