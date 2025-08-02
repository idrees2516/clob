//! Avellaneda-Stoikov Optimal Market Making Model
//!
//! This module implements the Avellaneda-Stoikov model for optimal market making,
//! which solves the Hamilton-Jacobi-Bellman (HJB) equation to find optimal bid-ask spreads
//! that maximize expected utility while managing inventory risk.
//!
//! The model provides:
//! - Closed-form solution for optimal spreads
//! - Reservation price calculation with inventory adjustment
//! - Market impact integration
//! - Dynamic parameter adjustment
//! - Adverse selection protection

use crate::math::{FixedPoint, DeterministicRng};
use crate::models::adverse_selection::{
    AdverseSelectionProtection, AdverseSelectionParams, AdverseSelectionState,
    TradeInfo, AdverseSelectionError
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Errors specific to the Avellaneda-Stoikov model
#[derive(Debug, Clone, PartialEq)]
pub enum AvellanedaStoikovError {
    /// Invalid model parameters
    InvalidParameters(String),
    /// Numerical instability detected
    NumericalInstability(String),
    /// Market data insufficient for calculation
    InsufficientData(String),
    /// Cache operation failed
    CacheError(String),
}

impl std::fmt::Display for AvellanedaStoikovError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AvellanedaStoikovError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            AvellanedaStoikovError::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            AvellanedaStoikovError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            AvellanedaStoikovError::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for AvellanedaStoikovError {}

/// Price type for market data
pub type Price = FixedPoint;

/// Volume type for order quantities
pub type Volume = i64;

/// Timestamp type for time-based calculations
pub type Timestamp = u64;

/// Asset identifier
pub type AssetId = u32;

/// Parameters for the Avellaneda-Stoikov model
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovParams {
    /// Risk aversion parameter (γ)
    pub gamma: FixedPoint,
    /// Market volatility (σ)
    pub sigma: FixedPoint,
    /// Market impact parameter (k)
    pub k: FixedPoint,
    /// Order arrival rate parameter (A)
    pub A: FixedPoint,
    /// Time horizon (T)
    pub T: FixedPoint,
    /// Minimum spread (in ticks)
    pub min_spread: FixedPoint,
    /// Maximum spread (in ticks)
    pub max_spread: FixedPoint,
    /// Tick size
    pub tick_size: FixedPoint,
}

impl Default for AvellanedaStoikovParams {
    fn default() -> Self {
        Self {
            gamma: FixedPoint::from_float(0.1),
            sigma: FixedPoint::from_float(0.2),
            k: FixedPoint::from_float(1.5),
            A: FixedPoint::from_float(140.0),
            T: FixedPoint::from_float(1.0),
            min_spread: FixedPoint::from_float(0.01),
            max_spread: FixedPoint::from_float(1.0),
            tick_size: FixedPoint::from_float(0.01),
        }
    }
}

impl AvellanedaStoikovParams {
    /// Validate parameters for mathematical consistency
    pub fn validate(&self) -> Result<(), AvellanedaStoikovError> {
        if self.gamma <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Risk aversion (gamma) must be positive".to_string()
            ));
        }
        
        if self.sigma <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Volatility (sigma) must be positive".to_string()
            ));
        }
        
        if self.k <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Market impact parameter (k) must be positive".to_string()
            ));
        }
        
        if self.A <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Order arrival rate (A) must be positive".to_string()
            ));
        }
        
        if self.T <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Time horizon (T) must be positive".to_string()
            ));
        }
        
        if self.min_spread >= self.max_spread {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Minimum spread must be less than maximum spread".to_string()
            ));
        }
        
        if self.tick_size <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Tick size must be positive".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Market state information required for quote calculation
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current mid price
    pub mid_price: Price,
    /// Current bid price
    pub bid_price: Price,
    /// Current ask price
    pub ask_price: Price,
    /// Bid volume
    pub bid_volume: Volume,
    /// Ask volume
    pub ask_volume: Volume,
    /// Last trade price
    pub last_trade_price: Price,
    /// Last trade volume
    pub last_trade_volume: Volume,
    /// Market timestamp
    pub timestamp: Timestamp,
    /// Sequence number for ordering
    pub sequence_number: u64,
    /// Current volatility estimate
    pub volatility: FixedPoint,
    /// Order flow imbalance
    pub order_flow_imbalance: FixedPoint,
    /// Microstructure noise estimate
    pub microstructure_noise: FixedPoint,
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: 1000,
            ask_volume: 1000,
            last_trade_price: FixedPoint::from_float(100.0),
            last_trade_volume: 100,
            timestamp: 0,
            sequence_number: 0,
            volatility: FixedPoint::from_float(0.2),
            order_flow_imbalance: FixedPoint::ZERO,
            microstructure_noise: FixedPoint::from_float(0.001),
        }
    }
}

/// Optimal quotes calculated by the Avellaneda-Stoikov model
#[derive(Debug, Clone)]
pub struct OptimalQuotes {
    /// Optimal bid price
    pub bid_price: Price,
    /// Optimal ask price
    pub ask_price: Price,
    /// Recommended bid size
    pub bid_size: Volume,
    /// Recommended ask size
    pub ask_size: Volume,
    /// Reservation price
    pub reservation_price: Price,
    /// Optimal spread (half-spread)
    pub optimal_spread: FixedPoint,
    /// Inventory skew adjustment
    pub inventory_skew: FixedPoint,
    /// Market impact adjustment
    pub market_impact_adjustment: FixedPoint,
    /// Adverse selection premium
    pub adverse_selection_premium: FixedPoint,
    /// Timestamp when quotes were calculated
    pub timestamp: Timestamp,
    /// Confidence score (0-1)
    pub confidence: FixedPoint,
}

/// Cache key for quote calculations
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct QuoteKey {
    mid_price_hash: u64,
    inventory: i64,
    volatility_hash: u64,
    time_to_maturity_hash: u64,
}

impl QuoteKey {
    fn new(mid_price: Price, inventory: i64, volatility: FixedPoint, time_to_maturity: FixedPoint) -> Self {
        Self {
            mid_price_hash: mid_price.to_bits(),
            inventory,
            volatility_hash: volatility.to_bits(),
            time_to_maturity_hash: time_to_maturity.to_bits(),
        }
    }
}

/// Market impact model parameters
#[derive(Debug, Clone)]
pub struct MarketImpactParams {
    /// Temporary impact coefficient (η)
    pub eta: FixedPoint,
    /// Temporary impact exponent (α)
    pub alpha: FixedPoint,
    /// Permanent impact coefficient (λ)
    pub lambda: FixedPoint,
    /// Cross-impact coefficient for correlated assets
    pub cross_impact_coeff: FixedPoint,
    /// Impact decay rate (for temporary impact)
    pub decay_rate: FixedPoint,
    /// Minimum participation rate threshold
    pub min_participation_rate: FixedPoint,
    /// Maximum participation rate threshold
    pub max_participation_rate: FixedPoint,
}

impl Default for MarketImpactParams {
    fn default() -> Self {
        Self {
            eta: FixedPoint::from_float(0.1),
            alpha: FixedPoint::from_float(0.5),
            lambda: FixedPoint::from_float(0.01),
            cross_impact_coeff: FixedPoint::from_float(0.05),
            decay_rate: FixedPoint::from_float(0.1),
            min_participation_rate: FixedPoint::from_float(0.01),
            max_participation_rate: FixedPoint::from_float(0.3),
        }
    }
}

impl MarketImpactParams {
    /// Validate market impact parameters
    pub fn validate(&self) -> Result<(), AvellanedaStoikovError> {
        if self.eta <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Temporary impact coefficient (eta) must be positive".to_string()
            ));
        }
        
        if self.alpha <= FixedPoint::ZERO || self.alpha > FixedPoint::ONE {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Impact exponent (alpha) must be in (0, 1]".to_string()
            ));
        }
        
        if self.lambda < FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Permanent impact coefficient (lambda) must be non-negative".to_string()
            ));
        }
        
        if self.decay_rate <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Decay rate must be positive".to_string()
            ));
        }
        
        if self.min_participation_rate >= self.max_participation_rate {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Minimum participation rate must be less than maximum".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Transaction cost analysis results
#[derive(Debug, Clone)]
pub struct TransactionCostAnalysis {
    /// Expected temporary impact cost
    pub temporary_impact_cost: FixedPoint,
    /// Expected permanent impact cost
    pub permanent_impact_cost: FixedPoint,
    /// Total expected impact cost
    pub total_impact_cost: FixedPoint,
    /// Optimal participation rate
    pub optimal_participation_rate: FixedPoint,
    /// Expected execution time
    pub expected_execution_time: FixedPoint,
    /// Risk-adjusted cost (including timing risk)
    pub risk_adjusted_cost: FixedPoint,
}

/// Market impact calculator for sophisticated impact modeling
#[derive(Debug)]
pub struct MarketImpactCalculator {
    /// Impact parameters
    params: MarketImpactParams,
    /// Historical impact measurements
    impact_history: Vec<(FixedPoint, FixedPoint, Timestamp)>, // (participation_rate, realized_impact, timestamp)
    /// Maximum history length
    max_history: usize,
}

impl MarketImpactCalculator {
    pub fn new(params: MarketImpactParams, max_history: usize) -> Result<Self, AvellanedaStoikovError> {
        params.validate()?;
        
        Ok(Self {
            params,
            impact_history: Vec::with_capacity(max_history),
            max_history,
        })
    }
    
    /// Calculate temporary impact: I_temp(v) = η*v^α
    pub fn calculate_temporary_impact(&self, participation_rate: FixedPoint) -> FixedPoint {
        if participation_rate <= FixedPoint::ZERO {
            return FixedPoint::ZERO;
        }
        
        // Clamp participation rate to reasonable bounds
        let clamped_rate = participation_rate
            .max(self.params.min_participation_rate)
            .min(self.params.max_participation_rate);
        
        self.params.eta * clamped_rate.powf(self.params.alpha.to_float())
    }
    
    /// Calculate permanent impact: I_perm(x) = λ*x
    pub fn calculate_permanent_impact(&self, executed_quantity: FixedPoint) -> FixedPoint {
        self.params.lambda * executed_quantity.abs()
    }
    
    /// Calculate combined impact with cross-asset effects
    pub fn calculate_combined_impact(
        &self,
        participation_rate: FixedPoint,
        executed_quantity: FixedPoint,
        cross_asset_volume: FixedPoint,
        correlation: FixedPoint,
    ) -> FixedPoint {
        let temp_impact = self.calculate_temporary_impact(participation_rate);
        let perm_impact = self.calculate_permanent_impact(executed_quantity);
        
        // Cross-asset impact: accounts for correlated trading
        let cross_impact = self.params.cross_impact_coeff * 
            cross_asset_volume * 
            correlation.abs() * 
            participation_rate.sqrt();
        
        temp_impact + perm_impact + cross_impact
    }
    
    /// Optimize participation rate to minimize total cost
    pub fn optimize_participation_rate(
        &self,
        total_quantity: FixedPoint,
        time_horizon: FixedPoint,
        volatility: FixedPoint,
        risk_aversion: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        if total_quantity <= FixedPoint::ZERO || time_horizon <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Quantity and time horizon must be positive".to_string()
            ));
        }
        
        // Use analytical solution for square-root law
        // Optimal rate minimizes: η*v^α + (σ²/2γ) * (Q/T)² / v
        // For α = 0.5, this gives: v* = (σ²*Q²/(2γ*T²*η))^(1/3)
        
        let timing_risk_coeff = volatility * volatility / (FixedPoint::from_float(2.0) * risk_aversion);
        let quantity_per_time = total_quantity / time_horizon;
        let quantity_squared = quantity_per_time * quantity_per_time;
        
        // Simplified optimization for α = 0.5 case
        if (self.params.alpha - FixedPoint::from_float(0.5)).abs() < FixedPoint::from_float(0.01) {
            let numerator = timing_risk_coeff * quantity_squared;
            let denominator = self.params.eta;
            let optimal_rate = (numerator / denominator).powf(1.0 / 3.0);
            
            // Clamp to bounds
            let clamped_rate = optimal_rate
                .max(self.params.min_participation_rate)
                .min(self.params.max_participation_rate);
            
            return Ok(clamped_rate);
        }
        
        // For general α, use numerical optimization (simplified grid search)
        let mut best_rate = self.params.min_participation_rate;
        let mut best_cost = FixedPoint::from_float(f64::INFINITY);
        
        let num_points = 100;
        let rate_step = (self.params.max_participation_rate - self.params.min_participation_rate) / 
            FixedPoint::from_int(num_points);
        
        for i in 0..=num_points {
            let rate = self.params.min_participation_rate + FixedPoint::from_int(i) * rate_step;
            
            // Calculate total cost: impact cost + timing risk
            let impact_cost = self.calculate_temporary_impact(rate) * total_quantity;
            let execution_time = total_quantity / rate;
            let timing_cost = timing_risk_coeff * execution_time;
            let total_cost = impact_cost + timing_cost;
            
            if total_cost < best_cost {
                best_cost = total_cost;
                best_rate = rate;
            }
        }
        
        Ok(best_rate)
    }
    
    /// Perform comprehensive transaction cost analysis
    pub fn analyze_transaction_costs(
        &self,
        total_quantity: FixedPoint,
        time_horizon: FixedPoint,
        volatility: FixedPoint,
        risk_aversion: FixedPoint,
    ) -> Result<TransactionCostAnalysis, AvellanedaStoikovError> {
        let optimal_rate = self.optimize_participation_rate(
            total_quantity, 
            time_horizon, 
            volatility, 
            risk_aversion
        )?;
        
        let temporary_impact_cost = self.calculate_temporary_impact(optimal_rate) * total_quantity;
        let permanent_impact_cost = self.calculate_permanent_impact(total_quantity);
        let total_impact_cost = temporary_impact_cost + permanent_impact_cost;
        
        let expected_execution_time = total_quantity / optimal_rate;
        
        // Risk-adjusted cost includes timing risk
        let timing_risk = volatility * volatility * expected_execution_time / 
            (FixedPoint::from_float(2.0) * risk_aversion);
        let risk_adjusted_cost = total_impact_cost + timing_risk;
        
        Ok(TransactionCostAnalysis {
            temporary_impact_cost,
            permanent_impact_cost,
            total_impact_cost,
            optimal_participation_rate: optimal_rate,
            expected_execution_time,
            risk_adjusted_cost,
        })
    }
    
    /// Update impact model with realized impact measurement
    pub fn update_with_realized_impact(
        &mut self,
        participation_rate: FixedPoint,
        realized_impact: FixedPoint,
        timestamp: Timestamp,
    ) {
        self.impact_history.push((participation_rate, realized_impact, timestamp));
        
        // Maintain history size
        if self.impact_history.len() > self.max_history {
            self.impact_history.remove(0);
        }
    }
    
    /// Calibrate impact parameters using historical data
    pub fn calibrate_parameters(&mut self) -> Result<(), AvellanedaStoikovError> {
        if self.impact_history.len() < 10 {
            return Err(AvellanedaStoikovError::InsufficientData(
                "Need at least 10 historical observations for calibration".to_string()
            ));
        }
        
        // Simple linear regression to estimate η and α
        // log(impact) = log(η) + α * log(participation_rate)
        
        let mut sum_log_rate = FixedPoint::ZERO;
        let mut sum_log_impact = FixedPoint::ZERO;
        let mut sum_log_rate_squared = FixedPoint::ZERO;
        let mut sum_log_rate_impact = FixedPoint::ZERO;
        let n = FixedPoint::from_int(self.impact_history.len() as i64);
        
        for (rate, impact, _) in &self.impact_history {
            if *rate > FixedPoint::ZERO && *impact > FixedPoint::ZERO {
                let log_rate = rate.ln();
                let log_impact = impact.ln();
                
                sum_log_rate = sum_log_rate + log_rate;
                sum_log_impact = sum_log_impact + log_impact;
                sum_log_rate_squared = sum_log_rate_squared + log_rate * log_rate;
                sum_log_rate_impact = sum_log_rate_impact + log_rate * log_impact;
            }
        }
        
        // Calculate regression coefficients
        let denominator = n * sum_log_rate_squared - sum_log_rate * sum_log_rate;
        if denominator.abs() < FixedPoint::from_float(1e-10) {
            return Err(AvellanedaStoikovError::NumericalInstability(
                "Cannot calibrate: insufficient variation in participation rates".to_string()
            ));
        }
        
        let alpha = (n * sum_log_rate_impact - sum_log_rate * sum_log_impact) / denominator;
        let log_eta = (sum_log_impact - alpha * sum_log_rate) / n;
        let eta = log_eta.exp();
        
        // Validate calibrated parameters
        if alpha > FixedPoint::ZERO && alpha <= FixedPoint::ONE && eta > FixedPoint::ZERO {
            self.params.alpha = alpha;
            self.params.eta = eta;
        }
        
        Ok(())
    }
    
    /// Get current impact parameters
    pub fn get_parameters(&self) -> &MarketImpactParams {
        &self.params
    }
    
    /// Update impact parameters
    pub fn update_parameters(&mut self, new_params: MarketImpactParams) -> Result<(), AvellanedaStoikovError> {
        new_params.validate()?;
        self.params = new_params;
        Ok(())
    }
}

/// Inventory tracker for position management
#[derive(Debug, Clone)]
pub struct InventoryTracker {
    /// Current inventory position
    pub position: i64,
    /// Average entry price
    pub average_price: Price,
    /// Unrealized P&L
    pub unrealized_pnl: FixedPoint,
    /// Realized P&L
    pub realized_pnl: FixedPoint,
    /// Last update timestamp
    pub last_update: Timestamp,
}

impl Default for InventoryTracker {
    fn default() -> Self {
        Self {
            position: 0,
            average_price: FixedPoint::ZERO,
            unrealized_pnl: FixedPoint::ZERO,
            realized_pnl: FixedPoint::ZERO,
            last_update: 0,
        }
    }
}

/// Realized volatility estimator with advanced features
#[derive(Debug)]
pub struct RealizedVolatilityEstimator {
    /// Price history for volatility calculation
    price_history: Vec<Price>,
    /// Return history
    return_history: Vec<FixedPoint>,
    /// Timestamp history for proper time-weighting
    timestamp_history: Vec<Timestamp>,
    /// Maximum history length
    max_history: usize,
    /// Current volatility estimate
    current_volatility: FixedPoint,
    /// EWMA decay factor
    decay_factor: FixedPoint,
    /// Intraday volatility pattern (24 hourly buckets)
    intraday_pattern: Vec<FixedPoint>,
    /// Regime-dependent volatility estimates
    regime_volatilities: HashMap<MarketRegime, FixedPoint>,
    /// Current market regime
    current_regime: MarketRegime,
    /// Rough volatility parameters
    rough_vol_enabled: bool,
    hurst_parameter: FixedPoint,
    /// Volatility forecasting horizon (in seconds)
    forecast_horizon: u64,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    Crisis,
    LowVolatility,
}

impl Default for MarketRegime {
    fn default() -> Self {
        MarketRegime::Normal
    }
}

impl RealizedVolatilityEstimator {
    pub fn new(max_history: usize, decay_factor: FixedPoint) -> Self {
        let mut regime_volatilities = HashMap::new();
        regime_volatilities.insert(MarketRegime::Normal, FixedPoint::from_float(0.2));
        regime_volatilities.insert(MarketRegime::HighVolatility, FixedPoint::from_float(0.4));
        regime_volatilities.insert(MarketRegime::Crisis, FixedPoint::from_float(0.8));
        regime_volatilities.insert(MarketRegime::LowVolatility, FixedPoint::from_float(0.1));
        
        Self {
            price_history: Vec::with_capacity(max_history),
            return_history: Vec::with_capacity(max_history),
            timestamp_history: Vec::with_capacity(max_history),
            max_history,
            current_volatility: FixedPoint::from_float(0.2),
            decay_factor,
            intraday_pattern: vec![FixedPoint::ONE; 24], // Initialize with neutral pattern
            regime_volatilities,
            current_regime: MarketRegime::Normal,
            rough_vol_enabled: false,
            hurst_parameter: FixedPoint::from_float(0.1), // Rough volatility Hurst parameter
            forecast_horizon: 300, // 5 minutes default
        }
    }
    
    /// Create estimator with rough volatility enabled
    pub fn new_with_rough_volatility(
        max_history: usize, 
        decay_factor: FixedPoint,
        hurst_parameter: FixedPoint,
    ) -> Self {
        let mut estimator = Self::new(max_history, decay_factor);
        estimator.rough_vol_enabled = true;
        estimator.hurst_parameter = hurst_parameter;
        estimator
    }
    
    /// Update volatility estimate with new price and timestamp
    pub fn update(&mut self, price: Price, timestamp: Timestamp) -> FixedPoint {
        self.price_history.push(price);
        self.timestamp_history.push(timestamp);
        
        if self.price_history.len() > 1 {
            let prev_price = self.price_history[self.price_history.len() - 2];
            let prev_timestamp = self.timestamp_history[self.timestamp_history.len() - 2];
            
            // Calculate time-adjusted return
            let time_diff = (timestamp - prev_timestamp) as f64 / 1_000_000_000.0; // Convert to seconds
            let log_return = (price / prev_price).ln();
            
            // Annualize return based on time difference
            let annualized_return = if time_diff > 0.0 {
                log_return / FixedPoint::from_float(time_diff.sqrt())
            } else {
                log_return
            };
            
            self.return_history.push(annualized_return);
            
            // Update current volatility using EWMA
            self.update_ewma_volatility(annualized_return);
            
            // Update intraday pattern
            self.update_intraday_pattern(timestamp, annualized_return);
            
            // Detect and update market regime
            self.update_market_regime();
            
            // Apply rough volatility adjustment if enabled
            if self.rough_vol_enabled {
                self.apply_rough_volatility_adjustment();
            }
        }
        
        // Maintain history size
        self.maintain_history_size();
        
        self.get_regime_adjusted_volatility()
    }
    
    /// Update EWMA volatility estimate
    fn update_ewma_volatility(&mut self, return_value: FixedPoint) {
        let squared_return = return_value * return_value;
        self.current_volatility = self.decay_factor * squared_return + 
            (FixedPoint::ONE - self.decay_factor) * self.current_volatility;
    }
    
    /// Update intraday volatility pattern
    fn update_intraday_pattern(&mut self, timestamp: Timestamp, return_value: FixedPoint) {
        // Extract hour of day from timestamp (simplified - assumes UTC)
        let seconds_in_day = 86400;
        let hour = ((timestamp / 1_000_000_000) % seconds_in_day) / 3600;
        let hour_index = (hour as usize).min(23);
        
        let squared_return = return_value * return_value;
        let pattern_decay = FixedPoint::from_float(0.99); // Slower decay for pattern
        
        self.intraday_pattern[hour_index] = pattern_decay * self.intraday_pattern[hour_index] +
            (FixedPoint::ONE - pattern_decay) * squared_return;
    }
    
    /// Detect and update market regime based on recent volatility
    fn update_market_regime(&mut self) {
        if self.return_history.len() < 20 {
            return; // Need sufficient history
        }
        
        // Calculate recent volatility (last 20 observations)
        let recent_returns = &self.return_history[self.return_history.len().saturating_sub(20)..];
        let recent_vol_squared: FixedPoint = recent_returns.iter()
            .map(|r| r * r)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_returns.len() as i64);
        
        let recent_vol = recent_vol_squared.sqrt();
        
        // Regime thresholds
        let crisis_threshold = FixedPoint::from_float(0.6);
        let high_vol_threshold = FixedPoint::from_float(0.3);
        let low_vol_threshold = FixedPoint::from_float(0.1);
        
        self.current_regime = if recent_vol > crisis_threshold {
            MarketRegime::Crisis
        } else if recent_vol > high_vol_threshold {
            MarketRegime::HighVolatility
        } else if recent_vol < low_vol_threshold {
            MarketRegime::LowVolatility
        } else {
            MarketRegime::Normal
        };
    }
    
    /// Apply rough volatility adjustment using fractional integration
    fn apply_rough_volatility_adjustment(&mut self) {
        if self.return_history.len() < 10 {
            return;
        }
        
        // Simplified rough volatility: σ²ₜ = σ²ₜ₋₁ + ν*ΔWᴴₜ
        // Where ΔWᴴₜ is fractional Brownian motion increment
        
        let nu = FixedPoint::from_float(0.1); // Volatility of volatility
        let recent_returns = &self.return_history[self.return_history.len().saturating_sub(10)..];
        
        // Calculate fractional increment (simplified)
        let fractional_increment = recent_returns.iter()
            .enumerate()
            .map(|(i, r)| {
                let weight = FixedPoint::from_float((i + 1) as f64).powf(-self.hurst_parameter.to_float());
                weight * r
            })
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_returns.len() as i64);
        
        // Apply rough volatility adjustment
        let rough_adjustment = nu * fractional_increment;
        self.current_volatility = (self.current_volatility + rough_adjustment).max(FixedPoint::from_float(0.01));
    }
    
    /// Get regime-adjusted volatility estimate
    fn get_regime_adjusted_volatility(&self) -> FixedPoint {
        let base_vol = self.current_volatility.sqrt();
        let regime_multiplier = match self.current_regime {
            MarketRegime::Crisis => FixedPoint::from_float(1.5),
            MarketRegime::HighVolatility => FixedPoint::from_float(1.2),
            MarketRegime::LowVolatility => FixedPoint::from_float(0.8),
            MarketRegime::Normal => FixedPoint::ONE,
        };
        
        base_vol * regime_multiplier
    }
    
    /// Maintain history size limits
    fn maintain_history_size(&mut self) {
        if self.price_history.len() > self.max_history {
            self.price_history.remove(0);
        }
        if self.return_history.len() > self.max_history {
            self.return_history.remove(0);
        }
        if self.timestamp_history.len() > self.max_history {
            self.timestamp_history.remove(0);
        }
    }
    
    /// Get current volatility estimate
    pub fn get_volatility(&self) -> FixedPoint {
        self.get_regime_adjusted_volatility()
    }
    
    /// Get volatility forecast for specified horizon
    pub fn forecast_volatility(&self, horizon_seconds: u64) -> FixedPoint {
        let base_vol = self.get_volatility();
        
        // Apply time scaling: vol scales with sqrt(time)
        let time_scaling = FixedPoint::from_float((horizon_seconds as f64 / self.forecast_horizon as f64).sqrt());
        
        // Apply intraday pattern if forecasting within day
        let pattern_adjustment = if horizon_seconds < 86400 {
            self.get_intraday_adjustment(horizon_seconds)
        } else {
            FixedPoint::ONE
        };
        
        base_vol * time_scaling * pattern_adjustment
    }
    
    /// Get intraday volatility adjustment
    fn get_intraday_adjustment(&self, horizon_seconds: u64) -> FixedPoint {
        let hour = (horizon_seconds / 3600) as usize;
        if hour < self.intraday_pattern.len() {
            self.intraday_pattern[hour].sqrt()
        } else {
            FixedPoint::ONE
        }
    }
    
    /// Get current market regime
    pub fn get_market_regime(&self) -> MarketRegime {
        self.current_regime
    }
    
    /// Get regime-specific volatility
    pub fn get_regime_volatility(&self, regime: MarketRegime) -> FixedPoint {
        self.regime_volatilities.get(&regime).copied()
            .unwrap_or(FixedPoint::from_float(0.2))
    }
    
    /// Update regime-specific volatility
    pub fn update_regime_volatility(&mut self, regime: MarketRegime, volatility: FixedPoint) {
        self.regime_volatilities.insert(regime, volatility);
    }
    
    /// Calculate realized volatility over specified window
    pub fn calculate_realized_volatility(&self, window_size: usize) -> FixedPoint {
        if self.return_history.len() < window_size {
            return self.current_volatility.sqrt();
        }
        
        let start_idx = self.return_history.len() - window_size;
        let window_returns = &self.return_history[start_idx..];
        
        let sum_squared: FixedPoint = window_returns.iter()
            .map(|r| r * r)
            .sum();
        
        (sum_squared / FixedPoint::from_int(window_size as i64)).sqrt()
    }
    
    /// Enable/disable rough volatility modeling
    pub fn set_rough_volatility(&mut self, enabled: bool, hurst_parameter: Option<FixedPoint>) {
        self.rough_vol_enabled = enabled;
        if let Some(h) = hurst_parameter {
            self.hurst_parameter = h;
        }
    }
}

/// Quote cache for performance optimization
type QuoteCache = HashMap<QuoteKey, (OptimalQuotes, Timestamp)>;

/// Dynamic parameter adjustment configuration
#[derive(Debug, Clone)]
pub struct DynamicParameterConfig {
    /// Enable adaptive risk aversion
    pub adaptive_risk_aversion: bool,
    /// Base risk aversion parameter
    pub base_gamma: FixedPoint,
    /// Risk aversion adjustment range [min_multiplier, max_multiplier]
    pub gamma_range: (FixedPoint, FixedPoint),
    /// Enable urgency factor modeling
    pub urgency_factor_enabled: bool,
    /// Urgency factor exponent (α ∈ [0.5, 1])
    pub urgency_alpha: FixedPoint,
    /// Minimum time to maturity for urgency (seconds)
    pub min_time_to_maturity: FixedPoint,
    /// Enable parameter validation and stability checks
    pub stability_checks_enabled: bool,
    /// Maximum allowed gamma for stability
    pub max_gamma: FixedPoint,
    /// Quote update frequency adjustment
    pub dynamic_update_frequency: bool,
    /// Base update interval (nanoseconds)
    pub base_update_interval: u64,
}

impl Default for DynamicParameterConfig {
    fn default() -> Self {
        Self {
            adaptive_risk_aversion: true,
            base_gamma: FixedPoint::from_float(0.1),
            gamma_range: (FixedPoint::from_float(0.01), FixedPoint::from_float(1.0)),
            urgency_factor_enabled: true,
            urgency_alpha: FixedPoint::from_float(0.7),
            min_time_to_maturity: FixedPoint::from_float(1.0), // 1 second
            stability_checks_enabled: true,
            max_gamma: FixedPoint::from_float(10.0),
            dynamic_update_frequency: true,
            base_update_interval: 1_000_000, // 1ms in nanoseconds
        }
    }
}

/// Market condition indicators for adaptive parameter adjustment
#[derive(Debug, Clone)]
pub struct MarketConditionIndicators {
    /// Current volatility level
    pub volatility: FixedPoint,
    /// Volatility regime
    pub volatility_regime: MarketRegime,
    /// Order flow imbalance
    pub order_flow_imbalance: FixedPoint,
    /// Bid-ask spread tightness
    pub spread_tightness: FixedPoint,
    /// Market depth
    pub market_depth: FixedPoint,
    /// Recent price momentum
    pub price_momentum: FixedPoint,
    /// Liquidity stress indicator
    pub liquidity_stress: FixedPoint,
    /// Time of day factor
    pub time_of_day_factor: FixedPoint,
}

impl Default for MarketConditionIndicators {
    fn default() -> Self {
        Self {
            volatility: FixedPoint::from_float(0.2),
            volatility_regime: MarketRegime::Normal,
            order_flow_imbalance: FixedPoint::ZERO,
            spread_tightness: FixedPoint::ONE,
            market_depth: FixedPoint::ONE,
            price_momentum: FixedPoint::ZERO,
            liquidity_stress: FixedPoint::ZERO,
            time_of_day_factor: FixedPoint::ONE,
        }
    }
}

/// Main Avellaneda-Stoikov engine with dynamic parameter adjustment
pub struct AvellanedaStoikovEngine {
    /// Model parameters
    params: AvellanedaStoikovParams,
    /// Market impact calculator
    market_impact_calculator: MarketImpactCalculator,
    /// Inventory tracker
    inventory_tracker: InventoryTracker,
    /// Volatility estimator
    volatility_estimator: RealizedVolatilityEstimator,
    /// Quote cache for performance
    quote_cache: QuoteCache,
    /// Cache TTL in nanoseconds
    cache_ttl: u64,
    /// Performance metrics
    calculation_count: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    /// Dynamic parameter adjustment configuration
    dynamic_config: DynamicParameterConfig,
    /// Current market condition indicators
    market_conditions: MarketConditionIndicators,
    /// Last parameter update timestamp
    last_parameter_update: AtomicU64,
    /// Current effective gamma (after adjustments)
    effective_gamma: FixedPoint,
    /// Parameter adjustment history for stability monitoring
    gamma_history: Vec<(FixedPoint, Timestamp)>,
    /// Maximum parameter history length
    max_param_history: usize,
    /// Adverse selection protection engine
    adverse_selection_protection: AdverseSelectionProtection,
}

impl AvellanedaStoikovEngine {
    /// Create a new Avellaneda-Stoikov engine
    pub fn new(params: AvellanedaStoikovParams) -> Result<Self, AvellanedaStoikovError> {
        params.validate()?;
        
        let market_impact_calculator = MarketImpactCalculator::new(
            MarketImpactParams::default(),
            1000 // max history
        )?;
        
        let dynamic_config = DynamicParameterConfig::default();
        let effective_gamma = params.gamma;
        
        // Initialize adverse selection protection with default parameters
        let adverse_selection_params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0); // 10 quotes per second default
        let adverse_selection_protection = AdverseSelectionProtection::new(
            adverse_selection_params, 
            base_frequency
        ).map_err(|e| AvellanedaStoikovError::InvalidParameters(format!("Adverse selection error: {}", e)))?;
        
        Ok(Self {
            params,
            market_impact_calculator,
            inventory_tracker: InventoryTracker::default(),
            volatility_estimator: RealizedVolatilityEstimator::new(1000, FixedPoint::from_float(0.94)),
            quote_cache: HashMap::new(),
            cache_ttl: 1_000_000, // 1ms in nanoseconds
            calculation_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            dynamic_config,
            market_conditions: MarketConditionIndicators::default(),
            last_parameter_update: AtomicU64::new(0),
            effective_gamma,
            gamma_history: Vec::new(),
            max_param_history: 1000,
            adverse_selection_protection,
        })
    }
    
    /// Create a new Avellaneda-Stoikov engine with dynamic parameter configuration
    pub fn new_with_dynamic_config(
        params: AvellanedaStoikovParams,
        dynamic_config: DynamicParameterConfig,
    ) -> Result<Self, AvellanedaStoikovError> {
        params.validate()?;
        
        let market_impact_calculator = MarketImpactCalculator::new(
            MarketImpactParams::default(),
            1000 // max history
        )?;
        
        let effective_gamma = params.gamma;
        
        // Initialize adverse selection protection with default parameters
        let adverse_selection_params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0); // 10 quotes per second default
        let adverse_selection_protection = AdverseSelectionProtection::new(
            adverse_selection_params, 
            base_frequency
        ).map_err(|e| AvellanedaStoikovError::InvalidParameters(format!("Adverse selection error: {}", e)))?;
        
        Ok(Self {
            params,
            market_impact_calculator,
            inventory_tracker: InventoryTracker::default(),
            volatility_estimator: RealizedVolatilityEstimator::new(1000, FixedPoint::from_float(0.94)),
            quote_cache: HashMap::new(),
            cache_ttl: 1_000_000, // 1ms in nanoseconds
            calculation_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            dynamic_config,
            market_conditions: MarketConditionIndicators::default(),
            last_parameter_update: AtomicU64::new(0),
            effective_gamma,
            gamma_history: Vec::new(),
            max_param_history: 1000,
            adverse_selection_protection,
        })
    }
    
    /// Create a new Avellaneda-Stoikov engine with custom market impact parameters
    pub fn new_with_impact_params(
        params: AvellanedaStoikovParams,
        impact_params: MarketImpactParams,
    ) -> Result<Self, AvellanedaStoikovError> {
        params.validate()?;
        impact_params.validate()?;
        
        let market_impact_calculator = MarketImpactCalculator::new(impact_params, 1000)?;
        let dynamic_config = DynamicParameterConfig::default();
        let effective_gamma = params.gamma;
        
        // Initialize adverse selection protection with default parameters
        let adverse_selection_params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0); // 10 quotes per second default
        let adverse_selection_protection = AdverseSelectionProtection::new(
            adverse_selection_params, 
            base_frequency
        ).map_err(|e| AvellanedaStoikovError::InvalidParameters(format!("Adverse selection error: {}", e)))?;
        
        Ok(Self {
            params,
            market_impact_calculator,
            inventory_tracker: InventoryTracker::default(),
            volatility_estimator: RealizedVolatilityEstimator::new(1000, FixedPoint::from_float(0.94)),
            quote_cache: HashMap::new(),
            cache_ttl: 1_000_000, // 1ms in nanoseconds
            calculation_count: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            dynamic_config,
            market_conditions: MarketConditionIndicators::default(),
            last_parameter_update: AtomicU64::new(0),
            effective_gamma,
            gamma_history: Vec::new(),
            max_param_history: 1000,
            adverse_selection_protection,
        })
    }
    
    /// Update market condition indicators for dynamic parameter adjustment
    pub fn update_market_conditions(&mut self, market_state: &MarketState) {
        // Update volatility from estimator
        self.market_conditions.volatility = self.volatility_estimator.get_volatility();
        self.market_conditions.volatility_regime = self.volatility_estimator.get_market_regime();
        
        // Update order flow imbalance
        self.market_conditions.order_flow_imbalance = market_state.order_flow_imbalance;
        
        // Calculate spread tightness (inverse of relative spread)
        let spread = market_state.ask_price - market_state.bid_price;
        let relative_spread = spread / market_state.mid_price;
        self.market_conditions.spread_tightness = FixedPoint::ONE / (FixedPoint::ONE + relative_spread);
        
        // Calculate market depth indicator
        let total_volume = market_state.bid_volume + market_state.ask_volume;
        self.market_conditions.market_depth = FixedPoint::from_int(total_volume).ln().max(FixedPoint::ONE);
        
        // Calculate price momentum (simplified)
        if let Some(last_price) = self.volatility_estimator.price_history.last() {
            let momentum = (market_state.mid_price / last_price - FixedPoint::ONE) * FixedPoint::from_float(100.0);
            self.market_conditions.price_momentum = momentum;
        }
        
        // Calculate liquidity stress indicator
        let vol_stress = if self.market_conditions.volatility > FixedPoint::from_float(0.3) {
            (self.market_conditions.volatility - FixedPoint::from_float(0.3)) / FixedPoint::from_float(0.3)
        } else {
            FixedPoint::ZERO
        };
        
        let spread_stress = if relative_spread > FixedPoint::from_float(0.01) {
            (relative_spread - FixedPoint::from_float(0.01)) / FixedPoint::from_float(0.01)
        } else {
            FixedPoint::ZERO
        };
        
        self.market_conditions.liquidity_stress = (vol_stress + spread_stress) / FixedPoint::from_float(2.0);
        
        // Time of day factor (simplified - assumes market hours)
        let hour = ((market_state.timestamp / 1_000_000_000) % 86400) / 3600;
        self.market_conditions.time_of_day_factor = match hour {
            9..=11 | 14..=16 => FixedPoint::from_float(1.2), // High activity periods
            12..=13 => FixedPoint::from_float(0.8), // Lunch period
            _ => FixedPoint::ONE,
        };
    }
    
    /// Calculate adaptive risk aversion based on market conditions
    pub fn calculate_adaptive_risk_aversion(&self, time_to_maturity: FixedPoint) -> FixedPoint {
        if !self.dynamic_config.adaptive_risk_aversion {
            return self.params.gamma;
        }
        
        let mut gamma_multiplier = FixedPoint::ONE;
        
        // Volatility adjustment: increase risk aversion in high volatility
        match self.market_conditions.volatility_regime {
            MarketRegime::Crisis => gamma_multiplier = gamma_multiplier * FixedPoint::from_float(2.0),
            MarketRegime::HighVolatility => gamma_multiplier = gamma_multiplier * FixedPoint::from_float(1.5),
            MarketRegime::LowVolatility => gamma_multiplier = gamma_multiplier * FixedPoint::from_float(0.8),
            MarketRegime::Normal => {},
        }
        
        // Liquidity stress adjustment
        let stress_adjustment = FixedPoint::ONE + self.market_conditions.liquidity_stress;
        gamma_multiplier = gamma_multiplier * stress_adjustment;
        
        // Order flow imbalance adjustment
        let imbalance_adjustment = FixedPoint::ONE + self.market_conditions.order_flow_imbalance.abs() * FixedPoint::from_float(0.5);
        gamma_multiplier = gamma_multiplier * imbalance_adjustment;
        
        // Time of day adjustment
        gamma_multiplier = gamma_multiplier * self.market_conditions.time_of_day_factor;
        
        // Apply urgency factor if enabled
        if self.dynamic_config.urgency_factor_enabled {
            let urgency_factor = self.calculate_urgency_factor(time_to_maturity);
            gamma_multiplier = gamma_multiplier * urgency_factor;
        }
        
        // Calculate adjusted gamma
        let adjusted_gamma = self.dynamic_config.base_gamma * gamma_multiplier;
        
        // Apply bounds
        let bounded_gamma = adjusted_gamma
            .max(self.dynamic_config.gamma_range.0)
            .min(self.dynamic_config.gamma_range.1);
        
        // Apply stability check
        if self.dynamic_config.stability_checks_enabled {
            self.validate_gamma_stability(bounded_gamma).unwrap_or(self.params.gamma)
        } else {
            bounded_gamma
        }
    }
    
    /// Calculate urgency factor based on time to maturity
    pub fn calculate_urgency_factor(&self, time_to_maturity: FixedPoint) -> FixedPoint {
        if time_to_maturity <= self.dynamic_config.min_time_to_maturity {
            // Maximum urgency when time is very short
            return FixedPoint::from_float(3.0);
        }
        
        // Urgency factor: u(t) = (T-t)^(-α) where α ∈ [0.5, 1]
        let normalized_time = time_to_maturity / self.dynamic_config.min_time_to_maturity;
        let urgency = normalized_time.powf(-self.dynamic_config.urgency_alpha.to_float());
        
        // Cap urgency factor to reasonable bounds
        urgency.max(FixedPoint::from_float(0.5)).min(FixedPoint::from_float(3.0))
    }
    
    /// Validate gamma stability to prevent oscillations
    fn validate_gamma_stability(&self, proposed_gamma: FixedPoint) -> Result<FixedPoint, AvellanedaStoikovError> {
        // Check maximum gamma constraint
        if proposed_gamma > self.dynamic_config.max_gamma {
            return Err(AvellanedaStoikovError::InvalidParameters(
                format!("Proposed gamma {} exceeds maximum {}", 
                    proposed_gamma.to_float(), 
                    self.dynamic_config.max_gamma.to_float())
            ));
        }
        
        // Check for rapid oscillations in gamma history
        if self.gamma_history.len() >= 5 {
            let recent_gammas: Vec<FixedPoint> = self.gamma_history
                .iter()
                .rev()
                .take(5)
                .map(|(gamma, _)| *gamma)
                .collect();
            
            // Calculate variance of recent gamma values
            let mean_gamma: FixedPoint = recent_gammas.iter().sum::<FixedPoint>() / FixedPoint::from_int(5);
            let variance: FixedPoint = recent_gammas.iter()
                .map(|g| (*g - mean_gamma) * (*g - mean_gamma))
                .sum::<FixedPoint>() / FixedPoint::from_int(5);
            
            // If variance is too high, use smoothed value
            let max_variance = (mean_gamma * FixedPoint::from_float(0.1)) * (mean_gamma * FixedPoint::from_float(0.1));
            if variance > max_variance {
                return Ok(mean_gamma); // Return smoothed value
            }
        }
        
        Ok(proposed_gamma)
    }
    
    /// Update effective gamma and maintain history
    pub fn update_effective_gamma(&mut self, new_gamma: FixedPoint, timestamp: Timestamp) {
        self.effective_gamma = new_gamma;
        self.gamma_history.push((new_gamma, timestamp));
        
        // Maintain history size
        if self.gamma_history.len() > self.max_param_history {
            self.gamma_history.remove(0);
        }
        
        self.last_parameter_update.store(timestamp, Ordering::Relaxed);
        
        // Clear cache when gamma changes significantly
        if let Some((prev_gamma, _)) = self.gamma_history.get(self.gamma_history.len().saturating_sub(2)) {
            let gamma_change = (new_gamma - prev_gamma).abs() / prev_gamma;
            if gamma_change > FixedPoint::from_float(0.05) { // 5% change threshold
                self.quote_cache.clear();
            }
        }
    }
    
    /// Calculate dynamic update frequency based on market conditions
    pub fn calculate_update_frequency(&self, time_to_maturity: FixedPoint) -> u64 {
        if !self.dynamic_config.dynamic_update_frequency {
            return self.dynamic_config.base_update_interval;
        }
        
        let mut frequency_multiplier = FixedPoint::ONE;
        
        // Increase frequency in high volatility
        match self.market_conditions.volatility_regime {
            MarketRegime::Crisis => frequency_multiplier = frequency_multiplier * FixedPoint::from_float(0.1), // 10x faster
            MarketRegime::HighVolatility => frequency_multiplier = frequency_multiplier * FixedPoint::from_float(0.5), // 2x faster
            MarketRegime::LowVolatility => frequency_multiplier = frequency_multiplier * FixedPoint::from_float(2.0), // 2x slower
            MarketRegime::Normal => {},
        }
        
        // Increase frequency when approaching maturity
        if time_to_maturity < FixedPoint::from_float(60.0) { // Less than 1 minute
            let urgency_multiplier = (time_to_maturity / FixedPoint::from_float(60.0)).max(FixedPoint::from_float(0.1));
            frequency_multiplier = frequency_multiplier * urgency_multiplier;
        }
        
        // Apply liquidity stress adjustment
        let stress_multiplier = FixedPoint::ONE / (FixedPoint::ONE + self.market_conditions.liquidity_stress);
        frequency_multiplier = frequency_multiplier * stress_multiplier;
        
        // Calculate final update interval
        let base_interval = FixedPoint::from_int(self.dynamic_config.base_update_interval as i64);
        let adjusted_interval = base_interval * frequency_multiplier;
        
        // Ensure reasonable bounds (100 microseconds to 10 seconds)
        adjusted_interval
            .max(FixedPoint::from_int(100_000)) // 100 microseconds
            .min(FixedPoint::from_int(10_000_000_000)) // 10 seconds
            .to_int() as u64
    }
    
    /// Get current dynamic parameter configuration
    pub fn get_dynamic_config(&self) -> &DynamicParameterConfig {
        &self.dynamic_config
    }
    
    /// Update dynamic parameter configuration
    pub fn update_dynamic_config(&mut self, new_config: DynamicParameterConfig) {
        self.dynamic_config = new_config;
        self.quote_cache.clear(); // Clear cache when configuration changes
    }
    
    /// Get current market conditions
    pub fn get_market_conditions(&self) -> &MarketConditionIndicators {
        &self.market_conditions
    }
    
    /// Get current effective gamma (after all adjustments)
    pub fn get_effective_gamma(&self) -> FixedPoint {
        self.effective_gamma
    }
    
    /// Get gamma adjustment history
    pub fn get_gamma_history(&self) -> &[(FixedPoint, Timestamp)] {
        &self.gamma_history
    }
    
    /// Calculate optimal quotes using the Avellaneda-Stoikov model with dynamic parameters
    pub fn calculate_optimal_quotes(
        &mut self,
        mid_price: Price,
        inventory: i64,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
        market_state: &MarketState,
    ) -> Result<OptimalQuotes, AvellanedaStoikovError> {
        self.calculation_count.fetch_add(1, Ordering::Relaxed);
        
        // Update market conditions for dynamic parameter adjustment
        self.update_market_conditions(market_state);
        
        // Calculate adaptive risk aversion
        let adaptive_gamma = self.calculate_adaptive_risk_aversion(time_to_maturity);
        self.update_effective_gamma(adaptive_gamma, market_state.timestamp);
        
        // Use real-time volatility estimate if available
        let effective_volatility = if self.volatility_estimator.return_history.len() > 10 {
            self.volatility_estimator.forecast_volatility(time_to_maturity.to_int() as u64)
        } else {
            volatility
        };
        
        // Check cache with dynamic parameters
        let cache_key = QuoteKey::new(mid_price, inventory, effective_volatility, time_to_maturity);
        let current_time = market_state.timestamp;
        
        // Adjust cache TTL based on market conditions
        let dynamic_cache_ttl = self.calculate_dynamic_cache_ttl(time_to_maturity);
        
        if let Some((cached_quotes, cache_time)) = self.quote_cache.get(&cache_key) {
            if current_time - cache_time < dynamic_cache_ttl {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(cached_quotes.clone());
            }
        }
        
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        
        // Validate inputs
        if mid_price <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Mid price must be positive".to_string()
            ));
        }
        
        if volatility <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Volatility must be positive".to_string()
            ));
        }
        
        if time_to_maturity <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Time to maturity must be positive".to_string()
            ));
        }
        
        // Calculate reservation price with dynamic gamma: r = S - q*γ*σ²*(T-t)
        let inventory_adjustment = FixedPoint::from_int(inventory) * 
            self.effective_gamma * 
            effective_volatility * effective_volatility * 
            time_to_maturity;
        
        let reservation_price = mid_price - inventory_adjustment;
        
        // Calculate optimal spread with dynamic gamma: δ* = γσ²(T-t) + (2/γ)ln(1 + γ/k)
        let risk_component = self.effective_gamma * effective_volatility * effective_volatility * time_to_maturity;
        
        // Calculate ln(1 + γ/k) safely with dynamic gamma
        let gamma_over_k = self.effective_gamma / self.params.k;
        let ln_term = if gamma_over_k < FixedPoint::from_float(1e-6) {
            gamma_over_k // Use first-order approximation for small values
        } else {
            (FixedPoint::ONE + gamma_over_k).ln()
        };
        
        let liquidity_component = (FixedPoint::from_float(2.0) / self.effective_gamma) * ln_term;
        
        let base_spread = risk_component + liquidity_component;
        
        // Apply market impact adjustment with effective volatility
        let market_impact_adjustment = self.calculate_market_impact_adjustment(
            inventory, 
            effective_volatility, 
            time_to_maturity
        )?;
        
        // Update adverse selection protection with current trade information
        let trade_info = TradeInfo {
            price: mid_price,
            volume: inventory,
            mid_price,
            volatility: effective_volatility,
            total_volume: market_state.bid_volume + market_state.ask_volume,
            order_flow_imbalance: market_state.order_flow_imbalance,
            timestamp: current_time,
        };
        
        let adverse_selection_state = self.adverse_selection_protection.update(trade_info)
            .map_err(|e| AvellanedaStoikovError::NumericalInstability(format!("Adverse selection error: {}", e)))?;
        
        let adverse_selection_premium = adverse_selection_state.premium;
        
        let total_spread = base_spread + market_impact_adjustment + adverse_selection_premium;
        
        // Ensure spread is within bounds
        let optimal_spread = total_spread.max(self.params.min_spread).min(self.params.max_spread);
        
        // Calculate inventory skew for asymmetric spreads
        let inventory_skew = self.calculate_inventory_skew(inventory, optimal_spread)?;
        
        // Calculate bid and ask prices
        let half_spread = optimal_spread / FixedPoint::from_float(2.0);
        let bid_adjustment = half_spread + inventory_skew;
        let ask_adjustment = half_spread - inventory_skew;
        
        let bid_price = reservation_price - bid_adjustment;
        let ask_price = reservation_price + ask_adjustment;
        
        // Round to tick size
        let bid_price = self.round_to_tick(bid_price);
        let ask_price = self.round_to_tick(ask_price);
        
        // Calculate recommended sizes (simplified for now)
        let base_size = 100; // This would be more sophisticated in practice
        let bid_size = base_size;
        let ask_size = base_size;
        
        // Calculate confidence score based on market conditions with effective volatility
        let confidence = self.calculate_confidence_score(market_state, effective_volatility)?;
        
        let quotes = OptimalQuotes {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            reservation_price,
            optimal_spread,
            inventory_skew,
            market_impact_adjustment,
            adverse_selection_premium,
            timestamp: current_time,
            confidence,
        };
        
        // Cache the result with dynamic TTL
        self.quote_cache.insert(cache_key, (quotes.clone(), current_time));
        
        // Clean old cache entries periodically
        if self.quote_cache.len() > 1000 {
            self.clean_cache(current_time);
        }
        
        Ok(quotes)
    }
    
    /// Calculate market impact adjustment for spread calculation
    fn calculate_market_impact_adjustment(
        &self,
        inventory: i64,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        // Estimate participation rate based on inventory and time horizon
        let inventory_abs = FixedPoint::from_int(inventory.abs());
        let estimated_participation_rate = if time_to_maturity > FixedPoint::ZERO {
            (inventory_abs / time_to_maturity).min(FixedPoint::from_float(0.3))
        } else {
            FixedPoint::from_float(0.1) // Default participation rate
        };
        
        // Calculate temporary impact component
        let temp_impact = self.market_impact_calculator.calculate_temporary_impact(estimated_participation_rate);
        
        // Calculate permanent impact component
        let perm_impact = self.market_impact_calculator.calculate_permanent_impact(inventory_abs);
        
        // Combined impact affects spread - higher impact means wider spreads
        let total_impact = temp_impact + perm_impact;
        
        // Scale impact by volatility and time horizon
        let impact_adjustment = total_impact * volatility * time_to_maturity.sqrt();
        
        Ok(impact_adjustment)
    }
    
    /// Calculate adverse selection premium based on market microstructure signals
    fn calculate_adverse_selection_premium(
        &self,
        market_state: &MarketState,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        // Information asymmetry measure: IA = |price_impact|/√(volume*volatility)
        let bid_ask_spread = market_state.ask_price - market_state.bid_price;
        let total_volume = FixedPoint::from_int(market_state.bid_volume + market_state.ask_volume);
        
        if total_volume <= FixedPoint::ZERO {
            return Ok(FixedPoint::ZERO);
        }
        
        // Estimate price impact from order flow imbalance
        let price_impact = market_state.order_flow_imbalance.abs() * bid_ask_spread;
        
        // Information asymmetry measure
        let volume_volatility_factor = (total_volume * volatility).sqrt();
        let information_asymmetry = if volume_volatility_factor > FixedPoint::ZERO {
            price_impact / volume_volatility_factor
        } else {
            FixedPoint::ZERO
        };
        
        // Adverse selection premium: AS = β*IA*σ*√(T-t)
        let beta = FixedPoint::from_float(0.5); // Adverse selection sensitivity parameter
        let adverse_selection_premium = beta * information_asymmetry * volatility * time_to_maturity.sqrt();
        
        // Cap the premium to prevent excessive spread widening
        let max_premium = bid_ask_spread * FixedPoint::from_float(0.5);
        Ok(adverse_selection_premium.min(max_premium))
    }
    
    /// Calculate inventory skew for asymmetric spread adjustment with dynamic gamma
    fn calculate_inventory_skew(
        &self,
        inventory: i64,
        optimal_spread: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        if inventory == 0 {
            return Ok(FixedPoint::ZERO);
        }
        
        // Skew factor: κ = q*γ*σ²*(T-t)/(2*δ*) using effective gamma
        let inventory_factor = FixedPoint::from_int(inventory) * self.effective_gamma;
        let skew_factor = inventory_factor / (FixedPoint::from_float(2.0) * optimal_spread);
        
        // Limit skew to prevent extreme asymmetry
        let max_skew = optimal_spread * FixedPoint::from_float(0.3);
        let inventory_skew = (skew_factor * optimal_spread).max(-max_skew).min(max_skew);
        
        Ok(inventory_skew)
    }
    
    /// Calculate dynamic cache TTL based on market conditions and time to maturity
    fn calculate_dynamic_cache_ttl(&self, time_to_maturity: FixedPoint) -> u64 {
        let base_ttl = self.cache_ttl as f64;
        let mut ttl_multiplier = 1.0;
        
        // Reduce TTL in high volatility (more frequent updates needed)
        match self.market_conditions.volatility_regime {
            MarketRegime::Crisis => ttl_multiplier *= 0.1,      // 10x shorter TTL
            MarketRegime::HighVolatility => ttl_multiplier *= 0.5,  // 2x shorter TTL
            MarketRegime::LowVolatility => ttl_multiplier *= 2.0,   // 2x longer TTL
            MarketRegime::Normal => {},
        }
        
        // Reduce TTL when approaching maturity
        if time_to_maturity < FixedPoint::from_float(60.0) { // Less than 1 minute
            let urgency_factor = (time_to_maturity.to_float() / 60.0).max(0.1);
            ttl_multiplier *= urgency_factor;
        }
        
        // Reduce TTL during high liquidity stress
        let stress_factor = 1.0 / (1.0 + self.market_conditions.liquidity_stress.to_float());
        ttl_multiplier *= stress_factor;
        
        // Apply bounds (10 microseconds to 10 seconds)
        let dynamic_ttl = (base_ttl * ttl_multiplier)
            .max(10_000.0)      // 10 microseconds minimum
            .min(10_000_000_000.0); // 10 seconds maximum
        
        dynamic_ttl as u64
    }
    
    /// Calculate confidence score based on market conditions
    fn calculate_confidence_score(
        &self,
        market_state: &MarketState,
        volatility: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        let mut confidence = FixedPoint::ONE;
        
        // Reduce confidence for high volatility
        let vol_threshold = FixedPoint::from_float(0.5);
        if volatility > vol_threshold {
            confidence = confidence * (vol_threshold / volatility);
        }
        
        // Reduce confidence for wide spreads (low liquidity)
        let spread = market_state.ask_price - market_state.bid_price;
        let relative_spread = spread / market_state.mid_price;
        let spread_threshold = FixedPoint::from_float(0.01); // 1%
        
        if relative_spread > spread_threshold {
            confidence = confidence * (spread_threshold / relative_spread);
        }
        
        // Reduce confidence for high order flow imbalance
        let imbalance_threshold = FixedPoint::from_float(0.5);
        let abs_imbalance = market_state.order_flow_imbalance.abs();
        if abs_imbalance > imbalance_threshold {
            confidence = confidence * (imbalance_threshold / abs_imbalance);
        }
        
        // Ensure confidence is between 0 and 1
        Ok(confidence.max(FixedPoint::from_float(0.1)).min(FixedPoint::ONE))
    }
    
    /// Round price to nearest tick size
    fn round_to_tick(&self, price: Price) -> Price {
        if self.params.tick_size <= FixedPoint::ZERO {
            return price;
        }
        
        let ticks = (price / self.params.tick_size).round();
        ticks * self.params.tick_size
    }
    
    /// Clean old entries from quote cache
    fn clean_cache(&mut self, current_time: Timestamp) {
        self.quote_cache.retain(|_, (_, cache_time)| {
            current_time - cache_time < self.cache_ttl * 10 // Keep entries for 10x TTL
        });
    }
    
    /// Update market impact parameters
    pub fn update_market_impact_parameters(
        &mut self,
        new_params: MarketImpactParams,
    ) -> Result<(), AvellanedaStoikovError> {
        self.market_impact_calculator.update_parameters(new_params)?;
        // Clear cache since impact parameters changed
        self.quote_cache.clear();
        Ok(())
    }
    
    /// Get current market impact parameters
    pub fn get_market_impact_parameters(&self) -> &MarketImpactParams {
        self.market_impact_calculator.get_parameters()
    }
    
    /// Perform transaction cost analysis for a given order
    pub fn analyze_transaction_costs(
        &self,
        total_quantity: FixedPoint,
        time_horizon: FixedPoint,
        volatility: FixedPoint,
    ) -> Result<TransactionCostAnalysis, AvellanedaStoikovError> {
        self.market_impact_calculator.analyze_transaction_costs(
            total_quantity,
            time_horizon,
            volatility,
            self.params.gamma,
        )
    }
    
    /// Update market impact model with realized impact measurement
    pub fn update_realized_impact(
        &mut self,
        participation_rate: FixedPoint,
        realized_impact: FixedPoint,
        timestamp: Timestamp,
    ) {
        self.market_impact_calculator.update_with_realized_impact(
            participation_rate,
            realized_impact,
            timestamp,
        );
    }
    
    /// Calibrate market impact parameters using historical data
    pub fn calibrate_market_impact_parameters(&mut self) -> Result<(), AvellanedaStoikovError> {
        self.market_impact_calculator.calibrate_parameters()?;
        // Clear cache since parameters changed
        self.quote_cache.clear();
        Ok(())
    }
    
    /// Calculate optimal participation rate for execution
    pub fn calculate_optimal_participation_rate(
        &self,
        total_quantity: FixedPoint,
        time_horizon: FixedPoint,
        volatility: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        self.market_impact_calculator.optimize_participation_rate(
            total_quantity,
            time_horizon,
            volatility,
            self.params.gamma,
        )
    }
    
    /// Update inventory tracker
    pub fn update_inventory(
        &mut self,
        trade_quantity: i64,
        trade_price: Price,
        timestamp: Timestamp,
    ) {
        let old_position = self.inventory_tracker.position;
        let new_position = old_position + trade_quantity;
        
        // Update average price using volume-weighted calculation
        if new_position != 0 {
            let old_value = FixedPoint::from_int(old_position) * self.inventory_tracker.average_price;
            let trade_value = FixedPoint::from_int(trade_quantity) * trade_price;
            let new_value = old_value + trade_value;
            self.inventory_tracker.average_price = new_value / FixedPoint::from_int(new_position);
        } else {
            self.inventory_tracker.average_price = FixedPoint::ZERO;
        }
        
        self.inventory_tracker.position = new_position;
        self.inventory_tracker.last_update = timestamp;
        
        // Clear cache since inventory changed
        self.quote_cache.clear();
    }
    
    /// Update volatility estimate
    pub fn update_volatility(&mut self, price: Price, timestamp: Timestamp) -> FixedPoint {
        self.volatility_estimator.update(price, timestamp)
    }
    
    /// Get current volatility estimate
    pub fn get_current_volatility(&self) -> FixedPoint {
        self.volatility_estimator.get_volatility()
    }
    
    /// Get current inventory position
    pub fn get_inventory(&self) -> &InventoryTracker {
        &self.inventory_tracker
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> (u64, u64, u64, f64) {
        let calculations = self.calculation_count.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        (calculations, hits, misses, hit_rate)
    }
    
    /// Reset performance metrics
    pub fn reset_performance_metrics(&self) {
        self.calculation_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
    
    /// Update model parameters
    pub fn update_parameters(&mut self, new_params: AvellanedaStoikovParams) -> Result<(), AvellanedaStoikovError> {
        new_params.validate()?;
        self.params = new_params;
        // Clear cache since parameters changed
        self.quote_cache.clear();
        Ok(())
    }
    
    /// Get current model parameters
    pub fn get_parameters(&self) -> &AvellanedaStoikovParams {
        &self.params
    }
    
    /// Validate parameter stability and detect potential issues
    pub fn validate_parameter_stability(&self) -> Result<(), AvellanedaStoikovError> {
        // Check if effective gamma is within reasonable bounds
        if self.effective_gamma <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::InvalidParameters(
                "Effective gamma must be positive".to_string()
            ));
        }
        
        if self.effective_gamma > self.dynamic_config.max_gamma {
            return Err(AvellanedaStoikovError::InvalidParameters(
                format!("Effective gamma {} exceeds maximum allowed {}", 
                    self.effective_gamma.to_float(),
                    self.dynamic_config.max_gamma.to_float())
            ));
        }
        
        // Check for parameter oscillations
        if self.gamma_history.len() >= 10 {
            let recent_changes: Vec<FixedPoint> = self.gamma_history
                .windows(2)
                .rev()
                .take(5)
                .map(|window| (window[1].0 - window[0].0).abs())
                .collect();
            
            let avg_change: FixedPoint = recent_changes.iter().sum::<FixedPoint>() / 
                FixedPoint::from_int(recent_changes.len() as i64);
            
            let max_allowed_change = self.effective_gamma * FixedPoint::from_float(0.1); // 10% of current gamma
            
            if avg_change > max_allowed_change {
                return Err(AvellanedaStoikovError::NumericalInstability(
                    format!("Parameter oscillation detected: average change {} exceeds threshold {}", 
                        avg_change.to_float(),
                        max_allowed_change.to_float())
                ));
            }
        }
        
        Ok(())
    }
    
    /// Get parameter adjustment statistics
    pub fn get_parameter_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("effective_gamma".to_string(), self.effective_gamma.to_float());
        stats.insert("base_gamma".to_string(), self.params.gamma.to_float());
        stats.insert("gamma_adjustment_ratio".to_string(), 
            (self.effective_gamma / self.params.gamma).to_float());
        
        if !self.gamma_history.is_empty() {
            let gamma_values: Vec<f64> = self.gamma_history.iter()
                .map(|(gamma, _)| gamma.to_float())
                .collect();
            
            let mean_gamma = gamma_values.iter().sum::<f64>() / gamma_values.len() as f64;
            let variance = gamma_values.iter()
                .map(|g| (g - mean_gamma).powi(2))
                .sum::<f64>() / gamma_values.len() as f64;
            
            stats.insert("gamma_mean".to_string(), mean_gamma);
            stats.insert("gamma_variance".to_string(), variance);
            stats.insert("gamma_std_dev".to_string(), variance.sqrt());
            
            if let (Some(first), Some(last)) = (gamma_values.first(), gamma_values.last()) {
                stats.insert("gamma_total_change".to_string(), (last - first).abs());
            }
        }
        
        stats.insert("volatility_regime".to_string(), match self.market_conditions.volatility_regime {
            MarketRegime::Normal => 0.0,
            MarketRegime::LowVolatility => 1.0,
            MarketRegime::HighVolatility => 2.0,
            MarketRegime::Crisis => 3.0,
        });
        
        stats.insert("liquidity_stress".to_string(), self.market_conditions.liquidity_stress.to_float());
        stats.insert("current_volatility".to_string(), self.market_conditions.volatility.to_float());
        
        stats
    }
    
    /// Reset dynamic parameter adjustment state
    pub fn reset_dynamic_state(&mut self) {
        self.effective_gamma = self.params.gamma;
        self.gamma_history.clear();
        self.market_conditions = MarketConditionIndicators::default();
        self.last_parameter_update.store(0, Ordering::Relaxed);
        self.quote_cache.clear();
    }
    
    /// Enable/disable specific dynamic features
    pub fn configure_dynamic_features(&mut self, 
        adaptive_risk_aversion: Option<bool>,
        urgency_factor: Option<bool>,
        stability_checks: Option<bool>,
        dynamic_update_frequency: Option<bool>,
    ) {
        if let Some(adaptive) = adaptive_risk_aversion {
            self.dynamic_config.adaptive_risk_aversion = adaptive;
        }
        if let Some(urgency) = urgency_factor {
            self.dynamic_config.urgency_factor_enabled = urgency;
        }
        if let Some(stability) = stability_checks {
            self.dynamic_config.stability_checks_enabled = stability;
        }
        if let Some(frequency) = dynamic_update_frequency {
            self.dynamic_config.dynamic_update_frequency = frequency;
        }
        
        // Clear cache when configuration changes
        self.quote_cache.clear();
    }
}
    
    /// Calculate adverse selection premium
    fn calculate_adverse_selection_premium(
        &self,
        market_state: &MarketState,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        // Information asymmetry measure based on order flow imbalance
        let order_flow_imbalance = market_state.order_flow_imbalance.abs();
        
        // Price impact measure
        let spread = market_state.ask_price - market_state.bid_price;
        let relative_spread = spread / market_state.mid_price;
        
        // Information asymmetry indicator
        let info_asymmetry = order_flow_imbalance * relative_spread;
        
        // Adverse selection premium: AS = β*IA*σ*√(T-t)
        let beta = FixedPoint::from_float(0.5); // Calibration parameter
        let premium = beta * info_asymmetry * volatility * time_to_maturity.sqrt();
        
        Ok(premium)
    }
    
    /// Calculate inventory skew for asymmetric spreads
    fn calculate_inventory_skew(
        &self,
        inventory: i64,
        optimal_spread: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        if inventory == 0 {
            return Ok(FixedPoint::ZERO);
        }
        
        // Skew factor: κ = q*γ*σ²*(T-t)/(2*δ*)
        let inventory_factor = FixedPoint::from_int(inventory);
        let skew_denominator = FixedPoint::from_float(2.0) * optimal_spread;
        
        if skew_denominator <= FixedPoint::ZERO {
            return Err(AvellanedaStoikovError::NumericalInstability(
                "Optimal spread too small for skew calculation".to_string()
            ));
        }
        
        // Simplified skew calculation (would include more factors in practice)
        let max_skew = optimal_spread / FixedPoint::from_float(4.0); // Limit skew to 25% of spread
        let raw_skew = inventory_factor * self.params.gamma / skew_denominator;
        
        // Clamp skew to reasonable bounds
        let skew = raw_skew.max(-max_skew).min(max_skew);
        
        Ok(skew)
    }
    
    /// Round price to tick size
    fn round_to_tick(&self, price: Price) -> Price {
        let ticks = (price / self.params.tick_size).round();
        ticks * self.params.tick_size
    }
    
    /// Calculate confidence score for quotes
    fn calculate_confidence_score(
        &self,
        market_state: &MarketState,
        volatility: FixedPoint,
    ) -> Result<FixedPoint, AvellanedaStoikovError> {
        // Base confidence
        let mut confidence = FixedPoint::from_float(0.8);
        
        // Reduce confidence for high volatility
        let vol_threshold = FixedPoint::from_float(0.5);
        if volatility > vol_threshold {
            let vol_penalty = (volatility - vol_threshold) / vol_threshold;
            confidence = confidence - vol_penalty * FixedPoint::from_float(0.3);
        }
        
        // Reduce confidence for wide spreads (illiquid markets)
        let spread = market_state.ask_price - market_state.bid_price;
        let relative_spread = spread / market_state.mid_price;
        let spread_threshold = FixedPoint::from_float(0.01); // 1%
        
        if relative_spread > spread_threshold {
            let spread_penalty = (relative_spread - spread_threshold) / spread_threshold;
            confidence = confidence - spread_penalty * FixedPoint::from_float(0.2);
        }
        
        // Ensure confidence is in [0, 1]
        confidence = confidence.max(FixedPoint::ZERO).min(FixedPoint::ONE);
        
        Ok(confidence)
    }
    
    /// Clean old entries from cache
    fn clean_cache(&mut self, current_time: Timestamp) {
        self.quote_cache.retain(|_, (_, cache_time)| {
            current_time - cache_time < self.cache_ttl * 10 // Keep entries for 10x TTL
        });
    }
    
    /// Update model parameters
    pub fn update_parameters(&mut self, new_params: AvellanedaStoikovParams) -> Result<(), AvellanedaStoikovError> {
        new_params.validate()?;
        self.params = new_params;
        self.quote_cache.clear(); // Clear cache when parameters change
        Ok(())
    }
    
    /// Update market impact parameters
    pub fn update_market_impact_parameters(&mut self, new_params: MarketImpactParams) {
        self.market_impact_params = new_params;
        self.quote_cache.clear(); // Clear cache when parameters change
    }
    
    /// Update inventory position
    pub fn update_inventory(&mut self, new_position: i64, price: Price, timestamp: Timestamp) {
        let old_position = self.inventory_tracker.position;
        let position_change = new_position - old_position;
        
        if position_change != 0 {
            // Update average price (simplified)
            if new_position != 0 {
                let total_value = FixedPoint::from_int(old_position) * self.inventory_tracker.average_price +
                    FixedPoint::from_int(position_change) * price;
                self.inventory_tracker.average_price = total_value / FixedPoint::from_int(new_position);
            } else {
                self.inventory_tracker.average_price = FixedPoint::ZERO;
            }
        }
        
        self.inventory_tracker.position = new_position;
        self.inventory_tracker.last_update = timestamp;
        
        // Clear cache when inventory changes significantly
        if position_change.abs() > 100 {
            self.quote_cache.clear();
        }
    }
    
    /// Get current inventory position
    pub fn get_inventory(&self) -> i64 {
        self.inventory_tracker.position
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> (u64, u64, u64, f64) {
        let calculations = self.calculation_count.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            hits as f64 / (hits + misses) as f64
        } else {
            0.0
        };
        
        (calculations, hits, misses, hit_rate)
    }
    
    /// Reset performance metrics
    pub fn reset_performance_metrics(&self) {
        self.calculation_count.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
    }
    
    /// Get current adverse selection protection state
    pub fn get_adverse_selection_state(&self) -> &AdverseSelectionState {
        self.adverse_selection_protection.get_state()
    }
    
    /// Update adverse selection protection parameters
    pub fn update_adverse_selection_params(
        &mut self, 
        params: AdverseSelectionParams
    ) -> Result<(), AvellanedaStoikovError> {
        self.adverse_selection_protection.update_parameters(params)
            .map_err(|e| AvellanedaStoikovError::InvalidParameters(format!("Adverse selection error: {}", e)))
    }
    
    /// Get adverse selection diagnostics for monitoring
    pub fn get_adverse_selection_diagnostics(&self) -> crate::models::adverse_selection::AdverseSelectionDiagnostics {
        self.adverse_selection_protection.get_diagnostics()
    }
    
    /// Reset adverse selection protection state
    pub fn reset_adverse_selection_protection(&mut self) {
        self.adverse_selection_protection.reset();
    }
    
    /// Check if adverse selection protection is currently active
    pub fn is_adverse_selection_protection_active(&self) -> bool {
        self.adverse_selection_protection.get_state().protection_active
    }
    
    /// Get current quote frequency adjustment due to adverse selection
    pub fn get_quote_frequency_adjustment(&self) -> FixedPoint {
        self.adverse_selection_protection.get_state().frequency_adjustment
    }
    
    /// Get current adverse selection premium
    pub fn get_adverse_selection_premium(&self) -> FixedPoint {
        self.adverse_selection_protection.get_state().premium
    }
    
    /// Get current toxicity level
    pub fn get_toxicity_level(&self) -> FixedPoint {
        self.adverse_selection_protection.get_state().toxicity_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parameter_validation() {
        let mut params = AvellanedaStoikovParams::default();
        assert!(params.validate().is_ok());
        
        params.gamma = FixedPoint::ZERO;
        assert!(params.validate().is_err());
        
        params.gamma = FixedPoint::from_float(0.1);
        params.sigma = FixedPoint::from_float(-0.1);
        assert!(params.validate().is_err());
    }
    
    #[test]
    fn test_engine_creation() {
        let params = AvellanedaStoikovParams::default();
        let engine = AvellanedaStoikovEngine::new(params);
        assert!(engine.is_ok());
    }
    
    #[test]
    fn test_basic_quote_calculation() {
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        let market_state = MarketState::default();
        let quotes = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            0, // No inventory
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        );
        
        assert!(quotes.is_ok());
        let quotes = quotes.unwrap();
        
        // Basic sanity checks
        assert!(quotes.ask_price > quotes.bid_price);
        assert!(quotes.optimal_spread > FixedPoint::ZERO);
        assert!(quotes.confidence > FixedPoint::ZERO);
        assert!(quotes.confidence <= FixedPoint::ONE);
    }
    
    #[test]
    fn test_inventory_skew() {
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        let market_state = MarketState::default();
        
        // Test with positive inventory (should skew quotes down)
        let quotes_long = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            100, // Long position
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        ).unwrap();
        
        // Test with negative inventory (should skew quotes up)
        let quotes_short = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            -100, // Short position
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        ).unwrap();
        
        // Long position should have lower reservation price
        assert!(quotes_long.reservation_price < quotes_short.reservation_price);
    }
    
    #[test]
    fn test_cache_functionality() {
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        let market_state = MarketState::default();
        
        // First calculation should be a cache miss
        let _quotes1 = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            0,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        ).unwrap();
        
        // Second identical calculation should be a cache hit
        let _quotes2 = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            0,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        ).unwrap();
        
        let (calculations, hits, misses, hit_rate) = engine.get_performance_metrics();
        assert_eq!(calculations, 2);
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 0.5);
    }
    
    #[test]
    fn test_adaptive_risk_aversion() {
        let params = AvellanedaStoikovParams::default();
        let mut config = DynamicParameterConfig::default();
        config.adaptive_risk_aversion = true;
        config.base_gamma = FixedPoint::from_float(0.1);
        
        let mut engine = AvellanedaStoikovEngine::new_with_dynamic_config(params, config).unwrap();
        
        // Test normal market conditions
        let mut market_state = MarketState::default();
        market_state.volatility = FixedPoint::from_float(0.2);
        
        engine.update_market_conditions(&market_state);
        let normal_gamma = engine.calculate_adaptive_risk_aversion(FixedPoint::from_float(60.0));
        
        // Test high volatility conditions
        market_state.volatility = FixedPoint::from_float(0.5);
        engine.update_market_conditions(&market_state);
        let high_vol_gamma = engine.calculate_adaptive_risk_aversion(FixedPoint::from_float(60.0));
        
        // High volatility should increase risk aversion
        assert!(high_vol_gamma > normal_gamma);
        assert!(high_vol_gamma <= engine.dynamic_config.max_gamma);
    }
    
    #[test]
    fn test_urgency_factor() {
        let params = AvellanedaStoikovParams::default();
        let mut config = DynamicParameterConfig::default();
        config.urgency_factor_enabled = true;
        config.urgency_alpha = FixedPoint::from_float(0.7);
        
        let engine = AvellanedaStoikovEngine::new_with_dynamic_config(params, config).unwrap();
        
        // Test different time horizons
        let long_horizon_urgency = engine.calculate_urgency_factor(FixedPoint::from_float(300.0)); // 5 minutes
        let short_horizon_urgency = engine.calculate_urgency_factor(FixedPoint::from_float(10.0));  // 10 seconds
        let very_short_urgency = engine.calculate_urgency_factor(FixedPoint::from_float(0.5));     // 0.5 seconds
        
        // Shorter time horizons should have higher urgency factors
        assert!(short_horizon_urgency > long_horizon_urgency);
        assert!(very_short_urgency > short_horizon_urgency);
        
        // Urgency should be bounded
        assert!(very_short_urgency <= FixedPoint::from_float(3.0));
        assert!(long_horizon_urgency >= FixedPoint::from_float(0.5));
    }
    
    #[test]
    fn test_realized_volatility_estimator() {
        let mut estimator = RealizedVolatilityEstimator::new(100, FixedPoint::from_float(0.94));
        
        // Add some price data
        let prices = vec![100.0, 100.5, 99.8, 101.2, 100.9, 99.5, 102.1];
        let mut timestamp = 1000000000u64;
        
        for price in prices {
            let vol = estimator.update(FixedPoint::from_float(price), timestamp);
            assert!(vol > FixedPoint::ZERO);
            timestamp += 1000000000; // 1 second intervals
        }
        
        // Test regime detection
        let regime = estimator.get_market_regime();
        assert!(matches!(regime, MarketRegime::Normal | MarketRegime::HighVolatility | MarketRegime::LowVolatility | MarketRegime::Crisis));
        
        // Test volatility forecasting
        let forecast = estimator.forecast_volatility(300); // 5 minutes
        assert!(forecast > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_parameter_stability_validation() {
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        // Test normal case
        assert!(engine.validate_parameter_stability().is_ok());
        
        // Test with extreme gamma
        engine.effective_gamma = FixedPoint::from_float(100.0); // Very high gamma
        assert!(engine.validate_parameter_stability().is_err());
        
        // Reset to normal
        engine.effective_gamma = FixedPoint::from_float(0.1);
        assert!(engine.validate_parameter_stability().is_ok());
    }
    
    #[test]
    fn test_dynamic_cache_ttl() {
        let params = AvellanedaStoikovParams::default();
        let engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        // Test normal conditions
        let normal_ttl = engine.calculate_dynamic_cache_ttl(FixedPoint::from_float(300.0));
        
        // Create engine with crisis conditions
        let mut crisis_engine = AvellanedaStoikovEngine::new(AvellanedaStoikovParams::default()).unwrap();
        crisis_engine.market_conditions.volatility_regime = MarketRegime::Crisis;
        let crisis_ttl = crisis_engine.calculate_dynamic_cache_ttl(FixedPoint::from_float(300.0));
        
        // Crisis should have shorter TTL
        assert!(crisis_ttl < normal_ttl);
        
        // Test urgency effect
        let urgent_ttl = engine.calculate_dynamic_cache_ttl(FixedPoint::from_float(10.0)); // 10 seconds
        assert!(urgent_ttl < normal_ttl);
    }
    
    #[test]
    fn test_market_condition_updates() {
        let params = AvellanedaStoikovParams::default();
        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
        
        let mut market_state = MarketState::default();
        market_state.order_flow_imbalance = FixedPoint::from_float(0.3);
        market_state.bid_volume = 500;
        market_state.ask_volume = 1500;
        
        engine.update_market_conditions(&market_state);
        
        // Check that conditions were updated
        assert_eq!(engine.market_conditions.order_flow_imbalance, FixedPoint::from_float(0.3));
        assert!(engine.market_conditions.spread_tightness > FixedPoint::ZERO);
        assert!(engine.market_conditions.market_depth > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_dynamic_quote_calculation() {
        let params = AvellanedaStoikovParams::default();
        let mut config = DynamicParameterConfig::default();
        config.adaptive_risk_aversion = true;
        config.urgency_factor_enabled = true;
        
        let mut engine = AvellanedaStoikovEngine::new_with_dynamic_config(params, config).unwrap();
        
        // Add some volatility history
        for i in 0..20 {
            let price = FixedPoint::from_float(100.0 + (i as f64 * 0.1));
            engine.volatility_estimator.update(price, 1000000000 + i * 1000000000);
        }
        
        let market_state = MarketState::default();
        
        // Test with normal time horizon
        let normal_quotes = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            100, // inventory
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(300.0), // 5 minutes
            &market_state,
        ).unwrap();
        
        // Test with urgent time horizon
        let urgent_quotes = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            100, // inventory
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(10.0), // 10 seconds
            &market_state,
        ).unwrap();
        
        // Urgent quotes should have different characteristics
        assert!(urgent_quotes.optimal_spread != normal_quotes.optimal_spread);
        assert!(urgent_quotes.confidence > FixedPoint::ZERO);
        assert!(urgent_quotes.confidence <= FixedPoint::ONE);
    }
}ws should have opposite signs
        assert!(skew_long.unwrap() * skew_short.unwrap() <= FixedPoint::ZERO);
    }
    
    #[test]
    fn test_adverse_selection_premium() {
        let engine = create_test_engine();
        let mut market_state = create_test_market_state();
        
        // Test with high order flow imbalance
        market_state.order_flow_imbalance = FixedPoint::from_float(0.8);
        
        let premium = engine.calculate_adverse_selection_premium(
            &market_state,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
        );
        
        assert!(premium.is_ok());
        let result = premium.unwrap();
        assert!(result >= FixedPoint::ZERO);
        
        // Test with zero imbalance
        market_state.order_flow_imbalance = FixedPoint::ZERO;
        let premium_zero = engine.calculate_adverse_selection_premium(
            &market_state,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
        );
        
        assert!(premium_zero.is_ok());
        // Should be lower with zero imbalance
        assert!(premium_zero.unwrap() <= result);
    }
    
    #[test]
    fn test_market_impact_parameter_updates() {
        let mut engine = create_test_engine();
        
        let new_params = MarketImpactParams {
            eta: FixedPoint::from_float(0.2),
            alpha: FixedPoint::from_float(0.6),
            lambda: FixedPoint::from_float(0.02),
            ..MarketImpactParams::default()
        };
        
        let result = engine.update_market_impact_parameters(new_params.clone());
        assert!(result.is_ok());
        
        let current_params = engine.get_market_impact_parameters();
        assert_eq!(current_params.eta, new_params.eta);
        assert_eq!(current_params.alpha, new_params.alpha);
        assert_eq!(current_params.lambda, new_params.lambda);
    }
    
    #[test]
    fn test_realized_impact_updates() {
        let mut engine = create_test_engine();
        
        // Add some realized impact measurements
        engine.update_realized_impact(
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.05),
            1000000000,
        );
        
        engine.update_realized_impact(
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.08),
            1000000001,
        );
        
        // This should work without errors
        // In a real implementation, we might check if the model adapts
    }
    
    #[test]
    fn test_inventory_updates() {
        let mut engine = create_test_engine();
        
        // Initial inventory should be zero
        assert_eq!(engine.get_inventory().position, 0);
        
        // Update with a buy trade
        engine.update_inventory(100, FixedPoint::from_float(100.5), 1000000000);
        assert_eq!(engine.get_inventory().position, 100);
        assert_eq!(engine.get_inventory().average_price, FixedPoint::from_float(100.5));
        
        // Update with another buy trade
        engine.update_inventory(50, FixedPoint::from_float(101.0), 1000000001);
        assert_eq!(engine.get_inventory().position, 150);
        
        // Average price should be weighted average
        let expected_avg = (FixedPoint::from_float(100.5) * FixedPoint::from_int(100) + 
                           FixedPoint::from_float(101.0) * FixedPoint::from_int(50)) / 
                           FixedPoint::from_int(150);
        assert!((engine.get_inventory().average_price - expected_avg).abs() < FixedPoint::from_float(0.01));
        
        // Update with a sell trade
        engine.update_inventory(-75, FixedPoint::from_float(100.8), 1000000002);
        assert_eq!(engine.get_inventory().position, 75);
    }
    
    #[test]
    fn test_confidence_score_calculation() {
        let engine = create_test_engine();
        let mut market_state = create_test_market_state();
        
        // Test normal market conditions
        let confidence = engine.calculate_confidence_score(&market_state, FixedPoint::from_float(0.2));
        assert!(confidence.is_ok());
        let score = confidence.unwrap();
        assert!(score > FixedPoint::ZERO);
        assert!(score <= FixedPoint::ONE);
        
        // Test high volatility (should reduce confidence)
        let high_vol_confidence = engine.calculate_confidence_score(&market_state, FixedPoint::from_float(0.8));
        assert!(high_vol_confidence.is_ok());
        assert!(high_vol_confidence.unwrap() < score);
        
        // Test wide spread (should reduce confidence)
        market_state.bid_price = FixedPoint::from_float(99.0);
        market_state.ask_price = FixedPoint::from_float(101.0);
        let wide_spread_confidence = engine.calculate_confidence_score(&market_state, FixedPoint::from_float(0.2));
        assert!(wide_spread_confidence.is_ok());
        assert!(wide_spread_confidence.unwrap() < score);
    }
    
    #[test]
    fn test_performance_metrics() {
        let mut engine = create_test_engine();
        let market_state = create_test_market_state();
        
        // Initial metrics should be zero
        let (calcs, hits, misses, hit_rate) = engine.get_performance_metrics();
        assert_eq!(calcs, 0);
        assert_eq!(hits, 0);
        assert_eq!(misses, 0);
        assert_eq!(hit_rate, 0.0);
        
        // Calculate some quotes
        let _ = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            0,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        );
        
        let (calcs_after, _, misses_after, _) = engine.get_performance_metrics();
        assert_eq!(calcs_after, 1);
        assert_eq!(misses_after, 1); // First call should be a cache miss
        
        // Calculate same quotes again (should hit cache)
        let _ = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            0,
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            &market_state,
        );
        
        let (calcs_final, hits_final, _, hit_rate_final) = engine.get_performance_metrics();
        assert_eq!(calcs_final, 2);
        assert_eq!(hits_final, 1); // Second call should be a cache hit
        assert!(hit_rate_final > 0.0);
    }
    
    #[test]
    fn test_parameter_validation() {
        let mut invalid_params = AvellanedaStoikovParams::default();
        
        // Test invalid gamma
        invalid_params.gamma = FixedPoint::ZERO;
        assert!(AvellanedaStoikovEngine::new(invalid_params.clone()).is_err());
        
        // Test invalid volatility
        invalid_params = AvellanedaStoikovParams::default();
        invalid_params.sigma = FixedPoint::from_float(-0.1);
        assert!(AvellanedaStoikovEngine::new(invalid_params.clone()).is_err());
        
        // Test invalid spread bounds
        invalid_params = AvellanedaStoikovParams::default();
        invalid_params.min_spread = FixedPoint::from_float(0.1);
        invalid_params.max_spread = FixedPoint::from_float(0.05);
        assert!(AvellanedaStoikovEngine::new(invalid_params).is_err());
    }
    
    #[test]
    fn test_market_impact_parameter_validation() {
        let mut invalid_params = MarketImpactParams::default();
        
        // Test invalid eta
        invalid_params.eta = FixedPoint::ZERO;
        assert!(MarketImpactCalculator::new(invalid_params.clone(), 1000).is_err());
        
        // Test invalid alpha
        invalid_params = MarketImpactParams::default();
        invalid_params.alpha = FixedPoint::from_float(1.5);
        assert!(MarketImpactCalculator::new(invalid_params.clone(), 1000).is_err());
        
        // Test invalid participation rate bounds
        invalid_params = MarketImpactParams::default();
        invalid_params.min_participation_rate = FixedPoint::from_float(0.5);
        invalid_params.max_participation_rate = FixedPoint::from_float(0.3);
        assert!(MarketImpactCalculator::new(invalid_params, 1000).is_err());
    }
}