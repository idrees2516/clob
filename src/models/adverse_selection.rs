//! Adverse Selection Protection Module
//!
//! This module implements sophisticated adverse selection detection and protection
//! mechanisms for market making strategies. It provides:
//!
//! - Information asymmetry detection using price impact analysis
//! - Adverse selection premium calculation with dynamic adjustments
//! - Dynamic spread widening based on toxic flow detection
//! - Quote frequency adjustment during adverse conditions
//!
//! The implementation follows the mathematical framework from requirement 1.8:
//! - Information asymmetry measure: IA = |price_impact|/√(volume*volatility)
//! - Adverse selection premium: AS = β*IA*σ*√(T-t)
//! - Dynamic spread adjustment: δ_adjusted = δ_base + AS
//! - Quote frequency reduction: f_new = f_base * exp(-AS/threshold)

use crate::math::FixedPoint;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Errors specific to adverse selection protection
#[derive(Debug, Clone, PartialEq)]
pub enum AdverseSelectionError {
    /// Invalid parameters provided
    InvalidParameters(String),
    /// Insufficient data for calculation
    InsufficientData(String),
    /// Numerical computation error
    NumericalError(String),
}

impl std::fmt::Display for AdverseSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdverseSelectionError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            AdverseSelectionError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            AdverseSelectionError::NumericalError(msg) => write!(f, "Numerical error: {}", msg),
        }
    }
}

impl std::error::Error for AdverseSelectionError {}

/// Parameters for adverse selection protection
#[derive(Debug, Clone)]
pub struct AdverseSelectionParams {
    /// Sensitivity parameter for adverse selection premium (β)
    pub beta: FixedPoint,
    /// Threshold for quote frequency reduction
    pub frequency_threshold: FixedPoint,
    /// Maximum adverse selection premium as fraction of spread
    pub max_premium_ratio: FixedPoint,
    /// Minimum quote frequency as fraction of base frequency
    pub min_frequency_ratio: FixedPoint,
    /// Window size for price impact analysis
    pub impact_window_size: usize,
    /// Decay factor for exponential weighting
    pub decay_factor: FixedPoint,
    /// Toxicity detection threshold
    pub toxicity_threshold: FixedPoint,
    /// Information asymmetry smoothing factor
    pub ia_smoothing_factor: FixedPoint,
}

impl Default for AdverseSelectionParams {
    fn default() -> Self {
        Self {
            beta: FixedPoint::from_float(0.5),
            frequency_threshold: FixedPoint::from_float(0.1),
            max_premium_ratio: FixedPoint::from_float(0.5),
            min_frequency_ratio: FixedPoint::from_float(0.1),
            impact_window_size: 100,
            decay_factor: FixedPoint::from_float(0.95),
            toxicity_threshold: FixedPoint::from_float(0.3),
            ia_smoothing_factor: FixedPoint::from_float(0.9),
        }
    }
}

impl AdverseSelectionParams {
    /// Validate parameters for mathematical consistency
    pub fn validate(&self) -> Result<(), AdverseSelectionError> {
        if self.beta <= FixedPoint::ZERO {
            return Err(AdverseSelectionError::InvalidParameters(
                "Beta parameter must be positive".to_string()
            ));
        }
        
        if self.frequency_threshold <= FixedPoint::ZERO {
            return Err(AdverseSelectionError::InvalidParameters(
                "Frequency threshold must be positive".to_string()
            ));
        }
        
        if self.max_premium_ratio <= FixedPoint::ZERO || self.max_premium_ratio > FixedPoint::ONE {
            return Err(AdverseSelectionError::InvalidParameters(
                "Max premium ratio must be in (0, 1]".to_string()
            ));
        }
        
        if self.min_frequency_ratio <= FixedPoint::ZERO || self.min_frequency_ratio > FixedPoint::ONE {
            return Err(AdverseSelectionError::InvalidParameters(
                "Min frequency ratio must be in (0, 1]".to_string()
            ));
        }
        
        if self.impact_window_size == 0 {
            return Err(AdverseSelectionError::InvalidParameters(
                "Impact window size must be positive".to_string()
            ));
        }
        
        if self.decay_factor <= FixedPoint::ZERO || self.decay_factor >= FixedPoint::ONE {
            return Err(AdverseSelectionError::InvalidParameters(
                "Decay factor must be in (0, 1)".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Trade information for adverse selection analysis
#[derive(Debug, Clone)]
pub struct TradeInfo {
    /// Trade price
    pub price: FixedPoint,
    /// Trade volume (signed: positive for buy, negative for sell)
    pub volume: i64,
    /// Mid price at time of trade
    pub mid_price: FixedPoint,
    /// Market volatility at time of trade
    pub volatility: FixedPoint,
    /// Total market volume (bid + ask)
    pub total_volume: i64,
    /// Order flow imbalance
    pub order_flow_imbalance: FixedPoint,
    /// Timestamp in nanoseconds
    pub timestamp: u64,
}

/// Price impact measurement
#[derive(Debug, Clone)]
pub struct PriceImpact {
    /// Immediate price impact
    pub immediate_impact: FixedPoint,
    /// Permanent price impact
    pub permanent_impact: FixedPoint,
    /// Temporary price impact
    pub temporary_impact: FixedPoint,
    /// Volume-normalized impact
    pub normalized_impact: FixedPoint,
    /// Timestamp of measurement
    pub timestamp: u64,
}

/// Information asymmetry measurement
#[derive(Debug, Clone)]
pub struct InformationAsymmetry {
    /// Raw information asymmetry measure
    pub raw_measure: FixedPoint,
    /// Smoothed information asymmetry
    pub smoothed_measure: FixedPoint,
    /// Confidence level (0-1)
    pub confidence: FixedPoint,
    /// Number of observations used
    pub observation_count: usize,
}

/// Adverse selection protection state
#[derive(Debug, Clone)]
pub struct AdverseSelectionState {
    /// Current adverse selection premium
    pub premium: FixedPoint,
    /// Quote frequency adjustment factor
    pub frequency_adjustment: FixedPoint,
    /// Toxicity level (0-1)
    pub toxicity_level: FixedPoint,
    /// Information asymmetry measure
    pub information_asymmetry: InformationAsymmetry,
    /// Is protection active
    pub protection_active: bool,
    /// Last update timestamp
    pub last_update: u64,
}

/// Comprehensive adverse selection protection engine
#[derive(Debug)]
pub struct AdverseSelectionProtection {
    /// Configuration parameters
    params: AdverseSelectionParams,
    /// Historical trade data for analysis
    trade_history: VecDeque<TradeInfo>,
    /// Price impact measurements
    impact_history: VecDeque<PriceImpact>,
    /// Current information asymmetry measure
    current_ia: FixedPoint,
    /// Smoothed information asymmetry
    smoothed_ia: FixedPoint,
    /// Current toxicity level
    toxicity_level: FixedPoint,
    /// Base quote frequency (quotes per second)
    base_frequency: FixedPoint,
    /// Current protection state
    protection_state: AdverseSelectionState,
    /// Statistics for calibration
    impact_statistics: ImpactStatistics,
}

/// Statistics for price impact analysis
#[derive(Debug, Clone)]
struct ImpactStatistics {
    /// Mean price impact
    mean_impact: FixedPoint,
    /// Standard deviation of price impact
    std_impact: FixedPoint,
    /// Skewness of price impact distribution
    skewness: FixedPoint,
    /// Kurtosis of price impact distribution
    kurtosis: FixedPoint,
    /// Number of observations
    count: usize,
}

impl Default for ImpactStatistics {
    fn default() -> Self {
        Self {
            mean_impact: FixedPoint::ZERO,
            std_impact: FixedPoint::ZERO,
            skewness: FixedPoint::ZERO,
            kurtosis: FixedPoint::from_float(3.0), // Normal distribution kurtosis
            count: 0,
        }
    }
}

impl AdverseSelectionProtection {
    /// Create new adverse selection protection engine
    pub fn new(params: AdverseSelectionParams, base_frequency: FixedPoint) -> Result<Self, AdverseSelectionError> {
        params.validate()?;
        
        if base_frequency <= FixedPoint::ZERO {
            return Err(AdverseSelectionError::InvalidParameters(
                "Base frequency must be positive".to_string()
            ));
        }
        
        let protection_state = AdverseSelectionState {
            premium: FixedPoint::ZERO,
            frequency_adjustment: FixedPoint::ONE,
            toxicity_level: FixedPoint::ZERO,
            information_asymmetry: InformationAsymmetry {
                raw_measure: FixedPoint::ZERO,
                smoothed_measure: FixedPoint::ZERO,
                confidence: FixedPoint::ZERO,
                observation_count: 0,
            },
            protection_active: false,
            last_update: Self::current_timestamp(),
        };
        
        Ok(Self {
            params,
            trade_history: VecDeque::with_capacity(params.impact_window_size),
            impact_history: VecDeque::with_capacity(params.impact_window_size),
            current_ia: FixedPoint::ZERO,
            smoothed_ia: FixedPoint::ZERO,
            toxicity_level: FixedPoint::ZERO,
            base_frequency,
            protection_state,
            impact_statistics: ImpactStatistics::default(),
        })
    }
    
    /// Update adverse selection protection with new trade information
    pub fn update(&mut self, trade: TradeInfo) -> Result<AdverseSelectionState, AdverseSelectionError> {
        // Add trade to history
        self.trade_history.push_back(trade.clone());
        
        // Maintain window size
        if self.trade_history.len() > self.params.impact_window_size {
            self.trade_history.pop_front();
        }
        
        // Calculate price impact for this trade
        let price_impact = self.calculate_price_impact(&trade)?;
        self.impact_history.push_back(price_impact.clone());
        
        // Maintain impact history size
        if self.impact_history.len() > self.params.impact_window_size {
            self.impact_history.pop_front();
        }
        
        // Update impact statistics
        self.update_impact_statistics(&price_impact);
        
        // Calculate information asymmetry measure
        self.update_information_asymmetry(&trade, &price_impact)?;
        
        // Update toxicity level
        self.update_toxicity_level(&trade)?;
        
        // Calculate adverse selection premium
        let premium = self.calculate_adverse_selection_premium(&trade)?;
        
        // Calculate quote frequency adjustment
        let frequency_adjustment = self.calculate_frequency_adjustment(premium)?;
        
        // Determine if protection should be active
        let protection_active = self.should_activate_protection()?;
        
        // Update protection state
        self.protection_state = AdverseSelectionState {
            premium,
            frequency_adjustment,
            toxicity_level: self.toxicity_level,
            information_asymmetry: InformationAsymmetry {
                raw_measure: self.current_ia,
                smoothed_measure: self.smoothed_ia,
                confidence: self.calculate_confidence(),
                observation_count: self.impact_history.len(),
            },
            protection_active,
            last_update: Self::current_timestamp(),
        };
        
        Ok(self.protection_state.clone())
    }
    
    /// Calculate price impact for a trade
    fn calculate_price_impact(&self, trade: &TradeInfo) -> Result<PriceImpact, AdverseSelectionError> {
        if trade.total_volume <= 0 {
            return Err(AdverseSelectionError::InvalidParameters(
                "Total volume must be positive".to_string()
            ));
        }
        
        // Immediate impact: difference between trade price and mid price
        let immediate_impact = (trade.price - trade.mid_price).abs();
        
        // Estimate permanent and temporary components
        // Permanent impact: typically 30-50% of immediate impact
        let permanent_ratio = FixedPoint::from_float(0.4);
        let permanent_impact = immediate_impact * permanent_ratio;
        
        // Temporary impact: remainder of immediate impact
        let temporary_impact = immediate_impact - permanent_impact;
        
        // Volume-normalized impact
        let volume_factor = FixedPoint::from_int(trade.volume.abs()) / 
            FixedPoint::from_int(trade.total_volume);
        let normalized_impact = immediate_impact / volume_factor.max(FixedPoint::from_float(0.01));
        
        Ok(PriceImpact {
            immediate_impact,
            permanent_impact,
            temporary_impact,
            normalized_impact,
            timestamp: trade.timestamp,
        })
    }
    
    /// Update information asymmetry measure
    fn update_information_asymmetry(
        &mut self, 
        trade: &TradeInfo, 
        impact: &PriceImpact
    ) -> Result<(), AdverseSelectionError> {
        if trade.volatility <= FixedPoint::ZERO {
            return Err(AdverseSelectionError::InvalidParameters(
                "Volatility must be positive".to_string()
            ));
        }
        
        // Information asymmetry measure: IA = |price_impact|/√(volume*volatility)
        let volume_volatility_factor = (FixedPoint::from_int(trade.total_volume) * trade.volatility).sqrt();
        
        if volume_volatility_factor <= FixedPoint::ZERO {
            return Err(AdverseSelectionError::NumericalError(
                "Volume-volatility factor is zero or negative".to_string()
            ));
        }
        
        // Use normalized impact for better stability
        self.current_ia = impact.normalized_impact / volume_volatility_factor;
        
        // Apply exponential smoothing
        if self.smoothed_ia == FixedPoint::ZERO {
            self.smoothed_ia = self.current_ia;
        } else {
            self.smoothed_ia = self.params.ia_smoothing_factor * self.smoothed_ia + 
                (FixedPoint::ONE - self.params.ia_smoothing_factor) * self.current_ia;
        }
        
        Ok(())
    }
    
    /// Update toxicity level based on order flow patterns
    fn update_toxicity_level(&mut self, trade: &TradeInfo) -> Result<(), AdverseSelectionError> {
        if self.trade_history.len() < 2 {
            self.toxicity_level = FixedPoint::ZERO;
            return Ok(());
        }
        
        // Calculate various toxicity indicators
        let volume_imbalance = self.calculate_volume_imbalance();
        let price_momentum = self.calculate_price_momentum();
        let order_flow_toxicity = trade.order_flow_imbalance.abs();
        
        // Combine indicators with dynamic weights
        let volume_weight = FixedPoint::from_float(0.4);
        let momentum_weight = FixedPoint::from_float(0.3);
        let flow_weight = FixedPoint::from_float(0.3);
        
        self.toxicity_level = volume_weight * volume_imbalance + 
            momentum_weight * price_momentum + 
            flow_weight * order_flow_toxicity;
        
        // Apply bounds [0, 1]
        self.toxicity_level = self.toxicity_level.max(FixedPoint::ZERO).min(FixedPoint::ONE);
        
        Ok(())
    }
    
    /// Calculate volume imbalance indicator
    fn calculate_volume_imbalance(&self) -> FixedPoint {
        if self.trade_history.len() < 10 {
            return FixedPoint::ZERO;
        }
        
        let recent_trades = self.trade_history.iter().rev().take(10);
        let mut buy_volume = FixedPoint::ZERO;
        let mut sell_volume = FixedPoint::ZERO;
        
        for trade in recent_trades {
            if trade.volume > 0 {
                buy_volume = buy_volume + FixedPoint::from_int(trade.volume);
            } else {
                sell_volume = sell_volume + FixedPoint::from_int(-trade.volume);
            }
        }
        
        let total_volume = buy_volume + sell_volume;
        if total_volume <= FixedPoint::ZERO {
            return FixedPoint::ZERO;
        }
        
        ((buy_volume - sell_volume) / total_volume).abs()
    }
    
    /// Calculate price momentum indicator
    fn calculate_price_momentum(&self) -> FixedPoint {
        if self.trade_history.len() < 5 {
            return FixedPoint::ZERO;
        }
        
        let recent_trades: Vec<_> = self.trade_history.iter().rev().take(5).collect();
        let first_price = recent_trades.last().unwrap().price;
        let last_price = recent_trades.first().unwrap().price;
        let avg_volatility = recent_trades.iter()
            .map(|t| t.volatility)
            .sum::<FixedPoint>() / FixedPoint::from_int(recent_trades.len() as i64);
        
        if avg_volatility <= FixedPoint::ZERO {
            return FixedPoint::ZERO;
        }
        
        // Normalized price momentum
        ((last_price - first_price) / first_price / avg_volatility).abs()
    }
    
    /// Calculate adverse selection premium
    fn calculate_adverse_selection_premium(&self, trade: &TradeInfo) -> Result<FixedPoint, AdverseSelectionError> {
        // Get time to maturity (simplified - assume 1 hour default)
        let time_to_maturity = FixedPoint::from_float(3600.0); // 1 hour in seconds
        
        // Adverse selection premium: AS = β*IA*σ*√(T-t)
        let premium = self.params.beta * 
            self.smoothed_ia * 
            trade.volatility * 
            time_to_maturity.sqrt();
        
        // Apply maximum premium constraint
        let current_spread = (trade.price - trade.mid_price).abs() * FixedPoint::from_float(2.0);
        let max_premium = current_spread * self.params.max_premium_ratio;
        
        Ok(premium.min(max_premium))
    }
    
    /// Calculate quote frequency adjustment
    fn calculate_frequency_adjustment(&self, premium: FixedPoint) -> Result<FixedPoint, AdverseSelectionError> {
        // Quote frequency reduction: f_new = f_base * exp(-AS/threshold)
        let adjustment_factor = if premium > FixedPoint::ZERO {
            let ratio = premium / self.params.frequency_threshold;
            (-ratio).exp()
        } else {
            FixedPoint::ONE
        };
        
        // Apply minimum frequency constraint
        Ok(adjustment_factor.max(self.params.min_frequency_ratio))
    }
    
    /// Determine if protection should be activated
    fn should_activate_protection(&self) -> Result<bool, AdverseSelectionError> {
        // Activate protection if:
        // 1. Toxicity level exceeds threshold
        // 2. Information asymmetry is significantly elevated
        // 3. Recent price impacts are abnormally high
        
        let toxicity_trigger = self.toxicity_level > self.params.toxicity_threshold;
        
        let ia_trigger = self.smoothed_ia > FixedPoint::from_float(0.2); // Threshold for IA
        
        let impact_trigger = if self.impact_statistics.count > 10 {
            let recent_impact = self.impact_history.back()
                .map(|i| i.normalized_impact)
                .unwrap_or(FixedPoint::ZERO);
            let threshold = self.impact_statistics.mean_impact + 
                FixedPoint::from_float(2.0) * self.impact_statistics.std_impact;
            recent_impact > threshold
        } else {
            false
        };
        
        Ok(toxicity_trigger || ia_trigger || impact_trigger)
    }
    
    /// Update impact statistics for calibration
    fn update_impact_statistics(&mut self, impact: &PriceImpact) {
        let new_impact = impact.normalized_impact;
        
        if self.impact_statistics.count == 0 {
            self.impact_statistics.mean_impact = new_impact;
            self.impact_statistics.std_impact = FixedPoint::ZERO;
            self.impact_statistics.count = 1;
            return;
        }
        
        // Update running statistics using Welford's algorithm
        let count = FixedPoint::from_int(self.impact_statistics.count as i64);
        let new_count = count + FixedPoint::ONE;
        
        let delta = new_impact - self.impact_statistics.mean_impact;
        let new_mean = self.impact_statistics.mean_impact + delta / new_count;
        let delta2 = new_impact - new_mean;
        
        // Update variance (using sample variance)
        if self.impact_statistics.count > 1 {
            let old_var = self.impact_statistics.std_impact * self.impact_statistics.std_impact;
            let new_var = (count * old_var + delta * delta2) / new_count;
            self.impact_statistics.std_impact = new_var.sqrt();
        }
        
        self.impact_statistics.mean_impact = new_mean;
        self.impact_statistics.count += 1;
        
        // Maintain reasonable history for statistics
        if self.impact_statistics.count > self.params.impact_window_size * 2 {
            // Reset statistics to prevent stale data
            self.impact_statistics.count = self.params.impact_window_size;
        }
    }
    
    /// Calculate confidence in current measurements
    fn calculate_confidence(&self) -> FixedPoint {
        let min_observations = 20;
        let observation_ratio = self.impact_history.len() as f64 / min_observations as f64;
        let base_confidence = observation_ratio.min(1.0);
        
        // Adjust confidence based on data quality
        let volatility_consistency = if self.impact_statistics.std_impact > FixedPoint::ZERO {
            let cv = self.impact_statistics.std_impact / self.impact_statistics.mean_impact.abs();
            (FixedPoint::ONE / (FixedPoint::ONE + cv)).to_float()
        } else {
            1.0
        };
        
        FixedPoint::from_float(base_confidence * volatility_consistency)
    }
    
    /// Get current protection state
    pub fn get_state(&self) -> &AdverseSelectionState {
        &self.protection_state
    }
    
    /// Get current parameters
    pub fn get_parameters(&self) -> &AdverseSelectionParams {
        &self.params
    }
    
    /// Update parameters
    pub fn update_parameters(&mut self, new_params: AdverseSelectionParams) -> Result<(), AdverseSelectionError> {
        new_params.validate()?;
        self.params = new_params;
        Ok(())
    }
    
    /// Reset protection state (useful for testing or regime changes)
    pub fn reset(&mut self) {
        self.trade_history.clear();
        self.impact_history.clear();
        self.current_ia = FixedPoint::ZERO;
        self.smoothed_ia = FixedPoint::ZERO;
        self.toxicity_level = FixedPoint::ZERO;
        self.impact_statistics = ImpactStatistics::default();
        
        self.protection_state = AdverseSelectionState {
            premium: FixedPoint::ZERO,
            frequency_adjustment: FixedPoint::ONE,
            toxicity_level: FixedPoint::ZERO,
            information_asymmetry: InformationAsymmetry {
                raw_measure: FixedPoint::ZERO,
                smoothed_measure: FixedPoint::ZERO,
                confidence: FixedPoint::ZERO,
                observation_count: 0,
            },
            protection_active: false,
            last_update: Self::current_timestamp(),
        };
    }
    
    /// Get current timestamp in nanoseconds
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
    
    /// Get diagnostic information for monitoring
    pub fn get_diagnostics(&self) -> AdverseSelectionDiagnostics {
        AdverseSelectionDiagnostics {
            trade_count: self.trade_history.len(),
            impact_count: self.impact_history.len(),
            mean_impact: self.impact_statistics.mean_impact,
            std_impact: self.impact_statistics.std_impact,
            current_ia: self.current_ia,
            smoothed_ia: self.smoothed_ia,
            toxicity_level: self.toxicity_level,
            protection_active: self.protection_state.protection_active,
            confidence: self.protection_state.information_asymmetry.confidence,
        }
    }
}

/// Diagnostic information for monitoring adverse selection protection
#[derive(Debug, Clone)]
pub struct AdverseSelectionDiagnostics {
    pub trade_count: usize,
    pub impact_count: usize,
    pub mean_impact: FixedPoint,
    pub std_impact: FixedPoint,
    pub current_ia: FixedPoint,
    pub smoothed_ia: FixedPoint,
    pub toxicity_level: FixedPoint,
    pub protection_active: bool,
    pub confidence: FixedPoint,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    include!("adverse_selection_tests.rs");
    
    #[test]
    fn test_adverse_selection_protection_creation() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        
        let protection = AdverseSelectionProtection::new(params, base_frequency);
        assert!(protection.is_ok());
        
        let protection = protection.unwrap();
        assert_eq!(protection.get_state().premium, FixedPoint::ZERO);
        assert_eq!(protection.get_state().frequency_adjustment, FixedPoint::ONE);
        assert!(!protection.get_state().protection_active);
    }
    
    #[test]
    fn test_parameter_validation() {
        let mut params = AdverseSelectionParams::default();
        
        // Test invalid beta
        params.beta = FixedPoint::ZERO;
        assert!(params.validate().is_err());
        
        // Test invalid frequency threshold
        params = AdverseSelectionParams::default();
        params.frequency_threshold = FixedPoint::ZERO;
        assert!(params.validate().is_err());
        
        // Test invalid max premium ratio
        params = AdverseSelectionParams::default();
        params.max_premium_ratio = FixedPoint::from_float(1.5);
        assert!(params.validate().is_err());
        
        // Test valid parameters
        params = AdverseSelectionParams::default();
        assert!(params.validate().is_ok());
    }
    
    #[test]
    fn test_trade_update() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        let trade = TradeInfo {
            price: FixedPoint::from_float(100.05),
            volume: 1000,
            mid_price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            total_volume: 10000,
            order_flow_imbalance: FixedPoint::from_float(0.1),
            timestamp: AdverseSelectionProtection::current_timestamp(),
        };
        
        let result = protection.update(trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        assert!(state.premium >= FixedPoint::ZERO);
        assert!(state.frequency_adjustment > FixedPoint::ZERO);
        assert!(state.frequency_adjustment <= FixedPoint::ONE);
    }
    
    #[test]
    fn test_information_asymmetry_calculation() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Create a trade with significant price impact
        let trade = TradeInfo {
            price: FixedPoint::from_float(100.10), // 10 cents above mid
            volume: 1000,
            mid_price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            total_volume: 5000, // Relatively small total volume
            order_flow_imbalance: FixedPoint::from_float(0.3),
            timestamp: AdverseSelectionProtection::current_timestamp(),
        };
        
        let result = protection.update(trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        // Should detect some information asymmetry due to large price impact
        assert!(state.information_asymmetry.raw_measure > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_toxicity_detection() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Create sequence of trades with increasing toxicity
        for i in 0..20 {
            let trade = TradeInfo {
                price: FixedPoint::from_float(100.0 + i as f64 * 0.01), // Trending price
                volume: if i % 2 == 0 { 1000 } else { -1000 }, // Alternating buy/sell
                mid_price: FixedPoint::from_float(100.0 + i as f64 * 0.005),
                volatility: FixedPoint::from_float(0.2),
                total_volume: 5000,
                order_flow_imbalance: FixedPoint::from_float(0.5), // High imbalance
                timestamp: AdverseSelectionProtection::current_timestamp() + i as u64 * 1000000,
            };
            
            let _ = protection.update(trade);
        }
        
        let state = protection.get_state();
        // Should detect elevated toxicity
        assert!(state.toxicity_level > FixedPoint::ZERO);
    }
    
    #[test]
    fn test_frequency_adjustment() {
        let params = AdverseSelectionParams::default();
        let base_frequency = FixedPoint::from_float(10.0);
        let mut protection = AdverseSelectionProtection::new(params, base_frequency).unwrap();
        
        // Create trade with high adverse selection premium
        let trade = TradeInfo {
            price: FixedPoint::from_float(100.20), // Large price impact
            volume: 2000,
            mid_price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.3), // High volatility
            total_volume: 3000, // Small total volume
            order_flow_imbalance: FixedPoint::from_float(0.8), // Very high imbalance
            timestamp: AdverseSelectionProtection::current_timestamp(),
        };
        
        let result = protection.update(trade);
        assert!(result.is_ok());
        
        let state = result.unwrap();
        // Should reduce quote frequency due to high adverse selection
        assert!(state.frequency_adjustment < FixedPoint::ONE);
        assert!(state.frequency_adjustment >= params.min_frequency_ratio);
    }
}