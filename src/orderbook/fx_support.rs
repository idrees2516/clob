//! Foreign Exchange (FX) and multi-currency support module
//!
//! This module provides comprehensive FX support including multi-currency position tracking,
//! FX hedging algorithms, carry trade optimization, and currency risk management.

use crate::orderbook::{Order, OrderId, Symbol, Side, Trade, OrderBookError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use thiserror::Error;

/// FX-specific errors
#[derive(Error, Debug)]
pub enum FxError {
    #[error("Currency not supported: {0}")]
    UnsupportedCurrency(String),
    #[error("Exchange rate not available: {0}/{1}")]
    ExchangeRateUnavailable(String, String),
    #[error("Insufficient liquidity for currency: {0}")]
    InsufficientLiquidity(String),
    #[error("Hedging error: {0}")]
    HedgingError(String),
    #[error("Carry trade error: {0}")]
    CarryTradeError(String),
    #[error("Order book error: {0}")]
    OrderBookError(#[from] OrderBookError),
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
}

pub type FxResult<T> = Result<T, FxError>;

/// Currency identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Currency(pub String);

impl Currency {
    pub fn new(code: &str) -> Self {
        Self(code.to_uppercase())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
    
    /// Major currencies
    pub fn usd() -> Self { Self("USD".to_string()) }
    pub fn eur() -> Self { Self("EUR".to_string()) }
    pub fn gbp() -> Self { Self("GBP".to_string()) }
    pub fn jpy() -> Self { Self("JPY".to_string()) }
    pub fn chf() -> Self { Self("CHF".to_string()) }
    pub fn cad() -> Self { Self("CAD".to_string()) }
    pub fn aud() -> Self { Self("AUD".to_string()) }
    pub fn nzd() -> Self { Self("NZD".to_string()) }
    
    /// Crypto currencies
    pub fn btc() -> Self { Self("BTC".to_string()) }
    pub fn eth() -> Self { Self("ETH".to_string()) }
    pub fn sol() -> Self { Self("SOL".to_string()) }
}

/// Currency pair for FX trading
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CurrencyPair {
    pub base: Currency,
    pub quote: Currency,
}

impl CurrencyPair {
    pub fn new(base: Currency, quote: Currency) -> Self {
        Self { base, quote }
    }
    
    pub fn from_symbol(symbol: &Symbol) -> Option<Self> {
        let parts: Vec<&str> = symbol.as_str().split('/').collect();
        if parts.len() == 2 {
            Some(Self {
                base: Currency::new(parts[0]),
                quote: Currency::new(parts[1]),
            })
        } else {
            None
        }
    }
    
    pub fn to_symbol(&self) -> Symbol {
        Symbol::new(&format!("{}/{}", self.base.as_str(), self.quote.as_str()))
    }
    
    pub fn inverse(&self) -> Self {
        Self {
            base: self.quote.clone(),
            quote: self.base.clone(),
        }
    }
}

/// Exchange rate with timestamp and source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRate {
    pub pair: CurrencyPair,
    pub rate: f64,
    pub bid: f64,
    pub ask: f64,
    pub timestamp: u64,
    pub source: String,
    pub confidence: f64, // 0.0 to 1.0
}

/// Multi-currency position tracker
#[derive(Debug, Clone, Default)]
pub struct MultiCurrencyPosition {
    /// Positions by currency (positive = long, negative = short)
    pub positions: HashMap<Currency, i64>,
    
    /// Base currency for P&L calculation
    pub base_currency: Currency,
    
    /// Last update timestamp
    pub last_update: u64,
    
    /// Realized P&L by currency
    pub realized_pnl: HashMap<Currency, i64>,
    
    /// Unrealized P&L by currency
    pub unrealized_pnl: HashMap<Currency, i64>,
}

impl MultiCurrencyPosition {
    pub fn new(base_currency: Currency) -> Self {
        Self {
            positions: HashMap::new(),
            base_currency,
            last_update: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            realized_pnl: HashMap::new(),
            unrealized_pnl: HashMap::new(),
        }
    }
    
    /// Update position from a trade
    pub fn update_from_trade(&mut self, trade: &Trade, pair: &CurrencyPair) {
        let base_change = match trade.side {
            Side::Buy => trade.quantity as i64,
            Side::Sell => -(trade.quantity as i64),
        };
        
        let quote_change = -base_change * (trade.price as i64) / 1_000_000; // Convert from fixed-point
        
        *self.positions.entry(pair.base.clone()).or_insert(0) += base_change;
        *self.positions.entry(pair.quote.clone()).or_insert(0) += quote_change;
        
        self.last_update = trade.timestamp;
    }
    
    /// Get position in specific currency
    pub fn get_position(&self, currency: &Currency) -> i64 {
        self.positions.get(currency).copied().unwrap_or(0)
    }
    
    /// Get total exposure in base currency
    pub fn get_total_exposure(&self, fx_rates: &FxRateProvider) -> FxResult<i64> {
        let mut total_exposure = 0i64;
        
        for (currency, position) in &self.positions {
            if currency == &self.base_currency {
                total_exposure += position;
            } else {
                let rate = fx_rates.get_rate(&CurrencyPair::new(currency.clone(), self.base_currency.clone()))?;
                total_exposure += (*position as f64 * rate.rate) as i64;
            }
        }
        
        Ok(total_exposure)
    }
    
    /// Calculate unrealized P&L
    pub fn calculate_unrealized_pnl(&mut self, fx_rates: &FxRateProvider) -> FxResult<()> {
        self.unrealized_pnl.clear();
        
        for (currency, position) in &self.positions {
            if *position != 0 {
                if currency == &self.base_currency {
                    self.unrealized_pnl.insert(currency.clone(), 0); // No P&L in base currency
                } else {
                    // Calculate P&L based on current exchange rate vs entry rate
                    let current_rate = fx_rates.get_rate(&CurrencyPair::new(currency.clone(), self.base_currency.clone()))?;
                    // Simplified P&L calculation (would need entry rates for accurate calculation)
                    let pnl = (*position as f64 * current_rate.rate * 0.01) as i64; // Placeholder calculation
                    self.unrealized_pnl.insert(currency.clone(), pnl);
                }
            }
        }
        
        Ok(())
    }
}

/// FX rate provider interface
pub trait FxRateProvider: Send + Sync {
    fn get_rate(&self, pair: &CurrencyPair) -> FxResult<ExchangeRate>;
    fn get_all_rates(&self) -> FxResult<Vec<ExchangeRate>>;
    fn subscribe_to_updates(&self, callback: Box<dyn Fn(ExchangeRate) + Send + Sync>);
}

/// Mock FX rate provider for testing
pub struct MockFxRateProvider {
    rates: Arc<Mutex<HashMap<CurrencyPair, ExchangeRate>>>,
}

impl MockFxRateProvider {
    pub fn new() -> Self {
        let mut rates = HashMap::new();
        
        // Add some default rates
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
        
        // Major FX pairs
        rates.insert(
            CurrencyPair::new(Currency::eur(), Currency::usd()),
            ExchangeRate {
                pair: CurrencyPair::new(Currency::eur(), Currency::usd()),
                rate: 1.0850,
                bid: 1.0849,
                ask: 1.0851,
                timestamp,
                source: "mock".to_string(),
                confidence: 1.0,
            }
        );
        
        rates.insert(
            CurrencyPair::new(Currency::gbp(), Currency::usd()),
            ExchangeRate {
                pair: CurrencyPair::new(Currency::gbp(), Currency::usd()),
                rate: 1.2650,
                bid: 1.2649,
                ask: 1.2651,
                timestamp,
                source: "mock".to_string(),
                confidence: 1.0,
            }
        );
        
        rates.insert(
            CurrencyPair::new(Currency::usd(), Currency::jpy()),
            ExchangeRate {
                pair: CurrencyPair::new(Currency::usd(), Currency::jpy()),
                rate: 150.25,
                bid: 150.24,
                ask: 150.26,
                timestamp,
                source: "mock".to_string(),
                confidence: 1.0,
            }
        );
        
        // Crypto pairs
        rates.insert(
            CurrencyPair::new(Currency::btc(), Currency::usd()),
            ExchangeRate {
                pair: CurrencyPair::new(Currency::btc(), Currency::usd()),
                rate: 50000.0,
                bid: 49995.0,
                ask: 50005.0,
                timestamp,
                source: "mock".to_string(),
                confidence: 0.95,
            }
        );
        
        Self {
            rates: Arc::new(Mutex::new(rates)),
        }
    }
    
    pub fn set_rate(&self, pair: CurrencyPair, rate: f64, spread: f64) {
        let mut rates = self.rates.lock().unwrap();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
        
        rates.insert(pair.clone(), ExchangeRate {
            pair,
            rate,
            bid: rate - spread / 2.0,
            ask: rate + spread / 2.0,
            timestamp,
            source: "mock".to_string(),
            confidence: 1.0,
        });
    }
}

impl FxRateProvider for MockFxRateProvider {
    fn get_rate(&self, pair: &CurrencyPair) -> FxResult<ExchangeRate> {
        let rates = self.rates.lock().unwrap();
        
        // Try direct lookup
        if let Some(rate) = rates.get(pair) {
            return Ok(rate.clone());
        }
        
        // Try inverse lookup
        let inverse_pair = pair.inverse();
        if let Some(inverse_rate) = rates.get(&inverse_pair) {
            return Ok(ExchangeRate {
                pair: pair.clone(),
                rate: 1.0 / inverse_rate.rate,
                bid: 1.0 / inverse_rate.ask,
                ask: 1.0 / inverse_rate.bid,
                timestamp: inverse_rate.timestamp,
                source: inverse_rate.source.clone(),
                confidence: inverse_rate.confidence,
            });
        }
        
        Err(FxError::ExchangeRateUnavailable(
            pair.base.as_str().to_string(),
            pair.quote.as_str().to_string()
        ))
    }
    
    fn get_all_rates(&self) -> FxResult<Vec<ExchangeRate>> {
        let rates = self.rates.lock().unwrap();
        Ok(rates.values().cloned().collect())
    }
    
    fn subscribe_to_updates(&self, _callback: Box<dyn Fn(ExchangeRate) + Send + Sync>) {
        // Mock implementation - would implement real-time updates in production
    }
}

impl Default for MockFxRateProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// FX hedging strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HedgingStrategy {
    /// No hedging
    None,
    /// Full hedging - hedge 100% of FX exposure
    Full,
    /// Partial hedging - hedge a percentage of FX exposure
    Partial { hedge_ratio: f64 },
    /// Dynamic hedging based on volatility
    Dynamic { min_ratio: f64, max_ratio: f64 },
    /// Carry trade optimization
    CarryOptimized { target_carry: f64 },
}

/// FX hedging manager
pub struct FxHedgingManager {
    fx_rates: Arc<dyn FxRateProvider>,
    positions: Arc<Mutex<HashMap<String, MultiCurrencyPosition>>>,
    hedging_strategy: HedgingStrategy,
    base_currency: Currency,
    hedge_orders: Arc<Mutex<Vec<HedgeOrder>>>,
}

/// Hedge order for FX risk management
#[derive(Debug, Clone)]
pub struct HedgeOrder {
    pub id: OrderId,
    pub pair: CurrencyPair,
    pub side: Side,
    pub quantity: u64,
    pub hedge_ratio: f64,
    pub created_at: u64,
    pub status: HedgeOrderStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HedgeOrderStatus {
    Pending,
    Submitted,
    Filled,
    Cancelled,
    Failed,
}

impl FxHedgingManager {
    pub fn new(
        fx_rates: Arc<dyn FxRateProvider>,
        hedging_strategy: HedgingStrategy,
        base_currency: Currency,
    ) -> Self {
        Self {
            fx_rates,
            positions: Arc::new(Mutex::new(HashMap::new())),
            hedging_strategy,
            base_currency,
            hedge_orders: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Update position and calculate hedging requirements
    pub fn update_position(&self, account_id: &str, trade: &Trade, pair: &CurrencyPair) -> FxResult<Vec<Order>> {
        let mut positions = self.positions.lock().unwrap();
        let position = positions.entry(account_id.to_string())
            .or_insert_with(|| MultiCurrencyPosition::new(self.base_currency.clone()));
        
        position.update_from_trade(trade, pair);
        drop(positions);
        
        // Calculate hedging requirements
        self.calculate_hedge_orders(account_id)
    }
    
    /// Calculate required hedge orders
    pub fn calculate_hedge_orders(&self, account_id: &str) -> FxResult<Vec<Order>> {
        let positions = self.positions.lock().unwrap();
        let position = positions.get(account_id)
            .ok_or_else(|| FxError::HedgingError("Position not found".to_string()))?;
        
        let mut hedge_orders = Vec::new();
        
        match &self.hedging_strategy {
            HedgingStrategy::None => {
                // No hedging required
            }
            HedgingStrategy::Full => {
                hedge_orders = self.calculate_full_hedge(position)?;
            }
            HedgingStrategy::Partial { hedge_ratio } => {
                hedge_orders = self.calculate_partial_hedge(position, *hedge_ratio)?;
            }
            HedgingStrategy::Dynamic { min_ratio, max_ratio } => {
                let dynamic_ratio = self.calculate_dynamic_hedge_ratio(*min_ratio, *max_ratio)?;
                hedge_orders = self.calculate_partial_hedge(position, dynamic_ratio)?;
            }
            HedgingStrategy::CarryOptimized { target_carry } => {
                hedge_orders = self.calculate_carry_optimized_hedge(position, *target_carry)?;
            }
        }
        
        Ok(hedge_orders)
    }
    
    /// Calculate full hedge orders
    fn calculate_full_hedge(&self, position: &MultiCurrencyPosition) -> FxResult<Vec<Order>> {
        let mut orders = Vec::new();
        
        for (currency, pos) in &position.positions {
            if currency != &self.base_currency && *pos != 0 {
                let pair = CurrencyPair::new(currency.clone(), self.base_currency.clone());
                let rate = self.fx_rates.get_rate(&pair)?;
                
                // Create hedge order to neutralize FX exposure
                let hedge_side = if *pos > 0 { Side::Sell } else { Side::Buy };
                let hedge_quantity = pos.abs() as u64;
                
                let order = Order {
                    id: OrderId::new(&format!("hedge_{}_{}", currency.as_str(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis())),
                    symbol: pair.to_symbol(),
                    side: hedge_side,
                    quantity: hedge_quantity,
                    price: ((rate.ask + rate.bid) / 2.0 * 1_000_000.0) as u64, // Convert to fixed-point
                    order_type: crate::orderbook::OrderType::Market,
                    time_in_force: crate::orderbook::TimeInForce::IOC,
                    timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
                };
                
                orders.push(order);
            }
        }
        
        Ok(orders)
    }
    
    /// Calculate partial hedge orders
    fn calculate_partial_hedge(&self, position: &MultiCurrencyPosition, hedge_ratio: f64) -> FxResult<Vec<Order>> {
        let full_hedge = self.calculate_full_hedge(position)?;
        
        // Scale down quantities by hedge ratio
        let partial_hedge = full_hedge.into_iter().map(|mut order| {
            order.quantity = (order.quantity as f64 * hedge_ratio) as u64;
            order
        }).collect();
        
        Ok(partial_hedge)
    }
    
    /// Calculate dynamic hedge ratio based on market conditions
    fn calculate_dynamic_hedge_ratio(&self, min_ratio: f64, max_ratio: f64) -> FxResult<f64> {
        // Simplified dynamic calculation - would use volatility, correlation, etc.
        let base_ratio = (min_ratio + max_ratio) / 2.0;
        
        // Could incorporate:
        // - FX volatility
        // - Correlation between currencies
        // - Market stress indicators
        // - Time to expiry of positions
        
        Ok(base_ratio)
    }
    
    /// Calculate carry trade optimized hedge
    fn calculate_carry_optimized_hedge(&self, position: &MultiCurrencyPosition, target_carry: f64) -> FxResult<Vec<Order>> {
        // Simplified carry trade optimization
        // In practice, would consider:
        // - Interest rate differentials
        // - Funding costs
        // - Rollover rates
        // - Expected returns vs risk
        
        let mut orders = Vec::new();
        
        for (currency, pos) in &position.positions {
            if currency != &self.base_currency && *pos != 0 {
                let carry_rate = self.get_carry_rate(currency)?;
                
                if carry_rate < target_carry {
                    // Reduce exposure to low-carry currencies
                    let reduction_ratio = (target_carry - carry_rate) / target_carry;
                    let hedge_quantity = (pos.abs() as f64 * reduction_ratio) as u64;
                    
                    if hedge_quantity > 0 {
                        let pair = CurrencyPair::new(currency.clone(), self.base_currency.clone());
                        let rate = self.fx_rates.get_rate(&pair)?;
                        let hedge_side = if *pos > 0 { Side::Sell } else { Side::Buy };
                        
                        let order = Order {
                            id: OrderId::new(&format!("carry_hedge_{}_{}", currency.as_str(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis())),
                            symbol: pair.to_symbol(),
                            side: hedge_side,
                            quantity: hedge_quantity,
                            price: ((rate.ask + rate.bid) / 2.0 * 1_000_000.0) as u64,
                            order_type: crate::orderbook::OrderType::Market,
                            time_in_force: crate::orderbook::TimeInForce::IOC,
                            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
                        };
                        
                        orders.push(order);
                    }
                }
            }
        }
        
        Ok(orders)
    }
    
    /// Get carry rate for currency (simplified)
    fn get_carry_rate(&self, currency: &Currency) -> FxResult<f64> {
        // Simplified carry rate calculation
        // In practice, would fetch real interest rates and funding costs
        let carry_rate = match currency.as_str() {
            "USD" => 0.05,  // 5% base rate
            "EUR" => 0.02,  // 2% rate
            "JPY" => -0.01, // -1% rate (negative rates)
            "GBP" => 0.04,  // 4% rate
            "AUD" => 0.06,  // 6% rate
            "NZD" => 0.055, // 5.5% rate
            "CHF" => 0.01,  // 1% rate
            "CAD" => 0.045, // 4.5% rate
            _ => 0.03,      // Default 3% rate
        };
        
        Ok(carry_rate)
    }
    
    /// Get current positions
    pub fn get_positions(&self) -> HashMap<String, MultiCurrencyPosition> {
        self.positions.lock().unwrap().clone()
    }
    
    /// Get position for specific account
    pub fn get_position(&self, account_id: &str) -> Option<MultiCurrencyPosition> {
        self.positions.lock().unwrap().get(account_id).cloned()
    }
    
    /// Calculate total FX exposure across all accounts
    pub fn calculate_total_fx_exposure(&self) -> FxResult<HashMap<Currency, i64>> {
        let positions = self.positions.lock().unwrap();
        let mut total_exposure = HashMap::new();
        
        for position in positions.values() {
            for (currency, amount) in &position.positions {
                *total_exposure.entry(currency.clone()).or_insert(0) += amount;
            }
        }
        
        Ok(total_exposure)
    }
}

/// Currency risk manager
pub struct CurrencyRiskManager {
    fx_hedging_manager: FxHedgingManager,
    risk_limits: CurrencyRiskLimits,
    current_exposures: Arc<Mutex<HashMap<Currency, i64>>>,
}

/// Currency risk limits
#[derive(Debug, Clone)]
pub struct CurrencyRiskLimits {
    pub max_exposure_per_currency: HashMap<Currency, u64>,
    pub max_total_fx_exposure: u64,
    pub max_carry_exposure: u64,
    pub max_correlation_exposure: f64,
    pub var_limit: u64, // Value at Risk limit
}

impl Default for CurrencyRiskLimits {
    fn default() -> Self {
        let mut max_exposure = HashMap::new();
        
        // Set default limits for major currencies
        max_exposure.insert(Currency::usd(), 10_000_000); // $10M
        max_exposure.insert(Currency::eur(), 8_000_000);  // €8M
        max_exposure.insert(Currency::gbp(), 6_000_000);  // £6M
        max_exposure.insert(Currency::jpy(), 1_000_000_000); // ¥1B
        
        Self {
            max_exposure_per_currency: max_exposure,
            max_total_fx_exposure: 50_000_000, // $50M total
            max_carry_exposure: 20_000_000,    // $20M carry exposure
            max_correlation_exposure: 0.8,     // 80% max correlation
            var_limit: 1_000_000,              // $1M VaR limit
        }
    }
}

impl CurrencyRiskManager {
    pub fn new(
        fx_hedging_manager: FxHedgingManager,
        risk_limits: CurrencyRiskLimits,
    ) -> Self {
        Self {
            fx_hedging_manager,
            risk_limits,
            current_exposures: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Check if trade is allowed under currency risk limits
    pub fn check_trade_risk(&self, trade: &Trade, pair: &CurrencyPair) -> FxResult<bool> {
        let mut exposures = self.current_exposures.lock().unwrap();
        
        // Calculate new exposure after trade
        let base_change = match trade.side {
            Side::Buy => trade.quantity as i64,
            Side::Sell => -(trade.quantity as i64),
        };
        
        let new_base_exposure = exposures.get(&pair.base).unwrap_or(&0) + base_change;
        let new_quote_exposure = exposures.get(&pair.quote).unwrap_or(&0) - base_change * (trade.price as i64) / 1_000_000;
        
        // Check individual currency limits
        if let Some(limit) = self.risk_limits.max_exposure_per_currency.get(&pair.base) {
            if new_base_exposure.abs() as u64 > *limit {
                return Ok(false);
            }
        }
        
        if let Some(limit) = self.risk_limits.max_exposure_per_currency.get(&pair.quote) {
            if new_quote_exposure.abs() as u64 > *limit {
                return Ok(false);
            }
        }
        
        // Check total FX exposure
        let total_exposure: u64 = exposures.values().map(|exp| exp.abs() as u64).sum();
        if total_exposure > self.risk_limits.max_total_fx_exposure {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Update exposure after trade
    pub fn update_exposure(&self, trade: &Trade, pair: &CurrencyPair) -> FxResult<()> {
        let mut exposures = self.current_exposures.lock().unwrap();
        
        let base_change = match trade.side {
            Side::Buy => trade.quantity as i64,
            Side::Sell => -(trade.quantity as i64),
        };
        
        let quote_change = -base_change * (trade.price as i64) / 1_000_000;
        
        *exposures.entry(pair.base.clone()).or_insert(0) += base_change;
        *exposures.entry(pair.quote.clone()).or_insert(0) += quote_change;
        
        Ok(())
    }
    
    /// Get current currency exposures
    pub fn get_exposures(&self) -> HashMap<Currency, i64> {
        self.current_exposures.lock().unwrap().clone()
    }
    
    /// Calculate Value at Risk (simplified)
    pub fn calculate_var(&self, confidence_level: f64, time_horizon_days: u32) -> FxResult<u64> {
        let exposures = self.current_exposures.lock().unwrap();
        
        // Simplified VaR calculation using historical simulation method
        // In practice, would use more sophisticated models (Monte Carlo, parametric, etc.)
        
        let mut total_var = 0.0;
        
        for (currency, exposure) in exposures.iter() {
            if *exposure != 0 {
                // Get historical volatility (simplified)
                let volatility = self.get_currency_volatility(currency)?;
                
                // Calculate VaR for this currency
                let z_score = self.get_z_score(confidence_level);
                let var = (*exposure as f64).abs() * volatility * z_score * (time_horizon_days as f64).sqrt();
                
                total_var += var * var; // Sum of squares for portfolio VaR
            }
        }
        
        Ok(total_var.sqrt() as u64)
    }
    
    /// Get currency volatility (simplified)
    fn get_currency_volatility(&self, currency: &Currency) -> FxResult<f64> {
        // Simplified volatility lookup
        let volatility = match currency.as_str() {
            "USD" => 0.08,  // 8% annual volatility
            "EUR" => 0.10,  // 10% annual volatility
            "GBP" => 0.12,  // 12% annual volatility
            "JPY" => 0.09,  // 9% annual volatility
            "CHF" => 0.07,  // 7% annual volatility
            "AUD" => 0.15,  // 15% annual volatility
            "NZD" => 0.16,  // 16% annual volatility
            "CAD" => 0.11,  // 11% annual volatility
            "BTC" => 0.80,  // 80% annual volatility
            "ETH" => 0.90,  // 90% annual volatility
            _ => 0.20,      // Default 20% volatility
        };
        
        Ok(volatility)
    }
    
    /// Get Z-score for confidence level
    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Simplified Z-score lookup for normal distribution
        match confidence_level {
            0.90 => 1.28,
            0.95 => 1.65,
            0.99 => 2.33,
            _ => 1.65, // Default to 95%
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_currency_pair_creation() {
        let pair = CurrencyPair::new(Currency::eur(), Currency::usd());
        assert_eq!(pair.base.as_str(), "EUR");
        assert_eq!(pair.quote.as_str(), "USD");
        
        let symbol = pair.to_symbol();
        assert_eq!(symbol.as_str(), "EUR/USD");
        
        let parsed_pair = CurrencyPair::from_symbol(&symbol).unwrap();
        assert_eq!(parsed_pair, pair);
    }
    
    #[test]
    fn test_multi_currency_position() {
        let mut position = MultiCurrencyPosition::new(Currency::usd());
        
        let trade = Trade {
            id: "test_trade".to_string(),
            buy_order_id: OrderId::new("buy_order"),
            sell_order_id: OrderId::new("sell_order"),
            symbol: Symbol::new("EUR/USD"),
            price: 1_085_000, // 1.085 in fixed-point
            quantity: 100_000, // 0.1 EUR
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            side: Side::Buy,
        };
        
        let pair = CurrencyPair::new(Currency::eur(), Currency::usd());
        position.update_from_trade(&trade, &pair);
        
        assert_eq!(position.get_position(&Currency::eur()), 100_000);
        assert!(position.get_position(&Currency::usd()) < 0); // Should be negative (spent USD)
    }
    
    #[test]
    fn test_fx_rate_provider() {
        let provider = MockFxRateProvider::new();
        
        let eur_usd = CurrencyPair::new(Currency::eur(), Currency::usd());
        let rate = provider.get_rate(&eur_usd).unwrap();
        
        assert!(rate.rate > 1.0);
        assert!(rate.bid < rate.ask);
        
        // Test inverse lookup
        let usd_eur = CurrencyPair::new(Currency::usd(), Currency::eur());
        let inverse_rate = provider.get_rate(&usd_eur).unwrap();
        
        assert!((rate.rate * inverse_rate.rate - 1.0).abs() < 0.001); // Should be approximately 1.0
    }
    
    #[tokio::test]
    async fn test_fx_hedging_manager() {
        let fx_rates = Arc::new(MockFxRateProvider::new());
        let hedging_strategy = HedgingStrategy::Full;
        let base_currency = Currency::usd();
        
        let hedging_manager = FxHedgingManager::new(fx_rates, hedging_strategy, base_currency);
        
        let trade = Trade {
            id: "test_trade".to_string(),
            buy_order_id: OrderId::new("buy_order"),
            sell_order_id: OrderId::new("sell_order"),
            symbol: Symbol::new("EUR/USD"),
            price: 1_085_000,
            quantity: 100_000,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            side: Side::Buy,
        };
        
        let pair = CurrencyPair::new(Currency::eur(), Currency::usd());
        let hedge_orders = hedging_manager.update_position("test_account", &trade, &pair).unwrap();
        
        assert!(!hedge_orders.is_empty(), "Should generate hedge orders");
        assert_eq!(hedge_orders[0].side, Side::Sell); // Should hedge by selling EUR
    }
}