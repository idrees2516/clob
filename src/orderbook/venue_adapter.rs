//! Cross-venue integration and adapter module
//!
//! This module provides venue-specific adapters, cross-venue arbitrage detection,
//! smart order routing, and venue-specific risk controls for multi-venue trading.

use crate::orderbook::{Order, OrderId, Symbol, Trade, OrderResult, OrderBookError};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;

/// Venue-specific errors
#[derive(Error, Debug)]
pub enum VenueError {
    #[error("Connection error: {0}")]
    ConnectionError(String),
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),
    #[error("Venue-specific error: {0}")]
    VenueSpecificError(String),
    #[error("Order book error: {0}")]
    OrderBookError(#[from] OrderBookError),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Network timeout")]
    NetworkTimeout,
    #[error("Invalid response format")]
    InvalidResponse,
}

pub type VenueResult<T> = Result<T, VenueError>;

/// Venue identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VenueId(pub String);

impl VenueId {
    pub fn new(id: &str) -> Self {
        Self(id.to_string())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Venue-specific order information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueOrder {
    pub venue_id: VenueId,
    pub venue_order_id: String,
    pub internal_order_id: OrderId,
    pub order: Order,
    pub status: VenueOrderStatus,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Venue order status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VenueOrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

/// Venue-specific trade information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueTrade {
    pub venue_id: VenueId,
    pub venue_trade_id: String,
    pub trade: Trade,
    pub fees: VenueFees,
    pub timestamp: u64,
}

/// Venue fee structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueFees {
    pub maker_fee: u64,    // Fixed-point representation
    pub taker_fee: u64,    // Fixed-point representation
    pub currency: String,
}

/// Venue capabilities and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueCapabilities {
    pub venue_id: VenueId,
    pub supported_symbols: Vec<Symbol>,
    pub min_order_size: HashMap<Symbol, u64>,
    pub max_order_size: HashMap<Symbol, u64>,
    pub tick_size: HashMap<Symbol, u64>,
    pub rate_limits: RateLimits,
    pub order_types: Vec<String>,
    pub time_in_force_options: Vec<String>,
    pub supports_post_only: bool,
    pub supports_ioc: bool,
    pub supports_fok: bool,
    pub latency_estimate_us: u64,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub orders_per_second: u32,
    pub requests_per_minute: u32,
    pub weight_per_order: u32,
    pub max_weight_per_minute: u32,
}

/// Venue adapter trait for implementing venue-specific logic
#[async_trait]
pub trait VenueAdapter: Send + Sync {
    /// Get venue capabilities
    async fn get_capabilities(&self) -> VenueResult<VenueCapabilities>;
    
    /// Submit order to venue
    async fn submit_order(&self, order: Order) -> VenueResult<VenueOrder>;
    
    /// Cancel order on venue
    async fn cancel_order(&self, venue_order_id: &str) -> VenueResult<VenueOrder>;
    
    /// Get order status from venue
    async fn get_order_status(&self, venue_order_id: &str) -> VenueResult<VenueOrder>;
    
    /// Get recent trades for symbol
    async fn get_recent_trades(&self, symbol: &Symbol, limit: usize) -> VenueResult<Vec<VenueTrade>>;
    
    /// Get current best bid/ask
    async fn get_best_bid_ask(&self, symbol: &Symbol) -> VenueResult<(Option<u64>, Option<u64>)>;
    
    /// Get order book depth
    async fn get_order_book(&self, symbol: &Symbol, depth: usize) -> VenueResult<VenueOrderBook>;
    
    /// Check connection health
    async fn health_check(&self) -> VenueResult<bool>;
    
    /// Get venue-specific fees
    async fn get_fees(&self, symbol: &Symbol) -> VenueResult<VenueFees>;
}

/// Venue order book representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueOrderBook {
    pub venue_id: VenueId,
    pub symbol: Symbol,
    pub bids: Vec<(u64, u64)>,  // (price, volume) pairs
    pub asks: Vec<(u64, u64)>,  // (price, volume) pairs
    pub timestamp: u64,
}

/// Cross-venue arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub buy_venue: VenueId,
    pub sell_venue: VenueId,
    pub symbol: Symbol,
    pub buy_price: u64,
    pub sell_price: u64,
    pub max_volume: u64,
    pub profit_estimate: i64,  // After fees and costs
    pub confidence: f64,       // 0.0 to 1.0
    pub detected_at: u64,
    pub expires_at: u64,
}

/// Smart order routing decision
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub venue_allocations: Vec<VenueAllocation>,
    pub expected_execution_price: u64,
    pub expected_fees: u64,
    pub estimated_latency_us: u64,
    pub confidence: f64,
}

/// Venue allocation for order routing
#[derive(Debug, Clone)]
pub struct VenueAllocation {
    pub venue_id: VenueId,
    pub quantity: u64,
    pub expected_price: u64,
    pub priority: u32,
}

/// Cross-venue arbitrage detector
pub struct ArbitrageDetector {
    venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>,
    min_profit_threshold: u64,
    max_position_size: u64,
    detection_window_ms: u64,
    recent_opportunities: Vec<ArbitrageOpportunity>,
}

impl ArbitrageDetector {
    pub fn new(
        venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>,
        min_profit_threshold: u64,
        max_position_size: u64,
    ) -> Self {
        Self {
            venue_adapters,
            min_profit_threshold,
            max_position_size,
            detection_window_ms: 5000, // 5 second window
            recent_opportunities: Vec::new(),
        }
    }
    
    /// Detect arbitrage opportunities across venues
    pub async fn detect_opportunities(&mut self, symbol: &Symbol) -> VenueResult<Vec<ArbitrageOpportunity>> {
        let mut venue_quotes = HashMap::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        // Collect quotes from all venues
        for (venue_id, adapter) in &self.venue_adapters {
            match adapter.get_best_bid_ask(symbol).await {
                Ok((bid, ask)) => {
                    venue_quotes.insert(venue_id.clone(), (bid, ask));
                }
                Err(e) => {
                    eprintln!("Failed to get quotes from venue {}: {}", venue_id.as_str(), e);
                    continue;
                }
            }
        }
        
        let mut opportunities = Vec::new();
        
        // Find arbitrage opportunities
        for (buy_venue, (_, buy_ask)) in &venue_quotes {
            if let Some(buy_price) = buy_ask {
                for (sell_venue, (sell_bid, _)) in &venue_quotes {
                    if let Some(sell_price) = sell_bid {
                        if buy_venue != sell_venue && sell_price > buy_price {
                            let profit = sell_price - buy_price;
                            
                            if profit >= self.min_profit_threshold {
                                // Get fees for both venues
                                let buy_fees = self.venue_adapters[buy_venue]
                                    .get_fees(symbol).await.unwrap_or_default();
                                let sell_fees = self.venue_adapters[sell_venue]
                                    .get_fees(symbol).await.unwrap_or_default();
                                
                                let total_fees = buy_fees.taker_fee + sell_fees.taker_fee;
                                let net_profit = profit as i64 - total_fees as i64;
                                
                                if net_profit > 0 {
                                    let opportunity = ArbitrageOpportunity {
                                        buy_venue: buy_venue.clone(),
                                        sell_venue: sell_venue.clone(),
                                        symbol: symbol.clone(),
                                        buy_price: *buy_price,
                                        sell_price: *sell_price,
                                        max_volume: self.max_position_size,
                                        profit_estimate: net_profit,
                                        confidence: self.calculate_confidence(&venue_quotes, buy_venue, sell_venue),
                                        detected_at: current_time,
                                        expires_at: current_time + self.detection_window_ms,
                                    };
                                    
                                    opportunities.push(opportunity);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by profit potential
        opportunities.sort_by(|a, b| b.profit_estimate.cmp(&a.profit_estimate));
        
        // Store recent opportunities
        self.recent_opportunities.extend(opportunities.clone());
        self.cleanup_expired_opportunities(current_time);
        
        Ok(opportunities)
    }
    
    /// Calculate confidence score for arbitrage opportunity
    fn calculate_confidence(&self, venue_quotes: &HashMap<VenueId, (Option<u64>, Option<u64>)>, 
                          buy_venue: &VenueId, sell_venue: &VenueId) -> f64 {
        // Simple confidence calculation based on spread consistency
        let mut confidence = 0.8; // Base confidence
        
        // Reduce confidence if spreads are very wide (indicating low liquidity)
        if let (Some((buy_bid, buy_ask)), Some((sell_bid, sell_ask))) = 
            (venue_quotes.get(buy_venue), venue_quotes.get(sell_venue)) {
            
            if let (Some(buy_bid), Some(buy_ask), Some(sell_bid), Some(sell_ask)) = 
                (buy_bid, buy_ask, sell_bid, sell_ask) {
                
                let buy_spread = buy_ask - buy_bid;
                let sell_spread = sell_ask - sell_bid;
                let avg_price = (buy_ask + sell_bid) / 2;
                
                let buy_spread_pct = (buy_spread as f64) / (avg_price as f64);
                let sell_spread_pct = (sell_spread as f64) / (avg_price as f64);
                
                // Reduce confidence for wide spreads
                if buy_spread_pct > 0.01 { confidence -= 0.2; }
                if sell_spread_pct > 0.01 { confidence -= 0.2; }
                
                // Reduce confidence if spreads are very different (liquidity imbalance)
                let spread_diff = (buy_spread_pct - sell_spread_pct).abs();
                if spread_diff > 0.005 { confidence -= 0.1; }
            }
        }
        
        confidence.max(0.0).min(1.0)
    }
    
    /// Clean up expired opportunities
    fn cleanup_expired_opportunities(&mut self, current_time: u64) {
        self.recent_opportunities.retain(|opp| opp.expires_at > current_time);
    }
    
    /// Get recent opportunities for analysis
    pub fn get_recent_opportunities(&self) -> &[ArbitrageOpportunity] {
        &self.recent_opportunities
    }
}

impl Default for VenueFees {
    fn default() -> Self {
        Self {
            maker_fee: 0,
            taker_fee: 0,
            currency: "USD".to_string(),
        }
    }
}

/// Smart order router for optimal execution across venues
pub struct SmartOrderRouter {
    venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>,
    venue_capabilities: HashMap<VenueId, VenueCapabilities>,
    routing_weights: HashMap<VenueId, f64>,
}

impl SmartOrderRouter {
    pub fn new(venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>) -> Self {
        Self {
            venue_adapters,
            venue_capabilities: HashMap::new(),
            routing_weights: HashMap::new(),
        }
    }
    
    /// Initialize router by fetching venue capabilities
    pub async fn initialize(&mut self) -> VenueResult<()> {
        for (venue_id, adapter) in &self.venue_adapters {
            match adapter.get_capabilities().await {
                Ok(capabilities) => {
                    self.venue_capabilities.insert(venue_id.clone(), capabilities);
                    self.routing_weights.insert(venue_id.clone(), 1.0); // Default weight
                }
                Err(e) => {
                    eprintln!("Failed to get capabilities for venue {}: {}", venue_id.as_str(), e);
                }
            }
        }
        Ok(())
    }
    
    /// Calculate optimal routing for an order
    pub async fn calculate_routing(&self, order: &Order) -> VenueResult<RoutingDecision> {
        let mut venue_scores = Vec::new();
        
        // Score each venue for this order
        for (venue_id, capabilities) in &self.venue_capabilities {
            if !capabilities.supported_symbols.contains(&order.symbol) {
                continue;
            }
            
            // Check order size constraints
            let min_size = capabilities.min_order_size.get(&order.symbol).unwrap_or(&0);
            let max_size = capabilities.max_order_size.get(&order.symbol).unwrap_or(&u64::MAX);
            
            if order.quantity < *min_size || order.quantity > *max_size {
                continue;
            }
            
            // Get current market data
            let adapter = &self.venue_adapters[venue_id];
            let (bid, ask) = adapter.get_best_bid_ask(&order.symbol).await?;
            let fees = adapter.get_fees(&order.symbol).await.unwrap_or_default();
            
            // Calculate score based on multiple factors
            let mut score = self.routing_weights.get(venue_id).unwrap_or(&1.0) * 100.0;
            
            // Price competitiveness
            match order.side {
                crate::orderbook::Side::Buy => {
                    if let Some(ask_price) = ask {
                        let price_score = if ask_price <= order.price {
                            50.0 // Can execute immediately
                        } else {
                            25.0 // Will need to wait
                        };
                        score += price_score;
                    }
                }
                crate::orderbook::Side::Sell => {
                    if let Some(bid_price) = bid {
                        let price_score = if bid_price >= order.price {
                            50.0 // Can execute immediately
                        } else {
                            25.0 // Will need to wait
                        };
                        score += price_score;
                    }
                }
            }
            
            // Fee competitiveness (lower fees = higher score)
            let fee_score = 100.0 - (fees.taker_fee as f64 / 1000.0); // Assuming fees in basis points
            score += fee_score * 0.3;
            
            // Latency score (lower latency = higher score)
            let latency_score = 100.0 - (capabilities.latency_estimate_us as f64 / 1000.0).min(100.0);
            score += latency_score * 0.2;
            
            venue_scores.push((venue_id.clone(), score, bid, ask, fees));
        }
        
        // Sort by score (highest first)
        venue_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        if venue_scores.is_empty() {
            return Err(VenueError::VenueSpecificError("No suitable venues found".to_string()));
        }
        
        // For now, route to the best venue (could implement splitting logic)
        let (best_venue, _score, bid, ask, fees) = &venue_scores[0];
        
        let expected_price = match order.side {
            crate::orderbook::Side::Buy => ask.unwrap_or(order.price),
            crate::orderbook::Side::Sell => bid.unwrap_or(order.price),
        };
        
        let allocation = VenueAllocation {
            venue_id: best_venue.clone(),
            quantity: order.quantity,
            expected_price,
            priority: 1,
        };
        
        let capabilities = &self.venue_capabilities[best_venue];
        
        Ok(RoutingDecision {
            venue_allocations: vec![allocation],
            expected_execution_price: expected_price,
            expected_fees: fees.taker_fee,
            estimated_latency_us: capabilities.latency_estimate_us,
            confidence: 0.8, // Could be more sophisticated
        })
    }
    
    /// Update venue weights based on performance
    pub fn update_venue_weight(&mut self, venue_id: &VenueId, weight: f64) {
        self.routing_weights.insert(venue_id.clone(), weight);
    }
    
    /// Get venue performance statistics
    pub fn get_venue_stats(&self, venue_id: &VenueId) -> Option<VenueStats> {
        self.venue_capabilities.get(venue_id).map(|caps| VenueStats {
            venue_id: venue_id.clone(),
            supported_symbols: caps.supported_symbols.len(),
            average_latency_us: caps.latency_estimate_us,
            current_weight: self.routing_weights.get(venue_id).copied().unwrap_or(1.0),
        })
    }
}

/// Venue performance statistics
#[derive(Debug, Clone)]
pub struct VenueStats {
    pub venue_id: VenueId,
    pub supported_symbols: usize,
    pub average_latency_us: u64,
    pub current_weight: f64,
}

/// Venue-specific risk controls
pub struct VenueRiskManager {
    venue_limits: HashMap<VenueId, VenueRiskLimits>,
    current_exposures: HashMap<VenueId, VenueExposure>,
}

/// Risk limits per venue
#[derive(Debug, Clone)]
pub struct VenueRiskLimits {
    pub max_position_per_symbol: HashMap<Symbol, u64>,
    pub max_daily_volume: u64,
    pub max_open_orders: u32,
    pub max_order_size: u64,
    pub allowed_symbols: Vec<Symbol>,
}

/// Current exposure per venue
#[derive(Debug, Clone, Default)]
pub struct VenueExposure {
    pub positions: HashMap<Symbol, i64>, // Signed position (positive = long)
    pub daily_volume: u64,
    pub open_orders: u32,
    pub last_reset: u64,
}

impl VenueRiskManager {
    pub fn new() -> Self {
        Self {
            venue_limits: HashMap::new(),
            current_exposures: HashMap::new(),
        }
    }
    
    /// Set risk limits for a venue
    pub fn set_venue_limits(&mut self, venue_id: VenueId, limits: VenueRiskLimits) {
        self.venue_limits.insert(venue_id, limits);
    }
    
    /// Check if order is allowed under risk limits
    pub fn check_order_risk(&self, venue_id: &VenueId, order: &Order) -> VenueResult<bool> {
        let limits = self.venue_limits.get(venue_id)
            .ok_or_else(|| VenueError::VenueSpecificError("No risk limits set for venue".to_string()))?;
        
        // Check symbol allowlist
        if !limits.allowed_symbols.contains(&order.symbol) {
            return Ok(false);
        }
        
        // Check order size
        if order.quantity > limits.max_order_size {
            return Ok(false);
        }
        
        // Check position limits
        if let Some(max_position) = limits.max_position_per_symbol.get(&order.symbol) {
            let current_exposure = self.current_exposures.get(venue_id)
                .and_then(|exp| exp.positions.get(&order.symbol))
                .unwrap_or(&0);
            
            let new_position = match order.side {
                crate::orderbook::Side::Buy => current_exposure + order.quantity as i64,
                crate::orderbook::Side::Sell => current_exposure - order.quantity as i64,
            };
            
            if new_position.abs() as u64 > *max_position {
                return Ok(false);
            }
        }
        
        // Check daily volume limits
        let current_exposure = self.current_exposures.get(venue_id).unwrap_or(&VenueExposure::default());
        if current_exposure.daily_volume + order.quantity > limits.max_daily_volume {
            return Ok(false);
        }
        
        // Check open orders limit
        if current_exposure.open_orders >= limits.max_open_orders {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Update exposure after order submission
    pub fn update_exposure(&mut self, venue_id: &VenueId, order: &Order, filled_quantity: u64) {
        let exposure = self.current_exposures.entry(venue_id.clone()).or_default();
        
        // Update position
        let position_change = match order.side {
            crate::orderbook::Side::Buy => filled_quantity as i64,
            crate::orderbook::Side::Sell => -(filled_quantity as i64),
        };
        
        *exposure.positions.entry(order.symbol.clone()).or_insert(0) += position_change;
        
        // Update daily volume
        exposure.daily_volume += filled_quantity;
        
        // Update open orders count (simplified - would need more sophisticated tracking)
        if filled_quantity < order.quantity {
            exposure.open_orders += 1;
        }
    }
    
    /// Reset daily limits (should be called daily)
    pub fn reset_daily_limits(&mut self) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        for exposure in self.current_exposures.values_mut() {
            exposure.daily_volume = 0;
            exposure.last_reset = current_time;
        }
    }
    
    /// Get current exposure for a venue
    pub fn get_venue_exposure(&self, venue_id: &VenueId) -> Option<&VenueExposure> {
        self.current_exposures.get(venue_id)
    }
}

impl Default for VenueRiskManager {
    fn default() -> Self {
        Self::new()
    }
}