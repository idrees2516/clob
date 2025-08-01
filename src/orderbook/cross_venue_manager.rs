//! Cross-venue integration manager
//!
//! This module provides the main coordination layer for cross-venue trading,
//! integrating arbitrage detection, smart order routing, and risk management.

use super::venue_adapter::{
    VenueAdapter, VenueId, ArbitrageDetector, SmartOrderRouter, VenueRiskManager,
    ArbitrageOpportunity, RoutingDecision, VenueRiskLimits, VenueError, VenueResult
};
use crate::orderbook::{Order, OrderId, Symbol, Trade, CLOBEngine, CLOBResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::sync::RwLock;
use tokio::time::interval;
use serde::{Serialize, Deserialize};

/// Cross-venue trading manager
pub struct CrossVenueManager {
    /// Venue adapters
    venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>,
    
    /// Arbitrage detector
    arbitrage_detector: Arc<Mutex<ArbitrageDetector>>,
    
    /// Smart order router
    order_router: Arc<RwLock<SmartOrderRouter>>,
    
    /// Risk manager
    risk_manager: Arc<Mutex<VenueRiskManager>>,
    
    /// Active arbitrage positions
    active_arbitrage: Arc<Mutex<HashMap<String, ArbitragePosition>>>,
    
    /// Performance tracking
    performance_tracker: Arc<Mutex<CrossVenuePerformance>>,
    
    /// Configuration
    config: CrossVenueConfig,
}

/// Configuration for cross-venue operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossVenueConfig {
    pub arbitrage_enabled: bool,
    pub smart_routing_enabled: bool,
    pub min_arbitrage_profit: u64,
    pub max_arbitrage_position: u64,
    pub arbitrage_timeout_ms: u64,
    pub routing_timeout_ms: u64,
    pub health_check_interval_ms: u64,
    pub performance_update_interval_ms: u64,
}

impl Default for CrossVenueConfig {
    fn default() -> Self {
        Self {
            arbitrage_enabled: true,
            smart_routing_enabled: true,
            min_arbitrage_profit: 1_000, // $0.001 minimum profit
            max_arbitrage_position: 1_000_000, // $1 maximum position
            arbitrage_timeout_ms: 5_000,
            routing_timeout_ms: 1_000,
            health_check_interval_ms: 30_000,
            performance_update_interval_ms: 60_000,
        }
    }
}

/// Active arbitrage position tracking
#[derive(Debug, Clone)]
pub struct ArbitragePosition {
    pub id: String,
    pub opportunity: ArbitrageOpportunity,
    pub buy_order_id: Option<String>,
    pub sell_order_id: Option<String>,
    pub status: ArbitrageStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub realized_pnl: i64,
}

/// Arbitrage position status
#[derive(Debug, Clone, PartialEq)]
pub enum ArbitrageStatus {
    Pending,
    BuyOrderSubmitted,
    SellOrderSubmitted,
    BothOrdersSubmitted,
    PartiallyFilled,
    Completed,
    Failed,
    Cancelled,
}

/// Cross-venue performance metrics
#[derive(Debug, Clone, Default)]
pub struct CrossVenuePerformance {
    pub total_arbitrage_opportunities: u64,
    pub successful_arbitrages: u64,
    pub failed_arbitrages: u64,
    pub total_arbitrage_profit: i64,
    pub total_routing_decisions: u64,
    pub successful_routes: u64,
    pub average_execution_latency_us: u64,
    pub venue_performance: HashMap<VenueId, VenuePerformance>,
}

/// Per-venue performance metrics
#[derive(Debug, Clone, Default)]
pub struct VenuePerformance {
    pub total_orders: u64,
    pub successful_orders: u64,
    pub failed_orders: u64,
    pub average_latency_us: u64,
    pub total_volume: u64,
    pub uptime_percentage: f64,
}

impl CrossVenueManager {
    /// Create a new cross-venue manager
    pub fn new(
        venue_adapters: HashMap<VenueId, Arc<dyn VenueAdapter>>,
        config: CrossVenueConfig,
    ) -> Self {
        let arbitrage_detector = Arc::new(Mutex::new(ArbitrageDetector::new(
            venue_adapters.clone(),
            config.min_arbitrage_profit,
            config.max_arbitrage_position,
        )));
        
        let order_router = Arc::new(RwLock::new(SmartOrderRouter::new(venue_adapters.clone())));
        let risk_manager = Arc::new(Mutex::new(VenueRiskManager::new()));
        
        Self {
            venue_adapters,
            arbitrage_detector,
            order_router,
            risk_manager,
            active_arbitrage: Arc::new(Mutex::new(HashMap::new())),
            performance_tracker: Arc::new(Mutex::new(CrossVenuePerformance::default())),
            config,
        }
    }
    
    /// Initialize the cross-venue manager
    pub async fn initialize(&self) -> VenueResult<()> {
        // Initialize order router
        let mut router = self.order_router.write().await;
        router.initialize().await?;
        drop(router);
        
        // Set up default risk limits for all venues
        let mut risk_manager = self.risk_manager.lock().unwrap();
        for venue_id in self.venue_adapters.keys() {
            let limits = VenueRiskLimits {
                max_position_per_symbol: HashMap::new(), // Would be configured per venue
                max_daily_volume: 10_000_000, // $10,000 daily limit
                max_open_orders: 100,
                max_order_size: 1_000_000, // $1,000 max order
                allowed_symbols: vec![
                    Symbol::new("BTC/USD"),
                    Symbol::new("ETH/USD"),
                    Symbol::new("SOL/USD"),
                ],
            };
            risk_manager.set_venue_limits(venue_id.clone(), limits);
        }
        drop(risk_manager);
        
        Ok(())
    }
    
    /// Start background tasks
    pub async fn start_background_tasks(&self) {
        if self.config.arbitrage_enabled {
            self.start_arbitrage_monitoring().await;
        }
        
        self.start_health_monitoring().await;
        self.start_performance_tracking().await;
    }
    
    /// Execute smart order routing for a single order
    pub async fn route_order(&self, order: Order) -> VenueResult<RoutingDecision> {
        if !self.config.smart_routing_enabled {
            return Err(VenueError::VenueSpecificError("Smart routing disabled".to_string()));
        }
        
        let start_time = SystemTime::now();
        
        // Check risk limits for all potential venues
        let risk_manager = self.risk_manager.lock().unwrap();
        let mut allowed_venues = Vec::new();
        
        for venue_id in self.venue_adapters.keys() {
            if risk_manager.check_order_risk(venue_id, &order)? {
                allowed_venues.push(venue_id.clone());
            }
        }
        drop(risk_manager);
        
        if allowed_venues.is_empty() {
            return Err(VenueError::VenueSpecificError("No venues allow this order due to risk limits".to_string()));
        }
        
        // Get routing decision
        let router = self.order_router.read().await;
        let mut routing_decision = router.calculate_routing(&order).await?;
        drop(router);
        
        // Filter routing decision to only include allowed venues
        routing_decision.venue_allocations.retain(|allocation| {
            allowed_venues.contains(&allocation.venue_id)
        });
        
        if routing_decision.venue_allocations.is_empty() {
            return Err(VenueError::VenueSpecificError("No suitable venues after risk filtering".to_string()));
        }
        
        // Update performance metrics
        let execution_time = start_time.elapsed().unwrap_or_default().as_micros() as u64;
        let mut performance = self.performance_tracker.lock().unwrap();
        performance.total_routing_decisions += 1;
        performance.successful_routes += 1;
        performance.average_execution_latency_us = 
            (performance.average_execution_latency_us + execution_time) / 2;
        drop(performance);
        
        Ok(routing_decision)
    }
    
    /// Execute arbitrage opportunity
    pub async fn execute_arbitrage(&self, opportunity: ArbitrageOpportunity) -> VenueResult<String> {
        if !self.config.arbitrage_enabled {
            return Err(VenueError::VenueSpecificError("Arbitrage disabled".to_string()));
        }
        
        let position_id = format!("arb_{}_{}", 
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis(),
            opportunity.symbol.as_str()
        );
        
        let mut position = ArbitragePosition {
            id: position_id.clone(),
            opportunity: opportunity.clone(),
            buy_order_id: None,
            sell_order_id: None,
            status: ArbitrageStatus::Pending,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            realized_pnl: 0,
        };
        
        // Store position
        {
            let mut active_positions = self.active_arbitrage.lock().unwrap();
            active_positions.insert(position_id.clone(), position.clone());
        }
        
        // Execute buy order
        let buy_order = Order {
            id: OrderId::new(&format!("buy_{}", position_id)),
            symbol: opportunity.symbol.clone(),
            side: crate::orderbook::Side::Buy,
            quantity: opportunity.max_volume,
            price: opportunity.buy_price,
            order_type: crate::orderbook::OrderType::Limit,
            time_in_force: crate::orderbook::TimeInForce::IOC, // Immediate or cancel
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
        };
        
        let buy_adapter = &self.venue_adapters[&opportunity.buy_venue];
        match buy_adapter.submit_order(buy_order).await {
            Ok(venue_order) => {
                position.buy_order_id = Some(venue_order.venue_order_id);
                position.status = ArbitrageStatus::BuyOrderSubmitted;
            }
            Err(e) => {
                position.status = ArbitrageStatus::Failed;
                self.update_arbitrage_position(position).await;
                return Err(e);
            }
        }
        
        // Execute sell order
        let sell_order = Order {
            id: OrderId::new(&format!("sell_{}", position_id)),
            symbol: opportunity.symbol.clone(),
            side: crate::orderbook::Side::Sell,
            quantity: opportunity.max_volume,
            price: opportunity.sell_price,
            order_type: crate::orderbook::OrderType::Limit,
            time_in_force: crate::orderbook::TimeInForce::IOC,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
        };
        
        let sell_adapter = &self.venue_adapters[&opportunity.sell_venue];
        match sell_adapter.submit_order(sell_order).await {
            Ok(venue_order) => {
                position.sell_order_id = Some(venue_order.venue_order_id);
                position.status = ArbitrageStatus::BothOrdersSubmitted;
            }
            Err(e) => {
                // Cancel buy order if sell order fails
                if let Some(buy_order_id) = &position.buy_order_id {
                    let _ = buy_adapter.cancel_order(buy_order_id).await;
                }
                position.status = ArbitrageStatus::Failed;
                self.update_arbitrage_position(position).await;
                return Err(e);
            }
        }
        
        self.update_arbitrage_position(position).await;
        
        // Update performance metrics
        let mut performance = self.performance_tracker.lock().unwrap();
        performance.total_arbitrage_opportunities += 1;
        drop(performance);
        
        Ok(position_id)
    }
    
    /// Monitor arbitrage opportunities
    async fn start_arbitrage_monitoring(&self) {
        let arbitrage_detector = Arc::clone(&self.arbitrage_detector);
        let manager = self.clone_for_background();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000)); // Check every second
            
            loop {
                interval.tick().await;
                
                let symbols = vec![
                    Symbol::new("BTC/USD"),
                    Symbol::new("ETH/USD"),
                    Symbol::new("SOL/USD"),
                ];
                
                for symbol in symbols {
                    let mut detector = arbitrage_detector.lock().unwrap();
                    match detector.detect_opportunities(&symbol).await {
                        Ok(opportunities) => {
                            drop(detector);
                            
                            for opportunity in opportunities {
                                if opportunity.confidence > 0.7 { // Only execute high-confidence opportunities
                                    if let Err(e) = manager.execute_arbitrage(opportunity).await {
                                        eprintln!("Failed to execute arbitrage: {}", e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("Failed to detect arbitrage opportunities for {}: {}", symbol.as_str(), e);
                            drop(detector);
                        }
                    }
                }
            }
        });
    }
    
    /// Monitor venue health
    async fn start_health_monitoring(&self) {
        let venue_adapters = self.venue_adapters.clone();
        let performance_tracker = Arc::clone(&self.performance_tracker);
        let interval_ms = self.config.health_check_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                for (venue_id, adapter) in &venue_adapters {
                    match adapter.health_check().await {
                        Ok(is_healthy) => {
                            let mut performance = performance_tracker.lock().unwrap();
                            let venue_perf = performance.venue_performance
                                .entry(venue_id.clone())
                                .or_default();
                            
                            if is_healthy {
                                venue_perf.uptime_percentage = 
                                    (venue_perf.uptime_percentage * 0.95) + (1.0 * 0.05); // Exponential moving average
                            } else {
                                venue_perf.uptime_percentage = venue_perf.uptime_percentage * 0.95;
                            }
                        }
                        Err(e) => {
                            eprintln!("Health check failed for venue {}: {}", venue_id.as_str(), e);
                            
                            let mut performance = performance_tracker.lock().unwrap();
                            let venue_perf = performance.venue_performance
                                .entry(venue_id.clone())
                                .or_default();
                            venue_perf.uptime_percentage = venue_perf.uptime_percentage * 0.95;
                        }
                    }
                }
            }
        });
    }
    
    /// Track performance metrics
    async fn start_performance_tracking(&self) {
        let performance_tracker = Arc::clone(&self.performance_tracker);
        let interval_ms = self.config.performance_update_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                // Update performance metrics (could include more sophisticated calculations)
                let performance = performance_tracker.lock().unwrap();
                println!("Cross-venue performance update: {:?}", *performance);
                drop(performance);
            }
        });
    }
    
    /// Update arbitrage position
    async fn update_arbitrage_position(&self, position: ArbitragePosition) {
        let mut active_positions = self.active_arbitrage.lock().unwrap();
        active_positions.insert(position.id.clone(), position);
    }
    
    /// Get current arbitrage positions
    pub fn get_active_arbitrage_positions(&self) -> HashMap<String, ArbitragePosition> {
        self.active_arbitrage.lock().unwrap().clone()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> CrossVenuePerformance {
        self.performance_tracker.lock().unwrap().clone()
    }
    
    /// Get venue adapters
    pub fn get_venue_adapters(&self) -> &HashMap<VenueId, Arc<dyn VenueAdapter>> {
        &self.venue_adapters
    }
    
    /// Clone for background tasks (simplified clone)
    fn clone_for_background(&self) -> Self {
        Self {
            venue_adapters: self.venue_adapters.clone(),
            arbitrage_detector: Arc::clone(&self.arbitrage_detector),
            order_router: Arc::clone(&self.order_router),
            risk_manager: Arc::clone(&self.risk_manager),
            active_arbitrage: Arc::clone(&self.active_arbitrage),
            performance_tracker: Arc::clone(&self.performance_tracker),
            config: self.config.clone(),
        }
    }
    
    /// Shutdown and cleanup
    pub async fn shutdown(&self) -> VenueResult<()> {
        // Cancel all active arbitrage positions
        let active_positions = self.active_arbitrage.lock().unwrap().clone();
        
        for (position_id, position) in active_positions {
            if position.status == ArbitrageStatus::BothOrdersSubmitted || 
               position.status == ArbitrageStatus::PartiallyFilled {
                
                // Cancel orders
                if let Some(buy_order_id) = &position.buy_order_id {
                    let buy_adapter = &self.venue_adapters[&position.opportunity.buy_venue];
                    let _ = buy_adapter.cancel_order(buy_order_id).await;
                }
                
                if let Some(sell_order_id) = &position.sell_order_id {
                    let sell_adapter = &self.venue_adapters[&position.opportunity.sell_venue];
                    let _ = sell_adapter.cancel_order(sell_order_id).await;
                }
            }
        }
        
        // Clear active positions
        self.active_arbitrage.lock().unwrap().clear();
        
        Ok(())
    }
}

/// Cross-venue integration tests and utilities
#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::venue_implementations::VenueAdapterFactory;
    
    #[tokio::test]
    async fn test_cross_venue_manager_initialization() {
        let venues = VenueAdapterFactory::create_test_venues();
        let config = CrossVenueConfig::default();
        
        let manager = CrossVenueManager::new(venues, config);
        assert!(manager.initialize().await.is_ok());
    }
    
    #[tokio::test]
    async fn test_arbitrage_detection() {
        let venues = VenueAdapterFactory::create_test_venues();
        let config = CrossVenueConfig::default();
        
        let manager = CrossVenueManager::new(venues, config);
        manager.initialize().await.unwrap();
        
        // The test venues are set up with price differences that should create arbitrage opportunities
        let mut detector = manager.arbitrage_detector.lock().unwrap();
        let opportunities = detector.detect_opportunities(&Symbol::new("BTC/USD")).await.unwrap();
        
        assert!(!opportunities.is_empty(), "Should detect arbitrage opportunities");
    }
    
    #[tokio::test]
    async fn test_smart_order_routing() {
        let venues = VenueAdapterFactory::create_test_venues();
        let config = CrossVenueConfig::default();
        
        let manager = CrossVenueManager::new(venues, config);
        manager.initialize().await.unwrap();
        
        let order = Order {
            id: OrderId::new("test_order"),
            symbol: Symbol::new("BTC/USD"),
            side: crate::orderbook::Side::Buy,
            quantity: 10_000, // 0.01 BTC
            price: 50_000_000_000, // $50,000
            order_type: crate::orderbook::OrderType::Limit,
            time_in_force: crate::orderbook::TimeInForce::GTC,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
        };
        
        let routing_decision = manager.route_order(order).await.unwrap();
        assert!(!routing_decision.venue_allocations.is_empty(), "Should provide routing decision");
    }
}