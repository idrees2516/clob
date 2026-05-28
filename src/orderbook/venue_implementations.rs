//! Concrete implementations of venue adapters for popular exchanges
//!
//! This module provides ready-to-use implementations for common trading venues,
//! including mock implementations for testing and development.

use super::venue_adapter::{
    VenueAdapter, VenueId, VenueOrder, VenueTrade, VenueCapabilities, VenueOrderBook,
    VenueFees, RateLimits, VenueOrderStatus, VenueError, VenueResult
};
use crate::orderbook::{Order, OrderId, Symbol, Side, Trade};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Mock venue adapter for testing and development
pub struct MockVenueAdapter {
    venue_id: VenueId,
    capabilities: VenueCapabilities,
    orders: Arc<Mutex<HashMap<String, VenueOrder>>>,
    trades: Arc<Mutex<Vec<VenueTrade>>>,
    market_data: Arc<Mutex<HashMap<Symbol, (Option<u64>, Option<u64>)>>>, // (bid, ask)
    order_counter: Arc<Mutex<u64>>,
    latency_simulation_us: u64,
}

impl MockVenueAdapter {
    pub fn new(venue_id: VenueId, supported_symbols: Vec<Symbol>) -> Self {
        let mut min_order_size = HashMap::new();
        let mut max_order_size = HashMap::new();
        let mut tick_size = HashMap::new();
        
        // Set default constraints for all symbols
        for symbol in &supported_symbols {
            min_order_size.insert(symbol.clone(), 1_000); // 0.001 in fixed-point
            max_order_size.insert(symbol.clone(), 1_000_000_000); // 1000 in fixed-point
            tick_size.insert(symbol.clone(), 100); // 0.0001 in fixed-point
        }
        
        let capabilities = VenueCapabilities {
            venue_id: venue_id.clone(),
            supported_symbols: supported_symbols.clone(),
            min_order_size,
            max_order_size,
            tick_size,
            rate_limits: RateLimits {
                orders_per_second: 10,
                requests_per_minute: 1200,
                weight_per_order: 1,
                max_weight_per_minute: 1200,
            },
            order_types: vec!["LIMIT".to_string(), "MARKET".to_string()],
            time_in_force_options: vec!["GTC".to_string(), "IOC".to_string(), "FOK".to_string()],
            supports_post_only: true,
            supports_ioc: true,
            supports_fok: true,
            latency_estimate_us: 500, // 0.5ms
        };
        
        // Initialize market data with some default prices
        let mut market_data = HashMap::new();
        for symbol in &supported_symbols {
            // Set some reasonable default prices based on symbol
            let base_price = match symbol.as_str() {
                "BTC/USD" => 50_000_000_000, // $50,000 in fixed-point
                "ETH/USD" => 3_000_000_000,  // $3,000 in fixed-point
                "SOL/USD" => 100_000_000,    // $100 in fixed-point
                _ => 1_000_000_000,          // $1,000 default
            };
            
            let spread = base_price / 1000; // 0.1% spread
            market_data.insert(symbol.clone(), (Some(base_price - spread/2), Some(base_price + spread/2)));
        }
        
        Self {
            venue_id,
            capabilities,
            orders: Arc::new(Mutex::new(HashMap::new())),
            trades: Arc::new(Mutex::new(Vec::new())),
            market_data: Arc::new(Mutex::new(market_data)),
            order_counter: Arc::new(Mutex::new(1)),
            latency_simulation_us: 500,
        }
    }
    
    /// Set market data for testing
    pub fn set_market_data(&self, symbol: Symbol, bid: Option<u64>, ask: Option<u64>) {
        let mut market_data = self.market_data.lock().unwrap();
        market_data.insert(symbol, (bid, ask));
    }
    
    /// Simulate order fill for testing
    pub fn simulate_fill(&self, venue_order_id: &str, filled_quantity: u64) -> VenueResult<()> {
        let mut orders = self.orders.lock().unwrap();
        if let Some(venue_order) = orders.get_mut(venue_order_id) {
            venue_order.status = if filled_quantity >= venue_order.order.quantity {
                VenueOrderStatus::Filled
            } else {
                VenueOrderStatus::PartiallyFilled
            };
            
            // Create a trade
            let trade = Trade {
                id: format!("trade_{}", venue_order_id),
                buy_order_id: if venue_order.order.side == Side::Buy { venue_order.internal_order_id } else { OrderId::new("counterparty") },
                sell_order_id: if venue_order.order.side == Side::Sell { venue_order.internal_order_id } else { OrderId::new("counterparty") },
                symbol: venue_order.order.symbol.clone(),
                price: venue_order.order.price,
                quantity: filled_quantity,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
                side: venue_order.order.side,
            };
            
            let venue_trade = VenueTrade {
                venue_id: self.venue_id.clone(),
                venue_trade_id: format!("venue_trade_{}", venue_order_id),
                trade,
                fees: VenueFees {
                    maker_fee: 25, // 0.025% in basis points
                    taker_fee: 50, // 0.05% in basis points
                    currency: "USD".to_string(),
                },
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            };
            
            let mut trades = self.trades.lock().unwrap();
            trades.push(venue_trade);
        }
        
        Ok(())
    }
}

#[async_trait]
impl VenueAdapter for MockVenueAdapter {
    async fn get_capabilities(&self) -> VenueResult<VenueCapabilities> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        Ok(self.capabilities.clone())
    }
    
    async fn submit_order(&self, order: Order) -> VenueResult<VenueOrder> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        // Generate venue order ID
        let mut counter = self.order_counter.lock().unwrap();
        let venue_order_id = format!("{}_{}", self.venue_id.as_str(), *counter);
        *counter += 1;
        
        let venue_order = VenueOrder {
            venue_id: self.venue_id.clone(),
            venue_order_id: venue_order_id.clone(),
            internal_order_id: order.id,
            order,
            status: VenueOrderStatus::Submitted,
            created_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
            updated_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
        };
        
        // Store order
        let mut orders = self.orders.lock().unwrap();
        orders.insert(venue_order_id, venue_order.clone());
        
        Ok(venue_order)
    }
    
    async fn cancel_order(&self, venue_order_id: &str) -> VenueResult<VenueOrder> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        let mut orders = self.orders.lock().unwrap();
        if let Some(venue_order) = orders.get_mut(venue_order_id) {
            venue_order.status = VenueOrderStatus::Cancelled;
            venue_order.updated_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
            Ok(venue_order.clone())
        } else {
            Err(VenueError::VenueSpecificError(format!("Order {} not found", venue_order_id)))
        }
    }
    
    async fn get_order_status(&self, venue_order_id: &str) -> VenueResult<VenueOrder> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        let orders = self.orders.lock().unwrap();
        orders.get(venue_order_id)
            .cloned()
            .ok_or_else(|| VenueError::VenueSpecificError(format!("Order {} not found", venue_order_id)))
    }
    
    async fn get_recent_trades(&self, symbol: &Symbol, limit: usize) -> VenueResult<Vec<VenueTrade>> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        let trades = self.trades.lock().unwrap();
        let recent_trades: Vec<VenueTrade> = trades.iter()
            .filter(|trade| &trade.trade.symbol == symbol)
            .rev()
            .take(limit)
            .cloned()
            .collect();
        
        Ok(recent_trades)
    }
    
    async fn get_best_bid_ask(&self, symbol: &Symbol) -> VenueResult<(Option<u64>, Option<u64>)> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        let market_data = self.market_data.lock().unwrap();
        Ok(market_data.get(symbol).copied().unwrap_or((None, None)))
    }
    
    async fn get_order_book(&self, symbol: &Symbol, depth: usize) -> VenueResult<VenueOrderBook> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        let market_data = self.market_data.lock().unwrap();
        let (bid, ask) = market_data.get(symbol).copied().unwrap_or((None, None));
        
        // Generate mock order book with some depth
        let mut bids = Vec::new();
        let mut asks = Vec::new();
        
        if let Some(best_bid) = bid {
            for i in 0..depth {
                let price = best_bid - (i as u64 * 1000); // Decrease by 0.001 each level
                let volume = 10_000 + (i as u64 * 1000); // Increase volume at worse prices
                bids.push((price, volume));
            }
        }
        
        if let Some(best_ask) = ask {
            for i in 0..depth {
                let price = best_ask + (i as u64 * 1000); // Increase by 0.001 each level
                let volume = 10_000 + (i as u64 * 1000); // Increase volume at worse prices
                asks.push((price, volume));
            }
        }
        
        Ok(VenueOrderBook {
            venue_id: self.venue_id.clone(),
            symbol: symbol.clone(),
            bids,
            asks,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64,
        })
    }
    
    async fn health_check(&self) -> VenueResult<bool> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        Ok(true) // Mock venue is always healthy
    }
    
    async fn get_fees(&self, _symbol: &Symbol) -> VenueResult<VenueFees> {
        // Simulate network latency
        tokio::time::sleep(tokio::time::Duration::from_micros(self.latency_simulation_us)).await;
        
        Ok(VenueFees {
            maker_fee: 25, // 0.025% in basis points
            taker_fee: 50, // 0.05% in basis points
            currency: "USD".to_string(),
        })
    }
}

/// Generic REST API venue adapter
pub struct RestApiVenueAdapter {
    venue_id: VenueId,
    base_url: String,
    api_key: String,
    secret_key: String,
    capabilities: VenueCapabilities,
    client: reqwest::Client,
}

impl RestApiVenueAdapter {
    pub fn new(
        venue_id: VenueId,
        base_url: String,
        api_key: String,
        secret_key: String,
        capabilities: VenueCapabilities,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            venue_id,
            base_url,
            api_key,
            secret_key,
            capabilities,
            client,
        }
    }
    
    /// Sign request for authenticated endpoints
    fn sign_request(&self, method: &str, path: &str, body: &str, timestamp: u64) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        
        let message = format!("{}{}{}{}", timestamp, method, path, body);
        let mut mac = HmacSha256::new_from_slice(self.secret_key.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        hex::encode(mac.finalize().into_bytes())
    }
}

#[async_trait]
impl VenueAdapter for RestApiVenueAdapter {
    async fn get_capabilities(&self) -> VenueResult<VenueCapabilities> {
        Ok(self.capabilities.clone())
    }
    
    async fn submit_order(&self, order: Order) -> VenueResult<VenueOrder> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        #[derive(Serialize)]
        struct OrderRequest {
            symbol: String,
            side: String,
            #[serde(rename = "type")]
            order_type: String,
            quantity: String,
            price: String,
            #[serde(rename = "timeInForce")]
            time_in_force: String,
        }
        
        let order_request = OrderRequest {
            symbol: order.symbol.as_str().to_string(),
            side: match order.side {
                Side::Buy => "BUY".to_string(),
                Side::Sell => "SELL".to_string(),
            },
            order_type: "LIMIT".to_string(),
            quantity: format!("{:.8}", order.quantity as f64 / 1_000_000.0), // Convert from fixed-point
            price: format!("{:.8}", order.price as f64 / 1_000_000.0), // Convert from fixed-point
            time_in_force: "GTC".to_string(),
        };
        
        let body = serde_json::to_string(&order_request)
            .map_err(|e| VenueError::SerializationError(e.to_string()))?;
        
        let path = "/api/v3/order";
        let signature = self.sign_request("POST", path, &body, timestamp);
        
        let response = self.client
            .post(&format!("{}{}", self.base_url, path))
            .header("X-API-KEY", &self.api_key)
            .header("X-TIMESTAMP", timestamp.to_string())
            .header("X-SIGNATURE", signature)
            .header("Content-Type", "application/json")
            .body(body)
            .send()
            .await
            .map_err(|e| VenueError::ConnectionError(e.to_string()))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VenueError::VenueSpecificError(format!("HTTP {}: {}", response.status(), error_text)));
        }
        
        #[derive(Deserialize)]
        struct OrderResponse {
            #[serde(rename = "orderId")]
            order_id: String,
            status: String,
        }
        
        let order_response: OrderResponse = response.json().await
            .map_err(|e| VenueError::InvalidResponse)?;
        
        let status = match order_response.status.as_str() {
            "NEW" => VenueOrderStatus::Submitted,
            "PARTIALLY_FILLED" => VenueOrderStatus::PartiallyFilled,
            "FILLED" => VenueOrderStatus::Filled,
            "CANCELED" => VenueOrderStatus::Cancelled,
            "REJECTED" => VenueOrderStatus::Rejected,
            "EXPIRED" => VenueOrderStatus::Expired,
            _ => VenueOrderStatus::Pending,
        };
        
        Ok(VenueOrder {
            venue_id: self.venue_id.clone(),
            venue_order_id: order_response.order_id,
            internal_order_id: order.id,
            order,
            status,
            created_at: timestamp * 1_000_000, // Convert to nanoseconds
            updated_at: timestamp * 1_000_000,
        })
    }
    
    async fn cancel_order(&self, venue_order_id: &str) -> VenueResult<VenueOrder> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let path = &format!("/api/v3/order/{}", venue_order_id);
        let signature = self.sign_request("DELETE", path, "", timestamp);
        
        let response = self.client
            .delete(&format!("{}{}", self.base_url, path))
            .header("X-API-KEY", &self.api_key)
            .header("X-TIMESTAMP", timestamp.to_string())
            .header("X-SIGNATURE", signature)
            .send()
            .await
            .map_err(|e| VenueError::ConnectionError(e.to_string()))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VenueError::VenueSpecificError(format!("HTTP {}: {}", response.status(), error_text)));
        }
        
        // For simplicity, return a cancelled order (would need to parse actual response)
        Ok(VenueOrder {
            venue_id: self.venue_id.clone(),
            venue_order_id: venue_order_id.to_string(),
            internal_order_id: OrderId::new("unknown"), // Would need to track this
            order: Order::default(), // Would need actual order data
            status: VenueOrderStatus::Cancelled,
            created_at: 0,
            updated_at: timestamp * 1_000_000,
        })
    }
    
    async fn get_order_status(&self, venue_order_id: &str) -> VenueResult<VenueOrder> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        let path = &format!("/api/v3/order/{}", venue_order_id);
        let signature = self.sign_request("GET", path, "", timestamp);
        
        let response = self.client
            .get(&format!("{}{}", self.base_url, path))
            .header("X-API-KEY", &self.api_key)
            .header("X-TIMESTAMP", timestamp.to_string())
            .header("X-SIGNATURE", signature)
            .send()
            .await
            .map_err(|e| VenueError::ConnectionError(e.to_string()))?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VenueError::VenueSpecificError(format!("HTTP {}: {}", response.status(), error_text)));
        }
        
        // Would implement actual response parsing here
        Err(VenueError::VenueSpecificError("Not implemented".to_string()))
    }
    
    async fn get_recent_trades(&self, symbol: &Symbol, limit: usize) -> VenueResult<Vec<VenueTrade>> {
        // Implementation would fetch recent trades from venue API
        Err(VenueError::VenueSpecificError("Not implemented".to_string()))
    }
    
    async fn get_best_bid_ask(&self, symbol: &Symbol) -> VenueResult<(Option<u64>, Option<u64>)> {
        let path = &format!("/api/v3/ticker/bookTicker?symbol={}", symbol.as_str());
        
        let response = self.client
            .get(&format!("{}{}", self.base_url, path))
            .send()
            .await
            .map_err(|e| VenueError::ConnectionError(e.to_string()))?;
        
        if !response.status().is_success() {
            return Err(VenueError::VenueSpecificError(format!("HTTP {}", response.status())));
        }
        
        #[derive(Deserialize)]
        struct TickerResponse {
            #[serde(rename = "bidPrice")]
            bid_price: String,
            #[serde(rename = "askPrice")]
            ask_price: String,
        }
        
        let ticker: TickerResponse = response.json().await
            .map_err(|_| VenueError::InvalidResponse)?;
        
        let bid = ticker.bid_price.parse::<f64>().ok()
            .map(|p| (p * 1_000_000.0) as u64); // Convert to fixed-point
        let ask = ticker.ask_price.parse::<f64>().ok()
            .map(|p| (p * 1_000_000.0) as u64); // Convert to fixed-point
        
        Ok((bid, ask))
    }
    
    async fn get_order_book(&self, symbol: &Symbol, depth: usize) -> VenueResult<VenueOrderBook> {
        // Implementation would fetch order book from venue API
        Err(VenueError::VenueSpecificError("Not implemented".to_string()))
    }
    
    async fn health_check(&self) -> VenueResult<bool> {
        let response = self.client
            .get(&format!("{}/api/v3/ping", self.base_url))
            .send()
            .await
            .map_err(|_| VenueError::ConnectionError("Health check failed".to_string()))?;
        
        Ok(response.status().is_success())
    }
    
    async fn get_fees(&self, _symbol: &Symbol) -> VenueResult<VenueFees> {
        // Would fetch actual fee schedule from venue
        Ok(VenueFees {
            maker_fee: 10, // 0.01%
            taker_fee: 20, // 0.02%
            currency: "USD".to_string(),
        })
    }
}

/// Venue adapter factory for creating different venue implementations
pub struct VenueAdapterFactory;

impl VenueAdapterFactory {
    /// Create a mock venue adapter for testing
    pub fn create_mock_venue(venue_id: VenueId, supported_symbols: Vec<Symbol>) -> Arc<dyn VenueAdapter> {
        Arc::new(MockVenueAdapter::new(venue_id, supported_symbols))
    }
    
    /// Create a generic REST API venue adapter
    pub fn create_rest_api_venue(
        venue_id: VenueId,
        base_url: String,
        api_key: String,
        secret_key: String,
        capabilities: VenueCapabilities,
    ) -> Arc<dyn VenueAdapter> {
        Arc::new(RestApiVenueAdapter::new(venue_id, base_url, api_key, secret_key, capabilities))
    }
    
    /// Create venue adapters for common exchanges
    pub fn create_binance_adapter(api_key: String, secret_key: String) -> Arc<dyn VenueAdapter> {
        let venue_id = VenueId::new("binance");
        let capabilities = VenueCapabilities {
            venue_id: venue_id.clone(),
            supported_symbols: vec![
                Symbol::new("BTCUSDT"),
                Symbol::new("ETHUSDT"),
                Symbol::new("SOLUSDT"),
            ],
            min_order_size: HashMap::new(), // Would be populated with actual values
            max_order_size: HashMap::new(),
            tick_size: HashMap::new(),
            rate_limits: RateLimits {
                orders_per_second: 10,
                requests_per_minute: 1200,
                weight_per_order: 1,
                max_weight_per_minute: 1200,
            },
            order_types: vec!["LIMIT".to_string(), "MARKET".to_string()],
            time_in_force_options: vec!["GTC".to_string(), "IOC".to_string(), "FOK".to_string()],
            supports_post_only: true,
            supports_ioc: true,
            supports_fok: true,
            latency_estimate_us: 50_000, // 50ms
        };
        
        Arc::new(RestApiVenueAdapter::new(
            venue_id,
            "https://api.binance.com".to_string(),
            api_key,
            secret_key,
            capabilities,
        ))
    }
    
    /// Create multiple mock venues for testing cross-venue functionality
    pub fn create_test_venues() -> HashMap<VenueId, Arc<dyn VenueAdapter>> {
        let mut venues = HashMap::new();
        
        let symbols = vec![
            Symbol::new("BTC/USD"),
            Symbol::new("ETH/USD"),
            Symbol::new("SOL/USD"),
        ];
        
        // Create mock venues with different characteristics
        let venue_a = VenueId::new("venue_a");
        let adapter_a = MockVenueAdapter::new(venue_a.clone(), symbols.clone());
        // Set slightly better prices on venue A
        adapter_a.set_market_data(Symbol::new("BTC/USD"), Some(49_995_000_000), Some(50_005_000_000));
        venues.insert(venue_a, Arc::new(adapter_a));
        
        let venue_b = VenueId::new("venue_b");
        let adapter_b = MockVenueAdapter::new(venue_b.clone(), symbols.clone());
        // Set slightly worse prices on venue B (arbitrage opportunity)
        adapter_b.set_market_data(Symbol::new("BTC/USD"), Some(49_990_000_000), Some(50_010_000_000));
        venues.insert(venue_b, Arc::new(adapter_b));
        
        let venue_c = VenueId::new("venue_c");
        let adapter_c = MockVenueAdapter::new(venue_c.clone(), symbols);
        // Set different prices on venue C
        adapter_c.set_market_data(Symbol::new("BTC/USD"), Some(50_000_000_000), Some(50_020_000_000));
        venues.insert(venue_c, Arc::new(adapter_c));
        
        venues
    }
}