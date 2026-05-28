use std::sync::Arc;
use rust_decimal_macros::dec;
use std::time::Duration;

use bid_ask_spread_estimation::models::lob_rough_vol::{
    LimitOrderBook, Order, OrderType, Side, TimeInForce, Price, 
    MatchingEngine, MarketDataEvent, OrderEvent, RoughVolatilityModel, RoughVolatilityParams,
    RateLimiter, RateLimitConfig, CircuitBreaker, OrderBookLevels, OrderBatch
};

#[test]
fn test_order_matching() {
    // Create a new order book
    let mut lob = LimitOrderBook::new(dec!(0.01), dec!(0.1)).unwrap();
    
    // Add a limit sell order
    let sell_order = Order::limit(
        Side::Ask, 
        dec!(100.0), 
        100, 
        None, 
        None, 
        None
    ).unwrap();
    
    // Add a limit buy order that should match
    let buy_order = Order::limit(
        Side::Bid, 
        dec!(100.0), 
        100, 
        None, 
        None, 
        None
    ).unwrap();
    
    // Process the sell order first
    let result = lob.process_order(sell_order);
    assert!(result.is_ok());
    assert_eq!(lob.best_bid(), None);
    assert!(lob.best_ask().is_some());
    
    // Process the buy order which should match
    let result = lob.process_order(buy_order);
    assert!(result.is_ok());
    
    // After matching, both sides should be empty
    assert_eq!(lob.best_bid(), None);
    assert_eq!(lob.best_ask(), None);
}

#[test]
fn test_volatility_simulation() {
    // Create volatility model parameters
    let params = RoughVolatilityParams {
        hurst: 0.1,
        nu: 0.3,
        kappa: 1.0,
        theta: 0.1,
        v0: 0.1,
        rho: -0.7,
        time_steps_per_day: 10,
        days: 1,
        seed: Some(42),
    };
    
    // Create and initialize the volatility model
    let mut model = RoughVolatilityModel::new(params).unwrap();
    
    // Simulate a price path
    let price_path = model.simulate_price_path(100.0).unwrap();
    
    // Check that we got the expected number of prices
    assert_eq!(price_path.len(), 10);
    
    // Check that all prices are positive
    assert!(price_path.iter().all(|&p| p > 0.0));
}

#[test]
fn test_matching_engine() {
    // Create a matching engine
    let mut engine = MatchingEngine::new();
    
    // Create a market data event
    let md_event = MarketDataEvent::PriceUpdate {
        timestamp: 1234567890,
        price: dec!(100.0),
    };
    
    // Add the market data event
    engine.add_market_data_event(md_event);
    
    // Create an order event
    let order = Order::market(
        Side::Bid,
        100,
        None,
        None,
        None
    ).unwrap();
    
    let order_event = OrderEvent::NewOrder(order);
    engine.add_order_event(order_event);
    
    // Process the events
    let result = engine.process_events();
    assert!(result.is_ok());
}

#[test]
fn test_rate_limiter() {
    // Create a rate limiter with strict limits
    let config = RateLimitConfig {
        orders_per_second: 2,
        orders_per_price_level: 10,
        max_order_size: 1000,
        max_price_deviation_pct: 10.0,
    };
    
    let limiter = RateLimiter::new(config);
    
    // Create a test order
    let order = Order::limit(
        Side::Bid,
        dec!(100.0),
        100,
        None,
        None,
        None
    ).unwrap();
    
    // First two orders should be allowed
    assert!(limiter.check_order(&order, Some(dec!(100.0))).is_ok());
    limiter.update_order(1);
    
    assert!(limiter.check_order(&order, Some(dec!(100.0))).is_ok());
    limiter.update_order(2);
    
    // Third order should be rate limited
    assert!(limiter.check_order(&order, Some(dec!(100.0))).is_err());
}

#[test]
fn test_circuit_breaker() {
    // Create a circuit breaker with a short cooldown
    let breaker = CircuitBreaker::new(Duration::from_millis(100));
    
    // Initially not triggered
    assert!(!breaker.is_triggered());
    
    // Trigger the circuit breaker
    breaker.trigger();
    assert!(breaker.is_triggered());
    
    // Should still be triggered before cooldown
    std::thread::sleep(Duration::from_millis(50));
    assert!(breaker.is_triggered());
    
    // Should reset after cooldown
    std::thread::sleep(Duration::from_millis(60));
    assert!(!breaker.is_triggered());
}

#[test]
fn test_order_batch() {
    // Create an order batch
    let mut batch = OrderBatch::new(10);
    
    // Add some orders to the batch
    for i in 0..5 {
        let order = Order::limit(
            Side::Bid,
            dec!(100.0 + i as f64),
            100 + i * 10,
            None,
            None,
            None
        ).unwrap();
        
        batch.add_order(order).unwrap();
    }
    
    // Process the batch
    let mut processed = 0;
    let result = batch.process(|order| {
        processed += 1;
        assert!(order.price >= dec!(100.0));
        assert!(order.quantity >= 100);
        Ok(None)
    });
    
    assert!(result.is_ok());
    assert_eq!(processed, 5);
}

#[test]
fn test_order_book_levels() {
    // Create order book levels
    let mut levels = OrderBookLevels::with_capacity(10);
    
    // Update with some levels
    let bids = vec![(Price::new(99.0, dec!(0.01)).unwrap(), 100)];
    let asks = vec![(Price::new(101.0, dec!(0.01)).unwrap(), 100)];
    
    levels.update(bids, asks);
    
    // Check the update was successful
    assert_eq!(levels.bids.len(), 1);
    assert_eq!(levels.asks.len(), 1);
    assert!(levels.last_update.elapsed() < Duration::from_secs(1));
}

#[test]
fn test_market_order_execution() {
    // Create a new order book
    let mut lob = LimitOrderBook::new(dec!(0.01), dec!(0.1)).unwrap();
    
    // Add a limit sell order
    let sell_order = Order::limit(
        Side::Ask, 
        dec!(100.0), 
        100, 
        None, 
        None, 
        None
    ).unwrap();
    
    // Process the sell order
    let result = lob.process_order(sell_order);
    assert!(result.is_ok());
    
    // Create a market buy order
    let buy_order = Order::market(
        Side::Bid,
        100,
        None,
        None,
        None
    ).unwrap();
    
    // Process the buy order which should match
    let result = lob.process_order(buy_order);
    assert!(result.is_ok());
    
    // After matching, the ask side should be empty
    assert_eq!(lob.best_ask(), None);
}
