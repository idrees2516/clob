//! Integration tests for order management system with CLOB and matching engine

use hf_quoting_liquidity_clob::orderbook::{
    OrderManager, Order, OrderId, Symbol, Side, TimeInForce, CancellationReason,
    OrderModification, RiskLimits, MatchingEngine, Trade,
    PRICE_SCALE, VOLUME_SCALE
};
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::Ordering;

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[test]
fn test_order_management_with_matching_engine() {
    let symbol = Symbol::new("BTCUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    let mut matching_engine = MatchingEngine::new();
    
    let base_time = get_timestamp();
    matching_engine.set_current_time(base_time);
    
    // Submit a buy order
    let buy_order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Buy,
        50000 * PRICE_SCALE,
        100 * VOLUME_SCALE,
        base_time,
    );
    
    let buy_result = order_manager.submit_order(buy_order).unwrap();
    assert_eq!(buy_result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    assert_eq!(order_manager.active_order_count, 1);
    
    // Submit a sell order that should match
    let sell_order = Order::new_limit(
        OrderId::new(2),
        symbol.clone(),
        Side::Sell,
        50000 * PRICE_SCALE,
        50 * VOLUME_SCALE,
        base_time + 1000,
    );
    
    // Process through matching engine
    let match_result = matching_engine.process_order(&mut order_manager.order_book, sell_order).unwrap();
    
    // Should have generated a trade
    assert_eq!(match_result.filled_size, 50 * VOLUME_SCALE);
    assert_eq!(match_result.trades.len(), 1);
    
    let trade = &match_result.trades[0];
    assert_eq!(trade.price, 50000 * PRICE_SCALE);
    assert_eq!(trade.size, 50 * VOLUME_SCALE);
    
    // Update position with the trade
    order_manager.update_position(trade);
    
    // Check position tracking
    assert_eq!(order_manager.position.size, -(50 * VOLUME_SCALE as i64)); // Short position
    assert_eq!(order_manager.position.avg_price, 50000 * PRICE_SCALE);
    assert_eq!(order_manager.position.total_volume, 50 * VOLUME_SCALE);
    assert_eq!(order_manager.position.trade_count, 1);
}

#[test]
fn test_comprehensive_order_lifecycle() {
    let symbol = Symbol::new("ETHUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    // 1. Submit initial orders
    let orders = vec![
        Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 2000 * PRICE_SCALE, 10 * VOLUME_SCALE, get_timestamp()),
        Order::new_limit(OrderId::new(2), symbol.clone(), Side::Sell, 2100 * PRICE_SCALE, 8 * VOLUME_SCALE, get_timestamp()),
        Order::new_limit(OrderId::new(3), symbol.clone(), Side::Buy, 1950 * PRICE_SCALE, 15 * VOLUME_SCALE, get_timestamp()),
    ];
    
    for order in orders {
        let result = order_manager.submit_order(order).unwrap();
        assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    }
    
    assert_eq!(order_manager.active_order_count, 3);
    assert_eq!(order_manager.total_orders_processed.load(Ordering::SeqCst), 3);
    
    // 2. Cancel one order
    let cancel_result = order_manager.cancel_order(OrderId::new(2), CancellationReason::UserRequested).unwrap();
    assert_eq!(cancel_result.order_id, OrderId::new(2));
    assert_eq!(order_manager.active_order_count, 2);
    
    // 3. Modify an order
    let modification = OrderModification {
        order_id: OrderId::new(3),
        new_price: Some(1900 * PRICE_SCALE),
        new_size: Some(20 * VOLUME_SCALE),
        new_time_in_force: None,
        timestamp: get_timestamp(),
        modification_id: Some("test_mod".to_string()),
    };
    
    let mod_result = order_manager.modify_order(modification).unwrap();
    assert_eq!(mod_result.original_order_id, OrderId::new(3));
    assert_ne!(mod_result.new_order_id, OrderId::new(3));
    assert_eq!(mod_result.new_order.price, 1900 * PRICE_SCALE);
    assert_eq!(mod_result.new_order.size, 20 * VOLUME_SCALE);
    
    // Original order should be cancelled, new order should be active
    assert_eq!(order_manager.get_order_state(OrderId::new(3)), 
               Some(&hf_quoting_liquidity_clob::orderbook::OrderState::Cancelled { 
                   reason: CancellationReason::Modification 
               }));
    assert_eq!(order_manager.get_order_state(mod_result.new_order_id), 
               Some(&hf_quoting_liquidity_clob::orderbook::OrderState::Active));
    
    // 4. Check comprehensive statistics
    let stats = order_manager.get_statistics();
    assert_eq!(stats.total_orders_processed, 4); // 3 original + 1 replacement
    assert_eq!(stats.active_order_count, 2);
    assert_eq!(stats.state_counts.active, 2);
    assert_eq!(stats.state_counts.cancelled, 2); // 1 user cancelled + 1 modification cancelled
    
    // 5. Cancel all remaining orders
    let cancel_all_results = order_manager.cancel_all_orders(CancellationReason::MarketClosed).unwrap();
    assert_eq!(cancel_all_results.len(), 2);
    assert_eq!(order_manager.active_order_count, 0);
    
    for result in cancel_all_results {
        assert_eq!(result.reason, CancellationReason::MarketClosed);
    }
}

#[test]
fn test_risk_management_integration() {
    let mut risk_limits = RiskLimits::default();
    risk_limits.max_order_size = 50 * VOLUME_SCALE;
    risk_limits.max_orders_per_symbol = 3;
    risk_limits.max_total_exposure = 500_000 * PRICE_SCALE;
    risk_limits.max_long_position = 100 * VOLUME_SCALE as i64;
    risk_limits.max_short_position = -100 * VOLUME_SCALE as i64;
    
    let symbol = Symbol::new("ADAUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), Some(risk_limits));
    
    // Test order size limit
    let large_order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Buy,
        100 * PRICE_SCALE,
        60 * VOLUME_SCALE, // Exceeds limit
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(large_order).unwrap();
    assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected);
    assert!(result.message.is_some());
    
    // Test order count limit
    for i in 1..=3 {
        let order = Order::new_limit(
            OrderId::new(i),
            symbol.clone(),
            Side::Buy,
            (100 - i) * PRICE_SCALE,
            10 * VOLUME_SCALE,
            get_timestamp(),
        );
        
        let result = order_manager.submit_order(order).unwrap();
        assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    }
    
    // Fourth order should be rejected
    let fourth_order = Order::new_limit(
        OrderId::new(4),
        symbol.clone(),
        Side::Buy,
        96 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(fourth_order).unwrap();
    assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk);
    
    assert_eq!(order_manager.active_order_count, 3);
}

#[test]
fn test_order_expiration_integration() {
    let symbol = Symbol::new("SOLUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    let current_time = get_timestamp();
    let expiration_time = current_time + 1000; // 1 microsecond from now
    
    // Create GTD order
    let mut gtd_order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Buy,
        50 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        current_time,
    );
    gtd_order.time_in_force = TimeInForce::GTD(expiration_time);
    
    let result = order_manager.submit_order(gtd_order).unwrap();
    assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    assert_eq!(order_manager.active_order_count, 1);
    
    // Wait for expiration (simulate time passing)
    std::thread::sleep(std::time::Duration::from_millis(1));
    
    // Handle expired orders
    let expired_results = order_manager.handle_expired_orders().unwrap();
    
    assert_eq!(expired_results.len(), 1);
    assert_eq!(expired_results[0].order_id, OrderId::new(1));
    assert_eq!(expired_results[0].reason, CancellationReason::Expired);
    assert_eq!(order_manager.active_order_count, 0);
    
    // Order should be in expired state first, then cancelled
    assert_eq!(order_manager.get_order_state(OrderId::new(1)), 
               Some(&hf_quoting_liquidity_clob::orderbook::OrderState::Cancelled { 
                   reason: CancellationReason::Expired 
               }));
}

#[test]
fn test_position_tracking_with_multiple_trades() {
    let symbol = Symbol::new("DOTUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    // Simulate multiple trades affecting position
    let trades = vec![
        // Buy 100 units at $10
        Trade::new(1, symbol.clone(), 10 * PRICE_SCALE, 100 * VOLUME_SCALE, get_timestamp(), 
                   OrderId::new(1), OrderId::new(100), false, 1),
        // Buy 50 units at $12
        Trade::new(2, symbol.clone(), 12 * PRICE_SCALE, 50 * VOLUME_SCALE, get_timestamp(), 
                   OrderId::new(2), OrderId::new(101), false, 2),
        // Sell 75 units at $15
        Trade::new(3, symbol.clone(), 15 * PRICE_SCALE, 75 * VOLUME_SCALE, get_timestamp(), 
                   OrderId::new(102), OrderId::new(3), true, 3),
    ];
    
    // Add our orders to the book for position tracking
    for i in 1..=3 {
        let order = Order::new_limit(
            OrderId::new(i),
            symbol.clone(),
            if i <= 2 { Side::Buy } else { Side::Sell },
            (10 + i * 2) * PRICE_SCALE,
            if i <= 2 { (100 + i * 50) } else { 75 } * VOLUME_SCALE,
            get_timestamp(),
        );
        order_manager.order_book.orders.insert(OrderId::new(i), order);
    }
    
    // Process trades
    for trade in &trades {
        order_manager.update_position(trade);
    }
    
    // Final position should be: +100 +50 -75 = +75 units
    assert_eq!(order_manager.position.size, 75 * VOLUME_SCALE as i64);
    
    // Average price should be weighted average of remaining position
    // Bought 150 units: 100@$10 + 50@$12 = $1600 total
    // Sold 75 units at $15, realizing profit
    // Remaining 75 units should have avg price of remaining long position
    let expected_avg_price = ((100 * 10 + 50 * 12) * PRICE_SCALE) / 150;
    assert_eq!(order_manager.position.avg_price, expected_avg_price);
    
    // Should have realized some profit from the sale
    assert!(order_manager.position.realized_pnl > 0);
    
    // Total volume should be sum of all trade sizes
    assert_eq!(order_manager.position.total_volume, 225 * VOLUME_SCALE); // 100 + 50 + 75
    assert_eq!(order_manager.position.trade_count, 3);
    
    // Test unrealized PnL calculation
    order_manager.update_unrealized_pnl(20 * PRICE_SCALE); // Current price $20
    
    // Unrealized PnL should be positive (position worth more than avg price)
    assert!(order_manager.position.unrealized_pnl > 0);
}

#[test]
fn test_emergency_halt_integration() {
    let symbol = Symbol::new("AVAXUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    // Submit multiple orders
    for i in 1..=5 {
        let order = Order::new_limit(
            OrderId::new(i),
            symbol.clone(),
            if i % 2 == 0 { Side::Buy } else { Side::Sell },
            (100 + i * 10) * PRICE_SCALE,
            (5 + i) * VOLUME_SCALE,
            get_timestamp(),
        );
        
        let result = order_manager.submit_order(order).unwrap();
        assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    }
    
    assert_eq!(order_manager.active_order_count, 5);
    
    // Trigger emergency halt
    let halt_results = order_manager.emergency_halt().unwrap();
    
    assert_eq!(halt_results.len(), 5);
    assert_eq!(order_manager.active_order_count, 0);
    
    // All orders should be cancelled with EmergencyHalt reason
    for result in halt_results {
        assert_eq!(result.reason, CancellationReason::EmergencyHalt);
    }
    
    // Risk limits should prevent new orders
    let emergency_order = Order::new_limit(
        OrderId::new(10),
        symbol.clone(),
        Side::Buy,
        200 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(emergency_order).unwrap();
    assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk);
    
    // Resume operations
    order_manager.resume_operations(None);
    
    // Should be able to submit orders again
    let resume_order = Order::new_limit(
        OrderId::new(11),
        symbol.clone(),
        Side::Buy,
        200 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(resume_order).unwrap();
    assert_eq!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted);
    assert_eq!(order_manager.active_order_count, 1);
}

#[test]
fn test_performance_metrics_integration() {
    let symbol = Symbol::new("MATICUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    // Submit orders and track performance
    let start_time = get_timestamp();
    
    for i in 1..=10 {
        let processing_start = get_timestamp();
        
        let order = Order::new_limit(
            OrderId::new(i),
            symbol.clone(),
            Side::Buy,
            (50 + i) * PRICE_SCALE,
            5 * VOLUME_SCALE,
            get_timestamp(),
        );
        
        let _result = order_manager.submit_order(order).unwrap();
        
        let processing_end = get_timestamp();
        let latency = processing_end - processing_start;
        
        order_manager.update_performance_metrics(latency);
    }
    
    // Check performance metrics
    assert!(order_manager.performance_metrics.avg_processing_latency_ns > 0);
    assert!(order_manager.performance_metrics.max_processing_latency_ns > 0);
    assert!(order_manager.performance_metrics.max_processing_latency_ns >= 
            order_manager.performance_metrics.avg_processing_latency_ns);
    
    // Fill rate should be 0% since no orders were filled
    assert_eq!(order_manager.performance_metrics.fill_rate, 0.0);
    
    let stats = order_manager.get_statistics();
    assert_eq!(stats.total_orders_processed, 10);
    assert_eq!(stats.active_order_count, 10);
    assert_eq!(stats.state_counts.active, 10);
}

#[test]
fn test_order_event_audit_trail() {
    let symbol = Symbol::new("LINKUSD").unwrap();
    let mut order_manager = OrderManager::new(symbol.clone(), None);
    
    // Submit an order
    let order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Buy,
        25 * PRICE_SCALE,
        20 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    order_manager.submit_order(order).unwrap();
    
    // Modify the order
    let modification = OrderModification {
        order_id: OrderId::new(1),
        new_price: Some(24 * PRICE_SCALE),
        new_size: None,
        new_time_in_force: None,
        timestamp: get_timestamp(),
        modification_id: Some("audit_test".to_string()),
    };
    
    let mod_result = order_manager.modify_order(modification).unwrap();
    
    // Cancel the new order
    order_manager.cancel_order(mod_result.new_order_id, CancellationReason::UserRequested).unwrap();
    
    // Check audit trail for original order
    let original_events = order_manager.get_order_events(OrderId::new(1));
    assert!(!original_events.is_empty());
    
    // Should have Created, Validated, RiskCheck, AddedToBook, Modified, Cancelled events
    let event_types: Vec<_> = original_events.iter().map(|e| &e.event_type).collect();
    assert!(event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Created));
    assert!(event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Validated));
    assert!(event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::RiskCheck));
    assert!(event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Modified));
    assert!(event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Cancelled));
    
    // Events should be in chronological order
    for i in 1..original_events.len() {
        assert!(original_events[i].sequence > original_events[i-1].sequence);
        assert!(original_events[i].timestamp >= original_events[i-1].timestamp);
    }
    
    // Check audit trail for new order
    let new_events = order_manager.get_order_events(mod_result.new_order_id);
    assert!(!new_events.is_empty());
    
    let new_event_types: Vec<_> = new_events.iter().map(|e| &e.event_type).collect();
    assert!(new_event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Created));
    assert!(new_event_types.contains(&&hf_quoting_liquidity_clob::orderbook::OrderEventType::Cancelled));
}