//! Comprehensive tests for the order management system

#[cfg(test)]
mod tests {
    use super::super::management::*;
    use crate::orderbook::types::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::sync::atomic::Ordering;

    fn test_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    fn test_symbol() -> Symbol {
        Symbol::new("BTCUSD").unwrap()
    }

    fn create_test_order(id: u64, side: Side, price: u64, size: u64) -> Order {
        Order::new_limit(
            OrderId::new(id),
            test_symbol(),
            side,
            price * PRICE_SCALE,
            size * VOLUME_SCALE,
            test_timestamp(),
        )
    }

    fn create_test_order_with_tif(id: u64, side: Side, price: u64, size: u64, tif: TimeInForce) -> Order {
        let mut order = create_test_order(id, side, price, size);
        order.time_in_force = tif;
        order
    }

    #[test]
    fn test_order_manager_creation() {
        let symbol = test_symbol();
        let manager = OrderManager::new(symbol.clone(), None);
        
        assert_eq!(manager.order_book.symbol, symbol);
        assert_eq!(manager.active_order_count, 0);
        assert_eq!(manager.total_orders_processed.load(Ordering::SeqCst), 0);
        assert!(manager.order_states.is_empty());
        assert!(manager.order_events.is_empty());
        assert_eq!(manager.position.size, 0);
        assert_eq!(manager.position.realized_pnl, 0);
    }

    #[test]
    fn test_order_submission_success() {
        let mut manager = OrderManager::new(test_symbol(), None);
        let order = create_test_order(1, Side::Buy, 50000, 1);
        
        let result = manager.submit_order(order.clone()).unwrap();
        
        assert_eq!(result.order_id, OrderId::new(1));
        assert_eq!(result.status, OrderSubmissionStatus::Accepted);
        assert!(result.message.is_none());
        assert!(result.trades.is_empty());
        
        // Check state tracking
        assert_eq!(manager.get_order_state(OrderId::new(1)), Some(&OrderState::Active));
        assert_eq!(manager.active_order_count, 1);
        assert_eq!(manager.total_orders_processed.load(Ordering::SeqCst), 1);
        
        // Check events were logged
        let events = manager.get_order_events(OrderId::new(1));
        assert!(!events.is_empty());
        
        // Should have Created, Validated, RiskCheck, and AddedToBook events
        let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
        assert!(event_types.contains(&&OrderEventType::Created));
        assert!(event_types.contains(&&OrderEventType::Validated));
        assert!(event_types.contains(&&OrderEventType::RiskCheck));
    }

    #[test]
    fn test_order_validation_failure() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Create order with size too large
        let mut order = create_test_order(1, Side::Buy, 50000, 1);
        order.size = manager.risk_limits.max_order_size + 1;
        
        let result = manager.submit_order(order).unwrap();
        
        assert_eq!(result.status, OrderSubmissionStatus::Rejected);
        assert!(result.message.is_some());
        assert_eq!(manager.get_order_state(OrderId::new(1)), Some(&OrderState::Rejected { reason: result.message.unwrap() }));
        assert_eq!(manager.active_order_count, 0);
    }

    #[test]
    fn test_order_cancellation() {
        let mut manager = OrderManager::new(test_symbol(), None);
        let order = create_test_order(1, Side::Buy, 50000, 1);
        
        // Submit order
        manager.submit_order(order).unwrap();
        assert_eq!(manager.active_order_count, 1);
        
        // Cancel order
        let cancel_result = manager.cancel_order(OrderId::new(1), CancellationReason::UserRequested).unwrap();
        
        assert_eq!(cancel_result.order_id, OrderId::new(1));
        assert_eq!(cancel_result.reason, CancellationReason::UserRequested);
        assert_eq!(manager.get_order_state(OrderId::new(1)), Some(&OrderState::Cancelled { reason: CancellationReason::UserRequested }));
        assert_eq!(manager.active_order_count, 0);
        
        // Check cancellation event was logged
        let events = manager.get_order_events(OrderId::new(1));
        let cancel_events: Vec<_> = events.iter()
            .filter(|e| e.event_type == OrderEventType::Cancelled)
            .collect();
        assert_eq!(cancel_events.len(), 1);
    }

    #[test]
    fn test_order_modification() {
        let mut manager = OrderManager::new(test_symbol(), None);
        let order = create_test_order(1, Side::Buy, 50000, 1);
        
        // Submit original order
        manager.submit_order(order).unwrap();
        
        // Modify order
        let modification = OrderModification {
            order_id: OrderId::new(1),
            new_price: Some(49000 * PRICE_SCALE),
            new_size: Some(2 * VOLUME_SCALE),
            new_time_in_force: None,
            timestamp: test_timestamp(),
            modification_id: Some("mod_1".to_string()),
        };
        
        let mod_result = manager.modify_order(modification).unwrap();
        
        assert_eq!(mod_result.original_order_id, OrderId::new(1));
        assert_ne!(mod_result.new_order_id, OrderId::new(1)); // Should be different
        assert_eq!(mod_result.new_order.price, 49000 * PRICE_SCALE);
        assert_eq!(mod_result.new_order.size, 2 * VOLUME_SCALE);
        assert_eq!(mod_result.modification_id, Some("mod_1".to_string()));
        
        // Original order should be cancelled
        assert_eq!(manager.get_order_state(OrderId::new(1)), Some(&OrderState::Cancelled { reason: CancellationReason::Modification }));
        
        // New order should be active
        assert_eq!(manager.get_order_state(mod_result.new_order_id), Some(&OrderState::Active));
    }

    #[test]
    fn test_position_tracking() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Create a trade that affects position
        let trade = Trade::new(
            1,
            test_symbol(),
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            test_timestamp(),
            OrderId::new(1), // Our order
            OrderId::new(2), // Counterparty
            false, // We're the taker (buyer)
            1,
        );
        
        // Add our order to the book first
        let order = create_test_order(1, Side::Buy, 50000, 100);
        manager.order_book.orders.insert(OrderId::new(1), order);
        
        // Update position with trade
        manager.update_position(&trade);
        
        assert_eq!(manager.position.size, 100 * VOLUME_SCALE as i64);
        assert_eq!(manager.position.avg_price, 50000 * PRICE_SCALE);
        assert_eq!(manager.position.total_volume, 100 * VOLUME_SCALE);
        assert_eq!(manager.position.trade_count, 1);
    }

    #[test]
    fn test_order_expiration() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Create GTD order that expires in the past
        let expiration_time = test_timestamp() - 1000; // 1 microsecond ago
        let order = create_test_order_with_tif(1, Side::Buy, 50000, 1, TimeInForce::GTD(expiration_time));
        
        // Submit order (should be accepted initially)
        manager.submit_order(order).unwrap();
        assert_eq!(manager.active_order_count, 1);
        
        // Handle expired orders
        let expired_results = manager.handle_expired_orders().unwrap();
        
        assert_eq!(expired_results.len(), 1);
        assert_eq!(expired_results[0].order_id, OrderId::new(1));
        assert_eq!(expired_results[0].reason, CancellationReason::Expired);
        assert_eq!(manager.get_order_state(OrderId::new(1)), Some(&OrderState::Cancelled { reason: CancellationReason::Expired }));
        assert_eq!(manager.active_order_count, 0);
    }

    #[test]
    fn test_rate_limiting() {
        let mut rate_limiter = OrderRateLimiter::new(2, 1000); // 2 orders per 1 microsecond
        let base_time = test_timestamp();
        
        // First order should be allowed
        assert!(rate_limiter.can_submit_order(base_time));
        rate_limiter.record_order(base_time);
        
        // Second order should be allowed
        assert!(rate_limiter.can_submit_order(base_time + 100));
        rate_limiter.record_order(base_time + 100);
        
        // Third order should be rejected (within window)
        assert!(!rate_limiter.can_submit_order(base_time + 200));
        
        // After window expires, should be allowed again
        assert!(rate_limiter.can_submit_order(base_time + 1500));
    }

    #[test]
    fn test_risk_limits_enforcement() {
        let mut risk_limits = RiskLimits::default();
        risk_limits.max_order_size = 10 * VOLUME_SCALE;
        risk_limits.max_orders_per_symbol = 2;
        
        let mut manager = OrderManager::new(test_symbol(), Some(risk_limits));
        
        // First order should be accepted
        let order1 = create_test_order(1, Side::Buy, 50000, 5);
        let result1 = manager.submit_order(order1).unwrap();
        assert_eq!(result1.status, OrderSubmissionStatus::Accepted);
        
        // Second order should be accepted
        let order2 = create_test_order(2, Side::Sell, 51000, 5);
        let result2 = manager.submit_order(order2).unwrap();
        assert_eq!(result2.status, OrderSubmissionStatus::Accepted);
        
        // Third order should be rejected (max orders exceeded)
        let order3 = create_test_order(3, Side::Buy, 49000, 5);
        let result3 = manager.submit_order(order3).unwrap();
        assert_eq!(result3.status, OrderSubmissionStatus::RejectedRisk);
        
        // Order with size too large should be rejected
        let order4 = create_test_order(4, Side::Buy, 50000, 15);
        let result4 = manager.submit_order(order4).unwrap();
        assert_eq!(result4.status, OrderSubmissionStatus::Rejected);
    }

    #[test]
    fn test_emergency_halt() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Submit multiple orders
        manager.submit_order(create_test_order(1, Side::Buy, 50000, 1)).unwrap();
        manager.submit_order(create_test_order(2, Side::Sell, 51000, 1)).unwrap();
        assert_eq!(manager.active_order_count, 2);
        
        // Trigger emergency halt
        let halt_results = manager.emergency_halt().unwrap();
        
        assert_eq!(halt_results.len(), 2);
        assert_eq!(manager.active_order_count, 0);
        
        // All orders should be cancelled with EmergencyHalt reason
        for result in halt_results {
            assert_eq!(result.reason, CancellationReason::EmergencyHalt);
        }
        
        // Risk limits should be set to prevent new orders
        assert_eq!(manager.risk_limits.max_orders_per_symbol, 0);
        assert_eq!(manager.risk_limits.max_order_size, 0);
        
        // New orders should be rejected
        let new_order = create_test_order(3, Side::Buy, 50000, 1);
        let result = manager.submit_order(new_order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::RejectedRisk);
    }

    #[test]
    fn test_resume_operations() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Trigger emergency halt
        manager.emergency_halt().unwrap();
        
        // Resume operations
        manager.resume_operations(None);
        
        // Risk limits should be restored to defaults
        let default_limits = RiskLimits::default();
        assert_eq!(manager.risk_limits.max_orders_per_symbol, default_limits.max_orders_per_symbol);
        assert_eq!(manager.risk_limits.max_order_size, default_limits.max_order_size);
        
        // New orders should be accepted again
        let order = create_test_order(1, Side::Buy, 50000, 1);
        let result = manager.submit_order(order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Accepted);
    }

    #[test]
    fn test_performance_metrics() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Update performance metrics
        manager.update_performance_metrics(1000); // 1 microsecond latency
        manager.update_performance_metrics(2000); // 2 microseconds latency
        
        assert_eq!(manager.performance_metrics.avg_processing_latency_ns, 1100); // EMA calculation
        assert_eq!(manager.performance_metrics.max_processing_latency_ns, 2000);
        
        // Submit some orders to test fill rate
        manager.submit_order(create_test_order(1, Side::Buy, 50000, 1)).unwrap();
        manager.submit_order(create_test_order(2, Side::Sell, 51000, 1)).unwrap();
        
        let stats = manager.get_statistics();
        assert_eq!(stats.total_orders_processed, 2);
        assert_eq!(stats.active_order_count, 2);
        assert!(stats.performance_metrics.fill_rate >= 0.0);
    }

    #[test]
    fn test_comprehensive_statistics() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Submit various orders
        manager.submit_order(create_test_order(1, Side::Buy, 50000, 1)).unwrap();
        manager.submit_order(create_test_order(2, Side::Sell, 51000, 1)).unwrap();
        manager.cancel_order(OrderId::new(1), CancellationReason::UserRequested).unwrap();
        
        let stats = manager.get_statistics();
        
        assert_eq!(stats.total_orders_processed, 2);
        assert_eq!(stats.active_order_count, 1);
        assert_eq!(stats.state_counts.active, 1);
        assert_eq!(stats.state_counts.cancelled, 1);
        assert!(stats.total_events > 0);
        assert_eq!(stats.position.size, 0);
    }

    #[test]
    fn test_order_event_logging() {
        let mut manager = OrderManager::new(test_symbol(), None);
        let order = create_test_order(1, Side::Buy, 50000, 1);
        
        // Submit and cancel order
        manager.submit_order(order).unwrap();
        manager.cancel_order(OrderId::new(1), CancellationReason::UserRequested).unwrap();
        
        let events = manager.get_order_events(OrderId::new(1));
        
        // Should have multiple events
        assert!(events.len() >= 4); // Created, Validated, RiskCheck, AddedToBook, Cancelled
        
        // Events should be in chronological order
        for i in 1..events.len() {
            assert!(events[i].sequence > events[i-1].sequence);
        }
        
        // Check specific event types exist
        let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
        assert!(event_types.contains(&&OrderEventType::Created));
        assert!(event_types.contains(&&OrderEventType::Validated));
        assert!(event_types.contains(&&OrderEventType::RiskCheck));
        assert!(event_types.contains(&&OrderEventType::Cancelled));
    }

    #[test]
    fn test_get_active_orders() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Submit multiple orders
        manager.submit_order(create_test_order(1, Side::Buy, 50000, 1)).unwrap();
        manager.submit_order(create_test_order(2, Side::Sell, 51000, 1)).unwrap();
        manager.submit_order(create_test_order(3, Side::Buy, 49000, 1)).unwrap();
        
        // Cancel one order
        manager.cancel_order(OrderId::new(2), CancellationReason::UserRequested).unwrap();
        
        let active_orders = manager.get_active_orders();
        
        assert_eq!(active_orders.len(), 2);
        
        let active_ids: Vec<OrderId> = active_orders.iter().map(|(id, _)| *id).collect();
        assert!(active_ids.contains(&OrderId::new(1)));
        assert!(active_ids.contains(&OrderId::new(3)));
        assert!(!active_ids.contains(&OrderId::new(2))); // Cancelled order
    }

    #[test]
    fn test_cancel_all_orders() {
        let mut manager = OrderManager::new(test_symbol(), None);
        
        // Submit multiple orders
        manager.submit_order(create_test_order(1, Side::Buy, 50000, 1)).unwrap();
        manager.submit_order(create_test_order(2, Side::Sell, 51000, 1)).unwrap();
        manager.submit_order(create_test_order(3, Side::Buy, 49000, 1)).unwrap();
        
        assert_eq!(manager.active_order_count, 3);
        
        // Cancel all orders
        let cancel_results = manager.cancel_all_orders(CancellationReason::MarketClosed).unwrap();
        
        assert_eq!(cancel_results.len(), 3);
        assert_eq!(manager.active_order_count, 0);
        
        // All orders should be cancelled with MarketClosed reason
        for result in cancel_results {
            assert_eq!(result.reason, CancellationReason::MarketClosed);
        }
        
        // No active orders should remain
        let active_orders = manager.get_active_orders();
        assert!(active_orders.is_empty());
    }
}