//! Integration tests for the complete CLOB engine
//!
//! These tests verify that all components work together correctly:
//! - Order submission and matching
//! - Market data generation
//! - State management
//! - Risk controls

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn get_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    #[test]
    fn test_clob_engine_creation() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        assert_eq!(engine.symbol, symbol);
        assert_eq!(engine.total_operations, 0);
        assert_eq!(engine.get_active_orders().len(), 0);
    }

    #[test]
    fn test_order_submission_and_matching() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Submit a sell order
        let sell_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Sell,
            50000 * PRICE_SCALE,  // $50,000
            100 * VOLUME_SCALE,   // 100 units
            timestamp,
        );
        
        let result = engine.submit_order(sell_order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Accepted);
        assert!(result.trades.is_empty());
        
        // Check market data
        let (best_bid, best_ask) = engine.get_best_bid_ask();
        assert_eq!(best_bid, None);
        assert_eq!(best_ask, Some(50000 * PRICE_SCALE));
        
        // Submit a matching buy order
        let buy_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,  // $50,000
            50 * VOLUME_SCALE,    // 50 units (partial fill)
            timestamp + 1000,
        );
        
        let result = engine.submit_order(buy_order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Filled);
        assert_eq!(result.trades.len(), 1);
        
        let trade = &result.trades[0];
        assert_eq!(trade.price, 50000 * PRICE_SCALE);
        assert_eq!(trade.size, 50 * VOLUME_SCALE);
        
        // Check updated market data
        let (best_bid, best_ask) = engine.get_best_bid_ask();
        assert_eq!(best_bid, None);
        assert_eq!(best_ask, Some(50000 * PRICE_SCALE)); // Remaining 50 units
        
        // Check spread
        let spread = engine.get_spread();
        assert!(spread.is_none()); // No bid, so no spread
        
        // Check VWAP
        let vwap = engine.get_current_vwap();
        assert_eq!(vwap, Some(50000 * PRICE_SCALE));
    }

    #[test]
    fn test_order_cancellation() {
        let symbol = Symbol::new("ETHUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Submit an order
        let order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            3000 * PRICE_SCALE,   // $3,000
            10 * VOLUME_SCALE,    // 10 units
            timestamp,
        );
        
        let submit_result = engine.submit_order(order).unwrap();
        assert_eq!(submit_result.status, OrderSubmissionStatus::Accepted);
        
        // Verify order is active
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 1);
        
        // Cancel the order
        let cancel_result = engine.cancel_order(
            OrderId::new(1),
            CancellationReason::UserRequested
        ).unwrap();
        
        assert_eq!(cancel_result.order_id, OrderId::new(1));
        assert_eq!(cancel_result.reason, CancellationReason::UserRequested);
        
        // Verify order is no longer active
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 0);
        
        // Check order state
        let state = engine.get_order_state(OrderId::new(1)).unwrap();
        assert!(matches!(state, OrderState::Cancelled { .. }));
    }

    #[test]
    fn test_order_modification() {
        let symbol = Symbol::new("ADAUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Submit an order
        let order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Sell,
            1 * PRICE_SCALE,      // $1.00
            1000 * VOLUME_SCALE,  // 1000 units
            timestamp,
        );
        
        let submit_result = engine.submit_order(order).unwrap();
        assert_eq!(submit_result.status, OrderSubmissionStatus::Accepted);
        
        // Modify the order (change price and size)
        let modification = OrderModification {
            order_id: OrderId::new(1),
            new_price: Some(2 * PRICE_SCALE),     // $2.00
            new_size: Some(500 * VOLUME_SCALE),   // 500 units
            new_time_in_force: None,
            timestamp: timestamp + 1000,
            modification_id: Some("mod_1".to_string()),
        };
        
        let modify_result = engine.modify_order(modification).unwrap();
        assert_eq!(modify_result.original_order_id, OrderId::new(1));
        assert_eq!(modify_result.cancellation_result.reason, CancellationReason::Modification);
        assert_eq!(modify_result.submission_result.status, OrderSubmissionStatus::Accepted);
        
        // Verify the new order has correct parameters
        let new_order = &modify_result.new_order;
        assert_eq!(new_order.price, 2 * PRICE_SCALE);
        assert_eq!(new_order.size, 500 * VOLUME_SCALE);
        
        // Check market data reflects the new price
        let (_, best_ask) = engine.get_best_bid_ask();
        assert_eq!(best_ask, Some(2 * PRICE_SCALE));
    }

    #[test]
    fn test_market_data_generation() {
        let symbol = Symbol::new("SOLUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Create a spread by adding bid and ask orders
        let bid_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            100 * PRICE_SCALE,    // $100
            10 * VOLUME_SCALE,    // 10 units
            timestamp,
        );
        
        let ask_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            102 * PRICE_SCALE,    // $102
            15 * VOLUME_SCALE,    // 15 units
            timestamp + 1000,
        );
        
        engine.submit_order(bid_order).unwrap();
        engine.submit_order(ask_order).unwrap();
        
        // Check market data
        let (best_bid, best_ask) = engine.get_best_bid_ask();
        assert_eq!(best_bid, Some(100 * PRICE_SCALE));
        assert_eq!(best_ask, Some(102 * PRICE_SCALE));
        
        let spread = engine.get_spread().unwrap();
        assert_eq!(spread, 2 * PRICE_SCALE);
        
        let mid_price = engine.get_mid_price().unwrap();
        assert_eq!(mid_price, 101 * PRICE_SCALE);
        
        // Get market depth
        let depth = engine.get_market_depth(5);
        assert_eq!(depth.bids.len(), 1);
        assert_eq!(depth.asks.len(), 1);
        assert_eq!(depth.bids[0], (100 * PRICE_SCALE, 10 * VOLUME_SCALE));
        assert_eq!(depth.asks[0], (102 * PRICE_SCALE, 15 * VOLUME_SCALE));
        
        // Execute a trade to generate market data
        let market_buy = Order::new_market(
            OrderId::new(3),
            symbol.clone(),
            Side::Buy,
            5 * VOLUME_SCALE,     // 5 units
            timestamp + 2000,
        );
        
        let result = engine.submit_order(market_buy).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Filled);
        assert_eq!(result.trades.len(), 1);
        
        // Check VWAP after trade
        let vwap = engine.get_current_vwap().unwrap();
        assert_eq!(vwap, 102 * PRICE_SCALE); // Trade executed at ask price
        
        // Check daily stats
        let daily_stats = engine.get_daily_stats();
        assert_eq!(daily_stats.volume_24h, 5 * VOLUME_SCALE);
        assert_eq!(daily_stats.trade_count_24h, 1);
    }

    #[test]
    fn test_risk_limits() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        
        // Create custom risk limits
        let mut risk_limits = RiskLimits::default();
        risk_limits.max_order_size = 10 * VOLUME_SCALE; // Very small limit
        
        let mut engine = CLOBEngine::new(symbol.clone(), Some(risk_limits)).unwrap();
        
        let timestamp = get_timestamp();
        
        // Try to submit an order that exceeds size limit
        let large_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,  // $50,000
            100 * VOLUME_SCALE,   // 100 units (exceeds limit of 10)
            timestamp,
        );
        
        let result = engine.submit_order(large_order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Rejected);
        assert!(result.message.is_some());
        
        // Verify order was not added to book
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 0);
    }

    #[test]
    fn test_compressed_state() {
        let symbol = Symbol::new("ETHUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Add some orders to create state
        let orders = vec![
            Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 3000 * PRICE_SCALE, 10 * VOLUME_SCALE, timestamp),
            Order::new_limit(OrderId::new(2), symbol.clone(), Side::Buy, 2999 * PRICE_SCALE, 5 * VOLUME_SCALE, timestamp + 1000),
            Order::new_limit(OrderId::new(3), symbol.clone(), Side::Sell, 3001 * PRICE_SCALE, 8 * VOLUME_SCALE, timestamp + 2000),
            Order::new_limit(OrderId::new(4), symbol.clone(), Side::Sell, 3002 * PRICE_SCALE, 12 * VOLUME_SCALE, timestamp + 3000),
        ];
        
        for order in orders {
            engine.submit_order(order).unwrap();
        }
        
        // Get compressed state
        let compressed_state = engine.get_compressed_state();
        assert!(compressed_state.compressed_size() > 0);
        
        // Verify state integrity
        let integrity_check = engine.verify_integrity().unwrap();
        assert!(integrity_check);
        
        // Check that compressed state reflects the order book
        assert_eq!(compressed_state.bid_tree.level_count, 2); // 2 bid levels
        assert_eq!(compressed_state.ask_tree.level_count, 2); // 2 ask levels
        
        assert_eq!(compressed_state.best_bid, Some(3000 * PRICE_SCALE));
        assert_eq!(compressed_state.best_ask, Some(3001 * PRICE_SCALE));
    }

    #[test]
    fn test_engine_statistics() {
        let symbol = Symbol::new("ADAUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Submit some orders
        let order1 = Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 1 * PRICE_SCALE, 100 * VOLUME_SCALE, timestamp);
        let order2 = Order::new_limit(OrderId::new(2), symbol.clone(), Side::Sell, 2 * PRICE_SCALE, 50 * VOLUME_SCALE, timestamp + 1000);
        
        engine.submit_order(order1).unwrap();
        engine.submit_order(order2).unwrap();
        
        // Get engine statistics
        let stats = engine.get_engine_stats();
        
        assert_eq!(stats.symbol, symbol);
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.active_orders, 2);
        assert_eq!(stats.best_bid, Some(1 * PRICE_SCALE));
        assert_eq!(stats.best_ask, Some(2 * PRICE_SCALE));
        assert_eq!(stats.spread, Some(1 * PRICE_SCALE));
        assert_eq!(stats.mid_price, Some((3 * PRICE_SCALE) / 2));
        assert!(stats.compressed_state_size > 0);
    }

    #[test]
    fn test_emergency_cancel_all() {
        let symbol = Symbol::new("SOLUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let timestamp = get_timestamp();
        
        // Submit multiple orders
        let orders = vec![
            Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 100 * PRICE_SCALE, 10 * VOLUME_SCALE, timestamp),
            Order::new_limit(OrderId::new(2), symbol.clone(), Side::Buy, 99 * PRICE_SCALE, 5 * VOLUME_SCALE, timestamp + 1000),
            Order::new_limit(OrderId::new(3), symbol.clone(), Side::Sell, 101 * PRICE_SCALE, 8 * VOLUME_SCALE, timestamp + 2000),
            Order::new_limit(OrderId::new(4), symbol.clone(), Side::Sell, 102 * PRICE_SCALE, 12 * VOLUME_SCALE, timestamp + 3000),
        ];
        
        for order in orders {
            engine.submit_order(order).unwrap();
        }
        
        // Verify orders are active
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 4);
        
        // Cancel all orders
        let cancel_results = engine.cancel_all_orders(CancellationReason::EmergencyHalt).unwrap();
        assert_eq!(cancel_results.len(), 4);
        
        // Verify all orders are cancelled
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 0);
        
        // Check that market data reflects empty book
        let (best_bid, best_ask) = engine.get_best_bid_ask();
        assert_eq!(best_bid, None);
        assert_eq!(best_ask, None);
    }

    #[test]
    fn test_price_time_priority() {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let mut engine = CLOBEngine::new(symbol.clone(), None).unwrap();
        
        let base_timestamp = get_timestamp();
        
        // Add multiple orders at the same price with different timestamps
        let order1 = Order::new_limit(
            OrderId::new(1), 
            symbol.clone(), 
            Side::Sell, 
            50000 * PRICE_SCALE, 
            10 * VOLUME_SCALE, 
            base_timestamp
        );
        let order2 = Order::new_limit(
            OrderId::new(2), 
            symbol.clone(), 
            Side::Sell, 
            50000 * PRICE_SCALE, 
            15 * VOLUME_SCALE, 
            base_timestamp + 1000
        );
        
        engine.submit_order(order1).unwrap();
        engine.submit_order(order2).unwrap();
        
        // Submit a buy order that should match the first order (time priority)
        let buy_order = Order::new_limit(
            OrderId::new(3),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,
            10 * VOLUME_SCALE,
            base_timestamp + 2000,
        );
        
        let result = engine.submit_order(buy_order).unwrap();
        assert_eq!(result.status, OrderSubmissionStatus::Filled);
        assert_eq!(result.trades.len(), 1);
        
        // The trade should be with the first order (OrderId::new(1))
        let trade = &result.trades[0];
        assert_eq!(trade.seller_order_id, OrderId::new(1));
        
        // Second order should still be in the book
        let active_orders = engine.get_active_orders();
        assert_eq!(active_orders.len(), 1);
        assert_eq!(active_orders[0].0, OrderId::new(2));
    }
}