//! Order Management System Demo
//!
//! This example demonstrates the comprehensive order management capabilities
//! including:
//! - Order submission with validation and risk checks
//! - Order cancellation with book cleanup
//! - Order modification (cancel-replace) functionality
//! - Order status tracking and lifecycle management
//! - Position tracking and PnL calculation
//! - Performance monitoring and metrics
//! - Risk management and emergency controls

use hf_quoting_liquidity_clob::orderbook::{
    OrderManager, Order, OrderId, Symbol, Side, TimeInForce, CancellationReason,
    OrderModification, RiskLimits, PRICE_SCALE, VOLUME_SCALE
};
use std::time::{SystemTime, UNIX_EPOCH};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Order Management System Demo ===\n");
    
    // Create a trading symbol
    let symbol = Symbol::new("ETHUSD")?;
    println!("Trading Symbol: {}", symbol);
    
    // Create custom risk limits for demonstration
    let mut risk_limits = RiskLimits::default();
    risk_limits.max_order_size = 100 * VOLUME_SCALE;      // 100 ETH max
    risk_limits.max_orders_per_symbol = 10;               // 10 orders max
    risk_limits.max_total_exposure = 1_000_000 * PRICE_SCALE; // $1M exposure
    risk_limits.max_long_position = 500 * VOLUME_SCALE as i64;  // 500 ETH long
    risk_limits.max_short_position = -500 * VOLUME_SCALE as i64; // 500 ETH short
    risk_limits.max_orders_per_second = 5;                // 5 orders/sec
    
    // Create order manager with custom risk limits
    let mut order_manager = OrderManager::new(symbol.clone(), Some(risk_limits));
    
    println!("\n1. Order Submission and Validation:");
    demonstrate_order_submission(&mut order_manager)?;
    
    println!("\n2. Order Cancellation:");
    demonstrate_order_cancellation(&mut order_manager)?;
    
    println!("\n3. Order Modification (Cancel-Replace):");
    demonstrate_order_modification(&mut order_manager)?;
    
    println!("\n4. Risk Management Controls:");
    demonstrate_risk_management(&mut order_manager)?;
    
    println!("\n5. Order Expiration Handling:");
    demonstrate_order_expiration(&mut order_manager)?;
    
    println!("\n6. Position Tracking:");
    demonstrate_position_tracking(&mut order_manager)?;
    
    println!("\n7. Performance Monitoring:");
    demonstrate_performance_monitoring(&mut order_manager)?;
    
    println!("\n8. Emergency Controls:");
    demonstrate_emergency_controls(&mut order_manager)?;
    
    println!("\n9. Comprehensive Statistics:");
    demonstrate_statistics(&order_manager);
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demonstrate_order_submission(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Submitting various orders...");
    
    // Submit a valid buy order
    let buy_order = Order::new_limit(
        OrderId::new(1),
        order_manager.order_book.symbol.clone(),
        Side::Buy,
        2000 * PRICE_SCALE, // $2000
        10 * VOLUME_SCALE,   // 10 ETH
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(buy_order)?;
    println!("✓ Buy Order #{}: {} - Added to book", 
             result.order_id.as_u64(), 
             match result.status {
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted => "Accepted",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::PartiallyFilled => "Partially Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Filled => "Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected => "Rejected",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk => "Rejected (Risk)",
             });
    
    // Submit a valid sell order
    let sell_order = Order::new_limit(
        OrderId::new(2),
        order_manager.order_book.symbol.clone(),
        Side::Sell,
        2100 * PRICE_SCALE, // $2100
        8 * VOLUME_SCALE,    // 8 ETH
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(sell_order)?;
    println!("✓ Sell Order #{}: {} - Added to book", 
             result.order_id.as_u64(),
             match result.status {
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted => "Accepted",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::PartiallyFilled => "Partially Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Filled => "Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected => "Rejected",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk => "Rejected (Risk)",
             });
    
    // Try to submit an invalid order (too large)
    let invalid_order = Order::new_limit(
        OrderId::new(3),
        order_manager.order_book.symbol.clone(),
        Side::Buy,
        2000 * PRICE_SCALE,
        200 * VOLUME_SCALE, // Exceeds max_order_size
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(invalid_order)?;
    println!("✗ Large Order #{}: {} - {}", 
             result.order_id.as_u64(),
             match result.status {
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted => "Accepted",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::PartiallyFilled => "Partially Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Filled => "Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected => "Rejected",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk => "Rejected (Risk)",
             },
             result.message.unwrap_or_default());
    
    println!("Active orders: {}", order_manager.active_order_count);
    
    Ok(())
}

fn demonstrate_order_cancellation(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating order cancellation...");
    
    // Cancel the first order
    let cancel_result = order_manager.cancel_order(
        OrderId::new(1), 
        CancellationReason::UserRequested
    )?;
    
    println!("✓ Cancelled Order #{} - Reason: {:?}", 
             cancel_result.order_id.as_u64(),
             cancel_result.reason);
    
    // Check order state
    if let Some(state) = order_manager.get_order_state(OrderId::new(1)) {
        println!("  Order state: {:?}", state);
    }
    
    println!("Active orders after cancellation: {}", order_manager.active_order_count);
    
    Ok(())
}

fn demonstrate_order_modification(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating order modification...");
    
    // Submit a new order to modify
    let order = Order::new_limit(
        OrderId::new(4),
        order_manager.order_book.symbol.clone(),
        Side::Buy,
        1950 * PRICE_SCALE, // $1950
        15 * VOLUME_SCALE,   // 15 ETH
        get_timestamp(),
    );
    
    order_manager.submit_order(order)?;
    println!("✓ Submitted Order #4 for modification");
    
    // Modify the order (change price and size)
    let modification = OrderModification {
        order_id: OrderId::new(4),
        new_price: Some(1900 * PRICE_SCALE), // Lower price
        new_size: Some(20 * VOLUME_SCALE),    // Larger size
        new_time_in_force: None,
        timestamp: get_timestamp(),
        modification_id: Some("mod_001".to_string()),
    };
    
    let mod_result = order_manager.modify_order(modification)?;
    
    println!("✓ Modified Order #4:");
    println!("  Original Order ID: {}", mod_result.original_order_id.as_u64());
    println!("  New Order ID: {}", mod_result.new_order_id.as_u64());
    println!("  New Price: ${:.2}", mod_result.new_order.price as f64 / PRICE_SCALE as f64);
    println!("  New Size: {:.2} ETH", mod_result.new_order.size as f64 / VOLUME_SCALE as f64);
    println!("  Modification ID: {:?}", mod_result.modification_id);
    
    Ok(())
}

fn demonstrate_risk_management(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating risk management controls...");
    
    // Try to submit orders that exceed various limits
    println!("Testing risk limits:");
    
    // Test order count limit
    for i in 5..15 {
        let order = Order::new_limit(
            OrderId::new(i),
            order_manager.order_book.symbol.clone(),
            Side::Buy,
            (1900 - i * 10) * PRICE_SCALE,
            5 * VOLUME_SCALE,
            get_timestamp(),
        );
        
        let result = order_manager.submit_order(order)?;
        
        if matches!(result.status, hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk) {
            println!("✗ Order #{}: Risk limit exceeded - {}", 
                     result.order_id.as_u64(),
                     result.message.unwrap_or_default());
            break;
        } else {
            println!("✓ Order #{}: Accepted", result.order_id.as_u64());
        }
    }
    
    println!("Final active orders: {}", order_manager.active_order_count);
    
    Ok(())
}

fn demonstrate_order_expiration(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating order expiration...");
    
    // Create a GTD order that expires soon
    let expiration_time = get_timestamp() + 1_000_000; // 1 millisecond from now
    
    let mut gtd_order = Order::new_limit(
        OrderId::new(100),
        order_manager.order_book.symbol.clone(),
        Side::Sell,
        2200 * PRICE_SCALE,
        5 * VOLUME_SCALE,
        get_timestamp(),
    );
    gtd_order.time_in_force = TimeInForce::GTD(expiration_time);
    
    let result = order_manager.submit_order(gtd_order)?;
    println!("✓ Submitted GTD Order #{} (expires in 1ms)", result.order_id.as_u64());
    
    // Wait for expiration
    thread::sleep(Duration::from_millis(2));
    
    // Handle expired orders
    let expired_results = order_manager.handle_expired_orders()?;
    
    if !expired_results.is_empty() {
        for result in expired_results {
            println!("✓ Expired Order #{} - Reason: {:?}", 
                     result.order_id.as_u64(),
                     result.reason);
        }
    } else {
        println!("No orders expired");
    }
    
    Ok(())
}

fn demonstrate_position_tracking(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating position tracking...");
    
    // Simulate some trades to update position
    let trade1 = hf_quoting_liquidity_clob::orderbook::Trade::new(
        1,
        order_manager.order_book.symbol.clone(),
        2000 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        get_timestamp(),
        OrderId::new(2), // Our sell order
        OrderId::new(999), // External buyer
        true, // We're the maker
        1,
    );
    
    // Add our order to the book for position tracking
    let our_order = Order::new_limit(
        OrderId::new(2),
        order_manager.order_book.symbol.clone(),
        Side::Sell,
        2000 * PRICE_SCALE,
        10 * VOLUME_SCALE,
        get_timestamp(),
    );
    order_manager.order_book.orders.insert(OrderId::new(2), our_order);
    
    order_manager.update_position(&trade1);
    
    println!("Position after trade:");
    println!("  Size: {:.2} ETH", order_manager.position.size as f64 / VOLUME_SCALE as f64);
    println!("  Avg Price: ${:.2}", order_manager.position.avg_price as f64 / PRICE_SCALE as f64);
    println!("  Realized PnL: ${:.2}", order_manager.position.realized_pnl as f64 / PRICE_SCALE as f64);
    println!("  Total Volume: {:.2} ETH", order_manager.position.total_volume as f64 / VOLUME_SCALE as f64);
    println!("  Trade Count: {}", order_manager.position.trade_count);
    
    // Update unrealized PnL with current market price
    order_manager.update_unrealized_pnl(2050 * PRICE_SCALE);
    println!("  Unrealized PnL (at $2050): ${:.2}", order_manager.position.unrealized_pnl as f64 / PRICE_SCALE as f64);
    
    Ok(())
}

fn demonstrate_performance_monitoring(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating performance monitoring...");
    
    // Simulate some processing latencies
    order_manager.update_performance_metrics(500);  // 0.5 microseconds
    order_manager.update_performance_metrics(1200); // 1.2 microseconds
    order_manager.update_performance_metrics(800);  // 0.8 microseconds
    
    println!("Performance Metrics:");
    println!("  Avg Processing Latency: {:.2} μs", 
             order_manager.performance_metrics.avg_processing_latency_ns as f64 / 1000.0);
    println!("  Max Processing Latency: {:.2} μs", 
             order_manager.performance_metrics.max_processing_latency_ns as f64 / 1000.0);
    println!("  Orders Per Second: {:.2}", order_manager.performance_metrics.orders_per_second);
    println!("  Fill Rate: {:.2}%", order_manager.performance_metrics.fill_rate);
    
    Ok(())
}

fn demonstrate_emergency_controls(order_manager: &mut OrderManager) -> Result<(), Box<dyn std::error::Error>> {
    println!("Demonstrating emergency controls...");
    
    println!("Active orders before emergency halt: {}", order_manager.active_order_count);
    
    // Trigger emergency halt
    let halt_results = order_manager.emergency_halt()?;
    
    println!("✓ Emergency halt activated!");
    println!("  Cancelled {} orders", halt_results.len());
    println!("  Active orders after halt: {}", order_manager.active_order_count);
    
    // Try to submit a new order (should be rejected)
    let emergency_order = Order::new_limit(
        OrderId::new(200),
        order_manager.order_book.symbol.clone(),
        Side::Buy,
        2000 * PRICE_SCALE,
        1 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(emergency_order)?;
    println!("✗ New order during halt: {} - {}", 
             match result.status {
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted => "Accepted",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::PartiallyFilled => "Partially Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Filled => "Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected => "Rejected",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk => "Rejected (Risk)",
             },
             result.message.unwrap_or_default());
    
    // Resume operations
    order_manager.resume_operations(None);
    println!("✓ Operations resumed");
    
    // Try to submit order again (should work now)
    let resume_order = Order::new_limit(
        OrderId::new(201),
        order_manager.order_book.symbol.clone(),
        Side::Buy,
        2000 * PRICE_SCALE,
        1 * VOLUME_SCALE,
        get_timestamp(),
    );
    
    let result = order_manager.submit_order(resume_order)?;
    println!("✓ New order after resume: {}", 
             match result.status {
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Accepted => "Accepted",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::PartiallyFilled => "Partially Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Filled => "Filled",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::Rejected => "Rejected",
                 hf_quoting_liquidity_clob::orderbook::OrderSubmissionStatus::RejectedRisk => "Rejected (Risk)",
             });
    
    Ok(())
}

fn demonstrate_statistics(order_manager: &OrderManager) {
    println!("Comprehensive Order Manager Statistics:");
    
    let stats = order_manager.get_statistics();
    
    println!("\nOrder Book Statistics:");
    println!("  Symbol: {}", stats.order_book_stats.symbol);
    println!("  Total Orders in Book: {}", stats.order_book_stats.total_orders);
    println!("  Best Bid: ${:.2}", stats.order_book_stats.best_bid.map_or(0.0, |p| p as f64 / PRICE_SCALE as f64));
    println!("  Best Ask: ${:.2}", stats.order_book_stats.best_ask.map_or(0.0, |p| p as f64 / PRICE_SCALE as f64));
    
    println!("\nOrder Management Statistics:");
    println!("  Total Orders Processed: {}", stats.total_orders_processed);
    println!("  Active Orders: {}", stats.active_order_count);
    println!("  Total Events: {}", stats.total_events);
    
    println!("\nOrder State Breakdown:");
    println!("  Pending: {}", stats.state_counts.pending);
    println!("  Active: {}", stats.state_counts.active);
    println!("  Partially Filled: {}", stats.state_counts.partially_filled);
    println!("  Filled: {}", stats.state_counts.filled);
    println!("  Cancelled: {}", stats.state_counts.cancelled);
    println!("  Rejected: {}", stats.state_counts.rejected);
    println!("  Expired: {}", stats.state_counts.expired);
    println!("  Modifying: {}", stats.state_counts.modifying);
    
    println!("\nPosition Information:");
    println!("  Current Position: {:.2} ETH", stats.position.size as f64 / VOLUME_SCALE as f64);
    println!("  Average Price: ${:.2}", stats.position.avg_price as f64 / PRICE_SCALE as f64);
    println!("  Realized PnL: ${:.2}", stats.position.realized_pnl as f64 / PRICE_SCALE as f64);
    println!("  Unrealized PnL: ${:.2}", stats.position.unrealized_pnl as f64 / PRICE_SCALE as f64);
    println!("  Total Volume Traded: {:.2} ETH", stats.position.total_volume as f64 / VOLUME_SCALE as f64);
    println!("  Trade Count: {}", stats.position.trade_count);
    
    println!("\nRisk Limits:");
    println!("  Max Order Size: {:.2} ETH", stats.risk_limits.max_order_size as f64 / VOLUME_SCALE as f64);
    println!("  Max Orders Per Symbol: {}", stats.risk_limits.max_orders_per_symbol);
    println!("  Max Total Exposure: ${:.0}", stats.risk_limits.max_total_exposure as f64 / PRICE_SCALE as f64);
    println!("  Max Long Position: {:.2} ETH", stats.risk_limits.max_long_position as f64 / VOLUME_SCALE as f64);
    println!("  Max Short Position: {:.2} ETH", stats.risk_limits.max_short_position as f64 / VOLUME_SCALE as f64);
    
    println!("\nPerformance Metrics:");
    println!("  Avg Processing Latency: {:.2} μs", stats.performance_metrics.avg_processing_latency_ns as f64 / 1000.0);
    println!("  Max Processing Latency: {:.2} μs", stats.performance_metrics.max_processing_latency_ns as f64 / 1000.0);
    println!("  Orders Per Second: {:.2}", stats.performance_metrics.orders_per_second);
    println!("  Fill Rate: {:.2}%", stats.performance_metrics.fill_rate);
    println!("  Risk Check Failures: {}", stats.performance_metrics.risk_check_failures);
    println!("  Validation Failures: {}", stats.performance_metrics.validation_failures);
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}