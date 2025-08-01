//! Central Limit Order Book Engine Demo
//!
//! This example demonstrates the complete functionality of the integrated CLOB engine:
//! - Order submission and matching with price-time priority
//! - Real-time market data generation
//! - Order management with lifecycle tracking
//! - Risk controls and validation
//! - Compressed state for zkVM compatibility

use hf_quoting_liquidity_clob::orderbook::*;
use std::time::{SystemTime, UNIX_EPOCH};

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Central Limit Order Book Engine Demo");
    println!("========================================\n");

    // Create a new CLOB engine for BTC/USD
    let symbol = Symbol::new("BTCUSD")?;
    let mut engine = CLOBEngine::new(symbol.clone(), None)?;
    
    println!("✅ Created CLOB engine for {}", symbol);
    println!("   Engine ID: {}", engine.created_at);
    println!("   Initial state: {} operations processed\n", engine.total_operations);

    // Demo 1: Basic Order Submission
    println!("📝 Demo 1: Basic Order Submission");
    println!("----------------------------------");
    
    let timestamp = get_timestamp();
    
    // Submit a sell order (ask)
    let sell_order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Sell,
        50000 * PRICE_SCALE,  // $50,000
        100 * VOLUME_SCALE,   // 1.00 BTC
        timestamp,
    );
    
    let result = engine.submit_order(sell_order)?;
    println!("   Submitted SELL order: {} BTC @ $50,000", 100 as f64 / VOLUME_SCALE as f64);
    println!("   Status: {:?}", result.status);
    println!("   Trades: {}", result.trades.len());
    
    // Submit a buy order (bid)
    let buy_order = Order::new_limit(
        OrderId::new(2),
        symbol.clone(),
        Side::Buy,
        49500 * PRICE_SCALE,  // $49,500
        50 * VOLUME_SCALE,    // 0.50 BTC
        timestamp + 1000,
    );
    
    let result = engine.submit_order(buy_order)?;
    println!("   Submitted BUY order: {} BTC @ $49,500", 50 as f64 / VOLUME_SCALE as f64);
    println!("   Status: {:?}", result.status);
    
    // Check market data
    let (best_bid, best_ask) = engine.get_best_bid_ask();
    let spread = engine.get_spread();
    let mid_price = engine.get_mid_price();
    
    println!("\n📊 Market Data:");
    println!("   Best Bid: ${}", best_bid.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));
    println!("   Best Ask: ${}", best_ask.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));
    println!("   Spread: ${}", spread.map(|s| s as f64 / PRICE_SCALE as f64).unwrap_or(0.0));
    println!("   Mid Price: ${}", mid_price.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));

    // Demo 2: Order Matching and Trade Execution
    println!("\n🔄 Demo 2: Order Matching and Trade Execution");
    println!("----------------------------------------------");
    
    // Submit a market buy order that will match the sell order
    let market_buy = Order::new_market(
        OrderId::new(3),
        symbol.clone(),
        Side::Buy,
        25 * VOLUME_SCALE,    // 0.25 BTC
        timestamp + 2000,
    );
    
    let result = engine.submit_order(market_buy)?;
    println!("   Submitted MARKET BUY order: {} BTC", 25 as f64 / VOLUME_SCALE as f64);
    println!("   Status: {:?}", result.status);
    println!("   Trades executed: {}", result.trades.len());
    
    if !result.trades.is_empty() {
        let trade = &result.trades[0];
        println!("   Trade details:");
        println!("     Price: ${}", trade.price as f64 / PRICE_SCALE as f64);
        println!("     Size: {} BTC", trade.size as f64 / VOLUME_SCALE as f64);
        println!("     Buyer: #{}", trade.buyer_order_id.as_u64());
        println!("     Seller: #{}", trade.seller_order_id.as_u64());
        println!("     Is Buyer Maker: {}", trade.is_buyer_maker);
    }
    
    // Check VWAP after trade
    let vwap = engine.get_current_vwap();
    println!("   Current VWAP: ${}", vwap.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));

    // Demo 3: Market Depth and Order Book State
    println!("\n📈 Demo 3: Market Depth and Order Book State");
    println!("---------------------------------------------");
    
    let depth = engine.get_market_depth(5);
    println!("   Market Depth (Top 5 levels):");
    println!("   Bids:");
    for (i, (price, size)) in depth.bids.iter().enumerate() {
        println!("     {}: {} BTC @ ${}", i + 1, *size as f64 / VOLUME_SCALE as f64, *price as f64 / PRICE_SCALE as f64);
    }
    println!("   Asks:");
    for (i, (price, size)) in depth.asks.iter().enumerate() {
        println!("     {}: {} BTC @ ${}", i + 1, *size as f64 / VOLUME_SCALE as f64, *price as f64 / PRICE_SCALE as f64);
    }
    
    let active_orders = engine.get_active_orders();
    println!("   Active Orders: {}", active_orders.len());
    for (order_id, order) in active_orders {
        println!("     Order #{}: {} {} BTC @ ${}", 
                 order_id.as_u64(),
                 order.side,
                 order.size as f64 / VOLUME_SCALE as f64,
                 order.price as f64 / PRICE_SCALE as f64);
    }

    // Demo 4: Order Modification
    println!("\n✏️  Demo 4: Order Modification");
    println!("------------------------------");
    
    // Modify the existing buy order (change price and size)
    let modification = OrderModification {
        order_id: OrderId::new(2),
        new_price: Some(49800 * PRICE_SCALE),  // Increase price to $49,800
        new_size: Some(75 * VOLUME_SCALE),     // Increase size to 0.75 BTC
        new_time_in_force: None,
        timestamp: timestamp + 3000,
        modification_id: Some("mod_001".to_string()),
    };
    
    let result = engine.modify_order(modification)?;
    println!("   Modified order #{} -> #{}", 
             result.original_order_id.as_u64(),
             result.new_order_id.as_u64());
    println!("   New order: {} BTC @ ${}", 
             result.new_order.size as f64 / VOLUME_SCALE as f64,
             result.new_order.price as f64 / PRICE_SCALE as f64);
    println!("   Modification status: {:?}", result.submission_result.status);

    // Demo 5: Order Cancellation
    println!("\n❌ Demo 5: Order Cancellation");
    println!("------------------------------");
    
    // Cancel the modified order
    let cancel_result = engine.cancel_order(result.new_order_id, CancellationReason::UserRequested)?;
    println!("   Cancelled order #{}", cancel_result.order_id.as_u64());
    println!("   Reason: {:?}", cancel_result.reason);
    
    // Check order state
    let state = engine.get_order_state(cancel_result.order_id);
    println!("   Order state: {:?}", state);

    // Demo 6: Risk Controls
    println!("\n🛡️  Demo 6: Risk Controls");
    println!("-------------------------");
    
    // Create engine with strict risk limits
    let mut risk_limits = RiskLimits::default();
    risk_limits.max_order_size = 10 * VOLUME_SCALE;  // Max 0.10 BTC per order
    risk_limits.max_order_value = 1000 * PRICE_SCALE; // Max $1,000 per order
    
    let mut risk_engine = CLOBEngine::new(Symbol::new("ETHUSD")?, Some(risk_limits))?;
    
    // Try to submit an order that violates size limit
    let large_order = Order::new_limit(
        OrderId::new(100),
        Symbol::new("ETHUSD")?,
        Side::Buy,
        3000 * PRICE_SCALE,   // $3,000 (violates value limit)
        20 * VOLUME_SCALE,    // 0.20 ETH (violates size limit)
        timestamp + 4000,
    );
    
    let result = risk_engine.submit_order(large_order)?;
    println!("   Submitted order violating risk limits");
    println!("   Status: {:?}", result.status);
    if let Some(message) = result.message {
        println!("   Rejection reason: {}", message);
    }

    // Demo 7: Performance Metrics and Statistics
    println!("\n📊 Demo 7: Performance Metrics and Statistics");
    println!("----------------------------------------------");
    
    let stats = engine.get_engine_stats();
    println!("   Engine Statistics:");
    println!("     Symbol: {}", stats.symbol);
    println!("     Total Operations: {}", stats.total_operations);
    println!("     Active Orders: {}", stats.active_orders);
    println!("     Total Orders Processed: {}", stats.total_orders_processed);
    println!("     Last Trade Price: ${}", stats.last_trade_price.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));
    println!("     Compressed State Size: {} bytes", stats.compressed_state_size);
    
    let daily_stats = engine.get_daily_stats();
    println!("   Daily Statistics:");
    println!("     24h Volume: {} BTC", daily_stats.volume_24h as f64 / VOLUME_SCALE as f64);
    println!("     24h Trade Count: {}", daily_stats.trade_count_24h);
    println!("     24h High: ${}", daily_stats.high_24h.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));
    println!("     24h Low: ${}", daily_stats.low_24h.map(|p| p as f64 / PRICE_SCALE as f64).unwrap_or(0.0));

    // Demo 8: Compressed State for zkVM
    println!("\n🔒 Demo 8: Compressed State for zkVM");
    println!("------------------------------------");
    
    let compressed_state = engine.get_compressed_state();
    println!("   Compressed State:");
    println!("     Symbol ID: {}", compressed_state.symbol_id);
    println!("     Sequence: {}", compressed_state.sequence);
    println!("     Bid Levels: {}", compressed_state.bid_tree.level_count);
    println!("     Ask Levels: {}", compressed_state.ask_tree.level_count);
    println!("     State Root: {:02x?}", &compressed_state.state_root[..8]);
    println!("     Compressed Size: {} bytes", compressed_state.compressed_size());
    
    // Verify state integrity
    let integrity_check = engine.verify_integrity()?;
    println!("   State Integrity Check: {}", if integrity_check { "✅ PASSED" } else { "❌ FAILED" });

    // Demo 9: Emergency Controls
    println!("\n🚨 Demo 9: Emergency Controls");
    println!("-----------------------------");
    
    // Add a few more orders first
    let orders = vec![
        Order::new_limit(OrderId::new(10), symbol.clone(), Side::Buy, 49000 * PRICE_SCALE, 10 * VOLUME_SCALE, timestamp + 5000),
        Order::new_limit(OrderId::new(11), symbol.clone(), Side::Sell, 51000 * PRICE_SCALE, 15 * VOLUME_SCALE, timestamp + 6000),
        Order::new_limit(OrderId::new(12), symbol.clone(), Side::Buy, 48500 * PRICE_SCALE, 5 * VOLUME_SCALE, timestamp + 7000),
    ];
    
    for order in orders {
        engine.submit_order(order)?;
    }
    
    println!("   Added {} additional orders", 3);
    println!("   Active orders before emergency cancel: {}", engine.get_active_orders().len());
    
    // Emergency cancel all orders
    let cancel_results = engine.cancel_all_orders(CancellationReason::EmergencyHalt)?;
    println!("   Emergency cancelled {} orders", cancel_results.len());
    println!("   Active orders after emergency cancel: {}", engine.get_active_orders().len());
    
    // Final market state
    let (final_bid, final_ask) = engine.get_best_bid_ask();
    println!("   Final market state:");
    println!("     Best Bid: {}", if final_bid.is_some() { format!("${}", final_bid.unwrap() as f64 / PRICE_SCALE as f64) } else { "None".to_string() });
    println!("     Best Ask: {}", if final_ask.is_some() { format!("${}", final_ask.unwrap() as f64 / PRICE_SCALE as f64) } else { "None".to_string() });

    println!("\n🎉 Demo completed successfully!");
    println!("   Final engine statistics:");
    let final_stats = engine.get_engine_stats();
    println!("     Total operations: {}", final_stats.total_operations);
    println!("     Total orders processed: {}", final_stats.total_orders_processed);
    println!("     Engine uptime: {} ns", final_stats.last_update - final_stats.created_at);

    Ok(())
}