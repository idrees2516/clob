use hf_quoting_liquidity_clob::orderbook::{
    CentralLimitOrderBook, Order, OrderId, Symbol, Side, Trade, 
    PRICE_SCALE, VOLUME_SCALE
};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Core Order Book Data Structures Demo ===\n");
    
    // Create a new order book for BTCUSD
    let symbol = Symbol::new("BTCUSD")?;
    let mut book = CentralLimitOrderBook::new(symbol.clone());
    
    println!("Created order book for symbol: {}", symbol);
    println!("Initial state: {} orders, {} bid levels, {} ask levels\n", 
             book.total_orders, book.bids.len(), book.asks.len());
    
    // Helper function to get current timestamp
    let timestamp = || SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    
    // Add some limit orders to build the book
    println!("Adding limit orders to build the order book...");
    
    // Add bid orders (buy side)
    let bid_orders = vec![
        Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 50000 * PRICE_SCALE, 100 * VOLUME_SCALE, timestamp()),
        Order::new_limit(OrderId::new(2), symbol.clone(), Side::Buy, 49900 * PRICE_SCALE, 200 * VOLUME_SCALE, timestamp()),
        Order::new_limit(OrderId::new(3), symbol.clone(), Side::Buy, 49800 * PRICE_SCALE, 150 * VOLUME_SCALE, timestamp()),
    ];
    
    // Add ask orders (sell side)
    let ask_orders = vec![
        Order::new_limit(OrderId::new(4), symbol.clone(), Side::Sell, 50100 * PRICE_SCALE, 120 * VOLUME_SCALE, timestamp()),
        Order::new_limit(OrderId::new(5), symbol.clone(), Side::Sell, 50200 * PRICE_SCALE, 180 * VOLUME_SCALE, timestamp()),
        Order::new_limit(OrderId::new(6), symbol.clone(), Side::Sell, 50300 * PRICE_SCALE, 250 * VOLUME_SCALE, timestamp()),
    ];
    
    // Add all orders to the book
    for order in bid_orders.iter().chain(ask_orders.iter()) {
        let trades = book.add_order(order.clone())?;
        println!("Added order #{} ({} {} @ ${:.2}) - {} trades generated", 
                 order.id.as_u64(), 
                 order.side,
                 order.size as f64 / VOLUME_SCALE as f64,
                 order.price as f64 / PRICE_SCALE as f64,
                 trades.len());
    }
    
    println!("\nOrder book state after adding limit orders:");
    print_book_state(&book);
    
    // Demonstrate market depth
    println!("\nMarket Depth (Top 3 levels):");
    let depth = book.get_market_depth(3);
    
    println!("BIDS:");
    for (i, (price, size)) in depth.bids.iter().enumerate() {
        println!("  {}: ${:.2} x {:.6}", i + 1, *price as f64 / PRICE_SCALE as f64, *size as f64 / VOLUME_SCALE as f64);
    }
    
    println!("ASKS:");
    for (i, (price, size)) in depth.asks.iter().enumerate() {
        println!("  {}: ${:.2} x {:.6}", i + 1, *price as f64 / PRICE_SCALE as f64, *size as f64 / VOLUME_SCALE as f64);
    }
    
    // Add a market order that will generate trades
    println!("\n=== Adding Market Order ===");
    let market_buy = Order::new_market(
        OrderId::new(7),
        symbol.clone(),
        Side::Buy,
        200 * VOLUME_SCALE, // Buy 0.2 BTC at market
        timestamp(),
    );
    
    println!("Adding market buy order for {:.6} BTC...", 
             market_buy.size as f64 / VOLUME_SCALE as f64);
    
    let trades = book.add_order(market_buy)?;
    
    println!("Market order generated {} trades:", trades.len());
    for (i, trade) in trades.iter().enumerate() {
        println!("  Trade {}: {:.6} BTC @ ${:.2} (Buyer: #{}, Seller: #{}, Buyer Maker: {})",
                 i + 1,
                 trade.size as f64 / VOLUME_SCALE as f64,
                 trade.price as f64 / PRICE_SCALE as f64,
                 trade.buyer_order_id.as_u64(),
                 trade.seller_order_id.as_u64(),
                 trade.is_buyer_maker);
    }
    
    println!("\nOrder book state after market order:");
    print_book_state(&book);
    
    // Demonstrate order cancellation
    println!("\n=== Order Cancellation ===");
    println!("Cancelling order #2...");
    
    let cancelled_order = book.cancel_order(OrderId::new(2))?;
    println!("Cancelled order: #{} ({} {} @ ${:.2})",
             cancelled_order.id.as_u64(),
             cancelled_order.side,
             cancelled_order.size as f64 / VOLUME_SCALE as f64,
             cancelled_order.price as f64 / PRICE_SCALE as f64);
    
    println!("\nFinal order book state:");
    print_book_state(&book);
    
    // Show comprehensive statistics
    println!("\n=== Order Book Statistics ===");
    let stats = book.get_statistics();
    println!("Symbol: {}", stats.symbol);
    println!("Total Orders: {}", stats.total_orders);
    println!("Bid Levels: {}", stats.bid_levels);
    println!("Ask Levels: {}", stats.ask_levels);
    println!("Total Bid Volume: {:.6} BTC", stats.total_bid_volume as f64 / VOLUME_SCALE as f64);
    println!("Total Ask Volume: {:.6} BTC", stats.total_ask_volume as f64 / VOLUME_SCALE as f64);
    
    if let Some(price) = stats.best_bid {
        println!("Best Bid: ${:.2}", price as f64 / PRICE_SCALE as f64);
    }
    if let Some(price) = stats.best_ask {
        println!("Best Ask: ${:.2}", price as f64 / PRICE_SCALE as f64);
    }
    if let Some(spread) = stats.spread {
        println!("Spread: ${:.2}", spread as f64 / PRICE_SCALE as f64);
    }
    if let Some(mid) = stats.mid_price {
        println!("Mid Price: ${:.2}", mid as f64 / PRICE_SCALE as f64);
    }
    if let Some(price) = stats.last_trade_price {
        println!("Last Trade Price: ${:.2}", price as f64 / PRICE_SCALE as f64);
    }
    
    println!("Sequence Number: {}", stats.sequence_number);
    
    // Demonstrate volume calculation
    println!("\n=== Volume Analysis ===");
    let test_price = 50150 * PRICE_SCALE;
    let buy_volume = book.get_volume_at_or_better(Side::Buy, test_price);
    let sell_volume = book.get_volume_at_or_better(Side::Sell, test_price);
    
    println!("Volume available for buying at or below ${:.2}: {:.6} BTC", 
             test_price as f64 / PRICE_SCALE as f64,
             buy_volume as f64 / VOLUME_SCALE as f64);
    println!("Volume available for selling at or above ${:.2}: {:.6} BTC", 
             test_price as f64 / PRICE_SCALE as f64,
             sell_volume as f64 / VOLUME_SCALE as f64);
    
    // Validate the order book integrity
    println!("\n=== Validation ===");
    match book.validate() {
        Ok(()) => println!("✓ Order book validation passed - all internal consistency checks OK"),
        Err(e) => println!("✗ Order book validation failed: {}", e),
    }
    
    println!("\n=== Demo Complete ===");
    println!("This demo showcased the comprehensive core order book data structures:");
    println!("• CentralLimitOrderBook with price-time priority");
    println!("• PriceLevel with FIFO order queues");
    println!("• Order validation and comprehensive error handling");
    println!("• Trade execution with detailed audit information");
    println!("• Market depth calculation and analysis");
    println!("• Order cancellation and book maintenance");
    println!("• Volume analysis and liquidity assessment");
    println!("• Internal consistency validation");
    println!("• Comprehensive statistics and monitoring");
    
    Ok(())
}

fn print_book_state(book: &CentralLimitOrderBook) {
    println!("  Total Orders: {}", book.total_orders);
    println!("  Bid Levels: {} (Volume: {:.6} BTC)", 
             book.bids.len(), 
             book.total_bid_volume as f64 / VOLUME_SCALE as f64);
    println!("  Ask Levels: {} (Volume: {:.6} BTC)", 
             book.asks.len(), 
             book.total_ask_volume as f64 / VOLUME_SCALE as f64);
    
    if let Some(bid) = book.get_best_bid() {
        println!("  Best Bid: ${:.2}", bid as f64 / PRICE_SCALE as f64);
    }
    if let Some(ask) = book.get_best_ask() {
        println!("  Best Ask: ${:.2}", ask as f64 / PRICE_SCALE as f64);
    }
    if let Some(spread) = book.get_spread() {
        println!("  Spread: ${:.2}", spread as f64 / PRICE_SCALE as f64);
    }
}