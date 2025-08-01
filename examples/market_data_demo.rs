//! Market Data Generation Demo
//!
//! This example demonstrates the comprehensive market data generation capabilities
//! of the CLOB system, including:
//! - Real-time market depth calculation
//! - Best bid/ask price tracking
//! - Trade tick generation with volume-weighted prices
//! - Market statistics (OHLCV, volume profiles)

use hf_quoting_liquidity_clob::orderbook::{
    CentralLimitOrderBook, Order, OrderId, Symbol, Side, Trade,
    MarketDataGenerator, TradeTick, OHLCV, VolumeProfile,
    PRICE_SCALE, VOLUME_SCALE
};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Market Data Generation Demo ===\n");
    
    // Create a trading symbol
    let symbol = Symbol::new("BTCUSD")?;
    println!("Trading Symbol: {}", symbol);
    
    // Create order book and market data generator
    let mut order_book = CentralLimitOrderBook::new(symbol.clone());
    let mut market_data = MarketDataGenerator::new(symbol.clone());
    
    println!("\n1. Setting up initial order book with liquidity...");
    
    // Add some initial liquidity to the order book
    let initial_orders = vec![
        // Buy orders (bids)
        Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            49900 * PRICE_SCALE, // $49,900
            100 * VOLUME_SCALE,   // 1.0 BTC
            get_timestamp(),
        ),
        Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Buy,
            49800 * PRICE_SCALE, // $49,800
            200 * VOLUME_SCALE,   // 2.0 BTC
            get_timestamp(),
        ),
        Order::new_limit(
            OrderId::new(3),
            symbol.clone(),
            Side::Buy,
            49700 * PRICE_SCALE, // $49,700
            150 * VOLUME_SCALE,   // 1.5 BTC
            get_timestamp(),
        ),
        
        // Sell orders (asks)
        Order::new_limit(
            OrderId::new(4),
            symbol.clone(),
            Side::Sell,
            50100 * PRICE_SCALE, // $50,100
            80 * VOLUME_SCALE,    // 0.8 BTC
            get_timestamp(),
        ),
        Order::new_limit(
            OrderId::new(5),
            symbol.clone(),
            Side::Sell,
            50200 * PRICE_SCALE, // $50,200
            120 * VOLUME_SCALE,   // 1.2 BTC
            get_timestamp(),
        ),
        Order::new_limit(
            OrderId::new(6),
            symbol.clone(),
            Side::Sell,
            50300 * PRICE_SCALE, // $50,300
            90 * VOLUME_SCALE,    // 0.9 BTC
            get_timestamp(),
        ),
    ];
    
    // Add orders to the book
    for order in initial_orders {
        order_book.add_order(order)?;
    }
    
    // Update market data from order book
    market_data.update_from_order_book(&order_book)?;
    
    println!("\n2. Market Depth Analysis:");
    demonstrate_market_depth(&market_data, &order_book);
    
    println!("\n3. Best Bid/Ask Tracking:");
    demonstrate_best_bid_ask(&market_data);
    
    println!("\n4. Simulating Trade Activity:");
    demonstrate_trade_processing(&mut market_data)?;
    
    println!("\n5. Market Statistics (OHLCV):");
    demonstrate_market_statistics(&market_data);
    
    println!("\n6. Volume Profile Analysis:");
    demonstrate_volume_profile(&market_data);
    
    println!("\n7. Advanced Market Data Features:");
    demonstrate_advanced_features(&mut market_data, &mut order_book)?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demonstrate_market_depth(market_data: &MarketDataGenerator, order_book: &CentralLimitOrderBook) {
    let depth = market_data.get_market_depth(order_book, 3);
    
    println!("Market Depth (Top 3 levels):");
    println!("Bids:");
    for (i, (price, size)) in depth.bids.iter().enumerate() {
        println!("  Level {}: ${:.2} - {:.6} BTC", 
                 i + 1, 
                 *price as f64 / PRICE_SCALE as f64,
                 *size as f64 / VOLUME_SCALE as f64);
    }
    
    println!("Asks:");
    for (i, (price, size)) in depth.asks.iter().enumerate() {
        println!("  Level {}: ${:.2} - {:.6} BTC", 
                 i + 1, 
                 *price as f64 / PRICE_SCALE as f64,
                 *size as f64 / VOLUME_SCALE as f64);
    }
    
    if let Some(spread) = depth.spread() {
        println!("Spread: ${:.2}", spread as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(mid) = depth.mid_price() {
        println!("Mid Price: ${:.2}", mid as f64 / PRICE_SCALE as f64);
    }
}

fn demonstrate_best_bid_ask(market_data: &MarketDataGenerator) {
    let (best_bid, best_ask) = market_data.get_best_bid_ask();
    
    println!("Current Best Prices:");
    if let Some(bid) = best_bid {
        println!("  Best Bid: ${:.2}", bid as f64 / PRICE_SCALE as f64);
    }
    if let Some(ask) = best_ask {
        println!("  Best Ask: ${:.2}", ask as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(spread) = market_data.get_spread() {
        println!("  Spread: ${:.2}", spread as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(mid) = market_data.get_mid_price() {
        println!("  Mid Price: ${:.2}", mid as f64 / PRICE_SCALE as f64);
    }
}

fn demonstrate_trade_processing(market_data: &mut MarketDataGenerator) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing simulated trades...");
    
    // Simulate a series of trades
    let trades = vec![
        create_trade(1, 50000 * PRICE_SCALE, 50 * VOLUME_SCALE, get_timestamp()),
        create_trade(2, 50050 * PRICE_SCALE, 75 * VOLUME_SCALE, get_timestamp() + 1000),
        create_trade(3, 49950 * PRICE_SCALE, 100 * VOLUME_SCALE, get_timestamp() + 2000),
        create_trade(4, 50100 * PRICE_SCALE, 60 * VOLUME_SCALE, get_timestamp() + 3000),
        create_trade(5, 50025 * PRICE_SCALE, 80 * VOLUME_SCALE, get_timestamp() + 4000),
    ];
    
    for (i, trade) in trades.iter().enumerate() {
        let tick = market_data.process_trade(trade)?;
        
        println!("Trade {}: ${:.2} - {:.6} BTC", 
                 i + 1,
                 tick.price as f64 / PRICE_SCALE as f64,
                 tick.size as f64 / VOLUME_SCALE as f64);
        
        println!("  VWAP: ${:.2}", tick.vwap as f64 / PRICE_SCALE as f64);
        println!("  Side: {}", tick.side);
        println!("  Sequence: {}", tick.sequence);
    }
    
    // Show current VWAP
    if let Some(vwap) = market_data.get_current_vwap() {
        println!("\nCurrent VWAP: ${:.2}", vwap as f64 / PRICE_SCALE as f64);
    }
    
    Ok(())
}

fn demonstrate_market_statistics(market_data: &MarketDataGenerator) {
    println!("Market Statistics:");
    
    if let Some(ohlcv) = market_data.get_current_ohlcv() {
        println!("Current OHLCV Bar:");
        println!("  Open: ${:.2}", ohlcv.open as f64 / PRICE_SCALE as f64);
        println!("  High: ${:.2}", ohlcv.high as f64 / PRICE_SCALE as f64);
        println!("  Low: ${:.2}", ohlcv.low as f64 / PRICE_SCALE as f64);
        println!("  Close: ${:.2}", ohlcv.close as f64 / PRICE_SCALE as f64);
        println!("  Volume: {:.6} BTC", ohlcv.volume as f64 / VOLUME_SCALE as f64);
        println!("  VWAP: ${:.2}", ohlcv.vwap as f64 / PRICE_SCALE as f64);
        println!("  Trade Count: {}", ohlcv.trade_count);
    }
    
    let daily_stats = market_data.get_daily_stats();
    println!("\n24h Statistics:");
    if let Some(high) = daily_stats.high_24h {
        println!("  24h High: ${:.2}", high as f64 / PRICE_SCALE as f64);
    }
    if let Some(low) = daily_stats.low_24h {
        println!("  24h Low: ${:.2}", low as f64 / PRICE_SCALE as f64);
    }
    println!("  24h Volume: {:.6} BTC", daily_stats.volume_24h as f64 / VOLUME_SCALE as f64);
    println!("  24h Trade Count: {}", daily_stats.trade_count_24h);
}

fn demonstrate_volume_profile(market_data: &MarketDataGenerator) {
    let profile = market_data.get_volume_profile();
    
    println!("Volume Profile:");
    println!("  Total Volume: {:.6} BTC", profile.total_volume as f64 / VOLUME_SCALE as f64);
    
    if let Some(poc) = profile.poc {
        println!("  Point of Control (POC): ${:.2}", poc as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(vah) = profile.vah {
        println!("  Value Area High (VAH): ${:.2}", vah as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(val) = profile.val {
        println!("  Value Area Low (VAL): ${:.2}", val as f64 / PRICE_SCALE as f64);
    }
    
    // Show top 5 price levels by volume
    let mut price_volumes: Vec<_> = profile.price_volumes.iter().collect();
    price_volumes.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("\nTop 5 Price Levels by Volume:");
    for (i, (price, volume)) in price_volumes.iter().take(5).enumerate() {
        println!("  {}: ${:.2} - {:.6} BTC", 
                 i + 1,
                 **price as f64 / PRICE_SCALE as f64,
                 **volume as f64 / VOLUME_SCALE as f64);
    }
}

fn demonstrate_advanced_features(
    market_data: &mut MarketDataGenerator, 
    order_book: &mut CentralLimitOrderBook
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Advanced Market Data Features:");
    
    // Simulate market order that changes the order book
    let market_buy = Order::new_market(
        OrderId::new(100),
        market_data.symbol.clone(),
        Side::Buy,
        50 * VOLUME_SCALE, // 0.5 BTC
        get_timestamp(),
    );
    
    println!("\nExecuting market buy order for 0.5 BTC...");
    let trades = order_book.add_order(market_buy)?;
    
    // Update market data
    market_data.update_from_order_book(order_book)?;
    
    // Process resulting trades
    for trade in trades {
        let tick = market_data.process_trade(&trade)?;
        println!("Trade executed: ${:.2} - {:.6} BTC", 
                 tick.price as f64 / PRICE_SCALE as f64,
                 tick.size as f64 / VOLUME_SCALE as f64);
    }
    
    // Show updated best bid/ask
    println!("\nUpdated Best Prices:");
    let (best_bid, best_ask) = market_data.get_best_bid_ask();
    if let Some(bid) = best_bid {
        println!("  Best Bid: ${:.2}", bid as f64 / PRICE_SCALE as f64);
    }
    if let Some(ask) = best_ask {
        println!("  Best Ask: ${:.2}", ask as f64 / PRICE_SCALE as f64);
    }
    
    // Demonstrate utility functions
    println!("\nUtility Function Examples:");
    
    // Create some sample trades for utility demonstrations
    let sample_trades = vec![
        create_trade(10, 50000 * PRICE_SCALE, 100 * VOLUME_SCALE, get_timestamp()),
        create_trade(11, 50100 * PRICE_SCALE, 150 * VOLUME_SCALE, get_timestamp()),
        create_trade(12, 49900 * PRICE_SCALE, 200 * VOLUME_SCALE, get_timestamp()),
    ];
    
    if let Some(vwap) = hf_quoting_liquidity_clob::orderbook::market_data::utils::calculate_vwap(&sample_trades) {
        println!("  Sample VWAP: ${:.2}", vwap as f64 / PRICE_SCALE as f64);
    }
    
    if let Some(twap) = hf_quoting_liquidity_clob::orderbook::market_data::utils::calculate_twap(&sample_trades) {
        println!("  Sample TWAP: ${:.2}", twap as f64 / PRICE_SCALE as f64);
    }
    
    let sample_prices = vec![50000, 50100, 49900, 50050, 49950];
    if let Some(volatility) = hf_quoting_liquidity_clob::orderbook::market_data::utils::calculate_volatility(&sample_prices) {
        println!("  Sample Volatility: {:.6}", volatility);
    }
    
    Ok(())
}

fn create_trade(id: u64, price: u64, size: u64, timestamp: u64) -> Trade {
    Trade::new(
        id,
        Symbol::new("BTCUSD").unwrap(),
        price,
        size,
        timestamp,
        OrderId::new(id * 2),
        OrderId::new(id * 2 + 1),
        id % 2 == 0, // Alternate maker/taker
        id,
    )
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}