//! Cross-venue trading and FX support demonstration
//!
//! This example demonstrates the multi-asset and cross-venue support capabilities
//! including arbitrage detection, smart order routing, and FX hedging.

use hf_quoting_liquidity_clob::orderbook::{
    Order, OrderId, Symbol, Side, OrderType, TimeInForce,
    VenueAdapterFactory, CrossVenueManager, CrossVenueConfig,
    Currency, CurrencyPair, MockFxRateProvider, HedgingStrategy,
    FxHedgingManager, CurrencyRiskManager, CurrencyRiskLimits,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🌐 Cross-Venue Trading and FX Support Demo");
    println!("==========================================\n");
    
    // 1. Set up mock venues with different price characteristics
    println!("📊 Setting up mock trading venues...");
    let venues = VenueAdapterFactory::create_test_venues();
    
    // Display venue information
    for (venue_id, adapter) in &venues {
        let capabilities = adapter.get_capabilities().await?;
        println!("  • Venue: {} - Latency: {}μs - Symbols: {}", 
                venue_id.as_str(), 
                capabilities.latency_estimate_us,
                capabilities.supported_symbols.len());
    }
    
    // 2. Create cross-venue manager
    println!("\n🔄 Initializing cross-venue manager...");
    let config = CrossVenueConfig {
        arbitrage_enabled: true,
        smart_routing_enabled: true,
        min_arbitrage_profit: 5_000, // $0.005 minimum profit
        max_arbitrage_position: 100_000, // 0.1 BTC maximum
        arbitrage_timeout_ms: 5_000,
        routing_timeout_ms: 1_000,
        health_check_interval_ms: 10_000,
        performance_update_interval_ms: 30_000,
    };
    
    let cross_venue_manager = CrossVenueManager::new(venues, config);
    cross_venue_manager.initialize().await?;
    
    // 3. Set up FX support
    println!("💱 Setting up FX rate provider and hedging...");
    let fx_rates = Arc::new(MockFxRateProvider::new());
    
    // Add some crypto/fiat rates
    fx_rates.set_rate(
        CurrencyPair::new(Currency::btc(), Currency::usd()),
        50_000.0,
        10.0 // $10 spread
    );
    fx_rates.set_rate(
        CurrencyPair::new(Currency::eth(), Currency::usd()),
        3_000.0,
        2.0 // $2 spread
    );
    
    // Create FX hedging manager
    let hedging_strategy = HedgingStrategy::Partial { hedge_ratio: 0.8 }; // 80% hedging
    let fx_hedging_manager = FxHedgingManager::new(
        Arc::clone(&fx_rates),
        hedging_strategy,
        Currency::usd(),
    );
    
    // Create currency risk manager
    let risk_limits = CurrencyRiskLimits::default();
    let currency_risk_manager = CurrencyRiskManager::new(fx_hedging_manager, risk_limits);
    
    // 4. Demonstrate smart order routing
    println!("\n🎯 Demonstrating smart order routing...");
    let test_order = Order {
        id: OrderId::new("smart_route_test"),
        symbol: Symbol::new("BTC/USD"),
        side: Side::Buy,
        quantity: 50_000, // 0.05 BTC
        price: 50_000_000_000, // $50,000
        order_type: OrderType::Limit,
        time_in_force: TimeInForce::GTC,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64,
    };
    
    match cross_venue_manager.route_order(test_order.clone()).await {
        Ok(routing_decision) => {
            println!("  ✅ Routing decision generated:");
            println!("     Expected execution price: ${:.2}", 
                    routing_decision.expected_execution_price as f64 / 1_000_000.0);
            println!("     Expected fees: ${:.4}", 
                    routing_decision.expected_fees as f64 / 1_000_000.0);
            println!("     Estimated latency: {}μs", routing_decision.estimated_latency_us);
            println!("     Confidence: {:.1}%", routing_decision.confidence * 100.0);
            
            for (i, allocation) in routing_decision.venue_allocations.iter().enumerate() {
                println!("     Allocation {}: {} - Quantity: {:.6} BTC - Price: ${:.2}", 
                        i + 1,
                        allocation.venue_id.as_str(),
                        allocation.quantity as f64 / 1_000_000.0,
                        allocation.expected_price as f64 / 1_000_000.0);
            }
        }
        Err(e) => println!("  ❌ Routing failed: {}", e),
    }
    
    // 5. Demonstrate arbitrage detection
    println!("\n⚡ Demonstrating arbitrage detection...");
    
    // Start background arbitrage monitoring
    cross_venue_manager.start_background_tasks().await;
    
    // Wait a bit for arbitrage detection to run
    sleep(Duration::from_secs(2)).await;
    
    // Check for detected opportunities
    let active_positions = cross_venue_manager.get_active_arbitrage_positions();
    if !active_positions.is_empty() {
        println!("  ✅ Arbitrage opportunities detected and executed:");
        for (position_id, position) in &active_positions {
            println!("     Position {}: {} -> {}", 
                    position_id,
                    position.opportunity.buy_venue.as_str(),
                    position.opportunity.sell_venue.as_str());
            println!("       Symbol: {}", position.opportunity.symbol.as_str());
            println!("       Buy price: ${:.2}", position.opportunity.buy_price as f64 / 1_000_000.0);
            println!("       Sell price: ${:.2}", position.opportunity.sell_price as f64 / 1_000_000.0);
            println!("       Estimated profit: ${:.4}", position.opportunity.profit_estimate as f64 / 1_000_000.0);
            println!("       Status: {:?}", position.status);
        }
    } else {
        println!("  ℹ️  No arbitrage opportunities detected (prices may be too close)");
    }
    
    // 6. Demonstrate FX hedging
    println!("\n🛡️  Demonstrating FX hedging...");
    
    // Simulate a EUR/USD trade
    let eur_usd_trade = hf_quoting_liquidity_clob::orderbook::Trade {
        id: "eur_usd_trade_1".to_string(),
        buy_order_id: OrderId::new("buy_eur"),
        sell_order_id: OrderId::new("sell_eur"),
        symbol: Symbol::new("EUR/USD"),
        price: 1_085_000, // 1.085 in fixed-point
        quantity: 1_000_000, // 1.0 EUR
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64,
        side: Side::Buy,
    };
    
    let eur_usd_pair = CurrencyPair::new(Currency::eur(), Currency::usd());
    
    // Check if trade passes risk limits
    match currency_risk_manager.check_trade_risk(&eur_usd_trade, &eur_usd_pair) {
        Ok(allowed) => {
            if allowed {
                println!("  ✅ EUR/USD trade passes risk checks");
                
                // Update exposure
                currency_risk_manager.update_exposure(&eur_usd_trade, &eur_usd_pair)?;
                
                // Get hedge orders
                let hedge_orders = currency_risk_manager.fx_hedging_manager
                    .update_position("demo_account", &eur_usd_trade, &eur_usd_pair)?;
                
                if !hedge_orders.is_empty() {
                    println!("  🔄 Hedge orders generated:");
                    for (i, hedge_order) in hedge_orders.iter().enumerate() {
                        println!("     Hedge {}: {} {} {:.6} @ ${:.4}", 
                                i + 1,
                                match hedge_order.side { Side::Buy => "BUY", Side::Sell => "SELL" },
                                hedge_order.symbol.as_str(),
                                hedge_order.quantity as f64 / 1_000_000.0,
                                hedge_order.price as f64 / 1_000_000.0);
                    }
                } else {
                    println!("  ℹ️  No hedge orders required");
                }
            } else {
                println!("  ❌ EUR/USD trade rejected by risk limits");
            }
        }
        Err(e) => println!("  ❌ Risk check failed: {}", e),
    }
    
    // 7. Display current exposures
    println!("\n📈 Current currency exposures:");
    let exposures = currency_risk_manager.get_exposures();
    for (currency, exposure) in &exposures {
        if *exposure != 0 {
            println!("  • {}: {:.6}", currency.as_str(), *exposure as f64 / 1_000_000.0);
        }
    }
    
    // 8. Calculate and display VaR
    println!("\n📊 Risk metrics:");
    match currency_risk_manager.calculate_var(0.95, 1) {
        Ok(var) => println!("  • 1-day VaR (95%): ${:.2}", var as f64 / 1_000_000.0),
        Err(e) => println!("  ❌ VaR calculation failed: {}", e),
    }
    
    // 9. Display performance metrics
    println!("\n📈 Cross-venue performance metrics:");
    let performance = cross_venue_manager.get_performance_metrics();
    println!("  • Total arbitrage opportunities: {}", performance.total_arbitrage_opportunities);
    println!("  • Successful arbitrages: {}", performance.successful_arbitrages);
    println!("  • Total routing decisions: {}", performance.total_routing_decisions);
    println!("  • Successful routes: {}", performance.successful_routes);
    println!("  • Average execution latency: {}μs", performance.average_execution_latency_us);
    
    // Display per-venue performance
    if !performance.venue_performance.is_empty() {
        println!("  • Venue performance:");
        for (venue_id, venue_perf) in &performance.venue_performance {
            println!("    - {}: {:.1}% uptime, {} orders", 
                    venue_id.as_str(),
                    venue_perf.uptime_percentage * 100.0,
                    venue_perf.total_orders);
        }
    }
    
    // 10. Demonstrate carry trade optimization
    println!("\n💰 Demonstrating carry trade optimization...");
    
    // Create a carry-optimized hedging manager
    let carry_strategy = HedgingStrategy::CarryOptimized { target_carry: 0.04 }; // 4% target carry
    let carry_hedging_manager = FxHedgingManager::new(
        Arc::clone(&fx_rates),
        carry_strategy,
        Currency::usd(),
    );
    
    // Simulate positions in different currencies
    let currencies = vec![
        (Currency::aud(), 1_000_000i64), // Long AUD (high carry)
        (Currency::jpy(), -2_000_000i64), // Short JPY (low/negative carry)
        (Currency::eur(), 500_000i64),   // Long EUR (medium carry)
    ];
    
    println!("  Current positions for carry optimization:");
    for (currency, position) in &currencies {
        println!("    • {}: {:.6}", currency.as_str(), *position as f64 / 1_000_000.0);
    }
    
    // This would normally be done through actual trades, but for demo purposes
    // we'll just show what the carry optimization would suggest
    println!("  💡 Carry trade optimization would suggest rebalancing based on interest rate differentials");
    
    // 11. Clean up
    println!("\n🧹 Cleaning up...");
    cross_venue_manager.shutdown().await?;
    
    println!("\n✅ Cross-venue trading and FX support demo completed successfully!");
    println!("\nKey features demonstrated:");
    println!("  • Smart order routing across multiple venues");
    println!("  • Automated arbitrage detection and execution");
    println!("  • Multi-currency position tracking");
    println!("  • FX hedging with configurable strategies");
    println!("  • Currency risk management and limits");
    println!("  • Value at Risk (VaR) calculation");
    println!("  • Carry trade optimization");
    println!("  • Real-time performance monitoring");
    
    Ok(())
}

/// Helper function to format currency amounts
fn format_currency(amount: i64, currency: &str) -> String {
    match currency {
        "JPY" => format!("¥{:.0}", amount as f64 / 1_000_000.0),
        "EUR" => format!("€{:.6}", amount as f64 / 1_000_000.0),
        "GBP" => format!("£{:.6}", amount as f64 / 1_000_000.0),
        "USD" => format!("${:.6}", amount as f64 / 1_000_000.0),
        _ => format!("{:.6} {}", amount as f64 / 1_000_000.0, currency),
    }
}

/// Helper function to simulate market conditions
async fn simulate_market_volatility(fx_rates: &MockFxRateProvider) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Simulate some market volatility by updating rates
    let volatility = rng.gen_range(0.95..1.05); // ±5% volatility
    
    fx_rates.set_rate(
        CurrencyPair::new(Currency::btc(), Currency::usd()),
        50_000.0 * volatility,
        10.0 * volatility
    );
    
    fx_rates.set_rate(
        CurrencyPair::new(Currency::eur(), Currency::usd()),
        1.085 * volatility,
        0.002 * volatility
    );
}