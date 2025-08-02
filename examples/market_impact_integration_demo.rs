//! Market Impact Integration Demo
//!
//! This example demonstrates the market impact integration in the Avellaneda-Stoikov
//! market making model, showing how temporary and permanent impact functions are
//! incorporated into optimal quote calculation.

use advanced_trading_system::models::avellaneda_stoikov::{
    AvellanedaStoikovEngine, AvellanedaStoikovParams, MarketImpactParams, MarketState,
};
use advanced_trading_system::math::FixedPoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Market Impact Integration Demo ===\n");

    // Create market impact parameters
    let impact_params = MarketImpactParams {
        eta: FixedPoint::from_float(0.1),      // Temporary impact coefficient
        alpha: FixedPoint::from_float(0.5),    // Impact exponent (square-root law)
        lambda: FixedPoint::from_float(0.01),  // Permanent impact coefficient
        cross_impact_coeff: FixedPoint::from_float(0.05),
        decay_rate: FixedPoint::from_float(0.1),
        min_participation_rate: FixedPoint::from_float(0.01),
        max_participation_rate: FixedPoint::from_float(0.3),
    };

    println!("Market Impact Parameters:");
    println!("  Temporary impact coefficient (η): {:.4}", impact_params.eta.to_float());
    println!("  Impact exponent (α): {:.4}", impact_params.alpha.to_float());
    println!("  Permanent impact coefficient (λ): {:.4}", impact_params.lambda.to_float());
    println!("  Cross-impact coefficient: {:.4}", impact_params.cross_impact_coeff.to_float());
    println!();

    // Create Avellaneda-Stoikov engine with market impact
    let model_params = AvellanedaStoikovParams {
        gamma: FixedPoint::from_float(0.1),    // Risk aversion
        sigma: FixedPoint::from_float(0.2),    // Volatility
        k: FixedPoint::from_float(1.5),        // Market depth parameter
        A: FixedPoint::from_float(140.0),      // Order arrival rate
        T: FixedPoint::from_float(1.0),        // Time horizon
        min_spread: FixedPoint::from_float(0.001),
        max_spread: FixedPoint::from_float(0.1),
        tick_size: FixedPoint::from_float(0.01),
    };

    let mut engine = AvellanedaStoikovEngine::new_with_impact_params(
        model_params,
        impact_params,
    )?;

    println!("Avellaneda-Stoikov Model Parameters:");
    println!("  Risk aversion (γ): {:.4}", engine.get_parameters().gamma.to_float());
    println!("  Volatility (σ): {:.4}", engine.get_parameters().sigma.to_float());
    println!("  Market depth (k): {:.4}", engine.get_parameters().k.to_float());
    println!();

    // Create market state
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_volume: 1000,
        ask_volume: 1000,
        last_trade_price: FixedPoint::from_float(100.0),
        last_trade_volume: 100,
        timestamp: 1000000000,
        sequence_number: 1,
        volatility: FixedPoint::from_float(0.2),
        order_flow_imbalance: FixedPoint::from_float(0.1),
        microstructure_noise: FixedPoint::from_float(0.001),
    };

    println!("Market State:");
    println!("  Mid price: ${:.2}", market_state.mid_price.to_float());
    println!("  Bid-ask spread: ${:.4}", (market_state.ask_price - market_state.bid_price).to_float());
    println!("  Order flow imbalance: {:.3}", market_state.order_flow_imbalance.to_float());
    println!();

    // Demonstrate impact calculations for different participation rates
    println!("=== Market Impact Analysis ===");
    let participation_rates = [0.05, 0.1, 0.15, 0.2, 0.25];
    let quantity = FixedPoint::from_float(10000.0);

    println!("Participation Rate | Temporary Impact | Permanent Impact | Combined Impact");
    println!("------------------|------------------|------------------|----------------");

    for &rate in &participation_rates {
        let participation_rate = FixedPoint::from_float(rate);
        let impact_params = engine.get_market_impact_parameters();
        
        let temp_impact = impact_params.eta * participation_rate.powf(impact_params.alpha.to_float());
        let perm_impact = impact_params.lambda * quantity;
        let combined_impact = temp_impact + perm_impact;

        println!("      {:.2}%       |      {:.4}      |      {:.2}      |     {:.4}",
            rate * 100.0,
            temp_impact.to_float(),
            perm_impact.to_float(),
            combined_impact.to_float()
        );
    }
    println!();

    // Demonstrate transaction cost analysis
    println!("=== Transaction Cost Analysis ===");
    let total_quantity = FixedPoint::from_float(50000.0);
    let time_horizon = FixedPoint::from_float(1.0); // 1 hour
    let volatility = FixedPoint::from_float(0.2);

    let cost_analysis = engine.analyze_transaction_costs(
        total_quantity,
        time_horizon,
        volatility,
    )?;

    println!("Order Size: {:.0} shares", total_quantity.to_float());
    println!("Time Horizon: {:.1} hours", time_horizon.to_float());
    println!("Volatility: {:.1}%", volatility.to_float() * 100.0);
    println!();
    println!("Optimal Execution Strategy:");
    println!("  Optimal participation rate: {:.2}%", cost_analysis.optimal_participation_rate.to_float() * 100.0);
    println!("  Expected execution time: {:.2} hours", cost_analysis.expected_execution_time.to_float());
    println!();
    println!("Cost Breakdown:");
    println!("  Temporary impact cost: ${:.2}", cost_analysis.temporary_impact_cost.to_float());
    println!("  Permanent impact cost: ${:.2}", cost_analysis.permanent_impact_cost.to_float());
    println!("  Total impact cost: ${:.2}", cost_analysis.total_impact_cost.to_float());
    println!("  Risk-adjusted cost: ${:.2}", cost_analysis.risk_adjusted_cost.to_float());
    println!();

    // Demonstrate quote calculation with different inventory levels
    println!("=== Quote Calculation with Market Impact ===");
    let inventory_levels = [-1000, -500, 0, 500, 1000];
    let volatility = FixedPoint::from_float(0.2);
    let time_to_maturity = FixedPoint::from_float(0.1); // 6 minutes

    println!("Inventory | Bid Price | Ask Price | Spread | Impact Adj | AS Premium");
    println!("----------|-----------|-----------|--------|------------|------------");

    for &inventory in &inventory_levels {
        let quotes = engine.calculate_optimal_quotes(
            market_state.mid_price,
            inventory,
            volatility,
            time_to_maturity,
            &market_state,
        )?;

        println!("   {:5}   |  ${:.3}   |  ${:.3}   | ${:.4} |   ${:.4}   |   ${:.4}",
            inventory,
            quotes.bid_price.to_float(),
            quotes.ask_price.to_float(),
            (quotes.ask_price - quotes.bid_price).to_float(),
            quotes.market_impact_adjustment.to_float(),
            quotes.adverse_selection_premium.to_float()
        );
    }
    println!();

    // Demonstrate inventory updates and their effect on quotes
    println!("=== Inventory Management Demo ===");
    println!("Initial inventory: {}", engine.get_inventory().position);

    // Simulate some trades
    let trades = [
        (200, 100.1),   // Buy 200 at $100.10
        (300, 100.2),   // Buy 300 at $100.20
        (-150, 100.15), // Sell 150 at $100.15
    ];

    for (i, &(quantity, price)) in trades.iter().enumerate() {
        engine.update_inventory(quantity, FixedPoint::from_float(price), 1000000000 + i as u64);
        
        let inventory = engine.get_inventory();
        println!("After trade {}: Position = {}, Avg Price = ${:.3}, P&L = ${:.2}",
            i + 1,
            inventory.position,
            inventory.average_price.to_float(),
            inventory.unrealized_pnl.to_float()
        );

        // Calculate new quotes with updated inventory
        let quotes = engine.calculate_optimal_quotes(
            market_state.mid_price,
            inventory.position,
            volatility,
            time_to_maturity,
            &market_state,
        )?;

        println!("  New quotes: Bid ${:.3}, Ask ${:.3}, Skew: {:.4}",
            quotes.bid_price.to_float(),
            quotes.ask_price.to_float(),
            quotes.inventory_skew.to_float()
        );
        println!();
    }

    // Demonstrate performance metrics
    println!("=== Performance Metrics ===");
    let (calculations, hits, misses, hit_rate) = engine.get_performance_metrics();
    println!("Quote calculations: {}", calculations);
    println!("Cache hits: {}", hits);
    println!("Cache misses: {}", misses);
    println!("Cache hit rate: {:.2}%", hit_rate * 100.0);
    println!();

    // Demonstrate parameter updates
    println!("=== Dynamic Parameter Updates ===");
    let new_impact_params = MarketImpactParams {
        eta: FixedPoint::from_float(0.15),     // Increased temporary impact
        alpha: FixedPoint::from_float(0.6),    // Higher impact exponent
        lambda: FixedPoint::from_float(0.015), // Increased permanent impact
        ..impact_params
    };

    engine.update_market_impact_parameters(new_impact_params)?;
    println!("Updated market impact parameters");

    let updated_quotes = engine.calculate_optimal_quotes(
        market_state.mid_price,
        engine.get_inventory().position,
        volatility,
        time_to_maturity,
        &market_state,
    )?;

    println!("Quotes with updated parameters:");
    println!("  Bid: ${:.3}, Ask: ${:.3}", 
        updated_quotes.bid_price.to_float(),
        updated_quotes.ask_price.to_float()
    );
    println!("  Market impact adjustment: ${:.4}", 
        updated_quotes.market_impact_adjustment.to_float()
    );
    println!("  Confidence score: {:.3}", updated_quotes.confidence.to_float());

    println!("\n=== Demo Complete ===");
    Ok(())
}