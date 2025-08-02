//! Dynamic Parameter Adjustment Demo
//!
//! This example demonstrates the dynamic parameter adjustment capabilities
//! of the Avellaneda-Stoikov market making engine, including:
//! - Real-time volatility estimation using realized volatility
//! - Adaptive risk aversion based on market conditions
//! - Time-to-maturity effects with urgency factor modeling
//! - Parameter validation and stability checks

use std::collections::HashMap;
use trading_system::models::avellaneda_stoikov::{
    AvellanedaStoikovEngine, AvellanedaStoikovParams, DynamicParameterConfig,
    MarketState, MarketRegime, RealizedVolatilityEstimator,
};
use trading_system::math::FixedPoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Avellaneda-Stoikov Dynamic Parameter Adjustment Demo ===\n");

    // Create base parameters
    let mut params = AvellanedaStoikovParams::default();
    params.gamma = FixedPoint::from_float(0.1);
    params.sigma = FixedPoint::from_float(0.2);
    params.k = FixedPoint::from_float(1.5);
    params.T = FixedPoint::from_float(3600.0); // 1 hour

    // Configure dynamic parameter adjustment
    let mut dynamic_config = DynamicParameterConfig::default();
    dynamic_config.adaptive_risk_aversion = true;
    dynamic_config.urgency_factor_enabled = true;
    dynamic_config.stability_checks_enabled = true;
    dynamic_config.base_gamma = FixedPoint::from_float(0.1);
    dynamic_config.gamma_range = (FixedPoint::from_float(0.01), FixedPoint::from_float(2.0));
    dynamic_config.urgency_alpha = FixedPoint::from_float(0.7);

    println!("1. Creating Avellaneda-Stoikov engine with dynamic parameters...");
    let mut engine = AvellanedaStoikovEngine::new_with_dynamic_config(params, dynamic_config)?;
    
    println!("   Base gamma: {:.4}", engine.get_parameters().gamma.to_float());
    println!("   Dynamic features enabled: adaptive risk aversion, urgency factor, stability checks");
    println!();

    // Demonstrate real-time volatility estimation
    println!("2. Real-time volatility estimation...");
    demonstrate_volatility_estimation(&mut engine)?;
    println!();

    // Demonstrate adaptive risk aversion
    println!("3. Adaptive risk aversion based on market conditions...");
    demonstrate_adaptive_risk_aversion(&mut engine)?;
    println!();

    // Demonstrate urgency factor modeling
    println!("4. Time-to-maturity urgency factor modeling...");
    demonstrate_urgency_factor(&engine)?;
    println!();

    // Demonstrate parameter stability validation
    println!("5. Parameter stability validation...");
    demonstrate_parameter_stability(&mut engine)?;
    println!();

    // Demonstrate dynamic quote calculation
    println!("6. Dynamic quote calculation under different scenarios...");
    demonstrate_dynamic_quotes(&mut engine)?;
    println!();

    // Show parameter statistics
    println!("7. Parameter adjustment statistics...");
    show_parameter_statistics(&engine);

    println!("\n=== Demo completed successfully ===");
    Ok(())
}

fn demonstrate_volatility_estimation(engine: &mut AvellanedaStoikovEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Simulating price movements and volatility estimation...");
    
    // Simulate price series with different volatility regimes
    let price_scenarios = vec![
        ("Normal market", vec![100.0, 100.1, 99.9, 100.05, 99.95, 100.02, 99.98]),
        ("High volatility", vec![100.0, 102.0, 98.5, 103.2, 96.8, 104.1, 95.2]),
        ("Crisis conditions", vec![100.0, 95.0, 105.0, 90.0, 110.0, 85.0, 115.0]),
    ];

    for (scenario_name, prices) in price_scenarios {
        println!("   Scenario: {}", scenario_name);
        
        let mut timestamp = 1000000000u64;
        let mut volatilities = Vec::new();
        
        for price in prices {
            let vol = engine.update_volatility(FixedPoint::from_float(price), timestamp);
            volatilities.push(vol.to_float());
            timestamp += 1000000000; // 1 second intervals
        }
        
        let regime = engine.volatility_estimator.get_market_regime();
        let final_vol = volatilities.last().unwrap();
        
        println!("     Final volatility: {:.4}", final_vol);
        println!("     Detected regime: {:?}", regime);
        
        // Test volatility forecasting
        let forecast_5min = engine.volatility_estimator.forecast_volatility(300);
        let forecast_1hour = engine.volatility_estimator.forecast_volatility(3600);
        
        println!("     5-minute forecast: {:.4}", forecast_5min.to_float());
        println!("     1-hour forecast: {:.4}", forecast_1hour.to_float());
        println!();
    }

    Ok(())
}

fn demonstrate_adaptive_risk_aversion(engine: &mut AvellanedaStoikovEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing adaptive risk aversion under different market conditions...");
    
    let market_scenarios = vec![
        ("Normal conditions", MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: 1000,
            ask_volume: 1000,
            volatility: FixedPoint::from_float(0.2),
            order_flow_imbalance: FixedPoint::from_float(0.1),
            ..MarketState::default()
        }),
        ("High volatility", MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.8),
            ask_price: FixedPoint::from_float(100.2),
            bid_volume: 500,
            ask_volume: 500,
            volatility: FixedPoint::from_float(0.5),
            order_flow_imbalance: FixedPoint::from_float(0.3),
            ..MarketState::default()
        }),
        ("Crisis conditions", MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.5),
            ask_price: FixedPoint::from_float(100.5),
            bid_volume: 200,
            ask_volume: 200,
            volatility: FixedPoint::from_float(0.8),
            order_flow_imbalance: FixedPoint::from_float(0.6),
            ..MarketState::default()
        }),
    ];

    let time_horizons = vec![
        ("5 minutes", FixedPoint::from_float(300.0)),
        ("1 hour", FixedPoint::from_float(3600.0)),
        ("End of day", FixedPoint::from_float(21600.0)),
    ];

    for (scenario_name, market_state) in market_scenarios {
        println!("   Scenario: {}", scenario_name);
        engine.update_market_conditions(&market_state);
        
        for (time_name, time_to_maturity) in &time_horizons {
            let adaptive_gamma = engine.calculate_adaptive_risk_aversion(*time_to_maturity);
            let base_gamma = engine.get_parameters().gamma;
            let adjustment_ratio = adaptive_gamma / base_gamma;
            
            println!("     {} - Gamma: {:.4} ({}x base)", 
                time_name, 
                adaptive_gamma.to_float(), 
                adjustment_ratio.to_float()
            );
        }
        
        let conditions = engine.get_market_conditions();
        println!("     Liquidity stress: {:.3}", conditions.liquidity_stress.to_float());
        println!("     Spread tightness: {:.3}", conditions.spread_tightness.to_float());
        println!();
    }

    Ok(())
}

fn demonstrate_urgency_factor(engine: &AvellanedaStoikovEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing urgency factor for different time horizons...");
    
    let time_horizons = vec![
        ("10 hours", FixedPoint::from_float(36000.0)),
        ("1 hour", FixedPoint::from_float(3600.0)),
        ("10 minutes", FixedPoint::from_float(600.0)),
        ("1 minute", FixedPoint::from_float(60.0)),
        ("10 seconds", FixedPoint::from_float(10.0)),
        ("1 second", FixedPoint::from_float(1.0)),
        ("0.1 seconds", FixedPoint::from_float(0.1)),
    ];

    for (time_name, time_to_maturity) in time_horizons {
        let urgency_factor = engine.calculate_urgency_factor(time_to_maturity);
        let update_frequency = engine.calculate_update_frequency(time_to_maturity);
        
        println!("     {} - Urgency: {:.3}x, Update freq: {} ns", 
            time_name, 
            urgency_factor.to_float(),
            update_frequency
        );
    }

    Ok(())
}

fn demonstrate_parameter_stability(engine: &mut AvellanedaStoikovEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing parameter stability validation...");
    
    // Test normal stability
    match engine.validate_parameter_stability() {
        Ok(()) => println!("     ✓ Current parameters are stable"),
        Err(e) => println!("     ✗ Stability issue: {}", e),
    }
    
    // Simulate parameter history with some oscillation
    let mut timestamp = 1000000000u64;
    let gamma_values = vec![0.1, 0.12, 0.09, 0.13, 0.08, 0.14, 0.07, 0.15];
    
    for gamma in gamma_values {
        engine.update_effective_gamma(FixedPoint::from_float(gamma), timestamp);
        timestamp += 1000000000;
    }
    
    // Test stability after oscillations
    match engine.validate_parameter_stability() {
        Ok(()) => println!("     ✓ Parameters remain stable after adjustments"),
        Err(e) => println!("     ⚠ Stability warning: {}", e),
    }
    
    // Test extreme gamma (should fail)
    engine.effective_gamma = FixedPoint::from_float(50.0);
    match engine.validate_parameter_stability() {
        Ok(()) => println!("     ✗ Should have detected extreme gamma"),
        Err(e) => println!("     ✓ Correctly detected extreme parameter: {}", e),
    }
    
    // Reset to normal
    engine.effective_gamma = FixedPoint::from_float(0.1);
    
    Ok(())
}

fn demonstrate_dynamic_quotes(engine: &mut AvellanedaStoikovEngine) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Calculating quotes under different dynamic scenarios...");
    
    let scenarios = vec![
        ("Normal trading", MarketState::default(), 100, FixedPoint::from_float(3600.0)),
        ("High volatility", MarketState {
            volatility: FixedPoint::from_float(0.5),
            order_flow_imbalance: FixedPoint::from_float(0.3),
            ..MarketState::default()
        }, 500, FixedPoint::from_float(3600.0)),
        ("Urgent execution", MarketState::default(), 200, FixedPoint::from_float(10.0)),
        ("Large inventory", MarketState::default(), 2000, FixedPoint::from_float(1800.0)),
    ];

    for (scenario_name, market_state, inventory, time_to_maturity) in scenarios {
        println!("   Scenario: {}", scenario_name);
        
        let quotes = engine.calculate_optimal_quotes(
            FixedPoint::from_float(100.0),
            inventory,
            FixedPoint::from_float(0.2),
            time_to_maturity,
            &market_state,
        )?;
        
        println!("     Bid: {:.4}, Ask: {:.4}", 
            quotes.bid_price.to_float(), 
            quotes.ask_price.to_float()
        );
        println!("     Spread: {:.4}, Reservation: {:.4}", 
            quotes.optimal_spread.to_float(), 
            quotes.reservation_price.to_float()
        );
        println!("     Inventory skew: {:.4}, Confidence: {:.3}", 
            quotes.inventory_skew.to_float(), 
            quotes.confidence.to_float()
        );
        println!("     Market impact adj: {:.4}, Adverse selection: {:.4}", 
            quotes.market_impact_adjustment.to_float(),
            quotes.adverse_selection_premium.to_float()
        );
        println!();
    }

    Ok(())
}

fn show_parameter_statistics(engine: &AvellanedaStoikovEngine) {
    println!("   Current parameter statistics:");
    
    let stats = engine.get_parameter_statistics();
    
    for (key, value) in stats {
        println!("     {}: {:.6}", key, value);
    }
    
    let (calculations, hits, misses, hit_rate) = engine.get_performance_metrics();
    println!("   Performance metrics:");
    println!("     Calculations: {}", calculations);
    println!("     Cache hits: {} ({:.1}%)", hits, hit_rate * 100.0);
    println!("     Cache misses: {}", misses);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_parameter_demo() {
        // Test that the demo runs without panicking
        assert!(main().is_ok());
    }
}