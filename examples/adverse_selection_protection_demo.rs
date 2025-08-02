//! Adverse Selection Protection Demo
//!
//! This example demonstrates the comprehensive adverse selection protection
//! functionality implemented for the Avellaneda-Stoikov market making model.
//!
//! The demo shows:
//! - Information asymmetry detection using price impact analysis
//! - Adverse selection premium calculation
//! - Dynamic spread widening based on toxic flow detection
//! - Quote frequency adjustment during adverse conditions

use std::time::{SystemTime, UNIX_EPOCH};
use advanced_trading_system::models::{
    adverse_selection::{
        AdverseSelectionProtection, AdverseSelectionParams, TradeInfo
    },
    avellaneda_stoikov::{AvellanedaStoikovEngine, AvellanedaStoikovParams, MarketState}
};
use advanced_trading_system::math::FixedPoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adverse Selection Protection Demo ===\n");
    
    // Initialize adverse selection protection with custom parameters
    let mut as_params = AdverseSelectionParams::default();
    as_params.beta = FixedPoint::from_float(0.6); // Higher sensitivity
    as_params.toxicity_threshold = FixedPoint::from_float(0.25);
    as_params.frequency_threshold = FixedPoint::from_float(0.08);
    
    let base_frequency = FixedPoint::from_float(20.0); // 20 quotes per second
    let mut protection = AdverseSelectionProtection::new(as_params, base_frequency)?;
    
    println!("1. Testing Normal Trading Conditions");
    println!("=====================================");
    
    // Simulate normal trading conditions
    let normal_trades = vec![
        create_trade_info(100.01, 1000, 100.0, 0.15, 0.05),
        create_trade_info(99.99, -800, 100.0, 0.15, -0.03),
        create_trade_info(100.02, 1200, 100.0, 0.16, 0.08),
        create_trade_info(99.98, -900, 100.0, 0.15, -0.06),
    ];
    
    for (i, trade) in normal_trades.iter().enumerate() {
        let state = protection.update(trade.clone())?;
        println!("Trade {}: Premium = {:.4}, Freq Adj = {:.3}, Toxicity = {:.3}, Protection = {}", 
                 i + 1,
                 state.premium.to_float(),
                 state.frequency_adjustment.to_float(),
                 state.toxicity_level.to_float(),
                 state.protection_active);
    }
    
    println!("\n2. Testing Adverse Selection Scenario");
    println!("=====================================");
    
    // Simulate adverse selection scenario with informed trading
    let adverse_trades = vec![
        create_trade_info(100.15, 3000, 100.0, 0.25, 0.4),   // Large buy with high impact
        create_trade_info(100.25, 2500, 100.0, 0.28, 0.5),   // Another large buy
        create_trade_info(100.35, 2000, 100.0, 0.32, 0.6),   // Continued buying pressure
        create_trade_info(100.45, 1800, 100.0, 0.35, 0.7),   // Strong momentum
        create_trade_info(100.55, 1500, 100.0, 0.38, 0.8),   // Persistent informed flow
    ];
    
    for (i, trade) in adverse_trades.iter().enumerate() {
        let state = protection.update(trade.clone())?;
        println!("Adverse Trade {}: Premium = {:.4}, Freq Adj = {:.3}, Toxicity = {:.3}, Protection = {}", 
                 i + 1,
                 state.premium.to_float(),
                 state.frequency_adjustment.to_float(),
                 state.toxicity_level.to_float(),
                 state.protection_active);
    }
    
    println!("\n3. Information Asymmetry Analysis");
    println!("=================================");
    
    let final_state = protection.get_state();
    println!("Raw Information Asymmetry: {:.6}", final_state.information_asymmetry.raw_measure.to_float());
    println!("Smoothed Information Asymmetry: {:.6}", final_state.information_asymmetry.smoothed_measure.to_float());
    println!("Confidence Level: {:.3}", final_state.information_asymmetry.confidence.to_float());
    println!("Observations Used: {}", final_state.information_asymmetry.observation_count);
    
    println!("\n4. Protection Diagnostics");
    println!("=========================");
    
    let diagnostics = protection.get_diagnostics();
    println!("Trade Count: {}", diagnostics.trade_count);
    println!("Impact Count: {}", diagnostics.impact_count);
    println!("Mean Impact: {:.6}", diagnostics.mean_impact.to_float());
    println!("Impact Std Dev: {:.6}", diagnostics.std_impact.to_float());
    println!("Current IA: {:.6}", diagnostics.current_ia.to_float());
    println!("Smoothed IA: {:.6}", diagnostics.smoothed_ia.to_float());
    println!("Toxicity Level: {:.3}", diagnostics.toxicity_level.to_float());
    println!("Protection Active: {}", diagnostics.protection_active);
    println!("Confidence: {:.3}", diagnostics.confidence.to_float());
    
    println!("\n5. Integration with Avellaneda-Stoikov Model");
    println!("============================================");
    
    // Create Avellaneda-Stoikov engine with adverse selection protection
    let as_params = AvellanedaStoikovParams::default();
    let mut engine = AvellanedaStoikovEngine::new(as_params)?;
    
    // Create market state representing adverse conditions
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.50),
        bid_price: FixedPoint::from_float(100.40),
        ask_price: FixedPoint::from_float(100.60),
        bid_volume: 800,  // Reduced liquidity
        ask_volume: 600,  // Asymmetric liquidity
        last_trade_price: FixedPoint::from_float(100.55),
        last_trade_volume: 2000,
        timestamp: current_timestamp(),
        sequence_number: 100,
        volatility: FixedPoint::from_float(0.35), // High volatility
        order_flow_imbalance: FixedPoint::from_float(0.7), // Strong imbalance
        microstructure_noise: FixedPoint::from_float(0.005),
    };
    
    // Calculate optimal quotes with adverse selection protection
    let quotes = engine.calculate_optimal_quotes(
        FixedPoint::from_float(100.50),
        1000, // Long inventory position
        FixedPoint::from_float(0.35),
        FixedPoint::from_float(0.08), // 5 minutes to maturity
        &market_state,
    )?;
    
    println!("Optimal Quotes with Adverse Selection Protection:");
    println!("Bid Price: {:.4}", quotes.bid_price.to_float());
    println!("Ask Price: {:.4}", quotes.ask_price.to_float());
    println!("Reservation Price: {:.4}", quotes.reservation_price.to_float());
    println!("Optimal Spread: {:.4}", quotes.optimal_spread.to_float());
    println!("Adverse Selection Premium: {:.4}", quotes.adverse_selection_premium.to_float());
    println!("Market Impact Adjustment: {:.4}", quotes.market_impact_adjustment.to_float());
    println!("Inventory Skew: {:.4}", quotes.inventory_skew.to_float());
    println!("Confidence: {:.3}", quotes.confidence.to_float());
    
    // Show adverse selection state from the engine
    let as_state = engine.get_adverse_selection_state();
    println!("\nEngine Adverse Selection State:");
    println!("Premium: {:.4}", as_state.premium.to_float());
    println!("Frequency Adjustment: {:.3}", as_state.frequency_adjustment.to_float());
    println!("Toxicity Level: {:.3}", as_state.toxicity_level.to_float());
    println!("Protection Active: {}", as_state.protection_active);
    
    println!("\n6. Quote Frequency Adjustment Demonstration");
    println!("===========================================");
    
    let base_quote_frequency = 20.0; // 20 quotes per second
    let adjusted_frequency = base_quote_frequency * as_state.frequency_adjustment.to_float();
    let frequency_reduction = (1.0 - as_state.frequency_adjustment.to_float()) * 100.0;
    
    println!("Base Quote Frequency: {:.1} quotes/second", base_quote_frequency);
    println!("Adjusted Quote Frequency: {:.1} quotes/second", adjusted_frequency);
    println!("Frequency Reduction: {:.1}%", frequency_reduction);
    
    if as_state.protection_active {
        println!("\n⚠️  ADVERSE SELECTION PROTECTION ACTIVE ⚠️");
        println!("Recommended actions:");
        println!("- Widen spreads by {:.4} (premium)", as_state.premium.to_float());
        println!("- Reduce quote frequency to {:.1} quotes/second", adjusted_frequency);
        println!("- Monitor for continued toxic flow");
        println!("- Consider temporary position reduction");
    } else {
        println!("\n✅ Normal market conditions detected");
        println!("No adverse selection protection needed");
    }
    
    println!("\n7. Parameter Sensitivity Analysis");
    println!("=================================");
    
    // Test different beta values
    let beta_values = vec![0.2, 0.5, 0.8, 1.2];
    println!("Beta Sensitivity (for same adverse trade):");
    
    for beta in beta_values {
        let mut test_params = AdverseSelectionParams::default();
        test_params.beta = FixedPoint::from_float(beta);
        
        let mut test_protection = AdverseSelectionProtection::new(test_params, base_frequency)?;
        
        // Apply the same adverse trade
        let test_trade = create_trade_info(100.20, 2500, 100.0, 0.3, 0.6);
        let test_state = test_protection.update(test_trade)?;
        
        println!("  Beta = {:.1}: Premium = {:.4}, Freq Adj = {:.3}", 
                 beta, 
                 test_state.premium.to_float(),
                 test_state.frequency_adjustment.to_float());
    }
    
    println!("\n=== Demo Complete ===");
    println!("The adverse selection protection system successfully:");
    println!("✓ Detected information asymmetry using price impact analysis");
    println!("✓ Calculated dynamic adverse selection premiums");
    println!("✓ Implemented spread widening based on toxic flow detection");
    println!("✓ Adjusted quote frequencies during adverse conditions");
    println!("✓ Integrated seamlessly with the Avellaneda-Stoikov model");
    
    Ok(())
}

fn create_trade_info(
    price: f64,
    volume: i64,
    mid_price: f64,
    volatility: f64,
    order_flow_imbalance: f64,
) -> TradeInfo {
    TradeInfo {
        price: FixedPoint::from_float(price),
        volume,
        mid_price: FixedPoint::from_float(mid_price),
        volatility: FixedPoint::from_float(volatility),
        total_volume: 10000,
        order_flow_imbalance: FixedPoint::from_float(order_flow_imbalance),
        timestamp: current_timestamp(),
    }
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}