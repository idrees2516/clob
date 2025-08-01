//! High-Frequency Quoting Strategy Engine Demo
//! 
//! This example demonstrates how to use the complete HF quoting strategy engine
//! with optimal spread calculation, inventory management, adaptive parameters,
//! and risk management controls.

use hft_lob_ec::math::fixed_point::FixedPoint;
use hft_lob_ec::models::quoting_strategy::*;
use hft_lob_ec::math::optimization::ModelParameters;
use hft_lob_ec::orderbook::types::Side;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== High-Frequency Quoting Strategy Engine Demo ===\n");
    
    // 1. Initialize the strategy components
    println!("1. Initializing strategy components...");
    
    let model_params = ModelParameters {
        drift_coefficient: FixedPoint::from_float(0.05),
        volatility_coefficient: FixedPoint::from_float(0.2),
        inventory_penalty: FixedPoint::from_float(0.1),
        adverse_selection_cost: FixedPoint::from_float(0.01),
        market_impact_coefficient: FixedPoint::from_float(0.05),
        risk_aversion: FixedPoint::from_float(2.0),
        terminal_time: FixedPoint::from_float(1.0),
        max_inventory: FixedPoint::from_float(100.0),
    };
    
    let mut spread_calculator = OptimalSpreadCalculator::new(model_params);
    let risk_params = RiskParameters::default_conservative();
    let mut inventory_manager = InventoryManager::new(risk_params.clone(), 1000);
    let mut adaptive_engine = AdaptiveQuotingEngine::new(QuotingParameters::default());
    let mut risk_system = RiskManagementSystem::new(risk_params);
    
    println!("✓ Components initialized successfully\n");
    
    // 2. Create market scenario
    println!("2. Setting up market scenario...");
    
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_volume: FixedPoint::from_float(1000.0),
        ask_volume: FixedPoint::from_float(1000.0),
        spread: FixedPoint::from_float(0.1),
        volatility: FixedPoint::from_float(0.2),
        timestamp: 1000,
        order_flow_imbalance: FixedPoint::from_float(0.1),
        market_depth: vec![
            (FixedPoint::from_float(99.90), FixedPoint::from_float(500.0)),
            (FixedPoint::from_float(99.95), FixedPoint::from_float(1000.0)),
            (FixedPoint::from_float(100.05), FixedPoint::from_float(1000.0)),
            (FixedPoint::from_float(100.10), FixedPoint::from_float(500.0)),
        ],
    };
    
    let mut trading_state = TradingState::new();
    trading_state.inventory = FixedPoint::from_float(25.0);
    trading_state.cash = FixedPoint::from_float(2500.0);
    trading_state.update_pnl(market_state.mid_price);
    
    println!("Market State:");
    println!("  Mid Price: ${:.2}", market_state.mid_price.to_float());
    println!("  Spread: {:.4} ({:.2} bps)", market_state.spread.to_float(), market_state.spread.to_float() * 10000.0);
    println!("  Volatility: {:.1}%", market_state.volatility.to_float() * 100.0);
    println!("  Bid/Ask Volume: {:.0}/{:.0}", market_state.bid_volume.to_float(), market_state.ask_volume.to_float());
    
    println!("\nTrading State:");
    println!("  Inventory: {:.0} shares", trading_state.inventory.to_float());
    println!("  Cash: ${:.2}", trading_state.cash.to_float());
    println!("  Total PnL: ${:.2}\n", trading_state.total_pnl.to_float());
    
    // 3. Calculate optimal spreads
    println!("3. Calculating optimal bid-ask spreads...");
    
    let hawkes_intensities = vec![
        FixedPoint::from_float(1.2), // Buy order intensity
        FixedPoint::from_float(0.9), // Sell order intensity
    ];
    
    let spread_result = spread_calculator.compute_optimal_spreads(
        &market_state,
        &trading_state,
        &hawkes_intensities,
        1000,
    )?;
    
    let (bid_spread, ask_spread) = spread_result;
    
    println!("Optimal Spreads (from equations 9.18-9.21):");
    println!("  Bid Spread: {:.4} ({:.1} bps)", bid_spread.to_float(), bid_spread.to_float() * 10000.0);
    println!("  Ask Spread: {:.4} ({:.1} bps)", ask_spread.to_float(), ask_spread.to_float() * 10000.0);
    println!("  Asymmetry: {:.4} (inventory skew)", (ask_spread - bid_spread).to_float());
    
    // 4. Inventory management analysis
    println!("\n4. Analyzing inventory management...");
    
    let inventory_penalty = inventory_manager.compute_inventory_penalty(
        trading_state.inventory,
        market_state.volatility,
        FixedPoint::from_float(1.0), // Time to maturity
        FixedPoint::from_float(0.1), // Risk aversion
    );
    
    let (bid_skew, ask_skew) = inventory_manager.compute_inventory_skew_adjustment(
        trading_state.inventory,
        market_state.volatility,
        FixedPoint::from_float(0.1),
        FixedPoint::from_float(1.0),
    );
    
    println!("Inventory Analysis (equations 6.14-6.17):");
    println!("  Current Inventory: {:.0} shares", trading_state.inventory.to_float());
    println!("  Inventory Penalty: {:.6}", inventory_penalty.to_float());
    println!("  Bid Skew Adjustment: {:.4}", bid_skew.to_float());
    println!("  Ask Skew Adjustment: {:.4}", ask_skew.to_float());
    
    let position_check = inventory_manager.check_position_limits(&trading_state)?;
    println!("  Position Status: {:?}", if position_check { "Normal" } else { "Needs Reduction" });
    
    // 5. Adaptive parameter optimization
    println!("\n5. Computing adaptive quoting parameters...");
    
    let correlated_assets = vec![
        ("ETH", &market_state),
        ("SOL", &market_state),
    ];
    
    let adaptive_params = adaptive_engine.update_adaptive_parameters(
        &market_state,
        &trading_state,
        &correlated_assets,
        1000,
    )?;
    
    println!("Adaptive Parameters:");
    println!("  Base Spread: {:.4}", adaptive_params.base_spread.to_float());
    println!("  Quote Size: {:.0}", adaptive_params.quote_size.to_float());
    println!("  Spread Multiplier: {:.2}x", adaptive_params.spread_multiplier.to_float());
    println!("  Size Multiplier: {:.2}x", adaptive_params.size_multiplier.to_float());
    println!("  Volatility Adjustment: {:.2}x", adaptive_params.volatility_adjustment.to_float());
    
    // 6. Generate final quote pair
    println!("\n6. Generating optimized quote pair...");
    
    let final_bid_price = market_state.mid_price - bid_spread;
    let final_ask_price = market_state.mid_price + ask_spread;
    let final_bid_size = adaptive_params.quote_size * adaptive_params.size_multiplier;
    let final_ask_size = adaptive_params.quote_size * adaptive_params.size_multiplier;
    
    let quote_pair = QuotePair {
        bid_price: final_bid_price,
        ask_price: final_ask_price,
        bid_size: final_bid_size,
        ask_size: final_ask_size,
        bid_spread,
        ask_spread,
        confidence: FixedPoint::from_float(0.85),
        timestamp: 1000,
    };
    
    println!("Final Quote Pair:");
    println!("  Bid: ${:.3} @ {:.0} shares", quote_pair.bid_price.to_float(), quote_pair.bid_size.to_float());
    println!("  Ask: ${:.3} @ {:.0} shares", quote_pair.ask_price.to_float(), quote_pair.ask_size.to_float());
    println!("  Mid: ${:.3}", market_state.mid_price.to_float());
    println!("  Effective Spread: {:.4} ({:.1} bps)", 
             (quote_pair.ask_price - quote_pair.bid_price).to_float(),
             (quote_pair.ask_price - quote_pair.bid_price).to_float() / market_state.mid_price.to_float() * 10000.0);
    
    // 7. Risk management validation
    println!("\n7. Performing risk management checks...");
    
    let risk_check = risk_system.check_risk_limits(
        &trading_state,
        &market_state,
        &quote_pair,
        1000,
    )?;
    
    println!("Risk Assessment:");
    println!("  Overall Status: {:?}", risk_check.status);
    println!("  Risk Score: {:.3}/1.0", risk_check.risk_score.to_float());
    println!("  Warnings: {}", risk_check.warnings.len());
    println!("  Violations: {}", risk_check.violations.len());
    
    if !risk_check.warnings.is_empty() {
        println!("  Risk Warnings:");
        for warning in &risk_check.warnings {
            println!("    - {:?}", warning);
        }
    }
    
    if !risk_check.violations.is_empty() {
        println!("  Risk Violations:");
        for violation in &risk_check.violations {
            println!("    - {:?}", violation);
        }
    }
    
    if !risk_check.recommended_actions.is_empty() {
        println!("  Recommended Actions:");
        for action in &risk_check.recommended_actions {
            println!("    - {:?}", action);
        }
    }
    
    // 8. Performance metrics
    println!("\n8. Strategy performance summary...");
    
    let expected_pnl_per_trade = (bid_spread + ask_spread) / FixedPoint::from_float(2.0) * quote_pair.bid_size;
    let inventory_cost = inventory_penalty * trading_state.inventory.abs();
    let net_expected_pnl = expected_pnl_per_trade - inventory_cost;
    
    println!("Performance Metrics:");
    println!("  Expected PnL per Trade: ${:.4}", expected_pnl_per_trade.to_float());
    println!("  Inventory Holding Cost: ${:.4}", inventory_cost.to_float());
    println!("  Net Expected PnL: ${:.4}", net_expected_pnl.to_float());
    println!("  Quote Confidence: {:.1}%", quote_pair.confidence.to_float() * 100.0);
    
    // 9. Market regime analysis
    println!("\n9. Market regime analysis...");
    
    let mut regime_detector = MarketRegimeDetector::new();
    let detected_regime = regime_detector.detect_regime(&market_state);
    
    println!("Market Regime Analysis:");
    println!("  Detected Regime: {:?}", detected_regime);
    println!("  Regime Confidence: {:.1}%", regime_detector.regime_confidence.to_float() * 100.0);
    
    match detected_regime {
        MarketRegime::Normal => println!("  → Standard quoting parameters applied"),
        MarketRegime::HighVolatility => println!("  → Wider spreads and smaller sizes recommended"),
        MarketRegime::LowLiquidity => println!("  → Reduced quote sizes and careful positioning"),
        MarketRegime::Crisis => println!("  → Emergency risk controls activated"),
        MarketRegime::Recovery => println!("  → Gradual return to normal parameters"),
    }
    
    // 10. Simulation of high-volatility scenario
    println!("\n10. High-volatility scenario simulation...");
    
    let mut high_vol_market = market_state.clone();
    high_vol_market.volatility = FixedPoint::from_float(0.6); // 60% volatility
    high_vol_market.spread = FixedPoint::from_float(0.5); // Wide spread
    high_vol_market.bid_volume = FixedPoint::from_float(200.0); // Low liquidity
    high_vol_market.ask_volume = FixedPoint::from_float(200.0);
    
    let high_vol_spreads = spread_calculator.compute_optimal_spreads(
        &high_vol_market,
        &trading_state,
        &hawkes_intensities,
        2000,
    )?;
    
    let high_vol_params = adaptive_engine.update_adaptive_parameters(
        &high_vol_market,
        &trading_state,
        &[],
        2000,
    )?;
    
    println!("High Volatility Scenario:");
    println!("  Market Volatility: {:.1}%", high_vol_market.volatility.to_float() * 100.0);
    println!("  Optimal Bid Spread: {:.4} ({:.1} bps)", high_vol_spreads.0.to_float(), high_vol_spreads.0.to_float() * 10000.0);
    println!("  Optimal Ask Spread: {:.4} ({:.1} bps)", high_vol_spreads.1.to_float(), high_vol_spreads.1.to_float() * 10000.0);
    println!("  Spread Multiplier: {:.2}x", high_vol_params.spread_multiplier.to_float());
    println!("  Size Multiplier: {:.2}x", high_vol_params.size_multiplier.to_float());
    
    println!("\n=== Demo completed successfully! ===");
    println!("\nKey Features Demonstrated:");
    println!("✓ Optimal bid-ask spread calculation using closed-form solutions");
    println!("✓ Inventory management with penalty functions and position limits");
    println!("✓ Adaptive parameter adjustment based on market conditions");
    println!("✓ Comprehensive risk management with multiple circuit breakers");
    println!("✓ Market regime detection and strategy adaptation");
    println!("✓ Multi-asset correlation tracking and adjustments");
    println!("✓ Real-time performance monitoring and optimization");
    
    Ok(())
}

#[allow(dead_code)]
fn demonstrate_emergency_procedures() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Emergency Procedures Demo ===");
    
    let risk_params = RiskParameters {
        max_drawdown: FixedPoint::from_float(0.02), // 2% max drawdown
        max_inventory: FixedPoint::from_float(50.0),
        position_limit_hard: FixedPoint::from_float(45.0),
        ..RiskParameters::default_conservative()
    };
    
    let mut risk_system = RiskManagementSystem::new(risk_params);
    
    // Create emergency scenario
    let mut emergency_trading_state = TradingState::new();
    emergency_trading_state.inventory = FixedPoint::from_float(50.0); // Exceeds limit
    emergency_trading_state.total_pnl = FixedPoint::from_float(-150.0); // Large loss
    emergency_trading_state.realized_pnl = FixedPoint::from_float(-150.0);
    
    let emergency_market_state = MarketState {
        mid_price: FixedPoint::from_float(95.0), // Price dropped
        bid_price: FixedPoint::from_float(94.0),
        ask_price: FixedPoint::from_float(96.0),
        bid_volume: FixedPoint::from_float(50.0), // Low liquidity
        ask_volume: FixedPoint::from_float(50.0),
        spread: FixedPoint::from_float(2.0), // Wide spread
        volatility: FixedPoint::from_float(0.8), // High volatility
        timestamp: 5000,
        order_flow_imbalance: FixedPoint::from_float(0.5),
        market_depth: vec![],
    };
    
    let emergency_quote = QuotePair {
        bid_price: FixedPoint::from_float(94.0),
        ask_price: FixedPoint::from_float(96.0),
        bid_size: FixedPoint::from_float(100.0),
        ask_size: FixedPoint::from_float(100.0),
        bid_spread: FixedPoint::from_float(1.0),
        ask_spread: FixedPoint::from_float(1.0),
        confidence: FixedPoint::from_float(0.3),
        timestamp: 5000,
    };
    
    println!("Emergency Scenario:");
    println!("  Position: {:.0} shares (limit: 45)", emergency_trading_state.inventory.to_float());
    println!("  PnL: ${:.2}", emergency_trading_state.total_pnl.to_float());
    println!("  Market Price: ${:.2} (down from $100)", emergency_market_state.mid_price.to_float());
    println!("  Volatility: {:.1}%", emergency_market_state.volatility.to_float() * 100.0);
    
    let emergency_check = risk_system.check_risk_limits(
        &emergency_trading_state,
        &emergency_market_state,
        &emergency_quote,
        5000,
    )?;
    
    println!("\nRisk Assessment:");
    println!("  Status: {:?}", emergency_check.status);
    println!("  Violations: {}", emergency_check.violations.len());
    println!("  Risk Score: {:.3}", emergency_check.risk_score.to_float());
    
    // Execute emergency procedures
    for violation in &emergency_check.violations {
        println!("\nExecuting emergency procedures for: {:?}", violation);
        
        let emergency_actions = risk_system.execute_emergency_procedures(
            &emergency_trading_state,
            &emergency_market_state,
            violation,
        )?;
        
        println!("Emergency Actions:");
        for action in emergency_actions {
            println!("  - {:?}: {} shares, max impact {:.1}%", 
                     action.action_type, 
                     action.size.to_float(),
                     action.max_price_impact.to_float() * 100.0);
            println!("    Reason: {}", action.reason);
            println!("    Timeout: {}s", action.timeout_seconds);
        }
    }
    
    Ok(())
}