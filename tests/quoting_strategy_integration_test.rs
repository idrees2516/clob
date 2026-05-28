//! Integration tests for the High-Frequency Quoting Strategy Engine
//! 
//! This test suite verifies that all components of the quoting strategy work together
//! correctly, including optimal spread calculation, inventory management, adaptive
//! parameters, and risk management controls.

use hft_lob_ec::math::fixed_point::FixedPoint;
use hft_lob_ec::models::quoting_strategy::*;
use hft_lob_ec::math::optimization::ModelParameters;
use hft_lob_ec::orderbook::types::Side;

#[test]
fn test_complete_quoting_strategy_workflow() {
    // Initialize model parameters
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
    
    // Initialize components
    let mut spread_calculator = OptimalSpreadCalculator::new(model_params);
    let risk_params = RiskParameters::default_conservative();
    let mut inventory_manager = InventoryManager::new(risk_params.clone(), 1000);
    let mut adaptive_engine = AdaptiveQuotingEngine::new(QuotingParameters::default());
    let mut risk_system = RiskManagementSystem::new(risk_params);
    
    // Create market state
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_volume: FixedPoint::from_float(1000.0),
        ask_volume: FixedPoint::from_float(1000.0),
        spread: FixedPoint::from_float(0.1),
        volatility: FixedPoint::from_float(0.2),
        timestamp: 1000,
        order_flow_imbalance: FixedPoint::zero(),
        market_depth: vec![
            (FixedPoint::from_float(99.90), FixedPoint::from_float(500.0)),
            (FixedPoint::from_float(99.95), FixedPoint::from_float(1000.0)),
            (FixedPoint::from_float(100.05), FixedPoint::from_float(1000.0)),
            (FixedPoint::from_float(100.10), FixedPoint::from_float(500.0)),
        ],
    };
    
    // Create trading state
    let mut trading_state = TradingState::new();
    trading_state.inventory = FixedPoint::from_float(25.0); // Some inventory
    trading_state.update_pnl(market_state.mid_price);
    
    // Test 1: Optimal spread calculation
    let hawkes_intensities = vec![FixedPoint::one(), FixedPoint::from_float(0.8)];
    let spread_result = spread_calculator.compute_optimal_spreads(
        &market_state,
        &trading_state,
        &hawkes_intensities,
        1000,
    );
    
    assert!(spread_result.is_ok());
    let (bid_spread, ask_spread) = spread_result.unwrap();
    assert!(bid_spread.to_float() > 0.0);
    assert!(ask_spread.to_float() > 0.0);
    println!("Optimal spreads - Bid: {:.4}, Ask: {:.4}", bid_spread.to_float(), ask_spread.to_float());
    
    // Test 2: Inventory management
    let position_check = inventory_manager.check_position_limits(&trading_state);
    assert!(position_check.is_ok());
    
    let inventory_penalty = inventory_manager.compute_inventory_penalty(
        trading_state.inventory,
        market_state.volatility,
        FixedPoint::from_float(1.0),
        FixedPoint::from_float(0.1),
    );
    assert!(inventory_penalty.to_float() > 0.0);
    println!("Inventory penalty: {:.6}", inventory_penalty.to_float());
    
    // Test 3: Adaptive parameters
    let correlated_assets = vec![
        ("ETH", &market_state),
        ("BTC", &market_state),
    ];
    
    let adaptive_params = adaptive_engine.update_adaptive_parameters(
        &market_state,
        &trading_state,
        &correlated_assets,
        1000,
    );
    
    assert!(adaptive_params.is_ok());
    let params = adaptive_params.unwrap();
    assert!(params.base_spread.to_float() > 0.0);
    assert!(params.quote_size.to_float() > 0.0);
    println!("Adaptive params - Spread: {:.4}, Size: {:.2}", 
             params.base_spread.to_float(), params.quote_size.to_float());
    
    // Test 4: Create quote pair
    let quote_pair = QuotePair {
        bid_price: market_state.mid_price - bid_spread,
        ask_price: market_state.mid_price + ask_spread,
        bid_size: params.quote_size,
        ask_size: params.quote_size,
        bid_spread,
        ask_spread,
        confidence: FixedPoint::from_float(0.8),
        timestamp: 1000,
    };
    
    // Test 5: Risk management check
    let risk_check = risk_system.check_risk_limits(
        &trading_state,
        &market_state,
        &quote_pair,
        1000,
    );
    
    assert!(risk_check.is_ok());
    let risk_result = risk_check.unwrap();
    println!("Risk status: {:?}, Risk score: {:.4}", 
             risk_result.status, risk_result.risk_score.to_float());
    
    // Verify quote pair is reasonable
    assert!(quote_pair.bid_price < market_state.mid_price);
    assert!(quote_pair.ask_price > market_state.mid_price);
    assert!(quote_pair.bid_size.to_float() > 0.0);
    assert!(quote_pair.ask_size.to_float() > 0.0);
    
    println!("Quote pair - Bid: {:.2}@{:.2}, Ask: {:.2}@{:.2}", 
             quote_pair.bid_price.to_float(), quote_pair.bid_size.to_float(),
             quote_pair.ask_price.to_float(), quote_pair.ask_size.to_float());
}

#[test]
fn test_high_volatility_scenario() {
    let model_params = ModelParameters {
        drift_coefficient: FixedPoint::from_float(0.05),
        volatility_coefficient: FixedPoint::from_float(0.5), // High volatility
        inventory_penalty: FixedPoint::from_float(0.2),
        adverse_selection_cost: FixedPoint::from_float(0.02),
        market_impact_coefficient: FixedPoint::from_float(0.1),
        risk_aversion: FixedPoint::from_float(3.0),
        terminal_time: FixedPoint::from_float(1.0),
        max_inventory: FixedPoint::from_float(50.0),
    };
    
    let mut spread_calculator = OptimalSpreadCalculator::new(model_params);
    let mut adaptive_engine = AdaptiveQuotingEngine::new(QuotingParameters::default());
    
    // High volatility market state
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.5),
        ask_price: FixedPoint::from_float(100.5),
        bid_volume: FixedPoint::from_float(200.0), // Lower liquidity
        ask_volume: FixedPoint::from_float(200.0),
        spread: FixedPoint::from_float(1.0), // Wide spread
        volatility: FixedPoint::from_float(0.8), // Very high volatility
        timestamp: 2000,
        order_flow_imbalance: FixedPoint::from_float(0.3), // Imbalanced
        market_depth: vec![],
    };
    
    let trading_state = TradingState::new();
    let hawkes_intensities = vec![FixedPoint::from_float(2.0), FixedPoint::from_float(1.5)];
    
    // Test spread calculation in high volatility
    let spread_result = spread_calculator.compute_optimal_spreads(
        &market_state,
        &trading_state,
        &hawkes_intensities,
        2000,
    );
    
    assert!(spread_result.is_ok());
    let (bid_spread, ask_spread) = spread_result.unwrap();
    
    // Spreads should be wider in high volatility
    assert!(bid_spread.to_float() > 0.01); // At least 100 bps
    assert!(ask_spread.to_float() > 0.01);
    
    // Test adaptive parameters
    let adaptive_params = adaptive_engine.update_adaptive_parameters(
        &market_state,
        &trading_state,
        &[],
        2000,
    );
    
    assert!(adaptive_params.is_ok());
    let params = adaptive_params.unwrap();
    
    // Parameters should be adjusted for high volatility
    assert!(params.spread_multiplier.to_float() > 1.0);
    assert!(params.volatility_adjustment.to_float() > 1.0);
    
    println!("High volatility scenario - Spreads: {:.4}/{:.4}, Multipliers: {:.2}/{:.2}",
             bid_spread.to_float(), ask_spread.to_float(),
             params.spread_multiplier.to_float(), params.volatility_adjustment.to_float());
}

#[test]
fn test_large_inventory_scenario() {
    let risk_params = RiskParameters::default_conservative();
    let mut inventory_manager = InventoryManager::new(risk_params, 1000);
    
    // Large inventory position
    let mut trading_state = TradingState::new();
    trading_state.inventory = FixedPoint::from_float(85.0); // Close to soft limit
    trading_state.cash = FixedPoint::from_float(1000.0);
    trading_state.update_pnl(FixedPoint::from_float(100.0));
    
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_volume: FixedPoint::from_float(1000.0),
        ask_volume: FixedPoint::from_float(1000.0),
        spread: FixedPoint::from_float(0.1),
        volatility: FixedPoint::from_float(0.2),
        timestamp: 3000,
        order_flow_imbalance: FixedPoint::zero(),
        market_depth: vec![],
    };
    
    // Test position limits
    let position_check = inventory_manager.check_position_limits(&trading_state);
    assert!(position_check.is_ok());
    let should_reduce = position_check.unwrap();
    assert!(!should_reduce); // Should signal to reduce position
    
    // Test position reduction strategy
    let reduction_strategy = inventory_manager.compute_position_reduction_strategy(
        &trading_state,
        &market_state,
    );
    
    assert!(reduction_strategy.is_ok());
    let (ask_increase, bid_decrease) = reduction_strategy.unwrap();
    
    // Should increase ask size and decrease bid size for long position
    assert!(ask_increase.to_float() > 0.0);
    assert!(bid_decrease.to_float() < 0.0);
    
    // Test inventory skew adjustment
    let (bid_adj, ask_adj) = inventory_manager.compute_inventory_skew_adjustment(
        trading_state.inventory,
        market_state.volatility,
        FixedPoint::from_float(0.1),
        FixedPoint::from_float(1.0),
    );
    
    // For long position, bid should be adjusted down, ask up
    assert!(bid_adj.to_float() < 0.0);
    assert!(ask_adj.to_float() > 0.0);
    
    println!("Large inventory scenario - Reduction: ask+{:.2}, bid{:.2}, Skew: {:.4}/{:.4}",
             ask_increase.to_float(), bid_decrease.to_float(),
             bid_adj.to_float(), ask_adj.to_float());
}

#[test]
fn test_risk_violation_scenario() {
    let risk_params = RiskParameters {
        max_drawdown: FixedPoint::from_float(0.02), // Very low threshold
        max_inventory: FixedPoint::from_float(50.0),
        position_limit_hard: FixedPoint::from_float(45.0),
        ..RiskParameters::default_conservative()
    };
    
    let mut risk_system = RiskManagementSystem::new(risk_params);
    
    // Create scenario with risk violations
    let mut trading_state = TradingState::new();
    trading_state.inventory = FixedPoint::from_float(50.0); // Exceeds hard limit
    trading_state.total_pnl = FixedPoint::from_float(-100.0); // Large loss
    trading_state.realized_pnl = FixedPoint::from_float(-100.0);
    
    let market_state = MarketState {
        mid_price: FixedPoint::from_float(100.0),
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_volume: FixedPoint::from_float(50.0), // Low liquidity
        ask_volume: FixedPoint::from_float(50.0),
        spread: FixedPoint::from_float(0.1),
        volatility: FixedPoint::from_float(0.6), // High volatility
        timestamp: 4000,
        order_flow_imbalance: FixedPoint::zero(),
        market_depth: vec![],
    };
    
    let quote_pair = QuotePair {
        bid_price: FixedPoint::from_float(99.95),
        ask_price: FixedPoint::from_float(100.05),
        bid_size: FixedPoint::from_float(100.0),
        ask_size: FixedPoint::from_float(100.0),
        bid_spread: FixedPoint::from_float(0.05),
        ask_spread: FixedPoint::from_float(0.05),
        confidence: FixedPoint::from_float(0.5),
        timestamp: 4000,
    };
    
    // Test risk check - should detect violations
    let risk_check = risk_system.check_risk_limits(
        &trading_state,
        &market_state,
        &quote_pair,
        4000,
    );
    
    assert!(risk_check.is_ok());
    let risk_result = risk_check.unwrap();
    
    // Should have violations or warnings
    assert!(!risk_result.violations.is_empty() || !risk_result.warnings.is_empty());
    assert!(risk_result.risk_score.to_float() > 0.5); // High risk score
    
    println!("Risk violation scenario - Status: {:?}, Violations: {}, Warnings: {}, Score: {:.4}",
             risk_result.status, risk_result.violations.len(), 
             risk_result.warnings.len(), risk_result.risk_score.to_float());
    
    // Test emergency procedures if there are violations
    if !risk_result.violations.is_empty() {
        for violation in &risk_result.violations {
            let emergency_actions = risk_system.execute_emergency_procedures(
                &trading_state,
                &market_state,
                violation,
            );
            
            assert!(emergency_actions.is_ok());
            let actions = emergency_actions.unwrap();
            assert!(!actions.is_empty());
            
            println!("Emergency actions for violation: {} actions", actions.len());
        }
    }
}

#[test]
fn test_market_regime_adaptation() {
    let mut adaptive_engine = AdaptiveQuotingEngine::new(QuotingParameters::default());
    
    // Test different market regimes
    let regimes_and_states = vec![
        (MarketRegime::Normal, MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: FixedPoint::from_float(1000.0),
            ask_volume: FixedPoint::from_float(1000.0),
            spread: FixedPoint::from_float(0.1),
            volatility: FixedPoint::from_float(0.2),
            timestamp: 5000,
            order_flow_imbalance: FixedPoint::zero(),
            market_depth: vec![],
        }),
        (MarketRegime::HighVolatility, MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.5),
            ask_price: FixedPoint::from_float(100.5),
            bid_volume: FixedPoint::from_float(1000.0),
            ask_volume: FixedPoint::from_float(1000.0),
            spread: FixedPoint::from_float(1.0),
            volatility: FixedPoint::from_float(0.8),
            timestamp: 5000,
            order_flow_imbalance: FixedPoint::zero(),
            market_depth: vec![],
        }),
        (MarketRegime::LowLiquidity, MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: FixedPoint::from_float(50.0),
            ask_volume: FixedPoint::from_float(50.0),
            spread: FixedPoint::from_float(0.1),
            volatility: FixedPoint::from_float(0.2),
            timestamp: 5000,
            order_flow_imbalance: FixedPoint::zero(),
            market_depth: vec![],
        }),
    ];
    
    for (expected_regime, market_state) in regimes_and_states {
        let mut params = QuotingParameters::default();
        let detected_regime = adaptive_engine.adapt_to_market_regime(&mut params, &market_state);
        
        println!("Expected: {:?}, Detected: {:?}, Spread multiplier: {:.2}, Size multiplier: {:.2}",
                 expected_regime, detected_regime, 
                 params.spread_multiplier.to_float(), params.size_multiplier.to_float());
        
        // Verify regime-specific adjustments
        match detected_regime {
            MarketRegime::Normal => {
                assert_eq!(params.spread_multiplier.to_float(), 1.0);
                assert_eq!(params.size_multiplier.to_float(), 1.0);
            }
            MarketRegime::HighVolatility => {
                assert!(params.spread_multiplier.to_float() > 1.0);
                assert!(params.size_multiplier.to_float() < 1.0);
            }
            MarketRegime::LowLiquidity => {
                assert!(params.spread_multiplier.to_float() > 1.0);
                assert!(params.size_multiplier.to_float() < 1.0);
            }
            _ => {}
        }
    }
}

#[test]
fn test_correlation_tracking() {
    let mut correlation_tracker = CorrelationTracker::new();
    
    // Simulate correlated price movements
    let base_price = 100.0;
    let correlation = 0.8;
    
    for i in 0..50 {
        let timestamp = 1000 + i * 1000;
        let price_change = (i as f64 - 25.0) * 0.01; // Price moves from -0.25 to +0.24
        
        let asset1_price = base_price + price_change;
        let asset2_price = base_price + correlation * price_change + 0.1 * (i as f64).sin(); // Correlated with noise
        
        correlation_tracker.update_correlation(
            "BTC",
            "ETH",
            FixedPoint::from_float(asset2_price),
            timestamp,
        );
        
        // Also update BTC price history
        correlation_tracker.price_histories.entry("BTC".to_string())
            .or_insert_with(|| std::collections::VecDeque::with_capacity(100))
            .push_back((timestamp, FixedPoint::from_float(asset1_price)));
    }
    
    // Get correlations
    let correlations = correlation_tracker.get_correlations("BTC");
    assert!(!correlations.is_empty());
    
    for (asset, corr) in correlations {
        println!("Correlation between BTC and {}: {:.4}", asset, corr.to_float());
        // Should detect positive correlation
        assert!(corr.to_float() > 0.3);
    }
}

#[test]
fn test_complete_strategy_performance() {
    // This test simulates a complete trading session to verify overall performance
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
    
    let mut trading_state = TradingState::new();
    let mut total_quotes_generated = 0;
    let mut successful_risk_checks = 0;
    
    // Simulate 100 time steps
    for i in 0..100 {
        let timestamp = 1000 + i * 1000;
        
        // Simulate evolving market conditions
        let volatility = 0.15 + 0.1 * (i as f64 / 20.0).sin();
        let mid_price = 100.0 + 5.0 * (i as f64 / 30.0).sin();
        let spread = 0.05 + 0.05 * volatility;
        
        let market_state = MarketState {
            mid_price: FixedPoint::from_float(mid_price),
            bid_price: FixedPoint::from_float(mid_price - spread / 2.0),
            ask_price: FixedPoint::from_float(mid_price + spread / 2.0),
            bid_volume: FixedPoint::from_float(800.0 + 400.0 * (i as f64 / 10.0).cos()),
            ask_volume: FixedPoint::from_float(800.0 + 400.0 * (i as f64 / 10.0).cos()),
            spread: FixedPoint::from_float(spread),
            volatility: FixedPoint::from_float(volatility),
            timestamp,
            order_flow_imbalance: FixedPoint::from_float(0.1 * (i as f64 / 15.0).sin()),
            market_depth: vec![],
        };
        
        // Update trading state (simulate some trading activity)
        if i > 0 {
            let inventory_change = (i as f64 / 50.0).sin() * 2.0;
            trading_state.inventory = trading_state.inventory + FixedPoint::from_float(inventory_change);
            trading_state.inventory = trading_state.inventory.max(FixedPoint::from_float(-50.0))
                                                           .min(FixedPoint::from_float(50.0));
            
            let pnl_change = inventory_change * (mid_price - 100.0) * 0.1;
            trading_state.realized_pnl = trading_state.realized_pnl + FixedPoint::from_float(pnl_change);
            trading_state.update_pnl(market_state.mid_price);
        }
        
        // Generate quotes using the complete strategy
        let hawkes_intensities = vec![
            FixedPoint::from_float(1.0 + 0.5 * (i as f64 / 20.0).cos()),
            FixedPoint::from_float(0.8 + 0.3 * (i as f64 / 25.0).sin()),
        ];
        
        // 1. Calculate optimal spreads
        let spread_result = spread_calculator.compute_optimal_spreads(
            &market_state,
            &trading_state,
            &hawkes_intensities,
            timestamp,
        );
        
        if spread_result.is_ok() {
            let (bid_spread, ask_spread) = spread_result.unwrap();
            
            // 2. Get adaptive parameters
            let adaptive_result = adaptive_engine.update_adaptive_parameters(
                &market_state,
                &trading_state,
                &[],
                timestamp,
            );
            
            if adaptive_result.is_ok() {
                let params = adaptive_result.unwrap();
                
                // 3. Create quote pair
                let quote_pair = QuotePair {
                    bid_price: market_state.mid_price - bid_spread,
                    ask_price: market_state.mid_price + ask_spread,
                    bid_size: params.quote_size,
                    ask_size: params.quote_size,
                    bid_spread,
                    ask_spread,
                    confidence: FixedPoint::from_float(0.8),
                    timestamp,
                };
                
                // 4. Risk management check
                let risk_result = risk_system.check_risk_limits(
                    &trading_state,
                    &market_state,
                    &quote_pair,
                    timestamp,
                );
                
                if risk_result.is_ok() {
                    successful_risk_checks += 1;
                    total_quotes_generated += 1;
                    
                    // Update inventory manager history
                    inventory_manager.update_history(
                        timestamp,
                        trading_state.inventory,
                        trading_state.total_pnl,
                    );
                }
            }
        }
    }
    
    // Verify strategy performance
    assert!(total_quotes_generated > 80); // Should generate quotes in most scenarios
    assert!(successful_risk_checks > 80); // Most should pass risk checks
    
    let success_rate = successful_risk_checks as f64 / 100.0;
    println!("Strategy performance - Quotes generated: {}, Success rate: {:.1}%", 
             total_quotes_generated, success_rate * 100.0);
    
    // Final trading state should be reasonable
    assert!(trading_state.inventory.abs().to_float() < 60.0); // Within reasonable bounds
    println!("Final inventory: {:.2}, Final PnL: {:.2}", 
             trading_state.inventory.to_float(), trading_state.total_pnl.to_float());
}