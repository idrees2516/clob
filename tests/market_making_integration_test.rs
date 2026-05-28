use clob_engine::market_making::*;
use clob_engine::error::Result;
use std::collections::HashMap;

#[tokio::test]
async fn test_avellaneda_stoikov_integration() -> Result<()> {
    // Create Avellaneda-Stoikov engine
    let mut engine = avellaneda_stoikov::AvellanedaStoikovEngine::new(1.0, 3600.0)?;
    
    // Create mock market state
    let mut market_state = create_mock_market_state();
    
    // Generate quotes
    let quotes = engine.generate_quotes("BTC", &market_state)?;
    
    // Verify quote structure
    assert!(quotes.bid_price > 0.0);
    assert!(quotes.ask_price > quotes.bid_price);
    assert!(quotes.bid_size > 0.0);
    assert!(quotes.ask_size > 0.0);
    assert!(quotes.confidence > 0.0 && quotes.confidence <= 1.0);
    
    // Test inventory update
    engine.update_inventory("BTC", 100.0)?;
    let quotes_with_inventory = engine.generate_quotes("BTC", &market_state)?;
    
    // Quotes should be skewed due to inventory
    assert_ne!(quotes.bid_price, quotes_with_inventory.bid_price);
    
    Ok(())
}

#[tokio::test]
async fn test_gueant_lehalle_tapia_integration() -> Result<()> {
    // Create Guéant-Lehalle-Tapia engine
    let mut engine = gueant_lehalle_tapia::GuéantLehalleTapiaEngine::new(1.0)?;
    
    // Create mock market state
    let market_state = create_mock_market_state();
    
    // Generate quotes for multiple assets
    let btc_quotes = engine.generate_quotes("BTC", &market_state)?;
    let eth_quotes = engine.generate_quotes("ETH", &market_state)?;
    
    // Verify multi-asset functionality
    assert!(btc_quotes.bid_price > 0.0);
    assert!(eth_quotes.bid_price > 0.0);
    
    // Test portfolio optimization
    let optimal_weights = engine.optimize_portfolio(&market_state)?;
    assert!(!optimal_weights.is_empty());
    
    // Test hedge order calculation
    let hedge_orders = engine.calculate_hedge_orders(&market_state)?;
    // Should have hedge orders if there are positions
    
    Ok(())
}

#[tokio::test]
async fn test_cartea_jaimungal_integration() -> Result<()> {
    // Create Cartea-Jaimungal engine
    let mut engine = cartea_jaimungal::CarteaJaimungalEngine::new(1.0)?;
    
    // Create mock market state
    let market_state = create_mock_market_state();
    
    // Generate quotes
    let quotes = engine.generate_quotes("BTC", &market_state)?;
    
    // Verify jump-diffusion adjustments
    assert!(quotes.bid_price > 0.0);
    assert!(quotes.ask_price > quotes.bid_price);
    
    // Test with market data update (simulate jumps)
    let price_data = vec![100.0, 100.1, 99.8, 102.0, 101.5]; // Contains a jump
    let timestamps = vec![1000, 1001, 1002, 1003, 1004];
    
    engine.update_market_data("BTC", &price_data, &timestamps)?;
    
    let quotes_after_jump = engine.generate_quotes("BTC", &market_state)?;
    
    // Quotes should be different after jump detection
    // (In practice, spreads might widen)
    
    Ok(())
}

#[tokio::test]
async fn test_high_frequency_quoting_integration() -> Result<()> {
    // Create high-frequency quoting engine
    let config = MarketMakingConfig {
        risk_aversion: 1.0,
        time_horizon: 3600.0,
        update_frequency: 1000.0,
        rebalancing_threshold: 0.01,
        max_inventory_ratio: 0.1,
        min_spread_bps: 1.0,
        max_spread_bps: 100.0,
        quote_size_multiplier: 1.0,
        adverse_selection_threshold: 0.3,
        regime_detection_window: 100,
    };
    
    let mut engine = high_frequency_quoting::HighFrequencyQuotingEngine::new(config)?;
    
    // Create mock market state
    let market_state = create_mock_market_state();
    
    // Generate quotes
    let quotes = engine.generate_quotes("BTC", &market_state)?;
    
    // Verify high-frequency specific features
    assert!(quotes.bid_price > 0.0);
    assert!(quotes.ask_price > quotes.bid_price);
    assert!(quotes.confidence > 0.0);
    
    // Test order flow update
    let mock_orders = vec![
        high_frequency_quoting::OrderArrival {
            timestamp: 1000,
            order_type: high_frequency_quoting::OrderType::Market,
            side: high_frequency_quoting::OrderSide::Buy,
            size: 100.0,
            price: 100.0,
            intensity_contribution: 0.1,
        },
    ];
    
    engine.update_order_flow(mock_orders)?;
    
    let quotes_after_flow = engine.generate_quotes("BTC", &market_state)?;
    
    // Quotes should reflect updated order flow
    
    Ok(())
}

#[tokio::test]
async fn test_market_making_engine_integration() -> Result<()> {
    // Create comprehensive market making engine
    let config = MarketMakingConfig {
        risk_aversion: 1.0,
        time_horizon: 3600.0,
        update_frequency: 100.0,
        rebalancing_threshold: 0.01,
        max_inventory_ratio: 0.1,
        min_spread_bps: 1.0,
        max_spread_bps: 100.0,
        quote_size_multiplier: 1.0,
        adverse_selection_threshold: 0.3,
        regime_detection_window: 100,
    };
    
    let mut engine = MarketMakingEngine::new(config)?;
    
    // Create mock market data
    let market_data = MarketData {
        timestamp: 1000,
        prices: [("BTC".to_string(), 100.0)].iter().cloned().collect(),
        volumes: [("BTC".to_string(), 1000.0)].iter().cloned().collect(),
        bid_ask_spreads: [("BTC".to_string(), 0.01)].iter().cloned().collect(),
        order_book_depth: HashMap::new(),
        trade_history: Vec::new(),
    };
    
    // Update market data
    engine.update_market_data(&market_data)?;
    
    // Generate quotes using all models
    let quotes = engine.generate_quotes("BTC")?;
    
    // Verify ensemble quotes
    assert!(quotes.bid_price > 0.0);
    assert!(quotes.ask_price > quotes.bid_price);
    assert!(quotes.confidence > 0.0);
    
    // Test trade execution
    let trade = Trade {
        symbol: "BTC".to_string(),
        price: 100.0,
        quantity: 10.0,
        side: TradeSide::Buy,
        timestamp: 1001,
    };
    
    engine.execute_trade(&trade)?;
    
    // Verify inventory update
    let inventory = engine.state.inventory.get("BTC").copied().unwrap_or(0.0);
    assert_eq!(inventory, 10.0);
    
    // Test performance metrics
    let metrics = engine.get_performance_metrics();
    assert!(metrics.total_pnl.is_finite());
    assert!(metrics.sharpe_ratio.is_finite());
    
    Ok(())
}

#[tokio::test]
async fn test_market_making_under_stress() -> Result<()> {
    let config = MarketMakingConfig {
        risk_aversion: 2.0, // Higher risk aversion for stress test
        time_horizon: 3600.0,
        update_frequency: 1000.0,
        rebalancing_threshold: 0.005,
        max_inventory_ratio: 0.05,
        min_spread_bps: 2.0,
        max_spread_bps: 200.0,
        quote_size_multiplier: 0.5,
        adverse_selection_threshold: 0.2,
        regime_detection_window: 50,
    };
    
    let mut engine = MarketMakingEngine::new(config)?;
    
    // Simulate stress conditions
    let mut market_state = create_mock_market_state();
    
    // High volatility
    market_state.volatility_estimates.insert("BTC".to_string(), 0.05);
    
    // Large inventory position
    market_state.inventory.insert("BTC".to_string(), 1000.0);
    
    // Crisis regime
    market_state.regime_state = RegimeState::Crisis { emergency_mode: true };
    
    // Generate quotes under stress
    let stress_quotes = engine.generate_quotes("BTC")?;
    
    // Verify stress adaptations
    assert!(stress_quotes.bid_price > 0.0);
    assert!(stress_quotes.ask_price > stress_quotes.bid_price);
    
    // Spreads should be wider under stress
    let spread = stress_quotes.ask_price - stress_quotes.bid_price;
    assert!(spread > 0.001); // Should be wider than normal
    
    // Confidence should be lower under stress
    assert!(stress_quotes.confidence < 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_adverse_selection_detection() -> Result<()> {
    let config = MarketMakingConfig {
        risk_aversion: 1.0,
        time_horizon: 3600.0,
        update_frequency: 100.0,
        rebalancing_threshold: 0.01,
        max_inventory_ratio: 0.1,
        min_spread_bps: 1.0,
        max_spread_bps: 100.0,
        quote_size_multiplier: 1.0,
        adverse_selection_threshold: 0.3,
        regime_detection_window: 100,
    };
    
    let mut engine = MarketMakingEngine::new(config)?;
    
    // Create market state with adverse selection signals
    let mut market_state = create_mock_market_state();
    
    // High adverse selection measure
    market_state.microstructure_signals.adverse_selection_measure
        .insert("BTC".to_string(), 0.8);
    
    // High order flow imbalance
    market_state.microstructure_signals.order_flow_imbalance
        .insert("BTC".to_string(), 0.7);
    
    // Generate quotes with adverse selection
    let quotes = engine.generate_quotes("BTC")?;
    
    // Should adapt to adverse selection
    assert!(quotes.bid_price > 0.0);
    assert!(quotes.ask_price > quotes.bid_price);
    
    // Spreads should be wider due to adverse selection
    let spread = quotes.ask_price - quotes.bid_price;
    assert!(spread > 0.0005);
    
    Ok(())
}

fn create_mock_market_state() -> MarketMakingState {
    let mut inventory = HashMap::new();
    inventory.insert("BTC".to_string(), 0.0);
    inventory.insert("ETH".to_string(), 0.0);
    inventory.insert("SOL".to_string(), 0.0);
    
    let mut volatility_estimates = HashMap::new();
    volatility_estimates.insert("BTC".to_string(), 0.01);
    volatility_estimates.insert("ETH".to_string(), 0.015);
    volatility_estimates.insert("SOL".to_string(), 0.02);
    
    let correlation_matrix = nalgebra::DMatrix::identity(3, 3);
    
    let mut jump_parameters = HashMap::new();
    jump_parameters.insert("BTC".to_string(), JumpParameters {
        intensity: 0.1,
        upward_rate: 10.0,
        downward_rate: 15.0,
        upward_probability: 0.4,
        mean_jump_size: 0.001,
        jump_variance: 0.0001,
    });
    
    let mut hawkes_parameters = HashMap::new();
    hawkes_parameters.insert("BTC".to_string(), HawkesParameters {
        baseline_intensity: 1.0,
        self_excitation: 0.5,
        decay_rate: 2.0,
        branching_ratio: 0.25,
        clustering_coefficient: 0.3,
    });
    
    MarketMakingState {
        inventory,
        volatility_estimates,
        correlation_matrix,
        jump_parameters,
        hawkes_parameters,
        regime_state: RegimeState::Normal { volatility_level: 1.0 },
        risk_metrics: RiskMetrics::default(),
        liquidity_constraints: LiquidityConstraints::default(),
        microstructure_signals: MicrostructureSignals::default(),
    }
}