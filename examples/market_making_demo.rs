use clob_engine::market_making::*;
use clob_engine::error::Result;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Market Making Engine Demo");
    println!("============================");
    
    // Demo 1: Avellaneda-Stoikov Model
    println!("\n📊 Demo 1: Avellaneda-Stoikov Optimal Market Making");
    demo_avellaneda_stoikov().await?;
    
    // Demo 2: Guéant-Lehalle-Tapia Multi-Asset Framework
    println!("\n📈 Demo 2: Guéant-Lehalle-Tapia Multi-Asset Framework");
    demo_gueant_lehalle_tapia().await?;
    
    // Demo 3: Cartea-Jaimungal Jump-Diffusion Model
    println!("\n⚡ Demo 3: Cartea-Jaimungal Jump-Diffusion Model");
    demo_cartea_jaimungal().await?;
    
    // Demo 4: High-Frequency Quoting Under Liquidity Constraints
    println!("\n🏃 Demo 4: High-Frequency Quoting Under Liquidity Constraints");
    demo_high_frequency_quoting().await?;
    
    // Demo 5: Comprehensive Market Making Engine
    println!("\n🎯 Demo 5: Comprehensive Market Making Engine");
    demo_comprehensive_engine().await?;
    
    println!("\n✅ All demos completed successfully!");
    
    Ok(())
}

async fn demo_avellaneda_stoikov() -> Result<()> {
    println!("Creating Avellaneda-Stoikov engine with risk aversion γ=1.0, T=1 hour");
    
    let mut engine = avellaneda_stoikov::AvellanedaStoikovEngine::new(1.0, 3600.0)?;
    let market_state = create_demo_market_state();
    
    println!("Initial quotes (no inventory):");
    let initial_quotes = engine.generate_quotes("BTC", &market_state)?;
    print_quotes(&initial_quotes, "BTC");
    
    // Simulate inventory accumulation
    println!("\nSimulating inventory accumulation...");
    for inventory in [0.0, 50.0, 100.0, 200.0] {
        engine.update_inventory("BTC", inventory)?;
        let quotes = engine.generate_quotes("BTC", &market_state)?;
        println!("Inventory: {:.0}, Bid: {:.4}, Ask: {:.4}, Spread: {:.4}", 
                inventory, quotes.bid_price, quotes.ask_price, 
                quotes.ask_price - quotes.bid_price);
    }
    
    // Simulate volatility changes
    println!("\nSimulating volatility impact...");
    let mut volatile_state = market_state.clone();
    for vol in [0.01, 0.02, 0.05, 0.1] {
        volatile_state.volatility_estimates.insert("BTC".to_string(), vol);
        let quotes = engine.generate_quotes("BTC", &volatile_state)?;
        println!("Volatility: {:.3}, Spread: {:.4}, Confidence: {:.3}", 
                vol, quotes.ask_price - quotes.bid_price, quotes.confidence);
    }
    
    Ok(())
}

async fn demo_gueant_lehalle_tapia() -> Result<()> {
    println!("Creating Guéant-Lehalle-Tapia multi-asset engine");
    
    let mut engine = gueant_lehalle_tapia::GuéantLehalleTapiaEngine::new(1.0)?;
    let market_state = create_demo_market_state();
    
    // Generate quotes for multiple assets
    println!("Multi-asset quote generation:");
    for symbol in ["BTC", "ETH", "SOL"] {
        let quotes = engine.generate_quotes(symbol, &market_state)?;
        print_quotes(&quotes, symbol);
    }
    
    // Demonstrate portfolio optimization
    println!("\nPortfolio optimization:");
    let optimal_weights = engine.optimize_portfolio(&market_state)?;
    for (asset, weight) in optimal_weights {
        println!("{}: {:.3}", asset, weight);
    }
    
    // Demonstrate cross-asset hedging
    println!("\nCross-asset hedge orders:");
    engine.update_inventory("BTC", 100.0)?;
    let hedge_orders = engine.calculate_hedge_orders(&market_state)?;
    for hedge in hedge_orders {
        println!("Hedge {} with {:.2} shares, ratio: {:.3}, effectiveness: {:.3}",
                hedge.symbol, hedge.quantity, hedge.hedge_ratio, hedge.expected_effectiveness);
    }
    
    // Simulate correlation changes
    println!("\nCorrelation impact simulation:");
    let returns_matrix = nalgebra::DMatrix::from_row_slice(3, 3, &[
        0.01, 0.008, 0.005,
        0.008, 0.012, 0.006,
        0.005, 0.006, 0.015,
    ]);
    
    engine.update_correlations(&returns_matrix)?;
    println!("Updated correlations and recalculated quotes");
    
    Ok(())
}

async fn demo_cartea_jaimungal() -> Result<()> {
    println!("Creating Cartea-Jaimungal jump-diffusion engine");
    
    let mut engine = cartea_jaimungal::CarteaJaimungalEngine::new(1.0)?;
    let market_state = create_demo_market_state();
    
    println!("Initial quotes (no jumps detected):");
    let initial_quotes = engine.generate_quotes("BTC", &market_state)?;
    print_quotes(&initial_quotes, "BTC");
    
    // Simulate price data with jumps
    println!("\nSimulating price data with jumps...");
    let price_data_with_jumps = vec![
        100.0, 100.1, 100.05, 102.5, 102.3, 102.1, 101.9, 104.2, 104.0
    ];
    let timestamps: Vec<u64> = (1000..1009).collect();
    
    engine.update_market_data("BTC", &price_data_with_jumps, &timestamps)?;
    
    println!("Quotes after jump detection:");
    let jump_quotes = engine.generate_quotes("BTC", &market_state)?;
    print_quotes(&jump_quotes, "BTC");
    
    println!("Jump impact - Spread widening: {:.4} -> {:.4}",
            initial_quotes.ask_price - initial_quotes.bid_price,
            jump_quotes.ask_price - jump_quotes.bid_price);
    
    // Demonstrate regime switching
    println!("\nRegime switching demonstration:");
    let regime_data = vec![
        100.0, 99.8, 100.2, 99.9, 100.1, // Normal regime
        95.0, 98.0, 92.0, 96.0, 89.0,    // Crisis regime
    ];
    let regime_timestamps: Vec<u64> = (2000..2010).collect();
    
    engine.update_market_data("BTC", &regime_data, &regime_timestamps)?;
    let crisis_quotes = engine.generate_quotes("BTC", &market_state)?;
    
    println!("Crisis regime quotes:");
    print_quotes(&crisis_quotes, "BTC");
    
    Ok(())
}

async fn demo_high_frequency_quoting() -> Result<()> {
    println!("Creating high-frequency quoting engine with liquidity constraints");
    
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
    let market_state = create_demo_market_state();
    
    println!("Initial high-frequency quotes:");
    let hf_quotes = engine.generate_quotes("BTC", &market_state)?;
    print_quotes(&hf_quotes, "BTC");
    
    // Simulate order flow
    println!("\nSimulating order flow updates...");
    let order_flow = vec![
        high_frequency_quoting::OrderArrival {
            timestamp: 1000,
            order_type: high_frequency_quoting::OrderType::Market,
            side: high_frequency_quoting::OrderSide::Buy,
            size: 100.0,
            price: 100.0,
            intensity_contribution: 0.1,
        },
        high_frequency_quoting::OrderArrival {
            timestamp: 1001,
            order_type: high_frequency_quoting::OrderType::Market,
            side: high_frequency_quoting::OrderSide::Buy,
            size: 150.0,
            price: 100.05,
            intensity_contribution: 0.15,
        },
        high_frequency_quoting::OrderArrival {
            timestamp: 1002,
            order_type: high_frequency_quoting::OrderType::Limit,
            side: high_frequency_quoting::OrderSide::Sell,
            size: 80.0,
            price: 100.1,
            intensity_contribution: 0.08,
        },
    ];
    
    engine.update_order_flow(order_flow)?;
    
    let updated_quotes = engine.generate_quotes("BTC", &market_state)?;
    println!("Quotes after order flow update:");
    print_quotes(&updated_quotes, "BTC");
    
    // Demonstrate liquidity constraint impact
    println!("\nDemonstrating liquidity constraint impact:");
    engine.update_inventory("BTC", 800.0)?; // Large position
    let constrained_quotes = engine.generate_quotes("BTC", &market_state)?;
    
    println!("Quotes with large inventory (liquidity constrained):");
    print_quotes(&constrained_quotes, "BTC");
    
    Ok(())
}

async fn demo_comprehensive_engine() -> Result<()> {
    println!("Creating comprehensive market making engine with all models");
    
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
    
    // Create comprehensive market data
    let market_data = MarketData {
        timestamp: 1000,
        prices: [
            ("BTC".to_string(), 100.0),
            ("ETH".to_string(), 3000.0),
            ("SOL".to_string(), 150.0),
        ].iter().cloned().collect(),
        volumes: [
            ("BTC".to_string(), 1000.0),
            ("ETH".to_string(), 5000.0),
            ("SOL".to_string(), 2000.0),
        ].iter().cloned().collect(),
        bid_ask_spreads: [
            ("BTC".to_string(), 0.01),
            ("ETH".to_string(), 0.5),
            ("SOL".to_string(), 0.05),
        ].iter().cloned().collect(),
        order_book_depth: HashMap::new(),
        trade_history: Vec::new(),
    };
    
    engine.update_market_data(&market_data)?;
    
    println!("Ensemble quotes from all models:");
    let ensemble_quotes = engine.generate_quotes("BTC")?;
    print_quotes(&ensemble_quotes, "BTC");
    
    // Simulate trading activity
    println!("\nSimulating trading activity...");
    let trades = vec![
        Trade {
            symbol: "BTC".to_string(),
            price: 100.05,
            quantity: 10.0,
            side: TradeSide::Buy,
            timestamp: 1001,
        },
        Trade {
            symbol: "BTC".to_string(),
            price: 99.95,
            quantity: 15.0,
            side: TradeSide::Sell,
            timestamp: 1002,
        },
        Trade {
            symbol: "ETH".to_string(),
            price: 3001.0,
            quantity: 5.0,
            side: TradeSide::Buy,
            timestamp: 1003,
        },
    ];
    
    for trade in trades {
        engine.execute_trade(&trade)?;
        println!("Executed: {} {} {} @ {}", 
                trade.quantity, 
                match trade.side { TradeSide::Buy => "BUY", TradeSide::Sell => "SELL" },
                trade.symbol, 
                trade.price);
    }
    
    // Show performance metrics
    println!("\nPerformance metrics:");
    let metrics = engine.get_performance_metrics();
    println!("Total P&L: {:.2}", metrics.total_pnl);
    println!("Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("Max Drawdown: {:.3}", metrics.max_drawdown);
    println!("VaR (95%): {:.2}", metrics.var_95);
    println!("Fill Ratio: {:.3}", metrics.fill_ratio);
    println!("Adverse Selection Cost: {:.4}", metrics.adverse_selection_cost);
    
    // Demonstrate portfolio optimization
    println!("\nPortfolio optimization:");
    let optimal_allocation = engine.optimize_portfolio()?;
    for (asset, weight) in optimal_allocation {
        println!("{}: {:.3}", asset, weight);
    }
    
    // Demonstrate hedging
    println!("\nDynamic hedging:");
    let hedge_orders = engine.perform_hedging()?;
    for hedge in hedge_orders {
        println!("Hedge {} with {:.2} shares (ratio: {:.3})",
                hedge.symbol, hedge.quantity, hedge.hedge_ratio);
    }
    
    Ok(())
}

fn create_demo_market_state() -> MarketMakingState {
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

fn print_quotes(quotes: &QuoteSet, symbol: &str) {
    println!("{}: Bid: {:.4} ({:.0}) | Ask: {:.4} ({:.0}) | Spread: {:.4} | Confidence: {:.3} | Expected Profit: {:.6}",
            symbol,
            quotes.bid_price,
            quotes.bid_size,
            quotes.ask_price,
            quotes.ask_size,
            quotes.ask_price - quotes.bid_price,
            quotes.confidence,
            quotes.expected_profit);
}