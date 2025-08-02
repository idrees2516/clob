//! Order Flow Modeling with Hawkes Processes Demo
//! 
//! This example demonstrates the advanced order flow modeling capabilities
//! using Hawkes processes for real-time intensity forecasting and quote optimization.

use matching_engine::math::{
    FixedPoint, OrderType, OrderFlowEvent, OrderFlowHawkesParams, 
    RealTimeOrderFlowAnalyzer, DirectionalSignal, UrgencySignal
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Order Flow Modeling with Hawkes Processes Demo");
    println!("================================================\n");

    // Initialize Hawkes process parameters for order flow modeling
    let params = OrderFlowHawkesParams::new(
        FixedPoint::from_float(2.0),   // baseline_buy_intensity
        FixedPoint::from_float(1.8),   // baseline_sell_intensity
        FixedPoint::from_float(0.4),   // self_excitation
        FixedPoint::from_float(0.15),  // cross_excitation
        FixedPoint::from_float(3.0),   // decay_rate
        FixedPoint::from_float(2.0),   // impact_decay
        FixedPoint::from_float(0.02),  // volume_impact_scale
        FixedPoint::from_float(0.08),  // spread_impact_factor
    )?;

    println!("📊 Hawkes Process Parameters:");
    println!("   • Buy Intensity: {:.3}", params.hawkes_params.baseline_intensities[0].to_float());
    println!("   • Sell Intensity: {:.3}", params.hawkes_params.baseline_intensities[1].to_float());
    println!("   • Self-Excitation: {:.3}", params.self_excitation.to_float());
    println!("   • Cross-Excitation: {:.3}", params.cross_excitation.to_float());
    println!();

    // Create real-time order flow analyzer
    let mut analyzer = RealTimeOrderFlowAnalyzer::new(params, 200, 10);

    // Simulate realistic order flow scenarios
    println!("🎯 Simulating Order Flow Scenarios");
    println!("==================================\n");

    // Scenario 1: Normal market conditions
    println!("📈 Scenario 1: Normal Market Conditions");
    simulate_normal_market(&mut analyzer)?;

    // Scenario 2: Buy pressure buildup
    println!("\n📈 Scenario 2: Buy Pressure Buildup");
    simulate_buy_pressure(&mut analyzer)?;

    // Scenario 3: Market order clustering
    println!("\n📈 Scenario 3: Market Order Clustering");
    simulate_market_order_clustering(&mut analyzer)?;

    // Scenario 4: Adverse selection scenario
    println!("\n📈 Scenario 4: Adverse Selection Detection");
    simulate_adverse_selection(&mut analyzer)?;

    // Generate final recommendations
    println!("\n🎯 Final Quote Recommendations");
    println!("=============================");
    
    let recommendations = analyzer.generate_quote_recommendations(
        FixedPoint::from_float(100.0),  // current_mid_price
        FixedPoint::from_float(0.04),   // base_spread
        FixedPoint::from_float(0.3),    // risk_aversion
        FixedPoint::from_float(0.5),    // forecast_horizon
    )?;

    println!("💰 Optimal Quotes:");
    println!("   • Bid: ${:.4} (Size: {:.0})", 
             recommendations.bid_price.to_float(), 
             recommendations.bid_size.to_float());
    println!("   • Ask: ${:.4} (Size: {:.0})", 
             recommendations.ask_price.to_float(), 
             recommendations.ask_size.to_float());
    println!("   • Spread: {:.4} bps", recommendations.spread.to_float() * 10000.0);
    println!("   • Skew: {:.4}", recommendations.skew.to_float());
    println!("   • Confidence: {:.1}%", recommendations.confidence.to_float() * 100.0);
    println!("   • Update Frequency: {:.1} Hz", recommendations.recommended_update_frequency.to_float());

    println!("\n📊 Spread Decomposition:");
    println!("   • Intensity Adjustment: {:.4} bps", 
             recommendations.intensity_adjustment.to_float() * 10000.0);
    println!("   • Impact Adjustment: {:.4} bps", 
             recommendations.impact_adjustment.to_float() * 10000.0);
    println!("   • Adverse Selection: {:.4} bps", 
             recommendations.adverse_selection_adjustment.to_float() * 10000.0);

    // Performance metrics
    let metrics = analyzer.intensity_calculator.get_real_time_metrics();
    println!("\n⚡ Real-Time Metrics:");
    println!("   • Buy Intensity: {:.3}", metrics.buy_intensity.to_float());
    println!("   • Sell Intensity: {:.3}", metrics.sell_intensity.to_float());
    println!("   • Buy/Sell Imbalance: {:.3}", metrics.buy_sell_imbalance.to_float());
    println!("   • Market Order Pressure: {:.3}", metrics.market_order_pressure.to_float());
    println!("   • Intensity Volatility: {:.3}", metrics.intensity_volatility.to_float());

    println!("\n✅ Order Flow Modeling Demo Complete!");
    Ok(())
}

fn simulate_normal_market(analyzer: &mut RealTimeOrderFlowAnalyzer) -> Result<(), Box<dyn std::error::Error>> {
    let mut current_time = FixedPoint::zero();
    let mut current_price = FixedPoint::from_float(100.0);
    
    // Simulate 50 events with balanced buy/sell flow
    for i in 0..50 {
        let order_type = match i % 4 {
            0 => OrderType::Buy,
            1 => OrderType::Sell,
            2 => OrderType::Buy,
            3 => OrderType::Sell,
        };
        
        current_time = current_time + FixedPoint::from_float(0.1);
        current_price = current_price + FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.002);
        
        let event = OrderFlowEvent::new(
            current_time,
            order_type,
            FixedPoint::from_float(100.0 + rand::random::<f64>() * 50.0),
            current_price,
            FixedPoint::from_float(0.02 + rand::random::<f64>() * 0.01),
            FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.2),
            FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.001),
        );
        
        analyzer.process_event(event)?;
    }
    
    let signals = analyzer.generate_trading_signals();
    println!("   • Direction: {:?}", signals.directional_signal);
    println!("   • Urgency: {:?}", signals.urgency_signal);
    println!("   • Signal Strength: {:.2}", signals.signal_strength);
    
    Ok(())
}

fn simulate_buy_pressure(analyzer: &mut RealTimeOrderFlowAnalyzer) -> Result<(), Box<dyn std::error::Error>> {
    let mut current_time = FixedPoint::from_float(5.0);
    let mut current_price = FixedPoint::from_float(100.05);
    
    // Simulate increasing buy pressure
    for i in 0..30 {
        let order_type = if i < 20 {
            // 80% buy orders
            if rand::random::<f64>() < 0.8 { OrderType::Buy } else { OrderType::Sell }
        } else {
            // Add some market buy orders
            if rand::random::<f64>() < 0.6 { OrderType::MarketBuy } else { OrderType::Buy }
        };
        
        current_time = current_time + FixedPoint::from_float(0.05);
        current_price = current_price + FixedPoint::from_float(0.001); // Trending up
        
        let volume = if order_type == OrderType::MarketBuy {
            FixedPoint::from_float(200.0 + rand::random::<f64>() * 100.0) // Larger market orders
        } else {
            FixedPoint::from_float(80.0 + rand::random::<f64>() * 40.0)
        };
        
        let event = OrderFlowEvent::new(
            current_time,
            order_type,
            volume,
            current_price,
            FixedPoint::from_float(0.025 + rand::random::<f64>() * 0.015),
            FixedPoint::from_float(0.3 + rand::random::<f64>() * 0.2), // Positive imbalance
            FixedPoint::from_float(0.0005 + rand::random::<f64>() * 0.001),
        );
        
        analyzer.process_event(event)?;
    }
    
    let signals = analyzer.generate_trading_signals();
    println!("   • Direction: {:?}", signals.directional_signal);
    println!("   • Urgency: {:?}", signals.urgency_signal);
    println!("   • Signal Strength: {:.2}", signals.signal_strength);
    
    if matches!(signals.directional_signal, DirectionalSignal::Bullish) {
        println!("   ✅ Successfully detected buy pressure!");
    }
    
    Ok(())
}

fn simulate_market_order_clustering(analyzer: &mut RealTimeOrderFlowAnalyzer) -> Result<(), Box<dyn std::error::Error>> {
    let mut current_time = FixedPoint::from_float(8.0);
    let mut current_price = FixedPoint::from_float(100.08);
    
    // Simulate clustering of market orders
    for cluster in 0..3 {
        // Each cluster has 5-8 market orders in quick succession
        let cluster_size = 5 + (rand::random::<usize>() % 4);
        let cluster_type = if cluster % 2 == 0 { OrderType::MarketBuy } else { OrderType::MarketSell };
        
        for i in 0..cluster_size {
            current_time = current_time + FixedPoint::from_float(0.01); // Very fast succession
            
            let price_impact = if cluster_type == OrderType::MarketBuy {
                FixedPoint::from_float(0.002)
            } else {
                FixedPoint::from_float(-0.002)
            };
            current_price = current_price + price_impact;
            
            let event = OrderFlowEvent::new(
                current_time,
                cluster_type,
                FixedPoint::from_float(150.0 + rand::random::<f64>() * 100.0),
                current_price,
                FixedPoint::from_float(0.03 + rand::random::<f64>() * 0.02),
                FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.4),
                price_impact,
            );
            
            analyzer.process_event(event)?;
        }
        
        // Add some gap between clusters
        current_time = current_time + FixedPoint::from_float(0.5);
    }
    
    let signals = analyzer.generate_trading_signals();
    println!("   • Clustering Detected: {}", signals.clustering_detected);
    println!("   • Urgency: {:?}", signals.urgency_signal);
    
    if signals.clustering_detected {
        println!("   ✅ Successfully detected market order clustering!");
    }
    
    Ok(())
}

fn simulate_adverse_selection(analyzer: &mut RealTimeOrderFlowAnalyzer) -> Result<(), Box<dyn std::error::Error>> {
    let mut current_time = FixedPoint::from_float(12.0);
    let mut current_price = FixedPoint::from_float(100.12);
    
    // Simulate informed trading pattern (orders followed by price moves in same direction)
    for i in 0..25 {
        let is_informed = rand::random::<f64>() < 0.7; // 70% informed trades
        
        let order_type = if is_informed {
            // Informed trades: buy before price goes up, sell before price goes down
            let will_go_up = rand::random::<f64>() < 0.6;
            if will_go_up { OrderType::Buy } else { OrderType::Sell }
        } else {
            // Random trades
            if rand::random::<f64>() < 0.5 { OrderType::Buy } else { OrderType::Sell }
        };
        
        current_time = current_time + FixedPoint::from_float(0.2);
        
        // Price moves in direction of informed trades
        let price_change = if is_informed {
            match order_type {
                OrderType::Buy => FixedPoint::from_float(0.003 + rand::random::<f64>() * 0.002),
                OrderType::Sell => FixedPoint::from_float(-0.003 - rand::random::<f64>() * 0.002),
                _ => FixedPoint::zero(),
            }
        } else {
            FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.001)
        };
        
        current_price = current_price + price_change;
        
        let event = OrderFlowEvent::new(
            current_time,
            order_type,
            FixedPoint::from_float(120.0 + rand::random::<f64>() * 80.0),
            current_price,
            FixedPoint::from_float(0.025 + rand::random::<f64>() * 0.015),
            FixedPoint::from_float((rand::random::<f64>() - 0.5) * 0.3),
            price_change,
        );
        
        analyzer.process_event(event)?;
    }
    
    let signals = analyzer.generate_trading_signals();
    let results = analyzer.get_analysis_results();
    
    println!("   • Toxicity Warning: {}", signals.toxicity_warning);
    println!("   • Order Flow Toxicity: {:.3}", results.order_flow_toxicity.to_float());
    
    if signals.toxicity_warning {
        println!("   ⚠️  Successfully detected adverse selection!");
    }
    
    Ok(())
}