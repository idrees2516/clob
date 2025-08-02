//! High-Frequency Market Microstructure Analytics Demo
//! 
//! This example demonstrates the comprehensive market microstructure analysis capabilities
//! including real-time liquidity metrics, order flow analysis, market impact modeling,
//! and high-frequency volatility estimation.

use std::collections::HashMap;
use advanced_trading_system::microstructure::*;
use advanced_trading_system::math::FixedPoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 High-Frequency Market Microstructure Analytics Demo");
    println!("=====================================================\n");

    // Initialize analyzers
    let mut liquidity_calculator = LiquidityMetricsCalculator::new(1000, 60000);
    let mut order_flow_analyzer = OrderFlowAnalyzer::new(1000, FixedPoint::from_float(10000.0), 60000);
    let mut impact_analyzer = MarketImpactAnalyzer::new(1000, 5000, 60000);
    let mut volatility_estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));

    println!("📊 1. Real-Time Liquidity Metrics");
    println!("----------------------------------");

    // Simulate market data updates
    let base_timestamp = 1640995200000u64; // 2022-01-01 00:00:00 UTC
    
    for i in 0..50 {
        let timestamp = base_timestamp + i * 1000; // 1 second intervals
        
        // Simulate realistic market data with some volatility
        let base_price = 100.0;
        let volatility = 0.02;
        let price_change = (i as f64 * 0.1).sin() * volatility;
        let spread = 0.05 + (i as f64 * 0.05).cos().abs() * 0.03;
        
        let bid_price = base_price + price_change - spread / 2.0;
        let ask_price = base_price + price_change + spread / 2.0;
        let bid_volume = 1000.0 + (i as f64 * 0.2).sin() * 200.0;
        let ask_volume = 1200.0 + (i as f64 * 0.3).cos() * 300.0;

        let market_data = MarketData {
            timestamp,
            bid_price: FixedPoint::from_float(bid_price),
            ask_price: FixedPoint::from_float(ask_price),
            bid_volume: FixedPoint::from_float(bid_volume),
            ask_volume: FixedPoint::from_float(ask_volume),
            last_trade_price: FixedPoint::from_float((bid_price + ask_price) / 2.0),
            last_trade_volume: FixedPoint::from_float(100.0),
            trade_direction: Some(if i % 2 == 0 { TradeDirection::Buy } else { TradeDirection::Sell }),
        };

        // Create order book snapshot
        let orderbook = OrderBookSnapshot {
            timestamp,
            bids: vec![
                OrderBookLevel {
                    price: FixedPoint::from_float(bid_price),
                    volume: FixedPoint::from_float(bid_volume),
                    order_count: 5,
                },
                OrderBookLevel {
                    price: FixedPoint::from_float(bid_price - 0.05),
                    volume: FixedPoint::from_float(bid_volume * 1.5),
                    order_count: 8,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: FixedPoint::from_float(ask_price),
                    volume: FixedPoint::from_float(ask_volume),
                    order_count: 6,
                },
                OrderBookLevel {
                    price: FixedPoint::from_float(ask_price + 0.05),
                    volume: FixedPoint::from_float(ask_volume * 1.3),
                    order_count: 7,
                },
            ],
        };

        // Update liquidity calculator
        liquidity_calculator.update_market_data(market_data.clone());
        liquidity_calculator.update_orderbook(orderbook);

        // Update volatility estimator
        volatility_estimator.update_market_data(market_data.clone()).unwrap();

        // Simulate trades
        if i % 3 == 0 {
            let trade_price = if i % 6 == 0 { ask_price } else { bid_price };
            let trade_direction = if i % 6 == 0 { TradeDirection::Buy } else { TradeDirection::Sell };
            let trade_volume = 200.0 + (i as f64 * 0.4).sin().abs() * 300.0;

            let trade = Trade {
                timestamp: timestamp + 100, // Slightly after market data
                price: FixedPoint::from_float(trade_price),
                volume: FixedPoint::from_float(trade_volume),
                direction: trade_direction,
                trade_id: i + 1,
            };

            // Update analyzers with trade
            liquidity_calculator.update_trade(trade.clone());
            order_flow_analyzer.update_trade(trade.clone()).unwrap();
            impact_analyzer.analyze_trade_impact(trade).unwrap();
        }
    }

    // Calculate and display liquidity metrics
    if let Ok(liquidity_metrics) = liquidity_calculator.calculate_metrics() {
        println!("Bid-Ask Spread (absolute): {:.4}", liquidity_metrics.bid_ask_spread_absolute.to_float());
        println!("Bid-Ask Spread (relative): {:.4}%", liquidity_metrics.bid_ask_spread_relative.to_float() * 100.0);
        println!("Effective Spread: {:.4}", liquidity_metrics.effective_spread.to_float());
        println!("Realized Spread: {:.4}", liquidity_metrics.realized_spread.to_float());
        println!("Depth at Best Bid: {:.0}", liquidity_metrics.depth_at_best_bid.to_float());
        println!("Depth at Best Ask: {:.0}", liquidity_metrics.depth_at_best_ask.to_float());
        println!("Price Impact: {:.4}", liquidity_metrics.price_impact.to_float());
        println!("Market Quality Score: {:.2}%", liquidity_metrics.market_quality_score.to_float() * 100.0);
        
        if let Ok(summary) = liquidity_calculator.get_liquidity_summary() {
            println!("Liquidity Summary - Spread: {} bps, Depth: ${}, Quality: {}/100",
                    summary.spread_bps, summary.depth_usd, summary.quality_score);
        }
    }

    println!("\n📈 2. Order Flow Analysis");
    println!("-------------------------");

    // Calculate order flow metrics
    if let Ok(order_flow_metrics) = order_flow_analyzer.calculate_metrics() {
        println!("Order Flow Imbalance: {:.2}%", order_flow_metrics.order_flow_imbalance.to_float() * 100.0);
        println!("VPIN: {:.2}%", order_flow_metrics.vpin.to_float() * 100.0);
        println!("Toxic Flow Probability: {:.2}%", order_flow_metrics.toxic_flow_probability.to_float() * 100.0);
        println!("Institutional Flow Ratio: {:.2}%", order_flow_metrics.institutional_flow_ratio.to_float() * 100.0);
        println!("Buy Volume Ratio: {:.2}%", order_flow_metrics.buy_volume_ratio.to_float() * 100.0);
        println!("Sell Volume Ratio: {:.2}%", order_flow_metrics.sell_volume_ratio.to_float() * 100.0);
        println!("Classification Accuracy: {:.1}%", order_flow_metrics.trade_classification_accuracy.to_float() * 100.0);
        
        if let Ok(summary) = order_flow_analyzer.get_order_flow_summary() {
            println!("Order Flow Summary - Imbalance: {}%, VPIN: {}%, Toxic: {}%",
                    summary.imbalance_pct, summary.vpin_pct, summary.toxic_flow_pct);
        }
    }

    println!("\n💥 3. Market Impact Modeling");
    println!("-----------------------------");

    // Calculate market impact metrics
    if let Ok(impact_metrics) = impact_analyzer.calculate_metrics() {
        println!("Temporary Impact: {:.4}", impact_metrics.temporary_impact.to_float());
        println!("Permanent Impact: {:.4}", impact_metrics.permanent_impact.to_float());
        println!("Total Impact: {:.4}", impact_metrics.total_impact.to_float());
        println!("Impact Decay Rate: {:.4}", impact_metrics.impact_decay_rate.to_float());
        println!("Participation Rate: {:.2}%", impact_metrics.participation_rate.to_float() * 100.0);
        println!("Volume Impact Coefficient: {:.6}", impact_metrics.volume_impact_coefficient.to_float());
        
        if let Ok(summary) = impact_analyzer.get_impact_summary() {
            println!("Impact Summary - Temporary: {} bps, Permanent: {} bps, Total: {} bps",
                    summary.temporary_impact_bps, summary.permanent_impact_bps, summary.total_impact_bps);
        }
    }

    // Demonstrate impact prediction
    let predicted_impact = impact_analyzer.predict_impact(
        FixedPoint::from_float(5000.0), // Large order
        FixedPoint::from_float(100.0),
        TradeDirection::Buy,
    ).unwrap();
    println!("Predicted Impact for 5000 share buy order: {:.4}", predicted_impact.to_float());

    // Estimate model parameters
    if let Ok(params) = impact_analyzer.estimate_model_parameters() {
        println!("Model Parameters:");
        println!("  Temporary Impact Coefficient: {:.4}", params.temporary_impact_coefficient.to_float());
        println!("  Temporary Impact Exponent: {:.4}", params.temporary_impact_exponent.to_float());
        println!("  Permanent Impact Coefficient: {:.4}", params.permanent_impact_coefficient.to_float());
        println!("  Decay Rate: {:.4}", params.decay_rate.to_float());
    }

    println!("\n📊 4. High-Frequency Volatility Estimation");
    println!("-------------------------------------------");

    // Calculate volatility metrics
    if let Ok(volatility_metrics) = volatility_estimator.calculate_metrics(60000) {
        println!("Realized Volatility: {:.2}%", volatility_metrics.realized_volatility.to_float() * 100.0);
        println!("Bi-Power Variation: {:.2}%", volatility_metrics.bi_power_variation.to_float() * 100.0);
        println!("Jump-Robust Volatility: {:.2}%", volatility_metrics.jump_robust_volatility.to_float() * 100.0);
        println!("Microstructure Noise Variance: {:.6}", volatility_metrics.microstructure_noise_variance.to_float());
        println!("Optimal Sampling Frequency: {} ms", volatility_metrics.optimal_sampling_frequency);
        println!("Volatility of Volatility: {:.2}%", volatility_metrics.volatility_of_volatility.to_float() * 100.0);
        
        if let Ok(summary) = volatility_estimator.get_volatility_summary(60000) {
            println!("Volatility Summary - RV: {}%, Jump-Robust: {}%, Optimal Sampling: {} ms",
                    summary.realized_vol_pct, summary.jump_robust_vol_pct, summary.optimal_sampling_ms);
        }
    }

    // Display intraday volatility pattern
    println!("\nIntraday Volatility Pattern (by hour):");
    let pattern = volatility_estimator.get_intraday_pattern();
    for (hour, hourly_pattern) in pattern.iter().enumerate() {
        if hourly_pattern.observation_count > 0 {
            println!("  Hour {}: {:.2}% (std: {:.2}%, observations: {})",
                    hour,
                    hourly_pattern.average_volatility.to_float() * 100.0,
                    hourly_pattern.volatility_std.to_float() * 100.0,
                    hourly_pattern.observation_count);
        }
    }

    // Detect recent jumps
    let recent_jumps = volatility_estimator.detect_recent_jumps(30000);
    println!("\nRecent Jumps Detected: {}", recent_jumps.len());
    for (i, jump) in recent_jumps.iter().take(5).enumerate() {
        println!("  Jump {}: {:.4} at timestamp {}", i + 1, jump.log_return.to_float(), jump.timestamp);
    }

    // Calculate optimal sampling parameters
    if let Ok(optimal_params) = volatility_estimator.calculate_optimal_sampling_frequency() {
        println!("\nOptimal Sampling Parameters:");
        println!("  Optimal Frequency: {} ms", optimal_params.optimal_frequency_ms);
        println!("  Noise-to-Signal Ratio: {:.4}", optimal_params.noise_to_signal_ratio.to_float());
        println!("  Efficiency Gain: {:.2}x", optimal_params.efficiency_gain.to_float());
    }

    println!("\n🎯 5. Integrated Analysis");
    println!("-------------------------");

    // Demonstrate how different metrics can be combined for trading decisions
    if let (Ok(liquidity), Ok(order_flow), Ok(impact), Ok(volatility)) = (
        liquidity_calculator.get_liquidity_summary(),
        order_flow_analyzer.get_order_flow_summary(),
        impact_analyzer.get_impact_summary(),
        volatility_estimator.get_volatility_summary(60000),
    ) {
        println!("Market Condition Assessment:");
        
        // Liquidity assessment
        let liquidity_score = if liquidity.spread_bps < 10 && liquidity.depth_usd > 100000 {
            "High"
        } else if liquidity.spread_bps < 20 && liquidity.depth_usd > 50000 {
            "Medium"
        } else {
            "Low"
        };
        println!("  Liquidity: {} (spread: {} bps, depth: ${})", 
                liquidity_score, liquidity.spread_bps, liquidity.depth_usd);

        // Order flow assessment
        let flow_bias = if order_flow.imbalance_pct > 10 {
            "Bullish"
        } else if order_flow.imbalance_pct < -10 {
            "Bearish"
        } else {
            "Neutral"
        };
        println!("  Order Flow: {} (imbalance: {}%, toxic: {}%)", 
                flow_bias, order_flow.imbalance_pct, order_flow.toxic_flow_pct);

        // Impact assessment
        let impact_level = if impact.total_impact_bps < 5 {
            "Low"
        } else if impact.total_impact_bps < 15 {
            "Medium"
        } else {
            "High"
        };
        println!("  Market Impact: {} ({} bps total)", impact_level, impact.total_impact_bps);

        // Volatility assessment
        let vol_regime = if volatility.realized_vol_pct < 20 {
            "Low Vol"
        } else if volatility.realized_vol_pct < 40 {
            "Normal Vol"
        } else {
            "High Vol"
        };
        println!("  Volatility: {} ({}% realized)", vol_regime, volatility.realized_vol_pct);

        // Trading recommendation
        println!("\nTrading Recommendation:");
        if liquidity.quality_score > 70 && order_flow.toxic_flow_pct < 30 && impact.total_impact_bps < 10 {
            println!("  ✅ FAVORABLE - Good conditions for active trading");
            println!("     - High liquidity and low impact");
            println!("     - Manageable toxic flow");
            println!("     - Consider market making strategies");
        } else if order_flow.toxic_flow_pct > 60 || impact.total_impact_bps > 20 {
            println!("  ⚠️  CAUTION - Challenging market conditions");
            println!("     - High toxic flow or market impact detected");
            println!("     - Consider reducing position sizes");
            println!("     - Use passive execution strategies");
        } else {
            println!("  ℹ️  NEUTRAL - Standard market conditions");
            println!("     - Monitor conditions closely");
            println!("     - Use adaptive execution strategies");
        }
    }

    println!("\n✅ Microstructure Analytics Demo Complete!");
    println!("This demo showcased comprehensive market microstructure analysis");
    println!("capabilities for high-frequency trading applications.");

    Ok(())
}