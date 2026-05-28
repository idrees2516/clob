//! TWAP Execution Engine Demo
//! 
//! Demonstrates the sophisticated TWAP execution system with:
//! - Core TWAP implementation with time bucketing
//! - Volume forecasting and adaptive scheduling
//! - Market impact modeling and optimization
//! - Adaptive execution control with contingency planning

use advanced_trading_system::execution::{
    twap::{TWAPExecutor, TWAPConfig},
    volume_forecasting::{VolumeForecaster, VolumeDataPoint, AdaptationParameters},
    market_impact::{MarketImpactModel, MarketImpactParams},
    execution_control::{
        AdaptiveExecutionController, AdaptiveControlConfig, ContingencyPlan,
        ContingencyTrigger, ContingencyAction, ContingencyStatus, PriceMoveDirection,
    },
    Order, OrderSide, OrderType, TimeInForce, MarketState, MarketConditions,
    VolatilityRegime, LiquidityLevel, TrendDirection, MarketHours, NewsImpact,
};
use advanced_trading_system::math::fixed_point::FixedPoint;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 TWAP Execution Engine Demo");
    println!("=====================================\n");

    // Demo 1: Basic TWAP Execution
    demo_basic_twap_execution()?;
    
    // Demo 2: Volume Forecasting and Adaptation
    demo_volume_forecasting()?;
    
    // Demo 3: Market Impact Optimization
    demo_market_impact_optimization()?;
    
    // Demo 4: Adaptive Execution Control
    demo_adaptive_execution_control()?;
    
    // Demo 5: Complete Integration
    demo_complete_integration()?;

    println!("\n✅ All TWAP execution demos completed successfully!");
    Ok(())
}

fn demo_basic_twap_execution() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Basic TWAP Execution");
    println!("--------------------------------");

    // Create TWAP executor with configuration
    let mut twap_executor = TWAPExecutor::new();
    let config = TWAPConfig {
        execution_horizon: 3600,  // 1 hour
        num_buckets: 12,          // 5-minute buckets
        target_participation_rate: FixedPoint::from_float(0.15), // 15%
        max_participation_rate: FixedPoint::from_float(0.30),    // 30%
        min_order_size: 100,
        max_order_size: 5000,
        adaptive_scheduling: true,
        timing_randomization: FixedPoint::from_float(0.05), // 5%
    };

    // Create test order
    let order = Order {
        id: 1,
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 50000, // 50,000 shares
        price: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        time_in_force: TimeInForce::Day,
    };

    // Create market conditions
    let market_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    // Create execution plan
    let execution_plan = twap_executor.create_execution_plan(
        order.clone(),
        config,
        &market_conditions,
        None,
    )?;

    println!("📋 Execution Plan Created:");
    println!("  • Total Quantity: {} shares", execution_plan.order.quantity);
    println!("  • Number of Buckets: {}", execution_plan.time_buckets.len());
    println!("  • Execution Horizon: {} seconds", execution_plan.config.execution_horizon);
    
    // Display bucket allocation
    println!("\n🪣 Time Bucket Allocation:");
    for (i, bucket) in execution_plan.time_buckets.iter().take(5).enumerate() {
        println!("  Bucket {}: {} shares ({}% of total)", 
                i + 1, 
                bucket.target_volume,
                (bucket.target_volume as f64 / execution_plan.order.quantity as f64) * 100.0);
    }
    if execution_plan.time_buckets.len() > 5 {
        println!("  ... and {} more buckets", execution_plan.time_buckets.len() - 5);
    }

    // Simulate execution decisions
    let market_state = create_test_market_state("AAPL", 150.0);
    let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    
    let decision = twap_executor.execute_next_slice(current_time, &market_state)?;
    
    println!("\n⚡ Execution Decision:");
    println!("  • Should Execute: {}", decision.should_execute);
    println!("  • Order Size: {} shares", decision.order_size);
    println!("  • Participation Rate: {:.2}%", decision.participation_rate.to_float() * 100.0);
    println!("  • Urgency: {:.2}", decision.urgency.to_float());
    println!("  • Reason: {}", decision.reason);

    // Simulate execution and record
    if decision.should_execute {
        twap_executor.record_execution(
            decision.order_size,
            FixedPoint::from_float(150.05),
            10000, // Market volume
            current_time,
        )?;

        let status = twap_executor.get_execution_status().unwrap();
        println!("\n📈 Execution Status:");
        println!("  • Executed: {} shares ({:.1}%)", 
                status.executed_quantity,
                status.completion_percentage);
        println!("  • Remaining: {} shares", status.remaining_quantity);
        println!("  • Current Bucket: {}/{}", 
                status.current_bucket_index + 1, 
                status.total_buckets);
    }

    println!("\n✅ Basic TWAP execution demo completed\n");
    Ok(())
}

fn demo_volume_forecasting() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Demo 2: Volume Forecasting and Adaptation");
    println!("--------------------------------------------");

    // Create volume forecaster
    let mut volume_forecaster = VolumeForecaster::new(30); // 30 days history

    // Generate historical volume data
    let historical_data = generate_historical_volume_data(100)?;
    volume_forecaster.add_historical_data(historical_data)?;

    println!("📊 Historical Data Added: 100 data points");

    // Create market state and conditions
    let market_state = create_test_market_state("AAPL", 150.0);
    let market_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    // Forecast volume for next hour
    let volume_forecast = volume_forecaster.forecast_volume(
        3600, // 1 hour
        &market_state,
        &market_conditions,
    )?;

    println!("\n🔮 Volume Forecast (1 hour):");
    println!("  • Forecasted Volume: {} shares", volume_forecast.forecasted_volume);
    println!("  • Confidence Interval: {} - {} shares", 
            volume_forecast.confidence_interval.0,
            volume_forecast.confidence_interval.1);
    println!("  • Seasonality Adjustment: {:.2}x", 
            volume_forecast.seasonality_adjustment.to_float());
    println!("  • Market Condition Adjustment: {:.2}x", 
            volume_forecast.market_condition_adjustment.to_float());
    println!("  • Accuracy Score: {:.1}%", 
            volume_forecast.accuracy_score.to_float() * 100.0);

    // Generate adaptive bucket sizing
    let adaptive_sizing = volume_forecaster.generate_adaptive_bucket_sizing(
        50000, // Total quantity
        3600,  // Execution horizon
        12,    // Number of buckets
        &market_conditions,
    )?;

    println!("\n🪣 Adaptive Bucket Sizing:");
    println!("  • Number of Buckets: {}", adaptive_sizing.bucket_sizes.len());
    
    for (i, (&volume, &rate)) in adaptive_sizing.volume_allocations.iter()
        .zip(adaptive_sizing.participation_rates.iter())
        .take(5)
        .enumerate() {
        println!("  Bucket {}: {} shares, {:.1}% participation", 
                i + 1, volume, rate.to_float() * 100.0);
    }

    // Test participation rate adjustment
    let base_rate = FixedPoint::from_float(0.1);
    let mut high_vol_conditions = market_conditions.clone();
    high_vol_conditions.volatility_regime = VolatilityRegime::High;

    let adjusted_rate = volume_forecaster.adjust_participation_rate(
        base_rate,
        &market_state,
        &high_vol_conditions,
        FixedPoint::from_float(0.5), // 50% execution progress
    )?;

    println!("\n⚙️ Participation Rate Adjustment:");
    println!("  • Base Rate: {:.1}%", base_rate.to_float() * 100.0);
    println!("  • Adjusted Rate (High Vol): {:.1}%", adjusted_rate.to_float() * 100.0);
    println!("  • Adjustment Factor: {:.2}x", 
            (adjusted_rate.to_float() / base_rate.to_float()));

    println!("\n✅ Volume forecasting demo completed\n");
    Ok(())
}

fn demo_market_impact_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("💥 Demo 3: Market Impact Optimization");
    println!("-------------------------------------");

    // Create market impact model
    let impact_params = MarketImpactParams {
        temporary_impact_coeff: FixedPoint::from_float(0.1),
        temporary_impact_exponent: FixedPoint::from_float(0.5), // Square-root law
        permanent_impact_coeff: FixedPoint::from_float(0.01),
        permanent_impact_exponent: FixedPoint::from_float(1.0), // Linear
        cross_impact_coeff: FixedPoint::from_float(0.05),
        decay_rate: FixedPoint::from_float(0.1),
        volatility_scaling: FixedPoint::from_float(1.0),
        liquidity_scaling: FixedPoint::from_float(1.0),
    };

    let impact_model = MarketImpactModel::new(impact_params);

    // Create market state and conditions
    let market_state = create_test_market_state("AAPL", 150.0);
    let market_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    // Estimate market impact for different order sizes
    let order_sizes = vec![1000, 5000, 10000, 25000];
    let participation_rate = FixedPoint::from_float(0.15);

    println!("📊 Market Impact Estimates:");
    for &size in &order_sizes {
        let impact_estimate = impact_model.estimate_market_impact(
            size,
            participation_rate,
            &market_state,
            &market_conditions,
            300, // 5 minutes
        )?;

        println!("  {} shares:", size);
        println!("    • Total Impact: ${:.4} ({:.2} bps)", 
                impact_estimate.total_impact.to_float(),
                impact_estimate.impact_percentage.to_float());
        println!("    • Temporary: ${:.4}", impact_estimate.temporary_impact.to_float());
        println!("    • Permanent: ${:.4}", impact_estimate.permanent_impact.to_float());
        println!("    • Cross-Impact: ${:.4}", impact_estimate.cross_impact.to_float());
    }

    // Calculate optimal execution trajectory
    let trajectory = impact_model.calculate_optimal_trajectory(
        50000, // Total volume
        3600,  // Execution horizon
        &market_state,
        &market_conditions,
        FixedPoint::from_float(0.01), // Risk aversion
    )?;

    println!("\n🎯 Optimal Execution Trajectory:");
    println!("  • Time Points: {}", trajectory.time_points.len());
    println!("  • Total Expected Cost: ${:.2}", 
            trajectory.total_expected_cost.to_float());
    
    // Show first few execution rates
    for (i, (&rate, &cost)) in trajectory.execution_rates.iter()
        .zip(trajectory.expected_costs.iter())
        .take(5)
        .enumerate() {
        println!("  Point {}: Rate={:.0} shares/sec, Cost=${:.4}", 
                i + 1, rate.to_float(), cost.to_float());
    }

    // Simulate execution records for slippage analysis
    let execution_records = generate_execution_records(10)?;
    let benchmark_price = FixedPoint::from_float(150.0);
    let market_states = vec![market_state.clone(); 10];

    let slippage_analysis = impact_model.analyze_slippage(
        &execution_records,
        benchmark_price,
        &market_states,
    )?;

    println!("\n📉 Slippage Analysis:");
    println!("  • Total Slippage: ${:.4}", slippage_analysis.total_slippage.to_float());
    println!("  • Impact Slippage: ${:.4}", slippage_analysis.impact_slippage.to_float());
    println!("  • Timing Slippage: ${:.4}", slippage_analysis.timing_slippage.to_float());
    println!("  • Spread Slippage: ${:.4}", slippage_analysis.spread_slippage.to_float());
    println!("  • Time Periods Analyzed: {}", slippage_analysis.time_period_attribution.len());

    // Calculate execution costs
    let cost_breakdown = impact_model.calculate_execution_costs(
        &execution_records,
        benchmark_price,
        FixedPoint::from_float(5.0), // 5 bps transaction cost
    )?;

    println!("\n💰 Execution Cost Breakdown:");
    println!("  • Market Impact Cost: ${:.2}", cost_breakdown.market_impact_cost.to_float());
    println!("  • Timing Cost: ${:.2}", cost_breakdown.timing_cost.to_float());
    println!("  • Transaction Cost: ${:.2}", cost_breakdown.transaction_cost.to_float());
    println!("  • Total Cost: ${:.2}", cost_breakdown.total_cost.to_float());
    println!("  • Cost (bps): {:.1}", cost_breakdown.cost_basis_points.to_float());

    println!("\n✅ Market impact optimization demo completed\n");
    Ok(())
}

fn demo_adaptive_execution_control() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎛️ Demo 4: Adaptive Execution Control");
    println!("-------------------------------------");

    // Create adaptive control configuration
    let control_config = AdaptiveControlConfig {
        shortfall_threshold: FixedPoint::from_float(0.05), // 5%
        surplus_threshold: FixedPoint::from_float(0.05),   // 5%
        max_catchup_factor: FixedPoint::from_float(2.0),   // 2x
        max_slowdown_factor: FixedPoint::from_float(0.5),  // 0.5x
        volatility_sensitivity: FixedPoint::from_float(0.3),
        liquidity_sensitivity: FixedPoint::from_float(0.4),
        news_reaction_time: 300,  // 5 minutes
        risk_reaction_time: 60,   // 1 minute
        min_execution_rate: FixedPoint::from_float(0.01),
        max_execution_rate: FixedPoint::from_float(0.5),
    };

    let twap_config = TWAPConfig::default();
    let impact_params = MarketImpactParams::default();

    let mut controller = AdaptiveExecutionController::new(
        control_config,
        twap_config,
        impact_params,
    );

    println!("🎛️ Adaptive Controller Created");

    // Add contingency plans
    let volatility_plan = ContingencyPlan {
        id: "high_volatility".to_string(),
        trigger: ContingencyTrigger::VolatilitySpike { 
            threshold: FixedPoint::from_float(0.05) 
        },
        action: ContingencyAction::ReduceRate { 
            factor: FixedPoint::from_float(0.5) 
        },
        priority: 1,
        activation_time: None,
        expiration_time: None,
        status: ContingencyStatus::Inactive,
    };

    let news_plan = ContingencyPlan {
        id: "news_event".to_string(),
        trigger: ContingencyTrigger::NewsEvent { 
            impact_level: NewsImpact::High 
        },
        action: ContingencyAction::PauseExecution { duration: 300 },
        priority: 2,
        activation_time: None,
        expiration_time: None,
        status: ContingencyStatus::Inactive,
    };

    let price_move_plan = ContingencyPlan {
        id: "large_price_move".to_string(),
        trigger: ContingencyTrigger::PriceMove { 
            threshold: FixedPoint::from_float(2.0),
            direction: PriceMoveDirection::Either,
        },
        action: ContingencyAction::SwitchToMarketOrders,
        priority: 3,
        activation_time: None,
        expiration_time: None,
        status: ContingencyStatus::Inactive,
    };

    controller.add_contingency_plan(volatility_plan)?;
    controller.add_contingency_plan(news_plan)?;
    controller.add_contingency_plan(price_move_plan)?;

    println!("📋 Contingency Plans Added: 3");

    // Test normal market conditions
    let market_state = create_test_market_state("AAPL", 150.0);
    let normal_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    // Initialize TWAP execution first
    let order = Order {
        id: 1,
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 50000,
        price: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        time_in_force: TimeInForce::Day,
    };

    // This would normally be done internally, but for demo we need to initialize
    let _ = controller.twap_executor.create_execution_plan(
        order,
        TWAPConfig::default(),
        &normal_conditions,
        None,
    );

    let decision = controller.process_market_update(&market_state, &normal_conditions)?;

    println!("\n⚡ Normal Conditions Decision:");
    println!("  • Should Execute: {}", decision.should_execute);
    println!("  • Order Size: {} shares", decision.order_size);
    println!("  • Participation Rate: {:.2}%", decision.participation_rate.to_float() * 100.0);
    println!("  • Urgency: {:.2}", decision.urgency.to_float());

    // Test high volatility conditions
    let high_vol_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::High,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    let mut high_vol_market = market_state.clone();
    high_vol_market.volatility = FixedPoint::from_float(0.06); // High volatility

    let vol_decision = controller.process_market_update(&high_vol_market, &high_vol_conditions)?;

    println!("\n🌪️ High Volatility Decision:");
    println!("  • Should Execute: {}", vol_decision.should_execute);
    println!("  • Order Size: {} shares", vol_decision.order_size);
    println!("  • Participation Rate: {:.2}%", vol_decision.participation_rate.to_float() * 100.0);
    println!("  • Urgency: {:.2}", vol_decision.urgency.to_float());

    // Test catch-up mechanism
    let shortfall = FixedPoint::from_float(0.1); // 10% behind schedule
    let catchup_decision = controller.implement_catchup_mechanism(
        shortfall,
        &market_state,
        &normal_conditions,
    )?;

    println!("\n🏃 Catch-up Mechanism:");
    println!("  • Shortfall: {:.1}%", shortfall.to_float() * 100.0);
    println!("  • Order Size: {} shares", catchup_decision.order_size);
    println!("  • Participation Rate: {:.2}%", catchup_decision.participation_rate.to_float() * 100.0);
    println!("  • Urgency: {:.2}", catchup_decision.urgency.to_float());
    println!("  • Reason: {}", catchup_decision.reason);

    // Test slowdown mechanism
    let surplus = FixedPoint::from_float(0.08); // 8% ahead of schedule
    let slowdown_decision = controller.implement_slowdown_mechanism(
        surplus,
        &market_state,
        &normal_conditions,
    )?;

    println!("\n🐌 Slowdown Mechanism:");
    println!("  • Surplus: {:.1}%", surplus.to_float() * 100.0);
    println!("  • Order Size: {} shares", slowdown_decision.order_size);
    println!("  • Participation Rate: {:.2}%", slowdown_decision.participation_rate.to_float() * 100.0);
    println!("  • Urgency: {:.2}", slowdown_decision.urgency.to_float());
    println!("  • Reason: {}", slowdown_decision.reason);

    // Get execution status with control information
    let status = controller.get_execution_status_with_control()?;
    println!("\n📊 Execution Status with Control:");
    println!("  • Execution Mode: {:?}", status.control_state.mode);
    println!("  • Schedule Adherence: {:.2}", status.control_state.schedule_adherence.to_float());
    println!("  • Active Contingencies: {}", status.active_contingencies);
    println!("  • Total Adaptations: {}", status.performance_metrics.total_adaptations);
    println!("  • Response Time: {}ms", status.performance_metrics.response_time_ms);

    println!("\n✅ Adaptive execution control demo completed\n");
    Ok(())
}

fn demo_complete_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Demo 5: Complete TWAP System Integration");
    println!("==========================================");

    println!("🎯 Simulating complete TWAP execution workflow...");

    // Create comprehensive configuration
    let twap_config = TWAPConfig {
        execution_horizon: 1800,  // 30 minutes
        num_buckets: 6,           // 5-minute buckets
        target_participation_rate: FixedPoint::from_float(0.12),
        max_participation_rate: FixedPoint::from_float(0.25),
        min_order_size: 200,
        max_order_size: 8000,
        adaptive_scheduling: true,
        timing_randomization: FixedPoint::from_float(0.03),
    };

    let control_config = AdaptiveControlConfig::default();
    let impact_params = MarketImpactParams::default();

    let mut controller = AdaptiveExecutionController::new(
        control_config,
        twap_config,
        impact_params,
    );

    // Create large order
    let order = Order {
        id: 12345,
        symbol: "MSFT".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 100000, // 100,000 shares
        price: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        time_in_force: TimeInForce::Day,
    };

    println!("📋 Large Order Created:");
    println!("  • Symbol: {}", order.symbol);
    println!("  • Side: {:?}", order.side);
    println!("  • Quantity: {} shares", order.quantity);
    println!("  • Target Execution Time: 30 minutes");

    // Initialize execution plan
    let market_conditions = MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Weak_Up,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    };

    let _ = controller.twap_executor.create_execution_plan(
        order,
        TWAPConfig {
            execution_horizon: 1800,
            num_buckets: 6,
            target_participation_rate: FixedPoint::from_float(0.12),
            max_participation_rate: FixedPoint::from_float(0.25),
            min_order_size: 200,
            max_order_size: 8000,
            adaptive_scheduling: true,
            timing_randomization: FixedPoint::from_float(0.03),
        },
        &market_conditions,
        None,
    );

    // Simulate execution over multiple time periods
    let mut total_executed = 0u64;
    let mut execution_prices = Vec::new();
    let base_price = 280.0;

    println!("\n⏰ Simulating Execution Timeline:");
    println!("  Time    | Market | Decision | Executed | Cumulative | Mode");
    println!("  --------|--------|----------|----------|------------|----------");

    for i in 0..6 {
        let time_offset = i * 300; // 5-minute intervals
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() + time_offset;
        
        // Simulate market price movement
        let price_change = (i as f64 - 2.5) * 0.1; // Small price drift
        let current_price = base_price + price_change;
        
        // Create market state
        let mut market_state = create_test_market_state("MSFT", current_price);
        market_state.timestamp = current_time;
        
        // Simulate changing market conditions
        let mut conditions = market_conditions.clone();
        if i == 3 {
            conditions.volatility_regime = VolatilityRegime::High;
            market_state.volatility = FixedPoint::from_float(0.04);
        }
        if i == 4 {
            conditions.news_impact = NewsImpact::Medium;
        }

        // Process market update
        let decision = controller.process_market_update(&market_state, &conditions)?;
        
        // Simulate execution
        let executed_volume = if decision.should_execute {
            decision.order_size.min(20000) // Cap for simulation
        } else {
            0
        };

        if executed_volume > 0 {
            controller.twap_executor.record_execution(
                executed_volume,
                FixedPoint::from_float(current_price + 0.01), // Small slippage
                50000, // Market volume
                current_time,
            )?;
            
            total_executed += executed_volume;
            execution_prices.push(current_price + 0.01);
        }

        let status = controller.get_execution_status_with_control()?;
        
        println!("  {:02}:{}0   | ${:6.2} | {:8} | {:8} | {:10} | {:?}",
                i * 5,
                if i * 5 < 10 { "0" } else { "" },
                current_price,
                if decision.should_execute { "Execute" } else { "Skip" },
                executed_volume,
                total_executed,
                status.control_state.mode);
    }

    // Calculate final performance metrics
    let average_price = if !execution_prices.is_empty() {
        execution_prices.iter().sum::<f64>() / execution_prices.len() as f64
    } else {
        base_price
    };

    let benchmark_price = base_price;
    let implementation_shortfall = (average_price - benchmark_price) * total_executed as f64;

    println!("\n📊 Final Execution Summary:");
    println!("  • Total Executed: {} shares ({:.1}%)", 
            total_executed,
            (total_executed as f64 / 100000.0) * 100.0);
    println!("  • Average Price: ${:.4}", average_price);
    println!("  • Benchmark Price: ${:.4}", benchmark_price);
    println!("  • Implementation Shortfall: ${:.2}", implementation_shortfall);
    println!("  • Slippage: {:.1} bps", 
            ((average_price - benchmark_price) / benchmark_price) * 10000.0);

    let final_status = controller.get_execution_status_with_control()?;
    println!("\n🎛️ Final Control Metrics:");
    println!("  • Total Adaptations: {}", final_status.performance_metrics.total_adaptations);
    println!("  • Successful Catch-ups: {}", final_status.performance_metrics.successful_catchups);
    println!("  • Successful Slowdowns: {}", final_status.performance_metrics.successful_slowdowns);
    println!("  • Contingency Activations: {}", final_status.performance_metrics.contingency_activations);
    println!("  • Average Response Time: {}ms", final_status.performance_metrics.response_time_ms);

    println!("\n✅ Complete integration demo completed");
    Ok(())
}

// Helper functions

fn create_test_market_state(symbol: &str, price: f64) -> MarketState {
    MarketState {
        symbol: symbol.to_string(),
        bid_price: FixedPoint::from_float(price - 0.01),
        ask_price: FixedPoint::from_float(price + 0.01),
        bid_volume: 5000,
        ask_volume: 5000,
        last_price: FixedPoint::from_float(price),
        last_volume: 1000,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        volatility: FixedPoint::from_float(0.02),
        average_daily_volume: 2000000,
    }
}

fn generate_historical_volume_data(count: usize) -> Result<Vec<VolumeDataPoint>, Box<dyn std::error::Error>> {
    let mut data = Vec::with_capacity(count);
    let base_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    
    for i in 0..count {
        let timestamp = base_time - (i as u64 * 3600); // Hourly data going back
        let hour = ((timestamp / 3600) % 24) as u8;
        let day_of_week = (((timestamp / 86400) + 4) % 7) as u8; // Unix epoch was Thursday
        
        // Simulate volume patterns
        let base_volume = 10000;
        let hour_multiplier = match hour {
            9..=10 => 1.5,  // Market open
            11..=12 => 1.2,
            13..=14 => 0.8, // Lunch
            15..=16 => 1.3, // Market close
            _ => 1.0,
        };
        
        let volume = (base_volume as f64 * hour_multiplier) as u64 + (i as u64 * 10);
        
        data.push(VolumeDataPoint {
            timestamp,
            volume,
            hour_of_day: hour,
            day_of_week,
            is_holiday: false,
            market_conditions: MarketConditions {
                volatility_regime: VolatilityRegime::Normal,
                liquidity_level: LiquidityLevel::Normal,
                trend_direction: TrendDirection::Sideways,
                market_hours: MarketHours::Regular,
                news_impact: NewsImpact::None,
            },
        });
    }
    
    Ok(data)
}

fn generate_execution_records(count: usize) -> Result<Vec<advanced_trading_system::execution::market_impact::ExecutionRecord>, Box<dyn std::error::Error>> {
    use advanced_trading_system::execution::market_impact::ExecutionRecord;
    
    let mut records = Vec::with_capacity(count);
    let base_time = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let base_price = 150.0;
    
    for i in 0..count {
        records.push(ExecutionRecord {
            timestamp: base_time + (i as u64 * 60), // 1-minute intervals
            executed_volume: 1000 + (i as u64 * 100),
            execution_price: FixedPoint::from_float(base_price + (i as f64 * 0.01)),
            participation_rate: FixedPoint::from_float(0.1 + (i as f64 * 0.01)),
        });
    }
    
    Ok(records)
}