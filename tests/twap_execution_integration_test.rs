//! Integration tests for TWAP execution system

use advanced_trading_system::execution::{
    twap::{TWAPExecutor, TWAPConfig},
    volume_forecasting::VolumeForecaster,
    market_impact::MarketImpactModel,
    execution_control::AdaptiveExecutionController,
    Order, OrderSide, OrderType, TimeInForce, MarketState, MarketConditions,
    VolatilityRegime, LiquidityLevel, TrendDirection, MarketHours, NewsImpact,
};
use advanced_trading_system::math::fixed_point::FixedPoint;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_test_order() -> Order {
    Order {
        id: 1,
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 10000,
        price: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        time_in_force: TimeInForce::Day,
    }
}

fn create_test_market_state() -> MarketState {
    MarketState {
        symbol: "AAPL".to_string(),
        bid_price: FixedPoint::from_float(150.0),
        ask_price: FixedPoint::from_float(150.1),
        bid_volume: 1000,
        ask_volume: 1000,
        last_price: FixedPoint::from_float(150.05),
        last_volume: 500,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        volatility: FixedPoint::from_float(0.02),
        average_daily_volume: 1000000,
    }
}

fn create_test_market_conditions() -> MarketConditions {
    MarketConditions {
        volatility_regime: VolatilityRegime::Normal,
        liquidity_level: LiquidityLevel::Normal,
        trend_direction: TrendDirection::Sideways,
        market_hours: MarketHours::Regular,
        news_impact: NewsImpact::None,
    }
}

#[test]
fn test_twap_execution_integration() {
    let mut executor = TWAPExecutor::new();
    let order = create_test_order();
    let config = TWAPConfig::default();
    let market_conditions = create_test_market_conditions();
    let market_state = create_test_market_state();

    // Create execution plan
    let plan = executor.create_execution_plan(
        order.clone(),
        config.clone(),
        &market_conditions,
        None,
    ).unwrap();

    assert_eq!(plan.order.quantity, order.quantity);
    assert_eq!(plan.time_buckets.len(), config.num_buckets);

    // Get execution decision
    let decision = executor.execute_next_slice(
        market_state.timestamp,
        &market_state,
    ).unwrap();

    assert!(decision.order_size > 0);
    assert!(decision.participation_rate > FixedPoint::ZERO);

    // Record execution
    executor.record_execution(
        decision.order_size,
        FixedPoint::from_float(150.05),
        10000,
        market_state.timestamp,
    ).unwrap();

    let status = executor.get_execution_status().unwrap();
    assert_eq!(status.executed_quantity, decision.order_size);
}

#[test]
fn test_volume_forecasting_integration() {
    let mut forecaster = VolumeForecaster::new(30);
    let market_state = create_test_market_state();
    let market_conditions = create_test_market_conditions();

    // Test volume forecast without historical data
    let forecast = forecaster.forecast_volume(
        3600,
        &market_state,
        &market_conditions,
    ).unwrap();

    assert!(forecast.forecasted_volume > 0);
    assert!(forecast.confidence_interval.0 <= forecast.forecasted_volume);
    assert!(forecast.confidence_interval.1 >= forecast.forecasted_volume);

    // Test adaptive bucket sizing
    let bucket_sizing = forecaster.generate_adaptive_bucket_sizing(
        10000,
        3600,
        10,
        &market_conditions,
    ).unwrap();

    assert_eq!(bucket_sizing.bucket_sizes.len(), 10);
    assert_eq!(bucket_sizing.volume_allocations.len(), 10);
    
    let total_allocated: u64 = bucket_sizing.volume_allocations.iter().sum();
    assert_eq!(total_allocated, 10000);
}

#[test]
fn test_market_impact_integration() {
    let model = MarketImpactModel::default();
    let market_state = create_test_market_state();
    let market_conditions = create_test_market_conditions();

    // Test impact estimation
    let impact_estimate = model.estimate_market_impact(
        1000,
        FixedPoint::from_float(0.1),
        &market_state,
        &market_conditions,
        300,
    ).unwrap();

    assert!(impact_estimate.total_impact >= FixedPoint::ZERO);
    assert!(impact_estimate.temporary_impact >= FixedPoint::ZERO);
    assert!(impact_estimate.permanent_impact >= FixedPoint::ZERO);

    // Test optimal trajectory calculation
    let trajectory = model.calculate_optimal_trajectory(
        10000,
        3600,
        &market_state,
        &market_conditions,
        FixedPoint::from_float(0.01),
    ).unwrap();

    assert!(!trajectory.time_points.is_empty());
    assert_eq!(trajectory.time_points.len(), trajectory.execution_rates.len());
    assert!(trajectory.total_expected_cost >= FixedPoint::ZERO);
}

#[test]
fn test_adaptive_control_integration() {
    use advanced_trading_system::execution::{
        execution_control::AdaptiveControlConfig,
        market_impact::MarketImpactParams,
    };

    let control_config = AdaptiveControlConfig::default();
    let twap_config = TWAPConfig::default();
    let impact_params = MarketImpactParams::default();

    let mut controller = AdaptiveExecutionController::new(
        control_config,
        twap_config,
        impact_params,
    );

    let market_state = create_test_market_state();
    let market_conditions = create_test_market_conditions();

    // Initialize TWAP execution plan
    let order = create_test_order();
    let _ = controller.twap_executor.create_execution_plan(
        order,
        TWAPConfig::default(),
        &market_conditions,
        None,
    );

    // Test market update processing
    let decision = controller.process_market_update(
        &market_state,
        &market_conditions,
    ).unwrap();

    assert!(decision.participation_rate >= FixedPoint::ZERO);

    // Test catch-up mechanism
    let shortfall = FixedPoint::from_float(0.1);
    let catchup_decision = controller.implement_catchup_mechanism(
        shortfall,
        &market_state,
        &market_conditions,
    ).unwrap();

    assert_eq!(catchup_decision.urgency, FixedPoint::from_float(0.8));

    // Test slowdown mechanism
    let surplus = FixedPoint::from_float(0.1);
    let slowdown_decision = controller.implement_slowdown_mechanism(
        surplus,
        &market_state,
        &market_conditions,
    ).unwrap();

    assert_eq!(slowdown_decision.urgency, FixedPoint::from_float(0.2));
}

#[test]
fn test_end_to_end_twap_workflow() {
    use advanced_trading_system::execution::{
        execution_control::{AdaptiveControlConfig, ContingencyPlan, ContingencyTrigger, ContingencyAction, ContingencyStatus},
        market_impact::MarketImpactParams,
    };

    // Create complete TWAP system
    let control_config = AdaptiveControlConfig::default();
    let twap_config = TWAPConfig {
        execution_horizon: 1800,
        num_buckets: 6,
        target_participation_rate: FixedPoint::from_float(0.15),
        max_participation_rate: FixedPoint::from_float(0.30),
        min_order_size: 100,
        max_order_size: 5000,
        adaptive_scheduling: true,
        timing_randomization: FixedPoint::from_float(0.05),
    };
    let impact_params = MarketImpactParams::default();

    let mut controller = AdaptiveExecutionController::new(
        control_config,
        twap_config,
        impact_params,
    );

    // Add contingency plan
    let contingency_plan = ContingencyPlan {
        id: "test_volatility".to_string(),
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

    controller.add_contingency_plan(contingency_plan).unwrap();

    // Create and execute order
    let order = Order {
        id: 1,
        symbol: "AAPL".to_string(),
        side: OrderSide::Buy,
        order_type: OrderType::Market,
        quantity: 50000,
        price: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        time_in_force: TimeInForce::Day,
    };

    let market_conditions = create_test_market_conditions();
    
    // Initialize execution plan
    let _ = controller.twap_executor.create_execution_plan(
        order,
        TWAPConfig {
            execution_horizon: 1800,
            num_buckets: 6,
            target_participation_rate: FixedPoint::from_float(0.15),
            max_participation_rate: FixedPoint::from_float(0.30),
            min_order_size: 100,
            max_order_size: 5000,
            adaptive_scheduling: true,
            timing_randomization: FixedPoint::from_float(0.05),
        },
        &market_conditions,
        None,
    );

    // Simulate multiple execution cycles
    let mut total_executed = 0u64;
    
    for i in 0..3 {
        let mut market_state = create_test_market_state();
        market_state.timestamp += i * 300; // 5-minute intervals
        
        let decision = controller.process_market_update(
            &market_state,
            &market_conditions,
        ).unwrap();

        if decision.should_execute && decision.order_size > 0 {
            controller.twap_executor.record_execution(
                decision.order_size,
                FixedPoint::from_float(150.05),
                10000,
                market_state.timestamp,
            ).unwrap();
            
            total_executed += decision.order_size;
        }
    }

    // Verify execution progress
    let status = controller.get_execution_status_with_control().unwrap();
    assert!(total_executed > 0);
    
    if let Some(base_status) = status.base_status {
        assert_eq!(base_status.executed_quantity, total_executed);
        assert!(base_status.completion_percentage > 0.0);
    }
}