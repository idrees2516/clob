use hf_quoting_liquidity_clob::models::performance_metrics::{
    PerformanceCalculator, PerformanceData, TradeData, MarketData, TradeSide,
};

#[test]
fn test_performance_metrics_integration() {
    // Create simple test data
    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
    let prices = vec![100.0, 101.0, 100.5, 102.5, 101.5, 103.0];
    let positions = vec![0.0, 100.0, 100.0, 200.0, 150.0, 150.0];
    
    let trades = vec![
        TradeData {
            timestamp: 1.0,
            price: 101.0,
            quantity: 100.0,
            side: TradeSide::Buy,
            transaction_cost: 0.1,
            market_impact: 0.05,
            spread_at_trade: 0.02,
        },
    ];
    
    let market_data = vec![
        MarketData {
            timestamp: 1.0,
            mid_price: 100.5,
            bid_price: 100.49,
            ask_price: 100.51,
            bid_size: 1000.0,
            ask_size: 1000.0,
            spread: 0.02,
            depth: 2000.0,
            volatility: 0.15,
        },
    ];
    
    let timestamps = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    
    let performance_data = PerformanceData {
        returns,
        prices,
        positions,
        trades,
        market_data,
        timestamps,
        benchmark_returns: None,
        risk_free_rate: 0.02,
    };
    
    let calculator = PerformanceCalculator::default();
    let metrics = calculator.calculate_metrics(&performance_data).unwrap();
    
    // Basic assertions
    assert!(metrics.total_return.is_finite());
    assert!(metrics.volatility >= 0.0);
    assert!(metrics.max_drawdown >= 0.0);
    assert!(metrics.max_drawdown <= 1.0);
    assert!(metrics.total_trades > 0);
    assert!(metrics.transaction_costs.total_cost >= 0.0);
}

#[test]
fn test_sharpe_ratio_calculation() {
    let calculator = PerformanceCalculator::default();
    let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005];
    
    let sharpe = calculator.calculate_sharpe_ratio(&returns, 0.02).unwrap();
    assert!(sharpe.is_finite());
}

#[test]
fn test_max_drawdown_calculation() {
    let calculator = PerformanceCalculator::default();
    let returns = vec![0.1, -0.2, 0.05, -0.1, 0.15];
    
    let max_dd = calculator.calculate_max_drawdown(&returns).unwrap();
    assert!(max_dd >= 0.0);
    assert!(max_dd <= 1.0);
}

#[test]
fn test_var_calculation() {
    let calculator = PerformanceCalculator::default();
    let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 1000.0).collect();
    
    let (var_95, var_99) = calculator.calculate_var(&returns).unwrap();
    assert!(var_95 >= 0.0);
    assert!(var_99 >= var_95);
}

#[test]
fn test_transaction_cost_analysis() {
    let trades = vec![
        TradeData {
            timestamp: 1.0,
            price: 100.0,
            quantity: 100.0,
            side: TradeSide::Buy,
            transaction_cost: 0.1,
            market_impact: 0.05,
            spread_at_trade: 0.02,
        },
        TradeData {
            timestamp: 2.0,
            price: 101.0,
            quantity: -50.0,
            side: TradeSide::Sell,
            transaction_cost: 0.05,
            market_impact: 0.025,
            spread_at_trade: 0.02,
        },
    ];
    
    let performance_data = PerformanceData {
        returns: vec![0.01, -0.005],
        prices: vec![100.0, 101.0, 100.5],
        positions: vec![0.0, 100.0, 50.0],
        trades,
        market_data: vec![],
        timestamps: vec![0.0, 1.0, 2.0],
        benchmark_returns: None,
        risk_free_rate: 0.02,
    };
    
    let calculator = PerformanceCalculator::default();
    let tx_costs = calculator.calculate_transaction_costs(&performance_data).unwrap();
    
    assert!(tx_costs.total_cost > 0.0);
    assert!(tx_costs.cost_per_share > 0.0);
    assert!(!tx_costs.cost_breakdown.is_empty());
}