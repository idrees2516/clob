use bid_ask_spread_estimation::{
    SpreadEstimator, SpreadMetrics, TradeData,
    utils::{simulate_price_process, RollingWindow, VolatilityEstimator},
};
use chrono::Utc;
use rand::prelude::*;
use std::error::Error;

#[test]
fn test_end_to_end_estimation() -> Result<(), Box<dyn Error>> {
    // Generate synthetic data
    let initial_price = 100.0;
    let true_spread = 0.5;
    let volatility = 0.001;
    let n_steps = 1000;

    let data = simulate_price_process(initial_price, volatility, true_spread, n_steps)?;
    
    // Create estimator
    let window_size = 50;
    let confidence_level = 0.95;
    let estimator = SpreadEstimator::new(data, window_size, confidence_level)?;

    // Calculate spread estimates
    let metrics = estimator.estimate_spread()?;

    // Validate results
    assert!(metrics.spread > 0.0);
    assert!(metrics.confidence_interval.0 < metrics.confidence_interval.1);
    assert!((metrics.spread - true_spread).abs() / true_spread < 0.5);

    Ok(())
}

#[test]
fn test_rolling_window_estimation() -> Result<(), Box<dyn Error>> {
    let mut rng = thread_rng();
    let window_size = 20;
    let mut window = RollingWindow::new(window_size);

    // Add some random values
    for _ in 0..30 {
        let value = rng.gen_range(-1.0..1.0);
        let _ = window.add(value);
    }

    // Validate window behavior
    assert!(window.add(0.5).is_some());
    assert!(window.add(0.6).is_some());

    Ok(())
}

#[test]
fn test_volatility_estimation() -> Result<(), Box<dyn Error>> {
    let price_data: Vec<TradeData> = (0..100)
        .map(|i| TradeData {
            timestamp: Utc::now().timestamp() + i,
            price: 100.0 * (1.0 + 0.001 * (i as f64)).exp(),
            volume: 1000.0,
            direction: Some(if i % 2 == 0 { 1 } else { -1 }),
        })
        .collect();

    let estimator = VolatilityEstimator::new(&price_data, 20)?;
    let volatilities = estimator.estimate_volatility()?;

    assert!(!volatilities.is_empty());
    assert!(volatilities.iter().all(|&v| v >= 0.0));

    Ok(())
}

#[test]
fn test_error_handling() {
    // Test with empty data
    let empty_data: Vec<TradeData> = vec![];
    assert!(SpreadEstimator::new(empty_data, 50, 0.95).is_err());

    // Test with invalid window size
    let single_data = vec![TradeData {
        timestamp: Utc::now().timestamp(),
        price: 100.0,
        volume: 1000.0,
        direction: Some(1),
    }];
    assert!(SpreadEstimator::new(single_data, 0, 0.95).is_err());

    // Test with invalid confidence level
    let valid_data = vec![TradeData {
        timestamp: Utc::now().timestamp(),
        price: 100.0,
        volume: 1000.0,
        direction: Some(1),
    }];
    assert!(SpreadEstimator::new(valid_data, 1, 1.5).is_err());
}

#[test]
fn test_parallel_processing() -> Result<(), Box<dyn Error>> {
    let n_samples = 1000;
    let mut large_data: Vec<TradeData> = Vec::with_capacity(n_samples);
    let mut rng = thread_rng();
    let mut price = 100.0;
    let volatility = 0.001;
    let normal = rand_distr::Normal::new(0.0, volatility)?;

    for i in 0..n_samples {
        let noise = normal.sample(&mut rng);
        price *= (1.0 + noise).exp();
        large_data.push(TradeData {
            timestamp: Utc::now().timestamp() + i as i64,
            price,
            volume: rng.gen_range(100.0..1000.0),
            direction: Some(if rng.gen::<bool>() { 1 } else { -1 }),
        });
    }

    let estimator = SpreadEstimator::new(large_data, 50, 0.95)?;
    let metrics = estimator.estimate_spread()?;

    assert!(metrics.spread > 0.0);
    assert!(metrics.sample_size > 0);

    Ok(())
}

#[test]
fn test_metrics_consistency() -> Result<(), Box<dyn Error>> {
    let data = simulate_price_process(100.0, 0.001, 0.5, 1000)?;
    let estimator = SpreadEstimator::new(data.clone(), 50, 0.95)?;
    let metrics = estimator.estimate_spread()?;

    // Test relative metrics
    let mid_price = data.last().unwrap().price;
    let relative_metrics = metrics.relative_spread(mid_price);
    assert!(relative_metrics > 0.0);
    assert!(relative_metrics < 1.0);

    // Test confidence intervals
    assert!(metrics.confidence_interval.0 <= metrics.spread);
    assert!(metrics.confidence_interval.1 >= metrics.spread);

    Ok(())
}
