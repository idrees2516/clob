use bid_ask_spread_estimation::{SpreadEstimator, TradeData};
use chrono::Utc;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create sample trade data
    let data = generate_sample_data();

    // Initialize spread estimator
    let window_size = 50;
    let confidence_level = 0.95;
    let estimator = SpreadEstimator::new(data, window_size, confidence_level)?;

    // Calculate spread estimates
    let metrics = estimator.estimate_spread()?;

    // Print results
    println!("\nBid-Ask Spread Estimation Results:");
    println!("----------------------------------");
    println!("Estimated Spread: {:.6}", metrics.spread);
    println!(
        "95% Confidence Interval: ({:.6}, {:.6})",
        metrics.confidence_interval.0, metrics.confidence_interval.1
    );
    println!("Serial Covariance: {:.6}", metrics.serial_covariance);
    println!("Return Variance: {:.6}", metrics.variance);
    println!("Sample Size: {}", metrics.sample_size);

    Ok(())
}

fn generate_sample_data() -> Vec<TradeData> {
    let mut data = Vec::new();
    let mut price = 100.0;
    let mut rng = rand::thread_rng();
    let normal = rand_distr::Normal::new(0.0, 0.001).unwrap();

    for i in 0..1000 {
        let noise = normal.sample(&mut rng);
        price *= (1.0 + noise).exp();
        let direction = if rng.gen::<bool>() { 1 } else { -1 };
        let spread = 0.5;
        
        data.push(TradeData {
            timestamp: Utc::now().timestamp() + i,
            price: price + direction as f64 * spread / 2.0,
            volume: rng.gen_range(100.0..1000.0),
            direction: Some(direction),
        });
    }

    data
}
