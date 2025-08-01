use bid_ask_spread_estimation::{
    SpreadEstimator, TradeData,
    utils::{simulate_price_process, Bootstrap, VolatilityEstimator},
};
use chrono::Utc;
use std::error::Error;
use tracing::{info, Level};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Simulate market data
    info!("Generating simulated market data...");
    let initial_price = 100.0;
    let true_spread = 0.5;
    let volatility = 0.001;
    let n_steps = 5000;
    
    let data = simulate_price_process(initial_price, volatility, true_spread, n_steps)?;
    info!("Generated {} data points", data.len());

    // Estimate spreads with different window sizes
    let window_sizes = vec![20, 50, 100];
    let confidence_level = 0.95;

    for window_size in window_sizes {
        info!("Analyzing with window size: {}", window_size);
        let estimator = SpreadEstimator::new(data.clone(), window_size, confidence_level)?;
        let metrics = estimator.estimate_spread()?;

        println!("\nResults for window size {}:", window_size);
        println!("Estimated Spread: {:.6}", metrics.spread);
        println!(
            "95% Confidence Interval: ({:.6}, {:.6})",
            metrics.confidence_interval.0, metrics.confidence_interval.1
        );
    }

    // Estimate volatility
    info!("Calculating volatility estimates...");
    let vol_estimator = VolatilityEstimator::new(&data, 50)?;
    let volatilities = vol_estimator.estimate_volatility()?;
    
    println!("\nVolatility Analysis:");
    println!("Mean Volatility: {:.6}", 
             volatilities.iter().sum::<f64>() / volatilities.len() as f64);
    println!("Number of Estimates: {}", volatilities.len());

    // Perform bootstrap analysis
    info!("Performing bootstrap analysis...");
    let returns: Vec<f64> = data.windows(2)
        .map(|w| (w[1].price / w[0].price).ln())
        .collect();
    
    let bootstrap = Bootstrap::new(returns, 1000, 100);
    let samples = bootstrap.generate_samples();
    let (lower, upper) = bootstrap.compute_confidence_interval(&samples, confidence_level)?;

    println!("\nBootstrap Analysis:");
    println!("Bootstrap Confidence Interval: ({:.6}, {:.6})", lower, upper);
    println!("Number of Bootstrap Samples: {}", samples.len());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_usage() {
        main().unwrap();
    }
}
