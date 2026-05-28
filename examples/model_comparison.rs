use bid_ask_spread_estimation::{
    SpreadEstimator, TradeData,
    models::{GarchModel, HiddenMarkovModel, KalmanFilter},
    utils::simulate_price_process,
};
use nalgebra as na;
use rand::prelude::*;
use std::error::Error;
use tracing::{info, Level};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Generate synthetic data
    info!("Generating synthetic market data...");
    let initial_price = 100.0;
    let true_spread = 0.5;
    let volatility = 0.001;
    let n_steps = 5000;
    
    let data = simulate_price_process(initial_price, volatility, true_spread, n_steps)?;
    let returns: Vec<f64> = data.windows(2)
        .map(|w| (w[1].price / w[0].price).ln())
        .collect();

    // 1. Basic Roll estimator
    info!("Running Roll estimator...");
    let roll_estimator = SpreadEstimator::new(data.clone(), 50, 0.95)?;
    let roll_metrics = roll_estimator.estimate_spread()?;

    // 2. GARCH model
    info!("Fitting GARCH model...");
    let mut garch = GarchModel::new(0.0, 0.0, 0.0)?;
    garch.fit(returns.clone(), 1000, 1e-6)?;
    let garch_vols = garch.get_volatilities();

    // 3. Hidden Markov Model
    info!("Training HMM...");
    let n_states = 2;
    let transition_matrix = na::DMatrix::from_vec(2, 2, vec![0.9, 0.1, 0.1, 0.9]);
    let emission_means = na::DVector::from_vec(vec![-volatility, volatility]);
    let emission_vars = na::DVector::from_vec(vec![volatility, volatility]);
    let initial_probs = na::DVector::from_vec(vec![0.5, 0.5]);

    let mut hmm = HiddenMarkovModel::new(
        n_states,
        transition_matrix,
        emission_means,
        emission_vars,
        initial_probs,
    )?;
    let states = hmm.viterbi_algorithm(&returns)?;

    // 4. Kalman Filter
    info!("Applying Kalman filter...");
    let state_dim = 2; // [price, spread]
    let obs_dim = 1;   // observed price

    let transition_matrix = na::DMatrix::identity(state_dim, state_dim);
    let observation_matrix = na::DMatrix::from_vec(obs_dim, state_dim, vec![1.0, 0.0]);
    let process_noise = na::DMatrix::identity(state_dim, state_dim) * 1e-4;
    let measurement_noise = na::DMatrix::identity(obs_dim, obs_dim) * 1e-3;
    let initial_state = na::DVector::from_vec(vec![initial_price, true_spread]);
    let initial_cov = na::DMatrix::identity(state_dim, state_dim);

    let mut kf = KalmanFilter::new(
        state_dim,
        obs_dim,
        transition_matrix,
        observation_matrix,
        process_noise,
        measurement_noise,
        initial_state,
        initial_cov,
    )?;

    let measurements: Vec<na::DVector<f64>> = data.iter()
        .map(|d| na::DVector::from_vec(vec![d.price]))
        .collect();

    let filtered_states = kf.smooth(&measurements)?;

    // Print results
    println!("\nModel Comparison Results:");
    println!("========================");
    
    println!("\n1. Roll Estimator:");
    println!("Estimated Spread: {:.6}", roll_metrics.spread);
    println!("95% CI: ({:.6}, {:.6})", 
             roll_metrics.confidence_interval.0,
             roll_metrics.confidence_interval.1);

    println!("\n2. GARCH Model:");
    let (omega, alpha, beta) = garch.get_parameters();
    println!("Parameters: ω={:.6}, α={:.6}, β={:.6}", omega, alpha, beta);
    println!("Persistence: {:.6}", garch.persistence());
    println!("Unconditional Variance: {:.6}", garch.unconditional_variance()?);

    println!("\n3. HMM Results:");
    let state_counts: Vec<_> = states.iter()
        .fold(vec![0; n_states], |mut counts, &s| {
            counts[s] += 1;
            counts
        });
    for (i, count) in state_counts.iter().enumerate() {
        println!("State {}: {:.2}% of observations", 
                i, 
                *count as f64 / states.len() as f64 * 100.0);
    }

    println!("\n4. Kalman Filter Results:");
    let mean_spread: f64 = filtered_states.iter()
        .map(|s| s[1])
        .sum::<f64>() / filtered_states.len() as f64;
    println!("Mean Estimated Spread: {:.6}", mean_spread);

    // Compare accuracy
    let spread_rmse = (filtered_states.iter()
        .map(|s| (s[1] - true_spread).powi(2))
        .sum::<f64>() / filtered_states.len() as f64)
        .sqrt();
    println!("\nSpread RMSE: {:.6}", spread_rmse);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_comparison() {
        main().unwrap();
    }
}
