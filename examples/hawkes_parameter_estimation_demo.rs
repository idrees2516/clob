//! Hawkes Process Parameter Estimation Demo
//! 
//! This example demonstrates the comprehensive parameter estimation capabilities
//! for multivariate Hawkes processes, including maximum likelihood estimation,
//! expectation-maximization algorithm, cross-validation, and goodness-of-fit testing.

use std::error::Error;
use trading_system::math::{
    FixedPoint, DeterministicRng, HawkesEvent, MultivariateHawkesParams, KernelType,
    MultivariateHawkesSimulator, LBFGSOptimizer, HawkesMLEstimator, HawkesEMEstimator,
    HawkesCrossValidator, HawkesGoodnessOfFitTester, ValidationMetric
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Hawkes Process Parameter Estimation Demo ===\n");
    
    // Step 1: Create true parameters for data generation
    println!("1. Creating true Hawkes process parameters...");
    let true_params = create_true_parameters()?;
    print_parameters("True Parameters", &true_params);
    
    // Step 2: Generate synthetic event data
    println!("\n2. Generating synthetic event sequences...");
    let observation_time = FixedPoint::from_float(20.0);
    let n_sequences = 10;
    let event_sequences = generate_synthetic_data(&true_params, n_sequences, observation_time)?;
    
    let total_events: usize = event_sequences.iter().map(|seq| seq.len()).sum();
    println!("Generated {} sequences with {} total events", n_sequences, total_events);
    
    if total_events == 0 {
        println!("No events generated - demo cannot continue");
        return Ok(());
    }
    
    // Step 3: Maximum Likelihood Estimation
    println!("\n3. Performing Maximum Likelihood Estimation...");
    demonstrate_ml_estimation(&event_sequences, &true_params, observation_time)?;
    
    // Step 4: Expectation-Maximization Algorithm
    println!("\n4. Performing EM Algorithm Estimation...");
    demonstrate_em_estimation(&event_sequences, &true_params, observation_time)?;
    
    // Step 5: Cross-Validation for Model Selection
    println!("\n5. Performing Cross-Validation...");
    demonstrate_cross_validation(&event_sequences, &true_params, observation_time)?;
    
    // Step 6: Goodness-of-Fit Testing
    println!("\n6. Performing Goodness-of-Fit Testing...");
    demonstrate_goodness_of_fit(&event_sequences, &true_params, observation_time)?;
    
    println!("\n=== Demo completed successfully! ===");
    Ok(())
}

/// Create true parameters for synthetic data generation
fn create_true_parameters() -> Result<MultivariateHawkesParams, Box<dyn Error>> {
    let baseline_intensities = vec![
        FixedPoint::from_float(0.8),  // Buy order baseline intensity
        FixedPoint::from_float(0.6),  // Sell order baseline intensity
    ];
    
    let kernels = vec![
        vec![
            // Buy -> Buy excitation
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.3), 
                beta: FixedPoint::from_float(1.2) 
            },
            // Sell -> Buy excitation (cross-excitation)
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.15), 
                beta: FixedPoint::from_float(0.8) 
            },
        ],
        vec![
            // Buy -> Sell excitation (cross-excitation)
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.2), 
                beta: FixedPoint::from_float(0.9) 
            },
            // Sell -> Sell excitation
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.25), 
                beta: FixedPoint::from_float(1.0) 
            },
        ],
    ];
    
    let max_intensity = FixedPoint::from_float(20.0);
    
    Ok(MultivariateHawkesParams::new(
        baseline_intensities,
        kernels,
        max_intensity,
    )?)
}

/// Generate synthetic event data using true parameters
fn generate_synthetic_data(
    params: &MultivariateHawkesParams,
    n_sequences: usize,
    observation_time: FixedPoint,
) -> Result<Vec<Vec<HawkesEvent>>, Box<dyn Error>> {
    let mut sequences = Vec::new();
    let base_seed = 12345u64;
    
    for i in 0..n_sequences {
        let mut simulator = MultivariateHawkesSimulator::new(
            params.clone(),
            5000, // Large history buffer
        );
        
        let mut rng = DeterministicRng::new(base_seed + i as u64);
        
        match simulator.simulate_until(observation_time, &mut rng) {
            Ok(events) => {
                println!("  Sequence {}: {} events", i + 1, events.len());
                sequences.push(events);
            }
            Err(e) => {
                println!("  Sequence {}: simulation failed - {}", i + 1, e);
                sequences.push(Vec::new());
            }
        }
    }
    
    Ok(sequences)
}

/// Demonstrate Maximum Likelihood Estimation
fn demonstrate_ml_estimation(
    event_sequences: &[Vec<HawkesEvent>],
    true_params: &MultivariateHawkesParams,
    observation_time: FixedPoint,
) -> Result<(), Box<dyn Error>> {
    // Create initial parameters (perturbed from true values)
    let mut initial_params = true_params.clone();
    for intensity in &mut initial_params.baseline_intensities {
        *intensity = *intensity * FixedPoint::from_float(1.2); // 20% increase
    }
    
    // Configure L-BFGS optimizer
    let optimizer = LBFGSOptimizer::new(
        100,                                    // max iterations
        FixedPoint::from_float(1e-6),          // tolerance
        10,                                     // history size
    );
    
    // Create ML estimator with regularization
    let estimator = HawkesMLEstimator::new(optimizer)
        .with_regularization(FixedPoint::from_float(0.01));
    
    println!("  Using L-BFGS optimizer with regularization...");
    print_parameters("Initial Parameters", &initial_params);
    
    match estimator.estimate(event_sequences, &initial_params, observation_time) {
        Ok(estimated_params) => {
            print_parameters("ML Estimated Parameters", &estimated_params);
            compare_parameters("ML Estimation", true_params, &estimated_params);
        }
        Err(e) => {
            println!("  ML estimation failed: {}", e);
            println!("  This is common with limited synthetic data");
        }
    }
    
    Ok(())
}

/// Demonstrate Expectation-Maximization Algorithm
fn demonstrate_em_estimation(
    event_sequences: &[Vec<HawkesEvent>],
    true_params: &MultivariateHawkesParams,
    observation_time: FixedPoint,
) -> Result<(), Box<dyn Error>> {
    let estimator = HawkesEMEstimator::new(
        50,                                     // max iterations
        FixedPoint::from_float(1e-5),          // tolerance
    );
    
    println!("  Using EM algorithm...");
    
    match estimator.estimate(event_sequences, true_params, observation_time) {
        Ok(estimated_params) => {
            print_parameters("EM Estimated Parameters", &estimated_params);
            compare_parameters("EM Estimation", true_params, &estimated_params);
        }
        Err(e) => {
            println!("  EM estimation failed: {}", e);
            println!("  This is common with limited synthetic data");
        }
    }
    
    Ok(())
}

/// Demonstrate Cross-Validation for Model Selection
fn demonstrate_cross_validation(
    event_sequences: &[Vec<HawkesEvent>],
    true_params: &MultivariateHawkesParams,
    observation_time: FixedPoint,
) -> Result<(), Box<dyn Error>> {
    // Create candidate models with different parameters
    let mut candidate1 = true_params.clone();
    candidate1.baseline_intensities[0] = FixedPoint::from_float(0.7);
    
    let mut candidate2 = true_params.clone();
    candidate2.baseline_intensities[0] = FixedPoint::from_float(0.9);
    
    let mut candidate3 = true_params.clone();
    candidate3.baseline_intensities[1] = FixedPoint::from_float(0.5);
    
    let candidates = vec![candidate1, candidate2, candidate3];
    
    println!("  Testing {} candidate models with 5-fold cross-validation...", candidates.len());
    
    // Test different validation metrics
    let metrics = vec![
        ("AIC", ValidationMetric::AIC),
        ("BIC", ValidationMetric::BIC),
        ("Log-Likelihood", ValidationMetric::LogLikelihood),
    ];
    
    for (metric_name, metric) in metrics {
        println!("\n  Using {} metric:", metric_name);
        
        let validator = HawkesCrossValidator::new(5, metric);
        
        match validator.validate(event_sequences, &candidates, observation_time) {
            Ok(result) => {
                println!("    Best model: Candidate {} (index {})", 
                         result.best_model_idx + 1, result.best_model_idx);
                
                for (i, model_result) in result.results.iter().enumerate() {
                    println!("    Candidate {}: mean_score = {:.6}, std_score = {:.6}",
                             i + 1,
                             model_result.mean_score.to_float(),
                             model_result.std_score.to_float());
                }
            }
            Err(e) => {
                println!("    Cross-validation failed: {}", e);
            }
        }
    }
    
    Ok(())
}

/// Demonstrate Goodness-of-Fit Testing
fn demonstrate_goodness_of_fit(
    event_sequences: &[Vec<HawkesEvent>],
    fitted_params: &MultivariateHawkesParams,
    observation_time: FixedPoint,
) -> Result<(), Box<dyn Error>> {
    let tester = HawkesGoodnessOfFitTester::new(
        FixedPoint::from_float(0.05),          // 5% significance level
        1000,                                   // bootstrap samples
    );
    
    println!("  Performing comprehensive goodness-of-fit testing...");
    
    match tester.test_fit(event_sequences, fitted_params, observation_time) {
        Ok(result) => {
            println!("\n  Kolmogorov-Smirnov Test:");
            print_statistical_test(&result.ks_test);
            
            println!("\n  Anderson-Darling Test:");
            print_statistical_test(&result.ad_test);
            
            println!("\n  Bootstrap Test:");
            print_statistical_test(&result.bootstrap_test);
            
            println!("\n  Residual Analysis:");
            println!("    Number of residuals: {}", result.residual_analysis.residuals.len());
            println!("    Mean residual: {:.6}", result.residual_analysis.mean_residual.to_float());
            println!("    Std residual: {:.6}", result.residual_analysis.std_residual.to_float());
            println!("    Autocorrelation: {:.6}", result.residual_analysis.autocorrelation.to_float());
        }
        Err(e) => {
            println!("  Goodness-of-fit testing failed: {}", e);
        }
    }
    
    Ok(())
}

/// Print Hawkes process parameters in a readable format
fn print_parameters(title: &str, params: &MultivariateHawkesParams) {
    println!("\n  {}:", title);
    println!("    Dimension: {}", params.dimension());
    
    println!("    Baseline intensities:");
    for (i, &intensity) in params.baseline_intensities.iter().enumerate() {
        println!("      Process {}: {:.4}", i, intensity.to_float());
    }
    
    println!("    Excitation kernels:");
    for (i, row) in params.kernels.iter().enumerate() {
        for (j, kernel) in row.iter().enumerate() {
            match kernel {
                KernelType::Exponential { alpha, beta } => {
                    println!("      {} -> {}: Exp(α={:.4}, β={:.4})", 
                             i, j, alpha.to_float(), beta.to_float());
                }
                KernelType::PowerLaw { alpha, beta, cutoff } => {
                    println!("      {} -> {}: PowerLaw(α={:.4}, β={:.4}, c={:.4})", 
                             i, j, alpha.to_float(), beta.to_float(), cutoff.to_float());
                }
                KernelType::SumExponentials { alphas, betas } => {
                    println!("      {} -> {}: SumExp with {} terms", i, j, alphas.len());
                }
            }
        }
    }
}

/// Compare estimated parameters with true parameters
fn compare_parameters(
    method: &str,
    true_params: &MultivariateHawkesParams,
    estimated_params: &MultivariateHawkesParams,
) {
    println!("\n  {} Parameter Comparison:", method);
    
    // Compare baseline intensities
    println!("    Baseline Intensity Errors:");
    for i in 0..true_params.dimension() {
        let true_val = true_params.baseline_intensities[i].to_float();
        let est_val = estimated_params.baseline_intensities[i].to_float();
        let error = ((est_val - true_val) / true_val * 100.0).abs();
        println!("      Process {}: {:.2}% error", i, error);
    }
    
    // Compare kernel parameters (simplified for exponential kernels)
    println!("    Kernel Parameter Errors:");
    for i in 0..true_params.dimension() {
        for j in 0..true_params.dimension() {
            if let (KernelType::Exponential { alpha: true_alpha, beta: true_beta },
                    KernelType::Exponential { alpha: est_alpha, beta: est_beta }) = 
                (&true_params.kernels[i][j], &estimated_params.kernels[i][j]) {
                
                let alpha_error = ((est_alpha.to_float() - true_alpha.to_float()) / 
                                  true_alpha.to_float() * 100.0).abs();
                let beta_error = ((est_beta.to_float() - true_beta.to_float()) / 
                                 true_beta.to_float() * 100.0).abs();
                
                println!("      {} -> {}: α error = {:.2}%, β error = {:.2}%", 
                         i, j, alpha_error, beta_error);
            }
        }
    }
}

/// Print statistical test results
fn print_statistical_test(test: &trading_system::math::StatisticalTest) {
    println!("    Statistic: {:.6}", test.statistic.to_float());
    println!("    P-value: {:.6}", test.p_value.to_float());
    println!("    Critical value: {:.6}", test.critical_value.to_float());
    println!("    Significant: {}", if test.is_significant { "Yes" } else { "No" });
}