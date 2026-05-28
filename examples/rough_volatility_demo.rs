//! Rough Volatility Model Demonstration
//!
//! This example demonstrates the complete rough volatility implementation including:
//! - Fractional Brownian Motion generation using Cholesky decomposition
//! - Rough volatility path simulation with mean reversion
//! - Volatility clustering detection and analysis
//! - Hurst parameter estimation using R/S analysis and DFA
//! - Integration with SDE solvers for complete price dynamics

use bid_ask_spread_estimation::math::{
    FixedPoint, DeterministicRng,
    FractionalBrownianMotion, RoughVolatilitySimulator, 
    VolatilityClusteringDetector, HurstEstimator,
    RoughVolatilitySolver, RoughVolatilityState, RoughVolatilityParams,
    SDESolver
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rough Volatility Model Demonstration ===\n");

    // 1. Demonstrate Fractional Brownian Motion generation
    demonstrate_fbm_generation()?;
    
    // 2. Demonstrate rough volatility path simulation
    demonstrate_rough_volatility_simulation()?;
    
    // 3. Demonstrate volatility clustering detection
    demonstrate_volatility_clustering()?;
    
    // 4. Demonstrate Hurst parameter estimation
    demonstrate_hurst_estimation()?;
    
    // 5. Demonstrate integration with SDE solver
    demonstrate_sde_integration()?;
    
    // 6. Performance comparison
    demonstrate_performance_comparison()?;

    println!("\n=== Demonstration Complete ===");
    Ok(())
}

fn demonstrate_fbm_generation() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Fractional Brownian Motion Generation");
    println!("========================================");
    
    let hurst_values = [0.1, 0.2, 0.3, 0.4];
    let n_steps = 100;
    let mut rng = DeterministicRng::new(42);
    
    for &hurst in &hurst_values {
        let hurst_fp = FixedPoint::from_float(hurst);
        let fbm = FractionalBrownianMotion::new(hurst_fp, n_steps)?;
        
        let start_time = Instant::now();
        let path = fbm.generate_path(n_steps, &mut rng)?;
        let generation_time = start_time.elapsed();
        
        // Compute path statistics
        let final_value = path.last().unwrap().to_float();
        let path_variance = path.iter()
            .map(|&x| x.to_float().powi(2))
            .sum::<f64>() / path.len() as f64;
        
        println!("  Hurst = {:.1}: Final value = {:.4}, Variance = {:.6}, Time = {:?}",
                 hurst, final_value, path_variance, generation_time);
    }
    
    println!();
    Ok(())
}

fn demonstrate_rough_volatility_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Rough Volatility Path Simulation");
    println!("===================================");
    
    let hurst = FixedPoint::from_float(0.2);
    let vol_of_vol = FixedPoint::from_float(0.3);
    let mean_reversion = FixedPoint::from_float(2.0);
    let long_term_var = FixedPoint::from_float(0.04);
    let initial_log_vol = FixedPoint::from_float(0.2_f64.ln());
    
    let simulator = RoughVolatilitySimulator::new(
        hurst, vol_of_vol, mean_reversion, long_term_var, initial_log_vol, 200
    )?;
    
    let mut rng = DeterministicRng::new(123);
    let dt = FixedPoint::from_float(1.0 / 252.0); // Daily time step
    let n_steps = 100;
    
    let start_time = Instant::now();
    let vol_path = simulator.simulate_path(n_steps, dt, &mut rng)?;
    let simulation_time = start_time.elapsed();
    
    // Analyze volatility path
    let initial_vol = vol_path[0].to_float();
    let final_vol = vol_path.last().unwrap().to_float();
    let mean_vol = vol_path.iter().map(|v| v.to_float()).sum::<f64>() / vol_path.len() as f64;
    let max_vol = vol_path.iter().map(|v| v.to_float()).fold(0.0, f64::max);
    let min_vol = vol_path.iter().map(|v| v.to_float()).fold(f64::INFINITY, f64::min);
    
    println!("  Simulation parameters:");
    println!("    Hurst parameter: {:.2}", hurst.to_float());
    println!("    Vol of vol: {:.2}", vol_of_vol.to_float());
    println!("    Mean reversion: {:.2}", mean_reversion.to_float());
    println!("    Long-term variance: {:.4}", long_term_var.to_float());
    
    println!("  Results:");
    println!("    Initial volatility: {:.4}", initial_vol);
    println!("    Final volatility: {:.4}", final_vol);
    println!("    Mean volatility: {:.4}", mean_vol);
    println!("    Volatility range: [{:.4}, {:.4}]", min_vol, max_vol);
    println!("    Simulation time: {:?}", simulation_time);
    
    println!();
    Ok(())
}

fn demonstrate_volatility_clustering() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Volatility Clustering Detection");
    println!("==================================");
    
    // Generate synthetic returns with clustering
    let mut rng = DeterministicRng::new(456);
    let mut returns = Vec::new();
    let mut current_vol = 0.02; // 2% daily volatility
    
    for i in 0..200 {
        // Create volatility clustering by making volatility persistent
        let vol_shock = (rng.next_fixed().to_float() - 0.5) * 0.01;
        current_vol = (current_vol + vol_shock).max(0.005).min(0.1);
        
        // Add some regime changes
        if i == 50 || i == 150 {
            current_vol *= 2.0; // Volatility spike
        }
        
        // Generate return with current volatility
        let z = 2.0 * rng.next_fixed().to_float() - 1.0; // Uniform approximation to normal
        let return_val = current_vol * z;
        returns.push(FixedPoint::from_float(return_val));
    }
    
    let detector = VolatilityClusteringDetector::new(10, FixedPoint::from_float(2.0));
    
    // Detect clusters
    let start_time = Instant::now();
    let clusters = detector.detect_clusters(&returns)?;
    let detection_time = start_time.elapsed();
    
    let cluster_count = clusters.iter().filter(|&&x| x).count();
    let cluster_percentage = (cluster_count as f64 / clusters.len() as f64) * 100.0;
    
    println!("  Clustering analysis:");
    println!("    Total observations: {}", returns.len());
    println!("    Clustered periods: {} ({:.1}%)", cluster_count, cluster_percentage);
    println!("    Detection time: {:?}", detection_time);
    
    // Compute autocorrelations
    let autocorrs = detector.compute_volatility_autocorrelation(&returns)?;
    println!("  Volatility autocorrelations:");
    for (lag, autocorr) in autocorrs {
        println!("    Lag {}: {:.4}", lag, autocorr.to_float());
    }
    
    println!();
    Ok(())
}

fn demonstrate_hurst_estimation() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Hurst Parameter Estimation");
    println!("=============================");
    
    let estimator = HurstEstimator::new(10, 100);
    
    // Test with known Hurst parameters
    let true_hurst_values = [0.1, 0.2, 0.3, 0.4];
    
    for &true_hurst in &true_hurst_values {
        // Generate FBM series with known Hurst parameter
        let hurst_fp = FixedPoint::from_float(true_hurst);
        let fbm = FractionalBrownianMotion::new(hurst_fp, 200)?;
        let mut rng = DeterministicRng::new(789 + (true_hurst * 1000.0) as u64);
        let series = fbm.generate_path(150, &mut rng)?;
        
        // Estimate using R/S method
        let start_time = Instant::now();
        let estimated_rs = estimator.estimate_rs_method(&series)?;
        let rs_time = start_time.elapsed();
        
        // Estimate using DFA method
        let start_time = Instant::now();
        let estimated_dfa = estimator.estimate_dfa_method(&series)?;
        let dfa_time = start_time.elapsed();
        
        println!("  True Hurst = {:.1}:", true_hurst);
        println!("    R/S estimate: {:.3} (error: {:.3}, time: {:?})", 
                 estimated_rs.to_float(), 
                 (estimated_rs.to_float() - true_hurst).abs(),
                 rs_time);
        println!("    DFA estimate: {:.3} (error: {:.3}, time: {:?})", 
                 estimated_dfa.to_float(), 
                 (estimated_dfa.to_float() - true_hurst).abs(),
                 dfa_time);
    }
    
    println!();
    Ok(())
}

fn demonstrate_sde_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. SDE Integration with Advanced FBM");
    println!("====================================");
    
    let hurst = FixedPoint::from_float(0.15);
    
    // Compare basic vs advanced FBM solvers
    let mut basic_solver = RoughVolatilitySolver::new(50);
    let mut advanced_solver = RoughVolatilitySolver::with_advanced_fbm(50, hurst)?;
    
    let params = RoughVolatilityParams {
        hurst_parameter: hurst,
        vol_of_vol: FixedPoint::from_float(0.4),
        mean_reversion: FixedPoint::from_float(1.8),
        long_term_var: FixedPoint::from_float(0.05),
        correlation: FixedPoint::from_float(-0.7),
        initial_vol: FixedPoint::from_float(0.22),
    };
    
    let initial_state = RoughVolatilityState {
        price: FixedPoint::from_float(100.0),
        log_volatility: FixedPoint::from_float(0.22_f64.ln()),
        volatility: FixedPoint::from_float(0.22),
        time: FixedPoint::zero(),
        fbm_history: Vec::new(),
    };
    
    let t_span = (FixedPoint::zero(), FixedPoint::from_float(0.25)); // 3 months
    let n_steps = 60; // ~Daily steps
    
    // Basic solver
    let mut rng1 = DeterministicRng::new(999);
    let start_time = Instant::now();
    let basic_path = basic_solver.solve_path(t_span, initial_state.clone(), &params, n_steps, &mut rng1)?;
    let basic_time = start_time.elapsed();
    
    // Advanced solver
    let mut rng2 = DeterministicRng::new(999); // Same seed for comparison
    let start_time = Instant::now();
    let advanced_path = advanced_solver.solve_path(t_span, initial_state, &params, n_steps, &mut rng2)?;
    let advanced_time = start_time.elapsed();
    
    // Compare results
    let basic_final_price = basic_path.last().unwrap().price.to_float();
    let basic_final_vol = basic_path.last().unwrap().volatility.to_float();
    let advanced_final_price = advanced_path.last().unwrap().price.to_float();
    let advanced_final_vol = advanced_path.last().unwrap().volatility.to_float();
    
    println!("  Comparison of FBM methods:");
    println!("    Basic solver:");
    println!("      Final price: {:.2}", basic_final_price);
    println!("      Final volatility: {:.4}", basic_final_vol);
    println!("      Computation time: {:?}", basic_time);
    
    println!("    Advanced solver (Cholesky FBM):");
    println!("      Final price: {:.2}", advanced_final_price);
    println!("      Final volatility: {:.4}", advanced_final_vol);
    println!("      Computation time: {:?}", advanced_time);
    println!("      Using advanced FBM: {}", advanced_solver.use_advanced_fbm);
    
    // Compute path statistics
    let basic_vol_mean = basic_path.iter().map(|s| s.volatility.to_float()).sum::<f64>() / basic_path.len() as f64;
    let advanced_vol_mean = advanced_path.iter().map(|s| s.volatility.to_float()).sum::<f64>() / advanced_path.len() as f64;
    
    println!("    Mean volatility comparison:");
    println!("      Basic: {:.4}", basic_vol_mean);
    println!("      Advanced: {:.4}", advanced_vol_mean);
    
    println!();
    Ok(())
}

fn demonstrate_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Performance Comparison");
    println!("=========================");
    
    let hurst_values = [0.1, 0.2, 0.3, 0.4];
    let path_lengths = [50, 100, 200];
    
    println!("  FBM Generation Performance (microseconds per path):");
    println!("  Hurst\\Length   50      100     200");
    println!("  -----------   ----    ----    ----");
    
    for &hurst in &hurst_values {
        print!("  {:.1}         ", hurst);
        
        for &length in &path_lengths {
            let hurst_fp = FixedPoint::from_float(hurst);
            let fbm = FractionalBrownianMotion::new(hurst_fp, length)?;
            let mut rng = DeterministicRng::new(12345);
            
            // Warm up
            for _ in 0..5 {
                let _ = fbm.generate_path(length, &mut rng)?;
            }
            
            // Benchmark
            let n_trials = 100;
            let start_time = Instant::now();
            for _ in 0..n_trials {
                let _ = fbm.generate_path(length, &mut rng)?;
            }
            let total_time = start_time.elapsed();
            let avg_time_us = total_time.as_micros() / n_trials as u128;
            
            print!("{:6}  ", avg_time_us);
        }
        println!();
    }
    
    println!("\n  Memory usage scales as O(n²) for Cholesky decomposition");
    println!("  where n is the maximum path length.");
    
    println!();
    Ok(())
}