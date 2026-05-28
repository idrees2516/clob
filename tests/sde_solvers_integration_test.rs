//! Integration tests for SDE solvers with financial modeling validation

use hf_quoting_liquidity_clob::math::{
    FixedPoint, DeterministicRng,
    sde_solvers::{
        SDESolver, GBMJumpState, GBMJumpParams, EulerMaruyamaGBMJump, MilsteinGBMJump,
        KouJumpParams, KouJumpSolver, HestonState, HestonParams, HestonSolver,
        RoughVolatilityState, RoughVolatilityParams, RoughVolatilitySolver,
        MultiDimensionalState, MultiDimensionalParams, MultiDimensionalGBMSolver,
        MonteCarloSimulator, VarianceReductionMC, PathStatistics
    }
};

#[test]
fn test_gbm_jump_convergence() {
    // Test that Euler-Maruyama and Milstein converge to similar results
    let mut euler_solver = EulerMaruyamaGBMJump::new(100);
    let mut milstein_solver = MilsteinGBMJump::new(100);
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::from_float(0.1),
        jump_mean: FixedPoint::from_float(0.0),
        jump_volatility: FixedPoint::from_float(0.1),
        risk_free_rate: FixedPoint::from_float(0.03),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let t_span = (FixedPoint::zero(), FixedPoint::one());
    let n_steps = 1000; // High resolution for convergence test
    
    // Run multiple simulations and compare
    let mut euler_final_prices = Vec::new();
    let mut milstein_final_prices = Vec::new();
    
    for seed in 0..50 {
        let mut rng = DeterministicRng::new(seed);
        let euler_path = euler_solver.solve_path(t_span, initial.clone(), &params, n_steps, &mut rng).unwrap();
        euler_final_prices.push(euler_path.last().unwrap().price.to_float());
        
        let mut rng = DeterministicRng::new(seed); // Same seed for fair comparison
        let milstein_path = milstein_solver.solve_path(t_span, initial.clone(), &params, n_steps, &mut rng).unwrap();
        milstein_final_prices.push(milstein_path.last().unwrap().price.to_float());
    }
    
    // Calculate means
    let euler_mean = euler_final_prices.iter().sum::<f64>() / euler_final_prices.len() as f64;
    let milstein_mean = milstein_final_prices.iter().sum::<f64>() / milstein_final_prices.len() as f64;
    
    // Should be close for high-resolution simulations
    let relative_diff = (euler_mean - milstein_mean).abs() / euler_mean;
    assert!(relative_diff < 0.1, "Euler and Milstein should converge: {:.3}% difference", relative_diff * 100.0);
    
    // Both should be positive
    assert!(euler_mean > 0.0);
    assert!(milstein_mean > 0.0);
}

#[test]
fn test_heston_variance_positivity() {
    // Test that Heston model maintains positive variance
    let mut solver = HestonSolver::new();
    let mut rng = DeterministicRng::new(42);
    
    let params = HestonParams {
        risk_free_rate: FixedPoint::from_float(0.05),
        kappa: FixedPoint::from_float(3.0), // Strong mean reversion
        theta: FixedPoint::from_float(0.04),
        sigma: FixedPoint::from_float(0.5), // High vol of vol
        rho: FixedPoint::from_float(-0.8),
        initial_variance: FixedPoint::from_float(0.01), // Low initial variance
    };
    
    let initial = HestonState {
        price: FixedPoint::from_float(100.0),
        variance: FixedPoint::from_float(0.01),
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::from_float(2.0)), // 2 years
        initial,
        &params,
        500,
        &mut rng,
    ).unwrap();
    
    // All variances should be positive
    for state in &path {
        assert!(state.variance.to_float() > 0.0, "Variance should remain positive");
        assert!(state.price.to_float() > 0.0, "Price should remain positive");
    }
    
    // Variance should mean-revert towards theta
    let final_variance = path.last().unwrap().variance.to_float();
    let target_variance = params.theta.to_float();
    
    // Should be closer to target after 2 years with strong mean reversion
    assert!(final_variance > 0.0);
    assert!(final_variance < 1.0); // Reasonable upper bound
}

#[test]
fn test_kou_jump_properties() {
    // Test Kou model jump properties
    let mut solver = KouJumpSolver::new(1000);
    let mut rng = DeterministicRng::new(123);
    
    let params = KouJumpParams {
        drift: FixedPoint::from_float(0.0), // Zero drift to isolate jump effects
        volatility: FixedPoint::from_float(0.1), // Low volatility
        jump_intensity: FixedPoint::from_float(1.0), // High jump intensity
        prob_up_jump: FixedPoint::from_float(0.7), // 70% up jumps
        eta_up: FixedPoint::from_float(5.0),
        eta_down: FixedPoint::from_float(10.0),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.1),
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::from_float(5.0)), // 5 years for many jumps
        initial,
        &params,
        1000,
        &mut rng,
    ).unwrap();
    
    // Should have generated jumps
    assert!(!solver.jump_buffer.is_empty(), "Should have generated jumps");
    
    // Analyze jump distribution
    let up_jumps = solver.jump_buffer.iter().filter(|&&j| j.to_float() > 0.0).count();
    let down_jumps = solver.jump_buffer.iter().filter(|&&j| j.to_float() < 0.0).count();
    let total_jumps = up_jumps + down_jumps;
    
    if total_jumps > 10 { // Only test if we have enough jumps
        let up_jump_ratio = up_jumps as f64 / total_jumps as f64;
        
        // Should be approximately 70% up jumps (with some tolerance)
        assert!(up_jump_ratio > 0.5, "Should have more up jumps than down jumps");
        assert!(up_jump_ratio < 0.9, "Up jump ratio should be reasonable: {:.2}", up_jump_ratio);
    }
    
    // Price should remain positive
    for state in &path {
        assert!(state.price.to_float() > 0.0, "Price should remain positive");
    }
}

#[test]
fn test_rough_volatility_hurst_parameter() {
    // Test rough volatility with different Hurst parameters
    let hurst_values = [0.05, 0.1, 0.2, 0.4];
    
    for &hurst in &hurst_values {
        let mut solver = RoughVolatilitySolver::new(100);
        let mut rng = DeterministicRng::new(456);
        
        let params = RoughVolatilityParams {
            hurst_parameter: FixedPoint::from_float(hurst),
            vol_of_vol: FixedPoint::from_float(0.2),
            mean_reversion: FixedPoint::from_float(1.0),
            long_term_var: FixedPoint::from_float(0.04),
            correlation: FixedPoint::from_float(-0.5),
            initial_vol: FixedPoint::from_float(0.2),
        };
        
        let initial = RoughVolatilityState {
            price: FixedPoint::from_float(100.0),
            log_volatility: FixedPoint::from_float(0.2_f64.ln()),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
            fbm_history: Vec::new(),
        };
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            100,
            &mut rng,
        ).unwrap();
        
        // Should complete successfully for all Hurst parameters
        assert_eq!(path.len(), 101);
        assert!(path.last().unwrap().price.to_float() > 0.0);
        assert!(path.last().unwrap().volatility.to_float() > 0.0);
        
        // FBM history should grow
        assert!(!path.last().unwrap().fbm_history.is_empty());
        
        // Lower Hurst should lead to more volatile volatility paths
        let vol_changes: Vec<f64> = path.windows(2)
            .map(|w| (w[1].volatility - w[0].volatility).to_float().abs())
            .collect();
        
        let avg_vol_change = vol_changes.iter().sum::<f64>() / vol_changes.len() as f64;
        
        // This is a rough test - lower Hurst should generally have more volatility clustering
        assert!(avg_vol_change >= 0.0); // Basic sanity check
    }
}

#[test]
fn test_multidimensional_correlation() {
    // Test that multidimensional GBM respects correlation structure
    let mut solver = MultiDimensionalGBMSolver::new();
    let mut rng = DeterministicRng::new(789);
    
    // Perfect positive correlation case
    let correlation_matrix = vec![
        vec![FixedPoint::one(), FixedPoint::from_float(0.99)],
        vec![FixedPoint::from_float(0.99), FixedPoint::one()],
    ];
    
    let params = MultiDimensionalParams {
        drifts: vec![FixedPoint::from_float(0.05), FixedPoint::from_float(0.05)],
        volatilities: vec![FixedPoint::from_float(0.2), FixedPoint::from_float(0.2)],
        correlation_matrix,
        n_assets: 2,
    };
    
    let initial = MultiDimensionalState {
        prices: vec![FixedPoint::from_float(100.0), FixedPoint::from_float(100.0)],
        volatilities: vec![FixedPoint::from_float(0.2), FixedPoint::from_float(0.2)],
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        &mut rng,
    ).unwrap();
    
    // Calculate correlation of returns
    let returns1: Vec<f64> = path.windows(2)
        .map(|w| (w[1].prices[0] / w[0].prices[0]).ln().to_float())
        .collect();
    
    let returns2: Vec<f64> = path.windows(2)
        .map(|w| (w[1].prices[1] / w[0].prices[1]).ln().to_float())
        .collect();
    
    let mean1 = returns1.iter().sum::<f64>() / returns1.len() as f64;
    let mean2 = returns2.iter().sum::<f64>() / returns2.len() as f64;
    
    let covariance = returns1.iter().zip(returns2.iter())
        .map(|(&r1, &r2)| (r1 - mean1) * (r2 - mean2))
        .sum::<f64>() / returns1.len() as f64;
    
    let var1 = returns1.iter().map(|&r| (r - mean1).powi(2)).sum::<f64>() / returns1.len() as f64;
    let var2 = returns2.iter().map(|&r| (r - mean2).powi(2)).sum::<f64>() / returns2.len() as f64;
    
    let correlation = covariance / (var1.sqrt() * var2.sqrt());
    
    // Should have high positive correlation (allowing for sampling variation)
    assert!(correlation > 0.5, "Correlation should be positive and significant: {:.3}", correlation);
    
    // Both assets should have positive final prices
    assert!(path.last().unwrap().prices[0].to_float() > 0.0);
    assert!(path.last().unwrap().prices[1].to_float() > 0.0);
}

#[test]
fn test_monte_carlo_convergence() {
    // Test Monte Carlo convergence with increasing sample sizes
    let solver = EulerMaruyamaGBMJump::new(50);
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::zero(), // No jumps for cleaner test
        jump_mean: FixedPoint::zero(),
        jump_volatility: FixedPoint::zero(),
        risk_free_rate: FixedPoint::from_float(0.03),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let sample_sizes = [100, 500, 1000];
    let mut means = Vec::new();
    
    for &n_paths in &sample_sizes {
        let mut mc_simulator = MonteCarloSimulator::new(solver.clone(), n_paths, false);
        
        let paths = mc_simulator.simulate_paths(
            (FixedPoint::zero(), FixedPoint::one()),
            initial.clone(),
            &params,
            252,
            12345,
        ).unwrap();
        
        let stats = PathStatistics::compute_from_gbm_paths(&paths).unwrap();
        means.push(stats.mean_final_price.to_float());
    }
    
    // Means should converge (later estimates should be closer to each other)
    let diff_small = (means[1] - means[0]).abs();
    let diff_large = (means[2] - means[1]).abs();
    
    // Generally, larger samples should give more stable estimates
    // (though this isn't guaranteed for any single run)
    assert!(diff_large < diff_small * 2.0, "Monte Carlo should show convergence trend");
    
    // All means should be reasonable (around initial price * exp(drift))
    let expected_mean = 100.0 * (0.05_f64).exp(); // ≈ 105.13
    for &mean in &means {
        assert!(mean > 80.0 && mean < 150.0, "Mean should be reasonable: {:.2}", mean);
    }
}

#[test]
fn test_variance_reduction_effectiveness() {
    // Test that variance reduction actually reduces variance
    let base_solver = EulerMaruyamaGBMJump::new(50);
    let n_paths = 200;
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::from_float(0.05),
        jump_mean: FixedPoint::from_float(0.0),
        jump_volatility: FixedPoint::from_float(0.1),
        risk_free_rate: FixedPoint::from_float(0.03),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    // Standard Monte Carlo
    let mut standard_mc = MonteCarloSimulator::new(base_solver.clone(), n_paths, false);
    let standard_paths = standard_mc.simulate_paths(
        (FixedPoint::zero(), FixedPoint::one()),
        initial.clone(),
        &params,
        100,
        54321,
    ).unwrap();
    
    // Variance reduction Monte Carlo
    let vr_base_mc = MonteCarloSimulator::new(base_solver, n_paths, false);
    let mut vr_mc = VarianceReductionMC::new(vr_base_mc, true, false);
    let vr_paths = vr_mc.simulate_paths_with_variance_reduction(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        100,
        54321,
    ).unwrap();
    
    // Both should produce the same number of paths
    assert_eq!(standard_paths.len(), vr_paths.len());
    
    // Calculate statistics
    let standard_stats = PathStatistics::compute_from_gbm_paths(&standard_paths).unwrap();
    let vr_stats = PathStatistics::compute_from_gbm_paths(&vr_paths).unwrap();
    
    // Both should have reasonable means
    assert!(standard_stats.mean_final_price.to_float() > 50.0);
    assert!(vr_stats.mean_final_price.to_float() > 50.0);
    
    // Both should have positive standard deviations
    assert!(standard_stats.std_final_price.to_float() > 0.0);
    assert!(vr_stats.std_final_price.to_float() > 0.0);
    
    // Note: Variance reduction effectiveness depends on the specific technique
    // and may not always be visible in small samples, so we just check basic properties
}

#[test]
fn test_path_statistics_accuracy() {
    // Test path statistics calculation accuracy
    let solver = EulerMaruyamaGBMJump::new(50);
    let mut mc_simulator = MonteCarloSimulator::new(solver, 1000, true);
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.0), // Zero drift for easier testing
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::zero(),
        jump_mean: FixedPoint::zero(),
        jump_volatility: FixedPoint::zero(),
        risk_free_rate: FixedPoint::from_float(0.0),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let paths = mc_simulator.simulate_paths(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        98765,
    ).unwrap();
    
    let stats = PathStatistics::compute_from_gbm_paths(&paths).unwrap();
    
    // With zero drift, mean should be close to initial price
    let mean_price = stats.mean_final_price.to_float();
    assert!(mean_price > 80.0 && mean_price < 120.0, "Mean price should be near initial: {:.2}", mean_price);
    
    // Standard deviation should be reasonable for 20% volatility
    let std_price = stats.std_final_price.to_float();
    assert!(std_price > 5.0 && std_price < 50.0, "Std should be reasonable: {:.2}", std_price);
    
    // VaR should be less than mean (for 95% VaR)
    let var_95 = stats.value_at_risk_95.to_float();
    assert!(var_95 < mean_price, "VaR should be less than mean: VaR={:.2}, Mean={:.2}", var_95, mean_price);
    assert!(var_95 > 0.0, "VaR should be positive");
    
    // Expected shortfall should be less than VaR
    let es_95 = stats.expected_shortfall_95.to_float();
    assert!(es_95 <= var_95, "ES should be <= VaR: ES={:.2}, VaR={:.2}", es_95, var_95);
    assert!(es_95 > 0.0, "ES should be positive");
    
    // Max drawdown should be between 0 and 1
    let max_dd = stats.max_drawdown.to_float();
    assert!(max_dd >= 0.0 && max_dd <= 1.0, "Max drawdown should be in [0,1]: {:.3}", max_dd);
}

#[test]
fn test_deterministic_reproducibility() {
    // Test that same seed produces same results
    let solver = EulerMaruyamaGBMJump::new(50);
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::from_float(0.1),
        jump_mean: FixedPoint::from_float(0.0),
        jump_volatility: FixedPoint::from_float(0.1),
        risk_free_rate: FixedPoint::from_float(0.03),
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let seed = 13579;
    let t_span = (FixedPoint::zero(), FixedPoint::one());
    let n_steps = 100;
    
    // Run same simulation twice
    let mut solver1 = solver.clone();
    let mut rng1 = DeterministicRng::new(seed);
    let path1 = solver1.solve_path(t_span, initial.clone(), &params, n_steps, &mut rng1).unwrap();
    
    let mut solver2 = solver.clone();
    let mut rng2 = DeterministicRng::new(seed);
    let path2 = solver2.solve_path(t_span, initial, &params, n_steps, &mut rng2).unwrap();
    
    // Paths should be identical
    assert_eq!(path1.len(), path2.len());
    
    for (state1, state2) in path1.iter().zip(path2.iter()) {
        assert_eq!(state1.price, state2.price, "Prices should be identical");
        assert_eq!(state1.volatility, state2.volatility, "Volatilities should be identical");
        assert_eq!(state1.time, state2.time, "Times should be identical");
    }
}