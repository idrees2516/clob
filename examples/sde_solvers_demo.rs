//! SDE Solvers Demo
//!
//! This example demonstrates the comprehensive SDE solver capabilities
//! for financial modeling, including:
//! - Geometric Brownian Motion with jumps (Merton model)
//! - Kou jump-diffusion model with double exponential jumps
//! - Heston stochastic volatility model
//! - Rough volatility models with fractional Brownian motion
//! - Multi-dimensional correlated processes
//! - Monte Carlo simulation with variance reduction
//! - Statistical analysis and risk metrics

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SDE Solvers for Financial Modeling Demo ===\n");
    
    println!("1. Geometric Brownian Motion with Jumps (Merton Model):");
    demonstrate_gbm_with_jumps()?;
    
    println!("\n2. Kou Jump-Diffusion Model:");
    demonstrate_kou_model()?;
    
    println!("\n3. Heston Stochastic Volatility Model:");
    demonstrate_heston_model()?;
    
    println!("\n4. Rough Volatility Model:");
    demonstrate_rough_volatility()?;
    
    println!("\n5. Multi-Dimensional Correlated Processes:");
    demonstrate_multidimensional_gbm()?;
    
    println!("\n6. Monte Carlo Simulation with Statistics:");
    demonstrate_monte_carlo_simulation()?;
    
    println!("\n7. Variance Reduction Techniques:");
    demonstrate_variance_reduction()?;
    
    println!("\n8. Model Comparison and Analysis:");
    demonstrate_model_comparison()?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demonstrate_gbm_with_jumps() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating GBM with Merton jumps...");
    
    let mut euler_solver = EulerMaruyamaGBMJump::new(100);
    let mut milstein_solver = MilsteinGBMJump::new(100);
    let mut rng = DeterministicRng::new(42);
    
    let params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),           // 5% drift
        volatility: FixedPoint::from_float(0.2),       // 20% volatility
        jump_intensity: FixedPoint::from_float(0.1),   // 0.1 jumps per year
        jump_mean: FixedPoint::from_float(0.0),        // Zero mean jumps
        jump_volatility: FixedPoint::from_float(0.1),  // 10% jump volatility
        risk_free_rate: FixedPoint::from_float(0.03),  // 3% risk-free rate
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    // Simulate 1 year with daily steps
    let n_steps = 252;
    let t_span = (FixedPoint::zero(), FixedPoint::one());
    
    // Euler-Maruyama solution
    let euler_path = euler_solver.solve_path(t_span, initial.clone(), &params, n_steps, &mut rng)?;
    
    // Reset RNG for fair comparison
    rng = DeterministicRng::new(42);
    
    // Milstein solution
    let milstein_path = milstein_solver.solve_path(t_span, initial, &params, n_steps, &mut rng)?;
    
    println!("  Initial Price: ${:.2}", euler_path[0].price.to_float());
    println!("  Final Price (Euler): ${:.2}", euler_path.last().unwrap().price.to_float());
    println!("  Final Price (Milstein): ${:.2}", milstein_path.last().unwrap().price.to_float());
    println!("  Number of jumps (Euler): {}", euler_solver.jump_buffer.len());
    println!("  Number of jumps (Milstein): {}", milstein_solver.jump_buffer.len());
    
    // Calculate returns
    let euler_return = (euler_path.last().unwrap().price / euler_path[0].price).ln();
    let milstein_return = (milstein_path.last().unwrap().price / milstein_path[0].price).ln();
    
    println!("  Annual Return (Euler): {:.2}%", euler_return.to_float() * 100.0);
    println!("  Annual Return (Milstein): {:.2}%", milstein_return.to_float() * 100.0);
    
    Ok(())
}

fn demonstrate_kou_model() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating Kou double exponential jump-diffusion...");
    
    let mut solver = KouJumpSolver::new(100);
    let mut rng = DeterministicRng::new(123);
    
    let params = KouJumpParams {
        drift: FixedPoint::from_float(0.05),           // 5% drift
        volatility: FixedPoint::from_float(0.2),       // 20% volatility
        jump_intensity: FixedPoint::from_float(0.2),   // 0.2 jumps per year
        prob_up_jump: FixedPoint::from_float(0.6),     // 60% probability of up jump
        eta_up: FixedPoint::from_float(10.0),          // Upward jump rate
        eta_down: FixedPoint::from_float(5.0),         // Downward jump rate
    };
    
    let initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        &mut rng,
    )?;
    
    println!("  Initial Price: ${:.2}", path[0].price.to_float());
    println!("  Final Price: ${:.2}", path.last().unwrap().price.to_float());
    println!("  Number of jumps: {}", solver.jump_buffer.len());
    
    // Analyze jump sizes
    if !solver.jump_buffer.is_empty() {
        let avg_jump: f64 = solver.jump_buffer.iter()
            .map(|j| j.to_float())
            .sum::<f64>() / solver.jump_buffer.len() as f64;
        
        let max_jump = solver.jump_buffer.iter()
            .map(|j| j.to_float())
            .fold(f64::NEG_INFINITY, f64::max);
        
        let min_jump = solver.jump_buffer.iter()
            .map(|j| j.to_float())
            .fold(f64::INFINITY, f64::min);
        
        println!("  Average Jump Size: {:.4}", avg_jump);
        println!("  Max Jump Size: {:.4}", max_jump);
        println!("  Min Jump Size: {:.4}", min_jump);
    }
    
    let annual_return = (path.last().unwrap().price / path[0].price).ln();
    println!("  Annual Return: {:.2}%", annual_return.to_float() * 100.0);
    
    Ok(())
}

fn demonstrate_heston_model() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating Heston stochastic volatility model...");
    
    let mut solver = HestonSolver::new();
    let mut rng = DeterministicRng::new(456);
    
    let params = HestonParams {
        risk_free_rate: FixedPoint::from_float(0.05),     // 5% risk-free rate
        kappa: FixedPoint::from_float(2.0),               // Mean reversion speed
        theta: FixedPoint::from_float(0.04),              // Long-term variance (20% vol)
        sigma: FixedPoint::from_float(0.3),               // Vol of vol
        rho: FixedPoint::from_float(-0.7),                // Negative correlation
        initial_variance: FixedPoint::from_float(0.04),   // Initial variance
    };
    
    let initial = HestonState {
        price: FixedPoint::from_float(100.0),
        variance: FixedPoint::from_float(0.04),
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        &mut rng,
    )?;
    
    println!("  Initial Price: ${:.2}", path[0].price.to_float());
    println!("  Final Price: ${:.2}", path.last().unwrap().price.to_float());
    println!("  Initial Volatility: {:.2}%", (path[0].variance.to_float().sqrt() * 100.0));
    println!("  Final Volatility: {:.2}%", (path.last().unwrap().variance.to_float().sqrt() * 100.0));
    
    // Calculate average volatility
    let avg_variance: f64 = path.iter()
        .map(|state| state.variance.to_float())
        .sum::<f64>() / path.len() as f64;
    
    println!("  Average Volatility: {:.2}%", (avg_variance.sqrt() * 100.0));
    
    // Find min/max volatility
    let min_vol = path.iter()
        .map(|state| state.variance.to_float().sqrt())
        .fold(f64::INFINITY, f64::min);
    
    let max_vol = path.iter()
        .map(|state| state.variance.to_float().sqrt())
        .fold(f64::NEG_INFINITY, f64::max);
    
    println!("  Min Volatility: {:.2}%", min_vol * 100.0);
    println!("  Max Volatility: {:.2}%", max_vol * 100.0);
    
    let annual_return = (path.last().unwrap().price / path[0].price).ln();
    println!("  Annual Return: {:.2}%", annual_return.to_float() * 100.0);
    
    Ok(())
}

fn demonstrate_rough_volatility() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating rough volatility model...");
    
    let mut solver = RoughVolatilitySolver::new(50);
    let mut rng = DeterministicRng::new(789);
    
    let params = RoughVolatilityParams {
        hurst_parameter: FixedPoint::from_float(0.1),     // Rough parameter H < 0.5
        vol_of_vol: FixedPoint::from_float(0.3),          // Volatility of volatility
        mean_reversion: FixedPoint::from_float(2.0),      // Mean reversion speed
        long_term_var: FixedPoint::from_float(0.04),      // Long-term variance
        correlation: FixedPoint::from_float(-0.7),        // Price-vol correlation
        initial_vol: FixedPoint::from_float(0.2),         // Initial volatility
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
        100, // Fewer steps due to computational complexity
        &mut rng,
    )?;
    
    println!("  Initial Price: ${:.2}", path[0].price.to_float());
    println!("  Final Price: ${:.2}", path.last().unwrap().price.to_float());
    println!("  Initial Volatility: {:.2}%", path[0].volatility.to_float() * 100.0);
    println!("  Final Volatility: {:.2}%", path.last().unwrap().volatility.to_float() * 100.0);
    println!("  Hurst Parameter: {:.2}", params.hurst_parameter.to_float());
    println!("  FBM History Length: {}", path.last().unwrap().fbm_history.len());
    
    // Analyze volatility clustering
    let vol_changes: Vec<f64> = path.windows(2)
        .map(|w| (w[1].volatility - w[0].volatility).to_float())
        .collect();
    
    let avg_vol_change: f64 = vol_changes.iter().sum::<f64>() / vol_changes.len() as f64;
    let vol_change_std: f64 = {
        let variance = vol_changes.iter()
            .map(|&x| (x - avg_vol_change).powi(2))
            .sum::<f64>() / vol_changes.len() as f64;
        variance.sqrt()
    };
    
    println!("  Average Vol Change: {:.6}", avg_vol_change);
    println!("  Vol Change Std: {:.6}", vol_change_std);
    
    let annual_return = (path.last().unwrap().price / path[0].price).ln();
    println!("  Annual Return: {:.2}%", annual_return.to_float() * 100.0);
    
    Ok(())
}

fn demonstrate_multidimensional_gbm() -> Result<(), Box<dyn std::error::Error>> {
    println!("Simulating multi-dimensional correlated GBM...");
    
    let mut solver = MultiDimensionalGBMSolver::new();
    let mut rng = DeterministicRng::new(321);
    
    // 3-asset portfolio with different correlations
    let correlation_matrix = vec![
        vec![FixedPoint::one(), FixedPoint::from_float(0.6), FixedPoint::from_float(0.3)],
        vec![FixedPoint::from_float(0.6), FixedPoint::one(), FixedPoint::from_float(0.4)],
        vec![FixedPoint::from_float(0.3), FixedPoint::from_float(0.4), FixedPoint::one()],
    ];
    
    let params = MultiDimensionalParams {
        drifts: vec![
            FixedPoint::from_float(0.05),  // Asset 1: 5% drift
            FixedPoint::from_float(0.07),  // Asset 2: 7% drift
            FixedPoint::from_float(0.04),  // Asset 3: 4% drift
        ],
        volatilities: vec![
            FixedPoint::from_float(0.2),   // Asset 1: 20% vol
            FixedPoint::from_float(0.25),  // Asset 2: 25% vol
            FixedPoint::from_float(0.15),  // Asset 3: 15% vol
        ],
        correlation_matrix,
        n_assets: 3,
    };
    
    let initial = MultiDimensionalState {
        prices: vec![
            FixedPoint::from_float(100.0),  // Asset 1: $100
            FixedPoint::from_float(50.0),   // Asset 2: $50
            FixedPoint::from_float(200.0),  // Asset 3: $200
        ],
        volatilities: vec![
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.25),
            FixedPoint::from_float(0.15),
        ],
        time: FixedPoint::zero(),
    };
    
    let path = solver.solve_path(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        &mut rng,
    )?;
    
    println!("  Number of Assets: {}", params.n_assets);
    println!("  Simulation Steps: {}", path.len() - 1);
    
    for i in 0..params.n_assets {
        let initial_price = path[0].prices[i].to_float();
        let final_price = path.last().unwrap().prices[i].to_float();
        let annual_return = (final_price / initial_price).ln() * 100.0;
        
        println!("  Asset {} - Initial: ${:.2}, Final: ${:.2}, Return: {:.2}%", 
                 i + 1, initial_price, final_price, annual_return);
    }
    
    // Calculate portfolio return (equal weights)
    let portfolio_initial: f64 = path[0].prices.iter()
        .map(|p| p.to_float())
        .sum::<f64>() / params.n_assets as f64;
    
    let portfolio_final: f64 = path.last().unwrap().prices.iter()
        .map(|p| p.to_float())
        .sum::<f64>() / params.n_assets as f64;
    
    let portfolio_return = (portfolio_final / portfolio_initial).ln() * 100.0;
    println!("  Portfolio Return (equal weights): {:.2}%", portfolio_return);
    
    Ok(())
}

fn demonstrate_monte_carlo_simulation() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running Monte Carlo simulation with statistical analysis...");
    
    let solver = EulerMaruyamaGBMJump::new(50);
    let mut mc_simulator = MonteCarloSimulator::new(solver, 1000, true); // Use parallel processing
    
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
    
    println!("  Simulating {} paths...", mc_simulator.n_paths);
    
    let start_time = std::time::Instant::now();
    let paths = mc_simulator.simulate_paths(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        252,
        12345,
    )?;
    let simulation_time = start_time.elapsed();
    
    println!("  Simulation completed in {:.2}s", simulation_time.as_secs_f64());
    
    // Compute comprehensive statistics
    let stats = PathStatistics::compute_from_gbm_paths(&paths)?;
    
    println!("  Statistical Results:");
    println!("    Mean Final Price: ${:.2}", stats.mean_final_price.to_float());
    println!("    Std Final Price: ${:.2}", stats.std_final_price.to_float());
    println!("    Mean Volatility: {:.2}%", stats.mean_volatility.to_float() * 100.0);
    println!("    Max Drawdown: {:.2}%", stats.max_drawdown.to_float() * 100.0);
    println!("    95% VaR: ${:.2}", stats.value_at_risk_95.to_float());
    println!("    95% Expected Shortfall: ${:.2}", stats.expected_shortfall_95.to_float());
    
    // Calculate additional metrics
    let returns: Vec<f64> = paths.iter()
        .map(|path| (path.last().unwrap().price / path[0].price).ln().to_float())
        .collect();
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let return_std = {
        let variance = returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    };
    
    let sharpe_ratio = (mean_return - 0.03) / return_std; // Assuming 3% risk-free rate
    
    println!("    Mean Annual Return: {:.2}%", mean_return * 100.0);
    println!("    Return Volatility: {:.2}%", return_std * 100.0);
    println!("    Sharpe Ratio: {:.3}", sharpe_ratio);
    
    Ok(())
}

fn demonstrate_variance_reduction() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing standard vs variance reduction Monte Carlo...");
    
    let base_solver = EulerMaruyamaGBMJump::new(50);
    let n_paths = 500;
    
    // Standard Monte Carlo
    let mut standard_mc = MonteCarloSimulator::new(base_solver.clone(), n_paths, false);
    
    // Variance reduction Monte Carlo
    let vr_base_mc = MonteCarloSimulator::new(base_solver, n_paths, false);
    let mut vr_mc = VarianceReductionMC::new(vr_base_mc, true, false);
    
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
    
    // Run standard simulation
    println!("  Running standard Monte Carlo...");
    let start_time = std::time::Instant::now();
    let standard_paths = standard_mc.simulate_paths(
        (FixedPoint::zero(), FixedPoint::one()),
        initial.clone(),
        &params,
        100,
        54321,
    )?;
    let standard_time = start_time.elapsed();
    
    // Run variance reduction simulation
    println!("  Running variance reduction Monte Carlo...");
    let start_time = std::time::Instant::now();
    let vr_paths = vr_mc.simulate_paths_with_variance_reduction(
        (FixedPoint::zero(), FixedPoint::one()),
        initial,
        &params,
        100,
        54321,
    )?;
    let vr_time = start_time.elapsed();
    
    // Compare results
    let standard_stats = PathStatistics::compute_from_gbm_paths(&standard_paths)?;
    let vr_stats = PathStatistics::compute_from_gbm_paths(&vr_paths)?;
    
    println!("  Comparison Results:");
    println!("    Standard MC:");
    println!("      Time: {:.3}s", standard_time.as_secs_f64());
    println!("      Mean Price: ${:.2}", standard_stats.mean_final_price.to_float());
    println!("      Std Price: ${:.2}", standard_stats.std_final_price.to_float());
    
    println!("    Variance Reduction MC:");
    println!("      Time: {:.3}s", vr_time.as_secs_f64());
    println!("      Mean Price: ${:.2}", vr_stats.mean_final_price.to_float());
    println!("      Std Price: ${:.2}", vr_stats.std_final_price.to_float());
    
    let variance_reduction_ratio = (standard_stats.std_final_price / vr_stats.std_final_price).to_float();
    println!("    Variance Reduction Ratio: {:.3}", variance_reduction_ratio);
    
    Ok(())
}

fn demonstrate_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing different SDE models...");
    
    let mut rng = DeterministicRng::new(99999);
    let n_steps = 252;
    let t_span = (FixedPoint::zero(), FixedPoint::one());
    
    // 1. Standard GBM (no jumps)
    let mut gbm_solver = EulerMaruyamaGBMJump::new(10);
    let gbm_params = GBMJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::zero(), // No jumps
        jump_mean: FixedPoint::zero(),
        jump_volatility: FixedPoint::zero(),
        risk_free_rate: FixedPoint::from_float(0.03),
    };
    
    let gbm_initial = GBMJumpState {
        price: FixedPoint::from_float(100.0),
        volatility: FixedPoint::from_float(0.2),
        time: FixedPoint::zero(),
    };
    
    let gbm_path = gbm_solver.solve_path(t_span, gbm_initial, &gbm_params, n_steps, &mut rng)?;
    
    // 2. Heston model
    rng = DeterministicRng::new(99999); // Reset for fair comparison
    let mut heston_solver = HestonSolver::new();
    let heston_params = HestonParams {
        risk_free_rate: FixedPoint::from_float(0.05),
        kappa: FixedPoint::from_float(2.0),
        theta: FixedPoint::from_float(0.04),
        sigma: FixedPoint::from_float(0.3),
        rho: FixedPoint::from_float(-0.7),
        initial_variance: FixedPoint::from_float(0.04),
    };
    
    let heston_initial = HestonState {
        price: FixedPoint::from_float(100.0),
        variance: FixedPoint::from_float(0.04),
        time: FixedPoint::zero(),
    };
    
    let heston_path = heston_solver.solve_path(t_span, heston_initial, &heston_params, n_steps, &mut rng)?;
    
    // 3. Kou jump model
    rng = DeterministicRng::new(99999); // Reset for fair comparison
    let mut kou_solver = KouJumpSolver::new(100);
    let kou_params = KouJumpParams {
        drift: FixedPoint::from_float(0.05),
        volatility: FixedPoint::from_float(0.2),
        jump_intensity: FixedPoint::from_float(0.1),
        prob_up_jump: FixedPoint::from_float(0.6),
        eta_up: FixedPoint::from_float(10.0),
        eta_down: FixedPoint::from_float(5.0),
    };
    
    let kou_path = kou_solver.solve_path(t_span, gbm_initial, &kou_params, n_steps, &mut rng)?;
    
    // Compare results
    println!("  Model Comparison (1-year simulation):");
    
    let gbm_return = (gbm_path.last().unwrap().price / gbm_path[0].price).ln().to_float() * 100.0;
    let heston_return = (heston_path.last().unwrap().price / heston_path[0].price).ln().to_float() * 100.0;
    let kou_return = (kou_path.last().unwrap().price / kou_path[0].price).ln().to_float() * 100.0;
    
    println!("    Standard GBM:");
    println!("      Final Price: ${:.2}", gbm_path.last().unwrap().price.to_float());
    println!("      Annual Return: {:.2}%", gbm_return);
    
    println!("    Heston Model:");
    println!("      Final Price: ${:.2}", heston_path.last().unwrap().price.to_float());
    println!("      Annual Return: {:.2}%", heston_return);
    println!("      Final Volatility: {:.2}%", heston_path.last().unwrap().variance.to_float().sqrt() * 100.0);
    
    println!("    Kou Jump Model:");
    println!("      Final Price: ${:.2}", kou_path.last().unwrap().price.to_float());
    println!("      Annual Return: {:.2}%", kou_return);
    println!("      Number of Jumps: {}", kou_solver.jump_buffer.len());
    
    // Calculate path volatilities (realized volatility)
    let gbm_vol = calculate_realized_volatility(&gbm_path);
    let heston_vol = calculate_realized_volatility_heston(&heston_path);
    let kou_vol = calculate_realized_volatility(&kou_path);
    
    println!("    Realized Volatilities:");
    println!("      GBM: {:.2}%", gbm_vol * 100.0);
    println!("      Heston: {:.2}%", heston_vol * 100.0);
    println!("      Kou: {:.2}%", kou_vol * 100.0);
    
    Ok(())
}

fn calculate_realized_volatility(path: &[GBMJumpState]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    
    let returns: Vec<f64> = path.windows(2)
        .map(|w| (w[1].price / w[0].price).ln().to_float())
        .collect();
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    // Annualize (assuming daily returns)
    (variance * 252.0).sqrt()
}

fn calculate_realized_volatility_heston(path: &[HestonState]) -> f64 {
    if path.len() < 2 {
        return 0.0;
    }
    
    let returns: Vec<f64> = path.windows(2)
        .map(|w| (w[1].price / w[0].price).ln().to_float())
        .collect();
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    
    // Annualize (assuming daily returns)
    (variance * 252.0).sqrt()
}