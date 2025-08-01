use hf_quoting_liquidity_clob::math::fixed_point::FixedPoint;
use hf_quoting_liquidity_clob::models::parameter_optimization::*;
use std::collections::HashMap;

/// Example demonstrating parameter optimization for high-frequency quoting strategies
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("High-Frequency Quoting Parameter Optimization Demo");
    println!("==================================================");

    // Demo 1: Grid Search Optimization
    println!("\n1. Grid Search Optimization");
    println!("---------------------------");
    demo_grid_search()?;

    // Demo 2: Random Search Optimization
    println!("\n2. Random Search Optimization");
    println!("-----------------------------");
    demo_random_search()?;

    // Demo 3: Bayesian Optimization
    println!("\n3. Bayesian Optimization");
    println!("------------------------");
    demo_bayesian_optimization()?;

    // Demo 4: Cross-Validation Framework
    println!("\n4. Cross-Validation Framework");
    println!("-----------------------------");
    demo_cross_validation()?;

    // Demo 5: Walk-Forward Analysis
    println!("\n5. Walk-Forward Analysis");
    println!("------------------------");
    demo_walk_forward_analysis()?;

    println!("\nParameter optimization demo completed successfully!");
    Ok(())
}

/// Demo grid search optimization
fn demo_grid_search() -> Result<(), Box<dyn std::error::Error>> {
    // Define parameter space for quoting strategy
    let mut space = ParameterSpace::new();
    space.add_parameter(
        "inventory_penalty".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.01), FixedPoint::from_float(1.0))
    );
    space.add_parameter(
        "adverse_selection_cost".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.001), FixedPoint::from_float(0.1))
    );
    space.add_parameter(
        "volatility_coefficient".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.01), FixedPoint::from_float(0.5))
    );

    // Create grid search optimizer
    let optimizer = GridSearchOptimizer::new(space, 4); // 4^3 = 64 evaluations

    // Define objective function (simplified Sharpe ratio maximization)
    let objective = SharpeRatioObjective::new();

    // Run optimization
    let result = optimizer.optimize(&objective)?;

    println!("Grid Search Results:");
    println!("  Best Sharpe Ratio: {:.4}", result.best_value.to_float());
    println!("  Best Parameters:");
    for (name, value) in &result.best_parameters {
        println!("    {}: {:.6}", name, value.to_float());
    }
    println!("  Total Evaluations: {}", result.evaluations.len());

    Ok(())
}

/// Demo random search optimization
fn demo_random_search() -> Result<(), Box<dyn std::error::Error>> {
    // Define parameter space
    let mut space = ParameterSpace::new();
    space.add_parameter(
        "market_impact_coefficient".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.001), FixedPoint::from_float(0.1)).log_scale()
    );
    space.add_parameter(
        "risk_aversion".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.1), FixedPoint::from_float(10.0))
    );

    // Create random search optimizer
    let optimizer = RandomSearchOptimizer::new(space, 100).with_seed(42);

    // Define objective function
    let objective = InformationRatioObjective::new();

    // Run optimization
    let result = optimizer.optimize(&objective)?;

    println!("Random Search Results:");
    println!("  Best Information Ratio: {:.4}", result.best_value.to_float());
    println!("  Best Parameters:");
    for (name, value) in &result.best_parameters {
        println!("    {}: {:.6}", name, value.to_float());
    }
    println!("  Total Evaluations: {}", result.evaluations.len());

    // Show convergence history
    let mut best_so_far = FixedPoint::from_float(f64::NEG_INFINITY);
    let convergence_history: Vec<_> = result.evaluations.iter()
        .map(|eval| {
            if eval.objective_value > best_so_far {
                best_so_far = eval.objective_value;
            }
            best_so_far.to_float()
        })
        .collect();

    println!("  Convergence (every 20 evaluations):");
    for (i, &value) in convergence_history.iter().enumerate() {
        if i % 20 == 0 {
            println!("    Evaluation {}: {:.4}", i + 1, value);
        }
    }

    Ok(())
}

/// Demo Bayesian optimization
fn demo_bayesian_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Define parameter space
    let mut space = ParameterSpace::new();
    space.add_parameter(
        "drift_coefficient".to_string(),
        ParameterBounds::new(FixedPoint::from_float(-0.001), FixedPoint::from_float(0.001))
    );
    space.add_parameter(
        "volatility_coefficient".to_string(),
        ParameterBounds::new(FixedPoint::from_float(0.01), FixedPoint::from_float(0.2))
    );

    // Create Bayesian optimizer with RBF kernel
    let kernel = KernelFunction::RBF {
        length_scale: FixedPoint::from_float(0.1),
        variance: FixedPoint::from_float(1.0),
    };
    let acquisition = AcquisitionFunction::ExpectedImprovement;
    
    let mut optimizer = BayesianOptimizer::new(space, kernel, acquisition)
        .with_initial_points(8)
        .with_iterations(20)
        .with_seed(123);

    // Define objective function
    let objective = CalmarRatioObjective::new();

    // Run optimization
    let result = optimizer.optimize(&objective)?;

    println!("Bayesian Optimization Results:");
    println!("  Best Calmar Ratio: {:.4}", result.best_value.to_float());
    println!("  Best Parameters:");
    for (name, value) in &result.best_parameters {
        println!("    {}: {:.6}", name, value.to_float());
    }
    println!("  Total Evaluations: {}", result.evaluations.len());

    // Show acquisition function evolution
    println!("  Optimization Progress:");
    let initial_evals = 8;
    for (i, eval) in result.evaluations.iter().enumerate().skip(initial_evals) {
        if (i - initial_evals) % 5 == 0 {
            println!("    Iteration {}: {:.4}", i - initial_evals + 1, eval.objective_value.to_float());
        }
    }

    Ok(())
}

/// Demo cross-validation framework
fn demo_cross_validation() -> Result<(), Box<dyn std::error::Error>> {
    // Define test parameters
    let mut parameters = HashMap::new();
    parameters.insert("test_param".to_string(), FixedPoint::from_float(0.5));

    // Test K-Fold Cross-Validation
    let cv_kfold = CrossValidationFramework::new(5, ValidationMethod::KFold);
    let objective = StabilityTestObjective::new();
    let result_kfold = cv_kfold.validate(&objective, &parameters, 100)?;

    println!("K-Fold Cross-Validation (5 folds):");
    println!("  Mean Score: {:.4}", result_kfold.mean_score.to_float());
    println!("  Std Dev: {:.4}", result_kfold.std_dev.to_float());
    println!("  Fold Scores: {:?}", 
        result_kfold.fold_scores.iter().map(|x| x.to_float()).collect::<Vec<_>>());

    // Test Time Series Split
    let cv_ts = CrossValidationFramework::new(4, ValidationMethod::TimeSeriesSplit);
    let result_ts = cv_ts.validate(&objective, &parameters, 100)?;

    println!("\nTime Series Cross-Validation (4 folds):");
    println!("  Mean Score: {:.4}", result_ts.mean_score.to_float());
    println!("  Std Dev: {:.4}", result_ts.std_dev.to_float());
    println!("  Fold Scores: {:?}", 
        result_ts.fold_scores.iter().map(|x| x.to_float()).collect::<Vec<_>>());

    // Test Walk-Forward
    let cv_wf = CrossValidationFramework::new(3, ValidationMethod::WalkForward);
    let result_wf = cv_wf.validate(&objective, &parameters, 100)?;

    println!("\nWalk-Forward Cross-Validation (3 folds):");
    println!("  Mean Score: {:.4}", result_wf.mean_score.to_float());
    println!("  Std Dev: {:.4}", result_wf.std_dev.to_float());
    println!("  Fold Scores: {:?}", 
        result_wf.fold_scores.iter().map(|x| x.to_float()).collect::<Vec<_>>());

    Ok(())
}

/// Demo walk-forward analysis
fn demo_walk_forward_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Create walk-forward analyzer
    let wfa = WalkForwardAnalysis::new(
        20,  // window_size
        10,  // step_size
        30   // min_train_size
    );

    // Test parameters
    let mut parameters = HashMap::new();
    parameters.insert("strategy_param".to_string(), FixedPoint::from_float(1.2));

    // Define objective that simulates time-varying performance
    let objective = TimeVaryingObjective::new();

    // Run walk-forward analysis
    let result = wfa.analyze(&objective, &parameters, 200)?;

    println!("Walk-Forward Analysis Results:");
    println!("  Number of Windows: {}", result.windows.len());
    println!("  Mean Performance: {:.4}", result.mean_performance.to_float());
    println!("  Performance Std Dev: {:.4}", result.performance_std.to_float());
    println!("  Stability Ratio: {:.4}", result.stability_ratio.to_float());

    println!("\n  Window Performance Details:");
    for (i, window) in result.windows.iter().enumerate() {
        if i % 3 == 0 { // Show every 3rd window
            println!("    Window {}: Train[{}:{}], Test[{}:{}], Performance: {:.4}",
                i + 1,
                window.train_start,
                window.train_end,
                window.test_start,
                window.test_end,
                window.performance.to_float()
            );
        }
    }

    // Analyze performance stability
    let performance_trend = analyze_performance_trend(&result.windows);
    println!("\n  Performance Trend Analysis:");
    println!("    Trend Slope: {:.6}", performance_trend);
    if performance_trend > 0.001 {
        println!("    Strategy shows improving performance over time");
    } else if performance_trend < -0.001 {
        println!("    Strategy shows degrading performance over time");
    } else {
        println!("    Strategy shows stable performance over time");
    }

    Ok(())
}

/// Analyze performance trend across walk-forward windows
fn analyze_performance_trend(windows: &[WalkForwardWindow]) -> f64 {
    if windows.len() < 2 {
        return 0.0;
    }

    let n = windows.len() as f64;
    let sum_x = (0..windows.len()).sum::<usize>() as f64;
    let sum_y = windows.iter().map(|w| w.performance.to_float()).sum::<f64>();
    let sum_xy = windows.iter().enumerate()
        .map(|(i, w)| i as f64 * w.performance.to_float())
        .sum::<f64>();
    let sum_x2 = (0..windows.len()).map(|i| (i * i) as f64).sum::<f64>();

    // Linear regression slope
    (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
}

// Example objective functions for demonstration

/// Sharpe ratio objective (simplified)
struct SharpeRatioObjective;

impl SharpeRatioObjective {
    fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for SharpeRatioObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        let inventory_penalty = parameters.get("inventory_penalty")
            .copied().unwrap_or(FixedPoint::from_float(0.1));
        let adverse_selection = parameters.get("adverse_selection_cost")
            .copied().unwrap_or(FixedPoint::from_float(0.01));
        let volatility = parameters.get("volatility_coefficient")
            .copied().unwrap_or(FixedPoint::from_float(0.1));

        // Simplified Sharpe ratio calculation
        // In practice, this would run a full backtest
        let expected_return = FixedPoint::from_float(0.1) / inventory_penalty - adverse_selection;
        let risk = volatility + inventory_penalty * FixedPoint::from_float(0.1);
        let sharpe = expected_return / risk;

        Ok(sharpe)
    }

    fn name(&self) -> &str {
        "Sharpe Ratio"
    }
}

/// Information ratio objective
struct InformationRatioObjective;

impl InformationRatioObjective {
    fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for InformationRatioObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        let market_impact = parameters.get("market_impact_coefficient")
            .copied().unwrap_or(FixedPoint::from_float(0.01));
        let risk_aversion = parameters.get("risk_aversion")
            .copied().unwrap_or(FixedPoint::from_float(1.0));

        // Simplified information ratio
        let alpha = FixedPoint::from_float(0.05) - market_impact * FixedPoint::from_float(10.0);
        let tracking_error = FixedPoint::from_float(0.02) + market_impact / risk_aversion;
        let info_ratio = alpha / tracking_error;

        Ok(info_ratio)
    }

    fn name(&self) -> &str {
        "Information Ratio"
    }
}

/// Calmar ratio objective
struct CalmarRatioObjective;

impl CalmarRatioObjective {
    fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for CalmarRatioObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        let drift = parameters.get("drift_coefficient")
            .copied().unwrap_or(FixedPoint::zero());
        let volatility = parameters.get("volatility_coefficient")
            .copied().unwrap_or(FixedPoint::from_float(0.1));

        // Simplified Calmar ratio (return / max drawdown)
        let annual_return = drift * FixedPoint::from_float(252.0) + FixedPoint::from_float(0.05);
        let max_drawdown = volatility * FixedPoint::from_float(2.0); // Simplified
        let calmar = annual_return / max_drawdown.max(FixedPoint::from_float(0.001));

        Ok(calmar)
    }

    fn name(&self) -> &str {
        "Calmar Ratio"
    }
}

/// Stability test objective for cross-validation
struct StabilityTestObjective;

impl StabilityTestObjective {
    fn new() -> Self {
        Self
    }
}

impl ObjectiveFunction for StabilityTestObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        let param = parameters.get("test_param")
            .copied().unwrap_or(FixedPoint::from_float(0.5));

        // Simple quadratic function with some noise
        let base_value = -(param - FixedPoint::from_float(0.3)) * (param - FixedPoint::from_float(0.3));
        let noise = FixedPoint::from_float(0.01); // Small consistent noise
        
        Ok(base_value + noise)
    }

    fn name(&self) -> &str {
        "Stability Test"
    }
}

/// Time-varying objective for walk-forward analysis
struct TimeVaryingObjective {
    call_count: std::cell::RefCell<usize>,
}

impl TimeVaryingObjective {
    fn new() -> Self {
        Self {
            call_count: std::cell::RefCell::new(0),
        }
    }
}

impl ObjectiveFunction for TimeVaryingObjective {
    fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
        let param = parameters.get("strategy_param")
            .copied().unwrap_or(FixedPoint::from_float(1.0));

        // Simulate time-varying performance
        let mut count = self.call_count.borrow_mut();
        *count += 1;
        let time_factor = FixedPoint::from_float((*count as f64 * 0.01).sin() * 0.1);
        
        // Base performance that varies with parameter and time
        let base_performance = param * FixedPoint::from_float(0.8) + time_factor;
        
        Ok(base_performance)
    }

    fn name(&self) -> &str {
        "Time Varying"
    }
}