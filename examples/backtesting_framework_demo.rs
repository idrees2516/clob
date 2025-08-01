//! Backtesting Framework Demo
//! 
//! This example demonstrates the comprehensive backtesting framework implementation
//! including historical data replay, strategy simulation, performance attribution,
//! statistical testing, Bayesian optimization, and stress testing.

use hf_quoting_liquidity::models::backtesting_framework::{
    BacktestingFramework, BacktestingConfig, HistoricalMarketData, MarketTick,
    ObjectiveFunction, SimulationResults,
};
use hf_quoting_liquidity::models::parameter_optimization::{ParameterSpace, ParameterBounds};
use hf_quoting_liquidity::math::optimization::ModelParameters;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 High-Frequency Quoting Backtesting Framework Demo");
    println!("====================================================");

    // 1. Create sample historical market data
    let historical_data = create_sample_market_data();
    println!("✅ Created sample market data with {} ticks", historical_data.ticks.len());

    // 2. Initialize backtesting framework with default configuration
    let config = BacktestingConfig::default();
    let mut framework = BacktestingFramework::new(config)?;
    println!("✅ Initialized backtesting framework");

    // 3. Define strategy parameters
    let strategy_params = ModelParameters::default();
    println!("✅ Configured strategy parameters");

    // 4. Run comprehensive backtest
    println!("\n📊 Running comprehensive backtest...");
    let backtest_results = framework.run_comprehensive_backtest(
        &strategy_params,
        &historical_data,
    ).await?;

    // Display results
    display_backtest_results(&backtest_results);

    // 5. Demonstrate Bayesian optimization (Requirement 5.3)
    println!("\n🔍 Running Bayesian optimization for hyperparameter tuning...");
    let parameter_space = create_parameter_space();
    let objective_function: ObjectiveFunction = |results: &SimulationResults| {
        // Maximize Sharpe ratio (simplified)
        let returns: Vec<f64> = results.pnl_series.windows(2)
            .map(|w| if w[0].abs() > 1e-8 { (w[1] - w[0]) / w[0].abs() } else { 0.0 })
            .collect();
        
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_return = {
            let variance = returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            variance.sqrt()
        };
        
        if std_return > 1e-8 {
            mean_return / std_return
        } else {
            0.0
        }
    };

    let optimization_results = framework.optimize_parameters(
        &parameter_space,
        &historical_data,
        objective_function,
    ).await?;

    display_optimization_results(&optimization_results);

    // 6. Demonstrate walk-forward analysis
    println!("\n📈 Running walk-forward analysis...");
    let walk_forward_results = framework.walk_forward_analysis(
        &strategy_params,
        &historical_data,
    ).await?;

    display_walk_forward_results(&walk_forward_results);

    println!("\n🎉 Backtesting framework demo completed successfully!");
    println!("   - Historical data replay: ✅");
    println!("   - Strategy simulation: ✅");
    println!("   - Performance attribution: ✅");
    println!("   - Statistical testing: ✅");
    println!("   - Bayesian optimization: ✅");
    println!("   - Stress testing: ✅");
    println!("   - Walk-forward analysis: ✅");

    Ok(())
}

fn create_sample_market_data() -> HistoricalMarketData {
    let mut ticks = Vec::new();
    let mut rng = rand::thread_rng();
    
    let start_time = 1640995200000000u64; // 2022-01-01 00:00:00 UTC in microseconds
    let tick_interval = 1000000u64; // 1 second intervals
    
    let mut price = 100.0;
    let mut volatility = 0.02;
    
    for i in 0..10000 {
        // Simple random walk with mean reversion
        let price_change = rand::random::<f64>() * 0.002 - 0.001; // ±0.1%
        price += price_change;
        price = price.max(90.0).min(110.0); // Keep price in reasonable range
        
        // Dynamic volatility
        volatility = (volatility * 0.99 + rand::random::<f64>() * 0.001).max(0.001).min(0.05);
        
        let spread = price * 0.0001 + volatility * 0.1; // Spread based on volatility
        let bid_price = price - spread / 2.0;
        let ask_price = price + spread / 2.0;
        
        let tick = MarketTick {
            timestamp_us: start_time + i * tick_interval,
            mid_price: price,
            bid_price,
            ask_price,
            bid_size: 1000.0 + rand::random::<f64>() * 500.0,
            ask_size: 1000.0 + rand::random::<f64>() * 500.0,
            spread,
            depth: 10000.0 + rand::random::<f64>() * 5000.0,
            volatility,
        };
        
        ticks.push(tick);
    }
    
    HistoricalMarketData {
        start_time,
        end_time: start_time + (ticks.len() as u64 - 1) * tick_interval,
        ticks,
    }
}

fn create_parameter_space() -> ParameterSpace {
    ParameterSpace {
        bounds: vec![
            ParameterBounds {
                name: "drift_coefficient".to_string(),
                min: 0.01,
                max: 0.10,
            },
            ParameterBounds {
                name: "volatility_coefficient".to_string(),
                min: 0.1,
                max: 0.5,
            },
            ParameterBounds {
                name: "inventory_penalty".to_string(),
                min: 0.001,
                max: 0.05,
            },
            ParameterBounds {
                name: "adverse_selection_cost".to_string(),
                min: 0.0001,
                max: 0.01,
            },
        ],
    }
}

fn display_backtest_results(results: &hf_quoting_liquidity::models::backtesting_framework::ComprehensiveBacktestResults) {
    println!("\n📊 Backtest Results Summary:");
    println!("   Total Ticks Processed: {}", results.metadata.total_ticks);
    println!("   Total Trades: {}", results.simulation_results.trades.len());
    println!("   Final P&L: ${:.2}", results.simulation_results.pnl_series.last().unwrap_or(&0.0));
    println!("   Total P&L Attribution: ${:.2}", results.attribution_results.total_pnl);
    
    println!("\n📈 Performance Attribution:");
    for (component, value) in &results.attribution_results.attribution_breakdown {
        println!("   {}: ${:.2}", component, value);
    }
    
    println!("\n📊 Statistical Test Results:");
    println!("   Sharpe Ratio p-value: {:.4}", results.statistical_results.sharpe_ratio_pvalue);
    println!("   Returns Significant: {}", results.statistical_results.return_significance);
    
    println!("\n🚨 Stress Test Results:");
    println!("   Robustness Score: {:.2}", results.stress_results.robustness_score);
    for (scenario, result) in &results.stress_results.scenario_results {
        println!("   {}: Max DD {:.2}%, Recovery {} periods, VaR99 {:.2}%", 
                 scenario, result.max_drawdown * 100.0, result.recovery_time, result.var_99 * 100.0);
    }
}

fn display_optimization_results(results: &hf_quoting_liquidity::models::backtesting_framework::OptimizationResults) {
    println!("\n🔍 Bayesian Optimization Results:");
    println!("   Best Score: {:.6}", results.best_score);
    println!("   Optimization Iterations: {}", results.optimization_history.len());
    
    println!("\n🎯 Best Parameters:");
    for (param_name, param_value) in &results.best_parameters {
        println!("   {}: {:.6}", param_name, param_value.to_f64());
    }
    
    if results.optimization_history.len() >= 5 {
        println!("\n📈 Optimization Progress (last 5 iterations):");
        for iteration in results.optimization_history.iter().rev().take(5).rev() {
            println!("   Iteration {}: Score = {:.6}", iteration.iteration, iteration.score);
        }
    }
}

fn display_walk_forward_results(results: &hf_quoting_liquidity::models::backtesting_framework::WalkForwardResults) {
    println!("\n📈 Walk-Forward Analysis Results:");
    println!("   Total Periods: {}", results.periods.len());
    
    if !results.periods.is_empty() {
        let avg_pnl: f64 = results.periods.iter()
            .map(|period| period.results.simulation_results.pnl_series.last().unwrap_or(&0.0))
            .sum::<f64>() / results.periods.len() as f64;
        
        println!("   Average P&L per Period: ${:.2}", avg_pnl);
        
        let consistent_periods = results.periods.iter()
            .filter(|period| period.results.simulation_results.pnl_series.last().unwrap_or(&0.0) > &0.0)
            .count();
        
        println!("   Profitable Periods: {}/{} ({:.1}%)", 
                 consistent_periods, results.periods.len(), 
                 consistent_periods as f64 / results.periods.len() as f64 * 100.0);
    }
}