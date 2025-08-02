//! Advanced Inventory Management System Demo
//!
//! This example demonstrates the comprehensive inventory management system including:
//! - Portfolio risk metrics calculation (VaR, Expected Shortfall, Maximum Drawdown)
//! - Dynamic hedging strategies with multi-asset optimization
//! - Kelly criterion position sizing with risk overlay
//! - Real-time portfolio optimization with constraints

use advanced_trading_system::math::fixed_point::FixedPoint;
use advanced_trading_system::risk::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Inventory Management System Demo ===\n");
    
    // Create a sample portfolio
    let mut portfolio = create_sample_portfolio();
    println!("Initial Portfolio:");
    print_portfolio(&portfolio);
    
    // Create market data
    let market_data = create_sample_market_data();
    println!("\nMarket Data:");
    print_market_data(&market_data);
    
    // 1. Portfolio Risk Metrics
    println!("\n=== 1. Portfolio Risk Metrics ===");
    demo_risk_metrics(&mut portfolio, &market_data)?;
    
    // 2. Dynamic Hedging
    println!("\n=== 2. Dynamic Hedging Framework ===");
    demo_dynamic_hedging(&portfolio, &market_data)?;
    
    // 3. Kelly Criterion Position Sizing
    println!("\n=== 3. Kelly Criterion Position Sizing ===");
    demo_kelly_criterion(&portfolio, &market_data)?;
    
    // 4. Portfolio Optimization
    println!("\n=== 4. Portfolio Optimization ===");
    demo_portfolio_optimization(&portfolio, &market_data)?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

fn create_sample_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::new(FixedPoint::from_float(50000.0)); // $50k cash
    
    // Add some positions
    portfolio.add_position(Position::new(
        "AAPL".to_string(),
        200,
        FixedPoint::from_float(150.0),
        FixedPoint::from_float(155.0),
    ));
    
    portfolio.add_position(Position::new(
        "GOOGL".to_string(),
        50,
        FixedPoint::from_float(2800.0),
        FixedPoint::from_float(2850.0),
    ));
    
    portfolio.add_position(Position::new(
        "MSFT".to_string(),
        100,
        FixedPoint::from_float(300.0),
        FixedPoint::from_float(310.0),
    ));
    
    portfolio.add_position(Position::new(
        "TSLA".to_string(),
        75,
        FixedPoint::from_float(200.0),
        FixedPoint::from_float(195.0),
    ));
    
    portfolio
}

fn create_sample_market_data() -> MarketData {
    let mut market_data = MarketData::new();
    
    // Add current prices
    market_data.add_price("AAPL".to_string(), FixedPoint::from_float(155.0));
    market_data.add_price("GOOGL".to_string(), FixedPoint::from_float(2850.0));
    market_data.add_price("MSFT".to_string(), FixedPoint::from_float(310.0));
    market_data.add_price("TSLA".to_string(), FixedPoint::from_float(195.0));
    market_data.add_price("SPY".to_string(), FixedPoint::from_float(450.0));
    
    // Add volatilities
    market_data.add_volatility("AAPL".to_string(), FixedPoint::from_float(0.25));
    market_data.add_volatility("GOOGL".to_string(), FixedPoint::from_float(0.30));
    market_data.add_volatility("MSFT".to_string(), FixedPoint::from_float(0.22));
    market_data.add_volatility("TSLA".to_string(), FixedPoint::from_float(0.45));
    market_data.add_volatility("SPY".to_string(), FixedPoint::from_float(0.16));
    
    // Add correlations
    let correlations = vec![
        (("AAPL".to_string(), "GOOGL".to_string()), 0.65),
        (("AAPL".to_string(), "MSFT".to_string()), 0.70),
        (("AAPL".to_string(), "TSLA".to_string()), 0.45),
        (("AAPL".to_string(), "SPY".to_string()), 0.80),
        (("GOOGL".to_string(), "MSFT".to_string()), 0.75),
        (("GOOGL".to_string(), "TSLA".to_string()), 0.40),
        (("GOOGL".to_string(), "SPY".to_string()), 0.75),
        (("MSFT".to_string(), "TSLA".to_string()), 0.35),
        (("MSFT".to_string(), "SPY".to_string()), 0.85),
        (("TSLA".to_string(), "SPY".to_string()), 0.50),
    ];
    
    for ((asset1, asset2), corr) in correlations {
        market_data.add_correlation(asset1, asset2, FixedPoint::from_float(corr));
    }
    
    // Add sample historical returns (simplified)
    let assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"];
    let base_returns = [
        vec![0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.008, -0.005, 0.012],
        vec![0.012, -0.018, 0.02, -0.008, 0.003, 0.025, -0.012, 0.01, -0.003, 0.015],
        vec![0.008, -0.015, 0.012, -0.006, 0.004, 0.018, -0.01, 0.006, -0.002, 0.009],
        vec![0.025, -0.035, 0.03, -0.02, 0.01, 0.04, -0.025, 0.015, -0.01, 0.02],
        vec![0.006, -0.012, 0.009, -0.005, 0.003, 0.012, -0.008, 0.005, -0.002, 0.007],
    ];
    
    for (i, asset) in assets.iter().enumerate() {
        let returns: Vec<FixedPoint> = base_returns[i].iter()
            .map(|&r| FixedPoint::from_float(r))
            .collect();
        market_data.add_returns(asset.to_string(), returns);
    }
    
    market_data
}

fn demo_risk_metrics(portfolio: &mut Portfolio, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut calculator = RiskMetricsCalculator::new(1000, MonteCarloParams::default());
    
    // Calculate risk metrics using different methods
    let methods = [VaRMethod::Parametric, VaRMethod::Historical, VaRMethod::MonteCarlo];
    
    for method in &methods {
        println!("Risk Metrics ({:?}):", method);
        
        let risk_metrics = calculator.calculate_risk_metrics(portfolio, market_data, *method)?;
        
        println!("  VaR (95%): ${:.2}", risk_metrics.var_95.to_float());
        println!("  VaR (99%): ${:.2}", risk_metrics.var_99.to_float());
        println!("  Expected Shortfall (95%): ${:.2}", risk_metrics.expected_shortfall_95.to_float());
        println!("  Portfolio Volatility: {:.2}%", risk_metrics.portfolio_volatility.to_float() * 100.0);
        println!("  Maximum Drawdown: {:.2}%", risk_metrics.maximum_drawdown.to_float() * 100.0);
        println!("  Diversification Ratio: {:.2}", risk_metrics.diversification_ratio.to_float());
        
        println!("  Risk Contributions:");
        for (asset, contribution) in &risk_metrics.risk_contributions {
            println!("    {}: {:.2}%", asset, contribution.to_float() * 100.0);
        }
        println!();
    }
    
    Ok(())
}

fn demo_dynamic_hedging(portfolio: &Portfolio, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut hedging_framework = DynamicHedgingFramework::new(HedgingParams::default());
    
    // Set up hedging instruments
    hedging_framework.add_hedging_instrument("AAPL".to_string(), vec!["SPY".to_string()]);
    hedging_framework.add_hedging_instrument("GOOGL".to_string(), vec!["SPY".to_string()]);
    hedging_framework.add_hedging_instrument("MSFT".to_string(), vec!["SPY".to_string()]);
    hedging_framework.add_hedging_instrument("TSLA".to_string(), vec!["SPY".to_string()]);
    
    // Set transaction costs
    hedging_framework.set_transaction_cost("SPY".to_string(), FixedPoint::from_float(0.001));
    
    // Calculate hedge recommendations
    let recommendations = hedging_framework.calculate_hedge_recommendations(portfolio, market_data)?;
    
    println!("Hedge Recommendations:");
    for rec in &recommendations {
        println!("  {} -> {}", rec.target_asset, rec.hedge_asset);
        println!("    Hedge Ratio: {:.4}", rec.hedge_ratio.to_float());
        println!("    Hedge Quantity: {}", rec.hedge_quantity);
        println!("    Effectiveness: {:.2}%", rec.effectiveness.to_float() * 100.0);
        println!("    Transaction Cost: ${:.2}", rec.transaction_cost.to_float());
        println!();
    }
    
    // Demonstrate minimum variance hedge ratio calculation
    println!("Minimum Variance Hedge Ratios:");
    let assets = ["AAPL", "GOOGL", "MSFT", "TSLA"];
    for asset in &assets {
        let hedge_ratio = hedging_framework.calculate_minimum_variance_hedge_ratio(
            &asset.to_string(),
            &"SPY".to_string(),
            market_data,
        )?;
        println!("  {} vs SPY: {:.4}", asset, hedge_ratio.to_float());
    }
    
    Ok(())
}

fn demo_kelly_criterion(portfolio: &Portfolio, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let mut kelly_engine = KellyCriterionEngine::new(KellyParams::default());
    
    let available_capital = FixedPoint::from_float(200000.0); // $200k available
    
    // Calculate position recommendations
    let recommendations = kelly_engine.calculate_position_recommendations(
        portfolio,
        market_data,
        available_capital,
    )?;
    
    println!("Kelly Criterion Position Recommendations:");
    for rec in &recommendations {
        println!("  {}:", rec.asset_id);
        println!("    Position Fraction: {:.2}%", rec.position_fraction.to_float() * 100.0);
        println!("    Position Size: ${:.2}", rec.position_size.to_float());
        println!("    Raw Kelly: {:.2}%", rec.raw_kelly_fraction.to_float() * 100.0);
        println!("    Expected Return: {:.2}%", rec.expected_return.to_float() * 100.0);
        println!("    Risk (Volatility): {:.2}%", rec.risk.to_float() * 100.0);
        println!("    Sharpe Ratio: {:.2}", rec.sharpe_ratio.to_float());
        println!("    Confidence: {:.2}%", rec.confidence.to_float() * 100.0);
        println!();
    }
    
    // Multi-asset Kelly optimization
    let assets = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string(), "TSLA".to_string()];
    let multi_asset_result = kelly_engine.calculate_multi_asset_kelly(&assets, market_data)?;
    
    println!("Multi-Asset Kelly Optimization:");
    println!("  Expected Portfolio Return: {:.2}%", multi_asset_result.expected_portfolio_return.to_float() * 100.0);
    println!("  Portfolio Volatility: {:.2}%", multi_asset_result.portfolio_volatility.to_float() * 100.0);
    println!("  Portfolio Sharpe Ratio: {:.2}", multi_asset_result.portfolio_sharpe_ratio.to_float());
    println!("  Total Leverage: {:.2}x", multi_asset_result.total_leverage.to_float());
    println!("  Diversification Benefit: {:.2}", multi_asset_result.diversification_benefit.to_float());
    
    println!("  Optimal Position Fractions:");
    for (asset, fraction) in &multi_asset_result.position_fractions {
        println!("    {}: {:.2}%", asset, fraction.to_float() * 100.0);
    }
    
    Ok(())
}

fn demo_portfolio_optimization(portfolio: &Portfolio, market_data: &MarketData) -> Result<(), Box<dyn std::error::Error>> {
    let constraints = PortfolioConstraints::default();
    let optimizer = PortfolioOptimizer::new(
        OptimizationObjective::MaxSharpe,
        constraints,
        FixedPoint::from_float(2.0), // Risk aversion
    );
    
    let available_assets = vec!["AAPL".to_string(), "GOOGL".to_string(), "MSFT".to_string(), "TSLA".to_string()];
    
    // Optimize portfolio
    let optimization_result = optimizer.optimize_portfolio(portfolio, market_data, &available_assets)?;
    
    println!("Portfolio Optimization Results (Max Sharpe):");
    println!("  Expected Return: {:.2}%", optimization_result.expected_return.to_float() * 100.0);
    println!("  Portfolio Volatility: {:.2}%", optimization_result.portfolio_volatility.to_float() * 100.0);
    println!("  Sharpe Ratio: {:.2}", optimization_result.sharpe_ratio.to_float());
    println!("  Max Drawdown Estimate: {:.2}%", optimization_result.max_drawdown_estimate.to_float() * 100.0);
    println!("  Diversification Ratio: {:.2}", optimization_result.diversification_ratio.to_float());
    println!("  Turnover Required: {:.2}%", optimization_result.turnover.to_float() * 100.0);
    
    println!("  Optimal Weights:");
    for (asset, weight) in &optimization_result.weights {
        println!("    {}: {:.2}%", asset, weight.to_float() * 100.0);
    }
    
    if !optimization_result.constraint_violations.is_empty() {
        println!("  Constraint Violations:");
        for violation in &optimization_result.constraint_violations {
            println!("    - {}", violation);
        }
    }
    
    // Stress testing
    println!("\nStress Test Results:");
    let stress_results = optimizer.stress_test_portfolio(&optimization_result.weights, market_data)?;
    
    for result in stress_results.iter().take(5) { // Show first 5 scenarios
        println!("  {}:", result.scenario_name);
        println!("    Portfolio Loss: ${:.2} ({:.2}%)", 
                result.portfolio_loss.to_float(), 
                result.loss_percentage.to_float() * 100.0);
        if !result.worst_asset.is_empty() && result.worst_asset != "N/A" {
            println!("    Worst Asset: {}", result.worst_asset);
        }
        if result.correlation_breakdown {
            println!("    Correlation Breakdown: Yes");
        }
        println!();
    }
    
    // Performance metrics (using sample returns)
    let sample_returns = vec![
        FixedPoint::from_float(0.01),
        FixedPoint::from_float(-0.02),
        FixedPoint::from_float(0.015),
        FixedPoint::from_float(-0.01),
        FixedPoint::from_float(0.005),
        FixedPoint::from_float(0.02),
        FixedPoint::from_float(-0.015),
        FixedPoint::from_float(0.008),
        FixedPoint::from_float(-0.005),
        FixedPoint::from_float(0.012),
    ];
    
    let performance_metrics = optimizer.calculate_performance_metrics(
        &sample_returns,
        None,
        FixedPoint::from_float(0.02), // 2% risk-free rate
    )?;
    
    println!("Performance Metrics:");
    println!("  Total Return: {:.2}%", performance_metrics.total_return.to_float() * 100.0);
    println!("  Annualized Return: {:.2}%", performance_metrics.annualized_return.to_float() * 100.0);
    println!("  Volatility: {:.2}%", performance_metrics.volatility.to_float() * 100.0);
    println!("  Sharpe Ratio: {:.2}", performance_metrics.sharpe_ratio.to_float());
    println!("  Sortino Ratio: {:.2}", performance_metrics.sortino_ratio.to_float());
    println!("  Calmar Ratio: {:.2}", performance_metrics.calmar_ratio.to_float());
    println!("  Maximum Drawdown: {:.2}%", performance_metrics.max_drawdown.to_float() * 100.0);
    println!("  Win Rate: {:.2}%", performance_metrics.win_rate.to_float() * 100.0);
    println!("  Win/Loss Ratio: {:.2}", performance_metrics.win_loss_ratio.to_float());
    
    Ok(())
}

fn print_portfolio(portfolio: &Portfolio) {
    println!("  Cash: ${:.2}", portfolio.cash.to_float());
    println!("  Positions:");
    for (asset_id, position) in &portfolio.positions {
        println!("    {}: {} shares @ ${:.2} (Current: ${:.2})", 
                asset_id, 
                position.quantity, 
                position.average_price.to_float(),
                position.current_price.to_float());
        println!("      Market Value: ${:.2}, P&L: ${:.2}", 
                position.market_value().to_float(),
                position.unrealized_pnl.to_float());
    }
    println!("  Total Portfolio Value: ${:.2}", portfolio.total_market_value().to_float());
}

fn print_market_data(market_data: &MarketData) {
    println!("  Current Prices:");
    for (asset, price) in &market_data.prices {
        println!("    {}: ${:.2}", asset, price.to_float());
    }
    
    println!("  Volatilities:");
    for (asset, vol) in &market_data.volatilities {
        println!("    {}: {:.2}%", asset, vol.to_float() * 100.0);
    }
}