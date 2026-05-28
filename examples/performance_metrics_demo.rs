use hf_quoting_liquidity::models::performance_metrics::{
    PerformanceCalculator, PerformanceData, TradeData, MarketData, TradeSide,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== High-Frequency Trading Performance Metrics Demo ===\n");

    // Create sample trading data
    let performance_data = create_sample_trading_data();
    
    // Initialize performance calculator
    let calculator = PerformanceCalculator::new(252.0 * 24.0 * 60.0); // Minute-level data
    
    // Calculate comprehensive performance metrics
    println!("Calculating comprehensive performance metrics...\n");
    let metrics = calculator.calculate_metrics(&performance_data)?;
    
    // Display results
    display_performance_metrics(&metrics);
    
    Ok(())
}

fn create_sample_trading_data() -> PerformanceData {
    // Simulate a day of high-frequency trading returns (minute-level)
    let returns = vec![
        0.0001, -0.0002, 0.0003, -0.0001, 0.0002, 0.0001, -0.0003, 0.0004,
        -0.0001, 0.0002, -0.0002, 0.0001, 0.0003, -0.0001, 0.0002, -0.0002,
        0.0001, 0.0003, -0.0004, 0.0002, 0.0001, -0.0001, 0.0002, -0.0003,
        0.0004, -0.0002, 0.0001, 0.0002, -0.0001, 0.0003, -0.0002, 0.0001,
        0.0002, -0.0003, 0.0001, 0.0004, -0.0001, 0.0002, -0.0002, 0.0001,
        0.0003, -0.0001, 0.0002, -0.0004, 0.0003, 0.0001, -0.0002, 0.0002,
        -0.0001, 0.0003, -0.0002, 0.0001, 0.0002, -0.0003, 0.0004, -0.0001,
        0.0002, -0.0002, 0.0001, 0.0003, -0.0001, 0.0002, -0.0003, 0.0001,
    ];
    
    // Generate corresponding prices
    let mut prices = vec![100.0];
    for &ret in &returns {
        let last_price = *prices.last().unwrap();
        prices.push(last_price * (1.0 + ret));
    }
    
    // Generate position history (market making positions)
    let positions = vec![
        0.0, 100.0, 50.0, 150.0, 75.0, 125.0, 25.0, 175.0, 100.0, 200.0,
        150.0, 75.0, 125.0, 50.0, 100.0, 175.0, 125.0, 75.0, 150.0, 100.0,
        50.0, 125.0, 175.0, 100.0, 75.0, 150.0, 25.0, 175.0, 125.0, 50.0,
        100.0, 150.0, 75.0, 125.0, 175.0, 100.0, 50.0, 125.0, 75.0, 150.0,
        100.0, 175.0, 125.0, 50.0, 75.0, 150.0, 100.0, 125.0, 175.0, 75.0,
        150.0, 100.0, 50.0, 125.0, 175.0, 100.0, 75.0, 150.0, 125.0, 50.0,
        100.0, 175.0, 150.0, 75.0, 125.0,
    ];
    
    // Generate trade data
    let mut trades = Vec::new();
    for i in 1..positions.len() {
        let position_change = positions[i] - positions[i-1];
        if position_change.abs() > 1e-6 {
            trades.push(TradeData {
                timestamp: i as f64,
                price: prices[i],
                quantity: position_change,
                side: if position_change > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
                transaction_cost: position_change.abs() * 0.001, // 0.1% transaction cost
                market_impact: position_change.abs() * 0.0005, // 0.05% market impact
                spread_at_trade: prices[i] * 0.0002, // 2 bps spread
            });
        }
    }
    
    // Generate market data
    let mut market_data = Vec::new();
    for (i, &price) in prices.iter().enumerate() {
        market_data.push(MarketData {
            timestamp: i as f64,
            mid_price: price,
            bid_price: price * 0.9999,
            ask_price: price * 1.0001,
            bid_size: 1000.0,
            ask_size: 1000.0,
            spread: price * 0.0002,
            depth: 2000.0,
            volatility: 0.15,
        });
    }
    
    // Generate benchmark returns (market index)
    let benchmark_returns: Vec<f64> = returns.iter()
        .map(|&r| r * 0.8 + 0.00005) // Slightly different from strategy returns
        .collect();
    
    let timestamps: Vec<f64> = (0..prices.len()).map(|i| i as f64).collect();
    
    PerformanceData {
        returns,
        prices,
        positions,
        trades,
        market_data,
        timestamps,
        benchmark_returns: Some(benchmark_returns),
        risk_free_rate: 0.02, // 2% annual risk-free rate
    }
}

fn display_performance_metrics(metrics: &hf_quoting_liquidity::models::performance_metrics::ComprehensivePerformanceMetrics) {
    println!("📊 COMPREHENSIVE PERFORMANCE METRICS");
    println!("=====================================\n");
    
    // Basic Performance Metrics
    println!("🎯 BASIC PERFORMANCE");
    println!("-------------------");
    println!("Total Return:           {:.4}%", metrics.total_return * 100.0);
    println!("Annualized Return:      {:.4}%", metrics.annualized_return * 100.0);
    println!("Volatility (Ann.):      {:.4}%", metrics.volatility * 100.0);
    println!("Sharpe Ratio:           {:.4}", metrics.sharpe_ratio);
    println!("Information Ratio:      {:.4}", metrics.information_ratio);
    println!("Sortino Ratio:          {:.4}", metrics.sortino_ratio);
    println!();
    
    // Risk Metrics
    println!("⚠️  RISK METRICS");
    println!("---------------");
    println!("Maximum Drawdown:       {:.4}%", metrics.max_drawdown * 100.0);
    println!("Max DD Duration:        {} periods", metrics.max_drawdown_duration);
    println!("VaR (95%):             {:.4}%", metrics.var_95 * 100.0);
    println!("VaR (99%):             {:.4}%", metrics.var_99 * 100.0);
    println!("Expected Shortfall 95%: {:.4}%", metrics.expected_shortfall_95 * 100.0);
    println!("Expected Shortfall 99%: {:.4}%", metrics.expected_shortfall_99 * 100.0);
    println!("Skewness:              {:.4}", metrics.skewness);
    println!("Kurtosis:              {:.4}", metrics.kurtosis);
    println!();
    
    // Liquidity-Adjusted Metrics
    println!("💧 LIQUIDITY-ADJUSTED METRICS");
    println!("-----------------------------");
    println!("Liquidity-Adj. Return:  {:.4}%", metrics.liquidity_adjusted_return * 100.0);
    println!("Total Liquidity Cost:   {:.6}%", metrics.liquidity_cost * 100.0);
    println!("Market Impact Cost:     {:.6}%", metrics.market_impact_cost * 100.0);
    println!("Bid-Ask Spread Cost:    {:.6}%", metrics.bid_ask_spread_cost * 100.0);
    println!();
    
    // Transaction Cost Analysis
    println!("💰 TRANSACTION COST ANALYSIS");
    println!("----------------------------");
    println!("Total Cost:             ${:.4}", metrics.transaction_costs.total_cost);
    println!("Cost per Share:         ${:.6}", metrics.transaction_costs.cost_per_share);
    println!("Cost % of Volume:       {:.4}%", metrics.transaction_costs.cost_as_percentage_of_volume * 100.0);
    println!("Implementation Shortfall: {:.4}%", metrics.transaction_costs.implementation_shortfall * 100.0);
    println!("Timing Cost:            ${:.4}", metrics.transaction_costs.timing_cost);
    println!("Market Impact Cost:     ${:.4}", metrics.transaction_costs.market_impact_cost);
    println!("Spread Cost:            ${:.4}", metrics.transaction_costs.spread_cost);
    println!("Commission Cost:        ${:.4}", metrics.transaction_costs.commission_cost);
    
    println!("\nCost Breakdown:");
    for (category, cost) in &metrics.transaction_costs.cost_breakdown {
        println!("  {}: ${:.4}", category, cost);
    }
    println!();
    
    // Trading Activity Metrics
    println!("📈 TRADING ACTIVITY");
    println!("------------------");
    println!("Total Trades:           {}", metrics.total_trades);
    println!("Win Rate:               {:.2}%", metrics.win_rate * 100.0);
    println!("Profit Factor:          {:.4}", metrics.profit_factor);
    println!("Average Trade P&L:      ${:.4}", metrics.average_trade_pnl);
    println!("Largest Win:            ${:.4}", metrics.largest_win);
    println!("Largest Loss:           ${:.4}", metrics.largest_loss);
    println!();
    
    // Position and Inventory Metrics
    println!("📊 POSITION & INVENTORY");
    println!("----------------------");
    println!("Average Position:       {:.2} shares", metrics.average_position);
    println!("Position Volatility:    {:.2} shares", metrics.position_volatility);
    println!("Inventory Turnover:     {:.4}x", metrics.inventory_turnover);
    println!("Time in Market:         {:.2}%", metrics.time_in_market * 100.0);
    println!();
    
    // Market Making Metrics
    println!("🏪 MARKET MAKING METRICS");
    println!("-----------------------");
    println!("Fill Ratio:             {:.2}%", metrics.fill_ratio * 100.0);
    println!("Adverse Selection Cost: ${:.6}", metrics.adverse_selection_cost);
    println!("Inventory Risk Premium: {:.6}", metrics.inventory_risk_premium);
    println!("Quote Competitiveness:  {:.2}%", metrics.quote_competitiveness * 100.0);
    println!();
    
    // Performance Summary
    println!("📋 PERFORMANCE SUMMARY");
    println!("=====================");
    
    let risk_adjusted_return = if metrics.volatility > 0.0 {
        metrics.annualized_return / metrics.volatility
    } else {
        0.0
    };
    
    println!("Risk-Adjusted Return:   {:.4}", risk_adjusted_return);
    println!("Liquidity Efficiency:   {:.4}", 
        if metrics.liquidity_cost > 0.0 { 
            metrics.liquidity_adjusted_return / metrics.liquidity_cost 
        } else { 0.0 });
    
    let overall_score = (metrics.sharpe_ratio * 0.3) + 
                       (metrics.information_ratio * 0.2) + 
                       ((1.0 - metrics.max_drawdown) * 0.2) + 
                       (metrics.win_rate * 0.15) + 
                       (metrics.fill_ratio * 0.15);
    
    println!("Overall Performance Score: {:.4}/1.0", overall_score.max(0.0).min(1.0));
    
    // Performance Rating
    let rating = match overall_score {
        x if x >= 0.8 => "🌟 Excellent",
        x if x >= 0.6 => "✅ Good", 
        x if x >= 0.4 => "⚠️  Fair",
        x if x >= 0.2 => "❌ Poor",
        _ => "💀 Very Poor",
    };
    
    println!("Performance Rating:     {}", rating);
    println!();
    
    // Recommendations
    println!("💡 RECOMMENDATIONS");
    println!("==================");
    
    if metrics.sharpe_ratio < 1.0 {
        println!("• Consider improving risk-adjusted returns (Sharpe ratio < 1.0)");
    }
    
    if metrics.max_drawdown > 0.1 {
        println!("• Implement stronger risk controls (Max drawdown > 10%)");
    }
    
    if metrics.transaction_costs.cost_as_percentage_of_volume > 0.01 {
        println!("• Optimize execution to reduce transaction costs (> 1% of volume)");
    }
    
    if metrics.fill_ratio < 0.5 {
        println!("• Improve quote competitiveness to increase fill ratio");
    }
    
    if metrics.inventory_turnover > 10.0 {
        println!("• Consider reducing position churn to minimize costs");
    }
    
    if metrics.adverse_selection_cost > 0.001 {
        println!("• Enhance adverse selection protection mechanisms");
    }
    
    println!("\n=== Demo Complete ===");
}