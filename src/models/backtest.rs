use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, LOBSimulation},
    hft_strategy::{HFTStrategy, HFTParams, PerformanceMetrics},
    performance_metrics::{
        PerformanceCalculator, ComprehensivePerformanceMetrics, PerformanceData,
        TradeData, MarketData, TradeSide, TransactionCostMetrics,
    },
};

#[derive(Debug, Error)]
pub enum BacktestError {
    #[error("Backtest error: {0}")]
    BacktestError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

/// Parameters for backtesting
pub struct BacktestParams {
    pub initial_capital: f64,
    pub transaction_cost: f64,
    pub risk_free_rate: f64,
    pub margin_requirement: f64,
    pub max_leverage: f64,
}

/// Results of a backtest simulation
pub struct BacktestResults {
    pub performance_metrics: PerformanceMetrics,
    pub daily_pnl: Vec<f64>,
    pub position_history: Vec<f64>,
    pub market_impact: MarketImpactAnalysis,
    pub risk_metrics: RiskMetrics,
    pub transaction_costs: TransactionCostAnalysis,
}

/// Market impact analysis
pub struct MarketImpactAnalysis {
    pub permanent_impact: f64,
    pub temporary_impact: f64,
    pub spread_widening: f64,
    pub depth_reduction: f64,
    pub resilience: f64,
}

/// Risk metrics
pub struct RiskMetrics {
    pub var_95: f64,
    pub es_95: f64,
    pub beta: f64,
    pub volatility: f64,
    pub correlation_with_market: f64,
}

/// Transaction cost analysis
pub struct TransactionCostAnalysis {
    pub total_cost: f64,
    pub average_cost_per_trade: f64,
    pub cost_by_size: HashMap<String, f64>,
    pub implementation_shortfall: f64,
}

/// Backtester for HFT strategies
pub struct Backtester {
    params: BacktestParams,
    book: LimitOrderBook,
    strategy: HFTStrategy,
    volatility_model: RoughVolatilityModel,
}

impl Backtester {
    pub fn new(
        backtest_params: BacktestParams,
        lob_params: LOBParams,
        hft_params: HFTParams,
        volatility_params: RoughVolatilityParams,
        grid_points: usize,
        time_horizon: f64,
    ) -> Result<Self, BacktestError> {
        let book = LimitOrderBook::new(
            lob_params,
            volatility_params.clone(),
            grid_points,
            time_horizon,
        ).map_err(|e| BacktestError::BacktestError(e.to_string()))?;

        let strategy = HFTStrategy::new(hft_params)
            .map_err(|e| BacktestError::BacktestError(e.to_string()))?;

        let volatility_model = RoughVolatilityModel::new(
            volatility_params,
            grid_points,
            time_horizon,
        ).map_err(|e| BacktestError::BacktestError(e.to_string()))?;

        Ok(Self {
            params: backtest_params,
            book,
            strategy,
            volatility_model,
        })
    }

    /// Runs the backtest simulation
    pub fn run(
        &mut self,
        initial_price: f64,
        n_steps: usize,
        dt: f64,
    ) -> Result<BacktestResults, BacktestError> {
        // Simulate market conditions
        let (prices, volatility) = self.volatility_model.simulate_price_process(initial_price)
            .map_err(|e| BacktestError::BacktestError(e.to_string()))?;

        let mut daily_pnl = Vec::with_capacity(n_steps);
        let mut position_history = Vec::with_capacity(n_steps);
        let mut market_states = Vec::with_capacity(n_steps);

        // Run simulation
        for step in 0..n_steps {
            // Execute strategy
            let action = self.strategy.execute(
                &mut self.book,
                volatility[step],
                dt,
            ).map_err(|e| BacktestError::BacktestError(e.to_string()))?;

            // Record state
            daily_pnl.push(action.cash);
            position_history.push(action.position);
            
            // Record market state
            let state = self.book.compute_market_state()
                .map_err(|e| BacktestError::BacktestError(e.to_string()))?;
            market_states.push(state);
        }

        // Compute performance metrics
        let performance_metrics = self.strategy.compute_performance_metrics();

        // Analyze market impact
        let market_impact = self.analyze_market_impact(&market_states, &position_history)?;

        // Compute risk metrics
        let risk_metrics = self.compute_risk_metrics(&daily_pnl, &prices)?;

        // Analyze transaction costs
        let transaction_costs = self.analyze_transaction_costs(&daily_pnl, &position_history)?;

        Ok(BacktestResults {
            performance_metrics,
            daily_pnl,
            position_history,
            market_impact,
            risk_metrics,
            transaction_costs,
        })
    }

    /// Analyzes market impact of the strategy
    fn analyze_market_impact(
        &self,
        market_states: &[(f64, f64, f64)],
        position_history: &[f64],
    ) -> Result<MarketImpactAnalysis, BacktestError> {
        let n = market_states.len();
        if n < 2 {
            return Err(BacktestError::BacktestError(
                "Insufficient data for market impact analysis".to_string(),
            ));
        }

        // Compute average spread and depth before and after large trades
        let mut pre_trade_spread = 0.0;
        let mut post_trade_spread = 0.0;
        let mut pre_trade_depth = 0.0;
        let mut post_trade_depth = 0.0;
        let mut trade_count = 0;

        for i in 1..n {
            let position_change = (position_history[i] - position_history[i-1]).abs();
            if position_change > 0.0 {
                pre_trade_spread += market_states[i-1].1;
                post_trade_spread += market_states[i].1;
                pre_trade_depth += market_states[i-1].2;
                post_trade_depth += market_states[i].2;
                trade_count += 1;
            }
        }

        if trade_count == 0 {
            return Ok(MarketImpactAnalysis {
                permanent_impact: 0.0,
                temporary_impact: 0.0,
                spread_widening: 0.0,
                depth_reduction: 0.0,
                resilience: 1.0,
            });
        }

        let spread_widening = (post_trade_spread / trade_count as f64) /
            (pre_trade_spread / trade_count as f64) - 1.0;
        
        let depth_reduction = 1.0 - (post_trade_depth / trade_count as f64) /
            (pre_trade_depth / trade_count as f64);

        // Estimate permanent and temporary impact
        let permanent_impact = self.estimate_permanent_impact(market_states, position_history)?;
        let temporary_impact = self.estimate_temporary_impact(market_states, position_history)?;
        let resilience = self.estimate_market_resilience(market_states, position_history)?;

        Ok(MarketImpactAnalysis {
            permanent_impact,
            temporary_impact,
            spread_widening,
            depth_reduction,
            resilience,
        })
    }

    /// Estimates permanent price impact
    fn estimate_permanent_impact(
        &self,
        market_states: &[(f64, f64, f64)],
        position_history: &[f64],
    ) -> Result<f64, BacktestError> {
        let n = market_states.len();
        let window_size = 10; // Look at impact over 10 periods

        let mut total_impact = 0.0;
        let mut count = 0;

        for i in window_size..n {
            let position_change = (position_history[i] - position_history[i-1]).abs();
            if position_change > 0.0 {
                let pre_price = market_states[i-window_size].0;
                let post_price = market_states[i].0;
                total_impact += (post_price - pre_price).abs() / pre_price;
                count += 1;
            }
        }

        Ok(if count > 0 { total_impact / count as f64 } else { 0.0 })
    }

    /// Estimates temporary price impact
    fn estimate_temporary_impact(
        &self,
        market_states: &[(f64, f64, f64)],
        position_history: &[f64],
    ) -> Result<f64, BacktestError> {
        let n = market_states.len();
        let mut total_impact = 0.0;
        let mut count = 0;

        for i in 1..n {
            let position_change = (position_history[i] - position_history[i-1]).abs();
            if position_change > 0.0 {
                let mid_price_change = (market_states[i].0 - market_states[i-1].0).abs() /
                    market_states[i-1].0;
                total_impact += mid_price_change;
                count += 1;
            }
        }

        Ok(if count > 0 { total_impact / count as f64 } else { 0.0 })
    }

    /// Estimates market resilience
    fn estimate_market_resilience(
        &self,
        market_states: &[(f64, f64, f64)],
        position_history: &[f64],
    ) -> Result<f64, BacktestError> {
        let n = market_states.len();
        let window_size = 5; // Look at recovery over 5 periods
        
        let mut total_resilience = 0.0;
        let mut count = 0;

        for i in window_size..n {
            let position_change = (position_history[i-window_size] - position_history[i-window_size-1]).abs();
            if position_change > 0.0 {
                let initial_spread = market_states[i-window_size].1;
                let final_spread = market_states[i].1;
                total_resilience += 1.0 - (final_spread - initial_spread).abs() / initial_spread;
                count += 1;
            }
        }

        Ok(if count > 0 { total_resilience / count as f64 } else { 1.0 })
    }

    /// Computes risk metrics
    fn compute_risk_metrics(
        &self,
        daily_pnl: &[f64],
        market_prices: &[f64],
    ) -> Result<RiskMetrics, BacktestError> {
        let n = daily_pnl.len();
        if n < 2 {
            return Err(BacktestError::BacktestError(
                "Insufficient data for risk computation".to_string(),
            ));
        }

        // Compute returns
        let returns: Vec<f64> = daily_pnl.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let market_returns: Vec<f64> = market_prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Compute VaR and ES
        let mut sorted_returns = returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = ((1.0 - 0.95) * returns.len() as f64) as usize;
        let var_95 = sorted_returns[var_index];
        
        let es_95 = sorted_returns[..=var_index].iter().sum::<f64>() / (var_index + 1) as f64;

        // Compute volatility
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let volatility = (returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64)
            .sqrt();

        // Compute beta and correlation
        let (beta, correlation) = self.compute_market_relationship(&returns, &market_returns)?;

        Ok(RiskMetrics {
            var_95,
            es_95,
            beta,
            volatility,
            correlation_with_market: correlation,
        })
    }

    /// Computes beta and correlation with market
    fn compute_market_relationship(
        &self,
        returns: &[f64],
        market_returns: &[f64],
    ) -> Result<(f64, f64), BacktestError> {
        let n = returns.len();
        if n != market_returns.len() {
            return Err(BacktestError::BacktestError(
                "Return series lengths do not match".to_string(),
            ));
        }

        let mean_return = returns.iter().sum::<f64>() / n as f64;
        let mean_market = market_returns.iter().sum::<f64>() / n as f64;

        let mut covariance = 0.0;
        let mut variance_market = 0.0;
        let mut variance_return = 0.0;

        for i in 0..n {
            let dr = returns[i] - mean_return;
            let dm = market_returns[i] - mean_market;
            covariance += dr * dm;
            variance_market += dm * dm;
            variance_return += dr * dr;
        }

        covariance /= (n - 1) as f64;
        variance_market /= (n - 1) as f64;
        variance_return /= (n - 1) as f64;

        let beta = covariance / variance_market;
        let correlation = covariance / (variance_market * variance_return).sqrt();

        Ok((beta, correlation))
    }

    /// Analyzes transaction costs
    fn analyze_transaction_costs(
        &self,
        daily_pnl: &[f64],
        position_history: &[f64],
    ) -> Result<TransactionCostAnalysis, BacktestError> {
        let n = position_history.len();
        if n < 2 {
            return Err(BacktestError::BacktestError(
                "Insufficient data for transaction cost analysis".to_string(),
            ));
        }

        let mut total_cost = 0.0;
        let mut trade_count = 0;
        let mut cost_by_size = HashMap::new();

        for i in 1..n {
            let position_change = (position_history[i] - position_history[i-1]).abs();
            if position_change > 0.0 {
                let cost = position_change * self.params.transaction_cost;
                total_cost += cost;
                trade_count += 1;

                // Categorize costs by trade size
                let size_category = match position_change {
                    x if x < 100.0 => "Small",
                    x if x < 1000.0 => "Medium",
                    _ => "Large",
                };
                *cost_by_size.entry(size_category.to_string()).or_insert(0.0) += cost;
            }
        }

        // Compute implementation shortfall
        let theoretical_pnl = daily_pnl.last().unwrap() - daily_pnl[0];
        let actual_pnl = theoretical_pnl - total_cost;
        let implementation_shortfall = (theoretical_pnl - actual_pnl) / theoretical_pnl;

        Ok(TransactionCostAnalysis {
            total_cost,
            average_cost_per_trade: if trade_count > 0 { total_cost / trade_count as f64 } else { 0.0 },
            cost_by_size,
            implementation_shortfall,
        })
    }

    /// Calculate comprehensive performance metrics using the new performance calculator
    pub fn calculate_comprehensive_metrics(
        &self,
        daily_pnl: &[f64],
        position_history: &[f64],
        prices: &[f64],
        trades: Vec<TradeData>,
        market_data: Vec<MarketData>,
        timestamps: Vec<f64>,
        benchmark_returns: Option<Vec<f64>>,
    ) -> Result<ComprehensivePerformanceMetrics, BacktestError> {
        // Calculate returns from P&L
        let returns: Vec<f64> = daily_pnl.windows(2)
            .map(|w| if w[0] != 0.0 { (w[1] - w[0]) / w[0].abs() } else { 0.0 })
            .collect();

        let performance_data = PerformanceData {
            returns,
            prices: prices.to_vec(),
            positions: position_history.to_vec(),
            trades,
            market_data,
            timestamps,
            benchmark_returns,
            risk_free_rate: self.params.risk_free_rate,
        };

        let calculator = PerformanceCalculator::default();
        calculator.calculate_metrics(&performance_data)
            .map_err(|e| BacktestError::BacktestError(e.to_string()))
    }

    /// Enhanced backtest run with comprehensive metrics
    pub fn run_comprehensive(
        &mut self,
        initial_price: f64,
        n_steps: usize,
        dt: f64,
        benchmark_returns: Option<Vec<f64>>,
    ) -> Result<(BacktestResults, ComprehensivePerformanceMetrics), BacktestError> {
        // Run standard backtest
        let backtest_results = self.run(initial_price, n_steps, dt)?;

        // Collect trade data for comprehensive analysis
        let mut trades = Vec::new();
        let mut market_data = Vec::new();
        let mut timestamps = Vec::new();

        // Simulate trade data from position changes
        for (i, &pos) in backtest_results.position_history.iter().enumerate() {
            timestamps.push(i as f64 * dt);
            
            if i > 0 {
                let prev_pos = backtest_results.position_history[i-1];
                let position_change = pos - prev_pos;
                
                if position_change.abs() > 1e-6 {
                    let price = if i < backtest_results.daily_pnl.len() {
                        initial_price * (1.0 + backtest_results.daily_pnl[i] / 100.0)
                    } else {
                        initial_price
                    };
                    
                    trades.push(TradeData {
                        timestamp: i as f64 * dt,
                        price,
                        quantity: position_change,
                        side: if position_change > 0.0 { TradeSide::Buy } else { TradeSide::Sell },
                        transaction_cost: position_change.abs() * self.params.transaction_cost,
                        market_impact: position_change.abs() * 0.001, // Estimated 0.1% market impact
                        spread_at_trade: price * 0.0002, // Estimated 2 bps spread
                    });
                }
            }

            // Create market data snapshot
            let price = if i < backtest_results.daily_pnl.len() {
                initial_price * (1.0 + backtest_results.daily_pnl[i] / 100.0)
            } else {
                initial_price
            };
            
            market_data.push(MarketData {
                timestamp: i as f64 * dt,
                mid_price: price,
                bid_price: price * 0.9999, // Tight spread
                ask_price: price * 1.0001,
                bid_size: 1000.0,
                ask_size: 1000.0,
                spread: price * 0.0002,
                depth: 2000.0,
                volatility: 0.15, // Assumed volatility
            });
        }

        // Calculate comprehensive metrics
        let comprehensive_metrics = self.calculate_comprehensive_metrics(
            &backtest_results.daily_pnl,
            &backtest_results.position_history,
            &vec![initial_price; n_steps], // Simplified price series
            trades,
            market_data,
            timestamps,
            benchmark_returns,
        )?;

        Ok((backtest_results, comprehensive_metrics))
    }
}
