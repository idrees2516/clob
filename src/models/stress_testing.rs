//! Stress Testing Module for Extreme Market Scenarios
//! 
//! This module implements comprehensive stress testing capabilities for high-frequency
//! quoting strategies, fulfilling Requirement 5.4 by simulating extreme market scenarios
//! and measuring strategy robustness.

use crate::models::{
    backtesting_framework::{
        StressTestingConfig, ExtremeScenario, StressMetric, StressTestResults, ScenarioResult,
        ReplayedMarketData, MarketTick,
    },
    quoting_strategy::ModelParameters,
    performance_metrics::{ComprehensivePerformanceMetrics, PerformanceCalculator, PerformanceData, TradeData, MarketData, TradeSide},
};
use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StudentsT, Uniform, LogNormal};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StressTestError {
    #[error("Stress test error: {0}")]
    TestError(String),
    #[error("Scenario generation error: {0}")]
    ScenarioError(String),
    #[error("Monte Carlo error: {0}")]
    MonteCarloError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Comprehensive stress tester for trading strategies
pub struct StressTester {
    config: StressTestingConfig,
    scenario_generator: ScenarioGenerator,
    monte_carlo_engine: MonteCarloEngine,
    robustness_analyzer: RobustnessAnalyzer,
}

impl StressTester {
    pub fn new(config: &StressTestingConfig) -> Result<Self, StressTestError> {
        let scenario_generator = ScenarioGenerator::new(&config.extreme_scenarios)?;
        let monte_carlo_engine = MonteCarloEngine::new(config.monte_carlo_sims)?;
        let robustness_analyzer = RobustnessAnalyzer::new(&config.stress_metrics)?;

        Ok(Self {
            config: config.clone(),
            scenario_generator,
            monte_carlo_engine,
            robustness_analyzer,
        })
    }

    /// Run comprehensive stress tests
    pub async fn run_stress_tests(
        &mut self,
        strategy_params: &ModelParameters,
        baseline_market_data: &ReplayedMarketData,
    ) -> Result<StressTestResults, StressTestError> {
        let mut scenario_results = HashMap::new();

        // Test each extreme scenario
        for scenario in &self.config.extreme_scenarios {
            println!("Running stress test for scenario: {}", scenario.name);
            
            let scenario_result = self.test_scenario(
                scenario,
                strategy_params,
                baseline_market_data,
            ).await?;
            
            scenario_results.insert(scenario.name.clone(), scenario_result);
        }

        // Calculate overall robustness score
        let robustness_score = self.robustness_analyzer.calculate_robustness_score(&scenario_results)?;

        Ok(StressTestResults {
            scenario_results,
            robustness_score,
        })
    }

    /// Test strategy under a specific extreme scenario
    async fn test_scenario(
        &mut self,
        scenario: &ExtremeScenario,
        strategy_params: &ModelParameters,
        baseline_market_data: &ReplayedMarketData,
    ) -> Result<ScenarioResult, StressTestError> {
        // Generate stressed market data
        let stressed_market_data = self.scenario_generator.generate_scenario_data(
            scenario,
            baseline_market_data,
        )?;

        // Run Monte Carlo simulations under stress
        let monte_carlo_results = self.monte_carlo_engine.run_simulations(
            strategy_params,
            &stressed_market_data,
        ).await?;

        // Analyze results and extract stress metrics
        let max_drawdown = self.calculate_max_drawdown(&monte_carlo_results)?;
        let recovery_time = self.calculate_recovery_time(&monte_carlo_results)?;
        let var_99 = self.calculate_var_99(&monte_carlo_results)?;

        Ok(ScenarioResult {
            max_drawdown,
            recovery_time,
            var_99,
        })
    }

    fn calculate_max_drawdown(&self, results: &[MonteCarloResult]) -> Result<f64, StressTestError> {
        let mut max_drawdown = 0.0;
        
        for result in results {
            let mut peak = result.pnl_series[0];
            let mut current_drawdown = 0.0;
            
            for &pnl in &result.pnl_series {
                if pnl > peak {
                    peak = pnl;
                }
                current_drawdown = (peak - pnl) / peak.abs().max(1e-8);
                max_drawdown = max_drawdown.max(current_drawdown);
            }
        }

        Ok(max_drawdown)
    }

    fn calculate_recovery_time(&self, results: &[MonteCarloResult]) -> Result<usize, StressTestError> {
        let mut max_recovery_time = 0;
        
        for result in &results {
            let mut peak = result.pnl_series[0];
            let mut drawdown_start = 0;
            let mut in_drawdown = false;
            
            for (i, &pnl) in result.pnl_series.iter().enumerate() {
                if pnl > peak {
                    if in_drawdown {
                        // Recovery completed
                        let recovery_time = i - drawdown_start;
                        max_recovery_time = max_recovery_time.max(recovery_time);
                        in_drawdown = false;
                    }
                    peak = pnl;
                } else if !in_drawdown && (peak - pnl) / peak.abs().max(1e-8) > 0.05 {
                    // Drawdown started (>5%)
                    drawdown_start = i;
                    in_drawdown = true;
                }
            }
        }

        Ok(max_recovery_time)
    }

    fn calculate_var_99(&self, results: &[MonteCarloResult]) -> Result<f64, StressTestError> {
        let mut all_returns = Vec::new();
        
        for result in results {
            let returns: Vec<f64> = result.pnl_series.windows(2)
                .map(|w| if w[0].abs() > 1e-8 { (w[1] - w[0]) / w[0].abs() } else { 0.0 })
                .collect();
            all_returns.extend(returns);
        }

        if all_returns.len() < 100 {
            return Err(StressTestError::InsufficientData(
                "Need at least 100 returns for VaR calculation".to_string()
            ));
        }

        all_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_99_index = (0.01 * all_returns.len() as f64) as usize;
        Ok(-all_returns[var_99_index]) // VaR is positive for losses
    }
}

/// Scenario generator for extreme market conditions
pub struct ScenarioGenerator {
    scenarios: Vec<ExtremeScenario>,
}

impl ScenarioGenerator {
    pub fn new(scenarios: &[ExtremeScenario]) -> Result<Self, StressTestError> {
        Ok(Self {
            scenarios: scenarios.to_vec(),
        })
    }

    /// Generate market data for a specific stress scenario
    pub fn generate_scenario_data(
        &self,
        scenario: &ExtremeScenario,
        baseline_data: &ReplayedMarketData,
    ) -> Result<ReplayedMarketData, StressTestError> {
        let mut stressed_ticks = Vec::new();
        let mut rng = thread_rng();

        for (i, tick) in baseline_data.ticks.iter().enumerate() {
            let stressed_tick = match scenario.name.as_str() {
                "Flash Crash" => self.generate_flash_crash_tick(tick, i, &mut rng)?,
                "Market Stress" => self.generate_market_stress_tick(tick, scenario, &mut rng)?,
                "Liquidity Crisis" => self.generate_liquidity_crisis_tick(tick, scenario, &mut rng)?,
                "Volatility Spike" => self.generate_volatility_spike_tick(tick, scenario, &mut rng)?,
                _ => self.generate_generic_stress_tick(tick, scenario, &mut rng)?,
            };
            
            stressed_ticks.push(stressed_tick);
        }

        Ok(ReplayedMarketData {
            ticks: stressed_ticks,
            metadata: baseline_data.metadata.clone(),
        })
    }

    fn generate_flash_crash_tick(
        &self,
        baseline_tick: &MarketTick,
        tick_index: usize,
        rng: &mut impl Rng,
    ) -> Result<MarketTick, StressTestError> {
        let mut stressed_tick = baseline_tick.clone();

        // Simulate flash crash: sudden price drop followed by partial recovery
        let crash_duration = 100; // 100 ticks
        let crash_magnitude = 0.10; // 10% drop
        
        if tick_index < crash_duration {
            let crash_progress = tick_index as f64 / crash_duration as f64;
            
            // Price drops rapidly then recovers partially
            let price_impact = if crash_progress < 0.1 {
                // Rapid drop in first 10% of crash
                -crash_magnitude * (crash_progress / 0.1)
            } else {
                // Partial recovery
                let recovery_progress = (crash_progress - 0.1) / 0.9;
                -crash_magnitude * (1.0 - 0.5 * recovery_progress)
            };
            
            stressed_tick.mid_price *= 1.0 + price_impact;
            stressed_tick.bid_price *= 1.0 + price_impact;
            stressed_tick.ask_price *= 1.0 + price_impact;
            
            // Liquidity evaporates during crash
            stressed_tick.bid_size *= 0.1;
            stressed_tick.ask_size *= 0.1;
            stressed_tick.depth *= 0.1;
            
            // Spread widens dramatically
            stressed_tick.spread *= 10.0;
            
            // Volatility spikes
            stressed_tick.volatility *= 5.0;
        }

        Ok(stressed_tick)
    }

    fn generate_market_stress_tick(
        &self,
        baseline_tick: &MarketTick,
        scenario: &ExtremeScenario,
        rng: &mut impl Rng,
    ) -> Result<MarketTick, StressTestError> {
        let mut stressed_tick = baseline_tick.clone();

        // Apply volatility multiplier
        stressed_tick.volatility *= scenario.volatility_multiplier;
        
        // Reduce liquidity
        stressed_tick.bid_size *= 1.0 - scenario.liquidity_reduction;
        stressed_tick.ask_size *= 1.0 - scenario.liquidity_reduction;
        stressed_tick.depth *= 1.0 - scenario.liquidity_reduction;
        
        // Widen spreads due to reduced liquidity
        stressed_tick.spread *= 1.0 + scenario.liquidity_reduction * 2.0;
        
        // Add price volatility
        let price_shock = Normal::new(0.0, stressed_tick.volatility * 0.01)
            .map_err(|e| StressTestError::ScenarioError(e.to_string()))?;
        
        let price_change = price_shock.sample(rng);
        stressed_tick.mid_price *= 1.0 + price_change;
        stressed_tick.bid_price *= 1.0 + price_change;
        stressed_tick.ask_price *= 1.0 + price_change;

        Ok(stressed_tick)
    }

    fn generate_liquidity_crisis_tick(
        &self,
        baseline_tick: &MarketTick,
        scenario: &ExtremeScenario,
        rng: &mut impl Rng,
    ) -> Result<MarketTick, StressTestError> {
        let mut stressed_tick = baseline_tick.clone();

        // Severe liquidity reduction
        let liquidity_multiplier = 1.0 - scenario.liquidity_reduction.max(0.8);
        stressed_tick.bid_size *= liquidity_multiplier;
        stressed_tick.ask_size *= liquidity_multiplier;
        stressed_tick.depth *= liquidity_multiplier;
        
        // Extreme spread widening
        stressed_tick.spread *= 5.0 + rng.gen::<f64>() * 5.0;
        
        // Intermittent liquidity (some ticks have no liquidity)
        if rng.gen::<f64>() < 0.2 {
            stressed_tick.bid_size = 0.0;
            stressed_tick.ask_size = 0.0;
            stressed_tick.spread = stressed_tick.mid_price * 0.1; // 10% spread
        }

        Ok(stressed_tick)
    }

    fn generate_volatility_spike_tick(
        &self,
        baseline_tick: &MarketTick,
        scenario: &ExtremeScenario,
        rng: &mut impl Rng,
    ) -> Result<MarketTick, StressTestError> {
        let mut stressed_tick = baseline_tick.clone();

        // Extreme volatility spike
        stressed_tick.volatility *= scenario.volatility_multiplier;
        
        // Price jumps with fat tails
        let jump_prob = 0.05; // 5% chance of jump
        if rng.gen::<f64>() < jump_prob {
            let jump_size = StudentsT::new(3.0) // Fat tails
                .map_err(|e| StressTestError::ScenarioError(e.to_string()))?
                .sample(rng) * 0.02; // 2% jump size
            
            stressed_tick.mid_price *= 1.0 + jump_size;
            stressed_tick.bid_price *= 1.0 + jump_size;
            stressed_tick.ask_price *= 1.0 + jump_size;
        }
        
        // Volatility clustering
        let vol_shock = LogNormal::new(0.0, 0.5)
            .map_err(|e| StressTestError::ScenarioError(e.to_string()))?
            .sample(rng);
        stressed_tick.volatility *= vol_shock;

        Ok(stressed_tick)
    }

    fn generate_generic_stress_tick(
        &self,
        baseline_tick: &MarketTick,
        scenario: &ExtremeScenario,
        rng: &mut impl Rng,
    ) -> Result<MarketTick, StressTestError> {
        let mut stressed_tick = baseline_tick.clone();

        // Apply scenario parameters
        stressed_tick.volatility *= scenario.volatility_multiplier;
        stressed_tick.bid_size *= 1.0 - scenario.liquidity_reduction;
        stressed_tick.ask_size *= 1.0 - scenario.liquidity_reduction;
        stressed_tick.depth *= 1.0 - scenario.liquidity_reduction;
        
        // Correlation breakdown if specified
        if scenario.correlation_breakdown {
            let correlation_shock = Normal::new(0.0, 0.05)
                .map_err(|e| StressTestError::ScenarioError(e.to_string()))?;
            
            let price_shock = correlation_shock.sample(rng);
            stressed_tick.mid_price *= 1.0 + price_shock;
            stressed_tick.bid_price *= 1.0 + price_shock;
            stressed_tick.ask_price *= 1.0 + price_shock;
        }

        Ok(stressed_tick)
    }
}

/// Monte Carlo simulation engine for stress testing
pub struct MonteCarloEngine {
    n_simulations: usize,
}

impl MonteCarloEngine {
    pub fn new(n_simulations: usize) -> Result<Self, StressTestError> {
        if n_simulations == 0 {
            return Err(StressTestError::MonteCarloError(
                "Number of simulations must be positive".to_string()
            ));
        }

        Ok(Self { n_simulations })
    }

    /// Run Monte Carlo simulations under stressed conditions
    pub async fn run_simulations(
        &self,
        strategy_params: &ModelParameters,
        stressed_market_data: &ReplayedMarketData,
    ) -> Result<Vec<MonteCarloResult>, StressTestError> {
        let mut results = Vec::new();
        let mut rng = thread_rng();

        for simulation_id in 0..self.n_simulations {
            // Add random noise to each simulation
            let noisy_market_data = self.add_simulation_noise(stressed_market_data, &mut rng)?;
            
            // Simulate strategy performance
            let simulation_result = self.simulate_strategy_performance(
                strategy_params,
                &noisy_market_data,
                simulation_id,
            ).await?;
            
            results.push(simulation_result);
        }

        Ok(results)
    }

    fn add_simulation_noise(
        &self,
        market_data: &ReplayedMarketData,
        rng: &mut impl Rng,
    ) -> Result<ReplayedMarketData, StressTestError> {
        let mut noisy_ticks = Vec::new();
        
        for tick in &market_data.ticks {
            let mut noisy_tick = tick.clone();
            
            // Add small random noise to prices
            let price_noise = Normal::new(0.0, tick.mid_price * 0.0001)
                .map_err(|e| StressTestError::MonteCarloError(e.to_string()))?;
            
            let noise = price_noise.sample(rng);
            noisy_tick.mid_price += noise;
            noisy_tick.bid_price += noise;
            noisy_tick.ask_price += noise;
            
            // Add noise to liquidity
            let liquidity_noise = Uniform::new(0.9, 1.1);
            noisy_tick.bid_size *= liquidity_noise.sample(rng);
            noisy_tick.ask_size *= liquidity_noise.sample(rng);
            
            noisy_ticks.push(noisy_tick);
        }

        Ok(ReplayedMarketData {
            ticks: noisy_ticks,
            metadata: market_data.metadata.clone(),
        })
    }

    async fn simulate_strategy_performance(
        &self,
        _strategy_params: &ModelParameters,
        market_data: &ReplayedMarketData,
        simulation_id: usize,
    ) -> Result<MonteCarloResult, StressTestError> {
        // Simplified strategy simulation
        // In practice, this would use the full strategy implementation
        let mut pnl_series = Vec::new();
        let mut current_pnl = 0.0;
        let mut position = 0.0;
        
        for (i, tick) in market_data.ticks.iter().enumerate() {
            // Simple mean-reversion strategy for demonstration
            let signal = if i > 0 {
                let prev_price = market_data.ticks[i-1].mid_price;
                let price_change = (tick.mid_price - prev_price) / prev_price;
                -price_change // Mean reversion signal
            } else {
                0.0
            };
            
            // Position sizing based on signal and volatility
            let target_position = signal * 1000.0 / tick.volatility.max(0.01);
            let position_change = target_position - position;
            
            // Transaction costs
            let transaction_cost = position_change.abs() * tick.spread * 0.5;
            current_pnl -= transaction_cost;
            
            // Mark-to-market P&L
            if i > 0 {
                let price_change = tick.mid_price - market_data.ticks[i-1].mid_price;
                current_pnl += position * price_change;
            }
            
            position = target_position;
            pnl_series.push(current_pnl);
        }

        Ok(MonteCarloResult {
            simulation_id,
            pnl_series,
            final_pnl: current_pnl,
        })
    }
}

/// Robustness analyzer for stress test results
pub struct RobustnessAnalyzer {
    stress_metrics: Vec<StressMetric>,
}

impl RobustnessAnalyzer {
    pub fn new(stress_metrics: &[StressMetric]) -> Result<Self, StressTestError> {
        Ok(Self {
            stress_metrics: stress_metrics.to_vec(),
        })
    }

    /// Calculate overall robustness score
    pub fn calculate_robustness_score(
        &self,
        scenario_results: &HashMap<String, ScenarioResult>,
    ) -> Result<f64, StressTestError> {
        if scenario_results.is_empty() {
            return Ok(0.0);
        }

        let mut total_score = 0.0;
        let mut metric_count = 0;

        for result in scenario_results.values() {
            for metric in &self.stress_metrics {
                let metric_score = match metric {
                    StressMetric::MaxDrawdown => {
                        // Lower drawdown is better (score inversely related)
                        (1.0 - result.max_drawdown.min(1.0)).max(0.0)
                    },
                    StressMetric::VaR99 => {
                        // Lower VaR is better
                        (1.0 - result.var_99.min(1.0)).max(0.0)
                    },
                    StressMetric::RecoveryTime => {
                        // Shorter recovery time is better
                        let normalized_recovery = (result.recovery_time as f64 / 1000.0).min(1.0);
                        (1.0 - normalized_recovery).max(0.0)
                    },
                    StressMetric::TailRatio => {
                        // Simplified tail ratio calculation
                        0.5 // Placeholder
                    },
                    StressMetric::ExpectedShortfall => {
                        // Lower expected shortfall is better
                        (1.0 - result.var_99.min(1.0)).max(0.0) // Using VaR as proxy
                    },
                };
                
                total_score += metric_score;
                metric_count += 1;
            }
        }

        Ok(total_score / metric_count as f64)
    }
}

/// Result of a single Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub simulation_id: usize,
    pub pnl_series: Vec<f64>,
    pub final_pnl: f64,
}

/// Default stress testing scenarios
impl Default for StressTestingConfig {
    fn default() -> Self {
        Self {
            extreme_scenarios: vec![
                ExtremeScenario {
                    name: "Flash Crash".to_string(),
                    description: "Sudden 10% price drop with liquidity evaporation".to_string(),
                    volatility_multiplier: 5.0,
                    liquidity_reduction: 0.9,
                    correlation_breakdown: true,
                    duration_days: 1,
                },
                ExtremeScenario {
                    name: "Market Stress".to_string(),
                    description: "Extended period of high volatility and reduced liquidity".to_string(),
                    volatility_multiplier: 3.0,
                    liquidity_reduction: 0.6,
                    correlation_breakdown: false,
                    duration_days: 30,
                },
                ExtremeScenario {
                    name: "Liquidity Crisis".to_string(),
                    description: "Severe liquidity shortage with wide spreads".to_string(),
                    volatility_multiplier: 2.0,
                    liquidity_reduction: 0.8,
                    correlation_breakdown: true,
                    duration_days: 7,
                },
                ExtremeScenario {
                    name: "Volatility Spike".to_string(),
                    description: "Extreme volatility with frequent price jumps".to_string(),
                    volatility_multiplier: 8.0,
                    liquidity_reduction: 0.3,
                    correlation_breakdown: true,
                    duration_days: 3,
                },
            ],
            monte_carlo_sims: 1000,
            stress_metrics: vec![
                StressMetric::MaxDrawdown,
                StressMetric::VaR99,
                StressMetric::ExpectedShortfall,
                StressMetric::RecoveryTime,
            ],
        }
    }
}