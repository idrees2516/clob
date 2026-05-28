use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order},
    optimal_trading::{OptimalController, MarketImpact},
    toxicity::{ToxicityMetrics, ToxicityState},
};

#[derive(Debug, Error)]
pub enum RiskError {
    #[error("Position limit error: {0}")]
    PositionError(String),
    #[error("VaR calculation error: {0}")]
    VaRError(String),
    #[error("Margin error: {0}")]
    MarginError(String),
}

/// Risk limits for position management
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub position_bounds: Vec<Bound>,
    pub var_limits: HashMap<Timeframe, f64>,
    pub margin_requirements: MarginModel,
    pub stress_scenarios: Vec<StressScenario>,
}

#[derive(Debug, Clone)]
pub struct Bound {
    pub lower: f64,
    pub upper: f64,
    pub timeframe: Timeframe,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Timeframe {
    Intraday,
    Daily,
    Weekly,
    Monthly,
}

/// Margin model for collateral management
#[derive(Debug, Clone)]
pub struct MarginModel {
    pub initial_margin: f64,
    pub maintenance_margin: f64,
    pub variation_margin: f64,
    pub margin_multiplier: f64,
    pub margin_call_threshold: f64,
}

impl MarginModel {
    pub fn new(
        initial_margin: f64,
        maintenance_margin: f64,
        variation_margin: f64,
        margin_multiplier: f64,
        margin_call_threshold: f64,
    ) -> Self {
        Self {
            initial_margin,
            maintenance_margin,
            variation_margin,
            margin_multiplier,
            margin_call_threshold,
        }
    }

    pub fn calculate_margin(
        &self,
        position: f64,
        price: f64,
        volatility: f64,
    ) -> Result<MarginRequirement, RiskError> {
        let position_value = position.abs() * price;
        
        // Base margin requirements
        let initial = position_value * self.initial_margin;
        let maintenance = position_value * self.maintenance_margin;
        
        // Adjust for volatility
        let volatility_adjustment = 1.0 + (volatility - 0.2).max(0.0);
        let adjusted_initial = initial * volatility_adjustment;
        let adjusted_maintenance = maintenance * volatility_adjustment;
        
        // Variation margin based on P&L
        let variation = position_value * self.variation_margin;
        
        Ok(MarginRequirement {
            initial_margin: adjusted_initial,
            maintenance_margin: adjusted_maintenance,
            variation_margin: variation,
            total_requirement: adjusted_initial + variation,
        })
    }

    pub fn check_margin_call(
        &self,
        current_margin: f64,
        required_margin: f64,
    ) -> bool {
        current_margin < required_margin * self.margin_call_threshold
    }
}

#[derive(Debug, Clone)]
pub struct MarginRequirement {
    pub initial_margin: f64,
    pub maintenance_margin: f64,
    pub variation_margin: f64,
    pub total_requirement: f64,
}

/// Stress testing scenarios
#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub price_shock: f64,
    pub volatility_shock: f64,
    pub correlation_shock: f64,
    pub liquidity_shock: f64,
}

impl StressScenario {
    pub fn new(
        name: &str,
        price_shock: f64,
        volatility_shock: f64,
        correlation_shock: f64,
        liquidity_shock: f64,
    ) -> Self {
        Self {
            name: name.to_string(),
            price_shock,
            volatility_shock,
            correlation_shock,
            liquidity_shock,
        }
    }

    pub fn apply_shock(
        &self,
        position: f64,
        price: f64,
        volatility: f64,
    ) -> (f64, f64, f64) {
        let shocked_price = price * (1.0 + self.price_shock);
        let shocked_volatility = volatility * (1.0 + self.volatility_shock);
        let shocked_position = position * (1.0 - self.liquidity_shock);
        
        (shocked_position, shocked_price, shocked_volatility)
    }
}

/// Risk management framework
pub struct RiskManager {
    pub limits: RiskLimits,
    pub volatility_model: RoughVolatilityModel,
    pub var_calculator: VaRCalculator,
    pub position_manager: PositionManager,
}

impl RiskManager {
    pub fn new(
        limits: RiskLimits,
        volatility_model: RoughVolatilityModel,
        confidence_level: f64,
        holding_period: f64,
    ) -> Self {
        let var_calculator = VaRCalculator::new(confidence_level, holding_period);
        let position_manager = PositionManager::new(limits.position_bounds.clone());
        
        Self {
            limits,
            volatility_model,
            var_calculator,
            position_manager,
        }
    }

    pub fn check_risk_limits(
        &self,
        position: f64,
        price: f64,
        volatility: f64,
    ) -> Result<RiskMetrics, RiskError> {
        // Check position limits with time-varying thresholds
        self.position_manager.check_limits(position)?;
        
        // Calculate VaR with dynamic confidence levels
        let base_confidence = 0.99;
        let volatility_adjustment = (volatility / 0.2).min(1.2);
        let adjusted_confidence = (base_confidence * volatility_adjustment).min(0.999);
        
        let var = self.var_calculator.calculate_var(
            position,
            price,
            volatility,
        )?;
        
        // Calculate margin requirements with market stress adjustments
        let margin = self.limits.margin_requirements.calculate_margin(
            position,
            price,
            volatility,
        )?;
        
        // Run comprehensive stress tests
        let stress_results = self.run_stress_tests(position, price, volatility)?;
        
        // Compute risk utilization metrics
        let var_utilization = var.parametric_var / 
            self.limits.var_limits.get(&Timeframe::Daily).unwrap_or(&f64::INFINITY);
            
        let margin_utilization = margin.total_requirement /
            (margin.initial_margin * self.limits.margin_requirements.margin_multiplier);
            
        let position_utilization = position / self.position_manager.max_position();
        
        // Aggregate stress test results
        let max_stress_loss = stress_results.iter()
            .map(|r| r.pnl.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
            
        let avg_stress_var = stress_results.iter()
            .map(|r| r.var.parametric_var)
            .sum::<f64>() / stress_results.len() as f64;
            
        Ok(RiskMetrics {
            var,
            margin,
            stress_results,
            position_utilization,
            var_utilization,
            margin_utilization,
            max_stress_loss,
            avg_stress_var,
        })
    }

    fn run_stress_tests(
        &self,
        position: f64,
        price: f64,
        volatility: f64,
    ) -> Result<Vec<StressTestResult>, RiskError> {
        let mut results = Vec::new();
        
        for scenario in &self.limits.stress_scenarios {
            // Apply market shock scenarios
            let (shocked_position, shocked_price, shocked_volatility) = 
                scenario.apply_shock(position, price, volatility);
                
            // Calculate scenario P&L
            let base_value = position * price;
            let shocked_value = shocked_position * shocked_price;
            let pnl = shocked_value - base_value;
            
            // Compute scenario VaR
            let var = self.var_calculator.calculate_var(
                shocked_position,
                shocked_price,
                shocked_volatility,
            )?;
            
            // Calculate scenario margin requirements
            let margin = self.limits.margin_requirements.calculate_margin(
                shocked_position,
                shocked_price,
                shocked_volatility,
            )?;
            
            // Apply correlation shocks to portfolio
            let correlation_impact = if scenario.correlation_shock != 0.0 {
                let base_correlation = 0.5; // Example base correlation
                let shocked_correlation = base_correlation * (1.0 + scenario.correlation_shock);
                pnl * shocked_correlation
            } else {
                pnl
            };
            
            // Apply liquidity shocks
            let liquidity_adjusted_pnl = if scenario.liquidity_shock > 0.0 {
                let liquidity_cost = shocked_value * scenario.liquidity_shock;
                correlation_impact - liquidity_cost
            } else {
                correlation_impact
            };
            
            results.push(StressTestResult {
                scenario_name: scenario.name.clone(),
                pnl: liquidity_adjusted_pnl,
                var,
                margin,
            });
        }
        
        Ok(results)
    }
}

/// Value at Risk calculator
pub struct VaRCalculator {
    confidence_level: f64,
    holding_period: f64,
    historical_scenarios: Vec<HistoricalScenario>,
}

impl VaRCalculator {
    pub fn new(confidence_level: f64, holding_period: f64) -> Self {
        Self {
            confidence_level,
            holding_period,
            historical_scenarios: Vec::new(),
        }
    }

    pub fn calculate_var(
        &self,
        position: f64,
        price: f64,
        volatility: f64,
    ) -> Result<VaRMetrics, RiskError> {
        // Parametric VaR with Student-t distribution
        let position_value = position.abs() * price;
        let std_dev = volatility * price;
        let dof = 5.0; // Degrees of freedom for fat tails
        let t_multiplier = self.student_t_quantile(1.0 - self.confidence_level, dof);
        
        let parametric_var = position_value * std_dev * t_multiplier * 
            self.holding_period.sqrt();
            
        // Historical VaR with kernel density estimation
        let historical_var = if !self.historical_scenarios.is_empty() {
            self.calculate_historical_var_kde(
                position_value,
                &self.historical_scenarios,
                std_dev,
            )?
        } else {
            parametric_var
        };
        
        // Expected Shortfall with mixture model
        let es = self.calculate_expected_shortfall_mixture(
            position_value,
            std_dev,
            t_multiplier,
            dof,
        )?;
        
        Ok(VaRMetrics {
            parametric_var,
            historical_var,
            expected_shortfall: es,
            confidence_level: self.confidence_level,
            holding_period: self.holding_period,
        })
    }

    fn student_t_quantile(&self, p: f64, dof: f64) -> f64 {
        // Approximation of Student's t-distribution quantile
        let x = self.confidence_level_to_z_score(p);
        let h = x * x / dof;
        let a = (dof + 1.0) / (4.0 * dof);
        let b = (5.0 * dof + 2.0) / (96.0 * dof * dof);
        
        x * (1.0 + h * (a + h * b))
    }

    fn confidence_level_to_z_score(&self, confidence_level: f64) -> f64 {
        // Approximation of the inverse normal CDF
        let x = confidence_level - 0.5;
        let t = (1.0 - x * x).ln().abs().sqrt();
        
        2.515517 + 0.802853 * t + 0.010328 * t * t
    }

    fn calculate_historical_var_kde(
        &self,
        position_value: f64,
        scenarios: &[HistoricalScenario],
        bandwidth: f64,
    ) -> Result<f64, RiskError> {
        if scenarios.is_empty() {
            return Err(RiskError::VaRError(
                "No historical scenarios available".to_string(),
            ));
        }
        
        let returns: Vec<f64> = scenarios.iter()
            .map(|s| s.return_value)
            .collect();
            
        // Kernel density estimation
        let n = returns.len();
        let h = bandwidth * (4.0 / (3.0 * n as f64)).powf(0.2);
        
        let mut density_points = Vec::new();
        let grid_points: Vec<f64> = (0..100).map(|i| {
            let p = i as f64 / 100.0;
            returns[0] + p * (returns[n-1] - returns[0])
        }).collect();
        
        for x in grid_points {
            let density = returns.iter()
                .map(|&r| {
                    let z = (x - r) / h;
                    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
                })
                .sum::<f64>() / (n as f64 * h);
            density_points.push((x, density));
        }
        
        // Find VaR using numerical integration
        let mut cumsum = 0.0;
        let target = 1.0 - self.confidence_level;
        
        for i in 0..density_points.len()-1 {
            let (x1, y1) = density_points[i];
            let (x2, y2) = density_points[i+1];
            cumsum += 0.5 * (y1 + y2) * (x2 - x1);
            
            if cumsum >= target {
                let var = -position_value * x1;
                return Ok(var);
            }
        }
        
        Err(RiskError::VaRError("Failed to compute historical VaR".to_string()))
    }

    fn calculate_expected_shortfall_mixture(
        &self,
        position_value: f64,
        std_dev: f64,
        t_multiplier: f64,
        dof: f64,
    ) -> Result<f64, RiskError> {
        // Mixture of normal and Student-t distributions
        let normal_weight = 0.7;
        let t_weight = 0.3;
        
        // Normal component
        let normal_es = position_value * std_dev * 
            (-0.5 * t_multiplier * t_multiplier).exp() /
            ((1.0 - self.confidence_level) * (2.0 * std::f64::consts::PI).sqrt());
            
        // Student-t component
        let t_es = position_value * std_dev * 
            (dof + t_multiplier * t_multiplier) /
            ((dof - 1.0) * (1.0 - self.confidence_level)) *
            (1.0 + t_multiplier * t_multiplier / dof).powf(-(dof + 1.0) / 2.0);
            
        let mixture_es = normal_weight * normal_es + t_weight * t_es;
        
        Ok(mixture_es)
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalScenario {
    pub timestamp: f64,
    pub return_value: f64,
    pub volatility: f64,
}

#[derive(Debug, Clone)]
pub struct VaRMetrics {
    pub parametric_var: f64,
    pub historical_var: f64,
    pub expected_shortfall: f64,
    pub confidence_level: f64,
    pub holding_period: f64,
}

/// Position manager for limit monitoring
pub struct PositionManager {
    bounds: Vec<Bound>,
}

impl PositionManager {
    pub fn new(bounds: Vec<Bound>) -> Self {
        Self { bounds }
    }

    pub fn check_limits(&self, position: f64) -> Result<(), RiskError> {
        for bound in &self.bounds {
            if position < bound.lower || position > bound.upper {
                return Err(RiskError::PositionError(format!(
                    "Position {} exceeds bounds [{}, {}] for timeframe {:?}",
                    position, bound.lower, bound.upper, bound.timeframe
                )));
            }
        }
        Ok(())
    }

    pub fn max_position(&self) -> f64 {
        self.bounds.iter()
            .map(|b| b.upper.abs().max(b.lower.abs()))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

#[derive(Debug)]
pub struct RiskMetrics {
    pub var: VaRMetrics,
    pub margin: MarginRequirement,
    pub stress_results: Vec<StressTestResult>,
    pub position_utilization: f64,
    pub var_utilization: f64,
    pub margin_utilization: f64,
    pub max_stress_loss: f64,
    pub avg_stress_var: f64,
}

#[derive(Debug)]
pub struct StressTestResult {
    pub scenario_name: String,
    pub pnl: f64,
    pub var: VaRMetrics,
    pub margin: MarginRequirement,
}
