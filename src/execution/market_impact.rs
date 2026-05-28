//! Market Impact Modeling and Optimization
//! 
//! Implements sophisticated market impact models with:
//! - Market impact estimation using square-root law
//! - Temporary and permanent impact decomposition
//! - Execution cost optimization with impact-timing trade-off
//! - Slippage analysis and performance attribution

use super::{ExecutionError, MarketState, MarketConditions, ExecutionMetrics};
use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Market impact model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactParams {
    /// Temporary impact coefficient (η)
    pub temporary_impact_coeff: FixedPoint,
    /// Temporary impact exponent (α)
    pub temporary_impact_exponent: FixedPoint,
    /// Permanent impact coefficient (λ)
    pub permanent_impact_coeff: FixedPoint,
    /// Permanent impact exponent (β)
    pub permanent_impact_exponent: FixedPoint,
    /// Cross-impact coefficient for related assets
    pub cross_impact_coeff: FixedPoint,
    /// Impact decay rate (for temporary impact)
    pub decay_rate: FixedPoint,
    /// Volatility scaling factor
    pub volatility_scaling: FixedPoint,
    /// Liquidity scaling factor
    pub liquidity_scaling: FixedPoint,
}

impl Default for MarketImpactParams {
    fn default() -> Self {
        Self {
            temporary_impact_coeff: FixedPoint::from_float(0.1),
            temporary_impact_exponent: FixedPoint::from_float(0.5), // Square-root law
            permanent_impact_coeff: FixedPoint::from_float(0.01),
            permanent_impact_exponent: FixedPoint::from_float(1.0), // Linear
            cross_impact_coeff: FixedPoint::from_float(0.05),
            decay_rate: FixedPoint::from_float(0.1),
            volatility_scaling: FixedPoint::from_float(1.0),
            liquidity_scaling: FixedPoint::from_float(1.0),
        }
    }
}

/// Market impact estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactEstimate {
    /// Total market impact
    pub total_impact: FixedPoint,
    /// Temporary impact component
    pub temporary_impact: FixedPoint,
    /// Permanent impact component
    pub permanent_impact: FixedPoint,
    /// Cross-impact from related assets
    pub cross_impact: FixedPoint,
    /// Impact as percentage of price
    pub impact_percentage: FixedPoint,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (FixedPoint, FixedPoint),
}

/// Execution cost breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCostBreakdown {
    /// Market impact cost
    pub market_impact_cost: FixedPoint,
    /// Timing cost (opportunity cost)
    pub timing_cost: FixedPoint,
    /// Transaction costs (fees, commissions)
    pub transaction_cost: FixedPoint,
    /// Total execution cost
    pub total_cost: FixedPoint,
    /// Cost as basis points
    pub cost_basis_points: FixedPoint,
}

/// Slippage analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageAnalysis {
    /// Total slippage
    pub total_slippage: FixedPoint,
    /// Market impact slippage
    pub impact_slippage: FixedPoint,
    /// Timing slippage
    pub timing_slippage: FixedPoint,
    /// Spread slippage
    pub spread_slippage: FixedPoint,
    /// Slippage attribution by time period
    pub time_period_attribution: Vec<SlippagePeriod>,
}

/// Slippage for a specific time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippagePeriod {
    pub start_time: u64,
    pub end_time: u64,
    pub slippage: FixedPoint,
    pub volume_executed: u64,
    pub market_conditions: MarketConditions,
}

/// Optimal execution trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalExecutionTrajectory {
    /// Time points for execution
    pub time_points: Vec<u64>,
    /// Optimal execution rates at each time point
    pub execution_rates: Vec<FixedPoint>,
    /// Expected costs at each time point
    pub expected_costs: Vec<FixedPoint>,
    /// Risk measures at each time point
    pub risk_measures: Vec<FixedPoint>,
    /// Total expected cost
    pub total_expected_cost: FixedPoint,
}

/// Market impact model implementation
#[derive(Debug)]
pub struct MarketImpactModel {
    /// Model parameters
    params: MarketImpactParams,
    /// Historical impact observations
    impact_history: VecDeque<ImpactObservation>,
    /// Model calibration data
    calibration_data: CalibrationData,
    /// Performance metrics
    model_performance: ModelPerformance,
}

/// Historical impact observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactObservation {
    pub timestamp: u64,
    pub volume: u64,
    pub participation_rate: FixedPoint,
    pub observed_impact: FixedPoint,
    pub predicted_impact: FixedPoint,
    pub market_state: MarketState,
    pub market_conditions: MarketConditions,
}

/// Model calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub last_calibration: u64,
    pub calibration_r_squared: FixedPoint,
    pub parameter_confidence: Vec<FixedPoint>,
    pub residual_statistics: ResidualStatistics,
}

/// Residual statistics for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStatistics {
    pub mean_residual: FixedPoint,
    pub residual_std_dev: FixedPoint,
    pub max_residual: FixedPoint,
    pub residual_autocorrelation: FixedPoint,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    pub prediction_accuracy: FixedPoint,
    pub mean_absolute_error: FixedPoint,
    pub root_mean_square_error: FixedPoint,
    pub directional_accuracy: FixedPoint,
}

impl MarketImpactModel {
    /// Create new market impact model
    pub fn new(params: MarketImpactParams) -> Self {
        Self {
            params,
            impact_history: VecDeque::new(),
            calibration_data: CalibrationData {
                last_calibration: 0,
                calibration_r_squared: FixedPoint::ZERO,
                parameter_confidence: vec![FixedPoint::ZERO; 8],
                residual_statistics: ResidualStatistics {
                    mean_residual: FixedPoint::ZERO,
                    residual_std_dev: FixedPoint::ZERO,
                    max_residual: FixedPoint::ZERO,
                    residual_autocorrelation: FixedPoint::ZERO,
                },
            },
            model_performance: ModelPerformance {
                prediction_accuracy: FixedPoint::ZERO,
                mean_absolute_error: FixedPoint::ZERO,
                root_mean_square_error: FixedPoint::ZERO,
                directional_accuracy: FixedPoint::ZERO,
            },
        }
    }

    /// Estimate market impact for given execution parameters
    pub fn estimate_market_impact(
        &self,
        volume: u64,
        participation_rate: FixedPoint,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
        execution_horizon: u64,
    ) -> Result<MarketImpactEstimate, ExecutionError> {
        // Calculate temporary impact using square-root law: I_temp = η * v^α
        let normalized_volume = FixedPoint::from_integer(volume as i64) / 
                               FixedPoint::from_integer(market_state.average_daily_volume as i64);
        
        let temporary_impact = self.calculate_temporary_impact(
            normalized_volume,
            participation_rate,
            market_state,
            market_conditions,
        )?;

        // Calculate permanent impact: I_perm = λ * x^β
        let permanent_impact = self.calculate_permanent_impact(
            normalized_volume,
            market_state,
            market_conditions,
        )?;

        // Calculate cross-impact from related assets
        let cross_impact = self.calculate_cross_impact(
            volume,
            market_state,
            market_conditions,
        )?;

        // Total impact
        let total_impact = temporary_impact + permanent_impact + cross_impact;

        // Calculate impact as percentage of price
        let impact_percentage = total_impact / market_state.mid_price() * 
                               FixedPoint::from_integer(100);

        // Calculate confidence interval
        let confidence_interval = self.calculate_impact_confidence_interval(
            total_impact,
            volume,
            market_conditions,
        )?;

        Ok(MarketImpactEstimate {
            total_impact,
            temporary_impact,
            permanent_impact,
            cross_impact,
            impact_percentage,
            confidence_interval,
        })
    }

    /// Calculate optimal execution trajectory
    pub fn calculate_optimal_trajectory(
        &self,
        total_volume: u64,
        execution_horizon: u64,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
        risk_aversion: FixedPoint,
    ) -> Result<OptimalExecutionTrajectory, ExecutionError> {
        let num_time_points = 100; // Discretization
        let dt = execution_horizon as f64 / num_time_points as f64;
        
        let mut time_points = Vec::with_capacity(num_time_points);
        let mut execution_rates = Vec::with_capacity(num_time_points);
        let mut expected_costs = Vec::with_capacity(num_time_points);
        let mut risk_measures = Vec::with_capacity(num_time_points);

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Solve optimal execution using dynamic programming approach
        let mut remaining_volume = total_volume as f64;
        let mut total_expected_cost = 0.0;

        for i in 0..num_time_points {
            let t = current_time + (i as f64 * dt) as u64;
            let time_remaining = (num_time_points - i) as f64 * dt;
            
            // Calculate optimal execution rate using Almgren-Chriss model
            let optimal_rate = self.calculate_optimal_execution_rate(
                remaining_volume,
                time_remaining,
                market_state,
                market_conditions,
                risk_aversion,
            )?;

            let execution_volume = (optimal_rate.to_float() * dt).min(remaining_volume);
            
            // Calculate expected cost for this execution
            let participation_rate = if market_state.average_daily_volume > 0 {
                FixedPoint::from_float(execution_volume / (market_state.average_daily_volume as f64 * dt / 86400.0))
            } else {
                FixedPoint::from_float(0.1)
            };

            let impact_estimate = self.estimate_market_impact(
                execution_volume as u64,
                participation_rate,
                market_state,
                market_conditions,
                dt as u64,
            )?;

            let expected_cost = impact_estimate.total_impact.to_float() * execution_volume;
            
            // Calculate risk measure (variance of execution cost)
            let risk_measure = self.calculate_execution_risk(
                execution_volume,
                market_state,
                market_conditions,
                time_remaining,
            )?;

            time_points.push(t);
            execution_rates.push(optimal_rate);
            expected_costs.push(FixedPoint::from_float(expected_cost));
            risk_measures.push(risk_measure);

            remaining_volume -= execution_volume;
            total_expected_cost += expected_cost;

            if remaining_volume <= 0.0 {
                break;
            }
        }

        Ok(OptimalExecutionTrajectory {
            time_points,
            execution_rates,
            expected_costs,
            risk_measures,
            total_expected_cost: FixedPoint::from_float(total_expected_cost),
        })
    }

    /// Analyze execution slippage
    pub fn analyze_slippage(
        &self,
        execution_records: &[ExecutionRecord],
        benchmark_price: FixedPoint,
        market_states: &[MarketState],
    ) -> Result<SlippageAnalysis, ExecutionError> {
        if execution_records.is_empty() {
            return Err(ExecutionError::InvalidParameters(
                "No execution records provided".to_string()
            ));
        }

        let mut total_slippage = FixedPoint::ZERO;
        let mut impact_slippage = FixedPoint::ZERO;
        let mut timing_slippage = FixedPoint::ZERO;
        let mut spread_slippage = FixedPoint::ZERO;
        let mut time_period_attribution = Vec::new();

        // Group records by time periods (e.g., 5-minute intervals)
        let period_duration = 300; // 5 minutes
        let mut current_period_start = execution_records[0].timestamp;
        let mut period_records = Vec::new();

        for record in execution_records {
            if record.timestamp - current_period_start > period_duration {
                // Process current period
                if !period_records.is_empty() {
                    let period_analysis = self.analyze_period_slippage(
                        &period_records,
                        benchmark_price,
                        current_period_start,
                        current_period_start + period_duration,
                    )?;
                    
                    time_period_attribution.push(period_analysis);
                    period_records.clear();
                }
                current_period_start = record.timestamp;
            }
            period_records.push(record.clone());
        }

        // Process final period
        if !period_records.is_empty() {
            let period_analysis = self.analyze_period_slippage(
                &period_records,
                benchmark_price,
                current_period_start,
                execution_records.last().unwrap().timestamp,
            )?;
            time_period_attribution.push(period_analysis);
        }

        // Calculate aggregate slippage components
        for period in &time_period_attribution {
            total_slippage = total_slippage + period.slippage;
        }

        // Decompose slippage into components
        for record in execution_records {
            // Market impact slippage
            let predicted_impact = self.estimate_market_impact(
                record.executed_volume,
                record.participation_rate,
                &MarketState {
                    symbol: "".to_string(),
                    bid_price: benchmark_price,
                    ask_price: benchmark_price,
                    bid_volume: 0,
                    ask_volume: 0,
                    last_price: benchmark_price,
                    last_volume: 0,
                    timestamp: record.timestamp,
                    volatility: FixedPoint::from_float(0.02),
                    average_daily_volume: 1000000,
                },
                &MarketConditions {
                    volatility_regime: super::VolatilityRegime::Normal,
                    liquidity_level: super::LiquidityLevel::Normal,
                    trend_direction: super::TrendDirection::Sideways,
                    market_hours: super::MarketHours::Regular,
                    news_impact: super::NewsImpact::None,
                },
                300,
            )?;

            impact_slippage = impact_slippage + predicted_impact.total_impact;

            // Timing slippage (difference between execution price and benchmark)
            let execution_slippage = record.execution_price - benchmark_price;
            timing_slippage = timing_slippage + execution_slippage;

            // Spread slippage (estimated)
            let estimated_spread = benchmark_price * FixedPoint::from_float(0.001); // 10 bps
            spread_slippage = spread_slippage + estimated_spread / FixedPoint::from_integer(2);
        }

        Ok(SlippageAnalysis {
            total_slippage,
            impact_slippage,
            timing_slippage,
            spread_slippage,
            time_period_attribution,
        })
    }

    /// Calculate execution cost breakdown
    pub fn calculate_execution_costs(
        &self,
        execution_records: &[ExecutionRecord],
        benchmark_price: FixedPoint,
        transaction_cost_rate: FixedPoint, // basis points
    ) -> Result<ExecutionCostBreakdown, ExecutionError> {
        let mut market_impact_cost = FixedPoint::ZERO;
        let mut timing_cost = FixedPoint::ZERO;
        let mut transaction_cost = FixedPoint::ZERO;

        for record in execution_records {
            // Market impact cost
            let impact_estimate = self.estimate_market_impact(
                record.executed_volume,
                record.participation_rate,
                &MarketState {
                    symbol: "".to_string(),
                    bid_price: benchmark_price,
                    ask_price: benchmark_price,
                    bid_volume: 0,
                    ask_volume: 0,
                    last_price: benchmark_price,
                    last_volume: 0,
                    timestamp: record.timestamp,
                    volatility: FixedPoint::from_float(0.02),
                    average_daily_volume: 1000000,
                },
                &MarketConditions {
                    volatility_regime: super::VolatilityRegime::Normal,
                    liquidity_level: super::LiquidityLevel::Normal,
                    trend_direction: super::TrendDirection::Sideways,
                    market_hours: super::MarketHours::Regular,
                    news_impact: super::NewsImpact::None,
                },
                300,
            )?;

            let impact_cost = impact_estimate.total_impact * 
                             FixedPoint::from_integer(record.executed_volume as i64);
            market_impact_cost = market_impact_cost + impact_cost;

            // Timing cost (opportunity cost)
            let price_diff = record.execution_price - benchmark_price;
            let timing_cost_component = price_diff * 
                                       FixedPoint::from_integer(record.executed_volume as i64);
            timing_cost = timing_cost + timing_cost_component;

            // Transaction cost
            let notional = record.execution_price * 
                          FixedPoint::from_integer(record.executed_volume as i64);
            let transaction_cost_component = notional * transaction_cost_rate / 
                                           FixedPoint::from_integer(10000); // Convert from bps
            transaction_cost = transaction_cost + transaction_cost_component;
        }

        let total_cost = market_impact_cost + timing_cost + transaction_cost;
        
        // Calculate total notional for basis points calculation
        let total_notional: FixedPoint = execution_records.iter()
            .map(|r| r.execution_price * FixedPoint::from_integer(r.executed_volume as i64))
            .fold(FixedPoint::ZERO, |acc, x| acc + x);

        let cost_basis_points = if total_notional > FixedPoint::ZERO {
            total_cost / total_notional * FixedPoint::from_integer(10000)
        } else {
            FixedPoint::ZERO
        };

        Ok(ExecutionCostBreakdown {
            market_impact_cost,
            timing_cost,
            transaction_cost,
            total_cost,
            cost_basis_points,
        })
    }

    /// Record impact observation for model improvement
    pub fn record_impact_observation(
        &mut self,
        volume: u64,
        participation_rate: FixedPoint,
        observed_impact: FixedPoint,
        market_state: MarketState,
        market_conditions: MarketConditions,
    ) -> Result<(), ExecutionError> {
        // Calculate predicted impact
        let predicted_impact = self.estimate_market_impact(
            volume,
            participation_rate,
            &market_state,
            &market_conditions,
            300, // 5 minutes
        )?.total_impact;

        let observation = ImpactObservation {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            volume,
            participation_rate,
            observed_impact,
            predicted_impact,
            market_state,
            market_conditions,
        };

        self.impact_history.push_back(observation);

        // Maintain maximum history size
        while self.impact_history.len() > 10000 {
            self.impact_history.pop_front();
        }

        // Update model performance metrics
        self.update_model_performance()?;

        Ok(())
    }

    // Private helper methods

    fn calculate_temporary_impact(
        &self,
        normalized_volume: FixedPoint,
        participation_rate: FixedPoint,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<FixedPoint, ExecutionError> {
        // Base temporary impact: η * v^α
        let base_impact = self.params.temporary_impact_coeff * 
                         normalized_volume.pow(self.params.temporary_impact_exponent.to_float() as i32);

        // Volatility scaling
        let volatility_adjustment = match market_conditions.volatility_regime {
            super::VolatilityRegime::Low => FixedPoint::from_float(0.8),
            super::VolatilityRegime::High => FixedPoint::from_float(1.3),
            super::VolatilityRegime::Extreme => FixedPoint::from_float(1.8),
            _ => FixedPoint::from_integer(1),
        };

        // Liquidity scaling
        let liquidity_adjustment = match market_conditions.liquidity_level {
            super::LiquidityLevel::Low => FixedPoint::from_float(1.5),
            super::LiquidityLevel::High => FixedPoint::from_float(0.7),
            _ => FixedPoint::from_integer(1),
        };

        // Participation rate adjustment
        let participation_adjustment = FixedPoint::from_integer(1) + 
                                     (participation_rate - FixedPoint::from_float(0.1)) * 
                                     FixedPoint::from_float(2.0);

        let adjusted_impact = base_impact * volatility_adjustment * 
                             liquidity_adjustment * participation_adjustment;

        Ok(adjusted_impact)
    }

    fn calculate_permanent_impact(
        &self,
        normalized_volume: FixedPoint,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<FixedPoint, ExecutionError> {
        // Base permanent impact: λ * x^β
        let base_impact = self.params.permanent_impact_coeff * 
                         normalized_volume.pow(self.params.permanent_impact_exponent.to_float() as i32);

        // Market condition adjustments
        let condition_adjustment = match market_conditions.liquidity_level {
            super::LiquidityLevel::Low => FixedPoint::from_float(1.3),
            super::LiquidityLevel::High => FixedPoint::from_float(0.8),
            _ => FixedPoint::from_integer(1),
        };

        Ok(base_impact * condition_adjustment)
    }

    fn calculate_cross_impact(
        &self,
        volume: u64,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<FixedPoint, ExecutionError> {
        // Simplified cross-impact calculation
        // In practice, this would consider correlated assets
        let normalized_volume = FixedPoint::from_integer(volume as i64) / 
                               FixedPoint::from_integer(market_state.average_daily_volume as i64);
        
        let cross_impact = self.params.cross_impact_coeff * normalized_volume;
        
        Ok(cross_impact)
    }

    fn calculate_impact_confidence_interval(
        &self,
        total_impact: FixedPoint,
        volume: u64,
        market_conditions: &MarketConditions,
    ) -> Result<(FixedPoint, FixedPoint), ExecutionError> {
        // Simple confidence interval based on model uncertainty
        let uncertainty_factor = match market_conditions.volatility_regime {
            super::VolatilityRegime::Low => FixedPoint::from_float(0.1),
            super::VolatilityRegime::High => FixedPoint::from_float(0.3),
            super::VolatilityRegime::Extreme => FixedPoint::from_float(0.5),
            _ => FixedPoint::from_float(0.2),
        };

        let uncertainty = total_impact * uncertainty_factor;
        let lower_bound = total_impact - uncertainty;
        let upper_bound = total_impact + uncertainty;

        Ok((lower_bound, upper_bound))
    }

    fn calculate_optimal_execution_rate(
        &self,
        remaining_volume: f64,
        time_remaining: f64,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
        risk_aversion: FixedPoint,
    ) -> Result<FixedPoint, ExecutionError> {
        if time_remaining <= 0.0 {
            return Ok(FixedPoint::from_float(remaining_volume));
        }

        // Almgren-Chriss optimal execution rate
        let volatility = market_state.volatility.to_float();
        let gamma = risk_aversion.to_float();
        let eta = self.params.temporary_impact_coeff.to_float();
        let lambda = self.params.permanent_impact_coeff.to_float();

        // Calculate optimal rate: v* = X/T + (γσ²/η) * (X - ∫v dt)
        let base_rate = remaining_volume / time_remaining;
        let risk_adjustment = (gamma * volatility * volatility / eta) * remaining_volume;
        
        let optimal_rate = base_rate + risk_adjustment * 0.1; // Simplified

        Ok(FixedPoint::from_float(optimal_rate.max(0.0)))
    }

    fn calculate_execution_risk(
        &self,
        execution_volume: f64,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
        time_remaining: f64,
    ) -> Result<FixedPoint, ExecutionError> {
        // Risk measure based on price volatility and execution size
        let volatility = market_state.volatility;
        let volume_factor = FixedPoint::from_float(execution_volume / 1000.0); // Normalize
        let time_factor = FixedPoint::from_float((time_remaining / 3600.0).sqrt()); // Square root of time
        
        let risk_measure = volatility * volume_factor * time_factor;
        
        Ok(risk_measure)
    }

    fn analyze_period_slippage(
        &self,
        period_records: &[ExecutionRecord],
        benchmark_price: FixedPoint,
        start_time: u64,
        end_time: u64,
    ) -> Result<SlippagePeriod, ExecutionError> {
        let total_volume: u64 = period_records.iter().map(|r| r.executed_volume).sum();
        let weighted_price: FixedPoint = period_records.iter()
            .map(|r| r.execution_price * FixedPoint::from_integer(r.executed_volume as i64))
            .fold(FixedPoint::ZERO, |acc, x| acc + x) / 
            FixedPoint::from_integer(total_volume as i64);

        let slippage = weighted_price - benchmark_price;

        Ok(SlippagePeriod {
            start_time,
            end_time,
            slippage,
            volume_executed: total_volume,
            market_conditions: MarketConditions {
                volatility_regime: super::VolatilityRegime::Normal,
                liquidity_level: super::LiquidityLevel::Normal,
                trend_direction: super::TrendDirection::Sideways,
                market_hours: super::MarketHours::Regular,
                news_impact: super::NewsImpact::None,
            },
        })
    }

    fn update_model_performance(&mut self) -> Result<(), ExecutionError> {
        if self.impact_history.len() < 10 {
            return Ok(());
        }

        let recent_observations: Vec<_> = self.impact_history.iter()
            .rev()
            .take(100)
            .collect();

        let mut total_error = 0.0;
        let mut total_squared_error = 0.0;
        let mut correct_direction = 0;

        for obs in &recent_observations {
            let error = (obs.observed_impact - obs.predicted_impact).to_float();
            total_error += error.abs();
            total_squared_error += error * error;

            // Check directional accuracy
            if (obs.observed_impact > FixedPoint::ZERO) == (obs.predicted_impact > FixedPoint::ZERO) {
                correct_direction += 1;
            }
        }

        let n = recent_observations.len() as f64;
        self.model_performance.mean_absolute_error = FixedPoint::from_float(total_error / n);
        self.model_performance.root_mean_square_error = FixedPoint::from_float((total_squared_error / n).sqrt());
        self.model_performance.directional_accuracy = FixedPoint::from_float(correct_direction as f64 / n);

        Ok(())
    }
}

impl Default for MarketImpactModel {
    fn default() -> Self {
        Self::new(MarketImpactParams::default())
    }
}

/// Execution record for slippage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub timestamp: u64,
    pub executed_volume: u64,
    pub execution_price: FixedPoint,
    pub participation_rate: FixedPoint,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{VolatilityRegime, LiquidityLevel, TrendDirection, MarketHours, NewsImpact};

    fn create_test_market_state() -> MarketState {
        MarketState {
            symbol: "AAPL".to_string(),
            bid_price: FixedPoint::from_float(150.0),
            ask_price: FixedPoint::from_float(150.1),
            bid_volume: 1000,
            ask_volume: 1000,
            last_price: FixedPoint::from_float(150.05),
            last_volume: 500,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            volatility: FixedPoint::from_float(0.02),
            average_daily_volume: 1000000,
        }
    }

    fn create_test_market_conditions() -> MarketConditions {
        MarketConditions {
            volatility_regime: VolatilityRegime::Normal,
            liquidity_level: LiquidityLevel::Normal,
            trend_direction: TrendDirection::Sideways,
            market_hours: MarketHours::Regular,
            news_impact: NewsImpact::None,
        }
    }

    #[test]
    fn test_market_impact_model_creation() {
        let model = MarketImpactModel::default();
        assert!(model.impact_history.is_empty());
    }

    #[test]
    fn test_market_impact_estimation() {
        let model = MarketImpactModel::default();
        let market_state = create_test_market_state();
        let market_conditions = create_test_market_conditions();

        let estimate = model.estimate_market_impact(
            1000,
            FixedPoint::from_float(0.1),
            &market_state,
            &market_conditions,
            300,
        ).unwrap();

        assert!(estimate.total_impact >= FixedPoint::ZERO);
        assert!(estimate.temporary_impact >= FixedPoint::ZERO);
        assert!(estimate.permanent_impact >= FixedPoint::ZERO);
        assert!(estimate.confidence_interval.0 <= estimate.total_impact);
        assert!(estimate.confidence_interval.1 >= estimate.total_impact);
    }

    #[test]
    fn test_optimal_execution_trajectory() {
        let model = MarketImpactModel::default();
        let market_state = create_test_market_state();
        let market_conditions = create_test_market_conditions();

        let trajectory = model.calculate_optimal_trajectory(
            10000,
            3600,
            &market_state,
            &market_conditions,
            FixedPoint::from_float(0.01),
        ).unwrap();

        assert!(!trajectory.time_points.is_empty());
        assert_eq!(trajectory.time_points.len(), trajectory.execution_rates.len());
        assert_eq!(trajectory.time_points.len(), trajectory.expected_costs.len());
        assert!(trajectory.total_expected_cost >= FixedPoint::ZERO);
    }

    #[test]
    fn test_slippage_analysis() {
        let model = MarketImpactModel::default();
        let benchmark_price = FixedPoint::from_float(150.0);
        
        let execution_records = vec![
            ExecutionRecord {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                executed_volume: 100,
                execution_price: FixedPoint::from_float(150.05),
                participation_rate: FixedPoint::from_float(0.1),
            },
            ExecutionRecord {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() + 300,
                executed_volume: 200,
                execution_price: FixedPoint::from_float(150.08),
                participation_rate: FixedPoint::from_float(0.15),
            },
        ];

        let market_states = vec![create_test_market_state(); 2];

        let analysis = model.analyze_slippage(
            &execution_records,
            benchmark_price,
            &market_states,
        ).unwrap();

        assert!(analysis.total_slippage >= FixedPoint::ZERO);
        assert!(!analysis.time_period_attribution.is_empty());
    }

    #[test]
    fn test_execution_cost_breakdown() {
        let model = MarketImpactModel::default();
        let benchmark_price = FixedPoint::from_float(150.0);
        let transaction_cost_rate = FixedPoint::from_float(5.0); // 5 bps
        
        let execution_records = vec![
            ExecutionRecord {
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                executed_volume: 100,
                execution_price: FixedPoint::from_float(150.05),
                participation_rate: FixedPoint::from_float(0.1),
            },
        ];

        let cost_breakdown = model.calculate_execution_costs(
            &execution_records,
            benchmark_price,
            transaction_cost_rate,
        ).unwrap();

        assert!(cost_breakdown.total_cost >= FixedPoint::ZERO);
        assert!(cost_breakdown.market_impact_cost >= FixedPoint::ZERO);
        assert!(cost_breakdown.transaction_cost >= FixedPoint::ZERO);
        assert!(cost_breakdown.cost_basis_points >= FixedPoint::ZERO);
    }

    #[test]
    fn test_impact_observation_recording() {
        let mut model = MarketImpactModel::default();
        let market_state = create_test_market_state();
        let market_conditions = create_test_market_conditions();

        let result = model.record_impact_observation(
            1000,
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(0.05),
            market_state,
            market_conditions,
        );

        assert!(result.is_ok());
        assert_eq!(model.impact_history.len(), 1);
    }
}