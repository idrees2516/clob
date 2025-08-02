//! TWAP (Time-Weighted Average Price) Execution Engine
//! 
//! Implements sophisticated TWAP execution with:
//! - Configurable time bucketing
//! - Equal time interval division with volume allocation
//! - Execution rate calculation: v(t) = X/(T-t₀)
//! - Basic participation rate targeting

use super::{ExecutionError, Order, MarketState, MarketConditions, ExecutionMetrics, VolumePatterns};
use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// TWAP execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPConfig {
    /// Total execution horizon in seconds
    pub execution_horizon: u64,
    /// Number of time buckets to divide execution into
    pub num_buckets: usize,
    /// Target participation rate (0.0 to 1.0)
    pub target_participation_rate: FixedPoint,
    /// Maximum participation rate limit
    pub max_participation_rate: FixedPoint,
    /// Minimum order size
    pub min_order_size: u64,
    /// Maximum order size
    pub max_order_size: u64,
    /// Enable adaptive scheduling
    pub adaptive_scheduling: bool,
    /// Randomization factor for timing (0.0 to 1.0)
    pub timing_randomization: FixedPoint,
}

impl Default for TWAPConfig {
    fn default() -> Self {
        Self {
            execution_horizon: 3600, // 1 hour
            num_buckets: 60,         // 1-minute buckets
            target_participation_rate: FixedPoint::from_float(0.1), // 10%
            max_participation_rate: FixedPoint::from_float(0.25),   // 25%
            min_order_size: 100,
            max_order_size: 10000,
            adaptive_scheduling: true,
            timing_randomization: FixedPoint::from_float(0.1), // 10%
        }
    }
}

/// Time bucket for TWAP execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeBucket {
    /// Bucket index
    pub index: usize,
    /// Start time of bucket (Unix timestamp)
    pub start_time: u64,
    /// End time of bucket (Unix timestamp)
    pub end_time: u64,
    /// Target volume for this bucket
    pub target_volume: u64,
    /// Executed volume in this bucket
    pub executed_volume: u64,
    /// Target participation rate for this bucket
    pub participation_rate: FixedPoint,
    /// Bucket status
    pub status: BucketStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BucketStatus {
    Pending,
    Active,
    Completed,
    Skipped,
}

/// TWAP execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPExecutionPlan {
    /// Original order being executed
    pub order: Order,
    /// Configuration used
    pub config: TWAPConfig,
    /// Time buckets for execution
    pub time_buckets: Vec<TimeBucket>,
    /// Total remaining quantity
    pub remaining_quantity: u64,
    /// Execution start time
    pub start_time: u64,
    /// Expected completion time
    pub expected_completion_time: u64,
    /// Current bucket index
    pub current_bucket_index: usize,
}

/// TWAP execution decision for current time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionDecision {
    /// Whether to execute now
    pub should_execute: bool,
    /// Recommended order size
    pub order_size: u64,
    /// Recommended participation rate
    pub participation_rate: FixedPoint,
    /// Execution urgency (0.0 to 1.0)
    pub urgency: FixedPoint,
    /// Reason for decision
    pub reason: String,
}

/// TWAP Executor implementation
#[derive(Debug)]
pub struct TWAPExecutor {
    /// Current execution plan
    execution_plan: Option<TWAPExecutionPlan>,
    /// Execution history
    execution_history: VecDeque<ExecutionRecord>,
    /// Performance metrics
    metrics: ExecutionMetrics,
    /// Volume patterns for forecasting
    volume_patterns: Option<VolumePatterns>,
}

/// Record of execution activity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub timestamp: u64,
    pub bucket_index: usize,
    pub executed_volume: u64,
    pub execution_price: FixedPoint,
    pub market_volume: u64,
    pub participation_rate: FixedPoint,
}

impl TWAPExecutor {
    /// Create new TWAP executor
    pub fn new() -> Self {
        Self {
            execution_plan: None,
            execution_history: VecDeque::new(),
            metrics: ExecutionMetrics {
                total_executed: 0,
                average_price: FixedPoint::ZERO,
                execution_time: 0,
                implementation_shortfall: FixedPoint::ZERO,
                tracking_error: FixedPoint::ZERO,
                market_impact: FixedPoint::ZERO,
                timing_cost: FixedPoint::ZERO,
                opportunity_cost: FixedPoint::ZERO,
                slippage: FixedPoint::ZERO,
            },
            volume_patterns: None,
        }
    }

    /// Create TWAP execution plan
    pub fn create_execution_plan(
        &mut self,
        order: Order,
        config: TWAPConfig,
        market_conditions: &MarketConditions,
        historical_patterns: Option<&VolumePatterns>,
    ) -> Result<TWAPExecutionPlan, ExecutionError> {
        // Validate inputs
        if order.quantity == 0 {
            return Err(ExecutionError::InvalidParameters(
                "Order quantity must be greater than zero".to_string()
            ));
        }

        if config.execution_horizon == 0 {
            return Err(ExecutionError::InvalidParameters(
                "Execution horizon must be greater than zero".to_string()
            ));
        }

        if config.num_buckets == 0 {
            return Err(ExecutionError::InvalidParameters(
                "Number of buckets must be greater than zero".to_string()
            ));
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate bucket duration
        let bucket_duration = config.execution_horizon / config.num_buckets as u64;
        
        // Create time buckets with equal time intervals
        let mut time_buckets = Vec::with_capacity(config.num_buckets);
        let base_volume_per_bucket = order.quantity / config.num_buckets as u64;
        let remainder = order.quantity % config.num_buckets as u64;

        for i in 0..config.num_buckets {
            let start_time = current_time + (i as u64 * bucket_duration);
            let end_time = start_time + bucket_duration;
            
            // Distribute remainder across first few buckets
            let target_volume = if i < remainder as usize {
                base_volume_per_bucket + 1
            } else {
                base_volume_per_bucket
            };

            // Apply volume pattern adjustments if available
            let adjusted_volume = if let Some(patterns) = historical_patterns {
                let hour = ((start_time / 3600) % 24) as u8;
                let multiplier = patterns.get_hourly_multiplier(hour);
                ((target_volume as f64) * multiplier.to_float()) as u64
            } else {
                target_volume
            };

            time_buckets.push(TimeBucket {
                index: i,
                start_time,
                end_time,
                target_volume: adjusted_volume,
                executed_volume: 0,
                participation_rate: config.target_participation_rate,
                status: BucketStatus::Pending,
            });
        }

        // Normalize volumes to ensure total equals order quantity
        let total_target: u64 = time_buckets.iter().map(|b| b.target_volume).sum();
        if total_target != order.quantity {
            let adjustment_factor = order.quantity as f64 / total_target as f64;
            let mut adjusted_total = 0u64;
            
            for bucket in &mut time_buckets[..config.num_buckets - 1] {
                bucket.target_volume = ((bucket.target_volume as f64) * adjustment_factor) as u64;
                adjusted_total += bucket.target_volume;
            }
            
            // Assign remainder to last bucket
            time_buckets.last_mut().unwrap().target_volume = order.quantity - adjusted_total;
        }

        let execution_plan = TWAPExecutionPlan {
            order: order.clone(),
            config,
            time_buckets,
            remaining_quantity: order.quantity,
            start_time: current_time,
            expected_completion_time: current_time + config.execution_horizon,
            current_bucket_index: 0,
        };

        self.execution_plan = Some(execution_plan.clone());
        self.volume_patterns = historical_patterns.cloned();

        Ok(execution_plan)
    }

    /// Get execution decision for current time
    pub fn execute_next_slice(
        &mut self,
        current_time: u64,
        market_state: &MarketState,
    ) -> Result<ExecutionDecision, ExecutionError> {
        let plan = self.execution_plan.as_mut()
            .ok_or_else(|| ExecutionError::ExecutionControl(
                "No execution plan available".to_string()
            ))?;

        // Find current active bucket
        let current_bucket_index = self.find_current_bucket(current_time, plan)?;
        plan.current_bucket_index = current_bucket_index;

        if current_bucket_index >= plan.time_buckets.len() {
            return Ok(ExecutionDecision {
                should_execute: false,
                order_size: 0,
                participation_rate: FixedPoint::ZERO,
                urgency: FixedPoint::ZERO,
                reason: "Execution plan completed".to_string(),
            });
        }

        let current_bucket = &mut plan.time_buckets[current_bucket_index];
        current_bucket.status = BucketStatus::Active;

        // Calculate execution rate: v(t) = X/(T-t₀)
        let time_remaining = plan.expected_completion_time.saturating_sub(current_time);
        if time_remaining == 0 {
            return Ok(ExecutionDecision {
                should_execute: true,
                order_size: plan.remaining_quantity.min(plan.config.max_order_size),
                participation_rate: plan.config.max_participation_rate,
                urgency: FixedPoint::from_integer(1),
                reason: "Final execution - time expired".to_string(),
            });
        }

        let execution_rate = FixedPoint::from_integer(plan.remaining_quantity as i64) / 
                           FixedPoint::from_integer(time_remaining as i64);

        // Calculate bucket progress
        let bucket_elapsed = current_time.saturating_sub(current_bucket.start_time);
        let bucket_duration = current_bucket.end_time - current_bucket.start_time;
        let bucket_progress = if bucket_duration > 0 {
            FixedPoint::from_integer(bucket_elapsed as i64) / 
            FixedPoint::from_integer(bucket_duration as i64)
        } else {
            FixedPoint::from_integer(1)
        };

        // Calculate target execution for this bucket
        let bucket_target_remaining = current_bucket.target_volume - current_bucket.executed_volume;
        let bucket_time_remaining = current_bucket.end_time.saturating_sub(current_time);

        // Determine order size based on participation rate and market volume
        let estimated_market_volume = self.estimate_market_volume(market_state, bucket_time_remaining);
        let participation_volume = (estimated_market_volume.to_float() * 
                                  current_bucket.participation_rate.to_float()) as u64;

        let order_size = bucket_target_remaining
            .min(participation_volume)
            .min(plan.config.max_order_size)
            .max(plan.config.min_order_size);

        // Calculate urgency based on schedule adherence
        let schedule_adherence = self.calculate_schedule_adherence(plan, current_time);
        let urgency = if schedule_adherence < FixedPoint::from_float(0.8) {
            FixedPoint::from_float(0.8) // High urgency if behind schedule
        } else if schedule_adherence > FixedPoint::from_float(1.2) {
            FixedPoint::from_float(0.3) // Low urgency if ahead of schedule
        } else {
            FixedPoint::from_float(0.5) // Normal urgency
        };

        let should_execute = order_size >= plan.config.min_order_size && 
                           bucket_target_remaining > 0;

        Ok(ExecutionDecision {
            should_execute,
            order_size,
            participation_rate: current_bucket.participation_rate,
            urgency,
            reason: format!(
                "Bucket {}: target={}, remaining={}, participation_rate={:.2}%",
                current_bucket_index,
                bucket_target_remaining,
                plan.remaining_quantity,
                current_bucket.participation_rate.to_float() * 100.0
            ),
        })
    }

    /// Record execution activity
    pub fn record_execution(
        &mut self,
        executed_volume: u64,
        execution_price: FixedPoint,
        market_volume: u64,
        timestamp: u64,
    ) -> Result<(), ExecutionError> {
        let plan = self.execution_plan.as_mut()
            .ok_or_else(|| ExecutionError::ExecutionControl(
                "No execution plan available".to_string()
            ))?;

        if plan.current_bucket_index >= plan.time_buckets.len() {
            return Err(ExecutionError::ExecutionControl(
                "Invalid bucket index".to_string()
            ));
        }

        // Update current bucket
        let current_bucket = &mut plan.time_buckets[plan.current_bucket_index];
        current_bucket.executed_volume += executed_volume;
        
        // Update plan totals
        plan.remaining_quantity = plan.remaining_quantity.saturating_sub(executed_volume);

        // Calculate participation rate
        let participation_rate = if market_volume > 0 {
            FixedPoint::from_integer(executed_volume as i64) / 
            FixedPoint::from_integer(market_volume as i64)
        } else {
            FixedPoint::ZERO
        };

        // Record execution
        let record = ExecutionRecord {
            timestamp,
            bucket_index: plan.current_bucket_index,
            executed_volume,
            execution_price,
            market_volume,
            participation_rate,
        };

        self.execution_history.push_back(record);

        // Update metrics
        self.update_metrics(executed_volume, execution_price);

        // Mark bucket as completed if target reached
        if current_bucket.executed_volume >= current_bucket.target_volume {
            current_bucket.status = BucketStatus::Completed;
        }

        Ok(())
    }

    /// Calculate execution performance metrics
    pub fn calculate_performance_metrics(&self, benchmark_price: FixedPoint) -> ExecutionMetrics {
        let mut metrics = self.metrics.clone();
        
        if !self.execution_history.is_empty() {
            // Calculate implementation shortfall
            let total_executed = metrics.total_executed;
            if total_executed > 0 {
                let price_diff = metrics.average_price - benchmark_price;
                metrics.implementation_shortfall = price_diff * 
                    FixedPoint::from_integer(total_executed as i64);
            }

            // Calculate tracking error (standard deviation of execution prices)
            if self.execution_history.len() > 1 {
                let prices: Vec<f64> = self.execution_history.iter()
                    .map(|r| r.execution_price.to_float())
                    .collect();
                
                let mean_price = prices.iter().sum::<f64>() / prices.len() as f64;
                let variance = prices.iter()
                    .map(|&p| (p - mean_price).powi(2))
                    .sum::<f64>() / (prices.len() - 1) as f64;
                
                metrics.tracking_error = FixedPoint::from_float(variance.sqrt());
            }
        }

        metrics
    }

    /// Get current execution status
    pub fn get_execution_status(&self) -> Option<ExecutionStatus> {
        self.execution_plan.as_ref().map(|plan| {
            let completed_buckets = plan.time_buckets.iter()
                .filter(|b| b.status == BucketStatus::Completed)
                .count();
            
            let total_executed = plan.order.quantity - plan.remaining_quantity;
            let completion_percentage = if plan.order.quantity > 0 {
                (total_executed as f64 / plan.order.quantity as f64) * 100.0
            } else {
                0.0
            };

            ExecutionStatus {
                total_quantity: plan.order.quantity,
                executed_quantity: total_executed,
                remaining_quantity: plan.remaining_quantity,
                completion_percentage,
                current_bucket_index: plan.current_bucket_index,
                total_buckets: plan.time_buckets.len(),
                completed_buckets,
                start_time: plan.start_time,
                expected_completion_time: plan.expected_completion_time,
            }
        })
    }

    // Private helper methods

    fn find_current_bucket(&self, current_time: u64, plan: &TWAPExecutionPlan) -> Result<usize, ExecutionError> {
        for (index, bucket) in plan.time_buckets.iter().enumerate() {
            if current_time >= bucket.start_time && current_time < bucket.end_time {
                return Ok(index);
            }
        }
        
        // If past all buckets, return last bucket index
        if current_time >= plan.expected_completion_time {
            Ok(plan.time_buckets.len())
        } else {
            Ok(0)
        }
    }

    fn estimate_market_volume(&self, market_state: &MarketState, time_window: u64) -> FixedPoint {
        // Simple estimation based on average daily volume
        let seconds_per_day = 24 * 60 * 60;
        let volume_per_second = market_state.average_daily_volume as f64 / seconds_per_day as f64;
        let estimated_volume = volume_per_second * time_window as f64;
        
        FixedPoint::from_float(estimated_volume)
    }

    fn calculate_schedule_adherence(&self, plan: &TWAPExecutionPlan, current_time: u64) -> FixedPoint {
        let elapsed_time = current_time.saturating_sub(plan.start_time);
        let total_time = plan.expected_completion_time - plan.start_time;
        
        if total_time == 0 {
            return FixedPoint::from_integer(1);
        }

        let expected_progress = FixedPoint::from_integer(elapsed_time as i64) / 
                              FixedPoint::from_integer(total_time as i64);
        
        let actual_progress = if plan.order.quantity > 0 {
            FixedPoint::from_integer((plan.order.quantity - plan.remaining_quantity) as i64) / 
            FixedPoint::from_integer(plan.order.quantity as i64)
        } else {
            FixedPoint::ZERO
        };

        if expected_progress > FixedPoint::ZERO {
            actual_progress / expected_progress
        } else {
            FixedPoint::from_integer(1)
        }
    }

    fn update_metrics(&mut self, executed_volume: u64, execution_price: FixedPoint) {
        let old_total = self.metrics.total_executed;
        let new_total = old_total + executed_volume;
        
        // Update weighted average price
        if new_total > 0 {
            let old_weight = FixedPoint::from_integer(old_total as i64);
            let new_weight = FixedPoint::from_integer(executed_volume as i64);
            let total_weight = FixedPoint::from_integer(new_total as i64);
            
            self.metrics.average_price = (self.metrics.average_price * old_weight + 
                                        execution_price * new_weight) / total_weight;
        }
        
        self.metrics.total_executed = new_total;
    }
}

impl Default for TWAPExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatus {
    pub total_quantity: u64,
    pub executed_quantity: u64,
    pub remaining_quantity: u64,
    pub completion_percentage: f64,
    pub current_bucket_index: usize,
    pub total_buckets: usize,
    pub completed_buckets: usize,
    pub start_time: u64,
    pub expected_completion_time: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{OrderSide, OrderType, TimeInForce};

    fn create_test_order() -> Order {
        Order {
            id: 1,
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 10000,
            price: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            time_in_force: TimeInForce::Day,
        }
    }

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

    #[test]
    fn test_twap_executor_creation() {
        let executor = TWAPExecutor::new();
        assert!(executor.execution_plan.is_none());
        assert_eq!(executor.execution_history.len(), 0);
    }

    #[test]
    fn test_execution_plan_creation() {
        let mut executor = TWAPExecutor::new();
        let order = create_test_order();
        let config = TWAPConfig::default();
        let market_conditions = MarketConditions {
            volatility_regime: super::super::VolatilityRegime::Normal,
            liquidity_level: super::super::LiquidityLevel::Normal,
            trend_direction: super::super::TrendDirection::Sideways,
            market_hours: super::super::MarketHours::Regular,
            news_impact: super::super::NewsImpact::None,
        };

        let plan = executor.create_execution_plan(
            order.clone(),
            config.clone(),
            &market_conditions,
            None,
        ).unwrap();

        assert_eq!(plan.order.quantity, order.quantity);
        assert_eq!(plan.time_buckets.len(), config.num_buckets);
        assert_eq!(plan.remaining_quantity, order.quantity);
        
        // Verify total target volume equals order quantity
        let total_target: u64 = plan.time_buckets.iter().map(|b| b.target_volume).sum();
        assert_eq!(total_target, order.quantity);
    }

    #[test]
    fn test_execution_decision() {
        let mut executor = TWAPExecutor::new();
        let order = create_test_order();
        let config = TWAPConfig::default();
        let market_conditions = MarketConditions {
            volatility_regime: super::super::VolatilityRegime::Normal,
            liquidity_level: super::super::LiquidityLevel::Normal,
            trend_direction: super::super::TrendDirection::Sideways,
            market_hours: super::super::MarketHours::Regular,
            news_impact: super::super::NewsImpact::None,
        };
        let market_state = create_test_market_state();

        executor.create_execution_plan(order, config, &market_conditions, None).unwrap();
        
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let decision = executor.execute_next_slice(current_time, &market_state).unwrap();

        assert!(decision.order_size > 0);
        assert!(decision.participation_rate > FixedPoint::ZERO);
    }

    #[test]
    fn test_execution_recording() {
        let mut executor = TWAPExecutor::new();
        let order = create_test_order();
        let config = TWAPConfig::default();
        let market_conditions = MarketConditions {
            volatility_regime: super::super::VolatilityRegime::Normal,
            liquidity_level: super::super::LiquidityLevel::Normal,
            trend_direction: super::super::TrendDirection::Sideways,
            market_hours: super::super::MarketHours::Regular,
            news_impact: super::super::NewsImpact::None,
        };

        executor.create_execution_plan(order, config, &market_conditions, None).unwrap();
        
        let executed_volume = 100;
        let execution_price = FixedPoint::from_float(150.05);
        let market_volume = 1000;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        executor.record_execution(executed_volume, execution_price, market_volume, timestamp).unwrap();

        assert_eq!(executor.execution_history.len(), 1);
        assert_eq!(executor.metrics.total_executed, executed_volume);
        
        let status = executor.get_execution_status().unwrap();
        assert_eq!(status.executed_quantity, executed_volume);
        assert_eq!(status.remaining_quantity, 10000 - executed_volume);
    }

    #[test]
    fn test_invalid_parameters() {
        let mut executor = TWAPExecutor::new();
        let mut order = create_test_order();
        order.quantity = 0; // Invalid quantity
        
        let config = TWAPConfig::default();
        let market_conditions = MarketConditions {
            volatility_regime: super::super::VolatilityRegime::Normal,
            liquidity_level: super::super::LiquidityLevel::Normal,
            trend_direction: super::super::TrendDirection::Sideways,
            market_hours: super::super::MarketHours::Regular,
            news_impact: super::super::NewsImpact::None,
        };

        let result = executor.create_execution_plan(order, config, &market_conditions, None);
        assert!(result.is_err());
    }
}