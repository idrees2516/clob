//! Adaptive Execution Control System
//! 
//! Implements sophisticated execution control with:
//! - Catch-up mechanisms for execution shortfall
//! - Slowdown logic for execution surplus
//! - Market condition adaptation (volatility, liquidity changes)
//! - Contingency planning for adverse conditions

use super::{ExecutionError, MarketState, MarketConditions, ExecutionMetrics, Order};
use super::twap::{TWAPExecutor, TWAPConfig, ExecutionDecision, ExecutionStatus};
use super::volume_forecasting::VolumeForecaster;
use super::market_impact::{MarketImpactModel, MarketImpactParams};
use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Adaptive execution control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveControlConfig {
    /// Shortfall threshold for catch-up (percentage)
    pub shortfall_threshold: FixedPoint,
    /// Surplus threshold for slowdown (percentage)
    pub surplus_threshold: FixedPoint,
    /// Maximum catch-up acceleration factor
    pub max_catchup_factor: FixedPoint,
    /// Maximum slowdown factor
    pub max_slowdown_factor: FixedPoint,
    /// Volatility adaptation sensitivity
    pub volatility_sensitivity: FixedPoint,
    /// Liquidity adaptation sensitivity
    pub liquidity_sensitivity: FixedPoint,
    /// News impact reaction time (seconds)
    pub news_reaction_time: u64,
    /// Risk limit breach reaction time (seconds)
    pub risk_reaction_time: u64,
    /// Minimum execution rate (to avoid market timing risk)
    pub min_execution_rate: FixedPoint,
    /// Maximum execution rate (to avoid excessive impact)
    pub max_execution_rate: FixedPoint,
}

impl Default for AdaptiveControlConfig {
    fn default() -> Self {
        Self {
            shortfall_threshold: FixedPoint::from_float(0.05), // 5%
            surplus_threshold: FixedPoint::from_float(0.05),   // 5%
            max_catchup_factor: FixedPoint::from_float(2.0),   // 2x acceleration
            max_slowdown_factor: FixedPoint::from_float(0.5),  // 0.5x slowdown
            volatility_sensitivity: FixedPoint::from_float(0.3),
            liquidity_sensitivity: FixedPoint::from_float(0.4),
            news_reaction_time: 300,  // 5 minutes
            risk_reaction_time: 60,   // 1 minute
            min_execution_rate: FixedPoint::from_float(0.01), // 1% of remaining
            max_execution_rate: FixedPoint::from_float(0.5),  // 50% of remaining
        }
    }
}

/// Execution control state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionControlState {
    /// Current execution mode
    pub mode: ExecutionMode,
    /// Schedule adherence (1.0 = on schedule, <1.0 = behind, >1.0 = ahead)
    pub schedule_adherence: FixedPoint,
    /// Current adaptation factors
    pub adaptation_factors: AdaptationFactors,
    /// Active contingency plans
    pub active_contingencies: Vec<ContingencyPlan>,
    /// Last control update time
    pub last_update: u64,
}

/// Execution modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Normal,
    CatchUp,
    SlowDown,
    Pause,
    Emergency,
    Contingency,
}

/// Adaptation factors for different market conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationFactors {
    /// Volatility adaptation factor
    pub volatility_factor: FixedPoint,
    /// Liquidity adaptation factor
    pub liquidity_factor: FixedPoint,
    /// News impact factor
    pub news_factor: FixedPoint,
    /// Trend adaptation factor
    pub trend_factor: FixedPoint,
    /// Time-of-day factor
    pub time_factor: FixedPoint,
    /// Combined adaptation factor
    pub combined_factor: FixedPoint,
}

/// Contingency plan for adverse conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContingencyPlan {
    /// Plan identifier
    pub id: String,
    /// Trigger condition
    pub trigger: ContingencyTrigger,
    /// Action to take
    pub action: ContingencyAction,
    /// Plan priority (higher = more important)
    pub priority: u32,
    /// Activation time
    pub activation_time: Option<u64>,
    /// Expiration time
    pub expiration_time: Option<u64>,
    /// Plan status
    pub status: ContingencyStatus,
}

/// Contingency trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContingencyTrigger {
    VolatilitySpike { threshold: FixedPoint },
    LiquidityDrop { threshold: FixedPoint },
    NewsEvent { impact_level: super::NewsImpact },
    PriceMove { threshold: FixedPoint, direction: PriceMoveDirection },
    ExecutionShortfall { threshold: FixedPoint },
    RiskLimitBreach { limit_type: RiskLimitType },
    MarketClose { minutes_before: u32 },
    SystemError { error_type: String },
}

/// Price move direction for contingency triggers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceMoveDirection {
    Up,
    Down,
    Either,
}

/// Risk limit types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLimitType {
    Position,
    Loss,
    Concentration,
    Leverage,
}

/// Contingency actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContingencyAction {
    PauseExecution { duration: u64 },
    ReduceRate { factor: FixedPoint },
    IncreaseRate { factor: FixedPoint },
    SwitchToMarketOrders,
    CancelRemaining,
    HedgePosition { hedge_ratio: FixedPoint },
    NotifyOperator { message: String },
    EmergencyLiquidation,
}

/// Contingency plan status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContingencyStatus {
    Inactive,
    Active,
    Triggered,
    Completed,
    Expired,
}

/// Adaptive execution controller
#[derive(Debug)]
pub struct AdaptiveExecutionController {
    /// Configuration
    config: AdaptiveControlConfig,
    /// Current control state
    state: ExecutionControlState,
    /// TWAP executor
    twap_executor: TWAPExecutor,
    /// Volume forecaster
    volume_forecaster: VolumeForecaster,
    /// Market impact model
    impact_model: MarketImpactModel,
    /// Execution history
    execution_history: VecDeque<ExecutionEvent>,
    /// Market condition history
    market_condition_history: VecDeque<MarketConditionSnapshot>,
    /// Performance metrics
    performance_metrics: ControllerPerformanceMetrics,
}

/// Execution event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    pub timestamp: u64,
    pub event_type: ExecutionEventType,
    pub execution_mode: ExecutionMode,
    pub adaptation_factors: AdaptationFactors,
    pub market_conditions: MarketConditions,
    pub execution_decision: Option<ExecutionDecision>,
}

/// Types of execution events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEventType {
    ModeChange { from: ExecutionMode, to: ExecutionMode },
    CatchUpActivated { shortfall: FixedPoint },
    SlowDownActivated { surplus: FixedPoint },
    ContingencyTriggered { plan_id: String },
    MarketConditionChange { condition: String },
    ExecutionPaused { reason: String },
    ExecutionResumed,
    EmergencyStop { reason: String },
}

/// Market condition snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditionSnapshot {
    pub timestamp: u64,
    pub market_state: MarketState,
    pub market_conditions: MarketConditions,
    pub volatility_change: FixedPoint,
    pub liquidity_change: FixedPoint,
    pub price_change: FixedPoint,
}

/// Controller performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerPerformanceMetrics {
    pub total_adaptations: u32,
    pub successful_catchups: u32,
    pub successful_slowdowns: u32,
    pub contingency_activations: u32,
    pub average_schedule_adherence: FixedPoint,
    pub adaptation_effectiveness: FixedPoint,
    pub response_time_ms: u64,
}

impl AdaptiveExecutionController {
    /// Create new adaptive execution controller
    pub fn new(
        config: AdaptiveControlConfig,
        twap_config: TWAPConfig,
        impact_params: MarketImpactParams,
    ) -> Self {
        Self {
            config,
            state: ExecutionControlState {
                mode: ExecutionMode::Normal,
                schedule_adherence: FixedPoint::from_integer(1),
                adaptation_factors: AdaptationFactors {
                    volatility_factor: FixedPoint::from_integer(1),
                    liquidity_factor: FixedPoint::from_integer(1),
                    news_factor: FixedPoint::from_integer(1),
                    trend_factor: FixedPoint::from_integer(1),
                    time_factor: FixedPoint::from_integer(1),
                    combined_factor: FixedPoint::from_integer(1),
                },
                active_contingencies: Vec::new(),
                last_update: 0,
            },
            twap_executor: TWAPExecutor::new(),
            volume_forecaster: VolumeForecaster::default(),
            impact_model: MarketImpactModel::new(impact_params),
            execution_history: VecDeque::new(),
            market_condition_history: VecDeque::new(),
            performance_metrics: ControllerPerformanceMetrics {
                total_adaptations: 0,
                successful_catchups: 0,
                successful_slowdowns: 0,
                contingency_activations: 0,
                average_schedule_adherence: FixedPoint::from_integer(1),
                adaptation_effectiveness: FixedPoint::ZERO,
                response_time_ms: 0,
            },
        }
    }

    /// Process market update and adapt execution
    pub fn process_market_update(
        &mut self,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<ExecutionDecision, ExecutionError> {
        let start_time = SystemTime::now();
        let current_time = market_state.timestamp;

        // Record market condition snapshot
        self.record_market_snapshot(market_state, market_conditions)?;

        // Update adaptation factors
        self.update_adaptation_factors(market_state, market_conditions)?;

        // Check for contingency triggers
        self.check_contingency_triggers(market_state, market_conditions)?;

        // Calculate schedule adherence
        self.update_schedule_adherence()?;

        // Determine execution mode
        let new_mode = self.determine_execution_mode(market_state, market_conditions)?;
        if new_mode != self.state.mode {
            self.change_execution_mode(new_mode, market_conditions)?;
        }

        // Get base execution decision from TWAP executor
        let mut execution_decision = self.twap_executor
            .execute_next_slice(current_time, market_state)?;

        // Apply adaptive control adjustments
        execution_decision = self.apply_adaptive_adjustments(
            execution_decision,
            market_state,
            market_conditions,
        )?;

        // Record execution event
        self.record_execution_event(
            ExecutionEventType::MarketConditionChange {
                condition: format!("Mode: {:?}", self.state.mode),
            },
            Some(execution_decision.clone()),
            market_conditions,
        )?;

        // Update performance metrics
        let response_time = start_time.elapsed().unwrap_or_default().as_millis() as u64;
        self.performance_metrics.response_time_ms = response_time;
        self.state.last_update = current_time;

        Ok(execution_decision)
    }

    /// Implement catch-up mechanism for execution shortfall
    pub fn implement_catchup_mechanism(
        &mut self,
        shortfall: FixedPoint,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<ExecutionDecision, ExecutionError> {
        // Calculate catch-up factor based on shortfall severity
        let catchup_factor = self.calculate_catchup_factor(shortfall, market_conditions)?;
        
        // Get base execution decision
        let mut decision = self.twap_executor
            .execute_next_slice(market_state.timestamp, market_state)?;

        // Apply catch-up acceleration
        decision.order_size = ((decision.order_size as f64) * catchup_factor.to_float()) as u64;
        decision.participation_rate = decision.participation_rate * catchup_factor;
        decision.urgency = FixedPoint::from_float(0.8); // High urgency
        decision.reason = format!(
            "Catch-up mode: shortfall={:.2}%, factor={:.2}x",
            shortfall.to_float() * 100.0,
            catchup_factor.to_float()
        );

        // Ensure we don't exceed maximum limits
        decision.participation_rate = decision.participation_rate
            .min(FixedPoint::from_float(0.5)); // Cap at 50%
        
        // Record catch-up activation
        self.record_execution_event(
            ExecutionEventType::CatchUpActivated { shortfall },
            Some(decision.clone()),
            market_conditions,
        )?;

        self.performance_metrics.successful_catchups += 1;

        Ok(decision)
    }

    /// Implement slowdown mechanism for execution surplus
    pub fn implement_slowdown_mechanism(
        &mut self,
        surplus: FixedPoint,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<ExecutionDecision, ExecutionError> {
        // Calculate slowdown factor based on surplus
        let slowdown_factor = self.calculate_slowdown_factor(surplus, market_conditions)?;
        
        // Get base execution decision
        let mut decision = self.twap_executor
            .execute_next_slice(market_state.timestamp, market_state)?;

        // Apply slowdown
        decision.order_size = ((decision.order_size as f64) * slowdown_factor.to_float()) as u64;
        decision.participation_rate = decision.participation_rate * slowdown_factor;
        decision.urgency = FixedPoint::from_float(0.2); // Low urgency
        decision.reason = format!(
            "Slowdown mode: surplus={:.2}%, factor={:.2}x",
            surplus.to_float() * 100.0,
            slowdown_factor.to_float()
        );

        // Ensure minimum execution rate to avoid timing risk
        let min_rate = self.config.min_execution_rate;
        if decision.participation_rate < min_rate {
            decision.participation_rate = min_rate;
            decision.order_size = ((decision.order_size as f64) * 
                                 (min_rate.to_float() / slowdown_factor.to_float())) as u64;
        }

        // Record slowdown activation
        self.record_execution_event(
            ExecutionEventType::SlowDownActivated { surplus },
            Some(decision.clone()),
            market_conditions,
        )?;

        self.performance_metrics.successful_slowdowns += 1;

        Ok(decision)
    }

    /// Add contingency plan
    pub fn add_contingency_plan(&mut self, plan: ContingencyPlan) -> Result<(), ExecutionError> {
        // Validate plan
        if plan.id.is_empty() {
            return Err(ExecutionError::InvalidParameters(
                "Contingency plan ID cannot be empty".to_string()
            ));
        }

        // Check for duplicate IDs
        if self.state.active_contingencies.iter().any(|p| p.id == plan.id) {
            return Err(ExecutionError::InvalidParameters(
                format!("Contingency plan with ID '{}' already exists", plan.id)
            ));
        }

        self.state.active_contingencies.push(plan);
        Ok(())
    }

    /// Remove contingency plan
    pub fn remove_contingency_plan(&mut self, plan_id: &str) -> Result<(), ExecutionError> {
        let initial_len = self.state.active_contingencies.len();
        self.state.active_contingencies.retain(|p| p.id != plan_id);
        
        if self.state.active_contingencies.len() == initial_len {
            return Err(ExecutionError::InvalidParameters(
                format!("Contingency plan with ID '{}' not found", plan_id)
            ));
        }

        Ok(())
    }

    /// Get current execution status with control information
    pub fn get_execution_status_with_control(&self) -> Result<ExecutionStatusWithControl, ExecutionError> {
        let base_status = self.twap_executor.get_execution_status();
        
        Ok(ExecutionStatusWithControl {
            base_status,
            control_state: self.state.clone(),
            performance_metrics: self.performance_metrics.clone(),
            active_contingencies: self.state.active_contingencies.len(),
            recent_adaptations: self.get_recent_adaptations()?,
        })
    }

    // Private helper methods

    fn record_market_snapshot(
        &mut self,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        let current_time = market_state.timestamp;
        
        // Calculate changes from previous snapshot
        let (volatility_change, liquidity_change, price_change) = 
            if let Some(last_snapshot) = self.market_condition_history.back() {
                let vol_change = market_state.volatility - last_snapshot.market_state.volatility;
                let price_change = market_state.mid_price() - last_snapshot.market_state.mid_price();
                // Simplified liquidity change calculation
                let liq_change = FixedPoint::ZERO; // Would calculate based on order book depth
                (vol_change, liq_change, price_change)
            } else {
                (FixedPoint::ZERO, FixedPoint::ZERO, FixedPoint::ZERO)
            };

        let snapshot = MarketConditionSnapshot {
            timestamp: current_time,
            market_state: market_state.clone(),
            market_conditions: market_conditions.clone(),
            volatility_change,
            liquidity_change,
            price_change,
        };

        self.market_condition_history.push_back(snapshot);

        // Maintain history size
        while self.market_condition_history.len() > 1000 {
            self.market_condition_history.pop_front();
        }

        Ok(())
    }

    fn update_adaptation_factors(
        &mut self,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        // Volatility adaptation
        let vol_factor = match market_conditions.volatility_regime {
            super::VolatilityRegime::Low => FixedPoint::from_float(1.1),
            super::VolatilityRegime::High => FixedPoint::from_float(0.8),
            super::VolatilityRegime::Extreme => FixedPoint::from_float(0.5),
            _ => FixedPoint::from_integer(1),
        };

        // Liquidity adaptation
        let liq_factor = match market_conditions.liquidity_level {
            super::LiquidityLevel::Low => FixedPoint::from_float(0.7),
            super::LiquidityLevel::High => FixedPoint::from_float(1.2),
            _ => FixedPoint::from_integer(1),
        };

        // News impact adaptation
        let news_factor = match market_conditions.news_impact {
            super::NewsImpact::High => FixedPoint::from_float(0.5),
            super::NewsImpact::Extreme => FixedPoint::from_float(0.2),
            super::NewsImpact::Medium => FixedPoint::from_float(0.8),
            _ => FixedPoint::from_integer(1),
        };

        // Trend adaptation
        let trend_factor = match market_conditions.trend_direction {
            super::TrendDirection::Strong_Up | super::TrendDirection::Strong_Down => {
                FixedPoint::from_float(0.9) // Slightly reduce in strong trends
            },
            _ => FixedPoint::from_integer(1),
        };

        // Time-of-day adaptation
        let hour = ((market_state.timestamp / 3600) % 24) as u8;
        let time_factor = match hour {
            9..=10 => FixedPoint::from_float(0.8),  // Market open - reduce rate
            15..=16 => FixedPoint::from_float(1.1), // Market close - increase rate
            _ => FixedPoint::from_integer(1),
        };

        // Combined factor
        let combined_factor = vol_factor * liq_factor * news_factor * trend_factor * time_factor;

        self.state.adaptation_factors = AdaptationFactors {
            volatility_factor: vol_factor,
            liquidity_factor: liq_factor,
            news_factor,
            trend_factor,
            time_factor,
            combined_factor,
        };

        Ok(())
    }

    fn check_contingency_triggers(
        &mut self,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        let current_time = market_state.timestamp;
        
        for plan in &mut self.state.active_contingencies {
            if plan.status != ContingencyStatus::Inactive {
                continue;
            }

            let should_trigger = match &plan.trigger {
                ContingencyTrigger::VolatilitySpike { threshold } => {
                    market_state.volatility > *threshold
                },
                ContingencyTrigger::LiquidityDrop { threshold } => {
                    // Simplified liquidity check
                    market_state.spread() / market_state.mid_price() > *threshold
                },
                ContingencyTrigger::NewsEvent { impact_level } => {
                    market_conditions.news_impact >= *impact_level
                },
                ContingencyTrigger::PriceMove { threshold, direction } => {
                    if let Some(last_snapshot) = self.market_condition_history.back() {
                        let price_change = market_state.mid_price() - last_snapshot.market_state.mid_price();
                        let abs_change = if price_change >= FixedPoint::ZERO { price_change } else { -price_change };
                        
                        match direction {
                            PriceMoveDirection::Up => price_change > *threshold,
                            PriceMoveDirection::Down => price_change < -*threshold,
                            PriceMoveDirection::Either => abs_change > *threshold,
                        }
                    } else {
                        false
                    }
                },
                ContingencyTrigger::ExecutionShortfall { threshold } => {
                    self.state.schedule_adherence < (FixedPoint::from_integer(1) - *threshold)
                },
                ContingencyTrigger::RiskLimitBreach { .. } => {
                    // Would check actual risk limits
                    false
                },
                ContingencyTrigger::MarketClose { minutes_before } => {
                    // Would check actual market close time
                    false
                },
                ContingencyTrigger::SystemError { .. } => {
                    // Would check for system errors
                    false
                },
            };

            if should_trigger {
                plan.status = ContingencyStatus::Triggered;
                plan.activation_time = Some(current_time);
                
                self.execute_contingency_action(&plan.action, market_state, market_conditions)?;
                
                self.record_execution_event(
                    ExecutionEventType::ContingencyTriggered { plan_id: plan.id.clone() },
                    None,
                    market_conditions,
                )?;

                self.performance_metrics.contingency_activations += 1;
            }
        }

        Ok(())
    }

    fn execute_contingency_action(
        &mut self,
        action: &ContingencyAction,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        match action {
            ContingencyAction::PauseExecution { duration: _ } => {
                self.state.mode = ExecutionMode::Pause;
            },
            ContingencyAction::ReduceRate { factor } => {
                self.state.adaptation_factors.combined_factor = 
                    self.state.adaptation_factors.combined_factor * *factor;
            },
            ContingencyAction::IncreaseRate { factor } => {
                self.state.adaptation_factors.combined_factor = 
                    self.state.adaptation_factors.combined_factor * *factor;
            },
            ContingencyAction::SwitchToMarketOrders => {
                // Would switch order type to market orders
            },
            ContingencyAction::CancelRemaining => {
                // Would cancel remaining orders
            },
            ContingencyAction::HedgePosition { hedge_ratio: _ } => {
                // Would initiate hedging
            },
            ContingencyAction::NotifyOperator { message: _ } => {
                // Would send notification
            },
            ContingencyAction::EmergencyLiquidation => {
                self.state.mode = ExecutionMode::Emergency;
            },
        }

        Ok(())
    }

    fn update_schedule_adherence(&mut self) -> Result<(), ExecutionError> {
        if let Some(status) = self.twap_executor.get_execution_status() {
            let expected_progress = if status.total_quantity > 0 {
                // Calculate expected progress based on time elapsed
                let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
                let elapsed_time = current_time.saturating_sub(status.start_time);
                let total_time = status.expected_completion_time - status.start_time;
                
                if total_time > 0 {
                    FixedPoint::from_integer(elapsed_time as i64) / 
                    FixedPoint::from_integer(total_time as i64)
                } else {
                    FixedPoint::from_integer(1)
                }
            } else {
                FixedPoint::ZERO
            };

            let actual_progress = FixedPoint::from_float(status.completion_percentage / 100.0);
            
            self.state.schedule_adherence = if expected_progress > FixedPoint::ZERO {
                actual_progress / expected_progress
            } else {
                FixedPoint::from_integer(1)
            };
        }

        Ok(())
    }

    fn determine_execution_mode(
        &self,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<ExecutionMode, ExecutionError> {
        // Check for emergency conditions first
        if market_conditions.news_impact == super::NewsImpact::Extreme {
            return Ok(ExecutionMode::Emergency);
        }

        // Check for contingency mode
        if self.state.active_contingencies.iter()
            .any(|p| p.status == ContingencyStatus::Triggered) {
            return Ok(ExecutionMode::Contingency);
        }

        // Check for pause conditions
        if market_conditions.volatility_regime == super::VolatilityRegime::Extreme ||
           market_conditions.news_impact == super::NewsImpact::High {
            return Ok(ExecutionMode::Pause);
        }

        // Check schedule adherence for catch-up/slowdown
        let shortfall = FixedPoint::from_integer(1) - self.state.schedule_adherence;
        let surplus = self.state.schedule_adherence - FixedPoint::from_integer(1);

        if shortfall > self.config.shortfall_threshold {
            Ok(ExecutionMode::CatchUp)
        } else if surplus > self.config.surplus_threshold {
            Ok(ExecutionMode::SlowDown)
        } else {
            Ok(ExecutionMode::Normal)
        }
    }

    fn change_execution_mode(
        &mut self,
        new_mode: ExecutionMode,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        let old_mode = self.state.mode;
        self.state.mode = new_mode;

        self.record_execution_event(
            ExecutionEventType::ModeChange { from: old_mode, to: new_mode },
            None,
            market_conditions,
        )?;

        self.performance_metrics.total_adaptations += 1;

        Ok(())
    }

    fn apply_adaptive_adjustments(
        &self,
        mut decision: ExecutionDecision,
        market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<ExecutionDecision, ExecutionError> {
        // Apply adaptation factors
        let adjustment_factor = self.state.adaptation_factors.combined_factor;
        
        decision.order_size = ((decision.order_size as f64) * adjustment_factor.to_float()) as u64;
        decision.participation_rate = decision.participation_rate * adjustment_factor;

        // Apply mode-specific adjustments
        match self.state.mode {
            ExecutionMode::CatchUp => {
                decision.urgency = FixedPoint::from_float(0.8);
                decision.order_size = (decision.order_size as f64 * 1.5) as u64; // 50% increase
            },
            ExecutionMode::SlowDown => {
                decision.urgency = FixedPoint::from_float(0.3);
                decision.order_size = (decision.order_size as f64 * 0.7) as u64; // 30% decrease
            },
            ExecutionMode::Pause => {
                decision.should_execute = false;
                decision.order_size = 0;
                decision.reason = "Execution paused due to market conditions".to_string();
            },
            ExecutionMode::Emergency => {
                decision.should_execute = false;
                decision.order_size = 0;
                decision.reason = "Emergency stop activated".to_string();
            },
            _ => {},
        }

        // Ensure limits are respected
        decision.participation_rate = decision.participation_rate
            .max(self.config.min_execution_rate)
            .min(self.config.max_execution_rate);

        Ok(decision)
    }

    fn calculate_catchup_factor(
        &self,
        shortfall: FixedPoint,
        market_conditions: &MarketConditions,
    ) -> Result<FixedPoint, ExecutionError> {
        // Base catch-up factor based on shortfall severity
        let base_factor = FixedPoint::from_integer(1) + shortfall * FixedPoint::from_float(2.0);
        
        // Adjust for market conditions
        let condition_adjustment = match market_conditions.volatility_regime {
            super::VolatilityRegime::High => FixedPoint::from_float(0.8),
            super::VolatilityRegime::Extreme => FixedPoint::from_float(0.6),
            _ => FixedPoint::from_integer(1),
        };

        let adjusted_factor = base_factor * condition_adjustment;
        
        // Cap at maximum
        Ok(adjusted_factor.min(self.config.max_catchup_factor))
    }

    fn calculate_slowdown_factor(
        &self,
        surplus: FixedPoint,
        market_conditions: &MarketConditions,
    ) -> Result<FixedPoint, ExecutionError> {
        // Base slowdown factor based on surplus
        let base_factor = FixedPoint::from_integer(1) - surplus * FixedPoint::from_float(0.5);
        
        // Adjust for market conditions (less slowdown in volatile conditions)
        let condition_adjustment = match market_conditions.volatility_regime {
            super::VolatilityRegime::High => FixedPoint::from_float(1.2),
            super::VolatilityRegime::Extreme => FixedPoint::from_float(1.5),
            _ => FixedPoint::from_integer(1),
        };

        let adjusted_factor = base_factor * condition_adjustment;
        
        // Ensure minimum
        Ok(adjusted_factor.max(self.config.max_slowdown_factor))
    }

    fn record_execution_event(
        &mut self,
        event_type: ExecutionEventType,
        execution_decision: Option<ExecutionDecision>,
        market_conditions: &MarketConditions,
    ) -> Result<(), ExecutionError> {
        let event = ExecutionEvent {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            event_type,
            execution_mode: self.state.mode,
            adaptation_factors: self.state.adaptation_factors.clone(),
            market_conditions: market_conditions.clone(),
            execution_decision,
        };

        self.execution_history.push_back(event);

        // Maintain history size
        while self.execution_history.len() > 1000 {
            self.execution_history.pop_front();
        }

        Ok(())
    }

    fn get_recent_adaptations(&self) -> Result<Vec<ExecutionEvent>, ExecutionError> {
        let recent_events: Vec<_> = self.execution_history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        Ok(recent_events)
    }
}

/// Extended execution status with control information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatusWithControl {
    pub base_status: Option<ExecutionStatus>,
    pub control_state: ExecutionControlState,
    pub performance_metrics: ControllerPerformanceMetrics,
    pub active_contingencies: usize,
    pub recent_adaptations: Vec<ExecutionEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{OrderSide, OrderType, TimeInForce, VolatilityRegime, LiquidityLevel, TrendDirection, MarketHours, NewsImpact};

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
    fn test_adaptive_controller_creation() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        assert_eq!(controller.state.mode, ExecutionMode::Normal);
        assert!(controller.state.active_contingencies.is_empty());
    }

    #[test]
    fn test_contingency_plan_management() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let mut controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        
        let plan = ContingencyPlan {
            id: "test_plan".to_string(),
            trigger: ContingencyTrigger::VolatilitySpike { 
                threshold: FixedPoint::from_float(0.05) 
            },
            action: ContingencyAction::PauseExecution { duration: 300 },
            priority: 1,
            activation_time: None,
            expiration_time: None,
            status: ContingencyStatus::Inactive,
        };

        // Add plan
        let result = controller.add_contingency_plan(plan);
        assert!(result.is_ok());
        assert_eq!(controller.state.active_contingencies.len(), 1);

        // Remove plan
        let result = controller.remove_contingency_plan("test_plan");
        assert!(result.is_ok());
        assert_eq!(controller.state.active_contingencies.len(), 0);
    }

    #[test]
    fn test_market_update_processing() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let mut controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        
        // Initialize TWAP execution plan first
        let order = Order {
            id: 1,
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 10000,
            price: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            time_in_force: TimeInForce::Day,
        };
        
        let market_conditions = create_test_market_conditions();
        let _ = controller.twap_executor.create_execution_plan(
            order,
            TWAPConfig::default(),
            &market_conditions,
            None,
        );

        let market_state = create_test_market_state();
        
        let result = controller.process_market_update(&market_state, &market_conditions);
        assert!(result.is_ok());
        
        let decision = result.unwrap();
        assert!(decision.participation_rate >= FixedPoint::ZERO);
    }

    #[test]
    fn test_catchup_mechanism() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let mut controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        
        // Initialize TWAP execution plan
        let order = Order {
            id: 1,
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 10000,
            price: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            time_in_force: TimeInForce::Day,
        };
        
        let market_conditions = create_test_market_conditions();
        let _ = controller.twap_executor.create_execution_plan(
            order,
            TWAPConfig::default(),
            &market_conditions,
            None,
        );

        let market_state = create_test_market_state();
        let shortfall = FixedPoint::from_float(0.1); // 10% shortfall
        
        let result = controller.implement_catchup_mechanism(
            shortfall,
            &market_state,
            &market_conditions,
        );
        
        assert!(result.is_ok());
        let decision = result.unwrap();
        assert_eq!(decision.urgency, FixedPoint::from_float(0.8));
        assert!(decision.reason.contains("Catch-up mode"));
    }

    #[test]
    fn test_slowdown_mechanism() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let mut controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        
        // Initialize TWAP execution plan
        let order = Order {
            id: 1,
            symbol: "AAPL".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 10000,
            price: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            time_in_force: TimeInForce::Day,
        };
        
        let market_conditions = create_test_market_conditions();
        let _ = controller.twap_executor.create_execution_plan(
            order,
            TWAPConfig::default(),
            &market_conditions,
            None,
        );

        let market_state = create_test_market_state();
        let surplus = FixedPoint::from_float(0.1); // 10% surplus
        
        let result = controller.implement_slowdown_mechanism(
            surplus,
            &market_state,
            &market_conditions,
        );
        
        assert!(result.is_ok());
        let decision = result.unwrap();
        assert_eq!(decision.urgency, FixedPoint::from_float(0.2));
        assert!(decision.reason.contains("Slowdown mode"));
    }

    #[test]
    fn test_execution_status_with_control() {
        let config = AdaptiveControlConfig::default();
        let twap_config = TWAPConfig::default();
        let impact_params = MarketImpactParams::default();
        
        let controller = AdaptiveExecutionController::new(config, twap_config, impact_params);
        
        let result = controller.get_execution_status_with_control();
        assert!(result.is_ok());
        
        let status = result.unwrap();
        assert_eq!(status.control_state.mode, ExecutionMode::Normal);
        assert_eq!(status.active_contingencies, 0);
    }
}