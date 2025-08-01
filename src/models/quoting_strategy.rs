//! High-Frequency Quoting Strategy Engine
//! 
//! This module implements the optimal quoting algorithms from the research paper
//! "High Frequency Quoting Under Liquidity Constraints" with inventory management,
//! risk controls, and adaptive parameters based on market conditions.

use crate::math::{
    fixed_point::{FixedPoint, DeterministicRng},
    optimization::{ModelParameters, HJBSolver, StateGrid, BoundaryConditions, NumericalScheme, OptimalControls},
    hawkes_process::{MultivariateHawkesSimulator, MultivariateHawkesParams, KernelType, HawkesEvent},
    sde_solvers::{GBMJumpState, GBMJumpParams, RoughVolatilityState, RoughVolatilityParams},
};
use crate::orderbook::types::{Price, Volume, Side, OrderId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;
use std::sync::{Arc, Mutex};

#[derive(Error, Debug)]
pub enum QuotingError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Optimization error: {0}")]
    OptimizationError(String),
    #[error("Market data error: {0}")]
    MarketDataError(String),
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    #[error("Insufficient liquidity: {0}")]
    InsufficientLiquidity(String),
}

/// Market state information for quoting decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub mid_price: FixedPoint,
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_volume: FixedPoint,
    pub ask_volume: FixedPoint,
    pub spread: FixedPoint,
    pub volatility: FixedPoint,
    pub timestamp: u64,
    pub order_flow_imbalance: FixedPoint,
    pub market_depth: Vec<(FixedPoint, FixedPoint)>, // (price, volume) pairs
}

/// Trading state including inventory and PnL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingState {
    pub inventory: FixedPoint,
    pub cash: FixedPoint,
    pub unrealized_pnl: FixedPoint,
    pub realized_pnl: FixedPoint,
    pub total_pnl: FixedPoint,
    pub position_value: FixedPoint,
    pub last_trade_time: u64,
    pub trade_count: u64,
    pub max_inventory_reached: FixedPoint,
}

impl TradingState {
    pub fn new() -> Self {
        Self {
            inventory: FixedPoint::zero(),
            cash: FixedPoint::zero(),
            unrealized_pnl: FixedPoint::zero(),
            realized_pnl: FixedPoint::zero(),
            total_pnl: FixedPoint::zero(),
            position_value: FixedPoint::zero(),
            last_trade_time: 0,
            trade_count: 0,
            max_inventory_reached: FixedPoint::zero(),
        }
    }
    
    pub fn update_pnl(&mut self, mid_price: FixedPoint) {
        self.position_value = self.inventory * mid_price;
        self.unrealized_pnl = self.position_value;
        self.total_pnl = self.realized_pnl + self.unrealized_pnl;
    }
}

/// Quote pair (bid and ask)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotePair {
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_size: FixedPoint,
    pub ask_size: FixedPoint,
    pub bid_spread: FixedPoint,
    pub ask_spread: FixedPoint,
    pub confidence: FixedPoint,
    pub timestamp: u64,
}

/// Risk parameters for the quoting strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParameters {
    pub max_inventory: FixedPoint,
    pub max_position_value: FixedPoint,
    pub max_drawdown: FixedPoint,
    pub max_daily_loss: FixedPoint,
    pub position_limit_soft: FixedPoint,
    pub position_limit_hard: FixedPoint,
    pub volatility_threshold: FixedPoint,
    pub spread_threshold_min: FixedPoint,
    pub spread_threshold_max: FixedPoint,
    pub emergency_liquidation_threshold: FixedPoint,
}

impl RiskParameters {
    pub fn default_conservative() -> Self {
        Self {
            max_inventory: FixedPoint::from_float(100.0),
            max_position_value: FixedPoint::from_float(10000.0),
            max_drawdown: FixedPoint::from_float(0.05), // 5%
            max_daily_loss: FixedPoint::from_float(1000.0),
            position_limit_soft: FixedPoint::from_float(80.0),
            position_limit_hard: FixedPoint::from_float(95.0),
            volatility_threshold: FixedPoint::from_float(0.5),
            spread_threshold_min: FixedPoint::from_float(0.0001), // 1 bps
            spread_threshold_max: FixedPoint::from_float(0.01),   // 100 bps
            emergency_liquidation_threshold: FixedPoint::from_float(0.9),
        }
    }
}

/// Optimal bid-ask spread calculator implementing equations (9.18)-(9.21)
pub struct OptimalSpreadCalculator {
    pub model_params: ModelParameters,
    pub hjb_solver: Option<Arc<Mutex<HJBSolver>>>,
    pub optimal_controls: Option<OptimalControls>,
    pub last_calibration_time: u64,
    pub calibration_interval: u64,
}

impl OptimalSpreadCalculator {
    pub fn new(model_params: ModelParameters) -> Self {
        Self {
            model_params,
            hjb_solver: None,
            optimal_controls: None,
            last_calibration_time: 0,
            calibration_interval: 3600, // Recalibrate every hour
        }
    }
    
    /// Initialize HJB solver with state grid
    pub fn initialize_hjb_solver(&mut self, grid_size: (usize, usize, usize)) -> Result<(), QuotingError> {
        // Create state grid for HJB solver
        let inventory_range = (-self.model_params.max_inventory, self.model_params.max_inventory);
        let time_range = (FixedPoint::zero(), self.model_params.terminal_time);
        let price_range = (
            FixedPoint::from_float(50.0),  // Minimum price
            FixedPoint::from_float(200.0), // Maximum price
        );
        
        let grid = StateGrid::new(
            inventory_range,
            grid_size.0,
            time_range,
            grid_size.1,
            price_range,
            grid_size.2,
        );
        
        let boundary_conditions = BoundaryConditions::new();
        let dt = self.model_params.terminal_time / FixedPoint::from_float(grid_size.1 as f64);
        
        let solver = HJBSolver::new(
            grid,
            boundary_conditions,
            NumericalScheme::CrankNicolson,
            dt,
        );
        
        self.hjb_solver = Some(Arc::new(Mutex::new(solver)));
        Ok(())
    }
    
    /// Compute optimal bid-ask spreads using closed-form solutions from equations (9.18)-(9.21)
    pub fn compute_optimal_spreads(
        &mut self,
        market_state: &MarketState,
        trading_state: &TradingState,
        hawkes_intensities: &[FixedPoint],
        current_time: u64,
    ) -> Result<(FixedPoint, FixedPoint), QuotingError> {
        // Check if we need to recalibrate the HJB solution
        if current_time - self.last_calibration_time > self.calibration_interval {
            self.recalibrate_hjb_solution()?;
            self.last_calibration_time = current_time;
        }
        
        // Extract parameters
        let q = trading_state.inventory; // Current inventory
        let S = market_state.mid_price;  // Mid price
        let sigma = market_state.volatility; // Volatility
        let gamma = self.model_params.inventory_penalty; // Risk aversion
        let kappa = self.model_params.adverse_selection_cost; // Adverse selection
        let lambda = self.model_params.market_impact_coefficient; // Market impact
        
        // Time to maturity (simplified - could be dynamic)
        let T = self.model_params.terminal_time;
        let t = FixedPoint::from_float(current_time as f64 / 86400.0); // Convert to days
        let tau = T - t; // Time to maturity
        
        // Hawkes process intensities
        let lambda_buy = if hawkes_intensities.len() > 0 { hawkes_intensities[0] } else { FixedPoint::one() };
        let lambda_sell = if hawkes_intensities.len() > 1 { hawkes_intensities[1] } else { FixedPoint::one() };
        
        // Compute reservation price (inventory-adjusted mid price)
        // From equation (9.18): r = S - q * γ * σ² * τ
        let reservation_price = S - q * gamma * sigma * sigma * tau;
        
        // Compute optimal half-spread using closed-form solution
        // From equation (9.19): δ* = (γσ²τ + 2/γ * ln(1 + γ/κ)) / 2
        let risk_term = gamma * sigma * sigma * tau;
        let adverse_selection_term = FixedPoint::from_float(2.0) / gamma * 
            (FixedPoint::one() + gamma / kappa).ln();
        let base_half_spread = (risk_term + adverse_selection_term) / FixedPoint::from_float(2.0);
        
        // Market impact adjustment from equation (9.20)
        // δ_impact = λ * |q| * σ * √τ
        let impact_adjustment = lambda * q.abs() * sigma * tau.sqrt();
        
        // Liquidity-adjusted spread from equation (9.21)
        // Adjust based on order flow imbalance and market depth
        let imbalance_factor = FixedPoint::one() + market_state.order_flow_imbalance.abs() * FixedPoint::from_float(0.1);
        let depth_factor = if market_state.bid_volume + market_state.ask_volume > FixedPoint::zero() {
            FixedPoint::one() / (FixedPoint::one() + (market_state.bid_volume + market_state.ask_volume) * FixedPoint::from_float(0.01))
        } else {
            FixedPoint::from_float(2.0)
        };
        
        // Final half-spread with all adjustments
        let adjusted_half_spread = base_half_spread * imbalance_factor * depth_factor + impact_adjustment;
        
        // Inventory skew adjustment
        let inventory_skew = q * gamma * sigma * sigma * tau / FixedPoint::from_float(4.0);
        
        // Compute asymmetric spreads
        let bid_spread = adjusted_half_spread + inventory_skew;
        let ask_spread = adjusted_half_spread - inventory_skew;
        
        // Ensure minimum spread
        let min_spread = FixedPoint::from_float(0.0001); // 1 basis point
        let bid_spread = bid_spread.max(min_spread);
        let ask_spread = ask_spread.max(min_spread);
        
        // Apply Hawkes process intensity adjustment
        let intensity_adjustment_bid = FixedPoint::one() / (FixedPoint::one() + lambda_buy * FixedPoint::from_float(0.1));
        let intensity_adjustment_ask = FixedPoint::one() / (FixedPoint::one() + lambda_sell * FixedPoint::from_float(0.1));
        
        let final_bid_spread = bid_spread * intensity_adjustment_bid;
        let final_ask_spread = ask_spread * intensity_adjustment_ask;
        
        Ok((final_bid_spread, final_ask_spread))
    }
    
    /// Recalibrate HJB solution periodically
    fn recalibrate_hjb_solution(&mut self) -> Result<(), QuotingError> {
        if let Some(solver_arc) = &self.hjb_solver {
            let solver = solver_arc.lock().map_err(|e| {
                QuotingError::OptimizationError(format!("Failed to lock HJB solver: {}", e))
            })?;
            
            // Solve HJB equation
            let value_function = solver.solve_value_function(&self.model_params)
                .map_err(|e| QuotingError::OptimizationError(format!("HJB solver failed: {}", e)))?;
            
            // Extract optimal controls
            let controls = solver.compute_optimal_controls(&value_function, &self.model_params)
                .map_err(|e| QuotingError::OptimizationError(format!("Control extraction failed: {}", e)))?;
            
            self.optimal_controls = Some(controls);
        }
        
        Ok(())
    }
    
    /// Get optimal controls from pre-computed solution
    pub fn get_optimal_controls_from_hjb(
        &self,
        inventory: FixedPoint,
        time: FixedPoint,
        price: FixedPoint,
    ) -> Option<(FixedPoint, FixedPoint, FixedPoint)> {
        self.optimal_controls.as_ref().map(|controls| {
            controls.get_controls(inventory, time, price)
        })
    }
}

/// Inventory management system implementing equations (6.14)-(6.17)
pub struct InventoryManager {
    pub risk_params: RiskParameters,
    pub inventory_history: VecDeque<(u64, FixedPoint)>,
    pub pnl_history: VecDeque<(u64, FixedPoint)>,
    pub max_history_size: usize,
    pub emergency_mode: bool,
    pub last_rebalance_time: u64,
}

impl InventoryManager {
    pub fn new(risk_params: RiskParameters, max_history_size: usize) -> Self {
        Self {
            risk_params,
            inventory_history: VecDeque::with_capacity(max_history_size),
            pnl_history: VecDeque::with_capacity(max_history_size),
            max_history_size,
            emergency_mode: false,
            last_rebalance_time: 0,
        }
    }
    
    /// Compute inventory penalty function from equations (6.14)-(6.17)
    pub fn compute_inventory_penalty(
        &self,
        inventory: FixedPoint,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
        gamma: FixedPoint,
    ) -> FixedPoint {
        // Inventory penalty from equation (6.14): Φ(q) = γq²/2
        let quadratic_penalty = gamma * inventory * inventory / FixedPoint::from_float(2.0);
        
        // Time-dependent penalty from equation (6.15): Ψ(q,τ) = γσ²τq²/2
        let time_penalty = gamma * volatility * volatility * time_to_maturity * 
                          inventory * inventory / FixedPoint::from_float(2.0);
        
        // Asymmetric penalty for large positions from equation (6.16)
        let asymmetric_penalty = if inventory.abs() > self.risk_params.position_limit_soft {
            let excess = inventory.abs() - self.risk_params.position_limit_soft;
            gamma * excess * excess * excess / FixedPoint::from_float(6.0) // Cubic penalty
        } else {
            FixedPoint::zero()
        };
        
        // Liquidity penalty from equation (6.17): increases with position size
        let liquidity_penalty = gamma * inventory.abs() * volatility * 
                               (FixedPoint::one() + inventory.abs() / self.risk_params.max_inventory);
        
        quadratic_penalty + time_penalty + asymmetric_penalty + liquidity_penalty
    }
    
    /// Check position limits and enforce constraints
    pub fn check_position_limits(&mut self, trading_state: &TradingState) -> Result<bool, QuotingError> {
        let inventory_abs = trading_state.inventory.abs();
        
        // Hard position limit check
        if inventory_abs >= self.risk_params.position_limit_hard {
            self.emergency_mode = true;
            return Err(QuotingError::RiskLimitExceeded(
                format!("Hard position limit exceeded: {} >= {}", 
                       inventory_abs.to_float(), 
                       self.risk_params.position_limit_hard.to_float())
            ));
        }
        
        // Soft position limit warning
        if inventory_abs >= self.risk_params.position_limit_soft {
            return Ok(false); // Signal to reduce position
        }
        
        // Position value check
        if trading_state.position_value.abs() >= self.risk_params.max_position_value {
            return Err(QuotingError::RiskLimitExceeded(
                format!("Position value limit exceeded: {} >= {}", 
                       trading_state.position_value.to_float(), 
                       self.risk_params.max_position_value.to_float())
            ));
        }
        
        // Drawdown check
        let max_pnl = self.pnl_history.iter()
            .map(|(_, pnl)| *pnl)
            .fold(FixedPoint::zero(), |max, pnl| if pnl > max { pnl } else { max });
        
        if max_pnl > FixedPoint::zero() {
            let current_drawdown = (max_pnl - trading_state.total_pnl) / max_pnl;
            if current_drawdown >= self.risk_params.max_drawdown {
                return Err(QuotingError::RiskLimitExceeded(
                    format!("Maximum drawdown exceeded: {} >= {}", 
                           current_drawdown.to_float(), 
                           self.risk_params.max_drawdown.to_float())
                ));
            }
        }
        
        Ok(true)
    }
    
    /// Compute automatic position reduction strategy
    pub fn compute_position_reduction_strategy(
        &self,
        trading_state: &TradingState,
        market_state: &MarketState,
    ) -> Result<(FixedPoint, FixedPoint), QuotingError> {
        let inventory = trading_state.inventory;
        let inventory_abs = inventory.abs();
        
        // No reduction needed if within soft limits
        if inventory_abs <= self.risk_params.position_limit_soft {
            return Ok((FixedPoint::zero(), FixedPoint::zero()));
        }
        
        // Calculate target reduction
        let excess_inventory = inventory_abs - self.risk_params.position_limit_soft;
        let reduction_rate = if self.emergency_mode {
            FixedPoint::from_float(0.5) // Aggressive reduction in emergency
        } else {
            FixedPoint::from_float(0.2) // Gradual reduction normally
        };
        
        let target_reduction = excess_inventory * reduction_rate;
        
        // Determine which side to reduce
        if inventory > FixedPoint::zero() {
            // Long position - increase ask size, reduce bid size
            let ask_size_increase = target_reduction;
            let bid_size_decrease = target_reduction / FixedPoint::from_float(2.0);
            Ok((ask_size_increase, -bid_size_decrease))
        } else {
            // Short position - increase bid size, reduce ask size
            let bid_size_increase = target_reduction;
            let ask_size_decrease = target_reduction / FixedPoint::from_float(2.0);
            Ok((bid_size_increase, -ask_size_decrease))
        }
    }
    
    /// Implement inventory skew adjustment for quotes
    pub fn compute_inventory_skew_adjustment(
        &self,
        inventory: FixedPoint,
        volatility: FixedPoint,
        gamma: FixedPoint,
        time_to_maturity: FixedPoint,
    ) -> (FixedPoint, FixedPoint) {
        // Skew adjustment based on inventory level
        let base_skew = inventory * gamma * volatility * volatility * time_to_maturity;
        
        // Asymmetric adjustment for large positions
        let asymmetric_factor = if inventory.abs() > self.risk_params.position_limit_soft {
            FixedPoint::from_float(1.5) // Increase skew for large positions
        } else {
            FixedPoint::one()
        };
        
        let adjusted_skew = base_skew * asymmetric_factor;
        
        // Apply skew to bid and ask
        let bid_adjustment = -adjusted_skew / FixedPoint::from_float(2.0); // Widen bid for long positions
        let ask_adjustment = adjusted_skew / FixedPoint::from_float(2.0);  // Widen ask for short positions
        
        (bid_adjustment, ask_adjustment)
    }
    
    /// Update inventory and PnL history
    pub fn update_history(&mut self, timestamp: u64, inventory: FixedPoint, pnl: FixedPoint) {
        // Add new entries
        self.inventory_history.push_back((timestamp, inventory));
        self.pnl_history.push_back((timestamp, pnl));
        
        // Maintain history size
        if self.inventory_history.len() > self.max_history_size {
            self.inventory_history.pop_front();
        }
        if self.pnl_history.len() > self.max_history_size {
            self.pnl_history.pop_front();
        }
    }
    
    /// Check if emergency liquidation is needed
    pub fn should_emergency_liquidate(&self, trading_state: &TradingState) -> bool {
        let inventory_ratio = trading_state.inventory.abs() / self.risk_params.max_inventory;
        inventory_ratio >= self.risk_params.emergency_liquidation_threshold || self.emergency_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_spread_calculator() {
        let model_params = ModelParameters {
            drift_coefficient: FixedPoint::from_float(0.05),
            volatility_coefficient: FixedPoint::from_float(0.2),
            inventory_penalty: FixedPoint::from_float(0.1),
            adverse_selection_cost: FixedPoint::from_float(0.01),
            market_impact_coefficient: FixedPoint::from_float(0.05),
            risk_aversion: FixedPoint::from_float(2.0),
            terminal_time: FixedPoint::from_float(1.0),
            max_inventory: FixedPoint::from_float(100.0),
        };
        
        let mut calculator = OptimalSpreadCalculator::new(model_params);
        
        let market_state = MarketState {
            mid_price: FixedPoint::from_float(100.0),
            bid_price: FixedPoint::from_float(99.95),
            ask_price: FixedPoint::from_float(100.05),
            bid_volume: FixedPoint::from_float(1000.0),
            ask_volume: FixedPoint::from_float(1000.0),
            spread: FixedPoint::from_float(0.1),
            volatility: FixedPoint::from_float(0.2),
            timestamp: 1000,
            order_flow_imbalance: FixedPoint::zero(),
            market_depth: vec![],
        };
        
        let trading_state = TradingState::new();
        let hawkes_intensities = vec![FixedPoint::one(), FixedPoint::one()];
        
        let result = calculator.compute_optimal_spreads(
            &market_state,
            &trading_state,
            &hawkes_intensities,
            1000,
        );
        
        assert!(result.is_ok());
        let (bid_spread, ask_spread) = result.unwrap();
        assert!(bid_spread.to_float() > 0.0);
        assert!(ask_spread.to_float() > 0.0);
    }

    #[test]
    fn test_inventory_manager() {
        let risk_params = RiskParameters::default_conservative();
        let mut manager = InventoryManager::new(risk_params, 1000);
        
        let trading_state = TradingState {
            inventory: FixedPoint::from_float(50.0),
            cash: FixedPoint::from_float(1000.0),
            unrealized_pnl: FixedPoint::from_float(100.0),
            realized_pnl: FixedPoint::from_float(50.0),
            total_pnl: FixedPoint::from_float(150.0),
            position_value: FixedPoint::from_float(5000.0),
            last_trade_time: 1000,
            trade_count: 10,
            max_inventory_reached: FixedPoint::from_float(60.0),
        };
        
        // Test position limits
        let result = manager.check_position_limits(&trading_state);
        assert!(result.is_ok());
        
        // Test inventory penalty
        let penalty = manager.compute_inventory_penalty(
            FixedPoint::from_float(50.0),
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.1),
        );
        assert!(penalty.to_float() > 0.0);
        
        // Test skew adjustment
        let (bid_adj, ask_adj) = manager.compute_inventory_skew_adjustment(
            FixedPoint::from_float(50.0),
            FixedPoint::from_float(0.2),
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(1.0),
        );
        assert!(bid_adj.to_float() != 0.0 || ask_adj.to_float() != 0.0);
    }
}/// Ad
vanced inventory management with position reduction strategies
impl InventoryManager {
    /// Implement automatic position reduction strategies based on market conditions
    pub fn execute_position_reduction(
        &mut self,
        trading_state: &mut TradingState,
        market_state: &MarketState,
        current_time: u64,
    ) -> Result<Vec<PositionReductionAction>, QuotingError> {
        let mut actions = Vec::new();
        
        // Check if position reduction is needed
        let (bid_adjustment, ask_adjustment) = self.compute_position_reduction_strategy(
            trading_state, market_state
        )?;
        
        if bid_adjustment != FixedPoint::zero() || ask_adjustment != FixedPoint::zero() {
            // Create position reduction actions
            if trading_state.inventory > FixedPoint::zero() {
                // Long position - need to sell
                actions.push(PositionReductionAction {
                    action_type: ReductionActionType::IncreaseSellQuotes,
                    size_adjustment: ask_adjustment,
                    price_adjustment: FixedPoint::from_float(-0.0001), // Slightly more aggressive pricing
                    urgency: self.compute_reduction_urgency(trading_state),
                    timestamp: current_time,
                });
                
                // Reduce bid quotes to avoid accumulating more inventory
                actions.push(PositionReductionAction {
                    action_type: ReductionActionType::ReduceBuyQuotes,
                    size_adjustment: -bid_adjustment,
                    price_adjustment: FixedPoint::from_float(-0.0002), // Less aggressive bid
                    urgency: self.compute_reduction_urgency(trading_state),
                    timestamp: current_time,
                });
            } else if trading_state.inventory < FixedPoint::zero() {
                // Short position - need to buy
                actions.push(PositionReductionAction {
                    action_type: ReductionActionType::IncreaseBuyQuotes,
                    size_adjustment: bid_adjustment,
                    price_adjustment: FixedPoint::from_float(0.0001), // Slightly more aggressive pricing
                    urgency: self.compute_reduction_urgency(trading_state),
                    timestamp: current_time,
                });
                
                // Reduce ask quotes
                actions.push(PositionReductionAction {
                    action_type: ReductionActionType::ReduceSellQuotes,
                    size_adjustment: -ask_adjustment,
                    price_adjustment: FixedPoint::from_float(0.0002), // Less aggressive ask
                    urgency: self.compute_reduction_urgency(trading_state),
                    timestamp: current_time,
                });
            }
        }
        
        // Update last rebalance time
        self.last_rebalance_time = current_time;
        
        Ok(actions)
    }
    
    /// Compute urgency level for position reduction
    fn compute_reduction_urgency(&self, trading_state: &TradingState) -> ReductionUrgency {
        let inventory_ratio = trading_state.inventory.abs() / self.risk_params.max_inventory;
        
        if inventory_ratio >= self.risk_params.emergency_liquidation_threshold {
            ReductionUrgency::Emergency
        } else if inventory_ratio >= FixedPoint::from_float(0.8) {
            ReductionUrgency::High
        } else if inventory_ratio >= FixedPoint::from_float(0.6) {
            ReductionUrgency::Medium
        } else {
            ReductionUrgency::Low
        }
    }
    
    /// Implement emergency liquidation procedures
    pub fn execute_emergency_liquidation(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        let mut actions = Vec::new();
        
        if !self.should_emergency_liquidate(trading_state) {
            return Ok(actions);
        }
        
        let inventory = trading_state.inventory;
        let liquidation_size = inventory.abs();
        
        if inventory > FixedPoint::zero() {
            // Emergency sell
            actions.push(EmergencyAction {
                action_type: EmergencyActionType::MarketSell,
                size: liquidation_size,
                max_price_impact: FixedPoint::from_float(0.01), // Accept up to 1% impact
                timeout_seconds: 30,
                reason: "Emergency liquidation - long position".to_string(),
            });
        } else if inventory < FixedPoint::zero() {
            // Emergency buy
            actions.push(EmergencyAction {
                action_type: EmergencyActionType::MarketBuy,
                size: liquidation_size,
                max_price_impact: FixedPoint::from_float(0.01), // Accept up to 1% impact
                timeout_seconds: 30,
                reason: "Emergency liquidation - short position".to_string(),
            });
        }
        
        // Halt all quoting during emergency liquidation
        actions.push(EmergencyAction {
            action_type: EmergencyActionType::HaltQuoting,
            size: FixedPoint::zero(),
            max_price_impact: FixedPoint::zero(),
            timeout_seconds: 300, // 5 minutes
            reason: "Emergency liquidation in progress".to_string(),
        });
        
        Ok(actions)
    }
    
    /// Advanced inventory penalty with regime-dependent parameters
    pub fn compute_regime_adjusted_penalty(
        &self,
        inventory: FixedPoint,
        volatility: FixedPoint,
        time_to_maturity: FixedPoint,
        gamma: FixedPoint,
        market_regime: MarketRegime,
    ) -> FixedPoint {
        // Base penalty
        let base_penalty = self.compute_inventory_penalty(inventory, volatility, time_to_maturity, gamma);
        
        // Regime adjustment factor
        let regime_factor = match market_regime {
            MarketRegime::Normal => FixedPoint::one(),
            MarketRegime::HighVolatility => FixedPoint::from_float(1.5),
            MarketRegime::LowLiquidity => FixedPoint::from_float(2.0),
            MarketRegime::Crisis => FixedPoint::from_float(3.0),
            MarketRegime::Recovery => FixedPoint::from_float(0.8),
        };
        
        base_penalty * regime_factor
    }
    
    /// Compute optimal inventory target based on market conditions
    pub fn compute_optimal_inventory_target(
        &self,
        market_state: &MarketState,
        trading_state: &TradingState,
        forecast_horizon: FixedPoint,
    ) -> FixedPoint {
        // Base target is zero (market neutral)
        let mut target = FixedPoint::zero();
        
        // Adjust based on order flow imbalance
        let imbalance_adjustment = market_state.order_flow_imbalance * FixedPoint::from_float(0.1);
        target = target + imbalance_adjustment;
        
        // Adjust based on volatility regime
        let vol_adjustment = if market_state.volatility > FixedPoint::from_float(0.3) {
            // High volatility - reduce target inventory
            target * FixedPoint::from_float(0.5)
        } else {
            target
        };
        
        // Ensure target is within risk limits
        let max_target = self.risk_params.position_limit_soft * FixedPoint::from_float(0.5);
        vol_adjustment.max(-max_target).min(max_target)
    }
}

/// Position reduction action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReductionActionType {
    IncreaseBuyQuotes,
    IncreaseSellQuotes,
    ReduceBuyQuotes,
    ReduceSellQuotes,
    CancelAllOrders,
    ReduceQuoteSizes,
}

/// Position reduction urgency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReductionUrgency {
    Low,
    Medium,
    High,
    Emergency,
}

/// Position reduction action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionReductionAction {
    pub action_type: ReductionActionType,
    pub size_adjustment: FixedPoint,
    pub price_adjustment: FixedPoint,
    pub urgency: ReductionUrgency,
    pub timestamp: u64,
}

/// Emergency action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyActionType {
    MarketBuy,
    MarketSell,
    HaltQuoting,
    CancelAllOrders,
    ReducePositionSize,
}

/// Emergency action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAction {
    pub action_type: EmergencyActionType,
    pub size: FixedPoint,
    pub max_price_impact: FixedPoint,
    pub timeout_seconds: u64,
    pub reason: String,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    Normal,
    HighVolatility,
    LowLiquidity,
    Crisis,
    Recovery,
}

/// Market regime detector
pub struct MarketRegimeDetector {
    pub volatility_threshold_high: FixedPoint,
    pub volatility_threshold_low: FixedPoint,
    pub liquidity_threshold: FixedPoint,
    pub spread_threshold_crisis: FixedPoint,
    pub volume_history: VecDeque<FixedPoint>,
    pub volatility_history: VecDeque<FixedPoint>,
    pub spread_history: VecDeque<FixedPoint>,
    pub current_regime: MarketRegime,
    pub regime_confidence: FixedPoint,
}

impl MarketRegimeDetector {
    pub fn new() -> Self {
        Self {
            volatility_threshold_high: FixedPoint::from_float(0.4),
            volatility_threshold_low: FixedPoint::from_float(0.1),
            liquidity_threshold: FixedPoint::from_float(1000.0),
            spread_threshold_crisis: FixedPoint::from_float(0.01),
            volume_history: VecDeque::with_capacity(100),
            volatility_history: VecDeque::with_capacity(100),
            spread_history: VecDeque::with_capacity(100),
            current_regime: MarketRegime::Normal,
            regime_confidence: FixedPoint::from_float(0.5),
        }
    }
    
    /// Detect current market regime
    pub fn detect_regime(&mut self, market_state: &MarketState) -> MarketRegime {
        // Update history
        self.update_history(market_state);
        
        // Compute regime indicators
        let avg_volatility = self.compute_average_volatility();
        let avg_liquidity = self.compute_average_liquidity();
        let avg_spread = self.compute_average_spread();
        
        // Regime classification logic
        let new_regime = if avg_spread >= self.spread_threshold_crisis {
            MarketRegime::Crisis
        } else if avg_volatility >= self.volatility_threshold_high {
            MarketRegime::HighVolatility
        } else if avg_liquidity <= self.liquidity_threshold {
            MarketRegime::LowLiquidity
        } else if avg_volatility <= self.volatility_threshold_low && 
                  self.current_regime == MarketRegime::Crisis {
            MarketRegime::Recovery
        } else {
            MarketRegime::Normal
        };
        
        // Update regime with smoothing
        if new_regime != self.current_regime {
            self.regime_confidence = self.regime_confidence * FixedPoint::from_float(0.9);
            if self.regime_confidence < FixedPoint::from_float(0.3) {
                self.current_regime = new_regime;
                self.regime_confidence = FixedPoint::from_float(0.7);
            }
        } else {
            self.regime_confidence = (self.regime_confidence + FixedPoint::from_float(0.1))
                .min(FixedPoint::one());
        }
        
        self.current_regime
    }
    
    fn update_history(&mut self, market_state: &MarketState) {
        self.volatility_history.push_back(market_state.volatility);
        self.spread_history.push_back(market_state.spread);
        
        let total_volume = market_state.bid_volume + market_state.ask_volume;
        self.volume_history.push_back(total_volume);
        
        // Maintain history size
        if self.volatility_history.len() > 100 {
            self.volatility_history.pop_front();
        }
        if self.spread_history.len() > 100 {
            self.spread_history.pop_front();
        }
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }
    }
    
    fn compute_average_volatility(&self) -> FixedPoint {
        if self.volatility_history.is_empty() {
            return FixedPoint::from_float(0.2);
        }
        
        self.volatility_history.iter().sum::<FixedPoint>() / 
        FixedPoint::from_float(self.volatility_history.len() as f64)
    }
    
    fn compute_average_liquidity(&self) -> FixedPoint {
        if self.volume_history.is_empty() {
            return FixedPoint::from_float(1000.0);
        }
        
        self.volume_history.iter().sum::<FixedPoint>() / 
        FixedPoint::from_float(self.volume_history.len() as f64)
    }
    
    fn compute_average_spread(&self) -> FixedPoint {
        if self.spread_history.is_empty() {
            return FixedPoint::from_float(0.001);
        }
        
        self.spread_history.iter().sum::<FixedPoint>() / 
        FixedPoint::from_float(self.spread_history.len() as f64)
    }
}/// Adaptive
 quoting parameters engine
pub struct AdaptiveQuotingEngine {
    pub base_params: QuotingParameters,
    pub regime_detector: MarketRegimeDetector,
    pub volatility_estimator: VolatilityEstimator,
    pub liquidity_analyzer: LiquidityAnalyzer,
    pub correlation_tracker: CorrelationTracker,
    pub adaptation_speed: FixedPoint,
    pub last_update_time: u64,
    pub update_interval: u64,
}

/// Base quoting parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotingParameters {
    pub base_spread: FixedPoint,
    pub quote_size: FixedPoint,
    pub max_quote_size: FixedPoint,
    pub min_quote_size: FixedPoint,
    pub spread_multiplier: FixedPoint,
    pub size_multiplier: FixedPoint,
    pub skew_factor: FixedPoint,
    pub volatility_adjustment: FixedPoint,
    pub liquidity_adjustment: FixedPoint,
    pub regime_adjustment: FixedPoint,
}

impl QuotingParameters {
    pub fn default() -> Self {
        Self {
            base_spread: FixedPoint::from_float(0.001),      // 10 bps
            quote_size: FixedPoint::from_float(100.0),       // Base size
            max_quote_size: FixedPoint::from_float(1000.0),  // Max size
            min_quote_size: FixedPoint::from_float(10.0),    // Min size
            spread_multiplier: FixedPoint::one(),
            size_multiplier: FixedPoint::one(),
            skew_factor: FixedPoint::zero(),
            volatility_adjustment: FixedPoint::one(),
            liquidity_adjustment: FixedPoint::one(),
            regime_adjustment: FixedPoint::one(),
        }
    }
}

impl AdaptiveQuotingEngine {
    pub fn new(base_params: QuotingParameters) -> Self {
        Self {
            base_params,
            regime_detector: MarketRegimeDetector::new(),
            volatility_estimator: VolatilityEstimator::new(),
            liquidity_analyzer: LiquidityAnalyzer::new(),
            correlation_tracker: CorrelationTracker::new(),
            adaptation_speed: FixedPoint::from_float(0.1),
            last_update_time: 0,
            update_interval: 1000, // Update every second
        }
    }
    
    /// Implement volatility-based spread adjustment
    pub fn compute_volatility_adjusted_spread(
        &mut self,
        base_spread: FixedPoint,
        market_state: &MarketState,
        lookback_periods: usize,
    ) -> FixedPoint {
        // Update volatility estimate
        self.volatility_estimator.update(market_state.mid_price, market_state.timestamp);
        
        // Get current and historical volatility
        let current_vol = self.volatility_estimator.get_current_volatility();
        let historical_vol = self.volatility_estimator.get_historical_volatility(lookback_periods);
        
        // Volatility ratio
        let vol_ratio = if historical_vol > FixedPoint::zero() {
            current_vol / historical_vol
        } else {
            FixedPoint::one()
        };
        
        // Adjustment factor with bounds
        let vol_adjustment = vol_ratio.max(FixedPoint::from_float(0.5))
                                    .min(FixedPoint::from_float(3.0));
        
        // Apply square root scaling for volatility
        let sqrt_adjustment = FixedPoint::from_float(vol_adjustment.to_float().sqrt());
        
        base_spread * sqrt_adjustment
    }
    
    /// Build liquidity-aware quote sizing
    pub fn compute_liquidity_adjusted_size(
        &mut self,
        base_size: FixedPoint,
        market_state: &MarketState,
        side: Side,
    ) -> FixedPoint {
        // Update liquidity analysis
        self.liquidity_analyzer.update(market_state);
        
        // Get liquidity metrics
        let depth_ratio = self.liquidity_analyzer.get_depth_ratio(side);
        let volume_profile = self.liquidity_analyzer.get_volume_profile();
        let market_impact = self.liquidity_analyzer.estimate_market_impact(base_size, side);
        
        // Size adjustment based on available liquidity
        let depth_adjustment = depth_ratio.max(FixedPoint::from_float(0.1))
                                         .min(FixedPoint::from_float(2.0));
        
        // Volume profile adjustment
        let volume_adjustment = if volume_profile > FixedPoint::from_float(0.8) {
            FixedPoint::from_float(1.2) // Increase size in high volume periods
        } else if volume_profile < FixedPoint::from_float(0.3) {
            FixedPoint::from_float(0.7) // Reduce size in low volume periods
        } else {
            FixedPoint::one()
        };
        
        // Market impact constraint
        let impact_adjustment = if market_impact > FixedPoint::from_float(0.001) {
            FixedPoint::from_float(0.8) // Reduce size if impact is high
        } else {
            FixedPoint::one()
        };
        
        let adjusted_size = base_size * depth_adjustment * volume_adjustment * impact_adjustment;
        
        // Apply bounds
        adjusted_size.max(self.base_params.min_quote_size)
                    .min(self.base_params.max_quote_size)
    }
    
    /// Create market regime detection and adaptation
    pub fn adapt_to_market_regime(
        &mut self,
        params: &mut QuotingParameters,
        market_state: &MarketState,
    ) -> MarketRegime {
        let regime = self.regime_detector.detect_regime(market_state);
        
        // Regime-specific parameter adjustments
        match regime {
            MarketRegime::Normal => {
                params.spread_multiplier = FixedPoint::one();
                params.size_multiplier = FixedPoint::one();
                params.volatility_adjustment = FixedPoint::one();
            }
            MarketRegime::HighVolatility => {
                params.spread_multiplier = FixedPoint::from_float(1.5);
                params.size_multiplier = FixedPoint::from_float(0.8);
                params.volatility_adjustment = FixedPoint::from_float(1.3);
            }
            MarketRegime::LowLiquidity => {
                params.spread_multiplier = FixedPoint::from_float(1.2);
                params.size_multiplier = FixedPoint::from_float(0.6);
                params.volatility_adjustment = FixedPoint::from_float(1.1);
            }
            MarketRegime::Crisis => {
                params.spread_multiplier = FixedPoint::from_float(2.0);
                params.size_multiplier = FixedPoint::from_float(0.4);
                params.volatility_adjustment = FixedPoint::from_float(1.8);
            }
            MarketRegime::Recovery => {
                params.spread_multiplier = FixedPoint::from_float(0.9);
                params.size_multiplier = FixedPoint::from_float(1.1);
                params.volatility_adjustment = FixedPoint::from_float(0.9);
            }
        }
        
        regime
    }
    
    /// Implement correlation-based multi-asset adjustments
    pub fn compute_correlation_adjustments(
        &mut self,
        base_params: &QuotingParameters,
        primary_asset: &str,
        correlated_assets: &[(&str, &MarketState)],
    ) -> QuotingParameters {
        let mut adjusted_params = base_params.clone();
        
        // Update correlation tracking
        for (asset_name, market_state) in correlated_assets {
            self.correlation_tracker.update_correlation(
                primary_asset,
                asset_name,
                market_state.mid_price,
                market_state.timestamp,
            );
        }
        
        // Get correlation matrix
        let correlations = self.correlation_tracker.get_correlations(primary_asset);
        
        // Compute correlation-based adjustments
        let mut correlation_risk = FixedPoint::zero();
        let mut correlation_count = 0;
        
        for (_, correlation) in correlations {
            correlation_risk = correlation_risk + correlation.abs();
            correlation_count += 1;
        }
        
        if correlation_count > 0 {
            let avg_correlation = correlation_risk / FixedPoint::from_float(correlation_count as f64);
            
            // Adjust spread based on correlation risk
            let correlation_adjustment = FixedPoint::one() + avg_correlation * FixedPoint::from_float(0.2);
            adjusted_params.spread_multiplier = adjusted_params.spread_multiplier * correlation_adjustment;
            
            // Adjust size based on diversification
            let diversification_factor = FixedPoint::one() - avg_correlation * FixedPoint::from_float(0.1);
            adjusted_params.size_multiplier = adjusted_params.size_multiplier * diversification_factor;
        }
        
        adjusted_params
    }
    
    /// Update adaptive parameters based on all factors
    pub fn update_adaptive_parameters(
        &mut self,
        market_state: &MarketState,
        trading_state: &TradingState,
        correlated_assets: &[(&str, &MarketState)],
        current_time: u64,
    ) -> Result<QuotingParameters, QuotingError> {
        // Check if update is needed
        if current_time - self.last_update_time < self.update_interval {
            return Ok(self.base_params.clone());
        }
        
        let mut adapted_params = self.base_params.clone();
        
        // 1. Volatility-based adjustments
        adapted_params.base_spread = self.compute_volatility_adjusted_spread(
            adapted_params.base_spread,
            market_state,
            100, // 100-period lookback
        );
        
        // 2. Liquidity-based size adjustments
        adapted_params.quote_size = self.compute_liquidity_adjusted_size(
            adapted_params.quote_size,
            market_state,
            Side::Buy, // Use buy side as reference
        );
        
        // 3. Market regime adaptation
        let current_regime = self.adapt_to_market_regime(&mut adapted_params, market_state);
        
        // 4. Correlation-based adjustments
        if !correlated_assets.is_empty() {
            adapted_params = self.compute_correlation_adjustments(
                &adapted_params,
                "primary", // Primary asset identifier
                correlated_assets,
            );
        }
        
        // 5. Inventory-based skew adjustment
        let inventory_ratio = trading_state.inventory / FixedPoint::from_float(100.0); // Normalize
        adapted_params.skew_factor = inventory_ratio * FixedPoint::from_float(0.1);
        
        // 6. Apply adaptation speed smoothing
        self.apply_adaptation_smoothing(&mut adapted_params);
        
        // Update timestamp
        self.last_update_time = current_time;
        
        Ok(adapted_params)
    }
    
    /// Apply smoothing to parameter changes
    fn apply_adaptation_smoothing(&mut self, new_params: &mut QuotingParameters) {
        let alpha = self.adaptation_speed;
        let one_minus_alpha = FixedPoint::one() - alpha;
        
        // Smooth parameter transitions
        new_params.spread_multiplier = alpha * new_params.spread_multiplier + 
                                      one_minus_alpha * self.base_params.spread_multiplier;
        
        new_params.size_multiplier = alpha * new_params.size_multiplier + 
                                    one_minus_alpha * self.base_params.size_multiplier;
        
        new_params.volatility_adjustment = alpha * new_params.volatility_adjustment + 
                                          one_minus_alpha * self.base_params.volatility_adjustment;
        
        // Update base parameters for next iteration
        self.base_params = new_params.clone();
    }
}

/// Volatility estimator using EWMA and GARCH-like models
pub struct VolatilityEstimator {
    pub price_history: VecDeque<(u64, FixedPoint)>,
    pub return_history: VecDeque<FixedPoint>,
    pub volatility_history: VecDeque<FixedPoint>,
    pub ewma_lambda: FixedPoint,
    pub current_volatility: FixedPoint,
    pub max_history: usize,
}

impl VolatilityEstimator {
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(1000),
            return_history: VecDeque::with_capacity(1000),
            volatility_history: VecDeque::with_capacity(1000),
            ewma_lambda: FixedPoint::from_float(0.94), // Standard EWMA decay
            current_volatility: FixedPoint::from_float(0.2), // Initial estimate
            max_history: 1000,
        }
    }
    
    pub fn update(&mut self, price: FixedPoint, timestamp: u64) {
        // Add to price history
        self.price_history.push_back((timestamp, price));
        
        // Maintain history size
        if self.price_history.len() > self.max_history {
            self.price_history.pop_front();
        }
        
        // Compute return if we have previous price
        if self.price_history.len() >= 2 {
            let prev_price = self.price_history[self.price_history.len() - 2].1;
            let log_return = (price / prev_price).ln();
            
            self.return_history.push_back(log_return);
            if self.return_history.len() > self.max_history {
                self.return_history.pop_front();
            }
            
            // Update EWMA volatility
            let squared_return = log_return * log_return;
            self.current_volatility = self.ewma_lambda * self.current_volatility + 
                                     (FixedPoint::one() - self.ewma_lambda) * squared_return;
            
            self.volatility_history.push_back(self.current_volatility);
            if self.volatility_history.len() > self.max_history {
                self.volatility_history.pop_front();
            }
        }
    }
    
    pub fn get_current_volatility(&self) -> FixedPoint {
        self.current_volatility.sqrt()
    }
    
    pub fn get_historical_volatility(&self, periods: usize) -> FixedPoint {
        if self.volatility_history.len() < periods {
            return self.current_volatility.sqrt();
        }
        
        let start_idx = self.volatility_history.len() - periods;
        let sum: FixedPoint = self.volatility_history.iter()
            .skip(start_idx)
            .sum();
        
        (sum / FixedPoint::from_float(periods as f64)).sqrt()
    }
}

/// Liquidity analyzer for market depth and impact estimation
pub struct LiquidityAnalyzer {
    pub depth_history: VecDeque<(FixedPoint, FixedPoint)>, // (bid_depth, ask_depth)
    pub volume_history: VecDeque<FixedPoint>,
    pub spread_history: VecDeque<FixedPoint>,
    pub impact_model: MarketImpactModel,
    pub max_history: usize,
}

impl LiquidityAnalyzer {
    pub fn new() -> Self {
        Self {
            depth_history: VecDeque::with_capacity(100),
            volume_history: VecDeque::with_capacity(100),
            spread_history: VecDeque::with_capacity(100),
            impact_model: MarketImpactModel::new(),
            max_history: 100,
        }
    }
    
    pub fn update(&mut self, market_state: &MarketState) {
        // Update depth history
        self.depth_history.push_back((market_state.bid_volume, market_state.ask_volume));
        if self.depth_history.len() > self.max_history {
            self.depth_history.pop_front();
        }
        
        // Update volume history
        let total_volume = market_state.bid_volume + market_state.ask_volume;
        self.volume_history.push_back(total_volume);
        if self.volume_history.len() > self.max_history {
            self.volume_history.pop_front();
        }
        
        // Update spread history
        self.spread_history.push_back(market_state.spread);
        if self.spread_history.len() > self.max_history {
            self.spread_history.pop_front();
        }
        
        // Update impact model
        self.impact_model.update(market_state);
    }
    
    pub fn get_depth_ratio(&self, side: Side) -> FixedPoint {
        if self.depth_history.is_empty() {
            return FixedPoint::one();
        }
        
        let recent_depth = match side {
            Side::Buy => self.depth_history.back().unwrap().1, // Ask depth for buy orders
            Side::Sell => self.depth_history.back().unwrap().0, // Bid depth for sell orders
        };
        
        let avg_depth = self.depth_history.iter()
            .map(|(bid, ask)| match side {
                Side::Buy => *ask,
                Side::Sell => *bid,
            })
            .sum::<FixedPoint>() / FixedPoint::from_float(self.depth_history.len() as f64);
        
        if avg_depth > FixedPoint::zero() {
            recent_depth / avg_depth
        } else {
            FixedPoint::one()
        }
    }
    
    pub fn get_volume_profile(&self) -> FixedPoint {
        if self.volume_history.len() < 2 {
            return FixedPoint::from_float(0.5);
        }
        
        let recent_volume = self.volume_history.back().unwrap();
        let avg_volume = self.volume_history.iter().sum::<FixedPoint>() / 
                        FixedPoint::from_float(self.volume_history.len() as f64);
        
        if avg_volume > FixedPoint::zero() {
            (*recent_volume / avg_volume).min(FixedPoint::from_float(2.0))
        } else {
            FixedPoint::from_float(0.5)
        }
    }
    
    pub fn estimate_market_impact(&self, size: FixedPoint, side: Side) -> FixedPoint {
        self.impact_model.estimate_impact(size, side)
    }
}

/// Market impact model using square-root law
pub struct MarketImpactModel {
    pub impact_coefficient: FixedPoint,
    pub volume_history: VecDeque<FixedPoint>,
    pub price_impact_history: VecDeque<FixedPoint>,
}

impl MarketImpactModel {
    pub fn new() -> Self {
        Self {
            impact_coefficient: FixedPoint::from_float(0.1), // Initial estimate
            volume_history: VecDeque::with_capacity(100),
            price_impact_history: VecDeque::with_capacity(100),
        }
    }
    
    pub fn update(&mut self, market_state: &MarketState) {
        let total_volume = market_state.bid_volume + market_state.ask_volume;
        self.volume_history.push_back(total_volume);
        
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }
        
        // Estimate impact coefficient from spread and volume
        if total_volume > FixedPoint::zero() {
            let estimated_impact = market_state.spread / total_volume.sqrt();
            self.price_impact_history.push_back(estimated_impact);
            
            if self.price_impact_history.len() > 100 {
                self.price_impact_history.pop_front();
            }
            
            // Update coefficient using EWMA
            let alpha = FixedPoint::from_float(0.1);
            self.impact_coefficient = alpha * estimated_impact + 
                                     (FixedPoint::one() - alpha) * self.impact_coefficient;
        }
    }
    
    pub fn estimate_impact(&self, size: FixedPoint, _side: Side) -> FixedPoint {
        // Square-root law: impact = coefficient * sqrt(size / avg_volume)
        let avg_volume = if self.volume_history.is_empty() {
            FixedPoint::from_float(1000.0)
        } else {
            self.volume_history.iter().sum::<FixedPoint>() / 
            FixedPoint::from_float(self.volume_history.len() as f64)
        };
        
        if avg_volume > FixedPoint::zero() {
            self.impact_coefficient * (size / avg_volume).sqrt()
        } else {
            FixedPoint::zero()
        }
    }
}

/// Correlation tracker for multi-asset strategies
pub struct CorrelationTracker {
    pub price_histories: HashMap<String, VecDeque<(u64, FixedPoint)>>,
    pub correlation_matrix: HashMap<(String, String), FixedPoint>,
    pub correlation_window: usize,
    pub last_update_time: u64,
}

impl CorrelationTracker {
    pub fn new() -> Self {
        Self {
            price_histories: HashMap::new(),
            correlation_matrix: HashMap::new(),
            correlation_window: 100,
            last_update_time: 0,
        }
    }
    
    pub fn update_correlation(
        &mut self,
        asset1: &str,
        asset2: &str,
        price: FixedPoint,
        timestamp: u64,
    ) {
        // Update price history for asset2
        let history = self.price_histories.entry(asset2.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.correlation_window));
        
        history.push_back((timestamp, price));
        if history.len() > self.correlation_window {
            history.pop_front();
        }
        
        // Compute correlation if we have enough data for both assets
        if let Some(asset1_history) = self.price_histories.get(asset1) {
            if asset1_history.len() >= 20 && history.len() >= 20 {
                let correlation = self.compute_correlation(asset1_history, history);
                self.correlation_matrix.insert(
                    (asset1.to_string(), asset2.to_string()),
                    correlation,
                );
            }
        }
        
        self.last_update_time = timestamp;
    }
    
    pub fn get_correlations(&self, asset: &str) -> Vec<(String, FixedPoint)> {
        self.correlation_matrix.iter()
            .filter_map(|((a1, a2), &corr)| {
                if a1 == asset {
                    Some((a2.clone(), corr))
                } else if a2 == asset {
                    Some((a1.clone(), corr))
                } else {
                    None
                }
            })
            .collect()
    }
    
    fn compute_correlation(
        &self,
        history1: &VecDeque<(u64, FixedPoint)>,
        history2: &VecDeque<(u64, FixedPoint)>,
    ) -> FixedPoint {
        let min_len = history1.len().min(history2.len());
        if min_len < 10 {
            return FixedPoint::zero();
        }
        
        // Compute returns
        let returns1: Vec<FixedPoint> = history1.iter()
            .take(min_len)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[1].1 / w[0].1).ln())
            .collect();
        
        let returns2: Vec<FixedPoint> = history2.iter()
            .take(min_len)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[1].1 / w[0].1).ln())
            .collect();
        
        if returns1.len() != returns2.len() || returns1.is_empty() {
            return FixedPoint::zero();
        }
        
        // Compute correlation coefficient
        let n = FixedPoint::from_float(returns1.len() as f64);
        let mean1 = returns1.iter().sum::<FixedPoint>() / n;
        let mean2 = returns2.iter().sum::<FixedPoint>() / n;
        
        let mut numerator = FixedPoint::zero();
        let mut sum_sq1 = FixedPoint::zero();
        let mut sum_sq2 = FixedPoint::zero();
        
        for (r1, r2) in returns1.iter().zip(returns2.iter()) {
            let diff1 = *r1 - mean1;
            let diff2 = *r2 - mean2;
            
            numerator = numerator + diff1 * diff2;
            sum_sq1 = sum_sq1 + diff1 * diff1;
            sum_sq2 = sum_sq2 + diff2 * diff2;
        }
        
        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator > FixedPoint::zero() {
            numerator / denominator
        } else {
            FixedPoint::zero()
        }
    }
}/
// Comprehensive risk management system
pub struct RiskManagementSystem {
    pub risk_params: RiskParameters,
    pub drawdown_monitor: DrawdownMonitor,
    pub position_monitor: PositionMonitor,
    pub volatility_monitor: VolatilityMonitor,
    pub liquidity_monitor: LiquidityMonitor,
    pub circuit_breakers: CircuitBreakers,
    pub emergency_procedures: EmergencyProcedures,
    pub risk_metrics: RiskMetrics,
    pub alert_system: AlertSystem,
}

impl RiskManagementSystem {
    pub fn new(risk_params: RiskParameters) -> Self {
        Self {
            risk_params: risk_params.clone(),
            drawdown_monitor: DrawdownMonitor::new(risk_params.max_drawdown),
            position_monitor: PositionMonitor::new(risk_params.clone()),
            volatility_monitor: VolatilityMonitor::new(risk_params.volatility_threshold),
            liquidity_monitor: LiquidityMonitor::new(),
            circuit_breakers: CircuitBreakers::new(),
            emergency_procedures: EmergencyProcedures::new(),
            risk_metrics: RiskMetrics::new(),
            alert_system: AlertSystem::new(),
        }
    }
    
    /// Comprehensive risk check before placing quotes
    pub fn check_risk_limits(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
        proposed_quotes: &QuotePair,
        current_time: u64,
    ) -> Result<RiskCheckResult, QuotingError> {
        let mut warnings = Vec::new();
        let mut violations = Vec::new();
        
        // 1. Drawdown monitoring
        match self.drawdown_monitor.check_drawdown(trading_state, current_time) {
            Ok(DrawdownStatus::Normal) => {},
            Ok(DrawdownStatus::Warning(level)) => {
                warnings.push(RiskWarning::DrawdownWarning(level));
            },
            Err(e) => {
                violations.push(RiskViolation::DrawdownViolation(e.to_string()));
            }
        }
        
        // 2. Position monitoring
        match self.position_monitor.check_position_limits(trading_state, proposed_quotes) {
            Ok(PositionStatus::Normal) => {},
            Ok(PositionStatus::Warning(level)) => {
                warnings.push(RiskWarning::PositionWarning(level));
            },
            Err(e) => {
                violations.push(RiskViolation::PositionViolation(e.to_string()));
            }
        }
        
        // 3. Volatility monitoring
        match self.volatility_monitor.check_volatility_limits(market_state) {
            Ok(VolatilityStatus::Normal) => {},
            Ok(VolatilityStatus::Warning(level)) => {
                warnings.push(RiskWarning::VolatilityWarning(level));
            },
            Err(e) => {
                violations.push(RiskViolation::VolatilityViolation(e.to_string()));
            }
        }
        
        // 4. Liquidity monitoring
        match self.liquidity_monitor.check_liquidity_conditions(market_state) {
            Ok(LiquidityStatus::Normal) => {},
            Ok(LiquidityStatus::Warning(level)) => {
                warnings.push(RiskWarning::LiquidityWarning(level));
            },
            Err(e) => {
                violations.push(RiskViolation::LiquidityViolation(e.to_string()));
            }
        }
        
        // 5. Circuit breaker checks
        if let Some(breaker_action) = self.circuit_breakers.check_breakers(
            trading_state, market_state, current_time
        ) {
            violations.push(RiskViolation::CircuitBreakerTriggered(breaker_action));
        }
        
        // 6. Update risk metrics
        self.risk_metrics.update(trading_state, market_state, current_time);
        
        // 7. Send alerts if necessary
        if !warnings.is_empty() || !violations.is_empty() {
            self.alert_system.send_risk_alerts(&warnings, &violations, current_time);
        }
        
        // Determine overall risk status
        let risk_status = if !violations.is_empty() {
            RiskStatus::Violation
        } else if !warnings.is_empty() {
            RiskStatus::Warning
        } else {
            RiskStatus::Normal
        };
        
        Ok(RiskCheckResult {
            status: risk_status,
            warnings,
            violations,
            recommended_actions: self.compute_recommended_actions(&warnings, &violations),
            risk_score: self.risk_metrics.compute_overall_risk_score(),
        })
    }
    
    /// Execute emergency procedures if needed
    pub fn execute_emergency_procedures(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
        violation_type: &RiskViolation,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        self.emergency_procedures.execute(trading_state, market_state, violation_type)
    }
    
    fn compute_recommended_actions(
        &self,
        warnings: &[RiskWarning],
        violations: &[RiskViolation],
    ) -> Vec<RiskAction> {
        let mut actions = Vec::new();
        
        for warning in warnings {
            match warning {
                RiskWarning::DrawdownWarning(_) => {
                    actions.push(RiskAction::ReducePositionSize(FixedPoint::from_float(0.5)));
                    actions.push(RiskAction::WidenSpreads(FixedPoint::from_float(1.2)));
                }
                RiskWarning::PositionWarning(_) => {
                    actions.push(RiskAction::ReduceQuoteSizes(FixedPoint::from_float(0.7)));
                }
                RiskWarning::VolatilityWarning(_) => {
                    actions.push(RiskAction::WidenSpreads(FixedPoint::from_float(1.5)));
                    actions.push(RiskAction::ReduceQuoteSizes(FixedPoint::from_float(0.8)));
                }
                RiskWarning::LiquidityWarning(_) => {
                    actions.push(RiskAction::ReduceQuoteSizes(FixedPoint::from_float(0.6)));
                }
            }
        }
        
        for violation in violations {
            match violation {
                RiskViolation::DrawdownViolation(_) => {
                    actions.push(RiskAction::HaltTrading);
                    actions.push(RiskAction::EmergencyLiquidation);
                }
                RiskViolation::PositionViolation(_) => {
                    actions.push(RiskAction::EmergencyLiquidation);
                }
                RiskViolation::VolatilityViolation(_) => {
                    actions.push(RiskAction::HaltTrading);
                }
                RiskViolation::LiquidityViolation(_) => {
                    actions.push(RiskAction::HaltTrading);
                }
                RiskViolation::CircuitBreakerTriggered(_) => {
                    actions.push(RiskAction::HaltTrading);
                    actions.push(RiskAction::CancelAllOrders);
                }
            }
        }
        
        actions
    }
}

/// Real-time drawdown monitoring
pub struct DrawdownMonitor {
    pub max_allowed_drawdown: FixedPoint,
    pub peak_pnl: FixedPoint,
    pub current_drawdown: FixedPoint,
    pub drawdown_history: VecDeque<(u64, FixedPoint)>,
    pub consecutive_loss_count: u32,
    pub max_consecutive_losses: u32,
}

impl DrawdownMonitor {
    pub fn new(max_drawdown: FixedPoint) -> Self {
        Self {
            max_allowed_drawdown: max_drawdown,
            peak_pnl: FixedPoint::zero(),
            current_drawdown: FixedPoint::zero(),
            drawdown_history: VecDeque::with_capacity(1000),
            consecutive_loss_count: 0,
            max_consecutive_losses: 10,
        }
    }
    
    pub fn check_drawdown(
        &mut self,
        trading_state: &TradingState,
        timestamp: u64,
    ) -> Result<DrawdownStatus, QuotingError> {
        let current_pnl = trading_state.total_pnl;
        
        // Update peak PnL
        if current_pnl > self.peak_pnl {
            self.peak_pnl = current_pnl;
            self.consecutive_loss_count = 0;
        } else {
            self.consecutive_loss_count += 1;
        }
        
        // Calculate current drawdown
        self.current_drawdown = if self.peak_pnl > FixedPoint::zero() {
            (self.peak_pnl - current_pnl) / self.peak_pnl
        } else {
            FixedPoint::zero()
        };
        
        // Record drawdown history
        self.drawdown_history.push_back((timestamp, self.current_drawdown));
        if self.drawdown_history.len() > 1000 {
            self.drawdown_history.pop_front();
        }
        
        // Check limits
        if self.current_drawdown >= self.max_allowed_drawdown {
            return Err(QuotingError::RiskLimitExceeded(
                format!("Maximum drawdown exceeded: {:.2}% >= {:.2}%",
                       self.current_drawdown.to_float() * 100.0,
                       self.max_allowed_drawdown.to_float() * 100.0)
            ));
        }
        
        // Check consecutive losses
        if self.consecutive_loss_count >= self.max_consecutive_losses {
            return Err(QuotingError::RiskLimitExceeded(
                format!("Too many consecutive losses: {}", self.consecutive_loss_count)
            ));
        }
        
        // Warning levels
        let warning_threshold = self.max_allowed_drawdown * FixedPoint::from_float(0.8);
        if self.current_drawdown >= warning_threshold {
            Ok(DrawdownStatus::Warning(self.current_drawdown))
        } else {
            Ok(DrawdownStatus::Normal)
        }
    }
}

/// Position size monitoring and limits
pub struct PositionMonitor {
    pub risk_params: RiskParameters,
    pub position_history: VecDeque<(u64, FixedPoint)>,
    pub max_position_reached: FixedPoint,
    pub position_velocity: FixedPoint,
}

impl PositionMonitor {
    pub fn new(risk_params: RiskParameters) -> Self {
        Self {
            risk_params,
            position_history: VecDeque::with_capacity(100),
            max_position_reached: FixedPoint::zero(),
            position_velocity: FixedPoint::zero(),
        }
    }
    
    pub fn check_position_limits(
        &mut self,
        trading_state: &TradingState,
        proposed_quotes: &QuotePair,
    ) -> Result<PositionStatus, QuotingError> {
        let current_position = trading_state.inventory.abs();
        let position_value = trading_state.position_value.abs();
        
        // Update position tracking
        self.max_position_reached = self.max_position_reached.max(current_position);
        
        // Check hard limits
        if current_position >= self.risk_params.position_limit_hard {
            return Err(QuotingError::RiskLimitExceeded(
                format!("Hard position limit exceeded: {} >= {}",
                       current_position.to_float(),
                       self.risk_params.position_limit_hard.to_float())
            ));
        }
        
        if position_value >= self.risk_params.max_position_value {
            return Err(QuotingError::RiskLimitExceeded(
                format!("Position value limit exceeded: {} >= {}",
                       position_value.to_float(),
                       self.risk_params.max_position_value.to_float())
            ));
        }
        
        // Check if proposed quotes would exceed limits
        let potential_position_increase = proposed_quotes.bid_size.max(proposed_quotes.ask_size);
        let potential_new_position = current_position + potential_position_increase;
        
        if potential_new_position >= self.risk_params.position_limit_hard {
            return Err(QuotingError::RiskLimitExceeded(
                "Proposed quotes would exceed position limits".to_string()
            ));
        }
        
        // Warning levels
        if current_position >= self.risk_params.position_limit_soft {
            Ok(PositionStatus::Warning(current_position))
        } else {
            Ok(PositionStatus::Normal)
        }
    }
}

/// Volatility monitoring and circuit breakers
pub struct VolatilityMonitor {
    pub volatility_threshold: FixedPoint,
    pub volatility_history: VecDeque<(u64, FixedPoint)>,
    pub volatility_spike_count: u32,
    pub max_volatility_spikes: u32,
}

impl VolatilityMonitor {
    pub fn new(threshold: FixedPoint) -> Self {
        Self {
            volatility_threshold: threshold,
            volatility_history: VecDeque::with_capacity(100),
            volatility_spike_count: 0,
            max_volatility_spikes: 5,
        }
    }
    
    pub fn check_volatility_limits(
        &mut self,
        market_state: &MarketState,
    ) -> Result<VolatilityStatus, QuotingError> {
        let current_vol = market_state.volatility;
        
        // Update history
        self.volatility_history.push_back((market_state.timestamp, current_vol));
        if self.volatility_history.len() > 100 {
            self.volatility_history.pop_front();
        }
        
        // Check for volatility spike
        if current_vol > self.volatility_threshold {
            self.volatility_spike_count += 1;
            
            if self.volatility_spike_count >= self.max_volatility_spikes {
                return Err(QuotingError::RiskLimitExceeded(
                    format!("Too many volatility spikes: {}", self.volatility_spike_count)
                ));
            }
            
            return Ok(VolatilityStatus::Warning(current_vol));
        } else {
            // Reset spike count if volatility normalizes
            if self.volatility_spike_count > 0 {
                self.volatility_spike_count = self.volatility_spike_count.saturating_sub(1);
            }
        }
        
        Ok(VolatilityStatus::Normal)
    }
}

/// Liquidity monitoring
pub struct LiquidityMonitor {
    pub min_liquidity_threshold: FixedPoint,
    pub liquidity_history: VecDeque<(u64, FixedPoint)>,
    pub low_liquidity_count: u32,
}

impl LiquidityMonitor {
    pub fn new() -> Self {
        Self {
            min_liquidity_threshold: FixedPoint::from_float(100.0),
            liquidity_history: VecDeque::with_capacity(100),
            low_liquidity_count: 0,
        }
    }
    
    pub fn check_liquidity_conditions(
        &mut self,
        market_state: &MarketState,
    ) -> Result<LiquidityStatus, QuotingError> {
        let total_liquidity = market_state.bid_volume + market_state.ask_volume;
        
        // Update history
        self.liquidity_history.push_back((market_state.timestamp, total_liquidity));
        if self.liquidity_history.len() > 100 {
            self.liquidity_history.pop_front();
        }
        
        if total_liquidity < self.min_liquidity_threshold {
            self.low_liquidity_count += 1;
            
            if self.low_liquidity_count >= 10 {
                return Err(QuotingError::InsufficientLiquidity(
                    format!("Persistent low liquidity: {}", total_liquidity.to_float())
                ));
            }
            
            return Ok(LiquidityStatus::Warning(total_liquidity));
        } else {
            self.low_liquidity_count = 0;
        }
        
        Ok(LiquidityStatus::Normal)
    }
}

/// Circuit breakers for automatic trading halts
pub struct CircuitBreakers {
    pub price_circuit_breaker: PriceCircuitBreaker,
    pub volume_circuit_breaker: VolumeCircuitBreaker,
    pub pnl_circuit_breaker: PnLCircuitBreaker,
    pub time_circuit_breaker: TimeCircuitBreaker,
}

impl CircuitBreakers {
    pub fn new() -> Self {
        Self {
            price_circuit_breaker: PriceCircuitBreaker::new(FixedPoint::from_float(0.05)), // 5% price move
            volume_circuit_breaker: VolumeCircuitBreaker::new(FixedPoint::from_float(10.0)), // 10x volume spike
            pnl_circuit_breaker: PnLCircuitBreaker::new(FixedPoint::from_float(1000.0)), // $1000 loss
            time_circuit_breaker: TimeCircuitBreaker::new(3600), // 1 hour trading halt
        }
    }
    
    pub fn check_breakers(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
        current_time: u64,
    ) -> Option<CircuitBreakerAction> {
        // Check each circuit breaker
        if let Some(action) = self.price_circuit_breaker.check(market_state) {
            return Some(action);
        }
        
        if let Some(action) = self.volume_circuit_breaker.check(market_state) {
            return Some(action);
        }
        
        if let Some(action) = self.pnl_circuit_breaker.check(trading_state) {
            return Some(action);
        }
        
        if let Some(action) = self.time_circuit_breaker.check(current_time) {
            return Some(action);
        }
        
        None
    }
}

/// Price-based circuit breaker
pub struct PriceCircuitBreaker {
    pub price_move_threshold: FixedPoint,
    pub reference_price: Option<FixedPoint>,
    pub last_reset_time: u64,
}

impl PriceCircuitBreaker {
    pub fn new(threshold: FixedPoint) -> Self {
        Self {
            price_move_threshold: threshold,
            reference_price: None,
            last_reset_time: 0,
        }
    }
    
    pub fn check(&mut self, market_state: &MarketState) -> Option<CircuitBreakerAction> {
        if let Some(ref_price) = self.reference_price {
            let price_change = (market_state.mid_price - ref_price).abs() / ref_price;
            
            if price_change >= self.price_move_threshold {
                return Some(CircuitBreakerAction {
                    breaker_type: CircuitBreakerType::PriceMove,
                    trigger_value: price_change,
                    action: BreakerAction::HaltTrading,
                    duration_seconds: 300, // 5 minutes
                    reason: format!("Price moved {:.2}% from reference", price_change.to_float() * 100.0),
                });
            }
        } else {
            self.reference_price = Some(market_state.mid_price);
        }
        
        None
    }
}

/// Volume-based circuit breaker
pub struct VolumeCircuitBreaker {
    pub volume_spike_threshold: FixedPoint,
    pub average_volume: FixedPoint,
    pub volume_history: VecDeque<FixedPoint>,
}

impl VolumeCircuitBreaker {
    pub fn new(threshold: FixedPoint) -> Self {
        Self {
            volume_spike_threshold: threshold,
            average_volume: FixedPoint::from_float(1000.0),
            volume_history: VecDeque::with_capacity(100),
        }
    }
    
    pub fn check(&mut self, market_state: &MarketState) -> Option<CircuitBreakerAction> {
        let current_volume = market_state.bid_volume + market_state.ask_volume;
        
        // Update volume history
        self.volume_history.push_back(current_volume);
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }
        
        // Update average
        if !self.volume_history.is_empty() {
            self.average_volume = self.volume_history.iter().sum::<FixedPoint>() / 
                                 FixedPoint::from_float(self.volume_history.len() as f64);
        }
        
        // Check for volume spike
        if self.average_volume > FixedPoint::zero() {
            let volume_ratio = current_volume / self.average_volume;
            
            if volume_ratio >= self.volume_spike_threshold {
                return Some(CircuitBreakerAction {
                    breaker_type: CircuitBreakerType::VolumeSpike,
                    trigger_value: volume_ratio,
                    action: BreakerAction::ReduceQuoting,
                    duration_seconds: 60,
                    reason: format!("Volume spike: {:.1}x average", volume_ratio.to_float()),
                });
            }
        }
        
        None
    }
}

/// PnL-based circuit breaker
pub struct PnLCircuitBreaker {
    pub max_loss_threshold: FixedPoint,
    pub reference_pnl: Option<FixedPoint>,
}

impl PnLCircuitBreaker {
    pub fn new(threshold: FixedPoint) -> Self {
        Self {
            max_loss_threshold: threshold,
            reference_pnl: None,
        }
    }
    
    pub fn check(&mut self, trading_state: &TradingState) -> Option<CircuitBreakerAction> {
        if let Some(ref_pnl) = self.reference_pnl {
            let pnl_change = ref_pnl - trading_state.total_pnl;
            
            if pnl_change >= self.max_loss_threshold {
                return Some(CircuitBreakerAction {
                    breaker_type: CircuitBreakerType::PnLLoss,
                    trigger_value: pnl_change,
                    action: BreakerAction::HaltTrading,
                    duration_seconds: 1800, // 30 minutes
                    reason: format!("PnL loss exceeded: ${:.2}", pnl_change.to_float()),
                });
            }
        } else {
            self.reference_pnl = Some(trading_state.total_pnl);
        }
        
        None
    }
}

/// Time-based circuit breaker
pub struct TimeCircuitBreaker {
    pub max_trading_duration: u64,
    pub trading_start_time: Option<u64>,
    pub halt_until: Option<u64>,
}

impl TimeCircuitBreaker {
    pub fn new(max_duration: u64) -> Self {
        Self {
            max_trading_duration: max_duration,
            trading_start_time: None,
            halt_until: None,
        }
    }
    
    pub fn check(&mut self, current_time: u64) -> Option<CircuitBreakerAction> {
        // Check if we're in a halt period
        if let Some(halt_time) = self.halt_until {
            if current_time < halt_time {
                return Some(CircuitBreakerAction {
                    breaker_type: CircuitBreakerType::TimeHalt,
                    trigger_value: FixedPoint::from_float((halt_time - current_time) as f64),
                    action: BreakerAction::HaltTrading,
                    duration_seconds: halt_time - current_time,
                    reason: "Time-based trading halt in effect".to_string(),
                });
            } else {
                self.halt_until = None;
            }
        }
        
        // Check trading duration
        if let Some(start_time) = self.trading_start_time {
            if current_time - start_time >= self.max_trading_duration {
                self.halt_until = Some(current_time + 3600); // 1 hour halt
                return Some(CircuitBreakerAction {
                    breaker_type: CircuitBreakerType::TimeLimit,
                    trigger_value: FixedPoint::from_float((current_time - start_time) as f64),
                    action: BreakerAction::HaltTrading,
                    duration_seconds: 3600,
                    reason: "Maximum trading duration exceeded".to_string(),
                });
            }
        } else {
            self.trading_start_time = Some(current_time);
        }
        
        None
    }
}

/// Emergency procedures implementation
pub struct EmergencyProcedures {
    pub liquidation_engine: LiquidationEngine,
    pub order_cancellation: OrderCancellationEngine,
    pub position_reducer: PositionReducer,
}

impl EmergencyProcedures {
    pub fn new() -> Self {
        Self {
            liquidation_engine: LiquidationEngine::new(),
            order_cancellation: OrderCancellationEngine::new(),
            position_reducer: PositionReducer::new(),
        }
    }
    
    pub fn execute(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
        violation_type: &RiskViolation,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        let mut actions = Vec::new();
        
        match violation_type {
            RiskViolation::DrawdownViolation(_) => {
                // Immediate position liquidation
                actions.extend(self.liquidation_engine.emergency_liquidate(trading_state, market_state)?);
                actions.extend(self.order_cancellation.cancel_all_orders()?);
            }
            RiskViolation::PositionViolation(_) => {
                // Gradual position reduction
                actions.extend(self.position_reducer.reduce_position(trading_state, market_state)?);
            }
            RiskViolation::VolatilityViolation(_) => {
                // Halt trading temporarily
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::HaltQuoting,
                    size: FixedPoint::zero(),
                    max_price_impact: FixedPoint::zero(),
                    timeout_seconds: 300,
                    reason: "High volatility detected".to_string(),
                });
            }
            RiskViolation::LiquidityViolation(_) => {
                // Reduce quote sizes
                actions.extend(self.position_reducer.reduce_quote_sizes(market_state)?);
            }
            RiskViolation::CircuitBreakerTriggered(breaker_action) => {
                // Execute breaker-specific actions
                actions.extend(self.execute_breaker_actions(breaker_action)?);
            }
        }
        
        Ok(actions)
    }
    
    fn execute_breaker_actions(
        &self,
        breaker_action: &CircuitBreakerAction,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        let mut actions = Vec::new();
        
        match breaker_action.action {
            BreakerAction::HaltTrading => {
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::HaltQuoting,
                    size: FixedPoint::zero(),
                    max_price_impact: FixedPoint::zero(),
                    timeout_seconds: breaker_action.duration_seconds,
                    reason: breaker_action.reason.clone(),
                });
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::CancelAllOrders,
                    size: FixedPoint::zero(),
                    max_price_impact: FixedPoint::zero(),
                    timeout_seconds: 0,
                    reason: "Circuit breaker triggered".to_string(),
                });
            }
            BreakerAction::ReduceQuoting => {
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::ReducePositionSize,
                    size: FixedPoint::from_float(0.5),
                    max_price_impact: FixedPoint::from_float(0.005),
                    timeout_seconds: breaker_action.duration_seconds,
                    reason: breaker_action.reason.clone(),
                });
            }
        }
        
        Ok(actions)
    }
}

/// Liquidation engine for emergency situations
pub struct LiquidationEngine {
    pub max_liquidation_rate: FixedPoint,
    pub max_price_impact: FixedPoint,
}

impl LiquidationEngine {
    pub fn new() -> Self {
        Self {
            max_liquidation_rate: FixedPoint::from_float(0.5), // 50% per minute
            max_price_impact: FixedPoint::from_float(0.02),    // 2% max impact
        }
    }
    
    pub fn emergency_liquidate(
        &self,
        trading_state: &TradingState,
        market_state: &MarketState,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        let mut actions = Vec::new();
        let inventory = trading_state.inventory;
        
        if inventory.abs() > FixedPoint::zero() {
            let liquidation_size = inventory.abs() * self.max_liquidation_rate;
            
            if inventory > FixedPoint::zero() {
                // Liquidate long position
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::MarketSell,
                    size: liquidation_size,
                    max_price_impact: self.max_price_impact,
                    timeout_seconds: 60,
                    reason: "Emergency liquidation - long position".to_string(),
                });
            } else {
                // Liquidate short position
                actions.push(EmergencyAction {
                    action_type: EmergencyActionType::MarketBuy,
                    size: liquidation_size,
                    max_price_impact: self.max_price_impact,
                    timeout_seconds: 60,
                    reason: "Emergency liquidation - short position".to_string(),
                });
            }
        }
        
        Ok(actions)
    }
}

/// Order cancellation engine
pub struct OrderCancellationEngine;

impl OrderCancellationEngine {
    pub fn new() -> Self {
        Self
    }
    
    pub fn cancel_all_orders(&self) -> Result<Vec<EmergencyAction>, QuotingError> {
        Ok(vec![EmergencyAction {
            action_type: EmergencyActionType::CancelAllOrders,
            size: FixedPoint::zero(),
            max_price_impact: FixedPoint::zero(),
            timeout_seconds: 5,
            reason: "Emergency order cancellation".to_string(),
        }])
    }
}

/// Position reducer for gradual risk reduction
pub struct PositionReducer;

impl PositionReducer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn reduce_position(
        &self,
        trading_state: &TradingState,
        _market_state: &MarketState,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        let mut actions = Vec::new();
        let reduction_size = trading_state.inventory.abs() * FixedPoint::from_float(0.3);
        
        if trading_state.inventory > FixedPoint::zero() {
            actions.push(EmergencyAction {
                action_type: EmergencyActionType::MarketSell,
                size: reduction_size,
                max_price_impact: FixedPoint::from_float(0.01),
                timeout_seconds: 120,
                reason: "Gradual position reduction".to_string(),
            });
        } else if trading_state.inventory < FixedPoint::zero() {
            actions.push(EmergencyAction {
                action_type: EmergencyActionType::MarketBuy,
                size: reduction_size,
                max_price_impact: FixedPoint::from_float(0.01),
                timeout_seconds: 120,
                reason: "Gradual position reduction".to_string(),
            });
        }
        
        Ok(actions)
    }
    
    pub fn reduce_quote_sizes(
        &self,
        _market_state: &MarketState,
    ) -> Result<Vec<EmergencyAction>, QuotingError> {
        Ok(vec![EmergencyAction {
            action_type: EmergencyActionType::ReducePositionSize,
            size: FixedPoint::from_float(0.5),
            max_price_impact: FixedPoint::zero(),
            timeout_seconds: 300,
            reason: "Reduce quote sizes due to low liquidity".to_string(),
        }])
    }
}

/// Risk metrics computation and tracking
pub struct RiskMetrics {
    pub var_calculator: VaRCalculator,
    pub sharpe_calculator: SharpeCalculator,
    pub correlation_risk: FixedPoint,
    pub liquidity_risk: FixedPoint,
    pub overall_risk_score: FixedPoint,
}

impl RiskMetrics {
    pub fn new() -> Self {
        Self {
            var_calculator: VaRCalculator::new(),
            sharpe_calculator: SharpeCalculator::new(),
            correlation_risk: FixedPoint::zero(),
            liquidity_risk: FixedPoint::zero(),
            overall_risk_score: FixedPoint::zero(),
        }
    }
    
    pub fn update(
        &mut self,
        trading_state: &TradingState,
        market_state: &MarketState,
        timestamp: u64,
    ) {
        // Update VaR calculation
        self.var_calculator.update(trading_state.total_pnl, timestamp);
        
        // Update Sharpe ratio
        self.sharpe_calculator.update(trading_state.total_pnl, timestamp);
        
        // Update liquidity risk
        self.liquidity_risk = self.compute_liquidity_risk(market_state);
        
        // Compute overall risk score
        self.overall_risk_score = self.compute_overall_risk_score();
    }
    
    pub fn compute_overall_risk_score(&self) -> FixedPoint {
        let var_component = self.var_calculator.get_current_var() * FixedPoint::from_float(0.4);
        let sharpe_component = (FixedPoint::one() / (FixedPoint::one() + self.sharpe_calculator.get_current_sharpe().abs())) * FixedPoint::from_float(0.3);
        let liquidity_component = self.liquidity_risk * FixedPoint::from_float(0.3);
        
        var_component + sharpe_component + liquidity_component
    }
    
    fn compute_liquidity_risk(&self, market_state: &MarketState) -> FixedPoint {
        let total_depth = market_state.bid_volume + market_state.ask_volume;
        let spread_risk = market_state.spread / market_state.mid_price;
        
        if total_depth > FixedPoint::zero() {
            spread_risk + FixedPoint::one() / total_depth
        } else {
            FixedPoint::one()
        }
    }
}

/// Value at Risk calculator
pub struct VaRCalculator {
    pub returns_history: VecDeque<FixedPoint>,
    pub confidence_level: FixedPoint,
    pub current_var: FixedPoint,
}

impl VaRCalculator {
    pub fn new() -> Self {
        Self {
            returns_history: VecDeque::with_capacity(252), // One year of daily returns
            confidence_level: FixedPoint::from_float(0.95),
            current_var: FixedPoint::zero(),
        }
    }
    
    pub fn update(&mut self, pnl: FixedPoint, _timestamp: u64) {
        // Add return to history (simplified - should compute actual returns)
        self.returns_history.push_back(pnl);
        if self.returns_history.len() > 252 {
            self.returns_history.pop_front();
        }
        
        // Compute VaR using historical simulation
        if self.returns_history.len() >= 30 {
            let mut sorted_returns: Vec<FixedPoint> = self.returns_history.iter().cloned().collect();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let var_index = ((1.0 - self.confidence_level.to_float()) * sorted_returns.len() as f64) as usize;
            self.current_var = sorted_returns[var_index].abs();
        }
    }
    
    pub fn get_current_var(&self) -> FixedPoint {
        self.current_var
    }
}

/// Sharpe ratio calculator
pub struct SharpeCalculator {
    pub returns_history: VecDeque<FixedPoint>,
    pub current_sharpe: FixedPoint,
    pub risk_free_rate: FixedPoint,
}

impl SharpeCalculator {
    pub fn new() -> Self {
        Self {
            returns_history: VecDeque::with_capacity(252),
            current_sharpe: FixedPoint::zero(),
            risk_free_rate: FixedPoint::from_float(0.02), // 2% annual
        }
    }
    
    pub fn update(&mut self, pnl: FixedPoint, _timestamp: u64) {
        self.returns_history.push_back(pnl);
        if self.returns_history.len() > 252 {
            self.returns_history.pop_front();
        }
        
        if self.returns_history.len() >= 30 {
            let mean_return = self.returns_history.iter().sum::<FixedPoint>() / 
                             FixedPoint::from_float(self.returns_history.len() as f64);
            
            let variance = self.returns_history.iter()
                .map(|&r| {
                    let diff = r - mean_return;
                    diff * diff
                })
                .sum::<FixedPoint>() / FixedPoint::from_float(self.returns_history.len() as f64);
            
            let std_dev = variance.sqrt();
            
            if std_dev > FixedPoint::zero() {
                self.current_sharpe = (mean_return - self.risk_free_rate) / std_dev;
            }
        }
    }
    
    pub fn get_current_sharpe(&self) -> FixedPoint {
        self.current_sharpe
    }
}

/// Alert system for risk notifications
pub struct AlertSystem {
    pub alert_channels: Vec<AlertChannel>,
    pub alert_history: VecDeque<(u64, String)>,
    pub alert_cooldown: HashMap<String, u64>,
}

impl AlertSystem {
    pub fn new() -> Self {
        Self {
            alert_channels: vec![AlertChannel::Console, AlertChannel::Log],
            alert_history: VecDeque::with_capacity(1000),
            alert_cooldown: HashMap::new(),
        }
    }
    
    pub fn send_risk_alerts(
        &mut self,
        warnings: &[RiskWarning],
        violations: &[RiskViolation],
        timestamp: u64,
    ) {
        for warning in warnings {
            let message = format!("RISK WARNING: {:?}", warning);
            self.send_alert(AlertLevel::Warning, &message, timestamp);
        }
        
        for violation in violations {
            let message = format!("RISK VIOLATION: {:?}", violation);
            self.send_alert(AlertLevel::Critical, &message, timestamp);
        }
    }
    
    fn send_alert(&mut self, level: AlertLevel, message: &str, timestamp: u64) {
        // Check cooldown
        let cooldown_key = format!("{:?}_{}", level, message);
        if let Some(&last_sent) = self.alert_cooldown.get(&cooldown_key) {
            if timestamp - last_sent < 60 { // 1 minute cooldown
                return;
            }
        }
        
        // Send to all channels
        for channel in &self.alert_channels {
            match channel {
                AlertChannel::Console => {
                    println!("[{}] {}: {}", timestamp, level.as_str(), message);
                }
                AlertChannel::Log => {
                    // Would integrate with logging system
                    eprintln!("[{}] {}: {}", timestamp, level.as_str(), message);
                }
                AlertChannel::Email => {
                    // Would integrate with email system
                }
                AlertChannel::SMS => {
                    // Would integrate with SMS system
                }
                AlertChannel::Slack => {
                    // Would integrate with Slack API
                }
            }
        }
        
        // Update history and cooldown
        self.alert_history.push_back((timestamp, message.to_string()));
        if self.alert_history.len() > 1000 {
            self.alert_history.pop_front();
        }
        
        self.alert_cooldown.insert(cooldown_key, timestamp);
    }
}

// Supporting types and enums
#[derive(Debug, Clone)]
pub enum RiskStatus {
    Normal,
    Warning,
    Violation,
}

#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub status: RiskStatus,
    pub warnings: Vec<RiskWarning>,
    pub violations: Vec<RiskViolation>,
    pub recommended_actions: Vec<RiskAction>,
    pub risk_score: FixedPoint,
}

#[derive(Debug, Clone)]
pub enum RiskWarning {
    DrawdownWarning(FixedPoint),
    PositionWarning(FixedPoint),
    VolatilityWarning(FixedPoint),
    LiquidityWarning(FixedPoint),
}

#[derive(Debug, Clone)]
pub enum RiskViolation {
    DrawdownViolation(String),
    PositionViolation(String),
    VolatilityViolation(String),
    LiquidityViolation(String),
    CircuitBreakerTriggered(CircuitBreakerAction),
}

#[derive(Debug, Clone)]
pub enum RiskAction {
    ReducePositionSize(FixedPoint),
    WidenSpreads(FixedPoint),
    ReduceQuoteSizes(FixedPoint),
    HaltTrading,
    EmergencyLiquidation,
    CancelAllOrders,
}

#[derive(Debug, Clone)]
pub enum DrawdownStatus {
    Normal,
    Warning(FixedPoint),
}

#[derive(Debug, Clone)]
pub enum PositionStatus {
    Normal,
    Warning(FixedPoint),
}

#[derive(Debug, Clone)]
pub enum VolatilityStatus {
    Normal,
    Warning(FixedPoint),
}

#[derive(Debug, Clone)]
pub enum LiquidityStatus {
    Normal,
    Warning(FixedPoint),
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerAction {
    pub breaker_type: CircuitBreakerType,
    pub trigger_value: FixedPoint,
    pub action: BreakerAction,
    pub duration_seconds: u64,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum CircuitBreakerType {
    PriceMove,
    VolumeSpike,
    PnLLoss,
    TimeHalt,
    TimeLimit,
}

#[derive(Debug, Clone)]
pub enum BreakerAction {
    HaltTrading,
    ReduceQuoting,
}

#[derive(Debug, Clone)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

impl AlertLevel {
    fn as_str(&self) -> &'static str {
        match self {
            AlertLevel::Info => "INFO",
            AlertLevel::Warning => "WARNING",
            AlertLevel::Critical => "CRITICAL",
        }
    }
}

#[derive(Debug, Clone)]
pub enum AlertChannel {
    Console,
    Log,
    Email,
    SMS,
    Slack,
}