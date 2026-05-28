use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::{HashMap, BTreeMap};
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order},
    optimal_trading::{OptimalController, MarketImpact},
};

#[derive(Debug, Error)]
pub enum MicrostructureError {
    #[error("Tick size error: {0}")]
    TickSizeError(String),
    #[error("Queue position error: {0}")]
    QueueError(String),
    #[error("Order timing error: {0}")]
    TimingError(String),
}

/// Tick size grid for price discretization
#[derive(Debug, Clone)]
pub struct TickSizeGrid {
    pub base_tick: f64,
    pub price_levels: BTreeMap<f64, f64>,
    pub relative_ticks: bool,
}

impl TickSizeGrid {
    pub fn new(base_tick: f64, relative_ticks: bool) -> Self {
        Self {
            base_tick,
            price_levels: BTreeMap::new(),
            relative_ticks,
        }
    }

    pub fn add_price_level(&mut self, price: f64, tick_size: f64) {
        self.price_levels.insert(price, tick_size);
    }

    pub fn get_tick_size(&self, price: f64) -> f64 {
        if self.relative_ticks {
            let tick_size = self.price_levels
                .range(..=price)
                .next_back()
                .map(|(_, &tick)| tick)
                .unwrap_or(self.base_tick);
            price * tick_size
        } else {
            self.price_levels
                .range(..=price)
                .next_back()
                .map(|(_, &tick)| tick)
                .unwrap_or(self.base_tick)
        }
    }

    pub fn round_to_tick(&self, price: f64) -> f64 {
        let tick_size = self.get_tick_size(price);
        (price / tick_size).round() * tick_size
    }
}

/// Queue position value model
#[derive(Debug)]
pub struct QueuePositionModel {
    pub value_curve: Box<dyn QueueValueFunction>,
    pub position_weights: Vec<f64>,
    pub cancellation_rates: Vec<f64>,
    pub execution_probabilities: Vec<f64>,
}

/// Queue value function trait
pub trait QueueValueFunction: Send + Sync {
    fn compute_value(&self, position: usize, queue_size: usize) -> f64;
    fn compute_gradient(&self, position: usize, queue_size: usize) -> f64;
}

/// Exponential queue value function
#[derive(Debug)]
pub struct ExponentialQueueValue {
    decay_rate: f64,
    scale_factor: f64,
}

impl QueueValueFunction for ExponentialQueueValue {
    fn compute_value(&self, position: usize, queue_size: usize) -> f64 {
        if queue_size == 0 {
            return 0.0;
        }
        let relative_pos = position as f64 / queue_size as f64;
        self.scale_factor * (-self.decay_rate * relative_pos).exp()
    }

    fn compute_gradient(&self, position: usize, queue_size: usize) -> f64 {
        if queue_size == 0 {
            return 0.0;
        }
        let relative_pos = position as f64 / queue_size as f64;
        -self.decay_rate * self.scale_factor * (-self.decay_rate * relative_pos).exp() 
            / queue_size as f64
    }
}

/// Order timing optimizer
#[derive(Debug)]
pub struct OrderTimingOptimizer {
    pub arrival_rates: Vec<f64>,
    pub execution_costs: Vec<f64>,
    pub cancellation_costs: Vec<f64>,
    pub time_grid: Vec<f64>,
}

impl OrderTimingOptimizer {
    pub fn new(
        n_periods: usize,
        horizon: f64,
    ) -> Self {
        let time_grid: Vec<f64> = (0..=n_periods)
            .map(|i| i as f64 * horizon / n_periods as f64)
            .collect();
            
        Self {
            arrival_rates: vec![0.0; n_periods],
            execution_costs: vec![0.0; n_periods],
            cancellation_costs: vec![0.0; n_periods],
            time_grid,
        }
    }

    pub fn optimize_timing(
        &self,
        current_time: f64,
        order_size: f64,
    ) -> Result<OrderTimingStrategy, MicrostructureError> {
        let mut optimal_times = Vec::new();
        let mut optimal_sizes = Vec::new();
        
        // Dynamic programming solution
        let n_periods = self.time_grid.len() - 1;
        let mut value_function = vec![0.0; n_periods + 1];
        let mut policy = vec![0.0; n_periods];
        
        // Backward induction
        for t in (0..n_periods).rev() {
            let dt = self.time_grid[t + 1] - self.time_grid[t];
            let arrival_rate = self.arrival_rates[t];
            
            // Compute optimal order size for period
            let optimal_size = self.compute_optimal_size(
                order_size,
                arrival_rate,
                self.execution_costs[t],
                dt,
            )?;
            
            policy[t] = optimal_size;
            value_function[t] = self.compute_period_value(
                optimal_size,
                arrival_rate,
                self.execution_costs[t],
                self.cancellation_costs[t],
                dt,
            )?;
        }
        
        // Forward simulation
        let mut remaining_size = order_size;
        for t in 0..n_periods {
            if remaining_size > 0.0 {
                let size = policy[t].min(remaining_size);
                if size > 0.0 {
                    optimal_times.push(self.time_grid[t]);
                    optimal_sizes.push(size);
                    remaining_size -= size;
                }
            }
        }
        
        Ok(OrderTimingStrategy {
            execution_times: optimal_times,
            order_sizes: optimal_sizes,
        })
    }

    fn compute_optimal_size(
        &self,
        remaining_size: f64,
        arrival_rate: f64,
        execution_cost: f64,
        dt: f64,
    ) -> Result<f64, MicrostructureError> {
        // Optimal size based on arrival rate and costs
        let optimal_size = remaining_size * arrival_rate * dt 
            / (1.0 + execution_cost * arrival_rate * dt);
            
        Ok(optimal_size.min(remaining_size))
    }

    fn compute_period_value(
        &self,
        size: f64,
        arrival_rate: f64,
        execution_cost: f64,
        cancellation_cost: f64,
        dt: f64,
    ) -> Result<f64, MicrostructureError> {
        let execution_probability = 1.0 - (-arrival_rate * dt).exp();
        let expected_cost = execution_probability * execution_cost * size
            + (1.0 - execution_probability) * cancellation_cost * size;
            
        Ok(-expected_cost)
    }
}

#[derive(Debug)]
pub struct OrderTimingStrategy {
    pub execution_times: Vec<f64>,
    pub order_sizes: Vec<f64>,
}

/// Microstructure features aggregator
pub struct MicrostructureFeatures {
    pub tick_grid: TickSizeGrid,
    pub queue_value: QueuePositionModel,
    pub order_timing: OrderTimingOptimizer,
}

impl MicrostructureFeatures {
    pub fn new(
        tick_grid: TickSizeGrid,
        queue_value: QueuePositionModel,
        order_timing: OrderTimingOptimizer,
    ) -> Self {
        Self {
            tick_grid,
            queue_value,
            order_timing,
        }
    }

    pub fn compute_queue_priority(
        &self,
        price_level: f64,
        position: usize,
        queue_size: usize,
    ) -> Result<f64, MicrostructureError> {
        if position >= queue_size {
            return Err(MicrostructureError::QueueError(
                "Position exceeds queue size".to_string(),
            ));
        }
        
        let value = self.queue_value.value_curve.compute_value(position, queue_size);
        let gradient = self.queue_value.value_curve.compute_gradient(position, queue_size);
        
        let cancellation_rate = self.queue_value.cancellation_rates[
            position.min(self.queue_value.cancellation_rates.len() - 1)
        ];
        
        let execution_prob = self.queue_value.execution_probabilities[
            position.min(self.queue_value.execution_probabilities.len() - 1)
        ];
        
        let priority = value * (1.0 - cancellation_rate) * execution_prob;
        
        Ok(priority)
    }

    pub fn optimize_order_placement(
        &self,
        price: f64,
        size: f64,
        urgency: f64,
    ) -> Result<OrderPlacementStrategy, MicrostructureError> {
        let tick_size = self.tick_grid.get_tick_size(price);
        let rounded_price = self.tick_grid.round_to_tick(price);
        
        // Split order across price levels based on urgency
        let n_levels = (5.0 * (1.0 - urgency)).ceil() as usize;
        let mut price_levels = Vec::with_capacity(n_levels);
        let mut sizes = Vec::with_capacity(n_levels);
        
        let base_size = size / n_levels as f64;
        for i in 0..n_levels {
            let level_price = rounded_price + (i as f64) * tick_size;
            let level_size = base_size * (1.0 - 0.1 * i as f64);
            
            price_levels.push(level_price);
            sizes.push(level_size);
        }
        
        // Optimize timing for each level
        let mut timing_strategies = Vec::with_capacity(n_levels);
        for (price, size) in price_levels.iter().zip(sizes.iter()) {
            let timing = self.order_timing.optimize_timing(0.0, *size)?;
            timing_strategies.push(timing);
        }
        
        Ok(OrderPlacementStrategy {
            price_levels,
            sizes,
            timing_strategies,
        })
    }
}

#[derive(Debug)]
pub struct OrderPlacementStrategy {
    pub price_levels: Vec<f64>,
    pub sizes: Vec<f64>,
    pub timing_strategies: Vec<OrderTimingStrategy>,
}

use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::state_space::StateSpaceModel;

#[derive(Debug, Error)]
pub enum MicrostructureError {
    #[error("Microstructure error: {0}")]
    MicrostructureError(String),
    #[error("Data error: {0}")]
    DataError(String),
}

pub struct MarketMicrostructure {
    tick_size: f64,
    min_price_increment: f64,
    inventory_limit: f64,
    risk_aversion: f64,
    adverse_selection: f64,
    microstructure_features: MicrostructureFeatures,
}

impl MarketMicrostructure {
    pub fn new(
        tick_size: f64,
        min_price_increment: f64,
        inventory_limit: f64,
        risk_aversion: f64,
        adverse_selection: f64,
        microstructure_features: MicrostructureFeatures,
    ) -> Result<Self, MicrostructureError> {
        if tick_size <= 0.0 || min_price_increment <= 0.0 {
            return Err(MicrostructureError::MicrostructureError(
                "Price parameters must be positive".to_string(),
            ));
        }

        Ok(Self {
            tick_size,
            min_price_increment,
            inventory_limit,
            risk_aversion,
            adverse_selection,
            microstructure_features,
        })
    }

    pub fn analyze_order_flow(
        &self,
        prices: &[f64],
        volumes: &[f64],
        trade_signs: &[i8],
        timestamps: &[i64],
    ) -> Result<OrderFlowAnalysis, MicrostructureError> {
        let n = prices.len();
        if n != volumes.len() || n != trade_signs.len() || n != timestamps.len() {
            return Err(MicrostructureError::DataError(
                "Input vectors must have same length".to_string(),
            ));
        }

        // Compute order flow imbalance
        let imbalance = self.compute_order_imbalance(volumes, trade_signs)?;
        
        // Estimate price impact
        let price_impact = self.estimate_price_impact(
            prices,
            volumes,
            trade_signs,
            timestamps,
        )?;

        // Compute effective spreads
        let effective_spreads = self.compute_effective_spreads(
            prices,
            volumes,
            trade_signs,
        )?;

        // Analyze inventory effects
        let inventory = self.analyze_inventory_effects(
            volumes,
            trade_signs,
            timestamps,
        )?;

        Ok(OrderFlowAnalysis {
            order_imbalance: imbalance,
            price_impact,
            effective_spreads,
            inventory_analysis: inventory,
        })
    }

    fn compute_order_imbalance(
        &self,
        volumes: &[f64],
        trade_signs: &[i8],
    ) -> Result<OrderImbalance, MicrostructureError> {
        let n = volumes.len();
        
        // Compute signed volume
        let signed_volume: Vec<f64> = volumes.iter()
            .zip(trade_signs.iter())
            .map(|(&v, &s)| v * s as f64)
            .collect();

        // Compute cumulative imbalance
        let cumulative: Vec<f64> = signed_volume.iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc)
            })
            .collect();

        // Compute rolling statistics
        let window_sizes = vec![5, 10, 20];
        let mut rolling_stats = HashMap::new();

        for &window in &window_sizes {
            let means: Vec<f64> = signed_volume.windows(window)
                .map(|w| w.iter().sum::<f64>() / window as f64)
                .collect();
            
            let stds: Vec<f64> = signed_volume.windows(window)
                .map(|w| {
                    let mean = w.iter().sum::<f64>() / window as f64;
                    (w.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() 
                        / (window - 1) as f64).sqrt()
                })
                .collect();

            rolling_stats.insert(window, (means, stds));
        }

        Ok(OrderImbalance {
            signed_volume,
            cumulative_imbalance: cumulative,
            rolling_statistics: rolling_stats,
        })
    }

    fn estimate_price_impact(
        &self,
        prices: &[f64],
        volumes: &[f64],
        trade_signs: &[i8],
        timestamps: &[i64],
    ) -> Result<PriceImpact, MicrostructureError> {
        let n = prices.len();
        
        // Compute returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        // Compute temporary and permanent impact
        let mut temporary_impact = Vec::with_capacity(n - 1);
        let mut permanent_impact = Vec::with_capacity(n - 1);
        
        for i in 0..(n-1) {
            // Temporary impact: price reversal after trade
            let temp = if i < n-2 {
                (prices[i+1] - prices[i]) * trade_signs[i] as f64
            } else {
                0.0
            };
            temporary_impact.push(temp);

            // Permanent impact: sustained price change
            let window_size = 5.min(n - i - 1);
            let perm = if window_size > 0 {
                let future_price = prices[i + window_size];
                (future_price - prices[i]) * trade_signs[i] as f64
            } else {
                0.0
            };
            permanent_impact.push(perm);
        }

        // Estimate Kyle's lambda
        let signed_volume: Vec<f64> = volumes.iter()
            .zip(trade_signs.iter())
            .map(|(&v, &s)| v * s as f64)
            .collect();

        let (lambda, lambda_std_err) = self.estimate_kyle_lambda(
            &returns,
            &signed_volume,
        )?;

        Ok(PriceImpact {
            temporary_impact,
            permanent_impact,
            kyle_lambda: lambda,
            lambda_std_error: lambda_std_err,
        })
    }

    fn compute_effective_spreads(
        &self,
        prices: &[f64],
        volumes: &[f64],
        trade_signs: &[i8],
    ) -> Result<EffectiveSpreads, MicrostructureError> {
        let n = prices.len();
        
        // Compute VWAP for each trade
        let vwap: Vec<f64> = prices.iter()
            .zip(volumes.iter())
            .scan((0.0, 0.0), |acc, (&p, &v)| {
                acc.0 += p * v;
                acc.1 += v;
                Some(acc.0 / acc.1)
            })
            .collect();

        // Compute effective spreads
        let mut spreads = Vec::with_capacity(n);
        let mut realized_spreads = Vec::with_capacity(n);
        let mut adverse_selection = Vec::with_capacity(n);

        for i in 0..n {
            // Effective spread
            let spread = 2.0 * (prices[i] - vwap[i]).abs();
            spreads.push(spread);

            // Realized spread and adverse selection
            if i < n-1 {
                let midpoint = (prices[i] + prices[i+1]) / 2.0;
                let realized = 2.0 * trade_signs[i] as f64 * (prices[i] - midpoint);
                let selection = 2.0 * trade_signs[i] as f64 * (midpoint - vwap[i]);
                
                realized_spreads.push(realized);
                adverse_selection.push(selection);
            }
        }

        // Compute spread components
        let components = self.decompose_spread_components(
            &spreads,
            &realized_spreads,
            &adverse_selection,
        )?;

        Ok(EffectiveSpreads {
            spreads,
            realized_spreads,
            adverse_selection,
            components,
        })
    }

    fn analyze_inventory_effects(
        &self,
        volumes: &[f64],
        trade_signs: &[i8],
        timestamps: &[i64],
    ) -> Result<InventoryAnalysis, MicrostructureError> {
        let n = volumes.len();
        
        // Compute inventory path
        let inventory: Vec<f64> = volumes.iter()
            .zip(trade_signs.iter())
            .scan(0.0, |acc, (&v, &s)| {
                *acc += v * s as f64;
                Some(*acc)
            })
            .collect();

        // Compute inventory holding times
        let mut holding_times = Vec::new();
        let mut current_position = 0.0;
        let mut position_start = timestamps[0];

        for i in 0..n {
            let new_position = inventory[i];
            if (new_position * current_position) <= 0.0 && current_position != 0.0 {
                // Position crossed zero, record holding time
                holding_times.push(timestamps[i] - position_start);
                position_start = timestamps[i];
            }
            current_position = new_position;
        }

        // Compute inventory metrics
        let mean_inventory = inventory.iter().sum::<f64>() / n as f64;
        let inventory_volatility = (inventory.iter()
            .map(|&x| (x - mean_inventory).powi(2))
            .sum::<f64>() / (n - 1) as f64)
            .sqrt();

        let mean_holding_time = holding_times.iter().sum::<i64>() / holding_times.len() as i64;
        let max_inventory = inventory.iter().fold(0.0, |a, &b| a.max(b.abs()));

        // Analyze inventory control
        let control = self.analyze_inventory_control(&inventory)?;

        Ok(InventoryAnalysis {
            inventory_path: inventory,
            mean_inventory,
            inventory_volatility,
            mean_holding_time,
            max_inventory,
            holding_times,
            control_analysis: control,
        })
    }

    fn estimate_kyle_lambda(
        &self,
        returns: &[f64],
        signed_volume: &[f64],
    ) -> Result<(f64, f64), MicrostructureError> {
        let n = returns.len();
        if n != signed_volume.len() {
            return Err(MicrostructureError::DataError(
                "Returns and signed volume must have same length".to_string(),
            ));
        }

        // Create design matrix for regression
        let mut X = na::DMatrix::zeros(n, 2);
        for i in 0..n {
            X[(i, 0)] = 1.0;
            X[(i, 1)] = signed_volume[i];
        }

        let y = na::DVector::from_vec(returns.to_vec());

        // Compute OLS estimates
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let beta = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(MicrostructureError::MicrostructureError(
                "Singular matrix in Kyle's lambda estimation".to_string(),
            )),
        };

        // Compute standard error
        let residuals = y - &X * &beta;
        let sigma2 = residuals.dot(&residuals) / (n - 2) as f64;
        let var_beta = sigma2 * match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv,
            None => return Err(MicrostructureError::MicrostructureError(
                "Singular matrix in variance computation".to_string(),
            )),
        };

        Ok((beta[1], var_beta[(1, 1)].sqrt()))
    }

    fn decompose_spread_components(
        &self,
        spreads: &[f64],
        realized_spreads: &[f64],
        adverse_selection: &[f64],
    ) -> Result<SpreadComponents, MicrostructureError> {
        let n = spreads.len();
        
        // Compute component means
        let mean_spread = spreads.iter().sum::<f64>() / n as f64;
        let mean_realized = realized_spreads.iter().sum::<f64>() / realized_spreads.len() as f64;
        let mean_adverse = adverse_selection.iter().sum::<f64>() / adverse_selection.len() as f64;

        // Compute component proportions
        let order_processing = mean_realized / mean_spread;
        let information = mean_adverse / mean_spread;
        let inventory = 1.0 - order_processing - information;

        Ok(SpreadComponents {
            order_processing_component: order_processing,
            adverse_selection_component: information,
            inventory_component: inventory,
        })
    }

    fn analyze_inventory_control(
        &self,
        inventory: &[f64],
    ) -> Result<InventoryControl, MicrostructureError> {
        let n = inventory.len();
        
        // Compute mean reversion speed
        let (speed, speed_std_err) = self.estimate_mean_reversion(inventory)?;

        // Compute position limits
        let upper_limit = inventory.iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lower_limit = inventory.iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Analyze position targeting
        let target = 0.0; // Assume market makers target zero inventory
        let targeting_error = inventory.iter()
            .map(|&x| (x - target).powi(2))
            .sum::<f64>() / n as f64;

        Ok(InventoryControl {
            mean_reversion_speed: speed,
            speed_std_error: speed_std_err,
            upper_position_limit: upper_limit,
            lower_position_limit: lower_limit,
            targeting_error,
        })
    }

    fn estimate_mean_reversion(
        &self,
        inventory: &[f64],
    ) -> Result<(f64, f64), MicrostructureError> {
        let n = inventory.len();
        if n < 2 {
            return Err(MicrostructureError::DataError(
                "Insufficient data for mean reversion estimation".to_string(),
            ));
        }

        // Estimate AR(1) coefficient
        let mut X = na::DMatrix::zeros(n-1, 2);
        let mut y = na::DVector::zeros(n-1);

        for i in 0..(n-1) {
            X[(i, 0)] = 1.0;
            X[(i, 1)] = inventory[i];
            y[i] = inventory[i+1];
        }

        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let beta = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(MicrostructureError::MicrostructureError(
                "Singular matrix in mean reversion estimation".to_string(),
            )),
        };

        // Compute standard error
        let residuals = y - &X * &beta;
        let sigma2 = residuals.dot(&residuals) / (n - 3) as f64;
        let var_beta = sigma2 * match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv,
            None => return Err(MicrostructureError::MicrostructureError(
                "Singular matrix in variance computation".to_string(),
            )),
        };

        // Convert AR coefficient to mean reversion speed
        let speed = 1.0 - beta[1];
        let speed_std_err = var_beta[(1, 1)].sqrt();

        Ok((speed, speed_std_err))
    }
}

#[derive(Debug)]
pub struct OrderImbalance {
    pub signed_volume: Vec<f64>,
    pub cumulative_imbalance: Vec<f64>,
    pub rolling_statistics: HashMap<usize, (Vec<f64>, Vec<f64>)>,
}

#[derive(Debug)]
pub struct PriceImpact {
    pub temporary_impact: Vec<f64>,
    pub permanent_impact: Vec<f64>,
    pub kyle_lambda: f64,
    pub lambda_std_error: f64,
}

#[derive(Debug)]
pub struct EffectiveSpreads {
    pub spreads: Vec<f64>,
    pub realized_spreads: Vec<f64>,
    pub adverse_selection: Vec<f64>,
    pub components: SpreadComponents,
}

#[derive(Debug)]
pub struct SpreadComponents {
    pub order_processing_component: f64,
    pub adverse_selection_component: f64,
    pub inventory_component: f64,
}

#[derive(Debug)]
pub struct InventoryAnalysis {
    pub inventory_path: Vec<f64>,
    pub mean_inventory: f64,
    pub inventory_volatility: f64,
    pub mean_holding_time: i64,
    pub max_inventory: f64,
    pub holding_times: Vec<i64>,
    pub control_analysis: InventoryControl,
}

#[derive(Debug)]
pub struct InventoryControl {
    pub mean_reversion_speed: f64,
    pub speed_std_error: f64,
    pub upper_position_limit: f64,
    pub lower_position_limit: f64,
    pub targeting_error: f64,
}

#[derive(Debug)]
pub struct OrderFlowAnalysis {
    pub order_imbalance: OrderImbalance,
    pub price_impact: PriceImpact,
    pub effective_spreads: EffectiveSpreads,
    pub inventory_analysis: InventoryAnalysis,
}
