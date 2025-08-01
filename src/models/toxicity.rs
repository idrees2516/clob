use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order},
    optimal_trading::{OptimalController, MarketImpact},
    microstructure::{MicrostructureFeatures, OrderTimingStrategy},
};

#[derive(Debug, Error)]
pub enum ToxicityError {
    #[error("VPIN calculation error: {0}")]
    VPINError(String),
    #[error("Order flow error: {0}")]
    OrderFlowError(String),
    #[error("Adverse selection error: {0}")]
    AdverseSelectionError(String),
}

/// Volume-synchronized Probability of Informed Trading
#[derive(Debug)]
pub struct VPIN {
    pub bucket_size: f64,
    pub num_buckets: usize,
    pub alpha: f64,
    pub buy_volumes: Vec<f64>,
    pub sell_volumes: Vec<f64>,
    pub total_volumes: Vec<f64>,
    pub vpin_values: Vec<f64>,
    pub last_price: f64,
}

impl VPIN {
    pub fn new(bucket_size: f64, num_buckets: usize, alpha: f64) -> Self {
        Self {
            bucket_size,
            num_buckets,
            alpha,
            buy_volumes: Vec::new(),
            sell_volumes: Vec::new(),
            total_volumes: Vec::new(),
            vpin_values: Vec::new(),
            last_price: 0.0,
        }
    }

    pub fn update(
        &mut self,
        volume: f64,
        price: f64,
        direction: i8,
    ) -> Result<f64, ToxicityError> {
        // Update volume buckets
        let current_bucket = self.total_volumes.len();
        
        if current_bucket == 0 || 
            self.total_volumes[current_bucket - 1] >= self.bucket_size {
            // Start new bucket with exponential decay factor
            let decay = (-self.alpha * self.bucket_size).exp();
            if current_bucket > 0 {
                self.buy_volumes.push(self.buy_volumes[current_bucket-1] * decay);
                self.sell_volumes.push(self.sell_volumes[current_bucket-1] * decay);
                self.total_volumes.push(self.total_volumes[current_bucket-1] * decay);
            } else {
                self.buy_volumes.push(0.0);
                self.sell_volumes.push(0.0); 
                self.total_volumes.push(0.0);
            }
        }
        
        let idx = self.total_volumes.len() - 1;

        // Apply volume classification using tick rule and volume profile
        let tick = if idx > 0 {
            (price - self.last_price).signum()
        } else {
            0.0
        };
        self.last_price = price;

        let buy_prob = if direction > 0 {
            0.8 + 0.2 * tick
        } else {
            0.2 + 0.2 * tick
        };

        self.buy_volumes[idx] += volume * buy_prob;
        self.sell_volumes[idx] += volume * (1.0 - buy_prob);
        self.total_volumes[idx] += volume;
        
        // Maintain rolling window
        if self.total_volumes.len() > self.num_buckets {
            self.buy_volumes.remove(0);
            self.sell_volumes.remove(0);
            self.total_volumes.remove(0);
        }
        
        // Compute VPIN with exponential weights
        if self.total_volumes.len() >= self.num_buckets {
            let vpin = self.compute_vpin_weighted()?;
            self.vpin_values.push(vpin);
            Ok(vpin)
        } else {
            Ok(0.0)
        }
    }

    fn compute_vpin_weighted(&self) -> Result<f64, ToxicityError> {
        let n = self.num_buckets;
        if self.buy_volumes.len() < n || self.sell_volumes.len() < n {
            return Err(ToxicityError::VPINError(
                "Insufficient data for VPIN calculation".to_string(),
            ));
        }
        
        let start_idx = self.buy_volumes.len() - n;
        let mut sum_imbalance = 0.0;
        let mut sum_volume = 0.0;
        let mut sum_weights = 0.0;
        
        for (i, idx) in (start_idx..self.buy_volumes.len()).enumerate() {
            let weight = (-self.alpha * (n - i - 1) as f64).exp();
            let imbalance = (self.buy_volumes[idx] - self.sell_volumes[idx]).abs();
            sum_imbalance += imbalance * weight;
            sum_volume += self.total_volumes[idx] * weight;
            sum_weights += weight;
        }
        
        if sum_volume == 0.0 {
            return Err(ToxicityError::VPINError(
                "Zero total volume".to_string(),
            ));
        }
        
        Ok(sum_imbalance / sum_volume)
    }
}

/// Order flow imbalance indicators
#[derive(Debug)]
pub struct OrderFlowImbalance {
    pub time_window: f64,
    pub decay_factor: f64,
    pub volume_imbalance: f64,
    pub tick_imbalance: f64,
    pub order_imbalance: f64,
    pub flow_toxicity: f64,
    pub last_update: f64,
}

impl OrderFlowImbalance {
    pub fn new(time_window: f64, decay_factor: f64) -> Self {
        Self {
            time_window,
            decay_factor,
            volume_imbalance: 0.0,
            tick_imbalance: 0.0,
            order_imbalance: 0.0,
            flow_toxicity: 0.0,
            last_update: 0.0,
        }
    }

    pub fn update(
        &mut self,
        volume: f64,
        price: f64,
        prev_price: f64,
        direction: i8,
        timestamp: f64,
    ) -> Result<f64, ToxicityError> {
        // Update volume imbalance with non-linear impact
        let volume_impact = volume.powf(0.6) * direction as f64;
        self.volume_imbalance = self.decay_factor * self.volume_imbalance +
            (1.0 - self.decay_factor) * volume_impact;
            
        // Update tick imbalance with price impact sensitivity
        let tick_size = 0.01; // Minimum price increment
        let price_change = price - prev_price;
        let normalized_change = (price_change / tick_size).round();
        let tick_impact = normalized_change.signum() * normalized_change.abs().powf(0.8);
        self.tick_imbalance = self.decay_factor * self.tick_imbalance +
            (1.0 - self.decay_factor) * tick_impact;
            
        // Update order imbalance with time decay
        let time_weight = (-0.1 * (timestamp - self.last_update)).exp();
        self.order_imbalance = self.decay_factor * self.order_imbalance * time_weight +
            (1.0 - self.decay_factor) * direction as f64;
        self.last_update = timestamp;
        
        // Compute flow toxicity with dynamic weights
        self.flow_toxicity = self.compute_toxicity_dynamic()?;
        
        Ok(self.flow_toxicity)
    }

    fn compute_toxicity_dynamic(&self) -> Result<f64, ToxicityError> {
        // Dynamic weight calculation based on market conditions
        let vol_weight = (self.volume_imbalance.abs() / 5.0).min(1.0);
        let tick_weight = (self.tick_imbalance.abs() / 3.0).min(1.0);
        let order_weight = (self.order_imbalance.abs() / 2.0).min(1.0);
        
        let total_weight = vol_weight + tick_weight + order_weight;
        
        if total_weight == 0.0 {
            return Ok(0.0);
        }
        
        // Normalized weighted average with non-linear transformation
        let toxicity = (
            vol_weight * self.volume_imbalance.abs().powf(1.2) +
            tick_weight * self.tick_imbalance.abs().powf(1.1) +
            order_weight * self.order_imbalance.abs()
        ) / total_weight;
        
        Ok(toxicity.min(1.0))
    }
}

/// Adverse selection measures
#[derive(Debug)]
pub struct AdverseSelection {
    pub estimation_window: usize,
    pub num_price_levels: usize,
    pub price_impact: Vec<f64>,
    pub selection_cost: Vec<f64>,
    pub realized_spread: Vec<f64>,
    pub effective_spread: Vec<f64>,
    pub time_since_last: f64,
}

impl AdverseSelection {
    pub fn new(estimation_window: usize, num_price_levels: usize) -> Self {
        Self {
            estimation_window,
            num_price_levels,
            price_impact: Vec::new(),
            selection_cost: Vec::new(),
            realized_spread: Vec::new(),
            effective_spread: Vec::new(),
            time_since_last: 0.0,
        }
    }

    pub fn update(
        &mut self,
        trade_price: f64,
        mid_price: f64,
        future_price: f64,
        direction: i8,
        volume: f64,
    ) -> Result<f64, ToxicityError> {
        // Compute spreads with volume-weighted adjustments
        let volume_factor = volume.powf(0.7);
        let effective_spread = 2.0 * direction as f64 * (trade_price - mid_price) * volume_factor;
        
        // Realized spread with time decay
        let time_weight = (-0.05 * self.time_since_last).exp();
        let realized_spread = 2.0 * direction as f64 * (trade_price - future_price) * 
            volume_factor * time_weight;
            
        // Price impact with permanent and temporary components
        let permanent_impact = 0.3 * effective_spread;
        let temporary_impact = effective_spread - realized_spread - permanent_impact;
        let price_impact = permanent_impact + temporary_impact * (-0.1 * self.time_since_last).exp();
        
        // Update measures with exponential weights
        self.effective_spread.push(effective_spread);
        self.realized_spread.push(realized_spread);
        self.price_impact.push(price_impact);
        
        // Compute selection cost with adaptive thresholds
        let selection_cost = self.compute_selection_cost_adaptive(
            &self.price_impact,
            volume,
            mid_price,
        )?;
        self.selection_cost.push(selection_cost);
        
        // Maintain window size with weighted averaging
        if self.effective_spread.len() > self.estimation_window {
            let decay = (-0.01 * self.estimation_window as f64).exp();
            self.effective_spread = self.effective_spread.iter()
                .skip(1)
                .map(|&x| x * decay)
                .collect();
            self.realized_spread = self.realized_spread.iter()
                .skip(1)
                .map(|&x| x * decay)
                .collect();
            self.price_impact = self.price_impact.iter()
                .skip(1)
                .map(|&x| x * decay)
                .collect();
            self.selection_cost = self.selection_cost.iter()
                .skip(1)
                .map(|&x| x * decay)
                .collect();
        }
        
        Ok(selection_cost)
    }

    fn compute_selection_cost_adaptive(
        &mut self,
        price_impacts: &[f64],
        volume: f64,
        mid_price: f64,
    ) -> Result<f64, ToxicityError> {
        if price_impacts.is_empty() {
            return Err(ToxicityError::AdverseSelectionError(
                "No price impact data available".to_string(),
            ));
        }
        
        // Compute adaptive thresholds using quantiles
        let mut sorted_impacts = price_impacts.to_vec();
        sorted_impacts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let q25 = sorted_impacts[sorted_impacts.len() / 4];
        let q75 = sorted_impacts[3 * sorted_impacts.len() / 4];
        let iqr = q75 - q25;
        
        let lower_bound = q25 - 1.5 * iqr;
        let upper_bound = q75 + 1.5 * iqr;
        
        // Filter outliers and compute weighted average
        let filtered_impacts: Vec<f64> = price_impacts.iter()
            .filter(|&&x| x >= lower_bound && x <= upper_bound)
            .cloned()
            .collect();
            
        if filtered_impacts.is_empty() {
            return Ok(0.0);
        }
        
        // Compute volume-weighted average with volatility adjustment
        let volatility = sorted_impacts.iter()
            .map(|&x| (x - mid_price).powi(2))
            .sum::<f64>()
            .sqrt() / sorted_impacts.len() as f64;
            
        let vol_factor = (1.0 + volatility / mid_price).powf(0.5);
        let avg_impact = filtered_impacts.iter().sum::<f64>() / filtered_impacts.len() as f64;
        
        Ok(avg_impact * volume * vol_factor)
    }
}

/// Toxicity metric aggregator
pub struct ToxicityMetrics {
    pub vpin: VPIN,
    pub flow_imbalance: OrderFlowImbalance,
    pub adverse_selection: AdverseSelection,
}

impl ToxicityMetrics {
    pub fn new(
        vpin_params: (f64, usize, f64),
        flow_params: (f64, f64),
        selection_params: (usize, usize),
    ) -> Self {
        Self {
            vpin: VPIN::new(vpin_params.0, vpin_params.1, vpin_params.2),
            flow_imbalance: OrderFlowImbalance::new(flow_params.0, flow_params.1),
            adverse_selection: AdverseSelection::new(
                selection_params.0,
                selection_params.1,
            ),
        }
    }

    pub fn update(
        &mut self,
        trade: &Trade,
        market_state: &MarketState,
    ) -> Result<ToxicityState, ToxicityError> {
        // Update VPIN
        let vpin = self.vpin.update(
            trade.volume,
            trade.price,
            trade.direction,
        )?;
        
        // Update flow imbalance
        let flow_toxicity = self.flow_imbalance.update(
            trade.volume,
            trade.price,
            market_state.prev_price,
            trade.direction,
            trade.timestamp,
        )?;
        
        // Update adverse selection
        let selection_cost = self.adverse_selection.update(
            trade.price,
            market_state.mid_price,
            market_state.future_price,
            trade.direction,
            trade.volume,
        )?;
        
        Ok(ToxicityState {
            vpin,
            flow_toxicity,
            selection_cost,
            timestamp: trade.timestamp,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub price: f64,
    pub volume: f64,
    pub direction: i8,
    pub timestamp: f64,
}

#[derive(Debug, Clone)]
pub struct MarketState {
    pub mid_price: f64,
    pub prev_price: f64,
    pub future_price: f64,
}

#[derive(Debug, Clone)]
pub struct ToxicityState {
    pub vpin: f64,
    pub flow_toxicity: f64,
    pub selection_cost: f64,
    pub timestamp: f64,
}
