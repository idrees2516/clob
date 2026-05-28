use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Normal, ContinuousCDF};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum MicrostructureError {
    #[error("Computation error: {0}")]
    ComputationError(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
    #[error("Queue error: {0}")]
    QueueError(String),
}

#[derive(Debug, Clone)]
pub struct OrderFlow {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub direction: i8,
    pub tick_size: f64,
}

pub struct MarketMicrostructure {
    tick_size: f64,
    min_price_increment: f64,
    inventory_limit: f64,
    risk_aversion: f64,
    information_horizon: usize,
}

impl MarketMicrostructure {
    pub fn new(
        tick_size: f64,
        min_price_increment: f64,
        inventory_limit: f64,
        risk_aversion: f64,
        information_horizon: usize,
    ) -> Result<Self, MicrostructureError> {
        if tick_size <= 0.0 || min_price_increment <= 0.0 {
            return Err(MicrostructureError::InvalidData(
                "Price increments must be positive".to_string(),
            ));
        }

        Ok(Self {
            tick_size,
            min_price_increment,
            inventory_limit,
            risk_aversion,
            information_horizon,
        })
    }

    pub fn analyze_order_flow_imbalance(
        &self,
        orders: &[OrderFlow],
        window_size: usize,
    ) -> Result<Vec<f64>, MicrostructureError> {
        if orders.is_empty() {
            return Err(MicrostructureError::InvalidData("Empty order flow".to_string()));
        }

        let imbalances: Vec<f64> = orders
            .windows(window_size)
            .map(|window| {
                let buy_volume: f64 = window
                    .iter()
                    .filter(|o| o.direction > 0)
                    .map(|o| o.volume)
                    .sum();
                let sell_volume: f64 = window
                    .iter()
                    .filter(|o| o.direction < 0)
                    .map(|o| o.volume)
                    .sum();
                
                (buy_volume - sell_volume) / (buy_volume + sell_volume)
            })
            .collect();

        Ok(imbalances)
    }

    pub fn estimate_price_impact(
        &self,
        orders: &[OrderFlow],
        market_orders: bool,
    ) -> Result<Vec<f64>, MicrostructureError> {
        let impacts = orders.windows(2)
            .map(|window| {
                let volume_signed = window[0].volume * window[0].direction as f64;
                let price_change = window[1].price - window[0].price;
                let tick_normalized_impact = price_change / self.tick_size;
                
                if market_orders {
                    tick_normalized_impact / volume_signed.abs().sqrt()
                } else {
                    tick_normalized_impact / volume_signed.abs()
                }
            })
            .collect();

        Ok(impacts)
    }

    pub fn calculate_effective_spread(
        &self,
        orders: &[OrderFlow],
        midpoint_prices: &[f64],
    ) -> Result<Vec<f64>, MicrostructureError> {
        if orders.len() != midpoint_prices.len() {
            return Err(MicrostructureError::InvalidData(
                "Order and midpoint price lengths must match".to_string(),
            ));
        }

        let spreads: Vec<f64> = orders.iter()
            .zip(midpoint_prices.iter())
            .map(|(order, &mid)| {
                2.0 * order.direction as f64 * (order.price - mid)
            })
            .collect();

        Ok(spreads)
    }

    pub fn estimate_inventory_impact(
        &self,
        orders: &[OrderFlow],
    ) -> Result<Vec<f64>, MicrostructureError> {
        let mut inventory = 0.0;
        let mut price_adjustments = Vec::with_capacity(orders.len());

        for order in orders {
            inventory += order.volume * order.direction as f64;
            
            // Non-linear inventory impact based on risk aversion
            let impact = self.risk_aversion * inventory * 
                (1.0 + (inventory.abs() / self.inventory_limit).powi(2));
            
            price_adjustments.push(impact);
            
            // Reset inventory if it exceeds limits
            if inventory.abs() > self.inventory_limit {
                inventory *= 0.5; // Mean reversion
            }
        }

        Ok(price_adjustments)
    }

    pub fn compute_adverse_selection(
        &self,
        orders: &[OrderFlow],
        future_window: usize,
    ) -> Result<Vec<f64>, MicrostructureError> {
        let mut selection_components = Vec::with_capacity(orders.len());
        
        for i in 0..orders.len().saturating_sub(future_window) {
            let current_price = orders[i].price;
            let future_price = orders[i + future_window.min(orders.len() - 1)].price;
            
            let price_change = future_price - current_price;
            let signed_volume = orders[i].volume * orders[i].direction as f64;
            
            let adverse_selection = price_change * signed_volume.signum();
            selection_components.push(adverse_selection);
        }

        Ok(selection_components)
    }

    pub fn analyze_tick_clustering(
        &self,
        orders: &[OrderFlow],
    ) -> Result<HashMap<f64, f64>, MicrostructureError> {
        let mut tick_frequencies = HashMap::new();
        let total_trades = orders.len() as f64;

        for order in orders {
            let tick_multiple = (order.price / self.tick_size).round() * self.tick_size;
            *tick_frequencies.entry(tick_multiple).or_insert(0.0) += 1.0;
        }

        // Normalize frequencies
        for freq in tick_frequencies.values_mut() {
            *freq /= total_trades;
        }

        Ok(tick_frequencies)
    }

    pub fn estimate_information_content(
        &self,
        orders: &[OrderFlow],
    ) -> Result<Vec<f64>, MicrostructureError> {
        let mut information_measures = Vec::with_capacity(orders.len());
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| MicrostructureError::ComputationError(e.to_string()))?;

        for window in orders.windows(self.information_horizon) {
            let signed_volumes: Vec<f64> = window.iter()
                .map(|o| o.volume * o.direction as f64)
                .collect();
            
            let volume_autocorr = self.compute_autocorrelation(&signed_volumes);
            let price_changes: Vec<f64> = window.windows(2)
                .map(|w| w[1].price - w[0].price)
                .collect();
            
            let price_volume_corr = self.compute_correlation(&price_changes, &signed_volumes[..price_changes.len()]);
            
            let information_score = normal.cdf((volume_autocorr + price_volume_corr) / 2.0);
            information_measures.push(information_score);
        }

        Ok(information_measures)
    }

    pub fn calculate_queue_position(
        &self,
        order_id: u64,
        book: &LimitOrderBook,
        side: Side,
        price: f64,
    ) -> Result<usize, MicrostructureError> {
        // Find the queue for the price level
        let queue = match side {
            Side::Bid => book.buy_orders.get(&(price as i64)),
            Side::Ask => book.sell_orders.get(&(price as i64)),
        };
        if let Some(orders) = queue {
            for (pos, order) in orders.iter().enumerate() {
                if order.id == order_id {
                    return Ok(pos);
                }
            }
            Err(MicrostructureError::QueueError("Order not found in queue".to_string()))
        } else {
            Err(MicrostructureError::QueueError("Price level not found".to_string()))
        }
    }
    pub fn calculate_liquidity_risk(
        &self,
        book: &LimitOrderBook,
        side: Side,
        price: f64,
        size: f64,
    ) -> Result<f64, MicrostructureError> {
        // Compute available depth at price and above/below
        let depth = match side {
            Side::Bid => book.buy_orders.range(..=(price as i64)).map(|(_, v)| v.iter().map(|o| o.size).sum::<f64>()).sum(),
            Side::Ask => book.sell_orders.range((price as i64)..).map(|(_, v)| v.iter().map(|o| o.size).sum::<f64>()).sum(),
        };
        // Liquidity risk is high if size exceeds available depth
        Ok((size / (depth + 1e-6)).min(1.0))
    }

    fn compute_autocorrelation(&self, series: &[f64]) -> f64 {
        if series.len() < 2 {
            return 0.0;
        }

        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let variance = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (series.len() - 1) as f64;

        if variance == 0.0 {
            return 0.0;
        }

        let autocov = series.windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>() / (series.len() - 1) as f64;

        autocov / variance
    }

    fn compute_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;

        let mut covariance = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            covariance += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            return 0.0;
        }

        covariance / (var_x * var_y).sqrt()
    }
}
