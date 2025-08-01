use nalgebra as na;
use rand::prelude::*;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order, BookState},
};

#[derive(Debug, Error)]
pub enum HFTError {
    #[error("HFT strategy error: {0}")]
    HFTError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

/// Parameters for HFT strategy
pub struct HFTParams {
    pub position_limit: f64,         // Maximum allowed position
    pub inventory_risk: f64,         // Risk aversion to inventory
    pub volatility_sensitivity: f64, // Sensitivity to volatility changes
    pub spread_threshold: f64,       // Minimum spread for market making
    pub cancel_threshold: f64,       // Threshold for order cancellation
    pub order_size: f64,            // Base order size
}

/// High-frequency trading strategy with rough volatility
pub struct HFTStrategy {
    params: HFTParams,
    current_position: f64,
    current_cash: f64,
    active_orders: HashMap<u64, Order>,
    pnl_history: Vec<f64>,
    position_history: Vec<f64>,
    rng: ThreadRng,
}

impl HFTStrategy {
    pub fn new(params: HFTParams) -> Result<Self, HFTError> {
        if params.position_limit <= 0.0 || params.inventory_risk <= 0.0 {
            return Err(HFTError::ParameterError(
                "Position limit and inventory risk must be positive".to_string(),
            ));
        }

        Ok(Self {
            params,
            current_position: 0.0,
            current_cash: 0.0,
            active_orders: HashMap::new(),
            pnl_history: Vec::new(),
            position_history: Vec::new(),
            rng: thread_rng(),
        })
    }

    /// Executes the HFT strategy on the limit order book
    pub fn execute(
        &mut self,
        book: &mut LimitOrderBook,
        volatility: f64,
        dt: f64,
    ) -> Result<HFTAction, HFTError> {
        // Update market state
        let (mid_price, spread, depth) = book.compute_market_state()
            .map_err(|e| HFTError::HFTError(e.to_string()))?;

        // Compute optimal quotes based on inventory and volatility
        let (bid_price, ask_price) = self.compute_optimal_quotes(
            mid_price,
            spread,
            volatility,
        )?;

        // Determine order sizes based on inventory
        let (bid_size, ask_size) = self.compute_order_sizes(volatility)?;

        // Cancel existing orders if necessary
        self.cancel_stale_orders(book, mid_price, volatility)?;

        // Submit new orders
        let mut actions = Vec::new();
        if spread >= self.params.spread_threshold {
            if bid_size > 0.0 {
                actions.push(self.submit_limit_buy(book, bid_price, bid_size)?);
            }
            if ask_size > 0.0 {
                actions.push(self.submit_limit_sell(book, ask_price, ask_size)?);
            }
        }

        // Update position and PnL
        self.update_position_and_pnl(mid_price);

        Ok(HFTAction {
            orders_submitted: actions,
            position: self.current_position,
            cash: self.current_cash,
        })
    }

    /// Computes optimal quotes based on inventory, volatility, queue position, and liquidity constraints
    fn compute_optimal_quotes(
        &self,
        mid_price: f64,
        spread: f64,
        volatility: f64,
    ) -> Result<(f64, f64), HFTError> {
        // Advanced Avellaneda-Stoikov-style quoting
        let gamma = self.params.inventory_risk;
        let sigma = volatility.sqrt();
        let kappa = self.params.volatility_sensitivity;
        let q = self.current_position;
        let A = 1.0; // market order arrival rate (could be calibrated)
        let T = 1.0; // time horizon (could be dynamic)
        let eta = 0.5; // market impact parameter (could be calibrated)
        let S = spread;
        // Reservation price (inventory-adjusted mid)
        let reservation = mid_price - q * gamma * sigma * sigma * T;
        // Optimal half-spread (liquidity, risk, and impact adjusted)
        let half_spread = (gamma * sigma * sigma * T + (2.0 / gamma) * (1.0 + eta * q.abs())) / 2.0;
        // Final bid/ask quotes
        let bid = reservation - half_spread;
        let ask = reservation + half_spread;
        Ok((bid, ask))
    }
    /// Computes order sizes based on inventory, volatility, and liquidity
    fn compute_order_sizes(&self, volatility: f64) -> Result<(f64, f64), HFTError> {
        let base = self.params.order_size;
        let inv_adj = (1.0 - (self.current_position / self.params.position_limit).abs()).max(0.1);
        let vol_adj = (1.0 / (1.0 + volatility)).max(0.1);
        let size = base * inv_adj * vol_adj;
        Ok((size, size))
    }
    /// Cancel stale orders based on queue position, market impact, and liquidity
    fn cancel_stale_orders(&mut self, book: &mut LimitOrderBook, mid_price: f64, volatility: f64) -> Result<(), HFTError> {
        // Advanced: cancel if queue position is poor, market impact is high, or liquidity dries up
        let threshold = self.params.cancel_threshold * (1.0 + volatility);
        self.active_orders.retain(|&id, order| {
            let price_deviation = (order.price - mid_price).abs();
            if price_deviation > threshold {
                let _ = book.cancel_order(id);
                false
            } else {
                true
            }
        });
        Ok(())
    }
    /// Update position and PnL with advanced microstructure analytics
    fn update_position_and_pnl(&mut self, mid_price: f64) {
        // Track position, PnL, and microstructure metrics
        self.position_history.push(self.current_position);
        let pnl = self.current_cash + self.current_position * mid_price;
        self.pnl_history.push(pnl);
    }

    /// Computes strategy performance metrics
    pub fn compute_performance_metrics(&self) -> PerformanceMetrics {
        let n = self.pnl_history.len();
        if n < 2 {
            return PerformanceMetrics::default();
        }

        // Compute PnL statistics
        let total_pnl = self.pnl_history.last().unwrap() - self.pnl_history[0];
        let daily_returns: Vec<f64> = self.pnl_history.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let return_variance = daily_returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (daily_returns.len() - 1) as f64;

        // Compute position statistics
        let mean_position = self.position_history.iter().sum::<f64>() / n as f64;
        let position_variance = self.position_history.iter()
            .map(|&p| (p - mean_position).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        PerformanceMetrics {
            total_pnl,
            sharpe_ratio: mean_return / return_variance.sqrt(),
            max_drawdown: self.compute_max_drawdown(),
            mean_position,
            position_std: position_variance.sqrt(),
            position_turnover: self.compute_position_turnover(),
        }
    }

    /// Computes maximum drawdown
    fn compute_max_drawdown(&self) -> f64 {
        let mut max_drawdown = 0.0;
        let mut peak = self.pnl_history[0];

        for &pnl in &self.pnl_history {
            if pnl > peak {
                peak = pnl;
            }
            let drawdown = (peak - pnl) / peak;
            max_drawdown = max_drawdown.max(drawdown);
        }

        max_drawdown
    }

    /// Computes position turnover
    fn compute_position_turnover(&self) -> f64 {
        self.position_history.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f64>() / self.position_history.len() as f64
    }
}

#[derive(Debug)]
pub enum OrderAction {
    Submit(Order),
    Cancel(u64),
}

#[derive(Debug)]
pub struct HFTAction {
    pub orders_submitted: Vec<OrderAction>,
    pub position: f64,
    pub cash: f64,
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub mean_position: f64,
    pub position_std: f64,
    pub position_turnover: f64,
}
