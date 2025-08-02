use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use thiserror::Error;
use tracing::{debug, error, info, warn};

// Export modules
pub mod error;
pub mod execution;
pub mod infrastructure;
pub mod math;
pub mod models;
pub mod orderbook;
pub mod risk;
pub mod rollup;
pub mod utils;
pub mod zkvm;

#[derive(Debug, Error)]
pub enum SpreadEstimationError {
    #[error("Invalid price data: {0}")]
    InvalidPriceData(String),
    #[error("Insufficient data points: {0}")]
    InsufficientData(String),
    #[error("Computation error: {0}")]
    ComputationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeData {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub direction: Option<i8>,
}

#[derive(Debug, Clone)]
pub struct SpreadEstimator {
    price_data: Vec<TradeData>,
    window_size: usize,
    confidence_level: f64,
}

impl SpreadEstimator {
    pub fn new(price_data: Vec<TradeData>, window_size: usize, confidence_level: f64) -> Result<Self, SpreadEstimationError> {
        if price_data.len() < window_size {
            return Err(SpreadEstimationError::InsufficientData(
                "Price data length must be greater than window size".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&confidence_level) {
            return Err(SpreadEstimationError::InvalidPriceData(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }
        Ok(Self {
            price_data,
            window_size,
            confidence_level,
        })
    }

    pub fn estimate_spread(&self) -> Result<SpreadMetrics, SpreadEstimationError> {
        let returns = self.calculate_log_returns()?;
        let (serial_covariance, variance) = self.calculate_serial_statistics(&returns)?;
        let spread_estimate = self.calculate_spread_estimate(serial_covariance, variance)?;
        let confidence_interval = self.calculate_confidence_interval(&returns, spread_estimate)?;

        Ok(SpreadMetrics {
            spread: spread_estimate,
            confidence_interval,
            serial_covariance,
            variance,
            sample_size: returns.len(),
        })
    }

    fn calculate_log_returns(&self) -> Result<Vec<f64>, SpreadEstimationError> {
        self.price_data
            .windows(2)
            .map(|window| {
                if window[1].price <= 0.0 || window[0].price <= 0.0 {
                    Err(SpreadEstimationError::InvalidPriceData(
                        "Non-positive prices found in data".to_string(),
                    ))
                } else {
                    Ok((window[1].price / window[0].price).ln())
                }
            })
            .collect()
    }

    fn calculate_serial_statistics(&self, returns: &[f64]) -> Result<(f64, f64), SpreadEstimationError> {
        if returns.is_empty() {
            return Err(SpreadEstimationError::InsufficientData(
                "No returns data available".to_string(),
            ));
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        let serial_covariance = returns
            .windows(2)
            .map(|window| (window[0] - mean) * (window[1] - mean))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        Ok((serial_covariance, variance))
    }

    fn calculate_spread_estimate(
        &self,
        serial_covariance: f64,
        variance: f64,
    ) -> Result<f64, SpreadEstimationError> {
        if variance <= 0.0 {
            return Err(SpreadEstimationError::ComputationError(
                "Variance must be positive".to_string(),
            ));
        }

        let spread = 2.0 * (-serial_covariance).sqrt();
        if !spread.is_finite() {
            return Err(SpreadEstimationError::ComputationError(
                "Invalid spread calculation result".to_string(),
            ));
        }

        Ok(spread)
    }

    fn calculate_confidence_interval(
        &self,
        returns: &[f64],
        spread: f64,
    ) -> Result<(f64, f64), SpreadEstimationError> {
        let n = returns.len() as f64;
        let standard_error = spread / (2.0 * n.sqrt());
        
        // Using normal approximation for large samples
        let z_score = statrs::distribution::Normal::new(0.0, 1.0)
            .map_err(|e| SpreadEstimationError::ComputationError(e.to_string()))?
            .inverse_cdf(1.0 - (1.0 - self.confidence_level) / 2.0);

        let margin = z_score * standard_error;
        Ok((spread - margin, spread + margin))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadMetrics {
    pub spread: f64,
    pub confidence_interval: (f64, f64),
    pub serial_covariance: f64,
    pub variance: f64,
    pub sample_size: usize,
}

impl SpreadMetrics {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn relative_spread(&self, mid_price: f64) -> f64 {
        self.spread / mid_price
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn generate_test_data(n: usize, true_spread: f64, volatility: f64) -> Vec<TradeData> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, volatility).unwrap();
        let mut price = 100.0;
        let mut data = Vec::with_capacity(n);
        let base_timestamp = Utc::now().timestamp();

        for i in 0..n {
            let direction = if rng.gen::<bool>() { 1 } else { -1 };
            let noise = normal.sample(&mut rng);
            price *= (1.0 + noise).exp();
            let trade_price = price + direction as f64 * true_spread / 2.0;

            data.push(TradeData {
                timestamp: base_timestamp + i as i64,
                price: trade_price,
                volume: rng.gen_range(100.0..1000.0),
                direction: Some(direction),
            });
        }
        data
    }

    #[test]
    fn test_spread_estimation() {
        let true_spread = 0.5;
        let test_data = generate_test_data(1000, true_spread, 0.001);
        let estimator = SpreadEstimator::new(test_data, 50, 0.95).unwrap();
        let metrics = estimator.estimate_spread().unwrap();

        assert!(metrics.spread > 0.0);
        assert!(metrics.confidence_interval.0 < metrics.confidence_interval.1);
        assert!((metrics.spread - true_spread).abs() / true_spread < 0.5);
    }

    #[test]
    fn test_invalid_inputs() {
        let test_data = vec![
            TradeData {
                timestamp: 0,
                price: -1.0,
                volume: 100.0,
                direction: Some(1),
            },
        ];
        let estimator = SpreadEstimator::new(test_data, 1, 0.95).unwrap();
        assert!(estimator.estimate_spread().is_err());
    }
}