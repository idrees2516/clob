use crate::{SpreadEstimationError, TradeData};
use nalgebra as na;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::collections::VecDeque;

pub struct RollingWindow {
    window: VecDeque<f64>,
    size: usize,
}

impl RollingWindow {
    pub fn new(size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(size),
            size,
        }
    }

    pub fn add(&mut self, value: f64) -> Option<f64> {
        if self.window.len() == self.size {
            self.window.pop_front();
        }
        self.window.push_back(value);
        
        if self.window.len() == self.size {
            Some(self.compute_statistic())
        } else {
            None
        }
    }

    fn compute_statistic(&self) -> f64 {
        let mean = self.window.iter().sum::<f64>() / self.size as f64;
        let variance = self.window
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.size - 1) as f64;
        variance.sqrt()
    }
}

pub struct VolatilityEstimator {
    returns: Vec<f64>,
    window_size: usize,
}

impl VolatilityEstimator {
    pub fn new(price_data: &[TradeData], window_size: usize) -> Result<Self, SpreadEstimationError> {
        let returns = Self::calculate_returns(price_data)?;
        Ok(Self {
            returns,
            window_size,
        })
    }

    fn calculate_returns(price_data: &[TradeData]) -> Result<Vec<f64>, SpreadEstimationError> {
        price_data
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

    pub fn estimate_volatility(&self) -> Result<Vec<f64>, SpreadEstimationError> {
        if self.returns.len() < self.window_size {
            return Err(SpreadEstimationError::InsufficientData(
                "Insufficient data for volatility estimation".to_string(),
            ));
        }

        let mut rolling_window = RollingWindow::new(self.window_size);
        let mut volatilities = Vec::with_capacity(self.returns.len());

        for &ret in &self.returns {
            if let Some(vol) = rolling_window.add(ret) {
                volatilities.push(vol);
            }
        }

        Ok(volatilities)
    }
}

pub struct Bootstrap {
    data: Vec<f64>,
    n_samples: usize,
    sample_size: usize,
}

impl Bootstrap {
    pub fn new(data: Vec<f64>, n_samples: usize, sample_size: usize) -> Self {
        Self {
            data,
            n_samples,
            sample_size,
        }
    }

    pub fn generate_samples(&self) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        (0..self.n_samples)
            .into_par_iter()
            .map(|_| {
                (0..self.sample_size)
                    .map(|_| {
                        let idx = rng.gen_range(0..self.data.len());
                        self.data[idx]
                    })
                    .collect()
            })
            .collect()
    }

    pub fn compute_confidence_interval(
        &self,
        samples: &[Vec<f64>],
        confidence_level: f64,
    ) -> Result<(f64, f64), SpreadEstimationError> {
        let mut statistics: Vec<f64> = samples
            .iter()
            .map(|sample| {
                sample.iter().sum::<f64>() / sample.len() as f64
            })
            .collect();

        statistics.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = ((1.0 - confidence_level) / 2.0 * statistics.len() as f64) as usize;
        let upper_idx = ((1.0 + confidence_level) / 2.0 * statistics.len() as f64) as usize;

        Ok((statistics[lower_idx], statistics[upper_idx]))
    }
}

pub fn compute_autocorrelation(
    returns: &[f64],
    max_lag: usize,
) -> Result<Vec<f64>, SpreadEstimationError> {
    if returns.is_empty() {
        return Err(SpreadEstimationError::InsufficientData(
            "Empty returns series".to_string(),
        ));
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    if variance == 0.0 {
        return Err(SpreadEstimationError::ComputationError(
            "Zero variance in returns".to_string(),
        ));
    }

    let mut autocorr = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let covariance = returns
            .iter()
            .zip(returns.iter().skip(lag))
            .map(|(&x1, &x2)| (x1 - mean) * (x2 - mean))
            .sum::<f64>()
            / returns.len() as f64;
        autocorr.push(covariance / variance);
    }

    Ok(autocorr)
}

pub fn simulate_price_process(
    initial_price: f64,
    volatility: f64,
    spread: f64,
    n_steps: usize,
) -> Result<Vec<TradeData>, SpreadEstimationError> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, volatility)
        .map_err(|e| SpreadEstimationError::ComputationError(e.to_string()))?;

    let mut price = initial_price;
    let mut data = Vec::with_capacity(n_steps);
    let base_timestamp = chrono::Utc::now().timestamp();

    for i in 0..n_steps {
        let direction = if rng.gen::<bool>() { 1 } else { -1 };
        let noise = normal.sample(&mut rng);
        price *= (1.0 + noise).exp();
        
        let trade_price = price + direction as f64 * spread / 2.0;
        data.push(TradeData {
            timestamp: base_timestamp + i as i64,
            price: trade_price,
            volume: rng.gen_range(100.0..1000.0),
            direction: Some(direction),
        });
    }

    Ok(data)
}
