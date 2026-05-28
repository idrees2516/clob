use crate::error::EstimationError;
use nalgebra as na;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpreadMetrics {
    pub effective_spread: f64,
    pub quoted_spread: Option<f64>,
    pub realized_spread: Option<f64>,
    pub price_impact: Option<f64>,
    pub serial_correlation: f64,
    pub volatility: f64,
    pub confidence_interval: (f64, f64),
    pub sample_size: usize,
}

impl SpreadMetrics {
    pub fn new(
        effective_spread: f64,
        quoted_spread: Option<f64>,
        realized_spread: Option<f64>,
        price_impact: Option<f64>,
        serial_correlation: f64,
        volatility: f64,
        confidence_interval: (f64, f64),
        sample_size: usize,
    ) -> Self {
        Self {
            effective_spread,
            quoted_spread,
            realized_spread,
            price_impact,
            serial_correlation,
            volatility,
            confidence_interval,
            sample_size,
        }
    }

    pub fn relative_metrics(&self, mid_price: f64) -> RelativeSpreadMetrics {
        RelativeSpreadMetrics {
            relative_effective_spread: self.effective_spread / mid_price,
            relative_quoted_spread: self.quoted_spread.map(|s| s / mid_price),
            relative_realized_spread: self.realized_spread.map(|s| s / mid_price),
            relative_price_impact: self.price_impact.map(|p| p / mid_price),
            serial_correlation: self.serial_correlation,
            volatility: self.volatility,
            confidence_interval: (
                self.confidence_interval.0 / mid_price,
                self.confidence_interval.1 / mid_price,
            ),
            sample_size: self.sample_size,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeSpreadMetrics {
    pub relative_effective_spread: f64,
    pub relative_quoted_spread: Option<f64>,
    pub relative_realized_spread: Option<f64>,
    pub relative_price_impact: Option<f64>,
    pub serial_correlation: f64,
    pub volatility: f64,
    pub confidence_interval: (f64, f64),
    pub sample_size: usize,
}

pub fn calculate_serial_correlation(returns: &[f64]) -> Result<f64, EstimationError> {
    if returns.len() < 2 {
        return Err(EstimationError::InsufficientData(
            "Need at least 2 returns for serial correlation".to_string(),
        ));
    }

    let mean = returns.par_iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .par_iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    if variance == 0.0 {
        return Err(EstimationError::ComputationError(
            "Zero variance in returns".to_string(),
        ));
    }

    let covariance = returns
        .par_iter()
        .zip(returns[1..].par_iter())
        .map(|(&x1, &x2)| (x1 - mean) * (x2 - mean))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    Ok(covariance / variance)
}

pub fn calculate_realized_volatility(returns: &[f64], annualization_factor: f64) -> f64 {
    let squared_returns: f64 = returns.par_iter().map(|&x| x.powi(2)).sum();
    (squared_returns / returns.len() as f64 * annualization_factor).sqrt()
}

pub fn calculate_confidence_interval(
    spread: f64,
    std_error: f64,
    confidence_level: f64,
    df: usize,
) -> Result<(f64, f64), EstimationError> {
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(EstimationError::ComputationError(
            "Confidence level must be between 0 and 1".to_string(),
        ));
    }

    let alpha = 1.0 - confidence_level;
    let t_value = student_t_quantile(1.0 - alpha / 2.0, df);
    let margin = t_value * std_error;

    Ok((spread - margin, spread + margin))
}

fn student_t_quantile(p: f64, df: usize) -> f64 {
    // This is a simple approximation of the Student's t-distribution quantile
    // For production use, consider using a statistical library with more precise implementations
    let z = normal_quantile(p);
    let df = df as f64;
    
    if df > 100.0 {
        z
    } else {
        z + (z.powi(3) + z) / (4.0 * df)
    }
}

fn normal_quantile(p: f64) -> f64 {
    // Approximation of the normal distribution quantile (inverse CDF)
    // Using the Abramowitz and Stegun approximation
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    t - ((2.30753 + 0.27061 * t)
        / (1.0 + 0.99229 * t + 0.04481 * t.powi(2)))
}

pub fn calculate_price_impact(
    trade_price: f64,
    mid_price: f64,
    trade_direction: i8,
) -> Result<f64, EstimationError> {
    if trade_direction != 1 && trade_direction != -1 {
        return Err(EstimationError::ComputationError(
            "Trade direction must be 1 or -1".to_string(),
        ));
    }

    Ok(2.0 * trade_direction as f64 * (trade_price - mid_price))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_serial_correlation() {
        let returns = vec![0.01, 0.02, -0.01, 0.01, -0.02];
        let correlation = calculate_serial_correlation(&returns).unwrap();
        assert!(correlation.abs() <= 1.0);
    }

    #[test]
    fn test_realized_volatility() {
        let returns = vec![0.01, 0.02, -0.01, 0.01, -0.02];
        let vol = calculate_realized_volatility(&returns, 252.0);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_confidence_interval() {
        let (lower, upper) = calculate_confidence_interval(0.01, 0.002, 0.95, 100).unwrap();
        assert!(lower < 0.01);
        assert!(upper > 0.01);
        assert!(lower > 0.0);
    }

    #[test]
    fn test_price_impact() {
        let impact = calculate_price_impact(100.1, 100.0, 1).unwrap();
        assert_relative_eq!(impact, 0.2, epsilon = 1e-10);

        let impact_sell = calculate_price_impact(99.9, 100.0, -1).unwrap();
        assert_relative_eq!(impact_sell, 0.2, epsilon = 1e-10);
    }
}
