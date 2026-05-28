use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Normal, StudentsT, ContinuousCDF};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum StressError {
    #[error("Detection error: {0}")]
    DetectionError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

#[derive(Debug, Clone)]
pub struct MarketCondition {
    pub timestamp: i64,
    pub price: f64,
    pub volume: f64,
    pub spread: f64,
    pub volatility: f64,
    pub liquidity_score: f64,
}

pub struct MarketStressAnalyzer {
    volatility_threshold: f64,
    liquidity_threshold: f64,
    correlation_window: usize,
    regime_min_duration: usize,
}

impl MarketStressAnalyzer {
    pub fn new(
        volatility_threshold: f64,
        liquidity_threshold: f64,
        correlation_window: usize,
        regime_min_duration: usize,
    ) -> Result<Self, StressError> {
        if volatility_threshold <= 0.0 || liquidity_threshold <= 0.0 {
            return Err(StressError::InvalidParameters(
                "Thresholds must be positive".to_string(),
            ));
        }

        Ok(Self {
            volatility_threshold,
            liquidity_threshold,
            correlation_window,
            regime_min_duration,
        })
    }

    pub fn detect_stress_periods(
        &self,
        conditions: &[MarketCondition],
    ) -> Result<Vec<StressPeriod>, StressError> {
        let volatility_regimes = self.detect_volatility_regimes(conditions)?;
        let liquidity_regimes = self.detect_liquidity_regimes(conditions)?;
        let correlation_breaks = self.detect_correlation_breaks(conditions)?;
        
        self.combine_stress_indicators(
            conditions,
            &volatility_regimes,
            &liquidity_regimes,
            &correlation_breaks,
        )
    }

    fn detect_volatility_regimes(
        &self,
        conditions: &[MarketCondition],
    ) -> Result<Vec<RegimeChange>, StressError> {
        let volatilities: Vec<f64> = conditions.iter()
            .map(|c| c.volatility)
            .collect();

        let regime_changes = self.detect_regime_changes(
            &volatilities,
            self.volatility_threshold,
            self.regime_min_duration,
        )?;

        Ok(regime_changes)
    }

    fn detect_liquidity_regimes(
        &self,
        conditions: &[MarketCondition],
    ) -> Result<Vec<RegimeChange>, StressError> {
        let liquidity_scores: Vec<f64> = conditions.iter()
            .map(|c| c.liquidity_score)
            .collect();

        let regime_changes = self.detect_regime_changes(
            &liquidity_scores,
            self.liquidity_threshold,
            self.regime_min_duration,
        )?;

        Ok(regime_changes)
    }

    fn detect_correlation_breaks(
        &self,
        conditions: &[MarketCondition],
    ) -> Result<Vec<CorrelationBreak>, StressError> {
        let mut breaks = Vec::new();
        let window_size = self.correlation_window;

        for window in conditions.windows(2 * window_size).step_by(window_size) {
            let mid = window_size;
            let (first_half, second_half) = window.split_at(mid);

            let correlation_matrix_1 = self.compute_correlation_matrix(first_half)?;
            let correlation_matrix_2 = self.compute_correlation_matrix(second_half)?;

            let distance = self.compute_matrix_distance(&correlation_matrix_1, &correlation_matrix_2);
            
            if distance > 0.5 { // Significant correlation structure change
                breaks.push(CorrelationBreak {
                    start_time: first_half[0].timestamp,
                    end_time: second_half[second_half.len()-1].timestamp,
                    severity: distance,
                });
            }
        }

        Ok(breaks)
    }

    fn compute_correlation_matrix(
        &self,
        conditions: &[MarketCondition],
    ) -> Result<na::DMatrix<f64>, StressError> {
        let n = 4; // Number of variables: price, volume, spread, volatility
        let mut matrix = na::DMatrix::zeros(n, n);
        
        let variables: Vec<Vec<f64>> = vec![
            conditions.iter().map(|c| c.price).collect(),
            conditions.iter().map(|c| c.volume).collect(),
            conditions.iter().map(|c| c.spread).collect(),
            conditions.iter().map(|c| c.volatility).collect(),
        ];

        for i in 0..n {
            for j in 0..n {
                matrix[(i, j)] = self.compute_correlation(&variables[i], &variables[j]);
            }
        }

        Ok(matrix)
    }

    fn compute_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let mean_x = x.iter().sum::<f64>() / n as f64;
        let mean_y = y.iter().sum::<f64>() / n as f64;
        
        let mut covariance = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            covariance += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        if var_x == 0.0 || var_y == 0.0 {
            0.0
        } else {
            covariance / (var_x * var_y).sqrt()
        }
    }

    fn compute_matrix_distance(
        &self,
        matrix1: &na::DMatrix<f64>,
        matrix2: &na::DMatrix<f64>,
    ) -> f64 {
        let diff = matrix1 - matrix2;
        let frobenius_norm = diff.iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        
        frobenius_norm / (matrix1.nrows() * matrix1.ncols()) as f64
    }

    fn detect_regime_changes(
        &self,
        series: &[f64],
        threshold: f64,
        min_duration: usize,
    ) -> Result<Vec<RegimeChange>, StressError> {
        let mut changes = Vec::new();
        let mut current_regime = RegimeType::Normal;
        let mut regime_start = 0;
        
        let rolling_mean = self.compute_rolling_mean(series, 20);
        let rolling_std = self.compute_rolling_std(series, 20);

        for (i, &value) in series.iter().enumerate() {
            if i < 20 {
                continue;
            }

            let z_score = (value - rolling_mean[i]) / rolling_std[i];
            let new_regime = if z_score.abs() > threshold {
                RegimeType::Stress
            } else {
                RegimeType::Normal
            };

            if new_regime != current_regime && i - regime_start >= min_duration {
                changes.push(RegimeChange {
                    start_idx: regime_start,
                    end_idx: i,
                    regime_type: current_regime,
                    severity: z_score.abs(),
                });
                regime_start = i;
                current_regime = new_regime;
            }
        }

        Ok(changes)
    }

    fn compute_rolling_mean(&self, series: &[f64], window: usize) -> Vec<f64> {
        let mut means = vec![0.0; series.len()];
        
        for i in window..series.len() {
            let sum: f64 = series[i-window..i].iter().sum();
            means[i] = sum / window as f64;
        }

        means
    }

    fn compute_rolling_std(&self, series: &[f64], window: usize) -> Vec<f64> {
        let means = self.compute_rolling_mean(series, window);
        let mut stds = vec![0.0; series.len()];
        
        for i in window..series.len() {
            let variance: f64 = series[i-window..i]
                .iter()
                .map(|&x| (x - means[i]).powi(2))
                .sum::<f64>() / (window - 1) as f64;
            stds[i] = variance.sqrt();
        }

        stds
    }

    fn combine_stress_indicators(
        &self,
        conditions: &[MarketCondition],
        volatility_regimes: &[RegimeChange],
        liquidity_regimes: &[RegimeChange],
        correlation_breaks: &[CorrelationBreak],
    ) -> Result<Vec<StressPeriod>, StressError> {
        let mut stress_periods = Vec::new();
        let mut current_period: Option<StressPeriod> = None;

        for (i, condition) in conditions.iter().enumerate() {
            let vol_stress = volatility_regimes.iter()
                .any(|r| r.start_idx <= i && i <= r.end_idx && r.regime_type == RegimeType::Stress);
            
            let liq_stress = liquidity_regimes.iter()
                .any(|r| r.start_idx <= i && i <= r.end_idx && r.regime_type == RegimeType::Stress);
            
            let corr_stress = correlation_breaks.iter()
                .any(|b| b.start_time <= condition.timestamp && condition.timestamp <= b.end_time);

            let stress_score = self.compute_stress_score(vol_stress, liq_stress, corr_stress);

            match (&mut current_period, stress_score > 0.5) {
                (None, true) => {
                    current_period = Some(StressPeriod {
                        start_time: condition.timestamp,
                        end_time: condition.timestamp,
                        severity: stress_score,
                        indicators: StressIndicators {
                            volatility_stress: vol_stress,
                            liquidity_stress: liq_stress,
                            correlation_stress: corr_stress,
                        },
                    });
                },
                (Some(period), true) => {
                    period.end_time = condition.timestamp;
                    period.severity = period.severity.max(stress_score);
                },
                (Some(period), false) => {
                    stress_periods.push(period.clone());
                    current_period = None;
                },
                (None, false) => {},
            }
        }

        if let Some(period) = current_period {
            stress_periods.push(period);
        }

        Ok(stress_periods)
    }

    fn compute_stress_score(
        &self,
        vol_stress: bool,
        liq_stress: bool,
        corr_stress: bool,
    ) -> f64 {
        let weights = [0.4, 0.4, 0.2]; // Weights for each indicator
        let indicators = [vol_stress, liq_stress, corr_stress];
        
        indicators.iter()
            .zip(weights.iter())
            .map(|(&stress, &weight)| if stress { weight } else { 0.0 })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegimeType {
    Normal,
    Stress,
}

#[derive(Debug, Clone)]
pub struct RegimeChange {
    pub start_idx: usize,
    pub end_idx: usize,
    pub regime_type: RegimeType,
    pub severity: f64,
}

#[derive(Debug, Clone)]
pub struct CorrelationBreak {
    pub start_time: i64,
    pub end_time: i64,
    pub severity: f64,
}

#[derive(Debug, Clone)]
pub struct StressIndicators {
    pub volatility_stress: bool,
    pub liquidity_stress: bool,
    pub correlation_stress: bool,
}

#[derive(Debug, Clone)]
pub struct StressPeriod {
    pub start_time: i64,
    pub end_time: i64,
    pub severity: f64,
    pub indicators: StressIndicators,
}
