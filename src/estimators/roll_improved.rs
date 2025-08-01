use crate::error::EstimationError;
use crate::models::{GarchModel, KalmanFilter};
use crate::TradeData;
use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64;

pub struct ImprovedRollEstimator {
    data: Vec<TradeData>,
    window_size: usize,
    confidence_level: f64,
    use_kalman: bool,
    adjust_volatility: bool,
}

impl ImprovedRollEstimator {
    pub fn new(
        data: Vec<TradeData>,
        window_size: usize,
        confidence_level: f64,
        use_kalman: bool,
        adjust_volatility: bool,
    ) -> Result<Self, EstimationError> {
        if data.len() < window_size {
            return Err(EstimationError::InsufficientData(
                "Data length must be greater than window size".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&confidence_level) {
            return Err(EstimationError::InvalidParameters(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            data,
            window_size,
            confidence_level,
            use_kalman,
            adjust_volatility,
        })
    }

    pub fn estimate_spread(&self) -> Result<ImprovedSpreadMetrics, EstimationError> {
        let returns = self.calculate_returns()?;
        
        // Estimate base spread using Roll's method
        let (base_spread, serial_covariance, variance) = self.estimate_base_spread(&returns)?;
        
        // Apply Kalman filter if requested
        let kalman_spread = if self.use_kalman {
            Some(self.apply_kalman_filter()?)
        } else {
            None
        };
        
        // Adjust for time-varying volatility if requested
        let volatility_adjusted_spread = if self.adjust_volatility {
            Some(self.adjust_for_volatility(&returns, base_spread)?)
        } else {
            None
        };
        
        // Calculate confidence intervals
        let confidence_interval = self.calculate_confidence_interval(
            base_spread,
            &returns,
            serial_covariance,
            variance,
        )?;
        
        // Calculate additional metrics
        let effective_spread = self.calculate_effective_spread()?;
        let realized_spread = self.calculate_realized_spread()?;
        let price_impact = self.calculate_price_impact()?;
        
        Ok(ImprovedSpreadMetrics {
            base_spread,
            kalman_spread,
            volatility_adjusted_spread,
            effective_spread,
            realized_spread,
            price_impact,
            serial_covariance,
            variance,
            confidence_interval,
            sample_size: returns.len(),
        })
    }

    fn calculate_returns(&self) -> Result<Vec<f64>, EstimationError> {
        self.data
            .windows(2)
            .map(|window| {
                if window[1].price <= 0.0 || window[0].price <= 0.0 {
                    Err(EstimationError::InvalidParameters(
                        "Non-positive prices found".to_string(),
                    ))
                } else {
                    Ok((window[1].price / window[0].price).ln())
                }
            })
            .collect()
    }

    fn estimate_base_spread(
        &self,
        returns: &[f64],
    ) -> Result<(f64, f64, f64), EstimationError> {
        let mean = returns.par_iter().sum::<f64>() / returns.len() as f64;
        
        let variance = returns
            .par_iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        let serial_covariance = returns
            .par_iter()
            .zip(returns[1..].par_iter())
            .map(|(&x1, &x2)| (x1 - mean) * (x2 - mean))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        let spread = 2.0 * (-serial_covariance).sqrt();
        
        Ok((spread, serial_covariance, variance))
    }

    fn apply_kalman_filter(&self) -> Result<f64, EstimationError> {
        let state_dim = 2; // [price, spread]
        let obs_dim = 1;   // observed price
        
        let transition_matrix = na::DMatrix::identity(state_dim, state_dim);
        let observation_matrix = na::DMatrix::from_vec(obs_dim, state_dim, vec![1.0, 0.0]);
        
        // Initialize with reasonable values
        let process_noise = na::DMatrix::identity(state_dim, state_dim) * 1e-4;
        let measurement_noise = na::DMatrix::identity(obs_dim, obs_dim) * 1e-3;
        let initial_state = na::DVector::from_vec(vec![self.data[0].price, 0.5]);
        let initial_cov = na::DMatrix::identity(state_dim, state_dim);

        let mut kf = KalmanFilter::new(
            state_dim,
            obs_dim,
            transition_matrix,
            observation_matrix,
            process_noise,
            measurement_noise,
            initial_state,
            initial_cov,
        )?;

        let measurements: Vec<na::DVector<f64>> = self.data
            .iter()
            .map(|d| na::DVector::from_vec(vec![d.price]))
            .collect();

        let filtered_states = kf.smooth(&measurements)?;
        
        // Take the mean of the spread estimates
        let mean_spread = filtered_states
            .iter()
            .map(|s| s[1])
            .sum::<f64>() / filtered_states.len() as f64;

        Ok(mean_spread)
    }

    fn adjust_for_volatility(
        &self,
        returns: &[f64],
        base_spread: f64,
    ) -> Result<f64, EstimationError> {
        // Fit GARCH(1,1) model
        let mut garch = GarchModel::new(0.0, 0.0, 0.0)?;
        garch.fit(returns.to_vec(), 1000, 1e-6)?;
        
        let volatilities = garch.get_volatilities();
        let mean_vol = volatilities.iter().sum::<f64>() / volatilities.len() as f64;
        
        // Adjust spread estimate by volatility ratio
        let adjusted_spread = base_spread * (mean_vol / volatilities.last().unwrap()).sqrt();
        
        Ok(adjusted_spread)
    }

    fn calculate_confidence_interval(
        &self,
        spread: f64,
        returns: &[f64],
        serial_covariance: f64,
        variance: f64,
    ) -> Result<(f64, f64), EstimationError> {
        let n = returns.len() as f64;
        let standard_error = (
            (1.0 + 2.0 * serial_covariance.powi(2) / variance.powi(2))
            / n
        ).sqrt() * spread;

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| EstimationError::ComputationError(e.to_string()))?;
        let z_score = normal.inverse_cdf(1.0 - (1.0 - self.confidence_level) / 2.0);

        Ok((
            spread - z_score * standard_error,
            spread + z_score * standard_error,
        ))
    }

    fn calculate_effective_spread(&self) -> Result<f64, EstimationError> {
        let mid_prices: Vec<f64> = self.data
            .windows(2)
            .map(|w| (w[0].price + w[1].price) / 2.0)
            .collect();

        let effective_spreads: Vec<f64> = self.data
            .windows(2)
            .zip(mid_prices.iter())
            .filter_map(|(trades, &mid)| {
                trades[1].direction.map(|dir| {
                    2.0 * dir as f64 * (trades[1].price - mid)
                })
            })
            .collect();

        if effective_spreads.is_empty() {
            return Err(EstimationError::InsufficientData(
                "No trade direction information available".to_string(),
            ));
        }

        Ok(effective_spreads.iter().sum::<f64>() / effective_spreads.len() as f64)
    }

    fn calculate_realized_spread(&self) -> Result<f64, EstimationError> {
        let window = 5; // 5-minute window for realized spread
        let realized_spreads: Vec<f64> = self.data
            .windows(window + 1)
            .filter_map(|w| {
                w[0].direction.map(|dir| {
                    let mid_change = w[window].price - w[0].price;
                    2.0 * dir as f64 * (w[0].price - w[window].price) - mid_change
                })
            })
            .collect();

        if realized_spreads.is_empty() {
            return Err(EstimationError::InsufficientData(
                "Insufficient data for realized spread calculation".to_string(),
            ));
        }

        Ok(realized_spreads.iter().sum::<f64>() / realized_spreads.len() as f64)
    }

    fn calculate_price_impact(&self) -> Result<f64, EstimationError> {
        let impacts: Vec<f64> = self.data
            .windows(2)
            .filter_map(|w| {
                w[1].direction.map(|dir| {
                    let mid = (w[0].price + w[1].price) / 2.0;
                    2.0 * dir as f64 * (w[1].price - mid)
                })
            })
            .collect();

        if impacts.is_empty() {
            return Err(EstimationError::InsufficientData(
                "No trade direction information available".to_string(),
            ));
        }

        Ok(impacts.iter().sum::<f64>() / impacts.len() as f64)
    }
}

#[derive(Debug, Clone)]
pub struct ImprovedSpreadMetrics {
    pub base_spread: f64,
    pub kalman_spread: Option<f64>,
    pub volatility_adjusted_spread: Option<f64>,
    pub effective_spread: f64,
    pub realized_spread: f64,
    pub price_impact: f64,
    pub serial_covariance: f64,
    pub variance: f64,
    pub confidence_interval: (f64, f64),
    pub sample_size: usize,
}
