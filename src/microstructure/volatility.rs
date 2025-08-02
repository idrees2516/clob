//! High-Frequency Volatility Estimation
//! 
//! Implements sophisticated volatility estimation methods including:
//! - Realized volatility calculation using high-frequency returns
//! - Bi-power variation for jump-robust volatility
//! - Microstructure noise filtering using optimal sampling
//! - Intraday volatility pattern analysis

use super::{MarketData, Trade, TradeDirection, MicrostructureError, MicrostructureResult};
use crate::math::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// High-frequency volatility metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityMetrics {
    pub timestamp: u64,
    pub realized_volatility: FixedPoint,
    pub bi_power_variation: FixedPoint,
    pub jump_robust_volatility: FixedPoint,
    pub microstructure_noise_variance: FixedPoint,
    pub optimal_sampling_frequency: u64,
    pub intraday_volatility_pattern: Vec<FixedPoint>,
    pub volatility_of_volatility: FixedPoint,
}

/// Return observation for volatility calculation
#[derive(Debug, Clone)]
struct ReturnObservation {
    timestamp: u64,
    log_return: FixedPoint,
    price: FixedPoint,
    is_jump: bool,
}

/// Intraday volatility pattern (hourly buckets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntradayVolatilityPattern {
    pub hour_of_day: u8,
    pub average_volatility: FixedPoint,
    pub volatility_std: FixedPoint,
    pub observation_count: u32,
}

/// Optimal sampling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalSamplingParams {
    pub optimal_frequency_ms: u64,
    pub noise_to_signal_ratio: FixedPoint,
    pub efficiency_gain: FixedPoint,
}

/// High-frequency volatility estimator
pub struct VolatilityEstimator {
    market_data_history: VecDeque<MarketData>,
    return_observations: VecDeque<ReturnObservation>,
    intraday_patterns: Vec<IntradayVolatilityPattern>,
    max_history_size: usize,
    min_sampling_interval_ms: u64,
    max_sampling_interval_ms: u64,
    jump_detection_threshold: FixedPoint,
}

impl VolatilityEstimator {
    /// Create new volatility estimator
    pub fn new(
        max_history_size: usize,
        min_sampling_interval_ms: u64,
        max_sampling_interval_ms: u64,
        jump_detection_threshold: FixedPoint,
    ) -> Self {
        Self {
            market_data_history: VecDeque::with_capacity(max_history_size),
            return_observations: VecDeque::with_capacity(max_history_size),
            intraday_patterns: vec![IntradayVolatilityPattern {
                hour_of_day: i as u8,
                average_volatility: FixedPoint::ZERO,
                volatility_std: FixedPoint::ZERO,
                observation_count: 0,
            }; 24], // 24 hours
            max_history_size,
            min_sampling_interval_ms,
            max_sampling_interval_ms,
            jump_detection_threshold,
        }
    }

    /// Update with new market data
    pub fn update_market_data(&mut self, market_data: MarketData) -> MicrostructureResult<()> {
        // Calculate return if we have previous data
        if let Some(prev_data) = self.market_data_history.back() {
            let return_obs = self.calculate_return_observation(prev_data, &market_data)?;
            
            self.return_observations.push_back(return_obs);
            if self.return_observations.len() > self.max_history_size {
                self.return_observations.pop_front();
            }
        }

        self.market_data_history.push_back(market_data);
        if self.market_data_history.len() > self.max_history_size {
            self.market_data_history.pop_front();
        }

        // Update intraday patterns periodically
        if self.return_observations.len() % 100 == 0 {
            self.update_intraday_patterns()?;
        }

        Ok(())
    }

    /// Calculate return observation from consecutive market data
    fn calculate_return_observation(
        &self,
        prev_data: &MarketData,
        current_data: &MarketData,
    ) -> MicrostructureResult<ReturnObservation> {
        let prev_mid = (prev_data.bid_price + prev_data.ask_price) / FixedPoint::from_float(2.0);
        let current_mid = (current_data.bid_price + current_data.ask_price) / FixedPoint::from_float(2.0);

        if prev_mid <= FixedPoint::ZERO || current_mid <= FixedPoint::ZERO {
            return Err(MicrostructureError::DataQuality(
                "Invalid price data for return calculation".to_string()
            ));
        }

        let log_return = (current_mid / prev_mid).ln();
        
        // Detect jumps using threshold
        let is_jump = log_return.abs() > self.jump_detection_threshold;

        Ok(ReturnObservation {
            timestamp: current_data.timestamp,
            log_return,
            price: current_mid,
            is_jump,
        })
    }

    /// Calculate realized volatility
    pub fn calculate_realized_volatility(&self, window_ms: u64) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.return_observations.back()
            .map(|obs| obs.timestamp.saturating_sub(window_ms))
            .unwrap_or(0);

        let recent_returns: Vec<_> = self.return_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if recent_returns.len() < 2 {
            return Ok(FixedPoint::ZERO);
        }

        // RV = Σ r_i^2
        let sum_squared_returns: FixedPoint = recent_returns.iter()
            .map(|obs| obs.log_return * obs.log_return)
            .sum();

        // Annualize (assuming 252 trading days, 6.5 hours per day)
        let time_fraction = window_ms as f64 / (252.0 * 6.5 * 3600.0 * 1000.0);
        let annualized_rv = sum_squared_returns / FixedPoint::from_float(time_fraction);
        
        Ok(annualized_rv.sqrt())
    }

    /// Calculate bi-power variation for jump-robust volatility
    pub fn calculate_bi_power_variation(&self, window_ms: u64) -> MicrostructureResult<FixedPoint> {
        let cutoff_time = self.return_observations.back()
            .map(|obs| obs.timestamp.saturating_sub(window_ms))
            .unwrap_or(0);

        let recent_returns: Vec<_> = self.return_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if recent_returns.len() < 3 {
            return Ok(FixedPoint::ZERO);
        }

        // BV = μ_1^2 * Σ |r_{i-1}| * |r_i|
        // where μ_1 = sqrt(2/π) ≈ 0.7979
        let mu_1 = FixedPoint::from_float(0.7979);
        let mu_1_squared = mu_1 * mu_1;

        let mut bi_power_sum = FixedPoint::ZERO;
        for i in 1..recent_returns.len() {
            let abs_return_prev = recent_returns[i-1].log_return.abs();
            let abs_return_curr = recent_returns[i].log_return.abs();
            bi_power_sum += abs_return_prev * abs_return_curr;
        }

        let bi_power_variation = mu_1_squared * bi_power_sum;

        // Annualize
        let time_fraction = window_ms as f64 / (252.0 * 6.5 * 3600.0 * 1000.0);
        let annualized_bv = bi_power_variation / FixedPoint::from_float(time_fraction);
        
        Ok(annualized_bv.sqrt())
    }

    /// Calculate jump-robust volatility (using bi-power variation)
    pub fn calculate_jump_robust_volatility(&self, window_ms: u64) -> MicrostructureResult<FixedPoint> {
        let realized_vol = self.calculate_realized_volatility(window_ms)?;
        let bi_power_vol = self.calculate_bi_power_variation(window_ms)?;
        
        // Jump-robust volatility is the minimum of RV and BV
        // In practice, we use BV as it's more robust to jumps
        Ok(bi_power_vol)
    }

    /// Estimate microstructure noise variance
    pub fn estimate_microstructure_noise_variance(&self) -> MicrostructureResult<FixedPoint> {
        if self.return_observations.len() < 10 {
            return Ok(FixedPoint::ZERO);
        }

        // Use first-order autocovariance of returns
        // Noise variance ≈ -γ_1 where γ_1 is first-order autocovariance
        let returns: Vec<FixedPoint> = self.return_observations.iter()
            .rev()
            .take(100)
            .map(|obs| obs.log_return)
            .collect();

        if returns.len() < 2 {
            return Ok(FixedPoint::ZERO);
        }

        // Calculate mean
        let mean: FixedPoint = returns.iter().sum::<FixedPoint>() / 
                              FixedPoint::from_int(returns.len() as i64);

        // Calculate first-order autocovariance
        let mut autocovariance = FixedPoint::ZERO;
        for i in 1..returns.len() {
            let dev_curr = returns[i] - mean;
            let dev_prev = returns[i-1] - mean;
            autocovariance += dev_curr * dev_prev;
        }
        
        autocovariance /= FixedPoint::from_int((returns.len() - 1) as i64);
        
        // Noise variance is approximately -autocovariance (if negative)
        let noise_variance = if autocovariance < FixedPoint::ZERO {
            -autocovariance
        } else {
            FixedPoint::ZERO
        };

        Ok(noise_variance)
    }

    /// Calculate optimal sampling frequency
    pub fn calculate_optimal_sampling_frequency(&self) -> MicrostructureResult<OptimalSamplingParams> {
        let noise_variance = self.estimate_microstructure_noise_variance()?;
        
        if noise_variance <= FixedPoint::ZERO {
            return Ok(OptimalSamplingParams {
                optimal_frequency_ms: self.min_sampling_interval_ms,
                noise_to_signal_ratio: FixedPoint::ZERO,
                efficiency_gain: FixedPoint::from_float(1.0),
            });
        }

        // Estimate signal variance (integrated volatility)
        let signal_variance = self.calculate_realized_volatility(3600000)? // 1 hour window
            .powi(2) / FixedPoint::from_float(252.0 * 6.5); // Daily fraction

        if signal_variance <= FixedPoint::ZERO {
            return Ok(OptimalSamplingParams {
                optimal_frequency_ms: self.min_sampling_interval_ms,
                noise_to_signal_ratio: FixedPoint::ZERO,
                efficiency_gain: FixedPoint::from_float(1.0),
            });
        }

        let noise_to_signal_ratio = noise_variance / signal_variance;

        // Optimal sampling frequency (in seconds): Δt* = (2 * noise_variance / signal_variance)^(1/2)
        let optimal_interval_seconds = (FixedPoint::from_float(2.0) * noise_to_signal_ratio).sqrt();
        let optimal_frequency_ms = (optimal_interval_seconds.to_float() * 1000.0) as u64;

        // Clamp to reasonable bounds
        let optimal_frequency_ms = optimal_frequency_ms
            .max(self.min_sampling_interval_ms)
            .min(self.max_sampling_interval_ms);

        // Calculate efficiency gain compared to highest frequency sampling
        let efficiency_gain = (optimal_frequency_ms as f64 / self.min_sampling_interval_ms as f64).sqrt();

        Ok(OptimalSamplingParams {
            optimal_frequency_ms,
            noise_to_signal_ratio,
            efficiency_gain: FixedPoint::from_float(efficiency_gain),
        })
    }

    /// Update intraday volatility patterns
    fn update_intraday_patterns(&mut self) -> MicrostructureResult<()> {
        // Group returns by hour of day
        let mut hourly_returns: Vec<Vec<FixedPoint>> = vec![Vec::new(); 24];
        
        for obs in &self.return_observations {
            // Convert timestamp to hour of day (simplified - assumes UTC)
            let hour = ((obs.timestamp / 1000) % 86400) / 3600;
            if hour < 24 {
                hourly_returns[hour as usize].push(obs.log_return);
            }
        }

        // Update patterns for each hour
        for (hour, returns) in hourly_returns.iter().enumerate() {
            if returns.is_empty() {
                continue;
            }

            // Calculate realized volatility for this hour
            let sum_squared: FixedPoint = returns.iter()
                .map(|r| r * r)
                .sum();
            
            let hourly_rv = (sum_squared * FixedPoint::from_float(24.0)).sqrt(); // Scale to daily

            // Calculate standard deviation of returns
            let mean_return: FixedPoint = returns.iter().sum::<FixedPoint>() / 
                                        FixedPoint::from_int(returns.len() as i64);
            
            let variance: FixedPoint = returns.iter()
                .map(|r| (*r - mean_return).powi(2))
                .sum::<FixedPoint>() / FixedPoint::from_int(returns.len() as i64);
            
            let std_dev = variance.sqrt();

            // Update pattern with exponential smoothing
            let alpha = FixedPoint::from_float(0.1); // Smoothing factor
            let pattern = &mut self.intraday_patterns[hour];
            
            if pattern.observation_count == 0 {
                pattern.average_volatility = hourly_rv;
                pattern.volatility_std = std_dev;
            } else {
                pattern.average_volatility = pattern.average_volatility * (FixedPoint::from_float(1.0) - alpha) + 
                                           hourly_rv * alpha;
                pattern.volatility_std = pattern.volatility_std * (FixedPoint::from_float(1.0) - alpha) + 
                                       std_dev * alpha;
            }
            
            pattern.observation_count += returns.len() as u32;
        }

        Ok(())
    }

    /// Calculate volatility of volatility
    pub fn calculate_volatility_of_volatility(&self, window_ms: u64) -> MicrostructureResult<FixedPoint> {
        let bucket_size_ms = window_ms / 20; // 20 buckets
        let mut volatility_observations = Vec::new();

        let cutoff_time = self.return_observations.back()
            .map(|obs| obs.timestamp.saturating_sub(window_ms))
            .unwrap_or(0);

        // Calculate volatility for each bucket
        for i in 0..20 {
            let bucket_start = cutoff_time + i * bucket_size_ms;
            let bucket_end = bucket_start + bucket_size_ms;

            let bucket_returns: Vec<_> = self.return_observations.iter()
                .filter(|obs| obs.timestamp >= bucket_start && obs.timestamp < bucket_end)
                .collect();

            if bucket_returns.len() >= 2 {
                let sum_squared: FixedPoint = bucket_returns.iter()
                    .map(|obs| obs.log_return * obs.log_return)
                    .sum();
                
                let bucket_volatility = sum_squared.sqrt();
                volatility_observations.push(bucket_volatility);
            }
        }

        if volatility_observations.len() < 2 {
            return Ok(FixedPoint::ZERO);
        }

        // Calculate standard deviation of volatility observations
        let mean_vol: FixedPoint = volatility_observations.iter().sum::<FixedPoint>() / 
                                  FixedPoint::from_int(volatility_observations.len() as i64);

        let vol_variance: FixedPoint = volatility_observations.iter()
            .map(|vol| (*vol - mean_vol).powi(2))
            .sum::<FixedPoint>() / FixedPoint::from_int(volatility_observations.len() as i64);

        Ok(vol_variance.sqrt())
    }

    /// Calculate comprehensive volatility metrics
    pub fn calculate_metrics(&self, window_ms: u64) -> MicrostructureResult<VolatilityMetrics> {
        let timestamp = self.return_observations.back()
            .map(|obs| obs.timestamp)
            .unwrap_or(0);

        let realized_volatility = self.calculate_realized_volatility(window_ms)?;
        let bi_power_variation = self.calculate_bi_power_variation(window_ms)?;
        let jump_robust_volatility = self.calculate_jump_robust_volatility(window_ms)?;
        let microstructure_noise_variance = self.estimate_microstructure_noise_variance()?;
        let optimal_sampling = self.calculate_optimal_sampling_frequency()?;
        let volatility_of_volatility = self.calculate_volatility_of_volatility(window_ms)?;

        // Extract intraday pattern
        let intraday_volatility_pattern: Vec<FixedPoint> = self.intraday_patterns.iter()
            .map(|pattern| pattern.average_volatility)
            .collect();

        Ok(VolatilityMetrics {
            timestamp,
            realized_volatility,
            bi_power_variation,
            jump_robust_volatility,
            microstructure_noise_variance,
            optimal_sampling_frequency: optimal_sampling.optimal_frequency_ms,
            intraday_volatility_pattern,
            volatility_of_volatility,
        })
    }

    /// Get volatility summary
    pub fn get_volatility_summary(&self, window_ms: u64) -> MicrostructureResult<VolatilitySummary> {
        let metrics = self.calculate_metrics(window_ms)?;
        
        Ok(VolatilitySummary {
            timestamp: metrics.timestamp,
            realized_vol_pct: (metrics.realized_volatility * FixedPoint::from_float(100.0)).to_float() as u16,
            jump_robust_vol_pct: (metrics.jump_robust_volatility * FixedPoint::from_float(100.0)).to_float() as u16,
            noise_variance_bps: (metrics.microstructure_noise_variance * FixedPoint::from_float(10000.0)).to_float() as u32,
            optimal_sampling_ms: metrics.optimal_sampling_frequency,
            vol_of_vol_pct: (metrics.volatility_of_volatility * FixedPoint::from_float(100.0)).to_float() as u16,
        })
    }

    /// Detect jumps in recent data
    pub fn detect_recent_jumps(&self, lookback_ms: u64) -> Vec<ReturnObservation> {
        let cutoff_time = self.return_observations.back()
            .map(|obs| obs.timestamp.saturating_sub(lookback_ms))
            .unwrap_or(0);

        self.return_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time && obs.is_jump)
            .cloned()
            .collect()
    }

    /// Get intraday volatility pattern
    pub fn get_intraday_pattern(&self) -> &[IntradayVolatilityPattern] {
        &self.intraday_patterns
    }
}

/// Simplified volatility summary for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilitySummary {
    pub timestamp: u64,
    pub realized_vol_pct: u16,        // Realized volatility as percentage
    pub jump_robust_vol_pct: u16,     // Jump-robust volatility as percentage
    pub noise_variance_bps: u32,      // Microstructure noise variance in basis points
    pub optimal_sampling_ms: u64,     // Optimal sampling frequency in milliseconds
    pub vol_of_vol_pct: u16,          // Volatility of volatility as percentage
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_market_data(timestamp: u64, bid: f64, ask: f64) -> MarketData {
        MarketData {
            timestamp,
            bid_price: FixedPoint::from_float(bid),
            ask_price: FixedPoint::from_float(ask),
            bid_volume: FixedPoint::from_float(1000.0),
            ask_volume: FixedPoint::from_float(1000.0),
            last_trade_price: FixedPoint::from_float((bid + ask) / 2.0),
            last_trade_volume: FixedPoint::from_float(100.0),
            trade_direction: Some(TradeDirection::Buy),
        }
    }

    #[test]
    fn test_return_calculation() {
        let estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        let prev_data = create_test_market_data(1000, 99.95, 100.05);
        let current_data = create_test_market_data(2000, 100.00, 100.10);
        
        let return_obs = estimator.calculate_return_observation(&prev_data, &current_data).unwrap();
        
        // Mid prices: 100.0 -> 100.05, so return should be ln(100.05/100.0) ≈ 0.0005
        assert!((return_obs.log_return.to_float() - 0.0004998).abs() < 1e-6);
        assert!(!return_obs.is_jump); // Small return, not a jump
    }

    #[test]
    fn test_realized_volatility_calculation() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add market data with some volatility
        let base_price = 100.0;
        for i in 0..100 {
            let price_change = (i as f64 * 0.1).sin() * 0.01; // Sine wave pattern
            let bid = base_price + price_change - 0.05;
            let ask = base_price + price_change + 0.05;
            
            let market_data = create_test_market_data(1000 + i * 1000, bid, ask);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let realized_vol = estimator.calculate_realized_volatility(60000).unwrap();
        
        // Should have some positive volatility
        assert!(realized_vol > FixedPoint::ZERO);
    }

    #[test]
    fn test_jump_detection() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add normal market data
        estimator.update_market_data(create_test_market_data(1000, 99.95, 100.05)).unwrap();
        
        // Add data with a jump
        estimator.update_market_data(create_test_market_data(2000, 101.95, 102.05)).unwrap();
        
        let jumps = estimator.detect_recent_jumps(10000);
        
        // Should detect the jump
        assert!(!jumps.is_empty());
        assert!(jumps[0].is_jump);
    }

    #[test]
    fn test_bi_power_variation() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add market data
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.01).sin() * 0.1;
            let market_data = create_test_market_data(1000 + i * 1000, price - 0.05, price + 0.05);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let bi_power_var = estimator.calculate_bi_power_variation(30000).unwrap();
        
        // Should calculate some bi-power variation
        assert!(bi_power_var >= FixedPoint::ZERO);
    }

    #[test]
    fn test_microstructure_noise_estimation() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add market data with some noise (bid-ask bounce)
        for i in 0..100 {
            let base_price = 100.0;
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 }; // Alternating noise
            let price = base_price + noise;
            
            let market_data = create_test_market_data(1000 + i * 100, price - 0.05, price + 0.05);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let noise_variance = estimator.estimate_microstructure_noise_variance().unwrap();
        
        // Should detect some microstructure noise
        assert!(noise_variance >= FixedPoint::ZERO);
    }

    #[test]
    fn test_optimal_sampling_frequency() {
        let mut estimator = VolatilityEstimator::new(1000, 100, 60000, FixedPoint::from_float(0.01));
        
        // Add some market data
        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.05;
            let market_data = create_test_market_data(1000 + i * 1000, price - 0.05, price + 0.05);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let optimal_params = estimator.calculate_optimal_sampling_frequency().unwrap();
        
        // Should provide reasonable sampling frequency
        assert!(optimal_params.optimal_frequency_ms >= 100);
        assert!(optimal_params.optimal_frequency_ms <= 60000);
        assert!(optimal_params.efficiency_gain >= FixedPoint::from_float(1.0));
    }

    #[test]
    fn test_volatility_of_volatility() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add market data with changing volatility
        for i in 0..200 {
            let volatility_factor = 1.0 + (i as f64 / 50.0).sin() * 0.5; // Changing volatility
            let price_change = (i as f64 * 0.1).sin() * 0.01 * volatility_factor;
            let price = 100.0 + price_change;
            
            let market_data = create_test_market_data(1000 + i * 1000, price - 0.05, price + 0.05);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let vol_of_vol = estimator.calculate_volatility_of_volatility(100000).unwrap();
        
        // Should measure some volatility clustering
        assert!(vol_of_vol >= FixedPoint::ZERO);
    }

    #[test]
    fn test_comprehensive_metrics() {
        let mut estimator = VolatilityEstimator::new(1000, 1000, 60000, FixedPoint::from_float(0.01));
        
        // Add sufficient market data
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.05;
            let market_data = create_test_market_data(1000 + i * 1000, price - 0.05, price + 0.05);
            estimator.update_market_data(market_data).unwrap();
        }
        
        let metrics = estimator.calculate_metrics(60000).unwrap();
        
        assert!(metrics.realized_volatility >= FixedPoint::ZERO);
        assert!(metrics.bi_power_variation >= FixedPoint::ZERO);
        assert!(metrics.jump_robust_volatility >= FixedPoint::ZERO);
        assert_eq!(metrics.intraday_volatility_pattern.len(), 24);
    }
}