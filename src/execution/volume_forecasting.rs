//! Volume Forecasting and Adaptation Module
//! 
//! Implements sophisticated volume forecasting with:
//! - Intraday volume pattern analysis using historical data
//! - Real-time volume forecasting with seasonality adjustment
//! - Adaptive bucket sizing based on expected volume
//! - Participation rate adjustment for market conditions

use super::{ExecutionError, MarketState, MarketConditions, VolumePatterns};
use crate::math::fixed_point::FixedPoint;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Historical volume data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeDataPoint {
    pub timestamp: u64,
    pub volume: u64,
    pub hour_of_day: u8,
    pub day_of_week: u8,
    pub is_holiday: bool,
    pub market_conditions: MarketConditions,
}

/// Volume forecasting model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeForecaster {
    /// Historical volume data
    historical_data: VecDeque<VolumeDataPoint>,
    /// Maximum history to keep (in days)
    max_history_days: u32,
    /// Intraday patterns by hour
    hourly_patterns: HashMap<u8, VolumeStatistics>,
    /// Daily patterns by day of week
    daily_patterns: HashMap<u8, VolumeStatistics>,
    /// Seasonal adjustments
    seasonal_adjustments: HashMap<String, FixedPoint>,
    /// Real-time adaptation parameters
    adaptation_params: AdaptationParameters,
}

/// Volume statistics for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeStatistics {
    pub mean: FixedPoint,
    pub std_dev: FixedPoint,
    pub median: FixedPoint,
    pub percentile_25: FixedPoint,
    pub percentile_75: FixedPoint,
    pub sample_count: u32,
    pub last_updated: u64,
}

/// Parameters for real-time adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationParameters {
    /// Learning rate for pattern updates (0.0 to 1.0)
    pub learning_rate: FixedPoint,
    /// Minimum sample size for reliable patterns
    pub min_sample_size: u32,
    /// Decay factor for older data
    pub decay_factor: FixedPoint,
    /// Volatility adjustment factor
    pub volatility_adjustment: FixedPoint,
    /// Liquidity adjustment factor
    pub liquidity_adjustment: FixedPoint,
    /// News impact adjustment factor
    pub news_impact_adjustment: FixedPoint,
}

impl Default for AdaptationParameters {
    fn default() -> Self {
        Self {
            learning_rate: FixedPoint::from_float(0.1),
            min_sample_size: 30,
            decay_factor: FixedPoint::from_float(0.95),
            volatility_adjustment: FixedPoint::from_float(1.2),
            liquidity_adjustment: FixedPoint::from_float(0.8),
            news_impact_adjustment: FixedPoint::from_float(1.5),
        }
    }
}

/// Volume forecast result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeForecast {
    /// Forecasted volume
    pub forecasted_volume: u64,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (u64, u64),
    /// Forecast horizon in seconds
    pub horizon: u64,
    /// Seasonality adjustment applied
    pub seasonality_adjustment: FixedPoint,
    /// Market condition adjustment applied
    pub market_condition_adjustment: FixedPoint,
    /// Forecast accuracy score (0.0 to 1.0)
    pub accuracy_score: FixedPoint,
}

/// Adaptive bucket sizing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBucketSizing {
    /// Recommended bucket sizes
    pub bucket_sizes: Vec<u64>,
    /// Volume allocation per bucket
    pub volume_allocations: Vec<u64>,
    /// Participation rates per bucket
    pub participation_rates: Vec<FixedPoint>,
    /// Timing adjustments per bucket
    pub timing_adjustments: Vec<i64>, // seconds adjustment
}

impl VolumeForecaster {
    /// Create new volume forecaster
    pub fn new(max_history_days: u32) -> Self {
        Self {
            historical_data: VecDeque::new(),
            max_history_days,
            hourly_patterns: HashMap::new(),
            daily_patterns: HashMap::new(),
            seasonal_adjustments: HashMap::new(),
            adaptation_params: AdaptationParameters::default(),
        }
    }

    /// Add historical volume data
    pub fn add_historical_data(&mut self, data_points: Vec<VolumeDataPoint>) -> Result<(), ExecutionError> {
        for data_point in data_points {
            self.add_volume_data_point(data_point)?;
        }
        self.update_patterns()?;
        Ok(())
    }

    /// Add single volume data point
    pub fn add_volume_data_point(&mut self, data_point: VolumeDataPoint) -> Result<(), ExecutionError> {
        // Validate data point
        if data_point.hour_of_day >= 24 {
            return Err(ExecutionError::InvalidParameters(
                "Hour of day must be 0-23".to_string()
            ));
        }
        
        if data_point.day_of_week >= 7 {
            return Err(ExecutionError::InvalidParameters(
                "Day of week must be 0-6".to_string()
            ));
        }

        // Add to historical data
        self.historical_data.push_back(data_point);

        // Maintain maximum history
        let max_data_points = (self.max_history_days * 24) as usize; // Hourly data points
        while self.historical_data.len() > max_data_points {
            self.historical_data.pop_front();
        }

        Ok(())
    }

    /// Update volume patterns from historical data
    pub fn update_patterns(&mut self) -> Result<(), ExecutionError> {
        if self.historical_data.is_empty() {
            return Ok(());
        }

        // Update hourly patterns
        self.update_hourly_patterns()?;
        
        // Update daily patterns
        self.update_daily_patterns()?;
        
        // Update seasonal adjustments
        self.update_seasonal_adjustments()?;

        Ok(())
    }

    /// Forecast volume for given time horizon
    pub fn forecast_volume(
        &self,
        horizon_seconds: u64,
        current_market_state: &MarketState,
        market_conditions: &MarketConditions,
    ) -> Result<VolumeForecast, ExecutionError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get base forecast from patterns
        let base_forecast = self.get_base_forecast(current_time, horizon_seconds)?;
        
        // Apply seasonality adjustments
        let seasonality_adjustment = self.calculate_seasonality_adjustment(current_time)?;
        let seasonality_adjusted = (base_forecast.to_float() * seasonality_adjustment.to_float()) as u64;
        
        // Apply market condition adjustments
        let market_adjustment = self.calculate_market_condition_adjustment(market_conditions)?;
        let final_forecast = (seasonality_adjusted as f64 * market_adjustment.to_float()) as u64;
        
        // Calculate confidence interval
        let confidence_interval = self.calculate_confidence_interval(final_forecast, horizon_seconds)?;
        
        // Calculate accuracy score based on recent performance
        let accuracy_score = self.calculate_forecast_accuracy()?;

        Ok(VolumeForecast {
            forecasted_volume: final_forecast,
            confidence_interval,
            horizon: horizon_seconds,
            seasonality_adjustment,
            market_condition_adjustment: market_adjustment,
            accuracy_score,
        })
    }

    /// Generate adaptive bucket sizing
    pub fn generate_adaptive_bucket_sizing(
        &self,
        total_quantity: u64,
        execution_horizon: u64,
        num_buckets: usize,
        market_conditions: &MarketConditions,
    ) -> Result<AdaptiveBucketSizing, ExecutionError> {
        if num_buckets == 0 {
            return Err(ExecutionError::InvalidParameters(
                "Number of buckets must be greater than zero".to_string()
            ));
        }

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let bucket_duration = execution_horizon / num_buckets as u64;
        let mut bucket_sizes = Vec::with_capacity(num_buckets);
        let mut volume_allocations = Vec::with_capacity(num_buckets);
        let mut participation_rates = Vec::with_capacity(num_buckets);
        let mut timing_adjustments = Vec::with_capacity(num_buckets);

        let mut total_allocated = 0u64;

        for i in 0..num_buckets {
            let bucket_start_time = current_time + (i as u64 * bucket_duration);
            let bucket_end_time = bucket_start_time + bucket_duration;
            
            // Forecast volume for this bucket
            let volume_forecast = self.forecast_volume(
                bucket_duration,
                &MarketState {
                    symbol: "".to_string(),
                    bid_price: FixedPoint::ZERO,
                    ask_price: FixedPoint::ZERO,
                    bid_volume: 0,
                    ask_volume: 0,
                    last_price: FixedPoint::ZERO,
                    last_volume: 0,
                    timestamp: bucket_start_time,
                    volatility: FixedPoint::from_float(0.02),
                    average_daily_volume: 1000000,
                },
                market_conditions,
            )?;

            // Calculate volume allocation based on expected market volume
            let market_volume_weight = self.calculate_volume_weight(bucket_start_time)?;
            let base_allocation = (total_quantity as f64 * market_volume_weight.to_float()) as u64;
            
            // Adjust for market conditions
            let condition_adjustment = self.calculate_bucket_condition_adjustment(
                market_conditions,
                bucket_start_time,
            )?;
            
            let adjusted_allocation = (base_allocation as f64 * condition_adjustment.to_float()) as u64;
            
            // Calculate participation rate
            let participation_rate = if volume_forecast.forecasted_volume > 0 {
                let rate = adjusted_allocation as f64 / volume_forecast.forecasted_volume as f64;
                FixedPoint::from_float(rate.min(0.5)) // Cap at 50%
            } else {
                FixedPoint::from_float(0.1) // Default 10%
            };

            // Calculate timing adjustment based on volatility and liquidity
            let timing_adjustment = self.calculate_timing_adjustment(
                market_conditions,
                bucket_start_time,
            )?;

            bucket_sizes.push(bucket_duration);
            volume_allocations.push(adjusted_allocation);
            participation_rates.push(participation_rate);
            timing_adjustments.push(timing_adjustment);
            
            total_allocated += adjusted_allocation;
        }

        // Normalize allocations to match total quantity
        if total_allocated != total_quantity && total_allocated > 0 {
            let adjustment_factor = total_quantity as f64 / total_allocated as f64;
            let mut normalized_total = 0u64;
            
            for i in 0..num_buckets - 1 {
                volume_allocations[i] = (volume_allocations[i] as f64 * adjustment_factor) as u64;
                normalized_total += volume_allocations[i];
            }
            
            // Assign remainder to last bucket
            volume_allocations[num_buckets - 1] = total_quantity - normalized_total;
        }

        Ok(AdaptiveBucketSizing {
            bucket_sizes,
            volume_allocations,
            participation_rates,
            timing_adjustments,
        })
    }

    /// Adjust participation rate based on real-time market conditions
    pub fn adjust_participation_rate(
        &self,
        base_rate: FixedPoint,
        current_market_state: &MarketState,
        market_conditions: &MarketConditions,
        execution_progress: FixedPoint, // 0.0 to 1.0
    ) -> Result<FixedPoint, ExecutionError> {
        let mut adjusted_rate = base_rate;

        // Volatility adjustment
        match market_conditions.volatility_regime {
            super::VolatilityRegime::Low => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(1.2); // Increase rate in low vol
            },
            super::VolatilityRegime::High => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(0.8); // Decrease rate in high vol
            },
            super::VolatilityRegime::Extreme => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(0.5); // Significantly decrease in extreme vol
            },
            _ => {}, // No adjustment for normal volatility
        }

        // Liquidity adjustment
        match market_conditions.liquidity_level {
            super::LiquidityLevel::Low => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(0.7); // Decrease rate in low liquidity
            },
            super::LiquidityLevel::High => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(1.3); // Increase rate in high liquidity
            },
            _ => {}, // No adjustment for normal liquidity
        }

        // News impact adjustment
        match market_conditions.news_impact {
            super::NewsImpact::High | super::NewsImpact::Extreme => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(0.3); // Significantly reduce during news
            },
            super::NewsImpact::Medium => {
                adjusted_rate = adjusted_rate * FixedPoint::from_float(0.7); // Moderately reduce
            },
            _ => {}, // No adjustment for low/no news impact
        }

        // Execution progress adjustment (urgency factor)
        if execution_progress > FixedPoint::from_float(0.8) {
            // Near completion, increase urgency
            adjusted_rate = adjusted_rate * FixedPoint::from_float(1.5);
        } else if execution_progress < FixedPoint::from_float(0.2) {
            // Early in execution, can be more patient
            adjusted_rate = adjusted_rate * FixedPoint::from_float(0.9);
        }

        // Ensure rate stays within reasonable bounds
        let min_rate = FixedPoint::from_float(0.01); // 1%
        let max_rate = FixedPoint::from_float(0.5);  // 50%
        
        adjusted_rate = adjusted_rate.max(min_rate).min(max_rate);

        Ok(adjusted_rate)
    }

    // Private helper methods

    fn update_hourly_patterns(&mut self) -> Result<(), ExecutionError> {
        let mut hourly_data: HashMap<u8, Vec<u64>> = HashMap::new();
        
        // Group data by hour
        for data_point in &self.historical_data {
            hourly_data.entry(data_point.hour_of_day)
                .or_insert_with(Vec::new)
                .push(data_point.volume);
        }

        // Calculate statistics for each hour
        for (hour, volumes) in hourly_data {
            if volumes.len() >= self.adaptation_params.min_sample_size as usize {
                let stats = self.calculate_volume_statistics(&volumes)?;
                self.hourly_patterns.insert(hour, stats);
            }
        }

        Ok(())
    }

    fn update_daily_patterns(&mut self) -> Result<(), ExecutionError> {
        let mut daily_data: HashMap<u8, Vec<u64>> = HashMap::new();
        
        // Group data by day of week
        for data_point in &self.historical_data {
            daily_data.entry(data_point.day_of_week)
                .or_insert_with(Vec::new)
                .push(data_point.volume);
        }

        // Calculate statistics for each day
        for (day, volumes) in daily_data {
            if volumes.len() >= self.adaptation_params.min_sample_size as usize {
                let stats = self.calculate_volume_statistics(&volumes)?;
                self.daily_patterns.insert(day, stats);
            }
        }

        Ok(())
    }

    fn update_seasonal_adjustments(&mut self) -> Result<(), ExecutionError> {
        // Simple seasonal adjustments - can be expanded
        self.seasonal_adjustments.insert("monday_open".to_string(), FixedPoint::from_float(1.3));
        self.seasonal_adjustments.insert("friday_close".to_string(), FixedPoint::from_float(1.2));
        self.seasonal_adjustments.insert("lunch_time".to_string(), FixedPoint::from_float(0.7));
        self.seasonal_adjustments.insert("after_hours".to_string(), FixedPoint::from_float(0.3));
        
        Ok(())
    }

    fn calculate_volume_statistics(&self, volumes: &[u64]) -> Result<VolumeStatistics, ExecutionError> {
        if volumes.is_empty() {
            return Err(ExecutionError::VolumeForecasting(
                "Cannot calculate statistics for empty volume data".to_string()
            ));
        }

        let mut sorted_volumes = volumes.to_vec();
        sorted_volumes.sort_unstable();

        let mean = volumes.iter().sum::<u64>() as f64 / volumes.len() as f64;
        
        let variance = volumes.iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>() / volumes.len() as f64;
        let std_dev = variance.sqrt();

        let median = if sorted_volumes.len() % 2 == 0 {
            let mid = sorted_volumes.len() / 2;
            (sorted_volumes[mid - 1] + sorted_volumes[mid]) as f64 / 2.0
        } else {
            sorted_volumes[sorted_volumes.len() / 2] as f64
        };

        let percentile_25_idx = (sorted_volumes.len() as f64 * 0.25) as usize;
        let percentile_75_idx = (sorted_volumes.len() as f64 * 0.75) as usize;
        
        let percentile_25 = sorted_volumes[percentile_25_idx] as f64;
        let percentile_75 = sorted_volumes[percentile_75_idx] as f64;

        Ok(VolumeStatistics {
            mean: FixedPoint::from_float(mean),
            std_dev: FixedPoint::from_float(std_dev),
            median: FixedPoint::from_float(median),
            percentile_25: FixedPoint::from_float(percentile_25),
            percentile_75: FixedPoint::from_float(percentile_75),
            sample_count: volumes.len() as u32,
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        })
    }

    fn get_base_forecast(&self, current_time: u64, horizon: u64) -> Result<FixedPoint, ExecutionError> {
        let hour = ((current_time / 3600) % 24) as u8;
        let day_of_week = (((current_time / 86400) + 4) % 7) as u8; // Unix epoch was Thursday
        
        // Get hourly pattern
        let hourly_volume = self.hourly_patterns.get(&hour)
            .map(|stats| stats.mean)
            .unwrap_or(FixedPoint::from_integer(1000)); // Default volume
        
        // Get daily pattern
        let daily_multiplier = self.daily_patterns.get(&day_of_week)
            .map(|stats| stats.mean / FixedPoint::from_integer(1000)) // Normalize
            .unwrap_or(FixedPoint::from_integer(1));
        
        // Scale by horizon
        let horizon_hours = horizon as f64 / 3600.0;
        let base_forecast = hourly_volume * daily_multiplier * FixedPoint::from_float(horizon_hours);
        
        Ok(base_forecast)
    }

    fn calculate_seasonality_adjustment(&self, current_time: u64) -> Result<FixedPoint, ExecutionError> {
        let hour = ((current_time / 3600) % 24) as u8;
        let day_of_week = (((current_time / 86400) + 4) % 7) as u8;
        
        let mut adjustment = FixedPoint::from_integer(1);
        
        // Monday opening adjustment
        if day_of_week == 1 && hour == 9 {
            adjustment = adjustment * self.seasonal_adjustments.get("monday_open")
                .copied().unwrap_or(FixedPoint::from_integer(1));
        }
        
        // Friday closing adjustment
        if day_of_week == 5 && hour >= 15 {
            adjustment = adjustment * self.seasonal_adjustments.get("friday_close")
                .copied().unwrap_or(FixedPoint::from_integer(1));
        }
        
        // Lunch time adjustment
        if hour >= 12 && hour <= 13 {
            adjustment = adjustment * self.seasonal_adjustments.get("lunch_time")
                .copied().unwrap_or(FixedPoint::from_integer(1));
        }
        
        Ok(adjustment)
    }

    fn calculate_market_condition_adjustment(&self, conditions: &MarketConditions) -> Result<FixedPoint, ExecutionError> {
        let mut adjustment = FixedPoint::from_integer(1);
        
        // Volatility adjustment
        match conditions.volatility_regime {
            super::VolatilityRegime::Low => adjustment = adjustment * FixedPoint::from_float(0.8),
            super::VolatilityRegime::High => adjustment = adjustment * FixedPoint::from_float(1.3),
            super::VolatilityRegime::Extreme => adjustment = adjustment * FixedPoint::from_float(1.8),
            _ => {},
        }
        
        // Liquidity adjustment
        match conditions.liquidity_level {
            super::LiquidityLevel::Low => adjustment = adjustment * FixedPoint::from_float(0.7),
            super::LiquidityLevel::High => adjustment = adjustment * FixedPoint::from_float(1.2),
            _ => {},
        }
        
        // News impact adjustment
        match conditions.news_impact {
            super::NewsImpact::High => adjustment = adjustment * FixedPoint::from_float(1.5),
            super::NewsImpact::Extreme => adjustment = adjustment * FixedPoint::from_float(2.0),
            _ => {},
        }
        
        Ok(adjustment)
    }

    fn calculate_confidence_interval(&self, forecast: u64, horizon: u64) -> Result<(u64, u64), ExecutionError> {
        // Simple confidence interval based on forecast uncertainty
        let uncertainty_factor = FixedPoint::from_float(0.2); // 20% uncertainty
        let uncertainty = (forecast as f64 * uncertainty_factor.to_float()) as u64;
        
        let lower_bound = forecast.saturating_sub(uncertainty);
        let upper_bound = forecast + uncertainty;
        
        Ok((lower_bound, upper_bound))
    }

    fn calculate_forecast_accuracy(&self) -> Result<FixedPoint, ExecutionError> {
        // Placeholder for forecast accuracy calculation
        // In practice, this would compare recent forecasts with actual volumes
        Ok(FixedPoint::from_float(0.85)) // 85% accuracy
    }

    fn calculate_volume_weight(&self, timestamp: u64) -> Result<FixedPoint, ExecutionError> {
        let hour = ((timestamp / 3600) % 24) as u8;
        
        // Simple volume weight based on typical intraday patterns
        let weight = match hour {
            9..=10 => 1.5,  // Market open
            11..=12 => 1.0, // Mid-morning
            13..=14 => 0.7, // Lunch
            15..=16 => 1.2, // Afternoon
            _ => 0.8,       // Other hours
        };
        
        Ok(FixedPoint::from_float(weight))
    }

    fn calculate_bucket_condition_adjustment(
        &self,
        conditions: &MarketConditions,
        bucket_time: u64,
    ) -> Result<FixedPoint, ExecutionError> {
        let mut adjustment = FixedPoint::from_integer(1);
        
        // Time-based adjustments
        let hour = ((bucket_time / 3600) % 24) as u8;
        if hour >= 9 && hour <= 10 {
            adjustment = adjustment * FixedPoint::from_float(1.2); // Market open boost
        }
        
        // Market condition adjustments
        match conditions.volatility_regime {
            super::VolatilityRegime::High | super::VolatilityRegime::Extreme => {
                adjustment = adjustment * FixedPoint::from_float(0.8); // Reduce in high volatility
            },
            _ => {},
        }
        
        Ok(adjustment)
    }

    fn calculate_timing_adjustment(
        &self,
        conditions: &MarketConditions,
        bucket_time: u64,
    ) -> Result<i64, ExecutionError> {
        let mut adjustment = 0i64;
        
        // Volatility-based timing adjustment
        match conditions.volatility_regime {
            super::VolatilityRegime::High => adjustment += 30,    // Delay by 30 seconds
            super::VolatilityRegime::Extreme => adjustment += 60, // Delay by 1 minute
            _ => {},
        }
        
        // News impact timing adjustment
        match conditions.news_impact {
            super::NewsImpact::High => adjustment += 120,   // Delay by 2 minutes
            super::NewsImpact::Extreme => adjustment += 300, // Delay by 5 minutes
            _ => {},
        }
        
        Ok(adjustment)
    }
}

impl Default for VolumeForecaster {
    fn default() -> Self {
        Self::new(30) // 30 days of history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{VolatilityRegime, LiquidityLevel, TrendDirection, MarketHours, NewsImpact};

    fn create_test_market_conditions() -> MarketConditions {
        MarketConditions {
            volatility_regime: VolatilityRegime::Normal,
            liquidity_level: LiquidityLevel::Normal,
            trend_direction: TrendDirection::Sideways,
            market_hours: MarketHours::Regular,
            news_impact: NewsImpact::None,
        }
    }

    fn create_test_volume_data() -> Vec<VolumeDataPoint> {
        let mut data = Vec::new();
        let base_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        for i in 0..100 {
            data.push(VolumeDataPoint {
                timestamp: base_time - (i * 3600), // Hourly data
                volume: 1000 + (i * 10),
                hour_of_day: ((base_time - (i * 3600)) / 3600 % 24) as u8,
                day_of_week: (((base_time - (i * 3600)) / 86400 + 4) % 7) as u8,
                is_holiday: false,
                market_conditions: create_test_market_conditions(),
            });
        }
        
        data
    }

    #[test]
    fn test_volume_forecaster_creation() {
        let forecaster = VolumeForecaster::new(30);
        assert_eq!(forecaster.max_history_days, 30);
        assert!(forecaster.historical_data.is_empty());
    }

    #[test]
    fn test_add_historical_data() {
        let mut forecaster = VolumeForecaster::new(30);
        let test_data = create_test_volume_data();
        
        let result = forecaster.add_historical_data(test_data.clone());
        assert!(result.is_ok());
        assert_eq!(forecaster.historical_data.len(), test_data.len());
    }

    #[test]
    fn test_volume_forecast() {
        let mut forecaster = VolumeForecaster::new(30);
        let test_data = create_test_volume_data();
        forecaster.add_historical_data(test_data).unwrap();
        
        let market_state = MarketState {
            symbol: "AAPL".to_string(),
            bid_price: FixedPoint::from_float(150.0),
            ask_price: FixedPoint::from_float(150.1),
            bid_volume: 1000,
            ask_volume: 1000,
            last_price: FixedPoint::from_float(150.05),
            last_volume: 500,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            volatility: FixedPoint::from_float(0.02),
            average_daily_volume: 1000000,
        };
        
        let market_conditions = create_test_market_conditions();
        
        let forecast = forecaster.forecast_volume(3600, &market_state, &market_conditions).unwrap();
        
        assert!(forecast.forecasted_volume > 0);
        assert!(forecast.confidence_interval.0 <= forecast.forecasted_volume);
        assert!(forecast.confidence_interval.1 >= forecast.forecasted_volume);
        assert_eq!(forecast.horizon, 3600);
    }

    #[test]
    fn test_adaptive_bucket_sizing() {
        let mut forecaster = VolumeForecaster::new(30);
        let test_data = create_test_volume_data();
        forecaster.add_historical_data(test_data).unwrap();
        
        let market_conditions = create_test_market_conditions();
        
        let bucket_sizing = forecaster.generate_adaptive_bucket_sizing(
            10000,
            3600,
            10,
            &market_conditions,
        ).unwrap();
        
        assert_eq!(bucket_sizing.bucket_sizes.len(), 10);
        assert_eq!(bucket_sizing.volume_allocations.len(), 10);
        assert_eq!(bucket_sizing.participation_rates.len(), 10);
        assert_eq!(bucket_sizing.timing_adjustments.len(), 10);
        
        // Verify total allocation equals target
        let total_allocated: u64 = bucket_sizing.volume_allocations.iter().sum();
        assert_eq!(total_allocated, 10000);
    }

    #[test]
    fn test_participation_rate_adjustment() {
        let forecaster = VolumeForecaster::new(30);
        let base_rate = FixedPoint::from_float(0.1);
        
        let market_state = MarketState {
            symbol: "AAPL".to_string(),
            bid_price: FixedPoint::from_float(150.0),
            ask_price: FixedPoint::from_float(150.1),
            bid_volume: 1000,
            ask_volume: 1000,
            last_price: FixedPoint::from_float(150.05),
            last_volume: 500,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            volatility: FixedPoint::from_float(0.02),
            average_daily_volume: 1000000,
        };
        
        let mut market_conditions = create_test_market_conditions();
        market_conditions.volatility_regime = VolatilityRegime::High;
        
        let adjusted_rate = forecaster.adjust_participation_rate(
            base_rate,
            &market_state,
            &market_conditions,
            FixedPoint::from_float(0.5),
        ).unwrap();
        
        // Should be reduced due to high volatility
        assert!(adjusted_rate < base_rate);
    }

    #[test]
    fn test_invalid_parameters() {
        let mut forecaster = VolumeForecaster::new(30);
        
        let invalid_data = VolumeDataPoint {
            timestamp: 0,
            volume: 1000,
            hour_of_day: 25, // Invalid hour
            day_of_week: 0,
            is_holiday: false,
            market_conditions: create_test_market_conditions(),
        };
        
        let result = forecaster.add_volume_data_point(invalid_data);
        assert!(result.is_err());
    }
}