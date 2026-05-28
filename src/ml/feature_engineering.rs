//! Feature Engineering Module
//! 
//! Implements comprehensive feature extraction and engineering for market
//! microstructure data, including technical indicators, statistical features,
//! and domain-specific transformations.

use crate::ml::{MLError, MLResult};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    pub window_sizes: Vec<usize>,
    pub technical_indicators: Vec<String>,
    pub statistical_features: Vec<String>,
    pub microstructure_features: Vec<String>,
    pub normalization_method: String,
    pub feature_selection_threshold: f64,
    pub max_features: usize,
    pub rolling_window_size: usize,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            window_sizes: vec![5, 10, 20, 50, 100],
            technical_indicators: vec![
                "sma".to_string(),
                "ema".to_string(),
                "rsi".to_string(),
                "macd".to_string(),
                "bollinger_bands".to_string(),
                "atr".to_string(),
            ],
            statistical_features: vec![
                "mean".to_string(),
                "std".to_string(),
                "skewness".to_string(),
                "kurtosis".to_string(),
                "autocorrelation".to_string(),
            ],
            microstructure_features: vec![
                "bid_ask_spread".to_string(),
                "order_imbalance".to_string(),
                "trade_intensity".to_string(),
                "price_impact".to_string(),
                "volatility_clustering".to_string(),
            ],
            normalization_method: "z_score".to_string(),
            feature_selection_threshold: 0.05,
            max_features: 100,
            rolling_window_size: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub timestamp: SystemTime,
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub timestamp: SystemTime,
    pub price: f64,
    pub volume: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub trade_count: u32,
    pub vwap: f64,
}

/// Comprehensive feature engineering system for market data
pub struct FeatureEngineer {
    config: FeatureEngineeringConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    trade_history: VecDeque<MarketDataPoint>,
    feature_statistics: HashMap<String, FeatureStats>,
    technical_indicators: TechnicalIndicators,
    statistical_calculator: StatisticalFeatures,
    microstructure_analyzer: MicrostructureFeatures,
}

impl FeatureEngineer {
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self {
            technical_indicators: TechnicalIndicators::new(config.clone()),
            statistical_calculator: StatisticalFeatures::new(config.clone()),
            microstructure_analyzer: MicrostructureFeatures::new(config.clone()),
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            spread_history: VecDeque::new(),
            trade_history: VecDeque::new(),
            feature_statistics: HashMap::new(),
        }
    }

    /// Extract comprehensive feature vector from market data
    pub fn extract_features(&mut self, data_point: MarketDataPoint) -> MLResult<FeatureVector> {
        // Update history buffers
        self.update_history(&data_point);

        let mut features = Vec::new();
        let mut feature_names = Vec::new();

        // Extract technical indicator features
        let tech_features = self.technical_indicators.calculate(&self.price_history, &self.volume_history)?;
        features.extend(tech_features.values);
        feature_names.extend(tech_features.names);

        // Extract statistical features
        let stat_features = self.statistical_calculator.calculate(&self.price_history)?;
        features.extend(stat_features.values);
        feature_names.extend(stat_features.names);

        // Extract microstructure features
        let micro_features = self.microstructure_analyzer.calculate(&self.trade_history)?;
        features.extend(micro_features.values);
        feature_names.extend(micro_features.names);

        // Normalize features
        let normalized_features = self.normalize_features(&features, &feature_names)?;

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("data_quality".to_string(), self.assess_data_quality().to_string());
        metadata.insert("feature_count".to_string(), normalized_features.len().to_string());

        Ok(FeatureVector {
            timestamp: data_point.timestamp,
            features: normalized_features,
            feature_names,
            metadata,
        })
    }

    /// Update feature selection based on importance scores
    pub fn update_feature_selection(&mut self, importance_scores: &[(String, f64)]) -> MLResult<()> {
        // Sort features by importance
        let mut sorted_features = importance_scores.to_vec();
        sorted_features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top features
        let selected_features: Vec<String> = sorted_features
            .iter()
            .take(self.config.max_features)
            .filter(|(_, score)| *score > self.config.feature_selection_threshold)
            .map(|(name, _)| name.clone())
            .collect();

        // Update configuration to include only selected features
        self.update_feature_config(selected_features)?;

        Ok(())
    }

    /// Create polynomial and interaction features
    pub fn create_polynomial_features(&self, features: &[f64], degree: usize) -> MLResult<Vec<f64>> {
        let mut poly_features = features.to_vec();

        if degree >= 2 {
            // Add squared terms
            for &feature in features {
                poly_features.push(feature * feature);
            }

            // Add interaction terms
            for i in 0..features.len() {
                for j in (i + 1)..features.len() {
                    poly_features.push(features[i] * features[j]);
                }
            }
        }

        if degree >= 3 {
            // Add cubic terms
            for &feature in features {
                poly_features.push(feature * feature * feature);
            }
        }

        Ok(poly_features)
    }

    /// Apply feature transformations (log, sqrt, etc.)
    pub fn apply_transformations(&self, features: &[f64]) -> MLResult<Vec<f64>> {
        let mut transformed = Vec::new();

        for &feature in features {
            // Original feature
            transformed.push(feature);

            // Log transformation (for positive values)
            if feature > 0.0 {
                transformed.push(feature.ln());
            } else {
                transformed.push(0.0);
            }

            // Square root transformation (for non-negative values)
            if feature >= 0.0 {
                transformed.push(feature.sqrt());
            } else {
                transformed.push(0.0);
            }

            // Reciprocal transformation (avoid division by zero)
            if feature.abs() > 1e-10 {
                transformed.push(1.0 / feature);
            } else {
                transformed.push(0.0);
            }
        }

        Ok(transformed)
    }

    fn update_history(&mut self, data_point: &MarketDataPoint) {
        // Update price history
        self.price_history.push_back(data_point.price);
        if self.price_history.len() > self.config.rolling_window_size {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(data_point.volume);
        if self.volume_history.len() > self.config.rolling_window_size {
            self.volume_history.pop_front();
        }

        // Update spread history
        let spread = data_point.ask_price - data_point.bid_price;
        self.spread_history.push_back(spread);
        if self.spread_history.len() > self.config.rolling_window_size {
            self.spread_history.pop_front();
        }

        // Update trade history
        self.trade_history.push_back(data_point.clone());
        if self.trade_history.len() > self.config.rolling_window_size {
            self.trade_history.pop_front();
        }
    }

    fn normalize_features(&mut self, features: &[f64], feature_names: &[String]) -> MLResult<Vec<f64>> {
        let mut normalized = Vec::new();

        for (i, &feature) in features.iter().enumerate() {
            let feature_name = feature_names.get(i).unwrap_or(&format!("feature_{}", i));
            
            // Update feature statistics
            self.update_feature_stats(feature_name, feature);

            // Apply normalization
            let normalized_value = match self.config.normalization_method.as_str() {
                "z_score" => self.z_score_normalize(feature_name, feature),
                "min_max" => self.min_max_normalize(feature_name, feature),
                "robust" => self.robust_normalize(feature_name, feature),
                _ => feature, // No normalization
            };

            normalized.push(normalized_value);
        }

        Ok(normalized)
    }

    fn update_feature_stats(&mut self, feature_name: &str, value: f64) {
        let stats = self.feature_statistics
            .entry(feature_name.to_string())
            .or_insert_with(FeatureStats::new);
        
        stats.update(value);
    }

    fn z_score_normalize(&self, feature_name: &str, value: f64) -> f64 {
        if let Some(stats) = self.feature_statistics.get(feature_name) {
            if stats.std_dev > 1e-10 {
                (value - stats.mean) / stats.std_dev
            } else {
                0.0
            }
        } else {
            value
        }
    }

    fn min_max_normalize(&self, feature_name: &str, value: f64) -> f64 {
        if let Some(stats) = self.feature_statistics.get(feature_name) {
            let range = stats.max - stats.min;
            if range > 1e-10 {
                (value - stats.min) / range
            } else {
                0.0
            }
        } else {
            value
        }
    }

    fn robust_normalize(&self, feature_name: &str, value: f64) -> f64 {
        if let Some(stats) = self.feature_statistics.get(feature_name) {
            let iqr = stats.q75 - stats.q25;
            if iqr > 1e-10 {
                (value - stats.median) / iqr
            } else {
                0.0
            }
        } else {
            value
        }
    }

    fn assess_data_quality(&self) -> f64 {
        // Simple data quality assessment
        let price_quality = if self.price_history.len() > 10 {
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(10).cloned().collect();
            let price_changes: Vec<f64> = recent_prices.windows(2)
                .map(|w| (w[0] - w[1]).abs() / w[1])
                .collect();
            
            let avg_change = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
            if avg_change < 0.1 { 1.0 } else { 0.5 } // High quality if changes are reasonable
        } else {
            0.0
        };

        price_quality
    }

    fn update_feature_config(&mut self, selected_features: Vec<String>) -> MLResult<()> {
        // Update technical indicators
        self.config.technical_indicators.retain(|indicator| {
            selected_features.iter().any(|f| f.contains(indicator))
        });

        // Update statistical features
        self.config.statistical_features.retain(|stat| {
            selected_features.iter().any(|f| f.contains(stat))
        });

        // Update microstructure features
        self.config.microstructure_features.retain(|micro| {
            selected_features.iter().any(|f| f.contains(micro))
        });

        Ok(())
    }
}

/// Technical indicators calculator
pub struct TechnicalIndicators {
    config: FeatureEngineeringConfig,
    ema_state: HashMap<usize, f64>, // window_size -> current EMA value
}

impl TechnicalIndicators {
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self {
            config,
            ema_state: HashMap::new(),
        }
    }

    pub fn calculate(&mut self, prices: &VecDeque<f64>, volumes: &VecDeque<f64>) -> MLResult<FeatureSet> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        for &window in &self.config.window_sizes {
            if prices.len() >= window {
                let recent_prices: Vec<f64> = prices.iter().rev().take(window).cloned().collect();
                let recent_volumes: Vec<f64> = volumes.iter().rev().take(window).cloned().collect();

                // Simple Moving Average
                if self.config.technical_indicators.contains(&"sma".to_string()) {
                    let sma = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
                    features.push(sma);
                    names.push(format!("sma_{}", window));
                }

                // Exponential Moving Average
                if self.config.technical_indicators.contains(&"ema".to_string()) {
                    let ema = self.calculate_ema(&recent_prices, window);
                    features.push(ema);
                    names.push(format!("ema_{}", window));
                }

                // RSI
                if self.config.technical_indicators.contains(&"rsi".to_string()) {
                    let rsi = self.calculate_rsi(&recent_prices);
                    features.push(rsi);
                    names.push(format!("rsi_{}", window));
                }

                // MACD (only for larger windows)
                if window >= 26 && self.config.technical_indicators.contains(&"macd".to_string()) {
                    let (macd, signal, histogram) = self.calculate_macd(&recent_prices);
                    features.extend_from_slice(&[macd, signal, histogram]);
                    names.extend_from_slice(&[
                        format!("macd_{}", window),
                        format!("macd_signal_{}", window),
                        format!("macd_histogram_{}", window),
                    ]);
                }

                // Bollinger Bands
                if self.config.technical_indicators.contains(&"bollinger_bands".to_string()) {
                    let (upper, middle, lower) = self.calculate_bollinger_bands(&recent_prices);
                    features.extend_from_slice(&[upper, middle, lower]);
                    names.extend_from_slice(&[
                        format!("bb_upper_{}", window),
                        format!("bb_middle_{}", window),
                        format!("bb_lower_{}", window),
                    ]);
                }

                // Average True Range
                if self.config.technical_indicators.contains(&"atr".to_string()) {
                    let atr = self.calculate_atr(&recent_prices);
                    features.push(atr);
                    names.push(format!("atr_{}", window));
                }
            }
        }

        Ok(FeatureSet { values: features, names })
    }

    fn calculate_ema(&mut self, prices: &[f64], window: usize) -> f64 {
        let alpha = 2.0 / (window as f64 + 1.0);
        let current_price = prices.last().copied().unwrap_or(0.0);

        let ema = if let Some(&prev_ema) = self.ema_state.get(&window) {
            alpha * current_price + (1.0 - alpha) * prev_ema
        } else {
            // Initialize with SMA
            prices.iter().sum::<f64>() / prices.len() as f64
        };

        self.ema_state.insert(window, ema);
        ema
    }

    fn calculate_rsi(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 50.0; // Neutral RSI
        }

        let changes: Vec<f64> = prices.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        let gains: f64 = changes.iter().filter(|&&x| x > 0.0).sum();
        let losses: f64 = changes.iter().filter(|&&x| x < 0.0).map(|x| -x).sum();

        if losses == 0.0 {
            return 100.0;
        }

        let rs = gains / losses;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64, f64) {
        if prices.len() < 26 {
            return (0.0, 0.0, 0.0);
        }

        // Calculate EMAs
        let ema12 = self.calculate_simple_ema(prices, 12);
        let ema26 = self.calculate_simple_ema(prices, 26);
        let macd = ema12 - ema26;

        // Signal line (9-period EMA of MACD)
        let signal = macd * 0.2; // Simplified
        let histogram = macd - signal;

        (macd, signal, histogram)
    }

    fn calculate_simple_ema(&self, prices: &[f64], period: usize) -> f64 {
        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[0];

        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }

        ema
    }

    fn calculate_bollinger_bands(&self, prices: &[f64]) -> (f64, f64, f64) {
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();

        let upper = mean + 2.0 * std_dev;
        let lower = mean - 2.0 * std_dev;

        (upper, mean, lower)
    }

    fn calculate_atr(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let true_ranges: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .collect();

        true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
    }
}

/// Statistical features calculator
pub struct StatisticalFeatures {
    config: FeatureEngineeringConfig,
}

impl StatisticalFeatures {
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self { config }
    }

    pub fn calculate(&self, data: &VecDeque<f64>) -> MLResult<FeatureSet> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        for &window in &self.config.window_sizes {
            if data.len() >= window {
                let recent_data: Vec<f64> = data.iter().rev().take(window).cloned().collect();

                // Mean
                if self.config.statistical_features.contains(&"mean".to_string()) {
                    let mean = recent_data.iter().sum::<f64>() / recent_data.len() as f64;
                    features.push(mean);
                    names.push(format!("mean_{}", window));
                }

                // Standard deviation
                if self.config.statistical_features.contains(&"std".to_string()) {
                    let std_dev = self.calculate_std_dev(&recent_data);
                    features.push(std_dev);
                    names.push(format!("std_{}", window));
                }

                // Skewness
                if self.config.statistical_features.contains(&"skewness".to_string()) {
                    let skewness = self.calculate_skewness(&recent_data);
                    features.push(skewness);
                    names.push(format!("skewness_{}", window));
                }

                // Kurtosis
                if self.config.statistical_features.contains(&"kurtosis".to_string()) {
                    let kurtosis = self.calculate_kurtosis(&recent_data);
                    features.push(kurtosis);
                    names.push(format!("kurtosis_{}", window));
                }

                // Autocorrelation
                if self.config.statistical_features.contains(&"autocorrelation".to_string()) {
                    let autocorr = self.calculate_autocorrelation(&recent_data, 1);
                    features.push(autocorr);
                    names.push(format!("autocorr_{}", window));
                }
            }
        }

        Ok(FeatureSet { values: features, names })
    }

    fn calculate_std_dev(&self, data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    fn calculate_skewness(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std_dev = self.calculate_std_dev(data);

        if std_dev == 0.0 {
            return 0.0;
        }

        let skewness = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;

        skewness
    }

    fn calculate_kurtosis(&self, data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let std_dev = self.calculate_std_dev(data);

        if std_dev == 0.0 {
            return 0.0;
        }

        let kurtosis = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0; // Excess kurtosis

        kurtosis
    }

    fn calculate_autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        if data.len() <= lag {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        let covariance = data.iter()
            .take(data.len() - lag)
            .zip(data.iter().skip(lag))
            .map(|(x, y)| (x - mean) * (y - mean))
            .sum::<f64>() / (data.len() - lag) as f64;

        covariance / variance
    }
}

/// Microstructure features calculator
pub struct MicrostructureFeatures {
    config: FeatureEngineeringConfig,
}

impl MicrostructureFeatures {
    pub fn new(config: FeatureEngineeringConfig) -> Self {
        Self { config }
    }

    pub fn calculate(&self, trade_history: &VecDeque<MarketDataPoint>) -> MLResult<FeatureSet> {
        let mut features = Vec::new();
        let mut names = Vec::new();

        if trade_history.is_empty() {
            return Ok(FeatureSet { values: features, names });
        }

        // Bid-ask spread features
        if self.config.microstructure_features.contains(&"bid_ask_spread".to_string()) {
            let (avg_spread, spread_volatility) = self.calculate_spread_features(trade_history);
            features.extend_from_slice(&[avg_spread, spread_volatility]);
            names.extend_from_slice(&["avg_spread".to_string(), "spread_volatility".to_string()]);
        }

        // Order imbalance
        if self.config.microstructure_features.contains(&"order_imbalance".to_string()) {
            let imbalance = self.calculate_order_imbalance(trade_history);
            features.push(imbalance);
            names.push("order_imbalance".to_string());
        }

        // Trade intensity
        if self.config.microstructure_features.contains(&"trade_intensity".to_string()) {
            let intensity = self.calculate_trade_intensity(trade_history);
            features.push(intensity);
            names.push("trade_intensity".to_string());
        }

        // Price impact
        if self.config.microstructure_features.contains(&"price_impact".to_string()) {
            let impact = self.calculate_price_impact(trade_history);
            features.push(impact);
            names.push("price_impact".to_string());
        }

        // Volatility clustering
        if self.config.microstructure_features.contains(&"volatility_clustering".to_string()) {
            let clustering = self.calculate_volatility_clustering(trade_history);
            features.push(clustering);
            names.push("volatility_clustering".to_string());
        }

        Ok(FeatureSet { values: features, names })
    }

    fn calculate_spread_features(&self, trade_history: &VecDeque<MarketDataPoint>) -> (f64, f64) {
        let spreads: Vec<f64> = trade_history.iter()
            .map(|data| data.ask_price - data.bid_price)
            .collect();

        let avg_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let spread_variance = spreads.iter()
            .map(|s| (s - avg_spread).powi(2))
            .sum::<f64>() / spreads.len() as f64;
        let spread_volatility = spread_variance.sqrt();

        (avg_spread, spread_volatility)
    }

    fn calculate_order_imbalance(&self, trade_history: &VecDeque<MarketDataPoint>) -> f64 {
        let total_bid_volume: f64 = trade_history.iter().map(|data| data.bid_volume).sum();
        let total_ask_volume: f64 = trade_history.iter().map(|data| data.ask_volume).sum();
        let total_volume = total_bid_volume + total_ask_volume;

        if total_volume > 0.0 {
            (total_bid_volume - total_ask_volume) / total_volume
        } else {
            0.0
        }
    }

    fn calculate_trade_intensity(&self, trade_history: &VecDeque<MarketDataPoint>) -> f64 {
        if trade_history.len() < 2 {
            return 0.0;
        }

        let time_span = trade_history.back().unwrap().timestamp
            .duration_since(trade_history.front().unwrap().timestamp)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64();

        let total_trades: u32 = trade_history.iter().map(|data| data.trade_count).sum();
        total_trades as f64 / time_span
    }

    fn calculate_price_impact(&self, trade_history: &VecDeque<MarketDataPoint>) -> f64 {
        if trade_history.len() < 2 {
            return 0.0;
        }

        let price_changes: Vec<f64> = trade_history.windows(2)
            .map(|w| (w[1].price - w[0].price).abs())
            .collect();

        let volume_changes: Vec<f64> = trade_history.windows(2)
            .map(|w| w[1].volume)
            .collect();

        // Calculate correlation between volume and price impact
        if price_changes.len() != volume_changes.len() || price_changes.is_empty() {
            return 0.0;
        }

        let price_mean = price_changes.iter().sum::<f64>() / price_changes.len() as f64;
        let volume_mean = volume_changes.iter().sum::<f64>() / volume_changes.len() as f64;

        let covariance = price_changes.iter()
            .zip(volume_changes.iter())
            .map(|(p, v)| (p - price_mean) * (v - volume_mean))
            .sum::<f64>() / price_changes.len() as f64;

        let price_variance = price_changes.iter()
            .map(|p| (p - price_mean).powi(2))
            .sum::<f64>() / price_changes.len() as f64;

        let volume_variance = volume_changes.iter()
            .map(|v| (v - volume_mean).powi(2))
            .sum::<f64>() / volume_changes.len() as f64;

        if price_variance > 0.0 && volume_variance > 0.0 {
            covariance / (price_variance.sqrt() * volume_variance.sqrt())
        } else {
            0.0
        }
    }

    fn calculate_volatility_clustering(&self, trade_history: &VecDeque<MarketDataPoint>) -> f64 {
        if trade_history.len() < 10 {
            return 0.0;
        }

        let returns: Vec<f64> = trade_history.windows(2)
            .map(|w| (w[1].price / w[0].price).ln())
            .collect();

        let squared_returns: Vec<f64> = returns.iter().map(|r| r * r).collect();

        // Calculate autocorrelation of squared returns (GARCH effect)
        let mean_sq_return = squared_returns.iter().sum::<f64>() / squared_returns.len() as f64;
        let variance = squared_returns.iter()
            .map(|r| (r - mean_sq_return).powi(2))
            .sum::<f64>() / squared_returns.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        // Lag-1 autocorrelation
        let covariance = squared_returns.iter()
            .take(squared_returns.len() - 1)
            .zip(squared_returns.iter().skip(1))
            .map(|(r1, r2)| (r1 - mean_sq_return) * (r2 - mean_sq_return))
            .sum::<f64>() / (squared_returns.len() - 1) as f64;

        covariance / variance
    }
}

#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureStats {
    pub count: usize,
    pub sum: f64,
    pub sum_sq: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub values: VecDeque<f64>, // For quantile calculations
}

impl FeatureStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            mean: 0.0,
            std_dev: 0.0,
            median: 0.0,
            q25: 0.0,
            q75: 0.0,
            values: VecDeque::new(),
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Update running statistics
        self.mean = self.sum / self.count as f64;
        if self.count > 1 {
            let variance = (self.sum_sq - self.sum * self.sum / self.count as f64) / (self.count - 1) as f64;
            self.std_dev = variance.sqrt();
        }

        // Keep values for quantile calculations (limited buffer)
        self.values.push_back(value);
        if self.values.len() > 1000 {
            self.values.pop_front();
        }

        // Update quantiles
        self.update_quantiles();
    }

    fn update_quantiles(&mut self) {
        if self.values.is_empty() {
            return;
        }

        let mut sorted_values: Vec<f64> = self.values.iter().cloned().collect();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len();
        self.median = sorted_values[n / 2];
        self.q25 = sorted_values[n / 4];
        self.q75 = sorted_values[3 * n / 4];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_engineer_creation() {
        let config = FeatureEngineeringConfig::default();
        let engineer = FeatureEngineer::new(config);
        assert_eq!(engineer.config.window_sizes.len(), 5);
    }

    #[test]
    fn test_technical_indicators() {
        let config = FeatureEngineeringConfig::default();
        let mut indicators = TechnicalIndicators::new(config);
        
        let prices: VecDeque<f64> = (1..=20).map(|i| i as f64).collect();
        let volumes: VecDeque<f64> = (1..=20).map(|i| (i * 100) as f64).collect();
        
        let result = indicators.calculate(&prices, &volumes);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(!features.values.is_empty());
        assert_eq!(features.values.len(), features.names.len());
    }

    #[test]
    fn test_statistical_features() {
        let config = FeatureEngineeringConfig::default();
        let calculator = StatisticalFeatures::new(config);
        
        let data: VecDeque<f64> = (1..=50).map(|i| i as f64).collect();
        
        let result = calculator.calculate(&data);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(!features.values.is_empty());
    }

    #[test]
    fn test_feature_stats() {
        let mut stats = FeatureStats::new();
        
        for i in 1..=10 {
            stats.update(i as f64);
        }
        
        assert_eq!(stats.count, 10);
        assert_eq!(stats.mean, 5.5);
        assert!(stats.std_dev > 0.0);
    }
}