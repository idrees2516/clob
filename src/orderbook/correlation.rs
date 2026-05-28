//! Cross-Asset Correlation Analysis
//!
//! This module provides real-time correlation analysis between different trading symbols,
//! supporting risk management and portfolio optimization decisions.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::types::Symbol;

/// Errors related to correlation analysis
#[derive(Error, Debug, Clone)]
pub enum CorrelationError {
    #[error("Insufficient data for correlation calculation: {symbol}")]
    InsufficientData { symbol: String },
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
    #[error("Matrix calculation error: {0}")]
    MatrixError(String),
    #[error("Invalid time window: {0}")]
    InvalidTimeWindow(String),
}

/// Price observation for correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceObservation {
    pub price: f64,
    pub timestamp: u64,
    pub log_return: Option<f64>,
}

/// Correlation analyzer for multiple assets
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    /// Price history for each symbol
    price_histories: HashMap<Symbol, VecDeque<PriceObservation>>,
    
    /// Maximum number of observations to keep per symbol
    max_observations: usize,
    
    /// Minimum observations required for correlation calculation
    min_observations: usize,
    
    /// Time window for correlation calculation (nanoseconds)
    correlation_window: u64,
    
    /// Cached correlation matrix
    cached_correlation: Option<CachedCorrelation>,
    
    /// Cache validity duration (nanoseconds)
    cache_duration: u64,
}

/// Cached correlation matrix with timestamp
#[derive(Debug, Clone)]
struct CachedCorrelation {
    matrix: na::DMatrix<f64>,
    symbols: Vec<Symbol>,
    timestamp: u64,
}

impl CorrelationAnalyzer {
    /// Create a new correlation analyzer
    pub fn new() -> Self {
        Self {
            price_histories: HashMap::new(),
            max_observations: 10000,        // Keep last 10k observations
            min_observations: 30,           // Need at least 30 points for correlation
            correlation_window: 3600_000_000_000, // 1 hour in nanoseconds
            cached_correlation: None,
            cache_duration: 60_000_000_000, // Cache for 60 seconds
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(
        max_observations: usize,
        min_observations: usize,
        correlation_window_seconds: u64,
        cache_duration_seconds: u64,
    ) -> Self {
        Self {
            price_histories: HashMap::new(),
            max_observations,
            min_observations,
            correlation_window: correlation_window_seconds * 1_000_000_000,
            cached_correlation: None,
            cache_duration: cache_duration_seconds * 1_000_000_000,
        }
    }
    
    /// Add a new symbol for correlation tracking
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.price_histories.insert(symbol, VecDeque::new());
        // Invalidate cache when adding new symbols
        self.cached_correlation = None;
    }
    
    /// Add a price observation for a symbol
    pub fn add_price_observation(
        &mut self, 
        symbol: &Symbol, 
        price: u64, 
        timestamp: u64
    ) -> Result<(), CorrelationError> {
        let price_f64 = price as f64;
        
        let history = self.price_histories.get_mut(symbol)
            .ok_or_else(|| CorrelationError::SymbolNotFound(symbol.to_string()))?;
        
        // Calculate log return if we have previous price
        let log_return = if let Some(last_obs) = history.back() {
            if last_obs.price > 0.0 && price_f64 > 0.0 {
                Some((price_f64 / last_obs.price).ln())
            } else {
                None
            }
        } else {
            None
        };
        
        let observation = PriceObservation {
            price: price_f64,
            timestamp,
            log_return,
        };
        
        history.push_back(observation);
        
        // Maintain maximum size
        while history.len() > self.max_observations {
            history.pop_front();
        }
        
        // Remove old observations outside the correlation window
        let cutoff_time = timestamp.saturating_sub(self.correlation_window);
        while let Some(front) = history.front() {
            if front.timestamp < cutoff_time {
                history.pop_front();
            } else {
                break;
            }
        }
        
        // Invalidate cache when new data arrives
        self.cached_correlation = None;
        
        Ok(())
    }
    
    /// Calculate correlation matrix for all symbols
    pub fn calculate_correlation_matrix(&mut self) -> Result<na::DMatrix<f64>, String> {
        let current_time = current_timestamp();
        
        // Check if we can use cached result
        if let Some(ref cached) = self.cached_correlation {
            if current_time.saturating_sub(cached.timestamp) < self.cache_duration {
                return Ok(cached.matrix.clone());
            }
        }
        
        let symbols: Vec<Symbol> = self.price_histories.keys().cloned().collect();
        let n = symbols.len();
        
        if n == 0 {
            return Err("No symbols available for correlation calculation".to_string());
        }
        
        if n == 1 {
            // Single asset - correlation with itself is 1.0
            let mut matrix = na::DMatrix::zeros(1, 1);
            matrix[(0, 0)] = 1.0;
            return Ok(matrix);
        }
        
        // Extract log returns for each symbol
        let mut returns_data: Vec<Vec<f64>> = Vec::new();
        let mut min_length = usize::MAX;
        
        for symbol in &symbols {
            let history = &self.price_histories[symbol];
            
            if history.len() < self.min_observations {
                return Err(format!(
                    "Insufficient data for symbol {}: {} observations, need {}",
                    symbol, history.len(), self.min_observations
                ));
            }
            
            let returns: Vec<f64> = history
                .iter()
                .filter_map(|obs| obs.log_return)
                .collect();
            
            if returns.len() < self.min_observations {
                return Err(format!(
                    "Insufficient return data for symbol {}: {} returns, need {}",
                    symbol, returns.len(), self.min_observations
                ));
            }
            
            min_length = min_length.min(returns.len());
            returns_data.push(returns);
        }
        
        // Align all return series to the same length (use most recent data)
        for returns in &mut returns_data {
            if returns.len() > min_length {
                let start_idx = returns.len() - min_length;
                *returns = returns[start_idx..].to_vec();
            }
        }
        
        // Calculate correlation matrix
        let mut correlation_matrix = na::DMatrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    correlation_matrix[(i, j)] = 1.0;
                } else {
                    let correlation = calculate_pearson_correlation(
                        &returns_data[i], 
                        &returns_data[j]
                    )?;
                    correlation_matrix[(i, j)] = correlation;
                }
            }
        }
        
        // Cache the result
        self.cached_correlation = Some(CachedCorrelation {
            matrix: correlation_matrix.clone(),
            symbols: symbols.clone(),
            timestamp: current_time,
        });
        
        Ok(correlation_matrix)
    }
    
    /// Get correlation between two specific symbols
    pub fn get_pairwise_correlation(
        &mut self, 
        symbol1: &Symbol, 
        symbol2: &Symbol
    ) -> Result<f64, CorrelationError> {
        if symbol1 == symbol2 {
            return Ok(1.0);
        }
        
        let history1 = self.price_histories.get(symbol1)
            .ok_or_else(|| CorrelationError::SymbolNotFound(symbol1.to_string()))?;
        
        let history2 = self.price_histories.get(symbol2)
            .ok_or_else(|| CorrelationError::SymbolNotFound(symbol2.to_string()))?;
        
        if history1.len() < self.min_observations || history2.len() < self.min_observations {
            return Err(CorrelationError::InsufficientData {
                symbol: format!("{} or {}", symbol1, symbol2),
            });
        }
        
        let returns1: Vec<f64> = history1.iter()
            .filter_map(|obs| obs.log_return)
            .collect();
        
        let returns2: Vec<f64> = history2.iter()
            .filter_map(|obs| obs.log_return)
            .collect();
        
        if returns1.len() < self.min_observations || returns2.len() < self.min_observations {
            return Err(CorrelationError::InsufficientData {
                symbol: format!("{} or {} (returns)", symbol1, symbol2),
            });
        }
        
        // Align the series to the same length
        let min_len = returns1.len().min(returns2.len());
        let aligned_returns1 = &returns1[returns1.len() - min_len..];
        let aligned_returns2 = &returns2[returns2.len() - min_len..];
        
        calculate_pearson_correlation(aligned_returns1, aligned_returns2)
            .map_err(|e| CorrelationError::MatrixError(e))
    }
    
    /// Get rolling correlation over a specific window
    pub fn get_rolling_correlation(
        &self,
        symbol1: &Symbol,
        symbol2: &Symbol,
        window_size: usize,
    ) -> Result<Vec<f64>, CorrelationError> {
        let history1 = self.price_histories.get(symbol1)
            .ok_or_else(|| CorrelationError::SymbolNotFound(symbol1.to_string()))?;
        
        let history2 = self.price_histories.get(symbol2)
            .ok_or_else(|| CorrelationError::SymbolNotFound(symbol2.to_string()))?;
        
        let returns1: Vec<f64> = history1.iter()
            .filter_map(|obs| obs.log_return)
            .collect();
        
        let returns2: Vec<f64> = history2.iter()
            .filter_map(|obs| obs.log_return)
            .collect();
        
        let min_len = returns1.len().min(returns2.len());
        
        if min_len < window_size {
            return Err(CorrelationError::InsufficientData {
                symbol: format!("Need {} observations, have {}", window_size, min_len),
            });
        }
        
        let mut rolling_correlations = Vec::new();
        
        for i in window_size..=min_len {
            let window1 = &returns1[i - window_size..i];
            let window2 = &returns2[i - window_size..i];
            
            match calculate_pearson_correlation(window1, window2) {
                Ok(corr) => rolling_correlations.push(corr),
                Err(_) => rolling_correlations.push(0.0), // Default to 0 if calculation fails
            }
        }
        
        Ok(rolling_correlations)
    }
    
    /// Get symbols currently being tracked
    pub fn get_symbols(&self) -> Vec<Symbol> {
        self.price_histories.keys().cloned().collect()
    }
    
    /// Get number of observations for a symbol
    pub fn get_observation_count(&self, symbol: &Symbol) -> Option<usize> {
        self.price_histories.get(symbol).map(|h| h.len())
    }
    
    /// Clear all data
    pub fn clear(&mut self) {
        self.price_histories.clear();
        self.cached_correlation = None;
    }
}

/// Calculate Pearson correlation coefficient between two series
fn calculate_pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, String> {
    if x.len() != y.len() {
        return Err("Series must have the same length".to_string());
    }
    
    if x.is_empty() {
        return Err("Cannot calculate correlation for empty series".to_string());
    }
    
    let n = x.len() as f64;
    
    // Calculate means
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    // Calculate numerator and denominators
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        return Ok(0.0); // No correlation if either series has zero variance
    }
    
    let correlation = numerator / denominator;
    
    // Clamp to [-1, 1] to handle numerical precision issues
    Ok(correlation.max(-1.0).min(1.0))
}

/// Get current timestamp in nanoseconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_correlation_analyzer_creation() {
        let analyzer = CorrelationAnalyzer::new();
        assert_eq!(analyzer.get_symbols().len(), 0);
    }
    
    #[test]
    fn test_add_symbol() {
        let mut analyzer = CorrelationAnalyzer::new();
        let symbol = Symbol::new("BTCUSD").unwrap();
        
        analyzer.add_symbol(symbol.clone());
        assert_eq!(analyzer.get_symbols().len(), 1);
        assert_eq!(analyzer.get_observation_count(&symbol), Some(0));
    }
    
    #[test]
    fn test_price_observations() {
        let mut analyzer = CorrelationAnalyzer::new();
        let symbol = Symbol::new("BTCUSD").unwrap();
        
        analyzer.add_symbol(symbol.clone());
        
        let base_time = current_timestamp();
        let prices = vec![50000, 50100, 49900, 50200, 50050];
        
        for (i, &price) in prices.iter().enumerate() {
            analyzer.add_price_observation(
                &symbol, 
                price * 1_000_000, // Convert to fixed point
                base_time + (i as u64 * 1_000_000_000) // 1 second intervals
            ).unwrap();
        }
        
        assert_eq!(analyzer.get_observation_count(&symbol), Some(5));
    }
    
    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        
        let corr = calculate_pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
        
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // Perfect negative correlation
        let corr_neg = calculate_pearson_correlation(&x, &y_neg).unwrap();
        assert!((corr_neg + 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation_matrix() {
        let mut analyzer = CorrelationAnalyzer::with_params(1000, 5, 3600, 60);
        
        let btc = Symbol::new("BTCUSD").unwrap();
        let eth = Symbol::new("ETHUSD").unwrap();
        
        analyzer.add_symbol(btc.clone());
        analyzer.add_symbol(eth.clone());
        
        let base_time = current_timestamp();
        
        // Add correlated price data
        for i in 0..10 {
            let btc_price = (50000 + i * 100) * 1_000_000;
            let eth_price = (3000 + i * 10) * 1_000_000; // Positively correlated
            
            analyzer.add_price_observation(&btc, btc_price, base_time + i * 1_000_000_000).unwrap();
            analyzer.add_price_observation(&eth, eth_price, base_time + i * 1_000_000_000).unwrap();
        }
        
        let matrix = analyzer.calculate_correlation_matrix().unwrap();
        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        
        // Diagonal should be 1.0
        assert!((matrix[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((matrix[(1, 1)] - 1.0).abs() < 1e-10);
        
        // Off-diagonal should be positive (correlated data)
        assert!(matrix[(0, 1)] > 0.0);
        assert!(matrix[(1, 0)] > 0.0);
    }
}