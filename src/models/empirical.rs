use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order, TradeExecution},
};

#[derive(Debug, Error)]
pub enum EmpiricalError {
    #[error("Empirical analysis error: {0}")]
    EmpiricalError(String),
    #[error("Data error: {0}")]
    DataError(String),
}

/// Parameters for empirical analysis
pub struct EmpiricalParams {
    pub estimation_window: usize,
    pub significance_level: f64,
    pub min_observations: usize,
    pub max_lag: usize,
}

/// Results of empirical analysis
pub struct EmpiricalResults {
    pub variance_ratio: VarRatioResults,
    pub serial_correlation: SerialCorrelationResults,
    pub market_efficiency: MarketEfficiencyResults,
    pub microstructure: MicrostructureResults,
}

#[derive(Debug)]
pub struct VarRatioResults {
    pub ratios: Vec<f64>,
    pub p_values: Vec<f64>,
    pub is_significant: Vec<bool>,
    pub lags: Vec<usize>,
}

#[derive(Debug)]
pub struct SerialCorrelationResults {
    pub autocorrelations: Vec<f64>,
    pub partial_autocorr: Vec<f64>,
    pub ljung_box: f64,
    pub p_value: f64,
}

#[derive(Debug)]
pub struct MarketEfficiencyResults {
    pub hurst_exponent: f64,
    pub variance_ratio: f64,
    pub runs_test_stat: f64,
    pub runs_test_pval: f64,
}

#[derive(Debug)]
pub struct MicrostructureResults {
    pub bid_ask_bounce: f64,
    pub effective_spread: f64,
    pub price_impact: f64,
    pub order_flow_correlation: f64,
}

pub struct EmpiricalAnalyzer {
    params: EmpiricalParams,
}

impl EmpiricalAnalyzer {
    pub fn new(params: EmpiricalParams) -> Self {
        Self { params }
    }

    pub fn analyze_market_efficiency(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<EmpiricalResults, EmpiricalError> {
        // Compute returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
            
        // Variance ratio test
        let var_ratio = self.compute_variance_ratio(&returns)?;
        
        // Serial correlation analysis
        let serial_corr = self.analyze_serial_correlation(&returns)?;
        
        // Market efficiency tests
        let efficiency = self.test_market_efficiency(&returns, &volumes)?;
        
        // Microstructure analysis
        let microstructure = self.analyze_microstructure(prices, volumes)?;
        
        Ok(EmpiricalResults {
            variance_ratio: var_ratio,
            serial_correlation: serial_corr,
            market_efficiency: efficiency,
            microstructure: microstructure,
        })
    }

    fn compute_variance_ratio(
        &self,
        returns: &[f64],
    ) -> Result<VarRatioResults, EmpiricalError> {
        let n = returns.len();
        if n < self.params.min_observations {
            return Err(EmpiricalError::DataError(
                "Insufficient observations for variance ratio test".to_string(),
            ));
        }
        
        let mut ratios = Vec::new();
        let mut p_values = Vec::new();
        let mut is_significant = Vec::new();
        let mut lags = Vec::new();
        
        // Compute variance ratios for different lags
        for k in 1..=self.params.max_lag {
            // Compute k-period returns
            let k_returns: Vec<f64> = returns.windows(k)
                .map(|w| w.iter().sum())
                .collect();
                
            // Compute variances
            let var_1 = self.compute_variance(returns);
            let var_k = self.compute_variance(&k_returns) / k as f64;
            
            // Compute variance ratio
            let vr = var_k / var_1;
            
            // Compute standard error
            let phi = self.compute_variance_ratio_se(k, n, returns);
            
            // Compute test statistic and p-value
            let z_stat = (vr - 1.0) / phi;
            let p_value = 2.0 * (1.0 - standard_normal_cdf(z_stat.abs()));
            
            ratios.push(vr);
            p_values.push(p_value);
            is_significant.push(p_value < self.params.significance_level);
            lags.push(k);
        }
        
        Ok(VarRatioResults {
            ratios,
            p_values,
            is_significant,
            lags,
        })
    }

    fn analyze_serial_correlation(
        &self,
        returns: &[f64],
    ) -> Result<SerialCorrelationResults, EmpiricalError> {
        let n = returns.len();
        if n < self.params.min_observations {
            return Err(EmpiricalError::DataError(
                "Insufficient observations for serial correlation analysis".to_string(),
            ));
        }
        
        // Compute autocorrelations
        let mut autocorr = Vec::new();
        let mut partial_autocorr = Vec::new();
        
        for lag in 1..=self.params.max_lag {
            let ac = self.compute_autocorrelation(returns, lag);
            let pac = self.compute_partial_autocorr(returns, lag);
            
            autocorr.push(ac);
            partial_autocorr.push(pac);
        }
        
        // Compute Ljung-Box test
        let (lb_stat, lb_pval) = self.compute_ljung_box(returns, &autocorr);
        
        Ok(SerialCorrelationResults {
            autocorrelations: autocorr,
            partial_autocorr,
            ljung_box: lb_stat,
            p_value: lb_pval,
        })
    }

    fn test_market_efficiency(
        &self,
        returns: &[f64],
        volumes: &[f64],
    ) -> Result<MarketEfficiencyResults, EmpiricalError> {
        // Compute Hurst exponent
        let hurst = self.compute_hurst_exponent(returns)?;
        
        // Compute variance ratio
        let var_ratio = self.compute_variance_ratio(returns)?
            .ratios
            .iter()
            .sum::<f64>() / self.params.max_lag as f64;
            
        // Perform runs test
        let (runs_stat, runs_pval) = self.compute_runs_test(returns);
        
        Ok(MarketEfficiencyResults {
            hurst_exponent: hurst,
            variance_ratio: var_ratio,
            runs_test_stat: runs_stat,
            runs_test_pval: runs_pval,
        })
    }

    fn analyze_microstructure(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<MicrostructureResults, EmpiricalError> {
        // Compute bid-ask bounce
        let bid_ask_bounce = self.estimate_bid_ask_bounce(prices)?;
        
        // Compute effective spread
        let effective_spread = self.estimate_effective_spread(prices, volumes)?;
        
        // Compute price impact
        let price_impact = self.estimate_price_impact(prices, volumes)?;
        
        // Compute order flow correlation
        let order_flow_corr = self.compute_order_flow_correlation(volumes)?;
        
        Ok(MicrostructureResults {
            bid_ask_bounce,
            effective_spread,
            price_impact,
            order_flow_correlation: order_flow_corr,
        })
    }

    fn compute_hurst_exponent(
        &self,
        returns: &[f64],
    ) -> Result<f64, EmpiricalError> {
        let n = returns.len();
        if n < self.params.min_observations {
            return Err(EmpiricalError::DataError(
                "Insufficient observations for Hurst exponent estimation".to_string(),
            ));
        }
        
        let mut rs_values = Vec::new();
        let mut ns = Vec::new();
        
        // Compute R/S statistic for different time scales
        for k in 10..=n/4 {
            let chunks: Vec<_> = returns.chunks(k).collect();
            let mut rs_k = Vec::new();
            
            for chunk in chunks {
                if chunk.len() == k {
                    let mean = chunk.iter().sum::<f64>() / k as f64;
                    let std = (chunk.iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>() / k as f64)
                        .sqrt();
                        
                    let mut cum_dev = Vec::with_capacity(k);
                    let mut sum = 0.0;
                    for &x in chunk {
                        sum += x - mean;
                        cum_dev.push(sum);
                    }
                    
                    let range = cum_dev.iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, &x| {
                            (acc.0.min(x), acc.1.max(x))
                        });
                        
                    let rs = (range.1 - range.0) / std;
                    rs_k.push(rs);
                }
            }
            
            if !rs_k.is_empty() {
                let mean_rs = rs_k.iter().sum::<f64>() / rs_k.len() as f64;
                rs_values.push(mean_rs.ln());
                ns.push((k as f64).ln());
            }
        }
        
        // Estimate Hurst exponent using linear regression
        let (slope, _) = self.linear_regression(&ns, &rs_values);
        
        Ok(slope)
    }

    fn compute_variance(&self, data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64
    }

    fn compute_variance_ratio_se(
        &self,
        k: usize,
        n: usize,
        returns: &[f64],
    ) -> f64 {
        let mut sum = 0.0;
        for j in 1..k {
            let delta = (2.0 * (k - j) as f64 / k as f64).powi(2);
            let ac = self.compute_autocorrelation(returns, j);
            sum += delta * ac;
        }
        
        (2.0 * sum / n as f64).sqrt()
    }

    fn compute_autocorrelation(
        &self,
        data: &[f64],
        lag: usize,
    ) -> f64 {
        let n = data.len();
        if lag >= n {
            return 0.0;
        }
        
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = self.compute_variance(data);
        
        let mut sum = 0.0;
        for i in 0..n-lag {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        
        sum / ((n - lag) as f64 * var)
    }

    fn compute_partial_autocorr(
        &self,
        data: &[f64],
        lag: usize,
    ) -> f64 {
        if lag == 0 {
            return 1.0;
        }
        
        let n = data.len();
        let mut r = vec![0.0; lag + 1];
        for i in 0..=lag {
            r[i] = self.compute_autocorrelation(data, i);
        }
        
        let mut phi = vec![vec![0.0; lag + 1]; lag + 1];
        phi[1][1] = r[1];
        
        for k in 2..=lag {
            let mut sum = 0.0;
            for j in 1..k {
                sum += phi[k-1][j] * r[k-j];
            }
            phi[k][k] = (r[k] - sum) / (1.0 - r[k-1]);
            
            for j in 1..k {
                phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j];
            }
        }
        
        phi[lag][lag]
    }

    fn compute_ljung_box(
        &self,
        data: &[f64],
        autocorr: &[f64],
    ) -> (f64, f64) {
        let n = data.len();
        let mut q = 0.0;
        
        for (k, &ac) in autocorr.iter().enumerate() {
            q += ac.powi(2) / (n - k - 1) as f64;
        }
        
        q *= n as f64 * (n + 2) as f64;
        
        // Compute p-value using chi-square distribution
        let p_value = 1.0 - chi_square_cdf(q, autocorr.len() as f64);
        
        (q, p_value)
    }

    fn compute_runs_test(
        &self,
        returns: &[f64],
    ) -> (f64, f64) {
        let n = returns.len();
        let mut runs = 1;
        let mut pos = 0;
        let mut neg = 0;
        
        // Count positive and negative returns
        for &ret in returns {
            if ret > 0.0 {
                pos += 1;
            } else if ret < 0.0 {
                neg += 1;
            }
        }
        
        // Count runs
        for i in 1..n {
            if (returns[i] > 0.0 && returns[i-1] < 0.0) ||
               (returns[i] < 0.0 && returns[i-1] > 0.0) {
                runs += 1;
            }
        }
        
        // Compute test statistic
        let expected_runs = 2.0 * pos as f64 * neg as f64 / n as f64 + 1.0;
        let var_runs = (expected_runs - 1.0) * (expected_runs - 2.0) / (n - 1) as f64;
        let z = (runs as f64 - expected_runs) / var_runs.sqrt();
        
        // Compute p-value
        let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
        
        (z, p_value)
    }

    fn estimate_bid_ask_bounce(
        &self,
        prices: &[f64],
    ) -> Result<f64, EmpiricalError> {
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
            
        let first_order_acf = self.compute_autocorrelation(&returns, 1);
        
        // Roll's model implies first-order autocorrelation = -s²/4
        Ok(2.0 * (-first_order_acf).sqrt())
    }

    fn estimate_effective_spread(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<f64, EmpiricalError> {
        let n = prices.len();
        if n != volumes.len() {
            return Err(EmpiricalError::DataError(
                "Price and volume series must have same length".to_string(),
            ));
        }
        
        let mut spread_estimates = Vec::new();
        
        for window in prices.windows(3).zip(volumes.windows(3)) {
            let ((p0, p1, p2), (v0, v1, v2)) = window;
            
            // Compute signed volume
            let sign = if v1 > &0.0 { 1.0 } else { -1.0 };
            
            // Compute realized spread
            let half_spread = sign * (p1 - p0) / p0;
            
            // Adjust for price drift
            let drift = (p2 - p0) / (2.0 * p0);
            
            spread_estimates.push(2.0 * (half_spread - drift).abs());
        }
        
        Ok(spread_estimates.iter().sum::<f64>() / spread_estimates.len() as f64)
    }

    fn estimate_price_impact(
        &self,
        prices: &[f64],
        volumes: &[f64],
    ) -> Result<f64, EmpiricalError> {
        let n = prices.len();
        if n != volumes.len() {
            return Err(EmpiricalError::DataError(
                "Price and volume series must have same length".to_string(),
            ));
        }
        
        let mut impact_estimates = Vec::new();
        
        for window in prices.windows(2).zip(volumes.windows(2)) {
            let ((p0, p1), (v0, v1)) = window;
            
            // Compute volume-weighted price change
            let price_change = (p1 - p0) / p0;
            let vol_ratio = v1 / v0;
            
            impact_estimates.push(price_change / vol_ratio.abs().sqrt());
        }
        
        Ok(impact_estimates.iter().sum::<f64>() / impact_estimates.len() as f64)
    }

    fn compute_order_flow_correlation(
        &self,
        volumes: &[f64],
    ) -> Result<f64, EmpiricalError> {
        let n = volumes.len();
        if n < self.params.min_observations {
            return Err(EmpiricalError::DataError(
                "Insufficient observations for order flow correlation".to_string(),
            ));
        }
        
        // Compute sign of volume changes
        let signs: Vec<f64> = volumes.windows(2)
            .map(|w| if w[1] > w[0] { 1.0 } else { -1.0 })
            .collect();
            
        self.compute_autocorrelation(&signs, 1)
    }

    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
        let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;
        
        (slope, intercept)
    }
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / 2.0_f64.sqrt()))
}

fn chi_square_cdf(x: f64, df: f64) -> f64 {
    let gamma_df = gamma(df / 2.0);
    let incomplete_gamma = gamma_inc(df / 2.0, x / 2.0);
    incomplete_gamma / gamma_df
}

fn gamma_inc(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut term = 1.0 / a;
    let mut n = 1;
    
    while term.abs() > 1e-10 && n < 1000 {
        term *= x / (a + n as f64);
        sum += term;
        n += 1;
    }
    
    x.powf(a) * (-x).exp() * sum
}
