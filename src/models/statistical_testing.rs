//! Statistical Testing Module for Backtesting Framework
//! 
//! This module provides comprehensive statistical testing capabilities including:
//! - Bootstrap confidence intervals
//! - Hypothesis testing for performance metrics
//! - Multiple testing corrections
//! - Time series specific tests

use crate::models::backtesting_framework::{SimulationResults, StatisticalTestResults, StatisticalTestConfig, MultipleTestingCorrection};
use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal, StudentsT};
use statrs::distribution::{ContinuousCDF, StudentsT as StatrsStudentsT};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StatisticalTestError {
    #[error("Statistical test error: {0}")]
    TestError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
}

/// Statistical tester for backtesting results
pub struct StatisticalTester {
    config: StatisticalTestConfig,
}

impl StatisticalTester {
    pub fn new(config: &StatisticalTestConfig) -> Result<Self, StatisticalTestError> {
        if config.significance_level <= 0.0 || config.significance_level >= 1.0 {
            return Err(StatisticalTestError::InvalidParameters(
                "Significance level must be between 0 and 1".to_string()
            ));
        }

        Ok(Self {
            config: config.clone(),
        })
    }

    /// Test statistical significance of backtesting results
    pub async fn test_significance(
        &self,
        results: &SimulationResults,
    ) -> Result<StatisticalTestResults, StatisticalTestError> {
        if results.pnl_series.len() < 30 {
            return Err(StatisticalTestError::InsufficientData(
                "Need at least 30 observations for statistical testing".to_string()
            ));
        }

        // Calculate returns from P&L series
        let returns = self.calculate_returns(&results.pnl_series)?;

        // Perform various statistical tests
        let mut test_results = HashMap::new();

        // 1. Sharpe ratio significance test
        let sharpe_test = self.test_sharpe_ratio_significance(&returns)?;
        test_results.insert("sharpe_ratio".to_string(), sharpe_test);

        // 2. Return significance test (t-test)
        let return_test = self.test_return_significance(&returns)?;
        test_results.insert("returns".to_string(), return_test);

        // 3. Normality tests
        let normality_test = self.test_normality(&returns)?;
        test_results.insert("normality".to_string(), normality_test);

        // 4. Autocorrelation tests
        let autocorr_test = self.test_autocorrelation(&returns)?;
        test_results.insert("autocorrelation".to_string(), autocorr_test);

        // 5. Heteroscedasticity tests
        let hetero_test = self.test_heteroscedasticity(&returns)?;
        test_results.insert("heteroscedasticity".to_string(), hetero_test);

        // 6. Time series specific tests if enabled
        if self.config.time_series_tests {
            let ts_tests = self.time_series_tests(&returns)?;
            test_results.extend(ts_tests);
        }

        // 7. Bootstrap confidence intervals
        let bootstrap_results = self.bootstrap_confidence_intervals(&returns)?;

        // Apply multiple testing correction
        let corrected_pvalues = self.apply_multiple_testing_correction(&test_results)?;

        // Determine overall significance
        let sharpe_ratio_pvalue = corrected_pvalues.get("sharpe_ratio")
            .and_then(|test| test.get("p_value"))
            .copied()
            .unwrap_or(1.0);

        let return_significance = corrected_pvalues.get("returns")
            .and_then(|test| test.get("p_value"))
            .map(|&p| p < self.config.significance_level)
            .unwrap_or(false);

        Ok(StatisticalTestResults {
            sharpe_ratio_pvalue,
            return_significance,
            detailed_results: DetailedStatisticalResults {
                test_results: corrected_pvalues,
                bootstrap_results,
                summary_statistics: self.calculate_summary_statistics(&returns)?,
            },
        })
    }

    /// Calculate returns from P&L series
    fn calculate_returns(&self, pnl_series: &[f64]) -> Result<Vec<f64>, StatisticalTestError> {
        if pnl_series.len() < 2 {
            return Err(StatisticalTestError::InsufficientData(
                "Need at least 2 P&L observations".to_string()
            ));
        }

        let returns: Vec<f64> = pnl_series.windows(2)
            .map(|w| {
                if w[0].abs() > 1e-8 {
                    (w[1] - w[0]) / w[0].abs()
                } else {
                    0.0
                }
            })
            .collect();

        Ok(returns)
    }

    /// Test Sharpe ratio significance using bootstrap
    fn test_sharpe_ratio_significance(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let sharpe_ratio = self.calculate_sharpe_ratio(returns)?;
        
        // Bootstrap test for Sharpe ratio
        let mut bootstrap_sharpes = Vec::new();
        let mut rng = thread_rng();

        for _ in 0..self.config.bootstrap_iterations {
            let bootstrap_sample = self.bootstrap_sample(returns, &mut rng);
            let bootstrap_sharpe = self.calculate_sharpe_ratio(&bootstrap_sample)?;
            bootstrap_sharpes.push(bootstrap_sharpe);
        }

        // Calculate p-value (two-tailed test)
        let null_sharpe = 0.0;
        let extreme_count = bootstrap_sharpes.iter()
            .filter(|&&s| (s - null_sharpe).abs() >= (sharpe_ratio - null_sharpe).abs())
            .count();
        
        let p_value = extreme_count as f64 / self.config.bootstrap_iterations as f64;

        Ok(TestResult {
            statistic: sharpe_ratio,
            p_value,
            critical_value: self.calculate_critical_value(&bootstrap_sharpes)?,
            is_significant: p_value < self.config.significance_level,
            test_name: "Sharpe Ratio Bootstrap Test".to_string(),
        })
    }

    /// Test return significance using t-test
    fn test_return_significance(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let n = returns.len() as f64;
        let mean_return = returns.iter().sum::<f64>() / n;
        let std_return = self.calculate_std(returns, mean_return)?;
        
        // t-statistic for testing if mean return is significantly different from 0
        let t_statistic = mean_return / (std_return / n.sqrt());
        
        // Calculate p-value using t-distribution
        let t_dist = StatrsStudentsT::new(0.0, 1.0, n - 1.0)
            .map_err(|e| StatisticalTestError::TestError(e.to_string()))?;
        
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));
        
        // Critical value for two-tailed test
        let critical_value = t_dist.inverse_cdf(1.0 - self.config.significance_level / 2.0);

        Ok(TestResult {
            statistic: t_statistic,
            p_value,
            critical_value,
            is_significant: p_value < self.config.significance_level,
            test_name: "Return Significance t-test".to_string(),
        })
    }

    /// Test normality using Jarque-Bera test
    fn test_normality(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        
        // Calculate skewness and kurtosis
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        
        for &ret in returns {
            let diff = ret - mean;
            let diff2 = diff * diff;
            let diff3 = diff2 * diff;
            let diff4 = diff3 * diff;
            
            m2 += diff2;
            m3 += diff3;
            m4 += diff4;
        }
        
        m2 /= n;
        m3 /= n;
        m4 /= n;
        
        let skewness = m3 / m2.powf(1.5);
        let kurtosis = m4 / (m2 * m2) - 3.0;
        
        // Jarque-Bera statistic
        let jb_statistic = (n / 6.0) * (skewness * skewness + kurtosis * kurtosis / 4.0);
        
        // Chi-square distribution with 2 degrees of freedom
        // Approximate p-value calculation
        let p_value = (-jb_statistic / 2.0).exp();

        Ok(TestResult {
            statistic: jb_statistic,
            p_value,
            critical_value: 5.991, // Chi-square critical value at 5% significance
            is_significant: p_value < self.config.significance_level,
            test_name: "Jarque-Bera Normality Test".to_string(),
        })
    }

    /// Test for autocorrelation using Ljung-Box test
    fn test_autocorrelation(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let n = returns.len();
        let max_lag = (n / 4).min(20); // Use up to 20 lags or n/4, whichever is smaller
        
        let mean = returns.iter().sum::<f64>() / n as f64;
        
        // Calculate autocorrelations
        let mut autocorrs = Vec::new();
        for lag in 1..=max_lag {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            
            for i in 0..n {
                let x_i = returns[i] - mean;
                denominator += x_i * x_i;
                
                if i >= lag {
                    let x_lag = returns[i - lag] - mean;
                    numerator += x_i * x_lag;
                }
            }
            
            let autocorr = numerator / denominator;
            autocorrs.push(autocorr);
        }
        
        // Ljung-Box statistic
        let mut lb_statistic = 0.0;
        for (k, &rho_k) in autocorrs.iter().enumerate() {
            let lag = k + 1;
            lb_statistic += rho_k * rho_k / (n - lag) as f64;
        }
        lb_statistic *= n as f64 * (n as f64 + 2.0);
        
        // Approximate p-value using chi-square distribution
        let p_value = (-lb_statistic / 2.0).exp();

        Ok(TestResult {
            statistic: lb_statistic,
            p_value,
            critical_value: 31.41, // Chi-square critical value for 20 degrees of freedom at 5%
            is_significant: p_value < self.config.significance_level,
            test_name: "Ljung-Box Autocorrelation Test".to_string(),
        })
    }

    /// Test for heteroscedasticity using ARCH test
    fn test_heteroscedasticity(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let n = returns.len();
        let mean = returns.iter().sum::<f64>() / n as f64;
        
        // Calculate squared residuals
        let squared_residuals: Vec<f64> = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .collect();
        
        // Simple ARCH(1) test - regress squared residuals on lagged squared residuals
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let valid_n = n - 1;
        
        for i in 1..n {
            let x = squared_residuals[i - 1]; // lagged squared residual
            let y = squared_residuals[i];     // current squared residual
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        // Calculate regression coefficient
        let mean_x = sum_x / valid_n as f64;
        let mean_y = sum_y / valid_n as f64;
        
        let beta = (sum_xy - valid_n as f64 * mean_x * mean_y) / 
                   (sum_x2 - valid_n as f64 * mean_x * mean_x);
        
        // Calculate R-squared
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        
        for i in 1..n {
            let x = squared_residuals[i - 1];
            let y = squared_residuals[i];
            let y_pred = mean_y + beta * (x - mean_x);
            
            ss_tot += (y - mean_y).powi(2);
            ss_res += (y - y_pred).powi(2);
        }
        
        let r_squared = 1.0 - ss_res / ss_tot;
        
        // ARCH test statistic: n * R^2
        let arch_statistic = valid_n as f64 * r_squared;
        
        // Approximate p-value
        let p_value = (-arch_statistic / 2.0).exp();

        Ok(TestResult {
            statistic: arch_statistic,
            p_value,
            critical_value: 3.841, // Chi-square critical value with 1 df at 5%
            is_significant: p_value < self.config.significance_level,
            test_name: "ARCH Heteroscedasticity Test".to_string(),
        })
    }

    /// Additional time series tests
    fn time_series_tests(&self, returns: &[f64]) -> Result<HashMap<String, TestResult>, StatisticalTestError> {
        let mut tests = HashMap::new();

        // Unit root test (simplified ADF test)
        let unit_root_test = self.augmented_dickey_fuller_test(returns)?;
        tests.insert("unit_root".to_string(), unit_root_test);

        // Runs test for randomness
        let runs_test = self.runs_test(returns)?;
        tests.insert("runs".to_string(), runs_test);

        Ok(tests)
    }

    /// Simplified Augmented Dickey-Fuller test
    fn augmented_dickey_fuller_test(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let n = returns.len();
        if n < 10 {
            return Err(StatisticalTestError::InsufficientData(
                "Need at least 10 observations for ADF test".to_string()
            ));
        }

        // Calculate first differences
        let diff_returns: Vec<f64> = returns.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // Simple regression: Δy_t = α + βy_{t-1} + ε_t
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let valid_n = diff_returns.len();

        for i in 0..valid_n {
            let x = returns[i];           // y_{t-1}
            let y = diff_returns[i];      // Δy_t
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let mean_x = sum_x / valid_n as f64;
        let mean_y = sum_y / valid_n as f64;
        
        let beta = (sum_xy - valid_n as f64 * mean_x * mean_y) / 
                   (sum_x2 - valid_n as f64 * mean_x * mean_x);

        // Calculate standard error of beta
        let mut ss_res = 0.0;
        for i in 0..valid_n {
            let x = returns[i];
            let y = diff_returns[i];
            let y_pred = mean_y + beta * (x - mean_x);
            ss_res += (y - y_pred).powi(2);
        }

        let mse = ss_res / (valid_n - 2) as f64;
        let se_beta = (mse / (sum_x2 - valid_n as f64 * mean_x * mean_x)).sqrt();
        
        // ADF test statistic
        let adf_statistic = beta / se_beta;
        
        // Critical values (approximate)
        let critical_value = -2.86; // 5% critical value for ADF test
        let p_value = if adf_statistic < critical_value { 0.01 } else { 0.10 };

        Ok(TestResult {
            statistic: adf_statistic,
            p_value,
            critical_value,
            is_significant: adf_statistic < critical_value,
            test_name: "Augmented Dickey-Fuller Test".to_string(),
        })
    }

    /// Runs test for randomness
    fn runs_test(&self, returns: &[f64]) -> Result<TestResult, StatisticalTestError> {
        let median = {
            let mut sorted_returns = returns.to_vec();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted_returns.len() / 2;
            if sorted_returns.len() % 2 == 0 {
                (sorted_returns[mid - 1] + sorted_returns[mid]) / 2.0
            } else {
                sorted_returns[mid]
            }
        };

        // Convert to binary sequence (above/below median)
        let binary_seq: Vec<bool> = returns.iter()
            .map(|&r| r > median)
            .collect();

        // Count runs
        let mut runs = 1;
        for i in 1..binary_seq.len() {
            if binary_seq[i] != binary_seq[i - 1] {
                runs += 1;
            }
        }

        // Count positive and negative observations
        let n1 = binary_seq.iter().filter(|&&b| b).count() as f64;
        let n2 = binary_seq.iter().filter(|&&b| !b).count() as f64;
        let n = n1 + n2;

        // Expected runs and variance under null hypothesis
        let expected_runs = (2.0 * n1 * n2) / n + 1.0;
        let variance_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1.0));

        // Test statistic (standardized)
        let z_statistic = (runs as f64 - expected_runs) / variance_runs.sqrt();
        
        // Two-tailed p-value (approximate using normal distribution)
        let p_value = 2.0 * (1.0 - (z_statistic.abs() / 2.0_f64.sqrt()).exp());

        Ok(TestResult {
            statistic: z_statistic,
            p_value,
            critical_value: 1.96, // 5% critical value for two-tailed normal test
            is_significant: z_statistic.abs() > 1.96,
            test_name: "Runs Test for Randomness".to_string(),
        })
    }

    /// Bootstrap confidence intervals
    fn bootstrap_confidence_intervals(&self, returns: &[f64]) -> Result<BootstrapResults, StatisticalTestError> {
        let mut rng = thread_rng();
        let mut bootstrap_means = Vec::new();
        let mut bootstrap_stds = Vec::new();
        let mut bootstrap_sharpes = Vec::new();

        for _ in 0..self.config.bootstrap_iterations {
            let bootstrap_sample = self.bootstrap_sample(returns, &mut rng);
            
            let mean = bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64;
            let std = self.calculate_std(&bootstrap_sample, mean)?;
            let sharpe = if std > 1e-8 { mean / std } else { 0.0 };
            
            bootstrap_means.push(mean);
            bootstrap_stds.push(std);
            bootstrap_sharpes.push(sharpe);
        }

        // Sort for percentile calculation
        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_stds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        bootstrap_sharpes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = self.config.significance_level;
        let lower_idx = ((alpha / 2.0) * self.config.bootstrap_iterations as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.config.bootstrap_iterations as f64) as usize;

        Ok(BootstrapResults {
            mean_ci: (bootstrap_means[lower_idx], bootstrap_means[upper_idx]),
            std_ci: (bootstrap_stds[lower_idx], bootstrap_stds[upper_idx]),
            sharpe_ci: (bootstrap_sharpes[lower_idx], bootstrap_sharpes[upper_idx]),
            confidence_level: 1.0 - alpha,
        })
    }

    /// Apply multiple testing correction
    fn apply_multiple_testing_correction(
        &self,
        test_results: &HashMap<String, TestResult>,
    ) -> Result<HashMap<String, HashMap<String, f64>>, StatisticalTestError> {
        let mut corrected_results = HashMap::new();
        let p_values: Vec<f64> = test_results.values().map(|t| t.p_value).collect();

        let corrected_p_values = match self.config.multiple_testing_correction {
            MultipleTestingCorrection::None => p_values,
            MultipleTestingCorrection::Bonferroni => {
                p_values.iter().map(|&p| (p * p_values.len() as f64).min(1.0)).collect()
            },
            MultipleTestingCorrection::BenjaminiHochberg => {
                self.benjamini_hochberg_correction(&p_values)?
            },
        };

        for ((test_name, test_result), &corrected_p) in test_results.iter().zip(corrected_p_values.iter()) {
            let mut result_map = HashMap::new();
            result_map.insert("statistic".to_string(), test_result.statistic);
            result_map.insert("p_value".to_string(), corrected_p);
            result_map.insert("critical_value".to_string(), test_result.critical_value);
            result_map.insert("is_significant".to_string(), if corrected_p < self.config.significance_level { 1.0 } else { 0.0 });
            
            corrected_results.insert(test_name.clone(), result_map);
        }

        Ok(corrected_results)
    }

    /// Benjamini-Hochberg correction
    fn benjamini_hochberg_correction(&self, p_values: &[f64]) -> Result<Vec<f64>, StatisticalTestError> {
        let m = p_values.len();
        let mut indexed_p_values: Vec<(usize, f64)> = p_values.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by p-value
        indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut corrected = vec![0.0; m];
        let mut min_corrected = 1.0;

        // Apply correction in reverse order
        for (rank, &(original_idx, p_value)) in indexed_p_values.iter().enumerate().rev() {
            let corrected_p = (p_value * m as f64) / (rank + 1) as f64;
            min_corrected = min_corrected.min(corrected_p);
            corrected[original_idx] = min_corrected.min(1.0);
        }

        Ok(corrected)
    }

    // Helper methods

    fn bootstrap_sample(&self, data: &[f64], rng: &mut impl Rng) -> Vec<f64> {
        (0..data.len())
            .map(|_| data[rng.gen_range(0..data.len())])
            .collect()
    }

    fn calculate_sharpe_ratio(&self, returns: &[f64]) -> Result<f64, StatisticalTestError> {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std = self.calculate_std(returns, mean)?;
        
        if std > 1e-8 {
            Ok(mean / std)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_std(&self, data: &[f64], mean: f64) -> Result<f64, StatisticalTestError> {
        if data.len() < 2 {
            return Err(StatisticalTestError::InsufficientData(
                "Need at least 2 observations for standard deviation".to_string()
            ));
        }

        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;

        Ok(variance.sqrt())
    }

    fn calculate_critical_value(&self, bootstrap_values: &[f64]) -> Result<f64, StatisticalTestError> {
        let mut sorted_values = bootstrap_values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let critical_idx = ((1.0 - self.config.significance_level / 2.0) * sorted_values.len() as f64) as usize;
        Ok(sorted_values[critical_idx.min(sorted_values.len() - 1)])
    }

    fn calculate_summary_statistics(&self, returns: &[f64]) -> Result<SummaryStatistics, StatisticalTestError> {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let std = self.calculate_std(returns, mean)?;
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median = if sorted_returns.len() % 2 == 0 {
            let mid = sorted_returns.len() / 2;
            (sorted_returns[mid - 1] + sorted_returns[mid]) / 2.0
        } else {
            sorted_returns[sorted_returns.len() / 2]
        };

        let min = sorted_returns[0];
        let max = sorted_returns[sorted_returns.len() - 1];

        // Calculate skewness and kurtosis
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &ret in returns {
            let diff = ret - mean;
            m3 += diff.powi(3);
            m4 += diff.powi(4);
        }
        m3 /= n;
        m4 /= n;

        let skewness = m3 / std.powi(3);
        let kurtosis = m4 / std.powi(4) - 3.0;

        Ok(SummaryStatistics {
            count: returns.len(),
            mean,
            std,
            median,
            min,
            max,
            skewness,
            kurtosis,
        })
    }
}

// Data structures for statistical testing

#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub critical_value: f64,
    pub is_significant: bool,
    pub test_name: String,
}

#[derive(Debug)]
pub struct DetailedStatisticalResults {
    pub test_results: HashMap<String, HashMap<String, f64>>,
    pub bootstrap_results: BootstrapResults,
    pub summary_statistics: SummaryStatistics,
}

#[derive(Debug)]
pub struct BootstrapResults {
    pub mean_ci: (f64, f64),
    pub std_ci: (f64, f64),
    pub sharpe_ci: (f64, f64),
    pub confidence_level: f64,
}

#[derive(Debug)]
pub struct SummaryStatistics {
    pub count: usize,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

// Update the StatisticalTestResults to include detailed results
impl crate::models::backtesting_framework::StatisticalTestResults {
    pub fn new(sharpe_ratio_pvalue: f64, return_significance: bool) -> Self {
        Self {
            sharpe_ratio_pvalue,
            return_significance,
            detailed_results: DetailedStatisticalResults {
                test_results: HashMap::new(),
                bootstrap_results: BootstrapResults {
                    mean_ci: (0.0, 0.0),
                    std_ci: (0.0, 0.0),
                    sharpe_ci: (0.0, 0.0),
                    confidence_level: 0.95,
                },
                summary_statistics: SummaryStatistics {
                    count: 0,
                    mean: 0.0,
                    std: 0.0,
                    median: 0.0,
                    min: 0.0,
                    max: 0.0,
                    skewness: 0.0,
                    kurtosis: 0.0,
                },
            },
        }
    }
}