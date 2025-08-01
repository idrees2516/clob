use super::ValidationError;
use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use statrs::distribution::{StudentsT, FisherSnedecor, ChiSquare, ContinuousCDF};
use std::collections::HashMap;

pub struct DiagnosticTests {
    significance_level: f64,
}

impl DiagnosticTests {
    pub fn new(significance_level: f64) -> Result<Self, ValidationError> {
        if significance_level <= 0.0 || significance_level >= 1.0 {
            return Err(ValidationError::ValidationError(
                "Significance level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            significance_level,
        })
    }

    pub fn run_diagnostics(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
    ) -> Result<DiagnosticResults, ValidationError> {
        let normality_test = self.test_normality(residuals)?;
        let heteroskedasticity_test = self.test_heteroskedasticity(residuals, fitted_values)?;
        let autocorrelation_test = self.test_autocorrelation(residuals)?;
        let stationarity_test = self.test_stationarity(residuals)?;
        let structural_break_test = self.test_structural_breaks(residuals)?;

        Ok(DiagnosticResults {
            normality_test,
            heteroskedasticity_test,
            autocorrelation_test,
            stationarity_test,
            structural_break_test,
        })
    }

    fn test_normality(&self, residuals: &[f64]) -> Result<NormalityTest, ValidationError> {
        let n = residuals.len();
        
        // Compute sample moments
        let mean = residuals.iter().sum::<f64>() / n as f64;
        
        let centered: Vec<f64> = residuals.iter()
            .map(|&x| x - mean)
            .collect();
        
        let variance = centered.iter()
            .map(|&x| x.powi(2))
            .sum::<f64>() / (n - 1) as f64;
        
        let std_dev = variance.sqrt();
        
        let standardized: Vec<f64> = centered.iter()
            .map(|&x| x / std_dev)
            .collect();

        // Compute skewness and kurtosis
        let skewness = standardized.iter()
            .map(|&x| x.powi(3))
            .sum::<f64>() / n as f64;
        
        let kurtosis = standardized.iter()
            .map(|&x| x.powi(4))
            .sum::<f64>() / n as f64 - 3.0;

        // Compute Jarque-Bera statistic
        let jb_stat = n as f64 * (skewness.powi(2) / 6.0 + kurtosis.powi(2) / 24.0);
        
        // Compute p-value using chi-square distribution with 2 df
        let chi_square = ChiSquare::new(2.0)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let p_value = 1.0 - chi_square.cdf(jb_stat);

        Ok(NormalityTest {
            statistic: jb_stat,
            p_value,
            skewness,
            kurtosis,
            reject_h0: p_value < self.significance_level,
        })
    }

    fn test_heteroskedasticity(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
    ) -> Result<HeteroskedasticityTest, ValidationError> {
        let n = residuals.len();
        
        // Compute squared residuals
        let squared_residuals: Vec<f64> = residuals.iter()
            .map(|&x| x.powi(2))
            .collect();

        // Create design matrix for auxiliary regression
        let mut X = na::DMatrix::zeros(n, 2);
        for i in 0..n {
            X[(i, 0)] = 1.0;
            X[(i, 1)] = fitted_values[i];
        }

        // Perform auxiliary regression
        let y = na::DVector::from_vec(squared_residuals);
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let beta = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(ValidationError::StatisticalError(
                "Singular matrix in heteroskedasticity test".to_string(),
            )),
        };

        // Compute R-squared of auxiliary regression
        let fitted = &X * &beta;
        let residuals = y - fitted;
        
        let tss = y.iter()
            .map(|&x| (x - y.mean()).powi(2))
            .sum::<f64>();
        
        let rss = residuals.iter()
            .map(|&x| x.powi(2))
            .sum::<f64>();
        
        let r_squared = 1.0 - rss / tss;

        // Compute test statistic (n * R^2)
        let bp_stat = n as f64 * r_squared;
        
        // Compute p-value using chi-square distribution with 1 df
        let chi_square = ChiSquare::new(1.0)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let p_value = 1.0 - chi_square.cdf(bp_stat);

        Ok(HeteroskedasticityTest {
            statistic: bp_stat,
            p_value,
            r_squared,
            reject_h0: p_value < self.significance_level,
        })
    }

    fn test_autocorrelation(&self, residuals: &[f64]) -> Result<AutocorrelationTest, ValidationError> {
        let n = residuals.len();
        let max_lag = (n as f64).sqrt().floor() as usize;
        
        // Compute autocorrelations
        let mean = residuals.iter().sum::<f64>() / n as f64;
        let variance = residuals.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;

        let mut autocorr = Vec::with_capacity(max_lag);
        for lag in 1..=max_lag {
            let ac = residuals.windows(lag + 1)
                .map(|w| (w[0] - mean) * (w[lag] - mean))
                .sum::<f64>() / (n as f64 * variance);
            autocorr.push(ac);
        }

        // Compute Ljung-Box statistic
        let lb_stat = (n * (n + 2)) as f64 * autocorr.iter()
            .enumerate()
            .map(|(k, &ac)| ac.powi(2) / (n - k - 1) as f64)
            .sum::<f64>();

        // Compute p-value using chi-square distribution
        let chi_square = ChiSquare::new(max_lag as f64)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let p_value = 1.0 - chi_square.cdf(lb_stat);

        Ok(AutocorrelationTest {
            statistic: lb_stat,
            p_value,
            autocorrelations: autocorr,
            reject_h0: p_value < self.significance_level,
        })
    }

    fn test_stationarity(&self, series: &[f64]) -> Result<StationarityTest, ValidationError> {
        let n = series.len();
        
        // Compute first differences
        let diff: Vec<f64> = series.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        // Create design matrix for ADF regression
        let mut X = na::DMatrix::zeros(n - 1, 2);
        for i in 0..(n-1) {
            X[(i, 0)] = 1.0;
            X[(i, 1)] = series[i];
        }

        let y = na::DVector::from_vec(diff);
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let beta = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(ValidationError::StatisticalError(
                "Singular matrix in stationarity test".to_string(),
            )),
        };

        // Compute test statistic
        let residuals = y - &X * &beta;
        let sigma2 = residuals.iter()
            .map(|&x| x.powi(2))
            .sum::<f64>() / (n - 3) as f64;

        let var_beta = (XtX.try_inverse()
            .ok_or_else(|| ValidationError::StatisticalError(
                "Singular matrix in variance computation".to_string(),
            ))? * sigma2)[(1, 1)];

        let adf_stat = beta[1] / var_beta.sqrt();

        // Use pre-computed critical values (for simplicity)
        let critical_values = vec![-3.43, -2.86, -2.57]; // 1%, 5%, 10%
        let p_value = if adf_stat < critical_values[1] { 0.05 } else { 0.10 };

        Ok(StationarityTest {
            statistic: adf_stat,
            p_value,
            critical_values,
            reject_h0: p_value < self.significance_level,
        })
    }

    fn test_structural_breaks(&self, series: &[f64]) -> Result<StructuralBreakTest, ValidationError> {
        let n = series.len();
        let min_size = (0.15 * n as f64) as usize;
        
        // Compute CUSUM statistics
        let mean = series.iter().sum::<f64>() / n as f64;
        let std_dev = (series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64)
            .sqrt();

        let cusum: Vec<f64> = series.iter()
            .scan(0.0, |acc, &x| {
                *acc += (x - mean) / std_dev;
                Some(*acc)
            })
            .collect();

        // Find maximum absolute CUSUM statistic
        let max_cusum = cusum.iter()
            .map(|&x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        // Compute break point
        let break_point = cusum.iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Use pre-computed critical values (for simplicity)
        let critical_value = 1.36; // 5% significance level
        let p_value = if max_cusum > critical_value { 0.05 } else { 0.10 };

        Ok(StructuralBreakTest {
            statistic: max_cusum,
            p_value,
            break_point,
            cusum_path: cusum,
            reject_h0: p_value < self.significance_level,
        })
    }
}

#[derive(Debug)]
pub struct NormalityTest {
    pub statistic: f64,
    pub p_value: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct HeteroskedasticityTest {
    pub statistic: f64,
    pub p_value: f64,
    pub r_squared: f64,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct AutocorrelationTest {
    pub statistic: f64,
    pub p_value: f64,
    pub autocorrelations: Vec<f64>,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct StationarityTest {
    pub statistic: f64,
    pub p_value: f64,
    pub critical_values: Vec<f64>,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct StructuralBreakTest {
    pub statistic: f64,
    pub p_value: f64,
    pub break_point: usize,
    pub cusum_path: Vec<f64>,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct DiagnosticResults {
    pub normality_test: NormalityTest,
    pub heteroskedasticity_test: HeteroskedasticityTest,
    pub autocorrelation_test: AutocorrelationTest,
    pub stationarity_test: StationarityTest,
    pub structural_break_test: StructuralBreakTest,
}
