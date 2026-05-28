use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use statrs::distribution::{StudentsT, ContinuousCDF};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Statistical error: {0}")]
    StatisticalError(String),
}

pub struct ModelValidator {
    n_bootstrap: usize,
    n_cv_folds: usize,
    confidence_level: f64,
    random_seed: Option<u64>,
}

impl ModelValidator {
    pub fn new(
        n_bootstrap: usize,
        n_cv_folds: usize,
        confidence_level: f64,
        random_seed: Option<u64>,
    ) -> Result<Self, ValidationError> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(ValidationError::ValidationError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            n_bootstrap,
            n_cv_folds,
            confidence_level,
            random_seed,
        })
    }

    pub fn bootstrap_validation<F>(
        &self,
        data: &[f64],
        estimator: F,
    ) -> Result<BootstrapResults, ValidationError>
    where
        F: Fn(&[f64]) -> Result<f64, ValidationError> + Send + Sync,
    {
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n = data.len();
        let estimates: Vec<f64> = (0..self.n_bootstrap)
            .into_par_iter()
            .map(|_| {
                let bootstrap_sample: Vec<f64> = (0..n)
                    .map(|_| data[rng.gen_range(0..n)])
                    .collect();
                
                estimator(&bootstrap_sample)
            })
            .collect::<Result<Vec<f64>, ValidationError>>()?;

        let mean = estimates.par_iter().sum::<f64>() / self.n_bootstrap as f64;
        
        let variance = estimates.par_iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.n_bootstrap - 1) as f64;
        
        let std_error = variance.sqrt();

        // Compute confidence intervals
        let mut sorted_estimates = estimates.clone();
        sorted_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let lower_idx = ((1.0 - self.confidence_level) / 2.0 * self.n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 + self.confidence_level) / 2.0 * self.n_bootstrap as f64) as usize;

        Ok(BootstrapResults {
            mean,
            std_error,
            confidence_interval: (sorted_estimates[lower_idx], sorted_estimates[upper_idx]),
            estimates,
        })
    }

    pub fn cross_validation<F, T>(
        &self,
        data: &[T],
        estimator: F,
    ) -> Result<CrossValidationResults, ValidationError>
    where
        F: Fn(&[T], &[T]) -> Result<f64, ValidationError> + Send + Sync,
        T: Clone + Send + Sync,
    {
        let n = data.len();
        let fold_size = n / self.n_cv_folds;
        
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create random permutation of indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let scores: Vec<f64> = (0..self.n_cv_folds)
            .into_par_iter()
            .map(|fold| {
                let start = fold * fold_size;
                let end = if fold == self.n_cv_folds - 1 { n } else { start + fold_size };
                
                let test_indices: Vec<usize> = indices[start..end].to_vec();
                let train_indices: Vec<usize> = indices.iter()
                    .filter(|&&i| !test_indices.contains(&i))
                    .cloned()
                    .collect();

                let train_data: Vec<T> = train_indices.iter()
                    .map(|&i| data[i].clone())
                    .collect();
                let test_data: Vec<T> = test_indices.iter()
                    .map(|&i| data[i].clone())
                    .collect();

                estimator(&train_data, &test_data)
            })
            .collect::<Result<Vec<f64>, ValidationError>>()?;

        let mean_score = scores.par_iter().sum::<f64>() / self.n_cv_folds as f64;
        
        let variance = scores.par_iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>() / (self.n_cv_folds - 1) as f64;
        
        let std_error = (variance / self.n_cv_folds as f64).sqrt();

        // Compute confidence intervals using t-distribution
        let t_dist = StudentsT::new((self.n_cv_folds - 1) as f64)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let t_value = t_dist.inverse_cdf(1.0 - (1.0 - self.confidence_level) / 2.0);
        let margin = t_value * std_error;

        Ok(CrossValidationResults {
            mean_score,
            std_error,
            confidence_interval: (mean_score - margin, mean_score + margin),
            fold_scores: scores,
        })
    }

    pub fn out_of_sample_validation<F, T>(
        &self,
        train_data: &[T],
        test_data: &[T],
        estimator: F,
    ) -> Result<OutOfSampleResults, ValidationError>
    where
        F: Fn(&[T], &[T]) -> Result<Vec<f64>, ValidationError>,
        T: Clone,
    {
        let predictions = estimator(train_data, test_data)?;
        
        if predictions.is_empty() {
            return Err(ValidationError::ValidationError(
                "No predictions generated".to_string(),
            ));
        }

        let n = predictions.len();
        let mean = predictions.iter().sum::<f64>() / n as f64;
        
        let variance = predictions.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        
        let std_error = variance.sqrt();

        // Compute prediction intervals
        let normal = Normal::new(mean, std_error)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let simulated_predictions: Vec<f64> = (0..1000)
            .map(|_| normal.sample(&mut rng))
            .collect();

        let mut sorted_predictions = simulated_predictions.clone();
        sorted_predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let lower_idx = ((1.0 - self.confidence_level) / 2.0 * 1000.0) as usize;
        let upper_idx = ((1.0 + self.confidence_level) / 2.0 * 1000.0) as usize;

        Ok(OutOfSampleResults {
            mean,
            std_error,
            prediction_interval: (sorted_predictions[lower_idx], sorted_predictions[upper_idx]),
            predictions,
        })
    }

    pub fn compute_error_metrics(
        &self,
        actual: &[f64],
        predicted: &[f64],
    ) -> Result<ErrorMetrics, ValidationError> {
        if actual.len() != predicted.len() {
            return Err(ValidationError::ValidationError(
                "Actual and predicted values must have same length".to_string(),
            ));
        }

        let n = actual.len() as f64;
        
        let mse = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum::<f64>() / n;
        
        let rmse = mse.sqrt();
        
        let mae = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).abs())
            .sum::<f64>() / n;
        
        let mape = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| if a != 0.0 { (a - p).abs() / a.abs() } else { 0.0 })
            .sum::<f64>() / n * 100.0;

        let mean_actual = actual.iter().sum::<f64>() / n;
        let ss_tot = actual.iter()
            .map(|&a| (a - mean_actual).powi(2))
            .sum::<f64>();
        let ss_res = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum::<f64>();
        let r_squared = 1.0 - ss_res / ss_tot;

        Ok(ErrorMetrics {
            mse,
            rmse,
            mae,
            mape,
            r_squared,
        })
    }
}

#[derive(Debug)]
pub struct BootstrapResults {
    pub mean: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub estimates: Vec<f64>,
}

#[derive(Debug)]
pub struct CrossValidationResults {
    pub mean_score: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub fold_scores: Vec<f64>,
}

#[derive(Debug)]
pub struct OutOfSampleResults {
    pub mean: f64,
    pub std_error: f64,
    pub prediction_interval: (f64, f64),
    pub predictions: Vec<f64>,
}

#[derive(Debug)]
pub struct ErrorMetrics {
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub r_squared: f64,
}
