use super::{ValidationError, ModelValidator, ErrorMetrics};
use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{StudentsT, FisherSnedecor, ChiSquare, ContinuousCDF};
use std::collections::HashMap;

pub struct ModelComparison {
    validator: ModelValidator,
    significance_level: f64,
}

impl ModelComparison {
    pub fn new(
        validator: ModelValidator,
        significance_level: f64,
    ) -> Result<Self, ValidationError> {
        if significance_level <= 0.0 || significance_level >= 1.0 {
            return Err(ValidationError::ValidationError(
                "Significance level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            validator,
            significance_level,
        })
    }

    pub fn compare_models<F>(
        &self,
        data: &[f64],
        models: &HashMap<String, F>,
    ) -> Result<ComparisonResults, ValidationError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, ValidationError> + Send + Sync,
    {
        let mut model_metrics = HashMap::new();
        let mut model_predictions = HashMap::new();

        // Compute predictions and metrics for each model
        for (name, model) in models {
            let predictions = model(data)?;
            let metrics = self.validator.compute_error_metrics(data, &predictions)?;
            
            model_metrics.insert(name.clone(), metrics);
            model_predictions.insert(name.clone(), predictions);
        }

        // Perform statistical tests
        let dm_tests = self.diebold_mariano_tests(&model_predictions, data)?;
        let encompassing_tests = self.forecast_encompassing_tests(&model_predictions, data)?;
        let model_confidence = self.model_confidence_set(&model_predictions, data)?;

        Ok(ComparisonResults {
            model_metrics,
            dm_tests,
            encompassing_tests,
            model_confidence,
        })
    }

    fn diebold_mariano_tests(
        &self,
        predictions: &HashMap<String, Vec<f64>>,
        actual: &[f64],
    ) -> Result<HashMap<(String, String), DMTestResult>, ValidationError> {
        let mut results = HashMap::new();
        let models: Vec<&String> = predictions.keys().collect();

        for i in 0..models.len() {
            for j in (i+1)..models.len() {
                let model1 = &models[i];
                let model2 = &models[j];
                
                let pred1 = &predictions[*model1];
                let pred2 = &predictions[*model2];
                
                let dm_result = self.compute_dm_test(pred1, pred2, actual)?;
                results.insert(((*model1).clone(), (*model2).clone()), dm_result);
            }
        }

        Ok(results)
    }

    fn compute_dm_test(
        &self,
        pred1: &[f64],
        pred2: &[f64],
        actual: &[f64],
    ) -> Result<DMTestResult, ValidationError> {
        let n = actual.len();
        
        // Compute loss differentials
        let loss_diff: Vec<f64> = actual.iter()
            .zip(pred1.iter().zip(pred2.iter()))
            .map(|(&a, (&p1, &p2))| {
                let e1 = (a - p1).powi(2);
                let e2 = (a - p2).powi(2);
                e1 - e2
            })
            .collect();

        let mean_diff = loss_diff.iter().sum::<f64>() / n as f64;
        
        // Compute long-run variance estimate using Newey-West
        let max_lag = (n as f64).powf(1.0/3.0).floor() as usize;
        let mut variance = loss_diff.iter()
            .map(|&x| (x - mean_diff).powi(2))
            .sum::<f64>() / n as f64;
        
        for lag in 1..=max_lag {
            let weight = 1.0 - lag as f64 / (max_lag + 1) as f64;
            let autocovariance: f64 = loss_diff.windows(lag + 1)
                .map(|w| (w[0] - mean_diff) * (w[lag] - mean_diff))
                .sum::<f64>() / n as f64;
            
            variance += 2.0 * weight * autocovariance;
        }

        let dm_statistic = mean_diff / (variance / n as f64).sqrt();
        
        // Compute p-value using t-distribution
        let t_dist = StudentsT::new((n - 1) as f64)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let p_value = 2.0 * (1.0 - t_dist.cdf(dm_statistic.abs()));

        Ok(DMTestResult {
            statistic: dm_statistic,
            p_value,
            reject_h0: p_value < self.significance_level,
        })
    }

    fn forecast_encompassing_tests(
        &self,
        predictions: &HashMap<String, Vec<f64>>,
        actual: &[f64],
    ) -> Result<HashMap<(String, String), EncompassingTestResult>, ValidationError> {
        let mut results = HashMap::new();
        let models: Vec<&String> = predictions.keys().collect();

        for i in 0..models.len() {
            for j in (i+1)..models.len() {
                let model1 = &models[i];
                let model2 = &models[j];
                
                let pred1 = &predictions[*model1];
                let pred2 = &predictions[*model2];
                
                let result = self.compute_encompassing_test(pred1, pred2, actual)?;
                results.insert(((*model1).clone(), (*model2).clone()), result);
            }
        }

        Ok(results)
    }

    fn compute_encompassing_test(
        &self,
        pred1: &[f64],
        pred2: &[f64],
        actual: &[f64],
    ) -> Result<EncompassingTestResult, ValidationError> {
        let n = actual.len();
        
        // Construct design matrix for regression
        let mut X = na::DMatrix::zeros(n, 2);
        for i in 0..n {
            X[(i, 0)] = pred1[i];
            X[(i, 1)] = pred2[i];
        }
        let y = na::DVector::from_vec(actual.to_vec());

        // Compute OLS estimates
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let beta = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(ValidationError::StatisticalError(
                "Singular matrix in encompassing test".to_string(),
            )),
        };

        // Compute residuals and variance
        let residuals = y - &X * &beta;
        let sigma2 = residuals.dot(&residuals) / (n - 2) as f64;
        
        // Compute variance-covariance matrix
        let var_cov = XtX.try_inverse()
            .ok_or_else(|| ValidationError::StatisticalError(
                "Singular matrix in variance computation".to_string(),
            ))? * sigma2;

        // Compute test statistics
        let lambda1 = beta[0];
        let lambda2 = beta[1];
        let se1 = var_cov[(0, 0)].sqrt();
        let se2 = var_cov[(1, 1)].sqrt();

        let t1 = (lambda1 - 1.0) / se1;
        let t2 = lambda2 / se2;

        // Compute p-values
        let t_dist = StudentsT::new((n - 2) as f64)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let p1 = 2.0 * (1.0 - t_dist.cdf(t1.abs()));
        let p2 = 2.0 * (1.0 - t_dist.cdf(t2.abs()));

        Ok(EncompassingTestResult {
            lambda1,
            lambda2,
            se1,
            se2,
            t_stats: (t1, t2),
            p_values: (p1, p2),
            reject_h0: (p1 < self.significance_level, p2 < self.significance_level),
        })
    }

    fn model_confidence_set(
        &self,
        predictions: &HashMap<String, Vec<f64>>,
        actual: &[f64],
    ) -> Result<ModelConfidenceSet, ValidationError> {
        let models: Vec<String> = predictions.keys().cloned().collect();
        let n_models = models.len();
        
        // Compute loss matrix
        let mut loss_matrix = na::DMatrix::zeros(actual.len(), n_models);
        for (j, model) in models.iter().enumerate() {
            let pred = &predictions[model];
            for i in 0..actual.len() {
                loss_matrix[(i, j)] = (actual[i] - pred[i]).powi(2);
            }
        }

        // Compute test statistics for all pairs
        let mut t_stats = na::DMatrix::zeros(n_models, n_models);
        for i in 0..n_models {
            for j in (i+1)..n_models {
                let loss_diff = loss_matrix.column(i) - loss_matrix.column(j);
                let mean_diff = loss_diff.mean();
                let var = self.compute_variance_nw(&loss_diff.as_slice().to_vec())?;
                
                let t_stat = mean_diff / var.sqrt();
                t_stats[(i, j)] = t_stat;
                t_stats[(j, i)] = -t_stat;
            }
        }

        // Iteratively eliminate models
        let mut included: Vec<bool> = vec![true; n_models];
        let mut eliminated_order = Vec::new();
        let mut p_values = Vec::new();

        while included.iter().filter(|&&x| x).count() > 1 {
            // Compute test statistic for current set
            let max_t = included.iter()
                .enumerate()
                .filter(|&(_, &inc)| inc)
                .map(|(i, _)| {
                    included.iter()
                        .enumerate()
                        .filter(|&(_, &inc)| inc)
                        .map(|(j, _)| t_stats[(i, j)].abs())
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            // Compute p-value using bootstrap
            let p_value = self.compute_mcs_pvalue(&loss_matrix, &included, max_t)?;
            p_values.push(p_value);

            if p_value < self.significance_level {
                // Eliminate model with highest loss
                let mut max_loss = f64::NEG_INFINITY;
                let mut max_idx = 0;
                
                for (i, &inc) in included.iter().enumerate() {
                    if inc {
                        let loss = loss_matrix.column(i).mean();
                        if loss > max_loss {
                            max_loss = loss;
                            max_idx = i;
                        }
                    }
                }

                included[max_idx] = false;
                eliminated_order.push(models[max_idx].clone());
            } else {
                break;
            }
        }

        // Construct final set
        let mcs_models: Vec<String> = models.iter()
            .enumerate()
            .filter(|&(i, _)| included[i])
            .map(|(_, m)| m.clone())
            .collect();

        Ok(ModelConfidenceSet {
            included_models: mcs_models,
            eliminated_order,
            p_values,
        })
    }

    fn compute_variance_nw(&self, series: &[f64]) -> Result<f64, ValidationError> {
        let n = series.len();
        let mean = series.iter().sum::<f64>() / n as f64;
        
        // Compute autocovariances up to maximum lag
        let max_lag = (n as f64).powf(1.0/3.0).floor() as usize;
        let mut variance = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        for lag in 1..=max_lag {
            let weight = 1.0 - lag as f64 / (max_lag + 1) as f64;
            let autocovariance = series.windows(lag + 1)
                .map(|w| (w[0] - mean) * (w[lag] - mean))
                .sum::<f64>() / n as f64;
            
            variance += 2.0 * weight * autocovariance;
        }

        Ok(variance / n as f64)
    }

    fn compute_mcs_pvalue(
        &self,
        loss_matrix: &na::DMatrix<f64>,
        included: &[bool],
        observed_stat: f64,
    ) -> Result<f64, ValidationError> {
        let n_boot = 1000;
        let n = loss_matrix.nrows();
        let mut rng = rand::thread_rng();
        let mut exceeds = 0;

        for _ in 0..n_boot {
            // Generate bootstrap sample
            let indices: Vec<usize> = (0..n)
                .map(|_| rng.gen_range(0..n))
                .collect();
            
            let boot_matrix = indices.iter()
                .map(|&i| loss_matrix.row(i))
                .collect::<Vec<_>>();
            
            // Compute maximum t-statistic for bootstrap sample
            let mut max_t = 0.0;
            for (i, &inc_i) in included.iter().enumerate() {
                if !inc_i { continue; }
                
                for (j, &inc_j) in included.iter().enumerate() {
                    if !inc_j || i >= j { continue; }
                    
                    let diff: Vec<f64> = boot_matrix.iter()
                        .map(|row| row[i] - row[j])
                        .collect();
                    
                    let mean_diff = diff.iter().sum::<f64>() / n as f64;
                    let var = self.compute_variance_nw(&diff)?;
                    let t_stat = mean_diff / var.sqrt();
                    
                    max_t = max_t.max(t_stat.abs());
                }
            }

            if max_t > observed_stat {
                exceeds += 1;
            }
        }

        Ok(exceeds as f64 / n_boot as f64)
    }
}

#[derive(Debug)]
pub struct DMTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub reject_h0: bool,
}

#[derive(Debug)]
pub struct EncompassingTestResult {
    pub lambda1: f64,
    pub lambda2: f64,
    pub se1: f64,
    pub se2: f64,
    pub t_stats: (f64, f64),
    pub p_values: (f64, f64),
    pub reject_h0: (bool, bool),
}

#[derive(Debug)]
pub struct ModelConfidenceSet {
    pub included_models: Vec<String>,
    pub eliminated_order: Vec<String>,
    pub p_values: Vec<f64>,
}

#[derive(Debug)]
pub struct ComparisonResults {
    pub model_metrics: HashMap<String, ErrorMetrics>,
    pub dm_tests: HashMap<(String, String), DMTestResult>,
    pub encompassing_tests: HashMap<(String, String), EncompassingTestResult>,
    pub model_confidence: ModelConfidenceSet,
}
