use tokio::sync::{mpsc, oneshot};
use tokio::task;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Error)]
pub enum DistributedError {
    #[error("Task error: {0}")]
    TaskError(String),
    #[error("Communication error: {0}")]
    CommunicationError(String),
}

#[derive(Debug)]
pub enum ComputeTask {
    SpreadEstimation {
        data: Arc<Vec<f64>>,
        window_size: usize,
        response: oneshot::Sender<Result<Vec<f64>, DistributedError>>,
    },
    VolatilityComputation {
        returns: Arc<Vec<f64>>,
        window_size: usize,
        response: oneshot::Sender<Result<Vec<f64>, DistributedError>>,
    },
    CrossSectionalAnalysis {
        data: Arc<HashMap<String, Vec<f64>>>,
        response: oneshot::Sender<Result<CrossSectionalResult, DistributedError>>,
    },
}

#[derive(Debug)]
pub struct CrossSectionalResult {
    pub coefficients: Vec<f64>,
    pub t_statistics: Vec<f64>,
    pub r_squared: f64,
}

pub struct DistributedCompute {
    num_workers: usize,
    task_sender: mpsc::Sender<ComputeTask>,
    chunk_size: usize,
}

impl DistributedCompute {
    pub async fn new(num_workers: usize, chunk_size: usize) -> Result<Self, DistributedError> {
        let (tx, rx) = mpsc::channel(1000);
        let rx = Arc::new(tokio::sync::Mutex::new(rx));

        // Spawn worker tasks
        for worker_id in 0..num_workers {
            let rx = rx.clone();
            task::spawn(async move {
                Self::worker_loop(worker_id, rx).await;
            });
        }

        Ok(Self {
            num_workers,
            task_sender: tx,
            chunk_size,
        })
    }

    async fn worker_loop(
        worker_id: usize,
        rx: Arc<tokio::sync::Mutex<mpsc::Receiver<ComputeTask>>>,
    ) {
        loop {
            let task = {
                let mut rx = rx.lock().await;
                rx.recv().await
            };

            match task {
                Some(ComputeTask::SpreadEstimation { data, window_size, response }) => {
                    let result = Self::compute_spread_estimation(&data, window_size);
                    let _ = response.send(result);
                },
                Some(ComputeTask::VolatilityComputation { returns, window_size, response }) => {
                    let result = Self::compute_volatility(&returns, window_size);
                    let _ = response.send(result);
                },
                Some(ComputeTask::CrossSectionalAnalysis { data, response }) => {
                    let result = Self::compute_cross_sectional(&data);
                    let _ = response.send(result);
                },
                None => break,
            }
        }
    }

    pub async fn parallel_spread_estimation(
        &self,
        data: Vec<f64>,
        window_size: usize,
    ) -> Result<Vec<f64>, DistributedError> {
        let data = Arc::new(data);
        let chunks = (data.len() + self.chunk_size - 1) / self.chunk_size;
        let mut handles = Vec::with_capacity(chunks);

        for chunk_idx in 0..chunks {
            let (tx, rx) = oneshot::channel();
            let start = chunk_idx * self.chunk_size;
            let end = (start + self.chunk_size).min(data.len());
            let chunk_data = Arc::new(data[start..end].to_vec());

            self.task_sender.send(ComputeTask::SpreadEstimation {
                data: chunk_data,
                window_size,
                response: tx,
            }).await.map_err(|e| DistributedError::CommunicationError(e.to_string()))?;

            handles.push((start, rx));
        }

        let mut results = vec![0.0; data.len()];
        for (start, handle) in handles {
            let chunk_result = handle.await
                .map_err(|e| DistributedError::CommunicationError(e.to_string()))??;
            
            let end = (start + chunk_result.len()).min(results.len());
            results[start..end].copy_from_slice(&chunk_result);
        }

        Ok(results)
    }

    pub async fn parallel_volatility_computation(
        &self,
        returns: Vec<f64>,
        window_size: usize,
    ) -> Result<Vec<f64>, DistributedError> {
        let returns = Arc::new(returns);
        let chunks = (returns.len() + self.chunk_size - 1) / self.chunk_size;
        let mut handles = Vec::with_capacity(chunks);

        for chunk_idx in 0..chunks {
            let (tx, rx) = oneshot::channel();
            let start = chunk_idx * self.chunk_size;
            let end = (start + self.chunk_size + window_size).min(returns.len());
            let chunk_data = Arc::new(returns[start..end].to_vec());

            self.task_sender.send(ComputeTask::VolatilityComputation {
                returns: chunk_data,
                window_size,
                response: tx,
            }).await.map_err(|e| DistributedError::CommunicationError(e.to_string()))?;

            handles.push((start, rx));
        }

        let mut results = vec![0.0; returns.len()];
        for (start, handle) in handles {
            let chunk_result = handle.await
                .map_err(|e| DistributedError::CommunicationError(e.to_string()))??;
            
            let end = (start + chunk_result.len()).min(results.len());
            results[start..end].copy_from_slice(&chunk_result);
        }

        Ok(results)
    }

    pub async fn parallel_cross_sectional(
        &self,
        data: HashMap<String, Vec<f64>>,
    ) -> Result<CrossSectionalResult, DistributedError> {
        let data = Arc::new(data);
        let (tx, rx) = oneshot::channel();

        self.task_sender.send(ComputeTask::CrossSectionalAnalysis {
            data,
            response: tx,
        }).await.map_err(|e| DistributedError::CommunicationError(e.to_string()))?;

        rx.await.map_err(|e| DistributedError::CommunicationError(e.to_string()))?
    }

    fn compute_spread_estimation(
        data: &[f64],
        window_size: usize,
    ) -> Result<Vec<f64>, DistributedError> {
        if data.len() < window_size {
            return Err(DistributedError::TaskError(
                "Data length smaller than window size".to_string(),
            ));
        }

        let result: Vec<f64> = data.par_windows(window_size)
            .map(|window| {
                let mean = window.par_iter().sum::<f64>() / window.len() as f64;
                let variance = window.par_iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (window.len() - 1) as f64;
                2.0 * variance.sqrt()
            })
            .collect();

        Ok(result)
    }

    fn compute_volatility(
        returns: &[f64],
        window_size: usize,
    ) -> Result<Vec<f64>, DistributedError> {
        if returns.len() < window_size {
            return Err(DistributedError::TaskError(
                "Returns length smaller than window size".to_string(),
            ));
        }

        let result: Vec<f64> = returns.par_windows(window_size)
            .map(|window| {
                let mean = window.par_iter().sum::<f64>() / window.len() as f64;
                let variance = window.par_iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / (window.len() - 1) as f64;
                variance.sqrt()
            })
            .collect();

        Ok(result)
    }

    fn compute_cross_sectional(
        data: &HashMap<String, Vec<f64>>,
    ) -> Result<CrossSectionalResult, DistributedError> {
        if data.is_empty() {
            return Err(DistributedError::TaskError("Empty data".to_string()));
        }

        let n_obs = data.values().next().unwrap().len();
        let n_vars = data.len();

        // Prepare matrices for regression
        let mut X = vec![vec![0.0; n_vars]; n_obs];
        let mut y = vec![0.0; n_obs];

        for (i, (_, series)) in data.iter().enumerate() {
            for (j, &value) in series.iter().enumerate() {
                if i == 0 {
                    y[j] = value;
                } else {
                    X[j][i-1] = value;
                }
            }
        }

        // Compute OLS regression
        let (coefficients, t_stats, r_squared) = Self::compute_ols(&X, &y)?;

        Ok(CrossSectionalResult {
            coefficients,
            t_statistics: t_stats,
            r_squared,
        })
    }

    fn compute_ols(
        X: &[Vec<f64>],
        y: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>, f64), DistributedError> {
        let n = X.len();
        let p = X[0].len();

        // Compute X'X
        let mut XtX = vec![vec![0.0; p]; p];
        for i in 0..p {
            for j in 0..p {
                XtX[i][j] = (0..n)
                    .into_par_iter()
                    .map(|k| X[k][i] * X[k][j])
                    .sum();
            }
        }

        // Compute X'y
        let mut Xty = vec![0.0; p];
        for i in 0..p {
            Xty[i] = (0..n)
                .into_par_iter()
                .map(|k| X[k][i] * y[k])
                .sum();
        }

        // Solve system of equations
        let coefficients = Self::solve_system(&XtX, &Xty)?;

        // Compute residuals and R-squared
        let y_hat: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                (0..p).map(|j| coefficients[j] * X[i][j]).sum()
            })
            .collect();

        let residuals: Vec<f64> = y.iter()
            .zip(y_hat.iter())
            .map(|(&yi, &y_hati)| yi - y_hati)
            .collect();

        let tss: f64 = y.iter()
            .map(|&yi| yi.powi(2))
            .sum();
        let rss: f64 = residuals.iter()
            .map(|&r| r.powi(2))
            .sum();
        let r_squared = 1.0 - rss / tss;

        // Compute standard errors and t-statistics
        let mse = rss / (n - p) as f64;
        let std_errors: Vec<f64> = (0..p)
            .map(|i| (mse * XtX[i][i]).sqrt())
            .collect();
        let t_stats: Vec<f64> = coefficients.iter()
            .zip(std_errors.iter())
            .map(|(&b, &se)| b / se)
            .collect();

        Ok((coefficients, t_stats, r_squared))
    }

    fn solve_system(
        A: &[Vec<f64>],
        b: &[f64],
    ) -> Result<Vec<f64>, DistributedError> {
        let n = A.len();
        let mut L = vec![vec![0.0; n]; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n];

        // Cholesky decomposition
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if i == j {
                    for k in 0..j {
                        sum += L[j][k].powi(2);
                    }
                    L[i][j] = (A[i][i] - sum).sqrt();
                } else {
                    for k in 0..j {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (A[i][j] - sum) / L[j][j];
                }
            }
        }

        // Forward substitution
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += L[i][j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i][i];
        }

        // Backward substitution
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i+1)..n {
                sum += L[j][i] * x[j];
            }
            x[i] = (y[i] - sum) / L[i][i];
        }

        Ok(x)
    }
}
