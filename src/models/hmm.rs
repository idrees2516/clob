use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HMMError {
    #[error("Invalid state transition matrix: {0}")]
    InvalidTransitionMatrix(String),
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub struct HiddenMarkovModel {
    n_states: usize,
    transition_matrix: na::DMatrix<f64>,
    emission_means: na::DVector<f64>,
    emission_vars: na::DVector<f64>,
    initial_probs: na::DVector<f64>,
}

impl HiddenMarkovModel {
    pub fn new(
        n_states: usize,
        transition_matrix: na::DMatrix<f64>,
        emission_means: na::DVector<f64>,
        emission_vars: na::DVector<f64>,
        initial_probs: na::DVector<f64>,
    ) -> Result<Self, HMMError> {
        if transition_matrix.nrows() != n_states || transition_matrix.ncols() != n_states {
            return Err(HMMError::DimensionMismatch(
                "Transition matrix dimensions do not match number of states".to_string(),
            ));
        }

        if !Self::is_valid_probability_matrix(&transition_matrix) {
            return Err(HMMError::InvalidTransitionMatrix(
                "Invalid transition probabilities".to_string(),
            ));
        }

        Ok(Self {
            n_states,
            transition_matrix,
            emission_means,
            emission_vars,
            initial_probs,
        })
    }

    fn is_valid_probability_matrix(matrix: &na::DMatrix<f64>) -> bool {
        let rows_sum_to_one = matrix.row_iter().all(|row| {
            (row.iter().sum::<f64>() - 1.0).abs() < 1e-10
        });

        let all_probs_valid = matrix.iter().all(|&p| p >= 0.0 && p <= 1.0);

        rows_sum_to_one && all_probs_valid
    }

    pub fn forward_algorithm(&self, observations: &[f64]) -> Result<na::DMatrix<f64>, HMMError> {
        let t = observations.len();
        let mut alpha = na::DMatrix::zeros(self.n_states, t);
        
        // Initialize first column
        for i in 0..self.n_states {
            alpha[(i, 0)] = self.initial_probs[i] * self.emission_probability(observations[0], i);
        }

        // Forward pass
        for t in 1..observations.len() {
            for j in 0..self.n_states {
                let mut sum = 0.0;
                for i in 0..self.n_states {
                    sum += alpha[(i, t-1)] * self.transition_matrix[(i, j)];
                }
                alpha[(j, t)] = sum * self.emission_probability(observations[t], j);
            }
        }

        Ok(alpha)
    }

    pub fn backward_algorithm(&self, observations: &[f64]) -> Result<na::DMatrix<f64>, HMMError> {
        let t = observations.len();
        let mut beta = na::DMatrix::zeros(self.n_states, t);
        
        // Initialize last column
        for i in 0..self.n_states {
            beta[(i, t-1)] = 1.0;
        }

        // Backward pass
        for t in (0..observations.len()-1).rev() {
            for i in 0..self.n_states {
                let mut sum = 0.0;
                for j in 0..self.n_states {
                    sum += self.transition_matrix[(i, j)] 
                        * self.emission_probability(observations[t+1], j) 
                        * beta[(j, t+1)];
                }
                beta[(i, t)] = sum;
            }
        }

        Ok(beta)
    }

    pub fn viterbi_algorithm(&self, observations: &[f64]) -> Result<Vec<usize>, HMMError> {
        let t = observations.len();
        let mut delta = na::DMatrix::zeros(self.n_states, t);
        let mut psi = na::DMatrix::zeros(self.n_states, t);
        
        // Initialize
        for i in 0..self.n_states {
            delta[(i, 0)] = self.initial_probs[i] * self.emission_probability(observations[0], i);
            psi[(i, 0)] = 0.0;
        }

        // Forward pass
        for t in 1..observations.len() {
            for j in 0..self.n_states {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_index = 0;
                
                for i in 0..self.n_states {
                    let val = delta[(i, t-1)] 
                        * self.transition_matrix[(i, j)] 
                        * self.emission_probability(observations[t], j);
                    if val > max_val {
                        max_val = val;
                        max_index = i;
                    }
                }
                
                delta[(j, t)] = max_val;
                psi[(j, t)] = max_index as f64;
            }
        }

        // Backtrack
        let mut path = vec![0; t];
        let mut max_val = f64::NEG_INFINITY;
        
        for i in 0..self.n_states {
            if delta[(i, t-1)] > max_val {
                max_val = delta[(i, t-1)];
                path[t-1] = i;
            }
        }

        for t in (1..observations.len()).rev() {
            path[t-1] = psi[(path[t], t)] as usize;
        }

        Ok(path)
    }

    pub fn baum_welch_algorithm(
        &mut self,
        observations: &[f64],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<(), HMMError> {
        let mut prev_log_likelihood = f64::NEG_INFINITY;
        
        for _ in 0..max_iter {
            // E-step
            let alpha = self.forward_algorithm(observations)?;
            let beta = self.backward_algorithm(observations)?;
            
            // Calculate gamma and xi
            let (gamma, xi) = self.calculate_gamma_xi(observations, &alpha, &beta)?;
            
            // M-step
            self.update_parameters(observations, &gamma, &xi)?;
            
            // Check convergence
            let log_likelihood = self.calculate_log_likelihood(&alpha);
            if (log_likelihood - prev_log_likelihood).abs() < tolerance {
                return Ok(());
            }
            prev_log_likelihood = log_likelihood;
        }

        Err(HMMError::ConvergenceError(
            "Maximum iterations reached without convergence".to_string(),
        ))
    }

    fn emission_probability(&self, observation: f64, state: usize) -> f64 {
        let mean = self.emission_means[state];
        let var = self.emission_vars[state];
        let normal = Normal::new(mean, var.sqrt()).unwrap();
        normal.pdf(observation)
    }

    fn calculate_gamma_xi(
        &self,
        observations: &[f64],
        alpha: &na::DMatrix<f64>,
        beta: &na::DMatrix<f64>,
    ) -> Result<(na::DMatrix<f64>, Vec<na::DMatrix<f64>>), HMMError> {
        let t = observations.len();
        let mut gamma = na::DMatrix::zeros(self.n_states, t);
        let mut xi = vec![na::DMatrix::zeros(self.n_states, self.n_states); t-1];
        
        for t in 0..observations.len() {
            let mut sum = 0.0;
            for i in 0..self.n_states {
                gamma[(i, t)] = alpha[(i, t)] * beta[(i, t)];
                sum += gamma[(i, t)];
            }
            
            // Normalize gamma
            for i in 0..self.n_states {
                gamma[(i, t)] /= sum;
            }
            
            if t < observations.len() - 1 {
                let mut sum = 0.0;
                for i in 0..self.n_states {
                    for j in 0..self.n_states {
                        xi[t][(i, j)] = alpha[(i, t)] 
                            * self.transition_matrix[(i, j)]
                            * self.emission_probability(observations[t+1], j)
                            * beta[(j, t+1)];
                        sum += xi[t][(i, j)];
                    }
                }
                
                // Normalize xi
                for i in 0..self.n_states {
                    for j in 0..self.n_states {
                        xi[t][(i, j)] /= sum;
                    }
                }
            }
        }

        Ok((gamma, xi))
    }

    fn update_parameters(
        &mut self,
        observations: &[f64],
        gamma: &na::DMatrix<f64>,
        xi: &[na::DMatrix<f64>],
    ) -> Result<(), HMMError> {
        // Update transition matrix
        for i in 0..self.n_states {
            for j in 0..self.n_states {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                
                for t in 0..observations.len()-1 {
                    numerator += xi[t][(i, j)];
                    denominator += gamma[(i, t)];
                }
                
                self.transition_matrix[(i, j)] = numerator / denominator;
            }
        }

        // Update emission parameters
        for i in 0..self.n_states {
            let mut mean_num = 0.0;
            let mut mean_den = 0.0;
            
            for t in 0..observations.len() {
                mean_num += gamma[(i, t)] * observations[t];
                mean_den += gamma[(i, t)];
            }
            
            self.emission_means[i] = mean_num / mean_den;
            
            let mut var_num = 0.0;
            for t in 0..observations.len() {
                var_num += gamma[(i, t)] * (observations[t] - self.emission_means[i]).powi(2);
            }
            self.emission_vars[i] = var_num / mean_den;
        }

        Ok(())
    }

    fn calculate_log_likelihood(&self, alpha: &na::DMatrix<f64>) -> f64 {
        let t = alpha.ncols() - 1;
        alpha.column(t).iter().sum::<f64>().ln()
    }
}
