use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;
use crate::models::{
    rough_volatility::{RoughVolatilityModel, RoughVolatilityParams},
    limit_order_book::{LimitOrderBook, LOBParams, Order},
};

#[derive(Debug, Error)]
pub enum StateSpaceError {
    #[error("State space error: {0}")]
    StateSpaceError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

/// State space model parameters
#[derive(Debug, Clone)]
pub struct StateSpaceParams {
    pub transition_matrix: na::DMatrix<f64>,
    pub observation_matrix: na::DMatrix<f64>,
    pub state_covariance: na::DMatrix<f64>,
    pub observation_covariance: na::DMatrix<f64>,
    pub initial_state: na::DVector<f64>,
    pub initial_covariance: na::DMatrix<f64>,
}

/// Kalman filter results
#[derive(Debug)]
pub struct KalmanFilterResults {
    pub filtered_states: Vec<na::DVector<f64>>,
    pub filtered_covs: Vec<na::DMatrix<f64>>,
    pub predicted_states: Vec<na::DVector<f64>>,
    pub predicted_covs: Vec<na::DMatrix<f64>>,
    pub likelihoods: Vec<f64>,
    pub innovations: Vec<na::DVector<f64>>,
    pub innovation_covs: Vec<na::DMatrix<f64>>,
}

/// Particle filter results
#[derive(Debug)]
pub struct ParticleFilterResults {
    pub filtered_states: Vec<na::DVector<f64>>,
    pub filtered_covs: Vec<na::DMatrix<f64>>,
    pub weights: Vec<Vec<f64>>,
    pub effective_particles: Vec<f64>,
    pub resampled_indices: Vec<Vec<usize>>,
}

/// State space model for filtering and estimation
pub struct StateSpaceModel {
    params: StateSpaceParams,
    n_particles: usize,
    resampling_threshold: f64,
}

impl StateSpaceModel {
    pub fn new(
        params: StateSpaceParams,
        n_particles: usize,
        resampling_threshold: f64,
    ) -> Result<Self, StateSpaceError> {
        // Validate dimensions
        let (n_states, m_states) = params.transition_matrix.shape();
        let (n_obs, m_obs) = params.observation_matrix.shape();
        
        if n_states != m_states {
            return Err(StateSpaceError::ParameterError(
                "Transition matrix must be square".to_string(),
            ));
        }
        
        if m_obs != n_states {
            return Err(StateSpaceError::ParameterError(
                "Observation matrix dimensions mismatch".to_string(),
            ));
        }
        
        if params.state_covariance.shape() != (n_states, n_states) {
            return Err(StateSpaceError::ParameterError(
                "State covariance matrix dimensions mismatch".to_string(),
            ));
        }
        
        if params.observation_covariance.shape() != (n_obs, n_obs) {
            return Err(StateSpaceError::ParameterError(
                "Observation covariance matrix dimensions mismatch".to_string(),
            ));
        }
        
        Ok(Self {
            params,
            n_particles,
            resampling_threshold,
        })
    }

    /// Runs Kalman filter
    pub fn kalman_filter(
        &self,
        observations: &[na::DVector<f64>],
    ) -> Result<KalmanFilterResults, StateSpaceError> {
        let n = observations.len();
        let state_dim = self.params.transition_matrix.nrows();
        
        let mut filtered_states = Vec::with_capacity(n);
        let mut filtered_covs = Vec::with_capacity(n);
        let mut predicted_states = Vec::with_capacity(n);
        let mut predicted_covs = Vec::with_capacity(n);
        let mut likelihoods = Vec::with_capacity(n);
        let mut innovations = Vec::with_capacity(n);
        let mut innovation_covs = Vec::with_capacity(n);
        
        // Initialize
        let mut x_filtered = self.params.initial_state.clone();
        let mut P_filtered = self.params.initial_covariance.clone();
        
        for t in 0..n {
            // Prediction step
            let x_pred = &self.params.transition_matrix * &x_filtered;
            let P_pred = &self.params.transition_matrix * &P_filtered * 
                self.params.transition_matrix.transpose() + 
                &self.params.state_covariance;
            
            // Update step
            let innovation = observations[t] - &self.params.observation_matrix * &x_pred;
            let S = &self.params.observation_matrix * &P_pred * 
                self.params.observation_matrix.transpose() + 
                &self.params.observation_covariance;
            
            let K = match S.try_inverse() {
                Some(S_inv) => {
                    &P_pred * self.params.observation_matrix.transpose() * S_inv
                },
                None => return Err(StateSpaceError::StateSpaceError(
                    "Innovation covariance matrix is singular".to_string(),
                )),
            };
            
            x_filtered = &x_pred + &K * &innovation;
            P_filtered = &P_pred - &K * &self.params.observation_matrix * &P_pred;
            
            // Compute likelihood
            let log_likelihood = self.compute_gaussian_loglikelihood(
                &innovation,
                &S,
            )?;
            
            // Store results
            filtered_states.push(x_filtered.clone());
            filtered_covs.push(P_filtered.clone());
            predicted_states.push(x_pred);
            predicted_covs.push(P_pred);
            likelihoods.push(log_likelihood);
            innovations.push(innovation);
            innovation_covs.push(S);
        }
        
        Ok(KalmanFilterResults {
            filtered_states,
            filtered_covs,
            predicted_states,
            predicted_covs,
            likelihoods,
            innovations,
            innovation_covs,
        })
    }

    /// Runs particle filter
    pub fn particle_filter(
        &self,
        observations: &[na::DVector<f64>],
    ) -> Result<ParticleFilterResults, StateSpaceError> {
        let n = observations.len();
        let state_dim = self.params.transition_matrix.nrows();
        
        // Initialize particles
        let mut particles = self.initialize_particles()?;
        let mut weights = vec![1.0 / self.n_particles as f64; self.n_particles];
        
        let mut filtered_states = Vec::with_capacity(n);
        let mut filtered_covs = Vec::with_capacity(n);
        let mut all_weights = Vec::with_capacity(n);
        let mut effective_particles = Vec::with_capacity(n);
        let mut resampled_indices = Vec::with_capacity(n);
        
        for t in 0..n {
            // Propagate particles
            particles = self.propagate_particles(&particles)?;
            
            // Update weights
            weights = self.update_weights(
                &particles,
                &observations[t],
                &weights,
            )?;
            
            // Compute effective sample size
            let ess = self.compute_effective_sample_size(&weights);
            
            // Resample if necessary
            if ess < self.resampling_threshold * self.n_particles as f64 {
                let (new_particles, indices) = self.resample_particles(&particles, &weights)?;
                particles = new_particles;
                weights = vec![1.0 / self.n_particles as f64; self.n_particles];
                resampled_indices.push(indices);
            } else {
                resampled_indices.push(Vec::new());
            }
            
            // Compute filtered state and covariance
            let (state, cov) = self.compute_particle_statistics(&particles, &weights)?;
            
            // Store results
            filtered_states.push(state);
            filtered_covs.push(cov);
            all_weights.push(weights.clone());
            effective_particles.push(ess);
        }
        
        Ok(ParticleFilterResults {
            filtered_states,
            filtered_covs,
            weights: all_weights,
            effective_particles,
            resampled_indices,
        })
    }

    /// Initializes particles
    fn initialize_particles(&self) -> Result<Vec<na::DVector<f64>>, StateSpaceError> {
        let state_dim = self.params.initial_state.len();
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| StateSpaceError::StateSpaceError(e.to_string()))?;
        
        // Generate particles from initial distribution
        let chol = match self.params.initial_covariance.cholesky() {
            Some(c) => c.l(),
            None => return Err(StateSpaceError::StateSpaceError(
                "Initial covariance matrix is not positive definite".to_string(),
            )),
        };
        
        let particles: Vec<na::DVector<f64>> = (0..self.n_particles)
            .into_par_iter()
            .map(|_| {
                let noise: Vec<f64> = (0..state_dim)
                    .map(|_| normal.sample(&mut rand::thread_rng()))
                    .collect();
                let noise = na::DVector::from_vec(noise);
                &self.params.initial_state + &chol * noise
            })
            .collect();
            
        Ok(particles)
    }

    /// Propagates particles through state transition
    fn propagate_particles(
        &self,
        particles: &[na::DVector<f64>],
    ) -> Result<Vec<na::DVector<f64>>, StateSpaceError> {
        let state_dim = self.params.transition_matrix.nrows();
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0.0, 1.0)
            .map_err(|e| StateSpaceError::StateSpaceError(e.to_string()))?;
        
        // Generate state noise
        let chol = match self.params.state_covariance.cholesky() {
            Some(c) => c.l(),
            None => return Err(StateSpaceError::StateSpaceError(
                "State covariance matrix is not positive definite".to_string(),
            )),
        };
        
        let new_particles: Vec<na::DVector<f64>> = particles
            .par_iter()
            .map(|particle| {
                let noise: Vec<f64> = (0..state_dim)
                    .map(|_| normal.sample(&mut rand::thread_rng()))
                    .collect();
                let noise = na::DVector::from_vec(noise);
                &self.params.transition_matrix * particle + &chol * noise
            })
            .collect();
            
        Ok(new_particles)
    }

    /// Updates particle weights based on observations
    fn update_weights(
        &self,
        particles: &[na::DVector<f64>],
        observation: &na::DVector<f64>,
        prev_weights: &[f64],
    ) -> Result<Vec<f64>, StateSpaceError> {
        let obs_dim = observation.len();
        
        // Compute observation likelihood for each particle
        let log_weights: Vec<f64> = particles
            .par_iter()
            .map(|particle| {
                let predicted_obs = &self.params.observation_matrix * particle;
                let innovation = observation - predicted_obs;
                
                match self.compute_gaussian_loglikelihood(
                    &innovation,
                    &self.params.observation_covariance,
                ) {
                    Ok(ll) => ll,
                    Err(_) => f64::NEG_INFINITY,
                }
            })
            .collect();
            
        // Normalize weights
        let max_log_weight = log_weights.iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
        let weights: Vec<f64> = log_weights.iter()
            .zip(prev_weights)
            .map(|(&log_w, &prev_w)| {
                prev_w * (log_w - max_log_weight).exp()
            })
            .collect();
            
        let sum_weights: f64 = weights.iter().sum();
        
        if sum_weights <= 0.0 {
            return Err(StateSpaceError::StateSpaceError(
                "All particle weights are zero".to_string(),
            ));
        }
        
        Ok(weights.iter().map(|&w| w / sum_weights).collect())
    }

    /// Resamples particles based on their weights
    fn resample_particles(
        &self,
        particles: &[na::DVector<f64>],
        weights: &[f64],
    ) -> Result<(Vec<na::DVector<f64>>, Vec<usize>), StateSpaceError> {
        let mut rng = rand::thread_rng();
        let n = particles.len();
        
        // Compute cumulative weights
        let mut cum_weights = vec![0.0; n];
        cum_weights[0] = weights[0];
        for i in 1..n {
            cum_weights[i] = cum_weights[i-1] + weights[i];
        }
        
        // Generate sorted uniform random numbers
        let mut u: Vec<f64> = (0..n)
            .map(|i| {
                let r: f64 = rng.gen();
                (i as f64 + r) / n as f64
            })
            .collect();
            
        // Find indices using binary search
        let mut indices = Vec::with_capacity(n);
        let mut new_particles = Vec::with_capacity(n);
        
        for &u_i in &u {
            match cum_weights.binary_search_by(|&w| {
                w.partial_cmp(&u_i).unwrap()
            }) {
                Ok(i) => {
                    indices.push(i);
                    new_particles.push(particles[i].clone());
                },
                Err(i) => {
                    indices.push(i);
                    new_particles.push(particles[i].clone());
                },
            }
        }
        
        Ok((new_particles, indices))
    }

    /// Computes effective sample size
    fn compute_effective_sample_size(&self, weights: &[f64]) -> f64 {
        let sum_squared: f64 = weights.iter()
            .map(|&w| w * w)
            .sum();
            
        1.0 / sum_squared
    }

    /// Computes particle statistics (mean and covariance)
    fn compute_particle_statistics(
        &self,
        particles: &[na::DVector<f64>],
        weights: &[f64],
    ) -> Result<(na::DVector<f64>, na::DMatrix<f64>), StateSpaceError> {
        let state_dim = particles[0].len();
        
        // Compute weighted mean
        let mean = particles.iter()
            .zip(weights)
            .fold(na::DVector::zeros(state_dim), |acc, (p, &w)| {
                acc + w * p
            });
            
        // Compute weighted covariance
        let cov = particles.iter()
            .zip(weights)
            .fold(na::DMatrix::zeros(state_dim, state_dim), |acc, (p, &w)| {
                let diff = p - &mean;
                acc + w * &diff * diff.transpose()
            });
            
        Ok((mean, cov))
    }

    /// Computes Gaussian log-likelihood
    fn compute_gaussian_loglikelihood(
        &self,
        x: &na::DVector<f64>,
        cov: &na::DMatrix<f64>,
    ) -> Result<f64, StateSpaceError> {
        let n = x.len();
        
        match cov.try_inverse() {
            Some(cov_inv) => {
                let det = match cov.determinant() {
                    d if d <= 0.0 => return Err(StateSpaceError::StateSpaceError(
                        "Covariance matrix is not positive definite".to_string(),
                    )),
                    d => d,
                };
                
                let quad = x.transpose() * &cov_inv * x;
                Ok(-0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + 
                    det.ln() + quad[0]))
            },
            None => Err(StateSpaceError::StateSpaceError(
                "Covariance matrix is singular".to_string(),
            )),
        }
    }
}

/// Maximum likelihood estimation for state space models
impl StateSpaceModel {
    pub fn estimate_parameters(
        &self,
        observations: &[na::DVector<f64>],
        initial_params: StateSpaceParams,
    ) -> Result<(StateSpaceParams, f64), StateSpaceError> {
        let objective = |params: &[f64]| -> f64 {
            let state_space_params = self.vector_to_params(params);
            match state_space_params {
                Ok(params) => {
                    let model = StateSpaceModel::new(
                        params,
                        self.n_particles,
                        self.resampling_threshold,
                    );
                    
                    match model {
                        Ok(model) => {
                            match model.kalman_filter(observations) {
                                Ok(results) => {
                                    -results.likelihoods.iter().sum::<f64>()
                                },
                                Err(_) => f64::INFINITY,
                            }
                        },
                        Err(_) => f64::INFINITY,
                    }
                },
                Err(_) => f64::INFINITY,
            }
        };
        
        let initial_x = self.params_to_vector(&initial_params)?;
        
        // Use Nelder-Mead optimization
        let mut optimizer = neldermead::NelderMead::new(initial_x)
            .with_sd_tolerance(1e-8)
            .with_max_iterations(1000);
            
        let result = optimizer.minimize(&objective)
            .map_err(|e| StateSpaceError::StateSpaceError(e.to_string()))?;
            
        let optimal_params = self.vector_to_params(&result)?;
        let optimal_likelihood = -objective(&result);
        
        Ok((optimal_params, optimal_likelihood))
    }

    /// Converts parameters to vector for optimization
    fn params_to_vector(
        &self,
        params: &StateSpaceParams,
    ) -> Result<Vec<f64>, StateSpaceError> {
        let mut result = Vec::new();
        
        // Flatten transition matrix
        result.extend(params.transition_matrix.as_slice());
        
        // Flatten observation matrix
        result.extend(params.observation_matrix.as_slice());
        
        // Flatten state covariance
        result.extend(params.state_covariance.as_slice());
        
        // Flatten observation covariance
        result.extend(params.observation_covariance.as_slice());
        
        // Add initial state
        result.extend(params.initial_state.as_slice());
        
        // Flatten initial covariance
        result.extend(params.initial_covariance.as_slice());
        
        Ok(result)
    }

    /// Converts vector back to parameters
    fn vector_to_params(
        &self,
        x: &[f64],
    ) -> Result<StateSpaceParams, StateSpaceError> {
        let state_dim = self.params.transition_matrix.nrows();
        let obs_dim = self.params.observation_matrix.nrows();
        
        let mut idx = 0;
        
        // Extract transition matrix
        let transition_matrix = na::DMatrix::from_row_slice(
            state_dim,
            state_dim,
            &x[idx..idx + state_dim * state_dim],
        );
        idx += state_dim * state_dim;
        
        // Extract observation matrix
        let observation_matrix = na::DMatrix::from_row_slice(
            obs_dim,
            state_dim,
            &x[idx..idx + obs_dim * state_dim],
        );
        idx += obs_dim * state_dim;
        
        // Extract state covariance
        let state_covariance = na::DMatrix::from_row_slice(
            state_dim,
            state_dim,
            &x[idx..idx + state_dim * state_dim],
        );
        idx += state_dim * state_dim;
        
        // Extract observation covariance
        let observation_covariance = na::DMatrix::from_row_slice(
            obs_dim,
            obs_dim,
            &x[idx..idx + obs_dim * obs_dim],
        );
        idx += obs_dim * obs_dim;
        
        // Extract initial state
        let initial_state = na::DVector::from_row_slice(
            &x[idx..idx + state_dim],
        );
        idx += state_dim;
        
        // Extract initial covariance
        let initial_covariance = na::DMatrix::from_row_slice(
            state_dim,
            state_dim,
            &x[idx..idx + state_dim * state_dim],
        );
        
        Ok(StateSpaceParams {
            transition_matrix,
            observation_matrix,
            state_covariance,
            observation_covariance,
            initial_state,
            initial_covariance,
        })
    }
}
