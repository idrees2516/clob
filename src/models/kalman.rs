use nalgebra as na;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum KalmanError {
    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Matrix inversion error: {0}")]
    InversionError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub struct KalmanFilter {
    // State space dimensions
    state_dim: usize,
    obs_dim: usize,

    // System matrices
    transition_matrix: na::DMatrix<f64>,    // F
    observation_matrix: na::DMatrix<f64>,   // H
    process_noise_cov: na::DMatrix<f64>,    // Q
    measurement_noise_cov: na::DMatrix<f64>, // R

    // State estimate and covariance
    state_estimate: na::DVector<f64>,       // x
    error_covariance: na::DMatrix<f64>,     // P
}

impl KalmanFilter {
    pub fn new(
        state_dim: usize,
        obs_dim: usize,
        transition_matrix: na::DMatrix<f64>,
        observation_matrix: na::DMatrix<f64>,
        process_noise_cov: na::DMatrix<f64>,
        measurement_noise_cov: na::DMatrix<f64>,
        initial_state: na::DVector<f64>,
        initial_covariance: na::DMatrix<f64>,
    ) -> Result<Self, KalmanError> {
        // Validate dimensions
        if transition_matrix.nrows() != state_dim || transition_matrix.ncols() != state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid transition matrix dimensions".to_string(),
            ));
        }
        if observation_matrix.nrows() != obs_dim || observation_matrix.ncols() != state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid observation matrix dimensions".to_string(),
            ));
        }
        if process_noise_cov.nrows() != state_dim || process_noise_cov.ncols() != state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid process noise covariance dimensions".to_string(),
            ));
        }
        if measurement_noise_cov.nrows() != obs_dim || measurement_noise_cov.ncols() != obs_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid measurement noise covariance dimensions".to_string(),
            ));
        }
        if initial_state.nrows() != state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid initial state dimensions".to_string(),
            ));
        }
        if initial_covariance.nrows() != state_dim || initial_covariance.ncols() != state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid initial covariance dimensions".to_string(),
            ));
        }

        Ok(Self {
            state_dim,
            obs_dim,
            transition_matrix,
            observation_matrix,
            process_noise_cov,
            measurement_noise_cov,
            state_estimate: initial_state,
            error_covariance: initial_covariance,
        })
    }

    pub fn predict(&mut self) -> Result<(), KalmanError> {
        // Predict state: x = Fx
        self.state_estimate = &self.transition_matrix * &self.state_estimate;

        // Predict error covariance: P = FPF' + Q
        self.error_covariance = &self.transition_matrix * &self.error_covariance 
            * self.transition_matrix.transpose()
            + &self.process_noise_cov;

        Ok(())
    }

    pub fn update(&mut self, measurement: &na::DVector<f64>) -> Result<(), KalmanError> {
        if measurement.nrows() != self.obs_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid measurement dimension".to_string(),
            ));
        }

        // Innovation: y = z - Hx
        let innovation = measurement - &self.observation_matrix * &self.state_estimate;

        // Innovation covariance: S = HPH' + R
        let innovation_covariance = &self.observation_matrix * &self.error_covariance 
            * self.observation_matrix.transpose()
            + &self.measurement_noise_cov;

        // Kalman gain: K = PH'S^(-1)
        let kalman_gain = match innovation_covariance.try_inverse() {
            Some(inv) => &self.error_covariance * self.observation_matrix.transpose() * inv,
            None => return Err(KalmanError::InversionError(
                "Failed to invert innovation covariance".to_string(),
            )),
        };

        // Update state estimate: x = x + Ky
        self.state_estimate += &kalman_gain * innovation;

        // Update error covariance: P = (I - KH)P
        let identity = na::DMatrix::<f64>::identity(self.state_dim, self.state_dim);
        self.error_covariance = (&identity - &kalman_gain * &self.observation_matrix) 
            * &self.error_covariance;

        Ok(())
    }

    pub fn smooth(&self, measurements: &[na::DVector<f64>]) -> Result<Vec<na::DVector<f64>>, KalmanError> {
        let n = measurements.len();
        let mut smoothed_states = Vec::with_capacity(n);
        let mut forward_states = Vec::with_capacity(n);
        let mut forward_covs = Vec::with_capacity(n);

        // Forward pass
        let mut state = self.state_estimate.clone();
        let mut cov = self.error_covariance.clone();
        
        for measurement in measurements {
            // Predict
            state = &self.transition_matrix * &state;
            cov = &self.transition_matrix * &cov * self.transition_matrix.transpose() 
                + &self.process_noise_cov;
            
            // Store predicted state and covariance
            forward_states.push(state.clone());
            forward_covs.push(cov.clone());
            
            // Update
            let innovation = measurement - &self.observation_matrix * &state;
            let innovation_cov = &self.observation_matrix * &cov * self.observation_matrix.transpose() 
                + &self.measurement_noise_cov;
            
            let kalman_gain = match innovation_cov.try_inverse() {
                Some(inv) => &cov * self.observation_matrix.transpose() * inv,
                None => return Err(KalmanError::InversionError(
                    "Failed to invert innovation covariance during smoothing".to_string(),
                )),
            };
            
            state += &kalman_gain * innovation;
            cov = &cov - &kalman_gain * &self.observation_matrix * &cov;
        }

        // Backward pass (RTS smoother)
        let mut smoothed_state = state;
        let mut smoothed_cov = cov;
        smoothed_states.push(smoothed_state.clone());

        for t in (0..n-1).rev() {
            let forward_state = &forward_states[t];
            let forward_cov = &forward_covs[t];
            
            let predicted_state = &self.transition_matrix * forward_state;
            let predicted_cov = &self.transition_matrix * forward_cov * self.transition_matrix.transpose() 
                + &self.process_noise_cov;
            
            let smoother_gain = match predicted_cov.try_inverse() {
                Some(inv) => forward_cov * self.transition_matrix.transpose() * inv,
                None => return Err(KalmanError::InversionError(
                    "Failed to invert predicted covariance during smoothing".to_string(),
                )),
            };
            
            smoothed_state = forward_state + &smoother_gain * (&smoothed_state - &predicted_state);
            smoothed_cov = forward_cov + &smoother_gain * (&smoothed_cov - &predicted_cov) * smoother_gain.transpose();
            
            smoothed_states.push(smoothed_state.clone());
        }

        smoothed_states.reverse();
        Ok(smoothed_states)
    }

    pub fn get_state_estimate(&self) -> &na::DVector<f64> {
        &self.state_estimate
    }

    pub fn get_error_covariance(&self) -> &na::DMatrix<f64> {
        &self.error_covariance
    }

    pub fn set_transition_matrix(&mut self, matrix: na::DMatrix<f64>) -> Result<(), KalmanError> {
        if matrix.nrows() != self.state_dim || matrix.ncols() != self.state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid transition matrix dimensions".to_string(),
            ));
        }
        self.transition_matrix = matrix;
        Ok(())
    }

    pub fn set_process_noise(&mut self, matrix: na::DMatrix<f64>) -> Result<(), KalmanError> {
        if matrix.nrows() != self.state_dim || matrix.ncols() != self.state_dim {
            return Err(KalmanError::DimensionMismatch(
                "Invalid process noise dimensions".to_string(),
            ));
        }
        self.process_noise_cov = matrix;
        Ok(())
    }
}
