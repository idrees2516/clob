use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, StudentT};
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;

use crate::models::{
    estimators::{SpreadEstimator, RollEstimate, CSSEstimate, ModifiedEstimate},
    state_space::{StateSpaceModel, FilteringResult},
    microstructure::{MarketMicrostructure, OrderFlowAnalysis},
};

#[derive(Debug, Error)]
pub enum SimulationError {
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

pub struct MonteCarloSimulation {
    n_simulations: usize,
    sample_size: usize,
    true_spread: f64,
    serial_correlation: Vec<f64>,
    noise_std: f64,
    rng: ThreadRng,
}

impl MonteCarloSimulation {
    pub fn new(
        n_simulations: usize,
        sample_size: usize,
        true_spread: f64,
        serial_correlation: Vec<f64>,
        noise_std: f64,
    ) -> Result<Self, SimulationError> {
        if n_simulations == 0 || sample_size == 0 {
            return Err(SimulationError::ParameterError(
                "Sample size and number of simulations must be positive".to_string(),
            ));
        }

        if true_spread <= 0.0 {
            return Err(SimulationError::ParameterError(
                "True spread must be positive".to_string(),
            ));
        }

        Ok(Self {
            n_simulations,
            sample_size,
            true_spread,
            serial_correlation,
            noise_std,
            rng: thread_rng(),
        })
    }

    pub fn run_simulation(
        &mut self,
        estimator: &SpreadEstimator,
    ) -> Result<SimulationResults, SimulationError> {
        let mut roll_estimates = Vec::with_capacity(self.n_simulations);
        let mut css_estimates = Vec::with_capacity(self.n_simulations);
        let mut modified_estimates = Vec::with_capacity(self.n_simulations);

        // Run simulations in parallel
        let results: Vec<_> = (0..self.n_simulations)
            .into_par_iter()
            .map(|_| {
                let (returns, volumes, signs) = self.generate_sample()?;
                
                // Estimate spreads using different methods
                let roll = estimator.estimate_roll_spread(&returns)?;
                let css = estimator.estimate_css_spread(&returns, &volumes)?;
                let modified = estimator.estimate_modified_spread(
                    &returns,
                    &volumes,
                    &signs,
                )?;

                Ok((roll, css, modified))
            })
            .collect::<Result<Vec<_>, SimulationError>>()?;

        // Collect results
        for (roll, css, modified) in results {
            roll_estimates.push(roll);
            css_estimates.push(css);
            modified_estimates.push(modified);
        }

        // Compute simulation statistics
        let roll_stats = self.compute_estimator_statistics(&roll_estimates)?;
        let css_stats = self.compute_estimator_statistics(&css_estimates)?;
        let modified_stats = self.compute_estimator_statistics(&modified_estimates)?;

        // Analyze size and power
        let size_power = self.analyze_size_and_power(
            &roll_estimates,
            &css_estimates,
            &modified_estimates,
        )?;

        Ok(SimulationResults {
            roll_statistics: roll_stats,
            css_statistics: css_stats,
            modified_statistics: modified_stats,
            size_and_power: size_power,
            n_simulations: self.n_simulations,
            sample_size: self.sample_size,
            true_spread: self.true_spread,
        })
    }

    fn generate_sample(&mut self) -> Result<(Vec<f64>, Vec<f64>, Vec<i8>), SimulationError> {
        let mut returns = Vec::with_capacity(self.sample_size);
        let mut volumes = Vec::with_capacity(self.sample_size);
        let mut signs = Vec::with_capacity(self.sample_size);

        // Generate trade signs
        let sign_dist = rand::distributions::Bernoulli::new(0.5)
            .map_err(|e| SimulationError::SimulationError(e.to_string()))?;
        
        for _ in 0..self.sample_size {
            signs.push(if self.rng.sample(sign_dist) { 1 } else { -1 });
        }

        // Generate volumes from lognormal distribution
        let volume_dist = Normal::new(1.0, 0.5)
            .map_err(|e| SimulationError::SimulationError(e.to_string()))?;

        for _ in 0..self.sample_size {
            volumes.push(self.rng.sample(volume_dist).exp());
        }

        // Generate returns with serial correlation
        let noise_dist = Normal::new(0.0, self.noise_std)
            .map_err(|e| SimulationError::SimulationError(e.to_string()))?;

        let max_lag = self.serial_correlation.len();
        let mut innovations = Vec::with_capacity(self.sample_size);

        for _ in 0..self.sample_size {
            let noise = self.rng.sample(noise_dist);
            innovations.push(noise);
        }

        // Add serial correlation
        for t in 0..self.sample_size {
            let mut ret = innovations[t];
            
            // Add spread component
            ret += 0.5 * self.true_spread * signs[t] as f64;
            
            // Add serial correlation
            for (lag, &coef) in self.serial_correlation.iter().enumerate() {
                if t >= lag + 1 {
                    ret += coef * returns[t - lag - 1];
                }
            }

            returns.push(ret);
        }

        Ok((returns, volumes, signs))
    }

    fn compute_estimator_statistics<T>(
        &self,
        estimates: &[T],
    ) -> Result<EstimatorStatistics, SimulationError>
    where
        T: EstimatorResult,
    {
        let n = estimates.len();
        if n == 0 {
            return Err(SimulationError::SimulationError(
                "No estimates available for statistics computation".to_string(),
            ));
        }

        let spreads: Vec<f64> = estimates.iter()
            .map(|e| e.get_spread())
            .collect();

        let std_errors: Vec<f64> = estimates.iter()
            .map(|e| e.get_std_error())
            .collect();

        // Compute basic statistics
        let mean_spread = spreads.iter().sum::<f64>() / n as f64;
        let mean_std_error = std_errors.iter().sum::<f64>() / n as f64;

        let spread_variance = spreads.iter()
            .map(|&x| (x - mean_spread).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        let std_error_variance = std_errors.iter()
            .map(|&x| (x - mean_std_error).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        // Compute bias and RMSE
        let bias = mean_spread - self.true_spread;
        let rmse = (spreads.iter()
            .map(|&x| (x - self.true_spread).powi(2))
            .sum::<f64>() / n as f64)
            .sqrt();

        // Compute coverage probability
        let coverage = spreads.iter()
            .zip(std_errors.iter())
            .filter(|(&s, &se)| {
                let lower = s - 1.96 * se;
                let upper = s + 1.96 * se;
                lower <= self.true_spread && self.true_spread <= upper
            })
            .count() as f64 / n as f64;

        Ok(EstimatorStatistics {
            mean: mean_spread,
            std_dev: spread_variance.sqrt(),
            bias,
            rmse,
            mean_std_error,
            std_error_std: std_error_variance.sqrt(),
            coverage_probability: coverage,
        })
    }

    fn analyze_size_and_power(
        &self,
        roll_estimates: &[RollEstimate],
        css_estimates: &[CSSEstimate],
        modified_estimates: &[ModifiedEstimate],
    ) -> Result<SizeAndPower, SimulationError> {
        let alpha = 0.05; // Significance level
        
        // Compute rejection rates under null (size) and alternative (power)
        let roll_size = self.compute_rejection_rate(
            roll_estimates,
            self.true_spread,
            alpha,
        )?;

        let css_size = self.compute_rejection_rate(
            css_estimates,
            self.true_spread,
            alpha,
        )?;

        let modified_size = self.compute_rejection_rate(
            modified_estimates,
            self.true_spread,
            alpha,
        )?;

        // Compute power against local alternatives
        let alternatives = vec![0.8, 0.9, 1.1, 1.2];
        let mut roll_power = HashMap::new();
        let mut css_power = HashMap::new();
        let mut modified_power = HashMap::new();

        for alt in alternatives {
            let alt_spread = self.true_spread * alt;
            
            roll_power.insert(alt, self.compute_rejection_rate(
                roll_estimates,
                alt_spread,
                alpha,
            )?);

            css_power.insert(alt, self.compute_rejection_rate(
                css_estimates,
                alt_spread,
                alpha,
            )?);

            modified_power.insert(alt, self.compute_rejection_rate(
                modified_estimates,
                alt_spread,
                alpha,
            )?);
        }

        Ok(SizeAndPower {
            roll_size,
            css_size,
            modified_size,
            roll_power,
            css_power,
            modified_power,
        })
    }

    fn compute_rejection_rate<T>(
        &self,
        estimates: &[T],
        true_value: f64,
        alpha: f64,
    ) -> Result<f64, SimulationError>
    where
        T: EstimatorResult,
    {
        let n = estimates.len();
        if n == 0 {
            return Err(SimulationError::SimulationError(
                "No estimates available for rejection rate computation".to_string(),
            ));
        }

        let rejections = estimates.iter()
            .filter(|&e| {
                let spread = e.get_spread();
                let std_error = e.get_std_error();
                let t_stat = (spread - true_value) / std_error;
                t_stat.abs() > 1.96 // Using normal approximation
            })
            .count();

        Ok(rejections as f64 / n as f64)
    }
}

pub trait EstimatorResult {
    fn get_spread(&self) -> f64;
    fn get_std_error(&self) -> f64;
}

impl EstimatorResult for RollEstimate {
    fn get_spread(&self) -> f64 {
        self.spread
    }

    fn get_std_error(&self) -> f64 {
        self.std_error
    }
}

impl EstimatorResult for CSSEstimate {
    fn get_spread(&self) -> f64 {
        self.spread
    }

    fn get_std_error(&self) -> f64 {
        self.std_error
    }
}

impl EstimatorResult for ModifiedEstimate {
    fn get_spread(&self) -> f64 {
        self.spread
    }

    fn get_std_error(&self) -> f64 {
        self.std_error
    }
}

#[derive(Debug)]
pub struct EstimatorStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub bias: f64,
    pub rmse: f64,
    pub mean_std_error: f64,
    pub std_error_std: f64,
    pub coverage_probability: f64,
}

#[derive(Debug)]
pub struct SizeAndPower {
    pub roll_size: f64,
    pub css_size: f64,
    pub modified_size: f64,
    pub roll_power: HashMap<f64, f64>,
    pub css_power: HashMap<f64, f64>,
    pub modified_power: HashMap<f64, f64>,
}

#[derive(Debug)]
pub struct SimulationResults {
    pub roll_statistics: EstimatorStatistics,
    pub css_statistics: EstimatorStatistics,
    pub modified_statistics: EstimatorStatistics,
    pub size_and_power: SizeAndPower,
    pub n_simulations: usize,
    pub sample_size: usize,
    pub true_spread: f64,
}
