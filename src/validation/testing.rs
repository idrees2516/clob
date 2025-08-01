use super::ValidationError;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use statrs::distribution::{StudentsT, FisherSnedecor, ChiSquare, ContinuousCDF};
use std::collections::HashMap;

pub struct TestingFramework {
    rng: StdRng,
    n_simulations: usize,
    confidence_level: f64,
}

impl TestingFramework {
    pub fn new(
        seed: Option<u64>,
        n_simulations: usize,
        confidence_level: f64,
    ) -> Result<Self, ValidationError> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(ValidationError::ValidationError(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Ok(Self {
            rng,
            n_simulations,
            confidence_level,
        })
    }

    pub fn generate_test_data(
        &mut self,
        n_obs: usize,
        params: &SimulationParams,
    ) -> Result<TestData, ValidationError> {
        match params {
            SimulationParams::RandomWalk { volatility } => {
                self.generate_random_walk(n_obs, *volatility)
            },
            SimulationParams::ARMA { ar_params, ma_params, innovation_std } => {
                self.generate_arma(n_obs, ar_params, ma_params, *innovation_std)
            },
            SimulationParams::LongMemory { d, innovation_std } => {
                self.generate_long_memory(n_obs, *d, *innovation_std)
            },
            SimulationParams::StochasticVolatility { persistence, volatility_of_vol } => {
                self.generate_stochastic_volatility(n_obs, *persistence, *volatility_of_vol)
            },
        }
    }

    pub fn run_monte_carlo<F>(
        &mut self,
        test_fn: F,
        params: &SimulationParams,
        n_obs: usize,
    ) -> Result<MonteCarloResults, ValidationError>
    where
        F: Fn(&[f64]) -> Result<f64, ValidationError> + Send + Sync,
    {
        let results: Vec<Result<f64, ValidationError>> = (0..self.n_simulations)
            .into_par_iter()
            .map(|_| {
                let data = self.generate_test_data(n_obs, params)?;
                test_fn(&data.prices)
            })
            .collect();

        // Process results
        let mut valid_results = Vec::new();
        let mut errors = Vec::new();

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(value) => valid_results.push(value),
                Err(e) => errors.push((i, e)),
            }
        }

        if valid_results.is_empty() {
            return Err(ValidationError::ValidationError(
                "All simulations failed".to_string(),
            ));
        }

        // Compute statistics
        let mean = valid_results.iter().sum::<f64>() / valid_results.len() as f64;
        
        let variance = valid_results.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (valid_results.len() - 1) as f64;
        
        let std_error = (variance / valid_results.len() as f64).sqrt();

        // Compute confidence intervals
        let t_dist = StudentsT::new((valid_results.len() - 1) as f64)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;
        
        let t_value = t_dist.inverse_cdf(1.0 - (1.0 - self.confidence_level) / 2.0);
        let margin = t_value * std_error;

        Ok(MonteCarloResults {
            mean,
            std_error,
            confidence_interval: (mean - margin, mean + margin),
            n_success: valid_results.len(),
            n_failures: errors.len(),
            errors,
        })
    }

    fn generate_random_walk(
        &mut self,
        n_obs: usize,
        volatility: f64,
    ) -> Result<TestData, ValidationError> {
        let normal = Normal::new(0.0, volatility)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;

        let mut prices = Vec::with_capacity(n_obs);
        let mut returns = Vec::with_capacity(n_obs - 1);
        
        prices.push(100.0); // Initial price
        
        for _ in 1..n_obs {
            let return_t = normal.sample(&mut self.rng);
            let price_t = prices.last().unwrap() * (1.0 + return_t);
            
            returns.push(return_t);
            prices.push(price_t);
        }

        Ok(TestData { prices, returns })
    }

    fn generate_arma(
        &mut self,
        n_obs: usize,
        ar_params: &[f64],
        ma_params: &[f64],
        innovation_std: f64,
    ) -> Result<TestData, ValidationError> {
        let normal = Normal::new(0.0, innovation_std)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;

        let p = ar_params.len();
        let q = ma_params.len();
        let max_lag = p.max(q);

        let mut returns = vec![0.0; n_obs];
        let mut innovations = vec![0.0; n_obs];
        
        // Generate initial values
        for i in 0..max_lag {
            innovations[i] = normal.sample(&mut self.rng);
            returns[i] = innovations[i];
        }

        // Generate remaining values
        for t in max_lag..n_obs {
            let mut value = 0.0;
            
            // AR component
            for (i, &phi) in ar_params.iter().enumerate() {
                value += phi * returns[t - i - 1];
            }
            
            // MA component
            innovations[t] = normal.sample(&mut self.rng);
            for (i, &theta) in ma_params.iter().enumerate() {
                value += theta * innovations[t - i - 1];
            }
            
            value += innovations[t];
            returns[t] = value;
        }

        // Convert returns to prices
        let mut prices = Vec::with_capacity(n_obs);
        prices.push(100.0); // Initial price
        
        for &ret in &returns[1..] {
            let price_t = prices.last().unwrap() * (1.0 + ret);
            prices.push(price_t);
        }

        Ok(TestData { prices, returns })
    }

    fn generate_long_memory(
        &mut self,
        n_obs: usize,
        d: f64,
        innovation_std: f64,
    ) -> Result<TestData, ValidationError> {
        let normal = Normal::new(0.0, innovation_std)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;

        // Generate fractional differencing weights
        let max_lag = (n_obs as f64).sqrt() as usize;
        let mut weights = vec![1.0];
        let mut prev_weight = 1.0;
        
        for k in 1..=max_lag {
            let weight = prev_weight * (d - k as f64 + 1.0) / k as f64;
            weights.push(weight);
            prev_weight = weight;
        }

        // Generate returns using convolution
        let mut returns = vec![0.0; n_obs];
        let mut innovations = Vec::with_capacity(n_obs);
        
        for t in 0..n_obs {
            innovations.push(normal.sample(&mut self.rng));
            
            let mut value = 0.0;
            for (k, &weight) in weights.iter().enumerate() {
                if t >= k {
                    value += weight * innovations[t - k];
                }
            }
            returns[t] = value;
        }

        // Convert returns to prices
        let mut prices = Vec::with_capacity(n_obs);
        prices.push(100.0); // Initial price
        
        for &ret in &returns[1..] {
            let price_t = prices.last().unwrap() * (1.0 + ret);
            prices.push(price_t);
        }

        Ok(TestData { prices, returns })
    }

    fn generate_stochastic_volatility(
        &mut self,
        n_obs: usize,
        persistence: f64,
        volatility_of_vol: f64,
    ) -> Result<TestData, ValidationError> {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| ValidationError::StatisticalError(e.to_string()))?;

        let mut returns = Vec::with_capacity(n_obs);
        let mut log_variance = 0.0; // Initial log-variance
        
        for _ in 0..n_obs {
            // Update log-variance
            let vol_innovation = normal.sample(&mut self.rng) * volatility_of_vol;
            log_variance = persistence * log_variance + vol_innovation;
            
            // Generate return
            let volatility = log_variance.exp().sqrt();
            let return_t = normal.sample(&mut self.rng) * volatility;
            returns.push(return_t);
        }

        // Convert returns to prices
        let mut prices = Vec::with_capacity(n_obs);
        prices.push(100.0); // Initial price
        
        for &ret in &returns {
            let price_t = prices.last().unwrap() * (1.0 + ret);
            prices.push(price_t);
        }

        Ok(TestData { prices, returns })
    }

    pub fn compute_size_power<F>(
        &mut self,
        test_fn: F,
        params: &SimulationParams,
        n_obs: usize,
        significance_level: f64,
    ) -> Result<SizePowerAnalysis, ValidationError>
    where
        F: Fn(&[f64]) -> Result<(f64, f64), ValidationError> + Send + Sync,
    {
        let results: Vec<Result<(f64, f64), ValidationError>> = (0..self.n_simulations)
            .into_par_iter()
            .map(|_| {
                let data = self.generate_test_data(n_obs, params)?;
                test_fn(&data.prices)
            })
            .collect();

        let mut test_statistics = Vec::new();
        let mut p_values = Vec::new();
        let mut errors = Vec::new();

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok((stat, p_val)) => {
                    test_statistics.push(stat);
                    p_values.push(p_val);
                },
                Err(e) => errors.push((i, e)),
            }
        }

        if test_statistics.is_empty() {
            return Err(ValidationError::ValidationError(
                "All simulations failed".to_string(),
            ));
        }

        // Compute size/power
        let rejections = p_values.iter()
            .filter(|&&p| p < significance_level)
            .count();
        
        let rejection_rate = rejections as f64 / p_values.len() as f64;

        // Compute standard error of rejection rate
        let std_error = (rejection_rate * (1.0 - rejection_rate) / p_values.len() as f64).sqrt();

        Ok(SizePowerAnalysis {
            rejection_rate,
            std_error,
            n_success: test_statistics.len(),
            n_failures: errors.len(),
            errors,
        })
    }
}

#[derive(Debug)]
pub enum SimulationParams {
    RandomWalk {
        volatility: f64,
    },
    ARMA {
        ar_params: Vec<f64>,
        ma_params: Vec<f64>,
        innovation_std: f64,
    },
    LongMemory {
        d: f64,
        innovation_std: f64,
    },
    StochasticVolatility {
        persistence: f64,
        volatility_of_vol: f64,
    },
}

#[derive(Debug)]
pub struct TestData {
    pub prices: Vec<f64>,
    pub returns: Vec<f64>,
}

#[derive(Debug)]
pub struct MonteCarloResults {
    pub mean: f64,
    pub std_error: f64,
    pub confidence_interval: (f64, f64),
    pub n_success: usize,
    pub n_failures: usize,
    pub errors: Vec<(usize, ValidationError)>,
}

#[derive(Debug)]
pub struct SizePowerAnalysis {
    pub rejection_rate: f64,
    pub std_error: f64,
    pub n_success: usize,
    pub n_failures: usize,
    pub errors: Vec<(usize, ValidationError)>,
}
