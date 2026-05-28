use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::f64::consts::PI;

#[derive(Debug, Error)]
pub enum LongMemoryError {
    #[error("Estimation error: {0}")]
    EstimationError(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub struct FractionalIntegration {
    d: f64,
    ar_params: Vec<f64>,
    ma_params: Vec<f64>,
    max_lag: usize,
}

impl FractionalIntegration {
    pub fn new(d: f64, ar_params: Vec<f64>, ma_params: Vec<f64>, max_lag: usize) -> Result<Self, LongMemoryError> {
        if d <= -0.5 || d >= 0.5 {
            return Err(LongMemoryError::InvalidParameter(
                "Fractional integration parameter d must be in (-0.5, 0.5)".to_string(),
            ));
        }

        Ok(Self {
            d,
            ar_params,
            ma_params,
            max_lag,
        })
    }

    pub fn estimate_d(&self, data: &[f64]) -> Result<f64, LongMemoryError> {
        let periodogram = self.compute_periodogram(data);
        self.whittle_estimation(&periodogram)
    }

    fn compute_periodogram(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let frequencies: Vec<f64> = (1..=n/2)
            .map(|j| 2.0 * PI * j as f64 / n as f64)
            .collect();

        frequencies.par_iter()
            .map(|&freq| {
                let mut sum_cos = 0.0;
                let mut sum_sin = 0.0;
                for (t, &x) in data.iter().enumerate() {
                    let t_f = t as f64;
                    sum_cos += x * (freq * t_f).cos();
                    sum_sin += x * (freq * t_f).sin();
                }
                (sum_cos.powi(2) + sum_sin.powi(2)) / (2.0 * PI * n as f64)
            })
            .collect()
    }

    fn whittle_estimation(&self, periodogram: &[f64]) -> Result<f64, LongMemoryError> {
        let n = periodogram.len();
        let frequencies: Vec<f64> = (1..=n)
            .map(|j| 2.0 * PI * j as f64 / (2 * n) as f64)
            .collect();

        // Grid search for initial estimate
        let d_grid: Vec<f64> = (-0.4..=0.4).step_by(20)
            .map(|x| x as f64 / 100.0)
            .collect();

        let mut min_obj = f64::INFINITY;
        let mut best_d = 0.0;

        for &d in &d_grid {
            let obj = frequencies.iter()
                .zip(periodogram.iter())
                .map(|(&freq, &I)| {
                    let f = self.spectral_density(freq, d);
                    I / f + f.ln()
                })
                .sum::<f64>();

            if obj < min_obj {
                min_obj = obj;
                best_d = d;
            }
        }

        // Fine-tune using Newton-Raphson
        let mut d = best_d;
        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let (gradient, hessian) = self.compute_derivatives(&frequencies, periodogram, d);
            let delta = gradient / hessian;
            d -= delta;

            if delta.abs() < tolerance {
                break;
            }
        }

        if d <= -0.5 || d >= 0.5 {
            return Err(LongMemoryError::EstimationError(
                "Estimated d is outside valid range".to_string(),
            ));
        }

        Ok(d)
    }

    fn spectral_density(&self, freq: f64, d: f64) -> f64 {
        let factor = 2.0 * (1.0 - (freq).cos());
        let arma = self.compute_arma_spectrum(freq);
        factor.powf(-d) * arma
    }

    fn compute_arma_spectrum(&self, freq: f64) -> f64 {
        let ar_term: f64 = self.ar_params.iter()
            .enumerate()
            .map(|(p, &phi)| phi * (freq * (p + 1) as f64).cos())
            .sum::<f64>();
        
        let ma_term: f64 = self.ma_params.iter()
            .enumerate()
            .map(|(q, &theta)| theta * (freq * (q + 1) as f64).cos())
            .sum::<f64>();

        (1.0 + ma_term).powi(2) / (1.0 + ar_term).powi(2)
    }

    fn compute_derivatives(
        &self,
        frequencies: &[f64],
        periodogram: &[f64],
        d: f64,
    ) -> (f64, f64) {
        let mut gradient = 0.0;
        let mut hessian = 0.0;

        for (&freq, &I) in frequencies.iter().zip(periodogram.iter()) {
            let f = self.spectral_density(freq, d);
            let log_factor = 2.0 * (1.0 - freq.cos()).ln();
            
            gradient += -log_factor * (I / f - 1.0);
            hessian += log_factor.powi(2) * I / f;
        }

        (gradient, hessian)
    }

    pub fn fractional_difference(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];
        
        for t in 0..n {
            let mut sum = data[t];
            for k in 1..=t.min(self.max_lag) {
                let coef = self.binomial_coefficient(-self.d, k);
                sum += coef * data[t - k];
            }
            result[t] = sum;
        }
        
        result
    }

    fn binomial_coefficient(&self, alpha: f64, k: usize) -> f64 {
        let mut coef = 1.0;
        for i in 0..k {
            coef *= (alpha - i as f64) / (i + 1) as f64;
        }
        coef
    }
}
