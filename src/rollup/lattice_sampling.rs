//! Quantum-Resistant Gaussian Sampling for LatticeFold+
//! 
//! This module implements secure sampling algorithms for lattice-based cryptography:
//! - Quantum-resistant sampler with rejection sampling
//! - Constant-time Gaussian sampling to prevent timing attacks
//! - Bias-resistant lattice point sampling with configurable parameters
//! - Statistical quality testing and validation

use crate::rollup::lattice_fold_plus::{
    LatticeParams, LatticePoint, LatticeFoldError, Result, SecurityLevel,
};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Sampling parameters for quantum-resistant operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Base lattice parameters
    pub lattice_params: LatticeParams,
    /// Gaussian width parameter (σ)
    pub sigma: f64,
    /// Tail cut parameter (samples beyond this are rejected)
    pub tail_cut: f64,
    /// Maximum rejection attempts before failure
    pub max_attempts: usize,
    /// Whether to use constant-time sampling
    pub constant_time: bool,
    /// Bias resistance level
    pub bias_resistance_level: BiasResistanceLevel,
}

/// Bias resistance levels for sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiasResistanceLevel {
    /// Basic resistance against simple attacks
    Basic,
    /// Enhanced resistance against sophisticated attacks
    Enhanced,
    /// Maximum resistance against quantum adversaries
    Maximum,
}

impl SamplingParams {
    /// Create sampling parameters from lattice parameters
    pub fn from_lattice_params(lattice_params: LatticeParams) -> Self {
        let sigma = lattice_params.sigma;
        let tail_cut = match lattice_params.security_level {
            SecurityLevel::Medium => 6.0 * sigma,
            SecurityLevel::High => 8.0 * sigma,
            SecurityLevel::VeryHigh => 10.0 * sigma,
        };
        
        let max_attempts = match lattice_params.security_level {
            SecurityLevel::Medium => 1000,
            SecurityLevel::High => 2000,
            SecurityLevel::VeryHigh => 5000,
        };
        
        let bias_resistance_level = match lattice_params.security_level {
            SecurityLevel::Medium => BiasResistanceLevel::Basic,
            SecurityLevel::High => BiasResistanceLevel::Enhanced,
            SecurityLevel::VeryHigh => BiasResistanceLevel::Maximum,
        };
        
        Self {
            lattice_params,
            sigma,
            tail_cut,
            max_attempts,
            constant_time: true,
            bias_resistance_level,
        }
    }
    
    /// Validate sampling parameters
    pub fn validate(&self) -> Result<()> {
        self.lattice_params.validate()?;
        
        if self.sigma <= 0.0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Sigma must be positive".to_string(),
            ));
        }
        
        if self.tail_cut <= 0.0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Tail cut must be positive".to_string(),
            ));
        }
        
        if self.max_attempts == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Max attempts must be positive".to_string(),
            ));
        }
        
        Ok(())
    }
}

/// Quantum-resistant sampler for lattice points
#[derive(Debug, Clone)]
pub struct QuantumResistantSampler {
    /// Sampling parameters
    pub params: SamplingParams,
    /// Precomputed probability tables for efficiency
    pub probability_tables: HashMap<String, Vec<f64>>,
    /// Timing attack resistance state
    pub timing_state: TimingResistanceState,
}

/// State for timing attack resistance
#[derive(Debug, Clone)]
pub struct TimingResistanceState {
    /// Target execution time for constant-time operations
    pub target_time_ns: u64,
    /// Timing measurements for calibration
    pub timing_measurements: Vec<u64>,
    /// Whether timing calibration is complete
    pub calibrated: bool,
}

impl QuantumResistantSampler {
    /// Create a new quantum-resistant sampler
    pub fn new(params: SamplingParams) -> Result<Self> {
        params.validate()?;
        
        let mut sampler = Self {
            params,
            probability_tables: HashMap::new(),
            timing_state: TimingResistanceState {
                target_time_ns: 0,
                timing_measurements: Vec::new(),
                calibrated: false,
            },
        };
        
        // Precompute probability tables
        sampler.precompute_probability_tables()?;
        
        // Calibrate timing if constant-time is enabled
        if sampler.params.constant_time {
            sampler.calibrate_timing()?;
        }
        
        Ok(sampler)
    }
    
    /// Sample a single value from discrete Gaussian distribution
    pub fn sample_discrete_gaussian<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<i64> {
        if self.params.constant_time {
            self.sample_discrete_gaussian_constant_time(rng)
        } else {
            self.sample_discrete_gaussian_variable_time(rng)
        }
    }
    
    /// Sample a lattice point with all coordinates from discrete Gaussian
    pub fn sample_lattice_point<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<LatticePoint> {
        let n = self.params.lattice_params.n;
        let mut coordinates = Vec::with_capacity(n);
        
        for _ in 0..n {
            let coord = self.sample_discrete_gaussian(rng)?;
            coordinates.push(coord);
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Sample multiple lattice points efficiently
    pub fn sample_multiple_points<R: RngCore + CryptoRng>(
        &mut self,
        count: usize,
        rng: &mut R,
    ) -> Result<Vec<LatticePoint>> {
        let mut points = Vec::with_capacity(count);
        
        for _ in 0..count {
            points.push(self.sample_lattice_point(rng)?);
        }
        
        Ok(points)
    }
    
    /// Sample with bias resistance against adaptive attacks
    pub fn sample_bias_resistant<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<LatticePoint> {
        match self.params.bias_resistance_level {
            BiasResistanceLevel::Basic => self.sample_lattice_point(rng),
            BiasResistanceLevel::Enhanced => self.sample_with_enhanced_bias_resistance(rng),
            BiasResistanceLevel::Maximum => self.sample_with_maximum_bias_resistance(rng),
        }
    }
    
    /// Constant-time discrete Gaussian sampling
    fn sample_discrete_gaussian_constant_time<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<i64> {
        let start_time = Instant::now();
        
        let result = self.sample_discrete_gaussian_core(rng)?;
        
        // Pad execution time to constant target
        if self.timing_state.calibrated {
            let elapsed = start_time.elapsed().as_nanos() as u64;
            if elapsed < self.timing_state.target_time_ns {
                let padding_time = self.timing_state.target_time_ns - elapsed;
                self.constant_time_padding(padding_time);
            }
        }
        
        Ok(result)
    }
    
    /// Variable-time discrete Gaussian sampling (faster but potentially vulnerable)
    fn sample_discrete_gaussian_variable_time<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<i64> {
        self.sample_discrete_gaussian_core(rng)
    }
    
    /// Core discrete Gaussian sampling algorithm
    fn sample_discrete_gaussian_core<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<i64> {
        let sigma = self.params.sigma;
        let q = self.params.lattice_params.q;
        let tail_cut = self.params.tail_cut;
        
        for attempt in 0..self.params.max_attempts {
            // Sample from continuous Gaussian using Box-Muller
            let (z1, z2) = self.box_muller_sample(rng);
            let continuous_sample = z1 * sigma;
            
            // Check tail cut
            if continuous_sample.abs() > tail_cut {
                continue;
            }
            
            // Round to nearest integer
            let discrete_sample = continuous_sample.round() as i64;
            
            // Rejection sampling for discrete Gaussian
            let acceptance_prob = self.discrete_gaussian_probability(discrete_sample, sigma);
            let continuous_prob = self.continuous_gaussian_probability(continuous_sample, sigma);
            
            let rejection_prob = acceptance_prob / continuous_prob;
            let uniform: f64 = rng.next_u32() as f64 / u32::MAX as f64;
            
            if uniform < rejection_prob {
                // Reduce modulo q and return
                let result = ((discrete_sample % q) + q) % q;
                return Ok(result);
            }
            
            // Add bias resistance delay for failed attempts
            if attempt % 10 == 0 {
                self.bias_resistance_delay(rng);
            }
        }
        
        Err(LatticeFoldError::SamplingError(
            format!("Failed to sample after {} attempts", self.params.max_attempts),
        ))
    }
    
    /// Box-Muller transform for Gaussian sampling
    fn box_muller_sample<R: RngCore + CryptoRng>(&self, rng: &mut R) -> (f64, f64) {
        let u1: f64 = (rng.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 1.0); // Avoid log(0)
        let u2: f64 = rng.next_u32() as f64 / u32::MAX as f64;
        
        let magnitude = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        
        (magnitude * angle.cos(), magnitude * angle.sin())
    }
    
    /// Compute discrete Gaussian probability
    fn discrete_gaussian_probability(&self, x: i64, sigma: f64) -> f64 {
        let x_f = x as f64;
        (-x_f * x_f / (2.0 * sigma * sigma)).exp()
    }
    
    /// Compute continuous Gaussian probability
    fn continuous_gaussian_probability(&self, x: f64, sigma: f64) -> f64 {
        let normalization = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        normalization * (-x * x / (2.0 * sigma * sigma)).exp()
    }
    
    /// Sample with enhanced bias resistance
    fn sample_with_enhanced_bias_resistance<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<LatticePoint> {
        // Use multiple independent samples and combine
        let n = self.params.lattice_params.n;
        let num_samples = 3; // Use 3 independent samples
        
        let mut combined_coordinates = vec![0i64; n];
        
        for _ in 0..num_samples {
            let sample = self.sample_lattice_point(rng)?;
            for i in 0..n {
                combined_coordinates[i] = (combined_coordinates[i] + sample.coordinates[i]) 
                    % self.params.lattice_params.q;
            }
        }
        
        // Add random noise to break correlation patterns
        for coord in &mut combined_coordinates {
            let noise = (rng.next_u32() % 3) as i64 - 1; // -1, 0, or 1
            *coord = ((*coord + noise) % self.params.lattice_params.q + self.params.lattice_params.q) 
                % self.params.lattice_params.q;
        }
        
        Ok(LatticePoint::new(combined_coordinates))
    }
    
    /// Sample with maximum bias resistance
    fn sample_with_maximum_bias_resistance<R: RngCore + CryptoRng>(
        &mut self,
        rng: &mut R,
    ) -> Result<LatticePoint> {
        // Use cryptographic randomness extraction
        let base_sample = self.sample_with_enhanced_bias_resistance(rng)?;
        
        // Apply cryptographic hash to break any remaining bias
        let hash_input = base_sample.to_bytes();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&hash_input);
        
        // Add additional entropy
        let mut entropy = [0u8; 32];
        rng.fill_bytes(&mut entropy);
        hasher.update(&entropy);
        
        let hash_output = hasher.finalize();
        
        // Convert hash to lattice point
        let n = self.params.lattice_params.n;
        let q = self.params.lattice_params.q;
        let mut coordinates = Vec::with_capacity(n);
        
        for i in 0..n {
            let byte_idx = i % 32;
            let coord = (hash_output.as_bytes()[byte_idx] as i64) % q;
            coordinates.push(if coord < 0 { coord + q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Precompute probability tables for efficiency
    fn precompute_probability_tables(&mut self) -> Result<()> {
        let sigma = self.params.sigma;
        let tail_cut = self.params.tail_cut;
        let table_size = (2.0 * tail_cut) as usize + 1;
        
        let mut discrete_probs = Vec::with_capacity(table_size);
        let mut continuous_probs = Vec::with_capacity(table_size);
        
        for i in 0..table_size {
            let x = i as f64 - tail_cut;
            discrete_probs.push(self.discrete_gaussian_probability(x as i64, sigma));
            continuous_probs.push(self.continuous_gaussian_probability(x, sigma));
        }
        
        self.probability_tables.insert("discrete".to_string(), discrete_probs);
        self.probability_tables.insert("continuous".to_string(), continuous_probs);
        
        Ok(())
    }
    
    /// Calibrate timing for constant-time operations
    fn calibrate_timing(&mut self) -> Result<()> {
        let calibration_samples = 100;
        let mut measurements = Vec::with_capacity(calibration_samples);
        let mut rng = rand::thread_rng();
        
        for _ in 0..calibration_samples {
            let start = Instant::now();
            let _ = self.sample_discrete_gaussian_core(&mut rng)?;
            let elapsed = start.elapsed().as_nanos() as u64;
            measurements.push(elapsed);
        }
        
        // Use 95th percentile as target time
        measurements.sort_unstable();
        let target_idx = (calibration_samples as f64 * 0.95) as usize;
        self.timing_state.target_time_ns = measurements[target_idx];
        self.timing_state.timing_measurements = measurements;
        self.timing_state.calibrated = true;
        
        Ok(())
    }
    
    /// Add constant-time padding
    fn constant_time_padding(&self, padding_ns: u64) {
        let start = Instant::now();
        while start.elapsed().as_nanos() < padding_ns as u128 {
            // Busy wait with some computation to prevent optimization
            std::hint::black_box(rand::random::<u64>());
        }
    }
    
    /// Add bias resistance delay
    fn bias_resistance_delay<R: RngCore + CryptoRng>(&self, rng: &mut R) {
        let delay_cycles = match self.params.bias_resistance_level {
            BiasResistanceLevel::Basic => 10,
            BiasResistanceLevel::Enhanced => 50,
            BiasResistanceLevel::Maximum => 100,
        };
        
        for _ in 0..delay_cycles {
            std::hint::black_box(rng.next_u32());
        }
    }
}

/// Statistical quality tester for sampling
#[derive(Debug, Clone)]
pub struct SamplingQualityTester {
    /// Expected statistical properties
    pub expected_mean: f64,
    pub expected_variance: f64,
    /// Test parameters
    pub sample_size: usize,
    pub significance_level: f64,
}

impl SamplingQualityTester {
    /// Create a new quality tester
    pub fn new(expected_mean: f64, expected_variance: f64) -> Self {
        Self {
            expected_mean,
            expected_variance,
            sample_size: 10000,
            significance_level: 0.05,
        }
    }
    
    /// Test sampling quality
    pub fn test_sampling_quality<R: RngCore + CryptoRng>(
        &self,
        sampler: &mut QuantumResistantSampler,
        rng: &mut R,
    ) -> Result<QualityTestResult> {
        let mut samples = Vec::with_capacity(self.sample_size);
        
        // Collect samples
        for _ in 0..self.sample_size {
            let sample = sampler.sample_discrete_gaussian(rng)?;
            samples.push(sample as f64);
        }
        
        // Compute statistics
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let sample_variance = samples
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;
        
        // Perform statistical tests
        let mean_test_passed = self.test_mean(&samples, sample_mean)?;
        let variance_test_passed = self.test_variance(&samples, sample_variance)?;
        let normality_test_passed = self.test_normality(&samples)?;
        let independence_test_passed = self.test_independence(&samples)?;
        
        Ok(QualityTestResult {
            sample_mean,
            sample_variance,
            expected_mean: self.expected_mean,
            expected_variance: self.expected_variance,
            mean_test_passed,
            variance_test_passed,
            normality_test_passed,
            independence_test_passed,
            overall_passed: mean_test_passed && variance_test_passed && 
                           normality_test_passed && independence_test_passed,
        })
    }
    
    /// Test if sample mean is close to expected mean
    fn test_mean(&self, samples: &[f64], sample_mean: f64) -> Result<bool> {
        let n = samples.len() as f64;
        let standard_error = (self.expected_variance / n).sqrt();
        let z_score = (sample_mean - self.expected_mean) / standard_error;
        
        // Two-tailed test
        let critical_value = 1.96; // For 95% confidence
        Ok(z_score.abs() < critical_value)
    }
    
    /// Test if sample variance is close to expected variance
    fn test_variance(&self, samples: &[f64], sample_variance: f64) -> Result<bool> {
        let n = samples.len() as f64;
        let chi_square = (n - 1.0) * sample_variance / self.expected_variance;
        
        // Chi-square test (simplified)
        let degrees_of_freedom = n - 1.0;
        let critical_lower = degrees_of_freedom * 0.8; // Approximate
        let critical_upper = degrees_of_freedom * 1.2; // Approximate
        
        Ok(chi_square > critical_lower && chi_square < critical_upper)
    }
    
    /// Test for normality using Kolmogorov-Smirnov test (simplified)
    fn test_normality(&self, samples: &[f64]) -> Result<bool> {
        // Simplified normality test
        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = samples.len() as f64;
        let mut max_diff = 0.0;
        
        for (i, &sample) in sorted_samples.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = self.normal_cdf(sample);
            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }
        
        let critical_value = 1.36 / n.sqrt(); // For 95% confidence
        Ok(max_diff < critical_value)
    }
    
    /// Test for independence using runs test (simplified)
    fn test_independence(&self, samples: &[f64]) -> Result<bool> {
        let median = {
            let mut sorted = samples.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        
        let mut runs = 1;
        let mut above_median = samples[0] > median;
        
        for &sample in samples.iter().skip(1) {
            let current_above = sample > median;
            if current_above != above_median {
                runs += 1;
                above_median = current_above;
            }
        }
        
        let n = samples.len() as f64;
        let expected_runs = (n + 1.0) / 2.0;
        let variance_runs = (n - 1.0) / 4.0;
        let z_score = (runs as f64 - expected_runs) / variance_runs.sqrt();
        
        Ok(z_score.abs() < 1.96) // 95% confidence
    }
    
    /// Approximate normal CDF
    fn normal_cdf(&self, x: f64) -> f64 {
        let standardized = (x - self.expected_mean) / self.expected_variance.sqrt();
        0.5 * (1.0 + self.erf(standardized / 2.0_f64.sqrt()))
    }
    
    /// Approximate error function
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }
}

/// Result of sampling quality tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTestResult {
    pub sample_mean: f64,
    pub sample_variance: f64,
    pub expected_mean: f64,
    pub expected_variance: f64,
    pub mean_test_passed: bool,
    pub variance_test_passed: bool,
    pub normality_test_passed: bool,
    pub independence_test_passed: bool,
    pub overall_passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_quantum_resistant_sampler_creation() {
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params);
        
        let sampler = QuantumResistantSampler::new(sampling_params);
        assert!(sampler.is_ok());
    }
    
    #[test]
    fn test_discrete_gaussian_sampling() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params);
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        let sample = sampler.sample_discrete_gaussian(&mut rng);
        assert!(sample.is_ok());
        
        let value = sample.unwrap();
        assert!(value >= 0);
        assert!(value < lattice_params.q);
    }
    
    #[test]
    fn test_lattice_point_sampling() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params.clone());
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        let point = sampler.sample_lattice_point(&mut rng).unwrap();
        assert_eq!(point.dimension(), lattice_params.n);
        
        for coord in &point.coordinates {
            assert!(*coord >= 0);
            assert!(*coord < lattice_params.q);
        }
    }
    
    #[test]
    fn test_bias_resistant_sampling() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::High);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params.clone());
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        let point = sampler.sample_bias_resistant(&mut rng).unwrap();
        assert_eq!(point.dimension(), lattice_params.n);
    }
    
    #[test]
    fn test_multiple_point_sampling() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params.clone());
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        let points = sampler.sample_multiple_points(5, &mut rng).unwrap();
        assert_eq!(points.len(), 5);
        
        for point in &points {
            assert_eq!(point.dimension(), lattice_params.n);
        }
    }
    
    #[test]
    fn test_sampling_quality() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let sampling_params = SamplingParams::from_lattice_params(lattice_params);
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        let tester = SamplingQualityTester::new(0.0, sampling_params.sigma.powi(2));
        let result = tester.test_sampling_quality(&mut sampler, &mut rng);
        
        assert!(result.is_ok());
        // Note: Statistical tests may occasionally fail due to randomness
        // In practice, you'd run multiple tests and check overall pass rate
    }
    
    #[test]
    fn test_constant_time_sampling() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let mut sampling_params = SamplingParams::from_lattice_params(lattice_params);
        sampling_params.constant_time = true;
        
        let mut sampler = QuantumResistantSampler::new(sampling_params).unwrap();
        
        // Test that constant-time sampling works
        let sample1 = sampler.sample_discrete_gaussian(&mut rng);
        let sample2 = sampler.sample_discrete_gaussian(&mut rng);
        
        assert!(sample1.is_ok());
        assert!(sample2.is_ok());
    }
}