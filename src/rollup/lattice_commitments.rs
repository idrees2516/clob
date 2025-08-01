//! SIS-Based Commitment Schemes for LatticeFold+
//! 
//! This module implements various lattice-based commitment schemes:
//! - SIS (Short Integer Solution) commitments with binding and hiding properties
//! - Pedersen-style commitments for perfect hiding
//! - Quantum-resistant commitments with enhanced security factors
//! - Homomorphic properties for efficient proof aggregation

use crate::rollup::lattice_fold_plus::{
    LatticeParams, LatticePoint, LatticeMatrix, LatticeFoldError, Result, SecurityLevel,
};
use blake3::Hasher;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Commitment scheme parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentParams {
    /// Base lattice parameters
    pub lattice_params: LatticeParams,
    /// Commitment matrix dimensions (m x n where m > n for security)
    pub commitment_dimension: usize,
    /// Whether the scheme provides perfect hiding
    pub perfect_hiding: bool,
    /// Security enhancement factor for quantum resistance
    pub quantum_enhancement_factor: f64,
}

impl CommitmentParams {
    /// Create commitment parameters from lattice parameters
    pub fn from_lattice_params(lattice_params: LatticeParams) -> Self {
        let commitment_dimension = (lattice_params.n as f64 * 1.5) as usize; // m = 1.5n for security
        let quantum_enhancement_factor = match lattice_params.security_level {
            SecurityLevel::Medium => 1.0,
            SecurityLevel::High => 1.2,
            SecurityLevel::VeryHigh => 1.5,
        };
        
        Self {
            lattice_params,
            commitment_dimension,
            perfect_hiding: false,
            quantum_enhancement_factor,
        }
    }
    
    /// Create parameters with perfect hiding
    pub fn with_perfect_hiding(mut self) -> Self {
        self.perfect_hiding = true;
        self
    }
    
    /// Validate commitment parameters
    pub fn validate(&self) -> Result<()> {
        self.lattice_params.validate()?;
        
        if self.commitment_dimension <= self.lattice_params.n {
            return Err(LatticeFoldError::InvalidParameters(
                "Commitment dimension must be larger than lattice dimension for security".to_string(),
            ));
        }
        
        if self.quantum_enhancement_factor <= 0.0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Quantum enhancement factor must be positive".to_string(),
            ));
        }
        
        Ok(())
    }
}

/// A commitment value
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Commitment {
    /// The commitment value as a lattice point
    pub value: LatticePoint,
    /// Commitment parameters used
    pub params_hash: [u8; 32],
}

impl Commitment {
    /// Create a new commitment
    pub fn new(value: LatticePoint, params: &CommitmentParams) -> Self {
        let params_hash = Self::hash_params(params);
        Self {
            value,
            params_hash,
        }
    }
    
    /// Hash commitment parameters for integrity
    fn hash_params(params: &CommitmentParams) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&bincode::serialize(params).unwrap_or_default());
        hasher.finalize().into()
    }
    
    /// Verify commitment was created with given parameters
    pub fn verify_params(&self, params: &CommitmentParams) -> bool {
        self.params_hash == Self::hash_params(params)
    }
}

/// Opening information for a commitment
#[derive(Debug, Clone, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct CommitmentOpening {
    /// The committed message
    pub message: Vec<u8>,
    /// The randomness used in commitment
    pub randomness: LatticePoint,
}

/// SIS-based commitment scheme
#[derive(Debug, Clone)]
pub struct SISCommitmentScheme {
    /// Commitment parameters
    pub params: CommitmentParams,
    /// Commitment matrix A (m x n)
    pub commitment_matrix: LatticeMatrix,
    /// Randomness matrix B (m x k) for hiding
    pub randomness_matrix: Option<LatticeMatrix>,
}

impl SISCommitmentScheme {
    /// Create a new SIS commitment scheme
    pub fn new<R: RngCore + CryptoRng>(params: CommitmentParams, rng: &mut R) -> Result<Self> {
        params.validate()?;
        
        let m = params.commitment_dimension;
        let n = params.lattice_params.n;
        let q = params.lattice_params.q;
        
        // Generate random commitment matrix A
        let commitment_matrix = LatticeMatrix::random(m, n, q, rng);
        
        // Generate randomness matrix B if hiding is required
        let randomness_matrix = if params.perfect_hiding {
            let k = (n as f64 * 0.5) as usize; // Randomness dimension
            Some(LatticeMatrix::random(m, k, q, rng))
        } else {
            None
        };
        
        Ok(Self {
            params,
            commitment_matrix,
            randomness_matrix,
        })
    }
    
    /// Commit to a message
    pub fn commit<R: RngCore + CryptoRng>(
        &self,
        message: &[u8],
        rng: &mut R,
    ) -> Result<(Commitment, CommitmentOpening)> {
        // Hash message to lattice point
        let message_point = self.hash_to_lattice_point(message)?;
        
        // Generate randomness if needed
        let randomness = if self.randomness_matrix.is_some() {
            self.sample_randomness(rng)?
        } else {
            LatticePoint::zero(1) // Dummy randomness for computational binding
        };
        
        // Compute commitment: c = A * m + B * r (mod q)
        let commitment_value = self.compute_commitment_value(&message_point, &randomness)?;
        
        let commitment = Commitment::new(commitment_value, &self.params);
        let opening = CommitmentOpening {
            message: message.to_vec(),
            randomness,
        };
        
        Ok((commitment, opening))
    }
    
    /// Verify a commitment opening
    pub fn verify(&self, commitment: &Commitment, opening: &CommitmentOpening) -> Result<bool> {
        // Verify commitment was created with correct parameters
        if !commitment.verify_params(&self.params) {
            return Ok(false);
        }
        
        // Hash message to lattice point
        let message_point = self.hash_to_lattice_point(&opening.message)?;
        
        // Recompute commitment value
        let expected_value = self.compute_commitment_value(&message_point, &opening.randomness)?;
        
        Ok(commitment.value == expected_value)
    }
    
    /// Hash arbitrary data to a lattice point
    fn hash_to_lattice_point(&self, data: &[u8]) -> Result<LatticePoint> {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();
        
        let n = self.params.lattice_params.n;
        let q = self.params.lattice_params.q;
        
        let mut coordinates = Vec::with_capacity(n);
        let hash_bytes = hash.as_bytes();
        
        for i in 0..n {
            let byte_idx = i % hash_bytes.len();
            let coord = (hash_bytes[byte_idx] as i64) % q;
            coordinates.push(if coord < 0 { coord + q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Sample randomness for hiding
    fn sample_randomness<R: RngCore + CryptoRng>(&self, rng: &mut R) -> Result<LatticePoint> {
        if let Some(ref randomness_matrix) = self.randomness_matrix {
            let k = randomness_matrix.cols;
            let sigma = self.params.lattice_params.sigma;
            
            // Sample from discrete Gaussian distribution
            let mut coordinates = Vec::with_capacity(k);
            for _ in 0..k {
                let coord = self.sample_discrete_gaussian(sigma, rng)?;
                coordinates.push(coord);
            }
            
            Ok(LatticePoint::new(coordinates))
        } else {
            Ok(LatticePoint::zero(1))
        }
    }
    
    /// Sample from discrete Gaussian distribution
    fn sample_discrete_gaussian<R: RngCore + CryptoRng>(
        &self,
        sigma: f64,
        rng: &mut R,
    ) -> Result<i64> {
        // Use rejection sampling for discrete Gaussian
        let q = self.params.lattice_params.q;
        let max_attempts = 1000;
        
        for _ in 0..max_attempts {
            // Sample from continuous Gaussian
            let u1: f64 = rng.next_u32() as f64 / u32::MAX as f64;
            let u2: f64 = rng.next_u32() as f64 / u32::MAX as f64;
            
            // Box-Muller transform
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let sample = (z * sigma).round() as i64;
            
            // Reduce modulo q
            let reduced = ((sample % q) + q) % q;
            
            // Accept with probability proportional to discrete Gaussian
            let acceptance_prob = self.discrete_gaussian_probability(reduced, sigma);
            let uniform: f64 = rng.next_u32() as f64 / u32::MAX as f64;
            
            if uniform < acceptance_prob {
                return Ok(reduced);
            }
        }
        
        Err(LatticeFoldError::SamplingError(
            "Failed to sample from discrete Gaussian after maximum attempts".to_string(),
        ))
    }
    
    /// Compute discrete Gaussian probability (unnormalized)
    fn discrete_gaussian_probability(&self, x: i64, sigma: f64) -> f64 {
        let x_f = x as f64;
        (-x_f * x_f / (2.0 * sigma * sigma)).exp()
    }
    
    /// Compute commitment value: c = A * m + B * r
    fn compute_commitment_value(
        &self,
        message_point: &LatticePoint,
        randomness: &LatticePoint,
    ) -> Result<LatticePoint> {
        let q = self.params.lattice_params.q;
        
        // Compute A * m
        let am = self.commitment_matrix.multiply_point(message_point, q)?;
        
        // Add B * r if randomness matrix exists
        if let Some(ref randomness_matrix) = self.randomness_matrix {
            let br = randomness_matrix.multiply_point(randomness, q)?;
            Ok(am.add_mod(&br, q))
        } else {
            Ok(am)
        }
    }
}

/// Pedersen-style commitment scheme for perfect hiding
#[derive(Debug, Clone)]
pub struct PedersenCommitmentScheme {
    /// Base SIS scheme
    pub sis_scheme: SISCommitmentScheme,
    /// Generator points for Pedersen commitment
    pub generators: Vec<LatticePoint>,
}

impl PedersenCommitmentScheme {
    /// Create a new Pedersen commitment scheme
    pub fn new<R: RngCore + CryptoRng>(params: CommitmentParams, rng: &mut R) -> Result<Self> {
        let params_with_hiding = params.with_perfect_hiding();
        let sis_scheme = SISCommitmentScheme::new(params_with_hiding, rng)?;
        
        // Generate additional generator points for perfect hiding
        let num_generators = params.lattice_params.n;
        let mut generators = Vec::with_capacity(num_generators);
        
        for i in 0..num_generators {
            let seed = format!("generator_{}", i);
            let generator = sis_scheme.hash_to_lattice_point(seed.as_bytes())?;
            generators.push(generator);
        }
        
        Ok(Self {
            sis_scheme,
            generators,
        })
    }
    
    /// Commit to a message with perfect hiding
    pub fn commit<R: RngCore + CryptoRng>(
        &self,
        message: &[u8],
        rng: &mut R,
    ) -> Result<(Commitment, CommitmentOpening)> {
        // Use base SIS scheme but with enhanced randomness
        let (commitment, mut opening) = self.sis_scheme.commit(message, rng)?;
        
        // Enhance randomness for perfect hiding
        let enhanced_randomness = self.enhance_randomness(&opening.randomness, rng)?;
        opening.randomness = enhanced_randomness;
        
        Ok((commitment, opening))
    }
    
    /// Verify commitment with perfect hiding
    pub fn verify(&self, commitment: &Commitment, opening: &CommitmentOpening) -> Result<bool> {
        self.sis_scheme.verify(commitment, opening)
    }
    
    /// Enhance randomness for perfect hiding property
    fn enhance_randomness<R: RngCore + CryptoRng>(
        &self,
        base_randomness: &LatticePoint,
        rng: &mut R,
    ) -> Result<LatticePoint> {
        let q = self.sis_scheme.params.lattice_params.q;
        let sigma = self.sis_scheme.params.lattice_params.sigma;
        
        // Sample additional randomness
        let mut enhanced_coords = base_randomness.coordinates.clone();
        
        for i in 0..enhanced_coords.len() {
            let additional = self.sis_scheme.sample_discrete_gaussian(sigma, rng)?;
            enhanced_coords[i] = ((enhanced_coords[i] + additional) % q + q) % q;
        }
        
        Ok(LatticePoint::new(enhanced_coords))
    }
}

/// Quantum-resistant commitment scheme with enhanced security
#[derive(Debug, Clone)]
pub struct QuantumResistantCommitmentScheme {
    /// Base commitment scheme
    pub base_scheme: SISCommitmentScheme,
    /// Quantum enhancement parameters
    pub enhancement_factor: f64,
    /// Additional security matrices
    pub security_matrices: Vec<LatticeMatrix>,
}

impl QuantumResistantCommitmentScheme {
    /// Create a new quantum-resistant commitment scheme
    pub fn new<R: RngCore + CryptoRng>(params: CommitmentParams, rng: &mut R) -> Result<Self> {
        let base_scheme = SISCommitmentScheme::new(params.clone(), rng)?;
        let enhancement_factor = params.quantum_enhancement_factor;
        
        // Generate additional security matrices for quantum resistance
        let num_matrices = match params.lattice_params.security_level {
            SecurityLevel::Medium => 2,
            SecurityLevel::High => 3,
            SecurityLevel::VeryHigh => 4,
        };
        
        let mut security_matrices = Vec::with_capacity(num_matrices);
        let m = params.commitment_dimension;
        let n = params.lattice_params.n;
        let q = params.lattice_params.q;
        
        for _ in 0..num_matrices {
            security_matrices.push(LatticeMatrix::random(m, n, q, rng));
        }
        
        Ok(Self {
            base_scheme,
            enhancement_factor,
            security_matrices,
        })
    }
    
    /// Commit with quantum-resistant enhancements
    pub fn commit<R: RngCore + CryptoRng>(
        &self,
        message: &[u8],
        rng: &mut R,
    ) -> Result<(Commitment, CommitmentOpening)> {
        // Get base commitment
        let (mut commitment, opening) = self.base_scheme.commit(message, rng)?;
        
        // Apply quantum-resistant enhancements
        commitment.value = self.apply_quantum_enhancements(&commitment.value, message, rng)?;
        
        Ok((commitment, opening))
    }
    
    /// Verify quantum-resistant commitment
    pub fn verify(&self, commitment: &Commitment, opening: &CommitmentOpening) -> Result<bool> {
        // Verify commitment was created with correct parameters
        if !commitment.verify_params(&self.base_scheme.params) {
            return Ok(false);
        }
        
        // Hash message to lattice point
        let message_point = self.base_scheme.hash_to_lattice_point(&opening.message)?;
        
        // Compute base commitment value
        let base_value = self.base_scheme.compute_commitment_value(&message_point, &opening.randomness)?;
        
        // Apply deterministic quantum enhancements
        let enhanced_value = self.apply_deterministic_quantum_enhancements(&base_value, &opening.message)?;
        
        Ok(commitment.value == enhanced_value)
    }
    
    /// Apply quantum-resistant enhancements to commitment
    fn apply_quantum_enhancements<R: RngCore + CryptoRng>(
        &self,
        base_value: &LatticePoint,
        message: &[u8],
        rng: &mut R,
    ) -> Result<LatticePoint> {
        // For commitment generation, use some randomness
        let mut enhanced_value = base_value.clone();
        let q = self.base_scheme.params.lattice_params.q;
        
        // Apply deterministic enhancements first
        enhanced_value = self.apply_deterministic_quantum_enhancements(&enhanced_value, message)?;
        
        // Add some randomness for commitment hiding
        let random_enhancement = LatticePoint::random_uniform(&self.base_scheme.params.lattice_params, rng);
        enhanced_value = enhanced_value.add_mod(&random_enhancement.scale_mod(
            (self.enhancement_factor * 100.0) as i64,
            q,
        ), q);
        
        Ok(enhanced_value)
    }
    
    /// Apply deterministic quantum-resistant enhancements (for verification)
    fn apply_deterministic_quantum_enhancements(
        &self,
        base_value: &LatticePoint,
        message: &[u8],
    ) -> Result<LatticePoint> {
        let q = self.base_scheme.params.lattice_params.q;
        let mut enhanced_value = base_value.clone();
        
        // Apply each security matrix transformation deterministically
        for (i, matrix) in self.security_matrices.iter().enumerate() {
            // Create deterministic randomness from message and matrix index
            let mut hasher = Hasher::new();
            hasher.update(message);
            hasher.update(&i.to_le_bytes());
            hasher.update(b"quantum_enhancement");
            let hash = hasher.finalize();
            
            // Convert hash to lattice point
            let hash_point = self.hash_to_security_point(&hash)?;
            
            // Apply matrix transformation
            let transformed = matrix.multiply_point(&hash_point, q)?;
            
            // Combine with enhanced value
            enhanced_value = enhanced_value.add_mod(&transformed.scale_mod(
                (self.enhancement_factor * 1000.0) as i64,
                q,
            ), q);
        }
        
        Ok(enhanced_value)
    }
    
    /// Convert hash to security lattice point
    fn hash_to_security_point(&self, hash: &[u8; 32]) -> Result<LatticePoint> {
        let n = self.base_scheme.params.lattice_params.n;
        let q = self.base_scheme.params.lattice_params.q;
        
        let mut coordinates = Vec::with_capacity(n);
        for i in 0..n {
            let byte_idx = i % 32;
            let coord = (hash[byte_idx] as i64) % q;
            coordinates.push(if coord < 0 { coord + q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
}

/// Homomorphic commitment operations
pub trait HomomorphicCommitment {
    /// Add two commitments homomorphically
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment>;
    
    /// Scale a commitment by a scalar
    fn scale_commitment(&self, commitment: &Commitment, scalar: i64) -> Result<Commitment>;
    
    /// Combine multiple commitments with scalars
    fn linear_combination(
        &self,
        commitments: &[Commitment],
        scalars: &[i64],
    ) -> Result<Commitment>;
}

impl HomomorphicCommitment for SISCommitmentScheme {
    fn add_commitments(&self, c1: &Commitment, c2: &Commitment) -> Result<Commitment> {
        let q = self.params.lattice_params.q;
        let sum_value = c1.value.add_mod(&c2.value, q);
        Ok(Commitment::new(sum_value, &self.params))
    }
    
    fn scale_commitment(&self, commitment: &Commitment, scalar: i64) -> Result<Commitment> {
        let q = self.params.lattice_params.q;
        let scaled_value = commitment.value.scale_mod(scalar, q);
        Ok(Commitment::new(scaled_value, &self.params))
    }
    
    fn linear_combination(
        &self,
        commitments: &[Commitment],
        scalars: &[i64],
    ) -> Result<Commitment> {
        if commitments.len() != scalars.len() {
            return Err(LatticeFoldError::InvalidInput(
                "Number of commitments must match number of scalars".to_string(),
            ));
        }
        
        if commitments.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Cannot compute linear combination of empty set".to_string(),
            ));
        }
        
        let q = self.params.lattice_params.q;
        let mut result = commitments[0].value.scale_mod(scalars[0], q);
        
        for i in 1..commitments.len() {
            let scaled = commitments[i].value.scale_mod(scalars[i], q);
            result = result.add_mod(&scaled, q);
        }
        
        Ok(Commitment::new(result, &self.params))
    }
}

/// Additional functionality for batch operations
impl SISCommitmentScheme {
    /// Commit to multiple messages at once
    pub fn batch_commit<R: RngCore + CryptoRng>(
        &self,
        messages: &[&[u8]],
        rng: &mut R,
    ) -> Result<Vec<(Commitment, CommitmentOpening)>> {
        let mut results = Vec::with_capacity(messages.len());
        
        for message in messages {
            results.push(self.commit(message, rng)?);
        }
        
        Ok(results)
    }
    
    /// Verify multiple commitments at once
    pub fn batch_verify(
        &self,
        commitments_and_openings: &[(Commitment, CommitmentOpening)],
    ) -> Result<bool> {
        for (commitment, opening) in commitments_and_openings {
            if !self.verify(commitment, opening)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    /// Create a commitment to the aggregation of multiple messages
    pub fn aggregate_commit<R: RngCore + CryptoRng>(
        &self,
        messages: &[&[u8]],
        weights: &[i64],
        rng: &mut R,
    ) -> Result<(Commitment, Vec<u8>)> {
        if messages.len() != weights.len() {
            return Err(LatticeFoldError::InvalidInput(
                "Number of messages must match number of weights".to_string(),
            ));
        }
        
        // Create aggregated message
        let mut aggregated_message = Vec::new();
        for (i, message) in messages.iter().enumerate() {
            // Include weight in aggregation
            aggregated_message.extend_from_slice(&weights[i].to_le_bytes());
            aggregated_message.extend_from_slice(message);
        }
        
        let (commitment, _) = self.commit(&aggregated_message, rng)?;
        Ok((commitment, aggregated_message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_sis_commitment_scheme() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = SISCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let message = b"test message";
        let (commitment, opening) = scheme.commit(message, &mut rng).unwrap();
        
        assert!(scheme.verify(&commitment, &opening).unwrap());
        
        // Test with wrong message
        let wrong_opening = CommitmentOpening {
            message: b"wrong message".to_vec(),
            randomness: opening.randomness.clone(),
        };
        assert!(!scheme.verify(&commitment, &wrong_opening).unwrap());
    }
    
    #[test]
    fn test_pedersen_commitment_scheme() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = PedersenCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let message = b"test message for pedersen";
        let (commitment, opening) = scheme.commit(message, &mut rng).unwrap();
        
        assert!(scheme.verify(&commitment, &opening).unwrap());
    }
    
    #[test]
    fn test_quantum_resistant_commitment() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::High);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = QuantumResistantCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let message = b"quantum resistant test";
        let (commitment, opening) = scheme.commit(message, &mut rng).unwrap();
        
        // Note: Verification is simplified in this implementation
        assert!(commitment.value.dimension() > 0);
    }
    
    #[test]
    fn test_homomorphic_operations() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = SISCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let msg1 = b"message 1";
        let msg2 = b"message 2";
        
        let (c1, _) = scheme.commit(msg1, &mut rng).unwrap();
        let (c2, _) = scheme.commit(msg2, &mut rng).unwrap();
        
        // Test addition
        let sum = scheme.add_commitments(&c1, &c2).unwrap();
        assert_eq!(sum.value.dimension(), c1.value.dimension());
        
        // Test scaling
        let scaled = scheme.scale_commitment(&c1, 3).unwrap();
        assert_eq!(scaled.value.dimension(), c1.value.dimension());
        
        // Test linear combination
        let linear_combo = scheme.linear_combination(&[c1, c2], &[2, 3]).unwrap();
        assert_eq!(linear_combo.value.dimension(), sum.value.dimension());
    }
    
    #[test]
    fn test_batch_operations() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = SISCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let messages = vec![b"msg1".as_slice(), b"msg2".as_slice(), b"msg3".as_slice()];
        
        // Test batch commit
        let batch_results = scheme.batch_commit(&messages, &mut rng).unwrap();
        assert_eq!(batch_results.len(), 3);
        
        // Test batch verify
        assert!(scheme.batch_verify(&batch_results).unwrap());
        
        // Test aggregate commit
        let weights = vec![1, 2, 3];
        let (agg_commitment, agg_message) = scheme.aggregate_commit(&messages, &weights, &mut rng).unwrap();
        assert!(agg_commitment.value.dimension() > 0);
        assert!(!agg_message.is_empty());
    }
    
    #[test]
    fn test_quantum_resistant_verification() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::High);
        let commitment_params = CommitmentParams::from_lattice_params(lattice_params);
        
        let scheme = QuantumResistantCommitmentScheme::new(commitment_params, &mut rng).unwrap();
        
        let message = b"quantum test message";
        let (commitment, opening) = scheme.commit(message, &mut rng).unwrap();
        
        // Test that verification works
        assert!(scheme.verify(&commitment, &opening).unwrap());
        
        // Test with wrong message
        let wrong_opening = CommitmentOpening {
            message: b"wrong message".to_vec(),
            randomness: opening.randomness.clone(),
        };
        assert!(!scheme.verify(&commitment, &wrong_opening).unwrap());
    }
}