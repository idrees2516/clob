//! Challenge Generation and Cryptographic Transcripts for LatticeFold+
//! 
//! This module implements secure challenge generation for lattice-based proofs:
//! - ChallengeGenerator with Fiat-Shamir transformation support
//! - TranscriptChallengeGenerator for deterministic challenge derivation
//! - Structured challenge sampling for different proof contexts
//! - Cryptographic security and uniqueness guarantees

use crate::rollup::lattice_fold_plus::{
    LatticeParams, LatticePoint, LatticeFoldError, Result, SecurityLevel,
};
use blake3::Hasher;
use merlin::Transcript;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Challenge generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeParams {
    /// Base lattice parameters
    pub lattice_params: LatticeParams,
    /// Challenge space size (number of possible challenges)
    pub challenge_space_bits: usize,
    /// Whether to use Fiat-Shamir transformation
    pub fiat_shamir: bool,
    /// Domain separation tag for different proof contexts
    pub domain_separation: String,
    /// Security parameter for challenge generation
    pub security_parameter: usize,
}

impl ChallengeParams {
    /// Create challenge parameters from lattice parameters
    pub fn from_lattice_params(lattice_params: LatticeParams, domain: &str) -> Self {
        let challenge_space_bits = match lattice_params.security_level {
            SecurityLevel::Medium => 128,
            SecurityLevel::High => 192,
            SecurityLevel::VeryHigh => 256,
        };
        
        let security_parameter = lattice_params.security_level.classical_bits();
        
        Self {
            lattice_params,
            challenge_space_bits,
            fiat_shamir: true,
            domain_separation: domain.to_string(),
            security_parameter,
        }
    }
    
    /// Validate challenge parameters
    pub fn validate(&self) -> Result<()> {
        self.lattice_params.validate()?;
        
        if self.challenge_space_bits < 128 {
            return Err(LatticeFoldError::InvalidParameters(
                "Challenge space must be at least 128 bits for security".to_string(),
            ));
        }
        
        if self.security_parameter < 128 {
            return Err(LatticeFoldError::InvalidParameters(
                "Security parameter must be at least 128 bits".to_string(),
            ));
        }
        
        if self.domain_separation.is_empty() {
            return Err(LatticeFoldError::InvalidParameters(
                "Domain separation tag cannot be empty".to_string(),
            ));
        }
        
        Ok(())
    }
}

/// A cryptographic challenge
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct Challenge {
    /// Challenge value as bytes
    pub value: Vec<u8>,
    /// Challenge as integer (for arithmetic operations)
    pub integer_value: i64,
    /// Challenge as lattice point (for lattice operations)
    pub lattice_point: LatticePoint,
    /// Domain this challenge was generated for
    pub domain: String,
}

impl Challenge {
    /// Create a new challenge
    pub fn new(
        value: Vec<u8>,
        integer_value: i64,
        lattice_point: LatticePoint,
        domain: String,
    ) -> Self {
        Self {
            value,
            integer_value,
            lattice_point,
            domain,
        }
    }
    
    /// Get challenge as bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.value
    }
    
    /// Get challenge as integer
    pub fn as_integer(&self) -> i64 {
        self.integer_value
    }
    
    /// Get challenge as lattice point
    pub fn as_lattice_point(&self) -> &LatticePoint {
        &self.lattice_point
    }
    
    /// Verify challenge was generated for correct domain
    pub fn verify_domain(&self, expected_domain: &str) -> bool {
        self.domain == expected_domain
    }
}

/// Vector of challenges for batch operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeVector {
    /// Individual challenges
    pub challenges: Vec<Challenge>,
    /// Combined hash of all challenges
    pub combined_hash: [u8; 32],
}

impl ChallengeVector {
    /// Create a new challenge vector
    pub fn new(challenges: Vec<Challenge>) -> Self {
        let combined_hash = Self::compute_combined_hash(&challenges);
        Self {
            challenges,
            combined_hash,
        }
    }
    
    /// Get number of challenges
    pub fn len(&self) -> usize {
        self.challenges.len()
    }
    
    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.challenges.is_empty()
    }
    
    /// Get challenge at index
    pub fn get(&self, index: usize) -> Option<&Challenge> {
        self.challenges.get(index)
    }
    
    /// Compute combined hash of all challenges
    fn compute_combined_hash(challenges: &[Challenge]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"challenge_vector");
        
        for challenge in challenges {
            hasher.update(&challenge.value);
        }
        
        hasher.finalize().into()
    }
    
    /// Verify integrity of challenge vector
    pub fn verify_integrity(&self) -> bool {
        let expected_hash = Self::compute_combined_hash(&self.challenges);
        self.combined_hash == expected_hash
    }
}

/// Challenge generator for lattice-based proofs
#[derive(Debug, Clone)]
pub struct ChallengeGenerator {
    /// Challenge generation parameters
    pub params: ChallengeParams,
    /// Internal state for deterministic generation
    pub state: ChallengeState,
    /// Cache for frequently used challenges
    pub challenge_cache: HashMap<String, Challenge>,
}

/// Internal state for challenge generation
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ChallengeState {
    /// Current transcript state
    pub transcript_state: Vec<u8>,
    /// Challenge counter for uniqueness
    pub challenge_counter: u64,
    /// Random seed for non-deterministic generation
    pub random_seed: [u8; 32],
}

impl ChallengeGenerator {
    /// Create a new challenge generator
    pub fn new<R: RngCore + CryptoRng>(params: ChallengeParams, rng: &mut R) -> Result<Self> {
        params.validate()?;
        
        let mut random_seed = [0u8; 32];
        rng.fill_bytes(&mut random_seed);
        
        let state = ChallengeState {
            transcript_state: Vec::new(),
            challenge_counter: 0,
            random_seed,
        };
        
        Ok(Self {
            params,
            state,
            challenge_cache: HashMap::new(),
        })
    }
    
    /// Generate a single challenge
    pub fn generate_challenge(&mut self) -> Result<Challenge> {
        if self.params.fiat_shamir {
            self.generate_fiat_shamir_challenge()
        } else {
            self.generate_random_challenge()
        }
    }
    
    /// Generate a vector of challenges
    pub fn generate_challenge_vector(&mut self, count: usize) -> Result<ChallengeVector> {
        let mut challenges = Vec::with_capacity(count);
        
        for _ in 0..count {
            challenges.push(self.generate_challenge()?);
        }
        
        Ok(ChallengeVector::new(challenges))
    }
    
    /// Generate structured challenge for specific proof context
    pub fn generate_structured_challenge(&mut self, context: &str) -> Result<Challenge> {
        // Add context to transcript
        self.add_to_transcript(b"structured_challenge");
        self.add_to_transcript(context.as_bytes());
        
        // Generate challenge with context-specific domain
        let domain = format!("{}_{}", self.params.domain_separation, context);
        self.generate_challenge_with_domain(&domain)
    }
    
    /// Add data to transcript for Fiat-Shamir
    pub fn add_to_transcript(&mut self, data: &[u8]) {
        self.state.transcript_state.extend_from_slice(data);
    }
    
    /// Add lattice point to transcript
    pub fn add_lattice_point(&mut self, point: &LatticePoint, label: &[u8]) {
        self.add_to_transcript(label);
        self.add_to_transcript(&point.to_bytes());
    }
    
    /// Add commitment to transcript
    pub fn add_commitment(&mut self, commitment: &[u8], label: &[u8]) {
        self.add_to_transcript(label);
        self.add_to_transcript(commitment);
    }
    
    /// Reset transcript state
    pub fn reset_transcript(&mut self) {
        self.state.transcript_state.clear();
        self.state.challenge_counter = 0;
    }
    
    /// Generate Fiat-Shamir challenge from transcript
    fn generate_fiat_shamir_challenge(&mut self) -> Result<Challenge> {
        // Create hash input from transcript and counter
        let mut hasher = Hasher::new();
        hasher.update(b"fiat_shamir_challenge");
        hasher.update(&self.params.domain_separation.as_bytes());
        hasher.update(&self.state.transcript_state);
        hasher.update(&self.state.challenge_counter.to_le_bytes());
        
        let hash = hasher.finalize();
        
        // Increment counter for uniqueness
        self.state.challenge_counter += 1;
        
        // Convert hash to challenge
        self.hash_to_challenge(hash.as_bytes(), &self.params.domain_separation)
    }
    
    /// Generate random challenge (non-Fiat-Shamir)
    fn generate_random_challenge(&mut self) -> Result<Challenge> {
        // Use cryptographic randomness
        let mut hasher = Hasher::new();
        hasher.update(b"random_challenge");
        hasher.update(&self.state.random_seed);
        hasher.update(&self.state.challenge_counter.to_le_bytes());
        
        let hash = hasher.finalize();
        
        // Update seed for next challenge
        self.state.random_seed = hash.as_bytes()[0..32].try_into().unwrap();
        self.state.challenge_counter += 1;
        
        self.hash_to_challenge(hash.as_bytes(), &self.params.domain_separation)
    }
    
    /// Generate challenge with specific domain
    fn generate_challenge_with_domain(&mut self, domain: &str) -> Result<Challenge> {
        let mut hasher = Hasher::new();
        hasher.update(b"domain_challenge");
        hasher.update(domain.as_bytes());
        hasher.update(&self.state.transcript_state);
        hasher.update(&self.state.challenge_counter.to_le_bytes());
        
        let hash = hasher.finalize();
        self.state.challenge_counter += 1;
        
        self.hash_to_challenge(hash.as_bytes(), domain)
    }
    
    /// Convert hash to challenge in all required formats
    fn hash_to_challenge(&self, hash: &[u8], domain: &str) -> Result<Challenge> {
        // Challenge as bytes (first 32 bytes of hash)
        let value = hash[0..32].to_vec();
        
        // Challenge as integer (modulo q)
        let integer_value = {
            let mut int_bytes = [0u8; 8];
            int_bytes.copy_from_slice(&hash[0..8]);
            let raw_int = i64::from_le_bytes(int_bytes);
            ((raw_int % self.params.lattice_params.q) + self.params.lattice_params.q) 
                % self.params.lattice_params.q
        };
        
        // Challenge as lattice point
        let lattice_point = self.hash_to_lattice_point(hash)?;
        
        Ok(Challenge::new(
            value,
            integer_value,
            lattice_point,
            domain.to_string(),
        ))
    }
    
    /// Convert hash to lattice point
    fn hash_to_lattice_point(&self, hash: &[u8]) -> Result<LatticePoint> {
        let n = self.params.lattice_params.n;
        let q = self.params.lattice_params.q;
        
        let mut coordinates = Vec::with_capacity(n);
        
        // Use hash to generate coordinates
        for i in 0..n {
            let byte_idx = i % hash.len();
            let coord = (hash[byte_idx] as i64) % q;
            coordinates.push(if coord < 0 { coord + q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Get cached challenge if available
    pub fn get_cached_challenge(&self, key: &str) -> Option<&Challenge> {
        self.challenge_cache.get(key)
    }
    
    /// Cache a challenge for reuse
    pub fn cache_challenge(&mut self, key: String, challenge: Challenge) {
        self.challenge_cache.insert(key, challenge);
    }
    
    /// Clear challenge cache
    pub fn clear_cache(&mut self) {
        self.challenge_cache.clear();
    }
}

/// Transcript-based challenge generator using Merlin transcripts
#[derive(Debug)]
pub struct TranscriptChallengeGenerator {
    /// Merlin transcript for Fiat-Shamir
    pub transcript: Transcript,
    /// Challenge parameters
    pub params: ChallengeParams,
    /// Challenge counter
    pub counter: u64,
}

impl TranscriptChallengeGenerator {
    /// Create a new transcript challenge generator
    pub fn new(params: ChallengeParams) -> Result<Self> {
        params.validate()?;
        
        let transcript = Transcript::new(params.domain_separation.as_bytes());
        
        Ok(Self {
            transcript,
            params,
            counter: 0,
        })
    }
    
    /// Append message to transcript
    pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
        self.transcript.append_message(label, message);
    }
    
    /// Append lattice point to transcript
    pub fn append_lattice_point(&mut self, label: &'static [u8], point: &LatticePoint) {
        self.transcript.append_message(label, &point.to_bytes());
    }
    
    /// Generate challenge from current transcript state
    pub fn challenge_scalar(&mut self, label: &'static [u8]) -> Result<Challenge> {
        let mut challenge_bytes = vec![0u8; 32];
        self.transcript.challenge_bytes(label, &mut challenge_bytes);
        
        self.counter += 1;
        
        // Convert to challenge
        self.bytes_to_challenge(&challenge_bytes)
    }
    
    /// Generate challenge vector from transcript
    pub fn challenge_vector(&mut self, label: &'static [u8], count: usize) -> Result<ChallengeVector> {
        let mut challenges = Vec::with_capacity(count);
        
        for i in 0..count {
            let indexed_label = format!("{}_{}", std::str::from_utf8(label).unwrap_or("challenge"), i);
            let challenge = self.challenge_scalar(indexed_label.as_bytes())?;
            challenges.push(challenge);
        }
        
        Ok(ChallengeVector::new(challenges))
    }
    
    /// Generate structured challenge with context
    pub fn structured_challenge(&mut self, context: &str) -> Result<Challenge> {
        let label = format!("structured_{}", context);
        self.challenge_scalar(label.as_bytes())
    }
    
    /// Convert bytes to challenge
    fn bytes_to_challenge(&self, bytes: &[u8]) -> Result<Challenge> {
        // Challenge as bytes
        let value = bytes.to_vec();
        
        // Challenge as integer (modulo q)
        let integer_value = {
            let mut int_bytes = [0u8; 8];
            int_bytes.copy_from_slice(&bytes[0..8]);
            let raw_int = i64::from_le_bytes(int_bytes);
            ((raw_int % self.params.lattice_params.q) + self.params.lattice_params.q) 
                % self.params.lattice_params.q
        };
        
        // Challenge as lattice point
        let lattice_point = self.bytes_to_lattice_point(bytes)?;
        
        Ok(Challenge::new(
            value,
            integer_value,
            lattice_point,
            self.params.domain_separation.clone(),
        ))
    }
    
    /// Convert bytes to lattice point
    fn bytes_to_lattice_point(&self, bytes: &[u8]) -> Result<LatticePoint> {
        let n = self.params.lattice_params.n;
        let q = self.params.lattice_params.q;
        
        let mut coordinates = Vec::with_capacity(n);
        
        for i in 0..n {
            let byte_idx = i % bytes.len();
            let coord = (bytes[byte_idx] as i64) % q;
            coordinates.push(if coord < 0 { coord + q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Reset transcript to initial state
    pub fn reset(&mut self) {
        self.transcript = Transcript::new(self.params.domain_separation.as_bytes());
        self.counter = 0;
    }
    
    /// Get current transcript state
    pub fn get_state(&self) -> Vec<u8> {
        // Note: Merlin doesn't expose internal state directly
        // This is a placeholder for transcript state extraction
        format!("transcript_state_{}", self.counter).into_bytes()
    }
}

/// Challenge verification utilities
pub struct ChallengeVerifier {
    /// Expected challenge parameters
    pub params: ChallengeParams,
}

impl ChallengeVerifier {
    /// Create a new challenge verifier
    pub fn new(params: ChallengeParams) -> Result<Self> {
        params.validate()?;
        Ok(Self { params })
    }
    
    /// Verify challenge was generated correctly
    pub fn verify_challenge(&self, challenge: &Challenge, expected_domain: &str) -> Result<bool> {
        // Check domain
        if !challenge.verify_domain(expected_domain) {
            return Ok(false);
        }
        
        // Check value consistency
        if challenge.value.len() != 32 {
            return Ok(false);
        }
        
        // Check integer value is in correct range
        if challenge.integer_value < 0 || challenge.integer_value >= self.params.lattice_params.q {
            return Ok(false);
        }
        
        // Check lattice point dimension
        if challenge.lattice_point.dimension() != self.params.lattice_params.n {
            return Ok(false);
        }
        
        // Check lattice point coordinates are in range
        for &coord in &challenge.lattice_point.coordinates {
            if coord < 0 || coord >= self.params.lattice_params.q {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Verify challenge vector integrity
    pub fn verify_challenge_vector(&self, vector: &ChallengeVector) -> Result<bool> {
        // Check integrity hash
        if !vector.verify_integrity() {
            return Ok(false);
        }
        
        // Verify each individual challenge
        for challenge in &vector.challenges {
            if !self.verify_challenge(challenge, &self.params.domain_separation)? {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Verify challenge uniqueness in a set
    pub fn verify_uniqueness(&self, challenges: &[Challenge]) -> Result<bool> {
        let mut seen_values = std::collections::HashSet::new();
        
        for challenge in challenges {
            if !seen_values.insert(&challenge.value) {
                return Ok(false); // Duplicate found
            }
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_challenge_generator_creation() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "test");
        
        let generator = ChallengeGenerator::new(challenge_params, &mut rng);
        assert!(generator.is_ok());
    }
    
    #[test]
    fn test_challenge_generation() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params.clone(), "test");
        
        let mut generator = ChallengeGenerator::new(challenge_params, &mut rng).unwrap();
        
        let challenge = generator.generate_challenge().unwrap();
        
        assert_eq!(challenge.value.len(), 32);
        assert!(challenge.integer_value >= 0);
        assert!(challenge.integer_value < lattice_params.q);
        assert_eq!(challenge.lattice_point.dimension(), lattice_params.n);
        assert!(challenge.verify_domain("test"));
    }
    
    #[test]
    fn test_challenge_vector_generation() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "test");
        
        let mut generator = ChallengeGenerator::new(challenge_params, &mut rng).unwrap();
        
        let vector = generator.generate_challenge_vector(5).unwrap();
        
        assert_eq!(vector.len(), 5);
        assert!(vector.verify_integrity());
    }
    
    #[test]
    fn test_transcript_challenge_generator() {
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "transcript_test");
        
        let mut generator = TranscriptChallengeGenerator::new(challenge_params).unwrap();
        
        // Add some data to transcript
        generator.append_message(b"test_data", b"some test data");
        
        let challenge = generator.challenge_scalar(b"test_challenge").unwrap();
        
        assert_eq!(challenge.value.len(), 32);
        assert!(challenge.verify_domain("transcript_test"));
    }
    
    #[test]
    fn test_structured_challenge_generation() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "structured_test");
        
        let mut generator = ChallengeGenerator::new(challenge_params, &mut rng).unwrap();
        
        let challenge = generator.generate_structured_challenge("commitment").unwrap();
        
        assert!(challenge.verify_domain("structured_test_commitment"));
    }
    
    #[test]
    fn test_challenge_verification() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "verify_test");
        
        let mut generator = ChallengeGenerator::new(challenge_params.clone(), &mut rng).unwrap();
        let verifier = ChallengeVerifier::new(challenge_params).unwrap();
        
        let challenge = generator.generate_challenge().unwrap();
        
        assert!(verifier.verify_challenge(&challenge, "verify_test").unwrap());
        assert!(!verifier.verify_challenge(&challenge, "wrong_domain").unwrap());
    }
    
    #[test]
    fn test_fiat_shamir_determinism() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let mut challenge_params = ChallengeParams::from_lattice_params(lattice_params, "determinism_test");
        challenge_params.fiat_shamir = true;
        
        let mut generator1 = ChallengeGenerator::new(challenge_params.clone(), &mut rng).unwrap();
        let mut generator2 = ChallengeGenerator::new(challenge_params, &mut rng).unwrap();
        
        // Add same data to both transcripts
        generator1.add_to_transcript(b"test_data");
        generator2.add_to_transcript(b"test_data");
        
        let challenge1 = generator1.generate_challenge().unwrap();
        let challenge2 = generator2.generate_challenge().unwrap();
        
        // Should be the same due to Fiat-Shamir determinism
        assert_eq!(challenge1.value, challenge2.value);
    }
    
    #[test]
    fn test_challenge_uniqueness() {
        let mut rng = thread_rng();
        let lattice_params = LatticeParams::new(SecurityLevel::Medium);
        let challenge_params = ChallengeParams::from_lattice_params(lattice_params, "uniqueness_test");
        
        let mut generator = ChallengeGenerator::new(challenge_params.clone(), &mut rng).unwrap();
        let verifier = ChallengeVerifier::new(challenge_params).unwrap();
        
        let mut challenges = Vec::new();
        for _ in 0..10 {
            challenges.push(generator.generate_challenge().unwrap());
        }
        
        assert!(verifier.verify_uniqueness(&challenges).unwrap());
    }
}