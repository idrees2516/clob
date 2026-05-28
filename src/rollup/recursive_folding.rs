use crate::rollup::lattice_fold_plus::{LatticeParams, LatticePoint, LatticeMatrix, SecurityLevel};
use crate::rollup::lattice_commitments::{SISCommitmentScheme, Commitment, CommitmentOpening};
use crate::rollup::lattice_challenges::{ChallengeGenerator, Challenge, ChallengeParams};
use crate::error::LatticeFoldError;
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;
use merlin::Transcript;
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, LatticeFoldError>;

/// A folding scheme that can fold multiple lattice proofs into a single proof
#[derive(Clone, Debug)]
pub struct FoldingScheme {
    /// The lattice parameters
    pub params: LatticeParams,
    /// The commitment scheme used for folding
    pub commitment_scheme: SISCommitmentScheme,
    /// The challenge generator
    pub challenge_generator: ChallengeGenerator,
    /// Cached folding matrices for efficiency
    pub folding_matrices: HashMap<String, LatticeMatrix>,
}

/// A fold operation that combines multiple proofs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldOperation {
    /// The number of proofs being folded (arity)
    pub arity: usize,
    /// The folding matrices used in the operation
    pub matrices: Vec<LatticeMatrix>,
    /// The challenges used for linear combination
    pub challenges: Vec<Challenge>,
    /// The operation identifier for caching
    pub operation_id: String,
}

/// A proof that can be folded with other proofs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldableProof {
    /// The commitment to the proof data
    pub commitment: Commitment,
    /// The opening information for the commitment
    pub opening: CommitmentOpening,
    /// The lattice point representing the proof
    pub proof_point: LatticePoint,
    /// Additional metadata
    pub metadata: ProofMetadata,
}

/// Metadata associated with a foldable proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// The proof type identifier
    pub proof_type: String,
    /// The security level used
    pub security_level: SecurityLevel,
    /// The timestamp when the proof was created
    pub timestamp: u64,
    /// Additional custom data
    pub custom_data: HashMap<String, String>,
}

/// The result of a folding operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FoldingResult {
    /// The folded proof
    pub folded_proof: FoldableProof,
    /// The folding operation that was applied
    pub fold_operation: FoldOperation,
    /// The verification complexity (number of operations)
    pub verification_complexity: usize,
    /// The compression ratio achieved
    pub compression_ratio: f64,
}

impl FoldingScheme {
    /// Create a new folding scheme
    pub fn new<R: RngCore + CryptoRng>(params: LatticeParams, rng: &mut R) -> Result<Self> {
        let commitment_params = crate::rollup::lattice_commitments::CommitmentParams::from_lattice_params(params.clone());
        let commitment_scheme = SISCommitmentScheme::new(commitment_params, rng)?;
        
        let challenge_params = ChallengeParams::from_lattice_params(params.clone(), "folding");
        let challenge_generator = ChallengeGenerator::new(challenge_params, rng)?;
        
        Ok(Self {
            params,
            commitment_scheme,
            challenge_generator,
            folding_matrices: HashMap::new(),
        })
    }
    
    /// Fold multiple proofs into a single proof
    pub fn fold_proofs<R: RngCore + CryptoRng>(
        &mut self,
        proofs: &[FoldableProof],
        rng: &mut R,
    ) -> Result<FoldingResult> {
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Cannot fold empty set of proofs".to_string(),
            ));
        }
        
        if proofs.len() == 1 {
            return Ok(FoldingResult {
                folded_proof: proofs[0].clone(),
                fold_operation: FoldOperation {
                    arity: 1,
                    matrices: vec![LatticeMatrix::identity(self.params.n)],
                    challenges: vec![Challenge::new(
                        vec![1u8; 32],
                        1,
                        LatticePoint::new(vec![1; self.params.n]),
                        "identity".to_string(),
                    )],
                    operation_id: "identity".to_string(),
                },
                verification_complexity: 1,
                compression_ratio: 1.0,
            });
        }
        
        // Generate challenges for folding
        let mut challenges = Vec::with_capacity(proofs.len());
        let mut folding_matrices = Vec::with_capacity(proofs.len());
        
        // Add all proofs to transcript for challenge generation
        for (i, proof) in proofs.iter().enumerate() {
            self.challenge_generator.add_lattice_point(&proof.proof_point, format!("proof_{}", i).as_bytes());
            self.challenge_generator.add_commitment(&proof.commitment.value.to_bytes(), format!("commitment_{}", i).as_bytes());
        }
        
        // Generate challenges and matrices
        for i in 0..proofs.len() {
            let challenge = self.challenge_generator.generate_structured_challenge(&format!("fold_{}", i))?;
            challenges.push(challenge.clone());
            
            // Generate or retrieve folding matrix
            let matrix_key = format!("fold_{}_{}", i, challenge.as_integer());
            let matrix = if let Some(cached_matrix) = self.folding_matrices.get(&matrix_key) {
                cached_matrix.clone()
            } else {
                let matrix = self.generate_folding_matrix(&challenge, rng)?;
                self.folding_matrices.insert(matrix_key, matrix.clone());
                matrix
            };
            
            folding_matrices.push(matrix);
        }
        
        // Compute folded commitment
        let folded_commitment = self.fold_commitments(&proofs, &challenges)?;
        
        // Compute folded proof point
        let folded_proof_point = self.fold_proof_points(&proofs, &challenges, &folding_matrices)?;
        
        // Create opening for folded commitment
        let folded_opening = self.fold_openings(&proofs, &challenges)?;
        
        // Create metadata for folded proof
        let folded_metadata = ProofMetadata {
            proof_type: "folded".to_string(),
            security_level: self.params.security_level,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            custom_data: {
                let mut data = HashMap::new();
                data.insert("original_count".to_string(), proofs.len().to_string());
                data.insert("folding_arity".to_string(), proofs.len().to_string());
                data
            },
        };
        
        let folded_proof = FoldableProof {
            commitment: folded_commitment,
            opening: folded_opening,
            proof_point: folded_proof_point,
            metadata: folded_metadata,
        };
        
        let fold_operation = FoldOperation {
            arity: proofs.len(),
            matrices: folding_matrices,
            challenges,
            operation_id: format!("fold_{}", proofs.len()),
        };
        
        let verification_complexity = self.compute_verification_complexity(&fold_operation);
        let compression_ratio = self.compute_compression_ratio(proofs.len(), &fold_operation);
        
        Ok(FoldingResult {
            folded_proof,
            fold_operation,
            verification_complexity,
            compression_ratio,
        })
    }
    
    /// Verify a folded proof
    pub fn verify_folded_proof(
        &self,
        folded_proof: &FoldableProof,
        original_proofs: &[FoldableProof],
        fold_operation: &FoldOperation,
    ) -> Result<bool> {
        if original_proofs.len() != fold_operation.arity {
            return Ok(false);
        }
        
        // Recompute the folded commitment
        let expected_commitment = self.fold_commitments(original_proofs, &fold_operation.challenges)?;
        
        // Recompute the folded proof point
        let expected_proof_point = self.fold_proof_points(
            original_proofs,
            &fold_operation.challenges,
            &fold_operation.matrices,
        )?;
        
        // Verify commitment matches
        let commitment_valid = folded_proof.commitment.value == expected_commitment.value;
        
        // Verify proof point matches
        let proof_point_valid = folded_proof.proof_point == expected_proof_point;
        
        // Verify the commitment opening
        let opening_valid = self.commitment_scheme.verify(&folded_proof.commitment, &folded_proof.opening)?;
        
        Ok(commitment_valid && proof_point_valid && opening_valid)
    }
    
    /// Generate a folding matrix from a challenge
    fn generate_folding_matrix<R: RngCore + CryptoRng>(
        &self,
        challenge: &Challenge,
        rng: &mut R,
    ) -> Result<LatticeMatrix> {
        let n = self.params.n;
        let q = self.params.q;
        
        // Use challenge to seed matrix generation
        let mut matrix_data = Vec::with_capacity(n);
        let challenge_bytes = challenge.as_bytes();
        
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                // Combine challenge with position to generate matrix entry
                let seed = ((challenge.as_integer() as u64).wrapping_mul((i * n + j) as u64)) as i64;
                let entry = (seed % q + q) % q;
                row.push(entry);
            }
            matrix_data.push(row);
        }
        
        LatticeMatrix::new(matrix_data)
    }
    
    /// Fold multiple commitments using linear combination
    fn fold_commitments(
        &self,
        proofs: &[FoldableProof],
        challenges: &[Challenge],
    ) -> Result<Commitment> {
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput("No proofs to fold".to_string()));
        }
        
        let commitments: Vec<_> = proofs.iter().map(|p| &p.commitment).collect();
        let scalars: Vec<_> = challenges.iter().map(|c| c.as_integer()).collect();
        
        self.commitment_scheme.linear_combination(&commitments, &scalars)
    }
    
    /// Fold multiple proof points using linear combination with matrices
    fn fold_proof_points(
        &self,
        proofs: &[FoldableProof],
        challenges: &[Challenge],
        matrices: &[LatticeMatrix],
    ) -> Result<LatticePoint> {
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput("No proofs to fold".to_string()));
        }
        
        let q = self.params.q;
        let mut folded_point = LatticePoint::zero(self.params.n);
        
        for i in 0..proofs.len() {
            // Apply matrix transformation
            let transformed_point = matrices[i].multiply_point(&proofs[i].proof_point, q)?;
            
            // Scale by challenge
            let scaled_point = transformed_point.scale_mod(challenges[i].as_integer(), q);
            
            // Add to folded point
            folded_point = folded_point.add_mod(&scaled_point, q);
        }
        
        Ok(folded_point)
    }
    
    /// Fold multiple commitment openings
    fn fold_openings(
        &self,
        proofs: &[FoldableProof],
        challenges: &[Challenge],
    ) -> Result<CommitmentOpening> {
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput("No proofs to fold".to_string()));
        }
        
        // Combine messages
        let mut combined_message = Vec::new();
        for proof in proofs {
            combined_message.extend_from_slice(&proof.opening.message);
        }
        
        // Combine randomness using linear combination
        let q = self.params.q;
        let mut combined_randomness = LatticePoint::zero(proofs[0].opening.randomness.dimension());
        
        for i in 0..proofs.len() {
            let scaled_randomness = proofs[i].opening.randomness.scale_mod(challenges[i].as_integer(), q);
            combined_randomness = combined_randomness.add_mod(&scaled_randomness, q);
        }
        
        Ok(CommitmentOpening {
            message: combined_message,
            randomness: combined_randomness,
        })
    }
    
    /// Compute the verification complexity of a fold operation
    fn compute_verification_complexity(&self, fold_operation: &FoldOperation) -> usize {
        // Base cost for commitment verification
        let commitment_cost = fold_operation.arity * 10;
        
        // Cost for matrix operations
        let matrix_cost = fold_operation.arity * self.params.n * self.params.n;
        
        // Cost for challenge verification
        let challenge_cost = fold_operation.arity * 5;
        
        commitment_cost + matrix_cost + challenge_cost
    }
    
    /// Compute the compression ratio achieved by folding
    fn compute_compression_ratio(&self, original_count: usize, fold_operation: &FoldOperation) -> f64 {
        if original_count <= 1 {
            return 1.0;
        }
        
        // Original size: sum of all individual proofs
        let original_size = original_count * self.estimate_proof_size();
        
        // Folded size: single proof + fold operation overhead
        let folded_size = self.estimate_proof_size() + self.estimate_fold_operation_size(fold_operation);
        
        original_size as f64 / folded_size as f64
    }
    
    /// Estimate the size of a single proof in bytes
    fn estimate_proof_size(&self) -> usize {
        // Commitment size + opening size + proof point size + metadata
        let commitment_size = self.params.n * 8; // 8 bytes per coordinate
        let opening_size = 1000; // Estimated opening size
        let proof_point_size = self.params.n * 8;
        let metadata_size = 200; // Estimated metadata size
        
        commitment_size + opening_size + proof_point_size + metadata_size
    }
    
    /// Estimate the size of a fold operation in bytes
    fn estimate_fold_operation_size(&self, fold_operation: &FoldOperation) -> usize {
        let matrix_size = fold_operation.matrices.len() * self.params.n * self.params.n * 8;
        let challenge_size = fold_operation.challenges.len() * 32; // 32 bytes per challenge
        let metadata_size = 100; // Operation metadata
        
        matrix_size + challenge_size + metadata_size
    }
}

impl FoldableProof {
    /// Create a new foldable proof
    pub fn new<R: RngCore + CryptoRng>(
        data: &[u8],
        proof_type: &str,
        commitment_scheme: &SISCommitmentScheme,
        params: &LatticeParams,
        rng: &mut R,
    ) -> Result<Self> {
        // Create commitment to the data
        let (commitment, opening) = commitment_scheme.commit(data, rng)?;
        
        // Create proof point from data hash
        let proof_point = Self::data_to_lattice_point(data, params)?;
        
        let metadata = ProofMetadata {
            proof_type: proof_type.to_string(),
            security_level: params.security_level,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            custom_data: HashMap::new(),
        };
        
        Ok(Self {
            commitment,
            opening,
            proof_point,
            metadata,
        })
    }
    
    /// Convert data to a lattice point
    fn data_to_lattice_point(data: &[u8], params: &LatticeParams) -> Result<LatticePoint> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();
        
        let mut coordinates = Vec::with_capacity(params.n);
        let hash_bytes = hash.as_bytes();
        
        for i in 0..params.n {
            let byte_idx = i % hash_bytes.len();
            let coord = (hash_bytes[byte_idx] as i64) % params.q;
            coordinates.push(if coord < 0 { coord + params.q } else { coord });
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Get the size of this proof in bytes
    pub fn size_in_bytes(&self) -> usize {
        let commitment_size = self.commitment.value.to_bytes().len();
        let opening_size = self.opening.message.len() + self.opening.randomness.to_bytes().len();
        let proof_point_size = self.proof_point.to_bytes().len();
        let metadata_size = bincode::serialize(&self.metadata).unwrap_or_default().len();
        
        commitment_size + opening_size + proof_point_size + metadata_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_folding_scheme_creation() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        let scheme = FoldingScheme::new(params, &mut rng);
        assert!(scheme.is_ok());
    }
    
    #[test]
    fn test_single_proof_folding() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let mut scheme = FoldingScheme::new(params.clone(), &mut rng).unwrap();
        
        // Create a single proof
        let proof = FoldableProof::new(
            b"test data",
            "test",
            &scheme.commitment_scheme,
            &params,
            &mut rng,
        ).unwrap();
        
        let result = scheme.fold_proofs(&[proof], &mut rng).unwrap();
        assert_eq!(result.fold_operation.arity, 1);
        assert_eq!(result.compression_ratio, 1.0);
    }
    
    #[test]
    fn test_multiple_proof_folding() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let mut scheme = FoldingScheme::new(params.clone(), &mut rng).unwrap();
        
        // Create multiple proofs
        let mut proofs = Vec::new();
        for i in 0..5 {
            let data = format!("test data {}", i);
            let proof = FoldableProof::new(
                data.as_bytes(),
                "test",
                &scheme.commitment_scheme,
                &params,
                &mut rng,
            ).unwrap();
            proofs.push(proof);
        }
        
        let result = scheme.fold_proofs(&proofs, &mut rng).unwrap();
        assert_eq!(result.fold_operation.arity, 5);
        assert!(result.compression_ratio > 1.0);
        assert!(result.verification_complexity > 0);
    }
    
    #[test]
    fn test_folded_proof_verification() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let mut scheme = FoldingScheme::new(params.clone(), &mut rng).unwrap();
        
        // Create multiple proofs
        let mut proofs = Vec::new();
        for i in 0..3 {
            let data = format!("test data {}", i);
            let proof = FoldableProof::new(
                data.as_bytes(),
                "test",
                &scheme.commitment_scheme,
                &params,
                &mut rng,
            ).unwrap();
            proofs.push(proof);
        }
        
        // Fold the proofs
        let result = scheme.fold_proofs(&proofs, &mut rng).unwrap();
        
        // Verify the folded proof
        let is_valid = scheme.verify_folded_proof(
            &result.folded_proof,
            &proofs,
            &result.fold_operation,
        ).unwrap();
        
        assert!(is_valid);
    }
    
    #[test]
    fn test_proof_size_estimation() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let scheme = FoldingScheme::new(params.clone(), &mut rng).unwrap();
        
        let proof = FoldableProof::new(
            b"test data",
            "test",
            &scheme.commitment_scheme,
            &params,
            &mut rng,
        ).unwrap();
        
        let estimated_size = scheme.estimate_proof_size();
        let actual_size = proof.size_in_bytes();
        
        // Estimated size should be in the right ballpark
        assert!(estimated_size > actual_size / 2);
        assert!(estimated_size < actual_size * 2);
    }
}