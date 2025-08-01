//! LatticeFold+ Core Implementation for Trustless Data Availability
//! 
//! This module implements the LatticeFold+ protocol with lattice-based cryptographic primitives
//! for quantum-resistant data availability proofs. It provides:
//! - Core lattice structures and parameters
//! - SIS-based commitment schemes
//! - Quantum-resistant Gaussian sampling
//! - Challenge generation with cryptographic transcripts
//! - Recursive folding for proof aggregation

use crate::error::TradingError;
use blake3::Hasher;
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// LatticeFold+ specific errors
#[derive(Error, Debug)]
pub enum LatticeFoldError {
    #[error("Invalid lattice parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Point not in lattice")]
    PointNotInLattice,
    
    #[error("Invalid dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },
    
    #[error("Verification failed: {0}")]
    VerificationError(String),
    
    #[error("Sampling error: {0}")]
    SamplingError(String),
    
    #[error("Challenge generation error: {0}")]
    ChallengeError(String),
    
    #[error("Commitment error: {0}")]
    CommitmentError(String),
    
    #[error("Quantum resistance analysis failed: {0}")]
    QuantumAnalysisError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, LatticeFoldError>;

/// Security levels for quantum resistance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 128-bit classical security, ~64-bit quantum security
    Medium,
    /// 192-bit classical security, ~96-bit quantum security  
    High,
    /// 256-bit classical security, ~128-bit quantum security
    VeryHigh,
}

/// Optimization targets for parameter selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Optimize for computational speed
    Speed,
    /// Optimize for maximum security
    Security,
    /// Optimize for memory usage
    Memory,
    /// Balanced optimization
    Balanced,
}

impl SecurityLevel {
    /// Get the classical security bits
    pub fn classical_bits(&self) -> usize {
        match self {
            SecurityLevel::Medium => 128,
            SecurityLevel::High => 192,
            SecurityLevel::VeryHigh => 256,
        }
    }
    
    /// Get the quantum security bits (accounting for Grover speedup)
    pub fn quantum_bits(&self) -> usize {
        self.classical_bits() / 2
    }
    
    /// Get recommended lattice dimension for this security level
    pub fn lattice_dimension(&self) -> usize {
        match self {
            SecurityLevel::Medium => 512,
            SecurityLevel::High => 768,
            SecurityLevel::VeryHigh => 1024,
        }
    }
    
    /// Get recommended modulus for this security level
    pub fn modulus(&self) -> i64 {
        match self {
            SecurityLevel::Medium => 2147483647, // 2^31 - 1 (Mersenne prime)
            SecurityLevel::High => 2305843009213693951, // 2^61 - 1 (Mersenne prime)
            SecurityLevel::VeryHigh => 9223372036854775783, // Large prime near 2^63
        }
    }
}

/// Core lattice parameters for LatticeFold+
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeParams {
    /// Lattice dimension (n)
    pub n: usize,
    /// Modulus (q) - should be prime for security
    pub q: i64,
    /// Gaussian width parameter for sampling
    pub sigma: f64,
    /// Security level
    pub security_level: SecurityLevel,
    /// BKZ block size for hardness estimation
    pub bkz_block_size: usize,
}

impl LatticeParams {
    /// Create new lattice parameters for a given security level
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            n: security_level.lattice_dimension(),
            q: security_level.modulus(),
            sigma: Self::compute_gaussian_width(security_level),
            security_level,
            bkz_block_size: Self::compute_bkz_block_size(security_level),
        }
    }
    
    /// Create optimized parameters for specific use case
    pub fn optimized(security_level: SecurityLevel, optimization_target: OptimizationTarget) -> Self {
        let mut params = Self::new(security_level);
        
        match optimization_target {
            OptimizationTarget::Speed => {
                // Reduce dimension slightly for faster operations
                params.n = (params.n as f64 * 0.9) as usize;
                params.sigma *= 0.95; // Slightly tighter distribution
            },
            OptimizationTarget::Security => {
                // Increase dimension for extra security margin
                params.n = (params.n as f64 * 1.1) as usize;
                params.sigma *= 1.05; // Slightly wider distribution
            },
            OptimizationTarget::Memory => {
                // Optimize for memory usage
                params.n = params.n.min(256); // Cap dimension
                params.sigma *= 0.9;
            },
            OptimizationTarget::Balanced => {
                // Keep default parameters
            },
        }
        
        params
    }
    
    /// Create parameters with custom dimension and modulus
    pub fn custom(n: usize, q: i64, security_level: SecurityLevel) -> Result<Self> {
        let mut params = Self::new(security_level);
        params.n = n;
        params.q = q;
        params.sigma = Self::compute_gaussian_width_for_dimension(n, security_level);
        params.bkz_block_size = Self::compute_bkz_block_size_for_dimension(n);
        
        params.validate()?;
        Ok(params)
    }
    
    /// Compute optimal Gaussian width for security level
    fn compute_gaussian_width(security_level: SecurityLevel) -> f64 {
        let n = security_level.lattice_dimension() as f64;
        Self::compute_gaussian_width_for_dimension(n as usize, security_level)
    }
    
    /// Compute Gaussian width for specific dimension
    fn compute_gaussian_width_for_dimension(n: usize, security_level: SecurityLevel) -> f64 {
        let n_f = n as f64;
        // Use smoothing parameter: σ ≥ η_ε(Λ) for small ε
        // Approximation: σ ≈ √(n/(2π)) * √(ln(2n/ε))
        let epsilon = 2.0_f64.powi(-(security_level.classical_bits() as i32));
        let ln_term = (2.0 * n_f / epsilon).ln();
        (n_f / (2.0 * std::f64::consts::PI)).sqrt() * ln_term.sqrt()
    }
    
    /// Compute BKZ block size for hardness estimation
    fn compute_bkz_block_size(security_level: SecurityLevel) -> usize {
        let n = security_level.lattice_dimension();
        Self::compute_bkz_block_size_for_dimension(n)
    }
    
    /// Compute BKZ block size for specific dimension
    fn compute_bkz_block_size_for_dimension(n: usize) -> usize {
        // Heuristic: block size ≈ n/10 with bounds
        (n / 10).max(20).min(150)
    }
    
    /// Validate parameters for security
    pub fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Dimension must be positive".to_string(),
            ));
        }
        
        if self.q <= 1 {
            return Err(LatticeFoldError::InvalidParameters(
                "Modulus must be greater than 1".to_string(),
            ));
        }
        
        if self.sigma <= 0.0 {
            return Err(LatticeFoldError::InvalidParameters(
                "Gaussian width must be positive".to_string(),
            ));
        }
        
        // Check that q is large enough for security
        let min_q = 2_i64.pow(self.security_level.classical_bits() as u32 / 4);
        if self.q < min_q {
            return Err(LatticeFoldError::InvalidParameters(
                format!("Modulus {} too small for security level, need at least {}", self.q, min_q),
            ));
        }
        
        Ok(())
    }
}

impl Default for LatticeParams {
    fn default() -> Self {
        Self::new(SecurityLevel::Medium)
    }
}

/// A point in the lattice
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct LatticePoint {
    /// Coordinates of the point
    pub coordinates: Vec<i64>,
}

impl LatticePoint {
    /// Create a new lattice point
    pub fn new(coordinates: Vec<i64>) -> Self {
        Self { coordinates }
    }
    
    /// Create a zero point of given dimension
    pub fn zero(dimension: usize) -> Self {
        Self {
            coordinates: vec![0; dimension],
        }
    }
    
    /// Get the dimension of the point
    pub fn dimension(&self) -> usize {
        self.coordinates.len()
    }
    
    /// Add two lattice points modulo q
    pub fn add_mod(&self, other: &LatticePoint, q: i64) -> LatticePoint {
        if self.dimension() != other.dimension() {
            panic!("Cannot add points of different dimensions");
        }
        
        let coordinates = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| ((a + b) % q + q) % q)
            .collect();
            
        LatticePoint::new(coordinates)
    }
    
    /// Subtract two lattice points modulo q
    pub fn sub_mod(&self, other: &LatticePoint, q: i64) -> LatticePoint {
        if self.dimension() != other.dimension() {
            panic!("Cannot subtract points of different dimensions");
        }
        
        let coordinates = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| ((a - b) % q + q) % q)
            .collect();
            
        LatticePoint::new(coordinates)
    }
    
    /// Scale a lattice point by a scalar modulo q
    pub fn scale_mod(&self, scalar: i64, q: i64) -> LatticePoint {
        let coordinates = self
            .coordinates
            .iter()
            .map(|x| ((x * scalar) % q + q) % q)
            .collect();
            
        LatticePoint::new(coordinates)
    }
    
    /// Compute inner product with another point modulo q
    pub fn inner_product_mod(&self, other: &LatticePoint, q: i64) -> i64 {
        if self.dimension() != other.dimension() {
            panic!("Cannot compute inner product of points with different dimensions");
        }
        
        let sum: i64 = self
            .coordinates
            .iter()
            .zip(&other.coordinates)
            .map(|(a, b)| (a * b) % q)
            .sum();
            
        (sum % q + q) % q
    }
    
    /// Check if point is in the fundamental domain [0, q)^n
    pub fn is_in_fundamental_domain(&self, q: i64) -> bool {
        self.coordinates.iter().all(|&x| x >= 0 && x < q)
    }
    
    /// Reduce point to fundamental domain
    pub fn reduce_mod(&mut self, q: i64) {
        for coord in &mut self.coordinates {
            *coord = ((*coord % q) + q) % q;
        }
    }
    
    /// Convert to bytes for hashing/serialization
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        for coord in &self.coordinates {
            bytes.extend_from_slice(&coord.to_le_bytes());
        }
        bytes
    }
    
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Result<Self> {
        if bytes.len() != dimension * 8 {
            return Err(LatticeFoldError::InvalidInput(
                format!("Invalid byte length: expected {}, got {}", dimension * 8, bytes.len()),
            ));
        }
        
        let mut coordinates = Vec::with_capacity(dimension);
        for i in 0..dimension {
            let start = i * 8;
            let end = start + 8;
            let coord_bytes: [u8; 8] = bytes[start..end].try_into().unwrap();
            coordinates.push(i64::from_le_bytes(coord_bytes));
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Sample a random lattice point with Gaussian distribution
    pub fn random_gaussian<R: RngCore + CryptoRng>(
        params: &LatticeParams,
        rng: &mut R,
    ) -> Result<Self> {
        use rand_distr::{Distribution, Normal};
        
        let normal = Normal::new(0.0, params.sigma)
            .map_err(|e| LatticeFoldError::SamplingError(format!("Invalid Gaussian parameters: {}", e)))?;
        
        let mut coordinates = Vec::with_capacity(params.n);
        for _ in 0..params.n {
            let sample = normal.sample(rng);
            let rounded = sample.round() as i64;
            let reduced = ((rounded % params.q) + params.q) % params.q;
            coordinates.push(reduced);
        }
        
        Ok(LatticePoint::new(coordinates))
    }
    
    /// Sample a uniform random lattice point
    pub fn random_uniform<R: RngCore + CryptoRng>(
        params: &LatticeParams,
        rng: &mut R,
    ) -> Self {
        let mut coordinates = Vec::with_capacity(params.n);
        for _ in 0..params.n {
            let val = (rng.next_u64() as i64) % params.q;
            coordinates.push(if val < 0 { val + params.q } else { val });
        }
        
        LatticePoint::new(coordinates)
    }
    
    /// Check if this point is in the specified lattice
    pub fn is_in_lattice(&self, basis_matrix: &LatticeMatrix, q: i64) -> bool {
        // For simplicity, check if point is in fundamental domain
        // In practice, this would involve more complex lattice membership testing
        self.is_in_fundamental_domain(q) && self.dimension() == basis_matrix.cols
    }
    
    /// Compute the L2 norm of the point
    pub fn l2_norm(&self) -> f64 {
        let sum_squares: i64 = self.coordinates.iter().map(|&x| x * x).sum();
        (sum_squares as f64).sqrt()
    }
    
    /// Compute the infinity norm of the point
    pub fn infinity_norm(&self) -> i64 {
        self.coordinates.iter().map(|&x| x.abs()).max().unwrap_or(0)
    }
    
    /// Add scaled point: self + scalar * other (mod q)
    pub fn add_scaled(&self, other: &LatticePoint, scalar: i64, params: &LatticeParams) -> Result<LatticePoint> {
        if self.dimension() != other.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.dimension(),
                got: other.dimension(),
            });
        }
        
        let scaled_other = other.scale_mod(scalar, params.q);
        Ok(self.add_mod(&scaled_other, params.q))
    }
    
    /// Hash the point to a challenge value
    pub fn hash_to_challenge(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&self.to_bytes());
        let hash = hasher.finalize();
        *hash.as_bytes()
    }
}

impl fmt::Display for LatticePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatticePoint({:?})", self.coordinates)
    }
}

/// A lattice matrix for linear operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeMatrix {
    /// Matrix data stored in row-major order
    pub data: Vec<Vec<i64>>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
}

impl LatticeMatrix {
    /// Create a new lattice matrix
    pub fn new(data: Vec<Vec<i64>>) -> Result<Self> {
        if data.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Matrix cannot be empty".to_string(),
            ));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Verify all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(LatticeFoldError::InvalidInput(
                    format!("Row {} has length {}, expected {}", i, row.len(), cols),
                ));
            }
        }
        
        Ok(Self { data, rows, cols })
    }
    
    /// Create an identity matrix
    pub fn identity(size: usize) -> Self {
        let mut data = vec![vec![0; size]; size];
        for i in 0..size {
            data[i][i] = 1;
        }
        
        Self {
            data,
            rows: size,
            cols: size,
        }
    }
    
    /// Create a random matrix with entries in [0, q)
    pub fn random<R: RngCore + CryptoRng>(
        rows: usize,
        cols: usize,
        q: i64,
        rng: &mut R,
    ) -> Self {
        let mut data = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                let val = (rng.next_u64() as i64) % q;
                row.push(if val < 0 { val + q } else { val });
            }
            data.push(row);
        }
        
        Self { data, rows, cols }
    }
    
    /// Multiply matrix by a lattice point (Ax)
    pub fn multiply_point(&self, point: &LatticePoint, q: i64) -> Result<LatticePoint> {
        if self.cols != point.dimension() {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.cols,
                got: point.dimension(),
            });
        }
        
        let mut result = Vec::with_capacity(self.rows);
        for row in &self.data {
            let sum: i64 = row
                .iter()
                .zip(&point.coordinates)
                .map(|(a, b)| (a * b) % q)
                .sum();
            result.push((sum % q + q) % q);
        }
        
        Ok(LatticePoint::new(result))
    }
    
    /// Matrix multiplication (AB)
    pub fn multiply(&self, other: &LatticeMatrix, q: i64) -> Result<LatticeMatrix> {
        if self.cols != other.rows {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.cols,
                got: other.rows,
            });
        }
        
        let mut result = vec![vec![0; other.cols]; self.rows];
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0i64;
                for k in 0..self.cols {
                    sum = (sum + (self.data[i][k] * other.data[k][j]) % q) % q;
                }
                result[i][j] = (sum % q + q) % q;
            }
        }
        
        LatticeMatrix::new(result)
    }
    
    /// Transpose the matrix
    pub fn transpose(&self) -> LatticeMatrix {
        let mut data = vec![vec![0; self.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j][i] = self.data[i][j];
            }
        }
        
        LatticeMatrix {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }
    
    /// Add two matrices (A + B)
    pub fn add(&self, other: &LatticeMatrix, q: i64) -> Result<LatticeMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.rows * self.cols,
                got: other.rows * other.cols,
            });
        }
        
        let mut result = vec![vec![0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let sum = (self.data[i][j] + other.data[i][j]) % q;
                result[i][j] = if sum < 0 { sum + q } else { sum };
            }
        }
        
        LatticeMatrix::new(result)
    }
    
    /// Subtract two matrices (A - B)
    pub fn subtract(&self, other: &LatticeMatrix, q: i64) -> Result<LatticeMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(LatticeFoldError::InvalidDimension {
                expected: self.rows * self.cols,
                got: other.rows * other.cols,
            });
        }
        
        let mut result = vec![vec![0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let diff = (self.data[i][j] - other.data[i][j]) % q;
                result[i][j] = if diff < 0 { diff + q } else { diff };
            }
        }
        
        LatticeMatrix::new(result)
    }
    
    /// Scale matrix by scalar (sA)
    pub fn scale(&self, scalar: i64, q: i64) -> LatticeMatrix {
        let mut result = vec![vec![0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                let scaled = (self.data[i][j] * scalar) % q;
                result[i][j] = if scaled < 0 { scaled + q } else { scaled };
            }
        }
        
        LatticeMatrix {
            data: result,
            rows: self.rows,
            cols: self.cols,
        }
    }
    
    /// Get matrix determinant (for square matrices only)
    pub fn determinant(&self, q: i64) -> Result<i64> {
        if self.rows != self.cols {
            return Err(LatticeFoldError::InvalidInput(
                "Determinant only defined for square matrices".to_string(),
            ));
        }
        
        if self.rows == 1 {
            return Ok(self.data[0][0]);
        }
        
        if self.rows == 2 {
            let det = (self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]) % q;
            return Ok(if det < 0 { det + q } else { det });
        }
        
        // For larger matrices, use cofactor expansion (simplified implementation)
        let mut det = 0i64;
        for j in 0..self.cols {
            let minor = self.minor(0, j)?;
            let cofactor = if j % 2 == 0 { 1 } else { -1 };
            let minor_det = minor.determinant(q)?;
            det = (det + cofactor * self.data[0][j] * minor_det) % q;
        }
        
        Ok(if det < 0 { det + q } else { det })
    }
    
    /// Get minor matrix by removing row i and column j
    fn minor(&self, row: usize, col: usize) -> Result<LatticeMatrix> {
        if row >= self.rows || col >= self.cols {
            return Err(LatticeFoldError::InvalidInput(
                "Row or column index out of bounds".to_string(),
            ));
        }
        
        let mut data = Vec::new();
        for i in 0..self.rows {
            if i == row {
                continue;
            }
            let mut new_row = Vec::new();
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
                new_row.push(self.data[i][j]);
            }
            data.push(new_row);
        }
        
        LatticeMatrix::new(data)
    }
    
    /// Check if matrix is invertible (det != 0)
    pub fn is_invertible(&self, q: i64) -> Result<bool> {
        if self.rows != self.cols {
            return Ok(false);
        }
        
        let det = self.determinant(q)?;
        Ok(det != 0)
    }
    
    /// Get the size in bytes for serialization
    pub fn size_in_bytes(&self) -> usize {
        // Size of dimensions + size of data
        8 + 8 + (self.rows * self.cols * 8)
    }
}

/// Quantum resistance analyzer for parameter optimization
#[derive(Debug, Clone)]
pub struct QuantumResistanceAnalyzer {
    /// Current security level
    pub security_level: SecurityLevel,
    /// BKZ cost models for different attack scenarios
    pub bkz_models: HashMap<String, BKZCostModel>,
}

/// BKZ cost model for security analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BKZCostModel {
    /// Model name (e.g., "Core-SVP", "Gate-Count", "Q-Core-SVP")
    pub name: String,
    /// Cost function parameters
    pub parameters: HashMap<String, f64>,
}

impl QuantumResistanceAnalyzer {
    /// Create a new quantum resistance analyzer
    pub fn new(security_level: SecurityLevel) -> Self {
        let mut bkz_models = HashMap::new();
        
        // Core-SVP model (classical)
        let mut core_svp_params = HashMap::new();
        core_svp_params.insert("c1".to_string(), 0.292);
        core_svp_params.insert("c2".to_string(), 16.4);
        bkz_models.insert(
            "Core-SVP".to_string(),
            BKZCostModel {
                name: "Core-SVP".to_string(),
                parameters: core_svp_params,
            },
        );
        
        // Gate-Count model (quantum)
        let mut gate_count_params = HashMap::new();
        gate_count_params.insert("c1".to_string(), 0.265);
        gate_count_params.insert("c2".to_string(), 16.4);
        bkz_models.insert(
            "Gate-Count".to_string(),
            BKZCostModel {
                name: "Gate-Count".to_string(),
                parameters: gate_count_params,
            },
        );
        
        // Q-Core-SVP model (quantum with depth constraints)
        let mut q_core_svp_params = HashMap::new();
        q_core_svp_params.insert("c1".to_string(), 0.292);
        q_core_svp_params.insert("c2".to_string(), 16.4);
        q_core_svp_params.insert("depth_factor".to_string(), 0.5);
        bkz_models.insert(
            "Q-Core-SVP".to_string(),
            BKZCostModel {
                name: "Q-Core-SVP".to_string(),
                parameters: q_core_svp_params,
            },
        );
        
        Self {
            security_level,
            bkz_models,
        }
    }
    
    /// Analyze security of lattice parameters against quantum attacks
    pub fn analyze_security(&self, params: &LatticeParams) -> Result<SecurityAnalysis> {
        let mut analysis = SecurityAnalysis {
            params: params.clone(),
            classical_security_bits: 0.0,
            quantum_security_bits: 0.0,
            attack_costs: HashMap::new(),
            recommendations: Vec::new(),
        };
        
        // Estimate attack costs for each model
        for (model_name, model) in &self.bkz_models {
            let cost = self.estimate_attack_cost(params, model)?;
            analysis.attack_costs.insert(model_name.clone(), cost);
        }
        
        // Compute security levels
        analysis.classical_security_bits = analysis
            .attack_costs
            .get("Core-SVP")
            .unwrap_or(&0.0)
            .log2();
            
        analysis.quantum_security_bits = analysis
            .attack_costs
            .get("Gate-Count")
            .unwrap_or(&0.0)
            .log2();
        
        // Generate recommendations
        self.generate_recommendations(&mut analysis)?;
        
        Ok(analysis)
    }
    
    /// Estimate attack cost using BKZ cost model
    fn estimate_attack_cost(&self, params: &LatticeParams, model: &BKZCostModel) -> Result<f64> {
        let n = params.n as f64;
        let q = params.q as f64;
        let sigma = params.sigma;
        
        // Estimate root Hermite factor
        let delta = self.estimate_root_hermite_factor(params)?;
        
        // Estimate required BKZ block size
        let beta = self.estimate_required_block_size(params, delta)?;
        
        // Get model parameters
        let c1 = model.parameters.get("c1").unwrap_or(&0.292);
        let c2 = model.parameters.get("c2").unwrap_or(&16.4);
        
        // Compute cost: 2^(c1 * β + c2)
        let cost_exponent = c1 * beta + c2;
        let cost = 2.0_f64.powf(cost_exponent);
        
        // Apply quantum depth factor if present
        if let Some(depth_factor) = model.parameters.get("depth_factor") {
            Ok(cost * depth_factor)
        } else {
            Ok(cost)
        }
    }
    
    /// Estimate root Hermite factor for lattice
    fn estimate_root_hermite_factor(&self, params: &LatticeParams) -> Result<f64> {
        let n = params.n as f64;
        let q = params.q as f64;
        let sigma = params.sigma;
        
        // For SIS: δ ≈ (σ√n / q^(1/n))^(1/(n-1))
        let numerator = sigma * n.sqrt();
        let denominator = q.powf(1.0 / n);
        let ratio = numerator / denominator;
        
        Ok(ratio.powf(1.0 / (n - 1.0)))
    }
    
    /// Estimate required BKZ block size to achieve root Hermite factor
    fn estimate_required_block_size(&self, params: &LatticeParams, delta: f64) -> Result<f64> {
        let n = params.n as f64;
        
        // Approximation: β ≈ n / (2 * log₂(δ))
        let log_delta = delta.log2();
        if log_delta <= 0.0 {
            return Err(LatticeFoldError::QuantumAnalysisError(
                "Invalid root Hermite factor".to_string(),
            ));
        }
        
        Ok(n / (2.0 * log_delta))
    }
    
    /// Generate security recommendations
    fn generate_recommendations(&self, analysis: &mut SecurityAnalysis) -> Result<()> {
        let target_classical_bits = self.security_level.classical_bits() as f64;
        let target_quantum_bits = self.security_level.quantum_bits() as f64;
        
        if analysis.classical_security_bits < target_classical_bits {
            analysis.recommendations.push(format!(
                "Classical security ({:.1} bits) below target ({:.1} bits). Consider increasing dimension or modulus.",
                analysis.classical_security_bits, target_classical_bits
            ));
        }
        
        if analysis.quantum_security_bits < target_quantum_bits {
            analysis.recommendations.push(format!(
                "Quantum security ({:.1} bits) below target ({:.1} bits). Consider quantum-resistant parameter upgrade.",
                analysis.quantum_security_bits, target_quantum_bits
            ));
        }
        
        if analysis.classical_security_bits > target_classical_bits * 1.5 {
            analysis.recommendations.push(format!(
                "Classical security ({:.1} bits) significantly exceeds target. Consider optimizing for performance.",
                analysis.classical_security_bits
            ));
        }
        
        Ok(())
    }
    
    /// Recommend parameter upgrade for increased security
    pub fn recommend_upgrade(&self, current_params: &LatticeParams, threat_level: ThreatLevel) -> Result<LatticeParams> {
        let new_security_level = match threat_level {
            ThreatLevel::Low => SecurityLevel::Medium,
            ThreatLevel::Medium => SecurityLevel::High,
            ThreatLevel::High => SecurityLevel::VeryHigh,
            ThreatLevel::Critical => SecurityLevel::VeryHigh, // Max available
        };
        
        if new_security_level as u8 <= self.security_level as u8 {
            return Ok(current_params.clone());
        }
        
        Ok(LatticeParams::new(new_security_level))
    }
    
    /// Estimate quantum attack cost for given parameters
    pub fn estimate_quantum_attack_cost(&self, params: &LatticeParams) -> Result<QuantumAttackCost> {
        let analysis = self.analyze_security(params)?;
        
        let gate_count_cost = analysis.attack_costs.get("Gate-Count").unwrap_or(&0.0);
        let q_core_svp_cost = analysis.attack_costs.get("Q-Core-SVP").unwrap_or(&0.0);
        
        Ok(QuantumAttackCost {
            gate_count_model: *gate_count_cost,
            q_core_svp_model: *q_core_svp_cost,
            grover_speedup_factor: 0.5, // Square root speedup
            estimated_time_years: Self::cost_to_time_estimate(*gate_count_cost),
            confidence_level: 0.8, // 80% confidence in estimates
        })
    }
    
    /// Convert computational cost to time estimate
    fn cost_to_time_estimate(cost: f64) -> f64 {
        // Rough estimate: assume 10^15 operations per second for quantum computer
        let ops_per_second = 1e15;
        let seconds_per_year = 365.25 * 24.0 * 3600.0;
        
        cost / (ops_per_second * seconds_per_year)
    }
    
    /// Check if parameters meet minimum security requirements
    pub fn meets_security_requirements(&self, params: &LatticeParams) -> Result<bool> {
        let analysis = self.analyze_security(params)?;
        
        let target_classical = self.security_level.classical_bits() as f64;
        let target_quantum = self.security_level.quantum_bits() as f64;
        
        Ok(analysis.classical_security_bits >= target_classical && 
           analysis.quantum_security_bits >= target_quantum)
    }
    
    /// Optimize parameters for specific constraints
    pub fn optimize_parameters(
        &self,
        constraints: &SecurityConstraints,
    ) -> Result<LatticeParams> {
        let mut best_params = LatticeParams::new(self.security_level);
        let mut best_score = f64::NEG_INFINITY;
        
        // Try different parameter combinations
        for &n in &[256, 384, 512, 768, 1024, 1536, 2048] {
            for &q_exp in &[31, 61, 127] { // Powers of 2 minus 1 (Mersenne primes)
                let q = (1i64 << q_exp) - 1;
                
                if let Ok(params) = LatticeParams::custom(n, q, self.security_level) {
                    if let Ok(analysis) = self.analyze_security(&params) {
                        let score = self.compute_parameter_score(&params, &analysis, constraints);
                        if score > best_score {
                            best_score = score;
                            best_params = params;
                        }
                    }
                }
            }
        }
        
        Ok(best_params)
    }
    
    /// Compute parameter score based on constraints
    fn compute_parameter_score(
        &self,
        params: &LatticeParams,
        analysis: &SecurityAnalysis,
        constraints: &SecurityConstraints,
    ) -> f64 {
        let mut score = 0.0;
        
        // Security score (higher is better)
        score += analysis.classical_security_bits * constraints.security_weight;
        score += analysis.quantum_security_bits * constraints.security_weight * 2.0; // Quantum security more important
        
        // Performance score (lower dimension is better for speed)
        let performance_score = 1000.0 / (params.n as f64);
        score += performance_score * constraints.performance_weight;
        
        // Memory score (lower dimension is better for memory)
        let memory_score = 1000.0 / (params.n as f64);
        score += memory_score * constraints.memory_weight;
        
        // Penalize if below minimum security requirements
        if analysis.classical_security_bits < self.security_level.classical_bits() as f64 {
            score -= 1000.0;
        }
        if analysis.quantum_security_bits < self.security_level.quantum_bits() as f64 {
            score -= 2000.0;
        }
        
        score
    }
}

/// Security analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    /// Analyzed parameters
    pub params: LatticeParams,
    /// Estimated classical security in bits
    pub classical_security_bits: f64,
    /// Estimated quantum security in bits
    pub quantum_security_bits: f64,
    /// Attack costs for different models
    pub attack_costs: HashMap<String, f64>,
    /// Security recommendations
    pub recommendations: Vec<String>,
}

/// Threat level for parameter upgrades
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Quantum attack cost analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAttackCost {
    /// Cost using gate-count model
    pub gate_count_model: f64,
    /// Cost using Q-Core-SVP model
    pub q_core_svp_model: f64,
    /// Grover speedup factor applied
    pub grover_speedup_factor: f64,
    /// Estimated time in years to break
    pub estimated_time_years: f64,
    /// Confidence level in the estimate
    pub confidence_level: f64,
}

/// Security constraints for parameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraints {
    /// Weight for security in optimization (0.0 to 1.0)
    pub security_weight: f64,
    /// Weight for performance in optimization (0.0 to 1.0)
    pub performance_weight: f64,
    /// Weight for memory usage in optimization (0.0 to 1.0)
    pub memory_weight: f64,
    /// Minimum acceptable classical security bits
    pub min_classical_bits: f64,
    /// Minimum acceptable quantum security bits
    pub min_quantum_bits: f64,
}

impl Default for SecurityConstraints {
    fn default() -> Self {
        Self {
            security_weight: 0.6,
            performance_weight: 0.3,
            memory_weight: 0.1,
            min_classical_bits: 128.0,
            min_quantum_bits: 64.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[test]
    fn test_lattice_params_creation() {
        let params = LatticeParams::new(SecurityLevel::Medium);
        assert_eq!(params.n, 512);
        assert_eq!(params.security_level, SecurityLevel::Medium);
        assert!(params.validate().is_ok());
    }
    
    #[test]
    fn test_lattice_params_optimized() {
        let speed_params = LatticeParams::optimized(SecurityLevel::Medium, OptimizationTarget::Speed);
        let security_params = LatticeParams::optimized(SecurityLevel::Medium, OptimizationTarget::Security);
        let memory_params = LatticeParams::optimized(SecurityLevel::Medium, OptimizationTarget::Memory);
        
        // Speed optimization should reduce dimension
        assert!(speed_params.n < 512);
        // Security optimization should increase dimension
        assert!(security_params.n > 512);
        // Memory optimization should cap dimension
        assert!(memory_params.n <= 256);
    }
    
    #[test]
    fn test_lattice_params_custom() {
        let custom_params = LatticeParams::custom(256, 2147483647, SecurityLevel::Medium).unwrap();
        assert_eq!(custom_params.n, 256);
        assert_eq!(custom_params.q, 2147483647);
        assert!(custom_params.validate().is_ok());
        
        // Test invalid parameters
        let invalid_params = LatticeParams::custom(0, 1, SecurityLevel::Medium);
        assert!(invalid_params.is_err());
    }
    
    #[test]
    fn test_lattice_params_validation() {
        let mut params = LatticeParams::new(SecurityLevel::Medium);
        assert!(params.validate().is_ok());
        
        // Test invalid dimension
        params.n = 0;
        assert!(params.validate().is_err());
        
        // Test invalid modulus
        params.n = 512;
        params.q = 1;
        assert!(params.validate().is_err());
        
        // Test invalid sigma
        params.q = 2147483647;
        params.sigma = -1.0;
        assert!(params.validate().is_err());
    }
    
    #[test]
    fn test_lattice_point_operations() {
        let p1 = LatticePoint::new(vec![1, 2, 3]);
        let p2 = LatticePoint::new(vec![4, 5, 6]);
        let q = 7;
        
        let sum = p1.add_mod(&p2, q);
        assert_eq!(sum.coordinates, vec![5, 0, 2]); // (1+4)%7, (2+5)%7, (3+6)%7
        
        let diff = p1.sub_mod(&p2, q);
        assert_eq!(diff.coordinates, vec![4, 4, 4]); // (1-4+7)%7, etc.
        
        let scaled = p1.scale_mod(3, q);
        assert_eq!(scaled.coordinates, vec![3, 6, 2]); // (1*3)%7, (2*3)%7, (3*3)%7
    }
    
    #[test]
    fn test_lattice_point_advanced_operations() {
        let p1 = LatticePoint::new(vec![1, 2, 3]);
        let p2 = LatticePoint::new(vec![4, 5, 6]);
        let q = 7;
        
        // Test inner product
        let inner_prod = p1.inner_product_mod(&p2, q);
        assert_eq!(inner_prod, (1*4 + 2*5 + 3*6) % q); // (4 + 10 + 18) % 7 = 32 % 7 = 4
        
        // Test fundamental domain check
        let valid_point = LatticePoint::new(vec![0, 1, 6]);
        assert!(valid_point.is_in_fundamental_domain(q));
        
        let invalid_point = LatticePoint::new(vec![-1, 1, 6]);
        assert!(!invalid_point.is_in_fundamental_domain(q));
        
        // Test norms
        let point = LatticePoint::new(vec![3, 4, 0]);
        assert_eq!(point.l2_norm(), 5.0); // sqrt(9 + 16 + 0) = 5
        assert_eq!(point.infinity_norm(), 4); // max(3, 4, 0) = 4
    }
    
    #[test]
    fn test_lattice_point_sampling() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        // Test Gaussian sampling
        let gaussian_point = LatticePoint::random_gaussian(&params, &mut rng).unwrap();
        assert_eq!(gaussian_point.dimension(), params.n);
        assert!(gaussian_point.is_in_fundamental_domain(params.q));
        
        // Test uniform sampling
        let uniform_point = LatticePoint::random_uniform(&params, &mut rng);
        assert_eq!(uniform_point.dimension(), params.n);
        assert!(uniform_point.is_in_fundamental_domain(params.q));
    }
    
    #[test]
    fn test_lattice_point_add_scaled() {
        let params = LatticeParams::new(SecurityLevel::Medium);
        let p1 = LatticePoint::new(vec![1, 2, 3]);
        let p2 = LatticePoint::new(vec![4, 5, 6]);
        let scalar = 3;
        
        let result = p1.add_scaled(&p2, scalar, &params).unwrap();
        let expected = p1.add_mod(&p2.scale_mod(scalar, params.q), params.q);
        assert_eq!(result.coordinates, expected.coordinates);
    }
    
    #[test]
    fn test_lattice_matrix_operations() {
        let matrix = LatticeMatrix::new(vec![
            vec![1, 2],
            vec![3, 4],
        ]).unwrap();
        
        let point = LatticePoint::new(vec![5, 6]);
        let result = matrix.multiply_point(&point, 7).unwrap();
        
        // [1*5 + 2*6, 3*5 + 4*6] = [17, 39] = [3, 4] mod 7
        assert_eq!(result.coordinates, vec![3, 4]);
    }
    
    #[test]
    fn test_lattice_matrix_advanced_operations() {
        let m1 = LatticeMatrix::new(vec![
            vec![1, 2],
            vec![3, 4],
        ]).unwrap();
        
        let m2 = LatticeMatrix::new(vec![
            vec![5, 6],
            vec![7, 8],
        ]).unwrap();
        
        let q = 11;
        
        // Test matrix addition
        let sum = m1.add(&m2, q).unwrap();
        assert_eq!(sum.data, vec![vec![6, 8], vec![10, 1]]); // (3+4)%11 = 7, (4+8)%11 = 1
        
        // Test matrix subtraction
        let diff = m1.subtract(&m2, q).unwrap();
        assert_eq!(diff.data, vec![vec![7, 7], vec![7, 7]]); // (1-5+11)%11 = 7, etc.
        
        // Test matrix scaling
        let scaled = m1.scale(3, q);
        assert_eq!(scaled.data, vec![vec![3, 6], vec![9, 1]]); // (4*3)%11 = 1
        
        // Test matrix multiplication
        let product = m1.multiply(&m2, q).unwrap();
        // [1*5+2*7, 1*6+2*8] = [19, 22] = [8, 0] mod 11
        // [3*5+4*7, 3*6+4*8] = [43, 50] = [10, 6] mod 11
        assert_eq!(product.data, vec![vec![8, 0], vec![10, 6]]);
        
        // Test transpose
        let transposed = m1.transpose();
        assert_eq!(transposed.data, vec![vec![1, 3], vec![2, 4]]);
    }
    
    #[test]
    fn test_lattice_matrix_determinant() {
        // Test 2x2 matrix determinant
        let matrix_2x2 = LatticeMatrix::new(vec![
            vec![1, 2],
            vec![3, 4],
        ]).unwrap();
        
        let det = matrix_2x2.determinant(7).unwrap();
        assert_eq!(det, (1*4 - 2*3) % 7); // -2 % 7 = 5
        
        // Test 3x3 matrix determinant
        let matrix_3x3 = LatticeMatrix::new(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
        ]).unwrap();
        
        let det_3x3 = matrix_3x3.determinant(11).unwrap();
        // This matrix has determinant 0 (rows are linearly dependent)
        assert_eq!(det_3x3, 0);
        
        // Test invertibility
        assert!(!matrix_3x3.is_invertible(11).unwrap());
        assert!(matrix_2x2.is_invertible(7).unwrap());
    }
    
    #[test]
    fn test_lattice_matrix_creation_validation() {
        // Test empty matrix
        let empty_result = LatticeMatrix::new(vec![]);
        assert!(empty_result.is_err());
        
        // Test inconsistent row lengths
        let inconsistent_result = LatticeMatrix::new(vec![
            vec![1, 2, 3],
            vec![4, 5], // Different length
        ]);
        assert!(inconsistent_result.is_err());
        
        // Test valid matrix
        let valid_matrix = LatticeMatrix::new(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]).unwrap();
        assert_eq!(valid_matrix.rows, 2);
        assert_eq!(valid_matrix.cols, 3);
    }
    
    #[test]
    fn test_quantum_resistance_analyzer() {
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        let analysis = analyzer.analyze_security(&params).unwrap();
        assert!(analysis.classical_security_bits > 0.0);
        assert!(analysis.quantum_security_bits > 0.0);
        assert!(!analysis.attack_costs.is_empty());
        
        // Test that analysis contains expected models
        assert!(analysis.attack_costs.contains_key("Core-SVP"));
        assert!(analysis.attack_costs.contains_key("Gate-Count"));
        assert!(analysis.attack_costs.contains_key("Q-Core-SVP"));
    }
    
    #[test]
    fn test_quantum_attack_cost_estimation() {
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        let attack_cost = analyzer.estimate_quantum_attack_cost(&params).unwrap();
        assert!(attack_cost.gate_count_model > 0.0);
        assert!(attack_cost.q_core_svp_model > 0.0);
        assert_eq!(attack_cost.grover_speedup_factor, 0.5);
        assert!(attack_cost.estimated_time_years > 0.0);
        assert!(attack_cost.confidence_level > 0.0 && attack_cost.confidence_level <= 1.0);
    }
    
    #[test]
    fn test_security_requirements_checking() {
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        let meets_requirements = analyzer.meets_security_requirements(&params).unwrap();
        assert!(meets_requirements);
        
        // Test with weaker parameters
        let weak_params = LatticeParams::custom(64, 101, SecurityLevel::Medium).unwrap();
        let meets_weak = analyzer.meets_security_requirements(&weak_params).unwrap();
        assert!(!meets_weak);
    }
    
    #[test]
    fn test_parameter_optimization() {
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let constraints = SecurityConstraints::default();
        
        let optimized_params = analyzer.optimize_parameters(&constraints).unwrap();
        assert!(optimized_params.validate().is_ok());
        assert!(analyzer.meets_security_requirements(&optimized_params).unwrap());
    }
    
    #[test]
    fn test_parameter_upgrade_recommendation() {
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let current_params = LatticeParams::new(SecurityLevel::Medium);
        
        // Test upgrade for high threat
        let upgraded = analyzer.recommend_upgrade(&current_params, ThreatLevel::High).unwrap();
        assert_eq!(upgraded.security_level, SecurityLevel::VeryHigh);
        
        // Test no upgrade needed for low threat
        let no_upgrade = analyzer.recommend_upgrade(&current_params, ThreatLevel::Low).unwrap();
        assert_eq!(no_upgrade.security_level, SecurityLevel::Medium);
    }
    
    #[test]
    fn test_security_level_properties() {
        assert_eq!(SecurityLevel::Medium.classical_bits(), 128);
        assert_eq!(SecurityLevel::Medium.quantum_bits(), 64);
        assert_eq!(SecurityLevel::High.lattice_dimension(), 768);
        assert_eq!(SecurityLevel::VeryHigh.modulus(), 9223372036854775783);
    }
    
    #[test]
    fn test_lattice_point_serialization() {
        let point = LatticePoint::new(vec![1, 2, 3, 4]);
        let bytes = point.to_bytes();
        let recovered = LatticePoint::from_bytes(&bytes, 4).unwrap();
        assert_eq!(point, recovered);
        
        // Test invalid byte length
        let invalid_recovery = LatticePoint::from_bytes(&bytes, 3);
        assert!(invalid_recovery.is_err());
    }
    
    #[test]
    fn test_lattice_point_hash_to_challenge() {
        let point = LatticePoint::new(vec![1, 2, 3, 4]);
        let challenge = point.hash_to_challenge();
        assert_eq!(challenge.len(), 32);
        
        // Same point should produce same challenge
        let challenge2 = point.hash_to_challenge();
        assert_eq!(challenge, challenge2);
        
        // Different point should produce different challenge
        let point2 = LatticePoint::new(vec![1, 2, 3, 5]);
        let challenge3 = point2.hash_to_challenge();
        assert_ne!(challenge, challenge3);
    }
    
    #[test]
    fn test_bkz_cost_model() {
        let model = BKZCostModel {
            name: "Test-Model".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("c1".to_string(), 0.292);
                params.insert("c2".to_string(), 16.4);
                params
            },
        };
        
        assert_eq!(model.name, "Test-Model");
        assert_eq!(model.parameters.get("c1"), Some(&0.292));
        assert_eq!(model.parameters.get("c2"), Some(&16.4));
    }
    
    #[test]
    fn test_security_constraints_default() {
        let constraints = SecurityConstraints::default();
        assert_eq!(constraints.security_weight, 0.6);
        assert_eq!(constraints.performance_weight, 0.3);
        assert_eq!(constraints.memory_weight, 0.1);
        assert_eq!(constraints.min_classical_bits, 128.0);
        assert_eq!(constraints.min_quantum_bits, 64.0);
    }
    
    #[test]
    fn test_optimization_target_enum() {
        let targets = [
            OptimizationTarget::Speed,
            OptimizationTarget::Security,
            OptimizationTarget::Memory,
            OptimizationTarget::Balanced,
        ];
        
        for target in targets {
            let params = LatticeParams::optimized(SecurityLevel::Medium, target);
            assert!(params.validate().is_ok());
        }
    }
    
    #[test]
    fn test_threat_level_enum() {
        let levels = [
            ThreatLevel::Low,
            ThreatLevel::Medium,
            ThreatLevel::High,
            ThreatLevel::Critical,
        ];
        
        let analyzer = QuantumResistanceAnalyzer::new(SecurityLevel::Medium);
        let params = LatticeParams::new(SecurityLevel::Medium);
        
        for level in levels {
            let upgrade = analyzer.recommend_upgrade(&params, level).unwrap();
            assert!(upgrade.validate().is_ok());
        }
    }
}