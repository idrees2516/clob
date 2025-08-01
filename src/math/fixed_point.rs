// Fixed-point arithmetic for deterministic, circuit-friendly calculations
// 18 decimals of precision (1e18 scaling)
use std::ops::{Add, Sub, Mul, Div};
use std::fmt;
use sha2::{Sha256, Digest};
use bincode::{serialize, deserialize};

const SCALE: i128 = 1_000_000_000_000_000_000;

// Export constants for use in other modules
pub const FIXED_POINT_SCALE: u64 = 1_000_000_000_000_000_000;
pub const FIXED_POINT_BITS: u32 = 64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint(pub i128);

impl FixedPoint {
    pub fn new(val: i128) -> Self {
        FixedPoint(val * SCALE)
    }
    pub fn from_float(val: f64) -> Self {
        FixedPoint((val * SCALE as f64).round() as i128)
    }
    pub fn to_float(self) -> f64 {
        self.0 as f64 / SCALE as f64
    }
    pub fn from_parts(int: i128, frac: i128) -> Self {
        FixedPoint(int * SCALE + frac)
    }
    pub fn zero() -> Self { FixedPoint(0) }
    pub fn one() -> Self { FixedPoint(SCALE) }
    pub fn from_int(val: i32) -> Self { FixedPoint(val as i128 * SCALE) }
    pub fn from_raw(val: i64) -> Self { FixedPoint(val as i128) }
    pub fn abs(self) -> Self { FixedPoint(self.0.abs()) }
    pub fn max(self, other: Self) -> Self { FixedPoint(self.0.max(other.0)) }
    pub fn min(self, other: Self) -> Self { FixedPoint(self.0.min(other.0)) }
    pub fn sqrt(self) -> Self {
        if self.0 <= 0 { return FixedPoint::zero(); }
        // Newton-Raphson method for square root
        let mut x = FixedPoint(self.0 / 2);
        for _ in 0..10 {
            let x_new = (x + self / x) / FixedPoint::from_float(2.0);
            if (x_new - x).abs() < FixedPoint::from_float(1e-10) {
                break;
            }
            x = x_new;
        }
        x
    }
    pub fn pow(self, exp: Self) -> Self {
        // Use exp(ln(x) * y) for x^y
        if self <= FixedPoint::zero() {
            return FixedPoint::zero();
        }
        (self.ln() * exp).exp()
    }
    /// Fixed-point exponential using 8-term Taylor expansion (for small x)
    pub fn exp(self) -> Self {
        // exp(x) ≈ 1 + x + x^2/2! + x^3/3! + ... + x^8/8!
        let mut term = FixedPoint::one();
        let mut sum = FixedPoint::one();
        for i in 1..=8 {
            term = term * self / FixedPoint::from_float(i as f64);
            sum = sum + term;
        }
        sum
    }
    /// Fixed-point natural logarithm using 8-term series for ln(1+x), x in (-1,1)
    pub fn ln(self) -> Self {
        if self <= FixedPoint::zero() {
            return FixedPoint::from_float(-10.0); // Approximate -infinity
        }
        if self == FixedPoint::one() {
            return FixedPoint::zero();
        }
        
        // For values far from 1, use properties of logarithms
        let mut x = self;
        let mut result = FixedPoint::zero();
        
        // Normalize to range near 1
        while x > FixedPoint::from_float(2.0) {
            x = x / FixedPoint::from_float(2.0);
            result = result + FixedPoint::from_float(2.0_f64.ln());
        }
        
        // ln(1+y) series where y = x - 1
        let y = x - FixedPoint::one();
        let mut term = y;
        let mut sum = y;
        let mut sign = -1;
        for i in 2..=8 {
            term = term * y;
            let add = term / FixedPoint::from_float(i as f64);
            if sign > 0 {
                sum = sum + add;
            } else {
                sum = sum - add;
            }
            sign *= -1;
        }
        result + sum
    }
}

impl Add for FixedPoint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        FixedPoint(self.0 + rhs.0)
    }
}
impl Sub for FixedPoint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        FixedPoint(self.0 - rhs.0)
    }
}
impl Mul for FixedPoint {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        FixedPoint((self.0 * rhs.0) / SCALE)
    }
}
impl Div for FixedPoint {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        FixedPoint((self.0 * SCALE) / rhs.0)
    }
}

impl fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.18}", self.to_float())
    }
}

/// Deterministic PRNG: XorShift64*
pub struct DeterministicRng {
    state: u64,
}
impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        // Xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn next_fixed(&mut self) -> FixedPoint {
        // Uniform [0, 1)
        let val = self.next_u64() >> 1; // 63 bits
        FixedPoint((val as i128) * (SCALE / (1u128 << 63) as i128))
    }
    pub fn next_double(&mut self) -> f64 {
        self.next_fixed().to_float()
    }
    pub fn next_int_range(&mut self, min: i32, max: i32) -> i32 {
        let range = (max - min + 1) as u64;
        let val = self.next_u64() % range;
        min + val as i32
    }
    pub fn next_normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        static mut SPARE: Option<f64> = None;
        static mut HAS_SPARE: bool = false;
        
        unsafe {
            if HAS_SPARE {
                HAS_SPARE = false;
                return mean + std_dev * SPARE.unwrap();
            }
        }
        
        let u1 = self.next_double();
        let u2 = self.next_double();
        
        let mag = std_dev * (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        
        unsafe {
            SPARE = Some(z1);
            HAS_SPARE = true;
        }
        
        mean + z0
    }
    pub fn next_normal_fixed(&mut self, mean: FixedPoint, std_dev: FixedPoint) -> FixedPoint {
        let normal = self.next_normal(mean.to_float(), std_dev.to_float());
        FixedPoint::from_float(normal)
    }
    pub fn next_exponential_fixed(&mut self, rate: FixedPoint) -> FixedPoint {
        let u = self.next_fixed();
        -u.ln() / rate
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MerkleNode {
    pub hash: [u8; 32],
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    pub fn new_leaf(data: &[u8]) -> Self {
        let hash = Sha256::digest(data);
        MerkleNode { hash: hash.into(), left: None, right: None }
    }
    pub fn new_internal(left: MerkleNode, right: MerkleNode) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(&left.hash);
        hasher.update(&right.hash);
        let hash = hasher.finalize();
        MerkleNode { hash: hash.into(), left: Some(Box::new(left)), right: Some(Box::new(right)) }
    }
}

pub struct MerkleTree {
    pub root: MerkleNode,
    pub leaves: Vec<MerkleNode>,
}

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<MerkleNode>) -> Self {
        fn build(nodes: Vec<MerkleNode>) -> MerkleNode {
            if nodes.len() == 1 {
                return nodes[0].clone();
            }
            let mut next_level = Vec::new();
            for i in (0..nodes.len()).step_by(2) {
                if i + 1 < nodes.len() {
                    next_level.push(MerkleNode::new_internal(nodes[i].clone(), nodes[i+1].clone()));
                } else {
                    next_level.push(nodes[i].clone());
                }
            }
            build(next_level)
        }
        let root = build(leaves.clone());
        MerkleTree { root, leaves }
    }
    pub fn root_hash(&self) -> [u8; 32] {
        self.root.hash
    }
    /// Serialize the Merkle tree leaves to bytes (for DA/state snapshot)
    pub fn serialize_leaves(&self) -> Vec<u8> {
        let leaf_hashes: Vec<[u8; 32]> = self.leaves.iter().map(|n| n.hash).collect();
        serialize(&leaf_hashes).unwrap()
    }
    /// Deserialize leaves and reconstruct the Merkle tree
    pub fn from_serialized_leaves(data: &[u8]) -> Self {
        let leaf_hashes: Vec<[u8; 32]> = deserialize(data).unwrap();
        let leaves: Vec<MerkleNode> = leaf_hashes.into_iter().map(|h| MerkleNode { hash: h, left: None, right: None }).collect();
        MerkleTree::from_leaves(leaves)
    }
}

/// Verify a state transition given old root, new root, and a proof (placeholder)
pub fn verify_state_transition(old_root: [u8; 32], new_root: [u8; 32], _proof: &[u8]) -> bool {
    // In a real system, proof would be a Merkle proof or zk proof
    // Here, just check roots differ (placeholder)
    old_root != new_root
}

// --- Mathematical Framework ---

/// Geometric Brownian Motion (GBM) SDE solver
pub struct GBMParams {
    pub mu: FixedPoint,
    pub sigma: FixedPoint,
    pub dt: FixedPoint,
}

pub fn simulate_gbm(
    s0: FixedPoint,
    n_steps: usize,
    params: &GBMParams,
    rng: &mut crate::math::fixed_point::DeterministicRng,
) -> Vec<FixedPoint> {
    let mut s = s0;
    let mut path = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        // dS = mu*S*dt + sigma*S*dW
        let z = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0); // Uniform(-1,1)
        let drift = params.mu * s * params.dt;
        let diffusion = params.sigma * s * z * params.dt.sqrt();
        s = s + drift + diffusion;
        path.push(s);
    }
    path
}

/// Hawkes process (exponential kernel)
pub struct HawkesParams {
    pub base: FixedPoint,
    pub alpha: FixedPoint,
    pub beta: FixedPoint,
    pub dt: FixedPoint,
}

pub fn simulate_hawkes(
    n_steps: usize,
    params: &HawkesParams,
    rng: &mut crate::math::fixed_point::DeterministicRng,
) -> Vec<u32> {
    let mut intensity = params.base;
    let mut events = vec![0u32; n_steps];
    for t in 0..n_steps {
        let p = intensity * params.dt;
        if rng.next_fixed().to_float() < p.to_float() {
            events[t] = 1;
            intensity = params.base + params.alpha;
        } else {
            intensity = params.base + (intensity - params.base) * (-params.beta * params.dt).exp();
        }
    }
    events
}

// Placeholder for rough volatility (fractional Brownian motion)
pub struct RoughVolParams {
    pub hurst: FixedPoint,
    pub sigma: FixedPoint,
    pub dt: FixedPoint,
}

pub fn simulate_rough_vol(_n_steps: usize, _params: &RoughVolParams, _rng: &mut crate::math::fixed_point::DeterministicRng) -> Vec<FixedPoint> {
    // TODO: Implement fractional Brownian motion
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() {
        let a = FixedPoint::from_float(1.5);
        let b = FixedPoint::from_float(2.25);
        assert_eq!((a + b).to_float(), 3.75);
    }
    #[test]
    fn test_sub() {
        let a = FixedPoint::from_float(5.0);
        let b = FixedPoint::from_float(2.5);
        assert_eq!((a - b).to_float(), 2.5);
    }
    #[test]
    fn test_mul() {
        let a = FixedPoint::from_float(1.5);
        let b = FixedPoint::from_float(2.0);
        assert!(((a * b).to_float() - 3.0).abs() < 1e-12);
    }
    #[test]
    fn test_div() {
        let a = FixedPoint::from_float(3.0);
        let b = FixedPoint::from_float(1.5);
        assert!(((a / b).to_float() - 2.0).abs() < 1e-12);
    }
    #[test]
    fn test_zero_one() {
        assert_eq!(FixedPoint::zero().to_float(), 0.0);
        assert_eq!(FixedPoint::one().to_float(), 1.0);
    }
    #[test]
    fn test_exp_ln() {
        let x = FixedPoint::from_float(0.1);
        let exp_x = x.exp().to_float();
        let std_exp = (0.1f64).exp();
        assert!((exp_x - std_exp).abs() < 1e-3);
        let y = FixedPoint::from_float(1.1);
        let ln_y = y.ln().to_float();
        let std_ln = (1.1f64).ln();
        assert!((ln_y - std_ln).abs() < 1e-3);
    }
    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
        let val = rng1.next_fixed().to_float();
        assert!(val >= 0.0 && val < 1.0);
    }
}

#[cfg(test)]
mod merkle_tests {
    use super::*;
    #[test]
    fn test_merkle_tree() {
        let data = vec![b"a", b"b", b"c", b"d"];
        let leaves: Vec<MerkleNode> = data.iter().map(|d| MerkleNode::new_leaf(d)).collect();
        let tree = MerkleTree::from_leaves(leaves);
        let hash = tree.root_hash();
        assert_eq!(hash.len(), 32);
    }
}

#[cfg(test)]
mod merkle_serialization_tests {
    use super::*;
    #[test]
    fn test_serialize_deserialize() {
        let data = vec![b"x", b"y", b"z", b"w"];
        let leaves: Vec<MerkleNode> = data.iter().map(|d| MerkleNode::new_leaf(d)).collect();
        let tree = MerkleTree::from_leaves(leaves);
        let bytes = tree.serialize_leaves();
        let tree2 = MerkleTree::from_serialized_leaves(&bytes);
        assert_eq!(tree.root_hash(), tree2.root_hash());
    }
    #[test]
    fn test_state_transition_verification() {
        let data1 = vec![b"a", b"b", b"c", b"d"];
        let data2 = vec![b"a", b"b", b"c", b"e"];
        let leaves1: Vec<MerkleNode> = data1.iter().map(|d| MerkleNode::new_leaf(d)).collect();
        let leaves2: Vec<MerkleNode> = data2.iter().map(|d| MerkleNode::new_leaf(d)).collect();
        let tree1 = MerkleTree::from_leaves(leaves1);
        let tree2 = MerkleTree::from_leaves(leaves2);
        assert!(verify_state_transition(tree1.root_hash(), tree2.root_hash(), b"dummy"));
    }
}

#[cfg(test)]
mod math_framework_tests {
    use super::*;
    #[test]
    fn test_gbm_simulation() {
        let mut rng = DeterministicRng::new(42);
        let params = GBMParams {
            mu: FixedPoint::from_float(0.01),
            sigma: FixedPoint::from_float(0.1),
            dt: FixedPoint::from_float(0.01),
        };
        let path = simulate_gbm(FixedPoint::from_float(100.0), 10, &params, &mut rng);
        assert_eq!(path.len(), 10);
    }
    #[test]
    fn test_hawkes_simulation() {
        let mut rng = DeterministicRng::new(123);
        let params = HawkesParams {
            base: FixedPoint::from_float(0.1),
            alpha: FixedPoint::from_float(0.5),
            beta: FixedPoint::from_float(1.0),
            dt: FixedPoint::from_float(0.01),
        };
        let events = simulate_hawkes(20, &params, &mut rng);
        assert_eq!(events.len(), 20);
    }
} 