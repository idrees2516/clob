// Fixed-point arithmetic for deterministic, circuit-friendly calculations
// 18 decimals of precision (1e18 scaling)
use std::ops::{Add, Sub, Mul, Div};
use std::fmt;
use sha2::{Sha256, Digest};
use bincode::{serialize, deserialize};

/// Fixed-point precision configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedPointPrecision {
    /// 64-bit with 18 decimal places (1e18 scaling)
    Standard64,
    /// 128-bit with 36 decimal places (1e36 scaling)
    Extended128,
    /// Custom precision
    Custom { scale: i128, bits: u32 },
}

impl FixedPointPrecision {
    pub fn scale(&self) -> i128 {
        match self {
            FixedPointPrecision::Standard64 => 1_000_000_000_000_000_000,
            FixedPointPrecision::Extended128 => {
                // 1e36 for 128-bit precision
                let mut scale = 1i128;
                for _ in 0..36 {
                    scale *= 10;
                }
                scale
            }
            FixedPointPrecision::Custom { scale, .. } => *scale,
        }
    }

    pub fn bits(&self) -> u32 {
        match self {
            FixedPointPrecision::Standard64 => 64,
            FixedPointPrecision::Extended128 => 128,
            FixedPointPrecision::Custom { bits, .. } => *bits,
        }
    }
}

const SCALE: i128 = 1_000_000_000_000_000_000;

// Export constants for use in other modules
pub const FIXED_POINT_SCALE: u64 = 1_000_000_000_000_000_000;
pub const FIXED_POINT_BITS: u32 = 64;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedPoint {
    value: i128,
    precision: FixedPointPrecision,
}

/// Rounding modes for fixed-point arithmetic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingMode {
    /// Round to nearest, ties to even
    NearestEven,
    /// Round toward zero (truncate)
    TowardZero,
    /// Round toward positive infinity
    TowardPositive,
    /// Round toward negative infinity
    TowardNegative,
}

impl FixedPoint {
    pub fn new(val: i128) -> Self {
        Self::with_precision(val, FixedPointPrecision::Standard64)
    }

    pub fn with_precision(val: i128, precision: FixedPointPrecision) -> Self {
        FixedPoint {
            value: val * precision.scale(),
            precision,
        }
    }

    pub fn from_float(val: f64) -> Self {
        Self::from_float_with_precision(val, FixedPointPrecision::Standard64, RoundingMode::NearestEven)
    }

    pub fn from_float_with_precision(val: f64, precision: FixedPointPrecision, rounding: RoundingMode) -> Self {
        let scale = precision.scale() as f64;
        let scaled = val * scale;
        
        let rounded = match rounding {
            RoundingMode::NearestEven => {
                let truncated = scaled.trunc();
                let frac = scaled - truncated;
                if frac.abs() < 0.5 {
                    truncated
                } else if frac.abs() > 0.5 {
                    if scaled > 0.0 { truncated + 1.0 } else { truncated - 1.0 }
                } else {
                    // Tie - round to even
                    if (truncated as i128) % 2 == 0 {
                        truncated
                    } else if scaled > 0.0 {
                        truncated + 1.0
                    } else {
                        truncated - 1.0
                    }
                }
            }
            RoundingMode::TowardZero => scaled.trunc(),
            RoundingMode::TowardPositive => scaled.ceil(),
            RoundingMode::TowardNegative => scaled.floor(),
        };

        FixedPoint {
            value: rounded as i128,
            precision,
        }
    }

    pub fn to_float(self) -> f64 {
        self.value as f64 / self.precision.scale() as f64
    }

    pub fn from_parts(int: i128, frac: i128) -> Self {
        let precision = FixedPointPrecision::Standard64;
        FixedPoint {
            value: int * precision.scale() + frac,
            precision,
        }
    }

    pub fn zero() -> Self {
        FixedPoint {
            value: 0,
            precision: FixedPointPrecision::Standard64,
        }
    }

    pub fn one() -> Self {
        FixedPoint {
            value: SCALE,
            precision: FixedPointPrecision::Standard64,
        }
    }

    pub fn from_int(val: i32) -> Self {
        FixedPoint {
            value: val as i128 * SCALE,
            precision: FixedPointPrecision::Standard64,
        }
    }

    pub fn from_raw(val: i64) -> Self {
        FixedPoint {
            value: val as i128,
            precision: FixedPointPrecision::Standard64,
        }
    }

    pub fn to_raw(self) -> u64 {
        self.value as u64
    }

    pub fn abs(self) -> Self {
        FixedPoint {
            value: self.value.abs(),
            precision: self.precision,
        }
    }

    pub fn max(self, other: Self) -> Self {
        if self.value >= other.value {
            self
        } else {
            other
        }
    }

    pub fn min(self, other: Self) -> Self {
        if self.value <= other.value {
            self
        } else {
            other
        }
    }

    /// Check for overflow in arithmetic operations
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.value.checked_add(other.value).map(|v| FixedPoint {
            value: v,
            precision: self.precision,
        })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        self.value.checked_sub(other.value).map(|v| FixedPoint {
            value: v,
            precision: self.precision,
        })
    }

    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let scale = self.precision.scale();
        self.value.checked_mul(other.value)
            .and_then(|v| v.checked_div(scale))
            .map(|v| FixedPoint {
                value: v,
                precision: self.precision,
            })
    }

    pub fn checked_div(self, other: Self) -> Option<Self> {
        if other.value == 0 {
            return None;
        }
        let scale = self.precision.scale();
        self.value.checked_mul(scale)
            .and_then(|v| v.checked_div(other.value))
            .map(|v| FixedPoint {
                value: v,
                precision: self.precision,
            })
    }

    /// Get the raw internal value
    pub fn raw_value(&self) -> i128 {
        self.value
    }

    /// Get the precision configuration
    pub fn precision(&self) -> FixedPointPrecision {
        self.precision
    }
    pub fn sqrt(self) -> Self {
        if self.value <= 0 {
            return FixedPoint::zero();
        }
        
        // Newton-Raphson method for square root with better convergence
        let mut x = FixedPoint {
            value: self.value / 2,
            precision: self.precision,
        };
        
        let tolerance = FixedPoint::from_float_with_precision(
            1e-15, 
            self.precision, 
            RoundingMode::NearestEven
        );
        
        for _ in 0..20 {
            let x_new = (x + self / x) / FixedPoint::from_float(2.0);
            if (x_new - x).abs() < tolerance {
                break;
            }
            x = x_new;
        }
        x
    }

    pub fn pow(self, exp: Self) -> Self {
        if self.value <= 0 {
            return FixedPoint::zero();
        }
        if exp == FixedPoint::zero() {
            return FixedPoint::one();
        }
        if exp == FixedPoint::one() {
            return self;
        }
        
        // Use exp(ln(x) * y) for x^y
        (self.ln() * exp).exp()
    }

    /// Enhanced exponential function with range reduction
    pub fn exp(self) -> Self {
        // Range reduction: exp(x) = exp(x/2^k)^(2^k)
        let mut x = self;
        let mut k = 0;
        
        // Reduce to |x| < 1
        while x.abs().to_float() > 1.0 {
            x = x / FixedPoint::from_float(2.0);
            k += 1;
        }
        
        // Taylor series for exp(x): 1 + x + x^2/2! + x^3/3! + ...
        let mut term = FixedPoint::one();
        let mut sum = FixedPoint::one();
        
        for i in 1..=20 {
            term = term * x / FixedPoint::from_float(i as f64);
            sum = sum + term;
            
            // Early termination if term becomes negligible
            if term.abs().to_float() < 1e-15 {
                break;
            }
        }
        
        // Square k times to undo range reduction
        for _ in 0..k {
            sum = sum * sum;
        }
        
        sum
    }

    /// Enhanced natural logarithm with better convergence
    pub fn ln(self) -> Self {
        if self.value <= 0 {
            return FixedPoint::from_float(-100.0); // Approximate -infinity
        }
        if self == FixedPoint::one() {
            return FixedPoint::zero();
        }
        
        let mut x = self;
        let mut result = FixedPoint::zero();
        
        // Range reduction using ln(2^k * y) = k*ln(2) + ln(y)
        let mut k = 0;
        let ln2 = FixedPoint::from_float(2.0_f64.ln());
        
        // Normalize to range [1, 2)
        while x >= FixedPoint::from_float(2.0) {
            x = x / FixedPoint::from_float(2.0);
            k += 1;
        }
        while x < FixedPoint::one() {
            x = x * FixedPoint::from_float(2.0);
            k -= 1;
        }
        
        // Use ln(1+y) series where y = x - 1, |y| < 1
        let y = x - FixedPoint::one();
        let mut term = y;
        let mut sum = y;
        
        for i in 2..=20 {
            term = term * y;
            let add = term / FixedPoint::from_float(i as f64);
            if i % 2 == 0 {
                sum = sum - add;
            } else {
                sum = sum + add;
            }
            
            // Early termination
            if add.abs().to_float() < 1e-15 {
                break;
            }
        }
        
        result + FixedPoint::from_float(k as f64) * ln2 + sum
    }

    /// Sine function using Taylor series
    pub fn sin(self) -> Self {
        // Range reduction to [-π, π]
        let two_pi = FixedPoint::from_float(2.0 * std::f64::consts::PI);
        let mut x = self;
        
        // Reduce to [-π, π]
        while x > FixedPoint::from_float(std::f64::consts::PI) {
            x = x - two_pi;
        }
        while x < FixedPoint::from_float(-std::f64::consts::PI) {
            x = x + two_pi;
        }
        
        // Taylor series: sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ...
        let x_squared = x * x;
        let mut term = x;
        let mut sum = x;
        
        for i in 1..=10 {
            term = term * x_squared / FixedPoint::from_float(((2 * i) * (2 * i + 1)) as f64);
            if i % 2 == 1 {
                sum = sum - term;
            } else {
                sum = sum + term;
            }
        }
        
        sum
    }

    /// Cosine function using Taylor series
    pub fn cos(self) -> Self {
        // cos(x) = sin(x + π/2)
        let pi_half = FixedPoint::from_float(std::f64::consts::PI / 2.0);
        (self + pi_half).sin()
    }

    /// Tangent function
    pub fn tan(self) -> Self {
        let cos_x = self.cos();
        if cos_x.abs().to_float() < 1e-10 {
            // Return large value for tan(π/2 + nπ)
            return if self.to_float() > 0.0 {
                FixedPoint::from_float(1e10)
            } else {
                FixedPoint::from_float(-1e10)
            };
        }
        self.sin() / cos_x
    }

    /// Arctangent function using Taylor series and range reduction
    pub fn atan(self) -> Self {
        let x = self;
        
        // Use atan(x) = π/2 - atan(1/x) for |x| > 1
        if x.abs() > FixedPoint::one() {
            let pi_half = FixedPoint::from_float(std::f64::consts::PI / 2.0);
            let sign = if x.to_float() > 0.0 { FixedPoint::one() } else { -FixedPoint::one() };
            return sign * pi_half - (FixedPoint::one() / x.abs()).atan();
        }
        
        // Taylor series: atan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
        let x_squared = x * x;
        let mut term = x;
        let mut sum = x;
        
        for i in 1..=15 {
            term = term * x_squared;
            let denominator = FixedPoint::from_float((2 * i + 1) as f64);
            let add = term / denominator;
            
            if i % 2 == 1 {
                sum = sum - add;
            } else {
                sum = sum + add;
            }
            
            if add.abs().to_float() < 1e-15 {
                break;
            }
        }
        
        sum
    }

    /// Hyperbolic sine function
    pub fn sinh(self) -> Self {
        // sinh(x) = (e^x - e^(-x)) / 2
        let exp_x = self.exp();
        let exp_neg_x = (-self).exp();
        (exp_x - exp_neg_x) / FixedPoint::from_float(2.0)
    }

    /// Hyperbolic cosine function
    pub fn cosh(self) -> Self {
        // cosh(x) = (e^x + e^(-x)) / 2
        let exp_x = self.exp();
        let exp_neg_x = (-self).exp();
        (exp_x + exp_neg_x) / FixedPoint::from_float(2.0)
    }

    /// Hyperbolic tangent function
    pub fn tanh(self) -> Self {
        // tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        let exp_x = self.exp();
        let exp_neg_x = (-self).exp();
        (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
    }

    /// Inverse hyperbolic sine
    pub fn asinh(self) -> Self {
        // asinh(x) = ln(x + sqrt(x^2 + 1))
        let x_squared = self * self;
        let sqrt_term = (x_squared + FixedPoint::one()).sqrt();
        (self + sqrt_term).ln()
    }

    /// Inverse hyperbolic cosine (for x >= 1)
    pub fn acosh(self) -> Self {
        if self < FixedPoint::one() {
            return FixedPoint::zero(); // Undefined for x < 1
        }
        // acosh(x) = ln(x + sqrt(x^2 - 1))
        let x_squared = self * self;
        let sqrt_term = (x_squared - FixedPoint::one()).sqrt();
        (self + sqrt_term).ln()
    }

    /// Inverse hyperbolic tangent (for |x| < 1)
    pub fn atanh(self) -> Self {
        if self.abs() >= FixedPoint::one() {
            return FixedPoint::zero(); // Undefined for |x| >= 1
        }
        // atanh(x) = (1/2) * ln((1+x)/(1-x))
        let one_plus_x = FixedPoint::one() + self;
        let one_minus_x = FixedPoint::one() - self;
        (one_plus_x / one_minus_x).ln() / FixedPoint::from_float(2.0)
    }

    /// Gamma function approximation using Stirling's approximation
    pub fn gamma(self) -> Self {
        if self <= FixedPoint::zero() {
            return FixedPoint::from_float(f64::INFINITY);
        }
        
        // For small values, use recurrence relation: Γ(x+1) = x*Γ(x)
        let mut x = self;
        let mut factor = FixedPoint::one();
        
        while x < FixedPoint::from_float(7.0) {
            factor = factor * x;
            x = x + FixedPoint::one();
        }
        
        // Stirling's approximation: Γ(x) ≈ √(2π/x) * (x/e)^x
        let two_pi = FixedPoint::from_float(2.0 * std::f64::consts::PI);
        let e = FixedPoint::from_float(std::f64::consts::E);
        
        let sqrt_term = (two_pi / x).sqrt();
        let power_term = (x / e).pow(x);
        
        let result = sqrt_term * power_term;
        result / factor
    }

    /// Error function approximation
    pub fn erf(self) -> Self {
        // Abramowitz and Stegun approximation
        let a1 = FixedPoint::from_float(0.254829592);
        let a2 = FixedPoint::from_float(-0.284496736);
        let a3 = FixedPoint::from_float(1.421413741);
        let a4 = FixedPoint::from_float(-1.453152027);
        let a5 = FixedPoint::from_float(1.061405429);
        let p = FixedPoint::from_float(0.3275911);
        
        let sign = if self.to_float() >= 0.0 { FixedPoint::one() } else { -FixedPoint::one() };
        let x = self.abs();
        
        let t = FixedPoint::one() / (FixedPoint::one() + p * x);
        let y = FixedPoint::one() - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }

    /// Complementary error function
    pub fn erfc(self) -> Self {
        FixedPoint::one() - self.erf()
    }

    /// Normal cumulative distribution function
    pub fn normal_cdf(self) -> Self {
        // Φ(x) = (1/2) * (1 + erf(x/√2))
        let sqrt_2 = FixedPoint::from_float(std::f64::consts::SQRT_2);
        let erf_term = (self / sqrt_2).erf();
        (FixedPoint::one() + erf_term) / FixedPoint::from_float(2.0)
    }

    /// Normal probability density function
    pub fn normal_pdf(self) -> Self {
        // φ(x) = (1/√(2π)) * exp(-x²/2)
        let two_pi = FixedPoint::from_float(2.0 * std::f64::consts::PI);
        let normalization = FixedPoint::one() / two_pi.sqrt();
        let exponent = -(self * self) / FixedPoint::from_float(2.0);
        normalization * exponent.exp()
    }

    /// Round to nearest integer
    pub fn round(self) -> Self {
        let fractional_part = self.value % self.precision.scale();
        let half_scale = self.precision.scale() / 2;
        
        if fractional_part >= half_scale {
            // Round up
            FixedPoint {
                value: self.value - fractional_part + self.precision.scale(),
                precision: self.precision,
            }
        } else {
            // Round down
            FixedPoint {
                value: self.value - fractional_part,
                precision: self.precision,
            }
        }
    }

    /// Floor function (round toward negative infinity)
    pub fn floor(self) -> Self {
        let fractional_part = self.value % self.precision.scale();
        if fractional_part == 0 || self.value >= 0 {
            FixedPoint {
                value: self.value - fractional_part,
                precision: self.precision,
            }
        } else {
            FixedPoint {
                value: self.value - fractional_part - self.precision.scale(),
                precision: self.precision,
            }
        }
    }

    /// Ceiling function (round toward positive infinity)
    pub fn ceil(self) -> Self {
        let fractional_part = self.value % self.precision.scale();
        if fractional_part == 0 {
            self
        } else if self.value >= 0 {
            FixedPoint {
                value: self.value - fractional_part + self.precision.scale(),
                precision: self.precision,
            }
        } else {
            FixedPoint {
                value: self.value - fractional_part,
                precision: self.precision,
            }
        }
    }

    /// Truncate toward zero
    pub fn trunc(self) -> Self {
        let fractional_part = self.value % self.precision.scale();
        FixedPoint {
            value: self.value - fractional_part,
            precision: self.precision,
        }
    }

    /// Get fractional part
    pub fn fract(self) -> Self {
        let fractional_part = self.value % self.precision.scale();
        FixedPoint {
            value: fractional_part,
            precision: self.precision,
        }
    }

    /// Modulo operation
    pub fn modulo(self, other: Self) -> Self {
        let quotient = (self / other).trunc();
        self - quotient * other
    }
}

impl Add for FixedPoint {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        FixedPoint {
            value: self.value + rhs.value,
            precision: self.precision,
        }
    }
}

impl Sub for FixedPoint {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        FixedPoint {
            value: self.value - rhs.value,
            precision: self.precision,
        }
    }
}

impl Mul for FixedPoint {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let scale = self.precision.scale();
        FixedPoint {
            value: (self.value * rhs.value) / scale,
            precision: self.precision,
        }
    }
}

impl Div for FixedPoint {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let scale = self.precision.scale();
        FixedPoint {
            value: (self.value * scale) / rhs.value,
            precision: self.precision,
        }
    }
}

impl std::ops::Neg for FixedPoint {
    type Output = Self;
    fn neg(self) -> Self {
        FixedPoint {
            value: -self.value,
            precision: self.precision,
        }
    }
}

impl fmt::Display for FixedPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.18}", self.to_float())
    }
}

/// PRNG Algorithm Types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PRNGAlgorithm {
    XorShift64Star,
    MersenneTwister,
    Xorshift128Plus,
}

/// Deterministic PRNG with multiple algorithms
pub struct DeterministicRng {
    algorithm: PRNGAlgorithm,
    xorshift_state: u64,
    mt_state: MersenneTwisterState,
    xorshift128_state: [u64; 2],
}

/// Mersenne Twister state (MT19937-64)
struct MersenneTwisterState {
    mt: [u64; 312],
    mti: usize,
}

impl MersenneTwisterState {
    fn new(seed: u64) -> Self {
        let mut mt = [0u64; 312];
        mt[0] = seed;
        for i in 1..312 {
            mt[i] = 6364136223846793005u64
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, mti: 312 }
    }

    fn next_u64(&mut self) -> u64 {
        const MAG01: [u64; 2] = [0, 0xB5026F5AA96619E9];
        
        if self.mti >= 312 {
            for i in 0..156 {
                let x = (self.mt[i] & 0x8000000000000000) | (self.mt[i + 1] & 0x7FFFFFFFFFFFFFFF);
                self.mt[i] = self.mt[i + 156] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
            }
            for i in 156..311 {
                let x = (self.mt[i] & 0x8000000000000000) | (self.mt[i + 1] & 0x7FFFFFFFFFFFFFFF);
                self.mt[i] = self.mt[i - 156] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
            }
            let x = (self.mt[311] & 0x8000000000000000) | (self.mt[0] & 0x7FFFFFFFFFFFFFFF);
            self.mt[311] = self.mt[155] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
            self.mti = 0;
        }

        let mut x = self.mt[self.mti];
        self.mti += 1;

        x ^= (x >> 29) & 0x5555555555555555;
        x ^= (x << 17) & 0x71D67FFFEDA60000;
        x ^= (x << 37) & 0xFFF7EEE000000000;
        x ^= x >> 43;

        x
    }
}

impl DeterministicRng {
    pub fn new(seed: u64) -> Self {
        Self::with_algorithm(seed, PRNGAlgorithm::XorShift64Star)
    }

    pub fn with_algorithm(seed: u64, algorithm: PRNGAlgorithm) -> Self {
        Self {
            algorithm,
            xorshift_state: seed,
            mt_state: MersenneTwisterState::new(seed),
            xorshift128_state: [seed, seed.wrapping_add(1)],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        match self.algorithm {
            PRNGAlgorithm::XorShift64Star => self.next_xorshift64_star(),
            PRNGAlgorithm::MersenneTwister => self.mt_state.next_u64(),
            PRNGAlgorithm::Xorshift128Plus => self.next_xorshift128_plus(),
        }
    }

    fn next_xorshift64_star(&mut self) -> u64 {
        let mut x = self.xorshift_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.xorshift_state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn next_xorshift128_plus(&mut self) -> u64 {
        let mut s1 = self.xorshift128_state[0];
        let s0 = self.xorshift128_state[1];
        self.xorshift128_state[0] = s0;
        s1 ^= s1 << 23;
        self.xorshift128_state[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        self.xorshift128_state[1].wrapping_add(s0)
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

/// Box-Muller Generator for Gaussian random variables
pub struct BoxMullerGenerator {
    spare: Option<FixedPoint>,
    rng: DeterministicRng,
}

impl BoxMullerGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            spare: None,
            rng: DeterministicRng::new(seed),
        }
    }

    pub fn with_algorithm(seed: u64, algorithm: PRNGAlgorithm) -> Self {
        Self {
            spare: None,
            rng: DeterministicRng::with_algorithm(seed, algorithm),
        }
    }

    /// Generate standard normal random variable N(0,1)
    pub fn next_standard_normal(&mut self) -> FixedPoint {
        if let Some(spare) = self.spare.take() {
            return spare;
        }

        let u1 = self.rng.next_fixed();
        let u2 = self.rng.next_fixed();

        // Ensure u1 > 0 to avoid log(0)
        let u1 = if u1 <= FixedPoint::zero() {
            FixedPoint::from_float(1e-10)
        } else {
            u1
        };

        let mag = (-FixedPoint::from_float(2.0) * u1.ln()).sqrt();
        let two_pi_u2 = FixedPoint::from_float(2.0 * std::f64::consts::PI) * u2;
        
        let z0 = mag * FixedPoint::from_float(two_pi_u2.to_float().cos());
        let z1 = mag * FixedPoint::from_float(two_pi_u2.to_float().sin());

        self.spare = Some(z1);
        z0
    }

    /// Generate normal random variable N(mean, variance)
    pub fn next_normal(&mut self, mean: FixedPoint, std_dev: FixedPoint) -> FixedPoint {
        mean + std_dev * self.next_standard_normal()
    }

    /// Generate correlated normal variables
    pub fn next_correlated_normals(&mut self, correlation: FixedPoint) -> (FixedPoint, FixedPoint) {
        let z1 = self.next_standard_normal();
        let z2 = self.next_standard_normal();
        
        let x1 = z1;
        let x2 = correlation * z1 + (FixedPoint::one() - correlation * correlation).sqrt() * z2;
        
        (x1, x2)
    }
}

/// Poisson Jump Generator for jump processes
pub struct PoissonJumpGenerator {
    rng: DeterministicRng,
    intensity_cache: Vec<(FixedPoint, FixedPoint)>, // (intensity, exp(-intensity))
}

impl PoissonJumpGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: DeterministicRng::new(seed),
            intensity_cache: Vec::new(),
        }
    }

    pub fn with_algorithm(seed: u64, algorithm: PRNGAlgorithm) -> Self {
        Self {
            rng: DeterministicRng::with_algorithm(seed, algorithm),
            intensity_cache: Vec::new(),
        }
    }

    /// Generate number of jumps in time interval dt with given intensity
    pub fn next_poisson_count(&mut self, intensity: FixedPoint, dt: FixedPoint) -> u32 {
        let lambda_dt = intensity * dt;
        
        // Use inverse transform method for small lambda*dt
        if lambda_dt.to_float() < 10.0 {
            self.inverse_transform_poisson(lambda_dt)
        } else {
            // Use normal approximation for large lambda*dt
            self.normal_approximation_poisson(lambda_dt)
        }
    }

    fn inverse_transform_poisson(&mut self, lambda: FixedPoint) -> u32 {
        let exp_neg_lambda = (-lambda).exp();
        let mut k = 0u32;
        let mut p = FixedPoint::one();
        
        loop {
            k += 1;
            let u = self.rng.next_fixed();
            p = p * u;
            if p <= exp_neg_lambda {
                return k - 1;
            }
            if k > 1000 {
                // Safety break to avoid infinite loops
                return (lambda.to_float() as u32).saturating_add(1);
            }
        }
    }

    fn normal_approximation_poisson(&mut self, lambda: FixedPoint) -> u32 {
        // For large lambda, Poisson(lambda) ≈ N(lambda, lambda)
        let mean = lambda;
        let std_dev = lambda.sqrt();
        
        let normal_sample = self.rng.next_normal_fixed(mean, std_dev);
        (normal_sample.to_float().max(0.0) as u32)
    }

    /// Generate jump times in interval [0, T] with given intensity
    pub fn generate_jump_times(&mut self, intensity: FixedPoint, time_horizon: FixedPoint) -> Vec<FixedPoint> {
        let mut jump_times = Vec::new();
        let mut current_time = FixedPoint::zero();
        
        while current_time < time_horizon {
            // Generate next inter-arrival time (exponential distribution)
            let u = self.rng.next_fixed();
            let inter_arrival = -u.ln() / intensity;
            current_time = current_time + inter_arrival;
            
            if current_time < time_horizon {
                jump_times.push(current_time);
            }
        }
        
        jump_times
    }

    /// Generate compound Poisson process (jumps with sizes)
    pub fn generate_compound_jumps<F>(
        &mut self,
        intensity: FixedPoint,
        time_horizon: FixedPoint,
        jump_size_generator: F,
    ) -> Vec<(FixedPoint, FixedPoint)>
    where
        F: Fn(&mut DeterministicRng) -> FixedPoint,
    {
        let jump_times = self.generate_jump_times(intensity, time_horizon);
        let mut compound_jumps = Vec::with_capacity(jump_times.len());
        
        for time in jump_times {
            let size = jump_size_generator(&mut self.rng);
            compound_jumps.push((time, size));
        }
        
        compound_jumps
    }
}

/// Quasi-Monte Carlo sequence generators for variance reduction
pub struct QuasiMonteCarloGenerator {
    sequence_type: QMCSequenceType,
    dimension: usize,
    current_index: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum QMCSequenceType {
    Sobol,
    Halton,
}

impl QuasiMonteCarloGenerator {
    pub fn new(sequence_type: QMCSequenceType, dimension: usize) -> Self {
        Self {
            sequence_type,
            dimension,
            current_index: 0,
        }
    }

    /// Generate next point in the quasi-Monte Carlo sequence
    pub fn next_point(&mut self) -> Vec<FixedPoint> {
        self.current_index += 1;
        match self.sequence_type {
            QMCSequenceType::Sobol => self.generate_sobol_point(),
            QMCSequenceType::Halton => self.generate_halton_point(),
        }
    }

    fn generate_sobol_point(&self) -> Vec<FixedPoint> {
        // Simplified Sobol sequence implementation
        // In production, use a proper Sobol sequence library
        let mut point = Vec::with_capacity(self.dimension);
        
        for d in 0..self.dimension {
            let mut value = 0.0;
            let mut base = 0.5;
            let mut n = self.current_index;
            
            while n > 0 {
                if n & 1 == 1 {
                    value += base;
                }
                base *= 0.5;
                n >>= 1;
            }
            
            // Apply Gray code for better uniformity
            let gray_code = self.current_index ^ (self.current_index >> 1);
            let mut gray_value = 0.0;
            let mut gray_base = 0.5;
            let mut g = gray_code;
            
            while g > 0 {
                if g & 1 == 1 {
                    gray_value += gray_base;
                }
                gray_base *= 0.5;
                g >>= 1;
            }
            
            // Combine with dimension-specific scrambling
            let scrambled = (gray_value + (d as f64) * 0.1) % 1.0;
            point.push(FixedPoint::from_float(scrambled));
        }
        
        point
    }

    fn generate_halton_point(&self) -> Vec<FixedPoint> {
        let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        let mut point = Vec::with_capacity(self.dimension);
        
        for d in 0..self.dimension {
            let base = primes[d % primes.len()];
            let value = self.van_der_corput_sequence(self.current_index, base);
            point.push(FixedPoint::from_float(value));
        }
        
        point
    }

    fn van_der_corput_sequence(&self, n: u64, base: u64) -> f64 {
        let mut result = 0.0;
        let mut f = 1.0 / base as f64;
        let mut i = n;
        
        while i > 0 {
            result += f * (i % base) as f64;
            i /= base;
            f /= base as f64;
        }
        
        result
    }

    /// Reset sequence to beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Skip to specific index in sequence
    pub fn skip_to(&mut self, index: u64) {
        self.current_index = index;
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
    fn test_basic_arithmetic() {
        let a = FixedPoint::from_float(1.5);
        let b = FixedPoint::from_float(2.25);
        assert_eq!((a + b).to_float(), 3.75);
        
        let a = FixedPoint::from_float(5.0);
        let b = FixedPoint::from_float(2.5);
        assert_eq!((a - b).to_float(), 2.5);
        
        let a = FixedPoint::from_float(1.5);
        let b = FixedPoint::from_float(2.0);
        assert!(((a * b).to_float() - 3.0).abs() < 1e-12);
        
        let a = FixedPoint::from_float(3.0);
        let b = FixedPoint::from_float(1.5);
        assert!(((a / b).to_float() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_constants() {
        assert_eq!(FixedPoint::zero().to_float(), 0.0);
        assert_eq!(FixedPoint::one().to_float(), 1.0);
    }

    #[test]
    fn test_precision_configurations() {
        let standard = FixedPoint::from_float(1.5);
        assert_eq!(standard.precision(), FixedPointPrecision::Standard64);
        
        let extended = FixedPoint::with_precision(1500, FixedPointPrecision::Extended128);
        assert_eq!(extended.precision(), FixedPointPrecision::Extended128);
    }

    #[test]
    fn test_rounding_modes() {
        let val = 1.5;
        
        let nearest_even = FixedPoint::from_float_with_precision(
            val, FixedPointPrecision::Standard64, RoundingMode::NearestEven
        );
        let toward_zero = FixedPoint::from_float_with_precision(
            val, FixedPointPrecision::Standard64, RoundingMode::TowardZero
        );
        let toward_pos = FixedPoint::from_float_with_precision(
            val, FixedPointPrecision::Standard64, RoundingMode::TowardPositive
        );
        let toward_neg = FixedPoint::from_float_with_precision(
            val, FixedPointPrecision::Standard64, RoundingMode::TowardNegative
        );
        
        assert_eq!(nearest_even.to_float(), 2.0); // Round to even
        assert_eq!(toward_zero.to_float(), 1.0);
        assert_eq!(toward_pos.to_float(), 2.0);
        assert_eq!(toward_neg.to_float(), 1.0);
    }

    #[test]
    fn test_overflow_detection() {
        let max_val = FixedPoint::from_float(1e10);
        let small_val = FixedPoint::from_float(1e-10);
        
        assert!(max_val.checked_add(max_val).is_some());
        assert!(max_val.checked_mul(small_val).is_some());
    }

    #[test]
    fn test_mathematical_functions() {
        // Test exp and ln
        let x = FixedPoint::from_float(0.1);
        let exp_x = x.exp().to_float();
        let std_exp = (0.1f64).exp();
        assert!((exp_x - std_exp).abs() < 1e-3);
        
        let y = FixedPoint::from_float(1.1);
        let ln_y = y.ln().to_float();
        let std_ln = (1.1f64).ln();
        assert!((ln_y - std_ln).abs() < 1e-3);
        
        // Test sqrt
        let z = FixedPoint::from_float(4.0);
        assert!((z.sqrt().to_float() - 2.0).abs() < 1e-10);
        
        // Test trigonometric functions
        let pi_4 = FixedPoint::from_float(std::f64::consts::PI / 4.0);
        let sin_pi_4 = pi_4.sin().to_float();
        let cos_pi_4 = pi_4.cos().to_float();
        assert!((sin_pi_4 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        assert!((cos_pi_4 - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }

    #[test]
    fn test_hyperbolic_functions() {
        let x = FixedPoint::from_float(0.5);
        
        let sinh_x = x.sinh().to_float();
        let cosh_x = x.cosh().to_float();
        let tanh_x = x.tanh().to_float();
        
        assert!((sinh_x - 0.5_f64.sinh()).abs() < 1e-3);
        assert!((cosh_x - 0.5_f64.cosh()).abs() < 1e-3);
        assert!((tanh_x - 0.5_f64.tanh()).abs() < 1e-3);
    }

    #[test]
    fn test_special_functions() {
        let x = FixedPoint::from_float(0.5);
        
        // Test error function
        let erf_x = x.erf().to_float();
        assert!((erf_x - 0.5204998778130465).abs() < 1e-3); // Known value
        
        // Test normal CDF
        let norm_cdf = FixedPoint::zero().normal_cdf().to_float();
        assert!((norm_cdf - 0.5).abs() < 1e-6);
        
        // Test normal PDF
        let norm_pdf = FixedPoint::zero().normal_pdf().to_float();
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((norm_pdf - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rounding_functions() {
        let x = FixedPoint::from_float(2.7);
        assert_eq!(x.round().to_float(), 3.0);
        assert_eq!(x.floor().to_float(), 2.0);
        assert_eq!(x.ceil().to_float(), 3.0);
        assert_eq!(x.trunc().to_float(), 2.0);
        
        let neg_x = FixedPoint::from_float(-2.7);
        assert_eq!(neg_x.round().to_float(), -3.0);
        assert_eq!(neg_x.floor().to_float(), -3.0);
        assert_eq!(neg_x.ceil().to_float(), -2.0);
        assert_eq!(neg_x.trunc().to_float(), -2.0);
    }

    #[test]
    fn test_prng_algorithms() {
        let seed = 42u64;
        
        let mut xorshift = DeterministicRng::with_algorithm(seed, PRNGAlgorithm::XorShift64Star);
        let mut mersenne = DeterministicRng::with_algorithm(seed, PRNGAlgorithm::MersenneTwister);
        let mut xorshift128 = DeterministicRng::with_algorithm(seed, PRNGAlgorithm::Xorshift128Plus);
        
        // Test that different algorithms produce different sequences
        let x1 = xorshift.next_u64();
        let x2 = mersenne.next_u64();
        let x3 = xorshift128.next_u64();
        
        assert_ne!(x1, x2);
        assert_ne!(x2, x3);
        assert_ne!(x1, x3);
        
        // Test determinism
        let mut rng1 = DeterministicRng::new(42);
        let mut rng2 = DeterministicRng::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_box_muller_generator() {
        let mut generator = BoxMullerGenerator::new(123);
        
        // Generate many samples and check basic properties
        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(generator.next_standard_normal().to_float());
        }
        
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / samples.len() as f64;
        
        // Should be approximately N(0,1)
        assert!((mean - 0.0).abs() < 0.1);
        assert!((variance - 1.0).abs() < 0.2);
        
        // Test correlated normals
        let (x1, x2) = generator.next_correlated_normals(FixedPoint::from_float(0.5));
        assert!(x1.to_float().is_finite());
        assert!(x2.to_float().is_finite());
    }

    #[test]
    fn test_poisson_jump_generator() {
        let mut generator = PoissonJumpGenerator::new(456);
        
        // Test Poisson count generation
        let intensity = FixedPoint::from_float(2.0);
        let dt = FixedPoint::from_float(0.1);
        
        let mut counts = Vec::new();
        for _ in 0..1000 {
            counts.push(generator.next_poisson_count(intensity, dt));
        }
        
        let mean_count = counts.iter().sum::<u32>() as f64 / counts.len() as f64;
        let expected_mean = (intensity * dt).to_float();
        
        // Should be approximately Poisson(λ*dt)
        assert!((mean_count - expected_mean).abs() < 0.1);
        
        // Test jump time generation
        let jump_times = generator.generate_jump_times(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(10.0),
        );
        
        // Should have reasonable number of jumps
        assert!(jump_times.len() > 5 && jump_times.len() < 20);
        
        // Times should be sorted and within bounds
        for i in 1..jump_times.len() {
            assert!(jump_times[i] > jump_times[i-1]);
            assert!(jump_times[i] <= FixedPoint::from_float(10.0));
        }
    }

    #[test]
    fn test_quasi_monte_carlo() {
        let mut sobol = QuasiMonteCarloGenerator::new(QMCSequenceType::Sobol, 2);
        let mut halton = QuasiMonteCarloGenerator::new(QMCSequenceType::Halton, 2);
        
        // Generate points and check they're in [0,1]²
        for _ in 0..10 {
            let sobol_point = sobol.next_point();
            let halton_point = halton.next_point();
            
            assert_eq!(sobol_point.len(), 2);
            assert_eq!(halton_point.len(), 2);
            
            for &coord in &sobol_point {
                assert!(coord.to_float() >= 0.0 && coord.to_float() <= 1.0);
            }
            
            for &coord in &halton_point {
                assert!(coord.to_float() >= 0.0 && coord.to_float() <= 1.0);
            }
        }
        
        // Test reset functionality
        sobol.reset();
        let first_point = sobol.next_point();
        sobol.reset();
        let reset_point = sobol.next_point();
        
        assert_eq!(first_point[0].to_float(), reset_point[0].to_float());
        assert_eq!(first_point[1].to_float(), reset_point[1].to_float());
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