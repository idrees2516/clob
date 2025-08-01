//! Circuit-friendly mathematical operations
//! 
//! This module provides implementations of mathematical functions that are
//! compatible with zkVM circuit constraints. These functions are designed
//! to be deterministic and avoid operations that are difficult to represent
//! in arithmetic circuits.

use crate::math::fixed_point::{FixedPoint, DeterministicRng, FIXED_POINT_SCALE};
use std::collections::VecDeque;

/// Maximum number of iterations for convergent series
const MAX_ITERATIONS: usize = 20;

/// Circuit-friendly approximation of the error function
pub fn erf(x: FixedPoint) -> FixedPoint {
    // Use Abramowitz and Stegun approximation (maximum error: 1.5×10^-7)
    let abs_x = x.abs();
    let p = FixedPoint::from_float(0.3275911);
    
    let t = FixedPoint::one() / (FixedPoint::one() + p * abs_x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    
    let a1 = FixedPoint::from_float(0.254829592);
    let a2 = FixedPoint::from_float(-0.284496736);
    let a3 = FixedPoint::from_float(1.421413741);
    let a4 = FixedPoint::from_float(-1.453152027);
    let a5 = FixedPoint::from_float(1.061405429);
    
    let polynomial = FixedPoint::one() - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-abs_x * abs_x).exp();
    
    if x < FixedPoint::zero() {
        -polynomial
    } else {
        polynomial
    }
}

/// Circuit-friendly approximation of the inverse error function
pub fn erf_inv(x: FixedPoint) -> FixedPoint {
    // Approximation using Chebyshev polynomials
    // Valid for |x| < 1
    
    if x >= FixedPoint::one() {
        return FixedPoint::from_float(5.0); // Approximate infinity
    }
    if x <= -FixedPoint::one() {
        return FixedPoint::from_float(-5.0); // Approximate -infinity
    }
    
    let abs_x = x.abs();
    let sign = if x < FixedPoint::zero() { -FixedPoint::one() } else { FixedPoint::one() };
    
    // Coefficients for the approximation
    let c1 = FixedPoint::from_float(0.886226899);
    let c2 = FixedPoint::from_float(1.645349621);
    let c3 = FixedPoint::from_float(0.285375813);
    
    // Compute polynomial approximation
    let y = abs_x * abs_x;
    let term1 = c1 * abs_x;
    let term2 = c2 * abs_x * y;
    let term3 = c3 * abs_x * y * y;
    
    sign * (term1 + term2 + term3)
}

/// Circuit-friendly normal cumulative distribution function
pub fn normal_cdf(x: FixedPoint) -> FixedPoint {
    // Use error function relationship: CDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
    let sqrt_2 = FixedPoint::from_float(1.4142135623730951);
    let half = FixedPoint::from_float(0.5);
    
    half * (FixedPoint::one() + erf(x / sqrt_2))
}

/// Circuit-friendly normal probability density function
pub fn normal_pdf(x: FixedPoint) -> FixedPoint {
    // PDF(x) = (1/sqrt(2π)) * e^(-x²/2)
    let inv_sqrt_2pi = FixedPoint::from_float(0.3989422804014327);
    let x_squared = x * x;
    let exponent = -x_squared / FixedPoint::from_float(2.0);
    
    inv_sqrt_2pi * exponent.exp()
}

/// Circuit-friendly Black-Scholes option pricing formula
pub fn black_scholes(
    spot: FixedPoint,
    strike: FixedPoint,
    time_to_expiry: FixedPoint,
    risk_free_rate: FixedPoint,
    volatility: FixedPoint,
    is_call: bool,
) -> FixedPoint {
    if time_to_expiry <= FixedPoint::zero() {
        // At expiry, option value is max(0, spot-strike) for calls
        // or max(0, strike-spot) for puts
        if is_call {
            return (spot - strike).max(FixedPoint::zero());
        } else {
            return (strike - spot).max(FixedPoint::zero());
        }
    }
    
    let sqrt_time = time_to_expiry.sqrt();
    let vol_sqrt_time = volatility * sqrt_time;
    
    // Calculate d1 and d2
    let d1 = ((spot / strike).ln() + 
             (risk_free_rate + volatility * volatility / FixedPoint::from_float(2.0)) * time_to_expiry) / 
             vol_sqrt_time;
    
    let d2 = d1 - vol_sqrt_time;
    
    // Calculate option price
    let discount = (-risk_free_rate * time_to_expiry).exp();
    
    if is_call {
        spot * normal_cdf(d1) - strike * discount * normal_cdf(d2)
    } else {
        strike * discount * normal_cdf(-d2) - spot * normal_cdf(-d1)
    }
}

/// Circuit-friendly binomial tree option pricing
pub fn binomial_option_price(
    spot: FixedPoint,
    strike: FixedPoint,
    time_to_expiry: FixedPoint,
    risk_free_rate: FixedPoint,
    volatility: FixedPoint,
    steps: usize,
    is_call: bool,
) -> FixedPoint {
    let dt = time_to_expiry / FixedPoint::from_int(steps as i32);
    let discount = (-risk_free_rate * dt).exp();
    
    // Calculate up and down factors
    let up = (volatility * dt.sqrt()).exp();
    let down = FixedPoint::one() / up;
    
    // Risk-neutral probability
    let p_up = ((risk_free_rate * dt).exp() - down) / (up - down);
    let p_down = FixedPoint::one() - p_up;
    
    // Initialize terminal values
    let mut values = Vec::with_capacity(steps + 1);
    
    for i in 0..=steps {
        let price = spot * up.pow(FixedPoint::from_int((steps - i) as i32)) * 
                   down.pow(FixedPoint::from_int(i as i32));
        
        let payoff = if is_call {
            (price - strike).max(FixedPoint::zero())
        } else {
            (strike - price).max(FixedPoint::zero())
        };
        
        values.push(payoff);
    }
    
    // Work backwards through the tree
    for step in (0..steps).rev() {
        for i in 0..=step {
            values[i] = discount * (p_up * values[i] + p_down * values[i + 1]);
        }
    }
    
    values[0]
}

/// Circuit-friendly numerical integration using Simpson's rule
pub fn integrate<F>(
    f: F,
    a: FixedPoint,
    b: FixedPoint,
    steps: usize,
) -> FixedPoint 
where
    F: Fn(FixedPoint) -> FixedPoint,
{
    if steps == 0 || a == b {
        return FixedPoint::zero();
    }
    
    let h = (b - a) / FixedPoint::from_int(steps as i32);
    let mut sum = f(a) + f(b);
    
    for i in 1..steps {
        let x = a + FixedPoint::from_int(i as i32) * h;
        let weight = if i % 2 == 0 { FixedPoint::from_int(2) } else { FixedPoint::from_int(4) };
        sum = sum + weight * f(x);
    }
    
    sum * h / FixedPoint::from_int(3)
}

/// Circuit-friendly numerical differentiation
pub fn differentiate<F>(
    f: F,
    x: FixedPoint,
    h: FixedPoint,
) -> FixedPoint 
where
    F: Fn(FixedPoint) -> FixedPoint,
{
    // Central difference approximation
    (f(x + h) - f(x - h)) / (FixedPoint::from_int(2) * h)
}

/// Circuit-friendly root finding using Newton-Raphson method
pub fn newton_raphson<F, DF>(
    f: F,
    df: DF,
    initial_guess: FixedPoint,
    tolerance: FixedPoint,
    max_iterations: usize,
) -> Option<FixedPoint>
where
    F: Fn(FixedPoint) -> FixedPoint,
    DF: Fn(FixedPoint) -> FixedPoint,
{
    let mut x = initial_guess;
    
    for _ in 0..max_iterations {
        let f_x = f(x);
        
        if f_x.abs() < tolerance {
            return Some(x);
        }
        
        let df_x = df(x);
        
        if df_x == FixedPoint::zero() {
            return None; // Derivative is zero, can't continue
        }
        
        x = x - f_x / df_x;
    }
    
    None // Failed to converge
}

/// Generate a sequence of Sobol numbers for quasi-Monte Carlo methods
pub struct SobolSequence {
    dimension: usize,
    count: usize,
    direction_numbers: Vec<Vec<u32>>,
}

impl SobolSequence {
    /// Create a new Sobol sequence generator for the given dimension
    pub fn new(dimension: usize) -> Self {
        // Initialize direction numbers for each dimension
        let mut direction_numbers = Vec::with_capacity(dimension);
        
        // First dimension uses special sequence
        let mut first_dim = Vec::with_capacity(32);
        for i in 0..32 {
            first_dim.push(1u32 << (31 - i));
        }
        direction_numbers.push(first_dim);
        
        // Other dimensions use primitive polynomials
        for d in 1..dimension {
            let mut dim_numbers = Vec::with_capacity(32);
            
            // Simple initialization for demo purposes
            // In practice, would use proper primitive polynomials
            for i in 0..32 {
                dim_numbers.push((1u32 << (31 - i)) ^ (d as u32));
            }
            
            direction_numbers.push(dim_numbers);
        }
        
        Self {
            dimension,
            count: 0,
            direction_numbers,
        }
    }
    
    /// Get the next point in the sequence
    pub fn next(&mut self) -> Vec<FixedPoint> {
        self.count += 1;
        let c = self.count;
        
        let mut result = Vec::with_capacity(self.dimension);
        
        for d in 0..self.dimension {
            let direction = &self.direction_numbers[d];
            let mut x = 0u32;
            
            // Find position of least significant zero bit
            let mut c_temp = c;
            let mut bit = 0;
            while c_temp & 1 == 1 {
                c_temp >>= 1;
                bit += 1;
            }
            
            if bit < direction.len() {
                x ^= direction[bit];
            }
            
            // Convert to fixed point in [0,1)
            let fixed = FixedPoint::from_raw((x as u64 * FIXED_POINT_SCALE / (1u64 << 32)) as i64);
            result.push(fixed);
        }
        
        result
    }
    
    /// Reset the sequence
    pub fn reset(&mut self) {
        self.count = 0;
    }
}

/// Circuit-friendly Monte Carlo simulation for option pricing
pub fn monte_carlo_option_price(
    spot: FixedPoint,
    strike: FixedPoint,
    time_to_expiry: FixedPoint,
    risk_free_rate: FixedPoint,
    volatility: FixedPoint,
    paths: usize,
    steps: usize,
    is_call: bool,
    rng: &mut DeterministicRng,
) -> FixedPoint {
    let dt = time_to_expiry / FixedPoint::from_int(steps as i32);
    let drift = risk_free_rate - volatility * volatility / FixedPoint::from_float(2.0);
    let vol_sqrt_dt = volatility * dt.sqrt();
    let discount = (-risk_free_rate * time_to_expiry).exp();
    
    let mut sum_payoffs = FixedPoint::zero();
    
    for _ in 0..paths {
        let mut price = spot;
        
        for _ in 0..steps {
            let z = rng.next_normal_fixed(FixedPoint::zero(), FixedPoint::one());
            price = price * (drift * dt + vol_sqrt_dt * z).exp();
        }
        
        let payoff = if is_call {
            (price - strike).max(FixedPoint::zero())
        } else {
            (strike - price).max(FixedPoint::zero())
        };
        
        sum_payoffs = sum_payoffs + payoff;
    }
    
    discount * sum_payoffs / FixedPoint::from_int(paths as i32)
}

/// Circuit-friendly implementation of the Hawkes process intensity calculation
pub fn hawkes_intensity(
    baseline: FixedPoint,
    decay: FixedPoint,
    excitation: FixedPoint,
    current_time: FixedPoint,
    event_times: &[FixedPoint],
) -> FixedPoint {
    let mut intensity = baseline;
    
    for &event_time in event_times {
        if event_time < current_time {
            let dt = current_time - event_time;
            let kernel = excitation * (-decay * dt).exp();
            intensity = intensity + kernel;
        }
    }
    
    intensity
}

/// Circuit-friendly implementation of the Hawkes process simulation
pub fn simulate_hawkes_process(
    baseline: FixedPoint,
    decay: FixedPoint,
    excitation: FixedPoint,
    end_time: FixedPoint,
    max_intensity: FixedPoint,
    rng: &mut DeterministicRng,
) -> Vec<FixedPoint> {
    let mut events = Vec::new();
    let mut current_time = FixedPoint::zero();
    
    while current_time < end_time {
        // Calculate current intensity
        let intensity = hawkes_intensity(baseline, decay, excitation, current_time, &events);
        
        // Generate next event time using thinning algorithm
        let dt = rng.next_exponential_fixed(max_intensity);
        current_time = current_time + dt;
        
        if current_time >= end_time {
            break;
        }
        
        // Accept/reject step
        let u = rng.next_fixed();
        let acceptance_prob = intensity / max_intensity;
        
        if u <= acceptance_prob {
            events.push(current_time);
        }
    }
    
    events
}

/// Circuit-friendly implementation of the fractional Brownian motion
pub fn fractional_brownian_motion(
    hurst: FixedPoint,
    n_steps: usize,
    dt: FixedPoint,
    rng: &mut DeterministicRng,
) -> Vec<FixedPoint> {
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push(FixedPoint::zero()); // Start at 0
    
    // For H = 0.5, this is standard Brownian motion
    if (hurst - FixedPoint::from_float(0.5)).abs() < FixedPoint::from_float(0.001) {
        let sqrt_dt = dt.sqrt();
        
        for _ in 0..n_steps {
            let z = rng.next_normal_fixed(FixedPoint::zero(), FixedPoint::one());
            let increment = z * sqrt_dt;
            let next_value = *path.last().unwrap() + increment;
            path.push(next_value);
        }
        
        return path;
    }
    
    // For H != 0.5, use Cholesky decomposition approach
    // First, build covariance matrix
    let mut cov = vec![vec![FixedPoint::zero(); n_steps]; n_steps];
    
    for i in 0..n_steps {
        for j in 0..=i {
            let ti = FixedPoint::from_int((i + 1) as i32) * dt;
            let tj = FixedPoint::from_int((j + 1) as i32) * dt;
            
            // Covariance function for fBm
            let cov_ij = FixedPoint::from_float(0.5) * (
                ti.pow(FixedPoint::from_float(2.0) * hurst) +
                tj.pow(FixedPoint::from_float(2.0) * hurst) -
                (ti - tj).abs().pow(FixedPoint::from_float(2.0) * hurst)
            );
            
            cov[i][j] = cov_ij;
            if i != j {
                cov[j][i] = cov_ij;
            }
        }
    }
    
    // Simplified Cholesky decomposition for circuit compatibility
    let mut l = vec![vec![FixedPoint::zero(); n_steps]; n_steps];
    
    for i in 0..n_steps {
        for j in 0..=i {
            let mut sum = cov[i][j];
            
            for k in 0..j {
                sum = sum - l[i][k] * l[j][k];
            }
            
            if i == j {
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    
    // Generate correlated normal variables
    let mut z = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        z.push(rng.next_normal_fixed(FixedPoint::zero(), FixedPoint::one()));
    }
    
    // Compute fBm increments
    let mut increments = vec![FixedPoint::zero(); n_steps];
    
    for i in 0..n_steps {
        for j in 0..=i {
            increments[i] = increments[i] + l[i][j] * z[j];
        }
    }
    
    // Build path
    for i in 0..n_steps {
        let next_value = path[i] + increments[i];
        path.push(next_value);
    }
    
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        let x = FixedPoint::from_float(1.0);
        let result = erf(x);
        let expected = FixedPoint::from_float(0.8427007929); // erf(1.0)
        
        assert!((result - expected).abs() < FixedPoint::from_float(0.01));
    }

    #[test]
    fn test_normal_cdf() {
        let x = FixedPoint::zero();
        let result = normal_cdf(x);
        let expected = FixedPoint::from_float(0.5); // CDF(0) = 0.5
        
        assert!((result - expected).abs() < FixedPoint::from_float(0.01));
    }

    #[test]
    fn test_black_scholes() {
        let spot = FixedPoint::from_float(100.0);
        let strike = FixedPoint::from_float(100.0);
        let time = FixedPoint::from_float(1.0);
        let rate = FixedPoint::from_float(0.05);
        let vol = FixedPoint::from_float(0.2);
        
        let call_price = black_scholes(spot, strike, time, rate, vol, true);
        let put_price = black_scholes(spot, strike, time, rate, vol, false);
        
        // Check put-call parity
        let pv_strike = strike * (-rate * time).exp();
        let parity_diff = (call_price - put_price) - (spot - pv_strike);
        
        assert!(parity_diff.abs() < FixedPoint::from_float(0.1));
    }

    #[test]
    fn test_hawkes_intensity() {
        let baseline = FixedPoint::from_float(1.0);
        let decay = FixedPoint::from_float(2.0);
        let excitation = FixedPoint::from_float(0.5);
        
        let events = vec![
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(1.5),
            FixedPoint::from_float(2.0),
        ];
        
        let intensity = hawkes_intensity(baseline, decay, excitation, FixedPoint::from_float(3.0), &events);
        
        // Intensity should be greater than baseline
        assert!(intensity > baseline);
    }

    #[test]
    fn test_fractional_brownian_motion() {
        let mut rng = DeterministicRng::new(42);
        
        // Standard Brownian motion (H=0.5)
        let path1 = fractional_brownian_motion(
            FixedPoint::from_float(0.5),
            100,
            FixedPoint::from_float(0.01),
            &mut rng,
        );
        
        // Rough path (H=0.1)
        let path2 = fractional_brownian_motion(
            FixedPoint::from_float(0.1),
            100,
            FixedPoint::from_float(0.01),
            &mut rng,
        );
        
        assert_eq!(path1.len(), 101);
        assert_eq!(path2.len(), 101);
        
        // First point should be zero
        assert_eq!(path1[0], FixedPoint::zero());
        assert_eq!(path2[0], FixedPoint::zero());
    }
}