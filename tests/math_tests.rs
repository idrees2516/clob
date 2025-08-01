//! Tests for deterministic mathematical operations
//!
//! This file contains comprehensive tests for the fixed-point arithmetic
//! and circuit-friendly mathematical operations.

use hf_quoting_liquidity_clob::math::{
    FixedPoint, DeterministicRng, FIXED_POINT_SCALE,
    normal_cdf, normal_pdf, black_scholes, hawkes_intensity,
    OptimizationType, Constraint, GradientDescentOptimizer,
};

#[test]
fn test_fixed_point_basic_operations() {
    // Test creation
    let a = FixedPoint::from_int(5);
    let b = FixedPoint::from_float(5.0);
    assert_eq!(a, b);
    
    // Test arithmetic
    let c = FixedPoint::from_float(2.5);
    assert_eq!((a + c).to_float(), 7.5);
    assert_eq!((a - c).to_float(), 2.5);
    assert_eq!((a * c).to_float(), 12.5);
    assert_eq!((a / c).to_float(), 2.0);
    
    // Test comparison
    assert!(a > c);
    assert!(c < a);
    assert!(a >= b);
    assert!(a <= b);
    assert!(a == b);
    assert!(a != c);
    
    // Test constants
    assert_eq!(FixedPoint::zero().to_float(), 0.0);
    assert_eq!(FixedPoint::one().to_float(), 1.0);
}

#[test]
fn test_fixed_point_advanced_operations() {
    // Test sqrt
    let a = FixedPoint::from_float(16.0);
    assert!((a.sqrt().to_float() - 4.0).abs() < 1e-6);
    
    // Test exp
    let b = FixedPoint::from_float(1.0);
    assert!((b.exp().to_float() - 2.718281828459045).abs() < 1e-6);
    
    // Test ln
    let c = FixedPoint::from_float(2.718281828459045);
    assert!((c.ln().to_float() - 1.0).abs() < 1e-6);
    
    // Test pow
    let d = FixedPoint::from_float(2.0);
    let e = FixedPoint::from_float(3.0);
    assert!((d.pow(e).to_float() - 8.0).abs() < 1e-6);
}

#[test]
fn test_deterministic_rng() {
    // Test deterministic sequence
    let mut rng1 = DeterministicRng::new(42);
    let mut rng2 = DeterministicRng::new(42);
    
    for _ in 0..10 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
    }
    
    // Test different seeds
    let mut rng3 = DeterministicRng::new(43);
    assert_ne!(rng1.next_u64(), rng3.next_u64());
    
    // Test distributions
    let normal = rng1.next_normal(0.0, 1.0);
    assert!(normal.is_finite());
    
    let uniform = rng1.next_double();
    assert!(uniform >= 0.0 && uniform < 1.0);
    
    let range = rng1.next_int_range(5, 10);
    assert!(range >= 5 && range <= 10);
}

#[test]
fn test_normal_distribution_functions() {
    // Test normal CDF
    let cdf_0 = normal_cdf(FixedPoint::zero());
    assert!((cdf_0.to_float() - 0.5).abs() < 1e-6);
    
    let cdf_1 = normal_cdf(FixedPoint::one());
    assert!((cdf_1.to_float() - 0.8413).abs() < 1e-3);
    
    // Test normal PDF
    let pdf_0 = normal_pdf(FixedPoint::zero());
    assert!((pdf_0.to_float() - 0.3989).abs() < 1e-3);
}

#[test]
fn test_black_scholes_pricing() {
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
    
    // Check call price is reasonable
    assert!(call_price.to_float() > 5.0);
    assert!(call_price.to_float() < 15.0);
}

#[test]
fn test_hawkes_process() {
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
    
    // Intensity should decay over time
    let later_intensity = hawkes_intensity(baseline, decay, excitation, FixedPoint::from_float(10.0), &events);
    assert!(intensity > later_intensity);
}

#[test]
fn test_gradient_descent_optimization() {
    // Simple quadratic function: f(x,y) = x² + y²
    let objective = |x: &[FixedPoint]| -> FixedPoint {
        x[0] * x[0] + x[1] * x[1]
    };
    
    let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
        vec![x[0] * FixedPoint::from_float(2.0), x[1] * FixedPoint::from_float(2.0)]
    };
    
    let optimizer = GradientDescentOptimizer::new(
        FixedPoint::from_float(0.1),
        1000,
        FixedPoint::from_float(1e-6),
    );
    
    let initial_point = vec![FixedPoint::from_float(1.0), FixedPoint::from_float(1.0)];
    let constraints = vec![];
    
    let result = optimizer.optimize(
        objective,
        gradient,
        &initial_point,
        &constraints,
        OptimizationType::Minimize,
    );
    
    assert!(result.converged);
    assert!(result.objective_value < FixedPoint::from_float(0.01));
    assert!(result.solution[0].abs() < FixedPoint::from_float(0.1));
    assert!(result.solution[1].abs() < FixedPoint::from_float(0.1));
}

#[test]
fn test_constrained_optimization() {
    // Minimize f(x,y) = (x-2)² + (y-1)² subject to x + y <= 2
    let objective = |x: &[FixedPoint]| -> FixedPoint {
        let dx = x[0] - FixedPoint::from_float(2.0);
        let dy = x[1] - FixedPoint::from_float(1.0);
        dx * dx + dy * dy
    };
    
    let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
        vec![
            FixedPoint::from_float(2.0) * (x[0] - FixedPoint::from_float(2.0)),
            FixedPoint::from_float(2.0) * (x[1] - FixedPoint::from_float(1.0)),
        ]
    };
    
    let constraint_fn = Box::new(|x: &[FixedPoint]| -> FixedPoint {
        x[0] + x[1]
    });
    
    let constraints = vec![
        Constraint::LessEqual(constraint_fn, FixedPoint::from_float(2.0)),
        Constraint::Box(0, FixedPoint::zero(), FixedPoint::from_float(10.0)),
        Constraint::Box(1, FixedPoint::zero(), FixedPoint::from_float(10.0)),
    ];
    
    let optimizer = GradientDescentOptimizer::new(
        FixedPoint::from_float(0.1),
        1000,
        FixedPoint::from_float(1e-6),
    );
    
    let initial_point = vec![FixedPoint::from_float(0.5), FixedPoint::from_float(0.5)];
    
    let result = optimizer.optimize(
        objective,
        gradient,
        &initial_point,
        &constraints,
        OptimizationType::Minimize,
    );
    
    // Optimal solution should be on the constraint boundary
    let sum = result.solution[0] + result.solution[1];
    assert!((sum - FixedPoint::from_float(2.0)).abs() < FixedPoint::from_float(0.1));
}

#[test]
fn test_deterministic_results() {
    // This test verifies that our mathematical operations produce
    // deterministic results across different runs
    
    // Fixed-point arithmetic
    let a = FixedPoint::from_float(3.14159);
    let b = FixedPoint::from_float(2.71828);
    
    let sum1 = a + b;
    let sum2 = a + b;
    assert_eq!(sum1.value, sum2.value);
    
    let prod1 = a * b;
    let prod2 = a * b;
    assert_eq!(prod1.value, prod2.value);
    
    // Transcendental functions
    let exp1 = a.exp();
    let exp2 = a.exp();
    assert_eq!(exp1.value, exp2.value);
    
    let ln1 = b.ln();
    let ln2 = b.ln();
    assert_eq!(ln1.value, ln2.value);
    
    // Random number generation
    let mut rng1 = DeterministicRng::new(123);
    let mut rng2 = DeterministicRng::new(123);
    
    for _ in 0..100 {
        assert_eq!(rng1.next_u64(), rng2.next_u64());
        assert_eq!(rng1.next_fixed().value, rng2.next_fixed().value);
    }
    
    // Normal distribution functions
    let cdf1 = normal_cdf(a);
    let cdf2 = normal_cdf(a);
    assert_eq!(cdf1.value, cdf2.value);
    
    // Black-Scholes pricing
    let spot = FixedPoint::from_float(100.0);
    let strike = FixedPoint::from_float(100.0);
    let time = FixedPoint::from_float(1.0);
    let rate = FixedPoint::from_float(0.05);
    let vol = FixedPoint::from_float(0.2);
    
    let call1 = black_scholes(spot, strike, time, rate, vol, true);
    let call2 = black_scholes(spot, strike, time, rate, vol, true);
    assert_eq!(call1.value, call2.value);
}

#[test]
fn test_fixed_point_precision() {
    // Test that our fixed-point representation has sufficient precision
    // for financial calculations
    
    // Test small numbers
    let small = FixedPoint::from_float(0.0001);
    assert!(small.value > 0);
    assert_eq!(small.to_float(), 0.0001);
    
    // Test large numbers
    let large = FixedPoint::from_float(1_000_000.0);
    assert_eq!(large.to_float(), 1_000_000.0);
    
    // Test precision in calculations
    let a = FixedPoint::from_float(1.0);
    let b = FixedPoint::from_float(1.0 + 1e-6);
    assert!(a != b);
    
    // Test that we can represent typical financial values
    let price = FixedPoint::from_float(123.45);
    let quantity = FixedPoint::from_float(1000.0);
    let value = price * quantity;
    assert!((value.to_float() - 123450.0).abs() < 0.01);
}