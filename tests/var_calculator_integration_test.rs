use hf_quoting_liquidity_clob::risk::var_calculator::*;
use hf_quoting_liquidity_clob::math::fixed_point::FixedPoint;

#[test]
fn test_var_calculator_integration() {
    let calculator = VaRCalculator::new(100, 1000, VaRMethod::Historical);
    
    // Add some test returns
    for i in 0..50 {
        let return_val = (i as f64 - 25.0) * 0.002;
        calculator.add_return(FixedPoint::from_float(return_val));
    }
    
    // Test historical VaR
    let result = calculator.calculate_var(VaRMethod::Historical, ConfidenceLevel::Percent95);
    assert!(result.is_ok());
    
    let var_result = result.unwrap();
    assert!(var_result.var > FixedPoint::from_float(0.0));
    assert!(var_result.expected_shortfall >= var_result.var);
    assert_eq!(var_result.method, VaRMethod::Historical);
}

#[test]
fn test_monte_carlo_var() {
    let mut calculator = MonteCarloVaRCalculator::new(1000, 100, false);
    
    // Add sufficient test returns
    for i in 0..50 {
        let return_val = (i as f64 - 25.0) * 0.002;
        calculator.add_return(FixedPoint::from_float(return_val));
    }
    
    let result = calculator.calculate_var(ConfidenceLevel::Percent95);
    assert!(result.is_ok());
    
    let var_result = result.unwrap();
    assert!(var_result.var > FixedPoint::from_float(0.0));
    assert_eq!(var_result.method, VaRMethod::MonteCarlo);
}

#[test]
fn test_confidence_levels() {
    assert_eq!(ConfidenceLevel::Percent95.alpha(), FixedPoint::from_float(0.05));
    assert_eq!(ConfidenceLevel::Percent99.alpha(), FixedPoint::from_float(0.01));
    assert_eq!(ConfidenceLevel::Percent999.alpha(), FixedPoint::from_float(0.001));
}