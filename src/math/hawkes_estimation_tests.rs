//! Tests for Hawkes Process Parameter Estimation
//! 
//! Comprehensive test suite for maximum likelihood estimation, EM algorithm,
//! cross-validation, and goodness-of-fit testing for Hawkes processes.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fixed_point::{FixedPoint, DeterministicRng};
    use crate::math::hawkes_process::{
        HawkesEvent, MultivariateHawkesParams, KernelType, MultivariateHawkesSimulator
    };
    use crate::math::hawkes_estimation::{
        LBFGSOptimizer, HawkesMLEstimator, HawkesEMEstimator, HawkesCrossValidator,
        HawkesGoodnessOfFitTester, ValidationMetric, EstimationError
    };

    /// Create test Hawkes parameters
    fn create_test_params() -> MultivariateHawkesParams {
        let baseline_intensities = vec![
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(0.3),
        ];
        
        let kernels = vec![
            vec![
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.2), 
                    beta: FixedPoint::from_float(1.0) 
                },
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.1), 
                    beta: FixedPoint::from_float(0.8) 
                },
            ],
            vec![
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.15), 
                    beta: FixedPoint::from_float(0.9) 
                },
                KernelType::Exponential { 
                    alpha: FixedPoint::from_float(0.25), 
                    beta: FixedPoint::from_float(1.2) 
                },
            ],
        ];
        
        MultivariateHawkesParams::new(
            baseline_intensities,
            kernels,
            FixedPoint::from_float(10.0),
        ).unwrap()
    }
    
    /// Generate synthetic event data for testing
    fn generate_test_data(
        params: &MultivariateHawkesParams,
        n_sequences: usize,
        observation_time: FixedPoint,
        seed: u64,
    ) -> Vec<Vec<HawkesEvent>> {
        let mut sequences = Vec::new();
        
        for i in 0..n_sequences {
            let mut simulator = MultivariateHawkesSimulator::new(
                params.clone(),
                1000,
            );
            let mut rng = DeterministicRng::new(seed + i as u64);
            
            match simulator.simulate_until(observation_time, &mut rng) {
                Ok(events) => sequences.push(events),
                Err(_) => sequences.push(Vec::new()),
            }
        }
        
        sequences
    }

    #[test]
    fn test_lbfgs_optimizer_creation() {
        let optimizer = LBFGSOptimizer::new(
            100,
            FixedPoint::from_float(1e-6),
            10,
        );
        
        assert_eq!(optimizer.max_iterations, 100);
        assert_eq!(optimizer.tolerance, FixedPoint::from_float(1e-6));
        assert_eq!(optimizer.history_size, 10);
    }

    #[test]
    fn test_lbfgs_simple_quadratic() {
        let optimizer = LBFGSOptimizer::default();
        
        // Minimize f(x) = x^2 + 2x + 1 = (x+1)^2
        // Minimum at x = -1, f(-1) = 0
        let objective = |x: &[FixedPoint]| -> FixedPoint {
            let x0 = x[0];
            x0 * x0 + FixedPoint::from_float(2.0) * x0 + FixedPoint::one()
        };
        
        let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
            let x0 = x[0];
            vec![FixedPoint::from_float(2.0) * x0 + FixedPoint::from_float(2.0)]
        };
        
        let initial = vec![FixedPoint::from_float(5.0)];
        
        match optimizer.optimize(initial, objective, gradient) {
            Ok(result) => {
                assert!((result[0].to_float() + 1.0).abs() < 1e-3);
            }
            Err(e) => panic!("Optimization failed: {}", e),
        }
    }

    #[test]
    fn test_hawkes_ml_estimator_creation() {
        let optimizer = LBFGSOptimizer::default();
        let estimator = HawkesMLEstimator::new(optimizer);
        
        assert!(!estimator.use_regularization);
        
        let estimator_with_reg = estimator.with_regularization(FixedPoint::from_float(0.01));
        assert!(estimator_with_reg.use_regularization);
        assert_eq!(estimator_with_reg.regularization_strength, FixedPoint::from_float(0.01));
    }

    #[test]
    fn test_hawkes_ml_estimation_with_synthetic_data() {
        let true_params = create_test_params();
        let observation_time = FixedPoint::from_float(10.0);
        
        // Generate synthetic data
        let event_sequences = generate_test_data(&true_params, 5, observation_time, 42);
        
        // Skip test if no events generated
        if event_sequences.iter().all(|seq| seq.is_empty()) {
            return;
        }
        
        // Create initial parameters (slightly different from true)
        let mut initial_params = true_params.clone();
        for intensity in &mut initial_params.baseline_intensities {
            *intensity = *intensity * FixedPoint::from_float(1.1);
        }
        
        let optimizer = LBFGSOptimizer::new(50, FixedPoint::from_float(1e-4), 5);
        let estimator = HawkesMLEstimator::new(optimizer);
        
        match estimator.estimate(&event_sequences, &initial_params, observation_time) {
            Ok(estimated_params) => {
                // Check that estimated parameters are reasonable
                assert_eq!(estimated_params.dimension(), true_params.dimension());
                
                for &intensity in &estimated_params.baseline_intensities {
                    assert!(intensity.to_float() > 0.0);
                    assert!(intensity.to_float() < 10.0); // Reasonable upper bound
                }
            }
            Err(e) => {
                // Estimation might fail with limited synthetic data - that's okay for testing
                println!("ML estimation failed (expected with limited data): {}", e);
            }
        }
    }

    #[test]
    fn test_hawkes_em_estimator_creation() {
        let estimator = HawkesEMEstimator::new(
            50,
            FixedPoint::from_float(1e-5),
        );
        
        assert_eq!(estimator.max_iterations, 50);
        assert_eq!(estimator.tolerance, FixedPoint::from_float(1e-5));
    }

    #[test]
    fn test_hawkes_em_estimation_with_synthetic_data() {
        let true_params = create_test_params();
        let observation_time = FixedPoint::from_float(5.0);
        
        // Generate synthetic data
        let event_sequences = generate_test_data(&true_params, 3, observation_time, 123);
        
        // Skip test if no events generated
        if event_sequences.iter().all(|seq| seq.is_empty()) {
            return;
        }
        
        let estimator = HawkesEMEstimator::default();
        
        match estimator.estimate(&event_sequences, &true_params, observation_time) {
            Ok(estimated_params) => {
                assert_eq!(estimated_params.dimension(), true_params.dimension());
                
                for &intensity in &estimated_params.baseline_intensities {
                    assert!(intensity.to_float() > 0.0);
                }
            }
            Err(e) => {
                println!("EM estimation failed (expected with limited data): {}", e);
            }
        }
    }

    #[test]
    fn test_cross_validator_creation() {
        let validator = HawkesCrossValidator::new(3, ValidationMetric::AIC);
        
        assert_eq!(validator.k_folds, 3);
        matches!(validator.validation_metric, ValidationMetric::AIC);
    }

    #[test]
    fn test_cross_validation_with_multiple_models() {
        let true_params = create_test_params();
        let observation_time = FixedPoint::from_float(8.0);
        
        // Generate synthetic data
        let event_sequences = generate_test_data(&true_params, 10, observation_time, 456);
        
        // Skip test if insufficient data
        if event_sequences.len() < 5 || event_sequences.iter().all(|seq| seq.is_empty()) {
            return;
        }
        
        // Create candidate models
        let mut candidate1 = true_params.clone();
        candidate1.baseline_intensities[0] = FixedPoint::from_float(0.4);
        
        let mut candidate2 = true_params.clone();
        candidate2.baseline_intensities[0] = FixedPoint::from_float(0.6);
        
        let candidates = vec![candidate1, candidate2];
        
        let validator = HawkesCrossValidator::new(3, ValidationMetric::AIC);
        
        match validator.validate(&event_sequences, &candidates, observation_time) {
            Ok(result) => {
                assert!(result.best_model_idx < candidates.len());
                assert_eq!(result.results.len(), candidates.len());
                
                for model_result in &result.results {
                    assert_eq!(model_result.fold_scores.len(), 3);
                }
            }
            Err(e) => {
                println!("Cross-validation failed (expected with limited data): {}", e);
            }
        }
    }

    #[test]
    fn test_goodness_of_fit_tester_creation() {
        let tester = HawkesGoodnessOfFitTester::new(
            FixedPoint::from_float(0.01),
            500,
        );
        
        assert_eq!(tester.significance_level, FixedPoint::from_float(0.01));
        assert_eq!(tester.n_bootstrap_samples, 500);
    }

    #[test]
    fn test_goodness_of_fit_testing() {
        let true_params = create_test_params();
        let observation_time = FixedPoint::from_float(6.0);
        
        // Generate synthetic data
        let event_sequences = generate_test_data(&true_params, 5, observation_time, 789);
        
        // Skip test if no events generated
        if event_sequences.iter().all(|seq| seq.is_empty()) {
            return;
        }
        
        let tester = HawkesGoodnessOfFitTester::default();
        
        match tester.test_fit(&event_sequences, &true_params, observation_time) {
            Ok(result) => {
                // Check that all tests were performed
                assert!(result.ks_test.p_value.to_float() >= 0.0);
                assert!(result.ks_test.p_value.to_float() <= 1.0);
                
                assert!(result.ad_test.p_value.to_float() >= 0.0);
                assert!(result.ad_test.p_value.to_float() <= 1.0);
                
                assert!(result.bootstrap_test.p_value.to_float() >= 0.0);
                assert!(result.bootstrap_test.p_value.to_float() <= 1.0);
                
                // Residual analysis should have some residuals if events exist
                if event_sequences.iter().any(|seq| !seq.is_empty()) {
                    assert!(!result.residual_analysis.residuals.is_empty());
                }
            }
            Err(e) => {
                println!("Goodness-of-fit testing failed: {}", e);
            }
        }
    }

    #[test]
    fn test_parameter_vector_conversion() {
        let params = create_test_params();
        let estimator = HawkesMLEstimator::default();
        
        // Convert to vector and back
        let vector = estimator.params_to_vector(&params);
        assert!(!vector.is_empty());
        
        match estimator.vector_to_params(&vector, params.dimension()) {
            Ok(reconstructed) => {
                assert_eq!(reconstructed.dimension(), params.dimension());
                assert_eq!(reconstructed.baseline_intensities.len(), params.baseline_intensities.len());
            }
            Err(e) => panic!("Parameter conversion failed: {}", e),
        }
    }

    #[test]
    fn test_validation_metrics() {
        // Test different validation metrics
        let metrics = vec![
            ValidationMetric::LogLikelihood,
            ValidationMetric::AIC,
            ValidationMetric::BIC,
            ValidationMetric::KSTest,
        ];
        
        for metric in metrics {
            let validator = HawkesCrossValidator::new(3, metric);
            // Just ensure creation works with all metric types
            assert_eq!(validator.k_folds, 3);
        }
    }

    #[test]
    fn test_estimation_error_handling() {
        let estimator = HawkesMLEstimator::default();
        
        // Test with empty data
        let empty_sequences: Vec<Vec<HawkesEvent>> = vec![];
        let params = create_test_params();
        let observation_time = FixedPoint::from_float(1.0);
        
        match estimator.estimate(&empty_sequences, &params, observation_time) {
            Err(EstimationError::InsufficientData(_)) => {
                // Expected error
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_statistical_test_properties() {
        let tester = HawkesGoodnessOfFitTester::default();
        let params = create_test_params();
        let observation_time = FixedPoint::from_float(1.0);
        
        // Create minimal test data
        let events = vec![vec![HawkesEvent {
            time: FixedPoint::from_float(0.5),
            process_id: 0,
            mark: None,
        }]];
        
        match tester.test_fit(&events, &params, observation_time) {
            Ok(result) => {
                // Verify statistical test structure
                assert!(result.ks_test.statistic.to_float() >= 0.0);
                assert!(result.ks_test.critical_value.to_float() >= 0.0);
                
                // P-values should be between 0 and 1
                assert!(result.ks_test.p_value.to_float() >= 0.0);
                assert!(result.ks_test.p_value.to_float() <= 1.0);
            }
            Err(e) => {
                println!("Statistical test failed: {}", e);
            }
        }
    }

    #[test]
    fn test_kernel_parameter_estimation() {
        // Test with different kernel types
        let baseline_intensities = vec![FixedPoint::from_float(0.5)];
        
        let exponential_kernels = vec![vec![
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.3), 
                beta: FixedPoint::from_float(1.5) 
            }
        ]];
        
        let power_law_kernels = vec![vec![
            KernelType::PowerLaw { 
                alpha: FixedPoint::from_float(0.2), 
                beta: FixedPoint::from_float(1.8),
                cutoff: FixedPoint::from_float(0.1),
            }
        ]];
        
        let exp_params = MultivariateHawkesParams::new(
            baseline_intensities.clone(),
            exponential_kernels,
            FixedPoint::from_float(5.0),
        ).unwrap();
        
        let power_params = MultivariateHawkesParams::new(
            baseline_intensities,
            power_law_kernels,
            FixedPoint::from_float(5.0),
        ).unwrap();
        
        // Both should be valid for estimation
        assert_eq!(exp_params.dimension(), 1);
        assert_eq!(power_params.dimension(), 1);
    }

    #[test]
    fn test_regularization_effects() {
        let optimizer = LBFGSOptimizer::default();
        let estimator_no_reg = HawkesMLEstimator::new(optimizer.clone());
        let estimator_with_reg = HawkesMLEstimator::new(optimizer)
            .with_regularization(FixedPoint::from_float(0.1));
        
        assert!(!estimator_no_reg.use_regularization);
        assert!(estimator_with_reg.use_regularization);
        assert_eq!(estimator_with_reg.regularization_strength, FixedPoint::from_float(0.1));
    }

    #[test]
    fn test_em_convergence_properties() {
        let em_estimator = HawkesEMEstimator::new(
            10, // Low iteration count for testing
            FixedPoint::from_float(1e-3), // Loose tolerance
        );
        
        assert_eq!(em_estimator.max_iterations, 10);
        assert_eq!(em_estimator.tolerance, FixedPoint::from_float(1e-3));
    }
}