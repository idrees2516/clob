use hf_quoting_liquidity::models::parameter_optimization::*;
use hf_quoting_liquidity::math::fixed_point::FixedPoint;
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple quadratic objective function for testing
    struct QuadraticObjective {
        target: HashMap<String, FixedPoint>,
    }

    impl QuadraticObjective {
        fn new() -> Self {
            let mut target = HashMap::new();
            target.insert("x".to_string(), FixedPoint::from_float(0.5));
            target.insert("y".to_string(), FixedPoint::from_float(-0.3));
            
            Self { target }
        }
    }

    impl ObjectiveFunction for QuadraticObjective {
        fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
            let x = parameters.get("x").copied().unwrap_or(FixedPoint::zero());
            let y = parameters.get("y").copied().unwrap_or(FixedPoint::zero());
            
            let target_x = self.target.get("x").copied().unwrap();
            let target_y = self.target.get("y").copied().unwrap();
            
            // Negative quadratic distance from target (to maximize)
            let distance_sq = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y);
            Ok(-distance_sq)
        }
        
        fn name(&self) -> &str {
            "Quadratic"
        }
    }

    #[test]
    fn test_parameter_bounds_sampling() {
        let bounds = ParameterBounds::new(
            FixedPoint::from_float(-1.0),
            FixedPoint::from_float(1.0)
        );
        
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let sample = bounds.sample(&mut rng);
            assert!(sample >= bounds.min_value);
            assert!(sample <= bounds.max_value);
        }
    }

    #[test]
    fn test_parameter_bounds_log_scale() {
        let bounds = ParameterBounds::new(
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(100.0)
        ).log_scale();
        
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let sample = bounds.sample(&mut rng);
            assert!(sample >= bounds.min_value);
            assert!(sample <= bounds.max_value);
        }
    }

    #[test]
    fn test_parameter_space_creation() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "param1".to_string(),
            ParameterBounds::new(FixedPoint::from_float(0.0), FixedPoint::from_float(1.0))
        );
        space.add_parameter(
            "param2".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        
        assert_eq!(space.dimension(), 2);
        assert_eq!(space.parameter_names().len(), 2);
        
        let mut rng = rand::thread_rng();
        let sample = space.sample(&mut rng);
        assert_eq!(sample.len(), 2);
        assert!(sample.contains_key("param1"));
        assert!(sample.contains_key("param2"));
    }

    #[test]
    fn test_grid_search_optimization() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-1.0), FixedPoint::from_float(1.0))
        );
        space.add_parameter(
            "y".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-1.0), FixedPoint::from_float(1.0))
        );
        
        let optimizer = GridSearchOptimizer::new(space, 5);
        let objective = QuadraticObjective::new();
        let result = optimizer.optimize(&objective).unwrap();
        
        // Should find optimum near target (0.5, -0.3)
        let best_x = result.best_parameters.get("x").copied().unwrap();
        let best_y = result.best_parameters.get("y").copied().unwrap();
        
        // With 5 grid points, we expect reasonable approximation
        assert!((best_x - FixedPoint::from_float(0.5)).abs() < FixedPoint::from_float(0.6));
        assert!((best_y - FixedPoint::from_float(-0.3)).abs() < FixedPoint::from_float(0.6));
        
        // Should have evaluated 5x5 = 25 points
        assert_eq!(result.evaluations.len(), 25);
        assert!(result.convergence_info.converged);
    }

    #[test]
    fn test_random_search_optimization() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        space.add_parameter(
            "y".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        
        let optimizer = RandomSearchOptimizer::new(space, 100).with_seed(42);
        let objective = QuadraticObjective::new();
        let result = optimizer.optimize(&objective).unwrap();
        
        // Should find a reasonable solution
        let best_x = result.best_parameters.get("x").copied().unwrap();
        let best_y = result.best_parameters.get("y").copied().unwrap();
        
        // Random search should get reasonably close
        assert!((best_x - FixedPoint::from_float(0.5)).abs() < FixedPoint::from_float(1.0));
        assert!((best_y - FixedPoint::from_float(-0.3)).abs() < FixedPoint::from_float(1.0));
        
        assert_eq!(result.evaluations.len(), 100);
        assert!(result.convergence_info.converged);
    }

    #[test]
    fn test_gaussian_process_kernel_functions() {
        let rbf_kernel = KernelFunction::RBF {
            length_scale: FixedPoint::from_float(1.0),
            variance: FixedPoint::from_float(1.0),
        };
        
        let x1 = vec![FixedPoint::zero()];
        let x2 = vec![FixedPoint::zero()];
        let k_same = rbf_kernel.evaluate(&x1, &x2);
        
        let x3 = vec![FixedPoint::from_float(1.0)];
        let k_diff = rbf_kernel.evaluate(&x1, &x3);
        
        // Kernel should be larger for identical points
        assert!(k_same > k_diff);
        // Should not exceed variance
        assert!(k_same <= FixedPoint::from_float(1.0));
        
        // Test Matern kernels
        let matern32_kernel = KernelFunction::Matern32 {
            length_scale: FixedPoint::from_float(1.0),
            variance: FixedPoint::from_float(1.0),
        };
        
        let k_matern_same = matern32_kernel.evaluate(&x1, &x2);
        let k_matern_diff = matern32_kernel.evaluate(&x1, &x3);
        
        assert!(k_matern_same > k_matern_diff);
        assert!(k_matern_same <= FixedPoint::from_float(1.0));
    }

    #[test]
    fn test_gaussian_process_prediction() {
        let kernel = KernelFunction::RBF {
            length_scale: FixedPoint::from_float(1.0),
            variance: FixedPoint::from_float(1.0),
        };
        
        let mut gp = GaussianProcess::new(kernel, FixedPoint::from_float(1e-6));
        
        // Simple training data
        let x_train = vec![
            vec![FixedPoint::from_float(-1.0)],
            vec![FixedPoint::from_float(0.0)],
            vec![FixedPoint::from_float(1.0)],
        ];
        let y_train = vec![
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.0),
            FixedPoint::from_float(1.0),
        ];
        
        gp.fit(x_train, y_train).unwrap();
        
        // Predict at training point
        let (mean, variance) = gp.predict(&[FixedPoint::zero()]).unwrap();
        
        // Should predict close to training value at x=0
        assert!((mean - FixedPoint::zero()).abs() < FixedPoint::from_float(0.1));
        assert!(variance > FixedPoint::zero());
    }

    #[test]
    fn test_acquisition_functions() {
        let mean = FixedPoint::from_float(1.0);
        let variance = FixedPoint::from_float(0.25); // std_dev = 0.5
        let best_value = FixedPoint::from_float(0.5);
        
        // Expected Improvement
        let ei = AcquisitionFunction::ExpectedImprovement;
        let ei_value = ei.evaluate(mean, variance, best_value);
        assert!(ei_value > FixedPoint::zero());
        
        // Upper Confidence Bound
        let ucb = AcquisitionFunction::UpperConfidenceBound {
            beta: FixedPoint::from_float(2.0)
        };
        let ucb_value = ucb.evaluate(mean, variance, best_value);
        // UCB = mean + beta * std_dev = 1.0 + 2.0 * 0.5 = 2.0
        assert!((ucb_value - FixedPoint::from_float(2.0)).abs() < FixedPoint::from_float(0.1));
        
        // Probability of Improvement
        let pi = AcquisitionFunction::ProbabilityOfImprovement;
        let pi_value = pi.evaluate(mean, variance, best_value);
        assert!(pi_value > FixedPoint::zero());
        assert!(pi_value <= FixedPoint::one());
    }

    #[test]
    fn test_bayesian_optimization() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "x".to_string(),
            ParameterBounds::new(FixedPoint::from_float(-2.0), FixedPoint::from_float(2.0))
        );
        
        let kernel = KernelFunction::RBF {
            length_scale: FixedPoint::from_float(1.0),
            variance: FixedPoint::from_float(1.0),
        };
        
        let acquisition = AcquisitionFunction::ExpectedImprovement;
        
        let mut optimizer = BayesianOptimizer::new(space, kernel, acquisition)
            .with_initial_points(5)
            .with_iterations(10)
            .with_seed(42);
        
        // Simple 1D quadratic objective: maximize -(x-0.5)^2
        struct Simple1DObjective;
        impl ObjectiveFunction for Simple1DObjective {
            fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
                let x = parameters.get("x").copied().unwrap_or(FixedPoint::zero());
                let target = FixedPoint::from_float(0.5);
                Ok(-(x - target) * (x - target))
            }
            
            fn name(&self) -> &str {
                "Simple1D"
            }
        }
        
        let objective = Simple1DObjective;
        let result = optimizer.optimize(&objective).unwrap();
        
        // Should find optimum near x = 0.5
        let best_x = result.best_parameters.get("x").copied().unwrap();
        assert!((best_x - FixedPoint::from_float(0.5)).abs() < FixedPoint::from_float(0.5));
        
        // Should have evaluated initial_points + iterations = 15 points
        assert_eq!(result.evaluations.len(), 15);
        assert!(result.convergence_info.converged);
    }

    #[test]
    fn test_cross_validation_fold_creation() {
        let cv = CrossValidationFramework::new(5, ValidationMethod::KFold);
        let folds = cv.create_folds(100).unwrap();
        
        assert_eq!(folds.len(), 5);
        
        // Check that all indices are covered exactly once in test sets
        let mut all_test_indices = Vec::new();
        for (_, test_indices) in &folds {
            all_test_indices.extend(test_indices);
        }
        all_test_indices.sort();
        
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(all_test_indices, expected);
        
        // Check that train and test sets don't overlap
        for (train_indices, test_indices) in &folds {
            for &test_idx in test_indices {
                assert!(!train_indices.contains(&test_idx));
            }
        }
    }

    #[test]
    fn test_time_series_cross_validation() {
        let cv = CrossValidationFramework::new(3, ValidationMethod::TimeSeriesSplit);
        let folds = cv.create_folds(100).unwrap();
        
        // Time series split should have increasing train sizes
        let mut prev_train_size = 0;
        for (train_indices, test_indices) in &folds {
            assert!(train_indices.len() >= prev_train_size);
            assert!(!test_indices.is_empty());
            
            // Test indices should come after train indices
            let max_train = train_indices.iter().max().unwrap_or(&0);
            let min_test = test_indices.iter().min().unwrap_or(&100);
            assert!(max_train < min_test);
            
            prev_train_size = train_indices.len();
        }
    }

    #[test]
    fn test_walk_forward_analysis() {
        let wfa = WalkForwardAnalysis::new(20, 10, 30);
        
        // Simple objective that returns a constant
        struct ConstantObjective(FixedPoint);
        impl ObjectiveFunction for ConstantObjective {
            fn evaluate(&self, _parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
                Ok(self.0)
            }
            
            fn name(&self) -> &str {
                "Constant"
            }
        }
        
        let objective = ConstantObjective(FixedPoint::from_float(1.5));
        let parameters = HashMap::new();
        
        let result = wfa.analyze(&objective, &parameters, 100).unwrap();
        
        assert!(!result.windows.is_empty());
        
        // All windows should have the same performance
        for window in &result.windows {
            assert!((window.performance - FixedPoint::from_float(1.5)).abs() < FixedPoint::from_float(1e-6));
        }
        
        // Mean performance should be 1.5
        assert!((result.mean_performance - FixedPoint::from_float(1.5)).abs() < FixedPoint::from_float(1e-6));
        
        // Standard deviation should be near zero
        assert!(result.performance_std < FixedPoint::from_float(1e-6));
    }

    #[test]
    fn test_optimization_result_structure() {
        let mut space = ParameterSpace::new();
        space.add_parameter(
            "test_param".to_string(),
            ParameterBounds::new(FixedPoint::from_float(0.0), FixedPoint::from_float(1.0))
        );
        
        let optimizer = RandomSearchOptimizer::new(space, 10).with_seed(123);
        
        struct SimpleObjective;
        impl ObjectiveFunction for SimpleObjective {
            fn evaluate(&self, parameters: &HashMap<String, FixedPoint>) -> Result<FixedPoint, OptimizationError> {
                let param = parameters.get("test_param").copied().unwrap_or(FixedPoint::zero());
                Ok(param) // Just return the parameter value
            }
            
            fn name(&self) -> &str {
                "Simple"
            }
        }
        
        let result = optimizer.optimize(&SimpleObjective).unwrap();
        
        // Check result structure
        assert!(result.best_parameters.contains_key("test_param"));
        assert!(result.best_value >= FixedPoint::zero());
        assert!(result.best_value <= FixedPoint::one());
        assert_eq!(result.evaluations.len(), 10);
        assert!(result.convergence_info.converged);
        assert_eq!(result.convergence_info.iterations, 10);
        
        // Check that evaluations are properly recorded
        for eval in &result.evaluations {
            assert!(eval.parameters.contains_key("test_param"));
            assert!(eval.objective_value >= FixedPoint::zero());
            assert!(eval.objective_value <= FixedPoint::one());
        }
    }

    #[test]
    fn test_parameter_optimization_error_handling() {
        // Test invalid bounds
        let bounds = ParameterBounds::new(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.0) // max < min
        );
        
        // This should still work but produce unexpected results
        let mut rng = rand::thread_rng();
        let sample = bounds.sample(&mut rng);
        // The implementation doesn't validate bounds, so this will work
        assert!(sample >= FixedPoint::from_float(0.0));
        
        // Test empty parameter space
        let empty_space = ParameterSpace::new();
        assert_eq!(empty_space.dimension(), 0);
        assert!(empty_space.parameter_names().is_empty());
        
        let mut rng = rand::thread_rng();
        let empty_sample = empty_space.sample(&mut rng);
        assert!(empty_sample.is_empty());
    }
}