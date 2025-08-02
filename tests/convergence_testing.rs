use approx::{assert_relative_eq, assert_abs_diff_eq};
use std::collections::HashMap;

use crate::math::{
    sde_solvers::{EulerMaruyamaGBMJump, MilsteinGBMJump, SDESolver},
    fixed_point::FixedPoint,
    jump_diffusion::{GBMJumpState, GBMJumpParams},
    hawkes_process::MultivariateHawkesSimulator,
};
use crate::models::avellaneda_stoikov::AvellanedaStoikovEngine;
use crate::optimization::OptimizationAlgorithm;

/// Convergence testing for SDE solvers and optimization algorithms
#[cfg(test)]
mod convergence_testing {
    use super::*;

    #[test]
    fn test_euler_maruyama_strong_convergence() {
        // Test strong convergence of Euler-Maruyama scheme
        // Strong convergence rate should be O(√Δt) = O(1/√n)
        
        let initial_state = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            log_price: FixedPoint::from_float(100.0_f64.ln()),
        };

        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.0),
            jump_mean: FixedPoint::from_float(0.0),
            jump_std: FixedPoint::from_float(0.0),
        };

        let time_horizon = 1.0;
        let step_sizes = vec![1000, 2000, 4000, 8000];
        let n_paths = 1000;
        
        let mut errors = Vec::new();
        
        for &n_steps in &step_sizes {
            let dt = FixedPoint::from_float(time_horizon / n_steps as f64);
            let mut path_errors = Vec::new();
            
            for path in 0..n_paths {
                let mut solver = EulerMaruyamaGBMJump::new();
                let mut rng = crate::math::sde_solvers::DeterministicRng::new(path as u64);
                
                // Coarse path
                let mut state_coarse = initial_state;
                for _ in 0..n_steps {
                    state_coarse = solver.solve_step(
                        FixedPoint::zero(),
                        &state_coarse,
                        dt,
                        &params,
                        &mut rng,
                    ).unwrap();
                }
                
                // Fine path (double the steps)
                let mut rng_fine = crate::math::sde_solvers::DeterministicRng::new(path as u64);
                let dt_fine = dt / FixedPoint::from_float(2.0);
                let mut state_fine = initial_state;
                for _ in 0..(n_steps * 2) {
                    state_fine = solver.solve_step(
                        FixedPoint::zero(),
                        &state_fine,
                        dt_fine,
                        &params,
                        &mut rng_fine,
                    ).unwrap();
                }
                
                let error = (state_coarse.price - state_fine.price).abs();
                path_errors.push(error.to_float());
            }
            
            let mean_error = path_errors.iter().sum::<f64>() / n_paths as f64;
            errors.push(mean_error);
        }
        
        // Check convergence rate
        for i in 1..errors.len() {
            let ratio = errors[i-1] / errors[i];
            // Should be approximately √2 ≈ 1.414 for strong convergence O(√Δt)
            assert!(ratio >= 1.2 && ratio <= 1.8, 
                "Convergence ratio {} not in expected range [1.2, 1.8]", ratio);
        }
    }

    #[test]
    fn test_milstein_strong_convergence() {
        // Test strong convergence of Milstein scheme
        // Should have better convergence rate than Euler-Maruyama
        
        let initial_state = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            log_price: FixedPoint::from_float(100.0_f64.ln()),
        };

        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.3), // Higher volatility to see Milstein advantage
            jump_intensity: FixedPoint::from_float(0.0),
            jump_mean: FixedPoint::from_float(0.0),
            jump_std: FixedPoint::from_float(0.0),
        };

        let time_horizon = 1.0;
        let step_sizes = vec![500, 1000, 2000, 4000];
        let n_paths = 500;
        
        let mut euler_errors = Vec::new();
        let mut milstein_errors = Vec::new();
        
        for &n_steps in &step_sizes {
            let dt = FixedPoint::from_float(time_horizon / n_steps as f64);
            let mut euler_path_errors = Vec::new();
            let mut milstein_path_errors = Vec::new();
            
            for path in 0..n_paths {
                // Reference solution (very fine Milstein)
                let mut ref_solver = MilsteinGBMJump::new();
                let mut ref_rng = crate::math::sde_solvers::DeterministicRng::new(path as u64);
                let ref_dt = FixedPoint::from_float(time_horizon / (n_steps * 10) as f64);
                let mut ref_state = initial_state;
                
                for _ in 0..(n_steps * 10) {
                    ref_state = ref_solver.solve_step(
                        FixedPoint::zero(),
                        &ref_state,
                        ref_dt,
                        &params,
                        &mut ref_rng,
                    ).unwrap();
                }
                
                // Euler solution
                let mut euler_solver = EulerMaruyamaGBMJump::new();
                let mut euler_rng = crate::math::sde_solvers::DeterministicRng::new(path as u64);
                let mut euler_state = initial_state;
                
                for _ in 0..n_steps {
                    euler_state = euler_solver.solve_step(
                        FixedPoint::zero(),
                        &euler_state,
                        dt,
                        &params,
                        &mut euler_rng,
                    ).unwrap();
                }
                
                // Milstein solution
                let mut milstein_solver = MilsteinGBMJump::new();
                let mut milstein_rng = crate::math::sde_solvers::DeterministicRng::new(path as u64);
                let mut milstein_state = initial_state;
                
                for _ in 0..n_steps {
                    milstein_state = milstein_solver.solve_step(
                        FixedPoint::zero(),
                        &milstein_state,
                        dt,
                        &params,
                        &mut milstein_rng,
                    ).unwrap();
                }
                
                let euler_error = (euler_state.price - ref_state.price).abs().to_float();
                let milstein_error = (milstein_state.price - ref_state.price).abs().to_float();
                
                euler_path_errors.push(euler_error);
                milstein_path_errors.push(milstein_error);
            }
            
            let euler_mean_error = euler_path_errors.iter().sum::<f64>() / n_paths as f64;
            let milstein_mean_error = milstein_path_errors.iter().sum::<f64>() / n_paths as f64;
            
            euler_errors.push(euler_mean_error);
            milstein_errors.push(milstein_mean_error);
        }
        
        // Milstein should be more accurate than Euler for all step sizes
        for i in 0..step_sizes.len() {
            assert!(milstein_errors[i] <= euler_errors[i], 
                "Milstein error {} should be <= Euler error {} for step size {}", 
                milstein_errors[i], euler_errors[i], step_sizes[i]);
        }
    }

    #[test]
    fn test_hawkes_intensity_convergence() {
        // Test convergence of Hawkes process intensity calculation
        
        let baseline_intensity = FixedPoint::from_float(1.0);
        let excitation_strength = FixedPoint::from_float(0.5);
        let decay_rate = FixedPoint::from_float(2.0);
        
        let mut simulator = MultivariateHawkesSimulator::new(
            vec![baseline_intensity],
            vec![vec![excitation_strength]],
            vec![vec![decay_rate]],
        ).unwrap();
        
        // Add some events
        let event_times = vec![0.1, 0.3, 0.7, 1.2, 1.8];
        for &t in &event_times {
            simulator.add_event(0, FixedPoint::from_float(t));
        }
        
        let query_time = FixedPoint::from_float(2.0);
        
        // Calculate intensity with different levels of precision
        let precisions = vec![100, 1000, 10000];
        let mut intensities = Vec::new();
        
        for &precision in &precisions {
            simulator.set_calculation_precision(precision);
            let intensity = simulator.get_intensity(0, query_time);
            intensities.push(intensity.to_float());
        }
        
        // Higher precision should converge to more accurate value
        for i in 1..intensities.len() {
            let relative_change = (intensities[i] - intensities[i-1]).abs() / intensities[i-1];
            assert!(relative_change < 0.01, 
                "Intensity should converge with higher precision, but change was {}", 
                relative_change);
        }
        
        // Analytical check: intensity should be >= baseline
        for &intensity in &intensities {
            assert!(intensity >= baseline_intensity.to_float());
        }
    }

    #[test]
    fn test_optimization_algorithm_convergence() {
        // Test convergence of optimization algorithms
        
        // Simple quadratic function: f(x) = (x - 2)² + 1, minimum at x = 2
        let objective = |x: &[f64]| (x[0] - 2.0).powi(2) + 1.0;
        let gradient = |x: &[f64]| vec![2.0 * (x[0] - 2.0)];
        
        let initial_guess = vec![0.0];
        let tolerance_levels = vec![1e-2, 1e-4, 1e-6, 1e-8];
        
        for &tolerance in &tolerance_levels {
            let mut optimizer = crate::optimization::GradientDescent::new(
                0.1, // learning rate
                tolerance,
                1000, // max iterations
            );
            
            let result = optimizer.optimize(&objective, &gradient, initial_guess.clone()).unwrap();
            
            // Should converge closer to true minimum with tighter tolerance
            let error = (result.solution[0] - 2.0).abs();
            assert!(error <= tolerance * 10.0, 
                "Optimization error {} should be within tolerance {}", 
                error, tolerance);
        }
    }

    #[test]
    fn test_avellaneda_stoikov_parameter_convergence() {
        // Test that Avellaneda-Stoikov model converges to expected behavior
        // as parameters approach limiting cases
        
        let base_params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
            gamma: FixedPoint::from_float(1.0),
            sigma: FixedPoint::from_float(0.2),
            k: FixedPoint::from_float(0.1),
            A: FixedPoint::from_float(1.0),
            T: FixedPoint::from_float(1.0),
        };
        
        // Test convergence as risk aversion γ → 0 (risk-neutral limit)
        let gamma_values = vec![1.0, 0.1, 0.01, 0.001];
        let mut spreads = Vec::new();
        
        for &gamma in &gamma_values {
            let params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
                gamma: FixedPoint::from_float(gamma),
                ..base_params
            };
            
            let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
            let quotes = engine.calculate_optimal_quotes(
                FixedPoint::from_float(100.0),
                0, // Zero inventory
                FixedPoint::from_float(0.2),
                FixedPoint::from_float(1.0),
            ).unwrap();
            
            let spread = quotes.ask_price - quotes.bid_price;
            spreads.push(spread.to_float());
        }
        
        // Spreads should decrease as risk aversion decreases
        for i in 1..spreads.len() {
            assert!(spreads[i] < spreads[i-1], 
                "Spread should decrease as risk aversion decreases");
        }
        
        // Test convergence as time to maturity T → 0
        let time_values = vec![1.0, 0.1, 0.01, 0.001];
        let mut reservation_prices = Vec::new();
        
        let inventory = 100i64; // Non-zero inventory
        
        for &time_to_maturity in &time_values {
            let mut engine = AvellanedaStoikovEngine::new(base_params).unwrap();
            let quotes = engine.calculate_optimal_quotes(
                FixedPoint::from_float(100.0),
                inventory,
                FixedPoint::from_float(0.2),
                FixedPoint::from_float(time_to_maturity),
            ).unwrap();
            
            reservation_prices.push(quotes.reservation_price.to_float());
        }
        
        // As T → 0, reservation price should approach mid price (urgency effect)
        let mid_price = 100.0;
        for i in 1..reservation_prices.len() {
            let distance_to_mid = (reservation_prices[i] - mid_price).abs();
            let prev_distance_to_mid = (reservation_prices[i-1] - mid_price).abs();
            assert!(distance_to_mid <= prev_distance_to_mid, 
                "Reservation price should approach mid price as time decreases");
        }
    }

    #[test]
    fn test_monte_carlo_convergence() {
        // Test Monte Carlo convergence using Central Limit Theorem
        
        // Estimate π using Monte Carlo: π ≈ 4 * (points inside unit circle) / (total points)
        let sample_sizes = vec![1000, 10000, 100000, 1000000];
        let mut estimates = Vec::new();
        let mut standard_errors = Vec::new();
        
        for &n_samples in &sample_sizes {
            let mut inside_circle = 0;
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
            
            for _ in 0..n_samples {
                let x = rng.sample_uniform() * 2.0 - 1.0; // [-1, 1]
                let y = rng.sample_uniform() * 2.0 - 1.0; // [-1, 1]
                
                if x * x + y * y <= 1.0 {
                    inside_circle += 1;
                }
            }
            
            let pi_estimate = 4.0 * inside_circle as f64 / n_samples as f64;
            estimates.push(pi_estimate);
            
            // Standard error should decrease as O(1/√n)
            let p = inside_circle as f64 / n_samples as f64;
            let standard_error = 4.0 * (p * (1.0 - p) / n_samples as f64).sqrt();
            standard_errors.push(standard_error);
        }
        
        // Check that estimates converge to π
        let true_pi = std::f64::consts::PI;
        for i in 1..estimates.len() {
            let error = (estimates[i] - true_pi).abs();
            let prev_error = (estimates[i-1] - true_pi).abs();
            
            // Error should generally decrease (with high probability)
            // We use a relaxed condition due to randomness
            assert!(error <= prev_error * 1.5, 
                "Monte Carlo error should generally decrease with more samples");
        }
        
        // Standard errors should decrease as O(1/√n)
        for i in 1..standard_errors.len() {
            let ratio = standard_errors[i-1] / standard_errors[i];
            let expected_ratio = (sample_sizes[i] as f64 / sample_sizes[i-1] as f64).sqrt();
            
            assert_relative_eq!(ratio, expected_ratio, epsilon = 0.1);
        }
    }

    #[test]
    fn test_fixed_point_iteration_convergence() {
        // Test fixed-point iteration: x_{n+1} = g(x_n)
        // Use g(x) = cos(x), which has a unique fixed point
        
        let g = |x: f64| x.cos();
        let tolerance_levels = vec![1e-3, 1e-6, 1e-9, 1e-12];
        
        for &tolerance in &tolerance_levels {
            let mut x = 1.0; // Initial guess
            let mut iterations = 0;
            let max_iterations = 1000;
            
            loop {
                let x_new = g(x);
                let error = (x_new - x).abs();
                
                if error < tolerance || iterations >= max_iterations {
                    break;
                }
                
                x = x_new;
                iterations += 1;
            }
            
            // Should converge to the fixed point cos(x*) = x*
            let fixed_point_error = (g(x) - x).abs();
            assert!(fixed_point_error <= tolerance, 
                "Fixed point iteration should converge within tolerance");
            
            // Tighter tolerance should require more iterations but achieve better accuracy
            assert!(iterations < max_iterations, 
                "Fixed point iteration should converge within max iterations");
        }
    }

    #[test]
    fn test_numerical_differentiation_convergence() {
        // Test convergence of numerical differentiation
        
        let f = |x: f64| x.powi(3) + 2.0 * x.powi(2) - x + 1.0;
        let f_prime_analytical = |x: f64| 3.0 * x.powi(2) + 4.0 * x - 1.0;
        
        let x = 2.0;
        let analytical_derivative = f_prime_analytical(x);
        
        let step_sizes = vec![1e-1, 1e-2, 1e-3, 1e-4, 1e-5];
        let mut errors = Vec::new();
        
        for &h in &step_sizes {
            // Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            let numerical_derivative = (f(x + h) - f(x - h)) / (2.0 * h);
            let error = (numerical_derivative - analytical_derivative).abs();
            errors.push(error);
        }
        
        // Error should decrease quadratically with step size for central difference
        for i in 1..errors.len() {
            let ratio = errors[i-1] / errors[i];
            // Should be approximately (h_{i-1}/h_i)² = 10² = 100
            assert!(ratio >= 50.0 && ratio <= 200.0, 
                "Numerical differentiation should converge quadratically");
        }
    }
}

// Helper trait for optimization algorithms
pub trait OptimizationAlgorithm {
    fn optimize<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        initial: Vec<f64>,
    ) -> Result<OptimizationResult, OptimizationError>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>;
}

pub struct OptimizationResult {
    pub solution: Vec<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug)]
pub struct OptimizationError {
    pub message: String,
}

// Simple gradient descent implementation for testing
pub struct GradientDescent {
    learning_rate: f64,
    tolerance: f64,
    max_iterations: usize,
}

impl GradientDescent {
    pub fn new(learning_rate: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            learning_rate,
            tolerance,
            max_iterations,
        }
    }
}

impl OptimizationAlgorithm for GradientDescent {
    fn optimize<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        mut x: Vec<f64>,
    ) -> Result<OptimizationResult, OptimizationError>
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> Vec<f64>,
    {
        for iteration in 0..self.max_iterations {
            let grad = gradient(&x);
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            
            if grad_norm < self.tolerance {
                return Ok(OptimizationResult {
                    solution: x,
                    objective_value: objective(&x),
                    iterations: iteration,
                    converged: true,
                });
            }
            
            // Update: x = x - α * ∇f(x)
            for i in 0..x.len() {
                x[i] -= self.learning_rate * grad[i];
            }
        }
        
        Ok(OptimizationResult {
            solution: x,
            objective_value: objective(&x),
            iterations: self.max_iterations,
            converged: false,
        })
    }
}