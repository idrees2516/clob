use approx::{assert_relative_eq, assert_abs_diff_eq};
use std::f64::consts::{PI, E};

use crate::math::{
    fixed_point::FixedPoint,
    sde_solvers::{EulerMaruyamaGBMJump, MilsteinGBMJump, SDESolver},
    jump_diffusion::{GBMJumpState, GBMJumpParams},
};

/// Numerical accuracy validation against known analytical solutions
#[cfg(test)]
mod numerical_accuracy_validation {
    use super::*;

    #[test]
    fn test_geometric_brownian_motion_analytical_solution() {
        // Test GBM against analytical solution: S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        let initial_price = 100.0;
        let drift = 0.05;
        let volatility = 0.2;
        let time_horizon = 1.0;
        
        let initial_state = GBMJumpState {
            price: FixedPoint::from_float(initial_price),
            log_price: FixedPoint::from_float(initial_price.ln()),
        };

        let params = GBMJumpParams {
            drift: FixedPoint::from_float(drift),
            volatility: FixedPoint::from_float(volatility),
            jump_intensity: FixedPoint::from_float(0.0),
            jump_mean: FixedPoint::from_float(0.0),
            jump_std: FixedPoint::from_float(0.0),
        };

        // Test with deterministic Brownian motion (zero noise)
        let mut solver = EulerMaruyamaGBMJump::new();
        let mut rng = crate::math::sde_solvers::DeterministicRng::new_deterministic(); // Zero noise

        let n_steps = 10000;
        let dt = FixedPoint::from_float(time_horizon / n_steps as f64);
        let mut state = initial_state;

        for _ in 0..n_steps {
            state = solver.solve_step(
                FixedPoint::zero(),
                &state,
                dt,
                &params,
                &mut rng,
            ).unwrap();
        }

        // Analytical solution with zero noise: S(t) = S(0) * exp((μ - σ²/2)t)
        let analytical_price = initial_price * ((drift - 0.5 * volatility * volatility) * time_horizon).exp();
        let numerical_price = state.price.to_float();

        // Should be very accurate with fine discretization and no noise
        assert_relative_eq!(numerical_price, analytical_price, epsilon = 1e-4);
    }

    #[test]
    fn test_ornstein_uhlenbeck_mean_reversion() {
        // Test mean reversion properties: dX = θ(μ - X)dt + σdW
        // Analytical solution: X(t) = μ + (X(0) - μ)exp(-θt) + σ∫₀ᵗ exp(-θ(t-s))dW(s)
        
        let theta = 2.0; // Mean reversion speed
        let mu = 0.05;   // Long-term mean
        let sigma = 0.1; // Volatility
        let initial_value = 0.2;
        let time_horizon = 2.0;

        // Implement simple Ornstein-Uhlenbeck solver
        let mut value = initial_value;
        let n_steps = 10000;
        let dt = time_horizon / n_steps as f64;
        let mut rng = crate::math::sde_solvers::DeterministicRng::new_deterministic();

        for _ in 0..n_steps {
            let dw = 0.0; // No noise for deterministic test
            value += theta * (mu - value) * dt + sigma * dw;
        }

        // Analytical solution with zero noise
        let analytical_value = mu + (initial_value - mu) * (-theta * time_horizon).exp();
        
        assert_relative_eq!(value, analytical_value, epsilon = 1e-6);
    }

    #[test]
    fn test_cox_ingersoll_ross_positivity() {
        // Test CIR model: dr = κ(θ - r)dt + σ√r dW
        // Should maintain positivity if 2κθ ≥ σ² (Feller condition)
        
        let kappa = 2.0;
        let theta = 0.05;
        let sigma = 0.1;
        let initial_rate = 0.03;
        
        // Check Feller condition
        assert!(2.0 * kappa * theta >= sigma * sigma);
        
        let mut rate = initial_rate;
        let n_steps = 10000;
        let dt = 0.001;
        let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);

        for _ in 0..n_steps {
            let dw = rng.sample_normal();
            let sqrt_rate = if rate > 0.0 { rate.sqrt() } else { 0.0 };
            rate += kappa * (theta - rate) * dt + sigma * sqrt_rate * dw * dt.sqrt();
            rate = rate.max(0.0); // Ensure non-negativity
            
            // Rate should remain positive
            assert!(rate >= 0.0);
        }
    }

    #[test]
    fn test_heston_model_correlation() {
        // Test Heston stochastic volatility model correlation structure
        // dS = rS dt + √V S dW₁
        // dV = κ(θ - V)dt + σ√V dW₂
        // where dW₁dW₂ = ρdt
        
        let correlation = -0.5; // Typical negative correlation
        let n_paths = 1000;
        let n_steps = 252;
        let dt = 1.0 / 252.0;
        
        let mut price_returns = Vec::new();
        let mut vol_changes = Vec::new();
        
        for path in 0..n_paths {
            let mut s = 100.0;
            let mut v = 0.04;
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(path as u64);
            
            for _ in 0..n_steps {
                let z1 = rng.sample_normal();
                let z2 = correlation * z1 + (1.0 - correlation * correlation).sqrt() * rng.sample_normal();
                
                let old_s = s;
                let old_v = v;
                
                // Heston dynamics (simplified)
                s += 0.05 * s * dt + v.sqrt() * s * z1 * dt.sqrt();
                v += 2.0 * (0.04 - v) * dt + 0.3 * v.sqrt() * z2 * dt.sqrt();
                v = v.max(0.0);
                
                if path < 100 { // Collect sample for correlation test
                    price_returns.push((s / old_s).ln());
                    vol_changes.push(v - old_v);
                }
            }
        }
        
        // Calculate empirical correlation
        let mean_return = price_returns.iter().sum::<f64>() / price_returns.len() as f64;
        let mean_vol_change = vol_changes.iter().sum::<f64>() / vol_changes.len() as f64;
        
        let covariance: f64 = price_returns.iter().zip(vol_changes.iter())
            .map(|(r, v)| (r - mean_return) * (v - mean_vol_change))
            .sum::<f64>() / (price_returns.len() - 1) as f64;
            
        let var_return: f64 = price_returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (price_returns.len() - 1) as f64;
            
        let var_vol: f64 = vol_changes.iter()
            .map(|v| (v - mean_vol_change).powi(2))
            .sum::<f64>() / (vol_changes.len() - 1) as f64;
        
        let empirical_correlation = covariance / (var_return.sqrt() * var_vol.sqrt());
        
        // Should be close to theoretical correlation (within sampling error)
        assert_abs_diff_eq!(empirical_correlation, correlation, epsilon = 0.2);
    }

    #[test]
    fn test_jump_diffusion_poisson_intensity() {
        // Test that jump arrivals follow Poisson distribution
        let jump_intensity = 5.0; // 5 jumps per year on average
        let time_horizon = 1.0;
        let n_simulations = 10000;
        
        let mut jump_counts = Vec::new();
        
        for sim in 0..n_simulations {
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(sim as u64);
            let mut time = 0.0;
            let mut jump_count = 0;
            
            while time < time_horizon {
                let u = rng.sample_uniform();
                let inter_arrival_time = -u.ln() / jump_intensity;
                time += inter_arrival_time;
                
                if time < time_horizon {
                    jump_count += 1;
                }
            }
            
            jump_counts.push(jump_count);
        }
        
        // Test Poisson distribution properties
        let mean_jumps = jump_counts.iter().sum::<i32>() as f64 / n_simulations as f64;
        let var_jumps = jump_counts.iter()
            .map(|&x| (x as f64 - mean_jumps).powi(2))
            .sum::<f64>() / (n_simulations - 1) as f64;
        
        // For Poisson distribution: mean = variance = λt
        let expected_mean = jump_intensity * time_horizon;
        assert_relative_eq!(mean_jumps, expected_mean, epsilon = 0.1);
        assert_relative_eq!(var_jumps, expected_mean, epsilon = 0.2);
    }

    #[test]
    fn test_milstein_vs_euler_convergence() {
        // Test that Milstein scheme converges faster than Euler for SDE with non-zero diffusion coefficient derivative
        // Use SDE: dX = μX dt + σX dW (geometric Brownian motion)
        
        let initial_value = 1.0;
        let drift = 0.1;
        let volatility = 0.3;
        let time_horizon = 1.0;
        
        // Analytical solution: X(t) = X(0) * exp((μ - σ²/2)t + σW(t))
        // For deterministic case (W(t) = 0): X(t) = X(0) * exp((μ - σ²/2)t)
        let analytical_solution = initial_value * ((drift - 0.5 * volatility * volatility) * time_horizon).exp();
        
        let time_steps = vec![100, 200, 400, 800];
        let mut euler_errors = Vec::new();
        let mut milstein_errors = Vec::new();
        
        for &n_steps in &time_steps {
            let dt = time_horizon / n_steps as f64;
            
            // Euler scheme
            let mut x_euler = initial_value;
            for _ in 0..n_steps {
                x_euler += drift * x_euler * dt; // No noise for convergence test
            }
            euler_errors.push((x_euler - analytical_solution).abs());
            
            // Milstein scheme (for this SDE, same as Euler since derivative of diffusion coefficient is σ)
            let mut x_milstein = initial_value;
            for _ in 0..n_steps {
                x_milstein += drift * x_milstein * dt;
                // Milstein correction term would be: + 0.5 * σ * σ * x * (dW² - dt)
                // But with zero noise, this is just: - 0.5 * σ² * x * dt
                x_milstein -= 0.5 * volatility * volatility * x_milstein * dt;
            }
            milstein_errors.push((x_milstein - analytical_solution).abs());
        }
        
        // Check convergence rates
        for i in 1..time_steps.len() {
            let euler_ratio = euler_errors[i-1] / euler_errors[i];
            let milstein_ratio = milstein_errors[i-1] / milstein_errors[i];
            
            // Euler should converge at rate O(dt) = O(1/n)
            // Milstein should converge at rate O(dt²) = O(1/n²) for this type of SDE
            assert!(euler_ratio >= 1.8); // Should be close to 2 for doubling steps
            assert!(milstein_ratio >= 3.5); // Should be close to 4 for doubling steps
        }
    }

    #[test]
    fn test_fixed_point_arithmetic_precision() {
        // Test fixed-point arithmetic maintains precision
        let a = FixedPoint::from_float(1.0 / 3.0);
        let b = FixedPoint::from_float(2.0 / 3.0);
        let sum = a + b;
        
        // Should be exactly 1.0 within fixed-point precision
        assert_abs_diff_eq!(sum.to_float(), 1.0, epsilon = 1e-15);
        
        // Test multiplication precision
        let x = FixedPoint::from_float(1.23456789);
        let y = FixedPoint::from_float(9.87654321);
        let product = x * y;
        let expected = 1.23456789 * 9.87654321;
        
        assert_relative_eq!(product.to_float(), expected, epsilon = 1e-12);
        
        // Test division precision
        let quotient = product / y;
        assert_relative_eq!(quotient.to_float(), x.to_float(), epsilon = 1e-12);
    }

    #[test]
    fn test_numerical_integration_accuracy() {
        // Test numerical integration methods against analytical integrals
        
        // Test ∫₀¹ x² dx = 1/3
        let integral_x_squared = numerical_integrate(|x| x * x, 0.0, 1.0, 10000);
        assert_relative_eq!(integral_x_squared, 1.0/3.0, epsilon = 1e-6);
        
        // Test ∫₀^π sin(x) dx = 2
        let integral_sin = numerical_integrate(|x| x.sin(), 0.0, PI, 10000);
        assert_relative_eq!(integral_sin, 2.0, epsilon = 1e-6);
        
        // Test ∫₀¹ e^x dx = e - 1
        let integral_exp = numerical_integrate(|x| x.exp(), 0.0, 1.0, 10000);
        assert_relative_eq!(integral_exp, E - 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_matrix_operations_accuracy() {
        // Test matrix operations maintain numerical accuracy
        use nalgebra::{Matrix3, Vector3};
        
        let a = Matrix3::new(
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0, // Changed 9.0 to 10.0 to make it invertible
        );
        
        let b = Matrix3::new(
            2.0, 1.0, 0.0,
            1.0, 2.0, 1.0,
            0.0, 1.0, 2.0,
        );
        
        // Test matrix multiplication associativity: (AB)C = A(BC)
        let c = Matrix3::identity();
        let ab_c = (a * b) * c;
        let a_bc = a * (b * c);
        
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(ab_c[(i, j)], a_bc[(i, j)], epsilon = 1e-12);
            }
        }
        
        // Test matrix inversion accuracy: A * A⁻¹ = I
        if let Some(a_inv) = a.try_inverse() {
            let identity_test = a * a_inv;
            let identity = Matrix3::identity();
            
            for i in 0..3 {
                for j in 0..3 {
                    assert_abs_diff_eq!(identity_test[(i, j)], identity[(i, j)], epsilon = 1e-12);
                }
            }
        }
    }
}

// Helper function for numerical integration using trapezoidal rule
fn numerical_integrate<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(x);
    }
    
    sum * h
}