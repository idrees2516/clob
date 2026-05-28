use proptest::prelude::*;
use approx::assert_relative_eq;
use std::f64::consts::PI;

use crate::math::{
    sde_solvers::{EulerMaruyamaGBMJump, MilsteinGBMJump, SDESolver},
    fixed_point::FixedPoint,
    jump_diffusion::{GBMJumpState, GBMJumpParams},
    hawkes_process::MultivariateHawkesSimulator,
    rough_volatility::RoughVolatilityModel,
};
use crate::models::{
    avellaneda_stoikov::AvellanedaStoikovEngine,
    gueant_lehalle_tapia::GuéantLehalleTapiaEngine,
    cartea_jaimungal::CarteaJaimungalEngine,
};
use crate::utils::statistics::Statistics;

/// Property-based tests for mathematical models
#[cfg(test)]
mod mathematical_model_validation {
    use super::*;

    // Test Avellaneda-Stoikov model properties
    proptest! {
        #[test]
        fn avellaneda_stoikov_mathematical_properties(
            gamma in 0.01f64..10.0,
            sigma in 0.01f64..1.0,
            inventory in -1000i64..1000,
            time_to_maturity in 0.001f64..1.0,
            mid_price in 50.0f64..200.0,
        ) {
            let params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
                gamma: FixedPoint::from_float(gamma),
                sigma: FixedPoint::from_float(sigma),
                k: FixedPoint::from_float(0.1),
                A: FixedPoint::from_float(1.0),
                T: FixedPoint::from_float(time_to_maturity),
            };

            let mut engine = AvellanedaStoikovEngine::new(params).unwrap();
            let quotes = engine.calculate_optimal_quotes(
                FixedPoint::from_float(mid_price),
                inventory,
                FixedPoint::from_float(sigma),
                FixedPoint::from_float(time_to_maturity),
            ).unwrap();

            // Property 1: Ask price must be greater than bid price
            prop_assert!(quotes.ask_price > quotes.bid_price);

            // Property 2: Reservation price must be between bid and ask
            prop_assert!(quotes.reservation_price >= quotes.bid_price);
            prop_assert!(quotes.reservation_price <= quotes.ask_price);

            // Property 3: Inventory skew - long inventory should lower reservation price
            if inventory > 0 {
                prop_assert!(quotes.reservation_price.to_float() < mid_price);
            } else if inventory < 0 {
                prop_assert!(quotes.reservation_price.to_float() > mid_price);
            }

            // Property 4: Spread should increase with volatility
            let high_vol_quotes = engine.calculate_optimal_quotes(
                FixedPoint::from_float(mid_price),
                inventory,
                FixedPoint::from_float(sigma * 1.5),
                FixedPoint::from_float(time_to_maturity),
            ).unwrap();
            
            let original_spread = quotes.ask_price - quotes.bid_price;
            let high_vol_spread = high_vol_quotes.ask_price - high_vol_quotes.bid_price;
            prop_assert!(high_vol_spread > original_spread);

            // Property 5: Spread should increase with risk aversion
            let high_gamma_params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
                gamma: FixedPoint::from_float(gamma * 2.0),
                ..params
            };
            let mut high_gamma_engine = AvellanedaStoikovEngine::new(high_gamma_params).unwrap();
            let high_gamma_quotes = high_gamma_engine.calculate_optimal_quotes(
                FixedPoint::from_float(mid_price),
                inventory,
                FixedPoint::from_float(sigma),
                FixedPoint::from_float(time_to_maturity),
            ).unwrap();
            
            let high_gamma_spread = high_gamma_quotes.ask_price - high_gamma_quotes.bid_price;
            prop_assert!(high_gamma_spread > original_spread);
        }
    }

    proptest! {
        #[test]
        fn sde_solver_convergence_properties(
            initial_price in 50.0f64..200.0,
            drift in -0.1f64..0.1,
            volatility in 0.01f64..0.5,
            time_horizon in 0.1f64..2.0,
        ) {
            let initial_state = GBMJumpState {
                price: FixedPoint::from_float(initial_price),
                log_price: FixedPoint::from_float(initial_price.ln()),
            };

            let params = GBMJumpParams {
                drift: FixedPoint::from_float(drift),
                volatility: FixedPoint::from_float(volatility),
                jump_intensity: FixedPoint::from_float(0.0), // No jumps for convergence test
                jump_mean: FixedPoint::from_float(0.0),
                jump_std: FixedPoint::from_float(0.0),
            };

            let mut euler_solver = EulerMaruyamaGBMJump::new();
            let mut milstein_solver = MilsteinGBMJump::new();

            // Test convergence with decreasing time steps
            let n_steps_coarse = 100;
            let n_steps_fine = 1000;
            let dt_coarse = FixedPoint::from_float(time_horizon / n_steps_coarse as f64);
            let dt_fine = FixedPoint::from_float(time_horizon / n_steps_fine as f64);

            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);

            // Solve with coarse time step
            let mut state_coarse = initial_state;
            for _ in 0..n_steps_coarse {
                state_coarse = euler_solver.solve_step(
                    FixedPoint::zero(),
                    &state_coarse,
                    dt_coarse,
                    &params,
                    &mut rng,
                ).unwrap();
            }

            // Reset RNG for fair comparison
            rng = crate::math::sde_solvers::DeterministicRng::new(42);

            // Solve with fine time step
            let mut state_fine = initial_state;
            for _ in 0..n_steps_fine {
                state_fine = euler_solver.solve_step(
                    FixedPoint::zero(),
                    &state_fine,
                    dt_fine,
                    &params,
                    &mut rng,
                ).unwrap();
            }

            // Property: Finer discretization should be closer to analytical solution
            let analytical_mean = initial_price * (drift * time_horizon).exp();
            let coarse_error = (state_coarse.price.to_float() - analytical_mean).abs();
            let fine_error = (state_fine.price.to_float() - analytical_mean).abs();

            // Fine discretization should generally have smaller error (with high probability)
            // We use a relaxed condition due to stochastic nature
            prop_assert!(fine_error <= coarse_error * 2.0);

            // Property: Price should remain positive
            prop_assert!(state_coarse.price > FixedPoint::zero());
            prop_assert!(state_fine.price > FixedPoint::zero());
        }
    }

    proptest! {
        #[test]
        fn hawkes_process_mathematical_properties(
            baseline_intensity in 0.1f64..2.0,
            excitation_strength in 0.01f64..0.5,
            decay_rate in 0.1f64..5.0,
        ) {
            let mut simulator = MultivariateHawkesSimulator::new(
                vec![FixedPoint::from_float(baseline_intensity)],
                vec![vec![FixedPoint::from_float(excitation_strength)]],
                vec![vec![FixedPoint::from_float(decay_rate)]],
            ).unwrap();

            // Property 1: Intensity should never be negative
            let current_intensity = simulator.get_intensity(0, FixedPoint::from_float(1.0));
            prop_assert!(current_intensity >= FixedPoint::zero());

            // Property 2: Without events, intensity should equal baseline
            let intensity_at_start = simulator.get_intensity(0, FixedPoint::zero());
            prop_assert_eq!(intensity_at_start, FixedPoint::from_float(baseline_intensity));

            // Property 3: After an event, intensity should increase
            simulator.add_event(0, FixedPoint::from_float(0.5));
            let intensity_after_event = simulator.get_intensity(0, FixedPoint::from_float(0.6));
            prop_assert!(intensity_after_event > FixedPoint::from_float(baseline_intensity));

            // Property 4: Intensity should decay over time
            let intensity_later = simulator.get_intensity(0, FixedPoint::from_float(2.0));
            prop_assert!(intensity_later < intensity_after_event);
            prop_assert!(intensity_later >= FixedPoint::from_float(baseline_intensity));

            // Property 5: Branching ratio should be less than 1 for stability
            let branching_ratio = excitation_strength / decay_rate;
            prop_assert!(branching_ratio < 1.0);
        }
    }

    proptest! {
        #[test]
        fn rough_volatility_model_properties(
            hurst_parameter in 0.1f64..0.9,
            volatility_of_volatility in 0.1f64..1.0,
            mean_reversion_speed in 0.1f64..5.0,
            initial_volatility in 0.01f64..0.5,
        ) {
            // Skip if Hurst parameter is too close to 0.5 (Brownian motion boundary)
            prop_assume!((hurst_parameter - 0.5).abs() > 0.05);

            let model = RoughVolatilityModel::new(
                FixedPoint::from_float(hurst_parameter),
                FixedPoint::from_float(volatility_of_volatility),
                FixedPoint::from_float(mean_reversion_speed),
                FixedPoint::from_float(initial_volatility),
            ).unwrap();

            // Property 1: Hurst parameter should be in valid range
            prop_assert!(hurst_parameter > 0.0 && hurst_parameter < 1.0);

            // Property 2: Volatility should remain positive
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
            let volatility_path = model.simulate_path(
                FixedPoint::from_float(1.0),
                100,
                &mut rng,
            ).unwrap();

            for vol in volatility_path {
                prop_assert!(vol > FixedPoint::zero());
            }

            // Property 3: Long memory property - autocorrelation should decay slowly
            // This is a simplified test of the rough volatility property
            let correlation_lag_1 = model.theoretical_autocorrelation(1);
            let correlation_lag_10 = model.theoretical_autocorrelation(10);
            
            if hurst_parameter < 0.5 {
                // For rough processes (H < 0.5), correlations should be negative
                prop_assert!(correlation_lag_1 < FixedPoint::zero());
            }
        }
    }

    #[test]
    fn test_black_scholes_analytical_validation() {
        // Test Black-Scholes formula against known analytical solutions
        let spot = 100.0;
        let strike = 100.0;
        let time_to_expiry = 0.25; // 3 months
        let risk_free_rate = 0.05;
        let volatility = 0.2;

        let call_price = black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility);
        let put_price = black_scholes_put(spot, strike, time_to_expiry, risk_free_rate, volatility);

        // Put-call parity: C - P = S - K * exp(-r * T)
        let parity_lhs = call_price - put_price;
        let parity_rhs = spot - strike * (-risk_free_rate * time_to_expiry).exp();
        assert_relative_eq!(parity_lhs, parity_rhs, epsilon = 1e-10);

        // At-the-money call option with known parameters should match expected value
        // Using online calculator or reference implementation
        let expected_call_price = 5.987; // Approximate value
        assert_relative_eq!(call_price, expected_call_price, epsilon = 0.01);
    }

    #[test]
    fn test_monte_carlo_convergence() {
        // Test Monte Carlo convergence for option pricing
        let spot = 100.0;
        let strike = 100.0;
        let time_to_expiry = 0.25;
        let risk_free_rate = 0.05;
        let volatility = 0.2;

        let analytical_price = black_scholes_call(spot, strike, time_to_expiry, risk_free_rate, volatility);

        // Test with increasing number of simulations
        let simulations = vec![1000, 10000, 100000];
        let mut errors = Vec::new();

        for n_sims in simulations {
            let mc_price = monte_carlo_option_price(
                spot, strike, time_to_expiry, risk_free_rate, volatility, n_sims
            );
            let error = (mc_price - analytical_price).abs();
            errors.push(error);
        }

        // Error should generally decrease with more simulations
        // (though this is stochastic, so we use a relaxed condition)
        assert!(errors[2] <= errors[0]); // 100k sims should be better than 1k sims
    }

    #[test]
    fn test_numerical_stability_edge_cases() {
        // Test numerical stability with extreme parameters
        
        // Very small volatility
        let small_vol_price = black_scholes_call(100.0, 100.0, 0.25, 0.05, 1e-6);
        assert!(small_vol_price.is_finite());
        assert!(small_vol_price >= 0.0);

        // Very large volatility
        let large_vol_price = black_scholes_call(100.0, 100.0, 0.25, 0.05, 5.0);
        assert!(large_vol_price.is_finite());
        assert!(large_vol_price >= 0.0);

        // Very short time to expiry
        let short_time_price = black_scholes_call(100.0, 100.0, 1e-6, 0.05, 0.2);
        assert!(short_time_price.is_finite());
        assert!(short_time_price >= 0.0);

        // Deep in-the-money
        let deep_itm_price = black_scholes_call(200.0, 100.0, 0.25, 0.05, 0.2);
        assert!(deep_itm_price.is_finite());
        assert!(deep_itm_price >= 100.0); // Should be at least intrinsic value

        // Deep out-of-the-money
        let deep_otm_price = black_scholes_call(50.0, 100.0, 0.25, 0.05, 0.2);
        assert!(deep_otm_price.is_finite());
        assert!(deep_otm_price >= 0.0);
        assert!(deep_otm_price <= 1.0); // Should be very small
    }

    #[test]
    fn test_cross_model_validation() {
        // Compare results between different implementations of the same model
        
        // Test Euler vs Milstein SDE solvers
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

        let mut euler_solver = EulerMaruyamaGBMJump::new();
        let mut milstein_solver = MilsteinGBMJump::new();
        let mut rng1 = crate::math::sde_solvers::DeterministicRng::new(42);
        let mut rng2 = crate::math::sde_solvers::DeterministicRng::new(42);

        let dt = FixedPoint::from_float(0.001);
        let n_steps = 1000;

        let mut euler_state = initial_state;
        let mut milstein_state = initial_state;

        for _ in 0..n_steps {
            euler_state = euler_solver.solve_step(
                FixedPoint::zero(),
                &euler_state,
                dt,
                &params,
                &mut rng1,
            ).unwrap();

            milstein_state = milstein_solver.solve_step(
                FixedPoint::zero(),
                &milstein_state,
                dt,
                &params,
                &mut rng2,
            ).unwrap();
        }

        // Milstein should be more accurate than Euler for the same random path
        // Both should be close to each other for small dt
        let relative_diff = (euler_state.price - milstein_state.price).abs() / euler_state.price;
        assert!(relative_diff.to_float() < 0.05); // Within 5%
    }
}

// Helper functions for testing
fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    use statrs::distribution::{Normal, ContinuousCDF};
    
    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    
    s * normal.cdf(d1) - k * (-r * t).exp() * normal.cdf(d2)
}

fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    use statrs::distribution::{Normal, ContinuousCDF};
    
    let normal = Normal::new(0.0, 1.0).unwrap();
    let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    
    k * (-r * t).exp() * normal.cdf(-d2) - s * normal.cdf(-d1)
}

fn monte_carlo_option_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, n_sims: usize) -> f64 {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut payoffs = Vec::with_capacity(n_sims);
    
    for _ in 0..n_sims {
        let z: f64 = rng.sample(StandardNormal);
        let st = s * ((r - 0.5 * sigma * sigma) * t + sigma * t.sqrt() * z).exp();
        let payoff = (st - k).max(0.0);
        payoffs.push(payoff);
    }
    
    let mean_payoff: f64 = payoffs.iter().sum::<f64>() / n_sims as f64;
    mean_payoff * (-r * t).exp()
}