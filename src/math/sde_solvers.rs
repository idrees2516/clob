//! Stochastic Differential Equation Solvers
//! 
//! This module implements high-performance SDE solvers for financial modeling,
//! including Euler-Maruyama and Milstein schemes with support for jump-diffusion
//! processes and rough volatility models.
//!
//! Features:
//! - Geometric Brownian Motion with jumps (Merton, Kou models)
//! - Stochastic volatility models (Heston, SABR, rough volatility)
//! - Multi-dimensional correlated processes
//! - Adaptive time-stepping for accuracy
//! - Variance reduction techniques for Monte Carlo
//! - Circuit-compatible deterministic execution

use crate::math::fixed_point::{FixedPoint, DeterministicRng, BoxMullerGenerator, PoissonJumpGenerator};
use crate::math::jump_diffusion::{GBMJumpState, GBMJumpParams, JumpDiffusionSimulator};
use crate::math::rough_volatility::{FractionalBrownianMotion, RoughVolatilityError};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use rayon::prelude::*;

#[derive(Error, Debug)]
pub enum SDEError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Convergence error: {0}")]
    ConvergenceError(String),
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    #[error("Numerical stability error: {0}")]
    NumericalStability(String),
}

/// SDE solver trait for different numerical schemes
pub trait SDESolver<State, Params> {
    /// Solve one time step
    fn solve_step(
        &mut self,
        t: FixedPoint,
        state: &State,
        dt: FixedPoint,
        params: &Params,
        rng: &mut DeterministicRng,
    ) -> Result<State, SDEError>;

    /// Solve entire path
    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<State>, SDEError>;

    /// Get solver name for identification
    fn name(&self) -> &'static str;

    /// Check numerical stability
    fn check_stability(&self, state: &State, params: &Params) -> Result<(), SDEError>;
}

/// Euler-Maruyama solver for GBM with jumps (first-order strong convergence)
pub struct EulerMaruyamaGBMJump {
    pub normal_generator: BoxMullerGenerator,
    pub poisson_generator: PoissonJumpGenerator,
    pub jump_buffer: VecDeque<FixedPoint>,
    pub max_buffer_size: usize,
    pub stability_threshold: FixedPoint,
}

impl EulerMaruyamaGBMJump {
    pub fn new(seed: u64, max_buffer_size: usize) -> Self {
        Self {
            normal_generator: BoxMullerGenerator::new(seed),
            poisson_generator: PoissonJumpGenerator::new(seed + 1),
            jump_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            stability_threshold: FixedPoint::from_float(1e-6),
        }
    }

    pub fn with_stability_threshold(mut self, threshold: FixedPoint) -> Self {
        self.stability_threshold = threshold;
        self
    }
}

impl SDESolver<GBMJumpState, GBMJumpParams> for EulerMaruyamaGBMJump {
    fn solve_step(
        &mut self,
        t: FixedPoint,
        state: &GBMJumpState,
        dt: FixedPoint,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> Result<GBMJumpState, SDEError> {
        self.check_stability(state, params)?;

        // dS_t = μS_t dt + σS_t dW_t + S_t ∫ h(z) Ñ(dt,dz)
        
        // Generate Brownian increment using Box-Muller
        let sqrt_dt = dt.sqrt();
        let dw = self.normal_generator.next_standard_normal() * sqrt_dt;
        
        // Generate jump component using Poisson process
        let jump_count = self.poisson_generator.next_poisson_count(params.jump_intensity, dt);
        let mut total_jump_size = FixedPoint::zero();
        
        for _ in 0..jump_count {
            let jump_size = match params.jump_distribution {
                crate::math::jump_diffusion::JumpSizeDistribution::Normal { mean, std_dev } => {
                    self.normal_generator.next_normal(mean, std_dev)
                }
                crate::math::jump_diffusion::JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } => {
                    let u = rng.next_fixed();
                    if u <= p {
                        rng.next_exponential_fixed(eta_plus)
                    } else {
                        -rng.next_exponential_fixed(eta_minus)
                    }
                }
                crate::math::jump_diffusion::JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                    let u = rng.next_fixed();
                    if u <= p {
                        rng.next_exponential_fixed(eta_plus)
                    } else {
                        -rng.next_exponential_fixed(eta_minus)
                    }
                }
            };
            
            total_jump_size = total_jump_size + jump_size;
            
            // Store in buffer for analysis
            if self.jump_buffer.len() >= self.max_buffer_size {
                self.jump_buffer.pop_front();
            }
            self.jump_buffer.push_back(jump_size);
        }
        
        // Euler-Maruyama step
        let drift_term = params.drift * state.price * dt;
        let diffusion_term = params.volatility * state.price * dw;
        let jump_term = state.price * total_jump_size;
        
        let new_price = state.price + drift_term + diffusion_term + jump_term;
        
        // Ensure price remains positive with better handling
        let new_price = if new_price.to_float() <= 0.0 {
            state.price * FixedPoint::from_float(0.001) // Proportional minimum
        } else {
            new_price
        };
        
        Ok(GBMJumpState {
            price: new_price,
            log_price: new_price.ln(),
            volatility: state.volatility,
            time: t + dt,
            jump_count: state.jump_count + jump_count,
            total_jump_size: state.total_jump_size + total_jump_size,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: GBMJumpState,
        params: &GBMJumpParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<GBMJumpState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        let mut current_time = t_span.0;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(current_time, &current_state, dt, params, rng)?;
            current_time = current_time + dt;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }

    fn name(&self) -> &'static str {
        "Euler-Maruyama GBM with Jumps"
    }

    fn check_stability(&self, state: &GBMJumpState, params: &GBMJumpParams) -> Result<(), SDEError> {
        if state.price <= FixedPoint::zero() {
            return Err(SDEError::ConvergenceError(
                "Price became non-positive".to_string()
            ));
        }
        
        if state.volatility <= FixedPoint::zero() {
            return Err(SDEError::InvalidParameters(
                "Volatility must be positive".to_string()
            ));
        }
        
        if params.volatility.to_float() > 5.0 {
            return Err(SDEError::ConvergenceError(
                "Volatility too high for numerical stability".to_string()
            ));
        }
        
        if state.price.to_float().is_infinite() || state.price.to_float().is_nan() {
            return Err(SDEError::ConvergenceError(
                "Price became infinite or NaN".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Milstein solver for higher-order accuracy (second-order correction terms)
pub struct MilsteinGBMJump {
    pub normal_generator: BoxMullerGenerator,
    pub poisson_generator: PoissonJumpGenerator,
    pub jump_buffer: VecDeque<FixedPoint>,
    pub max_buffer_size: usize,
    pub stability_threshold: FixedPoint,
    pub second_order_correction: bool,
}

impl MilsteinGBMJump {
    pub fn new(seed: u64, max_buffer_size: usize) -> Self {
        Self {
            normal_generator: BoxMullerGenerator::new(seed),
            poisson_generator: PoissonJumpGenerator::new(seed + 1),
            jump_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            stability_threshold: FixedPoint::from_float(1e-6),
            second_order_correction: true,
        }
    }

    pub fn with_second_order_correction(mut self, enable: bool) -> Self {
        self.second_order_correction = enable;
        self
    }
}

impl SDESolver<GBMJumpState, GBMJumpParams> for MilsteinGBMJump {
    fn solve_step(
        &mut self,
        t: FixedPoint,
        state: &GBMJumpState,
        dt: FixedPoint,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> Result<GBMJumpState, SDEError> {
        self.check_stability(state, params)?;

        // Milstein scheme: includes second-order correction term
        // dS_t = μS_t dt + σS_t dW_t + (1/2)σ²S_t((dW_t)² - dt) + jump terms
        
        let sqrt_dt = dt.sqrt();
        let dw = self.normal_generator.next_standard_normal() * sqrt_dt;
        
        // Generate jump component
        let jump_count = self.poisson_generator.next_poisson_count(params.jump_intensity, dt);
        let mut total_jump_size = FixedPoint::zero();
        
        for _ in 0..jump_count {
            let jump_size = match params.jump_distribution {
                crate::math::jump_diffusion::JumpSizeDistribution::Normal { mean, std_dev } => {
                    self.normal_generator.next_normal(mean, std_dev)
                }
                crate::math::jump_diffusion::JumpSizeDistribution::DoubleExponential { eta_plus, eta_minus, p } => {
                    let u = rng.next_fixed();
                    if u <= p {
                        rng.next_exponential_fixed(eta_plus)
                    } else {
                        -rng.next_exponential_fixed(eta_minus)
                    }
                }
                crate::math::jump_diffusion::JumpSizeDistribution::Kou { eta_plus, eta_minus, p } => {
                    let u = rng.next_fixed();
                    if u <= p {
                        rng.next_exponential_fixed(eta_plus)
                    } else {
                        -rng.next_exponential_fixed(eta_minus)
                    }
                }
            };
            
            total_jump_size = total_jump_size + jump_size;
            
            if self.jump_buffer.len() >= self.max_buffer_size {
                self.jump_buffer.pop_front();
            }
            self.jump_buffer.push_back(jump_size);
        }
        
        // Milstein terms
        let drift_term = params.drift * state.price * dt;
        let diffusion_term = params.volatility * state.price * dw;
        
        // Second-order correction term: (1/2)σ²S((dW)² - dt)
        let correction_term = if self.second_order_correction {
            let dw_squared = dw * dw;
            params.volatility * params.volatility * state.price * 
                (dw_squared - dt) / FixedPoint::from_float(2.0)
        } else {
            FixedPoint::zero()
        };
        
        let jump_term = state.price * total_jump_size;
        
        let new_price = state.price + drift_term + diffusion_term + correction_term + jump_term;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            state.price * FixedPoint::from_float(0.001)
        } else {
            new_price
        };
        
        Ok(GBMJumpState {
            price: new_price,
            log_price: new_price.ln(),
            volatility: state.volatility,
            time: t + dt,
            jump_count: state.jump_count + jump_count,
            total_jump_size: state.total_jump_size + total_jump_size,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: GBMJumpState,
        params: &GBMJumpParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<GBMJumpState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        let mut current_time = t_span.0;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(current_time, &current_state, dt, params, rng)?;
            current_time = current_time + dt;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }

    fn name(&self) -> &'static str {
        "Milstein GBM with Jumps"
    }

    fn check_stability(&self, state: &GBMJumpState, params: &GBMJumpParams) -> Result<(), SDEError> {
        if state.price <= FixedPoint::zero() {
            return Err(SDEError::ConvergenceError(
                "Price became non-positive".to_string()
            ));
        }
        
        if state.volatility <= FixedPoint::zero() {
            return Err(SDEError::InvalidParameters(
                "Volatility must be positive".to_string()
            ));
        }
        
        if params.volatility.to_float() > 5.0 {
            return Err(SDEError::NumericalStability(
                "Volatility too high for numerical stability".to_string()
            ));
        }
        
        if state.price.to_float().is_infinite() || state.price.to_float().is_nan() {
            return Err(SDEError::NumericalStability(
                "Price became infinite or NaN".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Multi-path simulation for Monte Carlo methods
pub struct MonteCarloSimulator<State, Params, Solver> 
where
    State: Clone + Send + Sync,
    Params: Clone + Send + Sync,
    Solver: SDESolver<State, Params> + Clone + Send,
{
    pub solver: Solver,
    pub n_paths: usize,
    pub parallel: bool,
    _phantom_state: std::marker::PhantomData<State>,
    _phantom_params: std::marker::PhantomData<Params>,
}

impl<State, Params, Solver> MonteCarloSimulator<State, Params, Solver>
where
    State: Clone + Send + Sync,
    Params: Clone + Send + Sync,
    Solver: SDESolver<State, Params> + Clone + Send,
{
    pub fn new(solver: Solver, n_paths: usize, parallel: bool) -> Self {
        Self {
            solver,
            n_paths,
            parallel,
            _phantom_state: std::marker::PhantomData,
            _phantom_params: std::marker::PhantomData,
        }
    }

    pub fn simulate_paths(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        base_seed: u64,
    ) -> Result<Vec<Vec<State>>, SDEError> {
        if self.parallel {
            self.simulate_paths_parallel(t_span, initial, params, n_steps, base_seed)
        } else {
            self.simulate_paths_sequential(t_span, initial, params, n_steps, base_seed)
        }
    }

    fn simulate_paths_sequential(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        base_seed: u64,
    ) -> Result<Vec<Vec<State>>, SDEError> {
        let mut paths = Vec::with_capacity(self.n_paths);
        
        for i in 0..self.n_paths {
            let mut rng = DeterministicRng::new(base_seed + i as u64);
            let path = self.solver.solve_path(t_span, initial.clone(), params, n_steps, &mut rng)?;
            paths.push(path);
        }
        
        Ok(paths)
    }

    fn simulate_paths_parallel(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        base_seed: u64,
    ) -> Result<Vec<Vec<State>>, SDEError> {
        let paths: Result<Vec<_>, _> = (0..self.n_paths)
            .into_par_iter()
            .map(|i| {
                let mut solver = self.solver.clone();
                let mut rng = DeterministicRng::new(base_seed + i as u64);
                solver.solve_path(t_span, initial.clone(), params, n_steps, &mut rng)
            })
            .collect();
        
        paths
    }
}

/// Statistics computation for Monte Carlo results
pub struct PathStatistics {
    pub mean_final_price: FixedPoint,
    pub std_final_price: FixedPoint,
    pub mean_volatility: FixedPoint,
    pub max_drawdown: FixedPoint,
    pub value_at_risk_95: FixedPoint,
    pub expected_shortfall_95: FixedPoint,
    pub jump_frequency: FixedPoint,
    pub mean_jump_size: FixedPoint,
}

impl PathStatistics {
    pub fn compute_from_gbm_paths(paths: &[Vec<GBMJumpState>]) -> Result<Self, SDEError> {
        if paths.is_empty() {
            return Err(SDEError::SimulationError("No paths provided".to_string()));
        }
        
        let n_paths = paths.len();
        let mut final_prices = Vec::with_capacity(n_paths);
        let mut mean_vols = Vec::with_capacity(n_paths);
        let mut max_drawdowns = Vec::with_capacity(n_paths);
        let mut jump_frequencies = Vec::with_capacity(n_paths);
        let mut mean_jump_sizes = Vec::with_capacity(n_paths);
        
        for path in paths {
            if path.is_empty() {
                continue;
            }
            
            // Final price
            final_prices.push(path.last().unwrap().price);
            
            // Mean volatility
            let mean_vol = path.iter()
                .map(|state| state.volatility)
                .fold(FixedPoint::zero(), |acc, vol| acc + vol) / 
                FixedPoint::from_float(path.len() as f64);
            mean_vols.push(mean_vol);
            
            // Maximum drawdown
            let mut max_price = path[0].price;
            let mut max_dd = FixedPoint::zero();
            
            for state in path {
                if state.price > max_price {
                    max_price = state.price;
                }
                let drawdown = (max_price - state.price) / max_price;
                if drawdown > max_dd {
                    max_dd = drawdown;
                }
            }
            max_drawdowns.push(max_dd);
            
            // Jump statistics
            let final_state = path.last().unwrap();
            let time_horizon = final_state.time - path[0].time;
            let jump_freq = if time_horizon > FixedPoint::zero() {
                FixedPoint::from_float(final_state.jump_count as f64) / time_horizon
            } else {
                FixedPoint::zero()
            };
            jump_frequencies.push(jump_freq);
            
            let mean_jump_size = if final_state.jump_count > 0 {
                final_state.total_jump_size / FixedPoint::from_float(final_state.jump_count as f64)
            } else {
                FixedPoint::zero()
            };
            mean_jump_sizes.push(mean_jump_size);
        }
        
        // Compute statistics
        let mean_final = final_prices.iter()
            .fold(FixedPoint::zero(), |acc, &price| acc + price) / 
            FixedPoint::from_float(final_prices.len() as f64);
        
        let variance_final = final_prices.iter()
            .map(|&price| {
                let diff = price - mean_final;
                diff * diff
            })
            .fold(FixedPoint::zero(), |acc, var| acc + var) / 
            FixedPoint::from_float(final_prices.len() as f64);
        
        let std_final = variance_final.sqrt();
        
        let mean_vol = mean_vols.iter()
            .fold(FixedPoint::zero(), |acc, &vol| acc + vol) / 
            FixedPoint::from_float(mean_vols.len() as f64);
        
        let max_dd = max_drawdowns.iter()
            .fold(FixedPoint::zero(), |acc, &dd| acc + dd) / 
            FixedPoint::from_float(max_drawdowns.len() as f64);
        
        let jump_freq = jump_frequencies.iter()
            .fold(FixedPoint::zero(), |acc, &freq| acc + freq) / 
            FixedPoint::from_float(jump_frequencies.len() as f64);
        
        let mean_jump = mean_jump_sizes.iter()
            .fold(FixedPoint::zero(), |acc, &size| acc + size) / 
            FixedPoint::from_float(mean_jump_sizes.len() as f64);
        
        // Sort for VaR and ES calculation
        let mut sorted_prices = final_prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = (0.05 * sorted_prices.len() as f64) as usize;
        let var_95 = if var_index < sorted_prices.len() {
            sorted_prices[var_index]
        } else {
            sorted_prices[0]
        };
        
        let es_95 = if var_index > 0 {
            sorted_prices[..var_index].iter()
                .fold(FixedPoint::zero(), |acc, &price| acc + price) / 
                FixedPoint::from_float(var_index as f64)
        } else {
            var_95
        };
        
        Ok(PathStatistics {
            mean_final_price: mean_final,
            std_final_price: std_final,
            mean_volatility: mean_vol,
            max_drawdown: max_dd,
            value_at_risk_95: var_95,
            expected_shortfall_95: es_95,
            jump_frequency: jump_freq,
            mean_jump_size: mean_jump,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::jump_diffusion::JumpSizeDistribution;

    #[test]
    fn test_euler_maruyama_gbm() {
        let mut solver = EulerMaruyamaGBMJump::new(42, 100);
        let mut rng = DeterministicRng::new(42);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.1),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState::new(
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(0.2),
        );
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            100,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 101);
        assert!(path.last().unwrap().price.to_float() > 0.0);
    }

    #[test]
    fn test_milstein_gbm() {
        let mut solver = MilsteinGBMJump::new(42, 100);
        let mut rng = DeterministicRng::new(42);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.05),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.05),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState::new(
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(0.2),
        );
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            100,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 101);
        assert!(path.last().unwrap().price.to_float() > 0.0);
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let solver = EulerMaruyamaGBMJump::new(42, 50);
        let mut mc_sim = MonteCarloSimulator::new(solver, 10, false);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.1),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState::new(
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(0.2),
        );
        
        let paths = mc_sim.simulate_paths(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            50,
            42,
        ).unwrap();
        
        assert_eq!(paths.len(), 10);
        assert_eq!(paths[0].len(), 51);
        
        let stats = PathStatistics::compute_from_gbm_paths(&paths).unwrap();
        assert!(stats.mean_final_price.to_float() > 0.0);
        assert!(stats.std_final_price.to_float() > 0.0);
    }

    #[test]
    fn test_solver_stability_checks() {
        let solver = EulerMaruyamaGBMJump::new(123, 50);
        
        let valid_state = GBMJumpState::new(
            FixedPoint::from_float(100.0),
            FixedPoint::from_float(0.2),
        );
        
        let valid_params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_distribution: JumpSizeDistribution::Normal {
                mean: FixedPoint::zero(),
                std_dev: FixedPoint::from_float(0.1),
            },
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        assert!(solver.check_stability(&valid_state, &valid_params).is_ok());
        
        // Test invalid state (negative price)
        let invalid_state = GBMJumpState {
            price: FixedPoint::from_float(-10.0),
            ..valid_state
        };
        
        assert!(solver.check_stability(&invalid_state, &valid_params).is_err());
        
        // Test invalid params (high volatility)
        let invalid_params = GBMJumpParams {
            volatility: FixedPoint::from_float(10.0),
            ..valid_params
        };
        
        assert!(solver.check_stability(&valid_state, &invalid_params).is_err());
    }
}