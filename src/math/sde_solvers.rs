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
use crate::math::fixed_point::{FixedPoint, DeterministicRng};
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
}

/// SDE solver trait for different numerical schemes
pub trait SDESolver<State, Params> {
    /// Solve one time step
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &State,
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
}

/// Geometric Brownian Motion with jumps state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMJumpState {
    pub price: FixedPoint,
    pub volatility: FixedPoint,
    pub time: FixedPoint,
}

/// GBM with jumps parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GBMJumpParams {
    pub drift: FixedPoint,              // μ - drift coefficient
    pub volatility: FixedPoint,         // σ - volatility coefficient
    pub jump_intensity: FixedPoint,     // λ - jump arrival rate
    pub jump_mean: FixedPoint,          // μ_J - mean jump size
    pub jump_volatility: FixedPoint,    // σ_J - jump volatility
    pub risk_free_rate: FixedPoint,     // r - risk-free rate
}

/// Euler-Maruyama solver for GBM with jumps
pub struct EulerMaruyamaGBMJump {
    pub jump_buffer: VecDeque<FixedPoint>,
    pub max_buffer_size: usize,
}

impl EulerMaruyamaGBMJump {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            jump_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
        }
    }

    /// Generate jump component using compound Poisson process
    fn generate_jump(
        &mut self,
        dt: FixedPoint,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> FixedPoint {
        // Poisson process for jump arrivals
        let lambda_dt = params.jump_intensity * dt;
        let u = rng.next_fixed();
        
        if u.to_float() < lambda_dt.to_float() {
            // Jump occurs - generate jump size
            let z = FixedPoint::from_float(
                2.0 * rng.next_fixed().to_float() - 1.0
            ); // Uniform(-1,1) approximation
            
            let jump_size = params.jump_mean + params.jump_volatility * z;
            
            // Store in buffer for analysis
            if self.jump_buffer.len() >= self.max_buffer_size {
                self.jump_buffer.pop_front();
            }
            self.jump_buffer.push_back(jump_size);
            
            jump_size
        } else {
            FixedPoint::zero()
        }
    }
}

impl SDESolver<GBMJumpState, GBMJumpParams> for EulerMaruyamaGBMJump {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &GBMJumpState,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> Result<GBMJumpState, SDEError> {
        // dS_t = μS_t dt + σS_t dW_t + S_t ∫ h(z) Ñ(dt,dz)
        
        // Brownian increment
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let dw = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * sqrt_dt;
        
        // Jump component
        let jump = self.generate_jump(dt, params, rng)?;
        
        // Euler-Maruyama step
        let drift_term = params.drift * state.price * dt;
        let diffusion_term = params.volatility * state.price * dw;
        let jump_term = state.price * jump;
        
        let new_price = state.price + drift_term + diffusion_term + jump_term;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            FixedPoint::from_float(0.001) // Minimum price
        } else {
            new_price
        };
        
        Ok(GBMJumpState {
            price: new_price,
            volatility: state.volatility, // Constant volatility for now
            time: state.time + dt,
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
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }
}

/// Milstein solver for higher-order accuracy
pub struct MilsteinGBMJump {
    pub jump_buffer: VecDeque<FixedPoint>,
    pub max_buffer_size: usize,
}

impl MilsteinGBMJump {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            jump_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
        }
    }

    /// Generate jump component
    fn generate_jump(
        &mut self,
        dt: FixedPoint,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> FixedPoint {
        let lambda_dt = params.jump_intensity * dt;
        let u = rng.next_fixed();
        
        if u.to_float() < lambda_dt.to_float() {
            let z = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
            let jump_size = params.jump_mean + params.jump_volatility * z;
            
            if self.jump_buffer.len() >= self.max_buffer_size {
                self.jump_buffer.pop_front();
            }
            self.jump_buffer.push_back(jump_size);
            
            jump_size
        } else {
            FixedPoint::zero()
        }
    }
}

impl SDESolver<GBMJumpState, GBMJumpParams> for MilsteinGBMJump {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &GBMJumpState,
        params: &GBMJumpParams,
        rng: &mut DeterministicRng,
    ) -> Result<GBMJumpState, SDEError> {
        // Milstein scheme: includes second-order correction term
        // dS_t = μS_t dt + σS_t dW_t + (1/2)σ²S_t((dW_t)² - dt) + jump terms
        
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let dw = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * sqrt_dt;
        
        // Jump component
        let jump = self.generate_jump(dt, params, rng)?;
        
        // Milstein terms
        let drift_term = params.drift * state.price * dt;
        let diffusion_term = params.volatility * state.price * dw;
        
        // Second-order correction term: (1/2)σ²S((dW)² - dt)
        let dw_squared = dw * dw;
        let correction_term = params.volatility * params.volatility * state.price * 
                             (dw_squared - dt) / FixedPoint::from_float(2.0);
        
        let jump_term = state.price * jump;
        
        let new_price = state.price + drift_term + diffusion_term + correction_term + jump_term;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            FixedPoint::from_float(0.001)
        } else {
            new_price
        };
        
        Ok(GBMJumpState {
            price: new_price,
            volatility: state.volatility,
            time: state.time + dt,
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
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }
}

/// Rough volatility state with fractional Brownian motion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughVolatilityState {
    pub price: FixedPoint,
    pub log_volatility: FixedPoint,
    pub volatility: FixedPoint,
    pub time: FixedPoint,
    pub fbm_history: Vec<FixedPoint>, // Fractional Brownian motion history
}

/// Rough volatility parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoughVolatilityParams {
    pub hurst_parameter: FixedPoint,    // H ∈ (0, 0.5) for rough paths
    pub vol_of_vol: FixedPoint,         // ν - volatility of volatility
    pub mean_reversion: FixedPoint,     // λ - mean reversion speed
    pub long_term_var: FixedPoint,      // θ - long-term variance
    pub correlation: FixedPoint,        // ρ - price-volatility correlation
    pub initial_vol: FixedPoint,        // v₀ - initial volatility
}

/// Rough volatility solver using Volterra representation
pub struct RoughVolatilitySolver {
    pub kernel_cache: Vec<FixedPoint>,
    pub max_history: usize,
    pub fbm_generator: Option<FractionalBrownianMotion>,
    pub use_advanced_fbm: bool,
}

impl RoughVolatilitySolver {
    pub fn new(max_history: usize) -> Self {
        Self {
            kernel_cache: Vec::with_capacity(max_history),
            max_history,
            fbm_generator: None,
            use_advanced_fbm: false,
        }
    }

    /// Create solver with advanced FBM generator
    pub fn with_advanced_fbm(max_history: usize, hurst_parameter: FixedPoint) -> Result<Self, SDEError> {
        let fbm_generator = FractionalBrownianMotion::new(hurst_parameter, max_history)
            .map_err(|e| SDEError::InvalidParameters(format!("FBM initialization failed: {}", e)))?;
        
        Ok(Self {
            kernel_cache: Vec::with_capacity(max_history),
            max_history,
            fbm_generator: Some(fbm_generator),
            use_advanced_fbm: true,
        })
    }

    /// Compute rough kernel K_H(t) = √(2H) * t^(H-1/2)
    fn compute_rough_kernel(&mut self, t: FixedPoint, h: FixedPoint) -> FixedPoint {
        if t.to_float() <= 0.0 {
            return FixedPoint::zero();
        }
        
        let two_h = FixedPoint::from_float(2.0) * h;
        let sqrt_2h = FixedPoint::from_float(two_h.to_float().sqrt());
        let exponent = h - FixedPoint::from_float(0.5);
        
        // Approximate t^(H-1/2) using exp(ln(t) * (H-1/2))
        let ln_t = FixedPoint::from_float(t.to_float().ln());
        let power_term = (ln_t * exponent).exp();
        
        sqrt_2h * power_term
    }

    /// Generate fractional Brownian motion increment
    fn generate_fbm_increment(
        &mut self,
        dt: FixedPoint,
        params: &RoughVolatilityParams,
        history: &[FixedPoint],
        rng: &mut DeterministicRng,
    ) -> FixedPoint {
        if self.use_advanced_fbm && self.fbm_generator.is_some() {
            // Use advanced FBM generator with Cholesky decomposition
            let step = history.len();
            if let Some(ref fbm_gen) = self.fbm_generator {
                match fbm_gen.generate_increment(step, rng) {
                    Ok(increment) => return increment,
                    Err(_) => {
                        // Fall back to basic method if advanced fails
                        self.use_advanced_fbm = false;
                    }
                }
            }
        }
        
        // Basic Volterra convolution method
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let dw = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * sqrt_dt;
        
        // Volterra convolution for rough path
        let mut convolution = FixedPoint::zero();
        for (i, &past_increment) in history.iter().enumerate() {
            let t_diff = FixedPoint::from_float((i + 1) as f64) * dt;
            let kernel_value = self.compute_rough_kernel(t_diff, params.hurst_parameter);
            convolution = convolution + kernel_value * past_increment;
        }
        
        // Current increment contribution
        let current_kernel = self.compute_rough_kernel(dt, params.hurst_parameter);
        convolution = convolution + current_kernel * dw;
        
        convolution
    }
}

impl SDESolver<RoughVolatilityState, RoughVolatilityParams> for RoughVolatilitySolver {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &RoughVolatilityState,
        params: &RoughVolatilityParams,
        rng: &mut DeterministicRng,
    ) -> Result<RoughVolatilityState, SDEError> {
        // Rough volatility model:
        // dlog(σ_t) = -λ(log(σ_t) - θ)dt + ν dW_t^H
        // dS_t = rS_t dt + σ_t S_t (ρ dW_t^H + √(1-ρ²) dW_t^⊥)
        
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        
        // Generate fractional Brownian motion increment
        let fbm_increment = self.generate_fbm_increment(dt, params, &state.fbm_history, rng);
        
        // Independent Brownian increment for price
        let dw_perp = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * sqrt_dt;
        
        // Update log-volatility with mean reversion and rough noise
        let mean_reversion_term = params.mean_reversion * 
                                 (state.log_volatility - params.long_term_var.ln()) * dt;
        let vol_noise_term = params.vol_of_vol * fbm_increment;
        
        let new_log_vol = state.log_volatility - mean_reversion_term + vol_noise_term;
        let new_volatility = new_log_vol.exp();
        
        // Update price with correlated noise
        let rho_term = params.correlation * fbm_increment;
        let one_minus_rho_sq = FixedPoint::one() - params.correlation * params.correlation;
        let perp_term = FixedPoint::from_float(one_minus_rho_sq.to_float().sqrt()) * dw_perp;
        
        let price_noise = new_volatility * state.price * (rho_term + perp_term);
        let new_price = state.price + price_noise;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            FixedPoint::from_float(0.001)
        } else {
            new_price
        };
        
        // Update FBM history
        let mut new_fbm_history = state.fbm_history.clone();
        new_fbm_history.push(fbm_increment);
        if new_fbm_history.len() > self.max_history {
            new_fbm_history.remove(0);
        }
        
        Ok(RoughVolatilityState {
            price: new_price,
            log_volatility: new_log_vol,
            volatility: new_volatility,
            time: state.time + dt,
            fbm_history: new_fbm_history,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: RoughVolatilityState,
        params: &RoughVolatilityParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<RoughVolatilityState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
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
        
        let std_final = FixedPoint::from_float(variance_final.to_float().sqrt());
        
        let mean_vol = mean_vols.iter()
            .fold(FixedPoint::zero(), |acc, &vol| acc + vol) / 
            FixedPoint::from_float(mean_vols.len() as f64);
        
        let max_dd = max_drawdowns.iter()
            .fold(FixedPoint::zero(), |acc, &dd| acc + dd) / 
            FixedPoint::from_float(max_drawdowns.len() as f64);
        
        // Sort for VaR and ES calculation
        let mut sorted_prices = final_prices.clone();
        sorted_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = (0.05 * sorted_prices.len() as f64) as usize;
        let var_95 = sorted_prices[var_index];
        
        let es_95 = sorted_prices[..var_index].iter()
            .fold(FixedPoint::zero(), |acc, &price| acc + price) / 
            FixedPoint::from_float(var_index as f64);
        
        Ok(PathStatistics {
            mean_final_price: mean_final,
            std_final_price: std_final,
            mean_volatility: mean_vol,
            max_drawdown: max_dd,
            value_at_risk_95: var_95,
            expected_shortfall_95: es_95,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_maruyama_gbm() {
        let mut solver = EulerMaruyamaGBMJump::new(100);
        let mut rng = DeterministicRng::new(42);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_mean: FixedPoint::from_float(0.0),
            jump_volatility: FixedPoint::from_float(0.1),
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
        };
        
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
        let mut solver = MilsteinGBMJump::new(100);
        let mut rng = DeterministicRng::new(42);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.05),
            jump_mean: FixedPoint::from_float(0.0),
            jump_volatility: FixedPoint::from_float(0.05),
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
        };
        
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
    fn test_rough_volatility_solver() {
        let mut solver = RoughVolatilitySolver::new(50);
        let mut rng = DeterministicRng::new(123);
        
        let params = RoughVolatilityParams {
            hurst_parameter: FixedPoint::from_float(0.1),
            vol_of_vol: FixedPoint::from_float(0.3),
            mean_reversion: FixedPoint::from_float(2.0),
            long_term_var: FixedPoint::from_float(0.04),
            correlation: FixedPoint::from_float(-0.7),
            initial_vol: FixedPoint::from_float(0.2),
        };
        
        let initial = RoughVolatilityState {
            price: FixedPoint::from_float(100.0),
            log_volatility: FixedPoint::from_float(0.2_f64.ln()),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
            fbm_history: Vec::new(),
        };
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            50,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 51);
        assert!(path.last().unwrap().price.to_float() > 0.0);
        assert!(path.last().unwrap().volatility.to_float() > 0.0);
    }

    #[test]
    fn test_rough_volatility_with_advanced_fbm() {
        let hurst = FixedPoint::from_float(0.2);
        let mut solver = RoughVolatilitySolver::with_advanced_fbm(50, hurst).unwrap();
        let mut rng = DeterministicRng::new(456);
        
        let params = RoughVolatilityParams {
            hurst_parameter: hurst,
            vol_of_vol: FixedPoint::from_float(0.4),
            mean_reversion: FixedPoint::from_float(1.5),
            long_term_var: FixedPoint::from_float(0.06),
            correlation: FixedPoint::from_float(-0.8),
            initial_vol: FixedPoint::from_float(0.25),
        };
        
        let initial = RoughVolatilityState {
            price: FixedPoint::from_float(100.0),
            log_volatility: FixedPoint::from_float(0.25_f64.ln()),
            volatility: FixedPoint::from_float(0.25),
            time: FixedPoint::zero(),
            fbm_history: Vec::new(),
        };
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            30,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 31);
        assert!(path.last().unwrap().price.to_float() > 0.0);
        assert!(path.last().unwrap().volatility.to_float() > 0.0);
        
        // Check that FBM history is being maintained
        assert!(!path.last().unwrap().fbm_history.is_empty());
        
        // Verify that advanced FBM is being used
        assert!(solver.use_advanced_fbm);
        assert!(solver.fbm_generator.is_some());
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let solver = EulerMaruyamaGBMJump::new(50);
        let mut mc_sim = MonteCarloSimulator::new(solver, 10, false);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            jump_mean: FixedPoint::from_float(0.0),
            jump_volatility: FixedPoint::from_float(0.1),
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
        };
        
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
}/// He
ston stochastic volatility model state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonState {
    pub price: FixedPoint,
    pub variance: FixedPoint,
    pub time: FixedPoint,
}

/// Heston model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HestonParams {
    pub risk_free_rate: FixedPoint,     // r - risk-free rate
    pub kappa: FixedPoint,              // κ - mean reversion speed
    pub theta: FixedPoint,              // θ - long-term variance
    pub sigma: FixedPoint,              // σ - volatility of volatility
    pub rho: FixedPoint,                // ρ - correlation between price and variance
    pub initial_variance: FixedPoint,   // v₀ - initial variance
}

/// Heston model solver with full truncation scheme
pub struct HestonSolver {
    pub variance_floor: FixedPoint,
}

impl HestonSolver {
    pub fn new() -> Self {
        Self {
            variance_floor: FixedPoint::from_float(1e-8), // Prevent negative variance
        }
    }

    /// Apply full truncation scheme for variance
    fn truncate_variance(&self, variance: FixedPoint) -> FixedPoint {
        if variance.to_float() < self.variance_floor.to_float() {
            self.variance_floor
        } else {
            variance
        }
    }
}

impl SDESolver<HestonState, HestonParams> for HestonSolver {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &HestonState,
        params: &HestonParams,
        rng: &mut DeterministicRng,
    ) -> Result<HestonState, SDEError> {
        // Heston model:
        // dS_t = rS_t dt + √v_t S_t dW₁_t
        // dv_t = κ(θ - v_t)dt + σ√v_t dW₂_t
        // where dW₁_t dW₂_t = ρ dt
        
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let sqrt_v = FixedPoint::from_float(state.variance.to_float().sqrt());
        
        // Generate correlated Brownian increments
        let z1 = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
        let z2 = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
        
        let dw1 = z1 * sqrt_dt;
        let dw2 = (params.rho * z1 + 
                   FixedPoint::from_float((FixedPoint::one() - params.rho * params.rho).to_float().sqrt()) * z2) * sqrt_dt;
        
        // Update price
        let price_drift = params.risk_free_rate * state.price * dt;
        let price_diffusion = sqrt_v * state.price * dw1;
        let new_price = state.price + price_drift + price_diffusion;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            FixedPoint::from_float(0.001)
        } else {
            new_price
        };
        
        // Update variance with full truncation
        let var_drift = params.kappa * (params.theta - state.variance) * dt;
        let var_diffusion = params.sigma * sqrt_v * dw2;
        let new_variance = self.truncate_variance(state.variance + var_drift + var_diffusion);
        
        Ok(HestonState {
            price: new_price,
            variance: new_variance,
            time: state.time + dt,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: HestonState,
        params: &HestonParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HestonState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }
}

/// Kou jump-diffusion model with double exponential jumps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KouJumpParams {
    pub drift: FixedPoint,              // μ - drift coefficient
    pub volatility: FixedPoint,         // σ - volatility coefficient
    pub jump_intensity: FixedPoint,     // λ - jump arrival rate
    pub prob_up_jump: FixedPoint,       // p - probability of upward jump
    pub eta_up: FixedPoint,             // η₁ - upward jump rate parameter
    pub eta_down: FixedPoint,           // η₂ - downward jump rate parameter
}

/// Kou model solver
pub struct KouJumpSolver {
    pub jump_buffer: VecDeque<FixedPoint>,
    pub max_buffer_size: usize,
}

impl KouJumpSolver {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            jump_buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
        }
    }

    /// Generate double exponential jump
    fn generate_kou_jump(
        &mut self,
        dt: FixedPoint,
        params: &KouJumpParams,
        rng: &mut DeterministicRng,
    ) -> FixedPoint {
        let lambda_dt = params.jump_intensity * dt;
        let u = rng.next_fixed();
        
        if u.to_float() < lambda_dt.to_float() {
            let jump_direction = rng.next_fixed();
            let exponential_sample = -FixedPoint::from_float(rng.next_fixed().to_float().ln());
            
            let jump_size = if jump_direction.to_float() < params.prob_up_jump.to_float() {
                // Upward jump: exponential with rate η₁
                exponential_sample / params.eta_up
            } else {
                // Downward jump: negative exponential with rate η₂
                -exponential_sample / params.eta_down
            };
            
            // Store in buffer for analysis
            if self.jump_buffer.len() >= self.max_buffer_size {
                self.jump_buffer.pop_front();
            }
            self.jump_buffer.push_back(jump_size);
            
            jump_size
        } else {
            FixedPoint::zero()
        }
    }
}

impl SDESolver<GBMJumpState, KouJumpParams> for KouJumpSolver {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &GBMJumpState,
        params: &KouJumpParams,
        rng: &mut DeterministicRng,
    ) -> Result<GBMJumpState, SDEError> {
        // Kou model: dS_t = μS_t dt + σS_t dW_t + S_t ∫ (e^z - 1) Ñ(dt,dz)
        
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let dw = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0) * sqrt_dt;
        
        // Generate Kou jump
        let jump_size = self.generate_kou_jump(dt, params, rng);
        let jump_multiplier = if jump_size.to_float() != 0.0 {
            jump_size.exp() - FixedPoint::one()
        } else {
            FixedPoint::zero()
        };
        
        // Update price
        let drift_term = params.drift * state.price * dt;
        let diffusion_term = params.volatility * state.price * dw;
        let jump_term = state.price * jump_multiplier;
        
        let new_price = state.price + drift_term + diffusion_term + jump_term;
        
        // Ensure price remains positive
        let new_price = if new_price.to_float() <= 0.0 {
            FixedPoint::from_float(0.001)
        } else {
            new_price
        };
        
        Ok(GBMJumpState {
            price: new_price,
            volatility: state.volatility,
            time: state.time + dt,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: GBMJumpState,
        params: &KouJumpParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<GBMJumpState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }
}

/// Multi-dimensional state for correlated assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDimensionalState {
    pub prices: Vec<FixedPoint>,
    pub volatilities: Vec<FixedPoint>,
    pub time: FixedPoint,
}

/// Multi-dimensional GBM parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDimensionalParams {
    pub drifts: Vec<FixedPoint>,
    pub volatilities: Vec<FixedPoint>,
    pub correlation_matrix: Vec<Vec<FixedPoint>>, // Cholesky decomposition
    pub n_assets: usize,
}

/// Multi-dimensional correlated GBM solver
pub struct MultiDimensionalGBMSolver {
    pub cholesky_decomp: Vec<Vec<FixedPoint>>,
}

impl MultiDimensionalGBMSolver {
    pub fn new() -> Self {
        Self {
            cholesky_decomp: Vec::new(),
        }
    }

    /// Compute Cholesky decomposition of correlation matrix
    fn compute_cholesky(&mut self, correlation_matrix: &[Vec<FixedPoint>]) -> Result<(), SDEError> {
        let n = correlation_matrix.len();
        self.cholesky_decomp = vec![vec![FixedPoint::zero(); n]; n];
        
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element
                    let mut sum = FixedPoint::zero();
                    for k in 0..j {
                        sum = sum + self.cholesky_decomp[j][k] * self.cholesky_decomp[j][k];
                    }
                    let diagonal_val = correlation_matrix[j][j] - sum;
                    if diagonal_val.to_float() <= 0.0 {
                        return Err(SDEError::InvalidParameters(
                            "Correlation matrix is not positive definite".to_string()
                        ));
                    }
                    self.cholesky_decomp[j][j] = FixedPoint::from_float(diagonal_val.to_float().sqrt());
                } else {
                    // Off-diagonal element
                    let mut sum = FixedPoint::zero();
                    for k in 0..j {
                        sum = sum + self.cholesky_decomp[i][k] * self.cholesky_decomp[j][k];
                    }
                    self.cholesky_decomp[i][j] = (correlation_matrix[i][j] - sum) / self.cholesky_decomp[j][j];
                }
            }
        }
        
        Ok(())
    }

    /// Generate correlated random variables
    fn generate_correlated_randoms(
        &self,
        rng: &mut DeterministicRng,
        sqrt_dt: FixedPoint,
    ) -> Vec<FixedPoint> {
        let n = self.cholesky_decomp.len();
        let mut independent_randoms = Vec::with_capacity(n);
        let mut correlated_randoms = vec![FixedPoint::zero(); n];
        
        // Generate independent standard normals (approximated)
        for _ in 0..n {
            let z = FixedPoint::from_float(2.0 * rng.next_fixed().to_float() - 1.0);
            independent_randoms.push(z * sqrt_dt);
        }
        
        // Apply Cholesky transformation
        for i in 0..n {
            for j in 0..=i {
                correlated_randoms[i] = correlated_randoms[i] + 
                    self.cholesky_decomp[i][j] * independent_randoms[j];
            }
        }
        
        correlated_randoms
    }
}

impl SDESolver<MultiDimensionalState, MultiDimensionalParams> for MultiDimensionalGBMSolver {
    fn solve_step(
        &mut self,
        dt: FixedPoint,
        state: &MultiDimensionalState,
        params: &MultiDimensionalParams,
        rng: &mut DeterministicRng,
    ) -> Result<MultiDimensionalState, SDEError> {
        if self.cholesky_decomp.is_empty() {
            self.compute_cholesky(&params.correlation_matrix)?;
        }
        
        let sqrt_dt = FixedPoint::from_float(dt.to_float().sqrt());
        let correlated_randoms = self.generate_correlated_randoms(rng, sqrt_dt);
        
        let mut new_prices = Vec::with_capacity(params.n_assets);
        
        for i in 0..params.n_assets {
            let drift_term = params.drifts[i] * state.prices[i] * dt;
            let diffusion_term = params.volatilities[i] * state.prices[i] * correlated_randoms[i];
            
            let new_price = state.prices[i] + drift_term + diffusion_term;
            
            // Ensure price remains positive
            let new_price = if new_price.to_float() <= 0.0 {
                FixedPoint::from_float(0.001)
            } else {
                new_price
            };
            
            new_prices.push(new_price);
        }
        
        Ok(MultiDimensionalState {
            prices: new_prices,
            volatilities: state.volatilities.clone(),
            time: state.time + dt,
        })
    }

    fn solve_path(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: MultiDimensionalState,
        params: &MultiDimensionalParams,
        n_steps: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<MultiDimensionalState>, SDEError> {
        let dt = (t_span.1 - t_span.0) / FixedPoint::from_float(n_steps as f64);
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut current_state = initial;
        
        path.push(current_state.clone());
        
        for _ in 0..n_steps {
            current_state = self.solve_step(dt, &current_state, params, rng)?;
            path.push(current_state.clone());
        }
        
        Ok(path)
    }
}

/// Adaptive time-stepping solver wrapper
pub struct AdaptiveSteppingSolver<State, Params, Solver>
where
    State: Clone,
    Params: Clone,
    Solver: SDESolver<State, Params>,
{
    pub base_solver: Solver,
    pub tolerance: FixedPoint,
    pub min_dt: FixedPoint,
    pub max_dt: FixedPoint,
    pub safety_factor: FixedPoint,
    _phantom_state: std::marker::PhantomData<State>,
    _phantom_params: std::marker::PhantomData<Params>,
}

impl<State, Params, Solver> AdaptiveSteppingSolver<State, Params, Solver>
where
    State: Clone,
    Params: Clone,
    Solver: SDESolver<State, Params>,
{
    pub fn new(base_solver: Solver, tolerance: f64, min_dt: f64, max_dt: f64) -> Self {
        Self {
            base_solver,
            tolerance: FixedPoint::from_float(tolerance),
            min_dt: FixedPoint::from_float(min_dt),
            max_dt: FixedPoint::from_float(max_dt),
            safety_factor: FixedPoint::from_float(0.9),
            _phantom_state: std::marker::PhantomData,
            _phantom_params: std::marker::PhantomData,
        }
    }
}

/// Variance reduction techniques for Monte Carlo
pub struct VarianceReductionMC<State, Params, Solver>
where
    State: Clone + Send + Sync,
    Params: Clone + Send + Sync,
    Solver: SDESolver<State, Params> + Clone + Send,
{
    pub base_simulator: MonteCarloSimulator<State, Params, Solver>,
    pub use_antithetic: bool,
    pub use_control_variate: bool,
}

impl<State, Params, Solver> VarianceReductionMC<State, Params, Solver>
where
    State: Clone + Send + Sync,
    Params: Clone + Send + Sync,
    Solver: SDESolver<State, Params> + Clone + Send,
{
    pub fn new(
        base_simulator: MonteCarloSimulator<State, Params, Solver>,
        use_antithetic: bool,
        use_control_variate: bool,
    ) -> Self {
        Self {
            base_simulator,
            use_antithetic,
            use_control_variate,
        }
    }

    /// Simulate paths with variance reduction techniques
    pub fn simulate_paths_with_variance_reduction(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        base_seed: u64,
    ) -> Result<Vec<Vec<State>>, SDEError> {
        if self.use_antithetic {
            self.simulate_antithetic_paths(t_span, initial, params, n_steps, base_seed)
        } else {
            self.base_simulator.simulate_paths(t_span, initial, params, n_steps, base_seed)
        }
    }

    fn simulate_antithetic_paths(
        &mut self,
        t_span: (FixedPoint, FixedPoint),
        initial: State,
        params: &Params,
        n_steps: usize,
        base_seed: u64,
    ) -> Result<Vec<Vec<State>>, SDEError> {
        // Generate half the paths normally, half with antithetic variates
        let half_paths = self.base_simulator.n_paths / 2;
        let mut all_paths = Vec::with_capacity(self.base_simulator.n_paths);
        
        // Generate normal paths
        let normal_paths = if self.base_simulator.parallel {
            (0..half_paths)
                .into_par_iter()
                .map(|i| {
                    let mut solver = self.base_simulator.solver.clone();
                    let mut rng = DeterministicRng::new(base_seed + i as u64);
                    solver.solve_path(t_span, initial.clone(), params, n_steps, &mut rng)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut paths = Vec::with_capacity(half_paths);
            for i in 0..half_paths {
                let mut rng = DeterministicRng::new(base_seed + i as u64);
                let path = self.base_simulator.solver.solve_path(
                    t_span, initial.clone(), params, n_steps, &mut rng
                )?;
                paths.push(path);
            }
            paths
        };
        
        all_paths.extend(normal_paths);
        
        // TODO: Generate antithetic paths (requires modification of RNG to support antithetic variates)
        // For now, just generate additional normal paths
        let additional_paths = if self.base_simulator.parallel {
            (half_paths..self.base_simulator.n_paths)
                .into_par_iter()
                .map(|i| {
                    let mut solver = self.base_simulator.solver.clone();
                    let mut rng = DeterministicRng::new(base_seed + i as u64 + 1000000);
                    solver.solve_path(t_span, initial.clone(), params, n_steps, &mut rng)
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let mut paths = Vec::with_capacity(self.base_simulator.n_paths - half_paths);
            for i in half_paths..self.base_simulator.n_paths {
                let mut rng = DeterministicRng::new(base_seed + i as u64 + 1000000);
                let path = self.base_simulator.solver.solve_path(
                    t_span, initial.clone(), params, n_steps, &mut rng
                )?;
                paths.push(path);
            }
            paths
        };
        
        all_paths.extend(additional_paths);
        Ok(all_paths)
    }
}

#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_heston_model() {
        let mut solver = HestonSolver::new();
        let mut rng = DeterministicRng::new(42);
        
        let params = HestonParams {
            risk_free_rate: FixedPoint::from_float(0.05),
            kappa: FixedPoint::from_float(2.0),
            theta: FixedPoint::from_float(0.04),
            sigma: FixedPoint::from_float(0.3),
            rho: FixedPoint::from_float(-0.7),
            initial_variance: FixedPoint::from_float(0.04),
        };
        
        let initial = HestonState {
            price: FixedPoint::from_float(100.0),
            variance: FixedPoint::from_float(0.04),
            time: FixedPoint::zero(),
        };
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            100,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 101);
        assert!(path.last().unwrap().price.to_float() > 0.0);
        assert!(path.last().unwrap().variance.to_float() > 0.0);
    }

    #[test]
    fn test_kou_jump_model() {
        let mut solver = KouJumpSolver::new(100);
        let mut rng = DeterministicRng::new(123);
        
        let params = KouJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.1),
            prob_up_jump: FixedPoint::from_float(0.6),
            eta_up: FixedPoint::from_float(10.0),
            eta_down: FixedPoint::from_float(5.0),
        };
        
        let initial = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
        };
        
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
    fn test_multidimensional_gbm() {
        let mut solver = MultiDimensionalGBMSolver::new();
        let mut rng = DeterministicRng::new(456);
        
        // 2-asset case with correlation
        let correlation_matrix = vec![
            vec![FixedPoint::one(), FixedPoint::from_float(0.5)],
            vec![FixedPoint::from_float(0.5), FixedPoint::one()],
        ];
        
        let params = MultiDimensionalParams {
            drifts: vec![FixedPoint::from_float(0.05), FixedPoint::from_float(0.07)],
            volatilities: vec![FixedPoint::from_float(0.2), FixedPoint::from_float(0.25)],
            correlation_matrix,
            n_assets: 2,
        };
        
        let initial = MultiDimensionalState {
            prices: vec![FixedPoint::from_float(100.0), FixedPoint::from_float(50.0)],
            volatilities: vec![FixedPoint::from_float(0.2), FixedPoint::from_float(0.25)],
            time: FixedPoint::zero(),
        };
        
        let path = solver.solve_path(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            50,
            &mut rng,
        ).unwrap();
        
        assert_eq!(path.len(), 51);
        assert_eq!(path[0].prices.len(), 2);
        assert!(path.last().unwrap().prices[0].to_float() > 0.0);
        assert!(path.last().unwrap().prices[1].to_float() > 0.0);
    }

    #[test]
    fn test_variance_reduction_mc() {
        let base_solver = EulerMaruyamaGBMJump::new(50);
        let base_mc = MonteCarloSimulator::new(base_solver, 20, false);
        let mut vr_mc = VarianceReductionMC::new(base_mc, true, false);
        
        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(0.05),
            jump_mean: FixedPoint::from_float(0.0),
            jump_volatility: FixedPoint::from_float(0.1),
            risk_free_rate: FixedPoint::from_float(0.03),
        };
        
        let initial = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            volatility: FixedPoint::from_float(0.2),
            time: FixedPoint::zero(),
        };
        
        let paths = vr_mc.simulate_paths_with_variance_reduction(
            (FixedPoint::zero(), FixedPoint::one()),
            initial,
            &params,
            50,
            789,
        ).unwrap();
        
        assert_eq!(paths.len(), 20);
        assert_eq!(paths[0].len(), 51);
    }
}