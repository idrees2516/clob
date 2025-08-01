//! Optimization algorithms for high-frequency quoting strategies
//! 
//! This module implements various optimization algorithms including gradient descent,
//! Hamilton-Jacobi-Bellman equation solvers, and constrained optimization methods
//! for the optimal quoting problem.

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OptimizationError {
    #[error("Convergence failed: {0}")]
    ConvergenceFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

/// Optimization problem type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    Minimize,
    Maximize,
}

/// Constraint types for optimization problems
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Equality constraint: g(x) = value
    Equality(Box<dyn Fn(&[FixedPoint]) -> FixedPoint + Send + Sync>, FixedPoint),
    /// Inequality constraint: g(x) <= value
    LessEqual(Box<dyn Fn(&[FixedPoint]) -> FixedPoint + Send + Sync>, FixedPoint),
    /// Inequality constraint: g(x) >= value
    GreaterEqual(Box<dyn Fn(&[FixedPoint]) -> FixedPoint + Send + Sync>, FixedPoint),
    /// Box constraint: lower <= x[i] <= upper
    Box(usize, FixedPoint, FixedPoint),
}

/// Result of an optimization procedure
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final solution vector
    pub solution: Vec<FixedPoint>,
    /// Final objective value
    pub objective_value: FixedPoint,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final gradient norm
    pub gradient_norm: FixedPoint,
    /// Constraint violations (if any)
    pub constraint_violations: Vec<FixedPoint>,
}

/// Gradient descent optimizer with momentum and adaptive learning rate
pub struct GradientDescentOptimizer {
    /// Learning rate
    pub learning_rate: FixedPoint,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: FixedPoint,
    /// Momentum coefficient
    pub momentum: FixedPoint,
    /// Adaptive learning rate decay
    pub learning_rate_decay: FixedPoint,
    /// L2 regularization coefficient
    pub l2_regularization: FixedPoint,
}

impl GradientDescentOptimizer {
    /// Create a new gradient descent optimizer
    pub fn new(
        learning_rate: FixedPoint,
        max_iterations: usize,
        tolerance: FixedPoint,
    ) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance,
            momentum: FixedPoint::from_float(0.9),
            learning_rate_decay: FixedPoint::from_float(0.99),
            l2_regularization: FixedPoint::zero(),
        }
    }
    
    /// Set momentum coefficient
    pub fn with_momentum(mut self, momentum: FixedPoint) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Set learning rate decay
    pub fn with_decay(mut self, decay: FixedPoint) -> Self {
        self.learning_rate_decay = decay;
        self
    }
    
    /// Set L2 regularization
    pub fn with_l2_regularization(mut self, l2_reg: FixedPoint) -> Self {
        self.l2_regularization = l2_reg;
        self
    }
    
    /// Optimize the given objective function
    pub fn optimize<F, G>(
        &self,
        objective: F,
        gradient: G,
        initial_point: &[FixedPoint],
        constraints: &[Constraint],
        optimization_type: OptimizationType,
    ) -> OptimizationResult
    where
        F: Fn(&[FixedPoint]) -> FixedPoint,
        G: Fn(&[FixedPoint]) -> Vec<FixedPoint>,
    {
        let mut x = initial_point.to_vec();
        let mut velocity = vec![FixedPoint::zero(); x.len()];
        let mut current_lr = self.learning_rate;
        let mut converged = false;
        let mut final_gradient_norm = FixedPoint::zero();
        
        for iteration in 0..self.max_iterations {
            // Compute gradient
            let mut grad = gradient(&x);
            
            // Add L2 regularization to gradient
            if self.l2_regularization > FixedPoint::zero() {
                for i in 0..grad.len() {
                    grad[i] = grad[i] + self.l2_regularization * x[i];
                }
            }
            
            // Flip gradient for maximization
            if optimization_type == OptimizationType::Maximize {
                for g in &mut grad {
                    *g = -*g;
                }
            }
            
            // Compute gradient norm
            let grad_norm = grad.iter()
                .map(|&g| g * g)
                .fold(FixedPoint::zero(), |acc, g2| acc + g2)
                .sqrt();
            
            final_gradient_norm = grad_norm;
            
            // Check convergence
            if grad_norm < self.tolerance {
                converged = true;
                break;
            }
            
            // Update velocity with momentum
            for i in 0..velocity.len() {
                velocity[i] = self.momentum * velocity[i] - current_lr * grad[i];
            }
            
            // Update parameters
            for i in 0..x.len() {
                x[i] = x[i] + velocity[i];
            }
            
            // Apply constraints
            self.apply_constraints(&mut x, constraints);
            
            // Decay learning rate
            current_lr = current_lr * self.learning_rate_decay;
        }
        
        // Compute final objective value
        let mut final_objective = objective(&x);
        if optimization_type == OptimizationType::Maximize {
            final_objective = -final_objective;
        }
        
        // Check constraint violations
        let constraint_violations = self.check_constraint_violations(&x, constraints);
        
        OptimizationResult {
            solution: x,
            objective_value: final_objective,
            iterations: if converged { 
                self.max_iterations - (self.max_iterations - 1) 
            } else { 
                self.max_iterations 
            },
            converged,
            gradient_norm: final_gradient_norm,
            constraint_violations,
        }
    }
    
    /// Apply constraints to the current solution
    fn apply_constraints(&self, x: &mut [FixedPoint], constraints: &[Constraint]) {
        for constraint in constraints {
            match constraint {
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        x[*index] = x[*index].max(*lower).min(*upper);
                    }
                }
                // For other constraints, we use penalty methods (simplified)
                _ => {}
            }
        }
    }
    
    /// Check constraint violations
    fn check_constraint_violations(&self, x: &[FixedPoint], constraints: &[Constraint]) -> Vec<FixedPoint> {
        let mut violations = Vec::new();
        
        for constraint in constraints {
            let violation = match constraint {
                Constraint::Equality(f, target) => {
                    (f(x) - *target).abs()
                }
                Constraint::LessEqual(f, upper) => {
                    (f(x) - *upper).max(FixedPoint::zero())
                }
                Constraint::GreaterEqual(f, lower) => {
                    (*lower - f(x)).max(FixedPoint::zero())
                }
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        let val = x[*index];
                        if val < *lower {
                            *lower - val
                        } else if val > *upper {
                            val - *upper
                        } else {
                            FixedPoint::zero()
                        }
                    } else {
                        FixedPoint::zero()
                    }
                }
            };
            
            violations.push(violation);
        }
        
        violations
    }
}

/// State grid for finite difference methods
#[derive(Debug, Clone)]
pub struct StateGrid {
    /// Inventory grid points: q ∈ [-Q_max, Q_max]
    pub inventory_grid: Vec<FixedPoint>,
    /// Time grid points: t ∈ [0, T]
    pub time_grid: Vec<FixedPoint>,
    /// Price grid points: S ∈ [S_min, S_max]
    pub price_grid: Vec<FixedPoint>,
}

impl StateGrid {
    /// Create a new state grid
    pub fn new(
        inventory_range: (FixedPoint, FixedPoint),
        inventory_steps: usize,
        time_range: (FixedPoint, FixedPoint),
        time_steps: usize,
        price_range: (FixedPoint, FixedPoint),
        price_steps: usize,
    ) -> Self {
        let inventory_grid = Self::linspace(inventory_range.0, inventory_range.1, inventory_steps);
        let time_grid = Self::linspace(time_range.0, time_range.1, time_steps);
        let price_grid = Self::linspace(price_range.0, price_range.1, price_steps);
        
        Self {
            inventory_grid,
            time_grid,
            price_grid,
        }
    }
    
    /// Create linearly spaced points
    fn linspace(start: FixedPoint, end: FixedPoint, num: usize) -> Vec<FixedPoint> {
        if num == 0 {
            return Vec::new();
        }
        if num == 1 {
            return vec![start];
        }
        
        let step = (end - start) / FixedPoint::from_int((num - 1) as i32);
        (0..num)
            .map(|i| start + step * FixedPoint::from_int(i as i32))
            .collect()
    }
    
    /// Get grid dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.inventory_grid.len(), self.time_grid.len(), self.price_grid.len())
    }
    
    /// Convert 3D indices to flat index
    pub fn to_flat_index(&self, i: usize, j: usize, k: usize) -> usize {
        let (ni, nj, _) = self.dimensions();
        i * nj * self.price_grid.len() + j * self.price_grid.len() + k
    }
    
    /// Convert flat index to 3D indices
    pub fn from_flat_index(&self, flat: usize) -> (usize, usize, usize) {
        let (ni, nj, nk) = self.dimensions();
        let i = flat / (nj * nk);
        let j = (flat % (nj * nk)) / nk;
        let k = flat % nk;
        (i, j, k)
    }
}

/// Boundary conditions for PDE solving
#[derive(Debug, Clone)]
pub enum BoundaryCondition {
    /// Dirichlet: u = value
    Dirichlet(FixedPoint),
    /// Neumann: du/dn = value
    Neumann(FixedPoint),
    /// Robin: a*u + b*du/dn = value
    Robin(FixedPoint, FixedPoint, FixedPoint),
}

/// Boundary conditions specification
#[derive(Debug, Clone)]
pub struct BoundaryConditions {
    /// Boundary conditions for each face of the domain
    pub conditions: HashMap<String, BoundaryCondition>,
}

impl BoundaryConditions {
    pub fn new() -> Self {
        Self {
            conditions: HashMap::new(),
        }
    }
    
    pub fn add_condition(&mut self, face: String, condition: BoundaryCondition) {
        self.conditions.insert(face, condition);
    }
}

impl Default for BoundaryConditions {
    fn default() -> Self {
        Self::new()
    }
}

/// Numerical scheme for PDE solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericalScheme {
    /// Explicit Euler scheme
    ExplicitEuler,
    /// Implicit Euler scheme
    ImplicitEuler,
    /// Crank-Nicolson scheme
    CrankNicolson,
    /// Upwind scheme for convection
    Upwind,
}

/// Value function representation
#[derive(Debug, Clone)]
pub struct ValueFunction {
    /// Grid on which the value function is defined
    pub grid: StateGrid,
    /// Value function values at grid points
    pub values: Vec<FixedPoint>,
    /// Gradient of value function (if computed)
    pub gradient: Option<Vec<Vec<FixedPoint>>>,
}

impl ValueFunction {
    /// Create a new value function
    pub fn new(grid: StateGrid) -> Self {
        let total_points = grid.inventory_grid.len() * grid.time_grid.len() * grid.price_grid.len();
        Self {
            grid,
            values: vec![FixedPoint::zero(); total_points],
            gradient: None,
        }
    }
    
    /// Get value at specific grid point
    pub fn get_value(&self, i: usize, j: usize, k: usize) -> FixedPoint {
        let flat_index = self.grid.to_flat_index(i, j, k);
        self.values[flat_index]
    }
    
    /// Set value at specific grid point
    pub fn set_value(&mut self, i: usize, j: usize, k: usize, value: FixedPoint) {
        let flat_index = self.grid.to_flat_index(i, j, k);
        self.values[flat_index] = value;
    }
    
    /// Interpolate value at arbitrary point
    pub fn interpolate(&self, inventory: FixedPoint, time: FixedPoint, price: FixedPoint) -> FixedPoint {
        // Trilinear interpolation (simplified implementation)
        // In practice, would use more sophisticated interpolation
        
        // Find surrounding grid points
        let i = self.find_grid_index(&self.grid.inventory_grid, inventory);
        let j = self.find_grid_index(&self.grid.time_grid, time);
        let k = self.find_grid_index(&self.grid.price_grid, price);
        
        // For simplicity, return nearest neighbor
        self.get_value(i, j, k)
    }
    
    /// Find nearest grid index
    fn find_grid_index(&self, grid: &[FixedPoint], value: FixedPoint) -> usize {
        if grid.is_empty() {
            return 0;
        }
        
        let mut best_index = 0;
        let mut best_distance = (grid[0] - value).abs();
        
        for (i, &grid_value) in grid.iter().enumerate() {
            let distance = (grid_value - value).abs();
            if distance < best_distance {
                best_distance = distance;
                best_index = i;
            }
        }
        
        best_index
    }
}

/// Optimal controls extracted from value function
#[derive(Debug, Clone)]
pub struct OptimalControls {
    /// Grid on which controls are defined
    pub grid: StateGrid,
    /// Optimal bid spread δ*ᵇ(t,q,S)
    pub optimal_bid_spread: Vec<FixedPoint>,
    /// Optimal ask spread δ*ᵃ(t,q,S)
    pub optimal_ask_spread: Vec<FixedPoint>,
    /// Optimal quote intensity
    pub optimal_intensity: Vec<FixedPoint>,
}

impl OptimalControls {
    /// Create new optimal controls
    pub fn new(grid: StateGrid) -> Self {
        let total_points = grid.inventory_grid.len() * grid.time_grid.len() * grid.price_grid.len();
        Self {
            grid,
            optimal_bid_spread: vec![FixedPoint::zero(); total_points],
            optimal_ask_spread: vec![FixedPoint::zero(); total_points],
            optimal_intensity: vec![FixedPoint::zero(); total_points],
        }
    }
    
    /// Get optimal controls at specific state
    pub fn get_controls(&self, inventory: FixedPoint, time: FixedPoint, price: FixedPoint) -> (FixedPoint, FixedPoint, FixedPoint) {
        // Find nearest grid point (simplified)
        let i = self.find_grid_index(&self.grid.inventory_grid, inventory);
        let j = self.find_grid_index(&self.grid.time_grid, time);
        let k = self.find_grid_index(&self.grid.price_grid, price);
        
        let flat_index = self.grid.to_flat_index(i, j, k);
        
        (
            self.optimal_bid_spread[flat_index],
            self.optimal_ask_spread[flat_index],
            self.optimal_intensity[flat_index],
        )
    }
    
    /// Find nearest grid index
    fn find_grid_index(&self, grid: &[FixedPoint], value: FixedPoint) -> usize {
        if grid.is_empty() {
            return 0;
        }
        
        let mut best_index = 0;
        let mut best_distance = (grid[0] - value).abs();
        
        for (i, &grid_value) in grid.iter().enumerate() {
            let distance = (grid_value - value).abs();
            if distance < best_distance {
                best_distance = distance;
                best_index = i;
            }
        }
        
        best_index
    }
}

/// Model parameters for the HJB equation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    /// Drift coefficient μ
    pub drift_coefficient: FixedPoint,
    /// Volatility coefficient σ
    pub volatility_coefficient: FixedPoint,
    /// Inventory penalty γ
    pub inventory_penalty: FixedPoint,
    /// Adverse selection cost κ
    pub adverse_selection_cost: FixedPoint,
    /// Market impact coefficient Λ
    pub market_impact_coefficient: FixedPoint,
    /// Risk aversion parameter
    pub risk_aversion: FixedPoint,
    /// Terminal time T
    pub terminal_time: FixedPoint,
    /// Maximum inventory Q_max
    pub max_inventory: FixedPoint,
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self {
            drift_coefficient: FixedPoint::from_float(0.05),      // 5% annual drift
            volatility_coefficient: FixedPoint::from_float(0.2),  // 20% annual volatility
            inventory_penalty: FixedPoint::from_float(0.01),      // Inventory penalty coefficient
            adverse_selection_cost: FixedPoint::from_float(0.001), // Adverse selection cost
            market_impact_coefficient: FixedPoint::from_float(0.1), // Market impact coefficient
            risk_aversion: FixedPoint::from_float(1.0),           // Risk aversion parameter
            terminal_time: FixedPoint::from_float(1.0),           // 1 day terminal time
            max_inventory: FixedPoint::from_float(1000.0),        // Maximum inventory limit
        }
    }
}

/// Hamilton-Jacobi-Bellman equation solver
pub struct HJBSolver {
    /// State grid for discretization
    pub grid: StateGrid,
    /// Boundary conditions
    pub boundary_conditions: BoundaryConditions,
    /// Numerical scheme
    pub numerical_scheme: NumericalScheme,
    /// Time step size
    pub dt: FixedPoint,
    /// Convergence tolerance
    pub tolerance: FixedPoint,
    /// Maximum iterations
    pub max_iterations: usize,
}

impl HJBSolver {
    /// Create a new HJB solver
    pub fn new(
        grid: StateGrid,
        boundary_conditions: BoundaryConditions,
        numerical_scheme: NumericalScheme,
        dt: FixedPoint,
    ) -> Self {
        Self {
            grid,
            boundary_conditions,
            numerical_scheme,
            dt,
            tolerance: FixedPoint::from_float(1e-6),
            max_iterations: 10000,
        }
    }
    
    /// Solve the HJB equation to get the value function
    pub fn solve_value_function(&self, params: &ModelParameters) -> Result<ValueFunction, OptimizationError> {
        let mut value_function = ValueFunction::new(self.grid.clone());
        
        // Initialize terminal condition
        self.apply_terminal_condition(&mut value_function, params)?;
        
        // Time stepping (backward in time)
        let time_steps = self.grid.time_grid.len();
        
        for t_idx in (0..time_steps-1).rev() {
            // Solve one time step
            self.solve_time_step(&mut value_function, t_idx, params)?;
            
            // Apply boundary conditions
            self.apply_boundary_conditions(&mut value_function, t_idx)?;
        }
        
        Ok(value_function)
    }
    
    /// Apply terminal condition V(T, q, S) = -γq²/2
    fn apply_terminal_condition(&self, value_function: &mut ValueFunction, params: &ModelParameters) -> Result<(), OptimizationError> {
        let t_final = self.grid.time_grid.len() - 1;
        
        for (i, &inventory) in self.grid.inventory_grid.iter().enumerate() {
            for (k, &_price) in self.grid.price_grid.iter().enumerate() {
                // Terminal penalty: -γq²/2
                let terminal_value = -params.inventory_penalty * inventory * inventory / FixedPoint::from_float(2.0);
                value_function.set_value(i, t_final, k, terminal_value);
            }
        }
        
        Ok(())
    }
    
    /// Solve one time step of the HJB equation
    fn solve_time_step(&self, value_function: &mut ValueFunction, t_idx: usize, params: &ModelParameters) -> Result<(), OptimizationError> {
        let (ni, _, nk) = self.grid.dimensions();
        
        for i in 1..ni-1 {  // Skip boundary points
            for k in 1..nk-1 {  // Skip boundary points
                let inventory = self.grid.inventory_grid[i];
                let price = self.grid.price_grid[k];
                
                // Compute optimal controls at this point
                let (optimal_bid_spread, optimal_ask_spread) = self.compute_optimal_controls(
                    value_function, i, t_idx, k, inventory, price, params
                )?;
                
                // Compute HJB operator
                let hjb_value = self.compute_hjb_operator(
                    value_function, i, t_idx, k, inventory, price, 
                    optimal_bid_spread, optimal_ask_spread, params
                )?;
                
                // Update value function
                let current_value = value_function.get_value(i, t_idx + 1, k);
                let new_value = current_value + self.dt * hjb_value;
                value_function.set_value(i, t_idx, k, new_value);
            }
        }
        
        Ok(())
    }
    
    /// Compute optimal controls by maximizing the HJB operator
    fn compute_optimal_controls(
        &self,
        value_function: &ValueFunction,
        i: usize,
        j: usize,
        k: usize,
        inventory: FixedPoint,
        price: FixedPoint,
        params: &ModelParameters,
    ) -> Result<(FixedPoint, FixedPoint), OptimizationError> {
        // Simplified optimal control computation
        // In practice, would solve: max_{δᵇ,δᵃ} [utility improvement]
        
        // Compute value function derivatives (finite differences)
        let dv_dq = self.compute_inventory_derivative(value_function, i, j, k)?;
        let dv_ds = self.compute_price_derivative(value_function, i, j, k)?;
        
        // Optimal bid spread (simplified)
        let bid_spread = (params.adverse_selection_cost + 
                         params.inventory_penalty * inventory.abs()) / 
                        (FixedPoint::one() + dv_dq.abs());
        
        // Optimal ask spread (simplified)
        let ask_spread = (params.adverse_selection_cost + 
                         params.inventory_penalty * inventory.abs()) / 
                        (FixedPoint::one() + dv_dq.abs());
        
        Ok((bid_spread.max(FixedPoint::from_float(0.001)), 
            ask_spread.max(FixedPoint::from_float(0.001))))
    }
    
    /// Compute HJB operator
    fn compute_hjb_operator(
        &self,
        value_function: &ValueFunction,
        i: usize,
        j: usize,
        k: usize,
        inventory: FixedPoint,
        price: FixedPoint,
        bid_spread: FixedPoint,
        ask_spread: FixedPoint,
        params: &ModelParameters,
    ) -> Result<FixedPoint, OptimizationError> {
        // HJB operator: ∂V/∂t + sup_{δᵇ,δᵃ} L^{δᵇ,δᵃ} V = 0
        
        // Compute derivatives
        let dv_dt = self.compute_time_derivative(value_function, i, j, k)?;
        let dv_dq = self.compute_inventory_derivative(value_function, i, j, k)?;
        let dv_ds = self.compute_price_derivative(value_function, i, j, k)?;
        let d2v_ds2 = self.compute_second_price_derivative(value_function, i, j, k)?;
        
        // Drift term: μS ∂V/∂S
        let drift_term = params.drift_coefficient * price * dv_ds;
        
        // Diffusion term: (1/2)σ²S² ∂²V/∂S²
        let diffusion_term = params.volatility_coefficient * params.volatility_coefficient * 
                           price * price * d2v_ds2 / FixedPoint::from_float(2.0);
        
        // Market making terms (simplified)
        let bid_intensity = self.compute_arrival_intensity(bid_spread, params);
        let ask_intensity = self.compute_arrival_intensity(ask_spread, params);
        
        let bid_utility = bid_intensity * (bid_spread + dv_dq);
        let ask_utility = ask_intensity * (ask_spread - dv_dq);
        
        // Inventory penalty
        let penalty_term = -params.inventory_penalty * inventory * inventory / FixedPoint::from_float(2.0);
        
        let hjb_operator = drift_term + diffusion_term + bid_utility + ask_utility + penalty_term;
        
        Ok(-hjb_operator) // Negative because we're solving backward in time
    }
    
    /// Compute arrival intensity as a function of spread
    fn compute_arrival_intensity(&self, spread: FixedPoint, params: &ModelParameters) -> FixedPoint {
        // Exponential intensity function: λ(δ) = A * exp(-k * δ)
        let base_intensity = FixedPoint::from_float(1.0);
        let decay_rate = params.adverse_selection_cost;
        
        base_intensity * (-decay_rate * spread).exp()
    }
    
    /// Compute time derivative using finite differences
    fn compute_time_derivative(&self, value_function: &ValueFunction, i: usize, j: usize, k: usize) -> Result<FixedPoint, OptimizationError> {
        if j == 0 || j >= self.grid.time_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let v_plus = value_function.get_value(i, j + 1, k);
        let v_minus = value_function.get_value(i, j - 1, k);
        let dt = self.grid.time_grid[j + 1] - self.grid.time_grid[j - 1];
        
        Ok((v_plus - v_minus) / dt)
    }
    
    /// Compute inventory derivative using finite differences
    fn compute_inventory_derivative(&self, value_function: &ValueFunction, i: usize, j: usize, k: usize) -> Result<FixedPoint, OptimizationError> {
        if i == 0 || i >= self.grid.inventory_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let v_plus = value_function.get_value(i + 1, j, k);
        let v_minus = value_function.get_value(i - 1, j, k);
        let dq = self.grid.inventory_grid[i + 1] - self.grid.inventory_grid[i - 1];
        
        Ok((v_plus - v_minus) / dq)
    }
    
    /// Compute price derivative using finite differences
    fn compute_price_derivative(&self, value_function: &ValueFunction, i: usize, j: usize, k: usize) -> Result<FixedPoint, OptimizationError> {
        if k == 0 || k >= self.grid.price_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let v_plus = value_function.get_value(i, j, k + 1);
        let v_minus = value_function.get_value(i, j, k - 1);
        let ds = self.grid.price_grid[k + 1] - self.grid.price_grid[k - 1];
        
        Ok((v_plus - v_minus) / ds)
    }
    
    /// Compute second price derivative using finite differences
    fn compute_second_price_derivative(&self, value_function: &ValueFunction, i: usize, j: usize, k: usize) -> Result<FixedPoint, OptimizationError> {
        if k == 0 || k >= self.grid.price_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let v_plus = value_function.get_value(i, j, k + 1);
        let v_center = value_function.get_value(i, j, k);
        let v_minus = value_function.get_value(i, j, k - 1);
        
        let ds = self.grid.price_grid[k + 1] - self.grid.price_grid[k];
        
        Ok((v_plus - FixedPoint::from_float(2.0) * v_center + v_minus) / (ds * ds))
    }
    
    /// Apply boundary conditions
    fn apply_boundary_conditions(&self, value_function: &mut ValueFunction, t_idx: usize) -> Result<(), OptimizationError> {
        // Apply inventory boundary conditions (simplified)
        let (ni, _, nk) = self.grid.dimensions();
        
        for k in 0..nk {
            // Lower inventory boundary
            let boundary_value = value_function.get_value(1, t_idx, k);
            value_function.set_value(0, t_idx, k, boundary_value);
            
            // Upper inventory boundary
            let boundary_value = value_function.get_value(ni - 2, t_idx, k);
            value_function.set_value(ni - 1, t_idx, k, boundary_value);
        }
        
        // Apply price boundary conditions (simplified)
        for i in 0..ni {
            // Lower price boundary
            let boundary_value = value_function.get_value(i, t_idx, 1);
            value_function.set_value(i, t_idx, 0, boundary_value);
            
            // Upper price boundary
            let boundary_value = value_function.get_value(i, t_idx, nk - 2);
            value_function.set_value(i, t_idx, nk - 1, boundary_value);
        }
        
        Ok(())
    }
    
    /// Extract optimal controls from the value function
    pub fn compute_optimal_controls(&self, value_function: &ValueFunction, params: &ModelParameters) -> Result<OptimalControls, OptimizationError> {
        let mut controls = OptimalControls::new(self.grid.clone());
        let (ni, nj, nk) = self.grid.dimensions();
        
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let inventory = self.grid.inventory_grid[i];
                    let time = self.grid.time_grid[j];
                    let price = self.grid.price_grid[k];
                    
                    // Compute optimal controls at this grid point
                    let (bid_spread, ask_spread) = self.compute_optimal_controls(
                        value_function, i, j, k, inventory, price, params
                    )?;
                    
                    // Compute optimal intensity
                    let bid_intensity = self.compute_arrival_intensity(bid_spread, params);
                    let ask_intensity = self.compute_arrival_intensity(ask_spread, params);
                    let total_intensity = bid_intensity + ask_intensity;
                    
                    let flat_index = self.grid.to_flat_index(i, j, k);
                    controls.optimal_bid_spread[flat_index] = bid_spread;
                    controls.optimal_ask_spread[flat_index] = ask_spread;
                    controls.optimal_intensity[flat_index] = total_intensity;
                }
            }
        }
        
        Ok(controls)
    }
}

/// Advanced finite difference schemes for PDE solving
pub struct FiniteDifferenceSchemes;

impl FiniteDifferenceSchemes {
    /// Upwind scheme for convection terms
    pub fn upwind_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
        velocity: FixedPoint,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        if velocity > FixedPoint::zero() {
            // Forward difference
            let dx = grid[index] - grid[index - 1];
            (values[index] - values[index - 1]) / dx
        } else {
            // Backward difference
            let dx = grid[index + 1] - grid[index];
            (values[index + 1] - values[index]) / dx
        }
    }
    
    /// Central difference scheme
    pub fn central_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        let dx = grid[index + 1] - grid[index - 1];
        (values[index + 1] - values[index - 1]) / dx
    }
    
    /// Second derivative using central differences
    pub fn second_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        let dx = grid[index + 1] - grid[index];
        (values[index + 1] - FixedPoint::from_float(2.0) * values[index] + values[index - 1]) / (dx * dx)
    }
    
    /// Crank-Nicolson time stepping
    pub fn crank_nicolson_step(
        old_values: &[FixedPoint],
        new_values: &mut [FixedPoint],
        dt: FixedPoint,
        spatial_operator: impl Fn(&[FixedPoint], usize) -> FixedPoint,
    ) {
        for i in 1..old_values.len() - 1 {
            let old_spatial = spatial_operator(old_values, i);
            let new_spatial = spatial_operator(new_values, i);
            
            new_values[i] = old_values[i] + dt * (old_spatial + new_spatial) / FixedPoint::from_float(2.0);
        }
    }
    
    /// Implicit Euler time stepping
    pub fn implicit_euler_step(
        old_values: &[FixedPoint],
        new_values: &mut [FixedPoint],
        dt: FixedPoint,
        spatial_operator: impl Fn(&[FixedPoint], usize) -> FixedPoint,
    ) {
        // Simplified implicit step (would need matrix solver in practice)
        for i in 1..old_values.len() - 1 {
            let spatial = spatial_operator(old_values, i);
            new_values[i] = old_values[i] + dt * spatial;
        }
    }
}

/// Value function optimization using policy iteration
pub struct ValueFunctionOptimizer {
    pub hjb_solver: HJBSolver,
    pub max_policy_iterations: usize,
    pub policy_tolerance: FixedPoint,
}

impl ValueFunctionOptimizer {
    /// Create a new value function optimizer
    pub fn new(hjb_solver: HJBSolver) -> Self {
        Self {
            hjb_solver,
            max_policy_iterations: 100,
            policy_tolerance: FixedPoint::from_float(1e-6),
        }
    }
    
    /// Optimize value function using policy iteration
    pub fn optimize_value_function(&self, params: &ModelParameters) -> Result<(ValueFunction, OptimalControls), OptimizationError> {
        let mut value_function = ValueFunction::new(self.hjb_solver.grid.clone());
        let mut controls = OptimalControls::new(self.hjb_solver.grid.clone());
        
        // Initialize with terminal condition
        self.hjb_solver.apply_terminal_condition(&mut value_function, params)?;
        
        // Policy iteration loop
        for iteration in 0..self.max_policy_iterations {
            let old_controls = controls.clone();
            
            // Policy evaluation: solve for value function given current policy
            value_function = self.policy_evaluation(&controls, params)?;
            
            // Policy improvement: update controls based on new value function
            controls = self.hjb_solver.compute_optimal_controls(&value_function, params)?;
            
            // Check convergence
            let policy_change = self.compute_policy_change(&old_controls, &controls);
            if policy_change < self.policy_tolerance {
                break;
            }
        }
        
        Ok((value_function, controls))
    }
    
    /// Policy evaluation step
    fn policy_evaluation(&self, controls: &OptimalControls, params: &ModelParameters) -> Result<ValueFunction, OptimizationError> {
        let mut value_function = ValueFunction::new(self.hjb_solver.grid.clone());
        
        // Apply terminal condition
        self.hjb_solver.apply_terminal_condition(&mut value_function, params)?;
        
        // Backward time stepping with fixed policy
        let time_steps = self.hjb_solver.grid.time_grid.len();
        
        for t_idx in (0..time_steps-1).rev() {
            self.policy_evaluation_step(&mut value_function, controls, t_idx, params)?;
            self.hjb_solver.apply_boundary_conditions(&mut value_function, t_idx)?;
        }
        
        Ok(value_function)
    }
    
    /// Single policy evaluation step
    fn policy_evaluation_step(
        &self,
        value_function: &mut ValueFunction,
        controls: &OptimalControls,
        t_idx: usize,
        params: &ModelParameters,
    ) -> Result<(), OptimizationError> {
        let (ni, _, nk) = self.hjb_solver.grid.dimensions();
        
        for i in 1..ni-1 {
            for k in 1..nk-1 {
                let inventory = self.hjb_solver.grid.inventory_grid[i];
                let price = self.hjb_solver.grid.price_grid[k];
                
                // Get fixed controls for this state
                let flat_index = self.hjb_solver.grid.to_flat_index(i, t_idx, k);
                let bid_spread = controls.optimal_bid_spread[flat_index];
                let ask_spread = controls.optimal_ask_spread[flat_index];
                
                // Compute HJB operator with fixed controls
                let hjb_value = self.hjb_solver.compute_hjb_operator(
                    value_function, i, t_idx, k, inventory, price,
                    bid_spread, ask_spread, params
                )?;
                
                // Update value function
                let current_value = value_function.get_value(i, t_idx + 1, k);
                let new_value = current_value + self.hjb_solver.dt * hjb_value;
                value_function.set_value(i, t_idx, k, new_value);
            }
        }
        
        Ok(())
    }
    
    /// Compute change in policy between iterations
    fn compute_policy_change(&self, old_controls: &OptimalControls, new_controls: &OptimalControls) -> FixedPoint {
        let mut max_change = FixedPoint::zero();
        
        for i in 0..old_controls.optimal_bid_spread.len() {
            let bid_change = (old_controls.optimal_bid_spread[i] - new_controls.optimal_bid_spread[i]).abs();
            let ask_change = (old_controls.optimal_ask_spread[i] - new_controls.optimal_ask_spread[i]).abs();
            
            max_change = max_change.max(bid_change).max(ask_change);
        }
        
        max_change
    }
}

/// Constrained optimization solver using penalty methods
pub struct ConstrainedOptimizer {
    pub penalty_coefficient: FixedPoint,
    pub penalty_growth_rate: FixedPoint,
    pub max_penalty_iterations: usize,
    pub inner_optimizer: GradientDescentOptimizer,
}

impl ConstrainedOptimizer {
    /// Create a new constrained optimizer
    pub fn new(inner_optimizer: GradientDescentOptimizer) -> Self {
        Self {
            penalty_coefficient: FixedPoint::from_float(1.0),
            penalty_growth_rate: FixedPoint::from_float(10.0),
            max_penalty_iterations: 10,
            inner_optimizer,
        }
    }
    
    /// Solve constrained optimization problem using penalty methods
    pub fn optimize_constrained<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        initial_point: &[FixedPoint],
        constraints: &[Constraint],
        optimization_type: OptimizationType,
    ) -> OptimizationResult
    where
        F: Fn(&[FixedPoint]) -> FixedPoint + Clone,
        G: Fn(&[FixedPoint]) -> Vec<FixedPoint> + Clone,
    {
        let mut current_point = initial_point.to_vec();
        let mut penalty_coeff = self.penalty_coefficient;
        
        for _penalty_iter in 0..self.max_penalty_iterations {
            // Create penalized objective function
            let penalized_objective = {
                let obj = objective.clone();
                let penalty = penalty_coeff;
                move |x: &[FixedPoint]| -> FixedPoint {
                    let base_obj = obj(x);
                    let penalty_term = Self::compute_penalty_term(x, constraints, penalty);
                    base_obj + penalty_term
                }
            };
            
            // Create penalized gradient
            let penalized_gradient = {
                let grad = gradient.clone();
                let penalty = penalty_coeff;
                move |x: &[FixedPoint]| -> Vec<FixedPoint> {
                    let mut base_grad = grad(x);
                    let penalty_grad = Self::compute_penalty_gradient(x, constraints, penalty);
                    
                    for i in 0..base_grad.len() {
                        base_grad[i] = base_grad[i] + penalty_grad[i];
                    }
                    
                    base_grad
                }
            };
            
            // Solve penalized problem
            let result = self.inner_optimizer.optimize(
                penalized_objective,
                penalized_gradient,
                &current_point,
                &[], // No additional constraints for inner problem
                optimization_type,
            );
            
            current_point = result.solution;
            
            // Check constraint satisfaction
            let constraint_violations = self.inner_optimizer.check_constraint_violations(&current_point, constraints);
            let max_violation = constraint_violations.iter()
                .fold(FixedPoint::zero(), |acc, &v| acc.max(v));
            
            if max_violation < self.inner_optimizer.tolerance {
                return OptimizationResult {
                    solution: current_point,
                    objective_value: objective(&current_point),
                    iterations: result.iterations,
                    converged: true,
                    gradient_norm: result.gradient_norm,
                    constraint_violations,
                };
            }
            
            // Increase penalty coefficient
            penalty_coeff = penalty_coeff * self.penalty_growth_rate;
        }
        
        // Return best solution found
        let constraint_violations = self.inner_optimizer.check_constraint_violations(&current_point, constraints);
        OptimizationResult {
            solution: current_point,
            objective_value: objective(&current_point),
            iterations: self.max_penalty_iterations * self.inner_optimizer.max_iterations,
            converged: false,
            gradient_norm: FixedPoint::zero(),
            constraint_violations,
        }
    }
    
    /// Compute penalty term for constraints
    fn compute_penalty_term(x: &[FixedPoint], constraints: &[Constraint], penalty_coeff: FixedPoint) -> FixedPoint {
        let mut penalty = FixedPoint::zero();
        
        for constraint in constraints {
            let violation = match constraint {
                Constraint::Equality(f, target) => {
                    let diff = f(x) - *target;
                    diff * diff
                }
                Constraint::LessEqual(f, upper) => {
                    let violation = (f(x) - *upper).max(FixedPoint::zero());
                    violation * violation
                }
                Constraint::GreaterEqual(f, lower) => {
                    let violation = (*lower - f(x)).max(FixedPoint::zero());
                    violation * violation
                }
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        let val = x[*index];
                        if val < *lower {
                            let violation = *lower - val;
                            violation * violation
                        } else if val > *upper {
                            let violation = val - *upper;
                            violation * violation
                        } else {
                            FixedPoint::zero()
                        }
                    } else {
                        FixedPoint::zero()
                    }
                }
            };
            
            penalty = penalty + penalty_coeff * violation;
        }
        
        penalty
    }
    
    /// Compute gradient of penalty term
    fn compute_penalty_gradient(x: &[FixedPoint], constraints: &[Constraint], penalty_coeff: FixedPoint) -> Vec<FixedPoint> {
        let mut penalty_grad = vec![FixedPoint::zero(); x.len()];
        
        for constraint in constraints {
            match constraint {
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        let val = x[*index];
                        if val < *lower {
                            penalty_grad[*index] = penalty_grad[*index] - 
                                FixedPoint::from_float(2.0) * penalty_coeff * (*lower - val);
                        } else if val > *upper {
                            penalty_grad[*index] = penalty_grad[*index] + 
                                FixedPoint::from_float(2.0) * penalty_coeff * (val - *upper);
                        }
                    }
                }
                // For function constraints, would need numerical differentiation
                _ => {}
            }
        }
        
        penalty_grad
    }                  let price = self.grid.price_grid[k];
                    
                    let (bid_spread, ask_spread) = self.compute_optimal_controls(
                        value_function, i, j, k, inventory, price, params
                    )?;
                    
                    let flat_index = self.grid.to_flat_index(i, j, k);
                    controls.optimal_bid_spread[flat_index] = bid_spread;
                    controls.optimal_ask_spread[flat_index] = ask_spread;
                    
                    // Compute optimal intensity
                    let intensity = self.compute_arrival_intensity(bid_spread, params) + 
                                   self.compute_arrival_intensity(ask_spread, params);
                    controls.optimal_intensity[flat_index] = intensity;
                }
            }
        }
        
        Ok(controls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_optimizer() {
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
    fn test_state_grid() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-10.0), FixedPoint::from_float(10.0)), 21,
            (FixedPoint::zero(), FixedPoint::one()), 11,
            (FixedPoint::from_float(90.0), FixedPoint::from_float(110.0)), 21,
        );
        
        assert_eq!(grid.dimensions(), (21, 11, 21));
        assert_eq!(grid.inventory_grid[0], FixedPoint::from_float(-10.0));
        assert_eq!(grid.inventory_grid[20], FixedPoint::from_float(10.0));
        assert_eq!(grid.time_grid[0], FixedPoint::zero());
        assert_eq!(grid.time_grid[10], FixedPoint::one());
    }

    #[test]
    fn test_value_function() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-5.0), FixedPoint::from_float(5.0)), 11,
            (FixedPoint::zero(), FixedPoint::one()), 6,
            (FixedPoint::from_float(95.0), FixedPoint::from_float(105.0)), 11,
        );
        
        let mut value_function = ValueFunction::new(grid);
        
        // Test setting and getting values
        value_function.set_value(5, 3, 5, FixedPoint::from_float(10.0));
        assert_eq!(value_function.get_value(5, 3, 5), FixedPoint::from_float(10.0));
        
        // Test interpolation
        let interpolated = value_function.interpolate(
            FixedPoint::zero(),
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(100.0),
        );
        assert!(interpolated >= FixedPoint::zero());
    }

    #[test]
    fn test_hjb_solver_creation() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-5.0), FixedPoint::from_float(5.0)), 11,
            (FixedPoint::zero(), FixedPoint::one()), 6,
            (FixedPoint::from_float(95.0), FixedPoint::from_float(105.0)), 11,
        );
        
        let boundary_conditions = BoundaryConditions::new();
        
        let solver = HJBSolver::new(
            grid,
            boundary_conditions,
            NumericalScheme::ImplicitEuler,
            FixedPoint::from_float(0.01),
        );
        
        assert_eq!(solver.numerical_scheme, NumericalScheme::ImplicitEuler);
        assert_eq!(solver.dt, FixedPoint::from_float(0.01));
    }
}    
                let time = self.grid.time_grid[j];
                    let price = self.grid.price_grid[k];
                    
                    // Compute optimal controls at this grid point
                    let (bid_spread, ask_spread) = self.compute_optimal_controls(
                        value_function, i, j, k, inventory, price, params
                    )?;
                    
                    // Compute optimal intensity
                    let bid_intensity = self.compute_arrival_intensity(bid_spread, params);
                    let ask_intensity = self.compute_arrival_intensity(ask_spread, params);
                    let total_intensity = bid_intensity + ask_intensity;
                    
                    let flat_index = self.grid.to_flat_index(i, j, k);
                    controls.optimal_bid_spread[flat_index] = bid_spread;
                    controls.optimal_ask_spread[flat_index] = ask_spread;
                    controls.optimal_intensity[flat_index] = total_intensity;
                }
            }
        }
        
        Ok(controls)
    }
}

/// Advanced finite difference schemes for PDE solving
pub struct FiniteDifferenceSchemes;

impl FiniteDifferenceSchemes {
    /// Upwind scheme for convection terms
    pub fn upwind_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
        velocity: FixedPoint,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        if velocity > FixedPoint::zero() {
            // Forward difference
            let dx = grid[index] - grid[index - 1];
            (values[index] - values[index - 1]) / dx
        } else {
            // Backward difference
            let dx = grid[index + 1] - grid[index];
            (values[index + 1] - values[index]) / dx
        }
    }
    
    /// Central difference scheme
    pub fn central_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        let dx = grid[index + 1] - grid[index - 1];
        (values[index + 1] - values[index - 1]) / dx
    }
    
    /// Second derivative using central differences
    pub fn second_derivative(
        values: &[FixedPoint],
        grid: &[FixedPoint],
        index: usize,
    ) -> FixedPoint {
        if index == 0 || index >= values.len() - 1 {
            return FixedPoint::zero();
        }
        
        let dx = grid[index + 1] - grid[index];
        (values[index + 1] - FixedPoint::from_float(2.0) * values[index] + values[index - 1]) / (dx * dx)
    }
    
    /// Crank-Nicolson time stepping
    pub fn crank_nicolson_step(
        old_values: &[FixedPoint],
        new_values: &mut [FixedPoint],
        dt: FixedPoint,
        spatial_operator: impl Fn(&[FixedPoint], usize) -> FixedPoint,
    ) {
        for i in 1..old_values.len() - 1 {
            let old_spatial = spatial_operator(old_values, i);
            let new_spatial = spatial_operator(new_values, i);
            
            new_values[i] = old_values[i] + dt * (old_spatial + new_spatial) / FixedPoint::from_float(2.0);
        }
    }
    
    /// Implicit Euler time stepping
    pub fn implicit_euler_step(
        old_values: &[FixedPoint],
        new_values: &mut [FixedPoint],
        dt: FixedPoint,
        spatial_operator: impl Fn(&[FixedPoint], usize) -> FixedPoint,
    ) {
        // Simplified implicit step (would need matrix solver in practice)
        for i in 1..old_values.len() - 1 {
            let spatial = spatial_operator(old_values, i);
            new_values[i] = old_values[i] + dt * spatial;
        }
    }
}

/// Value function optimization using policy iteration
pub struct ValueFunctionOptimizer {
    pub hjb_solver: HJBSolver,
    pub max_policy_iterations: usize,
    pub policy_tolerance: FixedPoint,
}

impl ValueFunctionOptimizer {
    /// Create a new value function optimizer
    pub fn new(hjb_solver: HJBSolver) -> Self {
        Self {
            hjb_solver,
            max_policy_iterations: 100,
            policy_tolerance: FixedPoint::from_float(1e-6),
        }
    }
    
    /// Optimize value function using policy iteration
    pub fn optimize_value_function(&self, params: &ModelParameters) -> Result<(ValueFunction, OptimalControls), OptimizationError> {
        let mut value_function = ValueFunction::new(self.hjb_solver.grid.clone());
        let mut controls = OptimalControls::new(self.hjb_solver.grid.clone());
        
        // Initialize with terminal condition
        self.hjb_solver.apply_terminal_condition(&mut value_function, params)?;
        
        // Policy iteration loop
        for iteration in 0..self.max_policy_iterations {
            let old_controls = controls.clone();
            
            // Policy evaluation: solve for value function given current policy
            value_function = self.policy_evaluation(&controls, params)?;
            
            // Policy improvement: update controls based on new value function
            controls = self.hjb_solver.compute_optimal_controls(&value_function, params)?;
            
            // Check convergence
            let policy_change = self.compute_policy_change(&old_controls, &controls);
            if policy_change < self.policy_tolerance {
                break;
            }
        }
        
        Ok((value_function, controls))
    }
    
    /// Policy evaluation step
    fn policy_evaluation(&self, controls: &OptimalControls, params: &ModelParameters) -> Result<ValueFunction, OptimizationError> {
        let mut value_function = ValueFunction::new(self.hjb_solver.grid.clone());
        
        // Apply terminal condition
        self.hjb_solver.apply_terminal_condition(&mut value_function, params)?;
        
        // Backward time stepping with fixed policy
        let time_steps = self.hjb_solver.grid.time_grid.len();
        
        for t_idx in (0..time_steps-1).rev() {
            self.policy_evaluation_step(&mut value_function, controls, t_idx, params)?;
            self.hjb_solver.apply_boundary_conditions(&mut value_function, t_idx)?;
        }
        
        Ok(value_function)
    }
    
    /// Single policy evaluation step
    fn policy_evaluation_step(
        &self,
        value_function: &mut ValueFunction,
        controls: &OptimalControls,
        t_idx: usize,
        params: &ModelParameters,
    ) -> Result<(), OptimizationError> {
        let (ni, _, nk) = self.hjb_solver.grid.dimensions();
        
        for i in 1..ni-1 {
            for k in 1..nk-1 {
                let inventory = self.hjb_solver.grid.inventory_grid[i];
                let price = self.hjb_solver.grid.price_grid[k];
                
                // Get fixed controls for this state
                let flat_index = self.hjb_solver.grid.to_flat_index(i, t_idx, k);
                let bid_spread = controls.optimal_bid_spread[flat_index];
                let ask_spread = controls.optimal_ask_spread[flat_index];
                
                // Compute HJB operator with fixed controls
                let hjb_value = self.hjb_solver.compute_hjb_operator(
                    value_function, i, t_idx, k, inventory, price,
                    bid_spread, ask_spread, params
                )?;
                
                // Update value function
                let current_value = value_function.get_value(i, t_idx + 1, k);
                let new_value = current_value + self.hjb_solver.dt * hjb_value;
                value_function.set_value(i, t_idx, k, new_value);
            }
        }
        
        Ok(())
    }
    
    /// Compute change in policy between iterations
    fn compute_policy_change(&self, old_controls: &OptimalControls, new_controls: &OptimalControls) -> FixedPoint {
        let mut max_change = FixedPoint::zero();
        
        for i in 0..old_controls.optimal_bid_spread.len() {
            let bid_change = (old_controls.optimal_bid_spread[i] - new_controls.optimal_bid_spread[i]).abs();
            let ask_change = (old_controls.optimal_ask_spread[i] - new_controls.optimal_ask_spread[i]).abs();
            
            max_change = max_change.max(bid_change).max(ask_change);
        }
        
        max_change
    }
}

/// Constrained optimization solver using penalty methods
pub struct ConstrainedOptimizer {
    pub penalty_coefficient: FixedPoint,
    pub penalty_growth_rate: FixedPoint,
    pub max_penalty_iterations: usize,
    pub inner_optimizer: GradientDescentOptimizer,
}

impl ConstrainedOptimizer {
    /// Create a new constrained optimizer
    pub fn new(inner_optimizer: GradientDescentOptimizer) -> Self {
        Self {
            penalty_coefficient: FixedPoint::from_float(1.0),
            penalty_growth_rate: FixedPoint::from_float(10.0),
            max_penalty_iterations: 10,
            inner_optimizer,
        }
    }
    
    /// Solve constrained optimization problem using penalty methods
    pub fn optimize_constrained<F, G>(
        &mut self,
        objective: F,
        gradient: G,
        initial_point: &[FixedPoint],
        constraints: &[Constraint],
        optimization_type: OptimizationType,
    ) -> OptimizationResult
    where
        F: Fn(&[FixedPoint]) -> FixedPoint + Clone,
        G: Fn(&[FixedPoint]) -> Vec<FixedPoint> + Clone,
    {
        let mut current_point = initial_point.to_vec();
        let mut penalty_coeff = self.penalty_coefficient;
        
        for _penalty_iter in 0..self.max_penalty_iterations {
            // Create penalized objective function
            let penalized_objective = {
                let obj = objective.clone();
                let penalty = penalty_coeff;
                move |x: &[FixedPoint]| -> FixedPoint {
                    let base_obj = obj(x);
                    let penalty_term = Self::compute_penalty_term(x, constraints, penalty);
                    base_obj + penalty_term
                }
            };
            
            // Create penalized gradient
            let penalized_gradient = {
                let grad = gradient.clone();
                let penalty = penalty_coeff;
                move |x: &[FixedPoint]| -> Vec<FixedPoint> {
                    let mut base_grad = grad(x);
                    let penalty_grad = Self::compute_penalty_gradient(x, constraints, penalty);
                    
                    for i in 0..base_grad.len() {
                        base_grad[i] = base_grad[i] + penalty_grad[i];
                    }
                    
                    base_grad
                }
            };
            
            // Solve penalized problem
            let result = self.inner_optimizer.optimize(
                penalized_objective,
                penalized_gradient,
                &current_point,
                &[], // No additional constraints for inner problem
                optimization_type,
            );
            
            current_point = result.solution;
            
            // Check constraint satisfaction
            let constraint_violations = self.inner_optimizer.check_constraint_violations(&current_point, constraints);
            let max_violation = constraint_violations.iter()
                .fold(FixedPoint::zero(), |acc, &v| acc.max(v));
            
            if max_violation < self.inner_optimizer.tolerance {
                return OptimizationResult {
                    solution: current_point,
                    objective_value: objective(&current_point),
                    iterations: result.iterations,
                    converged: true,
                    gradient_norm: result.gradient_norm,
                    constraint_violations,
                };
            }
            
            // Increase penalty coefficient
            penalty_coeff = penalty_coeff * self.penalty_growth_rate;
        }
        
        // Return best solution found
        let constraint_violations = self.inner_optimizer.check_constraint_violations(&current_point, constraints);
        OptimizationResult {
            solution: current_point,
            objective_value: objective(&current_point),
            iterations: self.max_penalty_iterations * self.inner_optimizer.max_iterations,
            converged: false,
            gradient_norm: FixedPoint::zero(),
            constraint_violations,
        }
    }
    
    /// Compute penalty term for constraints
    fn compute_penalty_term(x: &[FixedPoint], constraints: &[Constraint], penalty_coeff: FixedPoint) -> FixedPoint {
        let mut penalty = FixedPoint::zero();
        
        for constraint in constraints {
            let violation = match constraint {
                Constraint::Equality(f, target) => {
                    let diff = f(x) - *target;
                    diff * diff
                }
                Constraint::LessEqual(f, upper) => {
                    let violation = (f(x) - *upper).max(FixedPoint::zero());
                    violation * violation
                }
                Constraint::GreaterEqual(f, lower) => {
                    let violation = (*lower - f(x)).max(FixedPoint::zero());
                    violation * violation
                }
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        let val = x[*index];
                        if val < *lower {
                            let violation = *lower - val;
                            violation * violation
                        } else if val > *upper {
                            let violation = val - *upper;
                            violation * violation
                        } else {
                            FixedPoint::zero()
                        }
                    } else {
                        FixedPoint::zero()
                    }
                }
            };
            
            penalty = penalty + penalty_coeff * violation;
        }
        
        penalty
    }
    
    /// Compute gradient of penalty term
    fn compute_penalty_gradient(x: &[FixedPoint], constraints: &[Constraint], penalty_coeff: FixedPoint) -> Vec<FixedPoint> {
        let mut penalty_grad = vec![FixedPoint::zero(); x.len()];
        
        for constraint in constraints {
            match constraint {
                Constraint::Box(index, lower, upper) => {
                    if *index < x.len() {
                        let val = x[*index];
                        if val < *lower {
                            penalty_grad[*index] = penalty_grad[*index] - 
                                FixedPoint::from_float(2.0) * penalty_coeff * (*lower - val);
                        } else if val > *upper {
                            penalty_grad[*index] = penalty_grad[*index] + 
                                FixedPoint::from_float(2.0) * penalty_coeff * (val - *upper);
                        }
                    }
                }
                // For function constraints, would need numerical differentiation
                _ => {}
            }
        }
        
        penalty_grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_optimizer() {
        let optimizer = GradientDescentOptimizer::new(
            FixedPoint::from_float(0.1),
            1000,
            FixedPoint::from_float(1e-6),
        );
        
        // Test quadratic function: f(x) = (x-2)^2
        let objective = |x: &[FixedPoint]| -> FixedPoint {
            let diff = x[0] - FixedPoint::from_float(2.0);
            diff * diff
        };
        
        let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
            let diff = x[0] - FixedPoint::from_float(2.0);
            vec![FixedPoint::from_float(2.0) * diff]
        };
        
        let initial = vec![FixedPoint::from_float(0.0)];
        let result = optimizer.optimize(
            objective,
            gradient,
            &initial,
            &[],
            OptimizationType::Minimize,
        );
        
        assert!(result.converged);
        assert!((result.solution[0] - FixedPoint::from_float(2.0)).abs() < FixedPoint::from_float(0.01));
    }
    
    #[test]
    fn test_state_grid_creation() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-10.0), FixedPoint::from_float(10.0)),
            21,
            (FixedPoint::from_float(0.0), FixedPoint::from_float(1.0)),
            11,
            (FixedPoint::from_float(90.0), FixedPoint::from_float(110.0)),
            21,
        );
        
        assert_eq!(grid.inventory_grid.len(), 21);
        assert_eq!(grid.time_grid.len(), 11);
        assert_eq!(grid.price_grid.len(), 21);
        
        // Check grid bounds
        assert_eq!(grid.inventory_grid[0], FixedPoint::from_float(-10.0));
        assert_eq!(grid.inventory_grid[20], FixedPoint::from_float(10.0));
        assert_eq!(grid.time_grid[0], FixedPoint::from_float(0.0));
        assert_eq!(grid.time_grid[10], FixedPoint::from_float(1.0));
    }
    
    #[test]
    fn test_hjb_solver_creation() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-5.0), FixedPoint::from_float(5.0)),
            11,
            (FixedPoint::from_float(0.0), FixedPoint::from_float(1.0)),
            101,
            (FixedPoint::from_float(95.0), FixedPoint::from_float(105.0)),
            11,
        );
        
        let boundary_conditions = BoundaryConditions::new();
        let solver = HJBSolver::new(
            grid,
            boundary_conditions,
            NumericalScheme::ImplicitEuler,
            FixedPoint::from_float(0.01),
        );
        
        assert_eq!(solver.numerical_scheme, NumericalScheme::ImplicitEuler);
        assert_eq!(solver.dt, FixedPoint::from_float(0.01));
    }
    
    #[test]
    fn test_value_function_operations() {
        let grid = StateGrid::new(
            (FixedPoint::from_float(-1.0), FixedPoint::from_float(1.0)),
            3,
            (FixedPoint::from_float(0.0), FixedPoint::from_float(1.0)),
            3,
            (FixedPoint::from_float(99.0), FixedPoint::from_float(101.0)),
            3,
        );
        
        let mut value_function = ValueFunction::new(grid);
        
        // Test setting and getting values
        value_function.set_value(1, 1, 1, FixedPoint::from_float(5.0));
        assert_eq!(value_function.get_value(1, 1, 1), FixedPoint::from_float(5.0));
        
        // Test interpolation (simplified nearest neighbor)
        let interpolated = value_function.interpolate(
            FixedPoint::from_float(0.0),
            FixedPoint::from_float(0.5),
            FixedPoint::from_float(100.0),
        );
        assert_eq!(interpolated, FixedPoint::from_float(5.0));
    }
    
    #[test]
    fn test_finite_difference_schemes() {
        let values = vec![
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(4.0),
            FixedPoint::from_float(9.0),
            FixedPoint::from_float(16.0),
            FixedPoint::from_float(25.0),
        ];
        let grid = vec![
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(3.0),
            FixedPoint::from_float(4.0),
            FixedPoint::from_float(5.0),
        ];
        
        // Test central derivative (should approximate 2x for x^2)
        let derivative = FiniteDifferenceSchemes::central_derivative(&values, &grid, 2);
        let expected = FixedPoint::from_float(6.0); // 2*3 = 6
        assert!((derivative - expected).abs() < FixedPoint::from_float(0.1));
        
        // Test second derivative (should be approximately 2 for x^2)
        let second_derivative = FiniteDifferenceSchemes::second_derivative(&values, &grid, 2);
        let expected_second = FixedPoint::from_float(2.0);
        assert!((second_derivative - expected_second).abs() < FixedPoint::from_float(0.1));
    }
    
    #[test]
    fn test_constrained_optimization() {
        let inner_optimizer = GradientDescentOptimizer::new(
            FixedPoint::from_float(0.1),
            1000,
            FixedPoint::from_float(1e-6),
        );
        
        let mut constrained_optimizer = ConstrainedOptimizer::new(inner_optimizer);
        
        // Minimize (x-3)^2 subject to x >= 1
        let objective = |x: &[FixedPoint]| -> FixedPoint {
            let diff = x[0] - FixedPoint::from_float(3.0);
            diff * diff
        };
        
        let gradient = |x: &[FixedPoint]| -> Vec<FixedPoint> {
            let diff = x[0] - FixedPoint::from_float(3.0);
            vec![FixedPoint::from_float(2.0) * diff]
        };
        
        let constraints = vec![
            Constraint::Box(0, FixedPoint::from_float(1.0), FixedPoint::from_float(10.0))
        ];
        
        let initial = vec![FixedPoint::from_float(0.0)];
        let result = constrained_optimizer.optimize_constrained(
            objective,
            gradient,
            &initial,
            &constraints,
            OptimizationType::Minimize,
        );
        
        // Should converge to x = 3 (unconstrained minimum)
        assert!((result.solution[0] - FixedPoint::from_float(3.0)).abs() < FixedPoint::from_float(0.1));
    }
    
    #[test]
    fn test_model_parameters_default() {
        let params = ModelParameters::default();
        
        assert_eq!(params.drift_coefficient, FixedPoint::from_float(0.05));
        assert_eq!(params.volatility_coefficient, FixedPoint::from_float(0.2));
        assert_eq!(params.inventory_penalty, FixedPoint::from_float(0.01));
        assert!(params.max_inventory > FixedPoint::zero());
    }
}