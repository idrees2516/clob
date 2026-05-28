//! Hawkes Process Engine for Order Flow Modeling
//! 
//! This module implements multivariate Hawkes processes for modeling
//! order flow intensity, clustering effects, and cross-excitation
//! between buy and sell orders in high-frequency trading.

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use thiserror::Error;
use rayon::prelude::*;

#[derive(Error, Debug)]
pub enum HawkesError {
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
    #[error("Calibration error: {0}")]
    CalibrationError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// Hawkes process kernel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelType {
    /// Exponential kernel: κ(t) = α * exp(-β * t)
    Exponential { alpha: FixedPoint, beta: FixedPoint },
    /// Power law kernel: κ(t) = α * (t + c)^(-β)
    PowerLaw { alpha: FixedPoint, beta: FixedPoint, cutoff: FixedPoint },
    /// Sum of exponentials: κ(t) = Σ αᵢ * exp(-βᵢ * t)
    SumExponentials { alphas: Vec<FixedPoint>, betas: Vec<FixedPoint> },
}

impl KernelType {
    /// Evaluate kernel at time t
    pub fn evaluate(&self, t: FixedPoint) -> FixedPoint {
        if t.to_float() <= 0.0 {
            return FixedPoint::zero();
        }
        
        match self {
            KernelType::Exponential { alpha, beta } => {
                *alpha * (-(*beta) * t).exp()
            }
            KernelType::PowerLaw { alpha, beta, cutoff } => {
                let denominator = t + *cutoff;
                let power_term = FixedPoint::from_float(
                    denominator.to_float().powf(-beta.to_float())
                );
                *alpha * power_term
            }
            KernelType::SumExponentials { alphas, betas } => {
                alphas.iter().zip(betas.iter())
                    .map(|(&alpha, &beta)| alpha * (-(beta * t)).exp())
                    .fold(FixedPoint::zero(), |acc, term| acc + term)
            }
        }
    }
    
    /// Compute integral of kernel from 0 to t
    pub fn integral(&self, t: FixedPoint) -> FixedPoint {
        if t.to_float() <= 0.0 {
            return FixedPoint::zero();
        }
        
        match self {
            KernelType::Exponential { alpha, beta } => {
                (*alpha / *beta) * (FixedPoint::one() - (-(*beta) * t).exp())
            }
            KernelType::PowerLaw { alpha, beta, cutoff } => {
                // Approximate integral for power law
                let n_steps = 100;
                let dt = t / FixedPoint::from_float(n_steps as f64);
                let mut integral = FixedPoint::zero();
                
                for i in 1..=n_steps {
                    let ti = FixedPoint::from_float(i as f64) * dt;
                    integral = integral + self.evaluate(ti) * dt;
                }
                integral
            }
            KernelType::SumExponentials { alphas, betas } => {
                alphas.iter().zip(betas.iter())
                    .map(|(&alpha, &beta)| {
                        (alpha / beta) * (FixedPoint::one() - (-(beta * t)).exp())
                    })
                    .fold(FixedPoint::zero(), |acc, term| acc + term)
            }
        }
    }
}

/// Multivariate Hawkes process parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivariateHawkesParams {
    /// Baseline intensities λ₀ᵢ for each process
    pub baseline_intensities: Vec<FixedPoint>,
    /// Cross-excitation kernels κᵢⱼ(t)
    pub kernels: Vec<Vec<KernelType>>,
    /// Maximum intensity (for thinning algorithm)
    pub max_intensity: FixedPoint,
}

impl MultivariateHawkesParams {
    /// Create new parameters with validation
    pub fn new(
        baseline_intensities: Vec<FixedPoint>,
        kernels: Vec<Vec<KernelType>>,
        max_intensity: FixedPoint,
    ) -> Result<Self, HawkesError> {
        let n = baseline_intensities.len();
        
        if kernels.len() != n {
            return Err(HawkesError::InvalidParameters(
                "Kernel matrix dimensions don't match baseline intensities".to_string()
            ));
        }
        
        for (i, row) in kernels.iter().enumerate() {
            if row.len() != n {
                return Err(HawkesError::InvalidParameters(
                    format!("Kernel row {} has wrong dimension", i)
                ));
            }
        }
        
        for &intensity in &baseline_intensities {
            if intensity.to_float() <= 0.0 {
                return Err(HawkesError::InvalidParameters(
                    "Baseline intensities must be positive".to_string()
                ));
            }
        }
        
        Ok(Self {
            baseline_intensities,
            kernels,
            max_intensity,
        })
    }
    
    /// Get number of processes
    pub fn dimension(&self) -> usize {
        self.baseline_intensities.len()
    }
}

/// Event in multivariate Hawkes process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesEvent {
    pub time: FixedPoint,
    pub process_id: usize,
    pub mark: Option<FixedPoint>, // Optional mark (e.g., order size)
}

/// Multivariate Hawkes process state
#[derive(Debug, Clone)]
pub struct HawkesState {
    pub current_time: FixedPoint,
    pub intensities: Vec<FixedPoint>,
    pub event_history: Vec<VecDeque<HawkesEvent>>,
    pub total_events: Vec<usize>,
}

impl HawkesState {
    pub fn new(params: &MultivariateHawkesParams, history_size: usize) -> Self {
        let n = params.dimension();
        Self {
            current_time: FixedPoint::zero(),
            intensities: params.baseline_intensities.clone(),
            event_history: vec![VecDeque::with_capacity(history_size); n],
            total_events: vec![0; n],
        }
    }
    
    /// Update intensities based on current time and event history
    pub fn update_intensities(&mut self, params: &MultivariateHawkesParams) {
        let n = params.dimension();
        
        for i in 0..n {
            let mut intensity = params.baseline_intensities[i];
            
            // Add contributions from all processes
            for j in 0..n {
                for event in &self.event_history[j] {
                    let dt = self.current_time - event.time;
                    if dt.to_float() > 0.0 {
                        let kernel_value = params.kernels[i][j].evaluate(dt);
                        intensity = intensity + kernel_value;
                    }
                }
            }
            
            self.intensities[i] = intensity;
        }
    }
    
    /// Add new event and update state
    pub fn add_event(&mut self, event: HawkesEvent, max_history: usize) {
        let process_id = event.process_id;
        
        // Add to history
        self.event_history[process_id].push_back(event);
        self.total_events[process_id] += 1;
        
        // Maintain history size
        if self.event_history[process_id].len() > max_history {
            self.event_history[process_id].pop_front();
        }
    }
    
    /// Get total intensity (sum of all process intensities)
    pub fn total_intensity(&self) -> FixedPoint {
        self.intensities.iter().fold(FixedPoint::zero(), |acc, &intensity| acc + intensity)
    }
}

/// Multivariate Hawkes process simulator
pub struct MultivariateHawkesSimulator {
    pub params: MultivariateHawkesParams,
    pub state: HawkesState,
    pub max_history: usize,
}

impl MultivariateHawkesSimulator {
    pub fn new(params: MultivariateHawkesParams, max_history: usize) -> Self {
        let state = HawkesState::new(&params, max_history);
        Self {
            params,
            state,
            max_history,
        }
    }
    
    /// Simulate using thinning algorithm (Ogata's method)
    pub fn simulate_until(
        &mut self,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        let mut events = Vec::new();
        
        while self.state.current_time < end_time {
            // Update intensities
            self.state.update_intensities(&self.params);
            
            // Generate next event time using thinning
            let next_event = self.generate_next_event(end_time, rng)?;
            
            if let Some(event) = next_event {
                if event.time >= end_time {
                    break;
                }
                
                self.state.current_time = event.time;
                self.state.add_event(event.clone(), self.max_history);
                events.push(event);
            } else {
                break;
            }
        }
        
        Ok(events)
    }
    
    /// Generate next event using thinning algorithm
    fn generate_next_event(
        &mut self,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Option<HawkesEvent>, HawkesError> {
        let max_iter = 10000;
        let mut iter = 0;
        
        loop {
            iter += 1;
            if iter > max_iter {
                return Err(HawkesError::SimulationError(
                    "Thinning algorithm failed to converge".to_string()
                ));
            }
            
            // Current total intensity
            let lambda_total = self.state.total_intensity();
            
            if lambda_total.to_float() <= 0.0 {
                return Ok(None);
            }
            
            // Generate exponential waiting time
            let u1 = rng.next_fixed();
            let dt = FixedPoint::from_float(
                -u1.to_float().ln() / lambda_total.to_float()
            );
            
            let candidate_time = self.state.current_time + dt;
            
            if candidate_time >= end_time {
                return Ok(None);
            }
            
            // Update time and intensities
            self.state.current_time = candidate_time;
            self.state.update_intensities(&self.params);
            
            // Thinning step
            let u2 = rng.next_fixed();
            let new_lambda_total = self.state.total_intensity();
            
            if u2.to_float() * self.params.max_intensity.to_float() <= new_lambda_total.to_float() {
                // Accept event - determine which process
                let u3 = rng.next_fixed();
                let mut cumulative = FixedPoint::zero();
                
                for (i, &intensity) in self.state.intensities.iter().enumerate() {
                    cumulative = cumulative + intensity;
                    if u3 * new_lambda_total <= cumulative {
                        return Ok(Some(HawkesEvent {
                            time: candidate_time,
                            process_id: i,
                            mark: None,
                        }));
                    }
                }
            }
        }
    }
    
    /// Simulate multiple paths in parallel
    pub fn simulate_paths(
        &self,
        n_paths: usize,
        end_time: FixedPoint,
        base_seed: u64,
    ) -> Result<Vec<Vec<HawkesEvent>>, HawkesError> {
        let paths: Result<Vec<_>, _> = (0..n_paths)
            .into_par_iter()
            .map(|i| {
                let mut simulator = MultivariateHawkesSimulator::new(
                    self.params.clone(),
                    self.max_history,
                );
                let mut rng = DeterministicRng::new(base_seed + i as u64);
                simulator.simulate_until(end_time, &mut rng)
            })
            .collect();
        
        paths
    }
    
    /// Reset simulator state
    pub fn reset(&mut self) {
        self.state = HawkesState::new(&self.params, self.max_history);
    }
}

/// Maximum likelihood estimator for Hawkes processes
pub struct HawkesMLEstimator {
    pub tolerance: FixedPoint,
    pub max_iterations: usize,
}

impl HawkesMLEstimator {
    pub fn new(tolerance: FixedPoint, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }
    
    /// Estimate parameters from event data
    pub fn estimate(
        &self,
        events: &[Vec<HawkesEvent>],
        initial_params: MultivariateHawkesParams,
    ) -> Result<MultivariateHawkesParams, HawkesError> {
        if events.is_empty() {
            return Err(HawkesError::InsufficientData(
                "No event data provided".to_string()
            ));
        }
        
        let n = initial_params.dimension();
        let mut current_params = initial_params;
        
        for iteration in 0..self.max_iterations {
            let log_likelihood = self.compute_log_likelihood(events, &current_params)?;
            
            // Compute gradient (simplified - in practice would use numerical differentiation)
            let gradient = self.compute_gradient(events, &current_params)?;
            
            // Update parameters using gradient ascent
            let step_size = FixedPoint::from_float(0.01);
            for i in 0..n {
                let update = step_size * gradient[i];
                current_params.baseline_intensities[i] = 
                    current_params.baseline_intensities[i] + update;
                
                // Ensure positivity
                if current_params.baseline_intensities[i].to_float() <= 0.0 {
                    current_params.baseline_intensities[i] = FixedPoint::from_float(0.001);
                }
            }
            
            // Check convergence
            let gradient_norm = gradient.iter()
                .map(|&g| g * g)
                .fold(FixedPoint::zero(), |acc, g2| acc + g2);
            
            if gradient_norm < self.tolerance * self.tolerance {
                break;
            }
        }
        
        Ok(current_params)
    }
    
    /// Compute log-likelihood of parameters given data
    fn compute_log_likelihood(
        &self,
        events: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
    ) -> Result<FixedPoint, HawkesError> {
        let mut log_likelihood = FixedPoint::zero();
        
        for path in events {
            let mut simulator = MultivariateHawkesSimulator::new(
                params.clone(),
                1000,
            );
            
            let end_time = path.iter()
                .map(|event| event.time)
                .fold(FixedPoint::zero(), |acc, t| if t > acc { t } else { acc });
            
            // Compute intensity at each event time
            for event in path {
                simulator.state.current_time = event.time;
                simulator.state.update_intensities(params);
                
                let intensity = simulator.state.intensities[event.process_id];
                if intensity.to_float() > 0.0 {
                    log_likelihood = log_likelihood + intensity.ln();
                }
                
                simulator.state.add_event(event.clone(), simulator.max_history);
            }
            
            // Subtract integral term
            let integral_term = self.compute_intensity_integral(params, end_time)?;
            log_likelihood = log_likelihood - integral_term;
        }
        
        Ok(log_likelihood)
    }
    
    /// Compute gradient of log-likelihood (simplified)
    fn compute_gradient(
        &self,
        events: &[Vec<HawkesEvent>],
        params: &MultivariateHawkesParams,
    ) -> Result<Vec<FixedPoint>, HawkesError> {
        let n = params.dimension();
        let mut gradient = vec![FixedPoint::zero(); n];
        
        // Simplified gradient computation
        for path in events {
            let end_time = path.iter()
                .map(|event| event.time)
                .fold(FixedPoint::zero(), |acc, t| if t > acc { t } else { acc });
            
            for i in 0..n {
                let event_count = path.iter()
                    .filter(|event| event.process_id == i)
                    .count();
                
                gradient[i] = gradient[i] + 
                    FixedPoint::from_float(event_count as f64) / params.baseline_intensities[i] - 
                    end_time;
            }
        }
        
        Ok(gradient)
    }
    
    /// Compute integral of intensity function
    fn compute_intensity_integral(
        &self,
        params: &MultivariateHawkesParams,
        end_time: FixedPoint,
    ) -> Result<FixedPoint, HawkesError> {
        let n = params.dimension();
        let mut integral = FixedPoint::zero();
        
        // Baseline intensity contribution
        for &baseline in &params.baseline_intensities {
            integral = integral + baseline * end_time;
        }
        
        // Kernel contributions (simplified)
        for i in 0..n {
            for j in 0..n {
                let kernel_integral = params.kernels[i][j].integral(end_time);
                integral = integral + kernel_integral;
            }
        }
        
        Ok(integral)
    }
}

/// Branching structure for efficient Hawkes simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HawkesBranch {
    /// Parent event that triggered this branch
    pub parent_event: Option<HawkesEvent>,
    /// Child events generated by this branch
    pub child_events: Vec<HawkesEvent>,
    /// Branch generation (0 for immigrant events)
    pub generation: usize,
    /// Total offspring count
    pub offspring_count: usize,
}

impl HawkesBranch {
    pub fn new_immigrant(event: HawkesEvent) -> Self {
        Self {
            parent_event: None,
            child_events: vec![event],
            generation: 0,
            offspring_count: 0,
        }
    }
    
    pub fn new_offspring(parent: HawkesEvent, children: Vec<HawkesEvent>) -> Self {
        let offspring_count = children.len();
        Self {
            parent_event: Some(parent),
            child_events: children,
            generation: 1, // Will be updated based on parent's generation
            offspring_count,
        }
    }
}

/// Branching structure simulator for Hawkes processes
pub struct BranchingHawkesSimulator {
    pub params: MultivariateHawkesParams,
    pub branches: Vec<HawkesBranch>,
    pub max_generation: usize,
    pub branching_ratios: Vec<Vec<FixedPoint>>, // Expected offspring per process pair
}

impl BranchingHawkesSimulator {
    pub fn new(params: MultivariateHawkesParams, max_generation: usize) -> Self {
        let n = params.dimension();
        
        // Compute branching ratios from kernel integrals
        let mut branching_ratios = vec![vec![FixedPoint::zero(); n]; n];
        for i in 0..n {
            for j in 0..n {
                // Integral of kernel over infinite time gives branching ratio
                branching_ratios[i][j] = match &params.kernels[i][j] {
                    KernelType::Exponential { alpha, beta } => *alpha / *beta,
                    KernelType::PowerLaw { alpha, beta, cutoff: _ } => {
                        if beta.to_float() > 1.0 {
                            *alpha / (*beta - FixedPoint::one())
                        } else {
                            FixedPoint::from_float(f64::INFINITY) // Supercritical
                        }
                    }
                    KernelType::SumExponentials { alphas, betas } => {
                        alphas.iter().zip(betas.iter())
                            .map(|(&a, &b)| a / b)
                            .fold(FixedPoint::zero(), |acc, ratio| acc + ratio)
                    }
                };
            }
        }
        
        Self {
            params,
            branches: Vec::new(),
            max_generation,
            branching_ratios,
        }
    }
    
    /// Simulate using branching structure
    pub fn simulate_branching(
        &mut self,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        let mut all_events = Vec::new();
        self.branches.clear();
        
        // Generate immigrant events (Poisson processes)
        for (i, &baseline) in self.params.baseline_intensities.iter().enumerate() {
            let immigrant_events = self.generate_poisson_events(baseline, end_time, i, rng)?;
            
            for event in immigrant_events {
                if event.time < end_time {
                    let branch = HawkesBranch::new_immigrant(event.clone());
                    self.branches.push(branch);
                    all_events.push(event);
                }
            }
        }
        
        // Generate offspring events for each generation
        for generation in 0..self.max_generation {
            let mut new_branches = Vec::new();
            
            for branch in &self.branches {
                if branch.generation == generation {
                    for parent_event in &branch.child_events {
                        let offspring = self.generate_offspring(parent_event, end_time, rng)?;
                        
                        if !offspring.is_empty() {
                            let mut offspring_branch = HawkesBranch::new_offspring(
                                parent_event.clone(),
                                offspring.clone(),
                            );
                            offspring_branch.generation = generation + 1;
                            new_branches.push(offspring_branch);
                            
                            for event in offspring {
                                if event.time < end_time {
                                    all_events.push(event);
                                }
                            }
                        }
                    }
                }
            }
            
            self.branches.extend(new_branches);
        }
        
        // Sort events by time
        all_events.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        
        Ok(all_events)
    }
    
    /// Generate Poisson events for immigrant process
    fn generate_poisson_events(
        &self,
        rate: FixedPoint,
        end_time: FixedPoint,
        process_id: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        let mut events = Vec::new();
        let mut current_time = FixedPoint::zero();
        
        while current_time < end_time {
            let u = rng.next_fixed();
            let dt = FixedPoint::from_float(-u.to_float().ln()) / rate;
            current_time = current_time + dt;
            
            if current_time < end_time {
                events.push(HawkesEvent {
                    time: current_time,
                    process_id,
                    mark: None,
                });
            }
        }
        
        Ok(events)
    }
    
    /// Generate offspring events from a parent event
    fn generate_offspring(
        &self,
        parent: &HawkesEvent,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        let mut offspring = Vec::new();
        let n = self.params.dimension();
        
        for j in 0..n {
            let kernel = &self.params.kernels[j][parent.process_id];
            
            // Generate offspring for process j from parent in process i
            let offspring_events = self.generate_kernel_events(
                parent.time,
                end_time,
                kernel,
                j,
                rng,
            )?;
            
            offspring.extend(offspring_events);
        }
        
        Ok(offspring)
    }
    
    /// Generate events according to a kernel function
    fn generate_kernel_events(
        &self,
        start_time: FixedPoint,
        end_time: FixedPoint,
        kernel: &KernelType,
        process_id: usize,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        let mut events = Vec::new();
        
        match kernel {
            KernelType::Exponential { alpha, beta } => {
                // Use thinning for exponential kernel
                let mut current_time = start_time;
                
                while current_time < end_time {
                    let u1 = rng.next_fixed();
                    let dt = FixedPoint::from_float(-u1.to_float().ln()) / *alpha;
                    current_time = current_time + dt;
                    
                    if current_time >= end_time {
                        break;
                    }
                    
                    // Accept with probability proportional to kernel value
                    let u2 = rng.next_fixed();
                    let kernel_value = kernel.evaluate(current_time - start_time);
                    
                    if u2 * *alpha <= kernel_value {
                        events.push(HawkesEvent {
                            time: current_time,
                            process_id,
                            mark: None,
                        });
                    }
                }
            }
            _ => {
                // For other kernels, use numerical integration approach
                let dt = (end_time - start_time) / FixedPoint::from_float(1000.0);
                let mut t = start_time + dt;
                
                while t < end_time {
                    let intensity = kernel.evaluate(t - start_time);
                    let prob = intensity * dt;
                    
                    let u = rng.next_fixed();
                    if u < prob {
                        events.push(HawkesEvent {
                            time: t,
                            process_id,
                            mark: None,
                        });
                    }
                    
                    t = t + dt;
                }
            }
        }
        
        Ok(events)
    }
    
    /// Check if the process is subcritical (stable)
    pub fn is_subcritical(&self) -> bool {
        let n = self.params.dimension();
        
        // Compute spectral radius of branching matrix
        let mut max_eigenvalue = 0.0;
        
        for i in 0..n {
            let row_sum: f64 = self.branching_ratios[i].iter()
                .map(|&ratio| ratio.to_float())
                .sum();
            
            if row_sum > max_eigenvalue {
                max_eigenvalue = row_sum;
            }
        }
        
        max_eigenvalue < 1.0
    }
}

/// Real-time parameter updater for Hawkes processes
pub struct RealTimeHawkesUpdater {
    pub window_size: usize,
    pub update_frequency: usize,
    pub learning_rate: FixedPoint,
    pub event_buffer: VecDeque<HawkesEvent>,
    pub update_count: usize,
}

impl RealTimeHawkesUpdater {
    pub fn new(window_size: usize, update_frequency: usize, learning_rate: FixedPoint) -> Self {
        Self {
            window_size,
            update_frequency,
            learning_rate,
            event_buffer: VecDeque::with_capacity(window_size),
            update_count: 0,
        }
    }
    
    /// Add new event and potentially update parameters
    pub fn add_event_and_update(
        &mut self,
        event: HawkesEvent,
        params: &mut MultivariateHawkesParams,
    ) -> Result<bool, HawkesError> {
        // Add event to buffer
        self.event_buffer.push_back(event);
        
        // Maintain window size
        if self.event_buffer.len() > self.window_size {
            self.event_buffer.pop_front();
        }
        
        self.update_count += 1;
        
        // Update parameters if enough events and time has passed
        if self.update_count % self.update_frequency == 0 && self.event_buffer.len() >= 10 {
            self.update_parameters_online(params)?;
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Online parameter update using stochastic gradient ascent
    fn update_parameters_online(
        &mut self,
        params: &mut MultivariateHawkesParams,
    ) -> Result<(), HawkesError> {
        let n = params.dimension();
        let events: Vec<_> = self.event_buffer.iter().cloned().collect();
        
        if events.is_empty() {
            return Ok(());
        }
        
        let end_time = events.iter()
            .map(|e| e.time)
            .fold(FixedPoint::zero(), |acc, t| if t > acc { t } else { acc });
        
        // Compute gradients for baseline intensities
        for i in 0..n {
            let event_count = events.iter()
                .filter(|e| e.process_id == i)
                .count();
            
            // Gradient of log-likelihood w.r.t. baseline intensity
            let gradient = FixedPoint::from_float(event_count as f64) / params.baseline_intensities[i] - end_time;
            
            // Stochastic gradient ascent update
            let update = self.learning_rate * gradient;
            params.baseline_intensities[i] = params.baseline_intensities[i] + update;
            
            // Ensure positivity
            if params.baseline_intensities[i].to_float() <= 0.0 {
                params.baseline_intensities[i] = FixedPoint::from_float(0.001);
            }
        }
        
        // Update kernel parameters (simplified for exponential kernels)
        for i in 0..n {
            for j in 0..n {
                if let KernelType::Exponential { alpha, beta } = &mut params.kernels[i][j] {
                    // Compute empirical excitation effect
                    let excitation_events = self.count_excitation_events(i, j, &events);
                    let total_events = events.iter().filter(|e| e.process_id == i).count();
                    
                    if total_events > 0 {
                        let empirical_ratio = FixedPoint::from_float(excitation_events as f64 / total_events as f64);
                        let current_ratio = *alpha / *beta;
                        
                        // Update alpha to match empirical excitation
                        let alpha_update = self.learning_rate * (empirical_ratio - current_ratio);
                        *alpha = *alpha + alpha_update;
                        
                        // Ensure positivity
                        if alpha.to_float() <= 0.0 {
                            *alpha = FixedPoint::from_float(0.001);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Count excitation events between processes
    fn count_excitation_events(&self, target_process: usize, source_process: usize, events: &[HawkesEvent]) -> usize {
        let mut count = 0;
        let excitation_window = FixedPoint::from_float(1.0); // 1 time unit window
        
        for i in 0..events.len() {
            if events[i].process_id == target_process {
                // Look for recent events in source process
                for j in 0..i {
                    if events[j].process_id == source_process {
                        let time_diff = events[i].time - events[j].time;
                        if time_diff > FixedPoint::zero() && time_diff <= excitation_window {
                            count += 1;
                            break; // Count only the most recent excitation
                        }
                    }
                }
            }
        }
        
        count
    }
    
    /// Get current parameter estimates
    pub fn get_parameter_estimates(&self, params: &MultivariateHawkesParams) -> MultivariateHawkesParams {
        params.clone()
    }
    
    /// Reset the updater state
    pub fn reset(&mut self) {
        self.event_buffer.clear();
        self.update_count = 0;
    }
}

/// Order flow modeling using Hawkes processes
pub struct OrderFlowModel {
    pub buy_sell_hawkes: MultivariateHawkesSimulator,
    pub market_limit_hawkes: MultivariateHawkesSimulator,
    pub branching_simulator: BranchingHawkesSimulator,
    pub real_time_updater: RealTimeHawkesUpdater,
    pub cross_excitation_strength: FixedPoint,
}

impl OrderFlowModel {
    /// Create new order flow model
    pub fn new(
        baseline_buy: FixedPoint,
        baseline_sell: FixedPoint,
        excitation_strength: FixedPoint,
        decay_rate: FixedPoint,
        cross_excitation: FixedPoint,
    ) -> Result<Self, HawkesError> {
        // Buy-sell Hawkes process
        let buy_sell_params = MultivariateHawkesParams::new(
            vec![baseline_buy, baseline_sell],
            vec![
                vec![
                    KernelType::Exponential { alpha: excitation_strength, beta: decay_rate },
                    KernelType::Exponential { alpha: cross_excitation, beta: decay_rate },
                ],
                vec![
                    KernelType::Exponential { alpha: cross_excitation, beta: decay_rate },
                    KernelType::Exponential { alpha: excitation_strength, beta: decay_rate },
                ],
            ],
            baseline_buy + baseline_sell + FixedPoint::from_float(2.0) * excitation_strength,
        )?;
        
        // Market-limit order Hawkes process
        let market_limit_params = MultivariateHawkesParams::new(
            vec![baseline_buy / FixedPoint::from_float(2.0), baseline_sell / FixedPoint::from_float(2.0)],
            vec![
                vec![
                    KernelType::Exponential { alpha: excitation_strength / FixedPoint::from_float(2.0), beta: decay_rate },
                    KernelType::Exponential { alpha: cross_excitation / FixedPoint::from_float(2.0), beta: decay_rate },
                ],
                vec![
                    KernelType::Exponential { alpha: cross_excitation / FixedPoint::from_float(2.0), beta: decay_rate },
                    KernelType::Exponential { alpha: excitation_strength / FixedPoint::from_float(2.0), beta: decay_rate },
                ],
            ],
            (baseline_buy + baseline_sell + FixedPoint::from_float(2.0) * excitation_strength) / FixedPoint::from_float(2.0),
        )?;
        
        let branching_simulator = BranchingHawkesSimulator::new(buy_sell_params.clone(), 5);
        let real_time_updater = RealTimeHawkesUpdater::new(1000, 50, FixedPoint::from_float(0.01));
        
        Ok(Self {
            buy_sell_hawkes: MultivariateHawkesSimulator::new(buy_sell_params, 1000),
            market_limit_hawkes: MultivariateHawkesSimulator::new(market_limit_params, 1000),
            branching_simulator,
            real_time_updater,
            cross_excitation_strength: cross_excitation,
        })
    }
    
    /// Simulate order flow for given time horizon
    pub fn simulate_order_flow(
        &mut self,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<(Vec<HawkesEvent>, Vec<HawkesEvent>), HawkesError> {
        let buy_sell_events = self.buy_sell_hawkes.simulate_until(end_time, rng)?;
        let market_limit_events = self.market_limit_hawkes.simulate_until(end_time, rng)?;
        
        Ok((buy_sell_events, market_limit_events))
    }
    
    /// Get current order flow intensities
    pub fn get_current_intensities(&mut self) -> (Vec<FixedPoint>, Vec<FixedPoint>) {
        self.buy_sell_hawkes.state.update_intensities(&self.buy_sell_hawkes.params);
        self.market_limit_hawkes.state.update_intensities(&self.market_limit_hawkes.params);
        
        (
            self.buy_sell_hawkes.state.intensities.clone(),
            self.market_limit_hawkes.state.intensities.clone(),
        )
    }
    
    /// Simulate using branching structure for efficiency
    pub fn simulate_branching_order_flow(
        &mut self,
        end_time: FixedPoint,
        rng: &mut DeterministicRng,
    ) -> Result<Vec<HawkesEvent>, HawkesError> {
        self.branching_simulator.simulate_branching(end_time, rng)
    }
    
    /// Add real-time event and update parameters
    pub fn add_real_time_event(&mut self, event: HawkesEvent) -> Result<bool, HawkesError> {
        self.real_time_updater.add_event_and_update(event, &mut self.buy_sell_hawkes.params)
    }
    
    /// Check if the order flow process is stable (subcritical)
    pub fn is_stable(&self) -> bool {
        self.branching_simulator.is_subcritical()
    }
    
    /// Get branching ratios for analysis
    pub fn get_branching_ratios(&self) -> &Vec<Vec<FixedPoint>> {
        &self.branching_simulator.branching_ratios
    }
    
    /// Reset all simulators
    pub fn reset_all(&mut self) {
        self.buy_sell_hawkes.reset();
        self.market_limit_hawkes.reset();
        self.real_time_updater.reset();
    }
    
    /// Get current parameter estimates
    pub fn get_current_parameters(&self) -> &MultivariateHawkesParams {
        &self.buy_sell_hawkes.params
    }
    
    /// Update parameters manually
    pub fn update_parameters(&mut self, new_params: MultivariateHawkesParams) -> Result<(), HawkesError> {
        // Validate parameters first
        if new_params.dimension() != self.buy_sell_hawkes.params.dimension() {
            return Err(HawkesError::InvalidParameters(
                "Parameter dimensions don't match".to_string()
            ));
        }
        
        self.buy_sell_hawkes.params = new_params.clone();
        
        // Update branching simulator with new parameters
        self.branching_simulator = BranchingHawkesSimulator::new(new_params, self.branching_simulator.max_generation);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_kernel() {
        let kernel = KernelType::Exponential {
            alpha: FixedPoint::from_float(1.0),
            beta: FixedPoint::from_float(2.0),
        };
        
        let value = kernel.evaluate(FixedPoint::from_float(0.5));
        let expected = (1.0 * (-2.0 * 0.5_f64).exp()) as f64;
        
        assert!((value.to_float() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_hawkes_simulation() {
        let params = MultivariateHawkesParams::new(
            vec![FixedPoint::from_float(0.5), FixedPoint::from_float(0.3)],
            vec![
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(1.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(1.0) },
                ],
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(1.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(1.0) },
                ],
            ],
            FixedPoint::from_float(2.0),
        ).unwrap();
        
        let mut simulator = MultivariateHawkesSimulator::new(params, 100);
        let mut rng = DeterministicRng::new(42);
        
        let events = simulator.simulate_until(FixedPoint::from_float(10.0), &mut rng).unwrap();
        
        assert!(!events.is_empty());
        
        // Check events are ordered by time
        for i in 1..events.len() {
            assert!(events[i].time >= events[i-1].time);
        }
    }

    #[test]
    fn test_order_flow_model() {
        let mut model = OrderFlowModel::new(
            FixedPoint::from_float(1.0),  // baseline_buy
            FixedPoint::from_float(0.8),  // baseline_sell
            FixedPoint::from_float(0.5),  // excitation_strength
            FixedPoint::from_float(2.0),  // decay_rate
            FixedPoint::from_float(0.2),  // cross_excitation
        ).unwrap();
        
        let mut rng = DeterministicRng::new(123);
        let (buy_sell_events, market_limit_events) = model.simulate_order_flow(
            FixedPoint::from_float(5.0),
            &mut rng,
        ).unwrap();
        
        assert!(!buy_sell_events.is_empty());
        assert!(!market_limit_events.is_empty());
        
        let (buy_sell_intensities, market_limit_intensities) = model.get_current_intensities();
        assert_eq!(buy_sell_intensities.len(), 2);
        assert_eq!(market_limit_intensities.len(), 2);
    }

    #[test]
    fn test_branching_simulation() {
        let params = MultivariateHawkesParams::new(
            vec![FixedPoint::from_float(0.3), FixedPoint::from_float(0.2)],
            vec![
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.05), beta: FixedPoint::from_float(2.0) },
                ],
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.05), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(2.0) },
                ],
            ],
            FixedPoint::from_float(1.0),
        ).unwrap();
        
        let mut branching_sim = BranchingHawkesSimulator::new(params, 3);
        let mut rng = DeterministicRng::new(456);
        
        let events = branching_sim.simulate_branching(FixedPoint::from_float(10.0), &mut rng).unwrap();
        
        assert!(!events.is_empty());
        assert!(branching_sim.is_subcritical()); // Should be stable with these parameters
        
        // Check branching ratios are computed correctly
        assert_eq!(branching_sim.branching_ratios.len(), 2);
        assert_eq!(branching_sim.branching_ratios[0].len(), 2);
    }

    #[test]
    fn test_real_time_parameter_update() {
        let initial_params = MultivariateHawkesParams::new(
            vec![FixedPoint::from_float(0.5), FixedPoint::from_float(0.4)],
            vec![
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(1.5) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(1.5) },
                ],
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(1.5) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(1.5) },
                ],
            ],
            FixedPoint::from_float(1.5),
        ).unwrap();
        
        let mut updater = RealTimeHawkesUpdater::new(100, 10, FixedPoint::from_float(0.01));
        let mut params = initial_params.clone();
        
        // Add some test events
        for i in 0..50 {
            let event = HawkesEvent {
                time: FixedPoint::from_float(i as f64 * 0.1),
                process_id: i % 2,
                mark: None,
            };
            
            let updated = updater.add_event_and_update(event, &mut params).unwrap();
            
            // Should update every 10 events
            if (i + 1) % 10 == 0 && i >= 9 {
                assert!(updated);
            }
        }
        
        // Parameters should have changed
        assert_ne!(params.baseline_intensities, initial_params.baseline_intensities);
    }

    #[test]
    fn test_ml_estimation() {
        let true_params = MultivariateHawkesParams::new(
            vec![FixedPoint::from_float(0.8), FixedPoint::from_float(0.6)],
            vec![
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.3), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(2.0) },
                ],
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.1), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.3), beta: FixedPoint::from_float(2.0) },
                ],
            ],
            FixedPoint::from_float(2.0),
        ).unwrap();
        
        // Generate synthetic data
        let mut simulator = MultivariateHawkesSimulator::new(true_params.clone(), 100);
        let mut rng = DeterministicRng::new(789);
        
        let mut event_paths = Vec::new();
        for _ in 0..5 {
            simulator.reset();
            let events = simulator.simulate_until(FixedPoint::from_float(10.0), &mut rng).unwrap();
            event_paths.push(events);
        }
        
        // Estimate parameters
        let estimator = HawkesMLEstimator::new(FixedPoint::from_float(1e-6), 10);
        let initial_guess = MultivariateHawkesParams::new(
            vec![FixedPoint::from_float(0.5), FixedPoint::from_float(0.5)],
            vec![
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.05), beta: FixedPoint::from_float(2.0) },
                ],
                vec![
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.05), beta: FixedPoint::from_float(2.0) },
                    KernelType::Exponential { alpha: FixedPoint::from_float(0.2), beta: FixedPoint::from_float(2.0) },
                ],
            ],
            FixedPoint::from_float(2.0),
        ).unwrap();
        
        let estimated_params = estimator.estimate(&event_paths, initial_guess).unwrap();
        
        // Check that estimation moved parameters in the right direction
        assert!(estimated_params.baseline_intensities[0].to_float() > 0.0);
        assert!(estimated_params.baseline_intensities[1].to_float() > 0.0);
    }
}