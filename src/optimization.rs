use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error("Failed to converge: {0}")]
    ConvergenceError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

pub trait ObjectiveFunction: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
}

pub struct LBFGS {
    max_iter: usize,
    m: usize,
    epsilon: f64,
    s_vectors: Vec<Vec<f64>>,
    y_vectors: Vec<Vec<f64>>,
    rho: Vec<f64>,
}

impl LBFGS {
    pub fn new(max_iter: usize, m: usize, epsilon: f64) -> Self {
        Self {
            max_iter,
            m,
            epsilon,
            s_vectors: Vec::new(),
            y_vectors: Vec::new(),
            rho: Vec::new(),
        }
    }

    pub fn optimize<F: ObjectiveFunction>(
        &mut self,
        objective: &F,
        initial_guess: Vec<f64>,
    ) -> Result<(Vec<f64>, f64), OptimizationError> {
        let n = initial_guess.len();
        let mut x = initial_guess;
        let mut grad = objective.gradient(&x);
        let mut prev_x = vec![0.0; n];
        let mut prev_grad = vec![0.0; n];

        for iter in 0..self.max_iter {
            // Store previous point
            prev_x.copy_from_slice(&x);
            prev_grad.copy_from_slice(&grad);

            // Compute search direction using L-BFGS two-loop recursion
            let mut q = grad.clone();
            let mut alpha = vec![0.0; self.s_vectors.len()];

            // First loop
            for i in (0..self.s_vectors.len()).rev() {
                alpha[i] = self.rho[i] * dot_product(&self.s_vectors[i], &q);
                for j in 0..n {
                    q[j] -= alpha[i] * self.y_vectors[i][j];
                }
            }

            // Scale the initial Hessian approximation
            if !self.s_vectors.is_empty() {
                let gamma = dot_product(&self.s_vectors.last().unwrap(), &self.y_vectors.last().unwrap())
                    / dot_product(&self.y_vectors.last().unwrap(), &self.y_vectors.last().unwrap());
                for j in 0..n {
                    q[j] *= gamma;
                }
            }

            // Second loop
            for i in 0..self.s_vectors.len() {
                let beta = self.rho[i] * dot_product(&self.y_vectors[i], &q);
                for j in 0..n {
                    q[j] += self.s_vectors[i][j] * (alpha[i] - beta);
                }
            }

            // Negate search direction
            for j in 0..n {
                q[j] = -q[j];
            }

            // Line search
            let direction = q;
            let step_size = self.line_search(objective, &x, &direction, &grad)?;

            // Update position
            for j in 0..n {
                x[j] += step_size * direction[j];
            }

            // Compute new gradient
            grad = objective.gradient(&x);

            // Update L-BFGS vectors
            let s = subtract(&x, &prev_x);
            let y = subtract(&grad, &prev_grad);
            let rho_k = 1.0 / dot_product(&y, &s);

            if self.s_vectors.len() == self.m {
                self.s_vectors.remove(0);
                self.y_vectors.remove(0);
                self.rho.remove(0);
            }
            self.s_vectors.push(s);
            self.y_vectors.push(y);
            self.rho.push(rho_k);

            // Check convergence
            let grad_norm = l2_norm(&grad);
            if grad_norm < self.epsilon {
                return Ok((x, objective.evaluate(&x)));
            }

            if iter == self.max_iter - 1 {
                return Err(OptimizationError::ConvergenceError(
                    "Maximum iterations reached".to_string(),
                ));
            }
        }

        Ok((x, objective.evaluate(&x)))
    }

    fn line_search<F: ObjectiveFunction>(
        &self,
        objective: &F,
        x: &[f64],
        direction: &[f64],
        grad: &[f64],
    ) -> Result<f64, OptimizationError> {
        let c1 = 1e-4;
        let c2 = 0.9;
        let mut alpha = 1.0;
        let mut alpha_prev = 0.0;
        let mut f_prev = objective.evaluate(x);
        let directional_derivative = dot_product(grad, direction);

        for _ in 0..20 {
            // Try current step size
            let mut x_new = x.to_vec();
            for j in 0..x.len() {
                x_new[j] += alpha * direction[j];
            }
            let f_new = objective.evaluate(&x_new);

            // Check Wolfe conditions
            if f_new <= f_prev + c1 * alpha * directional_derivative {
                let grad_new = objective.gradient(&x_new);
                let directional_derivative_new = dot_product(&grad_new, direction);
                if directional_derivative_new.abs() <= -c2 * directional_derivative {
                    return Ok(alpha);
                }
            }

            // Update step size using cubic interpolation
            if alpha_prev == 0.0 {
                alpha *= 0.5;
            } else {
                let alpha_cubic = self.cubic_interpolation(
                    alpha_prev,
                    f_prev,
                    directional_derivative,
                    alpha,
                    f_new,
                );
                alpha = alpha_cubic.max(0.1 * alpha).min(0.5 * alpha);
            }

            alpha_prev = alpha;
            f_prev = f_new;
        }

        Ok(alpha)
    }

    fn cubic_interpolation(
        &self,
        alpha0: f64,
        f0: f64,
        g0: f64,
        alpha1: f64,
        f1: f64,
    ) -> f64 {
        let d1 = g0 + f1 - f0;
        let d2 = d1 * d1 - g0 * alpha1;
        if d2.abs() < 1e-10 {
            return alpha1;
        }
        let d3 = alpha1 / (alpha0 * alpha0);
        (-d1 + (d2 * d3).sqrt()) / (3.0 * d3)
    }
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn l2_norm(x: &[f64]) -> f64 {
    dot_product(x, x).sqrt()
}

fn subtract(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

pub struct ParticleSwarm {
    n_particles: usize,
    max_iter: usize,
    c1: f64,
    c2: f64,
    w: f64,
    bounds: Vec<(f64, f64)>,
}

impl ParticleSwarm {
    pub fn new(
        n_particles: usize,
        max_iter: usize,
        c1: f64,
        c2: f64,
        w: f64,
        bounds: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            n_particles,
            max_iter,
            c1,
            c2,
            w,
            bounds,
        }
    }

    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
    ) -> Result<(Vec<f64>, f64), OptimizationError> {
        let dim = self.bounds.len();
        let mut rng = thread_rng();

        // Initialize particles
        let mut positions: Vec<Vec<f64>> = (0..self.n_particles)
            .map(|_| {
                self.bounds
                    .iter()
                    .map(|&(min, max)| rng.gen_range(min..max))
                    .collect()
            })
            .collect();

        let mut velocities: Vec<Vec<f64>> = (0..self.n_particles)
            .map(|_| {
                self.bounds
                    .iter()
                    .map(|&(min, max)| rng.gen_range(min - max..max - min))
                    .collect()
            })
            .collect();

        let mut personal_best_positions = positions.clone();
        let mut personal_best_values: Vec<f64> = positions
            .iter()
            .map(|pos| objective.evaluate(pos))
            .collect();

        let mut global_best_idx = personal_best_values
            .iter()
            .enumerate()
            .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        for _ in 0..self.max_iter {
            // Update particles in parallel
            positions
                .par_iter_mut()
                .zip(velocities.par_iter_mut())
                .enumerate()
                .for_each(|(i, (pos, vel))| {
                    let mut rng = thread_rng();
                    
                    // Update velocity
                    for j in 0..dim {
                        let r1 = rng.gen::<f64>();
                        let r2 = rng.gen::<f64>();
                        
                        vel[j] = self.w * vel[j]
                            + self.c1 * r1 * (personal_best_positions[i][j] - pos[j])
                            + self.c2 * r2 * (positions[global_best_idx][j] - pos[j]);
                    }

                    // Update position
                    for j in 0..dim {
                        pos[j] += vel[j];
                        pos[j] = pos[j].clamp(self.bounds[j].0, self.bounds[j].1);
                    }

                    // Update personal best
                    let value = objective.evaluate(pos);
                    if value < personal_best_values[i] {
                        personal_best_values[i] = value;
                        personal_best_positions[i] = pos.clone();
                    }
                });

            // Update global best
            global_best_idx = personal_best_values
                .iter()
                .enumerate()
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
        }

        Ok((
            positions[global_best_idx].clone(),
            personal_best_values[global_best_idx],
        ))
    }
}
