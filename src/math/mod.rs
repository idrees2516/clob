//! Mathematical utilities for high-frequency quoting
//!
//! This module provides mathematical tools and algorithms for implementing
//! the models described in "High Frequency Quoting Under Liquidity Constraints".

pub mod fixed_point;
pub mod circuit_math;
pub mod hawkes_process;
pub mod sde_solvers;
pub mod optimization;
pub mod rough_volatility;

// Re-export commonly used types
pub use fixed_point::{FixedPoint, DeterministicRng, FIXED_POINT_BITS, FIXED_POINT_SCALE};
pub use circuit_math::{normal_cdf, normal_pdf, black_scholes, hawkes_intensity};
pub use hawkes_process::{KernelType, MultivariateHawkesParams, HawkesEvent, HawkesState};
pub use sde_solvers::{SDESolver, GBMJumpState, GBMJumpParams, RoughVolatilityState, RoughVolatilityParams};
pub use optimization::{OptimizationType, Constraint, OptimizationResult, GradientDescentOptimizer, HJBSolver};
pub use rough_volatility::{
    FractionalBrownianMotion, RoughVolatilitySimulator, VolatilityClusteringDetector, 
    HurstEstimator, RoughVolatilityError
};