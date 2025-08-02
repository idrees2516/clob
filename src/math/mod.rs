//! Mathematical utilities for high-frequency quoting
//!
//! This module provides mathematical tools and algorithms for implementing
//! the models described in "High Frequency Quoting Under Liquidity Constraints".

pub mod fixed_point;
pub mod circuit_math;
pub mod hawkes_process;
pub mod hawkes_estimation;
pub mod order_flow_hawkes;
pub mod sde_solvers;
pub mod optimization;
pub mod rough_volatility;
pub mod jump_diffusion;

// Re-export commonly used types
pub use fixed_point::{
    FixedPoint, DeterministicRng, BoxMullerGenerator, PoissonJumpGenerator, QuasiMonteCarloGenerator,
    FixedPointPrecision, RoundingMode, PRNGAlgorithm, QMCSequenceType,
    FIXED_POINT_BITS, FIXED_POINT_SCALE
};
pub use circuit_math::{normal_cdf, normal_pdf, black_scholes, hawkes_intensity};
pub use hawkes_process::{KernelType, MultivariateHawkesParams, HawkesEvent, HawkesState};
pub use order_flow_hawkes::{
    OrderType, OrderFlowEvent, OrderFlowHawkesParams, OrderFlowIntensityCalculator,
    RealTimeOrderFlowAnalyzer, TradingSignals, DirectionalSignal, UrgencySignal,
    OrderFlowMetrics, QuoteRecommendations, OrderFlowAnalysisResults, IntensityTrend
};
pub use sde_solvers::{
    SDESolver, EulerMaruyamaGBMJump, MilsteinGBMJump, MonteCarloSimulator, PathStatistics, SDEError
};
pub use jump_diffusion::{
    GBMJumpState, GBMJumpParams, JumpEvent, JumpSizeDistribution, JumpDiffusionSimulator,
    BiPowerVariationJumpDetector, JumpClusteringDetector, JumpDiffusionError
};
pub use optimization::{OptimizationType, Constraint, OptimizationResult, GradientDescentOptimizer, HJBSolver};
pub use rough_volatility::{
    FractionalBrownianMotion, RoughVolatilitySimulator, VolatilityClusteringDetector, 
    HurstEstimator, RoughVolatilityError, RoughVolatilityState, RoughVolatilityParams
};
pub use hawkes_estimation::{
    LBFGSOptimizer, HawkesMLEstimator, HawkesEMEstimator, HawkesCrossValidator,
    HawkesGoodnessOfFitTester, ValidationMetric, CrossValidationResult, GoodnessOfFitResult,
    StatisticalTest, ResidualAnalysis, EstimationError
};