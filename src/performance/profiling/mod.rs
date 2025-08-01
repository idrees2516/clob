pub mod continuous_profiler;
pub mod flame_graph;
pub mod sampling_profiler;
pub mod regression_detector;
pub mod profile_analyzer;

pub use continuous_profiler::*;
pub use flame_graph::*;
pub use sampling_profiler::*;
pub use regression_detector::*;
pub use profile_analyzer::*;

// Re-export key types for convenience
pub use continuous_profiler::{ContinuousProfiler, ProfilerConfig, ProfilerStats, global_profiler};
pub use flame_graph::{FlameGraph, FlameGraphNode, FlameGraphBuilder};
pub use sampling_profiler::{SamplingProfiler, ProfileSample, SamplingConfig};
pub use regression_detector::{RegressionDetector, PerformanceRegression, RegressionType};
pub use profile_analyzer::{ProfileAnalyzer, ProfileAnalysis, PerformanceInsight};