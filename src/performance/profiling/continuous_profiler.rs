use super::timing::{now_nanos, NanosecondTimer, LatencyHistogram};
use super::sampling_profiler::{SamplingProfiler, ProfileSample, SamplingConfig};
use super::flame_graph::{FlameGraph, FlameGraphBuilder};
use super::regression_detector::{RegressionDetector, PerformanceRegression};
use super::profile_analyzer::{ProfileAnalyzer, ProfileAnalysis};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::thread;
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Continuous profiling system for real-time performance monitoring
/// Provides sampling-based profiling with minimal overhead
pub struct ContinuousProfiler {
    /// Profiler configuration
    config: ProfilerConfig,
    
    /// Profiler state
    state: ProfilerState,
    
    /// Sampling profiler
    sampler: Arc<SamplingProfiler>,
    
    /// Flame graph builder
    flame_graph_builder: Arc<FlameGraphBuilder>,
    
    /// Regression detector
    regression_detector: Arc<RegressionDetector>,
    
    /// Profile analyzer
    analyzer: Arc<ProfileAnalyzer>,
    
    /// Profiling thread handle
    profiler_thread: Option<std::thread::JoinHandle<()>>,
    
    /// Profile storage
    profiles: std::sync::RwLock<VecDeque<ProfileSnapshot>>,
    
    /// Performance metrics
    metrics: ProfilerMetrics,
}

/// Profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Sampling interval (nanoseconds)
    pub sampling_interval_ns: u64,
    
    /// Profile collection interval (nanoseconds)
    pub collection_interval_ns: u64,
    
    /// Maximum number of stored profiles
    pub max_stored_profiles: usize,
    
    /// Enable flame graph generation
    pub enable_flame_graphs: bool,
    
    /// Enable regression detection
    pub enable_regression_detection: bool,
    
    /// Enable automatic analysis
    pub enable_auto_analysis: bool,
    
    /// Profiling overhead limit (percentage)
    pub max_overhead_percent: f64,
    
    /// Functions to include in profiling
    pub include_functions: Vec<String>,
    
    /// Functions to exclude from profiling
    pub exclude_functions: Vec<String>,
    
    /// Minimum sample count for analysis
    pub min_samples_for_analysis: u32,
    
    /// Profile retention time (nanoseconds)
    pub profile_retention_ns: u64,
}

/// Profiler state
#[repr(align(64))]
struct ProfilerState {
    /// Whether profiler is running
    is_running: AtomicBool,
    
    /// Total samples collected
    total_samples: AtomicU64,
    
    /// Last profile collection time
    last_collection: AtomicU64,
    
    /// Profiling overhead (nanoseconds)
    profiling_overhead_ns: AtomicU64,
    
    /// Active profiling sessions
    active_sessions: AtomicUsize,
    
    /// Profile generation counter
    profile_counter: AtomicU64,
}

/// Profile snapshot
#[derive(Debug, Clone)]
pub struct ProfileSnapshot {
    /// Profile ID
    pub id: String,
    
    /// Collection timestamp
    pub timestamp: u64,
    
    /// Collection duration (nanoseconds)
    pub duration_ns: u64,
    
    /// Number of samples in this profile
    pub sample_count: u32,
    
    /// Flame graph (if enabled)
    pub flame_graph: Option<FlameGraph>,
    
    /// Profile analysis
    pub analysis: Option<ProfileAnalysis>,
    
    /// Detected regressions
    pub regressions: Vec<PerformanceRegression>,
    
    /// Profile metadata
    pub metadata: HashMap<String, String>,
    
    /// Raw samples
    pub samples: Vec<ProfileSample>,
}

/// Profiler performance metrics
#[repr(align(64))]
struct ProfilerMetrics {
    /// Profiling latency histogram
    profiling_latency: LatencyHistogram,
    
    /// Analysis latency histogram
    analysis_latency: LatencyHistogram,
    
    /// Flame graph generation latency
    flame_graph_latency: LatencyHistogram,
    
    /// Total profiles generated
    total_profiles: AtomicU64,
    
    /// Total regressions detected
    total_regressions: AtomicU64,
    
    /// Average overhead percentage
    avg_overhead_percent: AtomicU64,
}

/// Profiler statistics
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    pub is_running: bool,
    pub total_samples: u64,
    pub total_profiles: u64,
    pub total_regressions: u64,
    pub last_collection_ns: u64,
    pub avg_profiling_latency_ns: u64,
    pub avg_analysis_latency_ns: u64,
    pub avg_overhead_percent: f64,
    pub active_sessions: usize,
    pub stored_profiles: usize,
}

impl ContinuousProfiler {
    /// Create a new continuous profiler
    pub fn new(config: ProfilerConfig) -> Self {
        let sampling_config = SamplingConfig {
            sampling_interval_ns: config.sampling_interval_ns,
            max_stack_depth: 64,
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_io_profiling: false,
            sample_buffer_size: 10000,
        };

        Self {
            config: config.clone(),
            state: ProfilerState {
                is_running: AtomicBool::new(false),
                total_samples: AtomicU64::new(0),
                last_collection: AtomicU64::new(0),
                profiling_overhead_ns: AtomicU64::new(0),
                active_sessions: AtomicUsize::new(0),
                profile_counter: AtomicU64::new(0),
            },
            sampler: Arc::new(SamplingProfiler::new(sampling_config)),
            flame_graph_builder: Arc::new(FlameGraphBuilder::new()),
            regression_detector: Arc::new(RegressionDetector::new()),
            analyzer: Arc::new(ProfileAnalyzer::new()),
            profiler_thread: None,
            profiles: std::sync::RwLock::new(VecDeque::new()),
            metrics: ProfilerMetrics {
                profiling_latency: LatencyHistogram::for_trading(),
                analysis_latency: LatencyHistogram::for_trading(),
                flame_graph_latency: LatencyHistogram::for_trading(),
                total_profiles: AtomicU64::new(0),
                total_regressions: AtomicU64::new(0),
                avg_overhead_percent: AtomicU64::new(0),
            },
        }
    }

    /// Create profiler with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Start continuous profiling
    pub fn start(&mut self) -> Result<(), ProfilerError> {
        if self.state.is_running.load(Ordering::Acquire) {
            return Err(ProfilerError::AlreadyRunning);
        }

        self.state.is_running.store(true, Ordering::Release);
        
        // Start sampling profiler
        self.sampler.start()?;
        
        // Clone necessary data for profiling thread
        let config = self.config.clone();
        let sampler = self.sampler.clone();
        let flame_graph_builder = self.flame_graph_builder.clone();
        let regression_detector = self.regression_detector.clone();
        let analyzer = self.analyzer.clone();
        let is_running = Arc::new(AtomicBool::new(true));
        let is_running_clone = is_running.clone();
        
        // Start profiling thread
        let handle = thread::spawn(move || {
            Self::profiling_loop(
                config,
                sampler,
                flame_graph_builder,
                regression_detector,
                analyzer,
                is_running_clone,
            );
        });

        self.profiler_thread = Some(handle);
        Ok(())
    }

    /// Stop continuous profiling
    pub fn stop(&mut self) -> Result<(), ProfilerError> {
        if !self.state.is_running.load(Ordering::Acquire) {
            return Err(ProfilerError::NotRunning);
        }

        self.state.is_running.store(false, Ordering::Release);
        
        // Stop sampling profiler
        self.sampler.stop()?;
        
        // Wait for profiling thread to finish
        if let Some(handle) = self.profiler_thread.take() {
            handle.join().map_err(|_| ProfilerError::ThreadJoinError)?;
        }

        Ok(())
    }

    /// Get current profiler statistics
    pub fn get_stats(&self) -> ProfilerStats {
        let profiles = self.profiles.read().unwrap();
        
        ProfilerStats {
            is_running: self.state.is_running.load(Ordering::Acquire),
            total_samples: self.state.total_samples.load(Ordering::Acquire),
            total_profiles: self.metrics.total_profiles.load(Ordering::Acquire),
            total_regressions: self.metrics.total_regressions.load(Ordering::Acquire),
            last_collection_ns: self.state.last_collection.load(Ordering::Acquire),
            avg_profiling_latency_ns: self.metrics.profiling_latency.get_stats().average as u64,
            avg_analysis_latency_ns: self.metrics.analysis_latency.get_stats().average as u64,
            avg_overhead_percent: self.metrics.avg_overhead_percent.load(Ordering::Acquire) as f64 / 100.0,
            active_sessions: self.state.active_sessions.load(Ordering::Acquire),
            stored_profiles: profiles.len(),
        }
    }

    /// Get recent profiles
    pub fn get_recent_profiles(&self, limit: Option<usize>) -> Vec<ProfileSnapshot> {
        let profiles = self.profiles.read().unwrap();
        let limit = limit.unwrap_or(profiles.len());
        
        profiles.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get profile by ID
    pub fn get_profile(&self, profile_id: &str) -> Option<ProfileSnapshot> {
        let profiles = self.profiles.read().unwrap();
        profiles.iter()
            .find(|p| p.id == profile_id)
            .cloned()
    }

    /// Generate flame graph for current samples
    pub fn generate_flame_graph(&self) -> Result<FlameGraph, ProfilerError> {
        let samples = self.sampler.get_recent_samples(1000);
        
        let generation_start = now_nanos();
        let flame_graph = self.flame_graph_builder.build_from_samples(&samples)?;
        let generation_latency = now_nanos() - generation_start;
        
        self.metrics.flame_graph_latency.record(generation_latency);
        
        Ok(flame_graph)
    }

    /// Analyze current performance
    pub fn analyze_performance(&self) -> Result<ProfileAnalysis, ProfilerError> {
        let samples = self.sampler.get_recent_samples(5000);
        
        let analysis_start = now_nanos();
        let analysis = self.analyzer.analyze_samples(&samples)?;
        let analysis_latency = now_nanos() - analysis_start;
        
        self.metrics.analysis_latency.record(analysis_latency);
        
        Ok(analysis)
    }

    /// Detect performance regressions
    pub fn detect_regressions(&self) -> Vec<PerformanceRegression> {
        let profiles = self.profiles.read().unwrap();
        let recent_profiles: Vec<&ProfileSnapshot> = profiles.iter().rev().take(10).collect();
        
        if recent_profiles.len() < 2 {
            return Vec::new();
        }

        self.regression_detector.detect_regressions(&recent_profiles)
    }

    /// Clear old profiles
    pub fn cleanup_old_profiles(&self) {
        let current_time = now_nanos();
        let mut profiles = self.profiles.write().unwrap();
        
        profiles.retain(|profile| {
            let age = current_time - profile.timestamp;
            age < self.config.profile_retention_ns
        });
        
        // Also enforce maximum count
        while profiles.len() > self.config.max_stored_profiles {
            profiles.pop_front();
        }
    }

    /// Main profiling loop
    fn profiling_loop(
        config: ProfilerConfig,
        sampler: Arc<SamplingProfiler>,
        flame_graph_builder: Arc<FlameGraphBuilder>,
        regression_detector: Arc<RegressionDetector>,
        analyzer: Arc<ProfileAnalyzer>,
        is_running: Arc<AtomicBool>,
    ) {
        let collection_interval = Duration::from_nanos(config.collection_interval_ns);
        
        while is_running.load(Ordering::Acquire) {
            let collection_start = now_nanos();
            
            // Collect samples
            let samples = sampler.get_recent_samples(10000);
            
            if samples.len() >= config.min_samples_for_analysis as usize {
                // Generate profile snapshot
                let profile_id = format!("profile_{}", collection_start);
                let mut snapshot = ProfileSnapshot {
                    id: profile_id,
                    timestamp: collection_start,
                    duration_ns: 0,
                    sample_count: samples.len() as u32,
                    flame_graph: None,
                    analysis: None,
                    regressions: Vec::new(),
                    metadata: HashMap::new(),
                    samples: samples.clone(),
                };
                
                // Generate flame graph if enabled
                if config.enable_flame_graphs {
                    if let Ok(flame_graph) = flame_graph_builder.build_from_samples(&samples) {
                        snapshot.flame_graph = Some(flame_graph);
                    }
                }
                
                // Perform analysis if enabled
                if config.enable_auto_analysis {
                    if let Ok(analysis) = analyzer.analyze_samples(&samples) {
                        snapshot.analysis = Some(analysis);
                    }
                }
                
                // Detect regressions if enabled
                if config.enable_regression_detection {
                    // Note: This would need access to previous profiles for comparison
                    // For now, we'll leave this empty
                }
                
                snapshot.duration_ns = now_nanos() - collection_start;
                
                // Store profile (this would need to be done through a channel or shared state)
                // For now, we'll just simulate the storage
            }
            
            thread::sleep(collection_interval);
        }
    }

    /// Store profile snapshot
    fn store_profile(&self, profile: ProfileSnapshot) {
        let mut profiles = self.profiles.write().unwrap();
        
        profiles.push_back(profile);
        
        // Maintain size limit
        while profiles.len() > self.config.max_stored_profiles {
            profiles.pop_front();
        }
        
        self.metrics.total_profiles.fetch_add(1, Ordering::Relaxed);
    }

    /// Calculate profiling overhead
    fn calculate_overhead(&self, profiling_time: u64, total_time: u64) -> f64 {
        if total_time == 0 {
            return 0.0;
        }
        
        (profiling_time as f64 / total_time as f64) * 100.0
    }
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            sampling_interval_ns: 10_000_000, // 10ms
            collection_interval_ns: 60_000_000_000, // 1 minute
            max_stored_profiles: 1000,
            enable_flame_graphs: true,
            enable_regression_detection: true,
            enable_auto_analysis: true,
            max_overhead_percent: 5.0,
            include_functions: Vec::new(),
            exclude_functions: vec![
                "std::".to_string(),
                "__rust_".to_string(),
                "core::".to_string(),
            ],
            min_samples_for_analysis: 100,
            profile_retention_ns: 86400_000_000_000, // 24 hours
        }
    }
}

/// Profiler errors
#[derive(Debug, Clone)]
pub enum ProfilerError {
    AlreadyRunning,
    NotRunning,
    ThreadJoinError,
    SamplingError,
    AnalysisError,
    FlameGraphError,
    InsufficientSamples,
}

impl std::fmt::Display for ProfilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProfilerError::AlreadyRunning => write!(f, "Profiler is already running"),
            ProfilerError::NotRunning => write!(f, "Profiler is not running"),
            ProfilerError::ThreadJoinError => write!(f, "Failed to join profiler thread"),
            ProfilerError::SamplingError => write!(f, "Sampling error occurred"),
            ProfilerError::AnalysisError => write!(f, "Analysis error occurred"),
            ProfilerError::FlameGraphError => write!(f, "Flame graph generation error"),
            ProfilerError::InsufficientSamples => write!(f, "Insufficient samples for analysis"),
        }
    }
}

impl std::error::Error for ProfilerError {}

/// Global profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<Arc<std::sync::Mutex<ContinuousProfiler>>> = std::sync::OnceLock::new();

/// Get global profiler instance
pub fn global_profiler() -> &'static Arc<std::sync::Mutex<ContinuousProfiler>> {
    GLOBAL_PROFILER.get_or_init(|| {
        Arc::new(std::sync::Mutex::new(ContinuousProfiler::with_defaults()))
    })
}

/// Convenience function to start profiling
pub fn start_profiling() -> Result<(), ProfilerError> {
    let profiler = global_profiler();
    let mut profiler = profiler.lock().unwrap();
    profiler.start()
}

/// Convenience function to stop profiling
pub fn stop_profiling() -> Result<(), ProfilerError> {
    let profiler = global_profiler();
    let mut profiler = profiler.lock().unwrap();
    profiler.stop()
}

/// Convenience function to get profiler stats
pub fn get_profiler_stats() -> ProfilerStats {
    let profiler = global_profiler();
    let profiler = profiler.lock().unwrap();
    profiler.get_stats()
}

/// Profiling session for scoped profiling
pub struct ProfilingSession {
    session_id: String,
    start_time: u64,
    samples: Vec<ProfileSample>,
}

impl ProfilingSession {
    /// Start a new profiling session
    pub fn start(session_id: String) -> Self {
        Self {
            session_id,
            start_time: now_nanos(),
            samples: Vec::new(),
        }
    }

    /// Add sample to session
    pub fn add_sample(&mut self, sample: ProfileSample) {
        self.samples.push(sample);
    }

    /// Finish session and return profile
    pub fn finish(self) -> ProfileSnapshot {
        let duration = now_nanos() - self.start_time;
        
        ProfileSnapshot {
            id: self.session_id,
            timestamp: self.start_time,
            duration_ns: duration,
            sample_count: self.samples.len() as u32,
            flame_graph: None,
            analysis: None,
            regressions: Vec::new(),
            metadata: HashMap::new(),
            samples: self.samples,
        }
    }
}

/// Macro for profiling a code block
#[macro_export]
macro_rules! profile_block {
    ($name:expr, $code:block) => {{
        let _session = ProfilingSession::start($name.to_string());
        let start_time = now_nanos();
        let result = $code;
        let end_time = now_nanos();
        
        // Record sample (simplified)
        let sample = ProfileSample {
            timestamp: start_time,
            duration_ns: end_time - start_time,
            function_name: $name.to_string(),
            stack_trace: vec![$name.to_string()],
            cpu_usage: 0.0,
            memory_usage: 0,
            thread_id: std::thread::current().id().as_u64().get(),
        };
        
        // Add to global profiler if running
        // (This would need proper integration)
        
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = ContinuousProfiler::with_defaults();
        let stats = profiler.get_stats();
        
        assert!(!stats.is_running);
        assert_eq!(stats.total_samples, 0);
        assert_eq!(stats.total_profiles, 0);
    }

    #[test]
    fn test_profiler_config() {
        let config = ProfilerConfig {
            sampling_interval_ns: 5_000_000, // 5ms
            collection_interval_ns: 30_000_000_000, // 30 seconds
            max_stored_profiles: 500,
            enable_flame_graphs: false,
            enable_regression_detection: false,
            enable_auto_analysis: false,
            max_overhead_percent: 2.0,
            include_functions: vec!["trading::".to_string()],
            exclude_functions: vec!["std::".to_string()],
            min_samples_for_analysis: 50,
            profile_retention_ns: 43200_000_000_000, // 12 hours
        };
        
        let profiler = ContinuousProfiler::new(config.clone());
        assert_eq!(profiler.config.sampling_interval_ns, config.sampling_interval_ns);
        assert_eq!(profiler.config.max_stored_profiles, config.max_stored_profiles);
    }

    #[test]
    fn test_profiling_session() {
        let mut session = ProfilingSession::start("test_session".to_string());
        
        let sample = ProfileSample {
            timestamp: now_nanos(),
            duration_ns: 1000000,
            function_name: "test_function".to_string(),
            stack_trace: vec!["test_function".to_string()],
            cpu_usage: 50.0,
            memory_usage: 1024,
            thread_id: 1,
        };
        
        session.add_sample(sample);
        
        let profile = session.finish();
        assert_eq!(profile.id, "test_session");
        assert_eq!(profile.sample_count, 1);
        assert!(profile.duration_ns > 0);
    }

    #[test]
    fn test_global_profiler() {
        let profiler1 = global_profiler();
        let profiler2 = global_profiler();
        
        // Should be the same instance
        assert!(Arc::ptr_eq(profiler1, profiler2));
    }

    #[test]
    fn test_profile_cleanup() {
        let mut profiler = ContinuousProfiler::with_defaults();
        
        // Add some old profiles
        for i in 0..10 {
            let profile = ProfileSnapshot {
                id: format!("profile_{}", i),
                timestamp: now_nanos() - (i as u64 * 1_000_000_000), // 1 second apart
                duration_ns: 1000000,
                sample_count: 100,
                flame_graph: None,
                analysis: None,
                regressions: Vec::new(),
                metadata: HashMap::new(),
                samples: Vec::new(),
            };
            
            profiler.store_profile(profile);
        }
        
        let stats_before = profiler.get_stats();
        assert_eq!(stats_before.stored_profiles, 10);
        
        profiler.cleanup_old_profiles();
        
        let stats_after = profiler.get_stats();
        // All profiles should still be there since they're not old enough
        assert_eq!(stats_after.stored_profiles, 10);
    }

    #[test]
    fn test_overhead_calculation() {
        let profiler = ContinuousProfiler::with_defaults();
        
        let overhead = profiler.calculate_overhead(1_000_000, 100_000_000); // 1ms out of 100ms
        assert_eq!(overhead, 1.0);
        
        let zero_overhead = profiler.calculate_overhead(0, 100_000_000);
        assert_eq!(zero_overhead, 0.0);
        
        let no_time_overhead = profiler.calculate_overhead(1_000_000, 0);
        assert_eq!(no_time_overhead, 0.0);
    }

    #[test]
    fn test_profile_storage_limits() {
        let config = ProfilerConfig {
            max_stored_profiles: 3,
            ..Default::default()
        };
        
        let profiler = ContinuousProfiler::new(config);
        
        // Add more profiles than the limit
        for i in 0..5 {
            let profile = ProfileSnapshot {
                id: format!("profile_{}", i),
                timestamp: now_nanos(),
                duration_ns: 1000000,
                sample_count: 100,
                flame_graph: None,
                analysis: None,
                regressions: Vec::new(),
                metadata: HashMap::new(),
                samples: Vec::new(),
            };
            
            profiler.store_profile(profile);
        }
        
        let stats = profiler.get_stats();
        assert_eq!(stats.stored_profiles, 3); // Should be limited to max
        
        let profiles = profiler.get_recent_profiles(None);
        assert_eq!(profiles.len(), 3);
        
        // Should have the most recent profiles
        assert_eq!(profiles[0].id, "profile_4");
        assert_eq!(profiles[1].id, "profile_3");
        assert_eq!(profiles[2].id, "profile_2");
    }
}