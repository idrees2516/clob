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

/// Hardware performance counter monitoring
pub struct HardwarePerformanceCounters {
    /// CPU cycle counter
    cpu_cycles: AtomicU64,
    /// Instructions retired
    instructions_retired: AtomicU64,
    /// Cache misses
    cache_misses: AtomicU64,
    /// Branch mispredictions
    branch_mispredictions: AtomicU64,
    /// TLB misses
    tlb_misses: AtomicU64,
    /// Memory stall cycles
    memory_stall_cycles: AtomicU64,
    /// Last measurement timestamp
    last_measurement: AtomicU64,
    /// Measurement interval (nanoseconds)
    measurement_interval_ns: u64,
}

impl HardwarePerformanceCounters {
    pub fn new(measurement_interval_ns: u64) -> Self {
        Self {
            cpu_cycles: AtomicU64::new(0),
            instructions_retired: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            branch_mispredictions: AtomicU64::new(0),
            tlb_misses: AtomicU64::new(0),
            memory_stall_cycles: AtomicU64::new(0),
            last_measurement: AtomicU64::new(0),
            measurement_interval_ns,
        }
    }

    /// Read hardware performance counters
    #[cfg(target_arch = "x86_64")]
    pub fn read_counters(&self) -> HardwareCounterSnapshot {
        let current_time = now_nanos();
        let last_time = self.last_measurement.load(Ordering::Acquire);
        
        // Only read if enough time has passed
        if current_time - last_time < self.measurement_interval_ns {
            return self.get_cached_snapshot();
        }

        unsafe {
            // Read CPU cycles using RDTSC
            let cpu_cycles = std::arch::x86_64::_rdtsc();
            
            // Read performance monitoring counters (PMCs)
            // Note: This requires kernel support and proper setup
            let instructions = self.read_pmc(0); // PMC0 typically instructions
            let cache_misses = self.read_pmc(1); // PMC1 typically cache misses
            let branch_misses = self.read_pmc(2); // PMC2 typically branch mispredictions
            
            // Update atomic counters
            self.cpu_cycles.store(cpu_cycles, Ordering::Release);
            self.instructions_retired.store(instructions, Ordering::Release);
            self.cache_misses.store(cache_misses, Ordering::Release);
            self.branch_mispredictions.store(branch_misses, Ordering::Release);
            self.last_measurement.store(current_time, Ordering::Release);
            
            HardwareCounterSnapshot {
                timestamp: current_time,
                cpu_cycles,
                instructions_retired: instructions,
                cache_misses,
                branch_mispredictions: branch_misses,
                tlb_misses: 0, // Would need additional PMC setup
                memory_stall_cycles: 0, // Would need additional PMC setup
                ipc: if cpu_cycles > 0 { instructions as f64 / cpu_cycles as f64 } else { 0.0 },
                cache_miss_rate: if instructions > 0 { cache_misses as f64 / instructions as f64 } else { 0.0 },
                branch_miss_rate: if instructions > 0 { branch_misses as f64 / instructions as f64 } else { 0.0 },
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn read_counters(&self) -> HardwareCounterSnapshot {
        // Fallback for non-x86_64 architectures
        HardwareCounterSnapshot {
            timestamp: now_nanos(),
            cpu_cycles: 0,
            instructions_retired: 0,
            cache_misses: 0,
            branch_mispredictions: 0,
            tlb_misses: 0,
            memory_stall_cycles: 0,
            ipc: 0.0,
            cache_miss_rate: 0.0,
            branch_miss_rate: 0.0,
        }
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn read_pmc(&self, counter: u32) -> u64 {
        // Read Performance Monitoring Counter
        // Note: This requires proper kernel setup and permissions
        // For now, we'll return a simulated value
        match counter {
            0 => std::arch::x86_64::_rdtsc() / 2, // Simulate instructions
            1 => std::arch::x86_64::_rdtsc() / 100, // Simulate cache misses
            2 => std::arch::x86_64::_rdtsc() / 1000, // Simulate branch misses
            _ => 0,
        }
    }

    fn get_cached_snapshot(&self) -> HardwareCounterSnapshot {
        let timestamp = self.last_measurement.load(Ordering::Acquire);
        let cpu_cycles = self.cpu_cycles.load(Ordering::Acquire);
        let instructions = self.instructions_retired.load(Ordering::Acquire);
        let cache_misses = self.cache_misses.load(Ordering::Acquire);
        let branch_misses = self.branch_mispredictions.load(Ordering::Acquire);
        
        HardwareCounterSnapshot {
            timestamp,
            cpu_cycles,
            instructions_retired: instructions,
            cache_misses,
            branch_mispredictions: branch_misses,
            tlb_misses: self.tlb_misses.load(Ordering::Acquire),
            memory_stall_cycles: self.memory_stall_cycles.load(Ordering::Acquire),
            ipc: if cpu_cycles > 0 { instructions as f64 / cpu_cycles as f64 } else { 0.0 },
            cache_miss_rate: if instructions > 0 { cache_misses as f64 / instructions as f64 } else { 0.0 },
            branch_miss_rate: if instructions > 0 { branch_misses as f64 / instructions as f64 } else { 0.0 },
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareCounterSnapshot {
    pub timestamp: u64,
    pub cpu_cycles: u64,
    pub instructions_retired: u64,
    pub cache_misses: u64,
    pub branch_mispredictions: u64,
    pub tlb_misses: u64,
    pub memory_stall_cycles: u64,
    pub ipc: f64, // Instructions per cycle
    pub cache_miss_rate: f64,
    pub branch_miss_rate: f64,
}

/// Advanced regression detection with statistical analysis
pub struct AdvancedRegressionDetector {
    /// Historical performance baselines
    baselines: HashMap<String, PerformanceBaseline>,
    /// Statistical thresholds
    thresholds: RegressionThresholds,
    /// Detection algorithms
    algorithms: Vec<Box<dyn RegressionAlgorithm>>,
    /// Alert history
    alert_history: VecDeque<RegressionAlert>,
}

#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub metric_name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<u8, f64>, // 50th, 90th, 95th, 99th percentiles
    pub sample_count: u64,
    pub last_updated: u64,
    pub trend: TrendDirection,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RegressionThresholds {
    /// Percentage increase that triggers a warning
    pub warning_threshold_percent: f64,
    /// Percentage increase that triggers an alert
    pub alert_threshold_percent: f64,
    /// Number of consecutive violations before alerting
    pub consecutive_violations: u32,
    /// Minimum sample size for statistical significance
    pub min_sample_size: u32,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
}

#[derive(Debug, Clone)]
pub struct RegressionAlert {
    pub alert_id: String,
    pub timestamp: u64,
    pub metric_name: String,
    pub severity: AlertSeverity,
    pub current_value: f64,
    pub baseline_value: f64,
    pub percentage_change: f64,
    pub confidence: f64,
    pub description: String,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

pub trait RegressionAlgorithm: Send + Sync {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline,
        current_samples: &[f64],
    ) -> Option<RegressionAlert>;
    
    fn algorithm_name(&self) -> &str;
}

/// Statistical t-test based regression detection
pub struct TTestRegressionDetector {
    thresholds: RegressionThresholds,
}

impl TTestRegressionDetector {
    pub fn new(thresholds: RegressionThresholds) -> Self {
        Self { thresholds }
    }
}

impl RegressionAlgorithm for TTestRegressionDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline,
        current_samples: &[f64],
    ) -> Option<RegressionAlert> {
        if current_samples.len() < self.thresholds.min_sample_size as usize {
            return None;
        }

        let current_mean = current_samples.iter().sum::<f64>() / current_samples.len() as f64;
        let current_variance = current_samples.iter()
            .map(|x| (x - current_mean).powi(2))
            .sum::<f64>() / (current_samples.len() - 1) as f64;
        let current_std_dev = current_variance.sqrt();

        // Perform Welch's t-test
        let pooled_std_error = (
            (baseline.std_dev.powi(2) / baseline.sample_count as f64) +
            (current_variance / current_samples.len() as f64)
        ).sqrt();

        let t_statistic = (current_mean - baseline.mean) / pooled_std_error;
        
        // Calculate degrees of freedom (Welch-Satterthwaite equation)
        let df = (
            (baseline.std_dev.powi(2) / baseline.sample_count as f64 + 
             current_variance / current_samples.len() as f64).powi(2)
        ) / (
            (baseline.std_dev.powi(2) / baseline.sample_count as f64).powi(2) / (baseline.sample_count - 1) as f64 +
            (current_variance / current_samples.len() as f64).powi(2) / (current_samples.len() - 1) as f64
        );

        // Critical value for two-tailed test (simplified)
        let critical_value = match self.thresholds.confidence_level {
            0.95 => 1.96,
            0.99 => 2.58,
            0.999 => 3.29,
            _ => 1.96,
        };

        let percentage_change = ((current_mean - baseline.mean) / baseline.mean) * 100.0;
        
        if t_statistic.abs() > critical_value && percentage_change > self.thresholds.warning_threshold_percent {
            let severity = if percentage_change > self.thresholds.alert_threshold_percent {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(RegressionAlert {
                alert_id: format!("ttest_{}_{}", baseline.metric_name, now_nanos()),
                timestamp: now_nanos(),
                metric_name: baseline.metric_name.clone(),
                severity,
                current_value: current_mean,
                baseline_value: baseline.mean,
                percentage_change,
                confidence: self.thresholds.confidence_level,
                description: format!(
                    "Statistical regression detected: {} increased by {:.2}% (t-statistic: {:.3})",
                    baseline.metric_name, percentage_change, t_statistic
                ),
                suggested_actions: vec![
                    "Review recent code changes".to_string(),
                    "Check system resource usage".to_string(),
                    "Analyze performance profiles".to_string(),
                ],
            })
        } else {
            None
        }
    }

    fn algorithm_name(&self) -> &str {
        "t-test"
    }
}

/// Mann-Whitney U test for non-parametric regression detection
pub struct MannWhitneyRegressionDetector {
    thresholds: RegressionThresholds,
}

impl MannWhitneyRegressionDetector {
    pub fn new(thresholds: RegressionThresholds) -> Self {
        Self { thresholds }
    }

    fn mann_whitney_u_test(&self, baseline_samples: &[f64], current_samples: &[f64]) -> (f64, f64) {
        let n1 = baseline_samples.len();
        let n2 = current_samples.len();
        
        // Combine and rank all samples
        let mut combined: Vec<(f64, usize)> = baseline_samples.iter()
            .map(|&x| (x, 0))
            .chain(current_samples.iter().map(|&x| (x, 1)))
            .collect();
        
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Calculate ranks
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }
        
        // Sum ranks for current samples
        let r2: f64 = combined.iter()
            .zip(ranks.iter())
            .filter(|((_, group), _)| *group == 1)
            .map(|(_, &rank)| rank)
            .sum();
        
        let u2 = r2 - (n2 * (n2 + 1)) as f64 / 2.0;
        let u1 = (n1 * n2) as f64 - u2;
        
        let u = u1.min(u2);
        
        // Calculate z-score for large samples
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
        let z_score = (u - mean_u) / std_u;
        
        (u, z_score)
    }
}

impl RegressionAlgorithm for MannWhitneyRegressionDetector {
    fn detect_regression(
        &self,
        baseline: &PerformanceBaseline,
        current_samples: &[f64],
    ) -> Option<RegressionAlert> {
        if current_samples.len() < self.thresholds.min_sample_size as usize {
            return None;
        }

        // For this example, we'll use the baseline mean as a proxy for baseline samples
        // In practice, you'd store actual baseline samples
        let baseline_samples: Vec<f64> = (0..baseline.sample_count.min(1000))
            .map(|_| baseline.mean + (rand::random::<f64>() - 0.5) * baseline.std_dev * 2.0)
            .collect();

        let (_, z_score) = self.mann_whitney_u_test(&baseline_samples, current_samples);
        
        let current_mean = current_samples.iter().sum::<f64>() / current_samples.len() as f64;
        let percentage_change = ((current_mean - baseline.mean) / baseline.mean) * 100.0;
        
        // Critical value for Mann-Whitney U test (two-tailed)
        let critical_z = match self.thresholds.confidence_level {
            0.95 => 1.96,
            0.99 => 2.58,
            0.999 => 3.29,
            _ => 1.96,
        };
        
        if z_score.abs() > critical_z && percentage_change > self.thresholds.warning_threshold_percent {
            let severity = if percentage_change > self.thresholds.alert_threshold_percent {
                AlertSeverity::Critical
            } else {
                AlertSeverity::Warning
            };

            Some(RegressionAlert {
                alert_id: format!("mannwhitney_{}_{}", baseline.metric_name, now_nanos()),
                timestamp: now_nanos(),
                metric_name: baseline.metric_name.clone(),
                severity,
                current_value: current_mean,
                baseline_value: baseline.mean,
                percentage_change,
                confidence: self.thresholds.confidence_level,
                description: format!(
                    "Non-parametric regression detected: {} changed by {:.2}% (z-score: {:.3})",
                    baseline.metric_name, percentage_change, z_score
                ),
                suggested_actions: vec![
                    "Investigate performance anomalies".to_string(),
                    "Check for system configuration changes".to_string(),
                    "Review recent deployments".to_string(),
                ],
            })
        } else {
            None
        }
    }

    fn algorithm_name(&self) -> &str {
        "mann-whitney-u"
    }
}

impl AdvancedRegressionDetector {
    pub fn new() -> Self {
        let thresholds = RegressionThresholds {
            warning_threshold_percent: 10.0,
            alert_threshold_percent: 25.0,
            consecutive_violations: 3,
            min_sample_size: 30,
            confidence_level: 0.95,
        };

        let algorithms: Vec<Box<dyn RegressionAlgorithm>> = vec![
            Box::new(TTestRegressionDetector::new(thresholds.clone())),
            Box::new(MannWhitneyRegressionDetector::new(thresholds.clone())),
        ];

        Self {
            baselines: HashMap::new(),
            thresholds,
            algorithms,
            alert_history: VecDeque::new(),
        }
    }

    pub fn update_baseline(&mut self, metric_name: &str, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (samples.len() - 1) as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_samples = samples.to_vec();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut percentiles = HashMap::new();
        for &p in &[50, 90, 95, 99] {
            let index = ((p as f64 / 100.0) * (sorted_samples.len() - 1) as f64) as usize;
            percentiles.insert(p, sorted_samples[index]);
        }

        let baseline = PerformanceBaseline {
            metric_name: metric_name.to_string(),
            mean,
            std_dev,
            percentiles,
            sample_count: samples.len() as u64,
            last_updated: now_nanos(),
            trend: TrendDirection::Unknown,
        };

        self.baselines.insert(metric_name.to_string(), baseline);
    }

    pub fn detect_regressions(&mut self, metric_name: &str, current_samples: &[f64]) -> Vec<RegressionAlert> {
        let baseline = match self.baselines.get(metric_name) {
            Some(baseline) => baseline,
            None => return Vec::new(),
        };

        let mut alerts = Vec::new();
        
        for algorithm in &self.algorithms {
            if let Some(alert) = algorithm.detect_regression(baseline, current_samples) {
                alerts.push(alert);
            }
        }

        // Store alerts in history
        for alert in &alerts {
            self.alert_history.push_back(alert.clone());
            
            // Maintain history size
            while self.alert_history.len() > 1000 {
                self.alert_history.pop_front();
            }
        }

        alerts
    }

    pub fn get_alert_history(&self, limit: Option<usize>) -> Vec<&RegressionAlert> {
        let limit = limit.unwrap_or(self.alert_history.len());
        self.alert_history.iter().rev().take(limit).collect()
    }

    pub fn get_baseline(&self, metric_name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(metric_name)
    }
}

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