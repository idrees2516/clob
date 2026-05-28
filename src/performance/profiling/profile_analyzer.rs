use super::sampling_profiler::ProfileSample;
use super::timing::now_nanos;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Profile analyzer for extracting performance insights from profiling data
pub struct ProfileAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    
    /// Analysis cache for performance
    analysis_cache: std::sync::RwLock<HashMap<String, CachedAnalysis>>,
}

/// Configuration for profile analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Minimum samples required for analysis
    pub min_samples: usize,
    
    /// Top N functions to analyze
    pub top_functions_count: usize,
    
    /// Enable hotspot detection
    pub enable_hotspot_detection: bool,
    
    /// Enable bottleneck analysis
    pub enable_bottleneck_analysis: bool,
    
    /// Enable memory analysis
    pub enable_memory_analysis: bool,
    
    /// Enable concurrency analysis
    pub enable_concurrency_analysis: bool,
    
    /// Hotspot threshold (percentage of total time)
    pub hotspot_threshold_percent: f64,
    
    /// Functions to exclude from analysis
    pub exclude_functions: Vec<String>,
    
    /// Cache analysis results
    pub enable_caching: bool,
    
    /// Cache TTL (nanoseconds)
    pub cache_ttl_ns: u64,
}

/// Comprehensive profile analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileAnalysis {
    /// Analysis timestamp
    pub analyzed_at: u64,
    
    /// Analysis duration (nanoseconds)
    pub analysis_duration_ns: u64,
    
    /// Number of samples analyzed
    pub sample_count: usize,
    
    /// Performance insights
    pub insights: Vec<PerformanceInsight>,
    
    /// Function statistics
    pub function_stats: Vec<FunctionStatistics>,
    
    /// Hotspot analysis
    pub hotspots: Vec<PerformanceHotspot>,
    
    /// Bottleneck analysis
    pub bottlenecks: Vec<PerformanceBottleneck>,
    
    /// Memory analysis
    pub memory_analysis: Option<MemoryAnalysis>,
    
    /// Concurrency analysis
    pub concurrency_analysis: Option<ConcurrencyAnalysis>,
    
    /// Overall performance score (0-100)
    pub performance_score: f64,
    
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Individual performance insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsight {
    /// Insight type
    pub insight_type: InsightType,
    
    /// Insight severity
    pub severity: InsightSeverity,
    
    /// Insight title
    pub title: String,
    
    /// Detailed description
    pub description: String,
    
    /// Affected function or component
    pub affected_component: String,
    
    /// Potential impact (percentage improvement)
    pub potential_impact_percent: f64,
    
    /// Recommended actions
    pub recommendations: Vec<String>,
    
    /// Supporting data
    pub supporting_data: HashMap<String, f64>,
}

/// Types of performance insights
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// CPU hotspot detected
    CpuHotspot,
    
    /// Memory inefficiency
    MemoryInefficiency,
    
    /// I/O bottleneck
    IoBottleneck,
    
    /// Lock contention
    LockContention,
    
    /// Algorithmic inefficiency
    AlgorithmicInefficiency,
    
    /// Resource waste
    ResourceWaste,
    
    /// Scalability issue
    ScalabilityIssue,
    
    /// General optimization opportunity
    OptimizationOpportunity,
}

/// Insight severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InsightSeverity {
    Info = 1,
    Low = 2,
    Medium = 3,
    High = 4,
    Critical = 5,
}

/// Function performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionStatistics {
    /// Function name
    pub name: String,
    
    /// Total execution time (nanoseconds)
    pub total_time_ns: u64,
    
    /// Average execution time (nanoseconds)
    pub avg_time_ns: u64,
    
    /// Minimum execution time (nanoseconds)
    pub min_time_ns: u64,
    
    /// Maximum execution time (nanoseconds)
    pub max_time_ns: u64,
    
    /// Standard deviation of execution time
    pub std_dev_ns: f64,
    
    /// Number of calls
    pub call_count: u64,
    
    /// Percentage of total execution time
    pub time_percentage: f64,
    
    /// Average CPU usage
    pub avg_cpu_usage: f64,
    
    /// Average memory usage
    pub avg_memory_usage: u64,
    
    /// Call frequency (calls per second)
    pub call_frequency: f64,
}

/// Performance hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHotspot {
    /// Function name
    pub function_name: String,
    
    /// Time spent in this hotspot (nanoseconds)
    pub time_spent_ns: u64,
    
    /// Percentage of total time
    pub time_percentage: f64,
    
    /// Hotspot type
    pub hotspot_type: HotspotType,
    
    /// Optimization potential
    pub optimization_potential: f64,
    
    /// Recommended optimizations
    pub recommendations: Vec<String>,
}

/// Types of performance hotspots
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HotspotType {
    CpuIntensive,
    MemoryIntensive,
    IoIntensive,
    LockContention,
    AlgorithmicComplexity,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck location
    pub location: String,
    
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    
    /// Impact on overall performance
    pub impact_percentage: f64,
    
    /// Wait time or delay (nanoseconds)
    pub delay_ns: u64,
    
    /// Affected operations count
    pub affected_operations: u64,
    
    /// Root cause analysis
    pub root_cause: String,
    
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    CpuBound,
    MemoryBound,
    IoBound,
    NetworkBound,
    LockContention,
    ResourceStarvation,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Total memory usage (bytes)
    pub total_memory_usage: u64,
    
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    
    /// Average memory usage (bytes)
    pub avg_memory_usage: u64,
    
    /// Memory growth rate (bytes per second)
    pub memory_growth_rate: f64,
    
    /// Memory efficiency score (0-100)
    pub efficiency_score: f64,
    
    /// Top memory consumers
    pub top_consumers: Vec<MemoryConsumer>,
    
    /// Memory allocation patterns
    pub allocation_patterns: Vec<AllocationPattern>,
    
    /// Potential memory leaks
    pub potential_leaks: Vec<String>,
}

/// Memory consumer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsumer {
    /// Function name
    pub function_name: String,
    
    /// Memory usage (bytes)
    pub memory_usage: u64,
    
    /// Percentage of total memory
    pub percentage: f64,
    
    /// Allocation frequency
    pub allocation_frequency: f64,
}

/// Memory allocation pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Pattern type
    pub pattern_type: String,
    
    /// Frequency of this pattern
    pub frequency: u64,
    
    /// Average allocation size
    pub avg_allocation_size: u64,
    
    /// Pattern efficiency
    pub efficiency_score: f64,
}

/// Concurrency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyAnalysis {
    /// Thread utilization
    pub thread_utilization: HashMap<u64, f64>,
    
    /// Lock contention points
    pub lock_contention: Vec<LockContentionPoint>,
    
    /// Thread synchronization overhead
    pub sync_overhead_ns: u64,
    
    /// Parallelization efficiency
    pub parallelization_efficiency: f64,
    
    /// Scalability bottlenecks
    pub scalability_bottlenecks: Vec<String>,
}

/// Lock contention point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockContentionPoint {
    /// Lock location
    pub location: String,
    
    /// Contention time (nanoseconds)
    pub contention_time_ns: u64,
    
    /// Number of waiting threads
    pub waiting_threads: u32,
    
    /// Contention frequency
    pub contention_frequency: f64,
}

/// Cached analysis result
#[derive(Debug, Clone)]
struct CachedAnalysis {
    analysis: ProfileAnalysis,
    cached_at: u64,
    ttl_ns: u64,
}

impl ProfileAnalyzer {
    /// Create a new profile analyzer
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
            analysis_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            config,
            analysis_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Analyze profile samples
    pub fn analyze_samples(&self, samples: &[ProfileSample]) -> Result<ProfileAnalysis, AnalysisError> {
        if samples.len() < self.config.min_samples {
            return Err(AnalysisError::InsufficientSamples);
        }

        let analysis_start = now_nanos();
        
        // Check cache if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(samples);
            if let Some(cached) = self.get_cached_analysis(&cache_key) {
                return Ok(cached);
            }
        }

        // Perform analysis
        let function_stats = self.calculate_function_statistics(samples);
        let insights = self.generate_insights(samples, &function_stats);
        let hotspots = if self.config.enable_hotspot_detection {
            self.detect_hotspots(samples, &function_stats)
        } else {
            Vec::new()
        };
        let bottlenecks = if self.config.enable_bottleneck_analysis {
            self.analyze_bottlenecks(samples, &function_stats)
        } else {
            Vec::new()
        };
        let memory_analysis = if self.config.enable_memory_analysis {
            Some(self.analyze_memory_usage(samples))
        } else {
            None
        };
        let concurrency_analysis = if self.config.enable_concurrency_analysis {
            Some(self.analyze_concurrency(samples))
        } else {
            None
        };

        let performance_score = self.calculate_performance_score(&insights, &hotspots, &bottlenecks);
        
        let analysis_duration = now_nanos() - analysis_start;
        
        let mut metadata = HashMap::new();
        metadata.insert("analysis_version".to_string(), "1.0".to_string());
        metadata.insert("sample_timespan_ns".to_string(), 
                        self.calculate_sample_timespan(samples).to_string());
        metadata.insert("unique_functions".to_string(), 
                        function_stats.len().to_string());

        let analysis = ProfileAnalysis {
            analyzed_at: analysis_start,
            analysis_duration_ns: analysis_duration,
            sample_count: samples.len(),
            insights,
            function_stats,
            hotspots,
            bottlenecks,
            memory_analysis,
            concurrency_analysis,
            performance_score,
            metadata,
        };

        // Cache result if enabled
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(samples);
            self.cache_analysis(cache_key, analysis.clone());
        }

        Ok(analysis)
    }

    /// Calculate function statistics
    fn calculate_function_statistics(&self, samples: &[ProfileSample]) -> Vec<FunctionStatistics> {
        let mut function_data: HashMap<String, Vec<&ProfileSample>> = HashMap::new();
        
        // Group samples by function
        for sample in samples {
            if !self.should_exclude_function(&sample.function_name) {
                function_data.entry(sample.function_name.clone())
                    .or_insert_with(Vec::new)
                    .push(sample);
            }
        }

        let total_time: u64 = samples.iter().map(|s| s.duration_ns).sum();
        let sample_timespan = self.calculate_sample_timespan(samples);
        
        let mut stats = Vec::new();
        
        for (function_name, function_samples) in function_data {
            let durations: Vec<u64> = function_samples.iter().map(|s| s.duration_ns).collect();
            let cpu_usages: Vec<f64> = function_samples.iter().map(|s| s.cpu_usage).collect();
            let memory_usages: Vec<u64> = function_samples.iter().map(|s| s.memory_usage).collect();
            
            let total_function_time: u64 = durations.iter().sum();
            let avg_time = total_function_time / durations.len() as u64;
            let min_time = *durations.iter().min().unwrap_or(&0);
            let max_time = *durations.iter().max().unwrap_or(&0);
            
            let mean = avg_time as f64;
            let variance = durations.iter()
                .map(|&d| (d as f64 - mean).powi(2))
                .sum::<f64>() / durations.len() as f64;
            let std_dev = variance.sqrt();
            
            let time_percentage = if total_time > 0 {
                (total_function_time as f64 / total_time as f64) * 100.0
            } else {
                0.0
            };
            
            let avg_cpu = cpu_usages.iter().sum::<f64>() / cpu_usages.len() as f64;
            let avg_memory = memory_usages.iter().sum::<u64>() / memory_usages.len() as u64;
            
            let call_frequency = if sample_timespan > 0 {
                (function_samples.len() as f64) / (sample_timespan as f64 / 1_000_000_000.0)
            } else {
                0.0
            };
            
            stats.push(FunctionStatistics {
                name: function_name,
                total_time_ns: total_function_time,
                avg_time_ns: avg_time,
                min_time_ns: min_time,
                max_time_ns: max_time,
                std_dev_ns: std_dev,
                call_count: function_samples.len() as u64,
                time_percentage,
                avg_cpu_usage: avg_cpu,
                avg_memory_usage: avg_memory,
                call_frequency,
            });
        }
        
        // Sort by total time (descending)
        stats.sort_by(|a, b| b.total_time_ns.cmp(&a.total_time_ns));
        
        // Limit to top N functions
        stats.truncate(self.config.top_functions_count);
        
        stats
    }

    /// Generate performance insights
    fn generate_insights(&self, samples: &[ProfileSample], function_stats: &[FunctionStatistics]) -> Vec<PerformanceInsight> {
        let mut insights = Vec::new();
        
        // Analyze function performance
        for stat in function_stats {
            // High CPU usage insight
            if stat.avg_cpu_usage > 80.0 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::CpuHotspot,
                    severity: InsightSeverity::High,
                    title: format!("High CPU usage in {}", stat.name),
                    description: format!("Function {} is using {:.1}% CPU on average", 
                                       stat.name, stat.avg_cpu_usage),
                    affected_component: stat.name.clone(),
                    potential_impact_percent: stat.time_percentage * 0.5,
                    recommendations: vec![
                        "Consider algorithmic optimizations".to_string(),
                        "Profile for CPU-intensive operations".to_string(),
                        "Consider parallelization if applicable".to_string(),
                    ],
                    supporting_data: {
                        let mut data = HashMap::new();
                        data.insert("cpu_usage".to_string(), stat.avg_cpu_usage);
                        data.insert("time_percentage".to_string(), stat.time_percentage);
                        data
                    },
                });
            }
            
            // High memory usage insight
            if stat.avg_memory_usage > 100_000_000 { // 100MB
                insights.push(PerformanceInsight {
                    insight_type: InsightType::MemoryInefficiency,
                    severity: InsightSeverity::Medium,
                    title: format!("High memory usage in {}", stat.name),
                    description: format!("Function {} is using {:.1} MB of memory on average", 
                                       stat.name, stat.avg_memory_usage as f64 / 1_000_000.0),
                    affected_component: stat.name.clone(),
                    potential_impact_percent: 10.0,
                    recommendations: vec![
                        "Review memory allocation patterns".to_string(),
                        "Consider memory pooling".to_string(),
                        "Check for memory leaks".to_string(),
                    ],
                    supporting_data: {
                        let mut data = HashMap::new();
                        data.insert("memory_usage_mb".to_string(), stat.avg_memory_usage as f64 / 1_000_000.0);
                        data
                    },
                });
            }
            
            // High variance insight (inconsistent performance)
            let coefficient_of_variation = stat.std_dev_ns / stat.avg_time_ns as f64;
            if coefficient_of_variation > 0.5 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::OptimizationOpportunity,
                    severity: InsightSeverity::Medium,
                    title: format!("Inconsistent performance in {}", stat.name),
                    description: format!("Function {} has high performance variance (CV: {:.2})", 
                                       stat.name, coefficient_of_variation),
                    affected_component: stat.name.clone(),
                    potential_impact_percent: 15.0,
                    recommendations: vec![
                        "Investigate performance variance causes".to_string(),
                        "Consider caching frequently computed results".to_string(),
                        "Review conditional logic paths".to_string(),
                    ],
                    supporting_data: {
                        let mut data = HashMap::new();
                        data.insert("coefficient_of_variation".to_string(), coefficient_of_variation);
                        data.insert("std_dev_ns".to_string(), stat.std_dev_ns);
                        data
                    },
                });
            }
        }
        
        // Sort insights by severity and potential impact
        insights.sort_by(|a, b| {
            b.severity.cmp(&a.severity)
                .then_with(|| b.potential_impact_percent.partial_cmp(&a.potential_impact_percent)
                    .unwrap_or(std::cmp::Ordering::Equal))
        });
        
        insights
    }

    /// Detect performance hotspots
    fn detect_hotspots(&self, _samples: &[ProfileSample], function_stats: &[FunctionStatistics]) -> Vec<PerformanceHotspot> {
        let mut hotspots = Vec::new();
        
        for stat in function_stats {
            if stat.time_percentage >= self.config.hotspot_threshold_percent {
                let hotspot_type = if stat.avg_cpu_usage > 70.0 {
                    HotspotType::CpuIntensive
                } else if stat.avg_memory_usage > 50_000_000 {
                    HotspotType::MemoryIntensive
                } else {
                    HotspotType::AlgorithmicComplexity
                };
                
                let optimization_potential = (stat.time_percentage / 100.0) * 50.0; // Up to 50% improvement
                
                let recommendations = match hotspot_type {
                    HotspotType::CpuIntensive => vec![
                        "Optimize CPU-intensive algorithms".to_string(),
                        "Consider SIMD optimizations".to_string(),
                        "Profile assembly code".to_string(),
                    ],
                    HotspotType::MemoryIntensive => vec![
                        "Optimize memory access patterns".to_string(),
                        "Consider memory pooling".to_string(),
                        "Reduce memory allocations".to_string(),
                    ],
                    _ => vec![
                        "Review algorithm complexity".to_string(),
                        "Consider alternative algorithms".to_string(),
                        "Optimize data structures".to_string(),
                    ],
                };
                
                hotspots.push(PerformanceHotspot {
                    function_name: stat.name.clone(),
                    time_spent_ns: stat.total_time_ns,
                    time_percentage: stat.time_percentage,
                    hotspot_type,
                    optimization_potential,
                    recommendations,
                });
            }
        }
        
        hotspots
    }

    /// Analyze performance bottlenecks
    fn analyze_bottlenecks(&self, _samples: &[ProfileSample], function_stats: &[FunctionStatistics]) -> Vec<PerformanceBottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Find functions with high execution time and low CPU usage (potential I/O bottlenecks)
        for stat in function_stats {
            if stat.time_percentage > 10.0 && stat.avg_cpu_usage < 30.0 {
                bottlenecks.push(PerformanceBottleneck {
                    location: stat.name.clone(),
                    bottleneck_type: BottleneckType::IoBound,
                    impact_percentage: stat.time_percentage,
                    delay_ns: stat.avg_time_ns,
                    affected_operations: stat.call_count,
                    root_cause: "Low CPU usage with high execution time suggests I/O waiting".to_string(),
                    mitigation_strategies: vec![
                        "Implement asynchronous I/O".to_string(),
                        "Use connection pooling".to_string(),
                        "Cache frequently accessed data".to_string(),
                    ],
                });
            }
        }
        
        bottlenecks
    }

    /// Analyze memory usage patterns
    fn analyze_memory_usage(&self, samples: &[ProfileSample]) -> MemoryAnalysis {
        let memory_usages: Vec<u64> = samples.iter().map(|s| s.memory_usage).collect();
        
        let total_memory = memory_usages.iter().sum::<u64>();
        let avg_memory = total_memory / memory_usages.len() as u64;
        let peak_memory = *memory_usages.iter().max().unwrap_or(&0);
        
        // Calculate memory growth rate (simplified)
        let memory_growth_rate = if samples.len() > 1 {
            let first_memory = samples[0].memory_usage as f64;
            let last_memory = samples[samples.len() - 1].memory_usage as f64;
            let time_diff = (samples[samples.len() - 1].timestamp - samples[0].timestamp) as f64 / 1_000_000_000.0;
            
            if time_diff > 0.0 {
                (last_memory - first_memory) / time_diff
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        // Calculate efficiency score (simplified)
        let efficiency_score = if peak_memory > 0 {
            ((avg_memory as f64 / peak_memory as f64) * 100.0).min(100.0)
        } else {
            100.0
        };
        
        // Find top memory consumers
        let mut function_memory: HashMap<String, Vec<u64>> = HashMap::new();
        for sample in samples {
            function_memory.entry(sample.function_name.clone())
                .or_insert_with(Vec::new)
                .push(sample.memory_usage);
        }
        
        let mut top_consumers = Vec::new();
        for (function_name, usages) in function_memory {
            let avg_usage = usages.iter().sum::<u64>() / usages.len() as u64;
            let percentage = (avg_usage as f64 / avg_memory as f64) * 100.0;
            
            top_consumers.push(MemoryConsumer {
                function_name,
                memory_usage: avg_usage,
                percentage,
                allocation_frequency: usages.len() as f64,
            });
        }
        
        top_consumers.sort_by(|a, b| b.memory_usage.cmp(&a.memory_usage));
        top_consumers.truncate(10);
        
        MemoryAnalysis {
            total_memory_usage: total_memory,
            peak_memory_usage: peak_memory,
            avg_memory_usage: avg_memory,
            memory_growth_rate,
            efficiency_score,
            top_consumers,
            allocation_patterns: Vec::new(), // Simplified for now
            potential_leaks: Vec::new(),     // Simplified for now
        }
    }

    /// Analyze concurrency patterns
    fn analyze_concurrency(&self, samples: &[ProfileSample]) -> ConcurrencyAnalysis {
        let mut thread_utilization = HashMap::new();
        let mut thread_samples: HashMap<u64, Vec<&ProfileSample>> = HashMap::new();
        
        // Group samples by thread
        for sample in samples {
            thread_samples.entry(sample.thread_id)
                .or_insert_with(Vec::new)
                .push(sample);
        }
        
        // Calculate thread utilization
        for (thread_id, thread_sample_list) in &thread_samples {
            let avg_cpu = thread_sample_list.iter()
                .map(|s| s.cpu_usage)
                .sum::<f64>() / thread_sample_list.len() as f64;
            thread_utilization.insert(*thread_id, avg_cpu);
        }
        
        // Calculate parallelization efficiency
        let total_threads = thread_samples.len() as f64;
        let avg_utilization = thread_utilization.values().sum::<f64>() / total_threads;
        let parallelization_efficiency = (avg_utilization / 100.0).min(1.0);
        
        ConcurrencyAnalysis {
            thread_utilization,
            lock_contention: Vec::new(), // Simplified for now
            sync_overhead_ns: 0,         // Simplified for now
            parallelization_efficiency,
            scalability_bottlenecks: Vec::new(), // Simplified for now
        }
    }

    /// Calculate overall performance score
    fn calculate_performance_score(&self, insights: &[PerformanceInsight], hotspots: &[PerformanceHotspot], bottlenecks: &[PerformanceBottleneck]) -> f64 {
        let mut score = 100.0;
        
        // Deduct points for insights
        for insight in insights {
            let deduction = match insight.severity {
                InsightSeverity::Critical => 20.0,
                InsightSeverity::High => 15.0,
                InsightSeverity::Medium => 10.0,
                InsightSeverity::Low => 5.0,
                InsightSeverity::Info => 2.0,
            };
            score -= deduction;
        }
        
        // Deduct points for hotspots
        for hotspot in hotspots {
            score -= hotspot.time_percentage * 0.5;
        }
        
        // Deduct points for bottlenecks
        for bottleneck in bottlenecks {
            score -= bottleneck.impact_percentage * 0.3;
        }
        
        score.max(0.0).min(100.0)
    }

    /// Check if function should be excluded from analysis
    fn should_exclude_function(&self, function_name: &str) -> bool {
        self.config.exclude_functions.iter()
            .any(|pattern| function_name.contains(pattern))
    }

    /// Calculate timespan of samples
    fn calculate_sample_timespan(&self, samples: &[ProfileSample]) -> u64 {
        if samples.len() < 2 {
            return 0;
        }
        
        let min_timestamp = samples.iter().map(|s| s.timestamp).min().unwrap_or(0);
        let max_timestamp = samples.iter().map(|s| s.timestamp).max().unwrap_or(0);
        
        max_timestamp - min_timestamp
    }

    /// Generate cache key for samples
    fn generate_cache_key(&self, samples: &[ProfileSample]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        samples.len().hash(&mut hasher);
        
        // Hash first and last few samples for uniqueness
        let sample_count = samples.len().min(10);
        for sample in samples.iter().take(sample_count) {
            sample.function_name.hash(&mut hasher);
            sample.duration_ns.hash(&mut hasher);
        }
        
        format!("analysis_{:x}", hasher.finish())
    }

    /// Get cached analysis if available and valid
    fn get_cached_analysis(&self, cache_key: &str) -> Option<ProfileAnalysis> {
        let cache = self.analysis_cache.read().unwrap();
        
        if let Some(cached) = cache.get(cache_key) {
            let current_time = now_nanos();
            if current_time - cached.cached_at < cached.ttl_ns {
                return Some(cached.analysis.clone());
            }
        }
        
        None
    }

    /// Cache analysis result
    fn cache_analysis(&self, cache_key: String, analysis: ProfileAnalysis) {
        let mut cache = self.analysis_cache.write().unwrap();
        
        cache.insert(cache_key, CachedAnalysis {
            analysis,
            cached_at: now_nanos(),
            ttl_ns: self.config.cache_ttl_ns,
        });
        
        // Cleanup old cache entries
        let current_time = now_nanos();
        cache.retain(|_, cached| current_time - cached.cached_at < cached.ttl_ns);
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            min_samples: 10,
            top_functions_count: 20,
            enable_hotspot_detection: true,
            enable_bottleneck_analysis: true,
            enable_memory_analysis: true,
            enable_concurrency_analysis: true,
            hotspot_threshold_percent: 5.0,
            exclude_functions: vec![
                "std::".to_string(),
                "__rust_".to_string(),
                "core::".to_string(),
            ],
            enable_caching: true,
            cache_ttl_ns: 300_000_000_000, // 5 minutes
        }
    }
}

/// Analysis errors
#[derive(Debug, Clone)]
pub enum AnalysisError {
    InsufficientSamples,
    InvalidConfiguration,
    AnalysisTimeout,
    CacheError,
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::InsufficientSamples => write!(f, "Insufficient samples for analysis"),
            AnalysisError::InvalidConfiguration => write!(f, "Invalid analysis configuration"),
            AnalysisError::AnalysisTimeout => write!(f, "Analysis timed out"),
            AnalysisError::CacheError => write!(f, "Cache operation failed"),
        }
    }
}

impl std::error::Error for AnalysisError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_samples() -> Vec<ProfileSample> {
        vec![
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 1000000,
                function_name: "fast_function".to_string(),
                stack_trace: vec!["fast_function".to_string()],
                cpu_usage: 30.0,
                memory_usage: 1024,
                thread_id: 1,
            },
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 5000000,
                function_name: "slow_function".to_string(),
                stack_trace: vec!["slow_function".to_string()],
                cpu_usage: 90.0,
                memory_usage: 100_000_000,
                thread_id: 1,
            },
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 2000000,
                function_name: "medium_function".to_string(),
                stack_trace: vec!["medium_function".to_string()],
                cpu_usage: 50.0,
                memory_usage: 10_000_000,
                thread_id: 2,
            },
        ]
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ProfileAnalyzer::new();
        assert_eq!(analyzer.config.min_samples, 10);
        assert_eq!(analyzer.config.top_functions_count, 20);
        assert!(analyzer.config.enable_hotspot_detection);
    }

    #[test]
    fn test_function_statistics_calculation() {
        let analyzer = ProfileAnalyzer::new();
        let samples = create_test_samples();
        
        let stats = analyzer.calculate_function_statistics(&samples);
        
        assert_eq!(stats.len(), 3);
        
        // Should be sorted by total time (slow_function should be first)
        assert_eq!(stats[0].name, "slow_function");
        assert_eq!(stats[0].call_count, 1);
        assert_eq!(stats[0].total_time_ns, 5000000);
        assert_eq!(stats[0].avg_cpu_usage, 90.0);
        assert_eq!(stats[0].avg_memory_usage, 100_000_000);
    }

    #[test]
    fn test_insight_generation() {
        let analyzer = ProfileAnalyzer::new();
        let samples = create_test_samples();
        let function_stats = analyzer.calculate_function_statistics(&samples);
        
        let insights = analyzer.generate_insights(&samples, &function_stats);
        
        assert!(!insights.is_empty());
        
        // Should detect high CPU usage in slow_function
        let cpu_insight = insights.iter()
            .find(|i| i.insight_type == InsightType::CpuHotspot)
            .expect("Should detect CPU hotspot");
        
        assert_eq!(cpu_insight.affected_component, "slow_function");
        assert_eq!(cpu_insight.severity, InsightSeverity::High);
        
        // Should detect high memory usage in slow_function
        let memory_insight = insights.iter()
            .find(|i| i.insight_type == InsightType::MemoryInefficiency)
            .expect("Should detect memory inefficiency");
        
        assert_eq!(memory_insight.affected_component, "slow_function");
    }

    #[test]
    fn test_hotspot_detection() {
        let config = AnalysisConfig {
            hotspot_threshold_percent: 30.0, // Lower threshold for testing
            ..Default::default()
        };
        
        let analyzer = ProfileAnalyzer::with_config(config);
        let samples = create_test_samples();
        let function_stats = analyzer.calculate_function_statistics(&samples);
        
        let hotspots = analyzer.detect_hotspots(&samples, &function_stats);
        
        assert!(!hotspots.is_empty());
        
        let slow_function_hotspot = hotspots.iter()
            .find(|h| h.function_name == "slow_function")
            .expect("Should detect slow_function as hotspot");
        
        assert_eq!(slow_function_hotspot.hotspot_type, HotspotType::CpuIntensive);
        assert!(slow_function_hotspot.optimization_potential > 0.0);
        assert!(!slow_function_hotspot.recommendations.is_empty());
    }

    #[test]
    fn test_memory_analysis() {
        let analyzer = ProfileAnalyzer::new();
        let samples = create_test_samples();
        
        let memory_analysis = analyzer.analyze_memory_usage(&samples);
        
        assert!(memory_analysis.total_memory_usage > 0);
        assert!(memory_analysis.peak_memory_usage > 0);
        assert!(memory_analysis.avg_memory_usage > 0);
        assert!(memory_analysis.efficiency_score > 0.0);
        assert!(!memory_analysis.top_consumers.is_empty());
        
        // slow_function should be the top memory consumer
        let top_consumer = &memory_analysis.top_consumers[0];
        assert_eq!(top_consumer.function_name, "slow_function");
        assert_eq!(top_consumer.memory_usage, 100_000_000);
    }

    #[test]
    fn test_concurrency_analysis() {
        let analyzer = ProfileAnalyzer::new();
        let samples = create_test_samples();
        
        let concurrency_analysis = analyzer.analyze_concurrency(&samples);
        
        assert_eq!(concurrency_analysis.thread_utilization.len(), 2); // Two threads
        assert!(concurrency_analysis.thread_utilization.contains_key(&1));
        assert!(concurrency_analysis.thread_utilization.contains_key(&2));
        assert!(concurrency_analysis.parallelization_efficiency > 0.0);
    }

    #[test]
    fn test_performance_score_calculation() {
        let analyzer = ProfileAnalyzer::new();
        
        let insights = vec![
            PerformanceInsight {
                insight_type: InsightType::CpuHotspot,
                severity: InsightSeverity::High,
                title: "Test".to_string(),
                description: "Test".to_string(),
                affected_component: "test".to_string(),
                potential_impact_percent: 10.0,
                recommendations: Vec::new(),
                supporting_data: HashMap::new(),
            }
        ];
        
        let hotspots = vec![
            PerformanceHotspot {
                function_name: "test".to_string(),
                time_spent_ns: 1000000,
                time_percentage: 20.0,
                hotspot_type: HotspotType::CpuIntensive,
                optimization_potential: 10.0,
                recommendations: Vec::new(),
            }
        ];
        
        let bottlenecks = Vec::new();
        
        let score = analyzer.calculate_performance_score(&insights, &hotspots, &bottlenecks);
        
        // Should be less than 100 due to insights and hotspots
        assert!(score < 100.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_full_analysis() {
        let analyzer = ProfileAnalyzer::new();
        let samples = create_test_samples();
        
        let analysis = analyzer.analyze_samples(&samples).unwrap();
        
        assert_eq!(analysis.sample_count, 3);
        assert!(analysis.analysis_duration_ns > 0);
        assert!(!analysis.function_stats.is_empty());
        assert!(!analysis.insights.is_empty());
        assert!(analysis.memory_analysis.is_some());
        assert!(analysis.concurrency_analysis.is_some());
        assert!(analysis.performance_score > 0.0);
        assert!(analysis.performance_score <= 100.0);
    }

    #[test]
    fn test_insufficient_samples_error() {
        let analyzer = ProfileAnalyzer::new();
        let empty_samples: Vec<ProfileSample> = Vec::new();
        
        let result = analyzer.analyze_samples(&empty_samples);
        assert!(matches!(result, Err(AnalysisError::InsufficientSamples)));
    }

    #[test]
    fn test_function_exclusion() {
        let config = AnalysisConfig {
            exclude_functions: vec!["slow_".to_string()],
            ..Default::default()
        };
        
        let analyzer = ProfileAnalyzer::with_config(config);
        
        assert!(analyzer.should_exclude_function("slow_function"));
        assert!(!analyzer.should_exclude_function("fast_function"));
    }

    #[test]
    fn test_cache_key_generation() {
        let analyzer = ProfileAnalyzer::new();
        let samples1 = create_test_samples();
        let samples2 = create_test_samples();
        
        let key1 = analyzer.generate_cache_key(&samples1);
        let key2 = analyzer.generate_cache_key(&samples2);
        
        // Keys should be the same for identical samples
        assert_eq!(key1, key2);
        
        // Different samples should produce different keys
        let mut different_samples = create_test_samples();
        different_samples.push(ProfileSample {
            timestamp: now_nanos(),
            duration_ns: 999999,
            function_name: "different_function".to_string(),
            stack_trace: vec!["different_function".to_string()],
            cpu_usage: 25.0,
            memory_usage: 512,
            thread_id: 3,
        });
        
        let key3 = analyzer.generate_cache_key(&different_samples);
        assert_ne!(key1, key3);
    }
}