use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache performance counters
#[repr(C, align(64))]
pub struct CachePerformanceCounters {
    /// L1 cache hits
    pub l1_hits: AtomicU64,
    /// L1 cache misses
    pub l1_misses: AtomicU64,
    /// L2 cache hits
    pub l2_hits: AtomicU64,
    /// L2 cache misses
    pub l2_misses: AtomicU64,
    /// L3 cache hits
    pub l3_hits: AtomicU64,
    /// L3 cache misses
    pub l3_misses: AtomicU64,
    /// TLB hits
    pub tlb_hits: AtomicU64,
    /// TLB misses
    pub tlb_misses: AtomicU64,
    /// Memory stall cycles
    pub memory_stall_cycles: AtomicU64,
    /// Total memory accesses
    pub total_accesses: AtomicU64,
    
    _padding: [u8; 64 - (10 * 8) % 64],
}

impl CachePerformanceCounters {
    pub fn new() -> Self {
        Self {
            l1_hits: AtomicU64::new(0),
            l1_misses: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            l2_misses: AtomicU64::new(0),
            l3_hits: AtomicU64::new(0),
            l3_misses: AtomicU64::new(0),
            tlb_hits: AtomicU64::new(0),
            tlb_misses: AtomicU64::new(0),
            memory_stall_cycles: AtomicU64::new(0),
            total_accesses: AtomicU64::new(0),
            _padding: [0; 64 - (10 * 8) % 64],
        }
    }

    /// Calculate L1 cache hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let hits = self.l1_hits.load(Ordering::Relaxed) as f64;
        let misses = self.l1_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    /// Calculate L2 cache hit rate
    pub fn l2_hit_rate(&self) -> f64 {
        let hits = self.l2_hits.load(Ordering::Relaxed) as f64;
        let misses = self.l2_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    /// Calculate L3 cache hit rate
    pub fn l3_hit_rate(&self) -> f64 {
        let hits = self.l3_hits.load(Ordering::Relaxed) as f64;
        let misses = self.l3_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total > 0.0 { hits / total } else { 0.0 }
    }

    /// Calculate overall cache efficiency
    pub fn cache_efficiency(&self) -> f64 {
        let total_hits = self.l1_hits.load(Ordering::Relaxed) +
                        self.l2_hits.load(Ordering::Relaxed) +
                        self.l3_hits.load(Ordering::Relaxed);
        let total_accesses = self.total_accesses.load(Ordering::Relaxed);
        
        if total_accesses > 0 {
            total_hits as f64 / total_accesses as f64
        } else {
            0.0
        }
    }

    /// Record a cache access pattern
    pub fn record_access(&self, cache_level: CacheLevel, hit: bool) {
        self.total_accesses.fetch_add(1, Ordering::Relaxed);
        
        match (cache_level, hit) {
            (CacheLevel::L1, true) => { self.l1_hits.fetch_add(1, Ordering::Relaxed); },
            (CacheLevel::L1, false) => { self.l1_misses.fetch_add(1, Ordering::Relaxed); },
            (CacheLevel::L2, true) => { self.l2_hits.fetch_add(1, Ordering::Relaxed); },
            (CacheLevel::L2, false) => { self.l2_misses.fetch_add(1, Ordering::Relaxed); },
            (CacheLevel::L3, true) => { self.l3_hits.fetch_add(1, Ordering::Relaxed); },
            (CacheLevel::L3, false) => { self.l3_misses.fetch_add(1, Ordering::Relaxed); },
        }
    }
}

/// Cache levels for monitoring
#[derive(Clone, Copy, Debug)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
}

/// Cache miss analyzer for identifying performance bottlenecks
pub struct CacheMissAnalyzer {
    /// Performance counters
    counters: CachePerformanceCounters,
    /// Hot memory regions (frequently accessed)
    hot_regions: HashMap<usize, AccessPattern>,
    /// Cold memory regions (infrequently accessed)
    cold_regions: HashMap<usize, AccessPattern>,
    /// Analysis window duration
    window_duration: Duration,
    /// Last analysis time
    last_analysis: Instant,
}

/// Memory access pattern information
#[derive(Clone, Debug)]
pub struct AccessPattern {
    /// Number of accesses in current window
    pub access_count: u64,
    /// Cache miss rate
    pub miss_rate: f64,
    /// Average access latency
    pub avg_latency_ns: f64,
    /// Access stride pattern
    pub stride_pattern: Option<isize>,
    /// Last access timestamp
    pub last_access: Instant,
}

impl CacheMissAnalyzer {
    pub fn new(window_duration: Duration) -> Self {
        Self {
            counters: CachePerformanceCounters::new(),
            hot_regions: HashMap::new(),
            cold_regions: HashMap::new(),
            window_duration,
            last_analysis: Instant::now(),
        }
    }

    /// Record a memory access for analysis
    pub fn record_memory_access(
        &mut self,
        address: usize,
        cache_level: CacheLevel,
        hit: bool,
        latency_ns: f64,
    ) {
        self.counters.record_access(cache_level, hit);
        
        // Group addresses by cache line (64-byte aligned)
        let cache_line = address & !63;
        
        let pattern = self.hot_regions.entry(cache_line).or_insert_with(|| {
            AccessPattern {
                access_count: 0,
                miss_rate: 0.0,
                avg_latency_ns: 0.0,
                stride_pattern: None,
                last_access: Instant::now(),
            }
        });
        
        // Update access pattern
        pattern.access_count += 1;
        pattern.avg_latency_ns = (pattern.avg_latency_ns + latency_ns) / 2.0;
        pattern.last_access = Instant::now();
        
        if !hit {
            pattern.miss_rate = (pattern.miss_rate + 1.0) / 2.0;
        } else {
            pattern.miss_rate = pattern.miss_rate * 0.9; // Decay miss rate on hits
        }
    }

    /// Analyze cache performance and identify optimization opportunities
    pub fn analyze_performance(&mut self) -> CacheAnalysisReport {
        let now = Instant::now();
        if now.duration_since(self.last_analysis) < self.window_duration {
            return CacheAnalysisReport::default();
        }
        
        self.last_analysis = now;
        
        // Identify hot and cold regions
        let mut hot_regions = Vec::new();
        let mut cold_regions = Vec::new();
        
        for (&address, pattern) in &self.hot_regions {
            if pattern.access_count > 100 {
                hot_regions.push((address, pattern.clone()));
            } else if pattern.access_count < 10 {
                cold_regions.push((address, pattern.clone()));
            }
        }
        
        // Sort by access frequency
        hot_regions.sort_by(|a, b| b.1.access_count.cmp(&a.1.access_count));
        cold_regions.sort_by(|a, b| a.1.access_count.cmp(&b.1.access_count));
        
        // Generate optimization recommendations
        let recommendations = self.generate_recommendations(&hot_regions, &cold_regions);
        
        CacheAnalysisReport {
            l1_hit_rate: self.counters.l1_hit_rate(),
            l2_hit_rate: self.counters.l2_hit_rate(),
            l3_hit_rate: self.counters.l3_hit_rate(),
            overall_efficiency: self.counters.cache_efficiency(),
            hot_regions: hot_regions.into_iter().take(10).collect(),
            cold_regions: cold_regions.into_iter().take(10).collect(),
            recommendations,
        }
    }

    fn generate_recommendations(
        &self,
        hot_regions: &[(usize, AccessPattern)],
        cold_regions: &[(usize, AccessPattern)],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        // Check for high miss rate in hot regions
        for (address, pattern) in hot_regions {
            if pattern.miss_rate > 0.1 { // More than 10% miss rate
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::PrefetchOptimization,
                    address: *address,
                    description: format!(
                        "High miss rate ({:.1}%) in hot region. Consider prefetching.",
                        pattern.miss_rate * 100.0
                    ),
                    priority: if pattern.miss_rate > 0.2 { Priority::High } else { Priority::Medium },
                });
            }
            
            if pattern.avg_latency_ns > 100.0 { // High latency
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::DataLayoutOptimization,
                    address: *address,
                    description: format!(
                        "High average latency ({:.1}ns). Consider data layout optimization.",
                        pattern.avg_latency_ns
                    ),
                    priority: Priority::Medium,
                });
            }
        }
        
        // Check for cache pollution from cold regions
        if cold_regions.len() > hot_regions.len() * 2 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::CachePollutionReduction,
                address: 0,
                description: "Many cold regions detected. Consider data structure reorganization.".to_string(),
                priority: Priority::Low,
            });
        }
        
        recommendations
    }

    /// Get current performance counters
    pub fn get_counters(&self) -> &CachePerformanceCounters {
        &self.counters
    }

    /// Reset all counters and analysis data
    pub fn reset(&mut self) {
        self.hot_regions.clear();
        self.cold_regions.clear();
        // Note: We can't reset atomic counters easily, but we can create new ones
        self.counters = CachePerformanceCounters::new();
        self.last_analysis = Instant::now();
    }
}

/// Cache analysis report
#[derive(Debug, Default)]
pub struct CacheAnalysisReport {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub overall_efficiency: f64,
    pub hot_regions: Vec<(usize, AccessPattern)>,
    pub cold_regions: Vec<(usize, AccessPattern)>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Optimization recommendation
#[derive(Debug)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub address: usize,
    pub description: String,
    pub priority: Priority,
}

/// Types of optimization recommendations
#[derive(Debug)]
pub enum RecommendationType {
    PrefetchOptimization,
    DataLayoutOptimization,
    CachePollutionReduction,
    MemoryPoolReorganization,
    AlgorithmOptimization,
}

/// Priority levels for recommendations
#[derive(Debug)]
pub enum Priority {
    High,
    Medium,
    Low,
}

/// Cache-aware memory access tracker
pub struct CacheAccessTracker {
    analyzer: CacheMissAnalyzer,
    enabled: bool,
}

impl CacheAccessTracker {
    pub fn new() -> Self {
        Self {
            analyzer: CacheMissAnalyzer::new(Duration::from_secs(1)),
            enabled: true,
        }
    }

    /// Track a memory access (should be called from hot paths with care)
    #[inline(always)]
    pub fn track_access<T>(&mut self, ptr: *const T, hit: bool, latency_ns: f64) {
        if self.enabled {
            self.analyzer.record_memory_access(
                ptr as usize,
                CacheLevel::L1, // Assume L1 for simplicity
                hit,
                latency_ns,
            );
        }
    }

    /// Enable or disable tracking (useful for performance-critical sections)
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get analysis report
    pub fn get_analysis(&mut self) -> CacheAnalysisReport {
        self.analyzer.analyze_performance()
    }
}

/// Global cache performance monitor
static mut GLOBAL_CACHE_MONITOR: Option<CacheAccessTracker> = None;
static MONITOR_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize global cache monitoring
pub fn init_cache_monitoring() {
    MONITOR_INIT.call_once(|| {
        unsafe {
            GLOBAL_CACHE_MONITOR = Some(CacheAccessTracker::new());
        }
    });
}

/// Get global cache monitor (must call init_cache_monitoring first)
pub fn get_cache_monitor() -> Option<&'static mut CacheAccessTracker> {
    unsafe { GLOBAL_CACHE_MONITOR.as_mut() }
}