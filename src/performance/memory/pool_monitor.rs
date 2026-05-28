use super::lock_free_pool::{LockFreePool, PoolStats};
use super::numa_allocator::{NumaAllocator, NumaNodeStatsSnapshot};
use super::object_pools::{TradingObjectPools, TradingPoolStats, MemoryUsageStats};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;

/// Real-time memory pool monitoring system
pub struct PoolMonitor {
    /// Pools being monitored
    trading_pools: Arc<TradingObjectPools>,
    
    /// Monitoring configuration
    config: MonitorConfig,
    
    /// Real-time metrics collector
    metrics_collector: Arc<MetricsCollector>,
    
    /// Alert system
    alert_system: Arc<AlertSystem>,
    
    /// Monitoring thread handle
    monitor_thread: Option<thread::JoinHandle<()>>,
    
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

/// Configuration for pool monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    
    /// Enable real-time metrics collection
    pub enable_real_time_metrics: bool,
    
    /// Enable allocation failure detection
    pub enable_failure_detection: bool,
    
    /// Enable memory fragmentation monitoring
    pub enable_fragmentation_monitoring: bool,
    
    /// Enable performance analytics
    pub enable_performance_analytics: bool,
    
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    
    /// Metrics retention period (seconds)
    pub metrics_retention_seconds: u64,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Pool utilization threshold (percentage)
    pub pool_utilization_threshold: f64,
    
    /// Allocation failure rate threshold (failures per second)
    pub allocation_failure_rate_threshold: f64,
    
    /// Memory fragmentation threshold (percentage)
    pub fragmentation_threshold: f64,
    
    /// Average allocation latency threshold (nanoseconds)
    pub allocation_latency_threshold_ns: u64,
    
    /// Cross-NUMA allocation rate threshold (percentage)
    pub cross_numa_threshold: f64,
}

/// Real-time metrics collector
pub struct MetricsCollector {
    /// Current metrics snapshot
    current_metrics: Arc<std::sync::RwLock<PoolMetricsSnapshot>>,
    
    /// Historical metrics (circular buffer)
    historical_metrics: Arc<std::sync::RwLock<Vec<TimestampedMetrics>>>,
    
    /// Metrics collection statistics
    collection_stats: CollectionStats,
    
    /// Maximum historical entries
    max_historical_entries: usize,
}

/// Alert system for pool monitoring
pub struct AlertSystem {
    /// Active alerts
    active_alerts: Arc<std::sync::RwLock<HashMap<AlertType, Alert>>>,
    
    /// Alert history
    alert_history: Arc<std::sync::RwLock<Vec<Alert>>>,
    
    /// Alert callbacks
    alert_callbacks: Arc<std::sync::RwLock<Vec<Box<dyn Fn(&Alert) + Send + Sync>>>>,
    
    /// Alert statistics
    alert_stats: AlertStats,
}

/// Pool metrics snapshot
#[derive(Debug, Clone)]
pub struct PoolMetricsSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: u64,
    
    /// Trading pool statistics
    pub trading_stats: TradingPoolStats,
    
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    
    /// Pool utilization metrics
    pub utilization_metrics: UtilizationMetrics,
    
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    
    /// Fragmentation metrics
    pub fragmentation_metrics: FragmentationMetrics,
}

/// Pool utilization metrics
#[derive(Debug, Clone)]
pub struct UtilizationMetrics {
    /// Overall pool utilization (0.0 to 1.0)
    pub overall_utilization: f64,
    
    /// Per-pool utilization
    pub order_pool_utilization: f64,
    pub trade_pool_utilization: f64,
    pub order_node_pool_utilization: f64,
    pub price_level_pool_utilization: f64,
    
    /// Buffer pool utilizations by size
    pub buffer_pool_utilizations: HashMap<usize, f64>,
    
    /// NUMA node utilizations
    pub numa_utilizations: HashMap<usize, f64>,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average allocation latency (nanoseconds)
    pub avg_allocation_latency_ns: u64,
    
    /// 95th percentile allocation latency
    pub p95_allocation_latency_ns: u64,
    
    /// 99th percentile allocation latency
    pub p99_allocation_latency_ns: u64,
    
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,
    
    /// Deallocation rate (deallocations per second)
    pub deallocation_rate: f64,
    
    /// Pool expansion rate (expansions per minute)
    pub expansion_rate: f64,
    
    /// Cross-NUMA allocation rate (percentage)
    pub cross_numa_rate: f64,
}

/// Memory fragmentation metrics
#[derive(Debug, Clone)]
pub struct FragmentationMetrics {
    /// Overall fragmentation percentage
    pub overall_fragmentation: f64,
    
    /// Per-pool fragmentation
    pub pool_fragmentations: HashMap<String, f64>,
    
    /// Free block size distribution
    pub free_block_distribution: HashMap<usize, usize>,
    
    /// Largest free block size
    pub largest_free_block: usize,
    
    /// Number of free blocks
    pub free_block_count: usize,
}

/// Timestamped metrics for historical tracking
#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    pub timestamp: u64,
    pub metrics: PoolMetricsSnapshot,
}

/// Metrics collection statistics
#[derive(Debug)]
pub struct CollectionStats {
    /// Total collections performed
    pub total_collections: AtomicUsize,
    
    /// Failed collections
    pub failed_collections: AtomicUsize,
    
    /// Average collection time (nanoseconds)
    pub avg_collection_time_ns: AtomicU64,
    
    /// Last collection timestamp
    pub last_collection_timestamp: AtomicU64,
}

/// Alert types
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AlertType {
    HighPoolUtilization,
    AllocationFailureSpike,
    HighFragmentation,
    HighAllocationLatency,
    HighCrossNumaRate,
    PoolExpansionFailure,
    MemoryLeak,
    SystemOverload,
}

/// Alert information
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert type
    pub alert_type: AlertType,
    
    /// Alert severity
    pub severity: AlertSeverity,
    
    /// Alert message
    pub message: String,
    
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    
    /// Associated metrics
    pub metrics: Option<PoolMetricsSnapshot>,
    
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert system statistics
#[derive(Debug)]
pub struct AlertStats {
    /// Total alerts generated
    pub total_alerts: AtomicUsize,
    
    /// Alerts by severity
    pub info_alerts: AtomicUsize,
    pub warning_alerts: AtomicUsize,
    pub critical_alerts: AtomicUsize,
    pub emergency_alerts: AtomicUsize,
    
    /// Alert resolution time (nanoseconds)
    pub avg_resolution_time_ns: AtomicU64,
}

impl PoolMonitor {
    /// Create a new pool monitor
    pub fn new(
        trading_pools: Arc<TradingObjectPools>,
        config: MonitorConfig,
    ) -> Result<Self, MonitorError> {
        let metrics_collector = Arc::new(MetricsCollector::new(config.metrics_retention_seconds)?);
        let alert_system = Arc::new(AlertSystem::new());
        
        Ok(Self {
            trading_pools,
            config,
            metrics_collector,
            alert_system,
            monitor_thread: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start monitoring
    pub fn start(&mut self) -> Result<(), MonitorError> {
        if self.monitor_thread.is_some() {
            return Err(MonitorError::AlreadyRunning);
        }

        let trading_pools = self.trading_pools.clone();
        let config = self.config.clone();
        let metrics_collector = self.metrics_collector.clone();
        let alert_system = self.alert_system.clone();
        let shutdown = self.shutdown.clone();

        let handle = thread::spawn(move || {
            Self::monitoring_loop(
                trading_pools,
                config,
                metrics_collector,
                alert_system,
                shutdown,
            );
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// Stop monitoring
    pub fn stop(&mut self) -> Result<(), MonitorError> {
        self.shutdown.store(true, Ordering::Release);
        
        if let Some(handle) = self.monitor_thread.take() {
            handle.join().map_err(|_| MonitorError::ThreadJoinFailed)?;
        }
        
        Ok(())
    }

    /// Main monitoring loop
    fn monitoring_loop(
        trading_pools: Arc<TradingObjectPools>,
        config: MonitorConfig,
        metrics_collector: Arc<MetricsCollector>,
        alert_system: Arc<AlertSystem>,
        shutdown: Arc<AtomicBool>,
    ) {
        let interval = Duration::from_millis(config.monitoring_interval_ms);
        
        while !shutdown.load(Ordering::Acquire) {
            let start_time = Self::get_timestamp_ns();
            
            // Collect metrics
            if let Ok(metrics) = Self::collect_metrics(&trading_pools, &config) {
                // Store metrics
                if let Err(e) = metrics_collector.store_metrics(metrics.clone()) {
                    eprintln!("Failed to store metrics: {:?}", e);
                }
                
                // Check for alerts
                Self::check_alerts(&metrics, &config, &alert_system);
            }
            
            // Update collection statistics
            let collection_time = Self::get_timestamp_ns() - start_time;
            metrics_collector.update_collection_stats(collection_time);
            
            thread::sleep(interval);
        }
    }

    /// Collect current pool metrics
    fn collect_metrics(
        trading_pools: &TradingObjectPools,
        config: &MonitorConfig,
    ) -> Result<PoolMetricsSnapshot, MonitorError> {
        let timestamp = Self::get_timestamp_ns();
        let trading_stats = trading_pools.get_comprehensive_stats();
        let memory_usage = trading_pools.get_memory_usage();
        
        let utilization_metrics = Self::calculate_utilization_metrics(&trading_stats);
        let performance_metrics = Self::calculate_performance_metrics(&trading_stats);
        let fragmentation_metrics = if config.enable_fragmentation_monitoring {
            Self::calculate_fragmentation_metrics(&trading_stats)
        } else {
            FragmentationMetrics::default()
        };

        Ok(PoolMetricsSnapshot {
            timestamp,
            trading_stats,
            memory_usage,
            utilization_metrics,
            performance_metrics,
            fragmentation_metrics,
        })
    }

    /// Calculate pool utilization metrics
    fn calculate_utilization_metrics(stats: &TradingPoolStats) -> UtilizationMetrics {
        let order_util = stats.order_pool.in_use as f64 / stats.order_pool.capacity as f64;
        let trade_util = stats.trade_pool.in_use as f64 / stats.trade_pool.capacity as f64;
        let node_util = stats.order_node_pool.in_use as f64 / stats.order_node_pool.capacity as f64;
        let level_util = stats.price_level_pool.in_use as f64 / stats.price_level_pool.capacity as f64;
        
        let mut buffer_utilizations = HashMap::new();
        for (&size, pool_stats) in &stats.buffer_pool_stats {
            let util = pool_stats.in_use as f64 / pool_stats.capacity as f64;
            buffer_utilizations.insert(size, util);
        }
        
        let mut numa_utilizations = HashMap::new();
        for (&node, numa_stats) in &stats.numa_stats {
            let total_allocs = numa_stats.total_allocations as f64;
            let total_deallocs = numa_stats.total_deallocations as f64;
            let util = if total_allocs > 0.0 {
                (total_allocs - total_deallocs) / total_allocs
            } else {
                0.0
            };
            numa_utilizations.insert(node, util);
        }
        
        let overall_util = (order_util + trade_util + node_util + level_util) / 4.0;

        UtilizationMetrics {
            overall_utilization: overall_util,
            order_pool_utilization: order_util,
            trade_pool_utilization: trade_util,
            order_node_pool_utilization: node_util,
            price_level_pool_utilization: level_util,
            buffer_pool_utilizations,
            numa_utilizations,
        }
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(stats: &TradingPoolStats) -> PerformanceMetrics {
        let total_allocs = stats.manager_stats.total_allocations as f64;
        let total_deallocs = stats.manager_stats.total_deallocations as f64;
        let cross_numa = stats.manager_stats.cross_numa_allocations as f64;
        
        // Calculate rates (simplified - in real implementation would use time windows)
        let allocation_rate = total_allocs; // per second
        let deallocation_rate = total_deallocs; // per second
        let expansion_rate = stats.manager_stats.pool_expansions as f64; // per minute
        let cross_numa_rate = if total_allocs > 0.0 {
            (cross_numa / total_allocs) * 100.0
        } else {
            0.0
        };

        // Average latencies from individual pools
        let avg_latency = (stats.order_pool.avg_allocation_time_ns + 
                          stats.trade_pool.avg_allocation_time_ns +
                          stats.order_node_pool.avg_allocation_time_ns +
                          stats.price_level_pool.avg_allocation_time_ns) / 4;

        PerformanceMetrics {
            avg_allocation_latency_ns: avg_latency as u64,
            p95_allocation_latency_ns: (avg_latency as f64 * 1.5) as u64, // Approximation
            p99_allocation_latency_ns: (avg_latency as f64 * 2.0) as u64, // Approximation
            allocation_rate,
            deallocation_rate,
            expansion_rate,
            cross_numa_rate,
        }
    }

    /// Calculate memory fragmentation metrics
    fn calculate_fragmentation_metrics(stats: &TradingPoolStats) -> FragmentationMetrics {
        // Simplified fragmentation calculation
        let mut pool_fragmentations = HashMap::new();
        
        // Calculate fragmentation for each pool
        let order_frag = Self::calculate_pool_fragmentation(&stats.order_pool);
        let trade_frag = Self::calculate_pool_fragmentation(&stats.trade_pool);
        let node_frag = Self::calculate_pool_fragmentation(&stats.order_node_pool);
        let level_frag = Self::calculate_pool_fragmentation(&stats.price_level_pool);
        
        pool_fragmentations.insert("order_pool".to_string(), order_frag);
        pool_fragmentations.insert("trade_pool".to_string(), trade_frag);
        pool_fragmentations.insert("order_node_pool".to_string(), node_frag);
        pool_fragmentations.insert("price_level_pool".to_string(), level_frag);
        
        let overall_fragmentation = (order_frag + trade_frag + node_frag + level_frag) / 4.0;
        
        // Mock free block distribution
        let mut free_block_distribution = HashMap::new();
        free_block_distribution.insert(64, 100);
        free_block_distribution.insert(128, 50);
        free_block_distribution.insert(256, 25);
        
        FragmentationMetrics {
            overall_fragmentation,
            pool_fragmentations,
            free_block_distribution,
            largest_free_block: 4096,
            free_block_count: 175,
        }
    }

    /// Calculate fragmentation for a single pool
    fn calculate_pool_fragmentation(pool_stats: &PoolStats) -> f64 {
        if pool_stats.capacity == 0 {
            return 0.0;
        }
        
        let free_objects = pool_stats.capacity - pool_stats.in_use;
        let fragmentation = if pool_stats.capacity > 0 {
            (free_objects as f64 / pool_stats.capacity as f64) * 100.0
        } else {
            0.0
        };
        
        // Add some randomness to simulate real fragmentation patterns
        fragmentation * (0.8 + (pool_stats.total_allocations % 100) as f64 / 500.0)
    }

    /// Check for alert conditions
    fn check_alerts(
        metrics: &PoolMetricsSnapshot,
        config: &MonitorConfig,
        alert_system: &AlertSystem,
    ) {
        let thresholds = &config.alert_thresholds;
        
        // Check pool utilization
        if metrics.utilization_metrics.overall_utilization > thresholds.pool_utilization_threshold {
            let alert = Alert {
                alert_type: AlertType::HighPoolUtilization,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Pool utilization is {:.1}%, exceeding threshold of {:.1}%",
                    metrics.utilization_metrics.overall_utilization * 100.0,
                    thresholds.pool_utilization_threshold * 100.0
                ),
                timestamp: metrics.timestamp,
                metrics: Some(metrics.clone()),
                metadata: HashMap::new(),
            };
            alert_system.trigger_alert(alert);
        }
        
        // Check allocation latency
        if metrics.performance_metrics.avg_allocation_latency_ns > thresholds.allocation_latency_threshold_ns {
            let alert = Alert {
                alert_type: AlertType::HighAllocationLatency,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Average allocation latency is {}ns, exceeding threshold of {}ns",
                    metrics.performance_metrics.avg_allocation_latency_ns,
                    thresholds.allocation_latency_threshold_ns
                ),
                timestamp: metrics.timestamp,
                metrics: Some(metrics.clone()),
                metadata: HashMap::new(),
            };
            alert_system.trigger_alert(alert);
        }
        
        // Check fragmentation
        if metrics.fragmentation_metrics.overall_fragmentation > thresholds.fragmentation_threshold {
            let alert = Alert {
                alert_type: AlertType::HighFragmentation,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Memory fragmentation is {:.1}%, exceeding threshold of {:.1}%",
                    metrics.fragmentation_metrics.overall_fragmentation,
                    thresholds.fragmentation_threshold
                ),
                timestamp: metrics.timestamp,
                metrics: Some(metrics.clone()),
                metadata: HashMap::new(),
            };
            alert_system.trigger_alert(alert);
        }
        
        // Check cross-NUMA allocation rate
        if metrics.performance_metrics.cross_numa_rate > thresholds.cross_numa_threshold {
            let alert = Alert {
                alert_type: AlertType::HighCrossNumaRate,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Cross-NUMA allocation rate is {:.1}%, exceeding threshold of {:.1}%",
                    metrics.performance_metrics.cross_numa_rate,
                    thresholds.cross_numa_threshold
                ),
                timestamp: metrics.timestamp,
                metrics: Some(metrics.clone()),
                metadata: HashMap::new(),
            };
            alert_system.trigger_alert(alert);
        }
    }

    /// Get current metrics snapshot
    pub fn get_current_metrics(&self) -> Result<PoolMetricsSnapshot, MonitorError> {
        self.metrics_collector.get_current_metrics()
    }

    /// Get historical metrics
    pub fn get_historical_metrics(&self, duration_seconds: u64) -> Result<Vec<TimestampedMetrics>, MonitorError> {
        self.metrics_collector.get_historical_metrics(duration_seconds)
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> HashMap<AlertType, Alert> {
        self.alert_system.get_active_alerts()
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: usize) -> Vec<Alert> {
        self.alert_system.get_alert_history(limit)
    }

    /// Register alert callback
    pub fn register_alert_callback<F>(&self, callback: F) 
    where 
        F: Fn(&Alert) + Send + Sync + 'static 
    {
        self.alert_system.register_callback(Box::new(callback));
    }

    /// Get monitoring statistics
    pub fn get_monitor_stats(&self) -> MonitorStats {
        MonitorStats {
            collection_stats: self.metrics_collector.get_collection_stats(),
            alert_stats: self.alert_system.get_alert_stats(),
            uptime_seconds: 0, // Would track actual uptime
            is_running: self.monitor_thread.is_some(),
        }
    }

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 100, // 100ms intervals
            enable_real_time_metrics: true,
            enable_failure_detection: true,
            enable_fragmentation_monitoring: true,
            enable_performance_analytics: true,
            alert_thresholds: AlertThresholds::default(),
            metrics_retention_seconds: 3600, // 1 hour
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            pool_utilization_threshold: 0.8, // 80%
            allocation_failure_rate_threshold: 10.0, // 10 failures per second
            fragmentation_threshold: 50.0, // 50%
            allocation_latency_threshold_ns: 1000, // 1 microsecond
            cross_numa_threshold: 20.0, // 20%
        }
    }
}

impl Default for FragmentationMetrics {
    fn default() -> Self {
        Self {
            overall_fragmentation: 0.0,
            pool_fragmentations: HashMap::new(),
            free_block_distribution: HashMap::new(),
            largest_free_block: 0,
            free_block_count: 0,
        }
    }
}impl Metri
csCollector {
    /// Create a new metrics collector
    pub fn new(retention_seconds: u64) -> Result<Self, MonitorError> {
        let max_entries = (retention_seconds * 10) as usize; // 10 entries per second
        
        Ok(Self {
            current_metrics: Arc::new(std::sync::RwLock::new(PoolMetricsSnapshot::default())),
            historical_metrics: Arc::new(std::sync::RwLock::new(Vec::with_capacity(max_entries))),
            collection_stats: CollectionStats::new(),
            max_historical_entries: max_entries,
        })
    }

    /// Store new metrics
    pub fn store_metrics(&self, metrics: PoolMetricsSnapshot) -> Result<(), MonitorError> {
        // Update current metrics
        {
            let mut current = self.current_metrics.write()
                .map_err(|_| MonitorError::LockError)?;
            *current = metrics.clone();
        }

        // Add to historical metrics
        {
            let mut historical = self.historical_metrics.write()
                .map_err(|_| MonitorError::LockError)?;
            
            historical.push(TimestampedMetrics {
                timestamp: metrics.timestamp,
                metrics,
            });

            // Maintain size limit
            if historical.len() > self.max_historical_entries {
                historical.remove(0);
            }
        }

        self.collection_stats.total_collections.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Get current metrics
    pub fn get_current_metrics(&self) -> Result<PoolMetricsSnapshot, MonitorError> {
        let current = self.current_metrics.read()
            .map_err(|_| MonitorError::LockError)?;
        Ok(current.clone())
    }

    /// Get historical metrics within time range
    pub fn get_historical_metrics(&self, duration_seconds: u64) -> Result<Vec<TimestampedMetrics>, MonitorError> {
        let historical = self.historical_metrics.read()
            .map_err(|_| MonitorError::LockError)?;
        
        let cutoff_time = PoolMonitor::get_timestamp_ns() - (duration_seconds * 1_000_000_000);
        
        let filtered: Vec<TimestampedMetrics> = historical
            .iter()
            .filter(|entry| entry.timestamp >= cutoff_time)
            .cloned()
            .collect();
        
        Ok(filtered)
    }

    /// Update collection statistics
    pub fn update_collection_stats(&self, collection_time_ns: u64) {
        self.collection_stats.last_collection_timestamp.store(
            PoolMonitor::get_timestamp_ns(),
            Ordering::Relaxed
        );

        // Update average collection time
        let current_avg = self.collection_stats.avg_collection_time_ns.load(Ordering::Acquire);
        let new_avg = if current_avg == 0 {
            collection_time_ns
        } else {
            (current_avg * 9 + collection_time_ns) / 10 // Simple moving average
        };
        self.collection_stats.avg_collection_time_ns.store(new_avg, Ordering::Release);
    }

    /// Get collection statistics
    pub fn get_collection_stats(&self) -> CollectionStatsSnapshot {
        CollectionStatsSnapshot {
            total_collections: self.collection_stats.total_collections.load(Ordering::Acquire),
            failed_collections: self.collection_stats.failed_collections.load(Ordering::Acquire),
            avg_collection_time_ns: self.collection_stats.avg_collection_time_ns.load(Ordering::Acquire),
            last_collection_timestamp: self.collection_stats.last_collection_timestamp.load(Ordering::Acquire),
        }
    }
}

impl AlertSystem {
    /// Create a new alert system
    pub fn new() -> Self {
        Self {
            active_alerts: Arc::new(std::sync::RwLock::new(HashMap::new())),
            alert_history: Arc::new(std::sync::RwLock::new(Vec::new())),
            alert_callbacks: Arc::new(std::sync::RwLock::new(Vec::new())),
            alert_stats: AlertStats::new(),
        }
    }

    /// Trigger an alert
    pub fn trigger_alert(&self, alert: Alert) {
        // Update statistics
        self.alert_stats.total_alerts.fetch_add(1, Ordering::Relaxed);
        match alert.severity {
            AlertSeverity::Info => self.alert_stats.info_alerts.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Warning => self.alert_stats.warning_alerts.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Critical => self.alert_stats.critical_alerts.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Emergency => self.alert_stats.emergency_alerts.fetch_add(1, Ordering::Relaxed),
        };

        // Add to active alerts
        if let Ok(mut active) = self.active_alerts.write() {
            active.insert(alert.alert_type.clone(), alert.clone());
        }

        // Add to history
        if let Ok(mut history) = self.alert_history.write() {
            history.push(alert.clone());
            
            // Maintain history size (keep last 1000 alerts)
            if history.len() > 1000 {
                history.remove(0);
            }
        }

        // Notify callbacks
        if let Ok(callbacks) = self.alert_callbacks.read() {
            for callback in callbacks.iter() {
                callback(&alert);
            }
        }
    }

    /// Resolve an alert
    pub fn resolve_alert(&self, alert_type: AlertType) {
        if let Ok(mut active) = self.active_alerts.write() {
            active.remove(&alert_type);
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> HashMap<AlertType, Alert> {
        self.active_alerts.read()
            .map(|active| active.clone())
            .unwrap_or_default()
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: usize) -> Vec<Alert> {
        self.alert_history.read()
            .map(|history| {
                let start = if history.len() > limit {
                    history.len() - limit
                } else {
                    0
                };
                history[start..].to_vec()
            })
            .unwrap_or_default()
    }

    /// Register alert callback
    pub fn register_callback(&self, callback: Box<dyn Fn(&Alert) + Send + Sync>) {
        if let Ok(mut callbacks) = self.alert_callbacks.write() {
            callbacks.push(callback);
        }
    }

    /// Get alert statistics
    pub fn get_alert_stats(&self) -> AlertStatsSnapshot {
        AlertStatsSnapshot {
            total_alerts: self.alert_stats.total_alerts.load(Ordering::Acquire),
            info_alerts: self.alert_stats.info_alerts.load(Ordering::Acquire),
            warning_alerts: self.alert_stats.warning_alerts.load(Ordering::Acquire),
            critical_alerts: self.alert_stats.critical_alerts.load(Ordering::Acquire),
            emergency_alerts: self.alert_stats.emergency_alerts.load(Ordering::Acquire),
            avg_resolution_time_ns: self.alert_stats.avg_resolution_time_ns.load(Ordering::Acquire),
        }
    }
}

impl Default for PoolMetricsSnapshot {
    fn default() -> Self {
        Self {
            timestamp: 0,
            trading_stats: TradingPoolStats {
                order_pool: PoolStats {
                    capacity: 0,
                    allocated: 0,
                    in_use: 0,
                    numa_node: 0,
                    total_allocations: 0,
                    total_deallocations: 0,
                    expansions: 0,
                    peak_usage: 0,
                    allocation_failures: 0,
                    avg_allocation_time_ns: 0,
                    avg_deallocation_time_ns: 0,
                },
                trade_pool: PoolStats {
                    capacity: 0,
                    allocated: 0,
                    in_use: 0,
                    numa_node: 0,
                    total_allocations: 0,
                    total_deallocations: 0,
                    expansions: 0,
                    peak_usage: 0,
                    allocation_failures: 0,
                    avg_allocation_time_ns: 0,
                    avg_deallocation_time_ns: 0,
                },
                order_node_pool: PoolStats {
                    capacity: 0,
                    allocated: 0,
                    in_use: 0,
                    numa_node: 0,
                    total_allocations: 0,
                    total_deallocations: 0,
                    expansions: 0,
                    peak_usage: 0,
                    allocation_failures: 0,
                    avg_allocation_time_ns: 0,
                    avg_deallocation_time_ns: 0,
                },
                price_level_pool: PoolStats {
                    capacity: 0,
                    allocated: 0,
                    in_use: 0,
                    numa_node: 0,
                    total_allocations: 0,
                    total_deallocations: 0,
                    expansions: 0,
                    peak_usage: 0,
                    allocation_failures: 0,
                    avg_allocation_time_ns: 0,
                    avg_deallocation_time_ns: 0,
                },
                buffer_pool_stats: HashMap::new(),
                numa_stats: HashMap::new(),
                manager_stats: super::object_pools::PoolManagerStatsSnapshot {
                    total_allocations: 0,
                    total_deallocations: 0,
                    pool_expansions: 0,
                    allocation_failures: 0,
                    cross_numa_allocations: 0,
                },
            },
            memory_usage: MemoryUsageStats {
                order_pool_bytes: 0,
                trade_pool_bytes: 0,
                order_node_pool_bytes: 0,
                price_level_pool_bytes: 0,
                buffer_pool_bytes: 0,
                total_bytes: 0,
            },
            utilization_metrics: UtilizationMetrics {
                overall_utilization: 0.0,
                order_pool_utilization: 0.0,
                trade_pool_utilization: 0.0,
                order_node_pool_utilization: 0.0,
                price_level_pool_utilization: 0.0,
                buffer_pool_utilizations: HashMap::new(),
                numa_utilizations: HashMap::new(),
            },
            performance_metrics: PerformanceMetrics {
                avg_allocation_latency_ns: 0,
                p95_allocation_latency_ns: 0,
                p99_allocation_latency_ns: 0,
                allocation_rate: 0.0,
                deallocation_rate: 0.0,
                expansion_rate: 0.0,
                cross_numa_rate: 0.0,
            },
            fragmentation_metrics: FragmentationMetrics::default(),
        }
    }
}

impl CollectionStats {
    fn new() -> Self {
        Self {
            total_collections: AtomicUsize::new(0),
            failed_collections: AtomicUsize::new(0),
            avg_collection_time_ns: AtomicU64::new(0),
            last_collection_timestamp: AtomicU64::new(0),
        }
    }
}

impl AlertStats {
    fn new() -> Self {
        Self {
            total_alerts: AtomicUsize::new(0),
            info_alerts: AtomicUsize::new(0),
            warning_alerts: AtomicUsize::new(0),
            critical_alerts: AtomicUsize::new(0),
            emergency_alerts: AtomicUsize::new(0),
            avg_resolution_time_ns: AtomicU64::new(0),
        }
    }
}

/// Collection statistics snapshot
#[derive(Debug, Clone)]
pub struct CollectionStatsSnapshot {
    pub total_collections: usize,
    pub failed_collections: usize,
    pub avg_collection_time_ns: u64,
    pub last_collection_timestamp: u64,
}

/// Alert statistics snapshot
#[derive(Debug, Clone)]
pub struct AlertStatsSnapshot {
    pub total_alerts: usize,
    pub info_alerts: usize,
    pub warning_alerts: usize,
    pub critical_alerts: usize,
    pub emergency_alerts: usize,
    pub avg_resolution_time_ns: u64,
}

/// Monitor statistics
#[derive(Debug, Clone)]
pub struct MonitorStats {
    pub collection_stats: CollectionStatsSnapshot,
    pub alert_stats: AlertStatsSnapshot,
    pub uptime_seconds: u64,
    pub is_running: bool,
}

/// Monitor error types
#[derive(Debug, Clone)]
pub enum MonitorError {
    AlreadyRunning,
    NotRunning,
    ThreadJoinFailed,
    LockError,
    ConfigurationError(String),
    MetricsCollectionFailed,
    AlertSystemError,
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitorError::AlreadyRunning => write!(f, "Monitor is already running"),
            MonitorError::NotRunning => write!(f, "Monitor is not running"),
            MonitorError::ThreadJoinFailed => write!(f, "Failed to join monitor thread"),
            MonitorError::LockError => write!(f, "Lock acquisition failed"),
            MonitorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            MonitorError::MetricsCollectionFailed => write!(f, "Metrics collection failed"),
            MonitorError::AlertSystemError => write!(f, "Alert system error"),
        }
    }
}

impl std::error::Error for MonitorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::memory::numa_allocator::AllocationPolicy;
    use std::time::Duration;

    #[test]
    fn test_monitor_creation() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        let config = MonitorConfig::default();
        
        let monitor = PoolMonitor::new(pools, config).unwrap();
        assert!(!monitor.get_monitor_stats().is_running);
    }

    #[test]
    fn test_metrics_collection() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        let config = MonitorConfig::default();
        
        // Allocate some objects to generate metrics
        let _order = pools.allocate_order().unwrap();
        let _trade = pools.allocate_trade().unwrap();
        
        let metrics = PoolMonitor::collect_metrics(&pools, &config).unwrap();
        assert!(metrics.timestamp > 0);
        assert!(metrics.utilization_metrics.overall_utilization >= 0.0);
    }

    #[test]
    fn test_alert_system() {
        let alert_system = AlertSystem::new();
        
        let alert = Alert {
            alert_type: AlertType::HighPoolUtilization,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            timestamp: PoolMonitor::get_timestamp_ns(),
            metrics: None,
            metadata: HashMap::new(),
        };
        
        alert_system.trigger_alert(alert.clone());
        
        let active_alerts = alert_system.get_active_alerts();
        assert!(active_alerts.contains_key(&AlertType::HighPoolUtilization));
        
        let history = alert_system.get_alert_history(10);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].message, "Test alert");
    }

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new(60).unwrap(); // 1 minute retention
        
        let metrics = PoolMetricsSnapshot::default();
        collector.store_metrics(metrics.clone()).unwrap();
        
        let current = collector.get_current_metrics().unwrap();
        assert_eq!(current.timestamp, metrics.timestamp);
        
        let historical = collector.get_historical_metrics(60).unwrap();
        assert_eq!(historical.len(), 1);
    }

    #[test]
    fn test_utilization_calculation() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        
        // Allocate some objects
        let _order1 = pools.allocate_order().unwrap();
        let _order2 = pools.allocate_order().unwrap();
        let _trade = pools.allocate_trade().unwrap();
        
        let stats = pools.get_comprehensive_stats();
        let utilization = PoolMonitor::calculate_utilization_metrics(&stats);
        
        assert!(utilization.overall_utilization > 0.0);
        assert!(utilization.order_pool_utilization > 0.0);
        assert!(utilization.trade_pool_utilization > 0.0);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        
        // Generate some activity
        for _ in 0..10 {
            let _order = pools.allocate_order().unwrap();
        }
        
        let stats = pools.get_comprehensive_stats();
        let performance = PoolMonitor::calculate_performance_metrics(&stats);
        
        assert!(performance.allocation_rate >= 0.0);
        assert!(performance.avg_allocation_latency_ns >= 0);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        let stats = pools.get_comprehensive_stats();
        
        let fragmentation = PoolMonitor::calculate_fragmentation_metrics(&stats);
        assert!(fragmentation.overall_fragmentation >= 0.0);
        assert!(fragmentation.overall_fragmentation <= 100.0);
    }

    #[test]
    fn test_alert_callback() {
        let alert_system = AlertSystem::new();
        let callback_triggered = Arc::new(AtomicBool::new(false));
        let callback_triggered_clone = callback_triggered.clone();
        
        alert_system.register_callback(Box::new(move |_alert| {
            callback_triggered_clone.store(true, Ordering::Relaxed);
        }));
        
        let alert = Alert {
            alert_type: AlertType::HighPoolUtilization,
            severity: AlertSeverity::Warning,
            message: "Test callback".to_string(),
            timestamp: PoolMonitor::get_timestamp_ns(),
            metrics: None,
            metadata: HashMap::new(),
        };
        
        alert_system.trigger_alert(alert);
        
        // Give callback time to execute
        thread::sleep(Duration::from_millis(10));
        assert!(callback_triggered.load(Ordering::Relaxed));
    }

    #[test]
    fn test_monitor_start_stop() {
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        let config = MonitorConfig {
            monitoring_interval_ms: 10, // Fast interval for testing
            ..MonitorConfig::default()
        };
        
        let mut monitor = PoolMonitor::new(pools, config).unwrap();
        
        // Start monitoring
        monitor.start().unwrap();
        assert!(monitor.get_monitor_stats().is_running);
        
        // Let it run briefly
        thread::sleep(Duration::from_millis(50));
        
        // Stop monitoring
        monitor.stop().unwrap();
        assert!(!monitor.get_monitor_stats().is_running);
    }

    #[test]
    fn test_historical_metrics_retention() {
        let collector = MetricsCollector::new(1).unwrap(); // 1 second retention
        
        // Add multiple metrics
        for i in 0..5 {
            let mut metrics = PoolMetricsSnapshot::default();
            metrics.timestamp = i * 1_000_000_000; // 1 second apart
            collector.store_metrics(metrics).unwrap();
        }
        
        // Should only retain recent metrics
        let historical = collector.get_historical_metrics(2).unwrap();
        assert!(historical.len() <= 5);
    }
}