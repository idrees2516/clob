use super::nanosecond_timer::{NanosecondTimer, now_nanos};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;

/// Real-time performance metrics collector with lock-free counters
/// Designed for zero-overhead metrics collection in high-frequency trading
pub struct PerformanceMetrics {
    /// Metrics storage
    metrics: HashMap<String, Arc<MetricCounter>>,
    
    /// Timer for timestamp generation
    timer: Arc<NanosecondTimer>,
    
    /// Global metrics registry
    registry: Arc<MetricsRegistry>,
}

/// Lock-free metric counter for high-performance updates
#[repr(align(64))] // Cache-line aligned
pub struct MetricCounter {
    /// Counter value
    value: AtomicU64,
    
    /// Counter type
    counter_type: CounterType,
    
    /// Last update timestamp
    last_update: AtomicU64,
    
    /// Update count
    update_count: AtomicU64,
    
    /// Minimum value seen
    min_value: AtomicU64,
    
    /// Maximum value seen
    max_value: AtomicU64,
    
    /// Sum for average calculation
    sum: AtomicU64,
    
    /// Sum of squares for variance calculation
    sum_squares: AtomicU64,
}

/// Types of metric counters
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CounterType {
    /// Monotonic counter (always increasing)
    Counter,
    
    /// Gauge (can increase or decrease)
    Gauge,
    
    /// Histogram for latency measurements
    Histogram,
    
    /// Rate counter (events per second)
    Rate,
}

/// Metrics registry for managing all metrics
pub struct MetricsRegistry {
    /// All registered metrics
    metrics: std::sync::RwLock<HashMap<String, Arc<MetricCounter>>>,
    
    /// Registry statistics
    stats: RegistryStats,
}

/// Registry statistics
#[repr(align(64))]
pub struct RegistryStats {
    /// Total metrics registered
    pub total_metrics: AtomicUsize,
    
    /// Total updates across all metrics
    pub total_updates: AtomicU64,
    
    /// Registry creation time
    pub created_at: u64,
    
    /// Last cleanup time
    pub last_cleanup: AtomicU64,
}

impl PerformanceMetrics {
    /// Create a new performance metrics collector
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            timer: Arc::new(NanosecondTimer::new().expect("Failed to create timer")),
            registry: Arc::new(MetricsRegistry::new()),
        }
    }

    /// Register a new metric counter
    pub fn register_counter(&mut self, name: String, counter_type: CounterType) -> Arc<MetricCounter> {
        let counter = Arc::new(MetricCounter::new(counter_type));
        self.metrics.insert(name.clone(), counter.clone());
        self.registry.register_metric(name, counter.clone());
        counter
    }

    /// Get or create a metric counter
    pub fn get_or_create_counter(&mut self, name: &str, counter_type: CounterType) -> Arc<MetricCounter> {
        if let Some(counter) = self.metrics.get(name) {
            counter.clone()
        } else {
            self.register_counter(name.to_string(), counter_type)
        }
    }

    /// Record a latency measurement
    #[inline(always)]
    pub fn record_latency(&self, name: &str, latency_ns: u64) {
        if let Some(counter) = self.metrics.get(name) {
            counter.record_value(latency_ns);
        }
    }

    /// Increment a counter
    #[inline(always)]
    pub fn increment_counter(&self, name: &str) {
        if let Some(counter) = self.metrics.get(name) {
            counter.increment();
        }
    }

    /// Add to a counter
    #[inline(always)]
    pub fn add_to_counter(&self, name: &str, value: u64) {
        if let Some(counter) = self.metrics.get(name) {
            counter.add(value);
        }
    }

    /// Set a gauge value
    #[inline(always)]
    pub fn set_gauge(&self, name: &str, value: u64) {
        if let Some(counter) = self.metrics.get(name) {
            counter.set(value);
        }
    }

    /// Get current value of a metric
    pub fn get_value(&self, name: &str) -> Option<u64> {
        self.metrics.get(name).map(|counter| counter.get())
    }

    /// Get comprehensive statistics for a metric
    pub fn get_stats(&self, name: &str) -> Option<MetricStats> {
        self.metrics.get(name).map(|counter| counter.get_stats())
    }

    /// Get all metric statistics
    pub fn get_all_stats(&self) -> HashMap<String, MetricStats> {
        self.metrics
            .iter()
            .map(|(name, counter)| (name.clone(), counter.get_stats()))
            .collect()
    }

    /// Reset all metrics
    pub fn reset_all(&self) {
        for counter in self.metrics.values() {
            counter.reset();
        }
    }

    /// Get registry statistics
    pub fn get_registry_stats(&self) -> RegistryStatsSnapshot {
        self.registry.get_stats()
    }
}

impl MetricCounter {
    /// Create a new metric counter
    pub fn new(counter_type: CounterType) -> Self {
        Self {
            value: AtomicU64::new(0),
            counter_type,
            last_update: AtomicU64::new(now_nanos()),
            update_count: AtomicU64::new(0),
            min_value: AtomicU64::new(u64::MAX),
            max_value: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            sum_squares: AtomicU64::new(0),
        }
    }

    /// Increment the counter by 1
    #[inline(always)]
    pub fn increment(&self) {
        self.add(1);
    }

    /// Add a value to the counter
    #[inline(always)]
    pub fn add(&self, value: u64) {
        match self.counter_type {
            CounterType::Counter | CounterType::Rate => {
                self.value.fetch_add(value, Ordering::Relaxed);
            }
            CounterType::Gauge => {
                self.value.store(value, Ordering::Relaxed);
            }
            CounterType::Histogram => {
                self.record_value(value);
                return; // record_value handles the update
            }
        }
        
        self.update_metadata(value);
    }

    /// Set the counter to a specific value (for gauges)
    #[inline(always)]
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
        self.update_metadata(value);
    }

    /// Record a value for histogram metrics
    #[inline(always)]
    pub fn record_value(&self, value: u64) {
        // Update sum and count for average calculation
        self.sum.fetch_add(value, Ordering::Relaxed);
        self.update_count.fetch_add(1, Ordering::Relaxed);
        
        // Update min/max
        self.update_min_max(value);
        
        // Update sum of squares for variance (with overflow protection)
        let squares = value.saturating_mul(value);
        self.sum_squares.fetch_add(squares, Ordering::Relaxed);
        
        self.last_update.store(now_nanos(), Ordering::Relaxed);
    }

    /// Get current counter value
    #[inline(always)]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get comprehensive statistics
    pub fn get_stats(&self) -> MetricStats {
        let value = self.value.load(Ordering::Acquire);
        let update_count = self.update_count.load(Ordering::Acquire);
        let sum = self.sum.load(Ordering::Acquire);
        let sum_squares = self.sum_squares.load(Ordering::Acquire);
        let min_value = self.min_value.load(Ordering::Acquire);
        let max_value = self.max_value.load(Ordering::Acquire);
        let last_update = self.last_update.load(Ordering::Acquire);

        let (average, variance, std_dev) = if update_count > 0 {
            let avg = sum as f64 / update_count as f64;
            let var = if update_count > 1 {
                let mean_squares = sum_squares as f64 / update_count as f64;
                let mean_squared = avg * avg;
                (mean_squares - mean_squared).max(0.0)
            } else {
                0.0
            };
            let std = var.sqrt();
            (avg, var, std)
        } else {
            (0.0, 0.0, 0.0)
        };

        MetricStats {
            counter_type: self.counter_type,
            current_value: value,
            update_count,
            sum,
            average,
            variance,
            std_deviation: std_dev,
            min_value: if min_value == u64::MAX { 0 } else { min_value },
            max_value,
            last_update_ns: last_update,
        }
    }

    /// Reset the counter
    pub fn reset(&self) {
        self.value.store(0, Ordering::Release);
        self.update_count.store(0, Ordering::Release);
        self.sum.store(0, Ordering::Release);
        self.sum_squares.store(0, Ordering::Release);
        self.min_value.store(u64::MAX, Ordering::Release);
        self.max_value.store(0, Ordering::Release);
        self.last_update.store(now_nanos(), Ordering::Release);
    }

    /// Update metadata (min/max, timestamps)
    #[inline(always)]
    fn update_metadata(&self, value: u64) {
        self.update_count.fetch_add(1, Ordering::Relaxed);
        self.update_min_max(value);
        self.last_update.store(now_nanos(), Ordering::Relaxed);
    }

    /// Update min/max values
    #[inline(always)]
    fn update_min_max(&self, value: u64) {
        // Update minimum
        let mut current_min = self.min_value.load(Ordering::Relaxed);
        while value < current_min {
            match self.min_value.compare_exchange_weak(
                current_min,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_min) => current_min = new_min,
            }
        }

        // Update maximum
        let mut current_max = self.max_value.load(Ordering::Relaxed);
        while value > current_max {
            match self.max_value.compare_exchange_weak(
                current_max,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_max) => current_max = new_max,
            }
        }
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            metrics: std::sync::RwLock::new(HashMap::new()),
            stats: RegistryStats {
                total_metrics: AtomicUsize::new(0),
                total_updates: AtomicU64::new(0),
                created_at: now_nanos(),
                last_cleanup: AtomicU64::new(now_nanos()),
            },
        }
    }

    /// Register a metric
    pub fn register_metric(&self, name: String, counter: Arc<MetricCounter>) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.insert(name, counter);
        self.stats.total_metrics.store(metrics.len(), Ordering::Release);
    }

    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<Arc<MetricCounter>> {
        let metrics = self.metrics.read().unwrap();
        metrics.get(name).cloned()
    }

    /// Get all metric names
    pub fn get_metric_names(&self) -> Vec<String> {
        let metrics = self.metrics.read().unwrap();
        metrics.keys().cloned().collect()
    }

    /// Get registry statistics
    pub fn get_stats(&self) -> RegistryStatsSnapshot {
        RegistryStatsSnapshot {
            total_metrics: self.stats.total_metrics.load(Ordering::Acquire),
            total_updates: self.stats.total_updates.load(Ordering::Acquire),
            created_at: self.stats.created_at,
            last_cleanup: self.stats.last_cleanup.load(Ordering::Acquire),
        }
    }

    /// Cleanup unused metrics (optional)
    pub fn cleanup_unused(&self) {
        // In a real implementation, this would remove metrics with no recent updates
        self.stats.last_cleanup.store(now_nanos(), Ordering::Release);
    }
}

/// Metric statistics snapshot
#[derive(Debug, Clone)]
pub struct MetricStats {
    pub counter_type: CounterType,
    pub current_value: u64,
    pub update_count: u64,
    pub sum: u64,
    pub average: f64,
    pub variance: f64,
    pub std_deviation: f64,
    pub min_value: u64,
    pub max_value: u64,
    pub last_update_ns: u64,
}

/// Registry statistics snapshot
#[derive(Debug, Clone)]
pub struct RegistryStatsSnapshot {
    pub total_metrics: usize,
    pub total_updates: u64,
    pub created_at: u64,
    pub last_cleanup: u64,
}

/// Global metrics instance for high-performance access
static GLOBAL_METRICS: std::sync::OnceLock<Arc<std::sync::Mutex<PerformanceMetrics>>> = std::sync::OnceLock::new();

/// Get global metrics instance
pub fn global_metrics() -> &'static Arc<std::sync::Mutex<PerformanceMetrics>> {
    GLOBAL_METRICS.get_or_init(|| {
        Arc::new(std::sync::Mutex::new(PerformanceMetrics::new()))
    })
}

/// Convenience macros for metrics
#[macro_export]
macro_rules! increment_counter {
    ($name:expr) => {
        if let Ok(metrics) = global_metrics().try_lock() {
            metrics.increment_counter($name);
        }
    };
}

#[macro_export]
macro_rules! record_latency {
    ($name:expr, $latency:expr) => {
        if let Ok(metrics) = global_metrics().try_lock() {
            metrics.record_latency($name, $latency);
        }
    };
}

#[macro_export]
macro_rules! set_gauge {
    ($name:expr, $value:expr) => {
        if let Ok(metrics) = global_metrics().try_lock() {
            metrics.set_gauge($name, $value);
        }
    };
}

/// High-performance metrics collector for trading operations
pub struct TradingMetrics {
    /// Order processing latency
    pub order_latency: Arc<MetricCounter>,
    
    /// Trade execution latency
    pub trade_latency: Arc<MetricCounter>,
    
    /// Network latency
    pub network_latency: Arc<MetricCounter>,
    
    /// Orders per second
    pub orders_per_second: Arc<MetricCounter>,
    
    /// Trades per second
    pub trades_per_second: Arc<MetricCounter>,
    
    /// Memory usage
    pub memory_usage: Arc<MetricCounter>,
    
    /// CPU utilization
    pub cpu_utilization: Arc<MetricCounter>,
    
    /// Error count
    pub error_count: Arc<MetricCounter>,
}

impl TradingMetrics {
    /// Create trading-specific metrics
    pub fn new() -> Self {
        let mut metrics = PerformanceMetrics::new();
        
        Self {
            order_latency: metrics.register_counter("order_latency_ns".to_string(), CounterType::Histogram),
            trade_latency: metrics.register_counter("trade_latency_ns".to_string(), CounterType::Histogram),
            network_latency: metrics.register_counter("network_latency_ns".to_string(), CounterType::Histogram),
            orders_per_second: metrics.register_counter("orders_per_second".to_string(), CounterType::Rate),
            trades_per_second: metrics.register_counter("trades_per_second".to_string(), CounterType::Rate),
            memory_usage: metrics.register_counter("memory_usage_bytes".to_string(), CounterType::Gauge),
            cpu_utilization: metrics.register_counter("cpu_utilization_percent".to_string(), CounterType::Gauge),
            error_count: metrics.register_counter("error_count".to_string(), CounterType::Counter),
        }
    }

    /// Record order processing latency
    #[inline(always)]
    pub fn record_order_latency(&self, latency_ns: u64) {
        self.order_latency.record_value(latency_ns);
    }

    /// Record trade execution latency
    #[inline(always)]
    pub fn record_trade_latency(&self, latency_ns: u64) {
        self.trade_latency.record_value(latency_ns);
    }

    /// Record network latency
    #[inline(always)]
    pub fn record_network_latency(&self, latency_ns: u64) {
        self.network_latency.record_value(latency_ns);
    }

    /// Update orders per second
    #[inline(always)]
    pub fn update_orders_per_second(&self, count: u64) {
        self.orders_per_second.set(count);
    }

    /// Update trades per second
    #[inline(always)]
    pub fn update_trades_per_second(&self, count: u64) {
        self.trades_per_second.set(count);
    }

    /// Update memory usage
    #[inline(always)]
    pub fn update_memory_usage(&self, bytes: u64) {
        self.memory_usage.set(bytes);
    }

    /// Update CPU utilization
    #[inline(always)]
    pub fn update_cpu_utilization(&self, percent: u64) {
        self.cpu_utilization.set(percent);
    }

    /// Increment error count
    #[inline(always)]
    pub fn increment_errors(&self) {
        self.error_count.increment();
    }

    /// Get comprehensive trading statistics
    pub fn get_trading_stats(&self) -> TradingStatsSnapshot {
        TradingStatsSnapshot {
            order_latency: self.order_latency.get_stats(),
            trade_latency: self.trade_latency.get_stats(),
            network_latency: self.network_latency.get_stats(),
            orders_per_second: self.orders_per_second.get(),
            trades_per_second: self.trades_per_second.get(),
            memory_usage_bytes: self.memory_usage.get(),
            cpu_utilization_percent: self.cpu_utilization.get(),
            error_count: self.error_count.get(),
        }
    }
}

/// Trading statistics snapshot
#[derive(Debug, Clone)]
pub struct TradingStatsSnapshot {
    pub order_latency: MetricStats,
    pub trade_latency: MetricStats,
    pub network_latency: MetricStats,
    pub orders_per_second: u64,
    pub trades_per_second: u64,
    pub memory_usage_bytes: u64,
    pub cpu_utilization_percent: u64,
    pub error_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metric_counter_creation() {
        let counter = MetricCounter::new(CounterType::Counter);
        assert_eq!(counter.get(), 0);
        
        let stats = counter.get_stats();
        assert_eq!(stats.counter_type, CounterType::Counter);
        assert_eq!(stats.current_value, 0);
    }

    #[test]
    fn test_counter_increment() {
        let counter = MetricCounter::new(CounterType::Counter);
        
        counter.increment();
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        let stats = counter.get_stats();
        assert_eq!(stats.update_count, 2);
    }

    #[test]
    fn test_gauge_operations() {
        let gauge = MetricCounter::new(CounterType::Gauge);
        
        gauge.set(100);
        assert_eq!(gauge.get(), 100);
        
        gauge.set(50);
        assert_eq!(gauge.get(), 50);
        
        let stats = gauge.get_stats();
        assert_eq!(stats.update_count, 2);
    }

    #[test]
    fn test_histogram_statistics() {
        let histogram = MetricCounter::new(CounterType::Histogram);
        
        // Record some values
        let values = vec![10, 20, 30, 40, 50];
        for value in &values {
            histogram.record_value(*value);
        }
        
        let stats = histogram.get_stats();
        assert_eq!(stats.update_count, 5);
        assert_eq!(stats.sum, 150);
        assert_eq!(stats.average, 30.0);
        assert_eq!(stats.min_value, 10);
        assert_eq!(stats.max_value, 50);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        let counter = metrics.register_counter("test_counter".to_string(), CounterType::Counter);
        counter.increment();
        
        assert_eq!(metrics.get_value("test_counter"), Some(1));
        
        let stats = metrics.get_stats("test_counter").unwrap();
        assert_eq!(stats.current_value, 1);
    }

    #[test]
    fn test_trading_metrics() {
        let trading_metrics = TradingMetrics::new();
        
        trading_metrics.record_order_latency(1000);
        trading_metrics.record_trade_latency(2000);
        trading_metrics.update_orders_per_second(100);
        trading_metrics.increment_errors();
        
        let stats = trading_metrics.get_trading_stats();
        assert_eq!(stats.order_latency.update_count, 1);
        assert_eq!(stats.trade_latency.update_count, 1);
        assert_eq!(stats.orders_per_second, 100);
        assert_eq!(stats.error_count, 1);
    }

    #[test]
    fn test_concurrent_metrics() {
        let counter = Arc::new(MetricCounter::new(CounterType::Counter));
        let mut handles = vec![];

        // Spawn threads to increment concurrently
        for _ in 0..4 {
            let counter_clone = counter.clone();
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    counter_clone.increment();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(counter.get(), 4000);
        
        let stats = counter.get_stats();
        assert_eq!(stats.update_count, 4000);
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();
        let counter = Arc::new(MetricCounter::new(CounterType::Counter));
        
        registry.register_metric("test".to_string(), counter.clone());
        
        let retrieved = registry.get_metric("test").unwrap();
        assert!(Arc::ptr_eq(&counter, &retrieved));
        
        let names = registry.get_metric_names();
        assert!(names.contains(&"test".to_string()));
    }

    #[test]
    fn test_min_max_tracking() {
        let histogram = MetricCounter::new(CounterType::Histogram);
        
        histogram.record_value(100);
        histogram.record_value(50);
        histogram.record_value(200);
        histogram.record_value(75);
        
        let stats = histogram.get_stats();
        assert_eq!(stats.min_value, 50);
        assert_eq!(stats.max_value, 200);
    }

    #[test]
    fn test_variance_calculation() {
        let histogram = MetricCounter::new(CounterType::Histogram);
        
        // Record values with known variance
        let values = vec![10, 20, 30];
        for value in values {
            histogram.record_value(value);
        }
        
        let stats = histogram.get_stats();
        assert_eq!(stats.average, 20.0);
        
        // Variance should be approximately 66.67
        assert!((stats.variance - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_reset_functionality() {
        let counter = MetricCounter::new(CounterType::Counter);
        
        counter.add(100);
        assert_eq!(counter.get(), 100);
        
        counter.reset();
        assert_eq!(counter.get(), 0);
        
        let stats = counter.get_stats();
        assert_eq!(stats.update_count, 0);
        assert_eq!(stats.sum, 0);
    }

    #[test]
    fn test_high_frequency_updates() {
        let counter = Arc::new(MetricCounter::new(CounterType::Histogram));
        let start_time = now_nanos();
        
        // Perform many updates quickly
        for i in 0..10000 {
            counter.record_value(i % 100);
        }
        
        let end_time = now_nanos();
        let elapsed_ns = end_time - start_time;
        
        // Should complete quickly (less than 10ms)
        assert!(elapsed_ns < 10_000_000);
        
        let stats = counter.get_stats();
        assert_eq!(stats.update_count, 10000);
    }
}