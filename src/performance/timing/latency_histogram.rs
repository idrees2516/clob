use super::nanosecond_timer::now_nanos;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// High-performance latency histogram for real-time metrics collection
/// Uses lock-free atomic operations for zero-contention updates
#[repr(align(64))] // Cache-line aligned
pub struct LatencyHistogram {
    /// Histogram buckets (powers of 2 for fast bucket calculation)
    buckets: Vec<AtomicU64>,
    
    /// Bucket boundaries in nanoseconds
    bucket_boundaries: Vec<u64>,
    
    /// Total sample count
    total_count: AtomicU64,
    
    /// Sum of all values for average calculation
    total_sum: AtomicU64,
    
    /// Minimum value observed
    min_value: AtomicU64,
    
    /// Maximum value observed
    max_value: AtomicU64,
    
    /// Last update timestamp
    last_update: AtomicU64,
    
    /// Overflow bucket for values exceeding max boundary
    overflow_count: AtomicU64,
}

/// Percentile calculation result
#[derive(Debug, Clone)]
pub struct PercentileResult {
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub p99_9: u64,
    pub p99_99: u64,
}

/// Histogram statistics snapshot
#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub total_count: u64,
    pub total_sum: u64,
    pub average: f64,
    pub min_value: u64,
    pub max_value: u64,
    pub percentiles: PercentileResult,
    pub overflow_count: u64,
    pub last_update_ns: u64,
}

impl LatencyHistogram {
    /// Create a new latency histogram with default buckets
    /// Buckets cover 1ns to 1s with exponential spacing
    pub fn new() -> Self {
        Self::with_buckets(Self::default_buckets())
    }

    /// Create histogram with custom bucket boundaries
    pub fn with_buckets(bucket_boundaries: Vec<u64>) -> Self {
        let bucket_count = bucket_boundaries.len();
        let mut buckets = Vec::with_capacity(bucket_count);
        
        for _ in 0..bucket_count {
            buckets.push(AtomicU64::new(0));
        }

        Self {
            buckets,
            bucket_boundaries,
            total_count: AtomicU64::new(0),
            total_sum: AtomicU64::new(0),
            min_value: AtomicU64::new(u64::MAX),
            max_value: AtomicU64::new(0),
            last_update: AtomicU64::new(now_nanos()),
            overflow_count: AtomicU64::new(0),
        }
    }

    /// Create histogram optimized for trading latencies (1ns to 100ms)
    pub fn for_trading() -> Self {
        let buckets = vec![
            // Sub-microsecond buckets (1ns to 1μs)
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000,
            // Microsecond buckets (1μs to 1ms)
            2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
            // Millisecond buckets (1ms to 100ms)
            2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000,
        ];
        Self::with_buckets(buckets)
    }

    /// Record a latency measurement
    #[inline(always)]
    pub fn record(&self, latency_ns: u64) {
        // Find appropriate bucket using binary search
        let bucket_index = self.find_bucket_index(latency_ns);
        
        if bucket_index < self.buckets.len() {
            self.buckets[bucket_index].fetch_add(1, Ordering::Relaxed);
        } else {
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
        }

        // Update aggregated statistics
        self.total_count.fetch_add(1, Ordering::Relaxed);
        self.total_sum.fetch_add(latency_ns, Ordering::Relaxed);
        self.update_min_max(latency_ns);
        self.last_update.store(now_nanos(), Ordering::Relaxed);
    }

    /// Record multiple measurements efficiently
    #[inline(always)]
    pub fn record_batch(&self, latencies: &[u64]) {
        for &latency in latencies {
            self.record(latency);
        }
    }

    /// Calculate percentiles from current histogram data
    pub fn calculate_percentiles(&self) -> PercentileResult {
        let total_count = self.total_count.load(Ordering::Acquire);
        
        if total_count == 0 {
            return PercentileResult {
                p50: 0, p90: 0, p95: 0, p99: 0, p99_9: 0, p99_99: 0,
            };
        }

        // Calculate target counts for each percentile
        let p50_target = (total_count as f64 * 0.50) as u64;
        let p90_target = (total_count as f64 * 0.90) as u64;
        let p95_target = (total_count as f64 * 0.95) as u64;
        let p99_target = (total_count as f64 * 0.99) as u64;
        let p99_9_target = (total_count as f64 * 0.999) as u64;
        let p99_99_target = (total_count as f64 * 0.9999) as u64;

        let mut cumulative_count = 0u64;
        let mut p50 = 0u64;
        let mut p90 = 0u64;
        let mut p95 = 0u64;
        let mut p99 = 0u64;
        let mut p99_9 = 0u64;
        let mut p99_99 = 0u64;

        // Iterate through buckets to find percentiles
        for (i, bucket) in self.buckets.iter().enumerate() {
            let bucket_count = bucket.load(Ordering::Acquire);
            cumulative_count += bucket_count;

            let bucket_value = if i < self.bucket_boundaries.len() {
                self.bucket_boundaries[i]
            } else {
                u64::MAX
            };

            if p50 == 0 && cumulative_count >= p50_target {
                p50 = bucket_value;
            }
            if p90 == 0 && cumulative_count >= p90_target {
                p90 = bucket_value;
            }
            if p95 == 0 && cumulative_count >= p95_target {
                p95 = bucket_value;
            }
            if p99 == 0 && cumulative_count >= p99_target {
                p99 = bucket_value;
            }
            if p99_9 == 0 && cumulative_count >= p99_9_target {
                p99_9 = bucket_value;
            }
            if p99_99 == 0 && cumulative_count >= p99_99_target {
                p99_99 = bucket_value;
            }
        }

        PercentileResult { p50, p90, p95, p99, p99_9, p99_99 }
    }

    /// Get comprehensive histogram statistics
    pub fn get_stats(&self) -> HistogramStats {
        let total_count = self.total_count.load(Ordering::Acquire);
        let total_sum = self.total_sum.load(Ordering::Acquire);
        let min_value = self.min_value.load(Ordering::Acquire);
        let max_value = self.max_value.load(Ordering::Acquire);
        let overflow_count = self.overflow_count.load(Ordering::Acquire);
        let last_update = self.last_update.load(Ordering::Acquire);

        let average = if total_count > 0 {
            total_sum as f64 / total_count as f64
        } else {
            0.0
        };

        let percentiles = self.calculate_percentiles();

        HistogramStats {
            total_count,
            total_sum,
            average,
            min_value: if min_value == u64::MAX { 0 } else { min_value },
            max_value,
            percentiles,
            overflow_count,
            last_update_ns: last_update,
        }
    }

    /// Reset histogram to initial state
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Release);
        }
        
        self.total_count.store(0, Ordering::Release);
        self.total_sum.store(0, Ordering::Release);
        self.min_value.store(u64::MAX, Ordering::Release);
        self.max_value.store(0, Ordering::Release);
        self.overflow_count.store(0, Ordering::Release);
        self.last_update.store(now_nanos(), Ordering::Release);
    }

    /// Get bucket counts (for debugging/analysis)
    pub fn get_bucket_counts(&self) -> Vec<u64> {
        self.buckets
            .iter()
            .map(|bucket| bucket.load(Ordering::Acquire))
            .collect()
    }

    /// Get bucket boundaries
    pub fn get_bucket_boundaries(&self) -> &[u64] {
        &self.bucket_boundaries
    }

    /// Find bucket index for a given value using binary search
    #[inline(always)]
    fn find_bucket_index(&self, value: u64) -> usize {
        // Binary search for the appropriate bucket
        let mut left = 0;
        let mut right = self.bucket_boundaries.len();

        while left < right {
            let mid = left + (right - left) / 2;
            if value <= self.bucket_boundaries[mid] {
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        left
    }

    /// Update min/max values atomically
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

    /// Default bucket boundaries for general-purpose latency measurement
    fn default_buckets() -> Vec<u64> {
        let mut buckets = Vec::new();
        
        // Nanosecond buckets: 1ns to 1μs
        for i in 0..10 {
            buckets.push(10u64.pow(i) * 1); // 1, 10, 100, 1000ns
        }
        
        // Microsecond buckets: 1μs to 1ms
        for i in 0..10 {
            buckets.push(10u64.pow(i) * 1_000); // 1, 10, 100, 1000μs
        }
        
        // Millisecond buckets: 1ms to 1s
        for i in 0..10 {
            buckets.push(10u64.pow(i) * 1_000_000); // 1, 10, 100, 1000ms
        }

        buckets.sort_unstable();
        buckets.dedup();
        buckets
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-threaded histogram for concurrent access
pub struct ConcurrentLatencyHistogram {
    /// Per-thread histograms to avoid contention
    thread_histograms: Vec<Arc<LatencyHistogram>>,
    
    /// Number of threads/shards
    shard_count: usize,
}

impl ConcurrentLatencyHistogram {
    /// Create a new concurrent histogram with specified shard count
    pub fn new(shard_count: usize) -> Self {
        let mut thread_histograms = Vec::with_capacity(shard_count);
        
        for _ in 0..shard_count {
            thread_histograms.push(Arc::new(LatencyHistogram::for_trading()));
        }

        Self {
            thread_histograms,
            shard_count,
        }
    }

    /// Record a latency measurement (automatically sharded)
    #[inline(always)]
    pub fn record(&self, latency_ns: u64) {
        let shard_index = self.get_shard_index();
        self.thread_histograms[shard_index].record(latency_ns);
    }

    /// Get aggregated statistics across all shards
    pub fn get_aggregated_stats(&self) -> HistogramStats {
        let mut total_count = 0u64;
        let mut total_sum = 0u64;
        let mut min_value = u64::MAX;
        let mut max_value = 0u64;
        let mut overflow_count = 0u64;
        let mut last_update = 0u64;

        // Aggregate basic statistics
        for histogram in &self.thread_histograms {
            let stats = histogram.get_stats();
            total_count += stats.total_count;
            total_sum += stats.total_sum;
            min_value = min_value.min(stats.min_value);
            max_value = max_value.max(stats.max_value);
            overflow_count += stats.overflow_count;
            last_update = last_update.max(stats.last_update_ns);
        }

        let average = if total_count > 0 {
            total_sum as f64 / total_count as f64
        } else {
            0.0
        };

        // Calculate aggregated percentiles
        let percentiles = self.calculate_aggregated_percentiles();

        HistogramStats {
            total_count,
            total_sum,
            average,
            min_value: if min_value == u64::MAX { 0 } else { min_value },
            max_value,
            percentiles,
            overflow_count,
            last_update_ns: last_update,
        }
    }

    /// Reset all histograms
    pub fn reset_all(&self) {
        for histogram in &self.thread_histograms {
            histogram.reset();
        }
    }

    /// Get shard index for current thread
    #[inline(always)]
    fn get_shard_index(&self) -> usize {
        // Use thread ID for consistent sharding
        std::thread::current().id().as_u64().get() as usize % self.shard_count
    }

    /// Calculate percentiles across all shards
    fn calculate_aggregated_percentiles(&self) -> PercentileResult {
        // Collect all bucket counts across shards
        let bucket_count = self.thread_histograms[0].buckets.len();
        let mut aggregated_buckets = vec![0u64; bucket_count];
        let mut total_count = 0u64;

        for histogram in &self.thread_histograms {
            let bucket_counts = histogram.get_bucket_counts();
            for (i, count) in bucket_counts.iter().enumerate() {
                if i < aggregated_buckets.len() {
                    aggregated_buckets[i] += count;
                    total_count += count;
                }
            }
        }

        if total_count == 0 {
            return PercentileResult {
                p50: 0, p90: 0, p95: 0, p99: 0, p99_9: 0, p99_99: 0,
            };
        }

        // Calculate percentiles from aggregated buckets
        let p50_target = (total_count as f64 * 0.50) as u64;
        let p90_target = (total_count as f64 * 0.90) as u64;
        let p95_target = (total_count as f64 * 0.95) as u64;
        let p99_target = (total_count as f64 * 0.99) as u64;
        let p99_9_target = (total_count as f64 * 0.999) as u64;
        let p99_99_target = (total_count as f64 * 0.9999) as u64;

        let mut cumulative_count = 0u64;
        let mut p50 = 0u64;
        let mut p90 = 0u64;
        let mut p95 = 0u64;
        let mut p99 = 0u64;
        let mut p99_9 = 0u64;
        let mut p99_99 = 0u64;

        let boundaries = self.thread_histograms[0].get_bucket_boundaries();

        for (i, &bucket_count) in aggregated_buckets.iter().enumerate() {
            cumulative_count += bucket_count;

            let bucket_value = if i < boundaries.len() {
                boundaries[i]
            } else {
                u64::MAX
            };

            if p50 == 0 && cumulative_count >= p50_target {
                p50 = bucket_value;
            }
            if p90 == 0 && cumulative_count >= p90_target {
                p90 = bucket_value;
            }
            if p95 == 0 && cumulative_count >= p95_target {
                p95 = bucket_value;
            }
            if p99 == 0 && cumulative_count >= p99_target {
                p99 = bucket_value;
            }
            if p99_9 == 0 && cumulative_count >= p99_9_target {
                p99_9 = bucket_value;
            }
            if p99_99 == 0 && cumulative_count >= p99_99_target {
                p99_99 = bucket_value;
            }
        }

        PercentileResult { p50, p90, p95, p99, p99_9, p99_99 }
    }
}

/// Global histogram instances for common metrics
static ORDER_LATENCY_HISTOGRAM: std::sync::OnceLock<Arc<ConcurrentLatencyHistogram>> = std::sync::OnceLock::new();
static TRADE_LATENCY_HISTOGRAM: std::sync::OnceLock<Arc<ConcurrentLatencyHistogram>> = std::sync::OnceLock::new();
static NETWORK_LATENCY_HISTOGRAM: std::sync::OnceLock<Arc<ConcurrentLatencyHistogram>> = std::sync::OnceLock::new();

/// Get global order latency histogram
pub fn order_latency_histogram() -> &'static Arc<ConcurrentLatencyHistogram> {
    ORDER_LATENCY_HISTOGRAM.get_or_init(|| {
        Arc::new(ConcurrentLatencyHistogram::new(num_cpus::get()))
    })
}

/// Get global trade latency histogram
pub fn trade_latency_histogram() -> &'static Arc<ConcurrentLatencyHistogram> {
    TRADE_LATENCY_HISTOGRAM.get_or_init(|| {
        Arc::new(ConcurrentLatencyHistogram::new(num_cpus::get()))
    })
}

/// Get global network latency histogram
pub fn network_latency_histogram() -> &'static Arc<ConcurrentLatencyHistogram> {
    NETWORK_LATENCY_HISTOGRAM.get_or_init(|| {
        Arc::new(ConcurrentLatencyHistogram::new(num_cpus::get()))
    })
}

/// Convenience functions for recording latencies
#[inline(always)]
pub fn record_order_latency(latency_ns: u64) {
    order_latency_histogram().record(latency_ns);
}

#[inline(always)]
pub fn record_trade_latency(latency_ns: u64) {
    trade_latency_histogram().record(latency_ns);
}

#[inline(always)]
pub fn record_network_latency(latency_ns: u64) {
    network_latency_histogram().record(latency_ns);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_histogram_creation() {
        let histogram = LatencyHistogram::new();
        let stats = histogram.get_stats();
        
        assert_eq!(stats.total_count, 0);
        assert_eq!(stats.average, 0.0);
    }

    #[test]
    fn test_histogram_recording() {
        let histogram = LatencyHistogram::for_trading();
        
        histogram.record(1000); // 1μs
        histogram.record(5000); // 5μs
        histogram.record(10000); // 10μs
        
        let stats = histogram.get_stats();
        assert_eq!(stats.total_count, 3);
        assert_eq!(stats.total_sum, 16000);
        assert_eq!(stats.min_value, 1000);
        assert_eq!(stats.max_value, 10000);
    }

    #[test]
    fn test_percentile_calculation() {
        let histogram = LatencyHistogram::for_trading();
        
        // Record 100 values from 1000ns to 100000ns
        for i in 1..=100 {
            histogram.record(i * 1000);
        }
        
        let percentiles = histogram.calculate_percentiles();
        
        // P50 should be around 50000ns
        assert!(percentiles.p50 >= 40000 && percentiles.p50 <= 60000);
        
        // P99 should be around 99000ns
        assert!(percentiles.p99 >= 90000 && percentiles.p99 <= 110000);
    }

    #[test]
    fn test_bucket_finding() {
        let histogram = LatencyHistogram::for_trading();
        
        // Test bucket finding for various values
        let index_1ns = histogram.find_bucket_index(1);
        let index_1us = histogram.find_bucket_index(1000);
        let index_1ms = histogram.find_bucket_index(1_000_000);
        
        assert!(index_1ns < histogram.buckets.len());
        assert!(index_1us < histogram.buckets.len());
        assert!(index_1ms < histogram.buckets.len());
        assert!(index_1ns < index_1us);
        assert!(index_1us < index_1ms);
    }

    #[test]
    fn test_concurrent_histogram() {
        let histogram = ConcurrentLatencyHistogram::new(4);
        let histogram_arc = Arc::new(histogram);
        let mut handles = vec![];

        // Spawn threads to record concurrently
        for thread_id in 0..4 {
            let histogram_clone = histogram_arc.clone();
            let handle = thread::spawn(move || {
                for i in 0..1000 {
                    histogram_clone.record((thread_id * 1000 + i) as u64);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        let stats = histogram_arc.get_aggregated_stats();
        assert_eq!(stats.total_count, 4000);
    }

    #[test]
    fn test_histogram_reset() {
        let histogram = LatencyHistogram::for_trading();
        
        histogram.record(1000);
        histogram.record(2000);
        
        let stats_before = histogram.get_stats();
        assert_eq!(stats_before.total_count, 2);
        
        histogram.reset();
        
        let stats_after = histogram.get_stats();
        assert_eq!(stats_after.total_count, 0);
        assert_eq!(stats_after.total_sum, 0);
    }

    #[test]
    fn test_overflow_handling() {
        let small_buckets = vec![100, 200, 300]; // Very small buckets
        let histogram = LatencyHistogram::with_buckets(small_buckets);
        
        histogram.record(1000); // Should overflow
        
        let stats = histogram.get_stats();
        assert_eq!(stats.overflow_count, 1);
    }

    #[test]
    fn test_batch_recording() {
        let histogram = LatencyHistogram::for_trading();
        let latencies = vec![1000, 2000, 3000, 4000, 5000];
        
        histogram.record_batch(&latencies);
        
        let stats = histogram.get_stats();
        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.total_sum, 15000);
    }

    #[test]
    fn test_global_histograms() {
        record_order_latency(1000);
        record_trade_latency(2000);
        record_network_latency(3000);
        
        let order_stats = order_latency_histogram().get_aggregated_stats();
        let trade_stats = trade_latency_histogram().get_aggregated_stats();
        let network_stats = network_latency_histogram().get_aggregated_stats();
        
        assert!(order_stats.total_count > 0);
        assert!(trade_stats.total_count > 0);
        assert!(network_stats.total_count > 0);
    }

    #[test]
    fn test_high_frequency_recording() {
        let histogram = LatencyHistogram::for_trading();
        let start_time = now_nanos();
        
        // Record many values quickly
        for i in 0..10000 {
            histogram.record(i % 1000 + 1000);
        }
        
        let end_time = now_nanos();
        let elapsed_ns = end_time - start_time;
        
        // Should complete quickly (less than 10ms)
        assert!(elapsed_ns < 10_000_000);
        
        let stats = histogram.get_stats();
        assert_eq!(stats.total_count, 10000);
    }

    #[test]
    fn test_percentile_accuracy() {
        let histogram = LatencyHistogram::for_trading();
        
        // Record known distribution
        for _ in 0..50 {
            histogram.record(1000); // 50% at 1μs
        }
        for _ in 0..40 {
            histogram.record(2000); // 40% at 2μs
        }
        for _ in 0..9 {
            histogram.record(3000); // 9% at 3μs
        }
        histogram.record(10000); // 1% at 10μs
        
        let percentiles = histogram.calculate_percentiles();
        
        // P50 should be 1μs or 2μs
        assert!(percentiles.p50 <= 2000);
        
        // P99 should be close to 10μs
        assert!(percentiles.p99 >= 3000);
    }
}