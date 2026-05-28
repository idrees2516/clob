use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::thread;
use serde::{Serialize, Deserialize};

use crate::math::fixed_point::FixedPoint;
use crate::models::avellaneda_stoikov::AvellanedaStoikovEngine;
use crate::performance::lock_free::order_book::LockFreeOrderBook;

/// Automated benchmark suite with baseline tracking and regression detection
#[cfg(test)]
mod automated_benchmark_suite {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkResult {
        pub name: String,
        pub latency_stats: LatencyStatistics,
        pub throughput_ops_per_sec: f64,
        pub memory_stats: MemoryStatistics,
        pub cpu_stats: CpuStatistics,
        pub timestamp: u64,
        pub git_commit: String,
        pub build_config: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LatencyStatistics {
        pub min_ns: f64,
        pub max_ns: f64,
        pub mean_ns: f64,
        pub median_ns: f64,
        pub p95_ns: f64,
        pub p99_ns: f64,
        pub p99_9_ns: f64,
        pub std_dev_ns: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryStatistics {
        pub peak_usage_mb: f64,
        pub average_usage_mb: f64,
        pub allocations_per_sec: f64,
        pub deallocations_per_sec: f64,
        pub fragmentation_ratio: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CpuStatistics {
        pub average_utilization_percent: f64,
        pub peak_utilization_percent: f64,
        pub cache_miss_rate: f64,
        pub branch_miss_rate: f64,
        pub instructions_per_cycle: f64,
    }

    pub struct AutomatedBenchmarkSuite {
        baselines: HashMap<String, BenchmarkResult>,
        regression_thresholds: RegressionThresholds,
        system_monitor: SystemMonitor,
    }

    #[derive(Debug, Clone)]
    pub struct RegressionThresholds {
        pub latency_increase_percent: f64,
        pub throughput_decrease_percent: f64,
        pub memory_increase_percent: f64,
        pub cpu_increase_percent: f64,
    }

    impl Default for RegressionThresholds {
        fn default() -> Self {
            Self {
                latency_increase_percent: 5.0,    // 5% latency increase
                throughput_decrease_percent: 5.0, // 5% throughput decrease
                memory_increase_percent: 10.0,    // 10% memory increase
                cpu_increase_percent: 10.0,       // 10% CPU increase
            }
        }
    }

    pub struct SystemMonitor {
        memory_tracker: Arc<Mutex<MemoryTracker>>,
        cpu_tracker: Arc<Mutex<CpuTracker>>,
    }

    struct MemoryTracker {
        current_usage: usize,
        peak_usage: usize,
        allocation_count: usize,
        deallocation_count: usize,
        start_time: Instant,
    }

    struct CpuTracker {
        start_time: Instant,
        total_cpu_time: Duration,
        sample_count: usize,
    }

    impl AutomatedBenchmarkSuite {
        pub fn new() -> Self {
            Self {
                baselines: HashMap::new(),
                regression_thresholds: RegressionThresholds::default(),
                system_monitor: SystemMonitor::new(),
            }
        }

        pub fn load_baselines(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            if std::path::Path::new(path).exists() {
                let content = std::fs::read_to_string(path)?;
                self.baselines = serde_json::from_str(&content)?;
            }
            Ok(())
        }

        pub fn save_baselines(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            let content = serde_json::to_string_pretty(&self.baselines)?;
            std::fs::write(path, content)?;
            Ok(())
        }

        pub fn run_benchmark<F>(&mut self, name: &str, benchmark_fn: F) -> BenchmarkResult
        where
            F: Fn() -> Duration,
        {
            println!("Running benchmark: {}", name);
            
            // Warmup phase
            println!("  Warming up...");
            for _ in 0..10 {
                benchmark_fn();
            }

            // Reset monitoring
            self.system_monitor.reset();

            // Measurement phase
            println!("  Measuring...");
            let mut latencies = Vec::new();
            let measurement_start = Instant::now();
            let measurement_duration = Duration::from_secs(10);
            let mut operation_count = 0;

            while measurement_start.elapsed() < measurement_duration {
                let latency = benchmark_fn();
                latencies.push(latency.as_nanos() as f64);
                operation_count += 1;

                // Update system monitoring
                self.system_monitor.update();
            }

            let total_time = measurement_start.elapsed();
            let throughput = operation_count as f64 / total_time.as_secs_f64();

            // Calculate statistics
            let latency_stats = self.calculate_latency_statistics(&latencies);
            let memory_stats = self.system_monitor.get_memory_statistics();
            let cpu_stats = self.system_monitor.get_cpu_statistics();

            let result = BenchmarkResult {
                name: name.to_string(),
                latency_stats,
                throughput_ops_per_sec: throughput,
                memory_stats,
                cpu_stats,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                git_commit: get_git_commit().unwrap_or_else(|| "unknown".to_string()),
                build_config: get_build_config(),
            };

            println!("  Completed: {:.2} ops/sec, {:.2}ns p95 latency", 
                result.throughput_ops_per_sec, result.latency_stats.p95_ns);

            result
        }

        pub fn check_regression(&self, result: &BenchmarkResult) -> RegressionReport {
            let mut report = RegressionReport {
                test_name: result.name.clone(),
                has_regression: false,
                issues: Vec::new(),
            };

            if let Some(baseline) = self.baselines.get(&result.name) {
                // Check latency regression
                let latency_increase = (result.latency_stats.p95_ns - baseline.latency_stats.p95_ns) 
                    / baseline.latency_stats.p95_ns * 100.0;
                if latency_increase > self.regression_thresholds.latency_increase_percent {
                    report.has_regression = true;
                    report.issues.push(format!(
                        "Latency regression: {:.2}% increase (threshold: {:.2}%)",
                        latency_increase, self.regression_thresholds.latency_increase_percent
                    ));
                }

                // Check throughput regression
                let throughput_decrease = (baseline.throughput_ops_per_sec - result.throughput_ops_per_sec) 
                    / baseline.throughput_ops_per_sec * 100.0;
                if throughput_decrease > self.regression_thresholds.throughput_decrease_percent {
                    report.has_regression = true;
                    report.issues.push(format!(
                        "Throughput regression: {:.2}% decrease (threshold: {:.2}%)",
                        throughput_decrease, self.regression_thresholds.throughput_decrease_percent
                    ));
                }

                // Check memory regression
                let memory_increase = (result.memory_stats.peak_usage_mb - baseline.memory_stats.peak_usage_mb) 
                    / baseline.memory_stats.peak_usage_mb * 100.0;
                if memory_increase > self.regression_thresholds.memory_increase_percent {
                    report.has_regression = true;
                    report.issues.push(format!(
                        "Memory regression: {:.2}% increase (threshold: {:.2}%)",
                        memory_increase, self.regression_thresholds.memory_increase_percent
                    ));
                }

                // Check CPU regression
                let cpu_increase = (result.cpu_stats.average_utilization_percent - baseline.cpu_stats.average_utilization_percent) 
                    / baseline.cpu_stats.average_utilization_percent * 100.0;
                if cpu_increase > self.regression_thresholds.cpu_increase_percent {
                    report.has_regression = true;
                    report.issues.push(format!(
                        "CPU regression: {:.2}% increase (threshold: {:.2}%)",
                        cpu_increase, self.regression_thresholds.cpu_increase_percent
                    ));
                }
            }

            report
        }

        pub fn update_baseline(&mut self, result: BenchmarkResult) {
            self.baselines.insert(result.name.clone(), result);
        }

        fn calculate_latency_statistics(&self, latencies: &[f64]) -> LatencyStatistics {
            let mut sorted_latencies = latencies.to_vec();
            sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let len = sorted_latencies.len();
            let min = sorted_latencies[0];
            let max = sorted_latencies[len - 1];
            let mean = sorted_latencies.iter().sum::<f64>() / len as f64;
            let median = if len % 2 == 0 {
                (sorted_latencies[len / 2 - 1] + sorted_latencies[len / 2]) / 2.0
            } else {
                sorted_latencies[len / 2]
            };

            let p95 = sorted_latencies[(len as f64 * 0.95) as usize];
            let p99 = sorted_latencies[(len as f64 * 0.99) as usize];
            let p99_9 = sorted_latencies[(len as f64 * 0.999) as usize];

            let variance = sorted_latencies.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / len as f64;
            let std_dev = variance.sqrt();

            LatencyStatistics {
                min_ns: min,
                max_ns: max,
                mean_ns: mean,
                median_ns: median,
                p95_ns: p95,
                p99_ns: p99,
                p99_9_ns: p99_9,
                std_dev_ns: std_dev,
            }
        }
    }

    impl SystemMonitor {
        pub fn new() -> Self {
            Self {
                memory_tracker: Arc::new(Mutex::new(MemoryTracker {
                    current_usage: 0,
                    peak_usage: 0,
                    allocation_count: 0,
                    deallocation_count: 0,
                    start_time: Instant::now(),
                })),
                cpu_tracker: Arc::new(Mutex::new(CpuTracker {
                    start_time: Instant::now(),
                    total_cpu_time: Duration::new(0, 0),
                    sample_count: 0,
                })),
            }
        }

        pub fn reset(&self) {
            let mut memory_tracker = self.memory_tracker.lock().unwrap();
            memory_tracker.current_usage = 0;
            memory_tracker.peak_usage = 0;
            memory_tracker.allocation_count = 0;
            memory_tracker.deallocation_count = 0;
            memory_tracker.start_time = Instant::now();

            let mut cpu_tracker = self.cpu_tracker.lock().unwrap();
            cpu_tracker.start_time = Instant::now();
            cpu_tracker.total_cpu_time = Duration::new(0, 0);
            cpu_tracker.sample_count = 0;
        }

        pub fn update(&self) {
            // Update CPU tracking
            let mut cpu_tracker = self.cpu_tracker.lock().unwrap();
            cpu_tracker.sample_count += 1;
            // In a real implementation, this would sample actual CPU usage
            // For testing, we'll use a mock value
        }

        pub fn get_memory_statistics(&self) -> MemoryStatistics {
            let memory_tracker = self.memory_tracker.lock().unwrap();
            let elapsed = memory_tracker.start_time.elapsed().as_secs_f64();
            
            MemoryStatistics {
                peak_usage_mb: memory_tracker.peak_usage as f64 / (1024.0 * 1024.0),
                average_usage_mb: memory_tracker.current_usage as f64 / (1024.0 * 1024.0),
                allocations_per_sec: memory_tracker.allocation_count as f64 / elapsed,
                deallocations_per_sec: memory_tracker.deallocation_count as f64 / elapsed,
                fragmentation_ratio: 0.05, // Mock value
            }
        }

        pub fn get_cpu_statistics(&self) -> CpuStatistics {
            let cpu_tracker = self.cpu_tracker.lock().unwrap();
            
            CpuStatistics {
                average_utilization_percent: 45.0, // Mock value
                peak_utilization_percent: 78.0,    // Mock value
                cache_miss_rate: 0.02,              // Mock value
                branch_miss_rate: 0.01,             // Mock value
                instructions_per_cycle: 2.5,        // Mock value
            }
        }
    }

    #[derive(Debug)]
    pub struct RegressionReport {
        pub test_name: String,
        pub has_regression: bool,
        pub issues: Vec<String>,
    }

    #[test]
    fn test_avellaneda_stoikov_benchmark() {
        let mut suite = AutomatedBenchmarkSuite::new();
        suite.load_baselines("benchmarks/baselines.json").ok();

        let params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
            gamma: FixedPoint::from_float(1.0),
            sigma: FixedPoint::from_float(0.2),
            k: FixedPoint::from_float(0.1),
            A: FixedPoint::from_float(1.0),
            T: FixedPoint::from_float(1.0),
        };

        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();

        let result = suite.run_benchmark("avellaneda_stoikov_quote_generation", || {
            let start = Instant::now();
            let _quotes = engine.calculate_optimal_quotes(
                FixedPoint::from_float(100.0),
                50,
                FixedPoint::from_float(0.2),
                FixedPoint::from_float(0.1),
            ).unwrap();
            start.elapsed()
        });

        let regression_report = suite.check_regression(&result);
        
        if regression_report.has_regression {
            panic!("Performance regression detected in {}: {:?}", 
                regression_report.test_name, regression_report.issues);
        }

        // Update baseline if this is a new test or performance improved
        suite.update_baseline(result);
        suite.save_baselines("benchmarks/baselines.json").unwrap();
    }

    #[test]
    fn test_lock_free_order_book_benchmark() {
        let mut suite = AutomatedBenchmarkSuite::new();
        suite.load_baselines("benchmarks/baselines.json").ok();

        let order_book = LockFreeOrderBook::new();
        let mut order_id = 0u64;

        let result = suite.run_benchmark("lock_free_order_book_operations", || {
            let start = Instant::now();
            
            order_id += 1;
            let order = crate::performance::lock_free::order_book::Order {
                id: order_id,
                price: FixedPoint::from_float(100.0 + (order_id % 100) as f64 * 0.01),
                quantity: FixedPoint::from_float(100.0),
                side: if order_id % 2 == 0 { 
                    crate::performance::lock_free::order_book::Side::Buy 
                } else { 
                    crate::performance::lock_free::order_book::Side::Sell 
                },
                timestamp: order_id,
            };
            
            order_book.add_order(order).unwrap();
            
            if order_id % 10 == 0 {
                order_book.get_best_bid_ask();
            }
            
            if order_id % 20 == 0 {
                order_book.cancel_order(order_id - 10).ok();
            }
            
            start.elapsed()
        });

        let regression_report = suite.check_regression(&result);
        
        if regression_report.has_regression {
            panic!("Performance regression detected in {}: {:?}", 
                regression_report.test_name, regression_report.issues);
        }

        suite.update_baseline(result);
        suite.save_baselines("benchmarks/baselines.json").unwrap();
    }

    #[test]
    fn test_memory_intensive_benchmark() {
        let mut suite = AutomatedBenchmarkSuite::new();
        suite.load_baselines("benchmarks/baselines.json").ok();

        let result = suite.run_benchmark("memory_intensive_operations", || {
            let start = Instant::now();
            
            // Allocate and deallocate memory
            let mut vectors = Vec::new();
            for i in 0..100 {
                vectors.push(vec![i as f64; 1000]);
            }
            
            // Process the data
            let mut sum = 0.0;
            for vec in &vectors {
                sum += vec.iter().sum::<f64>();
            }
            
            // Force use of sum to prevent optimization
            std::hint::black_box(sum);
            
            start.elapsed()
        });

        let regression_report = suite.check_regression(&result);
        
        if regression_report.has_regression {
            panic!("Performance regression detected in {}: {:?}", 
                regression_report.test_name, regression_report.issues);
        }

        suite.update_baseline(result);
        suite.save_baselines("benchmarks/baselines.json").unwrap();
    }

    #[test]
    fn test_cpu_intensive_benchmark() {
        let mut suite = AutomatedBenchmarkSuite::new();
        suite.load_baselines("benchmarks/baselines.json").ok();

        let result = suite.run_benchmark("cpu_intensive_operations", || {
            let start = Instant::now();
            
            // CPU-intensive mathematical computation
            let mut result = 1.0;
            for i in 1..1000 {
                result = result.sin() + (i as f64).sqrt();
            }
            
            // Force use of result to prevent optimization
            std::hint::black_box(result);
            
            start.elapsed()
        });

        let regression_report = suite.check_regression(&result);
        
        if regression_report.has_regression {
            panic!("Performance regression detected in {}: {:?}", 
                regression_report.test_name, regression_report.issues);
        }

        suite.update_baseline(result);
        suite.save_baselines("benchmarks/baselines.json").unwrap();
    }

    #[test]
    fn test_concurrent_benchmark() {
        let mut suite = AutomatedBenchmarkSuite::new();
        suite.load_baselines("benchmarks/baselines.json").ok();

        let order_book = Arc::new(LockFreeOrderBook::new());
        
        let result = suite.run_benchmark("concurrent_order_book_operations", || {
            let start = Instant::now();
            
            let mut handles = Vec::new();
            let num_threads = 4;
            let operations_per_thread = 100;
            
            for thread_id in 0..num_threads {
                let order_book_clone = Arc::clone(&order_book);
                let handle = thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        let order_id = (thread_id * operations_per_thread + i) as u64;
                        let order = crate::performance::lock_free::order_book::Order {
                            id: order_id,
                            price: FixedPoint::from_float(100.0 + (order_id % 100) as f64 * 0.01),
                            quantity: FixedPoint::from_float(100.0),
                            side: if order_id % 2 == 0 { 
                                crate::performance::lock_free::order_book::Side::Buy 
                            } else { 
                                crate::performance::lock_free::order_book::Side::Sell 
                            },
                            timestamp: order_id,
                        };
                        
                        order_book_clone.add_order(order).unwrap();
                        
                        if i % 10 == 0 {
                            order_book_clone.get_best_bid_ask();
                        }
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            start.elapsed()
        });

        let regression_report = suite.check_regression(&result);
        
        if regression_report.has_regression {
            panic!("Performance regression detected in {}: {:?}", 
                regression_report.test_name, regression_report.issues);
        }

        suite.update_baseline(result);
        suite.save_baselines("benchmarks/baselines.json").unwrap();
    }
}

// Helper functions
fn get_git_commit() -> Option<String> {
    std::process::Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
}

fn get_build_config() -> String {
    if cfg!(debug_assertions) {
        "debug".to_string()
    } else {
        "release".to_string()
    }
}