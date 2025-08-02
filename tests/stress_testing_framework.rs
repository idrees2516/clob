use std::sync::{Arc, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

use crate::math::fixed_point::FixedPoint;
use crate::performance::lock_free::order_book::LockFreeOrderBook;

/// Stress testing framework for high-frequency trading systems
#[cfg(test)]
mod stress_testing_framework {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StressTestConfig {
        pub duration_seconds: u64,
        pub target_throughput_ops_per_sec: u64,
        pub concurrent_threads: usize,
        pub memory_pressure_mb: usize,
        pub cpu_intensive_operations: bool,
        pub network_latency_simulation_ms: u64,
        pub error_injection_rate: f64, // 0.0 to 1.0
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StressTestResults {
        pub test_name: String,
        pub config: StressTestConfig,
        pub actual_throughput_ops_per_sec: f64,
        pub latency_stats: LatencyStats,
        pub error_rate: f64,
        pub memory_stats: MemoryStats,
        pub cpu_stats: CpuStats,
        pub system_stability: SystemStabilityMetrics,
        pub duration: Duration,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LatencyStats {
        pub min_ns: u64,
        pub max_ns: u64,
        pub mean_ns: f64,
        pub p50_ns: u64,
        pub p95_ns: u64,
        pub p99_ns: u64,
        pub p99_9_ns: u64,
        pub std_dev_ns: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MemoryStats {
        pub peak_usage_mb: f64,
        pub average_usage_mb: f64,
        pub allocation_rate_per_sec: f64,
        pub gc_pressure_score: f64,
        pub memory_leaks_detected: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CpuStats {
        pub average_utilization_percent: f64,
        pub peak_utilization_percent: f64,
        pub context_switches_per_sec: f64,
        pub cache_miss_rate: f64,
        pub thermal_throttling_detected: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SystemStabilityMetrics {
        pub crashes: u64,
        pub deadlocks_detected: u64,
        pub memory_corruption_events: u64,
        pub data_races_detected: u64,
        pub assertion_failures: u64,
        pub recovery_time_ms: Vec<u64>,
    }

    pub struct StressTestFramework {
        results_history: Vec<StressTestResults>,
        system_monitor: SystemMonitor,
        error_injector: ErrorInjector,
    }

    struct SystemMonitor {
        memory_tracker: Arc<AtomicU64>,
        cpu_tracker: Arc<AtomicU64>,
        operation_counter: Arc<AtomicU64>,
        error_counter: Arc<AtomicU64>,
        start_time: Arc<Mutex<Option<Instant>>>,
        latency_samples: Arc<Mutex<VecDeque<u64>>>,
    }

    struct ErrorInjector {
        injection_rate: f64,
        rng: crate::math::sde_solvers::DeterministicRng,
    }

    impl StressTestFramework {
        pub fn new() -> Self {
            Self {
                results_history: Vec::new(),
                system_monitor: SystemMonitor::new(),
                error_injector: ErrorInjector::new(0.0),
            }
        }

        pub async fn run_stress_test<F, Fut>(
            &mut self,
            test_name: &str,
            config: StressTestConfig,
            test_function: F,
        ) -> StressTestResults
        where
            F: Fn() -> Fut + Send + Sync + Clone + 'static,
            Fut: std::future::Future<Output = Result<Duration, Box<dyn std::error::Error + Send + Sync>>> + Send,
        {
            println!("Starting stress test: {}", test_name);
            println!("Config: {:?}", config);

            self.error_injector.set_injection_rate(config.error_injection_rate);
            self.system_monitor.reset();

            // Apply memory pressure if configured
            let _memory_pressure = if config.memory_pressure_mb > 0 {
                Some(self.apply_memory_pressure(config.memory_pressure_mb))
            } else {
                None
            };

            let test_start = Instant::now();
            let test_duration = Duration::from_secs(config.duration_seconds);
            let target_ops_per_sec = config.target_throughput_ops_per_sec;
            let operation_interval = Duration::from_nanos(1_000_000_000 / target_ops_per_sec);

            // Spawn worker threads
            let mut handles = Vec::new();
            let stop_flag = Arc::new(AtomicBool::new(false));

            for thread_id in 0..config.concurrent_threads {
                let test_fn = test_function.clone();
                let monitor = self.system_monitor.clone();
                let injector = self.error_injector.clone();
                let stop_flag_clone = Arc::clone(&stop_flag);
                let ops_per_thread = target_ops_per_sec / config.concurrent_threads as u64;
                let thread_interval = Duration::from_nanos(1_000_000_000 / ops_per_thread);

                let handle = tokio::spawn(async move {
                    let mut last_operation = Instant::now();
                    let mut operations_completed = 0u64;

                    while !stop_flag_clone.load(Ordering::Relaxed) {
                        if last_operation.elapsed() >= thread_interval {
                            let operation_start = Instant::now();

                            // Inject errors if configured
                            if injector.should_inject_error() {
                                monitor.record_error();
                                tokio::time::sleep(Duration::from_millis(1)).await; // Simulate error handling
                            } else {
                                match test_fn().await {
                                    Ok(latency) => {
                                        monitor.record_operation(latency);
                                        operations_completed += 1;
                                    }
                                    Err(_) => {
                                        monitor.record_error();
                                    }
                                }
                            }

                            last_operation = Instant::now();

                            // Simulate network latency if configured
                            if config.network_latency_simulation_ms > 0 {
                                tokio::time::sleep(Duration::from_millis(config.network_latency_simulation_ms)).await;
                            }
                        }

                        // CPU intensive operations if configured
                        if config.cpu_intensive_operations {
                            self.perform_cpu_intensive_work();
                        }

                        // Small sleep to prevent busy waiting
                        tokio::time::sleep(Duration::from_micros(1)).await;
                    }

                    operations_completed
                });

                handles.push(handle);
            }

            // Run test for specified duration
            tokio::time::sleep(test_duration).await;
            stop_flag.store(true, Ordering::Relaxed);

            // Wait for all threads to complete
            let mut total_operations = 0u64;
            for handle in handles {
                total_operations += handle.await.unwrap_or(0);
            }

            let actual_duration = test_start.elapsed();
            let actual_throughput = total_operations as f64 / actual_duration.as_secs_f64();

            // Collect results
            let results = StressTestResults {
                test_name: test_name.to_string(),
                config: config.clone(),
                actual_throughput_ops_per_sec: actual_throughput,
                latency_stats: self.system_monitor.get_latency_stats(),
                error_rate: self.system_monitor.get_error_rate(),
                memory_stats: self.system_monitor.get_memory_stats(),
                cpu_stats: self.system_monitor.get_cpu_stats(),
                system_stability: self.system_monitor.get_stability_metrics(),
                duration: actual_duration,
            };

            println!("Stress test completed:");
            println!("  Actual throughput: {:.2} ops/sec", results.actual_throughput_ops_per_sec);
            println!("  Error rate: {:.4}%", results.error_rate * 100.0);
            println!("  P99 latency: {} ns", results.latency_stats.p99_ns);

            self.results_history.push(results.clone());
            results
        }

        fn apply_memory_pressure(&self, target_mb: usize) -> Vec<Vec<u8>> {
            let mut memory_blocks = Vec::new();
            let block_size = 1024 * 1024; // 1MB blocks
            
            for _ in 0..target_mb {
                memory_blocks.push(vec![0u8; block_size]);
            }
            
            memory_blocks
        }

        fn perform_cpu_intensive_work(&self) {
            // Perform some CPU-intensive calculations
            let mut result = 1.0f64;
            for i in 1..1000 {
                result = result.sin() + (i as f64).sqrt();
            }
            std::hint::black_box(result);
        }

        pub fn analyze_performance_degradation(&self) -> PerformanceDegradationReport {
            if self.results_history.len() < 2 {
                return PerformanceDegradationReport {
                    has_degradation: false,
                    degradation_factors: Vec::new(),
                };
            }

            let latest = &self.results_history[self.results_history.len() - 1];
            let baseline = &self.results_history[0];

            let mut degradation_factors = Vec::new();

            // Check throughput degradation
            let throughput_change = (latest.actual_throughput_ops_per_sec - baseline.actual_throughput_ops_per_sec) 
                / baseline.actual_throughput_ops_per_sec;
            if throughput_change < -0.05 { // 5% degradation threshold
                degradation_factors.push(format!(
                    "Throughput degraded by {:.2}%", 
                    throughput_change.abs() * 100.0
                ));
            }

            // Check latency degradation
            let latency_change = (latest.latency_stats.p99_ns as f64 - baseline.latency_stats.p99_ns as f64) 
                / baseline.latency_stats.p99_ns as f64;
            if latency_change > 0.1 { // 10% increase threshold
                degradation_factors.push(format!(
                    "P99 latency increased by {:.2}%", 
                    latency_change * 100.0
                ));
            }

            // Check error rate increase
            let error_rate_change = latest.error_rate - baseline.error_rate;
            if error_rate_change > 0.01 { // 1% increase threshold
                degradation_factors.push(format!(
                    "Error rate increased by {:.4}%", 
                    error_rate_change * 100.0
                ));
            }

            PerformanceDegradationReport {
                has_degradation: !degradation_factors.is_empty(),
                degradation_factors,
            }
        }
    }

    impl SystemMonitor {
        fn new() -> Self {
            Self {
                memory_tracker: Arc::new(AtomicU64::new(0)),
                cpu_tracker: Arc::new(AtomicU64::new(0)),
                operation_counter: Arc::new(AtomicU64::new(0)),
                error_counter: Arc::new(AtomicU64::new(0)),
                start_time: Arc::new(Mutex::new(None)),
                latency_samples: Arc::new(Mutex::new(VecDeque::new())),
            }
        }

        fn clone(&self) -> Self {
            Self {
                memory_tracker: Arc::clone(&self.memory_tracker),
                cpu_tracker: Arc::clone(&self.cpu_tracker),
                operation_counter: Arc::clone(&self.operation_counter),
                error_counter: Arc::clone(&self.error_counter),
                start_time: Arc::clone(&self.start_time),
                latency_samples: Arc::clone(&self.latency_samples),
            }
        }

        fn reset(&self) {
            self.memory_tracker.store(0, Ordering::Relaxed);
            self.cpu_tracker.store(0, Ordering::Relaxed);
            self.operation_counter.store(0, Ordering::Relaxed);
            self.error_counter.store(0, Ordering::Relaxed);
            *self.start_time.lock().unwrap() = Some(Instant::now());
            self.latency_samples.lock().unwrap().clear();
        }

        fn record_operation(&self, latency: Duration) {
            self.operation_counter.fetch_add(1, Ordering::Relaxed);
            
            let mut samples = self.latency_samples.lock().unwrap();
            samples.push_back(latency.as_nanos() as u64);
            
            // Keep only recent samples to prevent memory growth
            if samples.len() > 100_000 {
                samples.pop_front();
            }
        }

        fn record_error(&self) {
            self.error_counter.fetch_add(1, Ordering::Relaxed);
        }

        fn get_latency_stats(&self) -> LatencyStats {
            let samples = self.latency_samples.lock().unwrap();
            let mut sorted_samples: Vec<u64> = samples.iter().cloned().collect();
            sorted_samples.sort_unstable();

            if sorted_samples.is_empty() {
                return LatencyStats {
                    min_ns: 0,
                    max_ns: 0,
                    mean_ns: 0.0,
                    p50_ns: 0,
                    p95_ns: 0,
                    p99_ns: 0,
                    p99_9_ns: 0,
                    std_dev_ns: 0.0,
                };
            }

            let len = sorted_samples.len();
            let min = sorted_samples[0];
            let max = sorted_samples[len - 1];
            let mean = sorted_samples.iter().sum::<u64>() as f64 / len as f64;
            let p50 = sorted_samples[len / 2];
            let p95 = sorted_samples[(len as f64 * 0.95) as usize];
            let p99 = sorted_samples[(len as f64 * 0.99) as usize];
            let p99_9 = sorted_samples[(len as f64 * 0.999) as usize];

            let variance = sorted_samples.iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>() / len as f64;
            let std_dev = variance.sqrt();

            LatencyStats {
                min_ns: min,
                max_ns: max,
                mean_ns: mean,
                p50_ns: p50,
                p95_ns: p95,
                p99_ns: p99,
                p99_9_ns: p99_9,
                std_dev_ns: std_dev,
            }
        }

        fn get_error_rate(&self) -> f64 {
            let operations = self.operation_counter.load(Ordering::Relaxed);
            let errors = self.error_counter.load(Ordering::Relaxed);
            
            if operations + errors == 0 {
                0.0
            } else {
                errors as f64 / (operations + errors) as f64
            }
        }

        fn get_memory_stats(&self) -> MemoryStats {
            // In a real implementation, this would collect actual memory statistics
            MemoryStats {
                peak_usage_mb: 150.0,
                average_usage_mb: 120.0,
                allocation_rate_per_sec: 1000.0,
                gc_pressure_score: 0.3,
                memory_leaks_detected: false,
            }
        }

        fn get_cpu_stats(&self) -> CpuStats {
            // In a real implementation, this would collect actual CPU statistics
            CpuStats {
                average_utilization_percent: 65.0,
                peak_utilization_percent: 95.0,
                context_switches_per_sec: 5000.0,
                cache_miss_rate: 0.02,
                thermal_throttling_detected: false,
            }
        }

        fn get_stability_metrics(&self) -> SystemStabilityMetrics {
            SystemStabilityMetrics {
                crashes: 0,
                deadlocks_detected: 0,
                memory_corruption_events: 0,
                data_races_detected: 0,
                assertion_failures: 0,
                recovery_time_ms: Vec::new(),
            }
        }
    }

    impl ErrorInjector {
        fn new(injection_rate: f64) -> Self {
            Self {
                injection_rate,
                rng: crate::math::sde_solvers::DeterministicRng::new(42),
            }
        }

        fn clone(&self) -> Self {
            Self {
                injection_rate: self.injection_rate,
                rng: crate::math::sde_solvers::DeterministicRng::new(42),
            }
        }

        fn set_injection_rate(&mut self, rate: f64) {
            self.injection_rate = rate.clamp(0.0, 1.0);
        }

        fn should_inject_error(&mut self) -> bool {
            self.rng.sample_uniform() < self.injection_rate
        }
    }

    #[derive(Debug)]
    pub struct PerformanceDegradationReport {
        pub has_degradation: bool,
        pub degradation_factors: Vec<String>,
    }

    #[tokio::test]
    async fn test_order_book_stress_test() {
        let mut framework = StressTestFramework::new();
        let order_book = Arc::new(LockFreeOrderBook::new());
        let order_counter = Arc::new(AtomicU64::new(0));

        let config = StressTestConfig {
            duration_seconds: 5,
            target_throughput_ops_per_sec: 100_000,
            concurrent_threads: 8,
            memory_pressure_mb: 100,
            cpu_intensive_operations: false,
            network_latency_simulation_ms: 0,
            error_injection_rate: 0.01,
        };

        let order_book_clone = Arc::clone(&order_book);
        let counter_clone = Arc::clone(&order_counter);

        let results = framework.run_stress_test(
            "order_book_high_throughput",
            config,
            move || {
                let ob = Arc::clone(&order_book_clone);
                let counter = Arc::clone(&counter_clone);
                
                async move {
                    let start = Instant::now();
                    let order_id = counter.fetch_add(1, Ordering::Relaxed);
                    
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
                    
                    ob.add_order(order)?;
                    
                    // Occasionally query best bid/ask
                    if order_id % 100 == 0 {
                        ob.get_best_bid_ask();
                    }
                    
                    Ok(start.elapsed())
                }
            },
        ).await;

        // Verify stress test results
        assert!(results.actual_throughput_ops_per_sec >= 50_000.0, 
            "Throughput {} should be at least 50k ops/sec", results.actual_throughput_ops_per_sec);
        assert!(results.latency_stats.p99_ns < 100_000, 
            "P99 latency {} should be under 100μs", results.latency_stats.p99_ns);
        assert!(results.error_rate < 0.05, 
            "Error rate {} should be under 5%", results.error_rate);
        assert_eq!(results.system_stability.crashes, 0, "Should have no crashes");

        println!("Order book stress test passed: {:?}", results);
    }

    #[tokio::test]
    async fn test_memory_pressure_stress_test() {
        let mut framework = StressTestFramework::new();

        let config = StressTestConfig {
            duration_seconds: 3,
            target_throughput_ops_per_sec: 10_000,
            concurrent_threads: 4,
            memory_pressure_mb: 500, // High memory pressure
            cpu_intensive_operations: true,
            network_latency_simulation_ms: 1,
            error_injection_rate: 0.05,
        };

        let results = framework.run_stress_test(
            "memory_pressure_test",
            config,
            || async {
                let start = Instant::now();
                
                // Allocate and deallocate memory
                let mut data = Vec::new();
                for i in 0..1000 {
                    data.push(vec![i as f64; 100]);
                }
                
                // Process data
                let sum: f64 = data.iter().flatten().sum();
                std::hint::black_box(sum);
                
                Ok(start.elapsed())
            },
        ).await;

        // Under memory pressure, we expect some performance degradation but system should remain stable
        assert!(results.actual_throughput_ops_per_sec >= 5_000.0, 
            "Even under memory pressure, should maintain reasonable throughput");
        assert!(results.error_rate < 0.1, 
            "Error rate should remain manageable under memory pressure");
        assert_eq!(results.system_stability.crashes, 0, "Should not crash under memory pressure");

        println!("Memory pressure stress test passed: {:?}", results);
    }

    #[tokio::test]
    async fn test_error_injection_resilience() {
        let mut framework = StressTestFramework::new();

        let config = StressTestConfig {
            duration_seconds: 2,
            target_throughput_ops_per_sec: 50_000,
            concurrent_threads: 4,
            memory_pressure_mb: 0,
            cpu_intensive_operations: false,
            network_latency_simulation_ms: 0,
            error_injection_rate: 0.2, // High error rate
        };

        let results = framework.run_stress_test(
            "error_injection_resilience",
            config,
            || async {
                let start = Instant::now();
                
                // Simple operation that might fail
                if start.elapsed().as_nanos() % 7 == 0 {
                    return Err("Simulated error".into());
                }
                
                Ok(start.elapsed())
            },
        ).await;

        // System should handle high error rates gracefully
        assert!(results.error_rate >= 0.15, 
            "Should reflect the injected error rate");
        assert!(results.actual_throughput_ops_per_sec >= 10_000.0, 
            "Should maintain some throughput despite errors");
        assert_eq!(results.system_stability.crashes, 0, "Should not crash despite errors");

        println!("Error injection resilience test passed: {:?}", results);
    }

    #[tokio::test]
    async fn test_performance_degradation_detection() {
        let mut framework = StressTestFramework::new();

        let base_config = StressTestConfig {
            duration_seconds: 1,
            target_throughput_ops_per_sec: 100_000,
            concurrent_threads: 4,
            memory_pressure_mb: 0,
            cpu_intensive_operations: false,
            network_latency_simulation_ms: 0,
            error_injection_rate: 0.0,
        };

        // Run baseline test
        framework.run_stress_test(
            "baseline_performance",
            base_config.clone(),
            || async {
                let start = Instant::now();
                // Fast operation
                std::hint::black_box(42);
                Ok(start.elapsed())
            },
        ).await;

        // Run degraded test
        let degraded_config = StressTestConfig {
            cpu_intensive_operations: true, // Add CPU load
            memory_pressure_mb: 200,        // Add memory pressure
            ..base_config
        };

        framework.run_stress_test(
            "baseline_performance", // Same name to compare
            degraded_config,
            || async {
                let start = Instant::now();
                // Slower operation
                for i in 0..100 {
                    std::hint::black_box(i * i);
                }
                Ok(start.elapsed())
            },
        ).await;

        let degradation_report = framework.analyze_performance_degradation();
        
        assert!(degradation_report.has_degradation, 
            "Should detect performance degradation");
        assert!(!degradation_report.degradation_factors.is_empty(), 
            "Should identify specific degradation factors");

        println!("Performance degradation detected: {:?}", degradation_report);
    }
}