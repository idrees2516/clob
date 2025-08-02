use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::math::{
    sde_solvers::{EulerMaruyamaGBMJump, MilsteinGBMJump, SDESolver},
    fixed_point::FixedPoint,
    jump_diffusion::{GBMJumpState, GBMJumpParams},
    hawkes_process::MultivariateHawkesSimulator,
};
use crate::models::{
    avellaneda_stoikov::AvellanedaStoikovEngine,
    gueant_lehalle_tapia::GuéantLehalleTapiaEngine,
};
use crate::performance::lock_free::order_book::LockFreeOrderBook;

/// Performance regression testing framework
#[cfg(test)]
mod performance_regression_testing {
    use super::*;

    // Benchmark configuration
    const BENCHMARK_DURATION: Duration = Duration::from_secs(10);
    const WARMUP_DURATION: Duration = Duration::from_secs(2);
    const MEASUREMENT_SAMPLES: usize = 100;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceBaseline {
        pub test_name: String,
        pub mean_latency_ns: f64,
        pub throughput_ops_per_sec: f64,
        pub memory_usage_mb: f64,
        pub cpu_utilization_percent: f64,
        pub timestamp: u64,
        pub git_commit: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceMetrics {
        pub latency_p50_ns: f64,
        pub latency_p95_ns: f64,
        pub latency_p99_ns: f64,
        pub latency_p99_9_ns: f64,
        pub throughput_ops_per_sec: f64,
        pub memory_peak_mb: f64,
        pub memory_average_mb: f64,
        pub cpu_average_percent: f64,
        pub cpu_peak_percent: f64,
        pub cache_miss_rate: f64,
        pub branch_miss_rate: f64,
    }

    pub struct PerformanceBenchmarkSuite {
        baselines: HashMap<String, PerformanceBaseline>,
        regression_threshold: f64, // Maximum allowed performance degradation (e.g., 0.05 = 5%)
    }

    impl PerformanceBenchmarkSuite {
        pub fn new(regression_threshold: f64) -> Self {
            Self {
                baselines: HashMap::new(),
                regression_threshold,
            }
        }

        pub fn load_baselines(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            let content = std::fs::read_to_string(path)?;
            self.baselines = serde_json::from_str(&content)?;
            Ok(())
        }

        pub fn save_baselines(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
            let content = serde_json::to_string_pretty(&self.baselines)?;
            std::fs::write(path, content)?;
            Ok(())
        }

        pub fn check_regression(&self, test_name: &str, current_metrics: &PerformanceMetrics) -> bool {
            if let Some(baseline) = self.baselines.get(test_name) {
                let latency_regression = (current_metrics.latency_p95_ns - baseline.mean_latency_ns) / baseline.mean_latency_ns;
                let throughput_regression = (baseline.throughput_ops_per_sec - current_metrics.throughput_ops_per_sec) / baseline.throughput_ops_per_sec;
                
                latency_regression <= self.regression_threshold && throughput_regression <= self.regression_threshold
            } else {
                true // No baseline to compare against
            }
        }
    }

    fn benchmark_avellaneda_stoikov_quote_generation(c: &mut Criterion) {
        let mut group = c.benchmark_group("avellaneda_stoikov_quote_generation");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);
        group.sample_size(MEASUREMENT_SAMPLES);

        let params = crate::models::avellaneda_stoikov::AvellanedaStoikovParams {
            gamma: FixedPoint::from_float(1.0),
            sigma: FixedPoint::from_float(0.2),
            k: FixedPoint::from_float(0.1),
            A: FixedPoint::from_float(1.0),
            T: FixedPoint::from_float(1.0),
        };

        let mut engine = AvellanedaStoikovEngine::new(params).unwrap();

        // Benchmark single quote calculation
        group.bench_function("single_quote", |b| {
            b.iter(|| {
                engine.calculate_optimal_quotes(
                    black_box(FixedPoint::from_float(100.0)),
                    black_box(50),
                    black_box(FixedPoint::from_float(0.2)),
                    black_box(FixedPoint::from_float(0.1)),
                )
            })
        });

        // Benchmark batch quote calculations
        let batch_sizes = vec![10, 100, 1000];
        for batch_size in batch_sizes {
            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new("batch_quotes", batch_size),
                &batch_size,
                |b, &size| {
                    let prices: Vec<_> = (0..size).map(|i| FixedPoint::from_float(100.0 + i as f64)).collect();
                    let inventories: Vec<_> = (0..size).map(|i| (i as i64) - (size as i64) / 2).collect();
                    
                    b.iter(|| {
                        for i in 0..size {
                            black_box(engine.calculate_optimal_quotes(
                                black_box(prices[i]),
                                black_box(inventories[i]),
                                black_box(FixedPoint::from_float(0.2)),
                                black_box(FixedPoint::from_float(0.1)),
                            ));
                        }
                    })
                },
            );
        }

        group.finish();
    }

    fn benchmark_sde_solvers(c: &mut Criterion) {
        let mut group = c.benchmark_group("sde_solvers");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);

        let initial_state = GBMJumpState {
            price: FixedPoint::from_float(100.0),
            log_price: FixedPoint::from_float(100.0_f64.ln()),
        };

        let params = GBMJumpParams {
            drift: FixedPoint::from_float(0.05),
            volatility: FixedPoint::from_float(0.2),
            jump_intensity: FixedPoint::from_float(1.0),
            jump_mean: FixedPoint::from_float(0.0),
            jump_std: FixedPoint::from_float(0.1),
        };

        let dt = FixedPoint::from_float(0.001);

        // Benchmark Euler-Maruyama solver
        group.bench_function("euler_maruyama_single_step", |b| {
            let mut solver = EulerMaruyamaGBMJump::new();
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
            
            b.iter(|| {
                black_box(solver.solve_step(
                    black_box(FixedPoint::zero()),
                    black_box(&initial_state),
                    black_box(dt),
                    black_box(&params),
                    black_box(&mut rng),
                ))
            })
        });

        // Benchmark Milstein solver
        group.bench_function("milstein_single_step", |b| {
            let mut solver = MilsteinGBMJump::new();
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
            
            b.iter(|| {
                black_box(solver.solve_step(
                    black_box(FixedPoint::zero()),
                    black_box(&initial_state),
                    black_box(dt),
                    black_box(&params),
                    black_box(&mut rng),
                ))
            })
        });

        // Benchmark path simulation
        let path_lengths = vec![100, 1000, 10000];
        for path_length in path_lengths {
            group.throughput(Throughput::Elements(path_length as u64));
            group.bench_with_input(
                BenchmarkId::new("euler_path_simulation", path_length),
                &path_length,
                |b, &length| {
                    b.iter(|| {
                        let mut solver = EulerMaruyamaGBMJump::new();
                        let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
                        let mut state = initial_state;
                        
                        for _ in 0..length {
                            state = black_box(solver.solve_step(
                                FixedPoint::zero(),
                                &state,
                                dt,
                                &params,
                                &mut rng,
                            ).unwrap());
                        }
                        
                        black_box(state)
                    })
                },
            );
        }

        group.finish();
    }

    fn benchmark_hawkes_process(c: &mut Criterion) {
        let mut group = c.benchmark_group("hawkes_process");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);

        let mut simulator = MultivariateHawkesSimulator::new(
            vec![FixedPoint::from_float(1.0)],
            vec![vec![FixedPoint::from_float(0.5)]],
            vec![vec![FixedPoint::from_float(2.0)]],
        ).unwrap();

        // Add some events for realistic testing
        for i in 0..100 {
            simulator.add_event(0, FixedPoint::from_float(i as f64 * 0.01));
        }

        group.bench_function("intensity_calculation", |b| {
            b.iter(|| {
                black_box(simulator.get_intensity(
                    black_box(0),
                    black_box(FixedPoint::from_float(1.0)),
                ))
            })
        });

        group.bench_function("event_simulation", |b| {
            let mut rng = crate::math::sde_solvers::DeterministicRng::new(42);
            b.iter(|| {
                black_box(simulator.simulate_next_event(black_box(&mut rng)))
            })
        });

        group.finish();
    }

    fn benchmark_lock_free_order_book(c: &mut Criterion) {
        let mut group = c.benchmark_group("lock_free_order_book");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);

        let order_book = LockFreeOrderBook::new();

        // Benchmark order insertion
        group.bench_function("order_insertion", |b| {
            let mut order_id = 0u64;
            b.iter(|| {
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
                
                black_box(order_book.add_order(black_box(order)))
            })
        });

        // Benchmark order cancellation
        group.bench_function("order_cancellation", |b| {
            // Pre-populate order book
            for i in 1..=1000 {
                let order = crate::performance::lock_free::order_book::Order {
                    id: i,
                    price: FixedPoint::from_float(100.0 + (i % 100) as f64 * 0.01),
                    quantity: FixedPoint::from_float(100.0),
                    side: if i % 2 == 0 { 
                        crate::performance::lock_free::order_book::Side::Buy 
                    } else { 
                        crate::performance::lock_free::order_book::Side::Sell 
                    },
                    timestamp: i,
                };
                order_book.add_order(order).unwrap();
            }

            let mut cancel_id = 1u64;
            b.iter(|| {
                cancel_id = (cancel_id % 1000) + 1;
                black_box(order_book.cancel_order(black_box(cancel_id)))
            })
        });

        // Benchmark best bid/ask retrieval
        group.bench_function("best_bid_ask", |b| {
            b.iter(|| {
                black_box(order_book.get_best_bid_ask())
            })
        });

        group.finish();
    }

    fn benchmark_memory_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("memory_operations");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);

        // Benchmark memory allocation patterns
        let allocation_sizes = vec![64, 256, 1024, 4096];
        for size in allocation_sizes {
            group.throughput(Throughput::Bytes(size as u64));
            group.bench_with_input(
                BenchmarkId::new("allocation", size),
                &size,
                |b, &size| {
                    b.iter(|| {
                        let vec: Vec<u8> = black_box(vec![0u8; size]);
                        black_box(vec)
                    })
                },
            );
        }

        // Benchmark cache-friendly vs cache-unfriendly access patterns
        let array_size = 1024 * 1024; // 1MB array
        let array: Vec<u64> = (0..array_size).collect();

        group.bench_function("sequential_access", |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..array_size {
                    sum += black_box(array[i]);
                }
                black_box(sum)
            })
        });

        group.bench_function("random_access", |b| {
            let indices: Vec<usize> = (0..1000).map(|i| (i * 1009) % array_size).collect();
            b.iter(|| {
                let mut sum = 0u64;
                for &idx in &indices {
                    sum += black_box(array[idx]);
                }
                black_box(sum)
            })
        });

        group.finish();
    }

    fn benchmark_simd_operations(c: &mut Criterion) {
        let mut group = c.benchmark_group("simd_operations");
        group.warm_up_time(WARMUP_DURATION);
        group.measurement_time(BENCHMARK_DURATION);

        let data_sizes = vec![100, 1000, 10000];
        
        for size in data_sizes {
            let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
            
            group.throughput(Throughput::Elements(size as u64));
            
            // Scalar sum
            group.bench_with_input(
                BenchmarkId::new("scalar_sum", size),
                &data,
                |b, data| {
                    b.iter(|| {
                        let mut sum = 0.0;
                        for &x in data {
                            sum += black_box(x);
                        }
                        black_box(sum)
                    })
                },
            );

            // SIMD sum (if available)
            #[cfg(target_arch = "x86_64")]
            group.bench_with_input(
                BenchmarkId::new("simd_sum", size),
                &data,
                |b, data| {
                    b.iter(|| {
                        black_box(crate::performance::simd::arithmetic::sum_f64_avx2(black_box(data)))
                    })
                },
            );
        }

        group.finish();
    }

    #[test]
    fn test_performance_regression_detection() {
        let mut suite = PerformanceBenchmarkSuite::new(0.05); // 5% regression threshold

        // Mock baseline
        let baseline = PerformanceBaseline {
            test_name: "test_function".to_string(),
            mean_latency_ns: 1000.0,
            throughput_ops_per_sec: 1_000_000.0,
            memory_usage_mb: 100.0,
            cpu_utilization_percent: 50.0,
            timestamp: 1234567890,
            git_commit: "abc123".to_string(),
        };
        suite.baselines.insert("test_function".to_string(), baseline);

        // Test case 1: No regression
        let good_metrics = PerformanceMetrics {
            latency_p50_ns: 950.0,
            latency_p95_ns: 1020.0,
            latency_p99_ns: 1100.0,
            latency_p99_9_ns: 1200.0,
            throughput_ops_per_sec: 1_020_000.0,
            memory_peak_mb: 105.0,
            memory_average_mb: 98.0,
            cpu_average_percent: 48.0,
            cpu_peak_percent: 55.0,
            cache_miss_rate: 0.02,
            branch_miss_rate: 0.01,
        };
        assert!(suite.check_regression("test_function", &good_metrics));

        // Test case 2: Latency regression
        let bad_latency_metrics = PerformanceMetrics {
            latency_p95_ns: 1100.0, // 10% increase
            ..good_metrics.clone()
        };
        assert!(!suite.check_regression("test_function", &bad_latency_metrics));

        // Test case 3: Throughput regression
        let bad_throughput_metrics = PerformanceMetrics {
            throughput_ops_per_sec: 900_000.0, // 10% decrease
            ..good_metrics.clone()
        };
        assert!(!suite.check_regression("test_function", &bad_throughput_metrics));
    }

    #[test]
    fn test_memory_usage_monitoring() {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::sync::atomic::{AtomicUsize, Ordering};

        // Simple memory tracking allocator
        struct TrackingAllocator {
            allocated: AtomicUsize,
            peak: AtomicUsize,
        }

        impl TrackingAllocator {
            const fn new() -> Self {
                Self {
                    allocated: AtomicUsize::new(0),
                    peak: AtomicUsize::new(0),
                }
            }

            fn current_usage(&self) -> usize {
                self.allocated.load(Ordering::Relaxed)
            }

            fn peak_usage(&self) -> usize {
                self.peak.load(Ordering::Relaxed)
            }

            fn reset(&self) {
                self.allocated.store(0, Ordering::Relaxed);
                self.peak.store(0, Ordering::Relaxed);
            }
        }

        unsafe impl GlobalAlloc for TrackingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let ptr = System.alloc(layout);
                if !ptr.is_null() {
                    let size = layout.size();
                    let old_allocated = self.allocated.fetch_add(size, Ordering::Relaxed);
                    let new_allocated = old_allocated + size;
                    
                    // Update peak if necessary
                    let mut current_peak = self.peak.load(Ordering::Relaxed);
                    while new_allocated > current_peak {
                        match self.peak.compare_exchange_weak(
                            current_peak,
                            new_allocated,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(x) => current_peak = x,
                        }
                    }
                }
                ptr
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                System.dealloc(ptr, layout);
                self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
            }
        }

        // Test memory usage tracking
        let tracker = TrackingAllocator::new();
        tracker.reset();

        let initial_usage = tracker.current_usage();
        
        // Allocate some memory
        let _vec: Vec<u64> = vec![0; 1000];
        let after_allocation = tracker.current_usage();
        
        assert!(after_allocation > initial_usage);
        assert!(tracker.peak_usage() >= after_allocation);
    }

    #[test]
    fn test_cpu_utilization_monitoring() {
        use std::thread;
        use std::time::{Duration, Instant};

        fn cpu_intensive_task(duration: Duration) {
            let start = Instant::now();
            let mut counter = 0u64;
            
            while start.elapsed() < duration {
                // CPU-intensive computation
                counter = counter.wrapping_add(1);
                if counter % 1000000 == 0 {
                    // Prevent optimization
                    std::hint::black_box(counter);
                }
            }
        }

        // Measure CPU usage during intensive task
        let start_time = Instant::now();
        cpu_intensive_task(Duration::from_millis(100));
        let elapsed = start_time.elapsed();

        // The task should take approximately the requested time
        assert!(elapsed >= Duration::from_millis(90));
        assert!(elapsed <= Duration::from_millis(200));
    }

    criterion_group!(
        benches,
        benchmark_avellaneda_stoikov_quote_generation,
        benchmark_sde_solvers,
        benchmark_hawkes_process,
        benchmark_lock_free_order_book,
        benchmark_memory_operations,
        benchmark_simd_operations
    );
}

criterion_main!(performance_regression_testing::benches);