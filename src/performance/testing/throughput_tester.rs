//! Throughput Measurement Tools
//! 
//! Provides comprehensive throughput testing and measurement capabilities

use super::{TestConfig, ThroughputStats};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Throughput measurement sample
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    pub timestamp: Instant,
    pub operations_count: u64,
    pub duration: Duration,
    pub thread_id: usize,
}

/// Throughput test configuration
#[derive(Debug, Clone)]
pub struct ThroughputTestConfig {
    pub target_ops_per_second: u64,
    pub measurement_interval: Duration,
    pub thread_count: usize,
    pub batch_size: usize,
    pub ramp_up_duration: Duration,
    pub steady_state_duration: Duration,
    pub ramp_down_duration: Duration,
}

/// Throughput measurement tester
pub struct ThroughputTester {
    total_operations: AtomicU64,
    peak_ops_per_second: AtomicU64,
    samples: Arc<std::sync::Mutex<Vec<ThroughputSample>>>,
}

impl ThroughputTester {
    /// Create a new throughput tester
    pub fn new() -> Self {
        Self {
            total_operations: AtomicU64::new(0),
            peak_ops_per_second: AtomicU64::new(0),
            samples: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Run throughput test
    pub async fn run_test(&mut self, config: &TestConfig) -> Result<ThroughputStats, Box<dyn std::error::Error>> {
        println!("Starting throughput test for {:?}", config.duration);
        
        let throughput_config = ThroughputTestConfig {
            target_ops_per_second: config.target_throughput,
            measurement_interval: Duration::from_millis(100), // Measure every 100ms
            thread_count: config.thread_count,
            batch_size: config.batch_size,
            ramp_up_duration: Duration::from_secs(5),
            steady_state_duration: config.duration - Duration::from_secs(10), // Account for ramp up/down
            ramp_down_duration: Duration::from_secs(5),
        };

        // Reset counters
        self.total_operations.store(0, Ordering::Relaxed);
        self.peak_ops_per_second.store(0, Ordering::Relaxed);
        self.samples.lock().unwrap().clear();

        let start_time = Instant::now();

        // Start measurement collection
        let (tx, mut rx) = mpsc::channel(1000);
        let samples_clone = Arc::clone(&self.samples);
        
        let collection_task = tokio::spawn(async move {
            while let Some(sample) = rx.recv().await {
                samples_clone.lock().unwrap().push(sample);
            }
        });

        // Start throughput generation tasks
        let generation_tasks = self.start_throughput_tasks(&throughput_config, tx).await?;

        // Run test phases
        self.run_test_phases(&throughput_config).await?;

        // Stop generation tasks
        for task in generation_tasks {
            task.abort();
        }

        // Stop collection task
        collection_task.abort();

        let end_time = Instant::now();
        let total_duration = end_time - start_time;

        // Analyze results
        let stats = self.analyze_throughput_samples(total_duration)?;
        
        println!("Throughput test completed. Total operations: {}", stats.total_operations);
        Ok(stats)
    }

    /// Start throughput generation tasks
    async fn start_throughput_tasks(
        &self,
        config: &ThroughputTestConfig,
        tx: mpsc::Sender<ThroughputSample>,
    ) -> Result<Vec<tokio::task::JoinHandle<()>>, Box<dyn std::error::Error>> {
        let mut tasks = Vec::new();
        let ops_per_thread = config.target_ops_per_second / config.thread_count as u64;
        
        for thread_id in 0..config.thread_count {
            let tx_clone = tx.clone();
            let total_operations = Arc::clone(&self.total_operations);
            let peak_ops_per_second = Arc::clone(&self.peak_ops_per_second);
            let batch_size = config.batch_size;
            let measurement_interval = config.measurement_interval;
            
            let task = tokio::spawn(async move {
                let mut interval = tokio::time::interval(measurement_interval);
                let mut last_measurement = Instant::now();
                let mut operations_in_interval = 0u64;
                
                // Calculate delay between operations to achieve target throughput
                let target_delay = Duration::from_nanos(1_000_000_000 / ops_per_thread);
                let mut operation_interval = tokio::time::interval(target_delay);
                
                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            // Record throughput sample
                            let now = Instant::now();
                            let duration = now - last_measurement;
                            
                            if operations_in_interval > 0 {
                                let sample = ThroughputSample {
                                    timestamp: now,
                                    operations_count: operations_in_interval,
                                    duration,
                                    thread_id,
                                };
                                
                                // Calculate ops per second for this interval
                                let ops_per_second = (operations_in_interval as f64 / duration.as_secs_f64()) as u64;
                                
                                // Update peak if necessary
                                let current_peak = peak_ops_per_second.load(Ordering::Relaxed);
                                if ops_per_second > current_peak {
                                    peak_ops_per_second.store(ops_per_second, Ordering::Relaxed);
                                }
                                
                                if tx_clone.send(sample).await.is_err() {
                                    break;
                                }
                            }
                            
                            last_measurement = now;
                            operations_in_interval = 0;
                        }
                        _ = operation_interval.tick() => {
                            // Perform batch of operations
                            for _ in 0..batch_size {
                                Self::simulate_operation().await;
                                total_operations.fetch_add(1, Ordering::Relaxed);
                                operations_in_interval += 1;
                            }
                        }
                    }
                }
            });
            
            tasks.push(task);
        }
        
        Ok(tasks)
    }

    /// Run test phases (ramp-up, steady-state, ramp-down)
    async fn run_test_phases(&self, config: &ThroughputTestConfig) -> Result<(), Box<dyn std::error::Error>> {
        // Ramp-up phase
        println!("Ramp-up phase: {:?}", config.ramp_up_duration);
        tokio::time::sleep(config.ramp_up_duration).await;
        
        // Steady-state phase
        println!("Steady-state phase: {:?}", config.steady_state_duration);
        tokio::time::sleep(config.steady_state_duration).await;
        
        // Ramp-down phase
        println!("Ramp-down phase: {:?}", config.ramp_down_duration);
        tokio::time::sleep(config.ramp_down_duration).await;
        
        Ok(())
    }

    /// Simulate a single operation
    async fn simulate_operation() {
        // Simulate minimal operation latency
        tokio::time::sleep(Duration::from_nanos(50)).await;
    }

    /// Analyze throughput samples
    fn analyze_throughput_samples(&self, total_duration: Duration) -> Result<ThroughputStats, Box<dyn std::error::Error>> {
        let samples = self.samples.lock().unwrap();
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        let peak_ops_per_second = self.peak_ops_per_second.load(Ordering::Relaxed) as f64;
        
        // Calculate average operations per second
        let average_ops_per_second = if total_duration.as_secs_f64() > 0.0 {
            total_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };
        
        // Calculate overall operations per second
        let operations_per_second = average_ops_per_second;
        
        Ok(ThroughputStats {
            total_operations,
            operations_per_second,
            peak_ops_per_second,
            average_ops_per_second,
            duration: total_duration,
        })
    }

    /// Test sustained throughput
    pub async fn test_sustained_throughput(
        &mut self,
        target_ops_per_second: u64,
        duration: Duration,
        thread_count: usize,
    ) -> Result<ThroughputStats, Box<dyn std::error::Error>> {
        let config = TestConfig {
            test_name: "sustained_throughput".to_string(),
            duration,
            target_throughput: target_ops_per_second,
            thread_count,
            ..TestConfig::default()
        };
        
        self.run_test(&config).await
    }

    /// Test burst throughput
    pub async fn test_burst_throughput(
        &mut self,
        burst_ops: u64,
        burst_duration: Duration,
        thread_count: usize,
    ) -> Result<ThroughputStats, Box<dyn std::error::Error>> {
        println!("Starting burst throughput test: {} ops in {:?}", burst_ops, burst_duration);
        
        self.total_operations.store(0, Ordering::Relaxed);
        let start_time = Instant::now();
        
        // Create burst generation tasks
        let mut tasks = Vec::new();
        let ops_per_thread = burst_ops / thread_count as u64;
        
        for _ in 0..thread_count {
            let total_operations = Arc::clone(&self.total_operations);
            let ops_to_generate = ops_per_thread;
            
            let task = tokio::spawn(async move {
                for _ in 0..ops_to_generate {
                    Self::simulate_operation().await;
                    total_operations.fetch_add(1, Ordering::Relaxed);
                }
            });
            
            tasks.push(task);
        }
        
        // Wait for all tasks to complete or timeout
        let timeout_task = tokio::time::sleep(burst_duration);
        let completion_task = async {
            for task in tasks {
                let _ = task.await;
            }
        };
        
        tokio::select! {
            _ = timeout_task => {
                println!("Burst test timed out after {:?}", burst_duration);
            }
            _ = completion_task => {
                println!("Burst test completed all operations");
            }
        }
        
        let end_time = Instant::now();
        let actual_duration = end_time - start_time;
        let total_operations = self.total_operations.load(Ordering::Relaxed);
        
        let operations_per_second = total_operations as f64 / actual_duration.as_secs_f64();
        
        Ok(ThroughputStats {
            total_operations,
            operations_per_second,
            peak_ops_per_second: operations_per_second,
            average_ops_per_second: operations_per_second,
            duration: actual_duration,
        })
    }

    /// Generate throughput report
    pub fn generate_throughput_report(&self, stats: &ThroughputStats) -> String {
        format!(
            r#"
=== Throughput Test Report ===
Total Operations: {}
Test Duration: {:.2} seconds
Operations per Second: {:.2}
Peak Operations per Second: {:.2}
Average Operations per Second: {:.2}

Performance Assessment:
- Target 1M ops/sec: {}
- HFT competitive (>500K ops/sec): {}
- High throughput (>100K ops/sec): {}

Efficiency Metrics:
- Operations per millisecond: {:.2}
- Operations per microsecond: {:.6}
"#,
            stats.total_operations,
            stats.duration.as_secs_f64(),
            stats.operations_per_second,
            stats.peak_ops_per_second,
            stats.average_ops_per_second,
            if stats.operations_per_second >= 1_000_000.0 { "✓ PASS" } else { "✗ FAIL" },
            if stats.operations_per_second >= 500_000.0 { "✓ PASS" } else { "✗ FAIL" },
            if stats.operations_per_second >= 100_000.0 { "✓ PASS" } else { "✗ FAIL" },
            stats.operations_per_second / 1_000.0,
            stats.operations_per_second / 1_000_000.0,
        )
    }

    /// Test throughput under different load patterns
    pub async fn test_load_patterns(&mut self) -> Result<Vec<ThroughputStats>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        
        // Test different load patterns
        let patterns = vec![
            ("low_load", 10_000, Duration::from_secs(30)),
            ("medium_load", 100_000, Duration::from_secs(30)),
            ("high_load", 500_000, Duration::from_secs(30)),
            ("extreme_load", 1_000_000, Duration::from_secs(30)),
        ];
        
        for (pattern_name, target_ops, duration) in patterns {
            println!("Testing load pattern: {}", pattern_name);
            
            let stats = self.test_sustained_throughput(target_ops, duration, num_cpus::get()).await?;
            results.push(stats);
            
            // Cool down between tests
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_throughput_tester_creation() {
        let tester = ThroughputTester::new();
        assert_eq!(tester.total_operations.load(Ordering::Relaxed), 0);
    }
    
    #[tokio::test]
    async fn test_burst_throughput() {
        let mut tester = ThroughputTester::new();
        let stats = tester.test_burst_throughput(1000, Duration::from_secs(1), 4).await.unwrap();
        
        assert!(stats.total_operations > 0);
        assert!(stats.operations_per_second > 0.0);
        assert!(stats.duration <= Duration::from_secs(2)); // Should complete within reasonable time
    }
    
    #[tokio::test]
    async fn test_sustained_throughput() {
        let mut tester = ThroughputTester::new();
        let stats = tester.test_sustained_throughput(1000, Duration::from_secs(2), 2).await.unwrap();
        
        assert!(stats.total_operations > 0);
        assert!(stats.operations_per_second > 0.0);
        assert!(stats.duration >= Duration::from_secs(1)); // Should run for reasonable duration
    }
}