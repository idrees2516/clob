//! Automated Latency Testing
//! 
//! Provides comprehensive latency measurement and analysis for the CLOB system

use super::{TestConfig, LatencyStats};
use crate::performance::timing::nanosecond_timer::NanosecondTimer;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Latency measurement sample
#[derive(Debug, Clone)]
pub struct LatencySample {
    pub timestamp: u64,
    pub latency_ns: u64,
    pub operation_type: String,
    pub thread_id: usize,
}

/// Latency test configuration
#[derive(Debug, Clone)]
pub struct LatencyTestConfig {
    pub sample_rate: u64, // Samples per second
    pub measurement_duration: Duration,
    pub operation_types: Vec<String>,
    pub percentiles: Vec<f64>,
    pub enable_histogram: bool,
    pub histogram_buckets: usize,
}

/// Automated latency tester
pub struct LatencyTester {
    timer: NanosecondTimer,
    samples: Arc<std::sync::Mutex<VecDeque<LatencySample>>>,
    sample_count: AtomicU64,
}

impl LatencyTester {
    /// Create a new latency tester
    pub fn new() -> Self {
        Self {
            timer: NanosecondTimer::new(),
            samples: Arc::new(std::sync::Mutex::new(VecDeque::new())),
            sample_count: AtomicU64::new(0),
        }
    }

    /// Run automated latency test
    pub async fn run_test(&mut self, config: &TestConfig) -> Result<LatencyStats, Box<dyn std::error::Error>> {
        println!("Starting latency test for {:?}", config.duration);
        
        let latency_config = LatencyTestConfig {
            sample_rate: 10_000, // 10k samples per second
            measurement_duration: config.duration,
            operation_types: vec![
                "order_processing".to_string(),
                "order_matching".to_string(),
                "trade_execution".to_string(),
                "market_data_update".to_string(),
            ],
            percentiles: vec![50.0, 95.0, 99.0, 99.9],
            enable_histogram: true,
            histogram_buckets: 1000,
        };

        // Clear previous samples
        self.samples.lock().unwrap().clear();
        self.sample_count.store(0, Ordering::Relaxed);

        // Start measurement threads
        let (tx, mut rx) = mpsc::channel(10000);
        let samples_clone = Arc::clone(&self.samples);
        let sample_count_clone = Arc::clone(&self.sample_count);

        // Sample collection task
        let collection_task = tokio::spawn(async move {
            while let Some(sample) = rx.recv().await {
                {
                    let mut samples = samples_clone.lock().unwrap();
                    samples.push_back(sample);
                    
                    // Keep only recent samples to prevent memory growth
                    if samples.len() > 1_000_000 {
                        samples.pop_front();
                    }
                }
                sample_count_clone.fetch_add(1, Ordering::Relaxed);
            }
        });

        // Generate test load and measure latency
        let measurement_tasks = self.start_measurement_tasks(&latency_config, tx).await?;

        // Wait for test duration
        tokio::time::sleep(config.duration).await;

        // Stop measurement tasks
        for task in measurement_tasks {
            task.abort();
        }

        // Stop collection task
        collection_task.abort();

        // Analyze results
        let stats = self.analyze_latency_samples(&latency_config)?;
        
        println!("Latency test completed. Samples collected: {}", self.sample_count.load(Ordering::Relaxed));
        Ok(stats)
    }

    /// Start measurement tasks for different operation types
    async fn start_measurement_tasks(
        &self,
        config: &LatencyTestConfig,
        tx: mpsc::Sender<LatencySample>,
    ) -> Result<Vec<tokio::task::JoinHandle<()>>, Box<dyn std::error::Error>> {
        let mut tasks = Vec::new();
        
        for (thread_id, operation_type) in config.operation_types.iter().enumerate() {
            let tx_clone = tx.clone();
            let operation_type_clone = operation_type.clone();
            let timer = self.timer.clone();
            let sample_rate = config.sample_rate;
            
            let task = tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_nanos(1_000_000_000 / sample_rate));
                
                loop {
                    interval.tick().await;
                    
                    // Simulate operation and measure latency
                    let start_time = timer.now_nanos();
                    
                    // Simulate different operation types with varying latencies
                    let simulated_latency = match operation_type_clone.as_str() {
                        "order_processing" => Self::simulate_order_processing().await,
                        "order_matching" => Self::simulate_order_matching().await,
                        "trade_execution" => Self::simulate_trade_execution().await,
                        "market_data_update" => Self::simulate_market_data_update().await,
                        _ => 100, // Default 100ns
                    };
                    
                    let end_time = timer.now_nanos();
                    let measured_latency = end_time - start_time;
                    
                    let sample = LatencySample {
                        timestamp: start_time,
                        latency_ns: measured_latency + simulated_latency,
                        operation_type: operation_type_clone.clone(),
                        thread_id,
                    };
                    
                    if tx_clone.send(sample).await.is_err() {
                        break; // Channel closed
                    }
                }
            });
            
            tasks.push(task);
        }
        
        Ok(tasks)
    }

    /// Simulate order processing latency
    async fn simulate_order_processing() -> u64 {
        // Simulate order validation, risk checks, etc.
        tokio::time::sleep(Duration::from_nanos(200)).await;
        200 // 200ns base latency
    }

    /// Simulate order matching latency
    async fn simulate_order_matching() -> u64 {
        // Simulate order book traversal and matching
        tokio::time::sleep(Duration::from_nanos(300)).await;
        300 // 300ns base latency
    }

    /// Simulate trade execution latency
    async fn simulate_trade_execution() -> u64 {
        // Simulate trade settlement and reporting
        tokio::time::sleep(Duration::from_nanos(150)).await;
        150 // 150ns base latency
    }

    /// Simulate market data update latency
    async fn simulate_market_data_update() -> u64 {
        // Simulate market data processing and distribution
        tokio::time::sleep(Duration::from_nanos(100)).await;
        100 // 100ns base latency
    }

    /// Analyze collected latency samples
    fn analyze_latency_samples(&self, config: &LatencyTestConfig) -> Result<LatencyStats, Box<dyn std::error::Error>> {
        let samples = self.samples.lock().unwrap();
        
        if samples.is_empty() {
            return Ok(LatencyStats::default());
        }

        // Extract latency values and sort for percentile calculation
        let mut latencies: Vec<u64> = samples.iter().map(|s| s.latency_ns).collect();
        latencies.sort_unstable();

        let sample_count = latencies.len() as u64;
        let min_ns = *latencies.first().unwrap_or(&0);
        let max_ns = *latencies.last().unwrap_or(&0);
        
        // Calculate mean
        let sum: u64 = latencies.iter().sum();
        let mean_ns = sum as f64 / sample_count as f64;
        
        // Calculate median
        let median_ns = if sample_count > 0 {
            if sample_count % 2 == 0 {
                let mid1 = latencies[(sample_count / 2 - 1) as usize];
                let mid2 = latencies[(sample_count / 2) as usize];
                (mid1 + mid2) / 2
            } else {
                latencies[(sample_count / 2) as usize]
            }
        } else {
            0
        };
        
        // Calculate percentiles
        let p95_ns = self.calculate_percentile(&latencies, 95.0);
        let p99_ns = self.calculate_percentile(&latencies, 99.0);
        let p99_9_ns = self.calculate_percentile(&latencies, 99.9);
        
        // Calculate standard deviation
        let variance = latencies.iter()
            .map(|&x| {
                let diff = x as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>() / sample_count as f64;
        let std_dev_ns = variance.sqrt();

        Ok(LatencyStats {
            min_ns,
            max_ns,
            mean_ns,
            median_ns,
            p95_ns,
            p99_ns,
            p99_9_ns,
            std_dev_ns,
            sample_count,
        })
    }

    /// Calculate percentile from sorted latency values
    fn calculate_percentile(&self, sorted_latencies: &[u64], percentile: f64) -> u64 {
        if sorted_latencies.is_empty() {
            return 0;
        }
        
        let index = (percentile / 100.0 * (sorted_latencies.len() - 1) as f64).round() as usize;
        sorted_latencies[index.min(sorted_latencies.len() - 1)]
    }

    /// Generate detailed latency report
    pub fn generate_latency_report(&self, stats: &LatencyStats) -> String {
        format!(
            r#"
=== Latency Test Report ===
Sample Count: {}
Min Latency: {} ns
Max Latency: {} ns
Mean Latency: {:.2} ns
Median Latency: {} ns
95th Percentile: {} ns
99th Percentile: {} ns
99.9th Percentile: {} ns
Standard Deviation: {:.2} ns

Performance Assessment:
- Sub-microsecond target (<1000ns): {}
- HFT competitive (<500ns median): {}
- Extreme low latency (<100ns median): {}
"#,
            stats.sample_count,
            stats.min_ns,
            stats.max_ns,
            stats.mean_ns,
            stats.median_ns,
            stats.p95_ns,
            stats.p99_ns,
            stats.p99_9_ns,
            stats.std_dev_ns,
            if stats.p99_ns < 1000 { "✓ PASS" } else { "✗ FAIL" },
            if stats.median_ns < 500 { "✓ PASS" } else { "✗ FAIL" },
            if stats.median_ns < 100 { "✓ PASS" } else { "✗ FAIL" },
        )
    }

    /// Test specific operation latency
    pub async fn test_operation_latency(&mut self, operation_name: &str, iterations: u64) -> Result<LatencyStats, Box<dyn std::error::Error>> {
        let mut latencies = Vec::with_capacity(iterations as usize);
        
        for _ in 0..iterations {
            let start_time = self.timer.now_nanos();
            
            // Simulate the specific operation
            match operation_name {
                "order_processing" => Self::simulate_order_processing().await,
                "order_matching" => Self::simulate_order_matching().await,
                "trade_execution" => Self::simulate_trade_execution().await,
                "market_data_update" => Self::simulate_market_data_update().await,
                _ => 0,
            };
            
            let end_time = self.timer.now_nanos();
            latencies.push(end_time - start_time);
        }
        
        // Analyze results
        latencies.sort_unstable();
        let sample_count = latencies.len() as u64;
        let min_ns = *latencies.first().unwrap_or(&0);
        let max_ns = *latencies.last().unwrap_or(&0);
        let sum: u64 = latencies.iter().sum();
        let mean_ns = sum as f64 / sample_count as f64;
        
        let median_ns = if sample_count > 0 {
            if sample_count % 2 == 0 {
                let mid1 = latencies[(sample_count / 2 - 1) as usize];
                let mid2 = latencies[(sample_count / 2) as usize];
                (mid1 + mid2) / 2
            } else {
                latencies[(sample_count / 2) as usize]
            }
        } else {
            0
        };
        
        let p95_ns = self.calculate_percentile(&latencies, 95.0);
        let p99_ns = self.calculate_percentile(&latencies, 99.0);
        let p99_9_ns = self.calculate_percentile(&latencies, 99.9);
        
        let variance = latencies.iter()
            .map(|&x| {
                let diff = x as f64 - mean_ns;
                diff * diff
            })
            .sum::<f64>() / sample_count as f64;
        let std_dev_ns = variance.sqrt();

        Ok(LatencyStats {
            min_ns,
            max_ns,
            mean_ns,
            median_ns,
            p95_ns,
            p99_ns,
            p99_9_ns,
            std_dev_ns,
            sample_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_latency_tester_creation() {
        let tester = LatencyTester::new();
        assert_eq!(tester.sample_count.load(Ordering::Relaxed), 0);
    }
    
    #[tokio::test]
    async fn test_operation_latency_measurement() {
        let mut tester = LatencyTester::new();
        let stats = tester.test_operation_latency("order_processing", 100).await.unwrap();
        
        assert!(stats.sample_count == 100);
        assert!(stats.min_ns > 0);
        assert!(stats.max_ns >= stats.min_ns);
        assert!(stats.mean_ns > 0.0);
    }
    
    #[test]
    fn test_percentile_calculation() {
        let tester = LatencyTester::new();
        let latencies = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        
        assert_eq!(tester.calculate_percentile(&latencies, 50.0), 500);
        assert_eq!(tester.calculate_percentile(&latencies, 90.0), 900);
        assert_eq!(tester.calculate_percentile(&latencies, 95.0), 1000);
    }
}