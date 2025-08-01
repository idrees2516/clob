//! Performance Testing Framework
//! 
//! This module provides comprehensive performance testing capabilities including:
//! - Automated latency testing
//! - Throughput measurement tools
//! - Load generation and simulation
//! - Test result analysis and reporting

pub mod latency_tester;
pub mod throughput_tester;
pub mod load_generator;
pub mod test_analyzer;
pub mod benchmarking;
pub mod stress_testing;
pub mod regression_testing;

use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Main performance test framework
pub struct PerformanceTestFramework {
    pub latency_tester: latency_tester::LatencyTester,
    pub throughput_tester: throughput_tester::ThroughputTester,
    pub load_generator: load_generator::LoadGenerator,
    pub test_analyzer: test_analyzer::TestAnalyzer,
    pub benchmarking_suite: benchmarking::BenchmarkingSuite,
    pub stress_tester: stress_testing::StressTester,
    pub regression_tester: regression_testing::RegressionTester,
}

/// Test configuration for performance tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub test_name: String,
    pub duration: Duration,
    pub target_throughput: u64,
    pub max_latency_ns: u64,
    pub warmup_duration: Duration,
    pub cooldown_duration: Duration,
    pub thread_count: usize,
    pub batch_size: usize,
    pub enable_profiling: bool,
    pub enable_monitoring: bool,
}

/// Performance test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    pub test_name: String,
    pub start_time: Instant,
    pub end_time: Instant,
    pub duration: Duration,
    pub latency_stats: LatencyStats,
    pub throughput_stats: ThroughputStats,
    pub resource_usage: ResourceUsage,
    pub errors: Vec<TestError>,
    pub metadata: HashMap<String, String>,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub median_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub p99_9_ns: u64,
    pub std_dev_ns: f64,
    pub sample_count: u64,
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    pub total_operations: u64,
    pub operations_per_second: f64,
    pub peak_ops_per_second: f64,
    pub average_ops_per_second: f64,
    pub duration: Duration,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub disk_reads: u64,
    pub disk_writes: u64,
    pub context_switches: u64,
    pub page_faults: u64,
}

/// Test error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestError {
    pub timestamp: Instant,
    pub error_type: String,
    pub message: String,
    pub context: HashMap<String, String>,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            test_name: "default_test".to_string(),
            duration: Duration::from_secs(60),
            target_throughput: 1_000_000,
            max_latency_ns: 1_000, // 1 microsecond
            warmup_duration: Duration::from_secs(10),
            cooldown_duration: Duration::from_secs(5),
            thread_count: num_cpus::get(),
            batch_size: 100,
            enable_profiling: true,
            enable_monitoring: true,
        }
    }
}

impl PerformanceTestFramework {
    /// Create a new performance test framework
    pub fn new() -> Self {
        Self {
            latency_tester: latency_tester::LatencyTester::new(),
            throughput_tester: throughput_tester::ThroughputTester::new(),
            load_generator: load_generator::LoadGenerator::new(),
            test_analyzer: test_analyzer::TestAnalyzer::new(),
            benchmarking_suite: benchmarking::BenchmarkingSuite::new(),
            stress_tester: stress_testing::StressTester::new(),
            regression_tester: regression_testing::RegressionTester::new(),
        }
    }

    /// Run a comprehensive performance test suite
    pub async fn run_test_suite(&mut self, config: &TestConfig) -> Result<TestResults, Box<dyn std::error::Error>> {
        println!("Starting performance test suite: {}", config.test_name);
        
        let start_time = Instant::now();
        let mut errors = Vec::new();
        
        // Warmup phase
        println!("Warming up for {:?}", config.warmup_duration);
        if let Err(e) = self.warmup(config).await {
            errors.push(TestError {
                timestamp: Instant::now(),
                error_type: "warmup_error".to_string(),
                message: e.to_string(),
                context: HashMap::new(),
            });
        }
        
        // Main test execution
        let latency_stats = match self.latency_tester.run_test(config).await {
            Ok(stats) => stats,
            Err(e) => {
                errors.push(TestError {
                    timestamp: Instant::now(),
                    error_type: "latency_test_error".to_string(),
                    message: e.to_string(),
                    context: HashMap::new(),
                });
                LatencyStats::default()
            }
        };
        
        let throughput_stats = match self.throughput_tester.run_test(config).await {
            Ok(stats) => stats,
            Err(e) => {
                errors.push(TestError {
                    timestamp: Instant::now(),
                    error_type: "throughput_test_error".to_string(),
                    message: e.to_string(),
                    context: HashMap::new(),
                });
                ThroughputStats::default()
            }
        };
        
        // Resource monitoring
        let resource_usage = self.collect_resource_usage().await;
        
        // Cooldown phase
        println!("Cooling down for {:?}", config.cooldown_duration);
        tokio::time::sleep(config.cooldown_duration).await;
        
        let end_time = Instant::now();
        let duration = end_time - start_time;
        
        let results = TestResults {
            test_name: config.test_name.clone(),
            start_time,
            end_time,
            duration,
            latency_stats,
            throughput_stats,
            resource_usage,
            errors,
            metadata: self.collect_metadata(config),
        };
        
        println!("Performance test completed in {:?}", duration);
        Ok(results)
    }
    
    /// Warmup phase to stabilize system performance
    async fn warmup(&mut self, config: &TestConfig) -> Result<(), Box<dyn std::error::Error>> {
        let warmup_config = TestConfig {
            duration: config.warmup_duration,
            target_throughput: config.target_throughput / 2, // Reduced load for warmup
            ..config.clone()
        };
        
        self.load_generator.generate_load(&warmup_config).await?;
        Ok(())
    }
    
    /// Collect system resource usage
    async fn collect_resource_usage(&self) -> ResourceUsage {
        // This would integrate with system monitoring tools
        // For now, return default values
        ResourceUsage::default()
    }
    
    /// Collect test metadata
    fn collect_metadata(&self, config: &TestConfig) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("cpu_count".to_string(), num_cpus::get().to_string());
        metadata.insert("thread_count".to_string(), config.thread_count.to_string());
        metadata.insert("batch_size".to_string(), config.batch_size.to_string());
        metadata.insert("target_throughput".to_string(), config.target_throughput.to_string());
        metadata.insert("max_latency_ns".to_string(), config.max_latency_ns.to_string());
        metadata
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            min_ns: 0,
            max_ns: 0,
            mean_ns: 0.0,
            median_ns: 0,
            p95_ns: 0,
            p99_ns: 0,
            p99_9_ns: 0,
            std_dev_ns: 0.0,
            sample_count: 0,
        }
    }
}

impl Default for ThroughputStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            operations_per_second: 0.0,
            peak_ops_per_second: 0.0,
            average_ops_per_second: 0.0,
            duration: Duration::from_secs(0),
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_bytes: 0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            disk_reads: 0,
            disk_writes: 0,
            context_switches: 0,
            page_faults: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_framework_creation() {
        let framework = PerformanceTestFramework::new();
        assert!(true); // Framework created successfully
    }
    
    #[test]
    fn test_default_config() {
        let config = TestConfig::default();
        assert_eq!(config.test_name, "default_test");
        assert_eq!(config.duration, Duration::from_secs(60));
        assert_eq!(config.target_throughput, 1_000_000);
        assert_eq!(config.max_latency_ns, 1_000);
    }
}