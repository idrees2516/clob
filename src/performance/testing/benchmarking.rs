//! Benchmarking Suite
//! 
//! Provides baseline performance measurement and comparative benchmarking tools

use super::{TestConfig, TestResults, LatencyStats, ThroughputStats};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub description: String,
    pub iterations: u32,
    pub warmup_iterations: u32,
    pub measurement_duration: Duration,
    pub baseline_file: Option<String>,
    pub comparison_mode: ComparisonMode,
}

/// Comparison mode for benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonMode {
    Baseline,           // Establish new baseline
    Compare,            // Compare against existing baseline
    Regression,         // Check for performance regression
    Improvement,        // Validate performance improvement
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub timestamp: u64,
    pub measurements: Vec<BenchmarkMeasurement>,
    pub statistics: BenchmarkStatistics,
    pub comparison: Option<BenchmarkComparison>,
    pub verdict: BenchmarkVerdict,
}

/// Individual benchmark measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub iteration: u32,
    pub latency_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub timestamp: u64,
}

/// Benchmark statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStatistics {
    pub latency_stats: LatencyStats,
    pub throughput_stats: ThroughputStats,
    pub stability_coefficient: f64,
    pub performance_index: f64,
}

/// Benchmark comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub baseline_name: String,
    pub latency_change_percent: f64,
    pub throughput_change_percent: f64,
    pub performance_change_percent: f64,
    pub significance_level: f64,
    pub is_significant: bool,
}

/// Benchmark verdict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkVerdict {
    Pass,
    Fail,
    Regression,
    Improvement,
    Inconclusive,
}

/// Benchmarking suite
pub struct BenchmarkingSuite {
    baselines: HashMap<String, BenchmarkResults>,
    results_history: Vec<BenchmarkResults>,
}

impl BenchmarkingSuite {
    /// Create a new benchmarking suite
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            results_history: Vec::new(),
        }
    }

    /// Run a benchmark
    pub async fn run_benchmark(&mut self, config: BenchmarkConfig) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        println!("Running benchmark: {}", config.name);
        
        let start_time = Instant::now();
        let mut measurements = Vec::new();

        // Warmup phase
        println!("Warmup phase: {} iterations", config.warmup_iterations);
        for _ in 0..config.warmup_iterations {
            self.run_single_measurement().await?;
        }

        // Measurement phase
        println!("Measurement phase: {} iterations", config.iterations);
        for iteration in 0..config.iterations {
            let measurement = self.run_single_measurement().await?;
            measurements.push(BenchmarkMeasurement {
                iteration,
                latency_ns: measurement.0,
                throughput_ops_per_sec: measurement.1,
                cpu_usage_percent: measurement.2,
                memory_usage_bytes: measurement.3,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
            });

            // Progress reporting
            if iteration % (config.iterations / 10).max(1) == 0 {
                println!("Progress: {}/{} iterations", iteration + 1, config.iterations);
            }
        }

        // Calculate statistics
        let statistics = self.calculate_benchmark_statistics(&measurements);

        // Perform comparison if baseline exists
        let comparison = match config.comparison_mode {
            ComparisonMode::Baseline => {
                // Store as new baseline
                None
            }
            ComparisonMode::Compare | ComparisonMode::Regression | ComparisonMode::Improvement => {
                self.compare_with_baseline(&config.name, &statistics)
            }
        };

        // Determine verdict
        let verdict = self.determine_benchmark_verdict(&config, &statistics, &comparison);

        let results = BenchmarkResults {
            config: config.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            measurements,
            statistics,
            comparison,
            verdict,
        };

        // Store results
        if matches!(config.comparison_mode, ComparisonMode::Baseline) {
            self.baselines.insert(config.name.clone(), results.clone());
        }
        self.results_history.push(results.clone());

        let duration = start_time.elapsed();
        println!("Benchmark completed in {:?}", duration);

        Ok(results)
    }

    /// Run a single benchmark measurement
    async fn run_single_measurement(&self) -> Result<(u64, f64, f64, u64), Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        // Simulate workload
        self.simulate_benchmark_workload().await;
        
        let end_time = std::time::Instant::now();
        let latency_ns = (end_time - start_time).as_nanos() as u64;
        
        // Simulate metrics collection
        let throughput_ops_per_sec = 1_000_000.0 + (rand::random::<f64>() - 0.5) * 100_000.0;
        let cpu_usage_percent = 50.0 + (rand::random::<f64>() - 0.5) * 20.0;
        let memory_usage_bytes = 1_000_000_000 + (rand::random::<u64>() % 100_000_000);
        
        Ok((latency_ns, throughput_ops_per_sec, cpu_usage_percent, memory_usage_bytes))
    }

    /// Simulate benchmark workload
    async fn simulate_benchmark_workload(&self) {
        // Simulate CPU-intensive work
        let mut sum = 0u64;
        for i in 0..10000 {
            sum = sum.wrapping_add(i * i);
        }
        
        // Simulate I/O wait
        tokio::time::sleep(Duration::from_nanos(100)).await;
        
        // Prevent optimization
        std::hint::black_box(sum);
    }

    /// Calculate benchmark statistics
    fn calculate_benchmark_statistics(&self, measurements: &[BenchmarkMeasurement]) -> BenchmarkStatistics {
        if measurements.is_empty() {
            return BenchmarkStatistics {
                latency_stats: LatencyStats::default(),
                throughput_stats: ThroughputStats::default(),
                stability_coefficient: 0.0,
                performance_index: 0.0,
            };
        }

        // Extract latency values
        let mut latencies: Vec<u64> = measurements.iter().map(|m| m.latency_ns).collect();
        latencies.sort_unstable();

        let sample_count = latencies.len() as u64;
        let min_ns = *latencies.first().unwrap_or(&0);
        let max_ns = *latencies.last().unwrap_or(&0);
        
        // Calculate latency statistics
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

        let latency_stats = LatencyStats {
            min_ns,
            max_ns,
            mean_ns,
            median_ns,
            p95_ns,
            p99_ns,
            p99_9_ns,
            std_dev_ns,
            sample_count,
        };

        // Calculate throughput statistics
        let throughputs: Vec<f64> = measurements.iter().map(|m| m.throughput_ops_per_sec).collect();
        let total_operations = throughputs.iter().sum::<f64>() as u64;
        let operations_per_second = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let peak_ops_per_second = throughputs.iter().fold(0.0f64, |a, &b| a.max(b));
        let average_ops_per_second = operations_per_second;

        let throughput_stats = ThroughputStats {
            total_operations,
            operations_per_second,
            peak_ops_per_second,
            average_ops_per_second,
            duration: Duration::from_secs(measurements.len() as u64),
        };

        // Calculate stability coefficient (coefficient of variation)
        let stability_coefficient = if mean_ns > 0.0 {
            1.0 - (std_dev_ns / mean_ns).min(1.0)
        } else {
            0.0
        };

        // Calculate performance index (composite score)
        let latency_score = if median_ns > 0 { 1_000_000.0 / median_ns as f64 } else { 0.0 };
        let throughput_score = operations_per_second / 1_000_000.0; // Normalize to millions
        let performance_index = (latency_score + throughput_score) * stability_coefficient;

        BenchmarkStatistics {
            latency_stats,
            throughput_stats,
            stability_coefficient,
            performance_index,
        }
    }

    /// Calculate percentile from sorted values
    fn calculate_percentile(&self, sorted_values: &[u64], percentile: f64) -> u64 {
        if sorted_values.is_empty() {
            return 0;
        }
        
        let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Compare with baseline
    fn compare_with_baseline(&self, benchmark_name: &str, statistics: &BenchmarkStatistics) -> Option<BenchmarkComparison> {
        if let Some(baseline) = self.baselines.get(benchmark_name) {
            let baseline_stats = &baseline.statistics;
            
            // Calculate percentage changes
            let latency_change_percent = if baseline_stats.latency_stats.median_ns > 0 {
                ((statistics.latency_stats.median_ns as f64 - baseline_stats.latency_stats.median_ns as f64) 
                 / baseline_stats.latency_stats.median_ns as f64) * 100.0
            } else {
                0.0
            };
            
            let throughput_change_percent = if baseline_stats.throughput_stats.operations_per_second > 0.0 {
                ((statistics.throughput_stats.operations_per_second - baseline_stats.throughput_stats.operations_per_second) 
                 / baseline_stats.throughput_stats.operations_per_second) * 100.0
            } else {
                0.0
            };
            
            let performance_change_percent = if baseline_stats.performance_index > 0.0 {
                ((statistics.performance_index - baseline_stats.performance_index) 
                 / baseline_stats.performance_index) * 100.0
            } else {
                0.0
            };

            // Statistical significance test (simplified)
            let significance_level = self.calculate_significance(statistics, baseline_stats);
            let is_significant = significance_level > 0.95; // 95% confidence

            Some(BenchmarkComparison {
                baseline_name: baseline.config.name.clone(),
                latency_change_percent,
                throughput_change_percent,
                performance_change_percent,
                significance_level,
                is_significant,
            })
        } else {
            None
        }
    }

    /// Calculate statistical significance (simplified t-test approximation)
    fn calculate_significance(&self, current: &BenchmarkStatistics, baseline: &BenchmarkStatistics) -> f64 {
        // Simplified significance calculation
        // In a real implementation, this would perform proper statistical tests
        
        let latency_diff = (current.latency_stats.median_ns as f64 - baseline.latency_stats.median_ns as f64).abs();
        let latency_pooled_std = (current.latency_stats.std_dev_ns + baseline.latency_stats.std_dev_ns) / 2.0;
        
        if latency_pooled_std > 0.0 {
            let t_stat = latency_diff / latency_pooled_std;
            // Convert t-statistic to approximate confidence level
            (1.0 - (-t_stat.abs()).exp()).min(0.99)
        } else {
            0.5 // No confidence if no variance
        }
    }

    /// Determine benchmark verdict
    fn determine_benchmark_verdict(
        &self,
        config: &BenchmarkConfig,
        statistics: &BenchmarkStatistics,
        comparison: &Option<BenchmarkComparison>,
    ) -> BenchmarkVerdict {
        match &config.comparison_mode {
            ComparisonMode::Baseline => BenchmarkVerdict::Pass,
            ComparisonMode::Compare => {
                if let Some(comp) = comparison {
                    if comp.is_significant {
                        if comp.performance_change_percent > 5.0 {
                            BenchmarkVerdict::Improvement
                        } else if comp.performance_change_percent < -5.0 {
                            BenchmarkVerdict::Regression
                        } else {
                            BenchmarkVerdict::Pass
                        }
                    } else {
                        BenchmarkVerdict::Inconclusive
                    }
                } else {
                    BenchmarkVerdict::Fail
                }
            }
            ComparisonMode::Regression => {
                if let Some(comp) = comparison {
                    if comp.is_significant && comp.performance_change_percent < -5.0 {
                        BenchmarkVerdict::Regression
                    } else {
                        BenchmarkVerdict::Pass
                    }
                } else {
                    BenchmarkVerdict::Fail
                }
            }
            ComparisonMode::Improvement => {
                if let Some(comp) = comparison {
                    if comp.is_significant && comp.performance_change_percent > 5.0 {
                        BenchmarkVerdict::Improvement
                    } else {
                        BenchmarkVerdict::Fail
                    }
                } else {
                    BenchmarkVerdict::Fail
                }
            }
        }
    }

    /// Run standard benchmark suite
    pub async fn run_standard_benchmarks(&mut self) -> Result<Vec<BenchmarkResults>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Order processing benchmark
        let order_processing_config = BenchmarkConfig {
            name: "order_processing".to_string(),
            description: "Benchmark order processing latency".to_string(),
            iterations: 1000,
            warmup_iterations: 100,
            measurement_duration: Duration::from_secs(60),
            baseline_file: None,
            comparison_mode: ComparisonMode::Baseline,
        };
        results.push(self.run_benchmark(order_processing_config).await?);

        // Throughput benchmark
        let throughput_config = BenchmarkConfig {
            name: "throughput_test".to_string(),
            description: "Benchmark maximum throughput".to_string(),
            iterations: 500,
            warmup_iterations: 50,
            measurement_duration: Duration::from_secs(30),
            baseline_file: None,
            comparison_mode: ComparisonMode::Baseline,
        };
        results.push(self.run_benchmark(throughput_config).await?);

        // Memory efficiency benchmark
        let memory_config = BenchmarkConfig {
            name: "memory_efficiency".to_string(),
            description: "Benchmark memory usage efficiency".to_string(),
            iterations: 200,
            warmup_iterations: 20,
            measurement_duration: Duration::from_secs(45),
            baseline_file: None,
            comparison_mode: ComparisonMode::Baseline,
        };
        results.push(self.run_benchmark(memory_config).await?);

        Ok(results)
    }

    /// Run comparative benchmarks
    pub async fn run_comparative_benchmarks(&mut self, baseline_name: &str) -> Result<Vec<BenchmarkResults>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Compare order processing
        let order_processing_config = BenchmarkConfig {
            name: "order_processing".to_string(),
            description: "Compare order processing performance".to_string(),
            iterations: 1000,
            warmup_iterations: 100,
            measurement_duration: Duration::from_secs(60),
            baseline_file: Some(baseline_name.to_string()),
            comparison_mode: ComparisonMode::Compare,
        };
        results.push(self.run_benchmark(order_processing_config).await?);

        // Compare throughput
        let throughput_config = BenchmarkConfig {
            name: "throughput_test".to_string(),
            description: "Compare throughput performance".to_string(),
            iterations: 500,
            warmup_iterations: 50,
            measurement_duration: Duration::from_secs(30),
            baseline_file: Some(baseline_name.to_string()),
            comparison_mode: ComparisonMode::Compare,
        };
        results.push(self.run_benchmark(throughput_config).await?);

        Ok(results)
    }

    /// Generate benchmark report
    pub fn generate_benchmark_report(&self, results: &BenchmarkResults) -> String {
        let mut report = format!(
            r#"
# Benchmark Report: {}

**Description:** {}
**Timestamp:** {}
**Iterations:** {}
**Verdict:** {:?}

## Performance Statistics

### Latency
- **Median:** {:.0}ns
- **95th Percentile:** {:.0}ns
- **99th Percentile:** {:.0}ns
- **99.9th Percentile:** {:.0}ns
- **Standard Deviation:** {:.2}ns

### Throughput
- **Operations/Second:** {:.0}
- **Peak Performance:** {:.0} ops/sec
- **Average Performance:** {:.0} ops/sec

### Stability
- **Stability Coefficient:** {:.3}
- **Performance Index:** {:.2}
"#,
            results.config.name,
            results.config.description,
            results.timestamp,
            results.config.iterations,
            results.verdict,
            results.statistics.latency_stats.median_ns,
            results.statistics.latency_stats.p95_ns,
            results.statistics.latency_stats.p99_ns,
            results.statistics.latency_stats.p99_9_ns,
            results.statistics.latency_stats.std_dev_ns,
            results.statistics.throughput_stats.operations_per_second,
            results.statistics.throughput_stats.peak_ops_per_second,
            results.statistics.throughput_stats.average_ops_per_second,
            results.statistics.stability_coefficient,
            results.statistics.performance_index,
        );

        // Add comparison section if available
        if let Some(comparison) = &results.comparison {
            report.push_str(&format!(
                r#"
## Comparison with Baseline: {}

- **Latency Change:** {:.1}%
- **Throughput Change:** {:.1}%
- **Performance Change:** {:.1}%
- **Statistical Significance:** {:.1}% confidence
- **Significant Change:** {}

### Assessment
{}
"#,
                comparison.baseline_name,
                comparison.latency_change_percent,
                comparison.throughput_change_percent,
                comparison.performance_change_percent,
                comparison.significance_level * 100.0,
                if comparison.is_significant { "Yes" } else { "No" },
                match results.verdict {
                    BenchmarkVerdict::Improvement => "✓ Performance improved significantly",
                    BenchmarkVerdict::Regression => "⚠ Performance regression detected",
                    BenchmarkVerdict::Pass => "✓ Performance within acceptable range",
                    BenchmarkVerdict::Fail => "✗ Performance below expectations",
                    BenchmarkVerdict::Inconclusive => "? Results inconclusive",
                }
            ));
        }

        report
    }

    /// Export benchmark results
    pub fn export_results(&self, results: &BenchmarkResults, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            "json" => Ok(serde_json::to_string_pretty(results)?),
            "csv" => Ok(self.generate_csv_export(results)),
            "report" => Ok(self.generate_benchmark_report(results)),
            _ => Err("Unsupported export format".into()),
        }
    }

    /// Generate CSV export
    fn generate_csv_export(&self, results: &BenchmarkResults) -> String {
        let mut csv = String::from("iteration,latency_ns,throughput_ops_per_sec,cpu_usage_percent,memory_usage_bytes,timestamp\n");
        
        for measurement in &results.measurements {
            csv.push_str(&format!(
                "{},{},{:.2},{:.2},{},{}\n",
                measurement.iteration,
                measurement.latency_ns,
                measurement.throughput_ops_per_sec,
                measurement.cpu_usage_percent,
                measurement.memory_usage_bytes,
                measurement.timestamp
            ));
        }
        
        csv
    }

    /// Get benchmark history
    pub fn get_benchmark_history(&self, benchmark_name: &str) -> Vec<&BenchmarkResults> {
        self.results_history
            .iter()
            .filter(|r| r.config.name == benchmark_name)
            .collect()
    }

    /// Get all baselines
    pub fn get_baselines(&self) -> &HashMap<String, BenchmarkResults> {
        &self.baselines
    }

    /// Load baseline from file
    pub fn load_baseline(&mut self, name: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // In a real implementation, this would load from file
        // For now, we'll create a mock baseline
        let mock_baseline = BenchmarkResults {
            config: BenchmarkConfig {
                name: name.to_string(),
                description: "Loaded baseline".to_string(),
                iterations: 1000,
                warmup_iterations: 100,
                measurement_duration: Duration::from_secs(60),
                baseline_file: Some(file_path.to_string()),
                comparison_mode: ComparisonMode::Baseline,
            },
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            measurements: Vec::new(),
            statistics: BenchmarkStatistics {
                latency_stats: LatencyStats {
                    median_ns: 500,
                    p95_ns: 1000,
                    p99_ns: 2000,
                    ..LatencyStats::default()
                },
                throughput_stats: ThroughputStats {
                    operations_per_second: 1_000_000.0,
                    ..ThroughputStats::default()
                },
                stability_coefficient: 0.95,
                performance_index: 100.0,
            },
            comparison: None,
            verdict: BenchmarkVerdict::Pass,
        };

        self.baselines.insert(name.to_string(), mock_baseline);
        Ok(())
    }

    /// Save baseline to file
    pub fn save_baseline(&self, name: &str, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(baseline) = self.baselines.get(name) {
            let json = serde_json::to_string_pretty(baseline)?;
            std::fs::write(file_path, json)?;
            println!("Baseline '{}' saved to {}", name, file_path);
        }
        Ok(())
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default_benchmark".to_string(),
            description: "Default benchmark configuration".to_string(),
            iterations: 1000,
            warmup_iterations: 100,
            measurement_duration: Duration::from_secs(60),
            baseline_file: None,
            comparison_mode: ComparisonMode::Baseline,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_benchmarking_suite_creation() {
        let suite = BenchmarkingSuite::new();
        assert_eq!(suite.baselines.len(), 0);
        assert_eq!(suite.results_history.len(), 0);
    }

    #[tokio::test]
    async fn test_single_measurement() {
        let suite = BenchmarkingSuite::new();
        let result = suite.run_single_measurement().await.unwrap();
        
        assert!(result.0 > 0); // latency_ns
        assert!(result.1 > 0.0); // throughput_ops_per_sec
        assert!(result.2 >= 0.0 && result.2 <= 100.0); // cpu_usage_percent
        assert!(result.3 > 0); // memory_usage_bytes
    }

    #[tokio::test]
    async fn test_benchmark_run() {
        let mut suite = BenchmarkingSuite::new();
        let config = BenchmarkConfig {
            name: "test_benchmark".to_string(),
            description: "Test benchmark".to_string(),
            iterations: 10,
            warmup_iterations: 2,
            measurement_duration: Duration::from_secs(1),
            baseline_file: None,
            comparison_mode: ComparisonMode::Baseline,
        };

        let results = suite.run_benchmark(config).await.unwrap();
        assert_eq!(results.measurements.len(), 10);
        assert!(matches!(results.verdict, BenchmarkVerdict::Pass));
    }

    #[test]
    fn test_percentile_calculation() {
        let suite = BenchmarkingSuite::new();
        let values = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        
        assert_eq!(suite.calculate_percentile(&values, 50.0), 500);
        assert_eq!(suite.calculate_percentile(&values, 90.0), 900);
        assert_eq!(suite.calculate_percentile(&values, 95.0), 1000);
    }
}