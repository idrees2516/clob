//! Regression Testing Automation
//! 
//! Implements continuous performance testing, regression detection, and CI/CD integration

use super::{TestResults, TestConfig, LatencyStats, ThroughputStats, ResourceUsage, TestError};
use super::test_analyzer::{TestAnalysis, TestAnalyzer, AnalysisConfig};
use super::benchmarking::{BenchmarkingSuite, BenchmarkConfig, BenchmarkResults, ComparisonMode};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use tokio::fs;

/// Regression testing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    pub test_suite_name: String,
    pub baseline_threshold_percent: f64,
    pub regression_threshold_percent: f64,
    pub minimum_samples: usize,
    pub historical_window_size: usize,
    pub confidence_level: f64,
    pub auto_baseline_update: bool,
    pub notification_config: NotificationConfig,
    pub ci_integration: CiIntegrationConfig,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    pub enabled: bool,
    pub email_recipients: Vec<String>,
    pub slack_webhook: Option<String>,
    pub severity_levels: Vec<SeverityLevel>,
}

/// CI/CD integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiIntegrationConfig {
    pub enabled: bool,
    pub fail_build_on_regression: bool,
    pub generate_reports: bool,
    pub report_format: Vec<ReportFormat>,
    pub artifact_storage_path: Option<PathBuf>,
}

/// Severity levels for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeverityLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Junit,
    Markdown,
}

/// Regression test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResults {
    pub test_name: String,
    pub timestamp: u64,
    pub current_results: TestResults,
    pub baseline_results: Option<TestResults>,
    pub regression_analysis: RegressionAnalysis,
    pub verdict: RegressionVerdict,
    pub recommendations: Vec<String>,
    pub historical_context: HistoricalContext,
}

/// Regression analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub latency_regression: MetricRegression,
    pub throughput_regression: MetricRegression,
    pub resource_regression: MetricRegression,
    pub overall_regression_score: f64,
    pub statistical_significance: f64,
    pub trend_analysis: TrendAnalysis,
}

/// Individual metric regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRegression {
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub change_percent: f64,
    pub is_regression: bool,
    pub severity: RegressionSeverity,
    pub confidence: f64,
}

/// Regression severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    None,
    Minor,
    Moderate,
    Major,
    Critical,
}

/// Regression test verdict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionVerdict {
    Pass,
    Warning,
    Regression,
    CriticalRegression,
    NoBaseline,
    InsufficientData,
}

/// Historical context for regression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalContext {
    pub sample_count: usize,
    pub time_span_days: f64,
    pub trend_direction: TrendDirection,
    pub volatility_score: f64,
    pub recent_changes: Vec<HistoricalChange>,
}

/// Historical performance changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalChange {
    pub timestamp: u64,
    pub change_type: ChangeType,
    pub magnitude_percent: f64,
    pub description: String,
}

/// Types of performance changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Improvement,
    Regression,
    Spike,
    Gradual,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub slope: f64,
    pub correlation: f64,
    pub prediction_confidence: f64,
    pub next_period_forecast: f64,
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Regression tester
pub struct RegressionTester {
    config: RegressionConfig,
    test_analyzer: TestAnalyzer,
    benchmarking_suite: BenchmarkingSuite,
    historical_results: VecDeque<TestResults>,
    baseline_results: HashMap<String, TestResults>,
    regression_history: Vec<RegressionTestResults>,
}

impl RegressionTester {
    /// Create a new regression tester
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
            test_analyzer: TestAnalyzer::new(),
            benchmarking_suite: BenchmarkingSuite::new(),
            historical_results: VecDeque::new(),
            baseline_results: HashMap::new(),
            regression_history: Vec::new(),
        }
    }

    /// Create regression tester with custom configuration
    pub fn with_config(config: RegressionConfig) -> Self {
        Self {
            config,
            test_analyzer: TestAnalyzer::new(),
            benchmarking_suite: BenchmarkingSuite::new(),
            historical_results: VecDeque::new(),
            baseline_results: HashMap::new(),
            regression_history: Vec::new(),
        }
    }

    /// Run regression test
    pub async fn run_regression_test(&mut self, test_config: &TestConfig) -> Result<RegressionTestResults, Box<dyn std::error::Error>> {
        println!("Running regression test: {}", test_config.test_name);
        
        // Run the actual performance test
        let current_results = self.run_performance_test(test_config).await?;
        
        // Get baseline for comparison
        let baseline_results = self.get_baseline(&test_config.test_name);
        
        // Perform regression analysis
        let regression_analysis = self.analyze_regression(&current_results, baseline_results.as_ref()).await;
        
        // Determine verdict
        let verdict = self.determine_regression_verdict(&regression_analysis);
        
        // Generate recommendations
        let recommendations = self.generate_regression_recommendations(&regression_analysis, &verdict);
        
        // Build historical context
        let historical_context = self.build_historical_context(&test_config.test_name);
        
        let regression_results = RegressionTestResults {
            test_name: test_config.test_name.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            current_results: current_results.clone(),
            baseline_results: baseline_results.clone(),
            regression_analysis,
            verdict: verdict.clone(),
            recommendations,
            historical_context,
        };
        
        // Store results
        self.store_regression_results(&regression_results).await?;
        
        // Update historical data
        self.update_historical_data(current_results);
        
        // Handle notifications and CI integration
        self.handle_post_test_actions(&regression_results).await?;
        
        println!("Regression test completed: {:?}", verdict);
        Ok(regression_results)
    }

    /// Run the actual performance test
    async fn run_performance_test(&mut self, config: &TestConfig) -> Result<TestResults, Box<dyn std::error::Error>> {
        // This would integrate with the actual performance testing framework
        // For now, we'll simulate a test run
        
        let start_time = Instant::now();
        
        // Simulate test execution
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let end_time = Instant::now();
        let duration = end_time - start_time;
        
        // Generate simulated results
        let latency_stats = LatencyStats {
            min_ns: 100,
            max_ns: 5000,
            mean_ns: 1000.0,
            median_ns: 800,
            p95_ns: 2000,
            p99_ns: 3500,
            p99_9_ns: 4500,
            std_dev_ns: 500.0,
            sample_count: 10000,
        };
        
        let throughput_stats = ThroughputStats {
            total_operations: 1_000_000,
            operations_per_second: 1_000_000.0,
            peak_ops_per_second: 1_200_000.0,
            average_ops_per_second: 950_000.0,
            duration,
        };
        
        let resource_usage = ResourceUsage {
            cpu_usage_percent: 75.0,
            memory_usage_bytes: 2_000_000_000,
            network_bytes_sent: 1_000_000,
            network_bytes_received: 1_000_000,
            disk_reads: 1000,
            disk_writes: 500,
            context_switches: 10000,
            page_faults: 100,
        };
        
        Ok(TestResults {
            test_name: config.test_name.clone(),
            start_time,
            end_time,
            duration,
            latency_stats,
            throughput_stats,
            resource_usage,
            errors: Vec::new(),
            metadata: HashMap::new(),
        })
    }

    /// Get baseline results for comparison
    fn get_baseline(&self, test_name: &str) -> Option<TestResults> {
        self.baseline_results.get(test_name).cloned()
    }

    /// Analyze regression between current and baseline results
    async fn analyze_regression(&self, current: &TestResults, baseline: Option<&TestResults>) -> RegressionAnalysis {
        if let Some(baseline) = baseline {
            // Analyze latency regression
            let latency_regression = self.analyze_metric_regression(
                "median_latency",
                current.latency_stats.median_ns as f64,
                baseline.latency_stats.median_ns as f64,
                false, // Lower is better for latency
            );
            
            // Analyze throughput regression
            let throughput_regression = self.analyze_metric_regression(
                "throughput",
                current.throughput_stats.operations_per_second,
                baseline.throughput_stats.operations_per_second,
                true, // Higher is better for throughput
            );
            
            // Analyze resource regression
            let resource_regression = self.analyze_metric_regression(
                "cpu_usage",
                current.resource_usage.cpu_usage_percent,
                baseline.resource_usage.cpu_usage_percent,
                false, // Lower is better for resource usage
            );
            
            // Calculate overall regression score
            let overall_regression_score = self.calculate_overall_regression_score(
                &latency_regression,
                &throughput_regression,
                &resource_regression,
            );
            
            // Calculate statistical significance
            let statistical_significance = self.calculate_statistical_significance(current, baseline);
            
            // Perform trend analysis
            let trend_analysis = self.perform_trend_analysis(&current.test_name);
            
            RegressionAnalysis {
                latency_regression,
                throughput_regression,
                resource_regression,
                overall_regression_score,
                statistical_significance,
                trend_analysis,
            }
        } else {
            // No baseline available
            RegressionAnalysis {
                latency_regression: MetricRegression::no_baseline("median_latency"),
                throughput_regression: MetricRegression::no_baseline("throughput"),
                resource_regression: MetricRegression::no_baseline("cpu_usage"),
                overall_regression_score: 0.0,
                statistical_significance: 0.0,
                trend_analysis: TrendAnalysis::unknown(),
            }
        }
    }

    /// Analyze regression for a specific metric
    fn analyze_metric_regression(&self, metric_name: &str, current_value: f64, baseline_value: f64, higher_is_better: bool) -> MetricRegression {
        let change_percent = if baseline_value != 0.0 {
            ((current_value - baseline_value) / baseline_value) * 100.0
        } else {
            0.0
        };
        
        let is_regression = if higher_is_better {
            change_percent < -self.config.regression_threshold_percent
        } else {
            change_percent > self.config.regression_threshold_percent
        };
        
        let severity = self.determine_regression_severity(change_percent.abs());
        let confidence = self.calculate_metric_confidence(current_value, baseline_value);
        
        MetricRegression {
            metric_name: metric_name.to_string(),
            current_value,
            baseline_value,
            change_percent,
            is_regression,
            severity,
            confidence,
        }
    }

    /// Calculate overall regression score
    fn calculate_overall_regression_score(&self, latency: &MetricRegression, throughput: &MetricRegression, resource: &MetricRegression) -> f64 {
        // Weighted scoring: latency (50%), throughput (30%), resource (20%)
        let latency_score = if latency.is_regression { latency.change_percent.abs() } else { 0.0 };
        let throughput_score = if throughput.is_regression { throughput.change_percent.abs() } else { 0.0 };
        let resource_score = if resource.is_regression { resource.change_percent.abs() } else { 0.0 };
        
        (latency_score * 0.5 + throughput_score * 0.3 + resource_score * 0.2).min(100.0)
    }

    /// Calculate statistical significance
    fn calculate_statistical_significance(&self, current: &TestResults, baseline: &TestResults) -> f64 {
        // Simplified statistical significance calculation
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

    /// Perform trend analysis
    fn perform_trend_analysis(&self, test_name: &str) -> TrendAnalysis {
        let recent_results: Vec<&TestResults> = self.historical_results
            .iter()
            .filter(|r| r.test_name == test_name)
            .collect();
        
        if recent_results.len() < 3 {
            return TrendAnalysis::unknown();
        }
        
        // Calculate trend slope using linear regression
        let values: Vec<f64> = recent_results
            .iter()
            .map(|r| r.latency_stats.median_ns as f64)
            .collect();
        
        let (slope, correlation) = self.calculate_linear_trend(&values);
        
        let direction = if slope.abs() < 0.1 {
            TrendDirection::Stable
        } else if slope < 0.0 {
            TrendDirection::Improving // Lower latency is better
        } else {
            TrendDirection::Degrading
        };
        
        let prediction_confidence = correlation.abs().min(1.0);
        let next_period_forecast = values.last().unwrap_or(&0.0) + slope;
        
        TrendAnalysis {
            direction,
            slope,
            correlation,
            prediction_confidence,
            next_period_forecast,
        }
    }

    /// Calculate linear trend using simple linear regression
    fn calculate_linear_trend(&self, values: &[f64]) -> (f64, f64) {
        if values.len() < 2 {
            return (0.0, 0.0);
        }
        
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(values.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();
        let sum_y2: f64 = values.iter().map(|y| y * y).sum();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        let correlation_numerator = n * sum_xy - sum_x * sum_y;
        let correlation_denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        let correlation = if correlation_denominator != 0.0 {
            correlation_numerator / correlation_denominator
        } else {
            0.0
        };
        
        (slope, correlation)
    }

    /// Determine regression severity
    fn determine_regression_severity(&self, change_percent: f64) -> RegressionSeverity {
        match change_percent {
            x if x < 5.0 => RegressionSeverity::None,
            x if x < 10.0 => RegressionSeverity::Minor,
            x if x < 25.0 => RegressionSeverity::Moderate,
            x if x < 50.0 => RegressionSeverity::Major,
            _ => RegressionSeverity::Critical,
        }
    }

    /// Calculate metric confidence
    fn calculate_metric_confidence(&self, current: f64, baseline: f64) -> f64 {
        // Simple confidence calculation based on magnitude of values
        let magnitude = (current + baseline) / 2.0;
        let difference = (current - baseline).abs();
        
        if magnitude > 0.0 {
            (1.0 - (difference / magnitude).min(1.0)).max(0.0)
        } else {
            0.5
        }
    }

    /// Determine regression verdict
    fn determine_regression_verdict(&self, analysis: &RegressionAnalysis) -> RegressionVerdict {
        if analysis.statistical_significance < self.config.confidence_level {
            return RegressionVerdict::InsufficientData;
        }
        
        let has_critical_regression = matches!(
            analysis.latency_regression.severity,
            RegressionSeverity::Critical
        ) || matches!(
            analysis.throughput_regression.severity,
            RegressionSeverity::Critical
        );
        
        if has_critical_regression {
            return RegressionVerdict::CriticalRegression;
        }
        
        let has_regression = analysis.latency_regression.is_regression
            || analysis.throughput_regression.is_regression
            || analysis.resource_regression.is_regression;
        
        if has_regression {
            if analysis.overall_regression_score > 25.0 {
                RegressionVerdict::Regression
            } else {
                RegressionVerdict::Warning
            }
        } else {
            RegressionVerdict::Pass
        }
    }

    /// Generate regression recommendations
    fn generate_regression_recommendations(&self, analysis: &RegressionAnalysis, verdict: &RegressionVerdict) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match verdict {
            RegressionVerdict::CriticalRegression => {
                recommendations.push("CRITICAL: Immediate investigation required - performance has degraded significantly".to_string());
                recommendations.push("Consider reverting recent changes and conducting root cause analysis".to_string());
            }
            RegressionVerdict::Regression => {
                recommendations.push("Performance regression detected - investigate recent changes".to_string());
                
                if analysis.latency_regression.is_regression {
                    recommendations.push(format!(
                        "Latency increased by {:.1}% - optimize hot path operations",
                        analysis.latency_regression.change_percent
                    ));
                }
                
                if analysis.throughput_regression.is_regression {
                    recommendations.push(format!(
                        "Throughput decreased by {:.1}% - check for bottlenecks",
                        analysis.throughput_regression.change_percent.abs()
                    ));
                }
            }
            RegressionVerdict::Warning => {
                recommendations.push("Minor performance changes detected - monitor closely".to_string());
            }
            RegressionVerdict::Pass => {
                recommendations.push("Performance is within acceptable ranges".to_string());
            }
            RegressionVerdict::NoBaseline => {
                recommendations.push("No baseline available - establish baseline for future comparisons".to_string());
            }
            RegressionVerdict::InsufficientData => {
                recommendations.push("Insufficient data for reliable regression analysis - collect more samples".to_string());
            }
        }
        
        // Add trend-based recommendations
        match analysis.trend_analysis.direction {
            TrendDirection::Degrading => {
                recommendations.push("Degrading performance trend detected - proactive optimization recommended".to_string());
            }
            TrendDirection::Volatile => {
                recommendations.push("Performance is volatile - investigate system stability".to_string());
            }
            _ => {}
        }
        
        recommendations
    }

    /// Build historical context
    fn build_historical_context(&self, test_name: &str) -> HistoricalContext {
        let relevant_results: Vec<&TestResults> = self.historical_results
            .iter()
            .filter(|r| r.test_name == test_name)
            .collect();
        
        let sample_count = relevant_results.len();
        
        let time_span_days = if let (Some(first), Some(last)) = (relevant_results.first(), relevant_results.last()) {
            let duration = last.end_time.duration_since(first.start_time);
            duration.as_secs() as f64 / 86400.0 // Convert to days
        } else {
            0.0
        };
        
        let trend_direction = if sample_count >= 3 {
            let values: Vec<f64> = relevant_results
                .iter()
                .map(|r| r.latency_stats.median_ns as f64)
                .collect();
            let (slope, _) = self.calculate_linear_trend(&values);
            
            if slope.abs() < 0.1 {
                TrendDirection::Stable
            } else if slope < 0.0 {
                TrendDirection::Improving
            } else {
                TrendDirection::Degrading
            }
        } else {
            TrendDirection::Unknown
        };
        
        let volatility_score = self.calculate_volatility_score(&relevant_results);
        let recent_changes = self.identify_recent_changes(&relevant_results);
        
        HistoricalContext {
            sample_count,
            time_span_days,
            trend_direction,
            volatility_score,
            recent_changes,
        }
    }

    /// Calculate volatility score
    fn calculate_volatility_score(&self, results: &[&TestResults]) -> f64 {
        if results.len() < 2 {
            return 0.0;
        }
        
        let values: Vec<f64> = results
            .iter()
            .map(|r| r.latency_stats.median_ns as f64)
            .collect();
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let std_dev = variance.sqrt();
        
        // Coefficient of variation as volatility score
        if mean > 0.0 {
            (std_dev / mean * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Identify recent changes
    fn identify_recent_changes(&self, results: &[&TestResults]) -> Vec<HistoricalChange> {
        let mut changes = Vec::new();
        
        if results.len() < 2 {
            return changes;
        }
        
        for window in results.windows(2) {
            let prev = window[0];
            let curr = window[1];
            
            let latency_change = if prev.latency_stats.median_ns > 0 {
                ((curr.latency_stats.median_ns as f64 - prev.latency_stats.median_ns as f64) 
                 / prev.latency_stats.median_ns as f64) * 100.0
            } else {
                0.0
            };
            
            if latency_change.abs() > 10.0 {
                let change_type = if latency_change > 0.0 {
                    ChangeType::Regression
                } else {
                    ChangeType::Improvement
                };
                
                changes.push(HistoricalChange {
                    timestamp: curr.end_time.duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    change_type,
                    magnitude_percent: latency_change.abs(),
                    description: format!("Latency changed by {:.1}%", latency_change),
                });
            }
        }
        
        // Keep only the most recent changes
        changes.sort_by_key(|c| c.timestamp);
        changes.into_iter().rev().take(5).collect()
    }

    /// Store regression results
    async fn store_regression_results(&mut self, results: &RegressionTestResults) -> Result<(), Box<dyn std::error::Error>> {
        self.regression_history.push(results.clone());
        
        // Optionally save to file
        if let Some(storage_path) = &self.config.ci_integration.artifact_storage_path {
            let file_path = storage_path.join(format!("regression_{}_{}.json", results.test_name, results.timestamp));
            let json = serde_json::to_string_pretty(results)?;
            fs::write(file_path, json).await?;
        }
        
        Ok(())
    }

    /// Update historical data
    fn update_historical_data(&mut self, results: TestResults) {
        self.historical_results.push_back(results);
        
        // Maintain window size
        while self.historical_results.len() > self.config.historical_window_size {
            self.historical_results.pop_front();
        }
    }

    /// Handle post-test actions (notifications, CI integration)
    async fn handle_post_test_actions(&self, results: &RegressionTestResults) -> Result<(), Box<dyn std::error::Error>> {
        // Send notifications
        if self.config.notification_config.enabled {
            self.send_notifications(results).await?;
        }
        
        // Generate CI reports
        if self.config.ci_integration.enabled && self.config.ci_integration.generate_reports {
            self.generate_ci_reports(results).await?;
        }
        
        Ok(())
    }

    /// Send notifications
    async fn send_notifications(&self, results: &RegressionTestResults) -> Result<(), Box<dyn std::error::Error>> {
        let severity = self.determine_notification_severity(&results.verdict);
        
        if self.config.notification_config.severity_levels.contains(&severity) {
            let message = self.format_notification_message(results);
            
            // Send email notifications
            for recipient in &self.config.notification_config.email_recipients {
                println!("Sending email notification to: {}", recipient);
                // In a real implementation, this would send actual emails
            }
            
            // Send Slack notification
            if let Some(webhook) = &self.config.notification_config.slack_webhook {
                println!("Sending Slack notification to: {}", webhook);
                // In a real implementation, this would send to Slack
            }
            
            println!("Notification sent: {}", message);
        }
        
        Ok(())
    }

    /// Determine notification severity
    fn determine_notification_severity(&self, verdict: &RegressionVerdict) -> SeverityLevel {
        match verdict {
            RegressionVerdict::CriticalRegression => SeverityLevel::Emergency,
            RegressionVerdict::Regression => SeverityLevel::Critical,
            RegressionVerdict::Warning => SeverityLevel::Warning,
            _ => SeverityLevel::Info,
        }
    }

    /// Format notification message
    fn format_notification_message(&self, results: &RegressionTestResults) -> String {
        format!(
            "🔍 Regression Test Alert\n\
            Test: {}\n\
            Verdict: {:?}\n\
            Overall Regression Score: {:.1}%\n\
            Statistical Significance: {:.1}%\n\
            \n\
            📊 Key Metrics:\n\
            • Latency: {:.1}% change\n\
            • Throughput: {:.1}% change\n\
            • Resources: {:.1}% change\n\
            \n\
            💡 Recommendations:\n\
            {}",
            results.test_name,
            results.verdict,
            results.regression_analysis.overall_regression_score,
            results.regression_analysis.statistical_significance * 100.0,
            results.regression_analysis.latency_regression.change_percent,
            results.regression_analysis.throughput_regression.change_percent,
            results.regression_analysis.resource_regression.change_percent,
            results.recommendations.join("\n• ")
        )
    }

    /// Generate CI reports
    async fn generate_ci_reports(&self, results: &RegressionTestResults) -> Result<(), Box<dyn std::error::Error>> {
        for format in &self.config.ci_integration.report_format {
            match format {
                ReportFormat::Json => {
                    let json = serde_json::to_string_pretty(results)?;
                    self.save_report(&format!("regression_report_{}.json", results.timestamp), &json).await?;
                }
                ReportFormat::Html => {
                    let html = self.generate_html_report(results);
                    self.save_report(&format!("regression_report_{}.html", results.timestamp), &html).await?;
                }
                ReportFormat::Junit => {
                    let junit = self.generate_junit_report(results);
                    self.save_report(&format!("regression_report_{}.xml", results.timestamp), &junit).await?;
                }
                ReportFormat::Markdown => {
                    let markdown = self.generate_markdown_report(results);
                    self.save_report(&format!("regression_report_{}.md", results.timestamp), &markdown).await?;
                }
            }
        }
        
        Ok(())
    }

    /// Save report to file
    async fn save_report(&self, filename: &str, content: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(storage_path) = &self.config.ci_integration.artifact_storage_path {
            let file_path = storage_path.join(filename);
            fs::write(file_path, content).await?;
            println!("Report saved: {}", filename);
        }
        Ok(())
    }

    /// Generate HTML report
    fn generate_html_report(&self, results: &RegressionTestResults) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Regression Test Report - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .verdict {{ font-size: 24px; font-weight: bold; }}
        .pass {{ color: green; }}
        .warning {{ color: orange; }}
        .regression {{ color: red; }}
        .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .recommendations {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Regression Test Report</h1>
        <h2>{}</h2>
        <div class="verdict {:?}">{:?}</div>
        <p>Timestamp: {}</p>
    </div>
    
    <h3>Regression Analysis</h3>
    <div class="metric">
        <strong>Overall Regression Score:</strong> {:.1}%<br>
        <strong>Statistical Significance:</strong> {:.1}%
    </div>
    
    <h3>Metric Changes</h3>
    <div class="metric">
        <strong>Latency:</strong> {:.1}% change ({:.0}ns → {:.0}ns)
    </div>
    <div class="metric">
        <strong>Throughput:</strong> {:.1}% change ({:.0} → {:.0} ops/sec)
    </div>
    <div class="metric">
        <strong>Resource Usage:</strong> {:.1}% change ({:.1}% → {:.1}% CPU)
    </div>
    
    <h3>Recommendations</h3>
    <div class="recommendations">
        <ul>
            {}
        </ul>
    </div>
    
    <h3>Historical Context</h3>
    <div class="metric">
        <strong>Sample Count:</strong> {}<br>
        <strong>Time Span:</strong> {:.1} days<br>
        <strong>Trend:</strong> {:?}<br>
        <strong>Volatility:</strong> {:.1}%
    </div>
</body>
</html>"#,
            results.test_name,
            results.test_name,
            results.verdict,
            results.verdict,
            results.timestamp,
            results.regression_analysis.overall_regression_score,
            results.regression_analysis.statistical_significance * 100.0,
            results.regression_analysis.latency_regression.change_percent,
            results.regression_analysis.latency_regression.baseline_value,
            results.regression_analysis.latency_regression.current_value,
            results.regression_analysis.throughput_regression.change_percent,
            results.regression_analysis.throughput_regression.baseline_value,
            results.regression_analysis.throughput_regression.current_value,
            results.regression_analysis.resource_regression.change_percent,
            results.regression_analysis.resource_regression.baseline_value,
            results.regression_analysis.resource_regression.current_value,
            results.recommendations.iter().map(|r| format!("<li>{}</li>", r)).collect::<Vec<_>>().join(""),
            results.historical_context.sample_count,
            results.historical_context.time_span_days,
            results.historical_context.trend_direction,
            results.historical_context.volatility_score
        )
    }

    /// Generate JUnit report
    fn generate_junit_report(&self, results: &RegressionTestResults) -> String {
        let status = match results.verdict {
            RegressionVerdict::Pass => "passed",
            _ => "failed",
        };
        
        let failure_message = if !matches!(results.verdict, RegressionVerdict::Pass) {
            format!(
                r#"<failure message="Regression detected">
                Overall regression score: {:.1}%
                Latency change: {:.1}%
                Throughput change: {:.1}%
                </failure>"#,
                results.regression_analysis.overall_regression_score,
                results.regression_analysis.latency_regression.change_percent,
                results.regression_analysis.throughput_regression.change_percent
            )
        } else {
            String::new()
        };
        
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="RegressionTests" tests="1" failures="{}" time="1.0">
    <testcase name="{}" classname="RegressionTest" time="1.0">
        {}
    </testcase>
</testsuite>"#,
            if matches!(results.verdict, RegressionVerdict::Pass) { 0 } else { 1 },
            results.test_name,
            failure_message
        )
    }

    /// Generate Markdown report
    fn generate_markdown_report(&self, results: &RegressionTestResults) -> String {
        format!(
            r#"# Regression Test Report: {}

**Timestamp:** {}  
**Verdict:** {:?}  
**Overall Regression Score:** {:.1}%  
**Statistical Significance:** {:.1}%  

## Metric Analysis

### Latency
- **Change:** {:.1}%
- **Baseline:** {:.0}ns
- **Current:** {:.0}ns
- **Severity:** {:?}

### Throughput
- **Change:** {:.1}%
- **Baseline:** {:.0} ops/sec
- **Current:** {:.0} ops/sec
- **Severity:** {:?}

### Resource Usage
- **Change:** {:.1}%
- **Baseline:** {:.1}% CPU
- **Current:** {:.1}% CPU
- **Severity:** {:?}

## Recommendations

{}

## Historical Context

- **Sample Count:** {}
- **Time Span:** {:.1} days
- **Trend Direction:** {:?}
- **Volatility Score:** {:.1}%

## Recent Changes

{}
"#,
            results.test_name,
            results.timestamp,
            results.verdict,
            results.regression_analysis.overall_regression_score,
            results.regression_analysis.statistical_significance * 100.0,
            results.regression_analysis.latency_regression.change_percent,
            results.regression_analysis.latency_regression.baseline_value,
            results.regression_analysis.latency_regression.current_value,
            results.regression_analysis.latency_regression.severity,
            results.regression_analysis.throughput_regression.change_percent,
            results.regression_analysis.throughput_regression.baseline_value,
            results.regression_analysis.throughput_regression.current_value,
            results.regression_analysis.throughput_regression.severity,
            results.regression_analysis.resource_regression.change_percent,
            results.regression_analysis.resource_regression.baseline_value,
            results.regression_analysis.resource_regression.current_value,
            results.regression_analysis.resource_regression.severity,
            results.recommendations.iter().map(|r| format!("- {}", r)).collect::<Vec<_>>().join("\n"),
            results.historical_context.sample_count,
            results.historical_context.time_span_days,
            results.historical_context.trend_direction,
            results.historical_context.volatility_score,
            results.historical_context.recent_changes.iter()
                .map(|c| format!("- {} ({:.1}%): {}", 
                    match c.change_type {
                        ChangeType::Improvement => "📈 Improvement",
                        ChangeType::Regression => "📉 Regression",
                        ChangeType::Spike => "⚡ Spike",
                        ChangeType::Gradual => "📊 Gradual",
                    },
                    c.magnitude_percent,
                    c.description
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    /// Set baseline for a test
    pub fn set_baseline(&mut self, test_name: &str, results: TestResults) {
        self.baseline_results.insert(test_name.to_string(), results);
        println!("Baseline set for test: {}", test_name);
    }

    /// Update baseline automatically if configured
    pub fn update_baseline_if_needed(&mut self, results: &RegressionTestResults) {
        if self.config.auto_baseline_update && matches!(results.verdict, RegressionVerdict::Pass) {
            self.baseline_results.insert(results.test_name.clone(), results.current_results.clone());
            println!("Baseline automatically updated for test: {}", results.test_name);
        }
    }

    /// Get regression history
    pub fn get_regression_history(&self, test_name: Option<&str>) -> Vec<&RegressionTestResults> {
        match test_name {
            Some(name) => self.regression_history.iter().filter(|r| r.test_name == name).collect(),
            None => self.regression_history.iter().collect(),
        }
    }

    /// Run continuous regression testing
    pub async fn run_continuous_testing(&mut self, test_configs: Vec<TestConfig>, interval: Duration) -> Result<(), Box<dyn std::error::Error>> {
        println!("Starting continuous regression testing with {} tests", test_configs.len());
        
        loop {
            for config in &test_configs {
                match self.run_regression_test(config).await {
                    Ok(results) => {
                        println!("Continuous test completed: {} - {:?}", config.test_name, results.verdict);
                        
                        // Update baseline if needed
                        self.update_baseline_if_needed(&results);
                        
                        // Exit if critical regression and CI integration is configured to fail
                        if matches!(results.verdict, RegressionVerdict::CriticalRegression) 
                            && self.config.ci_integration.fail_build_on_regression {
                            return Err("Critical regression detected - failing build".into());
                        }
                    }
                    Err(e) => {
                        eprintln!("Continuous test failed: {} - {}", config.test_name, e);
                    }
                }
            }
            
            println!("Waiting {:?} before next test cycle", interval);
            tokio::time::sleep(interval).await;
        }
    }

    /// Export regression data
    pub fn export_regression_data(&self, format: &str) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            "json" => Ok(serde_json::to_string_pretty(&self.regression_history)?),
            "csv" => Ok(self.generate_csv_export()),
            _ => Err("Unsupported export format".into()),
        }
    }

    /// Generate CSV export
    fn generate_csv_export(&self) -> String {
        let mut csv = String::from("timestamp,test_name,verdict,overall_score,latency_change,throughput_change,resource_change,significance\n");
        
        for result in &self.regression_history {
            csv.push_str(&format!(
                "{},{},{:?},{:.1},{:.1},{:.1},{:.1},{:.3}\n",
                result.timestamp,
                result.test_name,
                result.verdict,
                result.regression_analysis.overall_regression_score,
                result.regression_analysis.latency_regression.change_percent,
                result.regression_analysis.throughput_regression.change_percent,
                result.regression_analysis.resource_regression.change_percent,
                result.regression_analysis.statistical_significance
            ));
        }
        
        csv
    }
}

impl MetricRegression {
    /// Create a metric regression for cases with no baseline
    fn no_baseline(metric_name: &str) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            current_value: 0.0,
            baseline_value: 0.0,
            change_percent: 0.0,
            is_regression: false,
            severity: RegressionSeverity::None,
            confidence: 0.0,
        }
    }
}

impl TrendAnalysis {
    /// Create unknown trend analysis
    fn unknown() -> Self {
        Self {
            direction: TrendDirection::Unknown,
            slope: 0.0,
            correlation: 0.0,
            prediction_confidence: 0.0,
            next_period_forecast: 0.0,
        }
    }
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            test_suite_name: "default_regression_suite".to_string(),
            baseline_threshold_percent: 5.0,
            regression_threshold_percent: 10.0,
            minimum_samples: 5,
            historical_window_size: 100,
            confidence_level: 0.95,
            auto_baseline_update: false,
            notification_config: NotificationConfig {
                enabled: true,
                email_recipients: Vec::new(),
                slack_webhook: None,
                severity_levels: vec![SeverityLevel::Warning, SeverityLevel::Critical, SeverityLevel::Emergency],
            },
            ci_integration: CiIntegrationConfig {
                enabled: true,
                fail_build_on_regression: true,
                generate_reports: true,
                report_format: vec![ReportFormat::Json, ReportFormat::Html],
                artifact_storage_path: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_regression_tester_creation() {
        let tester = RegressionTester::new();
        assert_eq!(tester.historical_results.len(), 0);
        assert_eq!(tester.baseline_results.len(), 0);
    }

    #[tokio::test]
    async fn test_regression_analysis_no_baseline() {
        let tester = RegressionTester::new();
        let test_results = create_mock_test_results("test");
        
        let analysis = tester.analyze_regression(&test_results, None).await;
        assert!(!analysis.latency_regression.is_regression);
        assert!(!analysis.throughput_regression.is_regression);
        assert_eq!(analysis.overall_regression_score, 0.0);
    }

    #[tokio::test]
    async fn test_regression_detection() {
        let tester = RegressionTester::new();
        
        let baseline = create_mock_test_results("test");
        let mut current = create_mock_test_results("test");
        
        // Simulate regression - increase latency by 20%
        current.latency_stats.median_ns = (baseline.latency_stats.median_ns as f64 * 1.2) as u64;
        
        let analysis = tester.analyze_regression(&current, Some(&baseline)).await;
        assert!(analysis.latency_regression.is_regression);
        assert!(analysis.latency_regression.change_percent > 15.0);
    }

    #[test]
    fn test_trend_calculation() {
        let tester = RegressionTester::new();
        let values = vec![100.0, 110.0, 120.0, 130.0, 140.0];
        let (slope, correlation) = tester.calculate_linear_trend(&values);
        
        assert!(slope > 0.0); // Increasing trend
        assert!(correlation > 0.8); // Strong correlation
    }

    #[test]
    fn test_regression_severity() {
        let tester = RegressionTester::new();
        
        assert!(matches!(tester.determine_regression_severity(3.0), RegressionSeverity::None));
        assert!(matches!(tester.determine_regression_severity(7.0), RegressionSeverity::Minor));
        assert!(matches!(tester.determine_regression_severity(15.0), RegressionSeverity::Moderate));
        assert!(matches!(tester.determine_regression_severity(35.0), RegressionSeverity::Major));
        assert!(matches!(tester.determine_regression_severity(60.0), RegressionSeverity::Critical));
    }

    fn create_mock_test_results(name: &str) -> TestResults {
        TestResults {
            test_name: name.to_string(),
            start_time: Instant::now(),
            end_time: Instant::now(),
            duration: Duration::from_secs(1),
            latency_stats: LatencyStats {
                min_ns: 100,
                max_ns: 2000,
                mean_ns: 800.0,
                median_ns: 750,
                p95_ns: 1500,
                p99_ns: 1800,
                p99_9_ns: 1950,
                std_dev_ns: 200.0,
                sample_count: 10000,
            },
            throughput_stats: ThroughputStats {
                total_operations: 1_000_000,
                operations_per_second: 1_000_000.0,
                peak_ops_per_second: 1_200_000.0,
                average_ops_per_second: 950_000.0,
                duration: Duration::from_secs(1),
            },
            resource_usage: ResourceUsage {
                cpu_usage_percent: 70.0,
                memory_usage_bytes: 1_000_000_000,
                network_bytes_sent: 500_000,
                network_bytes_received: 500_000,
                disk_reads: 100,
                disk_writes: 50,
                context_switches: 1000,
                page_faults: 10,
            },
            errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}