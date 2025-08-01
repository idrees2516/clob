//! Test Result Analysis and Reporting
//! 
//! Provides comprehensive analysis and reporting capabilities for performance test results

use super::{TestResults, LatencyStats, ThroughputStats, ResourceUsage, TestError};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Test analysis configuration
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub latency_thresholds: LatencyThresholds,
    pub throughput_thresholds: ThroughputThresholds,
    pub resource_thresholds: ResourceThresholds,
    pub generate_charts: bool,
    pub export_formats: Vec<ExportFormat>,
}

/// Latency performance thresholds
#[derive(Debug, Clone)]
pub struct LatencyThresholds {
    pub max_median_ns: u64,
    pub max_p95_ns: u64,
    pub max_p99_ns: u64,
    pub max_p99_9_ns: u64,
    pub max_std_dev_ns: f64,
}

/// Throughput performance thresholds
#[derive(Debug, Clone)]
pub struct ThroughputThresholds {
    pub min_ops_per_second: f64,
    pub min_peak_ops_per_second: f64,
    pub min_sustained_ops_per_second: f64,
}

/// Resource usage thresholds
#[derive(Debug, Clone)]
pub struct ResourceThresholds {
    pub max_cpu_usage_percent: f64,
    pub max_memory_usage_bytes: u64,
    pub max_context_switches: u64,
    pub max_page_faults: u64,
}

/// Export format options
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Markdown,
}

/// Comprehensive test analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAnalysis {
    pub test_name: String,
    pub analysis_timestamp: u64,
    pub overall_score: f64, // 0-100 performance score
    pub latency_analysis: LatencyAnalysis,
    pub throughput_analysis: ThroughputAnalysis,
    pub resource_analysis: ResourceAnalysis,
    pub error_analysis: ErrorAnalysis,
    pub recommendations: Vec<String>,
    pub pass_fail_status: TestStatus,
}

/// Latency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyAnalysis {
    pub score: f64,
    pub median_assessment: AssessmentResult,
    pub p95_assessment: AssessmentResult,
    pub p99_assessment: AssessmentResult,
    pub p99_9_assessment: AssessmentResult,
    pub consistency_score: f64, // Based on std deviation
    pub trend_analysis: TrendAnalysis,
}

/// Throughput analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputAnalysis {
    pub score: f64,
    pub ops_per_second_assessment: AssessmentResult,
    pub peak_performance_assessment: AssessmentResult,
    pub sustained_performance_assessment: AssessmentResult,
    pub efficiency_score: f64,
    pub scalability_assessment: String,
}

/// Resource usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalysis {
    pub score: f64,
    pub cpu_efficiency: AssessmentResult,
    pub memory_efficiency: AssessmentResult,
    pub system_stability: AssessmentResult,
    pub resource_utilization_score: f64,
}

/// Error analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub total_errors: u64,
    pub error_rate_percent: f64,
    pub error_categories: HashMap<String, u64>,
    pub critical_errors: Vec<String>,
    pub error_trend: TrendAnalysis,
}

/// Assessment result for individual metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssessmentResult {
    pub value: f64,
    pub threshold: f64,
    pub status: AssessmentStatus,
    pub deviation_percent: f64,
}

/// Assessment status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssessmentStatus {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
}

/// Trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub direction: TrendDirection,
    pub slope: f64,
    pub confidence: f64,
    pub description: String,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Overall test status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestStatus {
    Pass,
    Fail,
    Warning,
}

/// Test result analyzer
pub struct TestAnalyzer {
    config: AnalysisConfig,
    historical_results: Vec<TestResults>,
}

impl TestAnalyzer {
    /// Create a new test analyzer
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
            historical_results: Vec::new(),
        }
    }

    /// Create analyzer with custom configuration
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            config,
            historical_results: Vec::new(),
        }
    }

    /// Analyze test results comprehensively
    pub fn analyze_results(&mut self, results: &TestResults) -> TestAnalysis {
        println!("Analyzing test results for: {}", results.test_name);

        // Store results for historical analysis
        self.historical_results.push(results.clone());

        // Perform individual analyses
        let latency_analysis = self.analyze_latency(&results.latency_stats);
        let throughput_analysis = self.analyze_throughput(&results.throughput_stats);
        let resource_analysis = self.analyze_resource_usage(&results.resource_usage);
        let error_analysis = self.analyze_errors(&results.errors);

        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            &latency_analysis,
            &throughput_analysis,
            &resource_analysis,
            &error_analysis,
        );

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &latency_analysis,
            &throughput_analysis,
            &resource_analysis,
            &error_analysis,
        );

        // Determine pass/fail status
        let pass_fail_status = self.determine_test_status(&latency_analysis, &throughput_analysis, &error_analysis);

        TestAnalysis {
            test_name: results.test_name.clone(),
            analysis_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            overall_score,
            latency_analysis,
            throughput_analysis,
            resource_analysis,
            error_analysis,
            recommendations,
            pass_fail_status,
        }
    }

    /// Analyze latency performance
    fn analyze_latency(&self, stats: &LatencyStats) -> LatencyAnalysis {
        let thresholds = &self.config.latency_thresholds;

        let median_assessment = AssessmentResult {
            value: stats.median_ns as f64,
            threshold: thresholds.max_median_ns as f64,
            status: self.assess_latency_metric(stats.median_ns, thresholds.max_median_ns),
            deviation_percent: self.calculate_deviation(stats.median_ns as f64, thresholds.max_median_ns as f64),
        };

        let p95_assessment = AssessmentResult {
            value: stats.p95_ns as f64,
            threshold: thresholds.max_p95_ns as f64,
            status: self.assess_latency_metric(stats.p95_ns, thresholds.max_p95_ns),
            deviation_percent: self.calculate_deviation(stats.p95_ns as f64, thresholds.max_p95_ns as f64),
        };

        let p99_assessment = AssessmentResult {
            value: stats.p99_ns as f64,
            threshold: thresholds.max_p99_ns as f64,
            status: self.assess_latency_metric(stats.p99_ns, thresholds.max_p99_ns),
            deviation_percent: self.calculate_deviation(stats.p99_ns as f64, thresholds.max_p99_ns as f64),
        };

        let p99_9_assessment = AssessmentResult {
            value: stats.p99_9_ns as f64,
            threshold: thresholds.max_p99_9_ns as f64,
            status: self.assess_latency_metric(stats.p99_9_ns, thresholds.max_p99_9_ns),
            deviation_percent: self.calculate_deviation(stats.p99_9_ns as f64, thresholds.max_p99_9_ns as f64),
        };

        // Calculate consistency score based on standard deviation
        let consistency_score = if stats.mean_ns > 0.0 {
            let cv = stats.std_dev_ns / stats.mean_ns; // Coefficient of variation
            (1.0 - cv.min(1.0)) * 100.0
        } else {
            0.0
        };

        // Analyze trends if historical data is available
        let trend_analysis = self.analyze_latency_trend();

        // Calculate overall latency score
        let score = self.calculate_latency_score(&median_assessment, &p95_assessment, &p99_assessment, consistency_score);

        LatencyAnalysis {
            score,
            median_assessment,
            p95_assessment,
            p99_assessment,
            p99_9_assessment,
            consistency_score,
            trend_analysis,
        }
    }

    /// Analyze throughput performance
    fn analyze_throughput(&self, stats: &ThroughputStats) -> ThroughputAnalysis {
        let thresholds = &self.config.throughput_thresholds;

        let ops_per_second_assessment = AssessmentResult {
            value: stats.operations_per_second,
            threshold: thresholds.min_ops_per_second,
            status: self.assess_throughput_metric(stats.operations_per_second, thresholds.min_ops_per_second),
            deviation_percent: self.calculate_deviation(stats.operations_per_second, thresholds.min_ops_per_second),
        };

        let peak_performance_assessment = AssessmentResult {
            value: stats.peak_ops_per_second,
            threshold: thresholds.min_peak_ops_per_second,
            status: self.assess_throughput_metric(stats.peak_ops_per_second, thresholds.min_peak_ops_per_second),
            deviation_percent: self.calculate_deviation(stats.peak_ops_per_second, thresholds.min_peak_ops_per_second),
        };

        let sustained_performance_assessment = AssessmentResult {
            value: stats.average_ops_per_second,
            threshold: thresholds.min_sustained_ops_per_second,
            status: self.assess_throughput_metric(stats.average_ops_per_second, thresholds.min_sustained_ops_per_second),
            deviation_percent: self.calculate_deviation(stats.average_ops_per_second, thresholds.min_sustained_ops_per_second),
        };

        // Calculate efficiency score (sustained vs peak performance)
        let efficiency_score = if stats.peak_ops_per_second > 0.0 {
            (stats.average_ops_per_second / stats.peak_ops_per_second) * 100.0
        } else {
            0.0
        };

        // Assess scalability
        let scalability_assessment = self.assess_scalability(stats);

        // Calculate overall throughput score
        let score = self.calculate_throughput_score(
            &ops_per_second_assessment,
            &peak_performance_assessment,
            &sustained_performance_assessment,
            efficiency_score,
        );

        ThroughputAnalysis {
            score,
            ops_per_second_assessment,
            peak_performance_assessment,
            sustained_performance_assessment,
            efficiency_score,
            scalability_assessment,
        }
    }

    /// Analyze resource usage
    fn analyze_resource_usage(&self, usage: &ResourceUsage) -> ResourceAnalysis {
        let thresholds = &self.config.resource_thresholds;

        let cpu_efficiency = AssessmentResult {
            value: usage.cpu_usage_percent,
            threshold: thresholds.max_cpu_usage_percent,
            status: self.assess_resource_metric(usage.cpu_usage_percent, thresholds.max_cpu_usage_percent, false),
            deviation_percent: self.calculate_deviation(usage.cpu_usage_percent, thresholds.max_cpu_usage_percent),
        };

        let memory_efficiency = AssessmentResult {
            value: usage.memory_usage_bytes as f64,
            threshold: thresholds.max_memory_usage_bytes as f64,
            status: self.assess_resource_metric(usage.memory_usage_bytes as f64, thresholds.max_memory_usage_bytes as f64, false),
            deviation_percent: self.calculate_deviation(usage.memory_usage_bytes as f64, thresholds.max_memory_usage_bytes as f64),
        };

        let system_stability = AssessmentResult {
            value: usage.context_switches as f64,
            threshold: thresholds.max_context_switches as f64,
            status: self.assess_resource_metric(usage.context_switches as f64, thresholds.max_context_switches as f64, false),
            deviation_percent: self.calculate_deviation(usage.context_switches as f64, thresholds.max_context_switches as f64),
        };

        // Calculate resource utilization score
        let resource_utilization_score = self.calculate_resource_utilization_score(usage);

        // Calculate overall resource score
        let score = self.calculate_resource_score(&cpu_efficiency, &memory_efficiency, &system_stability, resource_utilization_score);

        ResourceAnalysis {
            score,
            cpu_efficiency,
            memory_efficiency,
            system_stability,
            resource_utilization_score,
        }
    }

    /// Analyze errors
    fn analyze_errors(&self, errors: &[TestError]) -> ErrorAnalysis {
        let total_errors = errors.len() as u64;
        
        // Calculate error rate (assuming we have access to total operations)
        let error_rate_percent = if total_errors > 0 {
            // This would need to be calculated based on total operations
            (total_errors as f64 / 1000.0) * 100.0 // Placeholder calculation
        } else {
            0.0
        };

        // Categorize errors
        let mut error_categories = HashMap::new();
        let mut critical_errors = Vec::new();

        for error in errors {
            *error_categories.entry(error.error_type.clone()).or_insert(0) += 1;
            
            // Identify critical errors
            if error.error_type.contains("critical") || error.error_type.contains("fatal") {
                critical_errors.push(error.message.clone());
            }
        }

        // Analyze error trends
        let error_trend = self.analyze_error_trend(errors);

        ErrorAnalysis {
            total_errors,
            error_rate_percent,
            error_categories,
            critical_errors,
            error_trend,
        }
    }

    /// Calculate overall performance score
    fn calculate_overall_score(
        &self,
        latency: &LatencyAnalysis,
        throughput: &ThroughputAnalysis,
        resource: &ResourceAnalysis,
        error: &ErrorAnalysis,
    ) -> f64 {
        // Weighted scoring: latency (40%), throughput (30%), resource (20%), errors (10%)
        let latency_weight = 0.4;
        let throughput_weight = 0.3;
        let resource_weight = 0.2;
        let error_weight = 0.1;

        let error_score = if error.total_errors == 0 {
            100.0
        } else {
            (100.0 - error.error_rate_percent).max(0.0)
        };

        (latency.score * latency_weight
            + throughput.score * throughput_weight
            + resource.score * resource_weight
            + error_score * error_weight)
            .min(100.0)
            .max(0.0)
    }

    /// Generate performance recommendations
    fn generate_recommendations(
        &self,
        latency: &LatencyAnalysis,
        throughput: &ThroughputAnalysis,
        resource: &ResourceAnalysis,
        error: &ErrorAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Latency recommendations
        if latency.score < 70.0 {
            recommendations.push("Consider optimizing hot path operations to reduce latency".to_string());
            if latency.consistency_score < 80.0 {
                recommendations.push("Investigate latency variance - consider lock-free algorithms".to_string());
            }
        }

        // Throughput recommendations
        if throughput.score < 70.0 {
            recommendations.push("Optimize throughput by implementing batch processing".to_string());
            if throughput.efficiency_score < 80.0 {
                recommendations.push("Improve sustained performance - investigate resource bottlenecks".to_string());
            }
        }

        // Resource recommendations
        if resource.score < 70.0 {
            if resource.cpu_efficiency.value > resource.cpu_efficiency.threshold {
                recommendations.push("High CPU usage detected - consider CPU affinity optimization".to_string());
            }
            if resource.memory_efficiency.value > resource.memory_efficiency.threshold {
                recommendations.push("High memory usage - implement memory pooling".to_string());
            }
        }

        // Error recommendations
        if error.total_errors > 0 {
            recommendations.push("Address error conditions to improve system reliability".to_string());
            if !error.critical_errors.is_empty() {
                recommendations.push("Critical errors detected - immediate investigation required".to_string());
            }
        }

        // General recommendations
        if recommendations.is_empty() {
            recommendations.push("Performance is within acceptable ranges - consider stress testing".to_string());
        }

        recommendations
    }

    /// Determine overall test status
    fn determine_test_status(
        &self,
        latency: &LatencyAnalysis,
        throughput: &ThroughputAnalysis,
        error: &ErrorAnalysis,
    ) -> TestStatus {
        // Fail conditions
        if error.total_errors > 0 && error.error_rate_percent > 1.0 {
            return TestStatus::Fail;
        }
        
        if latency.score < 50.0 || throughput.score < 50.0 {
            return TestStatus::Fail;
        }

        // Warning conditions
        if latency.score < 70.0 || throughput.score < 70.0 || error.total_errors > 0 {
            return TestStatus::Warning;
        }

        TestStatus::Pass
    }

    /// Export analysis results in specified format
    pub fn export_analysis(&self, analysis: &TestAnalysis, format: &ExportFormat) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            ExportFormat::Json => {
                Ok(serde_json::to_string_pretty(analysis)?)
            }
            ExportFormat::Csv => {
                Ok(self.generate_csv_report(analysis))
            }
            ExportFormat::Html => {
                Ok(self.generate_html_report(analysis))
            }
            ExportFormat::Markdown => {
                Ok(self.generate_markdown_report(analysis))
            }
        }
    }

    /// Generate comprehensive analysis report
    pub fn generate_comprehensive_report(&self, analysis: &TestAnalysis) -> String {
        format!(
            r#"
# Performance Test Analysis Report

## Test: {}
**Analysis Date:** {}
**Overall Score:** {:.1}/100
**Status:** {:?}

## Latency Analysis
- **Score:** {:.1}/100
- **Median Latency:** {:.0}ns (Target: {:.0}ns) - {:?}
- **95th Percentile:** {:.0}ns (Target: {:.0}ns) - {:?}
- **99th Percentile:** {:.0}ns (Target: {:.0}ns) - {:?}
- **99.9th Percentile:** {:.0}ns (Target: {:.0}ns) - {:?}
- **Consistency Score:** {:.1}%

## Throughput Analysis
- **Score:** {:.1}/100
- **Operations/Second:** {:.0} (Target: {:.0}) - {:?}
- **Peak Performance:** {:.0} ops/sec - {:?}
- **Sustained Performance:** {:.0} ops/sec - {:?}
- **Efficiency:** {:.1}%
- **Scalability:** {}

## Resource Analysis
- **Score:** {:.1}/100
- **CPU Usage:** {:.1}% (Max: {:.1}%) - {:?}
- **Memory Usage:** {:.0} bytes (Max: {:.0}) - {:?}
- **System Stability:** {:.0} context switches (Max: {:.0}) - {:?}
- **Resource Utilization:** {:.1}%

## Error Analysis
- **Total Errors:** {}
- **Error Rate:** {:.2}%
- **Critical Errors:** {}

## Recommendations
{}

## Trend Analysis
- **Latency Trend:** {} ({})
- **Error Trend:** {} ({})
"#,
            analysis.test_name,
            analysis.analysis_timestamp,
            analysis.overall_score,
            analysis.pass_fail_status,
            
            analysis.latency_analysis.score,
            analysis.latency_analysis.median_assessment.value,
            analysis.latency_analysis.median_assessment.threshold,
            analysis.latency_analysis.median_assessment.status,
            analysis.latency_analysis.p95_assessment.value,
            analysis.latency_analysis.p95_assessment.threshold,
            analysis.latency_analysis.p95_assessment.status,
            analysis.latency_analysis.p99_assessment.value,
            analysis.latency_analysis.p99_assessment.threshold,
            analysis.latency_analysis.p99_assessment.status,
            analysis.latency_analysis.p99_9_assessment.value,
            analysis.latency_analysis.p99_9_assessment.threshold,
            analysis.latency_analysis.p99_9_assessment.status,
            analysis.latency_analysis.consistency_score,
            
            analysis.throughput_analysis.score,
            analysis.throughput_analysis.ops_per_second_assessment.value,
            analysis.throughput_analysis.ops_per_second_assessment.threshold,
            analysis.throughput_analysis.ops_per_second_assessment.status,
            analysis.throughput_analysis.peak_performance_assessment.value,
            analysis.throughput_analysis.peak_performance_assessment.status,
            analysis.throughput_analysis.sustained_performance_assessment.value,
            analysis.throughput_analysis.sustained_performance_assessment.status,
            analysis.throughput_analysis.efficiency_score,
            analysis.throughput_analysis.scalability_assessment,
            
            analysis.resource_analysis.score,
            analysis.resource_analysis.cpu_efficiency.value,
            analysis.resource_analysis.cpu_efficiency.threshold,
            analysis.resource_analysis.cpu_efficiency.status,
            analysis.resource_analysis.memory_efficiency.value,
            analysis.resource_analysis.memory_efficiency.threshold,
            analysis.resource_analysis.memory_efficiency.status,
            analysis.resource_analysis.system_stability.value,
            analysis.resource_analysis.system_stability.threshold,
            analysis.resource_analysis.system_stability.status,
            analysis.resource_analysis.resource_utilization_score,
            
            analysis.error_analysis.total_errors,
            analysis.error_analysis.error_rate_percent,
            analysis.error_analysis.critical_errors.len(),
            
            analysis.recommendations.join("\n- "),
            
            analysis.latency_analysis.trend_analysis.direction,
            analysis.latency_analysis.trend_analysis.description,
            analysis.error_analysis.error_trend.direction,
            analysis.error_analysis.error_trend.description,
        )
    }

    // Helper methods for assessments and calculations
    fn assess_latency_metric(&self, value: u64, threshold: u64) -> AssessmentStatus {
        let ratio = value as f64 / threshold as f64;
        match ratio {
            r if r <= 0.5 => AssessmentStatus::Excellent,
            r if r <= 0.7 => AssessmentStatus::Good,
            r if r <= 1.0 => AssessmentStatus::Acceptable,
            r if r <= 1.5 => AssessmentStatus::Poor,
            _ => AssessmentStatus::Critical,
        }
    }

    fn assess_throughput_metric(&self, value: f64, threshold: f64) -> AssessmentStatus {
        let ratio = value / threshold;
        match ratio {
            r if r >= 1.5 => AssessmentStatus::Excellent,
            r if r >= 1.2 => AssessmentStatus::Good,
            r if r >= 1.0 => AssessmentStatus::Acceptable,
            r if r >= 0.8 => AssessmentStatus::Poor,
            _ => AssessmentStatus::Critical,
        }
    }

    fn assess_resource_metric(&self, value: f64, threshold: f64, higher_is_better: bool) -> AssessmentStatus {
        let ratio = if higher_is_better { value / threshold } else { threshold / value };
        match ratio {
            r if r >= 1.5 => AssessmentStatus::Excellent,
            r if r >= 1.2 => AssessmentStatus::Good,
            r if r >= 1.0 => AssessmentStatus::Acceptable,
            r if r >= 0.8 => AssessmentStatus::Poor,
            _ => AssessmentStatus::Critical,
        }
    }

    fn calculate_deviation(&self, value: f64, threshold: f64) -> f64 {
        if threshold != 0.0 {
            ((value - threshold) / threshold) * 100.0
        } else {
            0.0
        }
    }

    fn calculate_latency_score(&self, median: &AssessmentResult, p95: &AssessmentResult, p99: &AssessmentResult, consistency: f64) -> f64 {
        let median_score = self.assessment_to_score(&median.status);
        let p95_score = self.assessment_to_score(&p95.status);
        let p99_score = self.assessment_to_score(&p99.status);
        
        // Weighted average: median (40%), p95 (30%), p99 (20%), consistency (10%)
        (median_score * 0.4 + p95_score * 0.3 + p99_score * 0.2 + consistency * 0.1).min(100.0)
    }

    fn calculate_throughput_score(&self, ops: &AssessmentResult, peak: &AssessmentResult, sustained: &AssessmentResult, efficiency: f64) -> f64 {
        let ops_score = self.assessment_to_score(&ops.status);
        let peak_score = self.assessment_to_score(&peak.status);
        let sustained_score = self.assessment_to_score(&sustained.status);
        
        // Weighted average: ops (40%), sustained (30%), peak (20%), efficiency (10%)
        (ops_score * 0.4 + sustained_score * 0.3 + peak_score * 0.2 + efficiency * 0.1).min(100.0)
    }

    fn calculate_resource_score(&self, cpu: &AssessmentResult, memory: &AssessmentResult, stability: &AssessmentResult, utilization: f64) -> f64 {
        let cpu_score = self.assessment_to_score(&cpu.status);
        let memory_score = self.assessment_to_score(&memory.status);
        let stability_score = self.assessment_to_score(&stability.status);
        
        // Weighted average: CPU (40%), memory (30%), stability (20%), utilization (10%)
        (cpu_score * 0.4 + memory_score * 0.3 + stability_score * 0.2 + utilization * 0.1).min(100.0)
    }

    fn assessment_to_score(&self, status: &AssessmentStatus) -> f64 {
        match status {
            AssessmentStatus::Excellent => 100.0,
            AssessmentStatus::Good => 80.0,
            AssessmentStatus::Acceptable => 60.0,
            AssessmentStatus::Poor => 40.0,
            AssessmentStatus::Critical => 20.0,
        }
    }

    fn assess_scalability(&self, stats: &ThroughputStats) -> String {
        if stats.peak_ops_per_second > stats.average_ops_per_second * 1.5 {
            "Good scalability potential".to_string()
        } else if stats.peak_ops_per_second > stats.average_ops_per_second * 1.2 {
            "Moderate scalability".to_string()
        } else {
            "Limited scalability".to_string()
        }
    }

    fn calculate_resource_utilization_score(&self, usage: &ResourceUsage) -> f64 {
        // Simple utilization score based on balanced resource usage
        let cpu_normalized = (usage.cpu_usage_percent / 100.0).min(1.0);
        let memory_normalized = (usage.memory_usage_bytes as f64 / (8_000_000_000.0)).min(1.0); // Assume 8GB max
        
        // Optimal utilization is around 70-80%
        let optimal_range = 0.75;
        let cpu_efficiency = 1.0 - (cpu_normalized - optimal_range).abs();
        let memory_efficiency = 1.0 - (memory_normalized - optimal_range).abs();
        
        ((cpu_efficiency + memory_efficiency) / 2.0 * 100.0).max(0.0)
    }

    fn analyze_latency_trend(&self) -> TrendAnalysis {
        // Placeholder trend analysis - would analyze historical data
        TrendAnalysis {
            direction: TrendDirection::Stable,
            slope: 0.0,
            confidence: 0.8,
            description: "Insufficient historical data for trend analysis".to_string(),
        }
    }

    fn analyze_error_trend(&self, _errors: &[TestError]) -> TrendAnalysis {
        // Placeholder error trend analysis
        TrendAnalysis {
            direction: TrendDirection::Stable,
            slope: 0.0,
            confidence: 0.8,
            description: "No significant error trend detected".to_string(),
        }
    }

    fn generate_csv_report(&self, analysis: &TestAnalysis) -> String {
        format!(
            "Metric,Value,Threshold,Status,Score\n\
            Overall Score,{:.1},,{:?},{:.1}\n\
            Median Latency,{:.0},{:.0},{:?},{:.1}\n\
            P95 Latency,{:.0},{:.0},{:?},{:.1}\n\
            P99 Latency,{:.0},{:.0},{:?},{:.1}\n\
            Operations/Second,{:.0},{:.0},{:?},{:.1}\n\
            CPU Usage,{:.1},{:.1},{:?},{:.1}\n\
            Memory Usage,{:.0},{:.0},{:?},{:.1}\n\
            Total Errors,{},,,{:.1}",
            analysis.overall_score, analysis.pass_fail_status, analysis.overall_score,
            analysis.latency_analysis.median_assessment.value,
            analysis.latency_analysis.median_assessment.threshold,
            analysis.latency_analysis.median_assessment.status,
            analysis.latency_analysis.score,
            analysis.latency_analysis.p95_assessment.value,
            analysis.latency_analysis.p95_assessment.threshold,
            analysis.latency_analysis.p95_assessment.status,
            analysis.latency_analysis.score,
            analysis.latency_analysis.p99_assessment.value,
            analysis.latency_analysis.p99_assessment.threshold,
            analysis.latency_analysis.p99_assessment.status,
            analysis.latency_analysis.score,
            analysis.throughput_analysis.ops_per_second_assessment.value,
            analysis.throughput_analysis.ops_per_second_assessment.threshold,
            analysis.throughput_analysis.ops_per_second_assessment.status,
            analysis.throughput_analysis.score,
            analysis.resource_analysis.cpu_efficiency.value,
            analysis.resource_analysis.cpu_efficiency.threshold,
            analysis.resource_analysis.cpu_efficiency.status,
            analysis.resource_analysis.score,
            analysis.resource_analysis.memory_efficiency.value,
            analysis.resource_analysis.memory_efficiency.threshold,
            analysis.resource_analysis.memory_efficiency.status,
            analysis.resource_analysis.score,
            analysis.error_analysis.total_errors,
            100.0 - analysis.error_analysis.error_rate_percent,
        )
    }

    fn generate_html_report(&self, analysis: &TestAnalysis) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Analysis - {}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007acc; }}
        .pass {{ border-left-color: #28a745; }}
        .warning {{ border-left-color: #ffc107; }}
        .fail {{ border-left-color: #dc3545; }}
        .score {{ font-size: 24px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Test Analysis</h1>
        <h2>{}</h2>
        <div class="score">Overall Score: {:.1}/100</div>
        <p>Status: {:?}</p>
    </div>
    
    <h3>Latency Analysis</h3>
    <div class="metric">
        <strong>Score:</strong> {:.1}/100<br>
        <strong>Median:</strong> {:.0}ns (Target: {:.0}ns)<br>
        <strong>95th Percentile:</strong> {:.0}ns<br>
        <strong>99th Percentile:</strong> {:.0}ns<br>
        <strong>Consistency:</strong> {:.1}%
    </div>
    
    <h3>Throughput Analysis</h3>
    <div class="metric">
        <strong>Score:</strong> {:.1}/100<br>
        <strong>Operations/Second:</strong> {:.0}<br>
        <strong>Peak Performance:</strong> {:.0} ops/sec<br>
        <strong>Efficiency:</strong> {:.1}%
    </div>
    
    <h3>Recommendations</h3>
    <ul>
        {}
    </ul>
</body>
</html>"#,
            analysis.test_name,
            analysis.test_name,
            analysis.overall_score,
            analysis.pass_fail_status,
            analysis.latency_analysis.score,
            analysis.latency_analysis.median_assessment.value,
            analysis.latency_analysis.median_assessment.threshold,
            analysis.latency_analysis.p95_assessment.value,
            analysis.latency_analysis.p99_assessment.value,
            analysis.latency_analysis.consistency_score,
            analysis.throughput_analysis.score,
            analysis.throughput_analysis.ops_per_second_assessment.value,
            analysis.throughput_analysis.peak_performance_assessment.value,
            analysis.throughput_analysis.efficiency_score,
            analysis.recommendations.iter()
                .map(|r| format!("<li>{}</li>", r))
                .collect::<Vec<_>>()
                .join("\n        ")
        )
    }

    fn generate_markdown_report(&self, analysis: &TestAnalysis) -> String {
        self.generate_comprehensive_report(analysis)
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            latency_thresholds: LatencyThresholds {
                max_median_ns: 500,      // 500ns median target
                max_p95_ns: 1_000,       // 1μs p95 target
                max_p99_ns: 2_000,       // 2μs p99 target
                max_p99_9_ns: 5_000,     // 5μs p99.9 target
                max_std_dev_ns: 200.0,   // 200ns std dev target
            },
            throughput_thresholds: ThroughputThresholds {
                min_ops_per_second: 1_000_000.0,      // 1M ops/sec target
                min_peak_ops_per_second: 1_500_000.0,  // 1.5M peak target
                min_sustained_ops_per_second: 800_000.0, // 800K sustained target
            },
            resource_thresholds: ResourceThresholds {
                max_cpu_usage_percent: 80.0,
                max_memory_usage_bytes: 4_000_000_000, // 4GB
                max_context_switches: 10_000,
                max_page_faults: 1_000,
            },
            generate_charts: true,
            export_formats: vec![ExportFormat::Json, ExportFormat::Markdown],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = TestAnalyzer::new();
        assert_eq!(analyzer.historical_results.len(), 0);
    }

    #[test]
    fn test_assessment_scoring() {
        let analyzer = TestAnalyzer::new();
        assert_eq!(analyzer.assessment_to_score(&AssessmentStatus::Excellent), 100.0);
        assert_eq!(analyzer.assessment_to_score(&AssessmentStatus::Good), 80.0);
        assert_eq!(analyzer.assessment_to_score(&AssessmentStatus::Critical), 20.0);
    }

    #[test]
    fn test_deviation_calculation() {
        let analyzer = TestAnalyzer::new();
        assert_eq!(analyzer.calculate_deviation(110.0, 100.0), 10.0);
        assert_eq!(analyzer.calculate_deviation(90.0, 100.0), -10.0);
    }
}