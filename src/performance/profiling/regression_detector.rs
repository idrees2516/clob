use super::continuous_profiler::ProfileSnapshot;
use super::timing::now_nanos;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Performance regression detector for identifying performance degradations
pub struct RegressionDetector {
    /// Detection configuration
    config: RegressionConfig,
    
    /// Baseline performance data
    baselines: std::sync::RwLock<HashMap<String, PerformanceBaseline>>,
    
    /// Detection history
    detection_history: std::sync::Mutex<Vec<RegressionDetection>>,
}

/// Configuration for regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Minimum percentage change to consider a regression
    pub min_regression_percent: f64,
    
    /// Number of samples required for baseline
    pub baseline_sample_count: usize,
    
    /// Number of recent samples to compare against baseline
    pub comparison_sample_count: usize,
    
    /// Statistical significance threshold (p-value)
    pub significance_threshold: f64,
    
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    
    /// Minimum confidence level for regression detection
    pub min_confidence: f64,
    
    /// Functions to monitor for regressions
    pub monitored_functions: Vec<String>,
    
    /// Metrics to analyze
    pub monitored_metrics: Vec<String>,
}

/// Performance baseline for a function or metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Function or metric name
    pub name: String,
    
    /// Baseline mean value
    pub mean: f64,
    
    /// Baseline standard deviation
    pub std_dev: f64,
    
    /// Baseline median
    pub median: f64,
    
    /// Baseline 95th percentile
    pub p95: f64,
    
    /// Baseline 99th percentile
    pub p99: f64,
    
    /// Sample count used for baseline
    pub sample_count: usize,
    
    /// Baseline creation timestamp
    pub created_at: u64,
    
    /// Last update timestamp
    pub updated_at: u64,
    
    /// Historical values
    pub historical_values: Vec<f64>,
}

/// Detected performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    /// Regression type
    pub regression_type: RegressionType,
    
    /// Function or metric name
    pub name: String,
    
    /// Baseline value
    pub baseline_value: f64,
    
    /// Current value
    pub current_value: f64,
    
    /// Percentage change
    pub percentage_change: f64,
    
    /// Statistical significance (p-value)
    pub p_value: f64,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Detection timestamp
    pub detected_at: u64,
    
    /// Severity level
    pub severity: RegressionSeverity,
    
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of performance regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionType {
    /// Latency increase
    LatencyRegression,
    
    /// Throughput decrease
    ThroughputRegression,
    
    /// Memory usage increase
    MemoryRegression,
    
    /// CPU usage increase
    CpuRegression,
    
    /// Error rate increase
    ErrorRateRegression,
    
    /// General performance degradation
    GeneralRegression,
}

/// Regression severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Regression detection result
#[derive(Debug, Clone)]
pub struct RegressionDetection {
    /// Detection timestamp
    pub timestamp: u64,
    
    /// Number of regressions detected
    pub regression_count: usize,
    
    /// Detected regressions
    pub regressions: Vec<PerformanceRegression>,
    
    /// Detection duration (nanoseconds)
    pub detection_duration_ns: u64,
}

impl RegressionDetector {
    /// Create a new regression detector
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
            baselines: std::sync::RwLock::new(HashMap::new()),
            detection_history: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Create detector with custom configuration
    pub fn with_config(config: RegressionConfig) -> Self {
        Self {
            config,
            baselines: std::sync::RwLock::new(HashMap::new()),
            detection_history: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Update baseline from profile snapshots
    pub fn update_baselines(&self, profiles: &[&ProfileSnapshot]) -> Result<(), RegressionError> {
        if profiles.len() < self.config.baseline_sample_count {
            return Err(RegressionError::InsufficientSamples);
        }

        let mut baselines = self.baselines.write().unwrap();
        
        // Extract performance metrics from profiles
        let metrics = self.extract_metrics_from_profiles(profiles);
        
        for (metric_name, values) in metrics {
            if values.len() >= self.config.baseline_sample_count {
                let baseline = self.calculate_baseline(&metric_name, &values)?;
                baselines.insert(metric_name, baseline);
            }
        }

        Ok(())
    }

    /// Detect regressions in recent profiles
    pub fn detect_regressions(&self, profiles: &[&ProfileSnapshot]) -> Vec<PerformanceRegression> {
        let detection_start = now_nanos();
        let mut regressions = Vec::new();

        if profiles.len() < self.config.comparison_sample_count {
            return regressions;
        }

        let baselines = self.baselines.read().unwrap();
        let recent_metrics = self.extract_metrics_from_profiles(profiles);

        for (metric_name, recent_values) in recent_metrics {
            if let Some(baseline) = baselines.get(&metric_name) {
                if let Some(regression) = self.detect_metric_regression(baseline, &recent_values) {
                    regressions.push(regression);
                }
            }
        }

        // Record detection
        let detection = RegressionDetection {
            timestamp: detection_start,
            regression_count: regressions.len(),
            regressions: regressions.clone(),
            detection_duration_ns: now_nanos() - detection_start,
        };

        let mut history = self.detection_history.lock().unwrap();
        history.push(detection);

        // Maintain history size
        if history.len() > 1000 {
            history.remove(0);
        }

        regressions
    }

    /// Get current baselines
    pub fn get_baselines(&self) -> HashMap<String, PerformanceBaseline> {
        let baselines = self.baselines.read().unwrap();
        baselines.clone()
    }

    /// Get detection history
    pub fn get_detection_history(&self, limit: Option<usize>) -> Vec<RegressionDetection> {
        let history = self.detection_history.lock().unwrap();
        let limit = limit.unwrap_or(history.len());
        
        history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Clear all baselines
    pub fn clear_baselines(&self) {
        let mut baselines = self.baselines.write().unwrap();
        baselines.clear();
    }

    /// Get regression statistics
    pub fn get_stats(&self) -> RegressionStats {
        let baselines = self.baselines.read().unwrap();
        let history = self.detection_history.lock().unwrap();
        
        let total_detections = history.len();
        let total_regressions = history.iter().map(|d| d.regression_count).sum();
        
        let severity_distribution = history.iter()
            .flat_map(|d| &d.regressions)
            .fold(HashMap::new(), |mut acc, regression| {
                *acc.entry(regression.severity).or_insert(0) += 1;
                acc
            });

        let avg_detection_time = if !history.is_empty() {
            history.iter().map(|d| d.detection_duration_ns).sum::<u64>() / history.len() as u64
        } else {
            0
        };

        RegressionStats {
            baseline_count: baselines.len(),
            total_detections,
            total_regressions,
            severity_distribution,
            avg_detection_time_ns: avg_detection_time,
        }
    }

    /// Extract performance metrics from profiles
    fn extract_metrics_from_profiles(&self, profiles: &[&ProfileSnapshot]) -> HashMap<String, Vec<f64>> {
        let mut metrics = HashMap::new();

        for profile in profiles {
            // Extract latency metrics from samples
            for sample in &profile.samples {
                let latency_key = format!("{}_latency", sample.function_name);
                metrics.entry(latency_key).or_insert_with(Vec::new).push(sample.duration_ns as f64);
                
                // Extract CPU usage
                let cpu_key = format!("{}_cpu", sample.function_name);
                metrics.entry(cpu_key).or_insert_with(Vec::new).push(sample.cpu_usage);
                
                // Extract memory usage
                let memory_key = format!("{}_memory", sample.function_name);
                metrics.entry(memory_key).or_insert_with(Vec::new).push(sample.memory_usage as f64);
            }

            // Extract profile-level metrics
            metrics.entry("profile_duration".to_string()).or_insert_with(Vec::new).push(profile.duration_ns as f64);
            metrics.entry("sample_count".to_string()).or_insert_with(Vec::new).push(profile.sample_count as f64);
        }

        // Filter metrics based on configuration
        if !self.config.monitored_metrics.is_empty() {
            metrics.retain(|key, _| {
                self.config.monitored_metrics.iter().any(|pattern| key.contains(pattern))
            });
        }

        metrics
    }

    /// Calculate baseline from historical values
    fn calculate_baseline(&self, name: &str, values: &[f64]) -> Result<PerformanceBaseline, RegressionError> {
        if values.is_empty() {
            return Err(RegressionError::InsufficientSamples);
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let p95_index = ((sorted_values.len() as f64) * 0.95) as usize;
        let p95 = sorted_values.get(p95_index).copied().unwrap_or(sorted_values[sorted_values.len() - 1]);

        let p99_index = ((sorted_values.len() as f64) * 0.99) as usize;
        let p99 = sorted_values.get(p99_index).copied().unwrap_or(sorted_values[sorted_values.len() - 1]);

        Ok(PerformanceBaseline {
            name: name.to_string(),
            mean,
            std_dev,
            median,
            p95,
            p99,
            sample_count: values.len(),
            created_at: now_nanos(),
            updated_at: now_nanos(),
            historical_values: values.to_vec(),
        })
    }

    /// Detect regression for a specific metric
    fn detect_metric_regression(&self, baseline: &PerformanceBaseline, recent_values: &[f64]) -> Option<PerformanceRegression> {
        if recent_values.is_empty() {
            return None;
        }

        let recent_mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let percentage_change = ((recent_mean - baseline.mean) / baseline.mean) * 100.0;

        // Check if change exceeds threshold
        if percentage_change.abs() < self.config.min_regression_percent {
            return None;
        }

        // Perform statistical test (simplified t-test)
        let p_value = self.calculate_t_test_p_value(baseline, recent_values);
        
        if p_value > self.config.significance_threshold {
            return None;
        }

        // Determine regression type
        let regression_type = self.determine_regression_type(&baseline.name, percentage_change);
        
        // Calculate confidence
        let confidence = 1.0 - p_value;
        
        if confidence < self.config.min_confidence {
            return None;
        }

        // Determine severity
        let severity = self.determine_severity(percentage_change.abs());

        let mut context = HashMap::new();
        context.insert("baseline_samples".to_string(), baseline.sample_count.to_string());
        context.insert("recent_samples".to_string(), recent_values.len().to_string());
        context.insert("baseline_std_dev".to_string(), baseline.std_dev.to_string());

        Some(PerformanceRegression {
            regression_type,
            name: baseline.name.clone(),
            baseline_value: baseline.mean,
            current_value: recent_mean,
            percentage_change,
            p_value,
            confidence,
            detected_at: now_nanos(),
            severity,
            context,
        })
    }

    /// Calculate t-test p-value (simplified)
    fn calculate_t_test_p_value(&self, baseline: &PerformanceBaseline, recent_values: &[f64]) -> f64 {
        if recent_values.is_empty() || baseline.std_dev == 0.0 {
            return 1.0;
        }

        let recent_mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
        let recent_variance = recent_values.iter()
            .map(|v| (v - recent_mean).powi(2))
            .sum::<f64>() / recent_values.len() as f64;
        let recent_std_dev = recent_variance.sqrt();

        // Pooled standard error
        let n1 = baseline.sample_count as f64;
        let n2 = recent_values.len() as f64;
        let pooled_std_error = ((baseline.std_dev.powi(2) / n1) + (recent_std_dev.powi(2) / n2)).sqrt();

        if pooled_std_error == 0.0 {
            return 1.0;
        }

        // T-statistic
        let t_stat = ((recent_mean - baseline.mean) / pooled_std_error).abs();

        // Simplified p-value calculation (approximation)
        // In a real implementation, this would use proper statistical tables
        if t_stat > 2.576 {
            0.01 // p < 0.01
        } else if t_stat > 1.96 {
            0.05 // p < 0.05
        } else if t_stat > 1.645 {
            0.10 // p < 0.10
        } else {
            0.20 // p >= 0.20
        }
    }

    /// Determine regression type based on metric name and change
    fn determine_regression_type(&self, metric_name: &str, percentage_change: f64) -> RegressionType {
        if metric_name.contains("latency") && percentage_change > 0.0 {
            RegressionType::LatencyRegression
        } else if metric_name.contains("throughput") && percentage_change < 0.0 {
            RegressionType::ThroughputRegression
        } else if metric_name.contains("memory") && percentage_change > 0.0 {
            RegressionType::MemoryRegression
        } else if metric_name.contains("cpu") && percentage_change > 0.0 {
            RegressionType::CpuRegression
        } else if metric_name.contains("error") && percentage_change > 0.0 {
            RegressionType::ErrorRateRegression
        } else {
            RegressionType::GeneralRegression
        }
    }

    /// Determine severity based on percentage change
    fn determine_severity(&self, percentage_change: f64) -> RegressionSeverity {
        if percentage_change >= 50.0 {
            RegressionSeverity::Critical
        } else if percentage_change >= 25.0 {
            RegressionSeverity::High
        } else if percentage_change >= 10.0 {
            RegressionSeverity::Medium
        } else {
            RegressionSeverity::Low
        }
    }
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            min_regression_percent: 10.0,
            baseline_sample_count: 100,
            comparison_sample_count: 20,
            significance_threshold: 0.05,
            enable_trend_analysis: true,
            min_confidence: 0.90,
            monitored_functions: Vec::new(),
            monitored_metrics: vec![
                "latency".to_string(),
                "throughput".to_string(),
                "cpu".to_string(),
                "memory".to_string(),
            ],
        }
    }
}

/// Regression detection statistics
#[derive(Debug, Clone)]
pub struct RegressionStats {
    pub baseline_count: usize,
    pub total_detections: usize,
    pub total_regressions: usize,
    pub severity_distribution: HashMap<RegressionSeverity, usize>,
    pub avg_detection_time_ns: u64,
}

/// Regression detection errors
#[derive(Debug, Clone)]
pub enum RegressionError {
    InsufficientSamples,
    InvalidConfiguration,
    StatisticalError,
    BaselineNotFound,
}

impl std::fmt::Display for RegressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegressionError::InsufficientSamples => write!(f, "Insufficient samples for regression detection"),
            RegressionError::InvalidConfiguration => write!(f, "Invalid regression detection configuration"),
            RegressionError::StatisticalError => write!(f, "Statistical calculation error"),
            RegressionError::BaselineNotFound => write!(f, "Baseline not found for metric"),
        }
    }
}

impl std::error::Error for RegressionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sampling_profiler::ProfileSample;

    fn create_test_profile(sample_count: u32, avg_latency: f64) -> ProfileSnapshot {
        let mut samples = Vec::new();
        
        for i in 0..sample_count {
            samples.push(ProfileSample {
                timestamp: now_nanos(),
                duration_ns: (avg_latency + (i as f64 * 100.0)) as u64,
                function_name: "test_function".to_string(),
                stack_trace: vec!["test_function".to_string()],
                cpu_usage: 50.0,
                memory_usage: 1024,
                thread_id: 1,
            });
        }

        ProfileSnapshot {
            id: format!("profile_{}", now_nanos()),
            timestamp: now_nanos(),
            duration_ns: 1000000,
            sample_count,
            flame_graph: None,
            analysis: None,
            regressions: Vec::new(),
            metadata: HashMap::new(),
            samples,
        }
    }

    #[test]
    fn test_regression_detector_creation() {
        let detector = RegressionDetector::new();
        let stats = detector.get_stats();
        
        assert_eq!(stats.baseline_count, 0);
        assert_eq!(stats.total_detections, 0);
    }

    #[test]
    fn test_baseline_calculation() {
        let detector = RegressionDetector::new();
        let values = vec![100.0, 110.0, 90.0, 105.0, 95.0, 115.0, 85.0, 120.0];
        
        let baseline = detector.calculate_baseline("test_metric", &values).unwrap();
        
        assert_eq!(baseline.name, "test_metric");
        assert!(baseline.mean > 0.0);
        assert!(baseline.std_dev > 0.0);
        assert!(baseline.median > 0.0);
        assert!(baseline.p95 > baseline.median);
        assert!(baseline.p99 >= baseline.p95);
        assert_eq!(baseline.sample_count, 8);
    }

    #[test]
    fn test_baseline_update() {
        let detector = RegressionDetector::new();
        
        // Create baseline profiles
        let baseline_profiles: Vec<ProfileSnapshot> = (0..10)
            .map(|_| create_test_profile(10, 1000.0))
            .collect();
        let baseline_refs: Vec<&ProfileSnapshot> = baseline_profiles.iter().collect();
        
        let result = detector.update_baselines(&baseline_refs);
        assert!(result.is_ok());
        
        let baselines = detector.get_baselines();
        assert!(!baselines.is_empty());
        assert!(baselines.contains_key("test_function_latency"));
    }

    #[test]
    fn test_regression_detection() {
        let detector = RegressionDetector::new();
        
        // Create baseline profiles (low latency)
        let baseline_profiles: Vec<ProfileSnapshot> = (0..10)
            .map(|_| create_test_profile(10, 1000.0))
            .collect();
        let baseline_refs: Vec<&ProfileSnapshot> = baseline_profiles.iter().collect();
        
        detector.update_baselines(&baseline_refs).unwrap();
        
        // Create recent profiles with regression (high latency)
        let recent_profiles: Vec<ProfileSnapshot> = (0..5)
            .map(|_| create_test_profile(10, 2000.0)) // 100% increase
            .collect();
        let recent_refs: Vec<&ProfileSnapshot> = recent_profiles.iter().collect();
        
        let regressions = detector.detect_regressions(&recent_refs);
        
        assert!(!regressions.is_empty());
        
        let latency_regression = regressions.iter()
            .find(|r| r.name.contains("latency"))
            .expect("Should detect latency regression");
        
        assert_eq!(latency_regression.regression_type, RegressionType::LatencyRegression);
        assert!(latency_regression.percentage_change > 50.0); // Should be around 100%
        assert_eq!(latency_regression.severity, RegressionSeverity::Critical);
    }

    #[test]
    fn test_regression_types() {
        let detector = RegressionDetector::new();
        
        assert_eq!(
            detector.determine_regression_type("function_latency", 20.0),
            RegressionType::LatencyRegression
        );
        
        assert_eq!(
            detector.determine_regression_type("throughput_ops", -15.0),
            RegressionType::ThroughputRegression
        );
        
        assert_eq!(
            detector.determine_regression_type("memory_usage", 30.0),
            RegressionType::MemoryRegression
        );
        
        assert_eq!(
            detector.determine_regression_type("cpu_utilization", 25.0),
            RegressionType::CpuRegression
        );
    }

    #[test]
    fn test_severity_determination() {
        let detector = RegressionDetector::new();
        
        assert_eq!(detector.determine_severity(5.0), RegressionSeverity::Low);
        assert_eq!(detector.determine_severity(15.0), RegressionSeverity::Medium);
        assert_eq!(detector.determine_severity(30.0), RegressionSeverity::High);
        assert_eq!(detector.determine_severity(60.0), RegressionSeverity::Critical);
    }

    #[test]
    fn test_statistical_significance() {
        let detector = RegressionDetector::new();
        
        let baseline = PerformanceBaseline {
            name: "test".to_string(),
            mean: 100.0,
            std_dev: 10.0,
            median: 100.0,
            p95: 120.0,
            p99: 130.0,
            sample_count: 100,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            historical_values: vec![100.0; 100],
        };
        
        // Small change - should not be significant
        let small_change = vec![105.0; 20];
        let p_value_small = detector.calculate_t_test_p_value(&baseline, &small_change);
        assert!(p_value_small > 0.05);
        
        // Large change - should be significant
        let large_change = vec![150.0; 20];
        let p_value_large = detector.calculate_t_test_p_value(&baseline, &large_change);
        assert!(p_value_large <= 0.05);
    }

    #[test]
    fn test_metrics_extraction() {
        let detector = RegressionDetector::new();
        let profiles = vec![create_test_profile(5, 1000.0)];
        let profile_refs: Vec<&ProfileSnapshot> = profiles.iter().collect();
        
        let metrics = detector.extract_metrics_from_profiles(&profile_refs);
        
        assert!(metrics.contains_key("test_function_latency"));
        assert!(metrics.contains_key("test_function_cpu"));
        assert!(metrics.contains_key("test_function_memory"));
        assert!(metrics.contains_key("profile_duration"));
        assert!(metrics.contains_key("sample_count"));
        
        let latency_values = &metrics["test_function_latency"];
        assert_eq!(latency_values.len(), 5);
    }

    #[test]
    fn test_regression_config() {
        let config = RegressionConfig {
            min_regression_percent: 5.0,
            baseline_sample_count: 50,
            comparison_sample_count: 10,
            significance_threshold: 0.01,
            enable_trend_analysis: false,
            min_confidence: 0.95,
            monitored_functions: vec!["critical_function".to_string()],
            monitored_metrics: vec!["latency".to_string()],
        };
        
        let detector = RegressionDetector::with_config(config.clone());
        assert_eq!(detector.config.min_regression_percent, 5.0);
        assert_eq!(detector.config.baseline_sample_count, 50);
        assert!(!detector.config.enable_trend_analysis);
    }

    #[test]
    fn test_detection_history() {
        let detector = RegressionDetector::new();
        
        // Simulate some detections
        let mut history = detector.detection_history.lock().unwrap();
        for i in 0..5 {
            history.push(RegressionDetection {
                timestamp: now_nanos(),
                regression_count: i,
                regressions: Vec::new(),
                detection_duration_ns: 1000000,
            });
        }
        drop(history);
        
        let retrieved_history = detector.get_detection_history(Some(3));
        assert_eq!(retrieved_history.len(), 3);
        
        // Should be in reverse chronological order
        assert!(retrieved_history[0].regression_count >= retrieved_history[1].regression_count);
    }

    #[test]
    fn test_insufficient_samples_error() {
        let detector = RegressionDetector::new();
        let empty_profiles: Vec<&ProfileSnapshot> = Vec::new();
        
        let result = detector.update_baselines(&empty_profiles);
        assert!(matches!(result, Err(RegressionError::InsufficientSamples)));
    }
}