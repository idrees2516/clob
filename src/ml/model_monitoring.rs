//! Model Monitoring Module
//! 
//! Implements comprehensive model performance monitoring, alerting,
//! and health checks for machine learning models in production.

use crate::ml::{MLError, MLResult, ModelPerformance, MLMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMonitoringConfig {
    pub performance_window: usize,
    pub alert_thresholds: AlertThresholds,
    pub monitoring_frequency: Duration,
    pub health_check_interval: Duration,
    pub metrics_retention_period: Duration,
    pub anomaly_detection_sensitivity: f64,
    pub drift_detection_enabled: bool,
    pub auto_remediation_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub accuracy_drop: f64,
    pub precision_drop: f64,
    pub recall_drop: f64,
    pub latency_increase: f64,
    pub error_rate_increase: f64,
    pub memory_usage_threshold: f64,
    pub cpu_usage_threshold: f64,
}

impl Default for ModelMonitoringConfig {
    fn default() -> Self {
        Self {
            performance_window: 1000,
            alert_thresholds: AlertThresholds {
                accuracy_drop: 0.05,
                precision_drop: 0.05,
                recall_drop: 0.05,
                latency_increase: 2.0,
                error_rate_increase: 0.02,
                memory_usage_threshold: 0.8,
                cpu_usage_threshold: 0.8,
            },
            monitoring_frequency: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            metrics_retention_period: Duration::from_secs(86400 * 7), // 7 days
            anomaly_detection_sensitivity: 0.95,
            drift_detection_enabled: true,
            auto_remediation_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAlert {
    pub alert_id: String,
    pub model_id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub metrics: HashMap<String, f64>,
    pub resolved: bool,
    pub resolution_time: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    PerformanceDegradation,
    LatencyIncrease,
    ErrorRateIncrease,
    ResourceExhaustion,
    ModelDrift,
    DataQualityIssue,
    SystemFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealthStatus {
    pub model_id: String,
    pub overall_health: HealthScore,
    pub performance_health: HealthScore,
    pub latency_health: HealthScore,
    pub resource_health: HealthScore,
    pub data_quality_health: HealthScore,
    pub last_updated: SystemTime,
    pub active_alerts: Vec<ModelAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthScore {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl HealthScore {
    pub fn from_score(score: f64) -> Self {
        match score {
            s if s >= 0.9 => HealthScore::Excellent,
            s if s >= 0.8 => HealthScore::Good,
            s if s >= 0.6 => HealthScore::Fair,
            s if s >= 0.4 => HealthScore::Poor,
            _ => HealthScore::Critical,
        }
    }

    pub fn to_score(&self) -> f64 {
        match self {
            HealthScore::Excellent => 0.95,
            HealthScore::Good => 0.85,
            HealthScore::Fair => 0.7,
            HealthScore::Poor => 0.5,
            HealthScore::Critical => 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedMetrics {
    pub timestamp: SystemTime,
    pub metrics: MLMetrics,
    pub latency_ns: u64,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_used_mb: u64,
    pub memory_usage_ratio: f64,
    pub cpu_usage_ratio: f64,
    pub disk_usage_mb: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub model_id: String,
    pub avg_accuracy: f64,
    pub avg_precision: f64,
    pub avg_recall: f64,
    pub avg_f1_score: f64,
    pub avg_latency_ns: u64,
    pub p95_latency_ns: u64,
    pub p99_latency_ns: u64,
    pub total_predictions: usize,
    pub error_rate: f64,
    pub uptime_percentage: f64,
}

/// Comprehensive model monitoring system
pub struct ModelMonitor {
    config: ModelMonitoringConfig,
    model_metrics: HashMap<String, VecDeque<TimestampedMetrics>>,
    baseline_metrics: HashMap<String, MLMetrics>,
    active_alerts: HashMap<String, Vec<ModelAlert>>,
    health_status: HashMap<String, ModelHealthStatus>,
    anomaly_detector: AnomalyDetector,
    performance_tracker: PerformanceTracker,
    resource_monitor: ResourceMonitor,
    alert_manager: AlertManager,
    last_health_check: SystemTime,
}

impl ModelMonitor {
    pub fn new(config: ModelMonitoringConfig) -> Self {
        Self {
            anomaly_detector: AnomalyDetector::new(config.clone()),
            performance_tracker: PerformanceTracker::new(config.clone()),
            resource_monitor: ResourceMonitor::new(config.clone()),
            alert_manager: AlertManager::new(config.clone()),
            config,
            model_metrics: HashMap::new(),
            baseline_metrics: HashMap::new(),
            active_alerts: HashMap::new(),
            health_status: HashMap::new(),
            last_health_check: SystemTime::now(),
        }
    }

    /// Register a model for monitoring
    pub fn register_model(&mut self, model_id: String, baseline_metrics: MLMetrics) -> MLResult<()> {
        self.baseline_metrics.insert(model_id.clone(), baseline_metrics);
        self.model_metrics.insert(model_id.clone(), VecDeque::new());
        self.active_alerts.insert(model_id.clone(), Vec::new());
        
        // Initialize health status
        let health_status = ModelHealthStatus {
            model_id: model_id.clone(),
            overall_health: HealthScore::Good,
            performance_health: HealthScore::Good,
            latency_health: HealthScore::Good,
            resource_health: HealthScore::Good,
            data_quality_health: HealthScore::Good,
            last_updated: SystemTime::now(),
            active_alerts: Vec::new(),
        };
        
        self.health_status.insert(model_id, health_status);
        Ok(())
    }

    /// Record new metrics for a model
    pub fn record_metrics(
        &mut self,
        model_id: &str,
        metrics: MLMetrics,
        latency_ns: u64,
        resource_usage: ResourceUsage,
    ) -> MLResult<Vec<ModelAlert>> {
        let timestamped_metrics = TimestampedMetrics {
            timestamp: SystemTime::now(),
            metrics,
            latency_ns,
            resource_usage,
        };

        // Add to metrics buffer
        let model_metrics = self.model_metrics.get_mut(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not registered", model_id)))?;

        model_metrics.push_back(timestamped_metrics.clone());
        
        // Maintain window size
        while model_metrics.len() > self.config.performance_window {
            model_metrics.pop_front();
        }

        // Check for alerts
        let mut new_alerts = Vec::new();

        // Performance degradation alerts
        if let Some(baseline) = self.baseline_metrics.get(model_id) {
            new_alerts.extend(self.check_performance_alerts(model_id, &metrics, baseline)?);
        }

        // Latency alerts
        new_alerts.extend(self.check_latency_alerts(model_id, latency_ns, model_metrics)?);

        // Resource usage alerts
        new_alerts.extend(self.check_resource_alerts(model_id, &resource_usage)?);

        // Anomaly detection
        if model_metrics.len() >= 10 {
            new_alerts.extend(self.anomaly_detector.detect_anomalies(model_id, model_metrics)?);
        }

        // Update active alerts
        if let Some(alerts) = self.active_alerts.get_mut(model_id) {
            alerts.extend(new_alerts.clone());
        }

        // Update health status
        self.update_health_status(model_id)?;

        Ok(new_alerts)
    }

    /// Perform comprehensive health check
    pub fn health_check(&mut self) -> MLResult<HashMap<String, ModelHealthStatus>> {
        if self.last_health_check.elapsed().unwrap_or(Duration::ZERO) < self.config.health_check_interval {
            return Ok(self.health_status.clone());
        }

        for model_id in self.model_metrics.keys().cloned().collect::<Vec<_>>() {
            self.update_health_status(&model_id)?;
        }

        self.last_health_check = SystemTime::now();
        Ok(self.health_status.clone())
    }

    /// Get current health status for a model
    pub fn get_health_status(&self, model_id: &str) -> MLResult<ModelHealthStatus> {
        self.health_status.get(model_id)
            .cloned()
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))
    }

    /// Get performance summary for a model
    pub fn get_performance_summary(&self, model_id: &str) -> MLResult<PerformanceSummary> {
        let metrics = self.model_metrics.get(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))?;

        if metrics.is_empty() {
            return Err(MLError::InsufficientData("No metrics available".to_string()));
        }

        let recent_metrics: Vec<&MLMetrics> = metrics.iter()
            .rev()
            .take(100)
            .map(|tm| &tm.metrics)
            .collect();

        let avg_accuracy = recent_metrics.iter().map(|m| m.accuracy).sum::<f64>() / recent_metrics.len() as f64;
        let avg_precision = recent_metrics.iter().map(|m| m.precision).sum::<f64>() / recent_metrics.len() as f64;
        let avg_recall = recent_metrics.iter().map(|m| m.recall).sum::<f64>() / recent_metrics.len() as f64;
        let avg_f1 = recent_metrics.iter().map(|m| m.f1_score).sum::<f64>() / recent_metrics.len() as f64;

        let recent_latencies: Vec<u64> = metrics.iter()
            .rev()
            .take(100)
            .map(|tm| tm.latency_ns)
            .collect();

        let avg_latency = recent_latencies.iter().sum::<u64>() as f64 / recent_latencies.len() as f64;
        let p95_latency = self.calculate_percentile(&recent_latencies, 0.95);
        let p99_latency = self.calculate_percentile(&recent_latencies, 0.99);

        Ok(PerformanceSummary {
            model_id: model_id.to_string(),
            avg_accuracy,
            avg_precision,
            avg_recall,
            avg_f1_score: avg_f1,
            avg_latency_ns: avg_latency as u64,
            p95_latency_ns: p95_latency,
            p99_latency_ns: p99_latency,
            total_predictions: metrics.len(),
            error_rate: self.calculate_error_rate(metrics),
            uptime_percentage: self.calculate_uptime(model_id),
        })
    }

    /// Resolve an alert
    pub fn resolve_alert(&mut self, model_id: &str, alert_id: &str) -> MLResult<()> {
        if let Some(alerts) = self.active_alerts.get_mut(model_id) {
            for alert in alerts.iter_mut() {
                if alert.alert_id == alert_id {
                    alert.resolved = true;
                    alert.resolution_time = Some(SystemTime::now());
                    break;
                }
            }
        }

        // Update health status
        self.update_health_status(model_id)?;
        Ok(())
    }

    fn check_performance_alerts(
        &self,
        model_id: &str,
        current: &MLMetrics,
        baseline: &MLMetrics,
    ) -> MLResult<Vec<ModelAlert>> {
        let mut alerts = Vec::new();

        // Accuracy degradation
        if baseline.accuracy - current.accuracy > self.config.alert_thresholds.accuracy_drop {
            alerts.push(ModelAlert {
                alert_id: format!("acc_drop_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                model_id: model_id.to_string(),
                alert_type: AlertType::PerformanceDegradation,
                severity: AlertSeverity::High,
                message: format!("Accuracy dropped from {:.3} to {:.3}", baseline.accuracy, current.accuracy),
                timestamp: SystemTime::now(),
                metrics: {
                    let mut map = HashMap::new();
                    map.insert("baseline_accuracy".to_string(), baseline.accuracy);
                    map.insert("current_accuracy".to_string(), current.accuracy);
                    map.insert("drop".to_string(), baseline.accuracy - current.accuracy);
                    map
                },
                resolved: false,
                resolution_time: None,
            });
        }

        // Precision degradation
        if baseline.precision - current.precision > self.config.alert_thresholds.precision_drop {
            alerts.push(ModelAlert {
                alert_id: format!("prec_drop_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                model_id: model_id.to_string(),
                alert_type: AlertType::PerformanceDegradation,
                severity: AlertSeverity::Medium,
                message: format!("Precision dropped from {:.3} to {:.3}", baseline.precision, current.precision),
                timestamp: SystemTime::now(),
                metrics: {
                    let mut map = HashMap::new();
                    map.insert("baseline_precision".to_string(), baseline.precision);
                    map.insert("current_precision".to_string(), current.precision);
                    map.insert("drop".to_string(), baseline.precision - current.precision);
                    map
                },
                resolved: false,
                resolution_time: None,
            });
        }

        Ok(alerts)
    }

    fn check_latency_alerts(
        &self,
        model_id: &str,
        current_latency: u64,
        metrics_history: &VecDeque<TimestampedMetrics>,
    ) -> MLResult<Vec<ModelAlert>> {
        let mut alerts = Vec::new();

        if metrics_history.len() < 10 {
            return Ok(alerts);
        }

        // Calculate baseline latency (average of last 100 measurements)
        let recent_latencies: Vec<u64> = metrics_history.iter()
            .rev()
            .take(100)
            .map(|tm| tm.latency_ns)
            .collect();

        let baseline_latency = recent_latencies.iter().sum::<u64>() as f64 / recent_latencies.len() as f64;

        if current_latency as f64 > baseline_latency * self.config.alert_thresholds.latency_increase {
            alerts.push(ModelAlert {
                alert_id: format!("lat_inc_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                model_id: model_id.to_string(),
                alert_type: AlertType::LatencyIncrease,
                severity: AlertSeverity::Medium,
                message: format!("Latency increased from {:.0}ns to {}ns", baseline_latency, current_latency),
                timestamp: SystemTime::now(),
                metrics: {
                    let mut map = HashMap::new();
                    map.insert("baseline_latency_ns".to_string(), baseline_latency);
                    map.insert("current_latency_ns".to_string(), current_latency as f64);
                    map.insert("increase_factor".to_string(), current_latency as f64 / baseline_latency);
                    map
                },
                resolved: false,
                resolution_time: None,
            });
        }

        Ok(alerts)
    }

    fn check_resource_alerts(
        &self,
        model_id: &str,
        resource_usage: &ResourceUsage,
    ) -> MLResult<Vec<ModelAlert>> {
        let mut alerts = Vec::new();

        // Memory usage alert
        if resource_usage.memory_usage_ratio > self.config.alert_thresholds.memory_usage_threshold {
            alerts.push(ModelAlert {
                alert_id: format!("mem_high_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                model_id: model_id.to_string(),
                alert_type: AlertType::ResourceExhaustion,
                severity: AlertSeverity::High,
                message: format!("High memory usage: {:.1}%", resource_usage.memory_usage_ratio * 100.0),
                timestamp: SystemTime::now(),
                metrics: {
                    let mut map = HashMap::new();
                    map.insert("memory_usage_ratio".to_string(), resource_usage.memory_usage_ratio);
                    map.insert("memory_used_mb".to_string(), resource_usage.memory_used_mb as f64);
                    map
                },
                resolved: false,
                resolution_time: None,
            });
        }

        // CPU usage alert
        if resource_usage.cpu_usage_ratio > self.config.alert_thresholds.cpu_usage_threshold {
            alerts.push(ModelAlert {
                alert_id: format!("cpu_high_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                model_id: model_id.to_string(),
                alert_type: AlertType::ResourceExhaustion,
                severity: AlertSeverity::Medium,
                message: format!("High CPU usage: {:.1}%", resource_usage.cpu_usage_ratio * 100.0),
                timestamp: SystemTime::now(),
                metrics: {
                    let mut map = HashMap::new();
                    map.insert("cpu_usage_ratio".to_string(), resource_usage.cpu_usage_ratio);
                    map
                },
                resolved: false,
                resolution_time: None,
            });
        }

        Ok(alerts)
    }

    fn update_health_status(&mut self, model_id: &str) -> MLResult<()> {
        let metrics = self.model_metrics.get(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))?;

        let active_alerts = self.active_alerts.get(model_id)
            .map(|alerts| alerts.iter().filter(|a| !a.resolved).cloned().collect())
            .unwrap_or_else(Vec::new);

        // Calculate health scores
        let performance_health = self.calculate_performance_health(metrics);
        let latency_health = self.calculate_latency_health(metrics);
        let resource_health = self.calculate_resource_health(metrics);
        let data_quality_health = self.calculate_data_quality_health(metrics);

        // Overall health is the minimum of all health scores
        let overall_health_score = [
            performance_health.to_score(),
            latency_health.to_score(),
            resource_health.to_score(),
            data_quality_health.to_score(),
        ].iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let overall_health = HealthScore::from_score(overall_health_score);

        let health_status = ModelHealthStatus {
            model_id: model_id.to_string(),
            overall_health,
            performance_health,
            latency_health,
            resource_health,
            data_quality_health,
            last_updated: SystemTime::now(),
            active_alerts,
        };

        self.health_status.insert(model_id.to_string(), health_status);
        Ok(())
    }

    fn calculate_performance_health(&self, metrics: &VecDeque<TimestampedMetrics>) -> HealthScore {
        if metrics.is_empty() {
            return HealthScore::Poor;
        }

        let recent_metrics: Vec<&MLMetrics> = metrics.iter()
            .rev()
            .take(50)
            .map(|tm| &tm.metrics)
            .collect();

        let avg_accuracy = recent_metrics.iter().map(|m| m.accuracy).sum::<f64>() / recent_metrics.len() as f64;
        HealthScore::from_score(avg_accuracy)
    }

    fn calculate_latency_health(&self, metrics: &VecDeque<TimestampedMetrics>) -> HealthScore {
        if metrics.is_empty() {
            return HealthScore::Poor;
        }

        let recent_latencies: Vec<u64> = metrics.iter()
            .rev()
            .take(50)
            .map(|tm| tm.latency_ns)
            .collect();

        let avg_latency = recent_latencies.iter().sum::<u64>() as f64 / recent_latencies.len() as f64;
        
        // Convert latency to health score (lower is better)
        let latency_ms = avg_latency / 1_000_000.0;
        let health_score = match latency_ms {
            l if l < 1.0 => 0.95,
            l if l < 10.0 => 0.85,
            l if l < 50.0 => 0.7,
            l if l < 100.0 => 0.5,
            _ => 0.2,
        };

        HealthScore::from_score(health_score)
    }

    fn calculate_resource_health(&self, metrics: &VecDeque<TimestampedMetrics>) -> HealthScore {
        if metrics.is_empty() {
            return HealthScore::Poor;
        }

        let recent_usage: Vec<&ResourceUsage> = metrics.iter()
            .rev()
            .take(50)
            .map(|tm| &tm.resource_usage)
            .collect();

        let avg_memory_usage = recent_usage.iter().map(|r| r.memory_usage_ratio).sum::<f64>() / recent_usage.len() as f64;
        let avg_cpu_usage = recent_usage.iter().map(|r| r.cpu_usage_ratio).sum::<f64>() / recent_usage.len() as f64;

        let resource_score = 1.0 - (avg_memory_usage + avg_cpu_usage) / 2.0;
        HealthScore::from_score(resource_score)
    }

    fn calculate_data_quality_health(&self, _metrics: &VecDeque<TimestampedMetrics>) -> HealthScore {
        // Simplified data quality assessment
        // In practice, this would analyze data distribution, missing values, etc.
        HealthScore::Good
    }

    fn calculate_percentile(&self, values: &[u64], percentile: f64) -> u64 {
        if values.is_empty() {
            return 0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_unstable();
        
        let index = ((values.len() - 1) as f64 * percentile) as usize;
        sorted_values[index]
    }

    fn calculate_error_rate(&self, metrics: &VecDeque<TimestampedMetrics>) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        // Simplified error rate calculation
        let recent_accuracy: Vec<f64> = metrics.iter()
            .rev()
            .take(100)
            .map(|tm| tm.metrics.accuracy)
            .collect();

        let avg_accuracy = recent_accuracy.iter().sum::<f64>() / recent_accuracy.len() as f64;
        1.0 - avg_accuracy
    }

    fn calculate_uptime(&self, _model_id: &str) -> f64 {
        // Simplified uptime calculation
        // In practice, this would track actual service availability
        99.5
    }
}

/// Anomaly detection for model performance
pub struct AnomalyDetector {
    config: ModelMonitoringConfig,
}

impl AnomalyDetector {
    pub fn new(config: ModelMonitoringConfig) -> Self {
        Self { config }
    }

    pub fn detect_anomalies(
        &self,
        model_id: &str,
        metrics: &VecDeque<TimestampedMetrics>,
    ) -> MLResult<Vec<ModelAlert>> {
        let mut alerts = Vec::new();

        if metrics.len() < 20 {
            return Ok(alerts);
        }

        // Extract accuracy values for anomaly detection
        let accuracy_values: Vec<f64> = metrics.iter()
            .map(|tm| tm.metrics.accuracy)
            .collect();

        // Simple statistical anomaly detection using z-score
        let mean = accuracy_values.iter().sum::<f64>() / accuracy_values.len() as f64;
        let variance = accuracy_values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / accuracy_values.len() as f64;
        let std_dev = variance.sqrt();

        if let Some(latest_accuracy) = accuracy_values.last() {
            let z_score = (latest_accuracy - mean) / std_dev;
            
            if z_score.abs() > 2.0 { // 2 standard deviations
                alerts.push(ModelAlert {
                    alert_id: format!("anomaly_{}_{}", model_id, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()),
                    model_id: model_id.to_string(),
                    alert_type: AlertType::DataQualityIssue,
                    severity: if z_score.abs() > 3.0 { AlertSeverity::High } else { AlertSeverity::Medium },
                    message: format!("Anomalous accuracy detected: {:.3} (z-score: {:.2})", latest_accuracy, z_score),
                    timestamp: SystemTime::now(),
                    metrics: {
                        let mut map = HashMap::new();
                        map.insert("accuracy".to_string(), *latest_accuracy);
                        map.insert("z_score".to_string(), z_score);
                        map.insert("mean_accuracy".to_string(), mean);
                        map.insert("std_dev".to_string(), std_dev);
                        map
                    },
                    resolved: false,
                    resolution_time: None,
                });
            }
        }

        Ok(alerts)
    }
}

/// Performance tracking and trend analysis
pub struct PerformanceTracker {
    config: ModelMonitoringConfig,
}

impl PerformanceTracker {
    pub fn new(config: ModelMonitoringConfig) -> Self {
        Self { config }
    }

    pub fn analyze_trends(&self, metrics: &VecDeque<TimestampedMetrics>) -> MLResult<PerformanceTrend> {
        if metrics.len() < 10 {
            return Ok(PerformanceTrend {
                accuracy_trend: TrendDirection::Stable,
                latency_trend: TrendDirection::Stable,
                resource_trend: TrendDirection::Stable,
                confidence: 0.0,
            });
        }

        let accuracy_values: Vec<f64> = metrics.iter().map(|tm| tm.metrics.accuracy).collect();
        let latency_values: Vec<f64> = metrics.iter().map(|tm| tm.latency_ns as f64).collect();
        let memory_values: Vec<f64> = metrics.iter().map(|tm| tm.resource_usage.memory_usage_ratio).collect();

        let accuracy_trend = self.calculate_trend(&accuracy_values);
        let latency_trend = self.calculate_trend(&latency_values);
        let resource_trend = self.calculate_trend(&memory_values);

        Ok(PerformanceTrend {
            accuracy_trend,
            latency_trend,
            resource_trend,
            confidence: 0.8, // Simplified confidence calculation
        })
    }

    fn calculate_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 5 {
            return TrendDirection::Stable;
        }

        // Simple linear regression slope calculation
        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = values.iter()
            .enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = (0..values.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        if denominator == 0.0 {
            return TrendDirection::Stable;
        }

        let slope = numerator / denominator;
        
        if slope > 0.001 {
            TrendDirection::Increasing
        } else if slope < -0.001 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Resource monitoring
pub struct ResourceMonitor {
    config: ModelMonitoringConfig,
}

impl ResourceMonitor {
    pub fn new(config: ModelMonitoringConfig) -> Self {
        Self { config }
    }

    pub fn get_current_usage(&self) -> ResourceUsage {
        // Simplified resource monitoring
        // In practice, this would use system APIs to get actual resource usage
        ResourceUsage {
            memory_used_mb: 512,
            memory_usage_ratio: 0.3,
            cpu_usage_ratio: 0.2,
            disk_usage_mb: 1024,
            network_bytes_sent: 1000000,
            network_bytes_received: 2000000,
        }
    }
}

/// Alert management system
pub struct AlertManager {
    config: ModelMonitoringConfig,
}

impl AlertManager {
    pub fn new(config: ModelMonitoringConfig) -> Self {
        Self { config }
    }

    pub fn send_alert(&self, alert: &ModelAlert) -> MLResult<()> {
        // Simplified alert sending
        // In practice, this would integrate with notification systems
        println!("ALERT [{}]: {} - {}", alert.severity, alert.alert_type, alert.message);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub accuracy_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub resource_trend: TrendDirection,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_monitor_creation() {
        let config = ModelMonitoringConfig::default();
        let monitor = ModelMonitor::new(config);
        assert_eq!(monitor.model_metrics.len(), 0);
    }

    #[test]
    fn test_model_registration() {
        let config = ModelMonitoringConfig::default();
        let mut monitor = ModelMonitor::new(config);
        
        let baseline_metrics = MLMetrics {
            accuracy: 0.85,
            precision: 0.82,
            recall: 0.88,
            f1_score: 0.85,
            auc_roc: 0.90,
            log_likelihood: -0.5,
        };

        let result = monitor.register_model("test_model".to_string(), baseline_metrics);
        assert!(result.is_ok());
        assert_eq!(monitor.model_metrics.len(), 1);
    }

    #[test]
    fn test_health_score_conversion() {
        assert_eq!(HealthScore::from_score(0.95), HealthScore::Excellent);
        assert_eq!(HealthScore::from_score(0.85), HealthScore::Good);
        assert_eq!(HealthScore::from_score(0.7), HealthScore::Fair);
        assert_eq!(HealthScore::from_score(0.5), HealthScore::Poor);
        assert_eq!(HealthScore::from_score(0.2), HealthScore::Critical);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = ModelMonitoringConfig::default();
        let detector = AnomalyDetector::new(config);
        
        let mut metrics = VecDeque::new();
        
        // Add normal metrics
        for _ in 0..20 {
            metrics.push_back(TimestampedMetrics {
                timestamp: SystemTime::now(),
                metrics: MLMetrics {
                    accuracy: 0.85,
                    precision: 0.82,
                    recall: 0.88,
                    f1_score: 0.85,
                    auc_roc: 0.90,
                    log_likelihood: -0.5,
                },
                latency_ns: 1000000,
                resource_usage: ResourceUsage {
                    memory_used_mb: 512,
                    memory_usage_ratio: 0.3,
                    cpu_usage_ratio: 0.2,
                    disk_usage_mb: 1024,
                    network_bytes_sent: 1000000,
                    network_bytes_received: 2000000,
                },
            });
        }

        // Add anomalous metric
        metrics.push_back(TimestampedMetrics {
            timestamp: SystemTime::now(),
            metrics: MLMetrics {
                accuracy: 0.5, // Significantly lower
                precision: 0.45,
                recall: 0.55,
                f1_score: 0.5,
                auc_roc: 0.6,
                log_likelihood: -1.0,
            },
            latency_ns: 1000000,
            resource_usage: ResourceUsage {
                memory_used_mb: 512,
                memory_usage_ratio: 0.3,
                cpu_usage_ratio: 0.2,
                disk_usage_mb: 1024,
                network_bytes_sent: 1000000,
                network_bytes_received: 2000000,
            },
        });

        let result = detector.detect_anomalies("test_model", &metrics);
        assert!(result.is_ok());
        
        let alerts = result.unwrap();
        assert!(!alerts.is_empty());
    }
}