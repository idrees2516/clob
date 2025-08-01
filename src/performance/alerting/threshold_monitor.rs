use super::alert_manager::{AlertManager, AlertSeverity, fire_alert};
use super::timing::{now_nanos, PerformanceMetrics, MetricCounter};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Real-time threshold monitoring system for performance metrics
/// Provides sub-second detection of threshold violations
pub struct ThresholdMonitor {
    /// Threshold configurations
    thresholds: std::sync::RwLock<HashMap<String, ThresholdConfig>>,
    
    /// Monitor state
    state: MonitorState,
    
    /// Performance metrics reference
    metrics: Arc<PerformanceMetrics>,
    
    /// Alert manager reference
    alert_manager: Arc<std::sync::Mutex<AlertManager>>,
    
    /// Monitoring thread handle
    monitor_thread: Option<std::thread::JoinHandle<()>>,
}

/// Threshold configuration for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Metric name to monitor
    pub metric_name: String,
    
    /// Warning threshold
    pub warning_threshold: f64,
    
    /// Critical threshold
    pub critical_threshold: f64,
    
    /// Emergency threshold
    pub emergency_threshold: f64,
    
    /// Threshold type (greater than, less than, etc.)
    pub threshold_type: ThresholdType,
    
    /// Evaluation window (nanoseconds)
    pub evaluation_window_ns: u64,
    
    /// Minimum samples required for evaluation
    pub min_samples: u32,
    
    /// Consecutive violations required to trigger alert
    pub consecutive_violations: u32,
    
    /// Cooldown period between alerts (nanoseconds)
    pub cooldown_period_ns: u64,
    
    /// Whether this threshold is enabled
    pub enabled: bool,
    
    /// Custom alert message template
    pub alert_message_template: String,
}

/// Types of threshold comparisons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdType {
    /// Value must be greater than threshold
    GreaterThan,
    
    /// Value must be less than threshold
    LessThan,
    
    /// Value must be within range (threshold ± tolerance)
    WithinRange,
    
    /// Value must be outside range
    OutsideRange,
    
    /// Rate of change exceeds threshold
    RateOfChange,
    
    /// Percentile-based threshold
    Percentile,
}

/// Threshold violation information
#[derive(Debug, Clone)]
pub struct ThresholdViolation {
    /// Metric name
    pub metric_name: String,
    
    /// Current metric value
    pub current_value: f64,
    
    /// Violated threshold value
    pub threshold_value: f64,
    
    /// Violation severity
    pub severity: AlertSeverity,
    
    /// Violation timestamp
    pub timestamp: u64,
    
    /// Number of consecutive violations
    pub consecutive_count: u32,
    
    /// Evaluation window data
    pub window_data: WindowData,
}

/// Evaluation window data
#[derive(Debug, Clone)]
pub struct WindowData {
    /// Sample count in window
    pub sample_count: u32,
    
    /// Average value in window
    pub average: f64,
    
    /// Minimum value in window
    pub minimum: f64,
    
    /// Maximum value in window
    pub maximum: f64,
    
    /// Standard deviation in window
    pub std_deviation: f64,
    
    /// Window start timestamp
    pub window_start: u64,
    
    /// Window end timestamp
    pub window_end: u64,
}

/// Monitor state
#[repr(align(64))]
struct MonitorState {
    /// Whether monitoring is active
    is_running: AtomicBool,
    
    /// Monitoring interval (nanoseconds)
    monitoring_interval_ns: AtomicU64,
    
    /// Total evaluations performed
    total_evaluations: AtomicU64,
    
    /// Total violations detected
    total_violations: AtomicU64,
    
    /// Last evaluation timestamp
    last_evaluation: AtomicU64,
    
    /// Evaluation latency tracking
    evaluation_latency_sum: AtomicU64,
    evaluation_count: AtomicU64,
}

/// Threshold violation tracking
struct ViolationTracker {
    /// Consecutive violation count per metric
    consecutive_counts: HashMap<String, u32>,
    
    /// Last alert timestamp per metric
    last_alert_times: HashMap<String, u64>,
    
    /// Historical values for rate calculations
    value_history: HashMap<String, VecDeque<(u64, f64)>>,
}

use std::collections::VecDeque;

impl ThresholdMonitor {
    /// Create a new threshold monitor
    pub fn new(metrics: Arc<PerformanceMetrics>, alert_manager: Arc<std::sync::Mutex<AlertManager>>) -> Self {
        Self {
            thresholds: std::sync::RwLock::new(HashMap::new()),
            state: MonitorState {
                is_running: AtomicBool::new(false),
                monitoring_interval_ns: AtomicU64::new(100_000_000), // 100ms default
                total_evaluations: AtomicU64::new(0),
                total_violations: AtomicU64::new(0),
                last_evaluation: AtomicU64::new(0),
                evaluation_latency_sum: AtomicU64::new(0),
                evaluation_count: AtomicU64::new(0),
            },
            metrics,
            alert_manager,
            monitor_thread: None,
        }
    }

    /// Add a threshold configuration
    pub fn add_threshold(&self, config: ThresholdConfig) {
        let mut thresholds = self.thresholds.write().unwrap();
        thresholds.insert(config.metric_name.clone(), config);
    }

    /// Remove a threshold configuration
    pub fn remove_threshold(&self, metric_name: &str) -> bool {
        let mut thresholds = self.thresholds.write().unwrap();
        thresholds.remove(metric_name).is_some()
    }

    /// Update threshold configuration
    pub fn update_threshold(&self, config: ThresholdConfig) -> bool {
        let mut thresholds = self.thresholds.write().unwrap();
        if thresholds.contains_key(&config.metric_name) {
            thresholds.insert(config.metric_name.clone(), config);
            true
        } else {
            false
        }
    }

    /// Get threshold configuration
    pub fn get_threshold(&self, metric_name: &str) -> Option<ThresholdConfig> {
        let thresholds = self.thresholds.read().unwrap();
        thresholds.get(metric_name).cloned()
    }

    /// Get all threshold configurations
    pub fn get_all_thresholds(&self) -> Vec<ThresholdConfig> {
        let thresholds = self.thresholds.read().unwrap();
        thresholds.values().cloned().collect()
    }

    /// Start monitoring
    pub fn start_monitoring(&mut self) -> Result<(), MonitorError> {
        if self.state.is_running.load(Ordering::Acquire) {
            return Err(MonitorError::AlreadyRunning);
        }

        self.state.is_running.store(true, Ordering::Release);
        
        // Clone necessary data for the monitoring thread
        let state = Arc::new(MonitorState {
            is_running: AtomicBool::new(true),
            monitoring_interval_ns: AtomicU64::new(self.state.monitoring_interval_ns.load(Ordering::Acquire)),
            total_evaluations: AtomicU64::new(0),
            total_violations: AtomicU64::new(0),
            last_evaluation: AtomicU64::new(0),
            evaluation_latency_sum: AtomicU64::new(0),
            evaluation_count: AtomicU64::new(0),
        });
        
        let thresholds = Arc::new(std::sync::RwLock::new(HashMap::new()));
        {
            let source_thresholds = self.thresholds.read().unwrap();
            let mut target_thresholds = thresholds.write().unwrap();
            *target_thresholds = source_thresholds.clone();
        }
        
        let metrics = self.metrics.clone();
        let alert_manager = self.alert_manager.clone();

        // Start monitoring thread
        let handle = thread::spawn(move || {
            let mut violation_tracker = ViolationTracker::new();
            
            while state.is_running.load(Ordering::Acquire) {
                let evaluation_start = now_nanos();
                
                // Evaluate all thresholds
                let threshold_configs = {
                    let thresholds_guard = thresholds.read().unwrap();
                    thresholds_guard.values().cloned().collect::<Vec<_>>()
                };
                
                for config in threshold_configs {
                    if config.enabled {
                        if let Some(violation) = Self::evaluate_threshold(&config, &metrics, &mut violation_tracker) {
                            Self::handle_violation(violation, &alert_manager);
                            state.total_violations.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                
                // Update statistics
                let evaluation_latency = now_nanos() - evaluation_start;
                state.total_evaluations.fetch_add(1, Ordering::Relaxed);
                state.last_evaluation.store(evaluation_start, Ordering::Release);
                state.evaluation_latency_sum.fetch_add(evaluation_latency, Ordering::Relaxed);
                state.evaluation_count.fetch_add(1, Ordering::Relaxed);
                
                // Sleep until next evaluation
                let interval_ns = state.monitoring_interval_ns.load(Ordering::Acquire);
                let sleep_duration = Duration::from_nanos(interval_ns);
                thread::sleep(sleep_duration);
            }
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) -> Result<(), MonitorError> {
        if !self.state.is_running.load(Ordering::Acquire) {
            return Err(MonitorError::NotRunning);
        }

        self.state.is_running.store(false, Ordering::Release);
        
        if let Some(handle) = self.monitor_thread.take() {
            handle.join().map_err(|_| MonitorError::ThreadJoinError)?;
        }

        Ok(())
    }

    /// Set monitoring interval
    pub fn set_monitoring_interval(&self, interval_ns: u64) {
        self.state.monitoring_interval_ns.store(interval_ns, Ordering::Release);
    }

    /// Get monitoring statistics
    pub fn get_stats(&self) -> MonitorStats {
        let evaluation_count = self.state.evaluation_count.load(Ordering::Acquire);
        let avg_latency = if evaluation_count > 0 {
            self.state.evaluation_latency_sum.load(Ordering::Acquire) / evaluation_count
        } else {
            0
        };

        MonitorStats {
            is_running: self.state.is_running.load(Ordering::Acquire),
            total_evaluations: self.state.total_evaluations.load(Ordering::Acquire),
            total_violations: self.state.total_violations.load(Ordering::Acquire),
            last_evaluation: self.state.last_evaluation.load(Ordering::Acquire),
            avg_evaluation_latency_ns: avg_latency,
            monitoring_interval_ns: self.state.monitoring_interval_ns.load(Ordering::Acquire),
            active_thresholds: self.thresholds.read().unwrap().len(),
        }
    }

    /// Evaluate a single threshold
    fn evaluate_threshold(
        config: &ThresholdConfig,
        metrics: &Arc<PerformanceMetrics>,
        violation_tracker: &mut ViolationTracker,
    ) -> Option<ThresholdViolation> {
        // Get current metric value
        let current_value = metrics.get_value(&config.metric_name)?;
        let current_time = now_nanos();
        
        // Update value history for rate calculations
        violation_tracker.update_value_history(&config.metric_name, current_time, current_value as f64);
        
        // Determine if threshold is violated
        let (is_violated, severity, threshold_value) = Self::check_threshold_violation(config, current_value as f64, violation_tracker);
        
        if is_violated {
            // Check cooldown period
            if let Some(&last_alert_time) = violation_tracker.last_alert_times.get(&config.metric_name) {
                if current_time - last_alert_time < config.cooldown_period_ns {
                    return None; // Still in cooldown
                }
            }
            
            // Update consecutive violation count
            let consecutive_count = violation_tracker.increment_consecutive_count(&config.metric_name);
            
            // Check if we have enough consecutive violations
            if consecutive_count >= config.consecutive_violations {
                // Create window data
                let window_data = Self::create_window_data(config, violation_tracker);
                
                // Reset consecutive count and update last alert time
                violation_tracker.reset_consecutive_count(&config.metric_name);
                violation_tracker.last_alert_times.insert(config.metric_name.clone(), current_time);
                
                return Some(ThresholdViolation {
                    metric_name: config.metric_name.clone(),
                    current_value: current_value as f64,
                    threshold_value,
                    severity,
                    timestamp: current_time,
                    consecutive_count,
                    window_data,
                });
            }
        } else {
            // Reset consecutive count if no violation
            violation_tracker.reset_consecutive_count(&config.metric_name);
        }
        
        None
    }

    /// Check if threshold is violated
    fn check_threshold_violation(
        config: &ThresholdConfig,
        current_value: f64,
        violation_tracker: &ViolationTracker,
    ) -> (bool, AlertSeverity, f64) {
        match config.threshold_type {
            ThresholdType::GreaterThan => {
                if current_value > config.emergency_threshold {
                    (true, AlertSeverity::Emergency, config.emergency_threshold)
                } else if current_value > config.critical_threshold {
                    (true, AlertSeverity::Critical, config.critical_threshold)
                } else if current_value > config.warning_threshold {
                    (true, AlertSeverity::Warning, config.warning_threshold)
                } else {
                    (false, AlertSeverity::Info, 0.0)
                }
            }
            ThresholdType::LessThan => {
                if current_value < config.emergency_threshold {
                    (true, AlertSeverity::Emergency, config.emergency_threshold)
                } else if current_value < config.critical_threshold {
                    (true, AlertSeverity::Critical, config.critical_threshold)
                } else if current_value < config.warning_threshold {
                    (true, AlertSeverity::Warning, config.warning_threshold)
                } else {
                    (false, AlertSeverity::Info, 0.0)
                }
            }
            ThresholdType::RateOfChange => {
                if let Some(rate) = Self::calculate_rate_of_change(config, violation_tracker) {
                    if rate.abs() > config.emergency_threshold {
                        (true, AlertSeverity::Emergency, config.emergency_threshold)
                    } else if rate.abs() > config.critical_threshold {
                        (true, AlertSeverity::Critical, config.critical_threshold)
                    } else if rate.abs() > config.warning_threshold {
                        (true, AlertSeverity::Warning, config.warning_threshold)
                    } else {
                        (false, AlertSeverity::Info, 0.0)
                    }
                } else {
                    (false, AlertSeverity::Info, 0.0)
                }
            }
            _ => {
                // Other threshold types not implemented yet
                (false, AlertSeverity::Info, 0.0)
            }
        }
    }

    /// Calculate rate of change for a metric
    fn calculate_rate_of_change(config: &ThresholdConfig, violation_tracker: &ViolationTracker) -> Option<f64> {
        let history = violation_tracker.value_history.get(&config.metric_name)?;
        
        if history.len() < 2 {
            return None;
        }
        
        let (latest_time, latest_value) = history.back()?;
        let (earliest_time, earliest_value) = history.front()?;
        
        let time_diff = latest_time - earliest_time;
        if time_diff == 0 {
            return None;
        }
        
        let value_diff = latest_value - earliest_value;
        let rate = value_diff / (time_diff as f64 / 1_000_000_000.0); // Rate per second
        
        Some(rate)
    }

    /// Create window data for violation
    fn create_window_data(config: &ThresholdConfig, violation_tracker: &ViolationTracker) -> WindowData {
        if let Some(history) = violation_tracker.value_history.get(&config.metric_name) {
            let values: Vec<f64> = history.iter().map(|(_, v)| *v).collect();
            
            if !values.is_empty() {
                let sum: f64 = values.iter().sum();
                let average = sum / values.len() as f64;
                let minimum = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let maximum = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                
                let variance = values.iter()
                    .map(|v| (v - average).powi(2))
                    .sum::<f64>() / values.len() as f64;
                let std_deviation = variance.sqrt();
                
                let window_start = history.front().map(|(t, _)| *t).unwrap_or(0);
                let window_end = history.back().map(|(t, _)| *t).unwrap_or(0);
                
                return WindowData {
                    sample_count: values.len() as u32,
                    average,
                    minimum,
                    maximum,
                    std_deviation,
                    window_start,
                    window_end,
                };
            }
        }
        
        // Default empty window data
        WindowData {
            sample_count: 0,
            average: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            std_deviation: 0.0,
            window_start: 0,
            window_end: 0,
        }
    }

    /// Handle threshold violation
    fn handle_violation(violation: ThresholdViolation, alert_manager: &Arc<std::sync::Mutex<AlertManager>>) {
        let message = format!(
            "Threshold violation: {} = {:.2} (threshold: {:.2})",
            violation.metric_name,
            violation.current_value,
            violation.threshold_value
        );
        
        if let Ok(manager) = alert_manager.try_lock() {
            let _ = manager.fire_alert(
                format!("threshold_violation_{}", violation.metric_name),
                violation.metric_name.clone(),
                violation.current_value,
                violation.threshold_value,
                violation.severity,
                message,
            );
        }
    }
}

impl ViolationTracker {
    fn new() -> Self {
        Self {
            consecutive_counts: HashMap::new(),
            last_alert_times: HashMap::new(),
            value_history: HashMap::new(),
        }
    }

    fn increment_consecutive_count(&mut self, metric_name: &str) -> u32 {
        let count = self.consecutive_counts.entry(metric_name.to_string()).or_insert(0);
        *count += 1;
        *count
    }

    fn reset_consecutive_count(&mut self, metric_name: &str) {
        self.consecutive_counts.insert(metric_name.to_string(), 0);
    }

    fn update_value_history(&mut self, metric_name: &str, timestamp: u64, value: f64) {
        let history = self.value_history.entry(metric_name.to_string()).or_insert_with(VecDeque::new);
        
        history.push_back((timestamp, value));
        
        // Keep only recent values (last 1000 samples or 1 hour, whichever is smaller)
        let max_age = 3_600_000_000_000u64; // 1 hour in nanoseconds
        let max_samples = 1000;
        
        while history.len() > max_samples || 
              (history.len() > 1 && timestamp - history.front().unwrap().0 > max_age) {
            history.pop_front();
        }
    }
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitorStats {
    pub is_running: bool,
    pub total_evaluations: u64,
    pub total_violations: u64,
    pub last_evaluation: u64,
    pub avg_evaluation_latency_ns: u64,
    pub monitoring_interval_ns: u64,
    pub active_thresholds: usize,
}

/// Monitor errors
#[derive(Debug, Clone)]
pub enum MonitorError {
    AlreadyRunning,
    NotRunning,
    ThreadJoinError,
    InvalidConfiguration,
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitorError::AlreadyRunning => write!(f, "Monitor is already running"),
            MonitorError::NotRunning => write!(f, "Monitor is not running"),
            MonitorError::ThreadJoinError => write!(f, "Failed to join monitor thread"),
            MonitorError::InvalidConfiguration => write!(f, "Invalid monitor configuration"),
        }
    }
}

impl std::error::Error for MonitorError {}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            metric_name: String::new(),
            warning_threshold: 1000.0,
            critical_threshold: 2000.0,
            emergency_threshold: 5000.0,
            threshold_type: ThresholdType::GreaterThan,
            evaluation_window_ns: 60_000_000_000, // 60 seconds
            min_samples: 1,
            consecutive_violations: 3,
            cooldown_period_ns: 300_000_000_000, // 5 minutes
            enabled: true,
            alert_message_template: "Threshold violation detected".to_string(),
        }
    }
}

/// Convenience function to create latency threshold
pub fn create_latency_threshold(metric_name: String, warning_us: f64, critical_us: f64, emergency_us: f64) -> ThresholdConfig {
    ThresholdConfig {
        metric_name,
        warning_threshold: warning_us * 1000.0, // Convert to nanoseconds
        critical_threshold: critical_us * 1000.0,
        emergency_threshold: emergency_us * 1000.0,
        threshold_type: ThresholdType::GreaterThan,
        consecutive_violations: 1, // Immediate alert for latency
        cooldown_period_ns: 60_000_000_000, // 1 minute cooldown
        ..Default::default()
    }
}

/// Convenience function to create throughput threshold
pub fn create_throughput_threshold(metric_name: String, min_warning: f64, min_critical: f64, min_emergency: f64) -> ThresholdConfig {
    ThresholdConfig {
        metric_name,
        warning_threshold: min_warning,
        critical_threshold: min_critical,
        emergency_threshold: min_emergency,
        threshold_type: ThresholdType::LessThan,
        consecutive_violations: 5, // Require sustained low throughput
        cooldown_period_ns: 120_000_000_000, // 2 minute cooldown
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::alert_manager::AlertManager;

    #[test]
    fn test_threshold_config_creation() {
        let config = ThresholdConfig {
            metric_name: "test_metric".to_string(),
            warning_threshold: 100.0,
            critical_threshold: 200.0,
            emergency_threshold: 500.0,
            threshold_type: ThresholdType::GreaterThan,
            ..Default::default()
        };
        
        assert_eq!(config.metric_name, "test_metric");
        assert_eq!(config.warning_threshold, 100.0);
        assert_eq!(config.threshold_type, ThresholdType::GreaterThan);
    }

    #[test]
    fn test_threshold_monitor_creation() {
        let metrics = Arc::new(PerformanceMetrics::new());
        let alert_manager = Arc::new(std::sync::Mutex::new(AlertManager::with_defaults()));
        
        let monitor = ThresholdMonitor::new(metrics, alert_manager);
        let stats = monitor.get_stats();
        
        assert!(!stats.is_running);
        assert_eq!(stats.active_thresholds, 0);
    }

    #[test]
    fn test_threshold_management() {
        let metrics = Arc::new(PerformanceMetrics::new());
        let alert_manager = Arc::new(std::sync::Mutex::new(AlertManager::with_defaults()));
        let monitor = ThresholdMonitor::new(metrics, alert_manager);
        
        let config = ThresholdConfig {
            metric_name: "test_metric".to_string(),
            warning_threshold: 100.0,
            ..Default::default()
        };
        
        monitor.add_threshold(config.clone());
        
        let retrieved = monitor.get_threshold("test_metric").unwrap();
        assert_eq!(retrieved.metric_name, config.metric_name);
        assert_eq!(retrieved.warning_threshold, config.warning_threshold);
        
        assert!(monitor.remove_threshold("test_metric"));
        assert!(monitor.get_threshold("test_metric").is_none());
    }

    #[test]
    fn test_latency_threshold_creation() {
        let threshold = create_latency_threshold(
            "order_latency".to_string(),
            1.0, // 1μs warning
            2.0, // 2μs critical
            5.0, // 5μs emergency
        );
        
        assert_eq!(threshold.metric_name, "order_latency");
        assert_eq!(threshold.warning_threshold, 1000.0); // 1000ns
        assert_eq!(threshold.critical_threshold, 2000.0); // 2000ns
        assert_eq!(threshold.emergency_threshold, 5000.0); // 5000ns
        assert_eq!(threshold.threshold_type, ThresholdType::GreaterThan);
    }

    #[test]
    fn test_throughput_threshold_creation() {
        let threshold = create_throughput_threshold(
            "orders_per_second".to_string(),
            1000.0, // Warning below 1000 ops/s
            500.0,  // Critical below 500 ops/s
            100.0,  // Emergency below 100 ops/s
        );
        
        assert_eq!(threshold.metric_name, "orders_per_second");
        assert_eq!(threshold.warning_threshold, 1000.0);
        assert_eq!(threshold.threshold_type, ThresholdType::LessThan);
        assert_eq!(threshold.consecutive_violations, 5);
    }

    #[test]
    fn test_violation_tracker() {
        let mut tracker = ViolationTracker::new();
        
        // Test consecutive count tracking
        assert_eq!(tracker.increment_consecutive_count("test_metric"), 1);
        assert_eq!(tracker.increment_consecutive_count("test_metric"), 2);
        
        tracker.reset_consecutive_count("test_metric");
        assert_eq!(tracker.increment_consecutive_count("test_metric"), 1);
        
        // Test value history
        tracker.update_value_history("test_metric", 1000, 100.0);
        tracker.update_value_history("test_metric", 2000, 200.0);
        
        let history = tracker.value_history.get("test_metric").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history.front().unwrap().1, 100.0);
        assert_eq!(history.back().unwrap().1, 200.0);
    }

    #[test]
    fn test_window_data_creation() {
        let mut tracker = ViolationTracker::new();
        let config = ThresholdConfig::default();
        
        // Add some test data
        tracker.update_value_history("test_metric", 1000, 100.0);
        tracker.update_value_history("test_metric", 2000, 200.0);
        tracker.update_value_history("test_metric", 3000, 150.0);
        
        let window_data = ThresholdMonitor::create_window_data(&config, &tracker);
        
        assert_eq!(window_data.sample_count, 3);
        assert_eq!(window_data.average, 150.0);
        assert_eq!(window_data.minimum, 100.0);
        assert_eq!(window_data.maximum, 200.0);
    }
}