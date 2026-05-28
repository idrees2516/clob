use super::timing::{now_nanos, LatencyHistogram, PerformanceMetrics};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// High-performance alert manager for real-time performance monitoring
/// Designed for sub-second alert detection and processing
pub struct AlertManager {
    /// Active alerts
    active_alerts: std::sync::RwLock<HashMap<String, Alert>>,
    
    /// Alert history (ring buffer for memory efficiency)
    alert_history: std::sync::Mutex<VecDeque<Alert>>,
    
    /// Alert statistics
    stats: AlertStats,
    
    /// Alert configuration
    config: AlertConfig,
    
    /// Notification channels
    notification_channels: Vec<Box<dyn NotificationChannel>>,
    
    /// Alert correlation engine
    correlator: Arc<AlertCorrelator>,
    
    /// Escalation engine
    escalation: Arc<EscalationEngine>,
}

/// Individual alert representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert identifier
    pub id: String,
    
    /// Alert name/type
    pub name: String,
    
    /// Alert severity level
    pub severity: AlertSeverity,
    
    /// Current alert status
    pub status: AlertStatus,
    
    /// Alert message
    pub message: String,
    
    /// Metric that triggered the alert
    pub metric_name: String,
    
    /// Current metric value
    pub current_value: f64,
    
    /// Threshold that was violated
    pub threshold_value: f64,
    
    /// Alert creation timestamp (nanoseconds)
    pub created_at: u64,
    
    /// Last update timestamp (nanoseconds)
    pub updated_at: u64,
    
    /// Alert resolution timestamp (nanoseconds, 0 if not resolved)
    pub resolved_at: u64,
    
    /// Additional context data
    pub context: HashMap<String, String>,
    
    /// Number of times this alert has fired
    pub fire_count: u32,
    
    /// Alert duration (nanoseconds)
    pub duration_ns: u64,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info = 1,
    Warning = 2,
    Critical = 3,
    Emergency = 4,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
}

/// Alert manager statistics
#[repr(align(64))]
pub struct AlertStats {
    /// Total alerts created
    pub total_alerts: AtomicU64,
    
    /// Active alert count
    pub active_count: AtomicUsize,
    
    /// Alerts by severity
    pub info_count: AtomicU64,
    pub warning_count: AtomicU64,
    pub critical_count: AtomicU64,
    pub emergency_count: AtomicU64,
    
    /// Alert processing latency (nanoseconds)
    pub processing_latency: LatencyHistogram,
    
    /// Notification latency (nanoseconds)
    pub notification_latency: LatencyHistogram,
    
    /// Last alert timestamp
    pub last_alert_time: AtomicU64,
    
    /// Alert rate (alerts per second)
    pub alert_rate: AtomicU64,
}

/// Alert manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Maximum number of active alerts
    pub max_active_alerts: usize,
    
    /// Alert history size
    pub history_size: usize,
    
    /// Alert processing timeout (nanoseconds)
    pub processing_timeout_ns: u64,
    
    /// Notification timeout (nanoseconds)
    pub notification_timeout_ns: u64,
    
    /// Alert deduplication window (nanoseconds)
    pub deduplication_window_ns: u64,
    
    /// Enable alert correlation
    pub enable_correlation: bool,
    
    /// Enable alert escalation
    pub enable_escalation: bool,
    
    /// Minimum severity for notifications
    pub min_notification_severity: AlertSeverity,
}

/// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    /// Send alert notification
    fn send_alert(&self, alert: &Alert) -> Result<(), NotificationError>;
    
    /// Get channel name
    fn name(&self) -> &str;
    
    /// Check if channel is healthy
    fn is_healthy(&self) -> bool;
}

/// Notification errors
#[derive(Debug, Clone)]
pub enum NotificationError {
    ChannelUnavailable,
    SendTimeout,
    InvalidAlert,
    ConfigurationError,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(config: AlertConfig) -> Self {
        Self {
            active_alerts: std::sync::RwLock::new(HashMap::new()),
            alert_history: std::sync::Mutex::new(VecDeque::new()),
            stats: AlertStats::new(),
            config,
            notification_channels: Vec::new(),
            correlator: Arc::new(AlertCorrelator::new()),
            escalation: Arc::new(EscalationEngine::new()),
        }
    }

    /// Create alert manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(AlertConfig::default())
    }

    /// Add notification channel
    pub fn add_notification_channel(&mut self, channel: Box<dyn NotificationChannel>) {
        self.notification_channels.push(channel);
    }

    /// Fire a new alert
    pub fn fire_alert(&self, alert_name: String, metric_name: String, current_value: f64, threshold_value: f64, severity: AlertSeverity, message: String) -> Result<String, AlertError> {
        let processing_start = now_nanos();
        
        // Generate alert ID
        let alert_id = self.generate_alert_id(&alert_name, &metric_name);
        
        // Check for deduplication
        if self.is_duplicate_alert(&alert_id, current_value) {
            return Ok(alert_id);
        }

        // Create alert
        let mut alert = Alert {
            id: alert_id.clone(),
            name: alert_name,
            severity,
            status: AlertStatus::Active,
            message,
            metric_name,
            current_value,
            threshold_value,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            resolved_at: 0,
            context: HashMap::new(),
            fire_count: 1,
            duration_ns: 0,
        };

        // Check if this is a repeat alert
        if let Some(existing_alert) = self.get_active_alert(&alert_id) {
            alert.fire_count = existing_alert.fire_count + 1;
            alert.created_at = existing_alert.created_at;
        }

        // Apply correlation rules
        if self.config.enable_correlation {
            alert = self.correlator.correlate_alert(alert);
        }

        // Store alert
        self.store_alert(alert.clone())?;
        
        // Send notifications
        if severity >= self.config.min_notification_severity {
            self.send_notifications(&alert);
        }

        // Apply escalation if enabled
        if self.config.enable_escalation {
            self.escalation.process_alert(&alert);
        }

        // Update statistics
        self.update_stats(&alert, processing_start);

        Ok(alert_id)
    }

    /// Resolve an alert
    pub fn resolve_alert(&self, alert_id: &str) -> Result<(), AlertError> {
        let mut active_alerts = self.active_alerts.write().unwrap();
        
        if let Some(mut alert) = active_alerts.remove(alert_id) {
            alert.status = AlertStatus::Resolved;
            alert.resolved_at = now_nanos();
            alert.updated_at = now_nanos();
            alert.duration_ns = alert.resolved_at - alert.created_at;
            
            // Add to history
            self.add_to_history(alert);
            
            // Update statistics
            self.stats.active_count.fetch_sub(1, Ordering::Relaxed);
            
            Ok(())
        } else {
            Err(AlertError::AlertNotFound)
        }
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, alert_id: &str) -> Result<(), AlertError> {
        let mut active_alerts = self.active_alerts.write().unwrap();
        
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.updated_at = now_nanos();
            Ok(())
        } else {
            Err(AlertError::AlertNotFound)
        }
    }

    /// Suppress an alert
    pub fn suppress_alert(&self, alert_id: &str) -> Result<(), AlertError> {
        let mut active_alerts = self.active_alerts.write().unwrap();
        
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Suppressed;
            alert.updated_at = now_nanos();
            Ok(())
        } else {
            Err(AlertError::AlertNotFound)
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        let active_alerts = self.active_alerts.read().unwrap();
        active_alerts.values().cloned().collect()
    }

    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        let active_alerts = self.active_alerts.read().unwrap();
        active_alerts.values()
            .filter(|alert| alert.severity == severity)
            .cloned()
            .collect()
    }

    /// Get alert history
    pub fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.lock().unwrap();
        let limit = limit.unwrap_or(history.len());
        history.iter().rev().take(limit).cloned().collect()
    }

    /// Get alert statistics
    pub fn get_stats(&self) -> AlertStatsSnapshot {
        AlertStatsSnapshot {
            total_alerts: self.stats.total_alerts.load(Ordering::Acquire),
            active_count: self.stats.active_count.load(Ordering::Acquire),
            info_count: self.stats.info_count.load(Ordering::Acquire),
            warning_count: self.stats.warning_count.load(Ordering::Acquire),
            critical_count: self.stats.critical_count.load(Ordering::Acquire),
            emergency_count: self.stats.emergency_count.load(Ordering::Acquire),
            processing_latency: self.stats.processing_latency.get_stats(),
            notification_latency: self.stats.notification_latency.get_stats(),
            last_alert_time: self.stats.last_alert_time.load(Ordering::Acquire),
            alert_rate: self.stats.alert_rate.load(Ordering::Acquire),
        }
    }

    /// Clear all alerts
    pub fn clear_all_alerts(&self) {
        let mut active_alerts = self.active_alerts.write().unwrap();
        active_alerts.clear();
        self.stats.active_count.store(0, Ordering::Release);
    }

    /// Health check
    pub fn health_check(&self) -> AlertManagerHealth {
        let active_count = self.stats.active_count.load(Ordering::Acquire);
        let processing_stats = self.stats.processing_latency.get_stats();
        
        let healthy_channels = self.notification_channels.iter()
            .filter(|channel| channel.is_healthy())
            .count();

        AlertManagerHealth {
            is_healthy: active_count < self.config.max_active_alerts && healthy_channels > 0,
            active_alert_count: active_count,
            healthy_notification_channels: healthy_channels,
            total_notification_channels: self.notification_channels.len(),
            avg_processing_latency_ns: processing_stats.average as u64,
            last_alert_age_ns: now_nanos().saturating_sub(self.stats.last_alert_time.load(Ordering::Acquire)),
        }
    }

    /// Generate unique alert ID
    fn generate_alert_id(&self, alert_name: &str, metric_name: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        alert_name.hash(&mut hasher);
        metric_name.hash(&mut hasher);
        now_nanos().hash(&mut hasher);
        
        format!("alert_{:x}", hasher.finish())
    }

    /// Check for duplicate alerts
    fn is_duplicate_alert(&self, alert_id: &str, current_value: f64) -> bool {
        let active_alerts = self.active_alerts.read().unwrap();
        
        if let Some(existing_alert) = active_alerts.get(alert_id) {
            let time_diff = now_nanos() - existing_alert.updated_at;
            let value_diff = (current_value - existing_alert.current_value).abs();
            
            // Consider duplicate if within deduplication window and similar value
            time_diff < self.config.deduplication_window_ns && value_diff < 0.01
        } else {
            false
        }
    }

    /// Get active alert by ID
    fn get_active_alert(&self, alert_id: &str) -> Option<Alert> {
        let active_alerts = self.active_alerts.read().unwrap();
        active_alerts.get(alert_id).cloned()
    }

    /// Store alert in active alerts
    fn store_alert(&self, alert: Alert) -> Result<(), AlertError> {
        let mut active_alerts = self.active_alerts.write().unwrap();
        
        // Check capacity
        if active_alerts.len() >= self.config.max_active_alerts {
            return Err(AlertError::CapacityExceeded);
        }

        active_alerts.insert(alert.id.clone(), alert);
        self.stats.active_count.store(active_alerts.len(), Ordering::Release);
        
        Ok(())
    }

    /// Send notifications for alert
    fn send_notifications(&self, alert: &Alert) {
        let notification_start = now_nanos();
        
        for channel in &self.notification_channels {
            if channel.is_healthy() {
                if let Err(e) = channel.send_alert(alert) {
                    eprintln!("Failed to send alert notification via {}: {:?}", channel.name(), e);
                }
            }
        }
        
        let notification_latency = now_nanos() - notification_start;
        self.stats.notification_latency.record(notification_latency);
    }

    /// Add alert to history
    fn add_to_history(&self, alert: Alert) {
        let mut history = self.alert_history.lock().unwrap();
        
        history.push_back(alert);
        
        // Maintain history size limit
        while history.len() > self.config.history_size {
            history.pop_front();
        }
    }

    /// Update alert statistics
    fn update_stats(&self, alert: &Alert, processing_start: u64) {
        self.stats.total_alerts.fetch_add(1, Ordering::Relaxed);
        self.stats.last_alert_time.store(alert.created_at, Ordering::Release);
        
        // Update severity counters
        match alert.severity {
            AlertSeverity::Info => self.stats.info_count.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Warning => self.stats.warning_count.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Critical => self.stats.critical_count.fetch_add(1, Ordering::Relaxed),
            AlertSeverity::Emergency => self.stats.emergency_count.fetch_add(1, Ordering::Relaxed),
        };

        // Record processing latency
        let processing_latency = now_nanos() - processing_start;
        self.stats.processing_latency.record(processing_latency);
    }
}

impl AlertStats {
    fn new() -> Self {
        Self {
            total_alerts: AtomicU64::new(0),
            active_count: AtomicUsize::new(0),
            info_count: AtomicU64::new(0),
            warning_count: AtomicU64::new(0),
            critical_count: AtomicU64::new(0),
            emergency_count: AtomicU64::new(0),
            processing_latency: LatencyHistogram::for_trading(),
            notification_latency: LatencyHistogram::for_trading(),
            last_alert_time: AtomicU64::new(0),
            alert_rate: AtomicU64::new(0),
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            max_active_alerts: 10000,
            history_size: 100000,
            processing_timeout_ns: 1_000_000, // 1ms
            notification_timeout_ns: 10_000_000, // 10ms
            deduplication_window_ns: 60_000_000_000, // 60 seconds
            enable_correlation: true,
            enable_escalation: true,
            min_notification_severity: AlertSeverity::Warning,
        }
    }
}

/// Alert statistics snapshot
#[derive(Debug, Clone)]
pub struct AlertStatsSnapshot {
    pub total_alerts: u64,
    pub active_count: usize,
    pub info_count: u64,
    pub warning_count: u64,
    pub critical_count: u64,
    pub emergency_count: u64,
    pub processing_latency: super::timing::HistogramStats,
    pub notification_latency: super::timing::HistogramStats,
    pub last_alert_time: u64,
    pub alert_rate: u64,
}

/// Alert manager health status
#[derive(Debug, Clone)]
pub struct AlertManagerHealth {
    pub is_healthy: bool,
    pub active_alert_count: usize,
    pub healthy_notification_channels: usize,
    pub total_notification_channels: usize,
    pub avg_processing_latency_ns: u64,
    pub last_alert_age_ns: u64,
}

/// Alert-related errors
#[derive(Debug, Clone)]
pub enum AlertError {
    AlertNotFound,
    CapacityExceeded,
    InvalidConfiguration,
    ProcessingTimeout,
    NotificationFailed,
}

impl std::fmt::Display for AlertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertError::AlertNotFound => write!(f, "Alert not found"),
            AlertError::CapacityExceeded => write!(f, "Alert capacity exceeded"),
            AlertError::InvalidConfiguration => write!(f, "Invalid alert configuration"),
            AlertError::ProcessingTimeout => write!(f, "Alert processing timeout"),
            AlertError::NotificationFailed => write!(f, "Alert notification failed"),
        }
    }
}

impl std::error::Error for AlertError {}

/// Global alert manager instance
static GLOBAL_ALERT_MANAGER: std::sync::OnceLock<Arc<std::sync::Mutex<AlertManager>>> = std::sync::OnceLock::new();

/// Get global alert manager
pub fn global_alert_manager() -> &'static Arc<std::sync::Mutex<AlertManager>> {
    GLOBAL_ALERT_MANAGER.get_or_init(|| {
        Arc::new(std::sync::Mutex::new(AlertManager::with_defaults()))
    })
}

/// Convenience function to fire an alert
pub fn fire_alert(alert_name: String, metric_name: String, current_value: f64, threshold_value: f64, severity: AlertSeverity, message: String) -> Result<String, AlertError> {
    let manager = global_alert_manager();
    let manager = manager.lock().unwrap();
    manager.fire_alert(alert_name, metric_name, current_value, threshold_value, severity, message)
}

/// Convenience function to resolve an alert
pub fn resolve_alert(alert_id: &str) -> Result<(), AlertError> {
    let manager = global_alert_manager();
    let manager = manager.lock().unwrap();
    manager.resolve_alert(alert_id)
}

// Forward declarations to avoid circular imports
pub struct AlertCorrelator;
pub struct EscalationEngine;

impl AlertCorrelator {
    pub fn new() -> Self { Self }
    pub fn correlate_alert(&self, alert: Alert) -> Alert { alert }
}

impl EscalationEngine {
    pub fn new() -> Self { Self }
    pub fn process_alert(&self, _alert: &Alert) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let manager = AlertManager::with_defaults();
        
        let alert_id = manager.fire_alert(
            "high_latency".to_string(),
            "order_processing_latency".to_string(),
            2000.0,
            1000.0,
            AlertSeverity::Warning,
            "Order processing latency exceeded threshold".to_string(),
        ).unwrap();
        
        assert!(!alert_id.is_empty());
        
        let active_alerts = manager.get_active_alerts();
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_resolution() {
        let manager = AlertManager::with_defaults();
        
        let alert_id = manager.fire_alert(
            "test_alert".to_string(),
            "test_metric".to_string(),
            100.0,
            50.0,
            AlertSeverity::Critical,
            "Test alert".to_string(),
        ).unwrap();
        
        assert_eq!(manager.get_active_alerts().len(), 1);
        
        manager.resolve_alert(&alert_id).unwrap();
        assert_eq!(manager.get_active_alerts().len(), 0);
    }

    #[test]
    fn test_alert_acknowledgment() {
        let manager = AlertManager::with_defaults();
        
        let alert_id = manager.fire_alert(
            "test_alert".to_string(),
            "test_metric".to_string(),
            100.0,
            50.0,
            AlertSeverity::Info,
            "Test alert".to_string(),
        ).unwrap();
        
        manager.acknowledge_alert(&alert_id).unwrap();
        
        let active_alerts = manager.get_active_alerts();
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].status, AlertStatus::Acknowledged);
    }

    #[test]
    fn test_alert_statistics() {
        let manager = AlertManager::with_defaults();
        
        // Fire alerts of different severities
        manager.fire_alert("info_alert".to_string(), "metric1".to_string(), 1.0, 0.5, AlertSeverity::Info, "Info".to_string()).unwrap();
        manager.fire_alert("warning_alert".to_string(), "metric2".to_string(), 2.0, 1.0, AlertSeverity::Warning, "Warning".to_string()).unwrap();
        manager.fire_alert("critical_alert".to_string(), "metric3".to_string(), 3.0, 2.0, AlertSeverity::Critical, "Critical".to_string()).unwrap();
        
        let stats = manager.get_stats();
        assert_eq!(stats.total_alerts, 3);
        assert_eq!(stats.active_count, 3);
        assert_eq!(stats.info_count, 1);
        assert_eq!(stats.warning_count, 1);
        assert_eq!(stats.critical_count, 1);
    }

    #[test]
    fn test_alert_filtering() {
        let manager = AlertManager::with_defaults();
        
        manager.fire_alert("warning1".to_string(), "metric1".to_string(), 1.0, 0.5, AlertSeverity::Warning, "Warning 1".to_string()).unwrap();
        manager.fire_alert("warning2".to_string(), "metric2".to_string(), 2.0, 1.0, AlertSeverity::Warning, "Warning 2".to_string()).unwrap();
        manager.fire_alert("critical1".to_string(), "metric3".to_string(), 3.0, 2.0, AlertSeverity::Critical, "Critical 1".to_string()).unwrap();
        
        let warning_alerts = manager.get_alerts_by_severity(AlertSeverity::Warning);
        let critical_alerts = manager.get_alerts_by_severity(AlertSeverity::Critical);
        
        assert_eq!(warning_alerts.len(), 2);
        assert_eq!(critical_alerts.len(), 1);
    }

    #[test]
    fn test_health_check() {
        let manager = AlertManager::with_defaults();
        
        let health = manager.health_check();
        assert!(health.is_healthy);
        assert_eq!(health.active_alert_count, 0);
    }

    #[test]
    fn test_alert_history() {
        let manager = AlertManager::with_defaults();
        
        let alert_id = manager.fire_alert(
            "test_alert".to_string(),
            "test_metric".to_string(),
            100.0,
            50.0,
            AlertSeverity::Warning,
            "Test alert".to_string(),
        ).unwrap();
        
        manager.resolve_alert(&alert_id).unwrap();
        
        let history = manager.get_alert_history(Some(10));
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, AlertStatus::Resolved);
    }

    #[test]
    fn test_global_alert_manager() {
        let result = fire_alert(
            "global_test".to_string(),
            "global_metric".to_string(),
            100.0,
            50.0,
            AlertSeverity::Critical,
            "Global test alert".to_string(),
        );
        
        assert!(result.is_ok());
        
        let alert_id = result.unwrap();
        let resolve_result = resolve_alert(&alert_id);
        assert!(resolve_result.is_ok());
    }
}