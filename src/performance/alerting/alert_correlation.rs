use super::alert_manager::{Alert, AlertSeverity};
use super::timing::now_nanos;
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Alert correlation engine for reducing alert noise and identifying patterns
/// Groups related alerts and detects cascading failures
pub struct AlertCorrelator {
    /// Correlation rules
    rules: std::sync::RwLock<Vec<CorrelationRule>>,
    
    /// Active correlation groups
    active_groups: std::sync::RwLock<HashMap<String, CorrelationGroup>>,
    
    /// Correlation history
    history: std::sync::Mutex<VecDeque<CorrelatedAlert>>,
    
    /// Configuration
    config: CorrelationConfig,
}

/// Correlation rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationRule {
    /// Rule identifier
    pub id: String,
    
    /// Rule name
    pub name: String,
    
    /// Rule type
    pub rule_type: CorrelationRuleType,
    
    /// Metrics to correlate
    pub metrics: Vec<String>,
    
    /// Time window for correlation (nanoseconds)
    pub time_window_ns: u64,
    
    /// Minimum alerts required for correlation
    pub min_alerts: u32,
    
    /// Maximum alerts in correlation group
    pub max_alerts: u32,
    
    /// Correlation threshold (0.0 to 1.0)
    pub correlation_threshold: f64,
    
    /// Whether rule is enabled
    pub enabled: bool,
    
    /// Rule priority (higher = more important)
    pub priority: u32,
    
    /// Custom correlation logic
    pub custom_logic: Option<String>,
}

/// Types of correlation rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationRuleType {
    /// Temporal correlation (alerts within time window)
    Temporal,
    
    /// Causal correlation (one alert causes another)
    Causal,
    
    /// Pattern-based correlation (similar patterns)
    Pattern,
    
    /// Threshold-based correlation (similar values)
    Threshold,
    
    /// Metric family correlation (related metrics)
    MetricFamily,
    
    /// Custom correlation logic
    Custom,
}

/// Correlation group containing related alerts
#[derive(Debug, Clone)]
pub struct CorrelationGroup {
    /// Group identifier
    pub id: String,
    
    /// Group name
    pub name: String,
    
    /// Alerts in this group
    pub alerts: Vec<Alert>,
    
    /// Root cause alert (if identified)
    pub root_cause: Option<Alert>,
    
    /// Group severity (highest among alerts)
    pub severity: AlertSeverity,
    
    /// Group creation timestamp
    pub created_at: u64,
    
    /// Last update timestamp
    pub updated_at: u64,
    
    /// Correlation score (0.0 to 1.0)
    pub correlation_score: f64,
    
    /// Applied correlation rules
    pub applied_rules: Vec<String>,
    
    /// Group status
    pub status: CorrelationStatus,
}

/// Correlation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationStatus {
    Active,
    Resolved,
    Suppressed,
    Escalated,
}

/// Correlated alert result
#[derive(Debug, Clone)]
pub struct CorrelatedAlert {
    /// Original alert
    pub alert: Alert,
    
    /// Correlation group ID (if correlated)
    pub group_id: Option<String>,
    
    /// Correlation confidence (0.0 to 1.0)
    pub confidence: f64,
    
    /// Applied correlation rules
    pub applied_rules: Vec<String>,
    
    /// Whether this alert was suppressed due to correlation
    pub suppressed: bool,
    
    /// Correlation timestamp
    pub correlated_at: u64,
}

/// Correlation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Maximum active correlation groups
    pub max_active_groups: usize,
    
    /// Correlation history size
    pub history_size: usize,
    
    /// Default correlation window (nanoseconds)
    pub default_window_ns: u64,
    
    /// Minimum correlation confidence
    pub min_confidence: f64,
    
    /// Enable automatic rule learning
    pub enable_learning: bool,
    
    /// Group timeout (nanoseconds)
    pub group_timeout_ns: u64,
}

impl AlertCorrelator {
    /// Create a new alert correlator
    pub fn new() -> Self {
        Self {
            rules: std::sync::RwLock::new(Vec::new()),
            active_groups: std::sync::RwLock::new(HashMap::new()),
            history: std::sync::Mutex::new(VecDeque::new()),
            config: CorrelationConfig::default(),
        }
    }

    /// Create correlator with custom configuration
    pub fn with_config(config: CorrelationConfig) -> Self {
        Self {
            rules: std::sync::RwLock::new(Vec::new()),
            active_groups: std::sync::RwLock::new(HashMap::new()),
            history: std::sync::Mutex::new(VecDeque::new()),
            config,
        }
    }

    /// Add correlation rule
    pub fn add_rule(&self, rule: CorrelationRule) {
        let mut rules = self.rules.write().unwrap();
        rules.push(rule);
        
        // Sort by priority (highest first)
        rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove correlation rule
    pub fn remove_rule(&self, rule_id: &str) -> bool {
        let mut rules = self.rules.write().unwrap();
        if let Some(pos) = rules.iter().position(|r| r.id == rule_id) {
            rules.remove(pos);
            true
        } else {
            false
        }
    }

    /// Update correlation rule
    pub fn update_rule(&self, rule: CorrelationRule) -> bool {
        let mut rules = self.rules.write().unwrap();
        if let Some(existing) = rules.iter_mut().find(|r| r.id == rule.id) {
            *existing = rule;
            rules.sort_by(|a, b| b.priority.cmp(&a.priority));
            true
        } else {
            false
        }
    }

    /// Get all correlation rules
    pub fn get_rules(&self) -> Vec<CorrelationRule> {
        let rules = self.rules.read().unwrap();
        rules.clone()
    }

    /// Correlate an alert with existing alerts
    pub fn correlate_alert(&self, alert: Alert) -> Alert {
        let correlation_start = now_nanos();
        
        // Find matching correlation groups
        let matching_groups = self.find_matching_groups(&alert);
        
        if let Some(group_id) = matching_groups.first() {
            // Add alert to existing group
            self.add_to_group(group_id, alert.clone());
            
            // Record correlation
            let correlated = CorrelatedAlert {
                alert: alert.clone(),
                group_id: Some(group_id.clone()),
                confidence: 0.8, // TODO: Calculate actual confidence
                applied_rules: vec![], // TODO: Track applied rules
                suppressed: false,
                correlated_at: correlation_start,
            };
            
            self.add_to_history(correlated);
            
            return alert;
        }
        
        // Check if alert should start a new correlation group
        if let Some(group) = self.create_correlation_group(&alert) {
            let mut active_groups = self.active_groups.write().unwrap();
            active_groups.insert(group.id.clone(), group);
        }
        
        // Record uncorrelated alert
        let correlated = CorrelatedAlert {
            alert: alert.clone(),
            group_id: None,
            confidence: 0.0,
            applied_rules: vec![],
            suppressed: false,
            correlated_at: correlation_start,
        };
        
        self.add_to_history(correlated);
        
        alert
    }

    /// Get active correlation groups
    pub fn get_active_groups(&self) -> Vec<CorrelationGroup> {
        let groups = self.active_groups.read().unwrap();
        groups.values().cloned().collect()
    }

    /// Get correlation group by ID
    pub fn get_group(&self, group_id: &str) -> Option<CorrelationGroup> {
        let groups = self.active_groups.read().unwrap();
        groups.get(group_id).cloned()
    }

    /// Resolve correlation group
    pub fn resolve_group(&self, group_id: &str) -> bool {
        let mut groups = self.active_groups.write().unwrap();
        if let Some(group) = groups.get_mut(group_id) {
            group.status = CorrelationStatus::Resolved;
            group.updated_at = now_nanos();
            true
        } else {
            false
        }
    }

    /// Suppress correlation group
    pub fn suppress_group(&self, group_id: &str) -> bool {
        let mut groups = self.active_groups.write().unwrap();
        if let Some(group) = groups.get_mut(group_id) {
            group.status = CorrelationStatus::Suppressed;
            group.updated_at = now_nanos();
            true
        } else {
            false
        }
    }

    /// Get correlation statistics
    pub fn get_stats(&self) -> CorrelationStats {
        let groups = self.active_groups.read().unwrap();
        let history = self.history.lock().unwrap();
        
        let active_groups = groups.len();
        let total_correlations = history.iter()
            .filter(|c| c.group_id.is_some())
            .count();
        let suppressed_alerts = history.iter()
            .filter(|c| c.suppressed)
            .count();
        
        let avg_confidence = if !history.is_empty() {
            history.iter().map(|c| c.confidence).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };

        CorrelationStats {
            active_groups,
            total_correlations,
            suppressed_alerts,
            avg_confidence,
            rules_count: self.rules.read().unwrap().len(),
            history_size: history.len(),
        }
    }

    /// Cleanup expired correlation groups
    pub fn cleanup_expired_groups(&self) {
        let current_time = now_nanos();
        let mut groups = self.active_groups.write().unwrap();
        
        groups.retain(|_, group| {
            let age = current_time - group.created_at;
            age < self.config.group_timeout_ns && group.status == CorrelationStatus::Active
        });
    }

    /// Find matching correlation groups for an alert
    fn find_matching_groups(&self, alert: &Alert) -> Vec<String> {
        let groups = self.active_groups.read().unwrap();
        let rules = self.rules.read().unwrap();
        let mut matching_groups = Vec::new();
        
        for (group_id, group) in groups.iter() {
            if group.status != CorrelationStatus::Active {
                continue;
            }
            
            // Check if alert matches any rule for this group
            for rule in rules.iter() {
                if !rule.enabled {
                    continue;
                }
                
                if self.check_rule_match(rule, alert, group) {
                    matching_groups.push(group_id.clone());
                    break;
                }
            }
        }
        
        matching_groups
    }

    /// Check if an alert matches a correlation rule for a group
    fn check_rule_match(&self, rule: &CorrelationRule, alert: &Alert, group: &CorrelationGroup) -> bool {
        match rule.rule_type {
            CorrelationRuleType::Temporal => {
                self.check_temporal_correlation(rule, alert, group)
            }
            CorrelationRuleType::MetricFamily => {
                self.check_metric_family_correlation(rule, alert, group)
            }
            CorrelationRuleType::Threshold => {
                self.check_threshold_correlation(rule, alert, group)
            }
            CorrelationRuleType::Pattern => {
                self.check_pattern_correlation(rule, alert, group)
            }
            CorrelationRuleType::Causal => {
                self.check_causal_correlation(rule, alert, group)
            }
            CorrelationRuleType::Custom => {
                self.check_custom_correlation(rule, alert, group)
            }
        }
    }

    /// Check temporal correlation
    fn check_temporal_correlation(&self, rule: &CorrelationRule, alert: &Alert, group: &CorrelationGroup) -> bool {
        let time_diff = alert.created_at.saturating_sub(group.created_at);
        time_diff <= rule.time_window_ns
    }

    /// Check metric family correlation
    fn check_metric_family_correlation(&self, rule: &CorrelationRule, alert: &Alert, group: &CorrelationGroup) -> bool {
        // Check if alert metric is in the rule's metric list
        if !rule.metrics.contains(&alert.metric_name) {
            return false;
        }
        
        // Check if any group alert has a related metric
        group.alerts.iter().any(|group_alert| {
            rule.metrics.contains(&group_alert.metric_name)
        })
    }

    /// Check threshold correlation
    fn check_threshold_correlation(&self, rule: &CorrelationRule, alert: &Alert, group: &CorrelationGroup) -> bool {
        // Find alerts with similar threshold violations
        group.alerts.iter().any(|group_alert| {
            let value_diff = (alert.current_value - group_alert.current_value).abs();
            let threshold_diff = (alert.threshold_value - group_alert.threshold_value).abs();
            
            let value_similarity = 1.0 - (value_diff / alert.current_value.max(group_alert.current_value));
            let threshold_similarity = 1.0 - (threshold_diff / alert.threshold_value.max(group_alert.threshold_value));
            
            let overall_similarity = (value_similarity + threshold_similarity) / 2.0;
            overall_similarity >= rule.correlation_threshold
        })
    }

    /// Check pattern correlation
    fn check_pattern_correlation(&self, _rule: &CorrelationRule, _alert: &Alert, _group: &CorrelationGroup) -> bool {
        // TODO: Implement pattern matching logic
        false
    }

    /// Check causal correlation
    fn check_causal_correlation(&self, _rule: &CorrelationRule, _alert: &Alert, _group: &CorrelationGroup) -> bool {
        // TODO: Implement causal relationship detection
        false
    }

    /// Check custom correlation
    fn check_custom_correlation(&self, _rule: &CorrelationRule, _alert: &Alert, _group: &CorrelationGroup) -> bool {
        // TODO: Implement custom correlation logic evaluation
        false
    }

    /// Create a new correlation group for an alert
    fn create_correlation_group(&self, alert: &Alert) -> Option<CorrelationGroup> {
        let rules = self.rules.read().unwrap();
        
        // Check if any rule suggests creating a group for this alert
        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }
            
            if rule.metrics.contains(&alert.metric_name) {
                let group_id = format!("group_{}_{}", alert.metric_name, now_nanos());
                
                return Some(CorrelationGroup {
                    id: group_id,
                    name: format!("Correlation group for {}", alert.metric_name),
                    alerts: vec![alert.clone()],
                    root_cause: Some(alert.clone()),
                    severity: alert.severity,
                    created_at: now_nanos(),
                    updated_at: now_nanos(),
                    correlation_score: 1.0,
                    applied_rules: vec![rule.id.clone()],
                    status: CorrelationStatus::Active,
                });
            }
        }
        
        None
    }

    /// Add alert to existing correlation group
    fn add_to_group(&self, group_id: &str, alert: Alert) {
        let mut groups = self.active_groups.write().unwrap();
        
        if let Some(group) = groups.get_mut(group_id) {
            group.alerts.push(alert);
            group.updated_at = now_nanos();
            
            // Update group severity to highest among alerts
            let max_severity = group.alerts.iter()
                .map(|a| a.severity)
                .max()
                .unwrap_or(AlertSeverity::Info);
            group.severity = max_severity;
        }
    }

    /// Add correlated alert to history
    fn add_to_history(&self, correlated: CorrelatedAlert) {
        let mut history = self.history.lock().unwrap();
        
        history.push_back(correlated);
        
        // Maintain history size limit
        while history.len() > self.config.history_size {
            history.pop_front();
        }
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            max_active_groups: 1000,
            history_size: 10000,
            default_window_ns: 300_000_000_000, // 5 minutes
            min_confidence: 0.5,
            enable_learning: false,
            group_timeout_ns: 3600_000_000_000, // 1 hour
        }
    }
}

/// Correlation statistics
#[derive(Debug, Clone)]
pub struct CorrelationStats {
    pub active_groups: usize,
    pub total_correlations: usize,
    pub suppressed_alerts: usize,
    pub avg_confidence: f64,
    pub rules_count: usize,
    pub history_size: usize,
}

/// Predefined correlation rules for common scenarios
pub struct CommonCorrelationRules;

impl CommonCorrelationRules {
    /// Create latency correlation rule
    pub fn latency_correlation() -> CorrelationRule {
        CorrelationRule {
            id: "latency_correlation".to_string(),
            name: "Latency Correlation".to_string(),
            rule_type: CorrelationRuleType::MetricFamily,
            metrics: vec![
                "order_processing_latency".to_string(),
                "trade_execution_latency".to_string(),
                "network_latency".to_string(),
            ],
            time_window_ns: 60_000_000_000, // 1 minute
            min_alerts: 2,
            max_alerts: 10,
            correlation_threshold: 0.7,
            enabled: true,
            priority: 100,
            custom_logic: None,
        }
    }

    /// Create throughput correlation rule
    pub fn throughput_correlation() -> CorrelationRule {
        CorrelationRule {
            id: "throughput_correlation".to_string(),
            name: "Throughput Correlation".to_string(),
            rule_type: CorrelationRuleType::MetricFamily,
            metrics: vec![
                "orders_per_second".to_string(),
                "trades_per_second".to_string(),
                "messages_per_second".to_string(),
            ],
            time_window_ns: 120_000_000_000, // 2 minutes
            min_alerts: 2,
            max_alerts: 5,
            correlation_threshold: 0.8,
            enabled: true,
            priority: 90,
            custom_logic: None,
        }
    }

    /// Create resource correlation rule
    pub fn resource_correlation() -> CorrelationRule {
        CorrelationRule {
            id: "resource_correlation".to_string(),
            name: "Resource Correlation".to_string(),
            rule_type: CorrelationRuleType::MetricFamily,
            metrics: vec![
                "cpu_utilization".to_string(),
                "memory_usage".to_string(),
                "network_utilization".to_string(),
            ],
            time_window_ns: 180_000_000_000, // 3 minutes
            min_alerts: 2,
            max_alerts: 8,
            correlation_threshold: 0.6,
            enabled: true,
            priority: 80,
            custom_logic: None,
        }
    }

    /// Get all common correlation rules
    pub fn all_rules() -> Vec<CorrelationRule> {
        vec![
            Self::latency_correlation(),
            Self::throughput_correlation(),
            Self::resource_correlation(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::alert_manager::AlertStatus;

    fn create_test_alert(name: &str, metric: &str, value: f64, threshold: f64, severity: AlertSeverity) -> Alert {
        Alert {
            id: format!("alert_{}", name),
            name: name.to_string(),
            severity,
            status: AlertStatus::Active,
            message: format!("Test alert for {}", metric),
            metric_name: metric.to_string(),
            current_value: value,
            threshold_value: threshold,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            resolved_at: 0,
            context: HashMap::new(),
            fire_count: 1,
            duration_ns: 0,
        }
    }

    #[test]
    fn test_correlator_creation() {
        let correlator = AlertCorrelator::new();
        let stats = correlator.get_stats();
        
        assert_eq!(stats.active_groups, 0);
        assert_eq!(stats.rules_count, 0);
    }

    #[test]
    fn test_rule_management() {
        let correlator = AlertCorrelator::new();
        let rule = CommonCorrelationRules::latency_correlation();
        
        correlator.add_rule(rule.clone());
        
        let rules = correlator.get_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, rule.id);
        
        assert!(correlator.remove_rule(&rule.id));
        assert_eq!(correlator.get_rules().len(), 0);
    }

    #[test]
    fn test_alert_correlation() {
        let correlator = AlertCorrelator::new();
        correlator.add_rule(CommonCorrelationRules::latency_correlation());
        
        let alert1 = create_test_alert("high_latency_1", "order_processing_latency", 2000.0, 1000.0, AlertSeverity::Warning);
        let alert2 = create_test_alert("high_latency_2", "trade_execution_latency", 1500.0, 1000.0, AlertSeverity::Warning);
        
        // First alert should create a group
        let correlated1 = correlator.correlate_alert(alert1);
        assert_eq!(correlated1.name, "high_latency_1");
        
        // Second alert should join the group
        let correlated2 = correlator.correlate_alert(alert2);
        assert_eq!(correlated2.name, "high_latency_2");
        
        let stats = correlator.get_stats();
        assert_eq!(stats.active_groups, 1);
    }

    #[test]
    fn test_temporal_correlation() {
        let correlator = AlertCorrelator::new();
        let rule = CorrelationRule {
            id: "temporal_test".to_string(),
            name: "Temporal Test".to_string(),
            rule_type: CorrelationRuleType::Temporal,
            metrics: vec!["test_metric".to_string()],
            time_window_ns: 60_000_000_000, // 1 minute
            min_alerts: 1,
            max_alerts: 10,
            correlation_threshold: 0.5,
            enabled: true,
            priority: 100,
            custom_logic: None,
        };
        
        correlator.add_rule(rule);
        
        let alert = create_test_alert("temporal_test", "test_metric", 100.0, 50.0, AlertSeverity::Warning);
        let group = CorrelationGroup {
            id: "test_group".to_string(),
            name: "Test Group".to_string(),
            alerts: vec![],
            root_cause: None,
            severity: AlertSeverity::Warning,
            created_at: now_nanos() - 30_000_000_000, // 30 seconds ago
            updated_at: now_nanos(),
            correlation_score: 1.0,
            applied_rules: vec![],
            status: CorrelationStatus::Active,
        };
        
        let rules = correlator.get_rules();
        assert!(correlator.check_temporal_correlation(&rules[0], &alert, &group));
    }

    #[test]
    fn test_metric_family_correlation() {
        let correlator = AlertCorrelator::new();
        let rule = CommonCorrelationRules::latency_correlation();
        
        let alert = create_test_alert("latency_test", "order_processing_latency", 2000.0, 1000.0, AlertSeverity::Warning);
        let group_alert = create_test_alert("existing_alert", "trade_execution_latency", 1800.0, 1000.0, AlertSeverity::Warning);
        
        let group = CorrelationGroup {
            id: "test_group".to_string(),
            name: "Test Group".to_string(),
            alerts: vec![group_alert],
            root_cause: None,
            severity: AlertSeverity::Warning,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            correlation_score: 1.0,
            applied_rules: vec![],
            status: CorrelationStatus::Active,
        };
        
        assert!(correlator.check_metric_family_correlation(&rule, &alert, &group));
    }

    #[test]
    fn test_threshold_correlation() {
        let correlator = AlertCorrelator::new();
        let rule = CorrelationRule {
            id: "threshold_test".to_string(),
            name: "Threshold Test".to_string(),
            rule_type: CorrelationRuleType::Threshold,
            metrics: vec!["test_metric".to_string()],
            time_window_ns: 60_000_000_000,
            min_alerts: 1,
            max_alerts: 10,
            correlation_threshold: 0.8,
            enabled: true,
            priority: 100,
            custom_logic: None,
        };
        
        let alert = create_test_alert("threshold_test", "test_metric", 100.0, 50.0, AlertSeverity::Warning);
        let group_alert = create_test_alert("existing_alert", "test_metric", 105.0, 52.0, AlertSeverity::Warning);
        
        let group = CorrelationGroup {
            id: "test_group".to_string(),
            name: "Test Group".to_string(),
            alerts: vec![group_alert],
            root_cause: None,
            severity: AlertSeverity::Warning,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            correlation_score: 1.0,
            applied_rules: vec![],
            status: CorrelationStatus::Active,
        };
        
        assert!(correlator.check_threshold_correlation(&rule, &alert, &group));
    }

    #[test]
    fn test_group_management() {
        let correlator = AlertCorrelator::new();
        correlator.add_rule(CommonCorrelationRules::latency_correlation());
        
        let alert = create_test_alert("test_alert", "order_processing_latency", 2000.0, 1000.0, AlertSeverity::Critical);
        correlator.correlate_alert(alert);
        
        let groups = correlator.get_active_groups();
        assert_eq!(groups.len(), 1);
        
        let group_id = &groups[0].id;
        assert!(correlator.resolve_group(group_id));
        
        let resolved_group = correlator.get_group(group_id).unwrap();
        assert_eq!(resolved_group.status, CorrelationStatus::Resolved);
    }

    #[test]
    fn test_common_correlation_rules() {
        let rules = CommonCorrelationRules::all_rules();
        assert_eq!(rules.len(), 3);
        
        let latency_rule = &rules[0];
        assert_eq!(latency_rule.id, "latency_correlation");
        assert_eq!(latency_rule.rule_type, CorrelationRuleType::MetricFamily);
        assert!(latency_rule.metrics.contains(&"order_processing_latency".to_string()));
    }

    #[test]
    fn test_correlation_stats() {
        let correlator = AlertCorrelator::new();
        correlator.add_rule(CommonCorrelationRules::latency_correlation());
        
        let alert1 = create_test_alert("alert1", "order_processing_latency", 2000.0, 1000.0, AlertSeverity::Warning);
        let alert2 = create_test_alert("alert2", "trade_execution_latency", 1500.0, 1000.0, AlertSeverity::Critical);
        
        correlator.correlate_alert(alert1);
        correlator.correlate_alert(alert2);
        
        let stats = correlator.get_stats();
        assert_eq!(stats.active_groups, 1);
        assert_eq!(stats.rules_count, 1);
        assert!(stats.avg_confidence > 0.0);
    }
}