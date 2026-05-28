use super::alert_manager::{Alert, AlertSeverity};
use super::timing::now_nanos;
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Alert escalation engine for managing alert lifecycle and escalation policies
/// Automatically escalates alerts based on time, severity, and custom rules
pub struct EscalationEngine {
    /// Escalation policies
    policies: std::sync::RwLock<HashMap<String, EscalationPolicy>>,
    
    /// Active escalations
    active_escalations: std::sync::RwLock<HashMap<String, ActiveEscalation>>,
    
    /// Escalation history
    history: std::sync::Mutex<VecDeque<EscalationEvent>>,
    
    /// Configuration
    config: EscalationConfig,
}

/// Escalation policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    /// Policy identifier
    pub id: String,
    
    /// Policy name
    pub name: String,
    
    /// Escalation levels
    pub levels: Vec<EscalationLevel>,
    
    /// Conditions for applying this policy
    pub conditions: Vec<EscalationCondition>,
    
    /// Whether policy is enabled
    pub enabled: bool,
    
    /// Policy priority (higher = more important)
    pub priority: u32,
    
    /// Maximum escalation level
    pub max_level: u32,
    
    /// Auto-resolution timeout (nanoseconds)
    pub auto_resolve_timeout_ns: Option<u64>,
}

/// Escalation level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    /// Level number (0-based)
    pub level: u32,
    
    /// Level name
    pub name: String,
    
    /// Time to wait before escalating to this level (nanoseconds)
    pub escalation_delay_ns: u64,
    
    /// Actions to take at this level
    pub actions: Vec<EscalationAction>,
    
    /// Notification channels for this level
    pub notification_channels: Vec<String>,
    
    /// Whether to suppress lower-level notifications
    pub suppress_lower_levels: bool,
}

/// Escalation action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationAction {
    /// Send notification to specific channels
    Notify { channels: Vec<String> },
    
    /// Increase alert severity
    IncreaseSeverity { new_severity: AlertSeverity },
    
    /// Create incident ticket
    CreateIncident { system: String, priority: String },
    
    /// Execute custom script/command
    ExecuteScript { script: String, args: Vec<String> },
    
    /// Page on-call personnel
    PageOnCall { team: String, urgency: String },
    
    /// Auto-remediation action
    AutoRemediate { action: String, parameters: HashMap<String, String> },
    
    /// Suppress alert
    Suppress { reason: String },
    
    /// Custom action
    Custom { action_type: String, parameters: HashMap<String, String> },
}

/// Escalation condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCondition {
    /// Condition type
    pub condition_type: ConditionType,
    
    /// Condition parameters
    pub parameters: HashMap<String, String>,
    
    /// Whether condition must be met (AND) or can be met (OR)
    pub required: bool,
}

/// Types of escalation conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    /// Alert severity level
    Severity,
    
    /// Metric name pattern
    MetricPattern,
    
    /// Alert age
    AlertAge,
    
    /// Alert frequency
    AlertFrequency,
    
    /// Business hours
    BusinessHours,
    
    /// Custom condition
    Custom,
}

/// Active escalation tracking
#[derive(Debug, Clone)]
pub struct ActiveEscalation {
    /// Alert being escalated
    pub alert: Alert,
    
    /// Applied escalation policy
    pub policy_id: String,
    
    /// Current escalation level
    pub current_level: u32,
    
    /// Escalation start time
    pub started_at: u64,
    
    /// Last escalation time
    pub last_escalated_at: u64,
    
    /// Next escalation time
    pub next_escalation_at: u64,
    
    /// Escalation status
    pub status: EscalationStatus,
    
    /// Actions taken
    pub actions_taken: Vec<EscalationActionResult>,
    
    /// Escalation history
    pub level_history: Vec<EscalationLevelEvent>,
}

/// Escalation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationStatus {
    Active,
    Paused,
    Resolved,
    MaxLevelReached,
    PolicyViolation,
}

/// Result of an escalation action
#[derive(Debug, Clone)]
pub struct EscalationActionResult {
    /// Action that was executed
    pub action: EscalationAction,
    
    /// Execution timestamp
    pub executed_at: u64,
    
    /// Whether action succeeded
    pub success: bool,
    
    /// Error message if action failed
    pub error_message: Option<String>,
    
    /// Execution duration (nanoseconds)
    pub duration_ns: u64,
}

/// Escalation level event
#[derive(Debug, Clone)]
pub struct EscalationLevelEvent {
    /// Level that was reached
    pub level: u32,
    
    /// Timestamp when level was reached
    pub timestamp: u64,
    
    /// Actions executed at this level
    pub actions: Vec<EscalationActionResult>,
}

/// Escalation event for history
#[derive(Debug, Clone)]
pub struct EscalationEvent {
    /// Event type
    pub event_type: EscalationEventType,
    
    /// Alert ID
    pub alert_id: String,
    
    /// Policy ID
    pub policy_id: String,
    
    /// Escalation level
    pub level: u32,
    
    /// Event timestamp
    pub timestamp: u64,
    
    /// Additional context
    pub context: HashMap<String, String>,
}

/// Types of escalation events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EscalationEventType {
    Started,
    LevelReached,
    ActionExecuted,
    Paused,
    Resumed,
    Resolved,
    Failed,
}

/// Escalation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    /// Maximum active escalations
    pub max_active_escalations: usize,
    
    /// Escalation history size
    pub history_size: usize,
    
    /// Default escalation delay (nanoseconds)
    pub default_escalation_delay_ns: u64,
    
    /// Enable automatic escalation
    pub enable_auto_escalation: bool,
    
    /// Escalation processing interval (nanoseconds)
    pub processing_interval_ns: u64,
    
    /// Maximum escalation level
    pub max_escalation_level: u32,
}

impl EscalationEngine {
    /// Create a new escalation engine
    pub fn new() -> Self {
        Self {
            policies: std::sync::RwLock::new(HashMap::new()),
            active_escalations: std::sync::RwLock::new(HashMap::new()),
            history: std::sync::Mutex::new(VecDeque::new()),
            config: EscalationConfig::default(),
        }
    }

    /// Create escalation engine with custom configuration
    pub fn with_config(config: EscalationConfig) -> Self {
        Self {
            policies: std::sync::RwLock::new(HashMap::new()),
            active_escalations: std::sync::RwLock::new(HashMap::new()),
            history: std::sync::Mutex::new(VecDeque::new()),
            config,
        }
    }

    /// Add escalation policy
    pub fn add_policy(&self, policy: EscalationPolicy) {
        let mut policies = self.policies.write().unwrap();
        policies.insert(policy.id.clone(), policy);
    }

    /// Remove escalation policy
    pub fn remove_policy(&self, policy_id: &str) -> bool {
        let mut policies = self.policies.write().unwrap();
        policies.remove(policy_id).is_some()
    }

    /// Get escalation policy
    pub fn get_policy(&self, policy_id: &str) -> Option<EscalationPolicy> {
        let policies = self.policies.read().unwrap();
        policies.get(policy_id).cloned()
    }

    /// Get all escalation policies
    pub fn get_all_policies(&self) -> Vec<EscalationPolicy> {
        let policies = self.policies.read().unwrap();
        policies.values().cloned().collect()
    }

    /// Process alert for escalation
    pub fn process_alert(&self, alert: &Alert) -> Option<String> {
        // Find matching escalation policy
        let matching_policy = self.find_matching_policy(alert)?;
        
        // Check if alert is already being escalated
        let escalation_id = format!("escalation_{}_{}", alert.id, matching_policy.id);
        
        {
            let active_escalations = self.active_escalations.read().unwrap();
            if active_escalations.contains_key(&escalation_id) {
                return Some(escalation_id);
            }
        }
        
        // Start new escalation
        self.start_escalation(alert.clone(), matching_policy, escalation_id.clone());
        
        Some(escalation_id)
    }

    /// Process all active escalations
    pub fn process_escalations(&self) -> EscalationProcessResult {
        let current_time = now_nanos();
        let mut processed = 0;
        let mut escalated = 0;
        let mut resolved = 0;
        let mut failed = 0;
        
        let escalation_ids: Vec<String> = {
            let active_escalations = self.active_escalations.read().unwrap();
            active_escalations.keys().cloned().collect()
        };
        
        for escalation_id in escalation_ids {
            processed += 1;
            
            match self.process_single_escalation(&escalation_id, current_time) {
                EscalationResult::Escalated => escalated += 1,
                EscalationResult::Resolved => resolved += 1,
                EscalationResult::Failed => failed += 1,
                EscalationResult::NoAction => {}
            }
        }
        
        EscalationProcessResult {
            processed,
            escalated,
            resolved,
            failed,
        }
    }

    /// Get active escalations
    pub fn get_active_escalations(&self) -> Vec<ActiveEscalation> {
        let escalations = self.active_escalations.read().unwrap();
        escalations.values().cloned().collect()
    }

    /// Get escalation by ID
    pub fn get_escalation(&self, escalation_id: &str) -> Option<ActiveEscalation> {
        let escalations = self.active_escalations.read().unwrap();
        escalations.get(escalation_id).cloned()
    }

    /// Pause escalation
    pub fn pause_escalation(&self, escalation_id: &str) -> bool {
        let mut escalations = self.active_escalations.write().unwrap();
        
        if let Some(escalation) = escalations.get_mut(escalation_id) {
            escalation.status = EscalationStatus::Paused;
            
            self.record_event(EscalationEvent {
                event_type: EscalationEventType::Paused,
                alert_id: escalation.alert.id.clone(),
                policy_id: escalation.policy_id.clone(),
                level: escalation.current_level,
                timestamp: now_nanos(),
                context: HashMap::new(),
            });
            
            true
        } else {
            false
        }
    }

    /// Resume escalation
    pub fn resume_escalation(&self, escalation_id: &str) -> bool {
        let mut escalations = self.active_escalations.write().unwrap();
        
        if let Some(escalation) = escalations.get_mut(escalation_id) {
            escalation.status = EscalationStatus::Active;
            
            self.record_event(EscalationEvent {
                event_type: EscalationEventType::Resumed,
                alert_id: escalation.alert.id.clone(),
                policy_id: escalation.policy_id.clone(),
                level: escalation.current_level,
                timestamp: now_nanos(),
                context: HashMap::new(),
            });
            
            true
        } else {
            false
        }
    }

    /// Resolve escalation
    pub fn resolve_escalation(&self, escalation_id: &str) -> bool {
        let mut escalations = self.active_escalations.write().unwrap();
        
        if let Some(mut escalation) = escalations.remove(escalation_id) {
            escalation.status = EscalationStatus::Resolved;
            
            self.record_event(EscalationEvent {
                event_type: EscalationEventType::Resolved,
                alert_id: escalation.alert.id.clone(),
                policy_id: escalation.policy_id.clone(),
                level: escalation.current_level,
                timestamp: now_nanos(),
                context: HashMap::new(),
            });
            
            true
        } else {
            false
        }
    }

    /// Get escalation statistics
    pub fn get_stats(&self) -> EscalationStats {
        let active_escalations = self.active_escalations.read().unwrap();
        let history = self.history.lock().unwrap();
        
        let active_count = active_escalations.len();
        let total_escalations = history.len();
        
        let level_distribution = active_escalations.values()
            .fold(HashMap::new(), |mut acc, escalation| {
                *acc.entry(escalation.current_level).or_insert(0) += 1;
                acc
            });
        
        let avg_escalation_time = if !active_escalations.is_empty() {
            let current_time = now_nanos();
            let total_time: u64 = active_escalations.values()
                .map(|e| current_time - e.started_at)
                .sum();
            total_time / active_escalations.len() as u64
        } else {
            0
        };

        EscalationStats {
            active_escalations: active_count,
            total_escalations,
            policies_count: self.policies.read().unwrap().len(),
            level_distribution,
            avg_escalation_time_ns: avg_escalation_time,
        }
    }

    /// Find matching escalation policy for an alert
    fn find_matching_policy(&self, alert: &Alert) -> Option<EscalationPolicy> {
        let policies = self.policies.read().unwrap();
        let mut matching_policies: Vec<&EscalationPolicy> = policies.values()
            .filter(|policy| policy.enabled && self.check_policy_conditions(policy, alert))
            .collect();
        
        // Sort by priority (highest first)
        matching_policies.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        matching_policies.first().map(|&policy| policy.clone())
    }

    /// Check if policy conditions are met for an alert
    fn check_policy_conditions(&self, policy: &EscalationPolicy, alert: &Alert) -> bool {
        if policy.conditions.is_empty() {
            return true; // No conditions means policy applies to all alerts
        }
        
        let required_conditions: Vec<&EscalationCondition> = policy.conditions.iter()
            .filter(|c| c.required)
            .collect();
        
        let optional_conditions: Vec<&EscalationCondition> = policy.conditions.iter()
            .filter(|c| !c.required)
            .collect();
        
        // All required conditions must be met
        let required_met = required_conditions.iter()
            .all(|condition| self.check_condition(condition, alert));
        
        // At least one optional condition must be met (if any exist)
        let optional_met = optional_conditions.is_empty() || 
            optional_conditions.iter().any(|condition| self.check_condition(condition, alert));
        
        required_met && optional_met
    }

    /// Check if a specific condition is met
    fn check_condition(&self, condition: &EscalationCondition, alert: &Alert) -> bool {
        match condition.condition_type {
            ConditionType::Severity => {
                if let Some(min_severity) = condition.parameters.get("min_severity") {
                    let min_sev = match min_severity.as_str() {
                        "Info" => AlertSeverity::Info,
                        "Warning" => AlertSeverity::Warning,
                        "Critical" => AlertSeverity::Critical,
                        "Emergency" => AlertSeverity::Emergency,
                        _ => AlertSeverity::Info,
                    };
                    alert.severity >= min_sev
                } else {
                    false
                }
            }
            ConditionType::MetricPattern => {
                if let Some(pattern) = condition.parameters.get("pattern") {
                    alert.metric_name.contains(pattern)
                } else {
                    false
                }
            }
            ConditionType::AlertAge => {
                if let Some(min_age_str) = condition.parameters.get("min_age_seconds") {
                    if let Ok(min_age_seconds) = min_age_str.parse::<u64>() {
                        let alert_age = now_nanos() - alert.created_at;
                        alert_age >= min_age_seconds * 1_000_000_000
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            _ => false, // Other condition types not implemented
        }
    }

    /// Start new escalation
    fn start_escalation(&self, alert: Alert, policy: EscalationPolicy, escalation_id: String) {
        let current_time = now_nanos();
        
        let escalation = ActiveEscalation {
            alert: alert.clone(),
            policy_id: policy.id.clone(),
            current_level: 0,
            started_at: current_time,
            last_escalated_at: current_time,
            next_escalation_at: current_time + policy.levels.get(0)
                .map(|level| level.escalation_delay_ns)
                .unwrap_or(self.config.default_escalation_delay_ns),
            status: EscalationStatus::Active,
            actions_taken: Vec::new(),
            level_history: Vec::new(),
        };
        
        {
            let mut active_escalations = self.active_escalations.write().unwrap();
            active_escalations.insert(escalation_id, escalation);
        }
        
        self.record_event(EscalationEvent {
            event_type: EscalationEventType::Started,
            alert_id: alert.id,
            policy_id: policy.id,
            level: 0,
            timestamp: current_time,
            context: HashMap::new(),
        });
    }

    /// Process a single escalation
    fn process_single_escalation(&self, escalation_id: &str, current_time: u64) -> EscalationResult {
        let mut escalations = self.active_escalations.write().unwrap();
        
        let escalation = match escalations.get_mut(escalation_id) {
            Some(esc) => esc,
            None => return EscalationResult::NoAction,
        };
        
        // Skip if escalation is paused or resolved
        if escalation.status != EscalationStatus::Active {
            return EscalationResult::NoAction;
        }
        
        // Check if it's time to escalate
        if current_time < escalation.next_escalation_at {
            return EscalationResult::NoAction;
        }
        
        // Get policy
        let policy = {
            let policies = self.policies.read().unwrap();
            match policies.get(&escalation.policy_id) {
                Some(policy) => policy.clone(),
                None => return EscalationResult::Failed,
            }
        };
        
        // Check if we've reached maximum level
        if escalation.current_level >= policy.max_level {
            escalation.status = EscalationStatus::MaxLevelReached;
            return EscalationResult::NoAction;
        }
        
        // Get next level
        let next_level = escalation.current_level + 1;
        let level_config = match policy.levels.get(next_level as usize) {
            Some(level) => level,
            None => {
                escalation.status = EscalationStatus::MaxLevelReached;
                return EscalationResult::NoAction;
            }
        };
        
        // Execute level actions
        let mut action_results = Vec::new();
        for action in &level_config.actions {
            let result = self.execute_action(action, &escalation.alert);
            action_results.push(result);
        }
        
        // Update escalation
        escalation.current_level = next_level;
        escalation.last_escalated_at = current_time;
        escalation.actions_taken.extend(action_results.clone());
        
        // Set next escalation time
        if let Some(next_level_config) = policy.levels.get((next_level + 1) as usize) {
            escalation.next_escalation_at = current_time + next_level_config.escalation_delay_ns;
        }
        
        // Record level event
        let level_event = EscalationLevelEvent {
            level: next_level,
            timestamp: current_time,
            actions: action_results,
        };
        escalation.level_history.push(level_event);
        
        // Record escalation event
        self.record_event(EscalationEvent {
            event_type: EscalationEventType::LevelReached,
            alert_id: escalation.alert.id.clone(),
            policy_id: escalation.policy_id.clone(),
            level: next_level,
            timestamp: current_time,
            context: HashMap::new(),
        });
        
        EscalationResult::Escalated
    }

    /// Execute an escalation action
    fn execute_action(&self, action: &EscalationAction, _alert: &Alert) -> EscalationActionResult {
        let start_time = now_nanos();
        
        // Simulate action execution
        let (success, error_message) = match action {
            EscalationAction::Notify { channels: _ } => {
                // TODO: Implement notification sending
                (true, None)
            }
            EscalationAction::IncreaseSeverity { new_severity: _ } => {
                // TODO: Implement severity increase
                (true, None)
            }
            EscalationAction::CreateIncident { system: _, priority: _ } => {
                // TODO: Implement incident creation
                (true, None)
            }
            EscalationAction::ExecuteScript { script: _, args: _ } => {
                // TODO: Implement script execution
                (true, None)
            }
            EscalationAction::PageOnCall { team: _, urgency: _ } => {
                // TODO: Implement paging
                (true, None)
            }
            EscalationAction::AutoRemediate { action: _, parameters: _ } => {
                // TODO: Implement auto-remediation
                (true, None)
            }
            EscalationAction::Suppress { reason: _ } => {
                // TODO: Implement alert suppression
                (true, None)
            }
            EscalationAction::Custom { action_type: _, parameters: _ } => {
                // TODO: Implement custom actions
                (true, None)
            }
        };
        
        let duration = now_nanos() - start_time;
        
        EscalationActionResult {
            action: action.clone(),
            executed_at: start_time,
            success,
            error_message,
            duration_ns: duration,
        }
    }

    /// Record escalation event
    fn record_event(&self, event: EscalationEvent) {
        let mut history = self.history.lock().unwrap();
        
        history.push_back(event);
        
        // Maintain history size limit
        while history.len() > self.config.history_size {
            history.pop_front();
        }
    }
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            max_active_escalations: 1000,
            history_size: 10000,
            default_escalation_delay_ns: 300_000_000_000, // 5 minutes
            enable_auto_escalation: true,
            processing_interval_ns: 60_000_000_000, // 1 minute
            max_escalation_level: 5,
        }
    }
}

/// Escalation processing result
#[derive(Debug, Clone)]
pub struct EscalationProcessResult {
    pub processed: usize,
    pub escalated: usize,
    pub resolved: usize,
    pub failed: usize,
}

/// Single escalation result
#[derive(Debug, Clone)]
enum EscalationResult {
    Escalated,
    Resolved,
    Failed,
    NoAction,
}

/// Escalation statistics
#[derive(Debug, Clone)]
pub struct EscalationStats {
    pub active_escalations: usize,
    pub total_escalations: usize,
    pub policies_count: usize,
    pub level_distribution: HashMap<u32, usize>,
    pub avg_escalation_time_ns: u64,
}

/// Predefined escalation policies
pub struct CommonEscalationPolicies;

impl CommonEscalationPolicies {
    /// Create critical alert escalation policy
    pub fn critical_alert_policy() -> EscalationPolicy {
        EscalationPolicy {
            id: "critical_alert_policy".to_string(),
            name: "Critical Alert Escalation".to_string(),
            levels: vec![
                EscalationLevel {
                    level: 0,
                    name: "Initial Alert".to_string(),
                    escalation_delay_ns: 0,
                    actions: vec![
                        EscalationAction::Notify { 
                            channels: vec!["slack".to_string(), "email".to_string()] 
                        }
                    ],
                    notification_channels: vec!["slack".to_string()],
                    suppress_lower_levels: false,
                },
                EscalationLevel {
                    level: 1,
                    name: "Team Lead Notification".to_string(),
                    escalation_delay_ns: 300_000_000_000, // 5 minutes
                    actions: vec![
                        EscalationAction::Notify { 
                            channels: vec!["team_lead_email".to_string()] 
                        }
                    ],
                    notification_channels: vec!["team_lead_email".to_string()],
                    suppress_lower_levels: false,
                },
                EscalationLevel {
                    level: 2,
                    name: "On-Call Paging".to_string(),
                    escalation_delay_ns: 600_000_000_000, // 10 minutes
                    actions: vec![
                        EscalationAction::PageOnCall { 
                            team: "trading_team".to_string(),
                            urgency: "high".to_string()
                        }
                    ],
                    notification_channels: vec!["pager".to_string()],
                    suppress_lower_levels: true,
                },
            ],
            conditions: vec![
                EscalationCondition {
                    condition_type: ConditionType::Severity,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("min_severity".to_string(), "Critical".to_string());
                        params
                    },
                    required: true,
                }
            ],
            enabled: true,
            priority: 100,
            max_level: 2,
            auto_resolve_timeout_ns: Some(3600_000_000_000), // 1 hour
        }
    }

    /// Create latency escalation policy
    pub fn latency_escalation_policy() -> EscalationPolicy {
        EscalationPolicy {
            id: "latency_escalation_policy".to_string(),
            name: "Latency Alert Escalation".to_string(),
            levels: vec![
                EscalationLevel {
                    level: 0,
                    name: "Initial Latency Alert".to_string(),
                    escalation_delay_ns: 0,
                    actions: vec![
                        EscalationAction::Notify { 
                            channels: vec!["performance_channel".to_string()] 
                        }
                    ],
                    notification_channels: vec!["performance_channel".to_string()],
                    suppress_lower_levels: false,
                },
                EscalationLevel {
                    level: 1,
                    name: "Performance Team Alert".to_string(),
                    escalation_delay_ns: 180_000_000_000, // 3 minutes
                    actions: vec![
                        EscalationAction::Notify { 
                            channels: vec!["performance_team".to_string()] 
                        },
                        EscalationAction::AutoRemediate {
                            action: "restart_slow_components".to_string(),
                            parameters: HashMap::new(),
                        }
                    ],
                    notification_channels: vec!["performance_team".to_string()],
                    suppress_lower_levels: false,
                },
            ],
            conditions: vec![
                EscalationCondition {
                    condition_type: ConditionType::MetricPattern,
                    parameters: {
                        let mut params = HashMap::new();
                        params.insert("pattern".to_string(), "latency".to_string());
                        params
                    },
                    required: true,
                }
            ],
            enabled: true,
            priority: 80,
            max_level: 1,
            auto_resolve_timeout_ns: Some(1800_000_000_000), // 30 minutes
        }
    }

    /// Get all common escalation policies
    pub fn all_policies() -> Vec<EscalationPolicy> {
        vec![
            Self::critical_alert_policy(),
            Self::latency_escalation_policy(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::alert_manager::AlertStatus;

    fn create_test_alert(name: &str, metric: &str, severity: AlertSeverity) -> Alert {
        Alert {
            id: format!("alert_{}", name),
            name: name.to_string(),
            severity,
            status: AlertStatus::Active,
            message: format!("Test alert for {}", metric),
            metric_name: metric.to_string(),
            current_value: 100.0,
            threshold_value: 50.0,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            resolved_at: 0,
            context: HashMap::new(),
            fire_count: 1,
            duration_ns: 0,
        }
    }

    #[test]
    fn test_escalation_engine_creation() {
        let engine = EscalationEngine::new();
        let stats = engine.get_stats();
        
        assert_eq!(stats.active_escalations, 0);
        assert_eq!(stats.policies_count, 0);
    }

    #[test]
    fn test_policy_management() {
        let engine = EscalationEngine::new();
        let policy = CommonEscalationPolicies::critical_alert_policy();
        
        engine.add_policy(policy.clone());
        
        let retrieved = engine.get_policy(&policy.id).unwrap();
        assert_eq!(retrieved.id, policy.id);
        assert_eq!(retrieved.levels.len(), policy.levels.len());
        
        assert!(engine.remove_policy(&policy.id));
        assert!(engine.get_policy(&policy.id).is_none());
    }

    #[test]
    fn test_alert_processing() {
        let engine = EscalationEngine::new();
        let policy = CommonEscalationPolicies::critical_alert_policy();
        engine.add_policy(policy);
        
        let alert = create_test_alert("critical_test", "test_metric", AlertSeverity::Critical);
        let escalation_id = engine.process_alert(&alert);
        
        assert!(escalation_id.is_some());
        
        let escalations = engine.get_active_escalations();
        assert_eq!(escalations.len(), 1);
        assert_eq!(escalations[0].alert.id, alert.id);
    }

    #[test]
    fn test_condition_checking() {
        let engine = EscalationEngine::new();
        
        let condition = EscalationCondition {
            condition_type: ConditionType::Severity,
            parameters: {
                let mut params = HashMap::new();
                params.insert("min_severity".to_string(), "Critical".to_string());
                params
            },
            required: true,
        };
        
        let critical_alert = create_test_alert("critical", "test", AlertSeverity::Critical);
        let warning_alert = create_test_alert("warning", "test", AlertSeverity::Warning);
        
        assert!(engine.check_condition(&condition, &critical_alert));
        assert!(!engine.check_condition(&condition, &warning_alert));
    }

    #[test]
    fn test_escalation_control() {
        let engine = EscalationEngine::new();
        let policy = CommonEscalationPolicies::critical_alert_policy();
        engine.add_policy(policy);
        
        let alert = create_test_alert("control_test", "test_metric", AlertSeverity::Critical);
        let escalation_id = engine.process_alert(&alert).unwrap();
        
        // Test pause
        assert!(engine.pause_escalation(&escalation_id));
        let escalation = engine.get_escalation(&escalation_id).unwrap();
        assert_eq!(escalation.status, EscalationStatus::Paused);
        
        // Test resume
        assert!(engine.resume_escalation(&escalation_id));
        let escalation = engine.get_escalation(&escalation_id).unwrap();
        assert_eq!(escalation.status, EscalationStatus::Active);
        
        // Test resolve
        assert!(engine.resolve_escalation(&escalation_id));
        assert!(engine.get_escalation(&escalation_id).is_none());
    }

    #[test]
    fn test_common_policies() {
        let policies = CommonEscalationPolicies::all_policies();
        assert_eq!(policies.len(), 2);
        
        let critical_policy = &policies[0];
        assert_eq!(critical_policy.id, "critical_alert_policy");
        assert_eq!(critical_policy.levels.len(), 3);
        
        let latency_policy = &policies[1];
        assert_eq!(latency_policy.id, "latency_escalation_policy");
        assert_eq!(latency_policy.levels.len(), 2);
    }

    #[test]
    fn test_escalation_stats() {
        let engine = EscalationEngine::new();
        let policy = CommonEscalationPolicies::critical_alert_policy();
        engine.add_policy(policy);
        
        let alert1 = create_test_alert("test1", "metric1", AlertSeverity::Critical);
        let alert2 = create_test_alert("test2", "metric2", AlertSeverity::Critical);
        
        engine.process_alert(&alert1);
        engine.process_alert(&alert2);
        
        let stats = engine.get_stats();
        assert_eq!(stats.active_escalations, 2);
        assert_eq!(stats.policies_count, 1);
        assert!(stats.level_distribution.contains_key(&0));
    }

    #[test]
    fn test_escalation_processing() {
        let engine = EscalationEngine::new();
        let policy = CommonEscalationPolicies::critical_alert_policy();
        engine.add_policy(policy);
        
        let alert = create_test_alert("process_test", "test_metric", AlertSeverity::Critical);
        engine.process_alert(&alert);
        
        let result = engine.process_escalations();
        assert_eq!(result.processed, 1);
        // Note: escalated count would be 0 because escalation delay hasn't passed
    }
}