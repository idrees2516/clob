use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use crate::performance::scaling::{ResourceType, ScalingDirection};

/// Scaling event for logging and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    pub resource_type: ResourceType,
    pub direction: ScalingDirection,
    pub from_capacity: u64,
    pub to_capacity: u64,
    pub reason: String,
    pub success: bool,
    pub timestamp: Instant,
}

/// Scaling event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Enhanced scaling event with additional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedScalingEvent {
    pub event: ScalingEvent,
    pub severity: EventSeverity,
    pub duration_ms: Option<u64>,
    pub utilization_before: f64,
    pub utilization_after: Option<f64>,
    pub correlation_id: String,
    pub tags: HashMap<String, String>,
}

/// Scaling event logger and analyzer
pub struct ScalingEventLogger {
    events: Arc<RwLock<VecDeque<EnhancedScalingEvent>>>,
    event_counter: AtomicU64,
    max_events: usize,
    analysis_window: Duration,
}

impl ScalingEventLogger {
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(max_events))),
            event_counter: AtomicU64::new(0),
            max_events,
            analysis_window: Duration::from_hours(24),
        }
    }

    /// Log a scaling event
    pub async fn log_event(&self, event: ScalingEvent) {
        let enhanced_event = self.enhance_event(event).await;
        self.store_event(enhanced_event).await;
    }

    /// Log an enhanced scaling event
    pub async fn log_enhanced_event(&self, event: EnhancedScalingEvent) {
        self.store_event(event).await;
    }

    /// Enhance a basic scaling event with additional metadata
    async fn enhance_event(&self, event: ScalingEvent) -> EnhancedScalingEvent {
        let severity = self.determine_severity(&event);
        let correlation_id = self.generate_correlation_id();
        
        EnhancedScalingEvent {
            event,
            severity,
            duration_ms: None,
            utilization_before: 0.0, // Would be populated by caller
            utilization_after: None,
            correlation_id,
            tags: HashMap::new(),
        }
    }

    /// Store event in the circular buffer
    async fn store_event(&self, event: EnhancedScalingEvent) {
        let mut events = self.events.write().await;
        
        if events.len() >= self.max_events {
            events.pop_front();
        }
        
        events.push_back(event);
        self.event_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Determine event severity based on scaling characteristics
    fn determine_severity(&self, event: &ScalingEvent) -> EventSeverity {
        if !event.success {
            return EventSeverity::Error;
        }

        let scale_factor = event.to_capacity as f64 / event.from_capacity as f64;
        
        match scale_factor {
            x if x >= 2.0 || x <= 0.5 => EventSeverity::Warning, // Large scaling changes
            x if x >= 1.5 || x <= 0.7 => EventSeverity::Info,    // Moderate changes
            _ => EventSeverity::Info, // Small changes
        }
    }

    /// Generate correlation ID for event tracking
    fn generate_correlation_id(&self) -> String {
        let counter = self.event_counter.load(Ordering::Relaxed);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        
        format!("scale-{}-{}", timestamp, counter)
    }

    /// Get recent events within the analysis window
    pub async fn get_recent_events(&self) -> Vec<EnhancedScalingEvent> {
        let events = self.events.read().await;
        let cutoff = Instant::now() - self.analysis_window;
        
        events
            .iter()
            .filter(|event| event.event.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    /// Get events for a specific resource type
    pub async fn get_events_for_resource(&self, resource_type: ResourceType) -> Vec<EnhancedScalingEvent> {
        let events = self.events.read().await;
        
        events
            .iter()
            .filter(|event| event.event.resource_type == resource_type)
            .cloned()
            .collect()
    }

    /// Get events by severity level
    pub async fn get_events_by_severity(&self, severity: EventSeverity) -> Vec<EnhancedScalingEvent> {
        let events = self.events.read().await;
        
        events
            .iter()
            .filter(|event| event.severity == severity)
            .cloned()
            .collect()
    }

    /// Analyze scaling patterns and generate insights
    pub async fn analyze_scaling_patterns(&self) -> ScalingAnalysis {
        let events = self.get_recent_events().await;
        
        if events.is_empty() {
            return ScalingAnalysis::default();
        }

        let mut analysis = ScalingAnalysis::default();
        analysis.total_events = events.len();
        analysis.analysis_period = self.analysis_window;

        // Count events by resource type
        for event in &events {
            *analysis.events_by_resource.entry(event.event.resource_type).or_insert(0) += 1;
            *analysis.events_by_direction.entry(event.event.direction).or_insert(0) += 1;
            *analysis.events_by_severity.entry(event.severity).or_insert(0) += 1;
            
            if event.event.success {
                analysis.successful_events += 1;
            } else {
                analysis.failed_events += 1;
            }
        }

        // Calculate scaling frequency
        analysis.scaling_frequency = self.calculate_scaling_frequency(&events).await;
        
        // Detect scaling patterns
        analysis.patterns = self.detect_scaling_patterns(&events).await;
        
        // Calculate resource stability
        analysis.resource_stability = self.calculate_resource_stability(&events).await;

        analysis
    }

    /// Calculate scaling frequency for each resource
    async fn calculate_scaling_frequency(&self, events: &[EnhancedScalingEvent]) -> HashMap<ResourceType, f64> {
        let mut frequency = HashMap::new();
        let mut resource_events = HashMap::new();

        for event in events {
            *resource_events.entry(event.event.resource_type).or_insert(0) += 1;
        }

        let hours = self.analysis_window.as_secs_f64() / 3600.0;
        
        for (resource_type, count) in resource_events {
            frequency.insert(resource_type, count as f64 / hours);
        }

        frequency
    }

    /// Detect scaling patterns and anomalies
    async fn detect_scaling_patterns(&self, events: &[EnhancedScalingEvent]) -> Vec<ScalingPattern> {
        let mut patterns = Vec::new();

        // Detect oscillation patterns
        if let Some(oscillation) = self.detect_oscillation(events) {
            patterns.push(oscillation);
        }

        // Detect trending patterns
        if let Some(trend) = self.detect_trending(events) {
            patterns.push(trend);
        }

        // Detect failure patterns
        if let Some(failure_pattern) = self.detect_failure_patterns(events) {
            patterns.push(failure_pattern);
        }

        patterns
    }

    /// Detect oscillation patterns (rapid up/down scaling)
    fn detect_oscillation(&self, events: &[EnhancedScalingEvent]) -> Option<ScalingPattern> {
        let mut resource_events = HashMap::new();
        
        for event in events {
            resource_events.entry(event.event.resource_type)
                .or_insert_with(Vec::new)
                .push(event);
        }

        for (resource_type, resource_events) in resource_events {
            if resource_events.len() < 4 {
                continue;
            }

            let mut oscillations = 0;
            let mut last_direction = None;

            for event in &resource_events {
                if let Some(last_dir) = last_direction {
                    if last_dir != event.event.direction {
                        oscillations += 1;
                    }
                }
                last_direction = Some(event.event.direction);
            }

            if oscillations >= 3 {
                return Some(ScalingPattern {
                    pattern_type: PatternType::Oscillation,
                    resource_type,
                    description: format!(
                        "Detected {} oscillations in {} events",
                        oscillations, resource_events.len()
                    ),
                    severity: EventSeverity::Warning,
                    recommendation: "Consider adjusting scaling thresholds or cooldown periods".to_string(),
                });
            }
        }

        None
    }

    /// Detect trending patterns (consistent scaling direction)
    fn detect_trending(&self, events: &[EnhancedScalingEvent]) -> Option<ScalingPattern> {
        let mut resource_events = HashMap::new();
        
        for event in events {
            resource_events.entry(event.event.resource_type)
                .or_insert_with(Vec::new)
                .push(event);
        }

        for (resource_type, resource_events) in resource_events {
            if resource_events.len() < 3 {
                continue;
            }

            let up_count = resource_events.iter()
                .filter(|e| e.event.direction == ScalingDirection::Up)
                .count();
            
            let down_count = resource_events.iter()
                .filter(|e| e.event.direction == ScalingDirection::Down)
                .count();

            let trend_ratio = if up_count > down_count {
                up_count as f64 / resource_events.len() as f64
            } else {
                down_count as f64 / resource_events.len() as f64
            };

            if trend_ratio >= 0.8 {
                let direction = if up_count > down_count { "upward" } else { "downward" };
                return Some(ScalingPattern {
                    pattern_type: PatternType::Trending,
                    resource_type,
                    description: format!(
                        "Detected strong {} trend: {:.1}% of events",
                        direction, trend_ratio * 100.0
                    ),
                    severity: EventSeverity::Info,
                    recommendation: format!(
                        "Monitor {} resource usage patterns for capacity planning",
                        direction
                    ),
                });
            }
        }

        None
    }

    /// Detect failure patterns
    fn detect_failure_patterns(&self, events: &[EnhancedScalingEvent]) -> Option<ScalingPattern> {
        let failed_events: Vec<_> = events.iter()
            .filter(|e| !e.event.success)
            .collect();

        if failed_events.len() >= 3 {
            let failure_rate = failed_events.len() as f64 / events.len() as f64;
            
            return Some(ScalingPattern {
                pattern_type: PatternType::FailureCluster,
                resource_type: failed_events[0].event.resource_type, // Use first failure's resource
                description: format!(
                    "High failure rate: {:.1}% ({} failures out of {} events)",
                    failure_rate * 100.0,
                    failed_events.len(),
                    events.len()
                ),
                severity: EventSeverity::Error,
                recommendation: "Investigate scaling infrastructure and resource limits".to_string(),
            });
        }

        None
    }

    /// Calculate resource stability metrics
    async fn calculate_resource_stability(&self, events: &[EnhancedScalingEvent]) -> HashMap<ResourceType, f64> {
        let mut stability = HashMap::new();
        let mut resource_events = HashMap::new();

        for event in events {
            resource_events.entry(event.event.resource_type)
                .or_insert_with(Vec::new)
                .push(event);
        }

        for (resource_type, resource_events) in resource_events {
            if resource_events.is_empty() {
                continue;
            }

            // Calculate stability as inverse of scaling frequency
            let scaling_count = resource_events.len() as f64;
            let time_span = self.analysis_window.as_secs_f64();
            let stability_score = 1.0 / (1.0 + (scaling_count / time_span * 3600.0)); // Events per hour

            stability.insert(resource_type, stability_score);
        }

        stability
    }

    /// Get event statistics
    pub async fn get_event_stats(&self) -> EventStats {
        let events = self.events.read().await;
        let total_events = self.event_counter.load(Ordering::Relaxed);
        
        EventStats {
            total_events,
            stored_events: events.len(),
            max_capacity: self.max_events,
            analysis_window: self.analysis_window,
        }
    }

    /// Clear all events
    pub async fn clear_events(&self) {
        let mut events = self.events.write().await;
        events.clear();
    }
}

/// Scaling analysis results
#[derive(Debug, Clone, Default)]
pub struct ScalingAnalysis {
    pub total_events: usize,
    pub successful_events: usize,
    pub failed_events: usize,
    pub analysis_period: Duration,
    pub events_by_resource: HashMap<ResourceType, usize>,
    pub events_by_direction: HashMap<ScalingDirection, usize>,
    pub events_by_severity: HashMap<EventSeverity, usize>,
    pub scaling_frequency: HashMap<ResourceType, f64>,
    pub patterns: Vec<ScalingPattern>,
    pub resource_stability: HashMap<ResourceType, f64>,
}

/// Detected scaling pattern
#[derive(Debug, Clone)]
pub struct ScalingPattern {
    pub pattern_type: PatternType,
    pub resource_type: ResourceType,
    pub description: String,
    pub severity: EventSeverity,
    pub recommendation: String,
}

/// Types of scaling patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Oscillation,
    Trending,
    FailureCluster,
    Anomaly,
}

/// Event statistics
#[derive(Debug, Clone)]
pub struct EventStats {
    pub total_events: u64,
    pub stored_events: usize,
    pub max_capacity: usize,
    pub analysis_window: Duration,
}

impl Default for ScalingEventLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_logging() {
        let logger = ScalingEventLogger::new();
        
        let event = ScalingEvent {
            resource_type: ResourceType::Cpu,
            direction: ScalingDirection::Up,
            from_capacity: 4,
            to_capacity: 8,
            reason: "High utilization".to_string(),
            success: true,
            timestamp: Instant::now(),
        };

        logger.log_event(event).await;
        
        let events = logger.get_recent_events().await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.resource_type, ResourceType::Cpu);
    }

    #[tokio::test]
    async fn test_pattern_detection() {
        let logger = ScalingEventLogger::new();
        
        // Create oscillation pattern
        for i in 0..6 {
            let direction = if i % 2 == 0 {
                ScalingDirection::Up
            } else {
                ScalingDirection::Down
            };
            
            let event = ScalingEvent {
                resource_type: ResourceType::Cpu,
                direction,
                from_capacity: 4,
                to_capacity: if direction == ScalingDirection::Up { 8 } else { 4 },
                reason: "Test oscillation".to_string(),
                success: true,
                timestamp: Instant::now(),
            };

            logger.log_event(event).await;
        }

        let analysis = logger.analyze_scaling_patterns().await;
        assert!(!analysis.patterns.is_empty());
        
        let has_oscillation = analysis.patterns.iter()
            .any(|p| matches!(p.pattern_type, PatternType::Oscillation));
        assert!(has_oscillation);
    }

    #[tokio::test]
    async fn test_event_filtering() {
        let logger = ScalingEventLogger::new();
        
        // Add events for different resources
        let cpu_event = ScalingEvent {
            resource_type: ResourceType::Cpu,
            direction: ScalingDirection::Up,
            from_capacity: 4,
            to_capacity: 8,
            reason: "CPU scaling".to_string(),
            success: true,
            timestamp: Instant::now(),
        };

        let memory_event = ScalingEvent {
            resource_type: ResourceType::Memory,
            direction: ScalingDirection::Up,
            from_capacity: 1024,
            to_capacity: 2048,
            reason: "Memory scaling".to_string(),
            success: false,
            timestamp: Instant::now(),
        };

        logger.log_event(cpu_event).await;
        logger.log_event(memory_event).await;

        let cpu_events = logger.get_events_for_resource(ResourceType::Cpu).await;
        assert_eq!(cpu_events.len(), 1);

        let error_events = logger.get_events_by_severity(EventSeverity::Error).await;
        assert_eq!(error_events.len(), 1);
    }
}