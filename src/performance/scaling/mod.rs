use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub mod resource_scaler;
pub mod scaling_policy;
pub mod scaling_events;
pub mod load_monitor;
pub mod cpu_manager;
pub mod memory_pool_manager;
pub mod network_bandwidth_manager;

pub use resource_scaler::*;
pub use scaling_policy::*;
pub use scaling_events::*;
pub use load_monitor::*;
pub use cpu_manager::*;
pub use memory_pool_manager::*;
pub use network_bandwidth_manager::*;

/// Core scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Minimum scaling interval to prevent thrashing
    pub min_scaling_interval: Duration,
    /// Maximum resource utilization before scaling up
    pub scale_up_threshold: f64,
    /// Minimum resource utilization before scaling down
    pub scale_down_threshold: f64,
    /// Cooldown period after scaling operations
    pub cooldown_period: Duration,
    /// Enable/disable auto-scaling
    pub enabled: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_scaling_interval: Duration::from_secs(30),
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_period: Duration::from_secs(60),
            enabled: true,
        }
    }
}

/// Resource types that can be scaled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Cpu,
    Memory,
    Network,
    ThreadPool,
}

/// Scaling direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingDirection {
    Up,
    Down,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub resource_type: ResourceType,
    pub current_utilization: f64,
    pub capacity: u64,
    pub allocated: u64,
    pub timestamp: Instant,
}

/// Scaling decision result
#[derive(Debug, Clone)]
pub struct ScalingDecision {
    pub resource_type: ResourceType,
    pub direction: ScalingDirection,
    pub current_capacity: u64,
    pub target_capacity: u64,
    pub reason: String,
    pub confidence: f64,
}

/// Auto-scaling manager
pub struct AutoScaler {
    config: ScalingConfig,
    resource_scaler: Arc<ResourceScaler>,
    load_monitor: Arc<LoadMonitor>,
    scaling_events: Arc<ScalingEventLogger>,
    last_scaling: HashMap<ResourceType, Instant>,
    enabled: AtomicBool,
}

impl AutoScaler {
    pub fn new(
        config: ScalingConfig,
        resource_scaler: Arc<ResourceScaler>,
        load_monitor: Arc<LoadMonitor>,
        scaling_events: Arc<ScalingEventLogger>,
    ) -> Self {
        Self {
            config,
            resource_scaler,
            load_monitor,
            scaling_events,
            last_scaling: HashMap::new(),
            enabled: AtomicBool::new(true),
        }
    }

    /// Enable or disable auto-scaling
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if auto-scaling is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed) && self.config.enabled
    }

    /// Evaluate scaling decisions for all resources
    pub fn evaluate_scaling(&mut self) -> Vec<ScalingDecision> {
        if !self.is_enabled() {
            return Vec::new();
        }

        let mut decisions = Vec::new();
        let utilizations = self.load_monitor.get_current_utilization();

        for utilization in utilizations {
            if let Some(decision) = self.evaluate_resource_scaling(&utilization) {
                decisions.push(decision);
            }
        }

        decisions
    }

    /// Evaluate scaling for a specific resource
    fn evaluate_resource_scaling(&mut self, utilization: &ResourceUtilization) -> Option<ScalingDecision> {
        // Check cooldown period
        if let Some(last_scaling) = self.last_scaling.get(&utilization.resource_type) {
            if last_scaling.elapsed() < self.config.cooldown_period {
                return None;
            }
        }

        // Determine scaling direction
        let direction = if utilization.current_utilization > self.config.scale_up_threshold {
            ScalingDirection::Up
        } else if utilization.current_utilization < self.config.scale_down_threshold {
            ScalingDirection::Down
        } else {
            return None; // No scaling needed
        };

        // Calculate target capacity
        let target_capacity = self.calculate_target_capacity(utilization, direction);
        
        if target_capacity == utilization.capacity {
            return None; // No change needed
        }

        // Calculate confidence based on utilization trend
        let confidence = self.calculate_scaling_confidence(utilization);

        Some(ScalingDecision {
            resource_type: utilization.resource_type,
            direction,
            current_capacity: utilization.capacity,
            target_capacity,
            reason: format!(
                "Utilization {}% {} threshold {}%",
                (utilization.current_utilization * 100.0) as u32,
                match direction {
                    ScalingDirection::Up => "exceeds",
                    ScalingDirection::Down => "below",
                },
                match direction {
                    ScalingDirection::Up => (self.config.scale_up_threshold * 100.0) as u32,
                    ScalingDirection::Down => (self.config.scale_down_threshold * 100.0) as u32,
                }
            ),
            confidence,
        })
    }

    /// Calculate target capacity for scaling
    fn calculate_target_capacity(&self, utilization: &ResourceUtilization, direction: ScalingDirection) -> u64 {
        match direction {
            ScalingDirection::Up => {
                // Scale up by 50% or to handle current load with 20% headroom
                let scale_factor = 1.5;
                let headroom_target = (utilization.allocated as f64 / 0.8) as u64;
                std::cmp::max(
                    (utilization.capacity as f64 * scale_factor) as u64,
                    headroom_target
                )
            }
            ScalingDirection::Down => {
                // Scale down to 150% of current usage
                let target = (utilization.allocated as f64 * 1.5) as u64;
                std::cmp::max(target, utilization.capacity / 2) // Never scale down more than 50%
            }
        }
    }

    /// Calculate confidence in scaling decision
    fn calculate_scaling_confidence(&self, utilization: &ResourceUtilization) -> f64 {
        // Higher confidence for more extreme utilization values
        let threshold_distance = match utilization.current_utilization {
            x if x > self.config.scale_up_threshold => {
                (x - self.config.scale_up_threshold) / (1.0 - self.config.scale_up_threshold)
            }
            x if x < self.config.scale_down_threshold => {
                (self.config.scale_down_threshold - x) / self.config.scale_down_threshold
            }
            _ => 0.0,
        };

        // Confidence ranges from 0.5 to 1.0
        0.5 + (threshold_distance * 0.5)
    }

    /// Execute scaling decisions
    pub async fn execute_scaling(&mut self, decisions: Vec<ScalingDecision>) -> Vec<Result<(), String>> {
        let mut results = Vec::new();

        for decision in decisions {
            let result = self.resource_scaler.scale_resource(
                decision.resource_type,
                decision.target_capacity,
            ).await;

            // Log scaling event
            let event = ScalingEvent {
                resource_type: decision.resource_type,
                direction: decision.direction,
                from_capacity: decision.current_capacity,
                to_capacity: decision.target_capacity,
                reason: decision.reason.clone(),
                success: result.is_ok(),
                timestamp: Instant::now(),
            };

            self.scaling_events.log_event(event);

            if result.is_ok() {
                self.last_scaling.insert(decision.resource_type, Instant::now());
            }

            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_scaling_decision_evaluation() {
        let config = ScalingConfig::default();
        let resource_scaler = Arc::new(ResourceScaler::new());
        let load_monitor = Arc::new(LoadMonitor::new());
        let scaling_events = Arc::new(ScalingEventLogger::new());

        let mut auto_scaler = AutoScaler::new(
            config,
            resource_scaler,
            load_monitor,
            scaling_events,
        );

        // Test scale up decision
        let high_utilization = ResourceUtilization {
            resource_type: ResourceType::Cpu,
            current_utilization: 0.9,
            capacity: 100,
            allocated: 90,
            timestamp: Instant::now(),
        };

        let decision = auto_scaler.evaluate_resource_scaling(&high_utilization);
        assert!(decision.is_some());
        let decision = decision.unwrap();
        assert_eq!(decision.direction, ScalingDirection::Up);
        assert!(decision.target_capacity > 100);

        // Test scale down decision
        let low_utilization = ResourceUtilization {
            resource_type: ResourceType::Cpu,
            current_utilization: 0.2,
            capacity: 100,
            allocated: 20,
            timestamp: Instant::now(),
        };

        let decision = auto_scaler.evaluate_resource_scaling(&low_utilization);
        assert!(decision.is_some());
        let decision = decision.unwrap();
        assert_eq!(decision.direction, ScalingDirection::Down);
        assert!(decision.target_capacity < 100);
    }

    #[test]
    fn test_scaling_confidence_calculation() {
        let config = ScalingConfig::default();
        let resource_scaler = Arc::new(ResourceScaler::new());
        let load_monitor = Arc::new(LoadMonitor::new());
        let scaling_events = Arc::new(ScalingEventLogger::new());

        let auto_scaler = AutoScaler::new(
            config,
            resource_scaler,
            load_monitor,
            scaling_events,
        );

        // Test high utilization confidence
        let high_util = ResourceUtilization {
            resource_type: ResourceType::Cpu,
            current_utilization: 0.95,
            capacity: 100,
            allocated: 95,
            timestamp: Instant::now(),
        };

        let confidence = auto_scaler.calculate_scaling_confidence(&high_util);
        assert!(confidence > 0.8);

        // Test moderate utilization confidence
        let moderate_util = ResourceUtilization {
            resource_type: ResourceType::Cpu,
            current_utilization: 0.85,
            capacity: 100,
            allocated: 85,
            timestamp: Instant::now(),
        };

        let confidence = auto_scaler.calculate_scaling_confidence(&moderate_util);
        assert!(confidence > 0.5 && confidence < 0.8);
    }
}