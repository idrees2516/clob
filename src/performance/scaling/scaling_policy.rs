use std::time::Duration;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::performance::scaling::{ResourceType, ScalingDirection};

/// Scaling policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub name: String,
    pub resource_type: ResourceType,
    pub enabled: bool,
    pub thresholds: ScalingThresholds,
    pub constraints: ScalingConstraints,
    pub behavior: ScalingBehavior,
}

/// Scaling thresholds for different metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingThresholds {
    /// Scale up when utilization exceeds this threshold
    pub scale_up_threshold: f64,
    /// Scale down when utilization falls below this threshold
    pub scale_down_threshold: f64,
    /// Emergency scale up threshold for immediate action
    pub emergency_threshold: f64,
    /// Minimum sustained duration before scaling
    pub sustained_duration: Duration,
}

/// Scaling constraints and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConstraints {
    /// Minimum resource capacity
    pub min_capacity: u64,
    /// Maximum resource capacity
    pub max_capacity: u64,
    /// Maximum scaling step size (percentage)
    pub max_scale_step: f64,
    /// Minimum scaling step size (percentage)
    pub min_scale_step: f64,
    /// Cooldown period between scaling operations
    pub cooldown_period: Duration,
}

/// Scaling behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingBehavior {
    /// Scaling strategy
    pub strategy: ScalingStrategy,
    /// Predictive scaling settings
    pub predictive: Option<PredictiveScaling>,
    /// Reactive scaling settings
    pub reactive: ReactiveScaling,
}

/// Scaling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// React to current metrics only
    Reactive,
    /// Use predictive algorithms
    Predictive,
    /// Combine reactive and predictive
    Hybrid,
}

/// Predictive scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveScaling {
    /// Enable predictive scaling
    pub enabled: bool,
    /// Prediction horizon in minutes
    pub horizon_minutes: u32,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Historical data window for predictions
    pub history_window: Duration,
}

/// Reactive scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactiveScaling {
    /// Metric evaluation window
    pub evaluation_window: Duration,
    /// Number of data points required for decision
    pub min_data_points: u32,
    /// Metric aggregation method
    pub aggregation: MetricAggregation,
}

/// Metric aggregation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricAggregation {
    Average,
    Maximum,
    Minimum,
    Percentile(u8),
}

/// Policy manager for scaling policies
pub struct ScalingPolicyManager {
    policies: HashMap<String, ScalingPolicy>,
    default_policies: HashMap<ResourceType, ScalingPolicy>,
}

impl ScalingPolicyManager {
    pub fn new() -> Self {
        let mut manager = Self {
            policies: HashMap::new(),
            default_policies: HashMap::new(),
        };

        // Initialize default policies
        manager.initialize_default_policies();
        manager
    }

    /// Initialize default scaling policies for each resource type
    fn initialize_default_policies(&mut self) {
        // CPU scaling policy
        let cpu_policy = ScalingPolicy {
            name: "default-cpu".to_string(),
            resource_type: ResourceType::Cpu,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
                emergency_threshold: 0.95,
                sustained_duration: Duration::from_secs(60),
            },
            constraints: ScalingConstraints {
                min_capacity: 2,
                max_capacity: 64,
                max_scale_step: 0.5, // 50% max increase
                min_scale_step: 0.1, // 10% min change
                cooldown_period: Duration::from_secs(300),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(300),
                    min_data_points: 5,
                    aggregation: MetricAggregation::Average,
                },
            },
        };

        // Memory scaling policy
        let memory_policy = ScalingPolicy {
            name: "default-memory".to_string(),
            resource_type: ResourceType::Memory,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.85,
                scale_down_threshold: 0.4,
                emergency_threshold: 0.95,
                sustained_duration: Duration::from_secs(120),
            },
            constraints: ScalingConstraints {
                min_capacity: 1024, // 1GB
                max_capacity: 65536, // 64GB
                max_scale_step: 0.3, // 30% max increase
                min_scale_step: 0.1, // 10% min change
                cooldown_period: Duration::from_secs(600),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(600),
                    min_data_points: 3,
                    aggregation: MetricAggregation::Maximum,
                },
            },
        };

        // Network scaling policy
        let network_policy = ScalingPolicy {
            name: "default-network".to_string(),
            resource_type: ResourceType::Network,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.75,
                scale_down_threshold: 0.25,
                emergency_threshold: 0.9,
                sustained_duration: Duration::from_secs(30),
            },
            constraints: ScalingConstraints {
                min_capacity: 100, // 100 Mbps
                max_capacity: 10000, // 10 Gbps
                max_scale_step: 1.0, // 100% max increase
                min_scale_step: 0.2, // 20% min change
                cooldown_period: Duration::from_secs(180),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(180),
                    min_data_points: 6,
                    aggregation: MetricAggregation::Percentile(95),
                },
            },
        };

        // Thread pool scaling policy
        let thread_policy = ScalingPolicy {
            name: "default-threads".to_string(),
            resource_type: ResourceType::ThreadPool,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.2,
                emergency_threshold: 0.95,
                sustained_duration: Duration::from_secs(30),
            },
            constraints: ScalingConstraints {
                min_capacity: 4,
                max_capacity: 256,
                max_scale_step: 0.5, // 50% max increase
                min_scale_step: 0.1, // 10% min change
                cooldown_period: Duration::from_secs(120),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(120),
                    min_data_points: 4,
                    aggregation: MetricAggregation::Average,
                },
            },
        };

        self.default_policies.insert(ResourceType::Cpu, cpu_policy);
        self.default_policies.insert(ResourceType::Memory, memory_policy);
        self.default_policies.insert(ResourceType::Network, network_policy);
        self.default_policies.insert(ResourceType::ThreadPool, thread_policy);
    }

    /// Add or update a scaling policy
    pub fn add_policy(&mut self, policy: ScalingPolicy) {
        self.policies.insert(policy.name.clone(), policy);
    }

    /// Get a policy by name
    pub fn get_policy(&self, name: &str) -> Option<&ScalingPolicy> {
        self.policies.get(name)
    }

    /// Get default policy for a resource type
    pub fn get_default_policy(&self, resource_type: ResourceType) -> Option<&ScalingPolicy> {
        self.default_policies.get(&resource_type)
    }

    /// Get all policies for a resource type
    pub fn get_policies_for_resource(&self, resource_type: ResourceType) -> Vec<&ScalingPolicy> {
        self.policies
            .values()
            .filter(|policy| policy.resource_type == resource_type)
            .collect()
    }

    /// Remove a policy
    pub fn remove_policy(&mut self, name: &str) -> Option<ScalingPolicy> {
        self.policies.remove(name)
    }

    /// Update policy configuration
    pub fn update_policy<F>(&mut self, name: &str, update_fn: F) -> Result<(), String>
    where
        F: FnOnce(&mut ScalingPolicy),
    {
        if let Some(policy) = self.policies.get_mut(name) {
            update_fn(policy);
            Ok(())
        } else {
            Err(format!("Policy '{}' not found", name))
        }
    }

    /// Validate policy configuration
    pub fn validate_policy(&self, policy: &ScalingPolicy) -> Result<(), String> {
        // Validate thresholds
        if policy.thresholds.scale_up_threshold <= policy.thresholds.scale_down_threshold {
            return Err("Scale up threshold must be greater than scale down threshold".to_string());
        }

        if policy.thresholds.emergency_threshold <= policy.thresholds.scale_up_threshold {
            return Err("Emergency threshold must be greater than scale up threshold".to_string());
        }

        // Validate constraints
        if policy.constraints.min_capacity >= policy.constraints.max_capacity {
            return Err("Minimum capacity must be less than maximum capacity".to_string());
        }

        if policy.constraints.max_scale_step <= policy.constraints.min_scale_step {
            return Err("Maximum scale step must be greater than minimum scale step".to_string());
        }

        // Validate behavior
        match &policy.behavior.reactive.aggregation {
            MetricAggregation::Percentile(p) if *p > 100 => {
                return Err("Percentile must be between 0 and 100".to_string());
            }
            _ => {}
        }

        Ok(())
    }

    /// Export policies to JSON
    pub fn export_policies(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.policies)
    }

    /// Import policies from JSON
    pub fn import_policies(&mut self, json: &str) -> Result<(), serde_json::Error> {
        let policies: HashMap<String, ScalingPolicy> = serde_json::from_str(json)?;
        
        for (name, policy) in policies {
            if let Err(e) = self.validate_policy(&policy) {
                eprintln!("Warning: Invalid policy '{}': {}", name, e);
                continue;
            }
            self.policies.insert(name, policy);
        }

        Ok(())
    }

    /// Get policy statistics
    pub fn get_policy_stats(&self) -> PolicyStats {
        let total_policies = self.policies.len();
        let enabled_policies = self.policies.values().filter(|p| p.enabled).count();
        let disabled_policies = total_policies - enabled_policies;

        let mut policies_by_resource = HashMap::new();
        let mut policies_by_strategy = HashMap::new();

        for policy in self.policies.values() {
            *policies_by_resource.entry(policy.resource_type).or_insert(0) += 1;
            *policies_by_strategy.entry(policy.behavior.strategy.clone()).or_insert(0) += 1;
        }

        PolicyStats {
            total_policies,
            enabled_policies,
            disabled_policies,
            policies_by_resource,
            policies_by_strategy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyStats {
    pub total_policies: usize,
    pub enabled_policies: usize,
    pub disabled_policies: usize,
    pub policies_by_resource: HashMap<ResourceType, usize>,
    pub policies_by_strategy: HashMap<ScalingStrategy, usize>,
}

impl Default for ScalingPolicyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_manager_initialization() {
        let manager = ScalingPolicyManager::new();
        
        // Check that default policies are created
        assert!(manager.get_default_policy(ResourceType::Cpu).is_some());
        assert!(manager.get_default_policy(ResourceType::Memory).is_some());
        assert!(manager.get_default_policy(ResourceType::Network).is_some());
        assert!(manager.get_default_policy(ResourceType::ThreadPool).is_some());
    }

    #[test]
    fn test_policy_validation() {
        let manager = ScalingPolicyManager::new();
        
        // Valid policy
        let valid_policy = ScalingPolicy {
            name: "test-policy".to_string(),
            resource_type: ResourceType::Cpu,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
                emergency_threshold: 0.95,
                sustained_duration: Duration::from_secs(60),
            },
            constraints: ScalingConstraints {
                min_capacity: 2,
                max_capacity: 16,
                max_scale_step: 0.5,
                min_scale_step: 0.1,
                cooldown_period: Duration::from_secs(300),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(300),
                    min_data_points: 5,
                    aggregation: MetricAggregation::Average,
                },
            },
        };

        assert!(manager.validate_policy(&valid_policy).is_ok());

        // Invalid policy - thresholds
        let mut invalid_policy = valid_policy.clone();
        invalid_policy.thresholds.scale_up_threshold = 0.2; // Less than scale_down_threshold
        assert!(manager.validate_policy(&invalid_policy).is_err());
    }

    #[test]
    fn test_policy_crud_operations() {
        let mut manager = ScalingPolicyManager::new();
        
        let policy = ScalingPolicy {
            name: "test-policy".to_string(),
            resource_type: ResourceType::Cpu,
            enabled: true,
            thresholds: ScalingThresholds {
                scale_up_threshold: 0.8,
                scale_down_threshold: 0.3,
                emergency_threshold: 0.95,
                sustained_duration: Duration::from_secs(60),
            },
            constraints: ScalingConstraints {
                min_capacity: 2,
                max_capacity: 16,
                max_scale_step: 0.5,
                min_scale_step: 0.1,
                cooldown_period: Duration::from_secs(300),
            },
            behavior: ScalingBehavior {
                strategy: ScalingStrategy::Reactive,
                predictive: None,
                reactive: ReactiveScaling {
                    evaluation_window: Duration::from_secs(300),
                    min_data_points: 5,
                    aggregation: MetricAggregation::Average,
                },
            },
        };

        // Add policy
        manager.add_policy(policy.clone());
        assert!(manager.get_policy("test-policy").is_some());

        // Update policy
        let result = manager.update_policy("test-policy", |p| {
            p.enabled = false;
        });
        assert!(result.is_ok());
        assert!(!manager.get_policy("test-policy").unwrap().enabled);

        // Remove policy
        let removed = manager.remove_policy("test-policy");
        assert!(removed.is_some());
        assert!(manager.get_policy("test-policy").is_none());
    }
}