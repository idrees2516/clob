use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use crate::performance::scaling::{ResourceType, ScalingDirection};

/// Resource provisioning interface
#[async_trait::async_trait]
pub trait ResourceProvisioner: Send + Sync {
    async fn provision_cpu(&self, target_cores: u64) -> Result<(), String>;
    async fn provision_memory(&self, target_mb: u64) -> Result<(), String>;
    async fn provision_network(&self, target_bandwidth_mbps: u64) -> Result<(), String>;
    async fn provision_threads(&self, target_threads: u64) -> Result<(), String>;
    
    fn get_current_capacity(&self, resource_type: ResourceType) -> u64;
    fn get_provisioning_limits(&self, resource_type: ResourceType) -> (u64, u64); // (min, max)
}

/// Resource scaling automation with advanced decision making
pub struct ResourceScaler {
    provisioners: HashMap<ResourceType, Arc<dyn ResourceProvisioner>>,
    scaling_history: Arc<RwLock<Vec<ScalingOperation>>>,
    total_scaling_operations: AtomicU64,
    scaling_policies: Arc<RwLock<HashMap<ResourceType, ScalingDecisionPolicy>>>,
    load_predictor: Arc<LoadPredictor>,
}

#[derive(Debug, Clone)]
pub struct ScalingOperation {
    pub resource_type: ResourceType,
    pub direction: ScalingDirection,
    pub from_capacity: u64,
    pub to_capacity: u64,
    pub duration_ms: u64,
    pub success: bool,
    pub timestamp: std::time::Instant,
}

impl ResourceScaler {
    pub fn new() -> Self {
        Self {
            provisioners: HashMap::new(),
            scaling_history: Arc::new(RwLock::new(Vec::new())),
            total_scaling_operations: AtomicU64::new(0),
        }
    }

    /// Register a resource provisioner
    pub fn register_provisioner(
        &mut self,
        resource_type: ResourceType,
        provisioner: Arc<dyn ResourceProvisioner>,
    ) {
        self.provisioners.insert(resource_type, provisioner);
    }

    /// Scale a specific resource to target capacity
    pub async fn scale_resource(
        &self,
        resource_type: ResourceType,
        target_capacity: u64,
    ) -> Result<(), String> {
        let provisioner = self.provisioners.get(&resource_type)
            .ok_or_else(|| format!("No provisioner registered for {:?}", resource_type))?;

        let current_capacity = provisioner.get_current_capacity(resource_type);
        let (min_capacity, max_capacity) = provisioner.get_provisioning_limits(resource_type);

        // Validate target capacity
        if target_capacity < min_capacity || target_capacity > max_capacity {
            return Err(format!(
                "Target capacity {} outside limits [{}, {}] for {:?}",
                target_capacity, min_capacity, max_capacity, resource_type
            ));
        }

        if target_capacity == current_capacity {
            return Ok(()); // No scaling needed
        }

        let direction = if target_capacity > current_capacity {
            ScalingDirection::Up
        } else {
            ScalingDirection::Down
        };

        let start_time = std::time::Instant::now();
        
        // Perform the scaling operation
        let result = match resource_type {
            ResourceType::Cpu => provisioner.provision_cpu(target_capacity).await,
            ResourceType::Memory => provisioner.provision_memory(target_capacity).await,
            ResourceType::Network => provisioner.provision_network(target_capacity).await,
            ResourceType::ThreadPool => provisioner.provision_threads(target_capacity).await,
        };

        let duration = start_time.elapsed();
        let success = result.is_ok();

        // Record scaling operation
        let operation = ScalingOperation {
            resource_type,
            direction,
            from_capacity: current_capacity,
            to_capacity: target_capacity,
            duration_ms: duration.as_millis() as u64,
            success,
            timestamp: start_time,
        };

        self.record_scaling_operation(operation).await;
        self.total_scaling_operations.fetch_add(1, Ordering::Relaxed);

        result
    }

    /// Record scaling operation in history
    async fn record_scaling_operation(&self, operation: ScalingOperation) {
        let mut history = self.scaling_history.write().await;
        history.push(operation);

        // Keep only last 1000 operations
        if history.len() > 1000 {
            history.drain(0..history.len() - 1000);
        }
    }

    /// Get scaling history for analysis
    pub async fn get_scaling_history(&self) -> Vec<ScalingOperation> {
        self.scaling_history.read().await.clone()
    }

    /// Get scaling statistics
    pub async fn get_scaling_stats(&self) -> ScalingStats {
        let history = self.scaling_history.read().await;
        let total_operations = self.total_scaling_operations.load(Ordering::Relaxed);

        let mut stats = ScalingStats {
            total_operations,
            successful_operations: 0,
            failed_operations: 0,
            average_duration_ms: 0.0,
            operations_by_resource: HashMap::new(),
            operations_by_direction: HashMap::new(),
        };

        if history.is_empty() {
            return stats;
        }

        let mut total_duration = 0u64;
        
        for operation in history.iter() {
            if operation.success {
                stats.successful_operations += 1;
            } else {
                stats.failed_operations += 1;
            }

            total_duration += operation.duration_ms;

            *stats.operations_by_resource.entry(operation.resource_type).or_insert(0) += 1;
            *stats.operations_by_direction.entry(operation.direction).or_insert(0) += 1;
        }

        stats.average_duration_ms = total_duration as f64 / history.len() as f64;

        stats
    }

    /// Check if resource can be scaled
    pub fn can_scale_resource(&self, resource_type: ResourceType) -> bool {
        self.provisioners.contains_key(&resource_type)
    }

    /// Get current capacity for all resources
    pub fn get_all_capacities(&self) -> HashMap<ResourceType, u64> {
        let mut capacities = HashMap::new();
        
        for (resource_type, provisioner) in &self.provisioners {
            let capacity = provisioner.get_current_capacity(*resource_type);
            capacities.insert(*resource_type, capacity);
        }

        capacities
    }
}

#[derive(Debug, Clone)]
pub struct ScalingStats {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_duration_ms: f64,
    pub operations_by_resource: HashMap<ResourceType, u64>,
    pub operations_by_direction: HashMap<ScalingDirection, u64>,
}

/// Default CPU provisioner implementation
pub struct CpuProvisioner {
    current_cores: AtomicU64,
    min_cores: u64,
    max_cores: u64,
}

impl CpuProvisioner {
    pub fn new(initial_cores: u64, min_cores: u64, max_cores: u64) -> Self {
        Self {
            current_cores: AtomicU64::new(initial_cores),
            min_cores,
            max_cores,
        }
    }
}

#[async_trait::async_trait]
impl ResourceProvisioner for CpuProvisioner {
    async fn provision_cpu(&self, target_cores: u64) -> Result<(), String> {
        if target_cores < self.min_cores || target_cores > self.max_cores {
            return Err(format!(
                "Target cores {} outside limits [{}, {}]",
                target_cores, self.min_cores, self.max_cores
            ));
        }

        // Simulate CPU provisioning delay
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        self.current_cores.store(target_cores, Ordering::Relaxed);
        Ok(())
    }

    async fn provision_memory(&self, _target_mb: u64) -> Result<(), String> {
        Err("CPU provisioner cannot provision memory".to_string())
    }

    async fn provision_network(&self, _target_bandwidth_mbps: u64) -> Result<(), String> {
        Err("CPU provisioner cannot provision network".to_string())
    }

    async fn provision_threads(&self, _target_threads: u64) -> Result<(), String> {
        Err("CPU provisioner cannot provision threads".to_string())
    }

    fn get_current_capacity(&self, resource_type: ResourceType) -> u64 {
        match resource_type {
            ResourceType::Cpu => self.current_cores.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    fn get_provisioning_limits(&self, resource_type: ResourceType) -> (u64, u64) {
        match resource_type {
            ResourceType::Cpu => (self.min_cores, self.max_cores),
            _ => (0, 0),
        }
    }
}

/// Default memory provisioner implementation
pub struct MemoryProvisioner {
    current_memory_mb: AtomicU64,
    min_memory_mb: u64,
    max_memory_mb: u64,
}

impl MemoryProvisioner {
    pub fn new(initial_memory_mb: u64, min_memory_mb: u64, max_memory_mb: u64) -> Self {
        Self {
            current_memory_mb: AtomicU64::new(initial_memory_mb),
            min_memory_mb,
            max_memory_mb,
        }
    }
}

#[async_trait::async_trait]
impl ResourceProvisioner for MemoryProvisioner {
    async fn provision_cpu(&self, _target_cores: u64) -> Result<(), String> {
        Err("Memory provisioner cannot provision CPU".to_string())
    }

    async fn provision_memory(&self, target_mb: u64) -> Result<(), String> {
        if target_mb < self.min_memory_mb || target_mb > self.max_memory_mb {
            return Err(format!(
                "Target memory {} MB outside limits [{}, {}]",
                target_mb, self.min_memory_mb, self.max_memory_mb
            ));
        }

        // Simulate memory provisioning delay
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        
        self.current_memory_mb.store(target_mb, Ordering::Relaxed);
        Ok(())
    }

    async fn provision_network(&self, _target_bandwidth_mbps: u64) -> Result<(), String> {
        Err("Memory provisioner cannot provision network".to_string())
    }

    async fn provision_threads(&self, _target_threads: u64) -> Result<(), String> {
        Err("Memory provisioner cannot provision threads".to_string())
    }

    fn get_current_capacity(&self, resource_type: ResourceType) -> u64 {
        match resource_type {
            ResourceType::Memory => self.current_memory_mb.load(Ordering::Relaxed),
            _ => 0,
        }
    }

    fn get_provisioning_limits(&self, resource_type: ResourceType) -> (u64, u64) {
        match resource_type {
            ResourceType::Memory => (self.min_memory_mb, self.max_memory_mb),
            _ => (0, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_scaler_cpu_provisioning() {
        let mut scaler = ResourceScaler::new();
        let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
        
        scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner.clone());

        // Test scaling up
        let result = scaler.scale_resource(ResourceType::Cpu, 8).await;
        assert!(result.is_ok());
        assert_eq!(cpu_provisioner.get_current_capacity(ResourceType::Cpu), 8);

        // Test scaling down
        let result = scaler.scale_resource(ResourceType::Cpu, 6).await;
        assert!(result.is_ok());
        assert_eq!(cpu_provisioner.get_current_capacity(ResourceType::Cpu), 6);

        // Test invalid scaling
        let result = scaler.scale_resource(ResourceType::Cpu, 20).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scaling_history_tracking() {
        let mut scaler = ResourceScaler::new();
        let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
        
        scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner);

        // Perform multiple scaling operations
        let _ = scaler.scale_resource(ResourceType::Cpu, 8).await;
        let _ = scaler.scale_resource(ResourceType::Cpu, 6).await;
        let _ = scaler.scale_resource(ResourceType::Cpu, 10).await;

        let history = scaler.get_scaling_history().await;
        assert_eq!(history.len(), 3);

        let stats = scaler.get_scaling_stats().await;
        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.successful_operations, 3);
        assert_eq!(stats.failed_operations, 0);
    }
}