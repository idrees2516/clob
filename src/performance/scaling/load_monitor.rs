use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use crate::performance::scaling::{ResourceType, ResourceUtilization};

/// Load monitoring and metrics collection
pub struct LoadMonitor {
    metrics_collectors: HashMap<ResourceType, Arc<dyn MetricsCollector>>,
    historical_data: Arc<RwLock<HashMap<ResourceType, VecDeque<ResourceUtilization>>>>,
    collection_interval: Duration,
    history_retention: Duration,
    enabled: AtomicBool,
}

/// Trait for collecting resource metrics
#[async_trait::async_trait]
pub trait MetricsCollector: Send + Sync {
    async fn collect_metrics(&self) -> Result<ResourceUtilization, String>;
    fn get_resource_type(&self) -> ResourceType;
}

impl LoadMonitor {
    pub fn new() -> Self {
        Self {
            metrics_collectors: HashMap::new(),
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            collection_interval: Duration::from_secs(10),
            history_retention: Duration::from_hours(24),
            enabled: AtomicBool::new(true),
        }
    }

    /// Register a metrics collector for a resource type
    pub fn register_collector(&mut self, collector: Arc<dyn MetricsCollector>) {
        let resource_type = collector.get_resource_type();
        self.metrics_collectors.insert(resource_type, collector);
    }

    /// Start monitoring loop
    pub async fn start_monitoring(&self) {
        let mut interval = tokio::time::interval(self.collection_interval);
        
        while self.enabled.load(Ordering::Relaxed) {
            interval.tick().await;
            self.collect_all_metrics().await;
            self.cleanup_old_data().await;
        }
    }

    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Collect metrics from all registered collectors
    async fn collect_all_metrics(&self) {
        let mut tasks = Vec::new();
        
        for (resource_type, collector) in &self.metrics_collectors {
            let collector = Arc::clone(collector);
            let resource_type = *resource_type;
            let historical_data = Arc::clone(&self.historical_data);
            
            let task = tokio::spawn(async move {
                match collector.collect_metrics().await {
                    Ok(utilization) => {
                        let mut data = historical_data.write().await;
                        let resource_data = data.entry(resource_type).or_insert_with(VecDeque::new);
                        resource_data.push_back(utilization);
                        
                        // Limit history size
                        if resource_data.len() > 10000 {
                            resource_data.pop_front();
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to collect metrics for {:?}: {}", resource_type, e);
                    }
                }
            });
            
            tasks.push(task);
        }
        
        // Wait for all collection tasks to complete
        for task in tasks {
            let _ = task.await;
        }
    }

    /// Clean up old historical data
    async fn cleanup_old_data(&self) {
        let cutoff = Instant::now() - self.history_retention;
        let mut data = self.historical_data.write().await;
        
        for resource_data in data.values_mut() {
            while let Some(front) = resource_data.front() {
                if front.timestamp < cutoff {
                    resource_data.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Get current utilization for all resources
    pub fn get_current_utilization(&self) -> Vec<ResourceUtilization> {
        let mut utilizations = Vec::new();
        
        // This would typically be implemented with a non-blocking read
        // For now, we'll use a simplified approach
        if let Ok(data) = self.historical_data.try_read() {
            for resource_data in data.values() {
                if let Some(latest) = resource_data.back() {
                    utilizations.push(latest.clone());
                }
            }
        }
        
        utilizations
    }

    /// Get historical utilization for a specific resource
    pub async fn get_historical_utilization(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> Vec<ResourceUtilization> {
        let data = self.historical_data.read().await;
        let cutoff = Instant::now() - duration;
        
        if let Some(resource_data) = data.get(&resource_type) {
            resource_data
                .iter()
                .filter(|u| u.timestamp >= cutoff)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate average utilization over a time period
    pub async fn get_average_utilization(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> Option<f64> {
        let utilizations = self.get_historical_utilization(resource_type, duration).await;
        
        if utilizations.is_empty() {
            return None;
        }
        
        let sum: f64 = utilizations.iter().map(|u| u.current_utilization).sum();
        Some(sum / utilizations.len() as f64)
    }

    /// Get peak utilization over a time period
    pub async fn get_peak_utilization(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> Option<f64> {
        let utilizations = self.get_historical_utilization(resource_type, duration).await;
        
        utilizations
            .iter()
            .map(|u| u.current_utilization)
            .fold(None, |max, val| match max {
                None => Some(val),
                Some(max_val) => Some(max_val.max(val)),
            })
    }

    /// Detect utilization trends
    pub async fn detect_trend(
        &self,
        resource_type: ResourceType,
        duration: Duration,
    ) -> Option<UtilizationTrend> {
        let utilizations = self.get_historical_utilization(resource_type, duration).await;
        
        if utilizations.len() < 3 {
            return None;
        }
        
        // Simple linear regression to detect trend
        let n = utilizations.len() as f64;
        let x_sum: f64 = (0..utilizations.len()).map(|i| i as f64).sum();
        let y_sum: f64 = utilizations.iter().map(|u| u.current_utilization).sum();
        let xy_sum: f64 = utilizations
            .iter()
            .enumerate()
            .map(|(i, u)| i as f64 * u.current_utilization)
            .sum();
        let x_squared_sum: f64 = (0..utilizations.len()).map(|i| (i as f64).powi(2)).sum();
        
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_squared_sum - x_sum.powi(2));
        let intercept = (y_sum - slope * x_sum) / n;
        
        let trend_direction = if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };
        
        Some(UtilizationTrend {
            resource_type,
            direction: trend_direction,
            slope,
            intercept,
            confidence: calculate_trend_confidence(&utilizations, slope, intercept),
            duration,
        })
    }

    /// Get monitoring statistics
    pub async fn get_monitoring_stats(&self) -> MonitoringStats {
        let data = self.historical_data.read().await;
        let mut stats = MonitoringStats {
            total_data_points: 0,
            data_points_by_resource: HashMap::new(),
            oldest_data_point: None,
            newest_data_point: None,
            collection_interval: self.collection_interval,
            enabled: self.enabled.load(Ordering::Relaxed),
        };
        
        for (resource_type, resource_data) in data.iter() {
            let count = resource_data.len();
            stats.total_data_points += count;
            stats.data_points_by_resource.insert(*resource_type, count);
            
            if let Some(oldest) = resource_data.front() {
                match stats.oldest_data_point {
                    None => stats.oldest_data_point = Some(oldest.timestamp),
                    Some(current_oldest) => {
                        if oldest.timestamp < current_oldest {
                            stats.oldest_data_point = Some(oldest.timestamp);
                        }
                    }
                }
            }
            
            if let Some(newest) = resource_data.back() {
                match stats.newest_data_point {
                    None => stats.newest_data_point = Some(newest.timestamp),
                    Some(current_newest) => {
                        if newest.timestamp > current_newest {
                            stats.newest_data_point = Some(newest.timestamp);
                        }
                    }
                }
            }
        }
        
        stats
    }
}

/// CPU metrics collector
pub struct CpuMetricsCollector {
    cpu_count: u64,
}

impl CpuMetricsCollector {
    pub fn new(cpu_count: u64) -> Self {
        Self { cpu_count }
    }
}

#[async_trait::async_trait]
impl MetricsCollector for CpuMetricsCollector {
    async fn collect_metrics(&self) -> Result<ResourceUtilization, String> {
        // In a real implementation, this would collect actual CPU metrics
        // For now, we'll simulate CPU utilization
        let utilization = simulate_cpu_utilization();
        
        Ok(ResourceUtilization {
            resource_type: ResourceType::Cpu,
            current_utilization: utilization,
            capacity: self.cpu_count,
            allocated: (utilization * self.cpu_count as f64) as u64,
            timestamp: Instant::now(),
        })
    }

    fn get_resource_type(&self) -> ResourceType {
        ResourceType::Cpu
    }
}

/// Memory metrics collector
pub struct MemoryMetricsCollector {
    total_memory_mb: u64,
}

impl MemoryMetricsCollector {
    pub fn new(total_memory_mb: u64) -> Self {
        Self { total_memory_mb }
    }
}

#[async_trait::async_trait]
impl MetricsCollector for MemoryMetricsCollector {
    async fn collect_metrics(&self) -> Result<ResourceUtilization, String> {
        let utilization = simulate_memory_utilization();
        
        Ok(ResourceUtilization {
            resource_type: ResourceType::Memory,
            current_utilization: utilization,
            capacity: self.total_memory_mb,
            allocated: (utilization * self.total_memory_mb as f64) as u64,
            timestamp: Instant::now(),
        })
    }

    fn get_resource_type(&self) -> ResourceType {
        ResourceType::Memory
    }
}

/// Network metrics collector
pub struct NetworkMetricsCollector {
    bandwidth_mbps: u64,
}

impl NetworkMetricsCollector {
    pub fn new(bandwidth_mbps: u64) -> Self {
        Self { bandwidth_mbps }
    }
}

#[async_trait::async_trait]
impl MetricsCollector for NetworkMetricsCollector {
    async fn collect_metrics(&self) -> Result<ResourceUtilization, String> {
        let utilization = simulate_network_utilization();
        
        Ok(ResourceUtilization {
            resource_type: ResourceType::Network,
            current_utilization: utilization,
            capacity: self.bandwidth_mbps,
            allocated: (utilization * self.bandwidth_mbps as f64) as u64,
            timestamp: Instant::now(),
        })
    }

    fn get_resource_type(&self) -> ResourceType {
        ResourceType::Network
    }
}

/// Utilization trend analysis
#[derive(Debug, Clone)]
pub struct UtilizationTrend {
    pub resource_type: ResourceType,
    pub direction: TrendDirection,
    pub slope: f64,
    pub intercept: f64,
    pub confidence: f64,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Monitoring statistics
#[derive(Debug, Clone)]
pub struct MonitoringStats {
    pub total_data_points: usize,
    pub data_points_by_resource: HashMap<ResourceType, usize>,
    pub oldest_data_point: Option<Instant>,
    pub newest_data_point: Option<Instant>,
    pub collection_interval: Duration,
    pub enabled: bool,
}

/// Calculate confidence in trend analysis
fn calculate_trend_confidence(
    utilizations: &[ResourceUtilization],
    slope: f64,
    intercept: f64,
) -> f64 {
    if utilizations.len() < 3 {
        return 0.0;
    }
    
    let mut sum_squared_errors = 0.0;
    let mut sum_squared_total = 0.0;
    let mean_y: f64 = utilizations.iter().map(|u| u.current_utilization).sum::<f64>() / utilizations.len() as f64;
    
    for (i, utilization) in utilizations.iter().enumerate() {
        let predicted = slope * i as f64 + intercept;
        let actual = utilization.current_utilization;
        
        sum_squared_errors += (actual - predicted).powi(2);
        sum_squared_total += (actual - mean_y).powi(2);
    }
    
    if sum_squared_total == 0.0 {
        return 1.0;
    }
    
    let r_squared = 1.0 - (sum_squared_errors / sum_squared_total);
    r_squared.max(0.0).min(1.0)
}

/// Simulate CPU utilization (replace with actual system metrics)
fn simulate_cpu_utilization() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now().hash(&mut hasher);
    let hash = hasher.finish();
    
    // Generate pseudo-random utilization between 0.1 and 0.9
    0.1 + (hash % 80) as f64 / 100.0
}

/// Simulate memory utilization (replace with actual system metrics)
fn simulate_memory_utilization() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    (std::time::SystemTime::now(), "memory").hash(&mut hasher);
    let hash = hasher.finish();
    
    // Generate pseudo-random utilization between 0.2 and 0.8
    0.2 + (hash % 60) as f64 / 100.0
}

/// Simulate network utilization (replace with actual system metrics)
fn simulate_network_utilization() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    (std::time::SystemTime::now(), "network").hash(&mut hasher);
    let hash = hasher.finish();
    
    // Generate pseudo-random utilization between 0.05 and 0.7
    0.05 + (hash % 65) as f64 / 100.0
}

impl Default for LoadMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collection() {
        let mut monitor = LoadMonitor::new();
        let cpu_collector = Arc::new(CpuMetricsCollector::new(8));
        
        monitor.register_collector(cpu_collector);
        
        // Collect metrics once
        monitor.collect_all_metrics().await;
        
        let utilizations = monitor.get_current_utilization();
        assert_eq!(utilizations.len(), 1);
        assert_eq!(utilizations[0].resource_type, ResourceType::Cpu);
    }

    #[tokio::test]
    async fn test_historical_data() {
        let mut monitor = LoadMonitor::new();
        let cpu_collector = Arc::new(CpuMetricsCollector::new(8));
        
        monitor.register_collector(cpu_collector);
        
        // Collect multiple data points
        for _ in 0..5 {
            monitor.collect_all_metrics().await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        let history = monitor.get_historical_utilization(
            ResourceType::Cpu,
            Duration::from_secs(1),
        ).await;
        
        assert!(history.len() >= 5);
    }

    #[tokio::test]
    async fn test_trend_detection() {
        let mut monitor = LoadMonitor::new();
        
        // Manually add trending data
        let mut data = monitor.historical_data.write().await;
        let mut cpu_data = VecDeque::new();
        
        // Add increasing trend data
        for i in 0..10 {
            cpu_data.push_back(ResourceUtilization {
                resource_type: ResourceType::Cpu,
                current_utilization: 0.1 + (i as f64 * 0.05),
                capacity: 8,
                allocated: 1 + i,
                timestamp: Instant::now(),
            });
        }
        
        data.insert(ResourceType::Cpu, cpu_data);
        drop(data);
        
        let trend = monitor.detect_trend(ResourceType::Cpu, Duration::from_secs(60)).await;
        assert!(trend.is_some());
        
        let trend = trend.unwrap();
        assert_eq!(trend.direction, TrendDirection::Increasing);
        assert!(trend.slope > 0.0);
    }

    #[test]
    fn test_trend_confidence_calculation() {
        let utilizations = vec![
            ResourceUtilization {
                resource_type: ResourceType::Cpu,
                current_utilization: 0.1,
                capacity: 8,
                allocated: 1,
                timestamp: Instant::now(),
            },
            ResourceUtilization {
                resource_type: ResourceType::Cpu,
                current_utilization: 0.2,
                capacity: 8,
                allocated: 2,
                timestamp: Instant::now(),
            },
            ResourceUtilization {
                resource_type: ResourceType::Cpu,
                current_utilization: 0.3,
                capacity: 8,
                allocated: 3,
                timestamp: Instant::now(),
            },
        ];
        
        let confidence = calculate_trend_confidence(&utilizations, 0.1, 0.1);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}