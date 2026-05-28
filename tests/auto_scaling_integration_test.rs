use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use trading_system::performance::scaling::{
    AutoScaler, ScalingConfig, ResourceScaler, LoadMonitor, ScalingEventLogger,
    CpuUtilizationManager, MemoryPoolManager, NetworkBandwidthManager,
    CpuProvisioner, MemoryProvisioner, ResourceType, PoolExpansionPolicy,
    ExpandableLockFreePool, TrafficShapingConfig, QueueConfig, DropPolicy,
};

#[tokio::test]
async fn test_auto_scaling_integration() {
    // Initialize auto-scaling system
    let scaling_config = ScalingConfig {
        min_scaling_interval: Duration::from_millis(100),
        scale_up_threshold: 0.8,
        scale_down_threshold: 0.3,
        cooldown_period: Duration::from_millis(500),
        enabled: true,
    };

    let resource_scaler = Arc::new(ResourceScaler::new());
    let load_monitor = Arc::new(LoadMonitor::new());
    let scaling_events = Arc::new(ScalingEventLogger::new());

    // Set up resource provisioners
    let mut scaler = unsafe { &mut *(Arc::as_ptr(&resource_scaler) as *mut ResourceScaler) };
    let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
    scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner);

    let auto_scaler = AutoScaler::new(
        scaling_config,
        Arc::clone(&resource_scaler),
        Arc::clone(&load_monitor),
        Arc::clone(&scaling_events),
    );

    // Test that auto-scaler is properly initialized
    assert!(auto_scaler.is_enabled());

    // Test scaling decision evaluation (should be empty initially)
    let mut auto_scaler = auto_scaler;
    let decisions = auto_scaler.evaluate_scaling();
    assert!(decisions.is_empty()); // No utilization data yet

    println!("✅ Auto-scaling integration test passed");
}

#[tokio::test]
async fn test_cpu_utilization_management() {
    let cpu_manager = CpuUtilizationManager::new(4);
    
    // Test initial state
    let stats = cpu_manager.get_cpu_stats();
    assert_eq!(stats.core_count, 4);
    assert_eq!(stats.core_stats.len(), 4);
    
    // Test load imbalance detection
    let imbalance = stats.load_imbalance;
    assert!(imbalance.mean_utilization >= 0.0);
    assert!(imbalance.standard_deviation >= 0.0);

    println!("✅ CPU utilization management test passed");
}

#[tokio::test]
async fn test_memory_pool_management() {
    let mut memory_manager = MemoryPoolManager::new();
    
    // Register a test pool
    let pool = Arc::new(ExpandableLockFreePool::<u64>::new(100, 1000));
    let policy = PoolExpansionPolicy::default();
    
    memory_manager.register_pool("test_pool".to_string(), pool, Some(policy));
    
    // Test pool statistics
    let stats = memory_manager.get_pool_stats();
    assert_eq!(stats.len(), 1);
    assert!(stats.contains_key("test_pool"));
    
    let pool_stats = &stats["test_pool"];
    assert_eq!(pool_stats.name, "test_pool");
    assert_eq!(pool_stats.current_size, 100);
    assert_eq!(pool_stats.max_size, 1000);

    println!("✅ Memory pool management test passed");
}

#[tokio::test]
async fn test_network_bandwidth_management() {
    let mut network_manager = NetworkBandwidthManager::new();
    
    // Register test interfaces
    network_manager.register_interface("test_eth0".to_string(), 1000);
    network_manager.register_interface("test_eth1".to_string(), 10000);
    
    // Test interface statistics
    let stats = network_manager.get_bandwidth_stats();
    assert_eq!(stats.len(), 2);
    assert!(stats.contains_key("test_eth0"));
    assert!(stats.contains_key("test_eth1"));
    
    let eth0_stats = &stats["test_eth0"];
    assert_eq!(eth0_stats.name, "test_eth0");
    assert_eq!(eth0_stats.max_bandwidth_mbps, 1000);
    
    // Test traffic shaping configuration
    let shaping_config = TrafficShapingConfig {
        max_bandwidth_mbps: 800,
        burst_size_kb: 64,
        priority_queues: vec![
            QueueConfig {
                name: "high_priority".to_string(),
                priority: 0,
                guaranteed_bandwidth_mbps: 200,
                max_bandwidth_mbps: 400,
                max_queue_size: 1000,
                drop_policy: DropPolicy::TailDrop,
            },
        ],
        rate_limiting_enabled: true,
        congestion_control_enabled: true,
    };
    
    let result = network_manager.configure_traffic_shaping("test_eth0".to_string(), shaping_config).await;
    assert!(result.is_ok());

    println!("✅ Network bandwidth management test passed");
}

#[tokio::test]
async fn test_scaling_event_logging() {
    let logger = ScalingEventLogger::new();
    
    // Create test scaling event
    let event = trading_system::performance::scaling::ScalingEvent {
        resource_type: ResourceType::Cpu,
        direction: trading_system::performance::scaling::ScalingDirection::Up,
        from_capacity: 4,
        to_capacity: 8,
        reason: "Test scaling event".to_string(),
        success: true,
        timestamp: std::time::Instant::now(),
    };
    
    // Log the event
    logger.log_event(event).await;
    
    // Verify event was logged
    let events = logger.get_recent_events().await;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event.resource_type, ResourceType::Cpu);
    assert_eq!(events[0].event.from_capacity, 4);
    assert_eq!(events[0].event.to_capacity, 8);
    
    // Test event analysis
    let analysis = logger.analyze_scaling_patterns().await;
    assert_eq!(analysis.total_events, 1);
    assert_eq!(analysis.successful_events, 1);
    assert_eq!(analysis.failed_events, 0);

    println!("✅ Scaling event logging test passed");
}

#[tokio::test]
async fn test_resource_provisioning() {
    let mut resource_scaler = ResourceScaler::new();
    
    // Register CPU provisioner
    let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
    resource_scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner);
    
    // Test scaling operations
    let result = resource_scaler.scale_resource(ResourceType::Cpu, 8).await;
    assert!(result.is_ok());
    
    // Test invalid scaling
    let result = resource_scaler.scale_resource(ResourceType::Cpu, 20).await;
    assert!(result.is_err());
    
    // Test scaling statistics
    let stats = resource_scaler.get_scaling_stats().await;
    assert_eq!(stats.total_operations, 2);
    assert_eq!(stats.successful_operations, 1);
    assert_eq!(stats.failed_operations, 1);

    println!("✅ Resource provisioning test passed");
}

#[tokio::test]
async fn test_end_to_end_scaling_workflow() {
    // This test simulates a complete scaling workflow
    let scaling_config = ScalingConfig {
        min_scaling_interval: Duration::from_millis(10),
        scale_up_threshold: 0.7,
        scale_down_threshold: 0.3,
        cooldown_period: Duration::from_millis(50),
        enabled: true,
    };

    let resource_scaler = Arc::new(ResourceScaler::new());
    let load_monitor = Arc::new(LoadMonitor::new());
    let scaling_events = Arc::new(ScalingEventLogger::new());

    // Set up provisioners
    let mut scaler = unsafe { &mut *(Arc::as_ptr(&resource_scaler) as *mut ResourceScaler) };
    let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
    scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner);

    let mut auto_scaler = AutoScaler::new(
        scaling_config,
        Arc::clone(&resource_scaler),
        Arc::clone(&load_monitor),
        Arc::clone(&scaling_events),
    );

    // Initialize other managers
    let cpu_manager = CpuUtilizationManager::new(4);
    let mut memory_manager = MemoryPoolManager::new();
    let pool = Arc::new(ExpandableLockFreePool::<u64>::new(100, 1000));
    memory_manager.register_pool("test".to_string(), pool, None);
    
    let mut network_manager = NetworkBandwidthManager::new();
    network_manager.register_interface("test_eth".to_string(), 1000);

    // Test that all components are working together
    let cpu_stats = cpu_manager.get_cpu_stats();
    assert_eq!(cpu_stats.core_count, 4);
    
    let pool_stats = memory_manager.get_pool_stats();
    assert_eq!(pool_stats.len(), 1);
    
    let network_stats = network_manager.get_bandwidth_stats();
    assert_eq!(network_stats.len(), 1);
    
    // Test scaling evaluation
    let decisions = auto_scaler.evaluate_scaling();
    // Should be empty since we don't have high utilization
    assert!(decisions.is_empty());

    println!("✅ End-to-end scaling workflow test passed");
}

#[tokio::test]
async fn test_performance_under_load() {
    // Test that the scaling system performs well under simulated load
    let start_time = std::time::Instant::now();
    
    let cpu_manager = CpuUtilizationManager::new(8);
    let mut memory_manager = MemoryPoolManager::new();
    let mut network_manager = NetworkBandwidthManager::new();
    
    // Register multiple pools and interfaces
    for i in 0..10 {
        let pool = Arc::new(ExpandableLockFreePool::<u64>::new(100, 1000));
        memory_manager.register_pool(format!("pool_{}", i), pool, None);
        
        network_manager.register_interface(format!("eth{}", i), 1000);
    }
    
    // Perform multiple operations
    for _ in 0..100 {
        let _ = cpu_manager.get_cpu_stats();
        let _ = memory_manager.get_pool_stats();
        let _ = network_manager.get_bandwidth_stats();
    }
    
    let elapsed = start_time.elapsed();
    
    // Should complete within reasonable time (less than 1 second)
    assert!(elapsed < Duration::from_secs(1));
    
    println!("✅ Performance under load test passed in {:?}", elapsed);
}