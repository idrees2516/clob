use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use trading_system::performance::scaling::{
    AutoScaler, ScalingConfig, ResourceScaler, LoadMonitor, ScalingEventLogger,
    CpuUtilizationManager, MemoryPoolManager, NetworkBandwidthManager,
    CpuMetricsCollector, MemoryMetricsCollector, NetworkMetricsCollector,
    CpuProvisioner, MemoryProvisioner, ResourceType, PoolExpansionPolicy,
    TrafficShapingConfig, QueueConfig, DropPolicy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Auto-Scaling and Load Management Demo");
    println!("=========================================");

    // Initialize auto-scaling components
    let scaling_config = ScalingConfig {
        min_scaling_interval: Duration::from_secs(10),
        scale_up_threshold: 0.7,
        scale_down_threshold: 0.3,
        cooldown_period: Duration::from_secs(30),
        enabled: true,
    };

    let resource_scaler = Arc::new(ResourceScaler::new());
    let load_monitor = Arc::new(LoadMonitor::new());
    let scaling_events = Arc::new(ScalingEventLogger::new());

    // Set up resource provisioners
    setup_resource_provisioners(&resource_scaler).await;

    // Set up load monitoring
    setup_load_monitoring(&load_monitor).await;

    // Create auto-scaler
    let mut auto_scaler = AutoScaler::new(
        scaling_config,
        Arc::clone(&resource_scaler),
        Arc::clone(&load_monitor),
        Arc::clone(&scaling_events),
    );

    println!("✅ Auto-scaler initialized");

    // Initialize CPU utilization manager
    let cpu_manager = Arc::new(CpuUtilizationManager::new(8));
    println!("✅ CPU utilization manager initialized with 8 cores");

    // Initialize memory pool manager
    let mut memory_manager = MemoryPoolManager::new();
    setup_memory_pools(&mut memory_manager).await;
    let memory_manager = Arc::new(memory_manager);
    println!("✅ Memory pool manager initialized");

    // Initialize network bandwidth manager
    let mut network_manager = NetworkBandwidthManager::new();
    setup_network_interfaces(&mut network_manager).await;
    let network_manager = Arc::new(network_manager);
    println!("✅ Network bandwidth manager initialized");

    // Start management tasks
    println!("\n🔄 Starting management loops...");
    
    let cpu_task = {
        let cpu_manager = Arc::clone(&cpu_manager);
        tokio::spawn(async move {
            // Run CPU management for a limited time in demo
            tokio::select! {
                _ = cpu_manager.start_management() => {},
                _ = sleep(Duration::from_secs(30)) => {
                    println!("⏰ CPU management demo completed");
                }
            }
        })
    };

    let memory_task = {
        let memory_manager = Arc::clone(&memory_manager);
        tokio::spawn(async move {
            tokio::select! {
                _ = memory_manager.start_management() => {},
                _ = sleep(Duration::from_secs(30)) => {
                    println!("⏰ Memory management demo completed");
                }
            }
        })
    };

    let network_task = {
        let network_manager = Arc::clone(&network_manager);
        tokio::spawn(async move {
            tokio::select! {
                _ = network_manager.start_management() => {},
                _ = sleep(Duration::from_secs(30)) => {
                    println!("⏰ Network management demo completed");
                }
            }
        })
    };

    // Simulate auto-scaling decisions
    let scaling_task = tokio::spawn(async move {
        for i in 0..10 {
            println!("\n📊 Scaling evaluation round {}", i + 1);
            
            // Evaluate scaling decisions
            let decisions = auto_scaler.evaluate_scaling();
            
            if decisions.is_empty() {
                println!("   No scaling decisions needed");
            } else {
                println!("   Found {} scaling decisions:", decisions.len());
                for decision in &decisions {
                    println!("     - {:?}: {} (confidence: {:.1}%)", 
                        decision.resource_type, 
                        decision.reason,
                        decision.confidence * 100.0
                    );
                }

                // Execute scaling decisions
                let results = auto_scaler.execute_scaling(decisions).await;
                let successful = results.iter().filter(|r| r.is_ok()).count();
                let failed = results.len() - successful;
                
                println!("   Executed: {} successful, {} failed", successful, failed);
            }

            sleep(Duration::from_secs(5)).await;
        }
        
        println!("⏰ Auto-scaling demo completed");
    });

    // Monitor and display statistics
    let stats_task = tokio::spawn(async move {
        for i in 0..6 {
            sleep(Duration::from_secs(5)).await;
            
            println!("\n📈 System Statistics ({})", i + 1);
            
            // CPU statistics
            let cpu_stats = cpu_manager.get_cpu_stats();
            println!("   CPU: {:.1}% utilization across {} cores", 
                cpu_stats.overall_utilization * 100.0, 
                cpu_stats.core_count
            );
            
            if cpu_stats.load_imbalance.is_significant() {
                println!("   ⚠️  CPU load imbalance detected (std dev: {:.2})", 
                    cpu_stats.load_imbalance.standard_deviation
                );
            }

            // Memory statistics
            let pool_stats = memory_manager.get_pool_stats();
            for (name, stats) in &pool_stats {
                println!("   Memory Pool '{}': {:.1}% utilization ({}/{} objects)", 
                    name,
                    stats.utilization * 100.0,
                    stats.allocated_objects,
                    stats.current_size
                );
            }

            // Network statistics
            let bandwidth_stats = network_manager.get_bandwidth_stats();
            for (name, stats) in &bandwidth_stats {
                println!("   Network '{}': {:.1}% utilization (RX: {:.1} Mbps, TX: {:.1} Mbps)", 
                    name,
                    stats.utilization * 100.0,
                    stats.current_rx_mbps,
                    stats.current_tx_mbps
                );
            }

            // Memory pressure
            let memory_pressure = memory_manager.get_memory_pressure();
            if memory_pressure > 0.8 {
                println!("   🔴 High memory pressure: {:.1}%", memory_pressure * 100.0);
            } else if memory_pressure > 0.6 {
                println!("   🟡 Moderate memory pressure: {:.1}%", memory_pressure * 100.0);
            } else {
                println!("   🟢 Low memory pressure: {:.1}%", memory_pressure * 100.0);
            }

            // Network congestion
            let congestion_levels = network_manager.detect_congestion().await;
            for (interface, level) in &congestion_levels {
                match level {
                    trading_system::performance::scaling::CongestionLevel::None => {},
                    _ => println!("   🚦 Network '{}': {:?} congestion", interface, level),
                }
            }
        }
    });

    // Wait for all tasks to complete
    let _ = tokio::join!(cpu_task, memory_task, network_task, scaling_task, stats_task);

    // Display final statistics
    println!("\n📊 Final Statistics");
    println!("===================");

    // Scaling events
    let events = scaling_events.get_recent_events().await;
    println!("Total scaling events: {}", events.len());

    let analysis = scaling_events.analyze_scaling_patterns().await;
    println!("Successful events: {}", analysis.successful_events);
    println!("Failed events: {}", analysis.failed_events);

    if !analysis.patterns.is_empty() {
        println!("\nDetected patterns:");
        for pattern in &analysis.patterns {
            println!("  - {:?} for {:?}: {}", 
                pattern.pattern_type, 
                pattern.resource_type, 
                pattern.description
            );
            println!("    Recommendation: {}", pattern.recommendation);
        }
    }

    // Resource stability
    println!("\nResource stability scores:");
    for (resource, stability) in &analysis.resource_stability {
        println!("  - {:?}: {:.3}", resource, stability);
    }

    // Expansion history
    let expansion_history = memory_manager.get_expansion_history().await;
    if !expansion_history.is_empty() {
        println!("\nMemory pool expansions: {}", expansion_history.len());
        for event in expansion_history.iter().take(5) {
            println!("  - {}: {} -> {} objects ({:.1}% -> {:.1}%)",
                event.pool_name,
                event.from_size,
                event.to_size,
                event.utilization_before * 100.0,
                event.utilization_after * 100.0
            );
        }
    }

    println!("\n✅ Auto-scaling demo completed successfully!");
    Ok(())
}

async fn setup_resource_provisioners(resource_scaler: &Arc<ResourceScaler>) {
    let mut scaler = unsafe { &mut *(Arc::as_ptr(resource_scaler) as *mut ResourceScaler) };
    
    // Register CPU provisioner
    let cpu_provisioner = Arc::new(CpuProvisioner::new(4, 2, 16));
    scaler.register_provisioner(ResourceType::Cpu, cpu_provisioner);

    // Register memory provisioner
    let memory_provisioner = Arc::new(MemoryProvisioner::new(8192, 4096, 32768));
    scaler.register_provisioner(ResourceType::Memory, memory_provisioner);

    println!("✅ Resource provisioners registered");
}

async fn setup_load_monitoring(load_monitor: &Arc<LoadMonitor>) {
    let mut monitor = unsafe { &mut *(Arc::as_ptr(load_monitor) as *mut LoadMonitor) };
    
    // Register metrics collectors
    let cpu_collector = Arc::new(CpuMetricsCollector::new(8));
    monitor.register_collector(cpu_collector);

    let memory_collector = Arc::new(MemoryMetricsCollector::new(16384));
    monitor.register_collector(memory_collector);

    let network_collector = Arc::new(NetworkMetricsCollector::new(1000));
    monitor.register_collector(network_collector);

    println!("✅ Load monitoring collectors registered");
}

async fn setup_memory_pools(memory_manager: &mut MemoryPoolManager) {
    use trading_system::performance::scaling::ExpandableLockFreePool;
    
    // Create expandable pools for different object types
    let order_pool = Arc::new(ExpandableLockFreePool::<u64>::new(1000, 10000));
    let order_policy = PoolExpansionPolicy {
        expansion_trigger_threshold: 0.8,
        expansion_factor: 1.5,
        max_expansion_size: 10000,
        emergency_threshold: 0.95,
        emergency_factor: 2.0,
        ..Default::default()
    };
    memory_manager.register_pool("orders".to_string(), order_pool, Some(order_policy));

    let trade_pool = Arc::new(ExpandableLockFreePool::<u64>::new(500, 5000));
    let trade_policy = PoolExpansionPolicy {
        expansion_trigger_threshold: 0.75,
        expansion_factor: 1.3,
        max_expansion_size: 5000,
        ..Default::default()
    };
    memory_manager.register_pool("trades".to_string(), trade_pool, Some(trade_policy));

    println!("✅ Memory pools configured");
}

async fn setup_network_interfaces(network_manager: &mut NetworkBandwidthManager) {
    // Register network interfaces
    network_manager.register_interface("eth0".to_string(), 1000); // 1 Gbps
    network_manager.register_interface("eth1".to_string(), 10000); // 10 Gbps

    // Configure traffic shaping for high-speed interface
    let shaping_config = TrafficShapingConfig {
        max_bandwidth_mbps: 8000, // 80% of 10 Gbps
        burst_size_kb: 128,
        priority_queues: vec![
            QueueConfig {
                name: "critical".to_string(),
                priority: 0,
                guaranteed_bandwidth_mbps: 2000,
                max_bandwidth_mbps: 4000,
                max_queue_size: 1000,
                drop_policy: DropPolicy::WeightedRandomEarlyDetection,
            },
            QueueConfig {
                name: "high".to_string(),
                priority: 1,
                guaranteed_bandwidth_mbps: 1000,
                max_bandwidth_mbps: 3000,
                max_queue_size: 2000,
                drop_policy: DropPolicy::RandomEarlyDetection,
            },
            QueueConfig {
                name: "normal".to_string(),
                priority: 2,
                guaranteed_bandwidth_mbps: 500,
                max_bandwidth_mbps: 2000,
                max_queue_size: 5000,
                drop_policy: DropPolicy::TailDrop,
            },
        ],
        rate_limiting_enabled: true,
        congestion_control_enabled: true,
    };

    let _ = network_manager.configure_traffic_shaping("eth1".to_string(), shaping_config).await;
    println!("✅ Network interfaces configured");
}