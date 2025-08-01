use hf_quoting_liquidity::infrastructure::*;
use hf_quoting_liquidity::error::InfrastructureError;
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_zero_copy_networking() -> Result<(), InfrastructureError> {
    let mut networking = ZeroCopyNetworking::new();
    
    // Test basic initialization
    assert_eq!(networking.get_stats().await.connections_active, 0);
    
    // Test listening on a port
    let addr = "127.0.0.1:0".parse().unwrap();
    networking.listen("test_server".to_string(), addr).await?;
    
    // Give some time for the server to start
    sleep(Duration::from_millis(100)).await;
    
    let stats = networking.get_stats().await;
    assert_eq!(stats.bytes_sent, 0);
    assert_eq!(stats.bytes_received, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_numa_aware_allocator() -> Result<(), InfrastructureError> {
    let allocator = NumaAwareAllocator::new()?;
    allocator.initialize_pools()?;
    
    let stats = allocator.get_stats();
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.current_usage, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_threading_system() -> Result<(), InfrastructureError> {
    let threading_system = ThreadingSystem::new()?;
    
    // Test CPU topology detection
    let topology = threading_system.get_cpu_topology();
    assert!(topology.logical_cores > 0);
    assert!(topology.physical_cores > 0);
    
    // Create a thread pool
    threading_system.create_thread_pool(
        "test_pool".to_string(),
        4,
        None,
        ThreadPriority::Normal,
    )?;
    
    // Submit a test task
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    threading_system.submit_task("test_pool", move || {
        let _ = tx.try_send("task_completed");
    })?;
    
    // Wait for task completion
    tokio::select! {
        result = rx.recv() => {
            assert_eq!(result, Some("task_completed"));
        }
        _ = sleep(Duration::from_secs(1)) => {
            panic!("Task did not complete in time");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_monitoring_system() -> Result<(), InfrastructureError> {
    let monitoring = MonitoringSystem::new()?;
    
    // Test metric recording
    monitoring.record_metric("test_metric", 42.0, HashMap::new());
    monitoring.increment_counter("test_counter", HashMap::new());
    monitoring.set_gauge("test_gauge", 100.0, HashMap::new());
    monitoring.record_histogram("test_latency", 1.5, HashMap::new());
    
    // Test logging
    let mut fields = HashMap::new();
    fields.insert("component".to_string(), serde_json::Value::String("test".to_string()));
    monitoring.log(LogLevel::Info, "Test log message", fields);
    
    // Give some time for metrics to be processed
    sleep(Duration::from_millis(100)).await;
    
    let metrics = monitoring.get_metrics().await;
    assert!(metrics.contains_key("test_metric"));
    
    let alerts = monitoring.get_active_alerts().await;
    assert_eq!(alerts.len(), 0); // No alerts should be active initially
    
    let health_status = monitoring.get_health_status().await;
    assert_eq!(health_status.len(), 0); // No health checks configured initially
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_manager() -> Result<(), InfrastructureError> {
    let config_manager = ConfigurationManager::new(Environment::Testing)?;
    
    // Test setting and getting configuration
    config_manager.set("test_key", "test_value").await?;
    let value: String = config_manager.get("test_key")?;
    assert_eq!(value, "test_value");
    
    // Test getting with default
    let default_value: i32 = config_manager.get_or_default("non_existent_key");
    assert_eq!(default_value, 0);
    
    // Test configuration with complex types
    let trading_config = TradingConfig {
        max_order_size: 1000.0,
        min_order_size: 1.0,
        tick_size: 0.01,
        max_positions: HashMap::new(),
        trading_hours: TradingHours {
            start: "09:30".to_string(),
            end: "16:00".to_string(),
            timezone: "EST".to_string(),
            holidays: vec!["2024-12-25".to_string()],
        },
        order_timeout: Duration::from_secs(30),
        cancel_timeout: Duration::from_secs(5),
    };
    
    config_manager.set("trading_config", &trading_config).await?;
    let retrieved_config: TradingConfig = config_manager.get("trading_config")?;
    assert_eq!(retrieved_config.max_order_size, 1000.0);
    
    // Test snapshot creation and rollback
    let snapshot_id = config_manager.create_snapshot("Test snapshot".to_string()).await?;
    
    // Modify configuration
    config_manager.set("test_key", "modified_value").await?;
    let modified_value: String = config_manager.get("test_key")?;
    assert_eq!(modified_value, "modified_value");
    
    // Rollback to snapshot
    config_manager.rollback_to_snapshot(&snapshot_id).await?;
    let rolled_back_value: String = config_manager.get("test_key")?;
    assert_eq!(rolled_back_value, "test_value");
    
    // Test configuration deletion
    config_manager.delete("test_key").await?;
    assert!(config_manager.get::<String>("test_key").is_err());
    
    // Test audit log
    let audit_log = config_manager.get_audit_log();
    assert!(!audit_log.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_validation() -> Result<(), InfrastructureError> {
    let config_manager = ConfigurationManager::new(Environment::Testing)?;
    
    // Add a range validator
    let range_validator = Box::new(RangeValidator { min: 0.0, max: 100.0 });
    config_manager.add_validator("percentage".to_string(), range_validator);
    
    // Test valid value
    config_manager.set("percentage", 50.0).await?;
    let value: f64 = config_manager.get("percentage")?;
    assert_eq!(value, 50.0);
    
    // Test invalid value (should fail validation)
    let result = config_manager.set("percentage", 150.0).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_change_listener() -> Result<(), InfrastructureError> {
    let config_manager = ConfigurationManager::new(Environment::Testing)?;
    
    // Add a change listener
    let listener = Box::new(TradingConfigListener {
        watched_keys: vec!["trading_param".to_string()],
    });
    config_manager.add_change_listener(listener);
    
    // Change a watched configuration
    config_manager.set("trading_param", 42).await?;
    
    // Give some time for the listener to process
    sleep(Duration::from_millis(50)).await;
    
    Ok(())
}

#[tokio::test]
async fn test_buffer_pool() -> Result<(), InfrastructureError> {
    let buffer_pool = BufferPool::new();
    
    // Test buffer allocation and return
    let small_buffer = buffer_pool.get_small_buffer().await;
    assert_eq!(small_buffer.capacity(), 4096);
    
    let medium_buffer = buffer_pool.get_medium_buffer().await;
    assert_eq!(medium_buffer.capacity(), 65536);
    
    let large_buffer = buffer_pool.get_large_buffer().await;
    assert_eq!(large_buffer.capacity(), 1048576);
    
    // Return buffers to pool
    buffer_pool.return_small_buffer(small_buffer).await;
    buffer_pool.return_medium_buffer(medium_buffer).await;
    buffer_pool.return_large_buffer(large_buffer).await;
    
    Ok(())
}

#[tokio::test]
async fn test_message_pool() -> Result<(), InfrastructureError> {
    let message_pool = MessagePool::new();
    
    // Test order message pool
    let mut order_msg = message_pool.get_order_message().await;
    order_msg.order_id = 12345;
    order_msg.symbol = "BTCUSD".to_string();
    order_msg.price = 50000;
    order_msg.quantity = 100;
    
    message_pool.return_order_message(order_msg).await;
    
    // Get another message and verify it's been reset
    let reset_msg = message_pool.get_order_message().await;
    assert_eq!(reset_msg.order_id, 0);
    assert!(reset_msg.symbol.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_lock_free_arena() -> Result<(), InfrastructureError> {
    let arena = LockFreeArena::new(1024 * 1024, 0)?; // 1MB arena
    
    // Test allocation
    let ptr1 = arena.allocate(1024, 8);
    assert!(ptr1.is_some());
    
    let ptr2 = arena.allocate(2048, 16);
    assert!(ptr2.is_some());
    
    // Check usage
    let usage = arena.usage();
    assert!(usage > 0.0);
    assert!(usage < 1.0);
    
    // Reset arena
    arena.reset();
    let usage_after_reset = arena.usage();
    assert_eq!(usage_after_reset, 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_histogram_percentiles() -> Result<(), InfrastructureError> {
    let histogram = Histogram::new();
    
    // Record some latency values (in microseconds)
    for i in 1..=100 {
        histogram.record(i as f64);
    }
    
    // Test percentile calculations
    let p50 = histogram.get_percentile(50.0);
    let p95 = histogram.get_percentile(95.0);
    let p99 = histogram.get_percentile(99.0);
    
    assert!(p50 > 0.0);
    assert!(p95 > p50);
    assert!(p99 > p95);
    
    Ok(())
}

#[tokio::test]
async fn test_lock_free_queue() -> Result<(), InfrastructureError> {
    use lockfree::LockFreeQueue;
    
    let queue = LockFreeQueue::new();
    
    // Test enqueue and dequeue
    queue.enqueue("message1");
    queue.enqueue("message2");
    queue.enqueue("message3");
    
    assert_eq!(queue.len(), 3);
    assert!(!queue.is_empty());
    
    let msg1 = queue.dequeue();
    assert_eq!(msg1, Some("message1"));
    
    let msg2 = queue.dequeue();
    assert_eq!(msg2, Some("message2"));
    
    assert_eq!(queue.len(), 1);
    
    let msg3 = queue.dequeue();
    assert_eq!(msg3, Some("message3"));
    
    assert!(queue.is_empty());
    
    let empty_msg = queue.dequeue();
    assert_eq!(empty_msg, None);
    
    Ok(())
}

#[tokio::test]
async fn test_integrated_infrastructure_workflow() -> Result<(), InfrastructureError> {
    // Test a complete workflow using multiple infrastructure components
    
    // 1. Initialize configuration
    let config_manager = ConfigurationManager::new(Environment::Testing)?;
    config_manager.set("server_port", 8080u16).await?;
    config_manager.set("thread_pool_size", 8usize).await?;
    config_manager.set("buffer_size", 65536usize).await?;
    
    // 2. Initialize threading system
    let threading_system = ThreadingSystem::new()?;
    let pool_size: usize = config_manager.get("thread_pool_size")?;
    threading_system.create_thread_pool(
        "main_pool".to_string(),
        pool_size,
        None,
        ThreadPriority::High,
    )?;
    
    // 3. Initialize monitoring
    let monitoring = MonitoringSystem::new()?;
    monitoring.start().await?;
    
    // 4. Initialize networking
    let mut networking = ZeroCopyNetworking::new();
    let port: u16 = config_manager.get("server_port")?;
    let addr = format!("127.0.0.1:{}", port).parse().unwrap();
    networking.listen("main_server".to_string(), addr).await?;
    
    // 5. Simulate some activity
    monitoring.record_metric("connections", 1.0, HashMap::new());
    monitoring.increment_counter("requests_total", HashMap::new());
    
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    threading_system.submit_task("main_pool", move || {
        // Simulate some work
        std::thread::sleep(Duration::from_millis(10));
        let _ = tx.try_send("work_completed");
    })?;
    
    // 6. Wait for completion and verify
    tokio::select! {
        result = rx.recv() => {
            assert_eq!(result, Some("work_completed"));
        }
        _ = sleep(Duration::from_secs(2)) => {
            panic!("Integrated workflow did not complete in time");
        }
    }
    
    // 7. Check final state
    let metrics = monitoring.get_metrics().await;
    assert!(metrics.contains_key("connections"));
    
    let stats = networking.get_stats().await;
    assert_eq!(stats.connections_active, 0); // No actual connections made
    
    Ok(())
}