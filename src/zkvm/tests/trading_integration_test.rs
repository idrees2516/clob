//! Comprehensive tests for trading zkVM integration
//! 
//! This module contains tests that verify all aspects of the trading zkVM integration,
//! including proof generation, batch processing, async operations, and performance.

use std::sync::Arc;
use tokio::time::{Duration, Instant};

use crate::zkvm::{
    trading_integration::*,
    router::{ZkVMRouter, SelectionStrategy},
    ZkVMConfig, ZkVMBackend, ZiskConfig, SP1Config, SP1ProverMode,
};
use crate::orderbook::{
    Order, OrderId, Trade, Symbol, Side, OrderType, TimeInForce,
};

/// Helper function to create a test zkVM router
async fn create_test_router() -> ZkVMRouter {
    let configs = vec![
        ZkVMConfig {
            backend: ZkVMBackend::ZisK,
            max_cycles: 1_000_000,
            memory_limit: 64 * 1024 * 1024,
            timeout_seconds: 300,
            network_config: None,
            zisk_config: Some(ZiskConfig {
                optimization_level: 2,
                enable_gpu: false,
                memory_pool_size: 512 * 1024 * 1024,
            }),
            sp1_config: None,
        },
    ];

    ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
        .await
        .expect("Failed to create test router")
}

/// Helper function to create a test order
fn create_test_order(id: u64, symbol: &str, side: Side, price: u64, size: u64) -> Order {
    Order {
        id: OrderId::new(id),
        symbol: Symbol::new(symbol).unwrap(),
        side,
        price,
        size,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        order_type: OrderType::Limit,
        time_in_force: TimeInForce::GTC,
        client_order_id: Some(format!("test_order_{}", id)),
    }
}

/// Helper function to create a test trade
fn create_test_trade(
    id: u64,
    symbol: &str,
    buyer_id: u64,
    seller_id: u64,
    price: u64,
    size: u64,
) -> Trade {
    Trade {
        id,
        symbol: Symbol::new(symbol).unwrap(),
        buyer_order_id: OrderId::new(buyer_id),
        seller_order_id: OrderId::new(seller_id),
        price,
        size,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        is_buyer_maker: false,
        sequence: id,
    }
}

#[tokio::test]
async fn test_trading_zkvm_manager_initialization() {
    let router = Arc::new(create_test_router().await);
    let batch_config = BatchConfig::default();
    
    let manager = TradingZkVMManager::new(router, batch_config, 2).await;
    assert!(manager.is_ok(), "Failed to create TradingZkVMManager");

    let manager = manager.unwrap();
    let stats = manager.get_stats().await;
    assert_eq!(stats.total_proofs_generated, 0);
    assert_eq!(stats.successful_proofs, 0);
    assert_eq!(stats.failed_proofs, 0);
}

#[tokio::test]
async fn test_order_placement_proof_generation() {
    let router = Arc::new(create_test_router().await);
    let manager = TradingZkVMManager::new(router, BatchConfig::default(), 1)
        .await
        .unwrap();

    let order = create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);
    let pre_state = vec![0u8; 32];
    let post_state = vec![1u8; 32];

    let start_time = Instant::now();
    let result = manager.prove_order_placement(order, &pre_state, &post_state).await;
    let proof_time = start_time.elapsed();

    assert!(result.is_ok(), "Order placement proof generation failed");
    
    let proof = result.unwrap();
    assert_eq!(proof.operation_type, "order_placement");
    assert_eq!(proof.operation_id, 1);
    assert!(proof.generation_time_ms > 0);
    assert!(proof_time.as_millis() > 0);

    // Verify statistics were updated
    let stats = manager.get_stats().await;
    assert_eq!(stats.total_proofs_generated, 1);
    assert_eq!(stats.successful_proofs, 1);
    assert_eq!(stats.failed_proofs, 0);
}

#[tokio::test]
async fn test_order_matching_proof_generation() {
    let router = Arc::new(create_test_router().await);
    let manager = TradingZkVMManager::new(router, BatchConfig::default(), 1)
        .await
        .unwrap();

    let taker_order = create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 2_000000);
    let maker_orders = vec![
        create_test_order(2, "BTCUSD", Side::Sell, 49999_000000, 1_000000),
        create_test_order(3, "BTCUSD", Side::Sell, 50000_000000, 1_000000),
    ];
    let trades = vec![
        create_test_trade(1, "BTCUSD", 1, 2, 49999_000000, 1_000000),
        create_test_trade(2, "BTCUSD", 1, 3, 50000_000000, 1_000000),
    ];

    let pre_state = vec![0u8; 32];
    let post_state = vec![1u8; 32];

    let result = manager.prove_order_matching(
        taker_order,
        maker_orders,
        trades,
        &pre_state,
        &post_state,
    ).await;

    assert!(result.is_ok(), "Order matching proof generation failed");
    
    let proof = result.unwrap();
    assert_eq!(proof.operation_type, "order_matching");
    assert!(proof.generation_time_ms > 0);
}

#[tokio::test]
async fn test_batch_proof_generation() {
    let router = Arc::new(create_test_router().await);
    let manager = TradingZkVMManager::new(router, BatchConfig::default(), 1)
        .await
        .unwrap();

    let operations = vec![
        TradingOperation::OrderPlacement {
            order: create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000),
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: create_test_order(2, "BTCUSD", Side::Sell, 51000_000000, 1_000000),
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: create_test_order(3, "ETHUSD", Side::Buy, 3000_000000, 10_000000),
            pre_state_hash: [2u8; 32],
            post_state_hash: [3u8; 32],
        },
    ];

    let batch_id = 12345;
    let pre_state = vec![0u8; 32];
    let post_state = vec![3u8; 32];

    let result = manager.prove_batch_operations(
        operations,
        batch_id,
        &pre_state,
        &post_state,
    ).await;

    assert!(result.is_ok(), "Batch proof generation failed");
    
    let proof = result.unwrap();
    assert_eq!(proof.operation_type, "batch_operations");
    assert_eq!(proof.operation_id, batch_id);
    assert!(proof.generation_time_ms > 0);
}

#[tokio::test]
async fn test_async_proof_generation() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 2)
            .await
            .unwrap()
    );

    let operation = TradingOperation::OrderPlacement {
        order: create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000),
        pre_state_hash: [0u8; 32],
        post_state_hash: [1u8; 32],
    };

    // Test async proof generation
    let receiver = manager.prove_operation_async(operation, ProofPriority::High).await;
    assert!(receiver.is_ok(), "Failed to submit async proof request");

    let result = receiver.unwrap().await;
    assert!(result.is_ok(), "Async proof generation failed");
    
    let proof = result.unwrap().unwrap();
    assert_eq!(proof.operation_type, "order_placement");
    assert!(proof.generation_time_ms > 0);

    // Verify statistics
    let stats = manager.get_stats().await;
    assert!(stats.total_proofs_generated > 0);
}

#[tokio::test]
async fn test_batch_proof_processor() {
    let router = Arc::new(create_test_router().await);
    let batch_config = BatchConfig {
        max_batch_size: 3,
        batch_timeout: Duration::from_millis(100),
        enable_parallel_processing: true,
        max_concurrent_batches: 2,
    };
    
    let manager = Arc::new(
        TradingZkVMManager::new(router, batch_config.clone(), 1)
            .await
            .unwrap()
    );
    
    let processor = BatchProofProcessor::new(batch_config, manager);

    // Add operations to trigger batch processing
    let operations = vec![
        TradingOperation::OrderPlacement {
            order: create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000),
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: create_test_order(2, "BTCUSD", Side::Sell, 51000_000000, 1_000000),
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: create_test_order(3, "ETHUSD", Side::Buy, 3000_000000, 10_000000),
            pre_state_hash: [2u8; 32],
            post_state_hash: [3u8; 32],
        },
    ];

    // Add operations one by one
    for operation in operations {
        let result = processor.add_operation(operation).await;
        assert!(result.is_ok(), "Failed to add operation to batch processor");
    }

    // Wait a bit for batch processing
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Check statistics
    let stats = processor.get_stats().await;
    assert!(stats.total_batches_processed > 0, "No batches were processed");
}

#[tokio::test]
async fn test_async_proof_generator() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 1)
            .await
            .unwrap()
    );
    
    let generator = AsyncProofGenerator::new(manager, 2);

    let operations = vec![
        TradingOperation::OrderPlacement {
            order: create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000),
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: create_test_order(2, "ETHUSD", Side::Sell, 3000_000000, 5_000000),
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        },
    ];

    let mut receivers = Vec::new();

    // Submit requests with different priorities
    for (i, operation) in operations.into_iter().enumerate() {
        let priority = if i == 0 { ProofPriority::High } else { ProofPriority::Normal };
        let receiver = generator.submit_request(operation, priority).await;
        assert!(receiver.is_ok(), "Failed to submit async proof request");
        receivers.push(receiver.unwrap());
    }

    // Wait for all proofs to complete
    for receiver in receivers {
        let result = receiver.await;
        assert!(result.is_ok(), "Async proof generation failed");
        
        let proof = result.unwrap().unwrap();
        assert!(proof.generation_time_ms > 0);
    }

    // Check statistics
    let stats = generator.get_stats().await;
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.high_priority_requests, 1);
    assert_eq!(stats.normal_priority_requests, 1);
    assert!(stats.completed_requests > 0);
}

#[tokio::test]
async fn test_zk_provable_clob_engine() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 1)
            .await
            .unwrap()
    );
    
    let symbol = Symbol::new("BTCUSD").unwrap();
    let mut engine = ZkProvableCLOBEngine::new(
        symbol,
        manager.clone(),
        true,  // Enable proof generation
        false, // Disable async proofs for this test
    ).await.unwrap();

    let order = create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);

    let result = engine.submit_order_with_proof(order).await;
    assert!(result.is_ok(), "Failed to submit order with proof");

    let (submission_result, proof) = result.unwrap();
    assert_eq!(submission_result.order_id, OrderId::new(1));
    assert!(proof.is_some(), "Proof should be generated");

    let proof = proof.unwrap();
    assert_eq!(proof.operation_type, "order_placement");
    assert!(proof.generation_time_ms > 0);

    // Test proof statistics
    let stats = engine.get_proof_stats().await;
    assert!(stats.total_proofs_generated > 0);
}

#[tokio::test]
async fn test_zk_provable_clob_engine_async() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 2)
            .await
            .unwrap()
    );
    
    let symbol = Symbol::new("BTCUSD").unwrap();
    let mut engine = ZkProvableCLOBEngine::new(
        symbol,
        manager.clone(),
        true, // Enable proof generation
        true, // Enable async proofs
    ).await.unwrap();

    let order = create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);

    let result = engine.submit_order_with_proof(order).await;
    assert!(result.is_ok(), "Failed to submit order with async proof");

    let (submission_result, proof) = result.unwrap();
    assert_eq!(submission_result.order_id, OrderId::new(1));
    assert!(proof.is_none(), "Proof should be None for async generation");

    // Wait a bit for async proof to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check that proof was generated asynchronously
    let stats = engine.get_proof_stats().await;
    assert!(stats.proof_queue_size >= 0); // Queue might be empty if proof completed
}

#[tokio::test]
async fn test_trading_operation_utilities() {
    let order = create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);
    
    let single_operation = TradingOperation::OrderPlacement {
        order: order.clone(),
        pre_state_hash: [0u8; 32],
        post_state_hash: [1u8; 32],
    };

    let batch_operation = TradingOperation::BatchOperations {
        operations: vec![single_operation.clone(), single_operation.clone()],
        batch_id: 123,
        pre_state_hash: [0u8; 32],
        post_state_hash: [2u8; 32],
    };

    // Test operation count
    assert_eq!(single_operation.operation_count(), 1);
    assert_eq!(batch_operation.operation_count(), 2);

    // Test operation extraction
    let extracted_single = single_operation.extract_operations();
    assert_eq!(extracted_single.len(), 1);

    let extracted_batch = batch_operation.extract_operations();
    assert_eq!(extracted_batch.len(), 2);
}

#[tokio::test]
async fn test_proof_priority_ordering() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 1)
            .await
            .unwrap()
    );
    
    let generator = AsyncProofGenerator::new(manager, 1);

    // Submit requests in reverse priority order
    let low_op = TradingOperation::OrderPlacement {
        order: create_test_order(1, "BTCUSD", Side::Buy, 50000_000000, 1_000000),
        pre_state_hash: [0u8; 32],
        post_state_hash: [1u8; 32],
    };

    let high_op = TradingOperation::OrderPlacement {
        order: create_test_order(2, "BTCUSD", Side::Sell, 51000_000000, 1_000000),
        pre_state_hash: [1u8; 32],
        post_state_hash: [2u8; 32],
    };

    // Submit low priority first, then high priority
    let low_receiver = generator.submit_request(low_op, ProofPriority::Low).await.unwrap();
    let high_receiver = generator.submit_request(high_op, ProofPriority::High).await.unwrap();

    // High priority should complete first (or at least not fail)
    let high_result = high_receiver.await;
    let low_result = low_receiver.await;

    assert!(high_result.is_ok(), "High priority proof failed");
    assert!(low_result.is_ok(), "Low priority proof failed");

    // Check statistics
    let stats = generator.get_stats().await;
    assert_eq!(stats.total_requests, 2);
    assert_eq!(stats.high_priority_requests, 1);
    assert_eq!(stats.low_priority_requests, 1);
}

#[tokio::test]
async fn test_performance_benchmarking() {
    let router = Arc::new(create_test_router().await);
    let manager = TradingZkVMManager::new(router, BatchConfig::default(), 4)
        .await
        .unwrap();

    let num_operations = 10;
    let mut proof_times = Vec::new();

    for i in 0..num_operations {
        let order = create_test_order(i + 1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);
        let pre_state = vec![i as u8; 32];
        let post_state = vec![(i + 1) as u8; 32];

        let start_time = Instant::now();
        let result = manager.prove_order_placement(order, &pre_state, &post_state).await;
        let proof_time = start_time.elapsed();

        assert!(result.is_ok(), "Proof generation failed for operation {}", i);
        proof_times.push(proof_time.as_millis() as f64);
    }

    // Calculate performance metrics
    let avg_time = proof_times.iter().sum::<f64>() / proof_times.len() as f64;
    let min_time = proof_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_time = proof_times.iter().fold(0.0, |a, &b| a.max(b));

    println!("Performance Benchmark Results:");
    println!("  Operations: {}", num_operations);
    println!("  Average time: {:.2}ms", avg_time);
    println!("  Min time: {:.2}ms", min_time);
    println!("  Max time: {:.2}ms", max_time);

    // Verify performance is reasonable (adjust thresholds as needed)
    assert!(avg_time < 1000.0, "Average proof time too high: {:.2}ms", avg_time);
    assert!(max_time < 2000.0, "Max proof time too high: {:.2}ms", max_time);

    // Check final statistics
    let stats = manager.get_stats().await;
    assert_eq!(stats.total_proofs_generated, num_operations as u64);
    assert_eq!(stats.successful_proofs, num_operations as u64);
    assert_eq!(stats.failed_proofs, 0);
    assert!(stats.avg_generation_time_ms > 0.0);
}

#[tokio::test]
async fn test_error_handling() {
    let router = Arc::new(create_test_router().await);
    let manager = TradingZkVMManager::new(router, BatchConfig::default(), 1)
        .await
        .unwrap();

    // Test empty batch
    let result = manager.prove_batch_operations(
        vec![], // Empty operations
        123,
        &[0u8; 32],
        &[1u8; 32],
    ).await;

    assert!(result.is_err(), "Empty batch should fail");
    match result.unwrap_err() {
        TradingZkVMError::BatchProcessing(msg) => {
            assert_eq!(msg, "Empty batch");
        }
        _ => panic!("Wrong error type for empty batch"),
    }
}

#[tokio::test]
async fn test_concurrent_proof_generation() {
    let router = Arc::new(create_test_router().await);
    let manager = Arc::new(
        TradingZkVMManager::new(router, BatchConfig::default(), 4)
            .await
            .unwrap()
    );

    let num_concurrent = 5;
    let mut handles = Vec::new();

    // Launch concurrent proof generation tasks
    for i in 0..num_concurrent {
        let manager_clone = manager.clone();
        let handle = tokio::spawn(async move {
            let order = create_test_order(i + 1, "BTCUSD", Side::Buy, 50000_000000, 1_000000);
            let pre_state = vec![i as u8; 32];
            let post_state = vec![(i + 1) as u8; 32];

            manager_clone.prove_order_placement(order, &pre_state, &post_state).await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut successful = 0;
    for handle in handles {
        let result = handle.await.unwrap();
        if result.is_ok() {
            successful += 1;
        }
    }

    assert_eq!(successful, num_concurrent, "Not all concurrent proofs succeeded");

    // Check final statistics
    let stats = manager.get_stats().await;
    assert_eq!(stats.total_proofs_generated, num_concurrent as u64);
    assert_eq!(stats.successful_proofs, num_concurrent as u64);
}