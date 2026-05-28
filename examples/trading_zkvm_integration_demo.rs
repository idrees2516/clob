//! Trading zkVM Integration Demo
//! 
//! This example demonstrates the integration of the trading core with zkVM proof generation,
//! showing how to generate ZK proofs for order placement, matching, and batch operations
//! while maintaining sub-microsecond trading latency.

use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tracing::{info, warn, error};

use hft_limit_order_book::zkvm::{
    trading_integration::*,
    router::{ZkVMRouter, SelectionStrategy},
    ZkVMConfig, ZkVMBackend, ZiskConfig, SP1Config, SP1ProverMode,
};
use hft_limit_order_book::orderbook::{
    Order, OrderId, Trade, Symbol, Side, OrderType, TimeInForce,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting Trading zkVM Integration Demo");

    // Create zkVM configurations
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

    // Initialize zkVM router
    info!("Initializing zkVM router...");
    let router = Arc::new(
        ZkVMRouter::new(configs, SelectionStrategy::Balanced)
            .await
            .map_err(|e| format!("Failed to create zkVM router: {}", e))?
    );

    // Configure batch processing
    let batch_config = BatchConfig {
        max_batch_size: 50,
        batch_timeout: Duration::from_millis(5), // 5ms batching window
        enable_parallel_processing: true,
        max_concurrent_batches: 4,
    };

    // Initialize trading zkVM manager
    info!("Initializing trading zkVM manager...");
    let zkvm_manager = Arc::new(
        TradingZkVMManager::new(router, batch_config.clone(), 4)
            .await
            .map_err(|e| format!("Failed to create trading zkVM manager: {}", e))?
    );

    // Demo 1: Order Placement Proof Generation
    info!("\n=== Demo 1: Order Placement Proof Generation ===");
    demo_order_placement_proof(&zkvm_manager).await?;

    // Demo 2: Order Matching Proof Generation
    info!("\n=== Demo 2: Order Matching Proof Generation ===");
    demo_order_matching_proof(&zkvm_manager).await?;

    // Demo 3: Batch Proof Generation
    info!("\n=== Demo 3: Batch Proof Generation ===");
    demo_batch_proof_generation(&zkvm_manager).await?;

    // Demo 4: Async Proof Generation
    info!("\n=== Demo 4: Async Proof Generation ===");
    demo_async_proof_generation(&zkvm_manager).await?;

    // Display final statistics
    info!("\n=== Final Statistics ===");
    let stats = zkvm_manager.get_stats().await;
    info!("Total proofs generated: {}", stats.total_proofs_generated);
    info!("Successful proofs: {}", stats.successful_proofs);
    info!("Failed proofs: {}", stats.failed_proofs);
    info!("Average generation time: {:.2}ms", stats.avg_generation_time_ms);
    info!("Current queue size: {}", stats.proof_queue_size);

    info!("Trading zkVM Integration Demo completed successfully!");
    Ok(())
}

async fn demo_order_placement_proof(
    zkvm_manager: &Arc<TradingZkVMManager>
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating proof for order placement...");

    let order = Order {
        id: OrderId::new(1),
        symbol: Symbol::new("BTCUSD")?,
        side: Side::Buy,
        price: 50000 * 1_000_000, // $50,000 scaled
        size: 1 * 1_000_000,      // 1 BTC scaled
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
        order_type: OrderType::Limit,
        time_in_force: TimeInForce::GTC,
        client_order_id: Some("client_order_1".to_string()),
    };

    let pre_state = vec![0u8; 32];  // Placeholder pre-state
    let post_state = vec![1u8; 32]; // Placeholder post-state

    let start_time = Instant::now();
    let result = zkvm_manager.prove_order_placement(order, &pre_state, &post_state).await;
    let proof_time = start_time.elapsed();

    match result {
        Ok(proof) => {
            info!("✅ Order placement proof generated successfully!");
            info!("   Operation Type: {}", proof.operation_type);
            info!("   Operation ID: {}", proof.operation_id);
            info!("   zkVM Backend: {:?}", proof.zkvm_backend);
            info!("   Proof Size: {} bytes", proof.proof.size());
            info!("   Generation Time: {:.2}ms", proof_time.as_millis());
            info!("   Public Inputs Symbol: {}", proof.public_inputs.symbol);
        }
        Err(e) => {
            error!("❌ Failed to generate order placement proof: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn demo_order_matching_proof(
    zkvm_manager: &Arc<TradingZkVMManager>
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating proof for order matching...");

    let taker_order = Order {
        id: OrderId::new(2),
        symbol: Symbol::new("BTCUSD")?,
        side: Side::Buy,
        price: 50000 * 1_000_000,
        size: 2 * 1_000_000, // 2 BTC
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
        order_type: OrderType::Market,
        time_in_force: TimeInForce::IOC,
        client_order_id: Some("taker_order_1".to_string()),
    };

    let maker_orders = vec![
        Order {
            id: OrderId::new(3),
            symbol: Symbol::new("BTCUSD")?,
            side: Side::Sell,
            price: 49999 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64 - 1000,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: Some("maker_order_1".to_string()),
        },
        Order {
            id: OrderId::new(4),
            symbol: Symbol::new("BTCUSD")?,
            side: Side::Sell,
            price: 50000 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64 - 500,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: Some("maker_order_2".to_string()),
        },
    ];

    let trades = vec![
        Trade {
            id: 1,
            symbol: Symbol::new("BTCUSD")?,
            buyer_order_id: OrderId::new(2),
            seller_order_id: OrderId::new(3),
            price: 49999 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64,
            is_buyer_maker: false,
            sequence: 1,
        },
        Trade {
            id: 2,
            symbol: Symbol::new("BTCUSD")?,
            buyer_order_id: OrderId::new(2),
            seller_order_id: OrderId::new(4),
            price: 50000 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos() as u64 + 1,
            is_buyer_maker: false,
            sequence: 2,
        },
    ];

    let pre_state = vec![1u8; 32];
    let post_state = vec![2u8; 32];

    let start_time = Instant::now();
    let result = zkvm_manager.prove_order_matching(
        taker_order,
        maker_orders,
        trades,
        &pre_state,
        &post_state,
    ).await;
    let proof_time = start_time.elapsed();

    match result {
        Ok(proof) => {
            info!("✅ Order matching proof generated successfully!");
            info!("   Operation Type: {}", proof.operation_type);
            info!("   zkVM Backend: {:?}", proof.zkvm_backend);
            info!("   Proof Size: {} bytes", proof.proof.size());
            info!("   Generation Time: {:.2}ms", proof_time.as_millis());
        }
        Err(e) => {
            error!("❌ Failed to generate order matching proof: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn demo_batch_proof_generation(
    zkvm_manager: &Arc<TradingZkVMManager>
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Generating proof for batch operations...");

    let operations = vec![
        TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(5),
                symbol: Symbol::new("ETHUSD")?,
                side: Side::Buy,
                price: 3000 * 1_000_000,
                size: 10 * 1_000_000,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("batch_order_1".to_string()),
            },
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(6),
                symbol: Symbol::new("ETHUSD")?,
                side: Side::Sell,
                price: 3100 * 1_000_000,
                size: 5 * 1_000_000,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64 + 1000,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("batch_order_2".to_string()),
            },
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(7),
                symbol: Symbol::new("ETHUSD")?,
                side: Side::Buy,
                price: 2950 * 1_000_000,
                size: 15 * 1_000_000,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64 + 2000,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("batch_order_3".to_string()),
            },
            pre_state_hash: [2u8; 32],
            post_state_hash: [3u8; 32],
        },
    ];

    let batch_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos() as u64;

    let pre_state = vec![0u8; 32];
    let post_state = vec![3u8; 32];

    let start_time = Instant::now();
    let result = zkvm_manager.prove_batch_operations(
        operations,
        batch_id,
        &pre_state,
        &post_state,
    ).await;
    let proof_time = start_time.elapsed();

    match result {
        Ok(proof) => {
            info!("✅ Batch proof generated successfully!");
            info!("   Operation Type: {}", proof.operation_type);
            info!("   Batch ID: {}", proof.operation_id);
            info!("   zkVM Backend: {:?}", proof.zkvm_backend);
            info!("   Proof Size: {} bytes", proof.proof.size());
            info!("   Generation Time: {:.2}ms", proof_time.as_millis());
        }
        Err(e) => {
            error!("❌ Failed to generate batch proof: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}

async fn demo_async_proof_generation(
    zkvm_manager: &Arc<TradingZkVMManager>
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Demonstrating async proof generation...");

    let operations = vec![
        TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(8),
                symbol: Symbol::new("ADAUSD")?,
                side: Side::Buy,
                price: 1 * 1_000_000,
                size: 1000 * 1_000_000,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("async_order_1".to_string()),
            },
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        },
        TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(9),
                symbol: Symbol::new("ADAUSD")?,
                side: Side::Sell,
                price: (1.1 * 1_000_000.0) as u64,
                size: 500 * 1_000_000,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos() as u64 + 1000,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("async_order_2".to_string()),
            },
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        },
    ];

    let mut receivers = Vec::new();

    // Submit async proof requests
    for (i, operation) in operations.into_iter().enumerate() {
        let priority = match i {
            0 => ProofPriority::High,
            1 => ProofPriority::Normal,
            _ => ProofPriority::Low,
        };

        let receiver = zkvm_manager.prove_operation_async(operation, priority).await?;
        receivers.push(receiver);
        info!("   Submitted async proof request {} with priority {:?}", i + 1, priority);
    }

    // Wait for all proofs to complete
    info!("Waiting for async proofs to complete...");
    let start_time = Instant::now();

    for (i, receiver) in receivers.into_iter().enumerate() {
        match receiver.await {
            Ok(Ok(proof)) => {
                info!("   ✅ Async proof {} completed: {} ({}ms)", 
                      i + 1, proof.operation_type, proof.generation_time_ms);
            }
            Ok(Err(e)) => {
                warn!("   ⚠️ Async proof {} failed: {}", i + 1, e);
            }
            Err(e) => {
                error!("   ❌ Async proof {} channel error: {}", i + 1, e);
            }
        }
    }

    let total_time = start_time.elapsed();
    info!("All async proofs completed in {:.2}ms", total_time.as_millis());

    Ok(())
}