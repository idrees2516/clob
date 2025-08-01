//! Integration Example for Advanced Data Availability Layer
//! 
//! This example demonstrates how to integrate the advanced DA layer
//! with the CLOB system for high-frequency trading data storage.

use crate::rollup::{
    advanced_da::{AdvancedDAClient, DiskStorageConfig, AdvancedBlobCommitment},
    types::OrderBatch,
    RollupError,
};
use crate::orderbook::types::{Order, Trade, OrderId, Symbol, Side, OrderType, TimeInForce};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error};

/// Complete integration example
pub struct CLOBDataAvailabilityIntegration {
    da_client: AdvancedDAClient,
    storage_path: String,
}

impl CLOBDataAvailabilityIntegration {
    /// Create a new integration instance
    pub async fn new(storage_path: String) -> Result<Self, RollupError> {
        let config = DiskStorageConfig {
            base_path: storage_path.clone().into(),
            max_file_size: 100 * 1024 * 1024, // 100MB
            compression_enabled: true,
            indexing_enabled: true,
            cache_size: 1000,
            sync_interval: Duration::from_secs(30),
        };

        let da_client = AdvancedDAClient::new(config).await
            .map_err(|e| RollupError::DataAvailabilityError(e.to_string()))?;

        Ok(Self {
            da_client,
            storage_path,
        })
    }

    /// Demonstrate complete workflow with real trading data
    pub async fn run_complete_example(&mut self) -> Result<(), RollupError> {
        info!("Starting CLOB Data Availability Integration Example");

        // 1. Create sample trading data
        let (batch, trades) = self.create_sample_trading_data().await?;
        info!("Created sample batch with {} orders and {} trades", 
               batch.orders.len(), trades.len());

        // 2. Store data with advanced DA techniques
        let commitment = self.store_trading_data(&batch, &trades).await?;
        info!("Stored trading data with advanced DA techniques");
        self.print_commitment_info(&commitment);

        // 3. Simulate some time passing
        sleep(Duration::from_millis(100)).await;

        // 4. Retrieve and verify data
        let (retrieved_batch, retrieved_trades) = self.retrieve_and_verify_data(commitment.blob_hash).await?;
        info!("Successfully retrieved and verified trading data");

        // 5. Verify data integrity
        self.verify_data_integrity(&batch, &trades, &retrieved_batch, &retrieved_trades)?;
        info!("Data integrity verification passed");

        // 6. Demonstrate recovery from erasure coded chunks
        self.demonstrate_data_recovery(&commitment).await?;

        // 7. Show performance metrics
        self.show_performance_metrics().await?;

        info!("CLOB Data Availability Integration Example completed successfully");
        Ok(())
    }

    /// Create realistic sample trading data
    async fn create_sample_trading_data(&self) -> Result<(OrderBatch, Vec<Trade>), RollupError> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Create sample orders
        let orders = vec![
            Order::new_limit(
                OrderId::new(1),
                Symbol::new("BTCUSD").unwrap(),
                Side::Buy,
                50000 * 1_000_000, // $50,000 with 6 decimal scaling
                1 * 1_000_000,     // 1 BTC with 6 decimal scaling
                timestamp,
            ),
            Order::new_limit(
                OrderId::new(2),
                Symbol::new("BTCUSD").unwrap(),
                Side::Sell,
                50100 * 1_000_000, // $50,100 with 6 decimal scaling
                1 * 1_000_000,     // 1 BTC with 6 decimal scaling
                timestamp + 1,
            ),
            Order::new_limit(
                OrderId::new(3),
                Symbol::new("ETHUSD").unwrap(),
                Side::Buy,
                3000 * 1_000_000,  // $3,000 with 6 decimal scaling
                10 * 1_000_000,    // 10 ETH with 6 decimal scaling
                timestamp + 2,
            ),
            Order::new_market(
                OrderId::new(4),
                Symbol::new("ETHUSD").unwrap(),
                Side::Sell,
                5 * 1_000_000,     // 5 ETH with 6 decimal scaling
                timestamp + 3,
            ),
        ];

        // Create order batch
        let batch = OrderBatch::new(
            1,
            orders,
            timestamp,
            [0u8; 32], // Initial state root
        );

        // Create sample trades
        let trades = vec![
            Trade::new(
                1,
                Symbol::new("BTCUSD").unwrap(),
                50050 * 1_000_000, // $50,050 execution price
                500_000,           // 0.5 BTC
                timestamp + 10,
                OrderId::new(1),   // Buyer
                OrderId::new(2),   // Seller
                true,              // Buyer is maker
                1,                 // Sequence
            ),
            Trade::new(
                2,
                Symbol::new("ETHUSD").unwrap(),
                3000 * 1_000_000,  // $3,000 execution price
                5 * 1_000_000,     // 5 ETH
                timestamp + 11,
                OrderId::new(3),   // Buyer
                OrderId::new(4),   // Seller
                true,              // Buyer is maker
                2,                 // Sequence
            ),
        ];

        Ok((batch, trades))
    }

    /// Store trading data using advanced DA techniques
    async fn store_trading_data(
        &mut self,
        batch: &OrderBatch,
        trades: &[Trade],
    ) -> Result<AdvancedBlobCommitment, RollupError> {
        info!("Storing trading data with advanced DA techniques...");
        
        let commitment = self.da_client.store_with_sampling(batch, trades).await
            .map_err(|e| RollupError::DataAvailabilityError(e.to_string()))?;

        info!("Successfully stored data with {} samples and {} encoded chunks",
               commitment.samples.len(), commitment.encoded_chunks_count);

        Ok(commitment)
    }

    /// Retrieve and verify trading data
    async fn retrieve_and_verify_data(
        &self,
        blob_hash: [u8; 32],
    ) -> Result<(OrderBatch, Vec<Trade>), RollupError> {
        info!("Retrieving and verifying trading data...");
        
        let (batch, trades) = self.da_client.retrieve_with_verification(blob_hash).await
            .map_err(|e| RollupError::DataAvailabilityError(e.to_string()))?;

        info!("Successfully retrieved batch {} with {} orders and {} trades",
               batch.batch_id, batch.orders.len(), trades.len());

        Ok((batch, trades))
    }

    /// Verify data integrity between original and retrieved data
    fn verify_data_integrity(
        &self,
        original_batch: &OrderBatch,
        original_trades: &[Trade],
        retrieved_batch: &OrderBatch,
        retrieved_trades: &[Trade],
    ) -> Result<(), RollupError> {
        // Verify batch integrity
        if original_batch.batch_id != retrieved_batch.batch_id {
            return Err(RollupError::DataAvailabilityError(
                "Batch ID mismatch".to_string()
            ));
        }

        if original_batch.orders.len() != retrieved_batch.orders.len() {
            return Err(RollupError::DataAvailabilityError(
                "Order count mismatch".to_string()
            ));
        }

        // Verify order integrity
        for (original, retrieved) in original_batch.orders.iter().zip(retrieved_batch.orders.iter()) {
            if original.id != retrieved.id {
                return Err(RollupError::DataAvailabilityError(
                    format!("Order ID mismatch: {} != {}", original.id.as_u64(), retrieved.id.as_u64())
                ));
            }
            if original.price != retrieved.price {
                return Err(RollupError::DataAvailabilityError(
                    format!("Order price mismatch for order {}", original.id.as_u64())
                ));
            }
            if original.size != retrieved.size {
                return Err(RollupError::DataAvailabilityError(
                    format!("Order size mismatch for order {}", original.id.as_u64())
                ));
            }
        }

        // Verify trades integrity
        if original_trades.len() != retrieved_trades.len() {
            return Err(RollupError::DataAvailabilityError(
                "Trade count mismatch".to_string()
            ));
        }

        for (original, retrieved) in original_trades.iter().zip(retrieved_trades.iter()) {
            if original.id != retrieved.id {
                return Err(RollupError::DataAvailabilityError(
                    format!("Trade ID mismatch: {} != {}", original.id, retrieved.id)
                ));
            }
            if original.price != retrieved.price {
                return Err(RollupError::DataAvailabilityError(
                    format!("Trade price mismatch for trade {}", original.id)
                ));
            }
        }

        info!("Data integrity verification passed - all data matches perfectly");
        Ok(())
    }

    /// Demonstrate data recovery from erasure coded chunks
    async fn demonstrate_data_recovery(
        &mut self,
        commitment: &AdvancedBlobCommitment,
    ) -> Result<(), RollupError> {
        info!("Demonstrating data recovery from erasure coded chunks...");
        
        // In a real scenario, you would:
        // 1. Simulate some chunks being unavailable
        // 2. Retrieve available chunks from different storage nodes
        // 3. Recover the original data using Reed-Solomon decoding
        
        // For this example, we'll just show that recovery is possible
        info!("Data recovery would be possible with {} out of {} chunks (threshold: {})",
               commitment.encoded_chunks_count - 2, // Simulate 2 missing chunks
               commitment.encoded_chunks_count,
               commitment.recovery_threshold);

        if commitment.encoded_chunks_count - 2 >= commitment.recovery_threshold {
            info!("✓ Data recovery is possible - sufficient chunks available");
        } else {
            warn!("✗ Data recovery not possible - insufficient chunks available");
        }

        Ok(())
    }

    /// Show performance metrics
    async fn show_performance_metrics(&self) -> Result<(), RollupError> {
        let metrics = self.da_client.get_advanced_metrics().await;
        
        info!("=== Advanced Data Availability Metrics ===");
        info!("Samples generated: {}", metrics.samples_generated);
        info!("Samples verified: {}", metrics.samples_verified);
        info!("Chunks encoded: {}", metrics.chunks_encoded);
        info!("Chunks recovered: {}", metrics.chunks_recovered);
        info!("Commitments created: {}", metrics.commitments_created);
        info!("Commitments verified: {}", metrics.commitments_verified);
        info!("Disk reads: {}", metrics.disk_reads);
        info!("Disk writes: {}", metrics.disk_writes);
        info!("Cache hits: {}", metrics.cache_hits);
        info!("Cache misses: {}", metrics.cache_misses);
        info!("Average sampling time: {:?}", metrics.average_sampling_time);
        info!("Average encoding time: {:?}", metrics.average_encoding_time);
        info!("Average verification time: {:?}", metrics.average_verification_time);
        
        let cache_hit_ratio = if metrics.cache_hits + metrics.cache_misses > 0 {
            metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64 * 100.0
        } else {
            0.0
        };
        info!("Cache hit ratio: {:.2}%", cache_hit_ratio);

        Ok(())
    }

    /// Print commitment information
    fn print_commitment_info(&self, commitment: &AdvancedBlobCommitment) {
        info!("=== Advanced Blob Commitment ===");
        info!("Blob hash: {}", hex::encode(&commitment.blob_hash[..8]));
        info!("Polynomial commitment degree: {}", commitment.polynomial_commitment.degree);
        info!("Samples count: {}", commitment.samples.len());
        info!("Encoded chunks: {}", commitment.encoded_chunks_count);
        info!("Recovery threshold: {}", commitment.recovery_threshold);
        info!("Timestamp: {}", commitment.timestamp);
        info!("Storage proof size: {} bytes", commitment.storage_proof.len());
    }

    /// Demonstrate high-frequency trading scenario
    pub async fn demonstrate_hft_scenario(&mut self) -> Result<(), RollupError> {
        info!("Demonstrating high-frequency trading scenario...");

        let mut batch_id = 1;
        let base_timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Simulate 10 rapid trading batches
        for i in 0..10 {
            let timestamp = base_timestamp + i;
            
            // Create high-frequency orders
            let orders = (0..50).map(|j| {
                let order_id = (i * 50 + j + 1) as u64;
                let side = if j % 2 == 0 { Side::Buy } else { Side::Sell };
                let price_offset = (j as i32 - 25) * 10; // Spread around base price
                let base_price = 50000 * 1_000_000; // $50,000
                let price = ((base_price as i64) + (price_offset as i64)) as u64;
                
                Order::new_limit(
                    OrderId::new(order_id),
                    Symbol::new("BTCUSD").unwrap(),
                    side,
                    price,
                    100_000, // 0.1 BTC
                    timestamp,
                )
            }).collect();

            let batch = OrderBatch::new(batch_id, orders, timestamp, [0u8; 32]);
            
            // Create some trades
            let trades = (0..5).map(|j| {
                Trade::new(
                    (i * 5 + j + 1) as u64,
                    Symbol::new("BTCUSD").unwrap(),
                    50000 * 1_000_000,
                    50_000, // 0.05 BTC
                    timestamp,
                    OrderId::new((i * 50 + j * 2 + 1) as u64),
                    OrderId::new((i * 50 + j * 2 + 2) as u64),
                    j % 2 == 0,
                    (i * 5 + j + 1) as u64,
                )
            }).collect::<Vec<_>>();

            // Store with advanced DA
            let commitment = self.store_trading_data(&batch, &trades).await?;
            
            info!("Stored HFT batch {} with {} orders and {} trades (blob: {})",
                   batch_id, batch.orders.len(), trades.len(),
                   hex::encode(&commitment.blob_hash[..4]));

            batch_id += 1;
            
            // Small delay to simulate realistic timing
            sleep(Duration::from_millis(10)).await;
        }

        info!("High-frequency trading scenario completed - stored {} batches", batch_id - 1);
        Ok(())
    }
}

/// Run the complete integration example
pub async fn run_integration_example() -> Result<(), RollupError> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Create temporary storage directory
    let storage_path = format!("./temp_da_storage_{}", 
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());

    let mut integration = CLOBDataAvailabilityIntegration::new(storage_path.clone()).await?;

    // Run the complete example
    integration.run_complete_example().await?;

    // Run HFT scenario
    integration.demonstrate_hft_scenario().await?;

    // Show final metrics
    integration.show_performance_metrics().await?;

    // Cleanup (in a real system, you'd want to keep the data)
    if let Err(e) = std::fs::remove_dir_all(&storage_path) {
        warn!("Failed to cleanup storage directory: {}", e);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_integration_example() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().to_string_lossy().to_string();

        let mut integration = CLOBDataAvailabilityIntegration::new(storage_path).await.unwrap();
        
        // Test basic functionality
        let (batch, trades) = integration.create_sample_trading_data().await.unwrap();
        assert!(!batch.orders.is_empty());
        assert!(!trades.is_empty());

        let commitment = integration.store_trading_data(&batch, &trades).await.unwrap();
        assert!(!commitment.samples.is_empty());

        let (retrieved_batch, retrieved_trades) = integration
            .retrieve_and_verify_data(commitment.blob_hash)
            .await
            .unwrap();

        integration.verify_data_integrity(&batch, &trades, &retrieved_batch, &retrieved_trades).unwrap();
    }

    #[tokio::test]
    async fn test_hft_scenario() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path().to_string_lossy().to_string();

        let mut integration = CLOBDataAvailabilityIntegration::new(storage_path).await.unwrap();
        
        // This should complete without errors
        integration.demonstrate_hft_scenario().await.unwrap();
        
        let metrics = integration.da_client.get_advanced_metrics().await;
        assert!(metrics.samples_generated > 0);
        assert!(metrics.chunks_encoded > 0);
        assert!(metrics.commitments_created > 0);
    }
}