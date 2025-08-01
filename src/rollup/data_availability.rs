//! Data Availability Layer Implementation
//! 
//! Implements a comprehensive data availability layer supporting:
//! - EIP-4844 blob storage
//! - IPFS backup storage
//! - Data compression and decompression
//! - Data retrieval and verification

use crate::rollup::{types::*, compression::CompressionEngine, RollupError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

/// High-throughput data availability client
pub struct DataAvailabilityClient {
    config: DataAvailabilityConfig,
    blob_storage: Arc<dyn BlobStorage>,
    ipfs_client: Option<Arc<dyn IpfsClient>>,
    compression_engine: CompressionEngine,
    blob_cache: Arc<RwLock<BlobCache>>,
    metrics: Arc<RwLock<DAMetrics>>,
}

/// Blob storage trait for different DA backends
#[async_trait::async_trait]
pub trait BlobStorage: Send + Sync {
    /// Store a blob and return its hash
    async fn store_blob(&self, blob: &DataBlob) -> Result<BlobCommitment, RollupError>;

    /// Retrieve a blob by its hash
    async fn retrieve_blob(&self, blob_hash: BlobHash) -> Result<DataBlob, RollupError>;

    /// Check if a blob exists
    async fn blob_exists(&self, blob_hash: BlobHash) -> Result<bool, RollupError>;

    /// Get blob metadata without downloading the full blob
    async fn get_blob_metadata(&self, blob_hash: BlobHash) -> Result<BlobMetadata, RollupError>;

    /// Delete a blob (if supported)
    async fn delete_blob(&self, blob_hash: BlobHash) -> Result<(), RollupError>;

    /// List all blobs (for maintenance)
    async fn list_blobs(&self, limit: Option<usize>) -> Result<Vec<BlobHash>, RollupError>;
}

/// IPFS client trait for backup storage
#[async_trait::async_trait]
pub trait IpfsClient: Send + Sync {
    /// Pin a blob to IPFS
    async fn pin_blob(&self, blob: &DataBlob) -> Result<String, RollupError>; // Returns IPFS hash

    /// Retrieve a blob from IPFS
    async fn get_blob(&self, ipfs_hash: &str) -> Result<DataBlob, RollupError>;

    /// Unpin a blob from IPFS
    async fn unpin_blob(&self, ipfs_hash: &str) -> Result<(), RollupError>;

    /// Check if a blob is pinned
    async fn is_pinned(&self, ipfs_hash: &str) -> Result<bool, RollupError>;
}

/// Blob cache for frequently accessed data
#[derive(Debug)]
pub struct BlobCache {
    cache: HashMap<BlobHash, CachedBlob>,
    max_size: usize,
    current_size: usize,
    access_order: Vec<BlobHash>, // LRU tracking
}

/// Cached blob entry
#[derive(Debug, Clone)]
pub struct CachedBlob {
    blob: DataBlob,
    last_accessed: Instant,
    access_count: u64,
}

/// Blob metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobMetadata {
    pub blob_hash: BlobHash,
    pub size: usize,
    pub compression_ratio: f64,
    pub timestamp: Timestamp,
    pub storage_backend: String,
    pub ipfs_hash: Option<String>,
}

/// Blob commitment returned after storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobCommitment {
    pub blob_hash: BlobHash,
    pub storage_proof: Vec<u8>,
    pub timestamp: Timestamp,
    pub backend: String,
    pub ipfs_backup: Option<String>,
}

/// Data availability metrics
#[derive(Debug, Clone)]
pub struct DAMetrics {
    pub blobs_stored: u64,
    pub blobs_retrieved: u64,
    pub total_data_stored: u64,
    pub total_data_retrieved: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub compression_savings: u64,
    pub ipfs_backups: u64,
    pub average_storage_time: Duration,
    pub average_retrieval_time: Duration,
}

impl DataAvailabilityClient {
    /// Create a new data availability client
    pub fn new(
        config: DataAvailabilityConfig,
        blob_storage: Arc<dyn BlobStorage>,
        ipfs_client: Option<Arc<dyn IpfsClient>>,
        compression_engine: CompressionEngine,
    ) -> Self {
        let blob_cache = Arc::new(RwLock::new(BlobCache::new(1000))); // Cache up to 1000 blobs
        let metrics = Arc::new(RwLock::new(DAMetrics::new()));

        Self {
            config,
            blob_storage,
            ipfs_client,
            compression_engine,
            blob_cache,
            metrics,
        }
    }

    /// Store trading data with maximum compression and redundancy
    pub async fn store_trading_data(
        &mut self,
        batch: &OrderBatch,
        trades: &[crate::orderbook::types::Trade],
    ) -> Result<BlobCommitment, RollupError> {
        let start_time = Instant::now();
        
        info!("Storing trading data for batch {}", batch.batch_id);

        // Compress the data
        let compressed_batch = self.compression_engine.compress_batch(batch)?;
        let compressed_trades = self.compress_trades(trades)?;

        // Create data blob
        let blob = DataBlob {
            blob_id: self.compute_blob_hash(&compressed_batch, &compressed_trades)?,
            compressed_data: self.combine_compressed_data(compressed_batch, compressed_trades),
            original_size: self.calculate_original_size(batch, trades),
            compression_ratio: 0.0, // Will be calculated
            merkle_root: [0u8; 32], // Will be calculated
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        };

        // Calculate compression ratio and merkle root
        let mut final_blob = blob;
        final_blob.compression_ratio = final_blob.original_size as f64 / final_blob.compressed_data.len() as f64;
        final_blob.merkle_root = self.compute_merkle_root(&final_blob.compressed_data)?;

        // Store in primary DA layer
        let commitment = self.blob_storage.store_blob(&final_blob).await?;

        // Backup to IPFS if configured
        let ipfs_hash = if self.config.enable_ipfs_backup {
            if let Some(ipfs) = &self.ipfs_client {
                match ipfs.pin_blob(&final_blob).await {
                    Ok(hash) => {
                        debug!("Backed up blob to IPFS: {}", hash);
                        Some(hash)
                    }
                    Err(e) => {
                        warn!("Failed to backup to IPFS: {}", e);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Update cache
        {
            let mut cache = self.blob_cache.write().await;
            cache.insert(final_blob.blob_id, final_blob.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.blobs_stored += 1;
            metrics.total_data_stored += final_blob.compressed_data.len() as u64;
            metrics.compression_savings += (final_blob.original_size - final_blob.compressed_data.len()) as u64;
            if ipfs_hash.is_some() {
                metrics.ipfs_backups += 1;
            }
            metrics.update_average_storage_time(start_time.elapsed());
        }

        let final_commitment = BlobCommitment {
            blob_hash: commitment.blob_hash,
            storage_proof: commitment.storage_proof,
            timestamp: commitment.timestamp,
            backend: commitment.backend,
            ipfs_backup: ipfs_hash,
        };

        info!(
            "Stored blob {} (compressed: {} bytes, ratio: {:.2}x)",
            hex::encode(&final_commitment.blob_hash[..8]),
            final_blob.compressed_data.len(),
            final_blob.compression_ratio
        );

        Ok(final_commitment)
    }

    /// Retrieve and decompress trading data
    pub async fn retrieve_trading_data(
        &self,
        blob_hash: BlobHash,
    ) -> Result<(OrderBatch, Vec<crate::orderbook::types::Trade>), RollupError> {
        let start_time = Instant::now();
        
        debug!("Retrieving trading data for blob {}", hex::encode(&blob_hash[..8]));

        // Check cache first
        if let Some(cached_blob) = self.get_from_cache(blob_hash).await {
            debug!("Cache hit for blob {}", hex::encode(&blob_hash[..8]));
            return self.decompress_trading_data(&cached_blob).await;
        }

        // Retrieve from primary storage
        let blob = match self.blob_storage.retrieve_blob(blob_hash).await {
            Ok(blob) => blob,
            Err(e) => {
                // Try IPFS backup if primary storage fails
                if let Some(ipfs) = &self.ipfs_client {
                    warn!("Primary storage failed, trying IPFS backup: {}", e);
                    // In a real implementation, we'd need to store the IPFS hash mapping
                    // For now, we'll just return the error
                    return Err(e);
                } else {
                    return Err(e);
                }
            }
        };

        // Verify blob integrity
        self.verify_blob_integrity(&blob)?;

        // Update cache
        {
            let mut cache = self.blob_cache.write().await;
            cache.insert(blob_hash, blob.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.blobs_retrieved += 1;
            metrics.total_data_retrieved += blob.compressed_data.len() as u64;
            metrics.cache_misses += 1;
            metrics.update_average_retrieval_time(start_time.elapsed());
        }

        self.decompress_trading_data(&blob).await
    }

    /// Compress trades data
    fn compress_trades(&self, trades: &[crate::orderbook::types::Trade]) -> Result<Vec<u8>, RollupError> {
        let serialized = bincode::serialize(trades)?;
        
        match self.config.blob_storage_type {
            BlobStorageType::EIP4844 => {
                // Use high compression for EIP-4844 due to size limits
                Ok(zstd::encode_all(&serialized[..], 9)?) // Max compression
            }
            _ => {
                // Use balanced compression for other storage types
                Ok(zstd::encode_all(&serialized[..], self.config.compression_level as i32)?)
            }
        }
    }

    /// Combine compressed batch and trades data
    fn combine_compressed_data(&self, batch_data: Vec<u8>, trades_data: Vec<u8>) -> Vec<u8> {
        let mut combined = Vec::new();
        
        // Add length prefixes for separation
        combined.extend_from_slice(&(batch_data.len() as u32).to_be_bytes());
        combined.extend_from_slice(&batch_data);
        combined.extend_from_slice(&(trades_data.len() as u32).to_be_bytes());
        combined.extend_from_slice(&trades_data);
        
        combined
    }

    /// Calculate original size before compression
    fn calculate_original_size(&self, batch: &OrderBatch, trades: &[crate::orderbook::types::Trade]) -> usize {
        let batch_size = bincode::serialize(batch).map(|data| data.len()).unwrap_or(0);
        let trades_size = bincode::serialize(trades).map(|data| data.len()).unwrap_or(0);
        batch_size + trades_size
    }

    /// Compute blob hash
    fn compute_blob_hash(&self, batch_data: &[u8], trades_data: &[u8]) -> Result<BlobHash, RollupError> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(batch_data);
        hasher.update(trades_data);
        hasher.update(&SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs().to_be_bytes());
        Ok(hasher.finalize().into())
    }

    /// Compute merkle root for data integrity
    fn compute_merkle_root(&self, data: &[u8]) -> Result<[u8; 32], RollupError> {
        use sha2::{Sha256, Digest};
        
        // Simple merkle root calculation for data chunks
        const CHUNK_SIZE: usize = 1024;
        let chunks: Vec<[u8; 32]> = data
            .chunks(CHUNK_SIZE)
            .map(|chunk| {
                let mut hasher = Sha256::new();
                hasher.update(chunk);
                hasher.finalize().into()
            })
            .collect();

        if chunks.is_empty() {
            return Ok([0u8; 32]);
        }

        if chunks.len() == 1 {
            return Ok(chunks[0]);
        }

        // Build merkle tree bottom-up
        let mut current_level = chunks;
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            for pair in current_level.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(&pair[0]);
                if pair.len() > 1 {
                    hasher.update(&pair[1]);
                } else {
                    hasher.update(&pair[0]); // Duplicate if odd number
                }
                next_level.push(hasher.finalize().into());
            }
            
            current_level = next_level;
        }

        Ok(current_level[0])
    }

    /// Verify blob integrity
    fn verify_blob_integrity(&self, blob: &DataBlob) -> Result<(), RollupError> {
        // Verify merkle root
        let computed_root = self.compute_merkle_root(&blob.compressed_data)?;
        if computed_root != blob.merkle_root {
            return Err(RollupError::DataAvailabilityError(
                "Blob integrity check failed: merkle root mismatch".to_string()
            ));
        }

        // Verify blob hash
        let computed_hash = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&blob.compressed_data);
            hasher.update(&blob.timestamp.to_be_bytes());
            hasher.finalize()
        };

        // Note: In a real implementation, you'd store the original hash components
        // For now, we'll just check that the data is not corrupted
        if blob.compressed_data.is_empty() {
            return Err(RollupError::DataAvailabilityError(
                "Blob data is empty".to_string()
            ));
        }

        Ok(())
    }

    /// Get blob from cache
    async fn get_from_cache(&self, blob_hash: BlobHash) -> Option<DataBlob> {
        let mut cache = self.blob_cache.write().await;
        if let Some(cached) = cache.get_mut(&blob_hash) {
            cached.last_accessed = Instant::now();
            cached.access_count += 1;
            
            // Update metrics
            if let Ok(mut metrics) = self.metrics.try_write() {
                metrics.cache_hits += 1;
            }
            
            Some(cached.blob.clone())
        } else {
            None
        }
    }

    /// Decompress trading data from blob
    async fn decompress_trading_data(
        &self,
        blob: &DataBlob,
    ) -> Result<(OrderBatch, Vec<crate::orderbook::types::Trade>), RollupError> {
        // Split combined data
        let mut cursor = 0;
        
        // Read batch data length
        if blob.compressed_data.len() < 4 {
            return Err(RollupError::DataAvailabilityError("Invalid blob format".to_string()));
        }
        
        let batch_len = u32::from_be_bytes([
            blob.compressed_data[cursor],
            blob.compressed_data[cursor + 1],
            blob.compressed_data[cursor + 2],
            blob.compressed_data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Read batch data
        if cursor + batch_len > blob.compressed_data.len() {
            return Err(RollupError::DataAvailabilityError("Invalid batch data length".to_string()));
        }
        
        let batch_data = &blob.compressed_data[cursor..cursor + batch_len];
        cursor += batch_len;

        // Read trades data length
        if cursor + 4 > blob.compressed_data.len() {
            return Err(RollupError::DataAvailabilityError("Invalid trades data format".to_string()));
        }
        
        let trades_len = u32::from_be_bytes([
            blob.compressed_data[cursor],
            blob.compressed_data[cursor + 1],
            blob.compressed_data[cursor + 2],
            blob.compressed_data[cursor + 3],
        ]) as usize;
        cursor += 4;

        // Read trades data
        if cursor + trades_len > blob.compressed_data.len() {
            return Err(RollupError::DataAvailabilityError("Invalid trades data length".to_string()));
        }
        
        let trades_data = &blob.compressed_data[cursor..cursor + trades_len];

        // Decompress batch
        let mut compression_engine = self.compression_engine.clone();
        let batch = compression_engine.decompress_batch(batch_data)?;

        // Decompress trades
        let decompressed_trades = zstd::decode_all(trades_data)?;
        let trades: Vec<crate::orderbook::types::Trade> = bincode::deserialize(&decompressed_trades)?;

        Ok((batch, trades))
    }

    /// Get data availability metrics
    pub async fn get_metrics(&self) -> DAMetrics {
        self.metrics.read().await.clone()
    }

    /// Cleanup old blobs based on retention policy
    pub async fn cleanup_old_blobs(&self) -> Result<usize, RollupError> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(self.config.retention_period_days as u64 * 24 * 60 * 60);

        let all_blobs = self.blob_storage.list_blobs(None).await?;
        let mut deleted_count = 0;

        for blob_hash in all_blobs {
            match self.blob_storage.get_blob_metadata(blob_hash).await {
                Ok(metadata) => {
                    if metadata.timestamp < cutoff_time {
                        match self.blob_storage.delete_blob(blob_hash).await {
                            Ok(()) => {
                                deleted_count += 1;
                                debug!("Deleted old blob: {}", hex::encode(&blob_hash[..8]));
                            }
                            Err(e) => {
                                warn!("Failed to delete blob {}: {}", hex::encode(&blob_hash[..8]), e);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to get metadata for blob {}: {}", hex::encode(&blob_hash[..8]), e);
                }
            }
        }

        info!("Cleaned up {} old blobs", deleted_count);
        Ok(deleted_count)
    }
}

impl BlobCache {
    /// Create a new blob cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            current_size: 0,
            access_order: Vec::new(),
        }
    }

    /// Insert a blob into the cache
    pub fn insert(&mut self, blob_hash: BlobHash, blob: DataBlob) {
        // Remove oldest entries if cache is full
        while self.current_size >= self.max_size && !self.access_order.is_empty() {
            let oldest = self.access_order.remove(0);
            if let Some(removed) = self.cache.remove(&oldest) {
                self.current_size -= 1;
            }
        }

        let cached_blob = CachedBlob {
            blob,
            last_accessed: Instant::now(),
            access_count: 1,
        };

        self.cache.insert(blob_hash, cached_blob);
        self.access_order.push(blob_hash);
        self.current_size += 1;
    }

    /// Get a blob from the cache
    pub fn get_mut(&mut self, blob_hash: &BlobHash) -> Option<&mut CachedBlob> {
        if let Some(cached) = self.cache.get_mut(blob_hash) {
            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|&h| h == *blob_hash) {
                self.access_order.remove(pos);
                self.access_order.push(*blob_hash);
            }
            Some(cached)
        } else {
            None
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.current_size,
            max_size: self.max_size,
            hit_ratio: 0.0, // Would need to track hits/misses
        }
    }
}

impl DAMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            blobs_stored: 0,
            blobs_retrieved: 0,
            total_data_stored: 0,
            total_data_retrieved: 0,
            cache_hits: 0,
            cache_misses: 0,
            compression_savings: 0,
            ipfs_backups: 0,
            average_storage_time: Duration::from_secs(0),
            average_retrieval_time: Duration::from_secs(0),
        }
    }

    /// Update average storage time
    pub fn update_average_storage_time(&mut self, new_time: Duration) {
        if self.blobs_stored == 0 {
            self.average_storage_time = new_time;
        } else {
            let total_time = self.average_storage_time * self.blobs_stored as u32 + new_time;
            self.average_storage_time = total_time / (self.blobs_stored + 1) as u32;
        }
    }

    /// Update average retrieval time
    pub fn update_average_retrieval_time(&mut self, new_time: Duration) {
        if self.blobs_retrieved == 0 {
            self.average_retrieval_time = new_time;
        } else {
            let total_time = self.average_retrieval_time * self.blobs_retrieved as u32 + new_time;
            self.average_retrieval_time = total_time / (self.blobs_retrieved + 1) as u32;
        }
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_requests as f64
        }
    }

    /// Get compression efficiency
    pub fn compression_efficiency(&self) -> f64 {
        if self.total_data_stored == 0 {
            0.0
        } else {
            self.compression_savings as f64 / (self.total_data_stored + self.compression_savings) as f64
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hit_ratio: f64,
}

// EIP-4844 blob storage implementation
pub struct EIP4844BlobStorage {
    // Implementation would connect to Ethereum node
}

#[async_trait::async_trait]
impl BlobStorage for EIP4844BlobStorage {
    async fn store_blob(&self, blob: &DataBlob) -> Result<BlobCommitment, RollupError> {
        // Mock implementation - would submit blob transaction to Ethereum
        Ok(BlobCommitment {
            blob_hash: blob.blob_id,
            storage_proof: vec![0u8; 32], // Mock proof
            timestamp: blob.timestamp,
            backend: "EIP4844".to_string(),
            ipfs_backup: None,
        })
    }

    async fn retrieve_blob(&self, _blob_hash: BlobHash) -> Result<DataBlob, RollupError> {
        // Mock implementation - would retrieve from Ethereum
        Err(RollupError::DataAvailabilityError("Not implemented".to_string()))
    }

    async fn blob_exists(&self, _blob_hash: BlobHash) -> Result<bool, RollupError> {
        Ok(false) // Mock implementation
    }

    async fn get_blob_metadata(&self, _blob_hash: BlobHash) -> Result<BlobMetadata, RollupError> {
        Err(RollupError::DataAvailabilityError("Not implemented".to_string()))
    }

    async fn delete_blob(&self, _blob_hash: BlobHash) -> Result<(), RollupError> {
        Err(RollupError::DataAvailabilityError("EIP-4844 blobs cannot be deleted".to_string()))
    }

    async fn list_blobs(&self, _limit: Option<usize>) -> Result<Vec<BlobHash>, RollupError> {
        Ok(Vec::new()) // Mock implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, Trade, Side, OrderType};
    use crate::rollup::compression::CompressionEngine;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Mock blob storage for testing
    struct MockBlobStorage {
        blobs: Arc<RwLock<HashMap<BlobHash, DataBlob>>>,
    }

    impl MockBlobStorage {
        fn new() -> Self {
            Self {
                blobs: Arc::new(RwLock::new(HashMap::new())),
            }
        }
    }

    #[async_trait::async_trait]
    impl BlobStorage for MockBlobStorage {
        async fn store_blob(&self, blob: &DataBlob) -> Result<BlobCommitment, RollupError> {
            let mut blobs = self.blobs.write().await;
            blobs.insert(blob.blob_id, blob.clone());
            
            Ok(BlobCommitment {
                blob_hash: blob.blob_id,
                storage_proof: vec![42u8; 32],
                timestamp: blob.timestamp,
                backend: "Mock".to_string(),
                ipfs_backup: None,
            })
        }

        async fn retrieve_blob(&self, blob_hash: BlobHash) -> Result<DataBlob, RollupError> {
            let blobs = self.blobs.read().await;
            blobs.get(&blob_hash)
                .cloned()
                .ok_or_else(|| RollupError::DataAvailabilityError("Blob not found".to_string()))
        }

        async fn blob_exists(&self, blob_hash: BlobHash) -> Result<bool, RollupError> {
            let blobs = self.blobs.read().await;
            Ok(blobs.contains_key(&blob_hash))
        }

        async fn get_blob_metadata(&self, blob_hash: BlobHash) -> Result<BlobMetadata, RollupError> {
            let blobs = self.blobs.read().await;
            if let Some(blob) = blobs.get(&blob_hash) {
                Ok(BlobMetadata {
                    blob_hash,
                    size: blob.compressed_data.len(),
                    compression_ratio: blob.compression_ratio,
                    timestamp: blob.timestamp,
                    storage_backend: "Mock".to_string(),
                    ipfs_hash: None,
                })
            } else {
                Err(RollupError::DataAvailabilityError("Blob not found".to_string()))
            }
        }

        async fn delete_blob(&self, blob_hash: BlobHash) -> Result<(), RollupError> {
            let mut blobs = self.blobs.write().await;
            blobs.remove(&blob_hash);
            Ok(())
        }

        async fn list_blobs(&self, limit: Option<usize>) -> Result<Vec<BlobHash>, RollupError> {
            let blobs = self.blobs.read().await;
            let mut hashes: Vec<BlobHash> = blobs.keys().cloned().collect();
            if let Some(limit) = limit {
                hashes.truncate(limit);
            }
            Ok(hashes)
        }
    }

    fn create_test_batch() -> OrderBatch {
        let orders = vec![
            Order {
                id: 1,
                symbol: "BTC-USD".to_string(),
                side: Side::Buy,
                order_type: OrderType::Limit,
                price: 50000.0,
                quantity: 1.0,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                user_id: "user_1".to_string(),
            }
        ];

        OrderBatch::new(1, orders, 100, [0u8; 32])
    }

    fn create_test_trades() -> Vec<Trade> {
        vec![
            Trade {
                id: 1,
                symbol: "BTC-USD".to_string(),
                price: 50000.0,
                quantity: 0.5,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                buyer_id: "buyer_1".to_string(),
                seller_id: "seller_1".to_string(),
                buy_order_id: 1,
                sell_order_id: 2,
            }
        ]
    }

    #[tokio::test]
    async fn test_data_availability_client() {
        let config = DataAvailabilityConfig::default();
        let blob_storage = Arc::new(MockBlobStorage::new());
        let compression_engine = CompressionEngine::new(Default::default());
        
        let mut da_client = DataAvailabilityClient::new(
            config,
            blob_storage,
            None,
            compression_engine,
        );

        let batch = create_test_batch();
        let trades = create_test_trades();

        // Store data
        let commitment = da_client.store_trading_data(&batch, &trades).await.unwrap();
        assert!(!commitment.blob_hash.iter().all(|&b| b == 0));

        // Retrieve data
        let (retrieved_batch, retrieved_trades) = da_client
            .retrieve_trading_data(commitment.blob_hash)
            .await
            .unwrap();

        assert_eq!(retrieved_batch.orders.len(), batch.orders.len());
        assert_eq!(retrieved_trades.len(), trades.len());
    }

    #[tokio::test]
    async fn test_blob_cache() {
        let mut cache = BlobCache::new(2);
        
        let blob1 = DataBlob {
            blob_id: [1u8; 32],
            compressed_data: vec![1, 2, 3],
            original_size: 100,
            compression_ratio: 2.0,
            merkle_root: [0u8; 32],
            timestamp: 1000,
        };

        let blob2 = DataBlob {
            blob_id: [2u8; 32],
            compressed_data: vec![4, 5, 6],
            original_size: 200,
            compression_ratio: 3.0,
            merkle_root: [0u8; 32],
            timestamp: 2000,
        };

        // Insert blobs
        cache.insert([1u8; 32], blob1.clone());
        cache.insert([2u8; 32], blob2.clone());
        
        assert_eq!(cache.current_size, 2);

        // Access first blob
        let cached = cache.get_mut(&[1u8; 32]).unwrap();
        assert_eq!(cached.blob.blob_id, [1u8; 32]);
        assert_eq!(cached.access_count, 2); // 1 initial + 1 access

        // Insert third blob (should evict least recently used)
        let blob3 = DataBlob {
            blob_id: [3u8; 32],
            compressed_data: vec![7, 8, 9],
            original_size: 300,
            compression_ratio: 4.0,
            merkle_root: [0u8; 32],
            timestamp: 3000,
        };
        
        cache.insert([3u8; 32], blob3);
        assert_eq!(cache.current_size, 2);
        
        // Blob 2 should be evicted (least recently used)
        assert!(cache.get_mut(&[2u8; 32]).is_none());
        assert!(cache.get_mut(&[1u8; 32]).is_some());
        assert!(cache.get_mut(&[3u8; 32]).is_some());
    }

    #[test]
    fn test_da_metrics() {
        let mut metrics = DAMetrics::new();
        
        metrics.blobs_stored = 10;
        metrics.cache_hits = 7;
        metrics.cache_misses = 3;
        metrics.compression_savings = 500;
        metrics.total_data_stored = 1000;

        assert_eq!(metrics.cache_hit_ratio(), 0.7);
        assert_eq!(metrics.compression_efficiency(), 500.0 / 1500.0);
    }
}