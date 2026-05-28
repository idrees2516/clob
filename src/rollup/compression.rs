//! Data Compression Engine for Rollup
//! 
//! Implements domain-specific compression for trading data to minimize
//! data availability costs while maintaining fast decompression.

use crate::orderbook::types::{Order, Trade, Side, OrderType};
use crate::rollup::{types::*, RollupError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

/// Compression engine for trading data
pub struct CompressionEngine {
    config: CompressionConfig,
    symbol_dictionary: SymbolDictionary,
    price_dictionary: PriceDictionary,
}

/// Symbol dictionary for compression
#[derive(Debug, Clone)]
pub struct SymbolDictionary {
    symbol_to_id: HashMap<String, u32>,
    id_to_symbol: HashMap<u32, String>,
    next_id: u32,
}

/// Price dictionary for common price levels
#[derive(Debug, Clone)]
pub struct PriceDictionary {
    price_to_id: HashMap<u64, u16>, // Fixed-point price -> ID
    id_to_price: HashMap<u16, u64>,
    next_id: u16,
    price_scale: u64,
}

/// Compressed order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressedOrder {
    symbol_id: u32,
    side: u8,
    order_type: u8,
    price_delta: i32,  // Delta from previous price
    quantity_fp: u32,  // Fixed-point quantity
    timestamp_delta: u32, // Delta from batch timestamp
    user_id_hash: u32, // Hash of user ID
}

/// Compressed trade representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressedTrade {
    symbol_id: u32,
    price_delta: i32,
    quantity_fp: u32,
    timestamp_delta: u32,
    buyer_hash: u32,
    seller_hash: u32,
}

/// Batch compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressionMetadata {
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f64,
    algorithm: CompressionAlgorithm,
    symbol_dictionary: Vec<(u32, String)>,
    base_timestamp: u64,
    base_price: u64,
}

impl CompressionEngine {
    /// Create a new compression engine
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            symbol_dictionary: SymbolDictionary::new(),
            price_dictionary: PriceDictionary::new(1_000_000), // 6 decimal places
        }
    }

    /// Compress an order batch
    pub fn compress_batch(&mut self, batch: &OrderBatch) -> Result<Vec<u8>, RollupError> {
        if batch.orders.is_empty() {
            return Ok(Vec::new());
        }

        // Compress orders
        let compressed_orders = self.compress_orders(&batch.orders)?;
        
        // Create metadata
        let metadata = CompressionMetadata {
            original_size: bincode::serialize(batch)?.len(),
            compressed_size: 0, // Will be updated after final compression
            compression_ratio: 0.0,
            algorithm: self.config.algorithm.clone(),
            symbol_dictionary: self.symbol_dictionary.export_dictionary(),
            base_timestamp: batch.timestamp,
            base_price: self.calculate_base_price(&batch.orders),
        };

        // Combine compressed data and metadata
        let mut combined_data = Vec::new();
        combined_data.extend_from_slice(&bincode::serialize(&metadata)?);
        combined_data.extend_from_slice(&compressed_orders);

        // Apply final compression algorithm
        let final_compressed = match self.config.algorithm {
            CompressionAlgorithm::Zstd => {
                zstd::encode_all(&combined_data[..], self.config.level as i32)?
            }
            CompressionAlgorithm::Lz4 => {
                lz4_flex::compress_prepend_size(&combined_data)
            }
            CompressionAlgorithm::Brotli => {
                let mut compressed = Vec::new();
                let mut encoder = brotli::CompressorWriter::new(
                    &mut compressed,
                    4096, // buffer size
                    self.config.level as u32,
                    22, // window size
                );
                encoder.write_all(&combined_data)?;
                encoder.flush()?;
                drop(encoder);
                compressed
            }
            CompressionAlgorithm::Custom(_) => {
                return Err(RollupError::CompressionError("Custom algorithm not implemented".to_string()));
            }
        };

        Ok(final_compressed)
    }

    /// Decompress an order batch
    pub fn decompress_batch(&mut self, compressed_data: &[u8]) -> Result<OrderBatch, RollupError> {
        if compressed_data.is_empty() {
            return Ok(OrderBatch::new(0, Vec::new(), 0, [0u8; 32]));
        }

        // Decompress using the appropriate algorithm
        let decompressed_data = match self.config.algorithm {
            CompressionAlgorithm::Zstd => {
                zstd::decode_all(compressed_data)?
            }
            CompressionAlgorithm::Lz4 => {
                lz4_flex::decompress_size_prepended(compressed_data)
                    .map_err(|e| RollupError::CompressionError(format!("LZ4 decompression failed: {}", e)))?
            }
            CompressionAlgorithm::Brotli => {
                let mut decompressed = Vec::new();
                let mut decoder = brotli::Decompressor::new(compressed_data, 4096);
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
            CompressionAlgorithm::Custom(_) => {
                return Err(RollupError::CompressionError("Custom algorithm not implemented".to_string()));
            }
        };

        // Extract metadata and compressed orders
        let metadata: CompressionMetadata = bincode::deserialize(&decompressed_data[..std::mem::size_of::<CompressionMetadata>()])?;
        let compressed_orders = &decompressed_data[std::mem::size_of::<CompressionMetadata>()..];

        // Restore symbol dictionary
        self.symbol_dictionary.import_dictionary(metadata.symbol_dictionary);

        // Decompress orders
        let orders = self.decompress_orders(compressed_orders, &metadata)?;

        // Reconstruct batch
        Ok(OrderBatch {
            batch_id: 0, // Will be set by caller
            orders,
            timestamp: metadata.base_timestamp,
            l1_block_number: 0, // Will be set by caller
            state_root_before: [0u8; 32],
            state_root_after: [0u8; 32],
            sequencer_signature: None,
        })
    }

    /// Compress a list of orders
    fn compress_orders(&mut self, orders: &[Order]) -> Result<Vec<u8>, RollupError> {
        if orders.is_empty() {
            return Ok(Vec::new());
        }

        let mut compressed_orders = Vec::new();
        let base_timestamp = orders[0].timestamp;
        let base_price = self.calculate_base_price(orders);
        let mut prev_price = base_price;

        for order in orders {
            // Get or create symbol ID
            let symbol_id = self.symbol_dictionary.get_or_create_id(&order.symbol);

            // Calculate price delta
            let price_fp = (order.price * self.price_dictionary.price_scale as f64) as u64;
            let price_delta = price_fp as i64 - prev_price as i64;
            prev_price = price_fp;

            // Calculate timestamp delta
            let timestamp_delta = order.timestamp.saturating_sub(base_timestamp) as u32;

            // Hash user ID for privacy
            let user_id_hash = self.hash_string(&order.user_id);

            let compressed_order = CompressedOrder {
                symbol_id,
                side: match order.side {
                    Side::Buy => 0,
                    Side::Sell => 1,
                },
                order_type: match order.order_type {
                    OrderType::Market => 0,
                    OrderType::Limit => 1,
                    OrderType::StopLoss => 2,
                    OrderType::StopLimit => 3,
                },
                price_delta: price_delta as i32,
                quantity_fp: (order.quantity * self.price_dictionary.price_scale as f64) as u32,
                timestamp_delta,
                user_id_hash,
            };

            compressed_orders.push(compressed_order);
        }

        // Serialize compressed orders
        let serialized = bincode::serialize(&compressed_orders)?;

        // Apply delta encoding if enabled
        if self.config.enable_delta_encoding {
            Ok(self.apply_delta_encoding(&serialized))
        } else {
            Ok(serialized)
        }
    }

    /// Decompress a list of orders
    fn decompress_orders(&mut self, compressed_data: &[u8], metadata: &CompressionMetadata) -> Result<Vec<Order>, RollupError> {
        if compressed_data.is_empty() {
            return Ok(Vec::new());
        }

        // Reverse delta encoding if it was applied
        let data = if self.config.enable_delta_encoding {
            self.reverse_delta_encoding(compressed_data)
        } else {
            compressed_data.to_vec()
        };

        // Deserialize compressed orders
        let compressed_orders: Vec<CompressedOrder> = bincode::deserialize(&data)?;
        let mut orders = Vec::with_capacity(compressed_orders.len());
        let mut current_price = metadata.base_price;

        for (order_id, compressed_order) in compressed_orders.into_iter().enumerate() {
            // Reconstruct price
            current_price = (current_price as i64 + compressed_order.price_delta as i64) as u64;
            let price = current_price as f64 / self.price_dictionary.price_scale as f64;

            // Reconstruct other fields
            let symbol = self.symbol_dictionary.get_symbol(compressed_order.symbol_id)
                .ok_or_else(|| RollupError::CompressionError("Unknown symbol ID".to_string()))?;

            let side = match compressed_order.side {
                0 => Side::Buy,
                1 => Side::Sell,
                _ => return Err(RollupError::CompressionError("Invalid side value".to_string())),
            };

            let order_type = match compressed_order.order_type {
                0 => OrderType::Market,
                1 => OrderType::Limit,
                2 => OrderType::StopLoss,
                3 => OrderType::StopLimit,
                _ => return Err(RollupError::CompressionError("Invalid order type value".to_string())),
            };

            let quantity = compressed_order.quantity_fp as f64 / self.price_dictionary.price_scale as f64;
            let timestamp = metadata.base_timestamp + compressed_order.timestamp_delta as u64;

            let order = Order {
                id: order_id as u64,
                symbol: symbol.clone(),
                side,
                order_type,
                price,
                quantity,
                timestamp,
                user_id: format!("user_{:08x}", compressed_order.user_id_hash), // Reconstructed user ID
            };

            orders.push(order);
        }

        Ok(orders)
    }

    /// Calculate base price for delta encoding
    fn calculate_base_price(&self, orders: &[Order]) -> u64 {
        if orders.is_empty() {
            return 0;
        }

        // Use median price as base to minimize deltas
        let mut prices: Vec<u64> = orders.iter()
            .map(|order| (order.price * self.price_dictionary.price_scale as f64) as u64)
            .collect();
        prices.sort_unstable();
        
        prices[prices.len() / 2]
    }

    /// Apply delta encoding to reduce data size
    fn apply_delta_encoding(&self, data: &[u8]) -> Vec<u8> {
        if data.len() < 2 {
            return data.to_vec();
        }

        let mut encoded = Vec::with_capacity(data.len());
        encoded.push(data[0]); // First byte unchanged

        for i in 1..data.len() {
            let delta = data[i].wrapping_sub(data[i - 1]);
            encoded.push(delta);
        }

        encoded
    }

    /// Reverse delta encoding
    fn reverse_delta_encoding(&self, encoded_data: &[u8]) -> Vec<u8> {
        if encoded_data.len() < 2 {
            return encoded_data.to_vec();
        }

        let mut decoded = Vec::with_capacity(encoded_data.len());
        decoded.push(encoded_data[0]); // First byte unchanged

        for i in 1..encoded_data.len() {
            let value = decoded[i - 1].wrapping_add(encoded_data[i]);
            decoded.push(value);
        }

        decoded
    }

    /// Hash a string to a 32-bit value
    fn hash_string(&self, s: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get compression statistics
    pub fn get_compression_stats(&self) -> CompressionStats {
        CompressionStats {
            symbol_dictionary_size: self.symbol_dictionary.size(),
            price_dictionary_size: self.price_dictionary.size(),
            algorithm: self.config.algorithm.clone(),
            compression_level: self.config.level,
        }
    }
}

impl SymbolDictionary {
    /// Create a new symbol dictionary
    pub fn new() -> Self {
        Self {
            symbol_to_id: HashMap::new(),
            id_to_symbol: HashMap::new(),
            next_id: 1, // Start from 1, reserve 0 for special cases
        }
    }

    /// Get or create an ID for a symbol
    pub fn get_or_create_id(&mut self, symbol: &str) -> u32 {
        if let Some(&id) = self.symbol_to_id.get(symbol) {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.symbol_to_id.insert(symbol.to_string(), id);
            self.id_to_symbol.insert(id, symbol.to_string());
            id
        }
    }

    /// Get symbol by ID
    pub fn get_symbol(&self, id: u32) -> Option<&String> {
        self.id_to_symbol.get(&id)
    }

    /// Export dictionary for serialization
    pub fn export_dictionary(&self) -> Vec<(u32, String)> {
        self.id_to_symbol.iter().map(|(&id, symbol)| (id, symbol.clone())).collect()
    }

    /// Import dictionary from serialized data
    pub fn import_dictionary(&mut self, data: Vec<(u32, String)>) {
        self.symbol_to_id.clear();
        self.id_to_symbol.clear();
        self.next_id = 1;

        for (id, symbol) in data {
            self.symbol_to_id.insert(symbol.clone(), id);
            self.id_to_symbol.insert(id, symbol);
            if id >= self.next_id {
                self.next_id = id + 1;
            }
        }
    }

    /// Get dictionary size
    pub fn size(&self) -> usize {
        self.symbol_to_id.len()
    }
}

impl PriceDictionary {
    /// Create a new price dictionary
    pub fn new(price_scale: u64) -> Self {
        Self {
            price_to_id: HashMap::new(),
            id_to_price: HashMap::new(),
            next_id: 1,
            price_scale,
        }
    }

    /// Get dictionary size
    pub fn size(&self) -> usize {
        self.price_to_id.len()
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub symbol_dictionary_size: usize,
    pub price_dictionary_size: usize,
    pub algorithm: CompressionAlgorithm,
    pub compression_level: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, Side, OrderType};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_test_order(id: u64, symbol: &str, price: f64, quantity: f64, side: Side) -> Order {
        Order {
            id,
            symbol: symbol.to_string(),
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            user_id: format!("user_{}", id),
        }
    }

    #[test]
    fn test_symbol_dictionary() {
        let mut dict = SymbolDictionary::new();
        
        let btc_id = dict.get_or_create_id("BTC-USD");
        let eth_id = dict.get_or_create_id("ETH-USD");
        let btc_id2 = dict.get_or_create_id("BTC-USD");
        
        assert_eq!(btc_id, btc_id2); // Same symbol should get same ID
        assert_ne!(btc_id, eth_id); // Different symbols should get different IDs
        
        assert_eq!(dict.get_symbol(btc_id), Some(&"BTC-USD".to_string()));
        assert_eq!(dict.get_symbol(eth_id), Some(&"ETH-USD".to_string()));
    }

    #[test]
    fn test_order_compression() {
        let config = CompressionConfig::default();
        let mut engine = CompressionEngine::new(config);
        
        let orders = vec![
            create_test_order(1, "BTC-USD", 50000.0, 1.0, Side::Buy),
            create_test_order(2, "BTC-USD", 50001.0, 2.0, Side::Sell),
            create_test_order(3, "ETH-USD", 3000.0, 5.0, Side::Buy),
        ];
        
        let batch = OrderBatch::new(1, orders.clone(), 100, [0u8; 32]);
        
        // Compress and decompress
        let compressed = engine.compress_batch(&batch).unwrap();
        let decompressed = engine.decompress_batch(&compressed).unwrap();
        
        assert_eq!(decompressed.orders.len(), orders.len());
        
        // Check that compression actually reduces size
        let original_size = bincode::serialize(&batch).unwrap().len();
        assert!(compressed.len() < original_size);
    }

    #[test]
    fn test_delta_encoding() {
        let config = CompressionConfig::default();
        let engine = CompressionEngine::new(config);
        
        let data = vec![100, 101, 103, 102, 105];
        let encoded = engine.apply_delta_encoding(&data);
        let decoded = engine.reverse_delta_encoding(&encoded);
        
        assert_eq!(data, decoded);
        
        // Delta encoding should help with sequential data
        let sequential_data = vec![100, 101, 102, 103, 104];
        let encoded_sequential = engine.apply_delta_encoding(&sequential_data);
        
        // Most deltas should be 1, which compresses well
        assert_eq!(encoded_sequential[0], 100); // First byte unchanged
        for i in 1..encoded_sequential.len() {
            assert_eq!(encoded_sequential[i], 1); // All deltas are 1
        }
    }

    #[test]
    fn test_compression_algorithms() {
        let orders = vec![
            create_test_order(1, "BTC-USD", 50000.0, 1.0, Side::Buy),
            create_test_order(2, "BTC-USD", 50001.0, 2.0, Side::Sell),
        ];
        let batch = OrderBatch::new(1, orders, 100, [0u8; 32]);
        
        // Test different compression algorithms
        let algorithms = vec![
            CompressionAlgorithm::Zstd,
            CompressionAlgorithm::Lz4,
        ];
        
        for algorithm in algorithms {
            let mut config = CompressionConfig::default();
            config.algorithm = algorithm.clone();
            let mut engine = CompressionEngine::new(config);
            
            let compressed = engine.compress_batch(&batch).unwrap();
            let decompressed = engine.decompress_batch(&compressed).unwrap();
            
            assert_eq!(decompressed.orders.len(), batch.orders.len());
        }
    }
}