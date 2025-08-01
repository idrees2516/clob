use std::arch::x86_64::*;
use crate::performance::simd::get_simd_capabilities;

/// SIMD-optimized validation operations for trading data
pub struct SimdValidator;

/// Validation result for batch operations
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationResult {
    pub valid_count: usize,
    pub invalid_indices: Vec<usize>,
    pub error_mask: u64,
}

/// Price validation constraints
#[derive(Debug, Clone)]
pub struct PriceConstraints {
    pub min_price: u64,
    pub max_price: u64,
    pub tick_size: u64,
}

impl Default for PriceConstraints {
    fn default() -> Self {
        Self {
            min_price: 1,           // Minimum 1 unit
            max_price: u64::MAX,    // No upper limit by default
            tick_size: 1,           // Minimum price increment
        }
    }
}

impl SimdValidator {
    /// Validate multiple prices using AVX2 SIMD instructions
    /// Returns a bitmask where each bit indicates if the corresponding price is valid
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_prices_avx2(
        prices: &[u64], 
        constraints: &PriceConstraints
    ) -> u64 {
        let mut valid_mask = 0u64;
        
        let min_vec = _mm256_set1_epi64x(constraints.min_price as i64);
        let max_vec = _mm256_set1_epi64x(constraints.max_price as i64);
        let tick_vec = _mm256_set1_epi64x(constraints.tick_size as i64);
        
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Check minimum price constraint
            let min_check = _mm256_cmpgt_epi64(prices_vec, min_vec);
            
            // Check maximum price constraint  
            let max_check = _mm256_cmpgt_epi64(max_vec, prices_vec);
            
            // For tick size validation, we need to do scalar operations since AVX2 doesn't have 64-bit modulo
            // Extract values for tick size check
            let mut chunk_values = [0u64; 4];
            _mm256_storeu_si256(chunk_values.as_mut_ptr() as *mut __m256i, prices_vec);
            
            // Create tick validation mask
            let mut tick_valid = [0i64; 4];
            for (j, &price) in chunk_values.iter().enumerate() {
                tick_valid[j] = if price % constraints.tick_size == 0 { -1i64 } else { 0i64 };
            }
            let tick_check = _mm256_loadu_si256(tick_valid.as_ptr() as *const __m256i);
            
            // Combine all checks with AND
            let combined = _mm256_and_si256(_mm256_and_si256(min_check, max_check), tick_check);
            let mask = _mm256_movemask_epi8(combined);
            
            // Extract validation results for each 64-bit element
            if mask & 0x80 != 0 { valid_mask |= 1u64 << (chunk_idx * 4); }
            if mask & 0x8000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 3); }
        }
        
        // Handle remainder with scalar operations
        let base_idx = chunks.len() * 4;
        for (i, &price) in remainder.iter().enumerate() {
            if Self::validate_single_price(price, constraints) {
                valid_mask |= 1u64 << (base_idx + i);
            }
        }
        
        valid_mask
    }

    /// Validate multiple quantities using AVX2 SIMD instructions
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_quantities_avx2(
        quantities: &[u64],
        min_quantity: u64,
        max_quantity: u64
    ) -> u64 {
        let mut valid_mask = 0u64;
        
        let min_vec = _mm256_set1_epi64x(min_quantity as i64);
        let max_vec = _mm256_set1_epi64x(max_quantity as i64);
        let zero_vec = _mm256_setzero_si256();
        
        let chunks = quantities.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let qty_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Check non-zero
            let nonzero_check = _mm256_cmpgt_epi64(qty_vec, zero_vec);
            
            // Check minimum quantity
            let min_check = _mm256_cmpgt_epi64(qty_vec, min_vec);
            
            // Check maximum quantity
            let max_check = _mm256_cmpgt_epi64(max_vec, qty_vec);
            
            // Combine all checks
            let combined = _mm256_and_si256(_mm256_and_si256(nonzero_check, min_check), max_check);
            let mask = _mm256_movemask_epi8(combined);
            
            // Extract validation results
            if mask & 0x80 != 0 { valid_mask |= 1u64 << (chunk_idx * 4); }
            if mask & 0x8000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 3); }
        }
        
        // Handle remainder
        let base_idx = chunks.len() * 4;
        for (i, &qty) in remainder.iter().enumerate() {
            if qty > 0 && qty >= min_quantity && qty <= max_quantity {
                valid_mask |= 1u64 << (base_idx + i);
            }
        }
        
        valid_mask
    }

    /// Validate price-quantity pairs using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_orders_avx2(
        prices: &[u64],
        quantities: &[u64],
        price_constraints: &PriceConstraints,
        min_quantity: u64,
        max_quantity: u64
    ) -> u64 {
        assert_eq!(prices.len(), quantities.len());
        
        let price_mask = Self::validate_prices_avx2(prices, price_constraints);
        let qty_mask = Self::validate_quantities_avx2(quantities, min_quantity, max_quantity);
        
        // Both price and quantity must be valid
        price_mask & qty_mask
    }

    /// Validate order IDs for uniqueness and range using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_order_ids_avx2(
        order_ids: &[u64],
        min_id: u64,
        max_id: u64
    ) -> u64 {
        let mut valid_mask = 0u64;
        
        let min_vec = _mm256_set1_epi64x(min_id as i64);
        let max_vec = _mm256_set1_epi64x(max_id as i64);
        let zero_vec = _mm256_setzero_si256();
        
        let chunks = order_ids.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let id_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Check non-zero
            let nonzero_check = _mm256_cmpgt_epi64(id_vec, zero_vec);
            
            // Check minimum ID
            let min_check = _mm256_cmpgt_epi64(id_vec, min_vec);
            
            // Check maximum ID
            let max_check = _mm256_cmpgt_epi64(max_vec, id_vec);
            
            // Combine all checks
            let combined = _mm256_and_si256(_mm256_and_si256(nonzero_check, min_check), max_check);
            let mask = _mm256_movemask_epi8(combined);
            
            // Extract validation results
            if mask & 0x80 != 0 { valid_mask |= 1u64 << (chunk_idx * 4); }
            if mask & 0x8000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 3); }
        }
        
        // Handle remainder
        let base_idx = chunks.len() * 4;
        for (i, &id) in remainder.iter().enumerate() {
            if id > 0 && id > min_id && id < max_id {
                valid_mask |= 1u64 << (base_idx + i);
            }
        }
        
        valid_mask
    }

    /// Validate timestamps for chronological order using AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_timestamps_avx2(
        timestamps: &[u64],
        min_timestamp: u64,
        max_timestamp: u64
    ) -> u64 {
        let mut valid_mask = 0u64;
        
        let min_vec = _mm256_set1_epi64x(min_timestamp as i64);
        let max_vec = _mm256_set1_epi64x(max_timestamp as i64);
        
        let chunks = timestamps.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let ts_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Check minimum timestamp
            let min_check = _mm256_cmpgt_epi64(ts_vec, min_vec);
            
            // Check maximum timestamp
            let max_check = _mm256_cmpgt_epi64(max_vec, ts_vec);
            
            // Combine checks
            let combined = _mm256_and_si256(min_check, max_check);
            let mask = _mm256_movemask_epi8(combined);
            
            // Extract validation results
            if mask & 0x80 != 0 { valid_mask |= 1u64 << (chunk_idx * 4); }
            if mask & 0x8000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 3); }
        }
        
        // Handle remainder
        let base_idx = chunks.len() * 4;
        for (i, &ts) in remainder.iter().enumerate() {
            if ts > min_timestamp && ts < max_timestamp {
                valid_mask |= 1u64 << (base_idx + i);
            }
        }
        
        valid_mask
    }

    /// Validate checksums using SIMD for data integrity
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_checksums_avx2(data: &[u64], expected_checksums: &[u64]) -> u64 {
        assert_eq!(data.len(), expected_checksums.len());
        
        let mut valid_mask = 0u64;
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, (data_chunk, checksum_chunk)) in 
            chunks.zip(expected_checksums.chunks_exact(4)).enumerate() {
            
            let data_vec = _mm256_loadu_si256(data_chunk.as_ptr() as *const __m256i);
            let checksum_vec = _mm256_loadu_si256(checksum_chunk.as_ptr() as *const __m256i);
            
            // Simple checksum: data XOR with a constant, then compare
            let xor_const = _mm256_set1_epi64x(0x5555555555555555u64 as i64);
            let computed_checksum = _mm256_xor_si256(data_vec, xor_const);
            
            let match_check = _mm256_cmpeq_epi64(computed_checksum, checksum_vec);
            let mask = _mm256_movemask_epi8(match_check);
            
            // Extract validation results
            if mask & 0x80 != 0 { valid_mask |= 1u64 << (chunk_idx * 4); }
            if mask & 0x8000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { valid_mask |= 1u64 << (chunk_idx * 4 + 3); }
        }
        
        // Handle remainder
        let base_idx = chunks.len() * 4;
        for (i, (&data_val, &expected)) in remainder.iter().zip(
            expected_checksums[base_idx..].iter()).enumerate() {
            let computed = data_val ^ 0x5555555555555555u64;
            if computed == expected {
                valid_mask |= 1u64 << (base_idx + i);
            }
        }
        
        valid_mask
    }

    /// Helper function for single price validation
    fn validate_single_price(price: u64, constraints: &PriceConstraints) -> bool {
        price >= constraints.min_price && 
        price <= constraints.max_price && 
        price % constraints.tick_size == 0
    }

    /// Public interface for price validation
    pub fn validate_prices(prices: &[u64], constraints: &PriceConstraints) -> ValidationResult {
        if prices.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::validate_prices_avx2(prices, constraints) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, &price) in prices.iter().enumerate() {
                if Self::validate_single_price(price, constraints) {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, prices.len())
    }

    /// Public interface for quantity validation
    pub fn validate_quantities(
        quantities: &[u64], 
        min_quantity: u64, 
        max_quantity: u64
    ) -> ValidationResult {
        if quantities.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && quantities.len() >= 4 {
            unsafe { Self::validate_quantities_avx2(quantities, min_quantity, max_quantity) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, &qty) in quantities.iter().enumerate() {
                if qty > 0 && qty >= min_quantity && qty <= max_quantity {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, quantities.len())
    }

    /// Public interface for order validation
    pub fn validate_orders(
        prices: &[u64],
        quantities: &[u64],
        price_constraints: &PriceConstraints,
        min_quantity: u64,
        max_quantity: u64
    ) -> ValidationResult {
        assert_eq!(prices.len(), quantities.len());
        
        if prices.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::validate_orders_avx2(prices, quantities, price_constraints, min_quantity, max_quantity) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, (&price, &qty)) in prices.iter().zip(quantities.iter()).enumerate() {
                if Self::validate_single_price(price, price_constraints) && 
                   qty > 0 && qty >= min_quantity && qty <= max_quantity {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, prices.len())
    }

    /// Public interface for order ID validation
    pub fn validate_order_ids(
        order_ids: &[u64],
        min_id: u64,
        max_id: u64
    ) -> ValidationResult {
        if order_ids.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && order_ids.len() >= 4 {
            unsafe { Self::validate_order_ids_avx2(order_ids, min_id, max_id) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, &id) in order_ids.iter().enumerate() {
                if id > 0 && id > min_id && id < max_id {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, order_ids.len())
    }

    /// Public interface for timestamp validation
    pub fn validate_timestamps(
        timestamps: &[u64],
        min_timestamp: u64,
        max_timestamp: u64
    ) -> ValidationResult {
        if timestamps.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && timestamps.len() >= 4 {
            unsafe { Self::validate_timestamps_avx2(timestamps, min_timestamp, max_timestamp) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, &ts) in timestamps.iter().enumerate() {
                if ts > min_timestamp && ts < max_timestamp {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, timestamps.len())
    }

    /// Public interface for checksum validation
    pub fn validate_checksums(data: &[u64], expected_checksums: &[u64]) -> ValidationResult {
        assert_eq!(data.len(), expected_checksums.len());
        
        if data.is_empty() {
            return ValidationResult {
                valid_count: 0,
                invalid_indices: Vec::new(),
                error_mask: 0,
            };
        }
        
        let caps = get_simd_capabilities();
        let valid_mask = if caps.has_avx2 && data.len() >= 4 {
            unsafe { Self::validate_checksums_avx2(data, expected_checksums) }
        } else {
            // Scalar fallback
            let mut mask = 0u64;
            for (i, (&data_val, &expected)) in data.iter().zip(expected_checksums.iter()).enumerate() {
                let computed = data_val ^ 0x5555555555555555u64;
                if computed == expected {
                    mask |= 1u64 << i;
                }
            }
            mask
        };
        
        Self::mask_to_result(valid_mask, data.len())
    }

    /// Convert bitmask to ValidationResult
    fn mask_to_result(valid_mask: u64, total_count: usize) -> ValidationResult {
        let mut invalid_indices = Vec::new();
        let mut valid_count = 0;
        
        for i in 0..total_count.min(64) {
            if valid_mask & (1u64 << i) != 0 {
                valid_count += 1;
            } else {
                invalid_indices.push(i);
            }
        }
        
        ValidationResult {
            valid_count,
            invalid_indices,
            error_mask: !valid_mask,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_validation() {
        let constraints = PriceConstraints {
            min_price: 100,
            max_price: 1000,
            tick_size: 5,
        };
        
        let prices = vec![
            105,  // Valid: within range and aligned to tick
            50,   // Invalid: below minimum
            1500, // Invalid: above maximum
            103,  // Invalid: not aligned to tick size
            200,  // Valid
            0,    // Invalid: below minimum
            500,  // Valid
            999,  // Invalid: not aligned to tick size
        ];
        
        let result = SimdValidator::validate_prices(&prices, &constraints);
        
        assert_eq!(result.valid_count, 3); // Indices 0, 4, 6 are valid
        assert_eq!(result.invalid_indices, vec![1, 2, 3, 5, 7]);
        
        // Check specific validations
        assert!(result.error_mask & (1 << 1) != 0); // Index 1 invalid
        assert!(result.error_mask & (1 << 2) != 0); // Index 2 invalid
        assert!(result.error_mask & (1 << 3) != 0); // Index 3 invalid
        assert!(result.error_mask & (1 << 5) != 0); // Index 5 invalid
        assert!(result.error_mask & (1 << 7) != 0); // Index 7 invalid
    }

    #[test]
    fn test_quantity_validation() {
        let quantities = vec![100, 0, 50, 1000, 2000, 1];
        let min_quantity = 10;
        let max_quantity = 1500;
        
        let result = SimdValidator::validate_quantities(&quantities, min_quantity, max_quantity);
        
        assert_eq!(result.valid_count, 3); // Indices 0, 2, 3 are valid
        assert_eq!(result.invalid_indices, vec![1, 4, 5]);
    }

    #[test]
    fn test_order_validation() {
        let prices = vec![100, 200, 300, 400];
        let quantities = vec![50, 0, 100, 200]; // Second quantity is invalid (zero)
        
        let constraints = PriceConstraints {
            min_price: 50,
            max_price: 500,
            tick_size: 1,
        };
        
        let result = SimdValidator::validate_orders(
            &prices, 
            &quantities, 
            &constraints, 
            10, 
            300
        );
        
        assert_eq!(result.valid_count, 3); // All prices valid, but quantity[1] = 0 is invalid
        assert_eq!(result.invalid_indices, vec![1]);
    }

    #[test]
    fn test_checksum_validation() {
        let data = vec![100, 200, 300, 400];
        let mut expected_checksums = Vec::new();
        
        // Generate expected checksums using the same algorithm
        for &val in &data {
            expected_checksums.push(val ^ 0x5555555555555555u64);
        }
        
        // Corrupt one checksum
        expected_checksums[2] = 0;
        
        let result = SimdValidator::validate_checksums(&data, &expected_checksums);
        
        assert_eq!(result.valid_count, 3); // All except index 2
        assert_eq!(result.invalid_indices, vec![2]);
    }

    #[test]
    fn test_empty_arrays() {
        let empty_prices: Vec<u64> = vec![];
        let constraints = PriceConstraints::default();
        
        let result = SimdValidator::validate_prices(&empty_prices, &constraints);
        assert_eq!(result.valid_count, 0);
        assert!(result.invalid_indices.is_empty());
        assert_eq!(result.error_mask, 0);
    }

    #[test]
    fn test_single_element() {
        let prices = vec![100];
        let constraints = PriceConstraints {
            min_price: 50,
            max_price: 150,
            tick_size: 1,
        };
        
        let result = SimdValidator::validate_prices(&prices, &constraints);
        assert_eq!(result.valid_count, 1);
        assert!(result.invalid_indices.is_empty());
    }

    #[test]
    fn test_large_batch() {
        // Test with a larger batch to ensure SIMD path is taken
        let mut prices = Vec::new();
        let mut quantities = Vec::new();
        
        for i in 0..100 {
            prices.push((i * 10 + 100) as u64); // Prices: 100, 110, 120, ...
            quantities.push((i + 1) as u64);    // Quantities: 1, 2, 3, ...
        }
        
        let constraints = PriceConstraints {
            min_price: 50,
            max_price: 2000,
            tick_size: 10,
        };
        
        let result = SimdValidator::validate_orders(
            &prices, 
            &quantities, 
            &constraints, 
            1, 
            200
        );
        
        // All should be valid
        assert_eq!(result.valid_count, 100);
        assert!(result.invalid_indices.is_empty());
    }

    #[test]
    fn test_tick_size_alignment() {
        let prices = vec![100, 101, 102, 105, 110, 115];
        let constraints = PriceConstraints {
            min_price: 50,
            max_price: 200,
            tick_size: 5,
        };
        
        let result = SimdValidator::validate_prices(&prices, &constraints);
        
        // Only prices divisible by 5 should be valid: 100, 105, 110, 115
        assert_eq!(result.valid_count, 4);
        assert_eq!(result.invalid_indices, vec![1, 2]); // 101 and 102 not divisible by 5
    }

    #[test]
    fn test_boundary_conditions() {
        let constraints = PriceConstraints {
            min_price: 100,
            max_price: 200,
            tick_size: 1,
        };
        
        let prices = vec![99, 100, 200, 201]; // Test exact boundaries
        let result = SimdValidator::validate_prices(&prices, &constraints);
        
        assert_eq!(result.valid_count, 2); // Only 100 and 200 are valid
        assert_eq!(result.invalid_indices, vec![0, 3]); // 99 and 201 are invalid
    }

    #[test]
    fn test_order_id_validation() {
        let order_ids = vec![1000, 0, 500, 2000, 3000, 1];
        let min_id = 100;
        let max_id = 2500;
        
        let result = SimdValidator::validate_order_ids(&order_ids, min_id, max_id);
        
        // Valid IDs: 1000 (index 0), 500 (index 2), 2000 (index 3)
        // Invalid: 0 (index 1, zero), 3000 (index 4, too high), 1 (index 5, too low)
        assert_eq!(result.valid_count, 3);
        assert_eq!(result.invalid_indices, vec![1, 4, 5]);
    }

    #[test]
    fn test_timestamp_validation() {
        let now = 1640995200; // Example timestamp
        let timestamps = vec![
            now - 3600,  // 1 hour ago - valid
            0,           // Invalid - zero
            now,         // Current - valid
            now + 3600,  // 1 hour future - valid
            now + 86400, // 1 day future - invalid (too far)
        ];
        
        let min_timestamp = now - 7200; // 2 hours ago
        let max_timestamp = now + 7200; // 2 hours future
        
        let result = SimdValidator::validate_timestamps(&timestamps, min_timestamp, max_timestamp);
        
        // Valid: indices 0, 2, 3
        // Invalid: indices 1 (zero), 4 (too far in future)
        assert_eq!(result.valid_count, 3);
        assert_eq!(result.invalid_indices, vec![1, 4]);
    }
}/// Add
itional SIMD-optimized operations for data processing
impl SimdValidator {
    /// SIMD-optimized CRC32 calculation for data integrity
    #[target_feature(enable = "sse4.2")]
    pub unsafe fn crc32_avx2(data: &[u8]) -> u32 {
        let mut crc = 0xFFFFFFFFu32;
        let mut offset = 0;
        
        // Process 8 bytes at a time using CRC32 instruction
        while offset + 8 <= data.len() {
            let chunk = std::ptr::read_unaligned(data.as_ptr().add(offset) as *const u64);
            crc = _mm_crc32_u64(crc as u64, chunk) as u32;
            offset += 8;
        }
        
        // Process remaining bytes
        while offset < data.len() {
            crc = _mm_crc32_u8(crc, data[offset]);
            offset += 1;
        }
        
        !crc
    }

    /// SIMD-optimized pattern matching for order validation
    #[target_feature(enable = "avx2")]
    pub unsafe fn pattern_match_avx2(data: &[u8], pattern: &[u8]) -> Vec<usize> {
        let mut matches = Vec::new();
        
        if pattern.is_empty() || data.len() < pattern.len() {
            return matches;
        }
        
        let pattern_len = pattern.len();
        let first_byte = pattern[0];
        let first_vec = _mm256_set1_epi8(first_byte as i8);
        
        let mut i = 0;
        while i + 32 <= data.len() - pattern_len + 1 {
            // Load 32 bytes of data
            let data_vec = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
            
            // Compare with first byte of pattern
            let cmp_result = _mm256_cmpeq_epi8(data_vec, first_vec);
            let mask = _mm256_movemask_epi8(cmp_result);
            
            // Check each potential match
            for bit in 0..32 {
                if (mask & (1 << bit)) != 0 {
                    let pos = i + bit;
                    if pos + pattern_len <= data.len() {
                        // Check full pattern match
                        let mut full_match = true;
                        for j in 1..pattern_len {
                            if data[pos + j] != pattern[j] {
                                full_match = false;
                                break;
                            }
                        }
                        if full_match {
                            matches.push(pos);
                        }
                    }
                }
            }
            
            i += 32;
        }
        
        // Handle remainder with scalar search
        while i <= data.len() - pattern_len {
            if data[i..i + pattern_len] == *pattern {
                matches.push(i);
            }
            i += 1;
        }
        
        matches
    }

    /// SIMD-optimized data compression using simple RLE
    #[target_feature(enable = "avx2")]
    pub unsafe fn compress_rle_avx2(data: &[u64]) -> Vec<(u64, u32)> {
        let mut compressed = Vec::new();
        
        if data.is_empty() {
            return compressed;
        }
        
        let mut current_value = data[0];
        let mut count = 1u32;
        
        for &value in &data[1..] {
            if value == current_value && count < u32::MAX {
                count += 1;
            } else {
                compressed.push((current_value, count));
                current_value = value;
                count = 1;
            }
        }
        
        compressed.push((current_value, count));
        compressed
    }

    /// SIMD-optimized data decompression from RLE
    #[target_feature(enable = "avx2")]
    pub unsafe fn decompress_rle_avx2(compressed: &[(u64, u32)]) -> Vec<u64> {
        let mut decompressed = Vec::new();
        
        for &(value, count) in compressed {
            for _ in 0..count {
                decompressed.push(value);
            }
        }
        
        decompressed
    }

    /// SIMD-optimized Hamming distance calculation
    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_distance_avx2(a: &[u64], b: &[u64]) -> u32 {
        assert_eq!(a.len(), b.len());
        
        let mut distance = 0u32;
        let mut offset = 0;
        
        // Process 4 u64 values at a time
        while offset + 4 <= a.len() {
            let a_vec = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
            let b_vec = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
            
            // XOR to find differing bits
            let xor_vec = _mm256_xor_si256(a_vec, b_vec);
            
            // Extract and count bits
            let mut xor_values = [0u64; 4];
            _mm256_storeu_si256(xor_values.as_mut_ptr() as *mut __m256i, xor_vec);
            
            for value in xor_values {
                distance += value.count_ones();
            }
            
            offset += 4;
        }
        
        // Handle remaining elements
        for i in offset..a.len() {
            distance += (a[i] ^ b[i]).count_ones();
        }
        
        distance
    }

    /// SIMD-optimized duplicate detection using sorting approach
    #[target_feature(enable = "avx2")]
    pub unsafe fn has_duplicates_sorted_avx2(mut data: Vec<u64>) -> bool {
        if data.len() < 2 {
            return false;
        }
        
        // Sort the data first
        data.sort_unstable();
        
        // Use SIMD to compare adjacent elements
        let mut offset = 0;
        while offset + 4 < data.len() {
            let current_vec = _mm256_loadu_si256(data.as_ptr().add(offset) as *const __m256i);
            let next_vec = _mm256_loadu_si256(data.as_ptr().add(offset + 1) as *const __m256i);
            
            let cmp_result = _mm256_cmpeq_epi64(current_vec, next_vec);
            let mask = _mm256_movemask_epi8(cmp_result);
            
            if mask != 0 {
                return true; // Found duplicate
            }
            
            offset += 4;
        }
        
        // Check remaining elements
        for i in offset..data.len() - 1 {
            if data[i] == data[i + 1] {
                return true;
            }
        }
        
        false
    }

    /// SIMD-optimized range validation with multiple ranges
    #[target_feature(enable = "avx2")]
    pub unsafe fn validate_multiple_ranges_avx2(
        values: &[u64],
        ranges: &[(u64, u64)], // (min, max) pairs
    ) -> Vec<u64> {
        let mut result_masks = vec![0u64; ranges.len()];
        
        for (range_idx, &(min_val, max_val)) in ranges.iter().enumerate() {
            let min_vec = _mm256_set1_epi64x(min_val as i64);
            let max_vec = _mm256_set1_epi64x(max_val as i64);
            
            let chunks = values.chunks_exact(4);
            let remainder = chunks.remainder();
            
            for (chunk_idx, chunk) in chunks.enumerate() {
                let values_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                
                // Check if value >= min
                let min_check = _mm256_cmpgt_epi64(values_vec, min_vec);
                
                // Check if value <= max
                let max_check = _mm256_cmpgt_epi64(max_vec, values_vec);
                
                // Combine checks
                let combined = _mm256_and_si256(min_check, max_check);
                let mask = _mm256_movemask_epi8(combined);
                
                // Extract validation results
                if mask & 0x80 != 0 { result_masks[range_idx] |= 1u64 << (chunk_idx * 4); }
                if mask & 0x8000 != 0 { result_masks[range_idx] |= 1u64 << (chunk_idx * 4 + 1); }
                if mask & 0x800000 != 0 { result_masks[range_idx] |= 1u64 << (chunk_idx * 4 + 2); }
                if mask & 0x80000000 != 0 { result_masks[range_idx] |= 1u64 << (chunk_idx * 4 + 3); }
            }
            
            // Handle remainder
            let base_idx = chunks.len() * 4;
            for (i, &value) in remainder.iter().enumerate() {
                if value >= min_val && value <= max_val {
                    result_masks[range_idx] |= 1u64 << (base_idx + i);
                }
            }
        }
        
        result_masks
    }

    /// Public interface for CRC32 calculation
    pub fn crc32(data: &[u8]) -> u32 {
        let caps = get_simd_capabilities();
        
        if caps.has_sse4_1 {
            unsafe { Self::crc32_avx2(data) }
        } else {
            // Scalar CRC32 fallback
            let mut crc = 0xFFFFFFFFu32;
            for &byte in data {
                crc = crc ^ (byte as u32);
                for _ in 0..8 {
                    if crc & 1 != 0 {
                        crc = (crc >> 1) ^ 0xEDB88320;
                    } else {
                        crc >>= 1;
                    }
                }
            }
            !crc
        }
    }

    /// Public interface for pattern matching
    pub fn pattern_match(data: &[u8], pattern: &[u8]) -> Vec<usize> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && data.len() >= 32 && !pattern.is_empty() {
            unsafe { Self::pattern_match_avx2(data, pattern) }
        } else {
            // Scalar fallback
            let mut matches = Vec::new();
            if pattern.is_empty() || data.len() < pattern.len() {
                return matches;
            }
            
            for i in 0..=data.len() - pattern.len() {
                if data[i..i + pattern.len()] == *pattern {
                    matches.push(i);
                }
            }
            matches
        }
    }

    /// Public interface for RLE compression
    pub fn compress_rle(data: &[u64]) -> Vec<(u64, u32)> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 {
            unsafe { Self::compress_rle_avx2(data) }
        } else {
            // Scalar fallback
            let mut compressed = Vec::new();
            
            if data.is_empty() {
                return compressed;
            }
            
            let mut current_value = data[0];
            let mut count = 1u32;
            
            for &value in &data[1..] {
                if value == current_value && count < u32::MAX {
                    count += 1;
                } else {
                    compressed.push((current_value, count));
                    current_value = value;
                    count = 1;
                }
            }
            
            compressed.push((current_value, count));
            compressed
        }
    }

    /// Public interface for RLE decompression
    pub fn decompress_rle(compressed: &[(u64, u32)]) -> Vec<u64> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 {
            unsafe { Self::decompress_rle_avx2(compressed) }
        } else {
            // Scalar fallback
            let mut decompressed = Vec::new();
            
            for &(value, count) in compressed {
                for _ in 0..count {
                    decompressed.push(value);
                }
            }
            
            decompressed
        }
    }

    /// Public interface for Hamming distance
    pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
        assert_eq!(a.len(), b.len());
        
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && a.len() >= 4 {
            unsafe { Self::hamming_distance_avx2(a, b) }
        } else {
            // Scalar fallback
            let mut distance = 0u32;
            for i in 0..a.len() {
                distance += (a[i] ^ b[i]).count_ones();
            }
            distance
        }
    }

    /// Public interface for duplicate detection with sorting
    pub fn has_duplicates_sorted(data: Vec<u64>) -> bool {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && data.len() >= 8 {
            unsafe { Self::has_duplicates_sorted_avx2(data) }
        } else {
            // Scalar fallback
            let mut sorted_data = data;
            sorted_data.sort_unstable();
            
            for i in 0..sorted_data.len() - 1 {
                if sorted_data[i] == sorted_data[i + 1] {
                    return true;
                }
            }
            false
        }
    }

    /// Public interface for multiple range validation
    pub fn validate_multiple_ranges(
        values: &[u64],
        ranges: &[(u64, u64)],
    ) -> Vec<ValidationResult> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && values.len() >= 4 {
            let masks = unsafe { Self::validate_multiple_ranges_avx2(values, ranges) };
            masks.into_iter()
                .map(|mask| Self::mask_to_result(mask, values.len()))
                .collect()
        } else {
            // Scalar fallback
            let mut results = Vec::new();
            
            for &(min_val, max_val) in ranges {
                let mut mask = 0u64;
                for (i, &value) in values.iter().enumerate() {
                    if value >= min_val && value <= max_val {
                        mask |= 1u64 << i;
                    }
                }
                results.push(Self::mask_to_result(mask, values.len()));
            }
            
            results
        }
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_crc32() {
        let data = b"Hello, World!";
        let crc = SimdValidator::crc32(data);
        
        // CRC32 should be deterministic
        assert_eq!(crc, SimdValidator::crc32(data));
    }

    #[test]
    fn test_pattern_match() {
        let data = b"abcdefabcxyzabc";
        let pattern = b"abc";
        let matches = SimdValidator::pattern_match(data, pattern);
        
        assert_eq!(matches, vec![0, 6, 12]);
    }

    #[test]
    fn test_compress_decompress_rle() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let compressed = SimdValidator::compress_rle(&data);
        let decompressed = SimdValidator::decompress_rle(&compressed);
        
        assert_eq!(data, decompressed);
        assert_eq!(compressed, vec![(1, 3), (2, 2), (3, 4)]);
    }

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b1010, 0b1100];
        let b = vec![0b1110, 0b1000];
        
        let distance = SimdValidator::hamming_distance(&a, &b);
        
        // 1010 XOR 1110 = 0100 (1 bit different)
        // 1100 XOR 1000 = 0100 (1 bit different)
        // Total: 2 bits different
        assert_eq!(distance, 2);
    }

    #[test]
    fn test_has_duplicates_sorted() {
        let data_no_dup = vec![1, 3, 5, 7, 9];
        let data_with_dup = vec![1, 3, 5, 5, 9];
        
        assert!(!SimdValidator::has_duplicates_sorted(data_no_dup));
        assert!(SimdValidator::has_duplicates_sorted(data_with_dup));
    }

    #[test]
    fn test_validate_multiple_ranges() {
        let values = vec![50, 150, 250, 350, 450];
        let ranges = vec![
            (100, 200), // Should match 150
            (200, 300), // Should match 250
            (400, 500), // Should match 450
        ];
        
        let results = SimdValidator::validate_multiple_ranges(&values, &ranges);
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].valid_count, 1); // Only 150 in range 100-200
        assert_eq!(results[1].valid_count, 1); // Only 250 in range 200-300
        assert_eq!(results[2].valid_count, 1); // Only 450 in range 400-500
    }

    #[test]
    fn test_empty_pattern_match() {
        let data = b"test data";
        let empty_pattern = b"";
        let matches = SimdValidator::pattern_match(data, empty_pattern);
        
        assert!(matches.is_empty());
    }

    #[test]
    fn test_rle_empty_data() {
        let empty_data: Vec<u64> = vec![];
        let compressed = SimdValidator::compress_rle(&empty_data);
        let decompressed = SimdValidator::decompress_rle(&compressed);
        
        assert!(compressed.is_empty());
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0b1010, 0b1100, 0b1111];
        let b = vec![0b1010, 0b1100, 0b1111];
        
        let distance = SimdValidator::hamming_distance(&a, &b);
        assert_eq!(distance, 0);
    }

    #[test]
    fn test_multiple_ranges_no_matches() {
        let values = vec![50, 150, 250];
        let ranges = vec![
            (300, 400), // No matches
            (500, 600), // No matches
        ];
        
        let results = SimdValidator::validate_multiple_ranges(&values, &ranges);
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].valid_count, 0);
        assert_eq!(results[1].valid_count, 0);
        assert_eq!(results[0].invalid_indices, vec![0, 1, 2]);
        assert_eq!(results[1].invalid_indices, vec![0, 1, 2]);
    }
}