use std::arch::x86_64::*;
use crate::performance::simd::get_simd_capabilities;

/// SIMD-optimized price comparison operations for order book
pub struct SimdPriceComparator;

impl SimdPriceComparator {
    /// Compare multiple prices using SIMD instructions
    /// Returns a bitmask indicating which prices in `prices` are greater than `threshold`
    #[target_feature(enable = "avx2")]
    pub unsafe fn compare_prices_avx2(prices: &[u64], threshold: u64) -> u32 {
        let threshold_vec = _mm256_set1_epi64x(threshold as i64);
        let mut result_mask = 0u32;
        
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (i, chunk) in chunks.enumerate() {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let cmp_result = _mm256_cmpgt_epi64(prices_vec, threshold_vec);
            let mask = _mm256_movemask_epi8(cmp_result);
            
            // Extract comparison results for each 64-bit element
            result_mask |= ((mask & 0x80) >> 7) << (i * 4);
            result_mask |= ((mask & 0x8000) >> 15) << (i * 4 + 1);
            result_mask |= ((mask & 0x800000) >> 23) << (i * 4 + 2);
            result_mask |= ((mask & 0x80000000) >> 31) << (i * 4 + 3);
        }
        
        // Handle remainder with scalar operations
        for (i, &price) in remainder.iter().enumerate() {
            if price > threshold {
                result_mask |= 1 << (chunks.len() * 4 + i);
            }
        }
        
        result_mask
    }

    /// Find the best bid price from an array of prices using SIMD
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_max_price_avx2(prices: &[u64]) -> Option<u64> {
        if prices.is_empty() {
            return None;
        }

        let mut max_vec = _mm256_setzero_si256();
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            max_vec = _mm256_max_epu64(max_vec, prices_vec);
        }

        // Extract maximum from vector
        let mut max_values = [0u64; 4];
        _mm256_storeu_si256(max_values.as_mut_ptr() as *mut __m256i, max_vec);
        let mut max_price = max_values.iter().copied().max().unwrap_or(0);

        // Handle remainder
        for &price in remainder {
            max_price = max_price.max(price);
        }

        if max_price == 0 { None } else { Some(max_price) }
    }

    /// Find prices within a range using SIMD
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_prices_in_range_avx2(
        prices: &[u64], 
        min_price: u64, 
        max_price: u64
    ) -> Vec<usize> {
        let mut indices = Vec::new();
        
        let min_vec = _mm256_set1_epi64x(min_price as i64);
        let max_vec = _mm256_set1_epi64x(max_price as i64);
        
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for (chunk_idx, chunk) in chunks.enumerate() {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            
            // Check if price >= min_price
            let min_check = _mm256_cmpgt_epi64(prices_vec, min_vec);
            
            // Check if price <= max_price
            let max_check = _mm256_cmpgt_epi64(max_vec, prices_vec);
            
            // Combine checks (price >= min AND price <= max)
            let combined = _mm256_and_si256(min_check, max_check);
            let mask = _mm256_movemask_epi8(combined);
            
            // Extract matching indices
            if mask & 0x80 != 0 { indices.push(chunk_idx * 4); }
            if mask & 0x8000 != 0 { indices.push(chunk_idx * 4 + 1); }
            if mask & 0x800000 != 0 { indices.push(chunk_idx * 4 + 2); }
            if mask & 0x80000000 != 0 { indices.push(chunk_idx * 4 + 3); }
        }
        
        // Handle remainder
        let base_idx = chunks.len() * 4;
        for (i, &price) in remainder.iter().enumerate() {
            if price >= min_price && price <= max_price {
                indices.push(base_idx + i);
            }
        }
        
        indices
    }

    /// Count prices above threshold using SIMD
    #[target_feature(enable = "avx2")]
    pub unsafe fn count_prices_above_avx2(prices: &[u64], threshold: u64) -> usize {
        let threshold_vec = _mm256_set1_epi64x(threshold as i64);
        let mut count = 0;
        
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let cmp_result = _mm256_cmpgt_epi64(prices_vec, threshold_vec);
            let mask = _mm256_movemask_epi8(cmp_result);
            
            // Count set bits (each 64-bit comparison sets 8 bits)
            count += ((mask & 0x80) >> 7) as usize;
            count += ((mask & 0x8000) >> 15) as usize;
            count += ((mask & 0x800000) >> 23) as usize;
            count += ((mask & 0x80000000) >> 31) as usize;
        }
        
        // Handle remainder
        for &price in remainder {
            if price > threshold {
                count += 1;
            }
        }
        
        count
    }

    /// Find the best ask price from an array of prices using SIMD
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_min_price_avx2(prices: &[u64]) -> Option<u64> {
        if prices.is_empty() {
            return None;
        }

        let mut min_vec = _mm256_set1_epi64x(u64::MAX as i64);
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let prices_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            min_vec = _mm256_min_epu64(min_vec, prices_vec);
        }

        // Extract minimum from vector
        let mut min_values = [0u64; 4];
        _mm256_storeu_si256(min_values.as_mut_ptr() as *mut __m256i, min_vec);
        let mut min_price = min_values.iter().copied().min().unwrap_or(u64::MAX);

        // Handle remainder
        for &price in remainder {
            min_price = min_price.min(price);
        }

        if min_price == u64::MAX { None } else { Some(min_price) }
    }

    /// Public interface that automatically selects the best SIMD implementation
    pub fn compare_prices(prices: &[u64], threshold: u64) -> u32 {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::compare_prices_avx2(prices, threshold) }
        } else {
            // Fallback to scalar implementation
            let mut result_mask = 0u32;
            for (i, &price) in prices.iter().enumerate() {
                if price > threshold {
                    result_mask |= 1 << i;
                }
            }
            result_mask
        }
    }

    pub fn find_max_price(prices: &[u64]) -> Option<u64> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::find_max_price_avx2(prices) }
        } else {
            prices.iter().copied().max()
        }
    }

    pub fn find_min_price(prices: &[u64]) -> Option<u64> {
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::find_min_price_avx2(prices) }
        } else {
            prices.iter().copied().min()
        }
    }

    /// Public interface for finding prices in range
    pub fn find_prices_in_range(prices: &[u64], min_price: u64, max_price: u64) -> Vec<usize> {
        if prices.is_empty() {
            return Vec::new();
        }
        
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::find_prices_in_range_avx2(prices, min_price, max_price) }
        } else {
            // Scalar fallback
            prices.iter()
                .enumerate()
                .filter_map(|(i, &price)| {
                    if price >= min_price && price <= max_price {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    /// Public interface for counting prices above threshold
    pub fn count_prices_above(prices: &[u64], threshold: u64) -> usize {
        if prices.is_empty() {
            return 0;
        }
        
        let caps = get_simd_capabilities();
        
        if caps.has_avx2 && prices.len() >= 4 {
            unsafe { Self::count_prices_above_avx2(prices, threshold) }
        } else {
            // Scalar fallback
            prices.iter().filter(|&&price| price > threshold).count()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_comparison() {
        let prices = vec![100, 200, 150, 300, 50];
        let threshold = 175;
        
        let result = SimdPriceComparator::compare_prices(&prices, threshold);
        
        // Prices 200 (index 1) and 300 (index 3) are > 175
        assert_eq!(result & (1 << 1), 1 << 1); // 200 > 175
        assert_eq!(result & (1 << 3), 1 << 3); // 300 > 175
        assert_eq!(result & (1 << 0), 0);      // 100 <= 175
        assert_eq!(result & (1 << 2), 0);      // 150 <= 175
        assert_eq!(result & (1 << 4), 0);      // 50 <= 175
    }

    #[test]
    fn test_find_max_price() {
        let prices = vec![100, 200, 150, 300, 50];
        assert_eq!(SimdPriceComparator::find_max_price(&prices), Some(300));
        
        let empty_prices: Vec<u64> = vec![];
        assert_eq!(SimdPriceComparator::find_max_price(&empty_prices), None);
    }

    #[test]
    fn test_find_min_price() {
        let prices = vec![100, 200, 150, 300, 50];
        assert_eq!(SimdPriceComparator::find_min_price(&prices), Some(50));
        
        let empty_prices: Vec<u64> = vec![];
        assert_eq!(SimdPriceComparator::find_min_price(&empty_prices), None);
    }

    #[test]
    fn test_find_prices_in_range() {
        let prices = vec![50, 100, 150, 200, 250, 300];
        let indices = SimdPriceComparator::find_prices_in_range(&prices, 100, 250);
        
        // Prices 100, 150, 200, 250 are in range (indices 1, 2, 3, 4)
        assert_eq!(indices, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_count_prices_above() {
        let prices = vec![100, 200, 150, 300, 50];
        let count = SimdPriceComparator::count_prices_above(&prices, 175);
        
        // Prices 200 and 300 are above 175
        assert_eq!(count, 2);
    }
}