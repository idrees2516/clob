use trading_system::performance::simd::{
    SimdValidator, PriceConstraints, SimdPriceComparator, 
    SimdArithmetic, SimdMemoryOps, get_simd_capabilities
};

#[test]
fn test_simd_capabilities_detection() {
    let caps = get_simd_capabilities();
    println!("SIMD Capabilities:");
    println!("  SSE2: {}", caps.has_sse2);
    println!("  SSE4.1: {}", caps.has_sse4_1);
    println!("  AVX: {}", caps.has_avx);
    println!("  AVX2: {}", caps.has_avx2);
    println!("  AVX512F: {}", caps.has_avx512f);
    println!("  Best vector width: {} bytes", caps.best_vector_width());
    
    // Should at least have SSE2 on modern x86_64
    assert!(caps.has_sse2);
}

#[test]
fn test_comprehensive_price_validation() {
    let constraints = PriceConstraints {
        min_price: 100,
        max_price: 1000,
        tick_size: 5,
    };
    
    // Test with various batch sizes to ensure both SIMD and scalar paths work
    for batch_size in [1, 3, 4, 8, 16, 32] {
        let mut prices = Vec::new();
        let mut expected_valid = 0;
        
        for i in 0..batch_size {
            let price = 100 + (i * 10) as u64;
            if price <= 1000 && price % 5 == 0 {
                expected_valid += 1;
            }
            prices.push(price);
        }
        
        let result = SimdValidator::validate_prices(&prices, &constraints);
        assert_eq!(result.valid_count, expected_valid, 
                  "Failed for batch size {}", batch_size);
    }
}

#[test]
fn test_comprehensive_arithmetic_operations() {
    let prices = vec![100, 110, 120, 130, 140, 150, 160, 170];
    let volumes = vec![10, 20, 15, 25, 30, 20, 15, 10];
    
    // Test VWAP calculation
    let vwap = SimdArithmetic::vwap(&prices, &volumes);
    let expected_vwap = {
        let total_value: u64 = prices.iter().zip(volumes.iter())
            .map(|(&p, &v)| p * v)
            .sum();
        let total_volume: u64 = volumes.iter().sum();
        total_value / total_volume
    };
    assert_eq!(vwap, expected_vwap);
    
    // Test moving average
    let mut ma_result = vec![0u64; prices.len() - 2];
    SimdArithmetic::moving_average(&prices, 3, &mut ma_result);
    
    // First moving average should be (100+110+120)/3 = 110
    assert_eq!(ma_result[0], 110);
    
    // Test exponential moving average
    let mut ema_result = vec![0u64; prices.len()];
    SimdArithmetic::exponential_moving_average(&prices, 0.2, &mut ema_result);
    assert_eq!(ema_result[0], prices[0]); // First value should be unchanged
    
    // Test price differences
    let mut diff_result = vec![0i64; prices.len() - 1];
    SimdArithmetic::price_differences(&prices, &mut diff_result);
    assert_eq!(diff_result[0], 10); // 110 - 100 = 10
}

#[test]
fn test_comprehensive_memory_operations() {
    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Test array sum
    let sum = SimdMemoryOps::sum_u64_array(&data);
    assert_eq!(sum, 55); // 1+2+...+10 = 55
    
    // Test array search
    assert_eq!(SimdMemoryOps::find_u64(&data, 5), Some(4));
    assert_eq!(SimdMemoryOps::find_u64(&data, 11), None);
    
    // Test transform copy
    let mut transformed = vec![0u64; data.len()];
    SimdMemoryOps::transform_copy_u64(&data, &mut transformed, 2);
    let expected: Vec<u64> = data.iter().map(|&x| x * 2).collect();
    assert_eq!(transformed, expected);
    
    // Test memory operations
    let src = vec![1u8, 2, 3, 4, 5];
    let mut dst = vec![0u8; 5];
    SimdMemoryOps::fast_memcpy(&mut dst, &src);
    assert_eq!(dst, src);
    
    let mut buffer = vec![0u8; 10];
    SimdMemoryOps::fast_memset(&mut buffer, 0xFF);
    assert!(buffer.iter().all(|&x| x == 0xFF));
}

#[test]
fn test_comprehensive_price_comparison() {
    let prices = vec![50, 100, 150, 200, 250, 300, 350, 400];
    
    // Test price comparison
    let mask = SimdPriceComparator::compare_prices(&prices, 200);
    let above_200_count = mask.count_ones() as usize;
    let expected_count = prices.iter().filter(|&&p| p > 200).count();
    assert_eq!(above_200_count, expected_count);
    
    // Test min/max finding
    assert_eq!(SimdPriceComparator::find_min_price(&prices), Some(50));
    assert_eq!(SimdPriceComparator::find_max_price(&prices), Some(400));
    
    // Test range finding
    let in_range = SimdPriceComparator::find_prices_in_range(&prices, 150, 300);
    let expected_indices: Vec<usize> = prices.iter()
        .enumerate()
        .filter_map(|(i, &p)| if p >= 150 && p <= 300 { Some(i) } else { None })
        .collect();
    assert_eq!(in_range, expected_indices);
    
    // Test counting
    let count = SimdPriceComparator::count_prices_above(&prices, 250);
    let expected = prices.iter().filter(|&&p| p > 250).count();
    assert_eq!(count, expected);
}

#[test]
fn test_order_validation_integration() {
    let prices = vec![105, 110, 115, 120, 125];
    let quantities = vec![100, 200, 0, 150, 300]; // Third quantity is invalid
    let order_ids = vec![1001, 1002, 1003, 1004, 1005];
    let timestamps = vec![1000, 1001, 1002, 1003, 1004];
    
    let constraints = PriceConstraints {
        min_price: 100,
        max_price: 200,
        tick_size: 5,
    };
    
    // Test combined order validation
    let order_result = SimdValidator::validate_orders(
        &prices, &quantities, &constraints, 50, 250
    );
    
    // All prices are valid, but quantity[2] = 0 is invalid
    assert_eq!(order_result.valid_count, 4);
    assert_eq!(order_result.invalid_indices, vec![2]);
    
    // Test order ID validation
    let id_result = SimdValidator::validate_order_ids(&order_ids, 1000, 2000);
    assert_eq!(id_result.valid_count, 5); // All IDs should be valid
    
    // Test timestamp validation
    let ts_result = SimdValidator::validate_timestamps(&timestamps, 999, 1005);
    assert_eq!(ts_result.valid_count, 5); // All timestamps should be valid
}

#[test]
fn test_large_batch_performance() {
    // Test with larger batches to ensure SIMD optimizations are used
    let size = 1000;
    let mut prices = Vec::with_capacity(size);
    let mut quantities = Vec::with_capacity(size);
    
    for i in 0..size {
        prices.push((100 + i * 5) as u64);
        quantities.push((10 + i) as u64);
    }
    
    let constraints = PriceConstraints {
        min_price: 50,
        max_price: 10000,
        tick_size: 5,
    };
    
    // This should use SIMD optimizations for large batches
    let result = SimdValidator::validate_orders(
        &prices, &quantities, &constraints, 5, 2000
    );
    
    // Most orders should be valid (some quantities might exceed max)
    assert!(result.valid_count > size / 2);
    
    // Test arithmetic operations on large arrays
    let sum = SimdMemoryOps::sum_u64_array(&prices);
    let expected_sum: u64 = prices.iter().sum();
    assert_eq!(sum, expected_sum);
    
    // Test VWAP on large dataset
    let vwap = SimdArithmetic::vwap(&prices, &quantities);
    assert!(vwap > 0); // Should produce a reasonable result
}

#[test]
fn test_edge_cases() {
    // Test empty arrays
    let empty: Vec<u64> = vec![];
    let empty_result = SimdValidator::validate_prices(&empty, &PriceConstraints::default());
    assert_eq!(empty_result.valid_count, 0);
    
    // Test single element
    let single = vec![100];
    let single_result = SimdValidator::validate_prices(&single, &PriceConstraints::default());
    assert_eq!(single_result.valid_count, 1);
    
    // Test boundary conditions
    let constraints = PriceConstraints {
        min_price: 100,
        max_price: 200,
        tick_size: 1,
    };
    
    let boundary_prices = vec![99, 100, 200, 201];
    let boundary_result = SimdValidator::validate_prices(&boundary_prices, &constraints);
    assert_eq!(boundary_result.valid_count, 2); // Only 100 and 200 are valid
    assert_eq!(boundary_result.invalid_indices, vec![0, 3]);
}

#[test]
fn test_data_integrity() {
    // Test checksum validation
    let data = vec![100, 200, 300, 400, 500];
    let mut checksums = Vec::new();
    
    // Generate checksums using the same algorithm as the validator
    for &val in &data {
        checksums.push(val ^ 0x5555555555555555u64);
    }
    
    let result = SimdValidator::validate_checksums(&data, &checksums);
    assert_eq!(result.valid_count, data.len());
    
    // Corrupt one checksum
    checksums[2] = 0;
    let corrupted_result = SimdValidator::validate_checksums(&data, &checksums);
    assert_eq!(corrupted_result.valid_count, data.len() - 1);
    assert_eq!(corrupted_result.invalid_indices, vec![2]);
}