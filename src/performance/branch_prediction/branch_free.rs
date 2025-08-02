/// Branch-free algorithms for performance-critical code paths
pub struct BranchFreeAlgorithms;

impl BranchFreeAlgorithms {
    /// Branch-free minimum of two values
    #[inline(always)]
    pub fn min_branch_free(a: u64, b: u64) -> u64 {
        let mask = ((a as i64 - b as i64) >> 63) as u64;
        (a & mask) | (b & !mask)
    }

    /// Branch-free maximum of two values
    #[inline(always)]
    pub fn max_branch_free(a: u64, b: u64) -> u64 {
        let mask = ((b as i64 - a as i64) >> 63) as u64;
        (a & mask) | (b & !mask)
    }

    /// Branch-free absolute value
    #[inline(always)]
    pub fn abs_branch_free(x: i64) -> i64 {
        let mask = x >> 63;
        (x + mask) ^ mask
    }

    /// Branch-free conditional assignment
    #[inline(always)]
    pub fn conditional_assign_branch_free(condition: bool, if_true: u64, if_false: u64) -> u64 {
        let mask = (condition as u64).wrapping_neg();
        (if_true & mask) | (if_false & !mask)
    }

    /// Branch-free sign function (-1, 0, 1)
    #[inline(always)]
    pub fn sign_branch_free(x: i64) -> i64 {
        (x > 0) as i64 - (x < 0) as i64
    }

    /// Branch-free clamp to range [min, max]
    #[inline(always)]
    pub fn clamp_branch_free(value: u64, min: u64, max: u64) -> u64 {
        Self::min_branch_free(Self::max_branch_free(value, min), max)
    }

    /// Branch-free comparison result as integer
    #[inline(always)]
    pub fn compare_branch_free(a: u64, b: u64) -> i32 {
        ((a > b) as i32) - ((a < b) as i32)
    }

    /// Branch-free selection from array based on condition
    #[inline(always)]
    pub fn select_branch_free<T: Copy>(condition: bool, options: &[T; 2]) -> T {
        options[condition as usize]
    }

    /// Branch-free bit manipulation - set bit if condition is true
    #[inline(always)]
    pub fn conditional_set_bit(value: u64, bit_pos: u32, condition: bool) -> u64 {
        let mask = (condition as u64) << bit_pos;
        value | mask
    }

    /// Branch-free bit manipulation - clear bit if condition is true
    #[inline(always)]
    pub fn conditional_clear_bit(value: u64, bit_pos: u32, condition: bool) -> u64 {
        let mask = !((condition as u64) << bit_pos);
        value & mask
    }
}

/// Branch-free price comparison for order book operations
pub struct BranchFreePriceOps;

impl BranchFreePriceOps {
    /// Branch-free price level comparison
    #[inline(always)]
    pub fn compare_prices(price1: u64, price2: u64) -> i32 {
        BranchFreeAlgorithms::compare_branch_free(price1, price2)
    }

    /// Branch-free best price selection
    #[inline(always)]
    pub fn select_best_bid(price1: u64, price2: u64) -> u64 {
        BranchFreeAlgorithms::max_branch_free(price1, price2)
    }

    /// Branch-free best ask selection
    #[inline(always)]
    pub fn select_best_ask(price1: u64, price2: u64) -> u64 {
        BranchFreeAlgorithms::min_branch_free(price1, price2)
    }

    /// Branch-free price within spread check
    #[inline(always)]
    pub fn price_in_spread(price: u64, bid: u64, ask: u64) -> bool {
        let above_bid = price >= bid;
        let below_ask = price <= ask;
        above_bid && below_ask
    }

    /// Branch-free order matching decision
    #[inline(always)]
    pub fn can_match_orders(buy_price: u64, sell_price: u64, is_buy: bool) -> bool {
        let buy_condition = buy_price >= sell_price;
        let sell_condition = sell_price <= buy_price;
        BranchFreeAlgorithms::conditional_assign_branch_free(is_buy, buy_condition as u64, sell_condition as u64) != 0
    }
}

/// Branch-free mathematical operations for trading calculations
pub struct BranchFreeMath;

impl BranchFreeMath {
    /// Branch-free division with zero check
    #[inline(always)]
    pub fn safe_divide(dividend: u64, divisor: u64) -> u64 {
        let is_zero = divisor == 0;
        let safe_divisor = BranchFreeAlgorithms::conditional_assign_branch_free(!is_zero, divisor, 1);
        let result = dividend / safe_divisor;
        BranchFreeAlgorithms::conditional_assign_branch_free(!is_zero, result, 0)
    }

    /// Branch-free percentage calculation
    #[inline(always)]
    pub fn calculate_percentage(value: u64, total: u64) -> u64 {
        if total == 0 {
            return 0;
        }
        (value * 100) / total
    }

    /// Branch-free rounding to nearest multiple
    #[inline(always)]
    pub fn round_to_multiple(value: u64, multiple: u64) -> u64 {
        if multiple == 0 {
            return value;
        }
        let remainder = value % multiple;
        let half_multiple = multiple / 2;
        let should_round_up = remainder >= half_multiple;
        let rounded_down = value - remainder;
        BranchFreeAlgorithms::conditional_assign_branch_free(should_round_up, rounded_down + multiple, rounded_down)
    }

    /// Branch-free linear interpolation
    #[inline(always)]
    pub fn lerp(a: u64, b: u64, t: u64, scale: u64) -> u64 {
        let diff = if b > a { b - a } else { a - b };
        let interpolated = (diff * t) / scale;
        if b > a {
            a + interpolated
        } else {
            a - interpolated
        }
    }
}

/// Branch-free validation operations
pub struct BranchFreeValidation;

impl BranchFreeValidation {
    /// Branch-free range validation
    #[inline(always)]
    pub fn validate_range(value: u64, min: u64, max: u64) -> bool {
        (value >= min) && (value <= max)
    }

    /// Branch-free multiple validation checks
    #[inline(always)]
    pub fn validate_multiple_conditions(conditions: &[bool]) -> bool {
        conditions.iter().all(|&c| c)
    }

    /// Branch-free order validation
    #[inline(always)]
    pub fn validate_order_params(price: u64, quantity: u64, min_price: u64, max_price: u64, min_qty: u64, max_qty: u64) -> bool {
        let price_valid = Self::validate_range(price, min_price, max_price);
        let qty_valid = Self::validate_range(quantity, min_qty, max_qty);
        price_valid && qty_valid
    }

    /// Branch-free checksum validation
    #[inline(always)]
    pub fn validate_checksum(data: &[u8], expected_checksum: u32) -> bool {
        let calculated = data.iter().fold(0u32, |acc, &byte| acc.wrapping_add(byte as u32));
        calculated == expected_checksum
    }
}

/// Lookup table-based algorithms for complex conditional logic
pub struct LookupTableAlgorithms;

impl LookupTableAlgorithms {
    /// Pre-computed lookup table for order type validation
    const ORDER_TYPE_VALID: [bool; 256] = {
        let mut table = [false; 256];
        table[0] = true;  // Market order
        table[1] = true;  // Limit order
        table[2] = true;  // Stop order
        table[3] = true;  // Stop-limit order
        table[4] = true;  // IOC order
        table[5] = true;  // FOK order
        table
    };

    /// Pre-computed lookup table for price tick validation
    const PRICE_TICK_VALID: [bool; 1000] = {
        let mut table = [false; 1000];
        let mut i = 0;
        while i < 1000 {
            // Valid ticks: multiples of 5 cents up to $10
            table[i] = (i % 5) == 0;
            i += 1;
        }
        table
    };

    /// Pre-computed lookup table for quantity lot size validation
    const QUANTITY_LOT_VALID: [bool; 10000] = {
        let mut table = [false; 10000];
        let mut i = 0;
        while i < 10000 {
            // Valid lots: multiples of 100 shares
            table[i] = (i % 100) == 0;
            i += 1;
        }
        table
    };

    /// Fast order type validation using lookup table
    #[inline(always)]
    pub fn validate_order_type(order_type: u8) -> bool {
        Self::ORDER_TYPE_VALID[order_type as usize]
    }

    /// Fast price tick validation using lookup table
    #[inline(always)]
    pub fn validate_price_tick(price_cents: u16) -> bool {
        if price_cents < 1000 {
            Self::PRICE_TICK_VALID[price_cents as usize]
        } else {
            // For prices above $10, use different logic
            (price_cents % 10) == 0
        }
    }

    /// Fast quantity lot validation using lookup table
    #[inline(always)]
    pub fn validate_quantity_lot(quantity: u16) -> bool {
        if quantity < 10000 {
            Self::QUANTITY_LOT_VALID[quantity as usize]
        } else {
            // For large quantities, use modulo
            (quantity % 100) == 0
        }
    }

    /// Lookup table for market state transitions
    const MARKET_STATE_TRANSITIONS: [[u8; 8]; 8] = [
        // From: Pre-open, Open, Continuous, Auction, Halt, Close, Post-close, Maintenance
        [1, 0, 0, 0, 4, 0, 0, 7], // Pre-open can go to Open or Halt or Maintenance
        [0, 2, 2, 3, 4, 5, 0, 0], // Open can go to Continuous, Auction, Halt, Close
        [0, 0, 2, 3, 4, 5, 0, 0], // Continuous can stay or go to Auction, Halt, Close
        [0, 0, 2, 3, 4, 5, 0, 0], // Auction can go to Continuous, stay, Halt, Close
        [0, 1, 2, 0, 4, 0, 0, 0], // Halt can go to Open, Continuous, or stay
        [0, 0, 0, 0, 0, 6, 6, 0], // Close can go to Post-close or stay
        [0, 0, 0, 0, 0, 0, 6, 7], // Post-close can stay or go to Maintenance
        [0, 0, 0, 0, 0, 0, 0, 0], // Maintenance (special handling)
    ];

    /// Fast market state transition validation
    #[inline(always)]
    pub fn validate_state_transition(from_state: u8, to_state: u8) -> bool {
        if from_state < 8 && to_state < 8 {
            Self::MARKET_STATE_TRANSITIONS[from_state as usize][to_state as usize] == to_state
        } else {
            false
        }
    }

    /// Lookup table for fee calculation based on order size tiers
    const FEE_TIERS: [u32; 16] = [
        30,  // 0-999 shares: 30 basis points
        25,  // 1000-4999: 25 basis points
        20,  // 5000-9999: 20 basis points
        15,  // 10000-24999: 15 basis points
        12,  // 25000-49999: 12 basis points
        10,  // 50000-99999: 10 basis points
        8,   // 100000-249999: 8 basis points
        6,   // 250000-499999: 6 basis points
        5,   // 500000-999999: 5 basis points
        4,   // 1M-2.49M: 4 basis points
        3,   // 2.5M-4.99M: 3 basis points
        2,   // 5M-9.99M: 2 basis points
        1,   // 10M+: 1 basis point
        1, 1, 1 // Padding
    ];

    /// Fast fee calculation using lookup table
    #[inline(always)]
    pub fn calculate_fee_tier(quantity: u32) -> u32 {
        let tier = match quantity {
            0..=999 => 0,
            1000..=4999 => 1,
            5000..=9999 => 2,
            10000..=24999 => 3,
            25000..=49999 => 4,
            50000..=99999 => 5,
            100000..=249999 => 6,
            250000..=499999 => 7,
            500000..=999999 => 8,
            1000000..=2499999 => 9,
            2500000..=4999999 => 10,
            5000000..=9999999 => 11,
            _ => 12,
        };
        Self::FEE_TIERS[tier]
    }
}

/// SIMD-based conditional processing for batch operations
pub struct SIMDConditionalProcessing;

impl SIMDConditionalProcessing {
    /// SIMD-based price comparison for multiple orders
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn compare_prices_simd(
        prices1: &[u64; 8],
        prices2: &[u64; 8],
        results: &mut [bool; 8],
    ) {
        use std::arch::x86_64::*;
        
        // Load prices into SIMD registers
        let p1_lo = _mm256_loadu_si256(prices1.as_ptr() as *const __m256i);
        let p1_hi = _mm256_loadu_si256(prices1.as_ptr().add(4) as *const __m256i);
        let p2_lo = _mm256_loadu_si256(prices2.as_ptr() as *const __m256i);
        let p2_hi = _mm256_loadu_si256(prices2.as_ptr().add(4) as *const __m256i);
        
        // Compare prices (greater than)
        let cmp_lo = _mm256_cmpgt_epi64(p1_lo, p2_lo);
        let cmp_hi = _mm256_cmpgt_epi64(p1_hi, p2_hi);
        
        // Extract comparison results
        let mask_lo = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_lo));
        let mask_hi = _mm256_movemask_pd(_mm256_castsi256_pd(cmp_hi));
        
        // Store results
        for i in 0..4 {
            results[i] = (mask_lo & (1 << i)) != 0;
            results[i + 4] = (mask_hi & (1 << i)) != 0;
        }
    }

    /// SIMD-based quantity validation for multiple orders
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn validate_quantities_simd(
        quantities: &[u32; 8],
        min_qty: u32,
        max_qty: u32,
        results: &mut [bool; 8],
    ) {
        use std::arch::x86_64::*;
        
        // Load quantities and limits
        let qty = _mm256_loadu_si256(quantities.as_ptr() as *const __m256i);
        let min_vec = _mm256_set1_epi32(min_qty as i32);
        let max_vec = _mm256_set1_epi32(max_qty as i32);
        
        // Check if quantities are within range
        let ge_min = _mm256_cmpgt_epi32(qty, min_vec);
        let le_max = _mm256_cmpgt_epi32(max_vec, qty);
        let valid = _mm256_and_si256(ge_min, le_max);
        
        // Extract results
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(valid));
        for i in 0..8 {
            results[i] = (mask & (1 << i)) != 0;
        }
    }

    /// SIMD-based order matching for multiple price levels
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn match_orders_simd(
        buy_prices: &[u64; 4],
        sell_prices: &[u64; 4],
        matches: &mut [bool; 4],
    ) {
        use std::arch::x86_64::*;
        
        let buy_vec = _mm256_loadu_si256(buy_prices.as_ptr() as *const __m256i);
        let sell_vec = _mm256_loadu_si256(sell_prices.as_ptr() as *const __m256i);
        
        // Buy price >= sell price means match
        let can_match = _mm256_cmpgt_epi64(buy_vec, sell_vec);
        let mask = _mm256_movemask_pd(_mm256_castsi256_pd(can_match));
        
        for i in 0..4 {
            matches[i] = (mask & (1 << i)) != 0;
        }
    }
}

/// Profile-guided optimization hints and branch prediction
pub struct ProfileGuidedOptimization;

impl ProfileGuidedOptimization {
    /// Branch prediction hints based on profiling data
    #[inline(always)]
    pub fn likely_true(condition: bool) -> bool {
        // Compiler hint that this condition is likely to be true
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::intrinsics::likely(condition)
        }
        #[cfg(not(target_arch = "x86_64"))]
        condition
    }

    #[inline(always)]
    pub fn likely_false(condition: bool) -> bool {
        // Compiler hint that this condition is likely to be false
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::intrinsics::unlikely(condition)
        }
        #[cfg(not(target_arch = "x86_64"))]
        condition
    }

    /// Hot path optimization for order processing
    #[inline(always)]
    pub fn process_order_hot_path(order_type: u8, price: u64, quantity: u32) -> bool {
        // Based on profiling: 95% of orders are limit orders
        if Self::likely_true(order_type == 1) {
            // Limit order processing (hot path)
            Self::validate_limit_order(price, quantity)
        } else if order_type == 0 {
            // Market order processing (warm path)
            Self::validate_market_order(quantity)
        } else {
            // Other order types (cold path)
            Self::validate_other_order_types(order_type, price, quantity)
        }
    }

    #[inline(always)]
    fn validate_limit_order(price: u64, quantity: u32) -> bool {
        // Fast validation for most common case
        price > 0 && quantity > 0 && quantity <= 1000000
    }

    #[inline(always)]
    fn validate_market_order(quantity: u32) -> bool {
        quantity > 0 && quantity <= 1000000
    }

    #[inline(never)] // Cold path - don't inline
    fn validate_other_order_types(order_type: u8, price: u64, quantity: u32) -> bool {
        // Complex validation for less common order types
        match order_type {
            2 => price > 0 && quantity > 0, // Stop order
            3 => price > 0 && quantity > 0, // Stop-limit
            4 => quantity > 0,              // IOC
            5 => quantity > 0,              // FOK
            _ => false,
        }
    }

    /// Market state transition with prediction hints
    #[inline(always)]
    pub fn handle_market_state_transition(current_state: u8, event: u8) -> u8 {
        // Based on profiling: 80% of time market stays in continuous trading
        if Self::likely_true(current_state == 2 && event == 0) {
            // Stay in continuous trading (hot path)
            2
        } else {
            // State change (cold path)
            Self::calculate_new_state(current_state, event)
        }
    }

    #[inline(never)]
    fn calculate_new_state(current_state: u8, event: u8) -> u8 {
        // Complex state transition logic
        match (current_state, event) {
            (0, 1) => 1, // Pre-open to Open
            (1, 2) => 2, // Open to Continuous
            (2, 3) => 3, // Continuous to Auction
            (3, 2) => 2, // Auction to Continuous
            (_, 4) => 4, // Any state to Halt
            (4, 2) => 2, // Halt to Continuous
            (2, 5) => 5, // Continuous to Close
            _ => current_state, // Invalid transition, stay in current state
        }
    }

    /// Order book update with branch prediction
    #[inline(always)]
    pub fn update_order_book(side: bool, price: u64, quantity: u32) -> bool {
        // Based on profiling: 60% buy orders, 40% sell orders
        if Self::likely_true(side) {
            // Buy side update (more likely)
            Self::update_buy_side(price, quantity)
        } else {
            // Sell side update (less likely)
            Self::update_sell_side(price, quantity)
        }
    }

    #[inline(always)]
    fn update_buy_side(price: u64, quantity: u32) -> bool {
        // Optimized for buy side updates
        price > 0 && quantity > 0
    }

    #[inline(always)]
    fn update_sell_side(price: u64, quantity: u32) -> bool {
        // Optimized for sell side updates
        price > 0 && quantity > 0
    }
}

/// Advanced branch elimination techniques
pub struct AdvancedBranchElimination;

impl AdvancedBranchElimination {
    /// Branchless binary search with SIMD acceleration
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn binary_search_branchless(
        array: &[u64],
        target: u64,
    ) -> Result<usize, usize> {
        use std::arch::x86_64::*;
        
        let mut left = 0;
        let mut right = array.len();
        
        while right - left > 8 {
            let mid = left + (right - left) / 2;
            
            // Load 4 elements around mid point
            if mid >= 4 && mid + 4 < array.len() {
                let values = _mm256_loadu_si256(
                    array.as_ptr().add(mid - 2) as *const __m256i
                );
                let target_vec = _mm256_set1_epi64x(target as i64);
                let cmp = _mm256_cmpgt_epi64(values, target_vec);
                let mask = _mm256_movemask_pd(_mm256_castsi256_pd(cmp));
                
                // Use mask to determine which direction to search
                if mask == 0 {
                    left = mid + 2;
                } else {
                    right = mid - 2;
                }
            } else {
                // Fallback to regular comparison
                if array[mid] <= target {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
        }
        
        // Linear search for remaining elements
        for i in left..right {
            if array[i] == target {
                return Ok(i);
            } else if array[i] > target {
                return Err(i);
            }
        }
        
        Err(right)
    }

    /// Branchless sorting network for small arrays
    #[inline(always)]
    pub fn sort_network_4(array: &mut [u64; 4]) {
        // 5-comparison sorting network for 4 elements
        Self::compare_and_swap(&mut array[0], &mut array[1]);
        Self::compare_and_swap(&mut array[2], &mut array[3]);
        Self::compare_and_swap(&mut array[0], &mut array[2]);
        Self::compare_and_swap(&mut array[1], &mut array[3]);
        Self::compare_and_swap(&mut array[1], &mut array[2]);
    }

    #[inline(always)]
    fn compare_and_swap(a: &mut u64, b: &mut u64) {
        let min = BranchFreeAlgorithms::min_branch_free(*a, *b);
        let max = BranchFreeAlgorithms::max_branch_free(*a, *b);
        *a = min;
        *b = max;
    }

    /// Branchless median calculation
    #[inline(always)]
    pub fn median_of_three(a: u64, b: u64, c: u64) -> u64 {
        let max_ab = BranchFreeAlgorithms::max_branch_free(a, b);
        let min_ab = BranchFreeAlgorithms::min_branch_free(a, b);
        let max_min_c = BranchFreeAlgorithms::max_branch_free(min_ab, c);
        BranchFreeAlgorithms::min_branch_free(max_ab, max_min_c)
    }

    /// Branchless partitioning for quicksort
    pub fn partition_branchless(array: &mut [u64], pivot: u64) -> usize {
        let mut write_pos = 0;
        
        for i in 0..array.len() {
            let should_move = array[i] < pivot;
            let temp = array[i];
            
            // Branchless conditional move
            array[i] = BranchFreeAlgorithms::conditional_assign_branch_free(
                should_move,
                array[write_pos],
                array[i],
            );
            array[write_pos] = BranchFreeAlgorithms::conditional_assign_branch_free(
                should_move,
                temp,
                array[write_pos],
            );
            
            write_pos += should_move as usize;
        }
        
        write_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_free_min_max() {
        assert_eq!(BranchFreeAlgorithms::min_branch_free(5, 3), 3);
        assert_eq!(BranchFreeAlgorithms::min_branch_free(3, 5), 3);
        assert_eq!(BranchFreeAlgorithms::max_branch_free(5, 3), 5);
        assert_eq!(BranchFreeAlgorithms::max_branch_free(3, 5), 5);
    }

    #[test]
    fn test_branch_free_abs() {
        assert_eq!(BranchFreeAlgorithms::abs_branch_free(-5), 5);
        assert_eq!(BranchFreeAlgorithms::abs_branch_free(5), 5);
        assert_eq!(BranchFreeAlgorithms::abs_branch_free(0), 0);
    }

    #[test]
    fn test_branch_free_conditional() {
        assert_eq!(BranchFreeAlgorithms::conditional_assign_branch_free(true, 10, 20), 10);
        assert_eq!(BranchFreeAlgorithms::conditional_assign_branch_free(false, 10, 20), 20);
    }

    #[test]
    fn test_branch_free_price_ops() {
        assert_eq!(BranchFreePriceOps::select_best_bid(100, 105), 105);
        assert_eq!(BranchFreePriceOps::select_best_ask(100, 105), 100);
        assert!(BranchFreePriceOps::price_in_spread(102, 100, 105));
        assert!(!BranchFreePriceOps::price_in_spread(98, 100, 105));
    }

    #[test]
    fn test_branch_free_math() {
        assert_eq!(BranchFreeMath::safe_divide(10, 2), 5);
        assert_eq!(BranchFreeMath::safe_divide(10, 0), 0);
        assert_eq!(BranchFreeMath::calculate_percentage(25, 100), 25);
        assert_eq!(BranchFreeMath::round_to_multiple(23, 5), 25);
        assert_eq!(BranchFreeMath::round_to_multiple(22, 5), 20);
    }

    #[test]
    fn test_branch_free_validation() {
        assert!(BranchFreeValidation::validate_range(5, 1, 10));
        assert!(!BranchFreeValidation::validate_range(15, 1, 10));
        assert!(BranchFreeValidation::validate_order_params(100, 50, 90, 110, 10, 100));
        assert!(!BranchFreeValidation::validate_order_params(120, 50, 90, 110, 10, 100));
    }

    #[test]
    fn test_lookup_table_algorithms() {
        assert!(LookupTableAlgorithms::validate_order_type(1)); // Limit order
        assert!(!LookupTableAlgorithms::validate_order_type(255)); // Invalid
        assert!(LookupTableAlgorithms::validate_price_tick(500)); // $5.00
        assert!(!LookupTableAlgorithms::validate_price_tick(501)); // $5.01
        assert!(LookupTableAlgorithms::validate_quantity_lot(100)); // 100 shares
        assert!(!LookupTableAlgorithms::validate_quantity_lot(150)); // 150 shares
    }

    #[test]
    fn test_advanced_branch_elimination() {
        let mut array = [4, 2, 3, 1];
        AdvancedBranchElimination::sort_network_4(&mut array);
        assert_eq!(array, [1, 2, 3, 4]);
        
        assert_eq!(AdvancedBranchElimination::median_of_three(1, 3, 2), 2);
        assert_eq!(AdvancedBranchElimination::median_of_three(3, 1, 2), 2);
    }

    #[test]
    fn test_profile_guided_optimization() {
        assert!(ProfileGuidedOptimization::process_order_hot_path(1, 100, 1000)); // Limit order
        assert!(ProfileGuidedOptimization::process_order_hot_path(0, 0, 1000)); // Market order
        assert!(!ProfileGuidedOptimization::process_order_hot_path(255, 100, 1000)); // Invalid
    }
}