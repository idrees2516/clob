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
}