use trading_system::performance::simd::{
    SimdValidator, PriceConstraints, SimdPriceComparator, SimdArithmetic, SimdMemoryOps
};

fn main() {
    println!("SIMD Price Validation Demo");
    println!("==========================");

    // Define price constraints for a trading instrument
    let constraints = PriceConstraints {
        min_price: 100,      // Minimum price: $1.00 (in cents)
        max_price: 10000,    // Maximum price: $100.00 (in cents)
        tick_size: 5,        // Tick size: $0.05 (in cents)
    };

    // Sample prices to validate (in cents)
    let prices = vec![
        105,   // Valid: $1.05
        50,    // Invalid: below minimum
        15000, // Invalid: above maximum  
        103,   // Invalid: not aligned to tick size
        200,   // Valid: $2.00
        0,     // Invalid: zero price
        500,   // Valid: $5.00
        999,   // Invalid: not aligned to tick size
        1000,  // Valid: $10.00
        10005, // Invalid: above maximum
    ];

    println!("Validating {} prices...", prices.len());
    println!("Constraints: min=${:.2}, max=${:.2}, tick=${:.2}", 
             constraints.min_price as f64 / 100.0,
             constraints.max_price as f64 / 100.0,
             constraints.tick_size as f64 / 100.0);

    // Validate prices using SIMD
    let result = SimdValidator::validate_prices(&prices, &constraints);

    println!("\nValidation Results:");
    println!("Valid prices: {}/{}", result.valid_count, prices.len());
    println!("Invalid indices: {:?}", result.invalid_indices);

    // Show detailed results
    println!("\nDetailed Results:");
    for (i, &price) in prices.iter().enumerate() {
        let is_valid = (result.error_mask & (1 << i)) == 0;
        let status = if is_valid { "✓ VALID" } else { "✗ INVALID" };
        println!("  Price[{}]: ${:.2} - {}", i, price as f64 / 100.0, status);
    }

    // Demonstrate quantity validation
    println!("\n\nQuantity Validation Demo");
    println!("========================");

    let quantities = vec![100, 0, 50, 1000, 2000, 1];
    let min_quantity = 10;
    let max_quantity = 1500;

    println!("Validating {} quantities...", quantities.len());
    println!("Constraints: min={}, max={}", min_quantity, max_quantity);

    let qty_result = SimdValidator::validate_quantities(&quantities, min_quantity, max_quantity);

    println!("\nQuantity Results:");
    for (i, &qty) in quantities.iter().enumerate() {
        let is_valid = (qty_result.error_mask & (1 << i)) == 0;
        let status = if is_valid { "✓ VALID" } else { "✗ INVALID" };
        println!("  Quantity[{}]: {} - {}", i, qty, status);
    }

    // Demonstrate combined order validation
    println!("\n\nOrder Validation Demo");
    println!("=====================");

    let order_prices = vec![105, 200, 300, 400, 500];
    let order_quantities = vec![50, 0, 100, 200, 75]; // Second quantity is invalid (zero)

    println!("Validating {} orders...", order_prices.len());

    let order_result = SimdValidator::validate_orders(
        &order_prices,
        &order_quantities,
        &constraints,
        min_quantity,
        max_quantity
    );

    println!("\nOrder Results:");
    for (i, (&price, &qty)) in order_prices.iter().zip(order_quantities.iter()).enumerate() {
        let is_valid = (order_result.error_mask & (1 << i)) == 0;
        let status = if is_valid { "✓ VALID" } else { "✗ INVALID" };
        println!("  Order[{}]: ${:.2} x {} - {}", i, price as f64 / 100.0, qty, status);
    }

    // Demonstrate additional SIMD operations
    println!("\n\nAdditional SIMD Operations Demo");
    println!("===============================");

    // Price comparison operations
    let market_prices = vec![95, 100, 105, 110, 115, 120, 125];
    println!("Market prices: {:?}", market_prices);
    
    let prices_in_range = SimdPriceComparator::find_prices_in_range(&market_prices, 105, 120);
    println!("Prices between 105-120 (indices): {:?}", prices_in_range);
    
    let count_above = SimdPriceComparator::count_prices_above(&market_prices, 110);
    println!("Count of prices above 110: {}", count_above);

    // Arithmetic operations
    let trade_prices = vec![100, 110, 105, 115, 120];
    let trade_volumes = vec![50, 30, 40, 60, 25];
    
    let vwap = SimdArithmetic::vwap(&trade_prices, &trade_volumes);
    println!("Volume Weighted Average Price: {}", vwap);
    
    let mut ema_result = vec![0u64; trade_prices.len()];
    SimdArithmetic::exponential_moving_average(&trade_prices, 0.3, &mut ema_result);
    println!("Exponential Moving Average: {:?}", ema_result);

    // Memory operations
    let search_array = vec![1001, 1002, 1003, 1004, 1005];
    if let Some(index) = SimdMemoryOps::find_u64(&search_array, 1003) {
        println!("Found order ID 1003 at index: {}", index);
    }

    // Validation operations
    let order_ids = vec![1001, 1002, 0, 1004, 9999];
    let id_result = SimdValidator::validate_order_ids(&order_ids, 1000, 2000);
    println!("Valid order IDs: {}/{}", id_result.valid_count, order_ids.len());

    let now = 1640995200u64;
    let timestamps = vec![now - 100, now, now + 100, now + 10000];
    let ts_result = SimdValidator::validate_timestamps(&timestamps, now - 200, now + 200);
    println!("Valid timestamps: {}/{}", ts_result.valid_count, timestamps.len());

    println!("\nDemo completed successfully!");
}