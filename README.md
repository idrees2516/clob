# Limit Order Book with Rough Volatility

A high-performance Rust implementation of a Limit Order Book (LOB) with rough volatility modeling for financial markets. This implementation provides a realistic simulation of market microstructure with support for rough volatility processes.

## Features

- **Full Limit Order Book** implementation with price-time priority
- **Rough Volatility Modeling** using fractional Brownian motion
- **High-Performance** matching engine with event-driven architecture
- **Thread-Safe** implementation with concurrent access support
- **Comprehensive Order Types**:
  - Market orders
  - Limit orders
  - Time-in-force options (GTC, IOC, FOK)
- **Advanced Features**:
  - Circuit breakers for extreme market conditions
  - Rate limiting for order flow control
  - Batch order processing
  - Order book snapshots and level updates
- **Simulation Tools**:
  - Volatility-adaptive market data simulation
  - Realistic price impact modeling
  - Configurable market regimes

## Features

- Robust spread estimation using Roll's model with serial dependence adjustments
- Efficient parallel processing for large datasets
- Rolling window volatility estimation
- Bootstrap confidence intervals
- Autocorrelation analysis
- Price process simulation capabilities
- Comprehensive error handling
- Thread-safe implementations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
bid-ask-spread-estimation = { git = "https://github.com/yourusername/bid-ask-spread-estimation" }
```

Or install the binary:

```bash
cargo install --git https://github.com/yourusername/bid-ask-spread-estimation
```

## Usage

### Basic Usage

```rust
use bid_ask_spread_estimation::{
    LimitOrderBook, Order, OrderType, Side, 
    RoughVolatilityModel, RoughVolatilityParams
};
use rust_decimal_macros::dec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new order book with tick size 0.01 and lot size 0.1
    let mut lob = LimitOrderBook::new(dec!(0.01), dec!(0.1))?;
    
    // Initialize volatility model
    let params = RoughVolatilityParams {
        hurst: 0.1,          // Hurst parameter (0 < H < 0.5)
        nu: 0.3,             // Volatility of volatility
        kappa: 1.0,          // Mean reversion speed
        theta: 0.1,          // Long-term mean of volatility
        v0: 0.1,             // Initial volatility
        rho: -0.7,           // Price-volatility correlation
        time_steps_per_day: 390,  // Trading minutes
        days: 1,             // Simulation days
        seed: Some(42),      // Fixed seed for reproducibility
    };
    
    // Add limit orders
    let order1 = Order::limit(Side::Bid, dec!(99.5), 100, None, None, None)?;
    let order2 = Order::limit(Side::Ask, dec!(100.5), 100, None, None, None)?;
    
    lob.process_order(order1)?;
    lob.process_order(order2)?;
    
    // Get best bid/ask
    println!("Best Bid: {:?}", lob.best_bid());
    println!("Best Ask: {:?}", lob.best_ask());
    
    // Simulate market data updates
    for _ in 0..10 {
        lob.simulate_market_data(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?.as_secs())?;
    }
    
    Ok(())
}
```

## Advanced Features

### Volatility Simulation

```rust
// Create and configure volatility model
let mut vol_model = RoughVolatilityModel::new(params)?;

// Simulate price path
let price_path = vol_model.simulate_price_path(100.0)?;
```

### Circuit Breakers

```rust
// Create a circuit breaker with 5-second cooldown
let breaker = CircuitBreaker::new(std::time::Duration::from_secs(5));

// Check if trading is allowed
if !breaker.is_triggered() {
    // Place orders
} else {
    // Handle market closure
}
```

### Batch Order Processing

```rust
let mut batch = OrderBatch::new(100);

// Add multiple orders
for i in 0..50 {
    let order = Order::limit(
        Side::Bid, 
        dec!(100.0 - i as f64 * 0.1), 
        100, 
        None, 
        None, 
        None
    )?;
    batch.add_order(order)?;
}

// Process all orders efficiently
let trades = batch.process(|order| {
    // Custom order processing logic
    Ok(None)
})?;
```


## Error Handling

The library provides comprehensive error handling through several error types:

- `LOBError`: General limit order book errors
- `MatchingError`: Order matching and execution errors
- `VolatilityError`: Volatility model simulation errors
- `OptimizationError`: Circuit breakers and rate limiting errors

Each error type implements `std::error::Error` and provides detailed error messages.

## Dependencies

- **Core**:
  - `ndarray`: N-dimensional array computations
  - `rustfft`: Fast Fourier transforms for volatility modeling
  - `rand`: Random number generation
  - `rust_decimal`: High-precision decimal arithmetic

- **Performance**:
  - `parking_lot`: High-performance synchronization primitives
  - `rayon`: Data parallelism
  - `lru`: LRU cache for order book optimizations

- **Utilities**:
  - `serde`: Serialization/deserialization
  - `tracing`: Structured, event-based logging
  - `thiserror`: Ergonomic error definitions

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

## License

MIT License

## References

- [Rough Volatility in Financial Markets](https://arxiv.org/abs/1508.02511)
- [Limit Order Book Modeling](https://www.springer.com/gp/book/9783642198988)
- [Market Microstructure Theory](https://www.wiley.com/en-us/Market+Microstructure+Theory-p-9780631207613)
