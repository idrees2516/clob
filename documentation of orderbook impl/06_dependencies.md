# Project Dependencies

This document lists the external crates (libraries) that the project depends on, along with a brief description of their purpose.

## Core & Numerics

- **`nalgebra`**: A linear algebra library for Rust. Used for vector and matrix operations.
- **`ndarray`**: Provides N-dimensional arrays, essential for numerical computations, similar to NumPy in Python.
- **`num-complex`**: Provides complex number types.
- **`num-traits`**: Provides generic numeric traits.
- **`rust_decimal`**, **`rust_decimal_macros`**: For high-precision decimal arithmetic, which is crucial for financial calculations to avoid floating-point errors.
- **`statrs`**: A library of statistical functions.

## Performance & Concurrency

- **`parking_lot`**: Provides more efficient and smaller synchronization primitives (Mutex, RwLock, etc.) than the standard library.
- **`rayon`**: A data-parallelism library for Rust, making it easy to convert sequential computations into parallel ones.
- **`lru`**: An LRU (Least Recently Used) cache implementation, likely used for caching frequently accessed data to improve performance.
- **`rustfft`**: A high-performance Fast Fourier Transform (FFT) library, used in the rough volatility model.

## Random Number Generation

- **`rand`**: The main random number generation library.
- **`rand_distr`**: Provides various statistical distributions for random number generation.
- **`rand_xoshiro`**: Implements the Xoshiro family of pseudo-random number generators, known for their speed and statistical properties.
- **`ndarray-rand`**: Provides random number generation for `ndarray` arrays.

## Error Handling & Logging

- **`thiserror`**: A library for creating ergonomic custom error types.
- **`tracing`**, **`tracing-subscriber`**: A framework for instrumenting Rust programs to collect structured, event-based diagnostic information.

## Serialization & Data Handling

- **`serde`**, **`serde_json`**: The standard libraries for serializing and deserializing Rust data structures, including to and from JSON.
- **`csv`**: A library for reading and writing CSV files.
- **`chrono`**: A library for handling dates and times.

## Utilities

- **`approx`**: A library for approximate equality comparisons of floating-point numbers, useful for testing.
