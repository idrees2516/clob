# Core Components

This document provides a detailed breakdown of the core components and modules of the project.

## 1. `main.rs` and `lib.rs`

- **`main.rs`:** The entry point for the executable binary. This file is responsible for parsing command-line arguments, setting up the environment, and running the main application logic.
- **`lib.rs`:** The entry point for the library crate. It defines the public API of the project, making its components available for other Rust projects to use as a dependency.

## 2. `models/`

- **Description:** This directory contains the core financial models of the project. It is the heart of the simulation engine.
- **Components:**
    - **`lob/`:** The implementation of the Limit Order Book (LOB).
    - **`lob_rough_vol/`:** The implementation of the LOB with the rough volatility model.

## 3. `estimators/`

- **Description:** This directory contains the implementations of various bid-ask spread estimators.
- **Components:**
    - **`roll_improved.rs`:** An implementation of Roll's model, adjusted for serial dependence in price changes.

## 4. `compute/`

- **Description:** This directory provides abstractions for different computation backends, allowing the project to leverage various hardware for performance.
- **Components:**
    - **`gpu.rs`:** Code for offloading computations to a GPU.
    - **`distributed.rs`:** Code for distributing computations across multiple nodes.

## 5. `optimization.rs`

- **Description:** This file contains algorithms for numerical optimization. These are likely used for calibrating model parameters to market data.

## 6. `utils.rs`

- **Description:** A collection of utility functions and helper modules that are used throughout the project. This helps to avoid code duplication and keep the main logic clean.

## 7. `error.rs`

- **Description:** This file defines the custom error types used in the project. A robust error handling strategy is crucial for a financial application, and this module provides the necessary infrastructure.

## 8. `metrics.rs`

- **Description:** This file includes functions for calculating various performance and risk metrics. These are essential for evaluating the results of simulations and backtests.

## 9. `validation.rs` and `validation/`

- **Description:** These files contain the framework and tools for model validation and backtesting. This is a critical component for ensuring the accuracy and reliability of the financial models.
