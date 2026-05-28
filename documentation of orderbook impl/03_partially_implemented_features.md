# Partially Implemented Features

This document provides a detailed analysis of features that are functionally implemented but not yet fully integrated into the main application pipeline. The project contains several powerful, well-engineered modules for high-performance computing, optimization, and validation. However, they currently exist as standalone libraries awaiting integration.

## 1. GPU Computing (`src/compute/gpu.rs`)

The GPU computing module uses OpenCL to accelerate numerical computations. It is a foundational piece for offloading heavy workloads from the CPU.

### Current State: Implemented
- **OpenCL Integration:** Core integration with the `ocl` crate is complete, including context and queue management in the `GPUAccelerator` struct.
- **Kernel Implementations:** Several key computational kernels are written in OpenCL C:
  - Covariance matrix calculation.
  - Rolling window volatility.
  - A simple spread estimator.
- **Data Handling:** Functions exist to manage buffer creation and data transfer between the host and the GPU.

### Gaps & Next Steps
- **Full Integration:** The `GPUAccelerator` is not yet called by the main models or estimators. The application logic does not yet have a path to dispatch computations to the GPU.
- **Limited Scope:** The set of GPU-accelerated functions is small. More complex parts of the simulation (e.g., the rough volatility model) are not yet ported to OpenCL.
- **Configuration:** There is no user-facing configuration to enable/disable GPU usage or select a specific OpenCL device.

## 2. Distributed Computing (`src/compute/distributed.rs`)

This module provides a framework for parallel and asynchronous computation, designed to leverage multi-core processors effectively. It is misnamed as "distributed" and is currently a single-machine parallelism framework.

### Current State: Implemented
- **Async Worker Pool:** A robust asynchronous worker pool (`DistributedCompute`) is built using `tokio` for managing concurrent tasks.
- **Task-Based Parallelism:** It defines a `ComputeTask` enum for different workloads (spread estimation, volatility, etc.) and processes them in parallel.
- **Intra-Task Parallelism:** It intelligently uses `rayon` to parallelize computations *within* a single task, further maximizing CPU usage.
- **Numerical Solvers:** Includes from-scratch implementations of an OLS linear regression solver and a Cholesky decomposition solver for internal use.

### Gaps & Next Steps
- **True Distribution:** The module lacks a networking layer. It cannot distribute tasks across multiple machines, which is required for a true distributed system.
- **Integration:** Similar to the GPU module, the `DistributedCompute` worker pool is not yet integrated with the main application logic.
- **Fault Tolerance:** There are no mechanisms for handling node failures or retrying failed tasks, which are essential for a distributed system.

## 3. Optimization Algorithms (`src/optimization.rs`)

The project includes a self-contained, high-quality numerical optimization library with from-scratch implementations of two powerful algorithms.

### Current State: Implemented
- **`ObjectiveFunction` Trait:** A generic trait allows any model to be optimized, promoting modularity.
- **L-BFGS Algorithm:** A complete, from-scratch implementation of the Limited-memory BFGS quasi-Newton optimizer, including a line search sub-procedure. This is ideal for gradient-based optimization.
- **Particle Swarm Optimization (PSO):** A complete, `rayon`-parallelized implementation of the PSO metaheuristic. This is a powerful gradient-free optimizer.

### Gaps & Next Steps
- **Integration with Models:** The primary gap is that no models currently use this module for parameter calibration. For example, the `RoughVolatilityModel` parameters are hardcoded instead of being found by an optimizer.
- **Constrained Optimization:** The implemented algorithms are for unconstrained problems. Support for linear or non-linear constraints is missing.

## 4. Validation & Diagnostics Framework (`src/validation/`)

This is the most developed of the partial features—a comprehensive, multi-file framework for rigorous model validation and statistical diagnostics. It is a powerful, standalone econometric library.

### Current State: Implemented
- **Data Integrity (`validation.rs`):** Robust functions for parsing and validating input data to ensure quality (e.g., positive prices, monotonic timestamps).
- **Model Validation (`mod.rs`):** A `ModelValidator` that performs:
  - **Bootstrapping:** To estimate confidence intervals and standard errors.
  - **K-Fold Cross-Validation:** To test a model's out-of-sample predictive power.
- **Error Metrics:** A suite of standard error calculations (MSE, RMSE, MAE, R-squared).
- **Statistical Diagnostics (`diagnostics.rs`):** A `DiagnosticTests` module with from-scratch implementations of key econometric tests:
  - **Normality Test (Jarque-Bera):** To check if model residuals are normally distributed.
  - **Heteroskedasticity Test (Breusch-Pagan):** To check for non-constant variance in residuals.
  - **Autocorrelation Test (Ljung-Box):** To check for remaining patterns in residuals.
  - **Stationarity Test (Augmented Dickey-Fuller):** To check if a time series is stationary.
  - **Structural Break Test (CUSUM):** To find points of instability in a time series.

### Gaps & Next Steps
- **Full Integration:** The entire validation framework is a standalone engine. It needs to be integrated into a complete workflow where a model is trained, its parameters are optimized, and its performance and residuals are then passed to this framework for analysis.
- **Finance-Specific Metrics:** The framework could be extended with finance-specific metrics like Sharpe Ratio, Sortino Ratio, or Maximum Drawdown for backtesting trading strategies.

---

# Implementation Roadmap & Detailed Guidelines

The following guidelines provide **step-by-step** technical instructions, recommended crates, architectural considerations, and testing strategies for integrating and extending each partially implemented component.

## A. GPU Computing Roadmap
1. **Feature Flag & Configuration Layer**
   - Add a `compute` table to `Cargo.toml` with `features = ["gpu"]`.
   - Provide a `Config` struct (e.g., `src/config.rs`) that parses `RUST_GPU=on|off` env var or CLI flag.
2. **Kernel Coverage Expansion**
   - Port rough-volatility Monte-Carlo loop to OpenCL.  Use `ocl::builders::KernelBuilder` to generate kernels at runtime for different grid sizes.
   - Maintain a `kernels/` directory with `.cl` files and CI that runs `clippy` + `opencl-clang` syntax checks.
3. **Safe Abstraction Layer**
   - Wrap raw OpenCL buffers in a `GpuArray<T>` new-type implementing `Deref<Target=[T]>` for zero-copy host reads under `#[cfg(feature="gpu")]`.
4. **Integration Hooks**
   - In each estimator/model expose `compute_backend: ComputeBackend` enum (`Cpu`, `Gpu`) and pattern-match to call either `ndarray` routines or `GPUAccelerator` methods.
5. **Testing & Benchmarking**
   - Add criterion benchmarks comparing CPU vs GPU paths (goal: ≥ 10x speed-up for 1 M price ticks).
   - Use GitHub Actions with `actions-rs` and Intel OpenCL runtime to run smoke tests on pull-requests.

## B. Distributed Computing Roadmap
1. **Networking Layer**
   - Replace misleading module name to `parallel.rs` (keep alias for backward compatibility).
   - Introduce `serde + bincode` to serialize `ComputeTask` and `tokio::net::TcpStream` for peer-to-peer messaging.
   - Maintain cluster state in a `HashMap<NodeId, NodeStatus>` protected by `tokio::sync::RwLock`.
2. **Job Scheduler**
   - Implement a simple master-worker pattern: master partitions tasks → sends to workers → aggregates results.
   - Use `prost` + gRPC for production-grade RPC once prototype is stable.
3. **Fault Tolerance**
   - Store task metadata in an `sled` embedded DB; re-queue unfinished tasks on timeout.
4. **Integration**
   - Provide a thin façade `ComputeContext::submit(task)` used by models; internally route to local rayon pool or remote cluster based on config.
5. **Validation**
   - Use `docker-compose` with 3 nodes in CI to ensure deterministic aggregation results.

## C. Optimization Algorithms Roadmap
1. **Model Trait Integration**
   - Define `trait Calibratable { fn parameters(&self)->Vec<f64>; fn set_parameters(&mut self, p: &[f64]); fn objective(&self)->impl ObjectiveFunction; }`.
   - Implement for `RoughVolatilityModel` and future models.
2. **Constraint Support**
   - Introduce `argmin` crate (MIT lic.) to gain trust-region & box-constraint algorithms.
   - For L-BFGS add projection step; for PSO clip particle positions per bound.
3. **Hyper-Parameter Presets**
   - Provide TOML presets under `config/optim/*.toml` read via `serde`.
4. **Integration Tests**
   - Calibrate rough-volatility params on synthetic Heston paths; assert RMSE < 1e-3.

## D. Validation & Diagnostics Roadmap
1. **End-to-End Pipeline**
   - After each calibration run produce `ModelReport` containing parameters, residuals, diagnostics, and error metrics.
2. **Finance Metrics Extension**
   - Use `statrs` to compute downside deviation → Sortino ratio.
   - Implement rolling max-drawdown in `analysis.rs`.
3. **Walk-Forward Validation**
   - Split data chronologically: train 70 %, test 30 %, slide window by 1 day.
4. **Visualization Integration**
   - Serialize `DiagnosticResults` to JSON and render via `plotly-rs`.
5. **CI Gate**
   - Fail build if Jarque-Bera p-value < 0.05 or out-of-sample R² < 0.2.

