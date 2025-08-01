# Unimplemented Features

This document outlines potential features and improvements that are not yet implemented but could be valuable additions to the project.

## 1. Additional Spread Estimators

- **Description:** The project could be extended to include a wider range of bid-ask spread estimators. This would allow for more robust analysis and comparison of different estimation techniques.
- **Suggestions:**
    - **Gibbs Sampler-Based Estimator (Hasbrouck):** An estimator based on a Gibbs sampler, which is well-suited for microstructure analysis.
    - **High-Frequency Estimator (Corwin and Schultz):** An estimator that uses high and low prices to estimate the spread, suitable for high-frequency data.

## 2. Advanced Financial Models

- **Description:** The modeling capabilities of the project could be expanded to include other types of financial models.
- **Suggestions:**
    - **Agent-Based Models (ABM):** Simulating the behavior of individual market participants (agents) to understand emergent market dynamics.
    - **GARCH Models:** Incorporating GARCH models for volatility forecasting and analysis.

## 3. Data Integration

- **Description:** The project could benefit from modules that facilitate the integration of real-world data.
- **Suggestions:**
    - **Real-Time Market Data Feeds:** Connectors for popular market data providers (e.g., via WebSocket or FIX protocol).
    - **Historical Data Importers:** Tools for importing and parsing historical data from various sources and formats (e.g., TAQ data).

## 4. Comprehensive Backtesting Engine

- **Description:** A dedicated backtesting engine would allow users to rigorously test trading strategies against the simulated market.
- **Suggestions:**
    - **Strategy Definition API:** An API for defining custom trading strategies.
    - **Performance Metrics:** A comprehensive set of performance metrics, including Sharpe ratio, max drawdown, and alpha/beta.
    - **Portfolio Management:** Features for managing a portfolio of assets and tracking its performance over time.

## 5. Visualization Tools

- **Description:** A visualization component would make it easier to analyze and understand the results of simulations and analysis.
- **Suggestions:**
    - **Plotting Library Integration:** Integration with a plotting library (e.g., `plotters` in Rust) to generate charts and graphs.
    - **Interactive Dashboards:** A web-based dashboard for visualizing real-time market data, order book dynamics, and backtesting results.

## 6. Graphical User Interface (GUI)

- **Description:** A GUI would make the project more accessible to a broader audience.
- **Suggestions:**
    - **Desktop Application:** A native desktop application built with a framework like `Tauri` or `egui`.
    - **Web-Based Interface:** A web-based interface built with a framework like `Yew` or `Dioxus`.

---

# Implementation Roadmap & Detailed Guidelines

The following section provides **step-by-step** technical guidance, recommended Rust crates, architectural notes, and testing strategies for turning each unimplemented idea into production-ready functionality.

## 1. Additional Spread Estimators
1. **Design a Common Trait**  
   ```rust
   pub trait SpreadEstimator {
       fn estimate(&self, prices: &[f64]) -> Result<Vec<f64>, EstimationError>;
   }
   ```
2. **Hasbrouck Gibbs Sampler**  
   • Crates: `rand_distr`, `statrs`, `rayon` for parallel chains.  
   • Implement a Gibbs loop generating latent true prices & spreads.  
   • Validate convergence with Gelman-Rubin statistic.
3. **Corwin-Schultz High/Low Estimator**  
   • Crates: none needed beyond `ndarray`.  
   • Accept OHLC bars, compute β and γ terms per the paper, vectorise with `ndarray`.
4. **Benchmark & Compare**  
   • Add `criterion` benchmarks and store results in CI artifact for regression tracking.

## 2. Advanced Financial Models
1. **Agent-Based Model (ABM)**  
   • Create `src/models/abm.rs` with `Agent` trait (`decide`, `act`).  
   • Use `slotmap` for fast entity storage and `rand` for stochastic decision trees.  
   • Visualise order-book heat-maps via `plotters`.
2. **GARCH Volatility Module**  
   • Leverage `rugarch`-like design (but in Rust).  
   • Use `ndarray` for vectorised likelihood; optimise parameters via existing L-BFGS.
3. **Testing**  
   • Reproduce canonical results from Bollerslev (1986) paper as unit tests.

## 3. Data Integration
1. **Real-Time Feeds**  
   • Implement WebSocket connector using `tokio-tungstenite`.  
   • Support FIX via `fix-rs` crate.  
   • Normalise into internal `TradeData` struct.
2. **Historical Importers**  
   • Parse TAQ binary with `byteorder`; for CSV use `csv_async`.  
   • Provide CLI `market-import` with `structopt`.
3. **Data Lake**  
   • Store raw & cleaned data in `parquet` using `arrow2` for columnar efficiency.

## 4. Comprehensive Backtesting Engine
1. **Strategy API**  
   ```rust
   pub trait Strategy {
       fn on_tick(&mut self, tick: &TradeData) -> Vec<Order>;
   }
   ```
2. **Execution Simulator**  
   • Extend existing matching engine to route simulated orders.  
   • Track P&L, inventory, and fees.
3. **Performance Metrics**  
   • Compute Sharpe, Sortino, max drawdown using `statrs`.
4. **Batch Runs & Walk-Forward**  
   • Use `rayon` to simulate parameter grid; summarise results with the validation framework.

## 5. Visualization Tools
1. **Static Charts**  
   • Integrate `plotters` for PNG/SVG output.  
   • Provide helpers in `viz.rs` (candles, heat-maps, cumulative P&L).
2. **Interactive Dashboard**  
   • Use `egui` + `eframe` for a cross-platform desktop dashboard.  
   • Stream data over `tokio::sync::broadcast` channels.

## 6. Graphical User Interface (GUI)
1. **Desktop (Tauri)**  
   • Scaffold with `cargo tauri init`; call Rust core via FFI.  
   • Bundle static assets from `viz` module.
2. **Web (Yew/Dioxus)**  
   • Compile core logic to WASM with `wasm-bindgen`; expose simple FFI for estimators.  
   • Use WebSockets to stream live results from backend.
3. **Accessibility & Theme**  
   • Follow WCAG AA; provide dark/light mode toggle.

