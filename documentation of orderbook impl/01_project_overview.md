# Project Overview

## 1. Introduction

This project is evolving into a full-featured **electronic exchange platform** built around a high-performance Limit Order Book (LOB). In addition to its rich analytical tool-set, the codebase will expose production-grade exchange functionality—order intake APIs, real-time matching, risk controls, persistence, and market data dissemination—while retaining the advanced research capabilities for spread estimation and microstructure analysis.

The vision is to offer a unified engine that can:
- Operate as a standalone exchange for simulated or live environments.
- Serve as a research sandbox for quantitative finance and market microstructure studies.
- Provide an extensible foundation for bespoke trading venues, dark pools, or internal matching facilities.

## 2. Key Features

The project boasts a comprehensive set of features, including:

- **Full Limit Order Book (LOB):** A complete implementation of a LOB with price-time priority for order matching.
- **Rough Volatility Modeling:** Incorporates fractional Brownian motion to model the rough nature of volatility observed in financial markets.
- **High-Performance Matching Engine:** An event-driven matching engine optimized for <1 µs matching latency on commodity hardware.
- **Concurrency:** A thread-safe design that allows for concurrent access and processing of orders.
- **Advanced Order Types:** Supports a variety of order types, including Market, Limit, and different time-in-force options (GTC, IOC, FOK).
- **Exchange-Grade Risk Management:** Pre-trade risk checks (max order size, position limits), post-trade risk tracking, circuit breakers, and rate limiting.
- **Spread Estimation:** Implements Roll's model with adjustments for serial dependence to provide robust bid-ask spread estimations.
- **Market Data & Analytics:** Real-time market data streams (depth, trades, OHLC bars) plus built-in analytics for spread, volatility, and market impact.

## 3. Exchange Architecture

The project is structured as a modular Rust application, with a clear separation of concerns. The main components are:

- **`main.rs` / `lib.rs`:** The entry points for the binary and library crates, respectively.
- **`models/`:** Contains the core financial models, including the LOB and the rough volatility model.
- **`estimators/`:** Implements various bid-ask spread estimators.
- **`compute/`:** Provides abstractions for different computation backends (e.g., CPU, GPU, distributed).
- **`optimization/`:** Contains algorithms for numerical optimization.
- **`utils/`:** A collection of utility functions used throughout the project.
- **`error.rs`:** Defines custom error types for robust error handling.
- **`metrics.rs`:** Includes functions for calculating performance and risk metrics.
- **`validation.rs`:** Provides tools and methods for model validation.
