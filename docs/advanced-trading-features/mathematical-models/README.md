# Mathematical Models Documentation

## Overview

This section provides comprehensive documentation for all mathematical models implemented in the advanced trading features system. Each model includes mathematical derivations, implementation details, and usage examples.

## Models Implemented

### 1. Avellaneda-Stoikov Market Making Model

The Avellaneda-Stoikov model provides optimal bid-ask spread calculation for market makers under inventory risk.

**Mathematical Foundation:**

The model solves the Hamilton-Jacobi-Bellman (HJB) equation:

```
∂u/∂t + (1/2)σ²S²∂²u/∂S² + max{δᵇ,δᵃ}[λ(δ)(∂u/∂q ± ∂u/∂x) - γ(∂u/∂q)²] = 0
```

Where:
- `u(t,S,x,q)` = utility function
- `S` = mid-price
- `x` = cash position
- `q` = inventory
- `δᵇ,δᵃ` = bid/ask spreads
- `λ(δ)` = order arrival intensity
- `γ` = risk aversion parameter

**Closed-Form Solution:**

- Reservation price: `r = S - q*γ*σ²*(T-t)`
- Optimal spread: `δ* = γσ²(T-t) + (2/γ)ln(1 + γ/k)`
- Bid quote: `pᵇ = r - δᵇ/2`
- Ask quote: `pᵃ = r + δᵃ/2`

### 2. Guéant-Lehalle-Tapia Multi-Asset Framework

Extends market making to multiple correlated assets with portfolio optimization.

**Multi-Asset HJB Equation:**

```
∂u/∂t + (1/2)Σᵢⱼ σᵢσⱼρᵢⱼSᵢSⱼ∂²u/∂Sᵢ∂Sⱼ + Σᵢ max{δᵢᵇ,δᵢᵃ}[λᵢ(δᵢ)(∂u/∂qᵢ ± ∂u/∂xᵢ) - γᵢⱼqⱼ∂²u/∂qᵢ∂qⱼ] = 0
```

Where:
- `ρᵢⱼ` = correlation matrix between assets i,j
- `γᵢⱼ` = cross-asset risk aversion matrix
- `qᵢ` = inventory in asset i

### 3. Cartea-Jaimungal Jump-Diffusion Model

Incorporates jump risk into market making decisions.

**Jump-Diffusion SDE:**

```
dS_t = μS_t dt + σS_t dW_t + S_t ∫ h(z) Ñ(dt,dz)
```

Where:
- `Ñ(dt,dz)` = compensated Poisson random measure
- `h(z)` = jump size function with double exponential distribution

**Jump Detection:**

Uses bi-power variation test:
```
BV_t = Σ|r_{t-1}||r_t|
QV_t = Σr_t²
Z_t = (QV_t - BV_t)/√(θ*BV_t) ~ N(0,1)
```

### 4. SDE Solvers

Numerical methods for solving stochastic differential equations.

**Euler-Maruyama Scheme:**
```
X_{n+1} = X_n + a(X_n, t_n)Δt + b(X_n, t_n)ΔW_n
```

**Milstein Scheme:**
```
X_{n+1} = X_n + a(X_n, t_n)Δt + b(X_n, t_n)ΔW_n + (1/2)b(X_n, t_n)b'(X_n, t_n)[(ΔW_n)² - Δt]
```

### 5. Hawkes Processes

Self-exciting point processes for modeling order flow.

**Intensity Function:**
```
λ(t) = λ₀ + Σᵢ:tᵢ<t α*exp(-β(t-tᵢ))
```

Where:
- `λ₀` = baseline intensity
- `α` = self-excitation parameter
- `β` = decay rate

### 6. Rough Volatility Models

Fractional Brownian motion-based volatility modeling.

**Rough Volatility SDE:**
```
dlog(σ_t) = -λ(log(σ_t) - θ)dt + ν dW_t^H
```

Where:
- `H` = Hurst parameter ∈ (0, 1)
- `W_t^H` = fractional Brownian motion

## Implementation Notes

All models are implemented with:
- Fixed-point arithmetic for deterministic calculations
- Comprehensive error handling and validation
- Property-based testing for mathematical correctness
- Performance optimization for sub-microsecond latency

## References

1. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance, 8(3), 217-224.
2. Guéant, O., Lehalle, C. A., & Tapia, J. F. (2013). Dealing with the inventory risk: a solution to the market making problem. Mathematics and Financial Economics, 7(4), 477-507.
3. Cartea, Á., & Jaimungal, S. (2015). Risk metrics and fine tuning of high-frequency trading strategies. Mathematical Finance, 25(3), 576-611.