# Hawkes Process Parameter Estimation

## Overview

This document describes the comprehensive parameter estimation framework for multivariate Hawkes processes implemented in the trading system. The framework includes maximum likelihood estimation using L-BFGS optimization, expectation-maximization algorithms, cross-validation for model selection, and extensive goodness-of-fit testing.

## Mathematical Foundation

### Multivariate Hawkes Process

A multivariate Hawkes process is defined by its intensity functions:

```
λᵢ(t) = μᵢ + Σⱼ ∫₀ᵗ κᵢⱼ(t-s) dNⱼ(s)
```

Where:
- `λᵢ(t)` is the intensity of process i at time t
- `μᵢ` is the baseline intensity for process i
- `κᵢⱼ(t)` is the excitation kernel from process j to process i
- `Nⱼ(s)` is the counting process for process j

### Kernel Types

The implementation supports multiple kernel types:

1. **Exponential Kernel**: `κ(t) = α * exp(-β * t)`
2. **Power Law Kernel**: `κ(t) = α * (t + c)^(-β)`
3. **Sum of Exponentials**: `κ(t) = Σₖ αₖ * exp(-βₖ * t)`

## Parameter Estimation Methods

### 1. Maximum Likelihood Estimation (MLE)

The log-likelihood function for a multivariate Hawkes process is:

```
ℓ(θ) = Σᵢ Σₖ log λᵢ(tₖⁱ) - Σᵢ ∫₀ᵀ λᵢ(t) dt
```

Where `θ` represents all parameters and `tₖⁱ` are the event times for process i.

#### L-BFGS Optimization

The implementation uses the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) algorithm:

```rust
use trading_system::math::{LBFGSOptimizer, HawkesMLEstimator};

let optimizer = LBFGSOptimizer::new(
    1000,                                    // max iterations
    FixedPoint::from_float(1e-6),           // tolerance
    10,                                      // history size
);

let estimator = HawkesMLEstimator::new(optimizer)
    .with_regularization(FixedPoint::from_float(0.01));

let estimated_params = estimator.estimate(
    &event_sequences,
    &initial_params,
    observation_time,
)?;
```

#### Features:
- **Regularization**: L2 regularization to prevent overfitting
- **Parameter constraints**: Ensures positivity of intensities and kernel parameters
- **Numerical stability**: Handles edge cases and numerical precision issues
- **Convergence monitoring**: Tracks optimization progress and detects convergence

### 2. Expectation-Maximization (EM) Algorithm

The EM algorithm alternates between:
- **E-step**: Compute expected sufficient statistics given current parameters
- **M-step**: Update parameters to maximize expected log-likelihood

```rust
use trading_system::math::HawkesEMEstimator;

let estimator = HawkesEMEstimator::new(
    100,                                     // max iterations
    FixedPoint::from_float(1e-6),           // tolerance
);

let estimated_params = estimator.estimate(
    &event_sequences,
    &initial_params,
    observation_time,
)?;
```

#### Advantages:
- **Guaranteed convergence**: Each iteration increases the likelihood
- **Handles missing data**: Can work with incomplete observations
- **Interpretable updates**: Clear separation of immigrant and offspring events

### 3. Cross-Validation for Model Selection

K-fold cross-validation helps select the best model among candidates:

```rust
use trading_system::math::{HawkesCrossValidator, ValidationMetric};

let validator = HawkesCrossValidator::new(5, ValidationMetric::AIC);

let result = validator.validate(
    &event_sequences,
    &candidate_models,
    observation_time,
)?;

println!("Best model: {}", result.best_model_idx);
```

#### Validation Metrics:
- **Log-Likelihood**: Direct measure of model fit
- **AIC (Akaike Information Criterion)**: Balances fit and complexity
- **BIC (Bayesian Information Criterion)**: Stronger penalty for complexity
- **Kolmogorov-Smirnov Test**: Distribution-based goodness-of-fit

## Goodness-of-Fit Testing

Comprehensive testing framework to validate model assumptions:

```rust
use trading_system::math::HawkesGoodnessOfFitTester;

let tester = HawkesGoodnessOfFitTester::new(
    FixedPoint::from_float(0.05),           // significance level
    1000,                                    // bootstrap samples
);

let result = tester.test_fit(
    &event_sequences,
    &fitted_params,
    observation_time,
)?;
```

### Test Types:

#### 1. Kolmogorov-Smirnov Test
Tests whether inter-event times follow the expected distribution:
- **Null hypothesis**: Model fits the data
- **Test statistic**: Maximum difference between empirical and theoretical CDFs
- **Critical value**: Based on sample size and significance level

#### 2. Anderson-Darling Test
More sensitive to tail behavior than KS test:
- **Weighted statistic**: Gives more weight to tail deviations
- **Better power**: More likely to detect departures from the null hypothesis

#### 3. Residual Analysis
Examines model residuals for patterns:
- **Mean residual**: Should be close to zero for good fit
- **Residual autocorrelation**: Tests for temporal dependencies
- **Standardized residuals**: Should follow standard normal distribution

#### 4. Bootstrap Test
Non-parametric approach using resampling:
- **Bootstrap samples**: Generate synthetic data from fitted model
- **Empirical distribution**: Compare test statistics across samples
- **P-value estimation**: Proportion of bootstrap statistics exceeding observed

## Implementation Details

### Parameter Vector Conversion

Parameters are converted to optimization vectors for numerical methods:

```rust
// Convert parameters to vector
let param_vector = estimator.params_to_vector(&params);

// Optimize
let optimal_vector = optimizer.optimize(param_vector, objective, gradient)?;

// Convert back to parameters
let estimated_params = estimator.vector_to_params(&optimal_vector, dimension)?;
```

### Numerical Stability

Several measures ensure numerical stability:

1. **Parameter bounds**: Enforce positivity constraints
2. **Regularization**: Prevent parameter explosion
3. **Gradient clipping**: Limit gradient magnitudes
4. **Line search**: Ensure sufficient decrease in objective
5. **Condition monitoring**: Check matrix conditioning

### Performance Optimizations

- **Parallel processing**: Cross-validation folds run in parallel
- **Efficient caching**: Intensity calculations cached when possible
- **Memory management**: Circular buffers for event history
- **SIMD operations**: Vectorized mathematical computations

## Usage Examples

### Basic Parameter Estimation

```rust
use trading_system::math::*;

// Create true parameters
let true_params = MultivariateHawkesParams::new(
    vec![FixedPoint::from_float(0.5), FixedPoint::from_float(0.3)],
    vec![
        vec![
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.2), 
                beta: FixedPoint::from_float(1.0) 
            },
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.1), 
                beta: FixedPoint::from_float(0.8) 
            },
        ],
        vec![
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.15), 
                beta: FixedPoint::from_float(0.9) 
            },
            KernelType::Exponential { 
                alpha: FixedPoint::from_float(0.25), 
                beta: FixedPoint::from_float(1.2) 
            },
        ],
    ],
    FixedPoint::from_float(10.0),
)?;

// Generate synthetic data
let mut simulator = MultivariateHawkesSimulator::new(true_params.clone(), 1000);
let mut rng = DeterministicRng::new(42);
let events = simulator.simulate_until(FixedPoint::from_float(10.0), &mut rng)?;

// Estimate parameters
let optimizer = LBFGSOptimizer::default();
let estimator = HawkesMLEstimator::new(optimizer);
let estimated_params = estimator.estimate(
    &vec![events],
    &true_params,
    FixedPoint::from_float(10.0),
)?;
```

### Model Selection with Cross-Validation

```rust
// Create candidate models
let candidates = vec![
    create_model_with_baseline(0.4),
    create_model_with_baseline(0.5),
    create_model_with_baseline(0.6),
];

// Perform cross-validation
let validator = HawkesCrossValidator::new(5, ValidationMetric::AIC);
let cv_result = validator.validate(&event_sequences, &candidates, observation_time)?;

// Select best model
let best_model = &candidates[cv_result.best_model_idx];
```

### Comprehensive Model Validation

```rust
// Fit model using ML estimation
let fitted_params = ml_estimator.estimate(&training_data, &initial_params, obs_time)?;

// Test goodness-of-fit
let tester = HawkesGoodnessOfFitTester::default();
let gof_result = tester.test_fit(&test_data, &fitted_params, obs_time)?;

// Check all tests
if !gof_result.ks_test.is_significant && 
   !gof_result.ad_test.is_significant &&
   !gof_result.bootstrap_test.is_significant {
    println!("Model passes all goodness-of-fit tests!");
} else {
    println!("Model may not fit the data well");
}
```

## Error Handling

The framework provides comprehensive error handling:

```rust
use trading_system::math::EstimationError;

match estimator.estimate(&data, &params, obs_time) {
    Ok(result) => println!("Estimation successful"),
    Err(EstimationError::InsufficientData(msg)) => {
        println!("Not enough data: {}", msg);
    }
    Err(EstimationError::OptimizationFailed(msg)) => {
        println!("Optimization failed: {}", msg);
    }
    Err(EstimationError::NumericalInstability(msg)) => {
        println!("Numerical issues: {}", msg);
    }
    Err(EstimationError::ConvergenceFailed(msg)) => {
        println!("Failed to converge: {}", msg);
    }
    Err(e) => println!("Other error: {}", e),
}
```

## Best Practices

### Data Requirements
- **Minimum events**: At least 100 events per process for reliable estimation
- **Observation time**: Should be long enough to capture kernel decay
- **Multiple sequences**: Use multiple independent sequences when possible

### Parameter Initialization
- **Reasonable bounds**: Initialize parameters within reasonable ranges
- **Prior knowledge**: Use domain knowledge for initial guesses
- **Multiple starts**: Try different initializations to avoid local minima

### Model Selection
- **Cross-validation**: Always use cross-validation for model selection
- **Information criteria**: Prefer AIC/BIC for nested model comparison
- **Out-of-sample testing**: Reserve data for final model validation

### Numerical Considerations
- **Scaling**: Normalize time scales and intensities appropriately
- **Precision**: Use sufficient numerical precision for optimization
- **Convergence**: Monitor convergence carefully and adjust tolerances

## Performance Characteristics

### Computational Complexity
- **MLE optimization**: O(n * m * k) per iteration, where n = events, m = processes, k = parameters
- **EM algorithm**: O(n * m²) per iteration for sufficient statistics computation
- **Cross-validation**: Linear scaling with number of folds and candidates

### Memory Usage
- **Event storage**: O(n) for event sequences
- **History buffers**: O(h * m) where h = history size, m = processes
- **Optimization state**: O(k * l) where k = parameters, l = L-BFGS history

### Typical Performance
- **Small problems** (2 processes, <1000 events): < 1 second
- **Medium problems** (5 processes, <10000 events): < 30 seconds  
- **Large problems** (10+ processes, >50000 events): Several minutes

## Future Enhancements

### Planned Features
1. **Non-parametric kernels**: Support for general kernel shapes
2. **Marked processes**: Handle event marks (e.g., order sizes)
3. **Time-varying parameters**: Support for regime-switching models
4. **Bayesian estimation**: MCMC-based parameter estimation
5. **Online estimation**: Real-time parameter updates

### Research Directions
1. **Deep learning integration**: Neural network-based intensity modeling
2. **Causal inference**: Identifying causal relationships between processes
3. **High-dimensional scaling**: Efficient methods for many processes
4. **Robustness**: Estimation under model misspecification

This comprehensive parameter estimation framework provides the foundation for sophisticated Hawkes process modeling in high-frequency trading applications, enabling accurate modeling of order flow dynamics and cross-excitation effects between different market processes.