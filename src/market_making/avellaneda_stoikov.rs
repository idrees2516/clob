use crate::error::Result;
use crate::market_making::{MarketMakingState, QuoteSet};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Avellaneda-Stoikov optimal market making engine
/// Implements the complete Hamilton-Jacobi-Bellman equation solution
/// with inventory risk management and adverse selection protection
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovEngine {
    /// Risk aversion parameter (gamma)
    pub risk_aversion: f64,
    /// Time horizon for optimization
    pub time_horizon: f64,
    /// Current model parameters
    pub parameters: AvellanedaStoikovParameters,
    /// Historical data for parameter estimation
    pub historical_data: Vec<MarketDataPoint>,
    /// Model state
    pub state: AvellanedaStoikovState,
}

/// Complete Avellaneda-Stoikov parameters
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovParameters {
    /// Risk aversion parameter (γ)
    pub gamma: f64,
    /// Volatility estimate (σ)
    pub sigma: f64,
    /// Order arrival intensity baseline (λ₀)
    pub lambda_0: f64,
    /// Market impact parameter (k)
    pub k: f64,
    /// Adverse selection parameter (α)
    pub alpha: f64,
    /// Time decay parameter
    pub time_decay: f64,
    /// Inventory penalty multiplier
    pub inventory_penalty: f64,
    /// Microstructure adjustment factor
    pub microstructure_adjustment: f64,
    /// Jump risk premium
    pub jump_risk_premium: f64,
    /// Correlation adjustment factors
    pub correlation_adjustments: HashMap<String, f64>,
}

/// Avellaneda-Stoikov model state
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovState {
    /// Current inventory positions
    pub inventory: HashMap<String, f64>,
    /// Reservation prices
    pub reservation_prices: HashMap<String, f64>,
    /// Optimal spreads
    pub optimal_spreads: HashMap<String, f64>,
    /// Quote skewness factors
    pub skewness_factors: HashMap<String, f64>,
    /// Arrival rate estimates
    pub arrival_rates: HashMap<String, (f64, f64)>, // (bid, ask)
    /// Utility function values
    pub utility_values: HashMap<String, f64>,
    /// Risk metrics
    pub risk_metrics: AvellanedaStoikovRiskMetrics,
}

/// Risk metrics specific to Avellaneda-Stoikov model
#[derive(Debug, Clone)]
pub struct AvellanedaStoikovRiskMetrics {
    pub inventory_risk: f64,
    pub adverse_selection_risk: f64,
    pub timing_risk: f64,
    pub volatility_risk: f64,
    pub utility_variance: f64,
    pub expected_utility: f64,
}

/// Market data point for parameter estimation
#[derive(Debug, Clone)]
pub struct MarketDataPoint {
    pub timestamp: u64,
    pub mid_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub trade_volume: f64,
    pub volatility: f64,
}

impl AvellanedaStoikovEngine {
    /// Create new Avellaneda-Stoikov engine
    pub fn new(risk_aversion: f64, time_horizon: f64) -> Result<Self> {
        let parameters = AvellanedaStoikovParameters {
            gamma: risk_aversion,
            sigma: 0.01, // Initial volatility estimate
            lambda_0: 1.0, // Initial arrival rate
            k: 1.0, // Initial market impact
            alpha: 0.001, // Initial adverse selection
            time_decay: 1.0,
            inventory_penalty: 1.0,
            microstructure_adjustment: 0.0,
            jump_risk_premium: 0.0,
            correlation_adjustments: HashMap::new(),
        };
        
        let state = AvellanedaStoikovState {
            inventory: HashMap::new(),
            reservation_prices: HashMap::new(),
            optimal_spreads: HashMap::new(),
            skewness_factors: HashMap::new(),
            arrival_rates: HashMap::new(),
            utility_values: HashMap::new(),
            risk_metrics: AvellanedaStoikovRiskMetrics::default(),
        };
        
        Ok(Self {
            risk_aversion,
            time_horizon,
            parameters,
            historical_data: Vec::new(),
            state,
        })
    }
    
    /// Generate optimal quotes using Avellaneda-Stoikov model
    pub fn generate_quotes(&self, symbol: &str, market_state: &MarketMakingState) -> Result<QuoteSet> {
        // Get current market data
        let mid_price = self.get_mid_price(symbol, market_state)?;
        let inventory = market_state.inventory.get(symbol).copied().unwrap_or(0.0);
        let volatility = market_state.volatility_estimates.get(symbol).copied().unwrap_or(0.01);
        
        // Calculate reservation price
        let reservation_price = self.calculate_reservation_price(
            mid_price,
            inventory,
            volatility,
            symbol,
            market_state,
        )?;
        
        // Calculate optimal spread
        let optimal_spread = self.calculate_optimal_spread(
            volatility,
            inventory,
            symbol,
            market_state,
        )?;
        
        // Calculate quote skewness
        let skewness = self.calculate_quote_skewness(
            inventory,
            volatility,
            optimal_spread,
            symbol,
            market_state,
        )?;
        
        // Generate bid and ask prices
        let bid_price = reservation_price - (optimal_spread * (1.0 - skewness)) / 2.0;
        let ask_price = reservation_price + (optimal_spread * (1.0 + skewness)) / 2.0;
        
        // Calculate optimal quote sizes
        let (bid_size, ask_size) = self.calculate_optimal_sizes(
            inventory,
            volatility,
            optimal_spread,
            symbol,
            market_state,
        )?;
        
        // Calculate expected profit and confidence
        let expected_profit = self.calculate_expected_profit(
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            symbol,
            market_state,
        )?;
        
        let confidence = self.calculate_confidence(
            volatility,
            inventory,
            symbol,
            market_state,
        )?;
        
        Ok(QuoteSet {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            confidence,
            expected_profit,
        })
    }
    
    /// Calculate reservation price with comprehensive adjustments
    fn calculate_reservation_price(
        &self,
        mid_price: f64,
        inventory: f64,
        volatility: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Base reservation price: r₀ = S - q*γ*σ²*(T-t)
        let time_remaining = self.time_horizon; // Simplified for now
        let base_reservation = mid_price - inventory * self.parameters.gamma * volatility.powi(2) * time_remaining;
        
        // Jump risk adjustment: r₁ = r₀ - q*λⱼ*E[J]*(T-t)
        let jump_adjustment = if let Some(jump_params) = market_state.jump_parameters.get(symbol) {
            inventory * jump_params.intensity * jump_params.mean_jump_size * time_remaining
        } else {
            0.0
        };
        
        // Correlation adjustment: r₂ = r₁ - Σᵢ qᵢ*ρᵢ*σᵢ*σ*(T-t)
        let correlation_adjustment = self.calculate_correlation_adjustment(
            inventory,
            volatility,
            time_remaining,
            symbol,
            market_state,
        )?;
        
        // Microstructure adjustment
        let microstructure_adjustment = self.calculate_microstructure_adjustment(
            symbol,
            market_state,
        )?;
        
        // Adverse selection adjustment
        let adverse_selection_adjustment = self.calculate_adverse_selection_adjustment(
            inventory,
            volatility,
            symbol,
            market_state,
        )?;
        
        let final_reservation = base_reservation 
            - jump_adjustment 
            - correlation_adjustment 
            + microstructure_adjustment
            - adverse_selection_adjustment;
        
        Ok(final_reservation)
    }
    
    /// Calculate optimal spread using HJB equation solution
    fn calculate_optimal_spread(
        &self,
        volatility: f64,
        inventory: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        let time_remaining = self.time_horizon;
        
        // Base optimal spread: δ* = γσ²(T-t) + (2/γ)ln(1 + γ/k)
        let base_spread = self.parameters.gamma * volatility.powi(2) * time_remaining
            + (2.0 / self.parameters.gamma) * (1.0 + self.parameters.gamma / self.parameters.k).ln();
        
        // Adverse selection premium
        let adverse_selection_premium = self.calculate_adverse_selection_premium(
            volatility,
            time_remaining,
            symbol,
            market_state,
        )?;
        
        // Jump risk premium
        let jump_risk_premium = if let Some(jump_params) = market_state.jump_parameters.get(symbol) {
            jump_params.intensity * jump_params.jump_variance.sqrt() * time_remaining.sqrt()
        } else {
            0.0
        };
        
        // Inventory-dependent adjustment
        let inventory_adjustment = inventory.abs() * self.parameters.inventory_penalty * volatility;
        
        // Regime-dependent adjustment
        let regime_adjustment = match market_state.regime_state {
            crate::market_making::RegimeState::Normal { .. } => 1.0,
            crate::market_making::RegimeState::Stressed { volatility_multiplier } => volatility_multiplier,
            crate::market_making::RegimeState::Crisis { .. } => 2.0,
            crate::market_making::RegimeState::Recovery { stabilization_factor } => stabilization_factor,
        };
        
        let optimal_spread = (base_spread + adverse_selection_premium + jump_risk_premium + inventory_adjustment) * regime_adjustment;
        
        // Apply minimum and maximum spread constraints
        let min_spread = 0.0001; // 1 basis point minimum
        let max_spread = 0.01;   // 100 basis points maximum
        
        Ok(optimal_spread.max(min_spread).min(max_spread))
    }
    
    /// Calculate quote skewness based on inventory position
    fn calculate_quote_skewness(
        &self,
        inventory: f64,
        volatility: f64,
        optimal_spread: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        let time_remaining = self.time_horizon;
        
        // Skew factor: κ = q*γ*σ²*(T-t)/(2*δ*)
        let base_skew = inventory * self.parameters.gamma * volatility.powi(2) * time_remaining / (2.0 * optimal_spread);
        
        // Directional jump risk adjustment
        let jump_skew_adjustment = if let Some(jump_params) = market_state.jump_parameters.get(symbol) {
            if inventory > 0.0 {
                // Long position: more sensitive to negative jumps
                -jump_params.downward_rate * (1.0 - jump_params.upward_probability)
            } else {
                // Short position: more sensitive to positive jumps
                jump_params.upward_rate * jump_params.upward_probability
            }
        } else {
            0.0
        };
        
        // Order flow imbalance adjustment
        let ofi_adjustment = if let Some(ofi) = market_state.microstructure_signals.order_flow_imbalance.get(symbol) {
            ofi * 0.1 // Scale factor
        } else {
            0.0
        };
        
        let total_skew = base_skew + jump_skew_adjustment + ofi_adjustment;
        
        // Limit skewness to reasonable range
        Ok(total_skew.max(-0.5).min(0.5))
    }
    
    /// Calculate optimal quote sizes
    fn calculate_optimal_sizes(
        &self,
        inventory: f64,
        volatility: f64,
        optimal_spread: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<(f64, f64)> {
        // Base size calculation using Kelly criterion
        let base_size = self.calculate_kelly_size(volatility, optimal_spread)?;
        
        // Inventory-dependent size adjustment
        let inventory_factor = (-self.parameters.gamma * inventory.abs()).exp();
        
        // Liquidity constraint adjustment
        let liquidity_factor = if let Some(position_limit) = market_state.liquidity_constraints.position_limits.get(symbol) {
            (position_limit - inventory.abs()) / position_limit
        } else {
            1.0
        };
        
        // Asymmetric sizing based on inventory
        let bid_size = if inventory > 0.0 {
            // Long inventory: reduce bid size
            base_size * inventory_factor * liquidity_factor * 0.8
        } else {
            // Short inventory: normal bid size
            base_size * inventory_factor * liquidity_factor
        };
        
        let ask_size = if inventory < 0.0 {
            // Short inventory: reduce ask size
            base_size * inventory_factor * liquidity_factor * 0.8
        } else {
            // Long inventory: normal ask size
            base_size * inventory_factor * liquidity_factor
        };
        
        Ok((bid_size.max(0.01), ask_size.max(0.01)))
    }
    
    /// Calculate expected profit from quotes
    fn calculate_expected_profit(
        &self,
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Get arrival rates
        let (bid_arrival_rate, ask_arrival_rate) = self.calculate_arrival_rates(
            bid_price,
            ask_price,
            symbol,
            market_state,
        )?;
        
        // Calculate expected profit per unit time
        let bid_profit = bid_arrival_rate * bid_size * (ask_price - bid_price) * 0.5;
        let ask_profit = ask_arrival_rate * ask_size * (ask_price - bid_price) * 0.5;
        
        // Subtract expected adverse selection cost
        let adverse_selection_cost = self.calculate_expected_adverse_selection_cost(
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            symbol,
            market_state,
        )?;
        
        Ok(bid_profit + ask_profit - adverse_selection_cost)
    }
    
    /// Calculate confidence in the quotes
    fn calculate_confidence(
        &self,
        volatility: f64,
        inventory: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Base confidence inversely related to volatility
        let volatility_confidence = (-volatility * 10.0).exp();
        
        // Inventory confidence (lower for extreme positions)
        let inventory_confidence = (-inventory.abs() * 0.1).exp();
        
        // Model stability confidence
        let stability_confidence = if self.historical_data.len() > 100 {
            0.9
        } else {
            0.5
        };
        
        // Regime confidence
        let regime_confidence = match market_state.regime_state {
            crate::market_making::RegimeState::Normal { .. } => 0.9,
            crate::market_making::RegimeState::Stressed { .. } => 0.7,
            crate::market_making::RegimeState::Crisis { .. } => 0.5,
            crate::market_making::RegimeState::Recovery { .. } => 0.8,
        };
        
        Ok(volatility_confidence * inventory_confidence * stability_confidence * regime_confidence)
    }
    
    /// Update inventory position
    pub fn update_inventory(&mut self, symbol: &str, new_inventory: f64) -> Result<()> {
        self.state.inventory.insert(symbol.to_string(), new_inventory);
        
        // Recalculate risk metrics
        self.update_risk_metrics(symbol)?;
        
        Ok(())
    }
    
    /// Update model parameters based on market data
    pub fn update_parameters(&mut self, market_data: &MarketDataPoint, symbol: &str) -> Result<()> {
        self.historical_data.push(market_data.clone());
        
        // Keep only recent data for parameter estimation
        if self.historical_data.len() > 10000 {
            self.historical_data.drain(0..1000);
        }
        
        // Update volatility estimate using EWMA
        self.parameters.sigma = self.estimate_volatility()?;
        
        // Update arrival intensity
        self.parameters.lambda_0 = self.estimate_arrival_intensity()?;
        
        // Update market impact parameter
        self.parameters.k = self.estimate_market_impact()?;
        
        // Update adverse selection parameter
        self.parameters.alpha = self.estimate_adverse_selection()?;
        
        Ok(())
    }
    
    // Private helper methods
    fn get_mid_price(&self, symbol: &str, market_state: &MarketMakingState) -> Result<f64> {
        // Implementation would get current mid price from market state
        Ok(100.0) // Placeholder
    }
    
    fn calculate_correlation_adjustment(
        &self,
        inventory: f64,
        volatility: f64,
        time_remaining: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        let mut adjustment = 0.0;
        
        // Calculate correlation adjustment for each other asset
        for (other_symbol, other_inventory) in &market_state.inventory {
            if other_symbol != symbol {
                if let Some(other_volatility) = market_state.volatility_estimates.get(other_symbol) {
                    // Get correlation coefficient (simplified)
                    let correlation = 0.3; // Would be calculated from correlation matrix
                    adjustment += other_inventory * correlation * other_volatility * volatility * time_remaining;
                }
            }
        }
        
        Ok(adjustment)
    }
    
    fn calculate_microstructure_adjustment(
        &self,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Tick size effects
        let tick_adjustment = 0.0001; // Minimum tick size
        
        // Queue position effects
        let queue_adjustment = 0.0; // Would be calculated based on queue position
        
        Ok(tick_adjustment + queue_adjustment)
    }
    
    fn calculate_adverse_selection_adjustment(
        &self,
        inventory: f64,
        volatility: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        if let Some(adverse_selection_measure) = market_state.microstructure_signals.adverse_selection_measure.get(symbol) {
            Ok(adverse_selection_measure * volatility * inventory.abs())
        } else {
            Ok(0.0)
        }
    }
    
    fn calculate_adverse_selection_premium(
        &self,
        volatility: f64,
        time_remaining: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Information asymmetry measure
        let information_asymmetry = if let Some(price_impact) = market_state.microstructure_signals.price_impact.get(symbol) {
            price_impact.abs() / volatility.sqrt()
        } else {
            0.0
        };
        
        Ok(self.parameters.alpha * information_asymmetry * volatility * time_remaining.sqrt())
    }
    
    fn calculate_kelly_size(&self, volatility: f64, spread: f64) -> Result<f64> {
        // Kelly criterion: f* = (μ-r)/(γσ²)
        let expected_return = spread / 2.0; // Expected profit from spread
        let risk_free_rate = 0.02; // 2% annual risk-free rate
        
        let kelly_fraction = (expected_return - risk_free_rate) / (self.parameters.gamma * volatility.powi(2));
        
        // Scale to reasonable position size
        Ok(kelly_fraction.abs().min(0.1) * 1000.0) // Max 10% of capital, scaled to shares
    }
    
    fn calculate_arrival_rates(
        &self,
        bid_price: f64,
        ask_price: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<(f64, f64)> {
        // Exponential arrival rate model: λ(δ) = λ₀ * exp(-k*δ)
        let mid_price = (bid_price + ask_price) / 2.0;
        let bid_spread = mid_price - bid_price;
        let ask_spread = ask_price - mid_price;
        
        let bid_arrival_rate = self.parameters.lambda_0 * (-self.parameters.k * bid_spread).exp();
        let ask_arrival_rate = self.parameters.lambda_0 * (-self.parameters.k * ask_spread).exp();
        
        Ok((bid_arrival_rate, ask_arrival_rate))
    }
    
    fn calculate_expected_adverse_selection_cost(
        &self,
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        symbol: &str,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        let adverse_selection_rate = self.parameters.alpha;
        let average_size = (bid_size + ask_size) / 2.0;
        let spread = ask_price - bid_price;
        
        Ok(adverse_selection_rate * average_size * spread * 0.1)
    }
    
    fn update_risk_metrics(&mut self, symbol: &str) -> Result<()> {
        if let Some(inventory) = self.state.inventory.get(symbol) {
            // Update inventory risk
            self.state.risk_metrics.inventory_risk = inventory.abs() * self.parameters.gamma;
            
            // Update other risk metrics
            self.state.risk_metrics.adverse_selection_risk = self.parameters.alpha * inventory.abs();
            self.state.risk_metrics.timing_risk = self.parameters.sigma * self.time_horizon.sqrt();
            self.state.risk_metrics.volatility_risk = self.parameters.sigma.powi(2) * inventory.abs();
        }
        
        Ok(())
    }
    
    fn estimate_volatility(&self) -> Result<f64> {
        if self.historical_data.len() < 2 {
            return Ok(0.01);
        }
        
        // Calculate returns
        let returns: Vec<f64> = self.historical_data
            .windows(2)
            .map(|window| (window[1].mid_price / window[0].mid_price).ln())
            .collect();
        
        // Calculate EWMA volatility
        let lambda = 0.94; // Decay factor
        let mut ewma_var = returns[0].powi(2);
        
        for &return_val in returns.iter().skip(1) {
            ewma_var = lambda * ewma_var + (1.0 - lambda) * return_val.powi(2);
        }
        
        Ok(ewma_var.sqrt())
    }
    
    fn estimate_arrival_intensity(&self) -> Result<f64> {
        if self.historical_data.len() < 10 {
            return Ok(1.0);
        }
        
        // Estimate from trade frequency
        let time_span = self.historical_data.last().unwrap().timestamp - self.historical_data.first().unwrap().timestamp;
        let trade_count = self.historical_data.len() as f64;
        
        Ok(trade_count / (time_span as f64 / 1000.0)) // Trades per second
    }
    
    fn estimate_market_impact(&self) -> Result<f64> {
        // Simplified market impact estimation
        // Would use regression analysis on price impact vs volume
        Ok(0.001) // Placeholder
    }
    
    fn estimate_adverse_selection(&self) -> Result<f64> {
        // Simplified adverse selection estimation
        // Would analyze post-trade price movements
        Ok(0.0001) // Placeholder
    }
}

impl Default for AvellanedaStoikovRiskMetrics {
    fn default() -> Self {
        Self {
            inventory_risk: 0.0,
            adverse_selection_risk: 0.0,
            timing_risk: 0.0,
            volatility_risk: 0.0,
            utility_variance: 0.0,
            expected_utility: 0.0,
        }
    }
}