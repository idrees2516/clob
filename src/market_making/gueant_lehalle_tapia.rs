use crate::error::Result;
use crate::market_making::{MarketMakingState, QuoteSet, HedgeOrder, TradeSide};
use nalgebra::{DMatrix, DVector, Cholesky};
use std::collections::HashMap;

/// Guéant-Lehalle-Tapia multi-asset market making engine
/// Implements portfolio-level optimization with cross-asset correlations
/// and regime-dependent parameters for sophisticated multi-asset trading
#[derive(Debug, Clone)]
pub struct GuéantLehalleTapiaEngine {
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Model parameters
    pub parameters: GuéantLehalleTapiaParameters,
    /// Multi-asset state
    pub state: GuéantLehalleTapiaState,
    /// Asset universe
    pub asset_universe: Vec<String>,
    /// Correlation estimator
    pub correlation_estimator: CorrelationEstimator,
    /// Regime detector
    pub regime_detector: RegimeDetector,
    /// Portfolio optimizer
    pub portfolio_optimizer: PortfolioOptimizer,
}

/// Complete Guéant-Lehalle-Tapia parameters
#[derive(Debug, Clone)]
pub struct GuéantLehalleTapiaParameters {
    /// Cross-asset risk aversion matrix (γᵢⱼ)
    pub cross_asset_risk_aversion: DMatrix<f64>,
    /// Correlation matrix (ρᵢⱼ)
    pub correlation_matrix: DMatrix<f64>,
    /// Volatility vector (σᵢ)
    pub volatility_vector: DVector<f64>,
    /// Arrival intensity matrix (λᵢⱼ)
    pub arrival_intensity_matrix: DMatrix<f64>,
    /// Market impact parameters (kᵢ)
    pub market_impact_parameters: DVector<f64>,
    /// Regime transition matrix
    pub regime_transition_matrix: DMatrix<f64>,
    /// Regime-dependent parameters
    pub regime_parameters: HashMap<RegimeType, RegimeParameters>,
    /// Cointegration vectors
    pub cointegration_vectors: DMatrix<f64>,
    /// Mean reversion speeds
    pub mean_reversion_speeds: DVector<f64>,
    /// Cross-impact matrix
    pub cross_impact_matrix: DMatrix<f64>,
}

/// Multi-asset state
#[derive(Debug, Clone)]
pub struct GuéantLehalleTapiaState {
    /// Portfolio inventory vector
    pub portfolio_inventory: DVector<f64>,
    /// Cross-asset reservation prices
    pub reservation_prices: DVector<f64>,
    /// Optimal spreads vector
    pub optimal_spreads: DVector<f64>,
    /// Portfolio risk measure
    pub portfolio_risk: f64,
    /// Cross-asset correlations (real-time)
    pub realtime_correlations: DMatrix<f64>,
    /// Covariance matrix
    pub covariance_matrix: DMatrix<f64>,
    /// Portfolio utility
    pub portfolio_utility: f64,
    /// Hedge ratios
    pub hedge_ratios: DMatrix<f64>,
    /// Cross-asset momentum signals
    pub momentum_signals: DVector<f64>,
    /// Lead-lag relationships
    pub lead_lag_matrix: DMatrix<f64>,
}

/// Regime types for multi-asset framework
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RegimeType {
    Bull,
    Bear,
    Volatile,
    Calm,
    Crisis,
    Recovery,
}

/// Regime-specific parameters
#[derive(Debug, Clone)]
pub struct RegimeParameters {
    pub correlation_multiplier: f64,
    pub volatility_multiplier: f64,
    pub arrival_intensity_multiplier: f64,
    pub risk_aversion_adjustment: f64,
    pub spread_multiplier: f64,
}

/// Correlation estimator with multiple methods
#[derive(Debug, Clone)]
pub struct CorrelationEstimator {
    /// EWMA decay parameter
    pub lambda: f64,
    /// DCC-GARCH parameters
    pub dcc_parameters: DCCParameters,
    /// Shrinkage parameters
    pub shrinkage_parameters: ShrinkageParameters,
    /// Historical correlation data
    pub historical_correlations: Vec<DMatrix<f64>>,
}

/// DCC-GARCH parameters for dynamic correlation
#[derive(Debug, Clone)]
pub struct DCCParameters {
    pub alpha: f64,
    pub beta: f64,
    pub theta: f64,
}

/// Shrinkage estimation parameters
#[derive(Debug, Clone)]
pub struct ShrinkageParameters {
    pub shrinkage_intensity: f64,
    pub target_correlation: f64,
}

/// Regime detection system
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    /// Hidden Markov Model parameters
    pub hmm_parameters: HMMParameters,
    /// Current regime probabilities
    pub regime_probabilities: HashMap<RegimeType, f64>,
    /// Regime history
    pub regime_history: Vec<(u64, RegimeType)>,
}

/// Hidden Markov Model parameters
#[derive(Debug, Clone)]
pub struct HMMParameters {
    pub transition_matrix: DMatrix<f64>,
    pub emission_parameters: HashMap<RegimeType, EmissionParameters>,
    pub initial_probabilities: DVector<f64>,
}

/// Emission parameters for each regime
#[derive(Debug, Clone)]
pub struct EmissionParameters {
    pub mean_returns: DVector<f64>,
    pub covariance_matrix: DMatrix<f64>,
    pub volatility_parameters: DVector<f64>,
}

/// Portfolio optimizer for multi-asset allocation
#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    /// Optimization method
    pub method: OptimizationMethod,
    /// Constraints
    pub constraints: PortfolioConstraints,
    /// Objective function parameters
    pub objective_parameters: ObjectiveParameters,
}

/// Portfolio optimization methods
#[derive(Debug, Clone)]
pub enum OptimizationMethod {
    MeanVariance,
    BlackLitterman,
    RiskParity,
    MaximumDiversification,
    MinimumVariance,
}

/// Portfolio constraints
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    /// Individual position limits
    pub position_limits: DVector<f64>,
    /// Portfolio leverage limit
    pub leverage_limit: f64,
    /// Concentration limits
    pub concentration_limits: DVector<f64>,
    /// Sector limits
    pub sector_limits: HashMap<String, f64>,
    /// Turnover constraints
    pub turnover_limit: f64,
}

/// Objective function parameters
#[derive(Debug, Clone)]
pub struct ObjectiveParameters {
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Expected returns
    pub expected_returns: DVector<f64>,
    /// Transaction costs
    pub transaction_costs: DMatrix<f64>,
    /// Market impact costs
    pub market_impact_costs: DVector<f64>,
}

impl GuéantLehalleTapiaEngine {
    /// Create new Guéant-Lehalle-Tapia engine
    pub fn new(risk_aversion: f64) -> Result<Self> {
        let asset_universe = vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()];
        let n_assets = asset_universe.len();
        
        // Initialize correlation matrix as identity
        let correlation_matrix = DMatrix::identity(n_assets, n_assets);
        
        // Initialize other matrices
        let cross_asset_risk_aversion = DMatrix::from_diagonal(&DVector::from_element(n_assets, risk_aversion));
        let volatility_vector = DVector::from_element(n_assets, 0.01);
        let arrival_intensity_matrix = DMatrix::from_element(n_assets, n_assets, 1.0);
        let market_impact_parameters = DVector::from_element(n_assets, 0.001);
        let regime_transition_matrix = DMatrix::identity(6, 6); // 6 regimes
        let cointegration_vectors = DMatrix::zeros(n_assets, n_assets);
        let mean_reversion_speeds = DVector::from_element(n_assets, 0.1);
        let cross_impact_matrix = DMatrix::zeros(n_assets, n_assets);
        
        // Initialize regime parameters
        let mut regime_parameters = HashMap::new();
        regime_parameters.insert(RegimeType::Bull, RegimeParameters {
            correlation_multiplier: 0.8,
            volatility_multiplier: 0.9,
            arrival_intensity_multiplier: 1.2,
            risk_aversion_adjustment: 0.9,
            spread_multiplier: 0.8,
        });
        regime_parameters.insert(RegimeType::Bear, RegimeParameters {
            correlation_multiplier: 1.3,
            volatility_multiplier: 1.5,
            arrival_intensity_multiplier: 0.8,
            risk_aversion_adjustment: 1.2,
            spread_multiplier: 1.3,
        });
        regime_parameters.insert(RegimeType::Volatile, RegimeParameters {
            correlation_multiplier: 1.1,
            volatility_multiplier: 2.0,
            arrival_intensity_multiplier: 1.5,
            risk_aversion_adjustment: 1.5,
            spread_multiplier: 1.8,
        });
        regime_parameters.insert(RegimeType::Calm, RegimeParameters {
            correlation_multiplier: 0.7,
            volatility_multiplier: 0.6,
            arrival_intensity_multiplier: 0.9,
            risk_aversion_adjustment: 0.8,
            spread_multiplier: 0.7,
        });
        regime_parameters.insert(RegimeType::Crisis, RegimeParameters {
            correlation_multiplier: 1.8,
            volatility_multiplier: 3.0,
            arrival_intensity_multiplier: 0.5,
            risk_aversion_adjustment: 2.0,
            spread_multiplier: 2.5,
        });
        regime_parameters.insert(RegimeType::Recovery, RegimeParameters {
            correlation_multiplier: 1.0,
            volatility_multiplier: 1.2,
            arrival_intensity_multiplier: 1.1,
            risk_aversion_adjustment: 1.0,
            spread_multiplier: 1.0,
        });
        
        let parameters = GuéantLehalleTapiaParameters {
            cross_asset_risk_aversion,
            correlation_matrix,
            volatility_vector,
            arrival_intensity_matrix,
            market_impact_parameters,
            regime_transition_matrix,
            regime_parameters,
            cointegration_vectors,
            mean_reversion_speeds,
            cross_impact_matrix,
        };
        
        let state = GuéantLehalleTapiaState {
            portfolio_inventory: DVector::zeros(n_assets),
            reservation_prices: DVector::zeros(n_assets),
            optimal_spreads: DVector::zeros(n_assets),
            portfolio_risk: 0.0,
            realtime_correlations: DMatrix::identity(n_assets, n_assets),
            covariance_matrix: DMatrix::identity(n_assets, n_assets),
            portfolio_utility: 0.0,
            hedge_ratios: DMatrix::zeros(n_assets, n_assets),
            momentum_signals: DVector::zeros(n_assets),
            lead_lag_matrix: DMatrix::zeros(n_assets, n_assets),
        };
        
        let correlation_estimator = CorrelationEstimator {
            lambda: 0.94,
            dcc_parameters: DCCParameters {
                alpha: 0.01,
                beta: 0.95,
                theta: 0.01,
            },
            shrinkage_parameters: ShrinkageParameters {
                shrinkage_intensity: 0.1,
                target_correlation: 0.3,
            },
            historical_correlations: Vec::new(),
        };
        
        let regime_detector = RegimeDetector {
            hmm_parameters: HMMParameters {
                transition_matrix: DMatrix::identity(6, 6),
                emission_parameters: HashMap::new(),
                initial_probabilities: DVector::from_element(6, 1.0 / 6.0),
            },
            regime_probabilities: HashMap::new(),
            regime_history: Vec::new(),
        };
        
        let portfolio_optimizer = PortfolioOptimizer {
            method: OptimizationMethod::MeanVariance,
            constraints: PortfolioConstraints {
                position_limits: DVector::from_element(n_assets, 1000.0),
                leverage_limit: 10.0,
                concentration_limits: DVector::from_element(n_assets, 0.3),
                sector_limits: HashMap::new(),
                turnover_limit: 0.5,
            },
            objective_parameters: ObjectiveParameters {
                risk_aversion,
                expected_returns: DVector::zeros(n_assets),
                transaction_costs: DMatrix::from_element(n_assets, n_assets, 0.001),
                market_impact_costs: DVector::from_element(n_assets, 0.001),
            },
        };
        
        Ok(Self {
            risk_aversion,
            parameters,
            state,
            asset_universe,
            correlation_estimator,
            regime_detector,
            portfolio_optimizer,
        })
    }
    
    /// Generate quotes for multi-asset portfolio
    pub fn generate_quotes(&self, symbol: &str, market_state: &MarketMakingState) -> Result<QuoteSet> {
        let asset_index = self.get_asset_index(symbol)?;
        
        // Solve multi-dimensional HJB equation
        let (reservation_prices, optimal_spreads) = self.solve_multi_dimensional_hjb(market_state)?;
        
        // Get reservation price and spread for this asset
        let reservation_price = reservation_prices[asset_index];
        let optimal_spread = optimal_spreads[asset_index];
        
        // Calculate cross-asset adjustments
        let cross_asset_adjustment = self.calculate_cross_asset_adjustment(asset_index, market_state)?;
        
        // Apply regime-dependent adjustments
        let regime_adjustment = self.calculate_regime_adjustment(asset_index, market_state)?;
        
        // Calculate final prices
        let adjusted_reservation = reservation_price + cross_asset_adjustment;
        let adjusted_spread = optimal_spread * regime_adjustment;
        
        let bid_price = adjusted_reservation - adjusted_spread / 2.0;
        let ask_price = adjusted_reservation + adjusted_spread / 2.0;
        
        // Calculate optimal sizes with portfolio constraints
        let (bid_size, ask_size) = self.calculate_portfolio_optimal_sizes(
            asset_index,
            adjusted_spread,
            market_state,
        )?;
        
        // Calculate expected profit considering cross-asset effects
        let expected_profit = self.calculate_cross_asset_expected_profit(
            asset_index,
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            market_state,
        )?;
        
        // Calculate confidence with regime uncertainty
        let confidence = self.calculate_regime_weighted_confidence(asset_index, market_state)?;
        
        Ok(QuoteSet {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            confidence,
            expected_profit,
        })
    }
    
    /// Solve multi-dimensional Hamilton-Jacobi-Bellman equation
    fn solve_multi_dimensional_hjb(&self, market_state: &MarketMakingState) -> Result<(DVector<f64>, DVector<f64>)> {
        let n_assets = self.asset_universe.len();
        let time_remaining = 1.0; // Simplified
        
        // Get current portfolio state
        let inventory_vector = self.get_inventory_vector(market_state);
        let volatility_vector = self.get_volatility_vector(market_state);
        let correlation_matrix = &self.state.realtime_correlations;
        
        // Calculate covariance matrix: Σ = V^(1/2) * R * V^(1/2)
        let vol_diag = DMatrix::from_diagonal(&volatility_vector);
        let covariance_matrix = &vol_diag * correlation_matrix * &vol_diag;
        
        // Cross-asset reservation prices: rᵢ = Sᵢ - Σⱼ γᵢⱼqⱼσⱼ²(T-t)
        let mut reservation_prices = DVector::zeros(n_assets);
        for i in 0..n_assets {
            let mid_price = self.get_mid_price(i, market_state)?;
            let mut cross_asset_penalty = 0.0;
            
            for j in 0..n_assets {
                cross_asset_penalty += self.parameters.cross_asset_risk_aversion[(i, j)]
                    * inventory_vector[j]
                    * volatility_vector[j].powi(2)
                    * time_remaining;
            }
            
            reservation_prices[i] = mid_price - cross_asset_penalty;
        }
        
        // Optimal spreads: δᵢ* = arg max Σᵢ λᵢ(δᵢ)[δᵢ - γᵢᵢδᵢ²/2 - Σⱼ≠ᵢ γᵢⱼqⱼδᵢ]
        let mut optimal_spreads = DVector::zeros(n_assets);
        for i in 0..n_assets {
            let gamma_ii = self.parameters.cross_asset_risk_aversion[(i, i)];
            let lambda_i = self.parameters.arrival_intensity_matrix[(i, i)];
            let k_i = self.parameters.market_impact_parameters[i];
            
            // Base spread calculation
            let base_spread = gamma_ii * volatility_vector[i].powi(2) * time_remaining
                + (2.0 / gamma_ii) * (1.0 + gamma_ii / k_i).ln();
            
            // Cross-asset adjustment
            let mut cross_adjustment = 0.0;
            for j in 0..n_assets {
                if i != j {
                    cross_adjustment += self.parameters.cross_asset_risk_aversion[(i, j)]
                        * inventory_vector[j]
                        * base_spread;
                }
            }
            
            optimal_spreads[i] = (base_spread + cross_adjustment).max(0.0001);
        }
        
        Ok((reservation_prices, optimal_spreads))
    }
    
    /// Calculate cross-asset adjustment for reservation price
    fn calculate_cross_asset_adjustment(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        let mut adjustment = 0.0;
        let inventory_vector = self.get_inventory_vector(market_state);
        let volatility_vector = self.get_volatility_vector(market_state);
        
        // Cross-asset correlation effects
        for j in 0..self.asset_universe.len() {
            if j != asset_index {
                let correlation = self.state.realtime_correlations[(asset_index, j)];
                let cross_gamma = self.parameters.cross_asset_risk_aversion[(asset_index, j)];
                
                adjustment += cross_gamma * inventory_vector[j] * correlation
                    * volatility_vector[asset_index] * volatility_vector[j];
            }
        }
        
        // Cointegration adjustment
        if let Some(cointegration_weight) = self.get_cointegration_weight(asset_index) {
            let spread_deviation = self.calculate_spread_deviation(asset_index, market_state)?;
            let mean_reversion_speed = self.parameters.mean_reversion_speeds[asset_index];
            
            adjustment += cointegration_weight * spread_deviation * mean_reversion_speed;
        }
        
        // Lead-lag adjustment
        let lead_lag_adjustment = self.calculate_lead_lag_adjustment(asset_index, market_state)?;
        adjustment += lead_lag_adjustment;
        
        Ok(adjustment)
    }
    
    /// Calculate regime-dependent adjustment
    fn calculate_regime_adjustment(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        let mut weighted_adjustment = 0.0;
        
        // Weight by regime probabilities
        for (regime, probability) in &self.regime_detector.regime_probabilities {
            if let Some(regime_params) = self.parameters.regime_parameters.get(regime) {
                weighted_adjustment += probability * regime_params.spread_multiplier;
            }
        }
        
        // If no regime probabilities, use current regime
        if weighted_adjustment == 0.0 {
            match market_state.regime_state {
                crate::market_making::RegimeState::Normal { .. } => weighted_adjustment = 1.0,
                crate::market_making::RegimeState::Stressed { volatility_multiplier } => {
                    weighted_adjustment = volatility_multiplier
                }
                crate::market_making::RegimeState::Crisis { .. } => weighted_adjustment = 2.5,
                crate::market_making::RegimeState::Recovery { stabilization_factor } => {
                    weighted_adjustment = stabilization_factor
                }
            }
        }
        
        Ok(weighted_adjustment)
    }
    
    /// Calculate portfolio-optimal sizes
    fn calculate_portfolio_optimal_sizes(
        &self,
        asset_index: usize,
        spread: f64,
        market_state: &MarketMakingState,
    ) -> Result<(f64, f64)> {
        let inventory_vector = self.get_inventory_vector(market_state);
        let current_inventory = inventory_vector[asset_index];
        
        // Portfolio risk contribution
        let risk_contribution = self.calculate_risk_contribution(asset_index, market_state)?;
        
        // Position limit constraint
        let position_limit = self.portfolio_optimizer.constraints.position_limits[asset_index];
        let available_capacity = position_limit - current_inventory.abs();
        
        // Concentration constraint
        let concentration_limit = self.portfolio_optimizer.constraints.concentration_limits[asset_index];
        let total_portfolio_value = self.calculate_total_portfolio_value(market_state)?;
        let max_position_value = total_portfolio_value * concentration_limit;
        
        // Base size from Kelly criterion
        let volatility = self.get_volatility_vector(market_state)[asset_index];
        let kelly_size = self.calculate_multi_asset_kelly_size(asset_index, spread, volatility, market_state)?;
        
        // Apply constraints
        let constrained_size = kelly_size
            .min(available_capacity * 0.1) // Use 10% of available capacity
            .min(max_position_value / self.get_mid_price(asset_index, market_state)?);
        
        // Asymmetric sizing based on portfolio risk
        let risk_adjustment = (-risk_contribution * 0.1).exp();
        
        let bid_size = if current_inventory > 0.0 {
            constrained_size * risk_adjustment * 0.8 // Reduce bid size when long
        } else {
            constrained_size * risk_adjustment
        };
        
        let ask_size = if current_inventory < 0.0 {
            constrained_size * risk_adjustment * 0.8 // Reduce ask size when short
        } else {
            constrained_size * risk_adjustment
        };
        
        Ok((bid_size.max(0.01), ask_size.max(0.01)))
    }
    
    /// Calculate cross-asset expected profit
    fn calculate_cross_asset_expected_profit(
        &self,
        asset_index: usize,
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Direct profit from this asset
        let direct_profit = (ask_price - bid_price) * (bid_size + ask_size) / 2.0;
        
        // Cross-asset profit from hedging
        let hedge_profit = self.calculate_hedge_profit(asset_index, bid_size + ask_size, market_state)?;
        
        // Portfolio diversification benefit
        let diversification_benefit = self.calculate_diversification_benefit(asset_index, market_state)?;
        
        // Cross-impact cost
        let cross_impact_cost = self.calculate_cross_impact_cost(
            asset_index,
            bid_size + ask_size,
            market_state,
        )?;
        
        Ok(direct_profit + hedge_profit + diversification_benefit - cross_impact_cost)
    }
    
    /// Calculate regime-weighted confidence
    fn calculate_regime_weighted_confidence(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        let mut weighted_confidence = 0.0;
        
        for (regime, probability) in &self.regime_detector.regime_probabilities {
            let regime_confidence = match regime {
                RegimeType::Bull | RegimeType::Calm => 0.9,
                RegimeType::Bear | RegimeType::Recovery => 0.8,
                RegimeType::Volatile => 0.7,
                RegimeType::Crisis => 0.5,
            };
            
            weighted_confidence += probability * regime_confidence;
        }
        
        // Adjust for correlation stability
        let correlation_stability = self.calculate_correlation_stability()?;
        weighted_confidence *= correlation_stability;
        
        // Adjust for model complexity
        let model_complexity_penalty = 0.95; // Slight penalty for complex multi-asset model
        weighted_confidence *= model_complexity_penalty;
        
        Ok(weighted_confidence.max(0.1).min(1.0))
    }
    
    /// Optimize portfolio allocation
    pub fn optimize_portfolio(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        match self.portfolio_optimizer.method {
            OptimizationMethod::MeanVariance => self.optimize_mean_variance(market_state),
            OptimizationMethod::BlackLitterman => self.optimize_black_litterman(market_state),
            OptimizationMethod::RiskParity => self.optimize_risk_parity(market_state),
            OptimizationMethod::MaximumDiversification => self.optimize_maximum_diversification(market_state),
            OptimizationMethod::MinimumVariance => self.optimize_minimum_variance(market_state),
        }
    }
    
    /// Calculate hedge orders for portfolio
    pub fn calculate_hedge_orders(&self, market_state: &MarketMakingState) -> Result<Vec<HedgeOrder>> {
        let mut hedge_orders = Vec::new();
        let inventory_vector = self.get_inventory_vector(market_state);
        
        // Calculate minimum variance hedge ratios
        for i in 0..self.asset_universe.len() {
            let inventory_i = inventory_vector[i];
            
            if inventory_i.abs() > 0.01 { // Only hedge significant positions
                // Find best hedge asset
                let mut best_hedge_ratio = 0.0;
                let mut best_hedge_asset = 0;
                let mut best_effectiveness = 0.0;
                
                for j in 0..self.asset_universe.len() {
                    if i != j {
                        let hedge_ratio = self.calculate_minimum_variance_hedge_ratio(i, j)?;
                        let effectiveness = self.calculate_hedge_effectiveness(i, j, hedge_ratio)?;
                        
                        if effectiveness > best_effectiveness {
                            best_hedge_ratio = hedge_ratio;
                            best_hedge_asset = j;
                            best_effectiveness = effectiveness;
                        }
                    }
                }
                
                // Create hedge order if effectiveness is sufficient
                if best_effectiveness > 0.3 {
                    let hedge_quantity = -inventory_i * best_hedge_ratio;
                    let side = if hedge_quantity > 0.0 { TradeSide::Buy } else { TradeSide::Sell };
                    
                    hedge_orders.push(HedgeOrder {
                        symbol: self.asset_universe[best_hedge_asset].clone(),
                        quantity: hedge_quantity.abs(),
                        side,
                        hedge_ratio: best_hedge_ratio,
                        expected_effectiveness: best_effectiveness,
                    });
                }
            }
        }
        
        Ok(hedge_orders)
    }
    
    /// Update inventory for asset
    pub fn update_inventory(&mut self, symbol: &str, new_inventory: f64) -> Result<()> {
        if let Ok(asset_index) = self.get_asset_index(symbol) {
            self.state.portfolio_inventory[asset_index] = new_inventory;
            
            // Recalculate portfolio risk
            self.update_portfolio_risk()?;
            
            // Update hedge ratios
            self.update_hedge_ratios()?;
        }
        
        Ok(())
    }
    
    /// Update correlation matrix with new market data
    pub fn update_correlations(&mut self, returns_matrix: &DMatrix<f64>) -> Result<()> {
        // EWMA correlation update
        let new_correlation = self.calculate_ewma_correlation(returns_matrix)?;
        
        // DCC-GARCH update
        let dcc_correlation = self.calculate_dcc_correlation(returns_matrix)?;
        
        // Shrinkage estimation
        let shrunk_correlation = self.apply_shrinkage_estimation(&new_correlation)?;
        
        // Combine methods with weights
        let combined_correlation = 0.5 * shrunk_correlation + 0.3 * dcc_correlation + 0.2 * new_correlation;
        
        // Ensure positive definiteness
        let regularized_correlation = self.regularize_correlation_matrix(&combined_correlation)?;
        
        self.state.realtime_correlations = regularized_correlation;
        
        // Update covariance matrix
        self.update_covariance_matrix()?;
        
        Ok(())
    }
    
    /// Detect regime changes
    pub fn detect_regime_change(&mut self, market_data: &DMatrix<f64>) -> Result<RegimeType> {
        // Calculate regime indicators
        let volatility_regime = self.calculate_volatility_regime(market_data)?;
        let correlation_regime = self.calculate_correlation_regime(market_data)?;
        let momentum_regime = self.calculate_momentum_regime(market_data)?;
        
        // Update HMM with new observations
        self.update_hmm_probabilities(market_data)?;
        
        // Get most likely regime
        let most_likely_regime = self.regime_detector.regime_probabilities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(regime, _)| regime.clone())
            .unwrap_or(RegimeType::Calm);
        
        // Update regime history
        self.regime_detector.regime_history.push((
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            most_likely_regime.clone(),
        ));
        
        Ok(most_likely_regime)
    }
    
    // Private helper methods
    fn get_asset_index(&self, symbol: &str) -> Result<usize> {
        self.asset_universe
            .iter()
            .position(|s| s == symbol)
            .ok_or_else(|| crate::error::Error::InvalidSymbol(symbol.to_string()))
    }
    
    fn get_inventory_vector(&self, market_state: &MarketMakingState) -> DVector<f64> {
        let mut inventory_vector = DVector::zeros(self.asset_universe.len());
        
        for (i, symbol) in self.asset_universe.iter().enumerate() {
            inventory_vector[i] = market_state.inventory.get(symbol).copied().unwrap_or(0.0);
        }
        
        inventory_vector
    }
    
    fn get_volatility_vector(&self, market_state: &MarketMakingState) -> DVector<f64> {
        let mut volatility_vector = DVector::zeros(self.asset_universe.len());
        
        for (i, symbol) in self.asset_universe.iter().enumerate() {
            volatility_vector[i] = market_state.volatility_estimates.get(symbol).copied().unwrap_or(0.01);
        }
        
        volatility_vector
    }
    
    fn get_mid_price(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        // Placeholder implementation
        Ok(100.0 * (asset_index + 1) as f64)
    }
    
    fn get_cointegration_weight(&self, asset_index: usize) -> Option<f64> {
        // Placeholder implementation
        Some(0.1)
    }
    
    fn calculate_spread_deviation(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        // Placeholder implementation
        Ok(0.0)
    }
    
    fn calculate_lead_lag_adjustment(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        // Placeholder implementation
        Ok(0.0)
    }
    
    fn calculate_risk_contribution(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        let inventory_vector = self.get_inventory_vector(market_state);
        let portfolio_risk = inventory_vector.transpose() * &self.state.covariance_matrix * &inventory_vector;
        
        // Risk contribution: RC_i = q_i * (Σq)_i / sqrt(q^T Σ q)
        let covariance_times_inventory = &self.state.covariance_matrix * &inventory_vector;
        let risk_contribution = inventory_vector[asset_index] * covariance_times_inventory[asset_index] / portfolio_risk.sqrt();
        
        Ok(risk_contribution)
    }
    
    fn calculate_total_portfolio_value(&self, market_state: &MarketMakingState) -> Result<f64> {
        let mut total_value = 0.0;
        
        for (i, symbol) in self.asset_universe.iter().enumerate() {
            let inventory = market_state.inventory.get(symbol).copied().unwrap_or(0.0);
            let price = self.get_mid_price(i, market_state)?;
            total_value += inventory.abs() * price;
        }
        
        Ok(total_value)
    }
    
    fn calculate_multi_asset_kelly_size(
        &self,
        asset_index: usize,
        spread: f64,
        volatility: f64,
        market_state: &MarketMakingState,
    ) -> Result<f64> {
        // Multi-asset Kelly: f* = Σ^(-1)(μ-r)/γ
        let expected_return = spread / 2.0;
        let risk_free_rate = 0.02;
        let excess_return = expected_return - risk_free_rate;
        
        // Use inverse covariance matrix for multi-asset Kelly
        let covariance_inv = self.state.covariance_matrix.try_inverse()
            .ok_or_else(|| crate::error::Error::NumericalError("Singular covariance matrix".to_string()))?;
        
        let kelly_fraction = covariance_inv[(asset_index, asset_index)] * excess_return / self.risk_aversion;
        
        Ok(kelly_fraction.abs().min(0.05) * 1000.0) // Max 5% allocation, scaled to shares
    }
    
    fn calculate_hedge_profit(&self, asset_index: usize, position_size: f64, market_state: &MarketMakingState) -> Result<f64> {
        // Simplified hedge profit calculation
        Ok(position_size * 0.001) // 0.1% hedge profit
    }
    
    fn calculate_diversification_benefit(&self, asset_index: usize, market_state: &MarketMakingState) -> Result<f64> {
        // Diversification benefit from correlation < 1
        let avg_correlation = self.state.realtime_correlations.row(asset_index).mean();
        let diversification_benefit = (1.0 - avg_correlation) * 0.01; // 1% max benefit
        
        Ok(diversification_benefit)
    }
    
    fn calculate_cross_impact_cost(&self, asset_index: usize, position_size: f64, market_state: &MarketMakingState) -> Result<f64> {
        let mut cross_impact_cost = 0.0;
        
        for j in 0..self.asset_universe.len() {
            if j != asset_index {
                let cross_impact = self.parameters.cross_impact_matrix[(asset_index, j)];
                cross_impact_cost += cross_impact * position_size * position_size;
            }
        }
        
        Ok(cross_impact_cost)
    }
    
    fn calculate_correlation_stability(&self) -> Result<f64> {
        if self.correlation_estimator.historical_correlations.len() < 2 {
            return Ok(0.8);
        }
        
        // Calculate stability as inverse of correlation change
        let recent_corr = self.correlation_estimator.historical_correlations.last().unwrap();
        let previous_corr = &self.correlation_estimator.historical_correlations[self.correlation_estimator.historical_correlations.len() - 2];
        
        let correlation_change = (recent_corr - previous_corr).norm();
        let stability = (-correlation_change * 10.0).exp();
        
        Ok(stability.max(0.3).min(1.0))
    }
    
    // Portfolio optimization methods
    fn optimize_mean_variance(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        let mut optimal_weights = HashMap::new();
        
        // Simplified mean-variance optimization
        let expected_returns = &self.portfolio_optimizer.objective_parameters.expected_returns;
        let covariance_matrix = &self.state.covariance_matrix;
        let risk_aversion = self.portfolio_optimizer.objective_parameters.risk_aversion;
        
        // Optimal weights: w* = (1/γ) * Σ^(-1) * μ
        if let Some(covariance_inv) = covariance_matrix.try_inverse() {
            let optimal_weights_vector = (1.0 / risk_aversion) * covariance_inv * expected_returns;
            
            for (i, symbol) in self.asset_universe.iter().enumerate() {
                optimal_weights.insert(symbol.clone(), optimal_weights_vector[i]);
            }
        }
        
        Ok(optimal_weights)
    }
    
    fn optimize_black_litterman(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        // Placeholder for Black-Litterman implementation
        let mut optimal_weights = HashMap::new();
        
        for symbol in &self.asset_universe {
            optimal_weights.insert(symbol.clone(), 1.0 / self.asset_universe.len() as f64);
        }
        
        Ok(optimal_weights)
    }
    
    fn optimize_risk_parity(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        // Risk parity: equal risk contribution from each asset
        let mut optimal_weights = HashMap::new();
        let volatility_vector = self.get_volatility_vector(market_state);
        
        // Inverse volatility weighting as approximation
        let inv_vol_sum: f64 = volatility_vector.iter().map(|v| 1.0 / v).sum();
        
        for (i, symbol) in self.asset_universe.iter().enumerate() {
            let weight = (1.0 / volatility_vector[i]) / inv_vol_sum;
            optimal_weights.insert(symbol.clone(), weight);
        }
        
        Ok(optimal_weights)
    }
    
    fn optimize_maximum_diversification(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        // Maximum diversification: maximize diversification ratio
        let mut optimal_weights = HashMap::new();
        
        // Simplified: equal weights
        for symbol in &self.asset_universe {
            optimal_weights.insert(symbol.clone(), 1.0 / self.asset_universe.len() as f64);
        }
        
        Ok(optimal_weights)
    }
    
    fn optimize_minimum_variance(&self, market_state: &MarketMakingState) -> Result<HashMap<String, f64>> {
        // Minimum variance: minimize portfolio variance
        let mut optimal_weights = HashMap::new();
        let covariance_matrix = &self.state.covariance_matrix;
        
        // Optimal weights: w* = Σ^(-1) * 1 / (1^T * Σ^(-1) * 1)
        if let Some(covariance_inv) = covariance_matrix.try_inverse() {
            let ones = DVector::from_element(self.asset_universe.len(), 1.0);
            let numerator = covariance_inv * &ones;
            let denominator = ones.transpose() * &numerator;
            let optimal_weights_vector = numerator / denominator[0];
            
            for (i, symbol) in self.asset_universe.iter().enumerate() {
                optimal_weights.insert(symbol.clone(), optimal_weights_vector[i]);
            }
        }
        
        Ok(optimal_weights)
    }
    
    fn calculate_minimum_variance_hedge_ratio(&self, asset_i: usize, asset_j: usize) -> Result<f64> {
        // h* = Cov(S_i, S_j) / Var(S_j)
        let covariance_ij = self.state.covariance_matrix[(asset_i, asset_j)];
        let variance_j = self.state.covariance_matrix[(asset_j, asset_j)];
        
        Ok(covariance_ij / variance_j)
    }
    
    fn calculate_hedge_effectiveness(&self, asset_i: usize, asset_j: usize, hedge_ratio: f64) -> Result<f64> {
        // R² = 1 - Var(hedged) / Var(unhedged)
        let variance_i = self.state.covariance_matrix[(asset_i, asset_i)];
        let variance_j = self.state.covariance_matrix[(asset_j, asset_j)];
        let covariance_ij = self.state.covariance_matrix[(asset_i, asset_j)];
        
        let hedged_variance = variance_i + hedge_ratio.powi(2) * variance_j - 2.0 * hedge_ratio * covariance_ij;
        let effectiveness = 1.0 - hedged_variance / variance_i;
        
        Ok(effectiveness.max(0.0).min(1.0))
    }
    
    fn update_portfolio_risk(&mut self) -> Result<()> {
        let portfolio_variance = self.state.portfolio_inventory.transpose() 
            * &self.state.covariance_matrix 
            * &self.state.portfolio_inventory;
        
        self.state.portfolio_risk = portfolio_variance.sqrt();
        
        Ok(())
    }
    
    fn update_hedge_ratios(&mut self) -> Result<()> {
        let n_assets = self.asset_universe.len();
        let mut hedge_ratios = DMatrix::zeros(n_assets, n_assets);
        
        for i in 0..n_assets {
            for j in 0..n_assets {
                if i != j {
                    hedge_ratios[(i, j)] = self.calculate_minimum_variance_hedge_ratio(i, j)?;
                }
            }
        }
        
        self.state.hedge_ratios = hedge_ratios;
        
        Ok(())
    }
    
    fn calculate_ewma_correlation(&self, returns_matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let n_assets = returns_matrix.ncols();
        let mut correlation_matrix = DMatrix::identity(n_assets, n_assets);
        
        // EWMA correlation calculation
        let lambda = self.correlation_estimator.lambda;
        
        for i in 0..n_assets {
            for j in i+1..n_assets {
                let returns_i = returns_matrix.column(i);
                let returns_j = returns_matrix.column(j);
                
                // Calculate EWMA correlation
                let mut ewma_cov = 0.0;
                let mut ewma_var_i = 0.0;
                let mut ewma_var_j = 0.0;
                
                for k in 0..returns_i.len() {
                    let weight = lambda.powi(returns_i.len() - k - 1);
                    ewma_cov += weight * returns_i[k] * returns_j[k];
                    ewma_var_i += weight * returns_i[k].powi(2);
                    ewma_var_j += weight * returns_j[k].powi(2);
                }
                
                let correlation = ewma_cov / (ewma_var_i.sqrt() * ewma_var_j.sqrt());
                correlation_matrix[(i, j)] = correlation;
                correlation_matrix[(j, i)] = correlation;
            }
        }
        
        Ok(correlation_matrix)
    }
    
    fn calculate_dcc_correlation(&self, returns_matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Simplified DCC-GARCH implementation
        let n_assets = returns_matrix.ncols();
        let correlation_matrix = DMatrix::identity(n_assets, n_assets);
        
        // Full DCC-GARCH implementation would be more complex
        Ok(correlation_matrix)
    }
    
    fn apply_shrinkage_estimation(&self, sample_correlation: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let shrinkage_intensity = self.correlation_estimator.shrinkage_parameters.shrinkage_intensity;
        let target_correlation = self.correlation_estimator.shrinkage_parameters.target_correlation;
        
        let n_assets = sample_correlation.nrows();
        let mut target_matrix = DMatrix::from_element(n_assets, n_assets, target_correlation);
        
        // Set diagonal to 1
        for i in 0..n_assets {
            target_matrix[(i, i)] = 1.0;
        }
        
        // Shrinkage: Σ̂ = α * Σ_sample + (1-α) * Σ_target
        let shrunk_correlation = (1.0 - shrinkage_intensity) * sample_correlation + shrinkage_intensity * target_matrix;
        
        Ok(shrunk_correlation)
    }
    
    fn regularize_correlation_matrix(&self, correlation_matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Ensure positive definiteness using eigenvalue regularization
        let eigen = correlation_matrix.symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues;
        
        // Set minimum eigenvalue
        let min_eigenvalue = 0.001;
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] < min_eigenvalue {
                eigenvalues[i] = min_eigenvalue;
            }
        }
        
        // Reconstruct matrix
        let regularized_matrix = eigen.eigenvectors * DMatrix::from_diagonal(&eigenvalues) * eigen.eigenvectors.transpose();
        
        Ok(regularized_matrix)
    }
    
    fn update_covariance_matrix(&mut self) -> Result<()> {
        // Update covariance matrix from correlation and volatilities
        let volatility_vector = &self.parameters.volatility_vector;
        let vol_diag = DMatrix::from_diagonal(volatility_vector);
        
        self.state.covariance_matrix = &vol_diag * &self.state.realtime_correlations * &vol_diag;
        
        Ok(())
    }
    
    fn calculate_volatility_regime(&self, market_data: &DMatrix<f64>) -> Result<RegimeType> {
        // Calculate average volatility
        let mut total_volatility = 0.0;
        for col in market_data.column_iter() {
            let variance = col.variance();
            total_volatility += variance.sqrt();
        }
        let avg_volatility = total_volatility / market_data.ncols() as f64;
        
        // Classify regime based on volatility
        if avg_volatility < 0.01 {
            Ok(RegimeType::Calm)
        } else if avg_volatility < 0.02 {
            Ok(RegimeType::Bull)
        } else if avg_volatility < 0.05 {
            Ok(RegimeType::Volatile)
        } else {
            Ok(RegimeType::Crisis)
        }
    }
    
    fn calculate_correlation_regime(&self, market_data: &DMatrix<f64>) -> Result<RegimeType> {
        // Calculate average correlation
        let correlation_matrix = self.calculate_ewma_correlation(market_data)?;
        let mut total_correlation = 0.0;
        let mut count = 0;
        
        for i in 0..correlation_matrix.nrows() {
            for j in i+1..correlation_matrix.ncols() {
                total_correlation += correlation_matrix[(i, j)].abs();
                count += 1;
            }
        }
        
        let avg_correlation = if count > 0 { total_correlation / count as f64 } else { 0.0 };
        
        // Classify regime based on correlation
        if avg_correlation > 0.8 {
            Ok(RegimeType::Crisis)
        } else if avg_correlation > 0.6 {
            Ok(RegimeType::Bear)
        } else if avg_correlation > 0.4 {
            Ok(RegimeType::Volatile)
        } else {
            Ok(RegimeType::Bull)
        }
    }
    
    fn calculate_momentum_regime(&self, market_data: &DMatrix<f64>) -> Result<RegimeType> {
        // Calculate momentum indicators
        let mut positive_momentum_count = 0;
        let mut total_assets = 0;
        
        for col in market_data.column_iter() {
            if col.len() > 1 {
                let recent_return = col[col.len() - 1];
                if recent_return > 0.0 {
                    positive_momentum_count += 1;
                }
                total_assets += 1;
            }
        }
        
        let positive_ratio = if total_assets > 0 {
            positive_momentum_count as f64 / total_assets as f64
        } else {
            0.5
        };
        
        // Classify regime based on momentum
        if positive_ratio > 0.7 {
            Ok(RegimeType::Bull)
        } else if positive_ratio < 0.3 {
            Ok(RegimeType::Bear)
        } else {
            Ok(RegimeType::Calm)
        }
    }
    
    fn update_hmm_probabilities(&mut self, market_data: &DMatrix<f64>) -> Result<()> {
        // Simplified HMM update
        // Full implementation would use forward-backward algorithm
        
        // Update regime probabilities based on current observations
        let volatility_regime = self.calculate_volatility_regime(market_data)?;
        let correlation_regime = self.calculate_correlation_regime(market_data)?;
        let momentum_regime = self.calculate_momentum_regime(market_data)?;
        
        // Simple voting mechanism
        let mut regime_votes: HashMap<RegimeType, f64> = HashMap::new();
        *regime_votes.entry(volatility_regime).or_insert(0.0) += 1.0;
        *regime_votes.entry(correlation_regime).or_insert(0.0) += 1.0;
        *regime_votes.entry(momentum_regime).or_insert(0.0) += 1.0;
        
        // Normalize to probabilities
        let total_votes: f64 = regime_votes.values().sum();
        for (regime, votes) in regime_votes {
            self.regime_detector.regime_probabilities.insert(regime, votes / total_votes);
        }
        
        Ok(())
    }
}