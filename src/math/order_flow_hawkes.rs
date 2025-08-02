//! Order Flow Modeling with Hawkes Processes
//! 
//! This module implements specialized Hawkes process models for order flow
//! analysis, including buy/sell intensity coupling, market impact modeling,
//! and real-time intensity forecasting for quote optimization.

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use crate::math::hawkes_process::{
    HawkesEvent, MultivariateHawkesParams, MultivariateHawkesSimulator, KernelType, HawkesError
};
use serde::{Deserialize, Serialize};
use std::collections::{VecDeque, HashMap};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OrderFlowError {
    #[error("Invalid order flow parameters: {0}")]
    InvalidParameters(String),
    #[error("Insufficient order data: {0}")]
    InsufficientData(String),
    #[error("Forecasting error: {0}")]
    ForecastingError(String),
    #[error("Market impact calculation error: {0}")]
    MarketImpactError(String),
}

/// Order types for Hawkes process modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Buy = 0,
    Sell = 1,
    MarketBuy = 2,
    MarketSell = 3,
}

impl OrderType {
    pub fn is_buy(&self) -> bool {
        matches!(self, OrderType::Buy | OrderType::MarketBuy)
    }
    
    pub fn is_sell(&self) -> bool {
        matches!(self, OrderType::Sell | OrderType::MarketSell)
    }
    
    pub fn is_market_order(&self) -> bool {
        matches!(self, OrderType::MarketBuy | OrderType::MarketSell)
    }
}

/// Order flow event with additional market microstructure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowEvent {
    pub base_event: HawkesEvent,
    pub order_type: OrderType,
    pub volume: FixedPoint,
    pub price: FixedPoint,
    pub spread: FixedPoint,
    pub order_book_imbalance: FixedPoint,
    pub mid_price_change: FixedPoint,
}

impl OrderFlowEvent {
    pub fn new(
        time: FixedPoint,
        order_type: OrderType,
        volume: FixedPoint,
        price: FixedPoint,
        spread: FixedPoint,
        order_book_imbalance: FixedPoint,
        mid_price_change: FixedPoint,
    ) -> Self {
        let process_id = order_type as usize;
        Self {
            base_event: HawkesEvent {
                time,
                process_id,
                mark: Some(volume),
            },
            order_type,
            volume,
            price,
            spread,
            order_book_imbalance,
            mid_price_change,
        }
    }
}

/// Parameters for order flow Hawkes model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowHawkesParams {
    /// Base Hawkes parameters
    pub hawkes_params: MultivariateHawkesParams,
    /// Market impact decay parameter
    pub impact_decay: FixedPoint,
    /// Cross-excitation strength between buy/sell orders
    pub cross_excitation: FixedPoint,
    /// Self-excitation strength for same-side orders
    pub self_excitation: FixedPoint,
    /// Volume impact scaling factor
    pub volume_impact_scale: FixedPoint,
    /// Spread impact factor
    pub spread_impact_factor: FixedPoint,
}

impl OrderFlowHawkesParams {
    pub fn new(
        baseline_buy_intensity: FixedPoint,
        baseline_sell_intensity: FixedPoint,
        self_excitation: FixedPoint,
        cross_excitation: FixedPoint,
        decay_rate: FixedPoint,
        impact_decay: FixedPoint,
        volume_impact_scale: FixedPoint,
        spread_impact_factor: FixedPoint,
    ) -> Result<Self, OrderFlowError> {
        // Create 4x4 kernel matrix for [Buy, Sell, MarketBuy, MarketSell]
        let kernels = vec![
            vec![
                // Buy -> Buy (self-excitation)
                KernelType::Exponential { 
                    alpha: self_excitation, 
                    beta: decay_rate 
                },
                // Buy -> Sell (cross-excitation, typically negative)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.5), 
                    beta: decay_rate 
                },
                // Buy -> MarketBuy (positive feedback)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.3), 
                    beta: decay_rate * FixedPoint::from_float(2.0) 
                },
                // Buy -> MarketSell (negative feedback)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.2), 
                    beta: decay_rate * FixedPoint::from_float(2.0) 
                },
            ],
            vec![
                // Sell -> Buy (cross-excitation, typically negative)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.5), 
                    beta: decay_rate 
                },
                // Sell -> Sell (self-excitation)
                KernelType::Exponential { 
                    alpha: self_excitation, 
                    beta: decay_rate 
                },
                // Sell -> MarketBuy (negative feedback)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.2), 
                    beta: decay_rate * FixedPoint::from_float(2.0) 
                },
                // Sell -> MarketSell (positive feedback)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.3), 
                    beta: decay_rate * FixedPoint::from_float(2.0) 
                },
            ],
            vec![
                // MarketBuy -> Buy (positive feedback)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.4), 
                    beta: decay_rate * FixedPoint::from_float(3.0) 
                },
                // MarketBuy -> Sell (strong negative feedback)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.8), 
                    beta: decay_rate * FixedPoint::from_float(1.5) 
                },
                // MarketBuy -> MarketBuy (clustering)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.6), 
                    beta: decay_rate * FixedPoint::from_float(4.0) 
                },
                // MarketBuy -> MarketSell (immediate reversal)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.3), 
                    beta: decay_rate * FixedPoint::from_float(5.0) 
                },
            ],
            vec![
                // MarketSell -> Buy (strong negative feedback)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.8), 
                    beta: decay_rate * FixedPoint::from_float(1.5) 
                },
                // MarketSell -> Sell (positive feedback)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.4), 
                    beta: decay_rate * FixedPoint::from_float(3.0) 
                },
                // MarketSell -> MarketBuy (immediate reversal)
                KernelType::Exponential { 
                    alpha: cross_excitation * FixedPoint::from_float(-0.3), 
                    beta: decay_rate * FixedPoint::from_float(5.0) 
                },
                // MarketSell -> MarketSell (clustering)
                KernelType::Exponential { 
                    alpha: self_excitation * FixedPoint::from_float(0.6), 
                    beta: decay_rate * FixedPoint::from_float(4.0) 
                },
            ],
        ];
        
        let baseline_intensities = vec![
            baseline_buy_intensity,
            baseline_sell_intensity,
            baseline_buy_intensity * FixedPoint::from_float(0.1), // Market orders less frequent
            baseline_sell_intensity * FixedPoint::from_float(0.1),
        ];
        
        let max_intensity = (baseline_buy_intensity + baseline_sell_intensity) * FixedPoint::from_float(10.0);
        
        let hawkes_params = MultivariateHawkesParams::new(
            baseline_intensities,
            kernels,
            max_intensity,
        ).map_err(|e| OrderFlowError::InvalidParameters(e.to_string()))?;
        
        Ok(Self {
            hawkes_params,
            impact_decay,
            cross_excitation,
            self_excitation,
            volume_impact_scale,
            spread_impact_factor,
        })
    }
}

/// Order flow intensity calculator with market impact feedback
pub struct OrderFlowIntensityCalculator {
    pub params: OrderFlowHawkesParams,
    pub event_history: VecDeque<OrderFlowEvent>,
    pub current_intensities: Vec<FixedPoint>,
    pub market_impact_state: MarketImpactState,
    pub max_history: usize,
}

#[derive(Debug, Clone)]
pub struct MarketImpactState {
    pub cumulative_buy_impact: FixedPoint,
    pub cumulative_sell_impact: FixedPoint,
    pub recent_volume_imbalance: FixedPoint,
    pub spread_pressure: FixedPoint,
    pub last_update_time: FixedPoint,
}

impl MarketImpactState {
    pub fn new() -> Self {
        Self {
            cumulative_buy_impact: FixedPoint::zero(),
            cumulative_sell_impact: FixedPoint::zero(),
            recent_volume_imbalance: FixedPoint::zero(),
            spread_pressure: FixedPoint::zero(),
            last_update_time: FixedPoint::zero(),
        }
    }
    
    pub fn update_impact(
        &mut self,
        event: &OrderFlowEvent,
        params: &OrderFlowHawkesParams,
    ) {
        let dt = event.base_event.time - self.last_update_time;
        
        // Decay existing impacts
        let decay_factor = (-params.impact_decay * dt).exp();
        self.cumulative_buy_impact = self.cumulative_buy_impact * decay_factor;
        self.cumulative_sell_impact = self.cumulative_sell_impact * decay_factor;
        self.recent_volume_imbalance = self.recent_volume_imbalance * decay_factor;
        self.spread_pressure = self.spread_pressure * decay_factor;
        
        // Add new impact
        let volume_impact = event.volume * params.volume_impact_scale;
        let spread_impact = event.spread * params.spread_impact_factor;
        
        match event.order_type {
            OrderType::Buy | OrderType::MarketBuy => {
                self.cumulative_buy_impact = self.cumulative_buy_impact + volume_impact;
                self.recent_volume_imbalance = self.recent_volume_imbalance + event.volume;
            }
            OrderType::Sell | OrderType::MarketSell => {
                self.cumulative_sell_impact = self.cumulative_sell_impact + volume_impact;
                self.recent_volume_imbalance = self.recent_volume_imbalance - event.volume;
            }
        }
        
        // Update spread pressure based on order book imbalance
        self.spread_pressure = self.spread_pressure + 
            event.order_book_imbalance * spread_impact;
        
        self.last_update_time = event.base_event.time;
    }
    
    pub fn get_intensity_adjustment(&self, order_type: OrderType) -> FixedPoint {
        let base_adjustment = match order_type {
            OrderType::Buy => {
                // Higher sell impact reduces buy intensity
                -self.cumulative_sell_impact * FixedPoint::from_float(0.5) +
                // Positive volume imbalance increases buy intensity
                self.recent_volume_imbalance * FixedPoint::from_float(0.3)
            }
            OrderType::Sell => {
                // Higher buy impact reduces sell intensity
                -self.cumulative_buy_impact * FixedPoint::from_float(0.5) -
                // Positive volume imbalance decreases sell intensity
                self.recent_volume_imbalance * FixedPoint::from_float(0.3)
            }
            OrderType::MarketBuy => {
                // Market orders more sensitive to spread pressure
                -self.spread_pressure * FixedPoint::from_float(0.8) +
                self.recent_volume_imbalance * FixedPoint::from_float(0.2)
            }
            OrderType::MarketSell => {
                // Market orders more sensitive to spread pressure
                -self.spread_pressure * FixedPoint::from_float(0.8) -
                self.recent_volume_imbalance * FixedPoint::from_float(0.2)
            }
        };
        
        base_adjustment
    }
}

impl OrderFlowIntensityCalculator {
    pub fn new(params: OrderFlowHawkesParams, max_history: usize) -> Self {
        let current_intensities = params.hawkes_params.baseline_intensities.clone();
        
        Self {
            params,
            event_history: VecDeque::with_capacity(max_history),
            current_intensities,
            market_impact_state: MarketImpactState::new(),
            max_history,
        }
    }
    
    /// Add new order flow event and update intensities
    pub fn add_event(&mut self, event: OrderFlowEvent) -> Result<(), OrderFlowError> {
        // Update market impact state
        self.market_impact_state.update_impact(&event, &self.params);
        
        // Add to history
        self.event_history.push_back(event);
        
        // Maintain history size
        if self.event_history.len() > self.max_history {
            self.event_history.pop_front();
        }
        
        // Update intensities
        self.update_intensities()?;
        
        Ok(())
    }
    
    /// Update current intensities based on event history and market impact
    pub fn update_intensities(&mut self) -> Result<(), OrderFlowError> {
        let n = self.params.hawkes_params.dimension();
        
        if let Some(last_event) = self.event_history.back() {
            let current_time = last_event.base_event.time;
            
            for i in 0..n {
                let order_type = match i {
                    0 => OrderType::Buy,
                    1 => OrderType::Sell,
                    2 => OrderType::MarketBuy,
                    3 => OrderType::MarketSell,
                    _ => return Err(OrderFlowError::InvalidParameters(
                        "Invalid process index".to_string()
                    )),
                };
                
                // Start with baseline intensity
                let mut intensity = self.params.hawkes_params.baseline_intensities[i];
                
                // Add Hawkes process contributions
                for event in &self.event_history {
                    let dt = current_time - event.base_event.time;
                    if dt.to_float() > 0.0 {
                        let j = event.base_event.process_id;
                        let kernel_value = self.params.hawkes_params.kernels[i][j].evaluate(dt);
                        
                        // Scale by volume if available
                        let volume_scale = event.volume / FixedPoint::from_float(100.0); // Normalize
                        intensity = intensity + kernel_value * volume_scale;
                    }
                }
                
                // Add market impact adjustment
                let impact_adjustment = self.market_impact_state.get_intensity_adjustment(order_type);
                intensity = intensity + impact_adjustment;
                
                // Ensure non-negative intensity
                if intensity.to_float() < 0.0 {
                    intensity = FixedPoint::from_float(0.001);
                }
                
                self.current_intensities[i] = intensity;
            }
        }
        
        Ok(())
    }
    
    /// Get current intensity for specific order type
    pub fn get_intensity(&self, order_type: OrderType) -> FixedPoint {
        let index = order_type as usize;
        if index < self.current_intensities.len() {
            self.current_intensities[index]
        } else {
            FixedPoint::zero()
        }
    }
    
    /// Get buy/sell intensity ratio
    pub fn get_buy_sell_ratio(&self) -> FixedPoint {
        let buy_intensity = self.get_intensity(OrderType::Buy) + 
                           self.get_intensity(OrderType::MarketBuy);
        let sell_intensity = self.get_intensity(OrderType::Sell) + 
                            self.get_intensity(OrderType::MarketSell);
        
        if sell_intensity.to_float() > 0.0 {
            buy_intensity / sell_intensity
        } else {
            FixedPoint::from_float(f64::INFINITY)
        }
    }
    
    /// Forecast intensity for next time period
    pub fn forecast_intensity(
        &self,
        order_type: OrderType,
        forecast_horizon: FixedPoint,
    ) -> Result<FixedPoint, OrderFlowError> {
        if self.event_history.is_empty() {
            return Ok(self.params.hawkes_params.baseline_intensities[order_type as usize]);
        }
        
        let current_time = self.event_history.back().unwrap().base_event.time;
        let future_time = current_time + forecast_horizon;
        
        let i = order_type as usize;
        let mut forecasted_intensity = self.params.hawkes_params.baseline_intensities[i];
        
        // Add contributions from existing events
        for event in &self.event_history {
            let dt = future_time - event.base_event.time;
            if dt.to_float() > 0.0 {
                let j = event.base_event.process_id;
                let kernel_value = self.params.hawkes_params.kernels[i][j].evaluate(dt);
                
                // Scale by volume
                let volume_scale = event.volume / FixedPoint::from_float(100.0);
                forecasted_intensity = forecasted_intensity + kernel_value * volume_scale;
            }
        }
        
        // Add decayed market impact
        let decay_factor = (-self.params.impact_decay * forecast_horizon).exp();
        let impact_adjustment = self.market_impact_state.get_intensity_adjustment(order_type) * decay_factor;
        forecasted_intensity = forecasted_intensity + impact_adjustment;
        
        // Ensure non-negative
        if forecasted_intensity.to_float() < 0.0 {
            forecasted_intensity = FixedPoint::from_float(0.001);
        }
        
        Ok(forecasted_intensity)
    }
    
    /// Calculate optimal quote adjustment based on intensity forecast
    pub fn calculate_quote_adjustment(
        &self,
        forecast_horizon: FixedPoint,
        risk_aversion: FixedPoint,
    ) -> Result<(FixedPoint, FixedPoint), OrderFlowError> {
        let buy_forecast = self.forecast_intensity(OrderType::Buy, forecast_horizon)?;
        let sell_forecast = self.forecast_intensity(OrderType::Sell, forecast_horizon)?;
        let market_buy_forecast = self.forecast_intensity(OrderType::MarketBuy, forecast_horizon)?;
        let market_sell_forecast = self.forecast_intensity(OrderType::MarketSell, forecast_horizon)?;
        
        // Total buy and sell pressures
        let total_buy_pressure = buy_forecast + market_buy_forecast;
        let total_sell_pressure = sell_forecast + market_sell_forecast;
        
        // Calculate imbalance
        let total_pressure = total_buy_pressure + total_sell_pressure;
        let imbalance = if total_pressure.to_float() > 0.0 {
            (total_buy_pressure - total_sell_pressure) / total_pressure
        } else {
            FixedPoint::zero()
        };
        
        // Adjust quotes based on imbalance and risk aversion
        let base_adjustment = imbalance * risk_aversion * forecast_horizon;
        
        // Bid adjustment (negative imbalance widens bid)
        let bid_adjustment = -base_adjustment * FixedPoint::from_float(0.5);
        
        // Ask adjustment (positive imbalance widens ask)
        let ask_adjustment = base_adjustment * FixedPoint::from_float(0.5);
        
        Ok((bid_adjustment, ask_adjustment))
    }
    
    /// Calculate advanced market impact based on order flow intensity
    pub fn calculate_market_impact(
        &self,
        order_type: OrderType,
        volume: FixedPoint,
        forecast_horizon: FixedPoint,
    ) -> Result<FixedPoint, OrderFlowError> {
        // Get current and forecasted intensities
        let current_intensity = self.get_intensity(order_type);
        let forecasted_intensity = self.forecast_intensity(order_type, forecast_horizon)?;
        
        // Calculate intensity acceleration
        let intensity_change = forecasted_intensity - current_intensity;
        let acceleration_factor = if current_intensity.to_float() > 0.0 {
            intensity_change / current_intensity
        } else {
            FixedPoint::zero()
        };
        
        // Base impact proportional to volume and current intensity
        let base_impact = volume * current_intensity * self.params.volume_impact_scale;
        
        // Acceleration impact (higher when intensity is increasing)
        let acceleration_impact = base_impact * acceleration_factor * FixedPoint::from_float(0.3);
        
        // Cross-impact from opposite side
        let opposite_type = match order_type {
            OrderType::Buy => OrderType::Sell,
            OrderType::Sell => OrderType::Buy,
            OrderType::MarketBuy => OrderType::MarketSell,
            OrderType::MarketSell => OrderType::MarketBuy,
        };
        
        let opposite_intensity = self.get_intensity(opposite_type);
        let cross_impact = volume * opposite_intensity * self.params.cross_excitation * 
                          FixedPoint::from_float(0.2);
        
        // Market impact state adjustment
        let state_impact = match order_type {
            OrderType::Buy | OrderType::MarketBuy => {
                self.market_impact_state.cumulative_sell_impact * FixedPoint::from_float(0.1)
            }
            OrderType::Sell | OrderType::MarketSell => {
                self.market_impact_state.cumulative_buy_impact * FixedPoint::from_float(0.1)
            }
        };
        
        let total_impact = base_impact + acceleration_impact + cross_impact + state_impact;
        
        Ok(total_impact)
    }
    
    /// Get real-time order flow metrics for quote optimization
    pub fn get_real_time_metrics(&self) -> OrderFlowMetrics {
        let buy_intensity = self.get_intensity(OrderType::Buy);
        let sell_intensity = self.get_intensity(OrderType::Sell);
        let market_buy_intensity = self.get_intensity(OrderType::MarketBuy);
        let market_sell_intensity = self.get_intensity(OrderType::MarketSell);
        
        // Calculate various metrics
        let total_intensity = buy_intensity + sell_intensity + market_buy_intensity + market_sell_intensity;
        let buy_sell_imbalance = if total_intensity.to_float() > 0.0 {
            ((buy_intensity + market_buy_intensity) - (sell_intensity + market_sell_intensity)) / total_intensity
        } else {
            FixedPoint::zero()
        };
        
        let market_order_pressure = if total_intensity.to_float() > 0.0 {
            (market_buy_intensity + market_sell_intensity) / total_intensity
        } else {
            FixedPoint::zero()
        };
        
        // Calculate intensity volatility
        let intensities = vec![buy_intensity, sell_intensity, market_buy_intensity, market_sell_intensity];
        let mean_intensity = intensities.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                           FixedPoint::from_float(intensities.len() as f64);
        
        let variance = intensities.iter()
            .map(|&x| {
                let diff = x - mean_intensity;
                diff * diff
            })
            .fold(FixedPoint::zero(), |acc, x| acc + x) / 
            FixedPoint::from_float(intensities.len() as f64);
        
        let intensity_volatility = variance.sqrt();
        
        OrderFlowMetrics {
            buy_intensity,
            sell_intensity,
            market_buy_intensity,
            market_sell_intensity,
            buy_sell_imbalance,
            market_order_pressure,
            intensity_volatility,
            cumulative_buy_impact: self.market_impact_state.cumulative_buy_impact,
            cumulative_sell_impact: self.market_impact_state.cumulative_sell_impact,
            volume_imbalance: self.market_impact_state.recent_volume_imbalance,
            spread_pressure: self.market_impact_state.spread_pressure,
        }
    }
}

/// Real-time order flow analyzer
pub struct RealTimeOrderFlowAnalyzer {
    pub intensity_calculator: OrderFlowIntensityCalculator,
    pub update_frequency: usize,
    pub event_count: usize,
    pub last_analysis_time: FixedPoint,
    pub analysis_results: OrderFlowAnalysisResults,
}

#[derive(Debug, Clone, Default)]
pub struct OrderFlowAnalysisResults {
    pub buy_sell_ratio: FixedPoint,
    pub market_order_ratio: FixedPoint,
    pub intensity_trend: IntensityTrend,
    pub volume_weighted_spread: FixedPoint,
    pub order_flow_toxicity: FixedPoint,
    pub clustering_coefficient: FixedPoint,
}

/// Real-time order flow metrics for quote optimization
#[derive(Debug, Clone)]
pub struct OrderFlowMetrics {
    pub buy_intensity: FixedPoint,
    pub sell_intensity: FixedPoint,
    pub market_buy_intensity: FixedPoint,
    pub market_sell_intensity: FixedPoint,
    pub buy_sell_imbalance: FixedPoint,
    pub market_order_pressure: FixedPoint,
    pub intensity_volatility: FixedPoint,
    pub cumulative_buy_impact: FixedPoint,
    pub cumulative_sell_impact: FixedPoint,
    pub volume_imbalance: FixedPoint,
    pub spread_pressure: FixedPoint,
}

#[derive(Debug, Clone, Default)]
pub enum IntensityTrend {
    #[default]
    Stable,
    Increasing,
    Decreasing,
    Volatile,
}

impl RealTimeOrderFlowAnalyzer {
    pub fn new(
        params: OrderFlowHawkesParams,
        max_history: usize,
        update_frequency: usize,
    ) -> Self {
        Self {
            intensity_calculator: OrderFlowIntensityCalculator::new(params, max_history),
            update_frequency,
            event_count: 0,
            last_analysis_time: FixedPoint::zero(),
            analysis_results: OrderFlowAnalysisResults::default(),
        }
    }
    
    /// Process new order flow event
    pub fn process_event(&mut self, event: OrderFlowEvent) -> Result<bool, OrderFlowError> {
        self.intensity_calculator.add_event(event.clone())?;
        self.event_count += 1;
        
        // Update analysis if enough events have passed
        if self.event_count % self.update_frequency == 0 {
            self.update_analysis(event.base_event.time)?;
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Update comprehensive order flow analysis
    fn update_analysis(&mut self, current_time: FixedPoint) -> Result<(), OrderFlowError> {
        // Calculate buy/sell ratio
        self.analysis_results.buy_sell_ratio = self.intensity_calculator.get_buy_sell_ratio();
        
        // Calculate market order ratio
        let market_buy = self.intensity_calculator.get_intensity(OrderType::MarketBuy);
        let market_sell = self.intensity_calculator.get_intensity(OrderType::MarketSell);
        let limit_buy = self.intensity_calculator.get_intensity(OrderType::Buy);
        let limit_sell = self.intensity_calculator.get_intensity(OrderType::Sell);
        
        let total_market = market_buy + market_sell;
        let total_limit = limit_buy + limit_sell;
        let total_intensity = total_market + total_limit;
        
        self.analysis_results.market_order_ratio = if total_intensity.to_float() > 0.0 {
            total_market / total_intensity
        } else {
            FixedPoint::zero()
        };
        
        // Analyze intensity trend
        self.analysis_results.intensity_trend = self.analyze_intensity_trend(current_time)?;
        
        // Calculate volume-weighted spread
        self.analysis_results.volume_weighted_spread = self.calculate_volume_weighted_spread();
        
        // Calculate order flow toxicity
        self.analysis_results.order_flow_toxicity = self.calculate_order_flow_toxicity();
        
        // Calculate clustering coefficient
        self.analysis_results.clustering_coefficient = self.calculate_clustering_coefficient();
        
        self.last_analysis_time = current_time;
        
        Ok(())
    }
    
    fn analyze_intensity_trend(&self, current_time: FixedPoint) -> Result<IntensityTrend, OrderFlowError> {
        if self.intensity_calculator.event_history.len() < 10 {
            return Ok(IntensityTrend::Stable);
        }
        
        // Calculate intensity over recent windows
        let window_size = self.intensity_calculator.event_history.len() / 3;
        let recent_events: Vec<_> = self.intensity_calculator.event_history
            .iter()
            .rev()
            .take(window_size)
            .collect();
        
        let older_events: Vec<_> = self.intensity_calculator.event_history
            .iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .collect();
        
        if recent_events.is_empty() || older_events.is_empty() {
            return Ok(IntensityTrend::Stable);
        }
        
        let recent_avg_intensity = recent_events.iter()
            .map(|e| self.intensity_calculator.current_intensities[e.base_event.process_id])
            .fold(FixedPoint::zero(), |acc, i| acc + i) / FixedPoint::from_float(recent_events.len() as f64);
        
        let older_avg_intensity = older_events.iter()
            .map(|e| self.intensity_calculator.current_intensities[e.base_event.process_id])
            .fold(FixedPoint::zero(), |acc, i| acc + i) / FixedPoint::from_float(older_events.len() as f64);
        
        let change_ratio = if older_avg_intensity.to_float() > 0.0 {
            (recent_avg_intensity - older_avg_intensity) / older_avg_intensity
        } else {
            FixedPoint::zero()
        };
        
        // Calculate volatility
        let variance = recent_events.iter()
            .map(|e| {
                let intensity = self.intensity_calculator.current_intensities[e.base_event.process_id];
                let diff = intensity - recent_avg_intensity;
                diff * diff
            })
            .fold(FixedPoint::zero(), |acc, v| acc + v) / FixedPoint::from_float(recent_events.len() as f64);
        
        let volatility = variance.sqrt();
        let cv = if recent_avg_intensity.to_float() > 0.0 {
            volatility / recent_avg_intensity
        } else {
            FixedPoint::zero()
        };
        
        // Classify trend
        if cv.to_float() > 0.5 {
            Ok(IntensityTrend::Volatile)
        } else if change_ratio.to_float() > 0.1 {
            Ok(IntensityTrend::Increasing)
        } else if change_ratio.to_float() < -0.1 {
            Ok(IntensityTrend::Decreasing)
        } else {
            Ok(IntensityTrend::Stable)
        }
    }
    
    fn calculate_volume_weighted_spread(&self) -> FixedPoint {
        if self.intensity_calculator.event_history.is_empty() {
            return FixedPoint::zero();
        }
        
        let mut total_volume_spread = FixedPoint::zero();
        let mut total_volume = FixedPoint::zero();
        
        for event in &self.intensity_calculator.event_history {
            total_volume_spread = total_volume_spread + event.volume * event.spread;
            total_volume = total_volume + event.volume;
        }
        
        if total_volume.to_float() > 0.0 {
            total_volume_spread / total_volume
        } else {
            FixedPoint::zero()
        }
    }
    
    fn calculate_order_flow_toxicity(&self) -> FixedPoint {
        if self.intensity_calculator.event_history.len() < 5 {
            return FixedPoint::zero();
        }
        
        // Calculate correlation between order flow and subsequent price changes
        let mut sum_flow_price = FixedPoint::zero();
        let mut sum_flow = FixedPoint::zero();
        let mut sum_price = FixedPoint::zero();
        let mut sum_flow_sq = FixedPoint::zero();
        let mut sum_price_sq = FixedPoint::zero();
        let mut count = 0;
        
        let events: Vec<_> = self.intensity_calculator.event_history.iter().collect();
        
        for i in 0..(events.len() - 1) {
            let flow_sign = if events[i].order_type.is_buy() {
                FixedPoint::one()
            } else {
                -FixedPoint::one()
            };
            
            let price_change = events[i + 1].mid_price_change;
            
            sum_flow_price = sum_flow_price + flow_sign * price_change;
            sum_flow = sum_flow + flow_sign;
            sum_price = sum_price + price_change;
            sum_flow_sq = sum_flow_sq + flow_sign * flow_sign;
            sum_price_sq = sum_price_sq + price_change * price_change;
            count += 1;
        }
        
        if count == 0 {
            return FixedPoint::zero();
        }
        
        let n = FixedPoint::from_float(count as f64);
        let numerator = n * sum_flow_price - sum_flow * sum_price;
        let denominator_flow = n * sum_flow_sq - sum_flow * sum_flow;
        let denominator_price = n * sum_price_sq - sum_price * sum_price;
        
        if denominator_flow.to_float() > 0.0 && denominator_price.to_float() > 0.0 {
            let correlation = numerator / (denominator_flow * denominator_price).sqrt();
            correlation.abs() // Toxicity is absolute correlation
        } else {
            FixedPoint::zero()
        }
    }
    
    fn calculate_clustering_coefficient(&self) -> FixedPoint {
        if self.intensity_calculator.event_history.len() < 10 {
            return FixedPoint::zero();
        }
        
        // Calculate clustering by measuring how often similar order types follow each other
        let mut same_type_transitions = 0;
        let mut total_transitions = 0;
        
        let events: Vec<_> = self.intensity_calculator.event_history.iter().collect();
        
        for i in 0..(events.len() - 1) {
            if events[i].order_type == events[i + 1].order_type {
                same_type_transitions += 1;
            }
            total_transitions += 1;
        }
        
        if total_transitions > 0 {
            FixedPoint::from_float(same_type_transitions as f64 / total_transitions as f64)
        } else {
            FixedPoint::zero()
        }
    }
    
    /// Get current analysis results
    pub fn get_analysis_results(&self) -> &OrderFlowAnalysisResults {
        &self.analysis_results
    }
    
    /// Generate trading signals based on order flow analysis
    pub fn generate_trading_signals(&self) -> TradingSignals {
        let results = &self.analysis_results;
        
        let mut signals = TradingSignals::default();
        
        // Buy/sell pressure signal
        if results.buy_sell_ratio.to_float() > 1.2 {
            signals.directional_signal = DirectionalSignal::Bullish;
            signals.signal_strength = (results.buy_sell_ratio.to_float() - 1.0).min(1.0);
        } else if results.buy_sell_ratio.to_float() < 0.8 {
            signals.directional_signal = DirectionalSignal::Bearish;
            signals.signal_strength = (1.0 - results.buy_sell_ratio.to_float()).min(1.0);
        }
        
        // Market order urgency signal
        if results.market_order_ratio.to_float() > 0.3 {
            signals.urgency_signal = UrgencySignal::High;
        } else if results.market_order_ratio.to_float() > 0.15 {
            signals.urgency_signal = UrgencySignal::Medium;
        }
        
        // Toxicity warning
        if results.order_flow_toxicity.to_float() > 0.7 {
            signals.toxicity_warning = true;
        }
        
        // Clustering signal
        if results.clustering_coefficient.to_float() > 0.6 {
            signals.clustering_detected = true;
        }
        
        signals
    }
    
    /// Generate optimal quote recommendations based on Hawkes process analysis
    pub fn generate_quote_recommendations(
        &self,
        current_mid_price: FixedPoint,
        base_spread: FixedPoint,
        risk_aversion: FixedPoint,
        forecast_horizon: FixedPoint,
    ) -> Result<QuoteRecommendations, OrderFlowError> {
        let metrics = self.intensity_calculator.get_real_time_metrics();
        
        // Calculate intensity-based spread adjustments
        let intensity_spread_adjustment = self.calculate_intensity_spread_adjustment(&metrics)?;
        
        // Calculate market impact adjustments
        let impact_adjustment = self.calculate_impact_adjustment(&metrics, forecast_horizon)?;
        
        // Calculate adverse selection protection
        let adverse_selection_adjustment = self.calculate_adverse_selection_adjustment(&metrics)?;
        
        // Calculate final spread
        let adjusted_spread = base_spread + intensity_spread_adjustment + 
                             impact_adjustment + adverse_selection_adjustment;
        
        // Calculate skew based on order flow imbalance
        let skew = metrics.buy_sell_imbalance * risk_aversion * FixedPoint::from_float(0.5);
        
        // Calculate bid and ask prices
        let bid_price = current_mid_price - adjusted_spread / FixedPoint::from_float(2.0) - skew;
        let ask_price = current_mid_price + adjusted_spread / FixedPoint::from_float(2.0) + skew;
        
        // Calculate recommended sizes based on intensity
        let base_size = FixedPoint::from_float(100.0); // Base order size
        let intensity_factor = (metrics.buy_intensity + metrics.sell_intensity) / 
                              FixedPoint::from_float(2.0);
        
        let bid_size = base_size * (FixedPoint::one() + metrics.sell_intensity / intensity_factor);
        let ask_size = base_size * (FixedPoint::one() + metrics.buy_intensity / intensity_factor);
        
        // Calculate confidence based on data quality
        let confidence = self.calculate_recommendation_confidence(&metrics);
        
        Ok(QuoteRecommendations {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            spread: adjusted_spread,
            skew,
            confidence,
            intensity_adjustment: intensity_spread_adjustment,
            impact_adjustment,
            adverse_selection_adjustment,
            recommended_update_frequency: self.calculate_update_frequency(&metrics),
        })
    }
    
    fn calculate_intensity_spread_adjustment(&self, metrics: &OrderFlowMetrics) -> Result<FixedPoint, OrderFlowError> {
        // Higher intensity volatility requires wider spreads
        let volatility_adjustment = metrics.intensity_volatility * FixedPoint::from_float(0.1);
        
        // Market order pressure increases spreads
        let pressure_adjustment = metrics.market_order_pressure * FixedPoint::from_float(0.05);
        
        Ok(volatility_adjustment + pressure_adjustment)
    }
    
    fn calculate_impact_adjustment(&self, metrics: &OrderFlowMetrics, horizon: FixedPoint) -> Result<FixedPoint, OrderFlowError> {
        // Adjust for cumulative market impact
        let impact_factor = (metrics.cumulative_buy_impact + metrics.cumulative_sell_impact) / 
                           FixedPoint::from_float(2.0);
        
        // Scale by forecast horizon
        let time_decay = (-FixedPoint::from_float(0.5) * horizon).exp();
        
        Ok(impact_factor * time_decay * FixedPoint::from_float(0.02))
    }
    
    fn calculate_adverse_selection_adjustment(&self, metrics: &OrderFlowMetrics) -> Result<FixedPoint, OrderFlowError> {
        // Higher toxicity requires wider spreads for protection
        let toxicity_adjustment = self.analysis_results.order_flow_toxicity * FixedPoint::from_float(0.03);
        
        // Clustering increases adverse selection risk
        let clustering_adjustment = if self.analysis_results.clustering_coefficient.to_float() > 0.5 {
            FixedPoint::from_float(0.01)
        } else {
            FixedPoint::zero()
        };
        
        Ok(toxicity_adjustment + clustering_adjustment)
    }
    
    fn calculate_recommendation_confidence(&self, metrics: &OrderFlowMetrics) -> FixedPoint {
        let mut confidence = FixedPoint::from_float(1.0);
        
        // Reduce confidence if intensity is too low (insufficient data)
        let total_intensity = metrics.buy_intensity + metrics.sell_intensity + 
                             metrics.market_buy_intensity + metrics.market_sell_intensity;
        
        if total_intensity.to_float() < 0.1 {
            confidence = confidence * FixedPoint::from_float(0.5);
        }
        
        // Reduce confidence if volatility is too high (unstable conditions)
        if metrics.intensity_volatility.to_float() > 1.0 {
            confidence = confidence * FixedPoint::from_float(0.7);
        }
        
        // Reduce confidence if we have insufficient history
        if self.intensity_calculator.event_history.len() < 20 {
            confidence = confidence * FixedPoint::from_float(0.8);
        }
        
        confidence
    }
    
    fn calculate_update_frequency(&self, metrics: &OrderFlowMetrics) -> FixedPoint {
        // Base update frequency (updates per second)
        let mut frequency = FixedPoint::from_float(10.0);
        
        // Increase frequency with higher intensity volatility
        if metrics.intensity_volatility.to_float() > 0.5 {
            frequency = frequency * FixedPoint::from_float(2.0);
        }
        
        // Increase frequency with higher market order pressure
        if metrics.market_order_pressure.to_float() > 0.2 {
            frequency = frequency * FixedPoint::from_float(1.5);
        }
        
        // Cap at reasonable maximum
        if frequency.to_float() > 100.0 {
            frequency = FixedPoint::from_float(100.0);
        }
        
        frequency
    }
}

#[derive(Debug, Clone, Default)]
pub struct TradingSignals {
    pub directional_signal: DirectionalSignal,
    pub urgency_signal: UrgencySignal,
    pub signal_strength: f64,
    pub toxicity_warning: bool,
    pub clustering_detected: bool,
}

/// Comprehensive quote recommendations based on Hawkes process analysis
#[derive(Debug, Clone)]
pub struct QuoteRecommendations {
    pub bid_price: FixedPoint,
    pub ask_price: FixedPoint,
    pub bid_size: FixedPoint,
    pub ask_size: FixedPoint,
    pub spread: FixedPoint,
    pub skew: FixedPoint,
    pub confidence: FixedPoint,
    pub intensity_adjustment: FixedPoint,
    pub impact_adjustment: FixedPoint,
    pub adverse_selection_adjustment: FixedPoint,
    pub recommended_update_frequency: FixedPoint,
}

#[derive(Debug, Clone, Default)]
pub enum DirectionalSignal {
    #[default]
    Neutral,
    Bullish,
    Bearish,
}

#[derive(Debug, Clone, Default)]
pub enum UrgencySignal {
    #[default]
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_order_flow_hawkes_params() {
        let params = OrderFlowHawkesParams::new(
            FixedPoint::from_float(1.0),  // baseline_buy_intensity
            FixedPoint::from_float(0.8),  // baseline_sell_intensity
            FixedPoint::from_float(0.3),  // self_excitation
            FixedPoint::from_float(0.1),  // cross_excitation
            FixedPoint::from_float(2.0),  // decay_rate
            FixedPoint::from_float(1.5),  // impact_decay
            FixedPoint::from_float(0.01), // volume_impact_scale
            FixedPoint::from_float(0.05), // spread_impact_factor
        ).unwrap();
        
        assert_eq!(params.hawkes_params.dimension(), 4);
        assert!(params.self_excitation.to_float() > 0.0);
    }
    
    #[test]
    fn test_order_flow_intensity_calculator() {
        let params = OrderFlowHawkesParams::new(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.8),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(1.5),
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(0.05),
        ).unwrap();
        
        let mut calculator = OrderFlowIntensityCalculator::new(params, 100);
        
        // Add some buy orders
        for i in 0..5 {
            let event = OrderFlowEvent::new(
                FixedPoint::from_float(i as f64 * 0.1),
                OrderType::Buy,
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(100.0 + i as f64 * 0.01),
                FixedPoint::from_float(0.02),
                FixedPoint::from_float(0.1),
                FixedPoint::from_float(0.001),
            );
            calculator.add_event(event).unwrap();
        }
        
        let buy_intensity = calculator.get_intensity(OrderType::Buy);
        let sell_intensity = calculator.get_intensity(OrderType::Sell);
        
        // Buy intensity should be higher due to self-excitation
        assert!(buy_intensity.to_float() > sell_intensity.to_float());
        
        // Test forecasting
        let forecast = calculator.forecast_intensity(
            OrderType::Buy,
            FixedPoint::from_float(0.1),
        ).unwrap();
        
        assert!(forecast.to_float() > 0.0);
    }
    
    #[test]
    fn test_real_time_analyzer() {
        let params = OrderFlowHawkesParams::new(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.8),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(1.5),
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(0.05),
        ).unwrap();
        
        let mut analyzer = RealTimeOrderFlowAnalyzer::new(params, 100, 5);
        
        // Add alternating buy/sell events
        for i in 0..20 {
            let order_type = if i % 2 == 0 { OrderType::Buy } else { OrderType::Sell };
            let event = OrderFlowEvent::new(
                FixedPoint::from_float(i as f64 * 0.1),
                order_type,
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(0.02),
                FixedPoint::from_float(0.0),
                FixedPoint::from_float(0.001),
            );
            
            let updated = analyzer.process_event(event).unwrap();
            
            // Should update every 5 events
            if (i + 1) % 5 == 0 {
                assert!(updated);
            }
        }
        
        let results = analyzer.get_analysis_results();
        assert!(results.buy_sell_ratio.to_float() > 0.0);
        
        let signals = analyzer.generate_trading_signals();
        // Should be neutral with alternating orders
        matches!(signals.directional_signal, DirectionalSignal::Neutral);
    }
    
    #[test]
    fn test_market_impact_calculation() {
        let params = OrderFlowHawkesParams::new(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.8),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(1.5),
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(0.05),
        ).unwrap();
        
        let mut calculator = OrderFlowIntensityCalculator::new(params, 100);
        
        // Add some events to build history
        for i in 0..10 {
            let event = OrderFlowEvent::new(
                FixedPoint::from_float(i as f64 * 0.1),
                OrderType::Buy,
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(0.02),
                FixedPoint::from_float(0.1),
                FixedPoint::from_float(0.001),
            );
            calculator.add_event(event).unwrap();
        }
        
        // Test market impact calculation
        let impact = calculator.calculate_market_impact(
            OrderType::Buy,
            FixedPoint::from_float(500.0),
            FixedPoint::from_float(0.1),
        ).unwrap();
        
        assert!(impact.to_float() > 0.0);
        
        // Test real-time metrics
        let metrics = calculator.get_real_time_metrics();
        assert!(metrics.buy_intensity.to_float() > 0.0);
        assert!(metrics.intensity_volatility.to_float() >= 0.0);
    }
    
    #[test]
    fn test_quote_recommendations() {
        let params = OrderFlowHawkesParams::new(
            FixedPoint::from_float(1.0),
            FixedPoint::from_float(0.8),
            FixedPoint::from_float(0.3),
            FixedPoint::from_float(0.1),
            FixedPoint::from_float(2.0),
            FixedPoint::from_float(1.5),
            FixedPoint::from_float(0.01),
            FixedPoint::from_float(0.05),
        ).unwrap();
        
        let mut analyzer = RealTimeOrderFlowAnalyzer::new(params, 100, 5);
        
        // Add events with buy bias
        for i in 0..30 {
            let order_type = if i % 3 == 0 { OrderType::Sell } else { OrderType::Buy };
            let event = OrderFlowEvent::new(
                FixedPoint::from_float(i as f64 * 0.1),
                order_type,
                FixedPoint::from_float(100.0),
                FixedPoint::from_float(100.0 + i as f64 * 0.001),
                FixedPoint::from_float(0.02),
                FixedPoint::from_float(0.1),
                FixedPoint::from_float(0.001),
            );
            analyzer.process_event(event).unwrap();
        }
        
        // Generate quote recommendations
        let recommendations = analyzer.generate_quote_recommendations(
            FixedPoint::from_float(100.0),  // current_mid_price
            FixedPoint::from_float(0.02),   // base_spread
            FixedPoint::from_float(0.5),    // risk_aversion
            FixedPoint::from_float(0.1),    // forecast_horizon
        ).unwrap();
        
        // Verify recommendations are reasonable
        assert!(recommendations.ask_price > recommendations.bid_price);
        assert!(recommendations.spread.to_float() > 0.0);
        assert!(recommendations.confidence.to_float() > 0.0);
        assert!(recommendations.confidence.to_float() <= 1.0);
        assert!(recommendations.recommended_update_frequency.to_float() > 0.0);
        
        // With buy bias, should have positive skew (ask wider than bid)
        if recommendations.skew.to_float() > 0.0 {
            let mid_price = FixedPoint::from_float(100.0);
            let bid_distance = mid_price - recommendations.bid_price;
            let ask_distance = recommendations.ask_price - mid_price;
            assert!(ask_distance > bid_distance);
        }
    }
}