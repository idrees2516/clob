use nalgebra as na;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PerformanceError {
    #[error("Performance calculation error: {0}")]
    CalculationError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Comprehensive performance metrics for high-frequency trading strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensivePerformanceMetrics {
    // Basic performance metrics
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub information_ratio: f64,
    pub sortino_ratio: f64,
    
    // Risk metrics
    pub max_drawdown: f64,
    pub max_drawdown_duration: usize,
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    
    // Liquidity-adjusted metrics
    pub liquidity_adjusted_return: f64,
    pub liquidity_cost: f64,
    pub market_impact_cost: f64,
    pub bid_ask_spread_cost: f64,
    
    // Transaction cost analysis
    pub transaction_costs: TransactionCostMetrics,
    
    // Trading activity metrics
    pub total_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub average_trade_pnl: f64,
    pub largest_win: f64,
    pub largest_loss: f64,
    
    // Position and inventory metrics
    pub average_position: f64,
    pub position_volatility: f64,
    pub inventory_turnover: f64,
    pub time_in_market: f64,
    
    // Market making specific metrics
    pub fill_ratio: f64,
    pub adverse_selection_cost: f64,
    pub inventory_risk_premium: f64,
    pub quote_competitiveness: f64,
}

/// Transaction cost analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionCostMetrics {
    pub total_cost: f64,
    pub cost_per_share: f64,
    pub cost_as_percentage_of_volume: f64,
    pub implementation_shortfall: f64,
    pub timing_cost: f64,
    pub market_impact_cost: f64,
    pub spread_cost: f64,
    pub commission_cost: f64,
    pub cost_breakdown: HashMap<String, f64>,
}

/// Input data for performance calculation
#[derive(Debug, Clone)]
pub struct PerformanceData {
    pub returns: Vec<f64>,
    pub prices: Vec<f64>,
    pub positions: Vec<f64>,
    pub trades: Vec<TradeData>,
    pub market_data: Vec<MarketData>,
    pub timestamps: Vec<f64>,
    pub benchmark_returns: Option<Vec<f64>>,
    pub risk_free_rate: f64,
}

/// Individual trade data
#[derive(Debug, Clone)]
pub struct TradeData {
    pub timestamp: f64,
    pub price: f64,
    pub quantity: f64,
    pub side: TradeSide,
    pub transaction_cost: f64,
    pub market_impact: f64,
    pub spread_at_trade: f64,
}

/// Market data snapshot
#[derive(Debug, Clone)]
pub struct MarketData {
    pub timestamp: f64,
    pub mid_price: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub spread: f64,
    pub depth: f64,
    pub volatility: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Performance metrics calculator
pub struct PerformanceCalculator {
    pub annualization_factor: f64, // e.g., 252 for daily data, 252*24*60 for minute data
}

impl PerformanceCalculator {
    pub fn new(annualization_factor: f64) -> Self {
        Self { annualization_factor }
    }
    
    /// Calculate comprehensive performance metrics
    pub fn calculate_metrics(&self, data: &PerformanceData) -> Result<ComprehensivePerformanceMetrics, PerformanceError> {
        if data.returns.is_empty() {
            return Err(PerformanceError::InsufficientData("No return data provided".to_string()));
        }
        
        // Basic performance metrics
        let total_return = self.calculate_total_return(&data.returns)?;
        let annualized_return = self.calculate_annualized_return(&data.returns)?;
        let volatility = self.calculate_volatility(&data.returns)?;
        let sharpe_ratio = self.calculate_sharpe_ratio(&data.returns, data.risk_free_rate)?;
        let information_ratio = self.calculate_information_ratio(&data.returns, &data.benchmark_returns)?;
        let sortino_ratio = self.calculate_sortino_ratio(&data.returns, data.risk_free_rate)?;
        
        // Risk metrics
        let max_drawdown = self.calculate_max_drawdown(&data.returns)?;
        let max_drawdown_duration = self.calculate_max_drawdown_duration(&data.returns)?;
        let (var_95, var_99) = self.calculate_var(&data.returns)?;
        let (es_95, es_99) = self.calculate_expected_shortfall(&data.returns)?;
        let skewness = self.calculate_skewness(&data.returns)?;
        let kurtosis = self.calculate_kurtosis(&data.returns)?;
        
        // Liquidity-adjusted metrics
        let liquidity_metrics = self.calculate_liquidity_adjusted_metrics(data)?;
        
        // Transaction cost analysis
        let transaction_costs = self.calculate_transaction_costs(data)?;
        
        // Trading activity metrics
        let trading_metrics = self.calculate_trading_metrics(data)?;
        
        // Position metrics
        let position_metrics = self.calculate_position_metrics(data)?;
        
        // Market making metrics
        let market_making_metrics = self.calculate_market_making_metrics(data)?;
        
        Ok(ComprehensivePerformanceMetrics {
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            information_ratio,
            sortino_ratio,
            max_drawdown,
            max_drawdown_duration,
            var_95,
            var_99,
            expected_shortfall_95: es_95,
            expected_shortfall_99: es_99,
            skewness,
            kurtosis,
            liquidity_adjusted_return: liquidity_metrics.0,
            liquidity_cost: liquidity_metrics.1,
            market_impact_cost: liquidity_metrics.2,
            bid_ask_spread_cost: liquidity_metrics.3,
            transaction_costs,
            total_trades: trading_metrics.0,
            win_rate: trading_metrics.1,
            profit_factor: trading_metrics.2,
            average_trade_pnl: trading_metrics.3,
            largest_win: trading_metrics.4,
            largest_loss: trading_metrics.5,
            average_position: position_metrics.0,
            position_volatility: position_metrics.1,
            inventory_turnover: position_metrics.2,
            time_in_market: position_metrics.3,
            fill_ratio: market_making_metrics.0,
            adverse_selection_cost: market_making_metrics.1,
            inventory_risk_premium: market_making_metrics.2,
            quote_competitiveness: market_making_metrics.3,
        })
    }
    
    /// Calculate total return
    fn calculate_total_return(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.is_empty() {
            return Err(PerformanceError::InsufficientData("No returns provided".to_string()));
        }
        
        // Compound returns: (1 + r1) * (1 + r2) * ... - 1
        let total_return = returns.iter()
            .fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;
        
        Ok(total_return)
    }
    
    /// Calculate annualized return
    fn calculate_annualized_return(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.is_empty() {
            return Err(PerformanceError::InsufficientData("No returns provided".to_string()));
        }
        
        let total_return = self.calculate_total_return(returns)?;
        let periods = returns.len() as f64;
        let annualized_return = (1.0 + total_return).powf(self.annualization_factor / periods) - 1.0;
        
        Ok(annualized_return)
    }
    
    /// Calculate volatility (annualized)
    fn calculate_volatility(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.len() < 2 {
            return Err(PerformanceError::InsufficientData("Need at least 2 returns for volatility".to_string()));
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        let volatility = variance.sqrt() * self.annualization_factor.sqrt();
        Ok(volatility)
    }
    
    /// Calculate Sharpe ratio
    fn calculate_sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> Result<f64, PerformanceError> {
        let annualized_return = self.calculate_annualized_return(returns)?;
        let volatility = self.calculate_volatility(returns)?;
        
        if volatility == 0.0 {
            return Ok(0.0);
        }
        
        let sharpe_ratio = (annualized_return - risk_free_rate) / volatility;
        Ok(sharpe_ratio)
    }
    
    /// Calculate Information Ratio (vs benchmark)
    fn calculate_information_ratio(&self, returns: &[f64], benchmark_returns: &Option<Vec<f64>>) -> Result<f64, PerformanceError> {
        let benchmark = match benchmark_returns {
            Some(bench) => bench,
            None => return Ok(0.0), // No benchmark provided
        };
        
        if returns.len() != benchmark.len() {
            return Err(PerformanceError::InvalidParameter("Returns and benchmark lengths don't match".to_string()));
        }
        
        // Calculate excess returns
        let excess_returns: Vec<f64> = returns.iter()
            .zip(benchmark.iter())
            .map(|(&r, &b)| r - b)
            .collect();
        
        let mean_excess = excess_returns.iter().sum::<f64>() / excess_returns.len() as f64;
        let tracking_error = self.calculate_volatility(&excess_returns)?;
        
        if tracking_error == 0.0 {
            return Ok(0.0);
        }
        
        let information_ratio = mean_excess * self.annualization_factor.sqrt() / tracking_error;
        Ok(information_ratio)
    }
    
    /// Calculate Sortino ratio (downside deviation)
    fn calculate_sortino_ratio(&self, returns: &[f64], risk_free_rate: f64) -> Result<f64, PerformanceError> {
        let annualized_return = self.calculate_annualized_return(returns)?;
        
        // Calculate downside deviation
        let target_return = risk_free_rate / self.annualization_factor;
        let downside_variance = returns.iter()
            .map(|&r| if r < target_return { (r - target_return).powi(2) } else { 0.0 })
            .sum::<f64>() / returns.len() as f64;
        
        let downside_deviation = downside_variance.sqrt() * self.annualization_factor.sqrt();
        
        if downside_deviation == 0.0 {
            return Ok(0.0);
        }
        
        let sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation;
        Ok(sortino_ratio)
    }
    
    /// Calculate maximum drawdown
    fn calculate_max_drawdown(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.is_empty() {
            return Err(PerformanceError::InsufficientData("No returns provided".to_string()));
        }
        
        let mut cumulative_return = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;
        
        for &ret in returns {
            cumulative_return *= 1.0 + ret;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            let drawdown = (peak - cumulative_return) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        Ok(max_drawdown)
    }
    
    /// Calculate maximum drawdown duration
    fn calculate_max_drawdown_duration(&self, returns: &[f64]) -> Result<usize, PerformanceError> {
        if returns.is_empty() {
            return Err(PerformanceError::InsufficientData("No returns provided".to_string()));
        }
        
        let mut cumulative_return = 1.0;
        let mut peak = 1.0;
        let mut current_duration = 0;
        let mut max_duration = 0;
        
        for &ret in returns {
            cumulative_return *= 1.0 + ret;
            if cumulative_return > peak {
                peak = cumulative_return;
                current_duration = 0;
            } else {
                current_duration += 1;
                if current_duration > max_duration {
                    max_duration = current_duration;
                }
            }
        }
        
        Ok(max_duration)
    }
    
    /// Calculate Value at Risk (VaR) at 95% and 99% confidence levels
    fn calculate_var(&self, returns: &[f64]) -> Result<(f64, f64), PerformanceError> {
        if returns.len() < 20 {
            return Err(PerformanceError::InsufficientData("Need at least 20 returns for VaR".to_string()));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_95_index = ((1.0 - 0.95) * returns.len() as f64) as usize;
        let var_99_index = ((1.0 - 0.99) * returns.len() as f64) as usize;
        
        let var_95 = -sorted_returns[var_95_index]; // VaR is positive for losses
        let var_99 = -sorted_returns[var_99_index];
        
        Ok((var_95, var_99))
    }
    
    /// Calculate Expected Shortfall (Conditional VaR)
    fn calculate_expected_shortfall(&self, returns: &[f64]) -> Result<(f64, f64), PerformanceError> {
        if returns.len() < 20 {
            return Err(PerformanceError::InsufficientData("Need at least 20 returns for ES".to_string()));
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_95_index = ((1.0 - 0.95) * returns.len() as f64) as usize;
        let var_99_index = ((1.0 - 0.99) * returns.len() as f64) as usize;
        
        let es_95 = -sorted_returns[..=var_95_index].iter().sum::<f64>() / (var_95_index + 1) as f64;
        let es_99 = -sorted_returns[..=var_99_index].iter().sum::<f64>() / (var_99_index + 1) as f64;
        
        Ok((es_95, es_99))
    }
    
    /// Calculate skewness
    fn calculate_skewness(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.len() < 3 {
            return Err(PerformanceError::InsufficientData("Need at least 3 returns for skewness".to_string()));
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        if variance == 0.0 {
            return Ok(0.0);
        }
        
        let std_dev = variance.sqrt();
        let skewness = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>() / returns.len() as f64;
        
        Ok(skewness)
    }
    
    /// Calculate kurtosis
    fn calculate_kurtosis(&self, returns: &[f64]) -> Result<f64, PerformanceError> {
        if returns.len() < 4 {
            return Err(PerformanceError::InsufficientData("Need at least 4 returns for kurtosis".to_string()));
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        if variance == 0.0 {
            return Ok(0.0);
        }
        
        let std_dev = variance.sqrt();
        let kurtosis = returns.iter()
            .map(|&r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>() / returns.len() as f64 - 3.0; // Excess kurtosis
        
        Ok(kurtosis)
    }
}    

    /// Calculate liquidity-adjusted metrics
    fn calculate_liquidity_adjusted_metrics(&self, data: &PerformanceData) -> Result<(f64, f64, f64, f64), PerformanceError> {
        if data.market_data.is_empty() || data.trades.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }
        
        // Calculate total liquidity costs
        let mut total_market_impact = 0.0;
        let mut total_spread_cost = 0.0;
        let mut total_volume = 0.0;
        
        for trade in &data.trades {
            total_market_impact += trade.market_impact.abs();
            total_spread_cost += trade.spread_at_trade * trade.quantity.abs();
            total_volume += trade.quantity.abs();
        }
        
        let market_impact_cost = if total_volume > 0.0 { total_market_impact / total_volume } else { 0.0 };
        let bid_ask_spread_cost = if total_volume > 0.0 { total_spread_cost / total_volume } else { 0.0 };
        let liquidity_cost = market_impact_cost + bid_ask_spread_cost;
        
        // Calculate liquidity-adjusted return
        let total_return = self.calculate_total_return(&data.returns)?;
        let liquidity_adjusted_return = total_return - liquidity_cost;
        
        Ok((liquidity_adjusted_return, liquidity_cost, market_impact_cost, bid_ask_spread_cost))
    }
    
    /// Calculate transaction cost metrics
    fn calculate_transaction_costs(&self, data: &PerformanceData) -> Result<TransactionCostMetrics, PerformanceError> {
        if data.trades.is_empty() {
            return Ok(TransactionCostMetrics {
                total_cost: 0.0,
                cost_per_share: 0.0,
                cost_as_percentage_of_volume: 0.0,
                implementation_shortfall: 0.0,
                timing_cost: 0.0,
                market_impact_cost: 0.0,
                spread_cost: 0.0,
                commission_cost: 0.0,
                cost_breakdown: HashMap::new(),
            });
        }
        
        let mut total_cost = 0.0;
        let mut total_volume = 0.0;
        let mut total_market_impact = 0.0;
        let mut total_spread_cost = 0.0;
        let mut total_commission = 0.0;
        let mut timing_cost = 0.0;
        let mut cost_breakdown = HashMap::new();
        
        // Calculate costs from trades
        for (i, trade) in data.trades.iter().enumerate() {
            total_cost += trade.transaction_cost;
            total_volume += trade.quantity.abs();
            total_market_impact += trade.market_impact.abs();
            total_spread_cost += trade.spread_at_trade * trade.quantity.abs() * 0.5; // Half spread
            total_commission += trade.transaction_cost - trade.market_impact.abs();
            
            // Calculate timing cost (difference between decision time and execution time)
            if i > 0 {
                let price_move = (trade.price - data.trades[i-1].price).abs();
                timing_cost += price_move * trade.quantity.abs();
            }
        }
        
        // Calculate implementation shortfall
        let theoretical_value = data.trades.iter()
            .map(|t| t.price * t.quantity)
            .sum::<f64>();
        let actual_value = data.trades.iter()
            .map(|t| (t.price - t.transaction_cost) * t.quantity)
            .sum::<f64>();
        let implementation_shortfall = if theoretical_value != 0.0 {
            (theoretical_value - actual_value) / theoretical_value.abs()
        } else { 0.0 };
        
        // Cost breakdown
        cost_breakdown.insert("Market Impact".to_string(), total_market_impact);
        cost_breakdown.insert("Spread Cost".to_string(), total_spread_cost);
        cost_breakdown.insert("Commission".to_string(), total_commission);
        cost_breakdown.insert("Timing Cost".to_string(), timing_cost);
        
        Ok(TransactionCostMetrics {
            total_cost,
            cost_per_share: if total_volume > 0.0 { total_cost / total_volume } else { 0.0 },
            cost_as_percentage_of_volume: if total_volume > 0.0 { total_cost / (total_volume * data.trades.iter().map(|t| t.price).sum::<f64>() / data.trades.len() as f64) } else { 0.0 },
            implementation_shortfall,
            timing_cost,
            market_impact_cost: total_market_impact,
            spread_cost: total_spread_cost,
            commission_cost: total_commission,
            cost_breakdown,
        })
    }
    
    /// Calculate trading activity metrics
    fn calculate_trading_metrics(&self, data: &PerformanceData) -> Result<(usize, f64, f64, f64, f64, f64), PerformanceError> {
        if data.trades.is_empty() {
            return Ok((0, 0.0, 0.0, 0.0, 0.0, 0.0));
        }
        
        let total_trades = data.trades.len();
        
        // Calculate individual trade P&L
        let mut trade_pnls = Vec::new();
        let mut winning_trades = 0;
        let mut total_wins = 0.0;
        let mut total_losses = 0.0;
        let mut largest_win = 0.0;
        let mut largest_loss = 0.0;
        
        for (i, trade) in data.trades.iter().enumerate() {
            // Estimate trade P&L based on position changes
            let pnl = if i < data.returns.len() {
                data.returns[i] * trade.quantity.abs()
            } else {
                0.0
            };
            
            trade_pnls.push(pnl);
            
            if pnl > 0.0 {
                winning_trades += 1;
                total_wins += pnl;
                if pnl > largest_win {
                    largest_win = pnl;
                }
            } else if pnl < 0.0 {
                total_losses += pnl.abs();
                if pnl.abs() > largest_loss {
                    largest_loss = pnl.abs();
                }
            }
        }
        
        let win_rate = winning_trades as f64 / total_trades as f64;
        let profit_factor = if total_losses > 0.0 { total_wins / total_losses } else { 0.0 };
        let average_trade_pnl = trade_pnls.iter().sum::<f64>() / total_trades as f64;
        
        Ok((total_trades, win_rate, profit_factor, average_trade_pnl, largest_win, largest_loss))
    }
    
    /// Calculate position and inventory metrics
    fn calculate_position_metrics(&self, data: &PerformanceData) -> Result<(f64, f64, f64, f64), PerformanceError> {
        if data.positions.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }
        
        // Average position
        let average_position = data.positions.iter().sum::<f64>() / data.positions.len() as f64;
        
        // Position volatility
        let position_variance = data.positions.iter()
            .map(|&p| (p - average_position).powi(2))
            .sum::<f64>() / data.positions.len() as f64;
        let position_volatility = position_variance.sqrt();
        
        // Inventory turnover (total position changes / average position)
        let mut total_position_change = 0.0;
        for i in 1..data.positions.len() {
            total_position_change += (data.positions[i] - data.positions[i-1]).abs();
        }
        let inventory_turnover = if average_position.abs() > 0.0 {
            total_position_change / (average_position.abs() * data.positions.len() as f64)
        } else { 0.0 };
        
        // Time in market (percentage of time with non-zero position)
        let non_zero_positions = data.positions.iter().filter(|&&p| p.abs() > 1e-6).count();
        let time_in_market = non_zero_positions as f64 / data.positions.len() as f64;
        
        Ok((average_position, position_volatility, inventory_turnover, time_in_market))
    }
    
    /// Calculate market making specific metrics
    fn calculate_market_making_metrics(&self, data: &PerformanceData) -> Result<(f64, f64, f64, f64), PerformanceError> {
        if data.trades.is_empty() || data.market_data.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }
        
        // Fill ratio (executed trades / total quote opportunities)
        // Approximated as trades per market data point
        let fill_ratio = data.trades.len() as f64 / data.market_data.len() as f64;
        
        // Adverse selection cost
        let mut adverse_selection_cost = 0.0;
        let mut total_volume = 0.0;
        
        for trade in &data.trades {
            // Find corresponding market data
            if let Some(market_data) = data.market_data.iter()
                .find(|md| (md.timestamp - trade.timestamp).abs() < 1e-6) {
                
                let mid_price = market_data.mid_price;
                let adverse_selection = match trade.side {
                    TradeSide::Buy => (trade.price - mid_price).max(0.0),
                    TradeSide::Sell => (mid_price - trade.price).max(0.0),
                };
                adverse_selection_cost += adverse_selection * trade.quantity.abs();
                total_volume += trade.quantity.abs();
            }
        }
        
        adverse_selection_cost = if total_volume > 0.0 { adverse_selection_cost / total_volume } else { 0.0 };
        
        // Inventory risk premium (compensation for holding inventory)
        let inventory_risk_premium = if !data.positions.is_empty() && !data.returns.is_empty() {
            let position_variance = self.calculate_position_metrics(data)?.1.powi(2);
            let return_variance = self.calculate_volatility(&data.returns)?.powi(2);
            position_variance * return_variance
        } else { 0.0 };
        
        // Quote competitiveness (how often our quotes are at the best bid/ask)
        let mut competitive_quotes = 0;
        let mut total_quotes = 0;
        
        for market_data in &data.market_data {
            // Simplified: assume we're competitive if spread is tight
            if market_data.spread < market_data.mid_price * 0.001 { // Less than 10 bps
                competitive_quotes += 1;
            }
            total_quotes += 1;
        }
        
        let quote_competitiveness = if total_quotes > 0 { competitive_quotes as f64 / total_quotes as f64 } else { 0.0 };
        
        Ok((fill_ratio, adverse_selection_cost, inventory_risk_premium, quote_competitiveness))
    }
}

impl Default for PerformanceCalculator {
    fn default() -> Self {
        Self::new(252.0) // Default to daily data with 252 trading days per year
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    fn create_test_data() -> PerformanceData {
        PerformanceData {
            returns: vec![0.01, -0.005, 0.02, -0.01, 0.015, -0.008, 0.012],
            prices: vec![100.0, 101.0, 100.5, 102.5, 101.5, 103.0, 102.2, 103.4],
            positions: vec![0.0, 100.0, 100.0, 200.0, 150.0, 150.0, 100.0, 50.0],
            trades: vec![
                TradeData {
                    timestamp: 1.0,
                    price: 101.0,
                    quantity: 100.0,
                    side: TradeSide::Buy,
                    transaction_cost: 0.1,
                    market_impact: 0.05,
                    spread_at_trade: 0.02,
                },
                TradeData {
                    timestamp: 3.0,
                    price: 102.5,
                    quantity: 100.0,
                    side: TradeSide::Buy,
                    transaction_cost: 0.1,
                    market_impact: 0.05,
                    spread_at_trade: 0.02,
                },
            ],
            market_data: vec![
                MarketData {
                    timestamp: 1.0,
                    mid_price: 100.5,
                    bid_price: 100.49,
                    ask_price: 100.51,
                    bid_size: 1000.0,
                    ask_size: 1000.0,
                    spread: 0.02,
                    depth: 2000.0,
                    volatility: 0.15,
                },
            ],
            timestamps: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            benchmark_returns: Some(vec![0.008, -0.003, 0.015, -0.008, 0.012, -0.006, 0.010]),
            risk_free_rate: 0.02,
        }
    }
    
    #[test]
    fn test_total_return_calculation() {
        let calculator = PerformanceCalculator::default();
        let returns = vec![0.1, -0.05, 0.08];
        let total_return = calculator.calculate_total_return(&returns).unwrap();
        let expected = (1.1 * 0.95 * 1.08) - 1.0;
        assert_relative_eq!(total_return, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_volatility_calculation() {
        let calculator = PerformanceCalculator::default();
        let returns = vec![0.01, -0.01, 0.02, -0.02];
        let volatility = calculator.calculate_volatility(&returns).unwrap();
        assert!(volatility > 0.0);
    }
    
    #[test]
    fn test_sharpe_ratio_calculation() {
        let calculator = PerformanceCalculator::default();
        let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let sharpe = calculator.calculate_sharpe_ratio(&returns, 0.02).unwrap();
        assert!(sharpe.is_finite());
    }
    
    #[test]
    fn test_max_drawdown_calculation() {
        let calculator = PerformanceCalculator::default();
        let returns = vec![0.1, -0.2, 0.05, -0.1, 0.15];
        let max_dd = calculator.calculate_max_drawdown(&returns).unwrap();
        assert!(max_dd >= 0.0);
        assert!(max_dd <= 1.0);
    }
    
    #[test]
    fn test_comprehensive_metrics() {
        let calculator = PerformanceCalculator::default();
        let data = create_test_data();
        let metrics = calculator.calculate_metrics(&data).unwrap();
        
        assert!(metrics.total_return.is_finite());
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.volatility > 0.0);
        assert!(metrics.total_trades > 0);
    }
    
    #[test]
    fn test_var_calculation() {
        let calculator = PerformanceCalculator::default();
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) / 1000.0).collect();
        let (var_95, var_99) = calculator.calculate_var(&returns).unwrap();
        assert!(var_95 >= 0.0);
        assert!(var_99 >= var_95);
    }
    
    #[test]
    fn test_transaction_cost_analysis() {
        let calculator = PerformanceCalculator::default();
        let data = create_test_data();
        let tx_costs = calculator.calculate_transaction_costs(&data).unwrap();
        
        assert!(tx_costs.total_cost >= 0.0);
        assert!(tx_costs.cost_per_share >= 0.0);
        assert!(!tx_costs.cost_breakdown.is_empty());
    }
}