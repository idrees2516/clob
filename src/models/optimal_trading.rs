/// Strategy optimization components
pub struct StrategyOptimizer {
    pub market_maker: MarketMaker,
    pub objective_function: ObjectiveFunction,
    pub constraints: OptimizationConstraints,
    pub optimizer: Optimizer,
}

impl StrategyOptimizer {
    pub fn new(
        market_maker: MarketMaker,
        objective_function: ObjectiveFunction,
        constraints: OptimizationConstraints,
        optimizer: Optimizer,
    ) -> Self {
        Self {
            market_maker,
            objective_function,
            constraints,
            optimizer,
        }
    }

    pub fn optimize_parameters(
        &mut self,
        market_data: &MarketData,
        initial_params: &OptimizationParameters,
    ) -> Result<OptimizationResult, OptimalTradingError> {
        // Set up optimization problem
        let objective = |params: &[f64]| -> Result<f64, OptimalTradingError> {
            let parameters = OptimizationParameters::from_vec(params)?;
            self.update_strategy_parameters(&parameters)?;
            
            let performance = self.backtest_strategy(market_data)?;
            let objective_value = self.objective_function.evaluate(&performance)?;
            
            Ok(-objective_value) // Negative because we want to maximize
        };
        
        // Convert constraints to bounds
        let bounds = self.constraints.to_bounds()?;
        
        // Run optimization
        let result = self.optimizer.optimize(
            objective,
            initial_params.to_vec()?,
            &bounds,
        )?;
        
        // Convert result back to parameters
        let optimal_params = OptimizationParameters::from_vec(&result.parameters)?;
        let performance = self.backtest_strategy(market_data)?;
        
        Ok(OptimizationResult {
            parameters: optimal_params,
            performance,
            convergence: result.convergence,
            iterations: result.iterations,
        })
    }

    fn update_strategy_parameters(
        &mut self,
        params: &OptimizationParameters,
    ) -> Result<(), OptimalTradingError> {
        // Update market maker parameters
        self.market_maker.execution_model.urgency = params.urgency;
        self.market_maker.execution_model.market_participation = params.participation;
        self.market_maker.execution_model.price_improvement = params.improvement;
        
        // Update position manager parameters
        self.market_maker.position_manager.target_position = params.target_position;
        self.market_maker.position_manager.position_half_life = params.half_life;
        self.market_maker.position_manager.risk_aversion = params.risk_aversion;
        
        // Update risk limits
        self.market_maker.risk_limits.max_position = params.max_position;
        self.market_maker.risk_limits.max_turnover = params.max_turnover;
        self.market_maker.risk_limits.max_drawdown = params.max_drawdown;
        self.market_maker.risk_limits.var_limit = params.var_limit;
        
        Ok(())
    }

    fn backtest_strategy(
        &self,
        market_data: &MarketData,
    ) -> Result<StrategyPerformance, OptimalTradingError> {
        let mut performance = StrategyPerformance::new();
        let mut current_state = MarketState::new(
            market_data.initial_price,
            0.0,
            0.0,
            market_data.initial_volatility,
            0.0,
            market_data.initial_volume,
        );
        
        for (i, data) in market_data.time_series.iter().enumerate() {
            // Update market state
            current_state.mid_price = data.price;
            current_state.volatility = data.volatility;
            current_state.volume = data.volume;
            current_state.signal = data.signal;
            
            // Get quotes
            let quotes = self.market_maker.update_quotes(&current_state)?;
            
            // Simulate executions
            let executions = self.simulate_executions(
                &quotes,
                &current_state,
                data,
            )?;
            
            // Update performance metrics
            performance.update(&executions, &current_state)?;
            
            // Update position
            current_state.inventory += executions.net_volume;
            current_state.queue_position = executions.queue_position;
        }
        
        Ok(performance)
    }

    fn simulate_executions(
        &self,
        quotes: &QuotePair,
        state: &MarketState,
        data: &MarketTick,
    ) -> Result<ExecutionSummary, OptimalTradingError> {
        let mut summary = ExecutionSummary::default();
        
        // Simulate bid side executions
        if data.low_price <= quotes.bid {
            let bid_volume = data.volume * 0.5 * (quotes.bid - data.low_price)
                / (data.high_price - data.low_price);
            summary.bid_executions = bid_volume.min(data.volume * 0.3);
        }
        
        // Simulate ask side executions
        if data.high_price >= quotes.ask {
            let ask_volume = data.volume * 0.5 * (data.high_price - quotes.ask)
                / (data.high_price - data.low_price);
            summary.ask_executions = ask_volume.min(data.volume * 0.3);
        }
        
        // Update queue position based on order flow
        summary.queue_position = state.queue_position * 0.9 +
            0.1 * (summary.bid_executions - summary.ask_executions).abs();
            
        // Compute net volume
        summary.net_volume = summary.bid_executions - summary.ask_executions;
        
        Ok(summary)
    }

    /// Advanced optimal execution: dynamic participation, urgency, risk aversion, market impact, liquidity
    pub fn compute_optimal_execution(
        &self,
        state: &MarketState,
        data: &MarketTick,
        params: &OptimizationParameters,
    ) -> Result<ExecutionSummary, OptimalTradingError> {
        // Almgren-Chriss-style optimal execution
        let T = params.half_life;
        let gamma = params.risk_aversion;
        let eta = params.improvement;
        let sigma = state.volatility;
        let X = params.target_position - state.position;
        let N = (T / data.dt).ceil() as usize;
        let mut executions = Vec::with_capacity(N);
        let mut x = state.position;
        for i in 0..N {
            let t = i as f64 * data.dt;
            let participation = params.participation * (1.0 - (t / T).exp());
            let liquidity_adj = (1.0 / (1.0 + state.liquidity_risk)).max(0.1);
            let size = (X / N as f64) * participation * liquidity_adj;
            let impact = eta * size * size;
            let price = data.mid_price - gamma * sigma * (X - x) - impact;
            executions.push((price, size));
            x += size;
        }
        Ok(ExecutionSummary {
            executions,
            total_volume: executions.iter().map(|(_, s)| s.abs()).sum(),
            total_cost: executions.iter().map(|(p, s)| p * s).sum(),
            realized_impact: executions.iter().map(|(p, s)| (p - data.mid_price) * s).sum(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationParameters {
    pub urgency: f64,
    pub participation: f64,
    pub improvement: f64,
    pub target_position: f64,
    pub half_life: f64,
    pub risk_aversion: f64,
    pub max_position: f64,
    pub max_turnover: f64,
    pub max_drawdown: f64,
    pub var_limit: f64,
}

impl OptimizationParameters {
    pub fn to_vec(&self) -> Result<Vec<f64>, OptimalTradingError> {
        Ok(vec![
            self.urgency,
            self.participation,
            self.improvement,
            self.target_position,
            self.half_life,
            self.risk_aversion,
            self.max_position,
            self.max_turnover,
            self.max_drawdown,
            self.var_limit,
        ])
    }

    pub fn from_vec(params: &[f64]) -> Result<Self, OptimalTradingError> {
        if params.len() != 10 {
            return Err(OptimalTradingError::ParameterError(
                "Invalid parameter vector length".to_string(),
            ));
        }
        
        Ok(Self {
            urgency: params[0],
            participation: params[1],
            improvement: params[2],
            target_position: params[3],
            half_life: params[4],
            risk_aversion: params[5],
            max_position: params[6],
            max_turnover: params[7],
            max_drawdown: params[8],
            var_limit: params[9],
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    pub parameter_bounds: Vec<(f64, f64)>,
}

impl OptimizationConstraints {
    pub fn new(parameter_bounds: Vec<(f64, f64)>) -> Self {
        Self { parameter_bounds }
    }

    pub fn to_bounds(&self) -> Result<Vec<(f64, f64)>, OptimalTradingError> {
        Ok(self.parameter_bounds.clone())
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub parameters: OptimizationParameters,
    pub performance: StrategyPerformance,
    pub convergence: bool,
    pub iterations: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionSummary {
    pub bid_executions: f64,
    pub ask_executions: f64,
    pub net_volume: f64,
    pub queue_position: f64,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub initial_price: f64,
    pub initial_volatility: f64,
    pub initial_volume: f64,
    pub time_series: Vec<MarketTick>,
}

#[derive(Debug, Clone)]
pub struct MarketTick {
    pub price: f64,
    pub volatility: f64,
    pub volume: f64,
    pub signal: f64,
    pub high_price: f64,
    pub low_price: f64,
}

#[derive(Debug, Clone)]
pub struct StrategyPerformance {
    pub pnl: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub turnover: f64,
    pub avg_spread: f64,
    pub avg_position: f64,
    pub trade_count: usize,
}

impl StrategyPerformance {
    pub fn new() -> Self {
        Self {
            pnl: 0.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            turnover: 0.0,
            avg_spread: 0.0,
            avg_position: 0.0,
            trade_count: 0,
        }
    }

    pub fn update(
        &mut self,
        executions: &ExecutionSummary,
        state: &MarketState,
    ) -> Result<(), OptimalTradingError> {
        // Update PnL
        let trade_pnl = executions.bid_executions * state.mid_price -
            executions.ask_executions * state.mid_price;
        self.pnl += trade_pnl;
        
        // Update turnover
        self.turnover += executions.bid_executions.abs() + executions.ask_executions.abs();
        
        // Update position metrics
        self.avg_position = (self.avg_position * self.trade_count as f64 +
            state.inventory) / (self.trade_count + 1) as f64;
            
        // Update trade count
        if executions.bid_executions > 0.0 || executions.ask_executions > 0.0 {
            self.trade_count += 1;
        }
        
        Ok(())
    }
}

pub struct ObjectiveFunction {
    pub pnl_weight: f64,
    pub sharpe_weight: f64,
    pub drawdown_weight: f64,
}

impl ObjectiveFunction {
    pub fn new(
        pnl_weight: f64,
        sharpe_weight: f64,
        drawdown_weight: f64,
    ) -> Self {
        Self {
            pnl_weight,
            sharpe_weight,
            drawdown_weight,
        }
    }

    pub fn evaluate(
        &self,
        performance: &StrategyPerformance,
    ) -> Result<f64, OptimalTradingError> {
        let objective = 
            self.pnl_weight * performance.pnl +
            self.sharpe_weight * performance.sharpe_ratio -
            self.drawdown_weight * performance.max_drawdown;
            
        Ok(objective)
    }
}

pub struct Optimizer {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Optimizer {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    pub fn optimize<F>(
        &self,
        objective: F,
        initial_params: Vec<f64>,
        bounds: &[(f64, f64)],
    ) -> Result<OptimizerResult, OptimalTradingError>
    where
        F: Fn(&[f64]) -> Result<f64, OptimalTradingError>,
    {
        let mut current_params = initial_params.clone();
        let mut current_value = objective(&current_params)?;
        let mut iterations = 0;
        let mut converged = false;
        
        while iterations < self.max_iterations {
            let mut improved = false;
            
            // Coordinate descent
            for i in 0..current_params.len() {
                let original = current_params[i];
                let (lower, upper) = bounds[i];
                
                // Try positive direction
                current_params[i] = (original + self.tolerance).min(upper);
                let value_plus = objective(&current_params)?;
                
                // Try negative direction
                current_params[i] = (original - self.tolerance).max(lower);
                let value_minus = objective(&current_params)?;
                
                // Update if improved
                if value_plus > current_value {
                    current_value = value_plus;
                    improved = true;
                } else if value_minus > current_value {
                    current_params[i] = (original - self.tolerance).max(lower);
                    current_value = value_minus;
                    improved = true;
                } else {
                    current_params[i] = original;
                }
            }
            
            // Check convergence
            if !improved {
                converged = true;
                break;
            }
            
            iterations += 1;
        }
        
        Ok(OptimizerResult {
            parameters: current_params,
            value: current_value,
            convergence: converged,
            iterations,
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerResult {
    pub parameters: Vec<f64>,
    pub value: f64,
    pub convergence: bool,
    pub iterations: usize,
}
