use nalgebra as na;
use rand::prelude::*;
use rand_distr::{Normal, Exponential, Poisson};
use rayon::prelude::*;
use thiserror::Error;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;
use crate::models::rough_volatility::{RoughVolatilityModel, RoughVolatilityParams};
use crate::math::fixed_point::FixedPoint;

#[derive(Debug, Error)]
pub enum LOBError {
    #[error("Limit order book error: {0}")]
    LOBError(String),
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

/// Advanced order types
#[derive(Debug, Clone, PartialEq)]
pub enum AdvancedOrderType {
    Standard,         // Regular limit/market order
    Iceberg { peak: f64 }, // Only part of the order is visible
    Pegged { reference: PeggedReference, offset: f64 }, // Pegged to best bid/ask/mid
    PostOnly,         // Only adds liquidity, cancels if would match
    Hidden,           // Not visible in the book, matches like a marketable limit
}

#[derive(Debug, Clone, PartialEq)]
pub enum PeggedReference {
    BestBid,
    BestAsk,
    Mid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Order {
    pub id: u64,
    pub price: FixedPoint,
    pub size: FixedPoint,
    pub timestamp: u64,
    pub is_buy: bool,
}

impl Order {
    pub fn validate(&self) -> bool {
        self.size > FixedPoint::zero() && self.price >= FixedPoint::zero()
    }
}

#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: FixedPoint,
    pub orders: VecDeque<Order>, // FIFO queue
}

impl PriceLevel {
    pub fn new(price: FixedPoint) -> Self {
        PriceLevel { price, orders: VecDeque::new() }
    }
    pub fn add_order(&mut self, order: Order) {
        self.orders.push_back(order);
    }
    pub fn pop_order(&mut self) -> Option<Order> {
        self.orders.pop_front()
    }
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    pub price: FixedPoint,
    pub size: FixedPoint,
    pub timestamp: u64,
}

#[derive(Debug)]
pub struct OrderBook {
    pub bids: BTreeMap<FixedPoint, PriceLevel>, // descending order
    pub asks: BTreeMap<FixedPoint, PriceLevel>, // ascending order
    pub next_order_id: u64,
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            next_order_id: 1,
        }
    }
    pub fn insert_order(&mut self, mut order: Order) -> bool {
        if !order.validate() {
            return false;
        }
        order.id = self.next_order_id;
        self.next_order_id += 1;
        let book = if order.is_buy { &mut self.bids } else { &mut self.asks };
        book.entry(order.price)
            .or_insert_with(|| PriceLevel::new(order.price))
            .add_order(order);
        true
    }
    pub fn match_orders(&mut self, timestamp: u64) -> Vec<Trade> {
        let mut trades = Vec::new();
        while let (Some((&best_bid, bid_level)), Some((&best_ask, ask_level))) =
            (self.bids.iter_mut().next_back(), self.asks.iter_mut().next())
        {
            if best_bid < best_ask {
                break;
            }
            let mut bid_order = bid_level.orders.front().cloned().unwrap();
            let mut ask_order = ask_level.orders.front().cloned().unwrap();
            let trade_size = bid_order.size.min(ask_order.size);
            let trade_price = best_ask; // price-time priority: use resting order price
            let trade = Trade {
                buy_order_id: if bid_order.is_buy { bid_order.id } else { ask_order.id },
                sell_order_id: if !bid_order.is_buy { bid_order.id } else { ask_order.id },
                price: trade_price,
                size: trade_size,
                timestamp,
            };
            trades.push(trade);
            // Update or remove orders
            if bid_order.size > trade_size {
                bid_level.orders.front_mut().unwrap().size = bid_order.size - trade_size;
            } else {
                bid_level.pop_order();
            }
            if ask_order.size > trade_size {
                ask_level.orders.front_mut().unwrap().size = ask_order.size - trade_size;
            } else {
                ask_level.pop_order();
            }
            if bid_level.is_empty() {
                self.bids.remove(&best_bid);
            }
            if ask_level.is_empty() {
                self.asks.remove(&best_ask);
            }
        }
        trades
    }

    /// Returns the best bid (highest buy price) and best ask (lowest sell price)
    pub fn best_bid_ask(&self) -> (Option<FixedPoint>, Option<FixedPoint>) {
        let best_bid = self.bids.keys().next_back().cloned();
        let best_ask = self.asks.keys().next().cloned();
        (best_bid, best_ask)
    }

    /// Returns the total market depth on each side (sum of all order sizes)
    pub fn market_depth(&self) -> (FixedPoint, FixedPoint) {
        let bid_depth = self.bids.values().flat_map(|level| level.orders.iter()).map(|o| o.size).fold(FixedPoint::zero(), |a, b| a + b);
        let ask_depth = self.asks.values().flat_map(|level| level.orders.iter()).map(|o| o.size).fold(FixedPoint::zero(), |a, b| a + b);
        (bid_depth, ask_depth)
    }

    /// Returns the full order book depth as a vector of (price, size) pairs for each side
    pub fn depth_profile(&self, side: bool) -> Vec<(FixedPoint, FixedPoint)> {
        let book = if side { &self.bids } else { &self.asks };
        book.iter().map(|(price, level)| {
            let total = level.orders.iter().map(|o| o.size).fold(FixedPoint::zero(), |a, b| a + b);
            (*price, total)
        }).collect()
    }

    /// Generates a trade tick (last trade) from a Trade
    pub fn trade_tick(trade: &Trade) -> TradeTick {
        TradeTick {
            price: trade.price,
            size: trade.size,
            timestamp: trade.timestamp,
        }
    }

    /// Computes OHLCV statistics for a sequence of trades
    pub fn ohlcv(trades: &[Trade]) -> Option<OHLCV> {
        if trades.is_empty() { return None; }
        let open = trades.first().unwrap().price;
        let close = trades.last().unwrap().price;
        let high = trades.iter().map(|t| t.price).max().unwrap();
        let low = trades.iter().map(|t| t.price).min().unwrap();
        let volume = trades.iter().map(|t| t.size).fold(FixedPoint::zero(), |a, b| a + b);
        Some(OHLCV { open, high, low, close, volume })
    }

    /// Computes volume profile (histogram of volume by price)
    pub fn volume_profile(trades: &[Trade]) -> Vec<(FixedPoint, FixedPoint)> {
        use std::collections::BTreeMap;
        let mut profile = BTreeMap::new();
        for t in trades {
            *profile.entry(t.price).or_insert(FixedPoint::zero()) += t.size;
        }
        profile.into_iter().collect()
    }

    /// Cancel an order by ID. Returns true if cancelled.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        for book in [&mut self.bids, &mut self.asks] {
            for level in book.values_mut() {
                if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                    level.orders.remove(pos);
                    return true;
                }
            }
        }
        false
    }

    /// Modify an order by ID. Returns true if modified.
    pub fn modify_order(&mut self, order_id: u64, new_size: Option<FixedPoint>, new_price: Option<FixedPoint>) -> bool {
        for book in [&mut self.bids, &mut self.asks] {
            for (price, level) in book.iter_mut() {
                if let Some(pos) = level.orders.iter().position(|o| o.id == order_id) {
                    let mut order = level.orders.remove(pos).unwrap();
                    if let Some(size) = new_size { order.size = size; }
                    if let Some(price_val) = new_price { order.price = price_val; }
                    // Re-insert at new price if price changed
                    if let Some(price_val) = new_price {
                        let target_book = if order.is_buy { &mut self.bids } else { &mut self.asks };
                        target_book.entry(price_val).or_insert_with(|| PriceLevel::new(price_val)).add_order(order);
                    } else {
                        level.add_order(order);
                    }
                    // Clean up empty price level
                    if level.is_empty() { book.remove(price); }
                    return true;
                }
            }
        }
        false
    }

    /// Get the status of an order by ID
    pub fn get_order_status(&self, order_id: u64) -> OrderStatus {
        for book in [&self.bids, &self.asks] {
            for level in book.values() {
                for o in &level.orders {
                    if o.id == order_id {
                        return if o.size > FixedPoint::zero() {
                            OrderStatus::Active
                        } else {
                            OrderStatus::Filled
                        };
                    }
                }
            }
        }
        OrderStatus::Cancelled // If not found, assume cancelled (could be improved)
    }

    /// Risk check: reject order if size or price is out of bounds
    pub fn check_risk(&self, order: &Order, max_size: FixedPoint, min_price: FixedPoint, max_price: FixedPoint) -> Result<(), String> {
        if order.size > max_size {
            return Err("Order size exceeds limit".to_string());
        }
        if order.price < min_price || order.price > max_price {
            return Err("Order price out of bounds".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TradeTick {
    pub price: FixedPoint,
    pub size: FixedPoint,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct OHLCV {
    pub open: FixedPoint,
    pub high: FixedPoint,
    pub low: FixedPoint,
    pub close: FixedPoint,
    pub volume: FixedPoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderStatus {
    Active,
    PartiallyFilled(FixedPoint), // remaining size
    Filled,
    Cancelled,
    Rejected(String),
}

#[cfg(test)]
mod market_data_tests {
    use super::*;
    #[test]
    fn test_best_bid_ask_and_depth() {
        let mut book = OrderBook::new();
        let p1 = FixedPoint::from_float(100.0);
        let p2 = FixedPoint::from_float(101.0);
        let s1 = FixedPoint::from_float(2.0);
        let s2 = FixedPoint::from_float(3.0);
        book.insert_order(Order { id: 0, price: p1, size: s1, timestamp: 1, is_buy: true });
        book.insert_order(Order { id: 0, price: p2, size: s2, timestamp: 2, is_buy: false });
        let (best_bid, best_ask) = book.best_bid_ask();
        assert_eq!(best_bid.unwrap().to_float(), 100.0);
        assert_eq!(best_ask.unwrap().to_float(), 101.0);
        let (bid_depth, ask_depth) = book.market_depth();
        assert_eq!(bid_depth.to_float(), 2.0);
        assert_eq!(ask_depth.to_float(), 3.0);
    }
    #[test]
    fn test_depth_profile() {
        let mut book = OrderBook::new();
        let p = FixedPoint::from_float(100.0);
        book.insert_order(Order { id: 0, price: p, size: FixedPoint::from_float(1.0), timestamp: 1, is_buy: true });
        let profile = book.depth_profile(true);
        assert_eq!(profile.len(), 1);
        assert_eq!(profile[0].0.to_float(), 100.0);
        assert_eq!(profile[0].1.to_float(), 1.0);
    }
    #[test]
    fn test_trade_tick_and_ohlcv() {
        let t1 = Trade { buy_order_id: 1, sell_order_id: 2, price: FixedPoint::from_float(100.0), size: FixedPoint::from_float(1.0), timestamp: 1 };
        let t2 = Trade { buy_order_id: 3, sell_order_id: 4, price: FixedPoint::from_float(101.0), size: FixedPoint::from_float(2.0), timestamp: 2 };
        let tick = OrderBook::trade_tick(&t2);
        assert_eq!(tick.price.to_float(), 101.0);
        let ohlcv = OrderBook::ohlcv(&[t1.clone(), t2.clone()]).unwrap();
        assert_eq!(ohlcv.open.to_float(), 100.0);
        assert_eq!(ohlcv.close.to_float(), 101.0);
        assert_eq!(ohlcv.high.to_float(), 101.0);
        assert_eq!(ohlcv.low.to_float(), 100.0);
        assert_eq!(ohlcv.volume.to_float(), 3.0);
    }
    #[test]
    fn test_volume_profile() {
        let t1 = Trade { buy_order_id: 1, sell_order_id: 2, price: FixedPoint::from_float(100.0), size: FixedPoint::from_float(1.0), timestamp: 1 };
        let t2 = Trade { buy_order_id: 3, sell_order_id: 4, price: FixedPoint::from_float(101.0), size: FixedPoint::from_float(2.0), timestamp: 2 };
        let t3 = Trade { buy_order_id: 5, sell_order_id: 6, price: FixedPoint::from_float(100.0), size: FixedPoint::from_float(1.5), timestamp: 3 };
        let profile = OrderBook::volume_profile(&[t1, t2, t3]);
        assert_eq!(profile.len(), 2);
        assert_eq!(profile[0].0.to_float(), 100.0);
        assert!((profile[0].1.to_float() - 2.5).abs() < 1e-8);
        assert_eq!(profile[1].0.to_float(), 101.0);
        assert_eq!(profile[1].1.to_float(), 2.0);
    }
}

#[cfg(test)]
mod order_management_tests {
    use super::*;
    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new();
        let price = FixedPoint::from_float(100.0);
        let o = Order { id: 0, price, size: FixedPoint::from_float(1.0), timestamp: 1, is_buy: true };
        book.insert_order(o);
        let id = book.next_order_id - 1;
        assert!(book.cancel_order(id));
        assert_eq!(book.get_order_status(id), OrderStatus::Cancelled);
    }
    #[test]
    fn test_modify_order() {
        let mut book = OrderBook::new();
        let price = FixedPoint::from_float(100.0);
        let o = Order { id: 0, price, size: FixedPoint::from_float(1.0), timestamp: 1, is_buy: true };
        book.insert_order(o);
        let id = book.next_order_id - 1;
        assert!(book.modify_order(id, Some(FixedPoint::from_float(2.0)), None));
        let status = book.get_order_status(id);
        assert_eq!(status, OrderStatus::Active);
    }
    #[test]
    fn test_risk_check() {
        let book = OrderBook::new();
        let order = Order { id: 0, price: FixedPoint::from_float(100.0), size: FixedPoint::from_float(1.0), timestamp: 1, is_buy: true };
        assert!(book.check_risk(&order, FixedPoint::from_float(2.0), FixedPoint::from_float(50.0), FixedPoint::from_float(200.0)).is_ok());
        let bad_order = Order { id: 0, price: FixedPoint::from_float(10.0), size: FixedPoint::from_float(3.0), timestamp: 1, is_buy: true };
        assert!(book.check_risk(&bad_order, FixedPoint::from_float(2.0), FixedPoint::from_float(50.0), FixedPoint::from_float(200.0)).is_err());
    }
}

/// Parameters for the limit order book model
pub struct LOBParams {
    pub tick_size: f64,
    pub base_intensity: f64,         // Base arrival intensity λ
    pub intensity_decay: f64,        // Intensity decay parameter κ
    pub size_distribution: f64,      // Parameter for order size distribution
    pub cancellation_rate: f64,      // Order cancellation rate
    pub spread_sensitivity: f64,     // Spread sensitivity to volatility
    pub depth_sensitivity: f64,      // Depth sensitivity to volatility
}

/// High-frequency limit order book model with rough volatility
pub struct LimitOrderBook {
    params: LOBParams,
    volatility_model: RoughVolatilityModel,
    buy_orders: BTreeMap<i64, VecDeque<Order>>,    // Price levels -> FIFO queue
    sell_orders: BTreeMap<i64, VecDeque<Order>>,
    order_counter: u64,
    current_time: f64,
    rng: ThreadRng,
    // Optionally, store hidden and pegged orders separately for performance
    hidden_orders: Vec<Order>,
    pegged_orders: Vec<Order>,
    latency_queue: VecDeque<LatencyOrder>,
}

impl LimitOrderBook {
    pub fn new(
        params: LOBParams,
        volatility_params: RoughVolatilityParams,
        grid_points: usize,
        time_horizon: f64,
    ) -> Result<Self, LOBError> {
        // Validate parameters
        if params.tick_size <= 0.0 || params.base_intensity <= 0.0 || 
           params.intensity_decay <= 0.0 || params.size_distribution <= 0.0 {
            return Err(LOBError::ParameterError(
                "Model parameters must be positive".to_string(),
            ));
        }

        let volatility_model = RoughVolatilityModel::new(
            volatility_params,
            grid_points,
            time_horizon,
        ).map_err(|e| LOBError::LOBError(e.to_string()))?;

        Ok(Self {
            params,
            volatility_model,
            buy_orders: BTreeMap::new(),
            sell_orders: BTreeMap::new(),
            order_counter: 0,
            current_time: 0.0,
            rng: thread_rng(),
            hidden_orders: Vec::new(),
            pegged_orders: Vec::new(),
            latency_queue: VecDeque::new(),
        })
    }

    /// Simulates the limit order book dynamics
    pub fn simulate(
        &mut self,
        initial_price: f64,
        n_steps: usize,
        dt: f64,
    ) -> Result<LOBSimulation, LOBError> {
        let mut mid_prices = Vec::with_capacity(n_steps);
        let mut spreads = Vec::with_capacity(n_steps);
        let mut depths = Vec::with_capacity(n_steps);
        let mut volumes = Vec::with_capacity(n_steps);

        // Simulate rough volatility process
        let (_, volatility) = self.volatility_model.simulate_price_process(initial_price)
            .map_err(|e| LOBError::LOBError(e.to_string()))?;

        // Initialize order book
        self.initialize_book(initial_price)?;

        for step in 0..n_steps {
            // Update current time
            self.current_time = step as f64 * dt;
            
            // Generate order flow based on current volatility
            self.generate_order_flow(volatility[step], dt)?;
            
            // Process order matching
            self.match_orders()?;

            // Clean up cancelled orders
            self.process_cancellations(dt)?;

            // Process latency queue
            self.process_latency_queue();

            // Process pegged orders
            self.process_pegged_orders();

            // Record market state
            let (mid_price, spread, depth) = self.compute_market_state()?;
            let volume = self.compute_trading_volume()?;

            mid_prices.push(mid_price);
            spreads.push(spread);
            depths.push(depth);
            volumes.push(volume);
        }

        Ok(LOBSimulation {
            mid_prices,
            spreads,
            depths,
            volumes,
            final_state: self.get_book_state()?,
        })
    }

    /// Initializes the limit order book around the initial price
    fn initialize_book(&mut self, initial_price: f64) -> Result<(), LOBError> {
        let price_ticks = (initial_price / self.params.tick_size).round() as i64;
        let initial_levels = 10; // Number of price levels to initialize

        // Initialize buy orders below initial price
        for i in 0..initial_levels {
            let price_level = price_ticks - i - 1;
            let orders = self.generate_initial_orders(
                price_level as f64 * self.params.tick_size,
                true,
            )?;
            self.buy_orders.insert(price_level, orders);
        }

        // Initialize sell orders above initial price
        for i in 0..initial_levels {
            let price_level = price_ticks + i + 1;
            let orders = self.generate_initial_orders(
                price_level as f64 * self.params.tick_size,
                false,
            )?;
            self.sell_orders.insert(price_level, orders);
        }

        Ok(())
    }

    /// Generates initial orders at a price level
    fn generate_initial_orders(
        &mut self,
        price: f64,
        is_buy: bool,
    ) -> Result<Vec<Order>, LOBError> {
        let n_orders = Poisson::new(5.0) // Average number of orders per level
            .map_err(|e| LOBError::LOBError(e.to_string()))?
            .sample(&mut self.rng) as usize;

        let size_dist = Exponential::new(1.0 / self.params.size_distribution)
            .map_err(|e| LOBError::LOBError(e.to_string()))?;

        let mut orders = Vec::with_capacity(n_orders);
        for _ in 0..n_orders {
            let order = Order {
                id: self.order_counter,
                price: FixedPoint::from_float(price),
                size: FixedPoint::from_float(size_dist.sample(&mut self.rng)),
                timestamp: self.current_time as u64,
                is_buy,
            };
            orders.push(order);
            self.order_counter += 1;
        }

        Ok(orders)
    }

    /// Generates order flow based on current volatility
    fn generate_order_flow(
        &mut self,
        volatility: f64,
        dt: f64,
    ) -> Result<(), LOBError> {
        // Compute arrival intensities based on volatility
        let base_intensity = self.params.base_intensity * 
            (1.0 + self.params.intensity_decay * volatility.sqrt());

        // Generate limit orders
        let n_limit_orders = Poisson::new(base_intensity * dt)
            .map_err(|e| LOBError::LOBError(e.to_string()))?
            .sample(&mut self.rng) as usize;

        for _ in 0..n_limit_orders {
            self.submit_limit_order(volatility)?;
        }

        // Generate market orders
        let n_market_orders = Poisson::new(base_intensity * dt * 0.5) // Assume market orders arrive at half the rate
            .map_err(|e| LOBError::LOBError(e.to_string()))?
            .sample(&mut self.rng) as usize;

        for _ in 0..n_market_orders {
            self.submit_market_order(volatility)?;
        }

        Ok(())
    }

    /// Submits a new limit order
    fn submit_limit_order(&mut self, volatility: f64) -> Result<(), LOBError> {
        // Determine order side
        let is_buy = self.rng.gen_bool(0.5);

        // Generate order size
        let size_dist = Exponential::new(1.0 / self.params.size_distribution)
            .map_err(|e| LOBError::LOBError(e.to_string()))?;
        let size = FixedPoint::from_float(size_dist.sample(&mut self.rng));

        // Determine price level based on current market state and volatility
        let (mid_price, spread, _) = self.compute_market_state()?;
        let price_std = volatility.sqrt() * mid_price * self.params.spread_sensitivity;
        
        let price_offset = Normal::new(0.0, price_std)
            .map_err(|e| LOBError::LOBError(e.to_string()))?
            .sample(&mut self.rng);

        let price = FixedPoint::from_float(if is_buy {
            mid_price - spread/2.0 + price_offset
        } else {
            mid_price + spread/2.0 + price_offset
        });

        // Round price to nearest tick
        let price_level = (price / FixedPoint::from_float(self.params.tick_size)).round() as i64;

        // Create and insert order
        let order = Order {
            id: self.order_counter,
            price,
            size,
            timestamp: self.current_time as u64,
            is_buy,
        };
        self.order_counter += 1;

        if is_buy {
            self.buy_orders
                .entry(price_level)
                .or_insert_with(VecDeque::new)
                .push_back(order);
        } else {
            self.sell_orders
                .entry(price_level)
                .or_insert_with(VecDeque::new)
                .push_back(order);
        }

        Ok(())
    }

    /// Submits a market order
    fn submit_market_order(&mut self, volatility: f64) -> Result<(), LOBError> {
        // Determine order side
        let is_buy = self.rng.gen_bool(0.5);

        // Generate order size
        let size_dist = Exponential::new(1.0 / self.params.size_distribution)
            .map_err(|e| LOBError::LOBError(e.to_string()))?;
        let size = FixedPoint::from_float(size_dist.sample(&mut self.rng));

        if is_buy {
            self.execute_market_buy(size)?;
        } else {
            self.execute_market_sell(size)?;
        }

        Ok(())
    }

    /// Executes a market buy order
    fn execute_market_buy(&mut self, size: FixedPoint) -> Result<(), LOBError> {
        let mut remaining_size = size;

        while remaining_size > FixedPoint::zero() {
            // Get best ask
            let best_ask = match self.sell_orders.iter_mut().next() {
                Some((&price_level, orders)) => {
                    // Match against available orders
                    let mut matched_size = FixedPoint::zero();
                    orders.retain(|order| {
                        if remaining_size > FixedPoint::zero() {
                            let match_size = remaining_size.min(order.size);
                            remaining_size -= match_size;
                            matched_size += match_size;
                            order.size > match_size // Keep order if partially filled
                        } else {
                            true
                        }
                    });

                    // Remove empty price level
                    if orders.is_empty() {
                        self.sell_orders.remove(&price_level);
                    }

                    Some(price_level)
                }
                None => None,
            };

            if best_ask.is_none() || remaining_size <= FixedPoint::zero() {
                break;
            }
        }

        Ok(())
    }

    /// Executes a market sell order
    fn execute_market_sell(&mut self, size: FixedPoint) -> Result<(), LOBError> {
        let mut remaining_size = size;

        while remaining_size > FixedPoint::zero() {
            // Get best bid
            let best_bid = match self.buy_orders.iter_mut().next_back() {
                Some((&price_level, orders)) => {
                    // Match against available orders
                    let mut matched_size = FixedPoint::zero();
                    orders.retain(|order| {
                        if remaining_size > FixedPoint::zero() {
                            let match_size = remaining_size.min(order.size);
                            remaining_size -= match_size;
                            matched_size += match_size;
                            order.size > match_size // Keep order if partially filled
                        } else {
                            true
                        }
                    });

                    // Remove empty price level
                    if orders.is_empty() {
                        self.buy_orders.remove(&price_level);
                    }

                    Some(price_level)
                }
                None => None,
            };

            if best_bid.is_none() || remaining_size <= FixedPoint::zero() {
                break;
            }
        }

        Ok(())
    }

    /// Matches crossing orders in the book
    fn match_orders(&mut self) -> Result<(), LOBError> {
        // First, match visible orders (VecDeque FIFO)
        loop {
            let best_bid = self.buy_orders.iter().next_back().map(|(&p, _)| p);
            let best_ask = self.sell_orders.iter().next().map(|(&p, _)| p);
            match (best_bid, best_ask) {
                (Some(bid), Some(ask)) if bid >= ask => {
                    self.match_price_levels(bid, ask)?;
                }
                _ => break,
            }
        }
        // Then, match hidden orders opportunistically
        self.match_hidden_orders();
        // Pegged orders should be re-priced periodically (not implemented here)
        Ok(())
    }

    /// Matches orders at crossing price levels
    fn match_price_levels(&mut self, bid: i64, ask: i64) -> Result<(), LOBError> {
        if let (Some(buy_orders), Some(sell_orders)) = (
            self.buy_orders.get_mut(&bid),
            self.sell_orders.get_mut(&ask),
        ) {
            let mut i = 0;
            let mut j = 0;

            while i < buy_orders.len() && j < sell_orders.len() {
                let match_size = buy_orders[i].size.min(sell_orders[j].size);

                // Update order sizes
                buy_orders[i].size -= match_size;
                sell_orders[j].size -= match_size;

                // Move to next order if fully matched
                if buy_orders[i].size <= FixedPoint::zero() {
                    i += 1;
                }
                if sell_orders[j].size <= FixedPoint::zero() {
                    j += 1;
                }
            }

            // Remove fully matched orders
            buy_orders.retain(|order| order.size > FixedPoint::zero());
            sell_orders.retain(|order| order.size > FixedPoint::zero());

            // Remove empty price levels
            if buy_orders.is_empty() {
                self.buy_orders.remove(&bid);
            }
            if sell_orders.is_empty() {
                self.sell_orders.remove(&ask);
            }
        }

        Ok(())
    }

    /// Processes order cancellations
    fn process_cancellations(&mut self, dt: f64) -> Result<(), LOBError> {
        let cancel_prob = 1.0 - (-self.params.cancellation_rate * dt).exp();

        // Process buy orders
        for orders in self.buy_orders.values_mut() {
            orders.retain(|_| !self.rng.gen_bool(cancel_prob));
        }
        self.buy_orders.retain(|_, orders| !orders.is_empty());

        // Process sell orders
        for orders in self.sell_orders.values_mut() {
            orders.retain(|_| !self.rng.gen_bool(cancel_prob));
        }
        self.sell_orders.retain(|_, orders| !orders.is_empty());

        Ok(())
    }

    /// Computes current market state
    fn compute_market_state(&self) -> Result<(f64, f64, f64), LOBError> {
        let best_bid = self.buy_orders.iter().next_back()
            .map(|(&p, _)| p as f64 * self.params.tick_size)
            .unwrap_or(0.0);

        let best_ask = self.sell_orders.iter().next()
            .map(|(&p, _)| p as f64 * self.params.tick_size)
            .unwrap_or(f64::INFINITY);

        let mid_price = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        
        // Compute market depth
        let depth = self.compute_market_depth()?;

        Ok((mid_price, spread, depth))
    }

    /// Computes market depth
    fn compute_market_depth(&self) -> Result<f64, LOBError> {
        let buy_depth: f64 = self.buy_orders.values()
            .flat_map(|orders| orders.iter())
            .map(|order| order.size.to_float())
            .sum();

        let sell_depth: f64 = self.sell_orders.values()
            .flat_map(|orders| orders.iter())
            .map(|order| order.size.to_float())
            .sum();

        Ok(buy_depth + sell_depth)
    }

    /// Computes trading volume in the current time step
    fn compute_trading_volume(&self) -> Result<f64, LOBError> {
        // In a real implementation, you would track and return actual trading volume
        // Here we return a placeholder based on current market state
        let (_, spread, depth) = self.compute_market_state()?;
        Ok(depth / (1.0 + spread)) // Simple proxy for trading volume
    }

    /// Returns current book state
    fn get_book_state(&self) -> Result<BookState, LOBError> {
        Ok(BookState {
            buy_levels: self.buy_orders.clone(),
            sell_levels: self.sell_orders.clone(),
        })
    }

    /// Computes market impact
    pub fn compute_market_impact(
        &self,
        order_size: f64,
        is_buy: bool,
        params: &MarketImpactParams,
    ) -> Result<(f64, f64), LOBError> {
        let sign = if is_buy { 1.0 } else { -1.0 };
        let volume = self.compute_volume()?;
        
        // Compute permanent impact using square-root model with nonlinear adjustment
        let permanent = params.permanent_impact * sign * 
            (order_size / volume).powf(0.5 + params.nonlinear_factor);
            
        // Compute temporary impact with volume-weighted adjustment
        let temporary = params.temporary_impact * sign *
            (order_size / volume).powf(0.5) * 
            (1.0 + self.compute_order_imbalance()?).abs();
            
        Ok((permanent, temporary))
    }

    /// Simulates Hawkes process
    pub fn simulate_hawkes_process(
        &mut self,
        dt: f64,
        n_steps: usize,
        base_intensity: f64,
        alpha: f64,
        beta: f64,
    ) -> Result<Vec<(f64, OrderType)>, LOBError> {
        let mut rng = thread_rng();
        let mut intensity = base_intensity;
        let mut events = Vec::new();
        let mut t = 0.0;
        
        while events.len() < n_steps {
            // Generate next event time
            let u: f64 = rng.gen();
            let dt_next = -1.0 / intensity * u.ln();
            t += dt_next;
            
            // Generate event type based on current book state
            let (bid_prob, ask_prob) = self.compute_order_probabilities()?;
            let event_type = match rng.gen::<f64>() {
                x if x < bid_prob => OrderType::Limit(true),  // Bid
                x if x < bid_prob + ask_prob => OrderType::Limit(false), // Ask
                _ => OrderType::Market(rng.gen_bool(0.5)),
            };
            
            events.push((t, event_type));
            
            // Update intensity
            intensity = base_intensity + 
                events.iter()
                    .map(|(s, _)| alpha * (-beta * (t - s)).exp())
                    .sum::<f64>();
        }
        
        Ok(events)
    }

    /// Computes order probabilities
    pub fn compute_order_probabilities(&self) -> Result<(f64, f64), LOBError> {
        let imbalance = self.compute_order_imbalance()?;
        let spread = self.compute_spread()?;
        let volatility = self.compute_volatility()?;
        
        // Compute base probabilities using logistic function
        let bid_base = 1.0 / (1.0 + (-2.0 * imbalance).exp());
        let ask_base = 1.0 - bid_base;
        
        // Adjust for spread and volatility
        let spread_factor = (-0.5 * spread / volatility).exp();
        let bid_prob = bid_base * spread_factor;
        let ask_prob = ask_base * spread_factor;
        
        // Normalize
        let total = bid_prob + ask_prob;
        Ok((bid_prob / total, ask_prob / total))
    }

    /// Processes self-exciting orders
    pub fn process_self_exciting_orders(
        &mut self,
        events: &[(f64, OrderType)],
        impact_params: &MarketImpactParams,
    ) -> Result<Vec<TradeExecution>, LOBError> {
        let mut executions = Vec::new();
        let mut price_impact = 0.0;
        
        for &(t, ref event_type) in events {
            match event_type {
                OrderType::Market(is_buy) => {
                    let size = self.generate_order_size(is_buy);
                    let (permanent, temporary) = self.compute_market_impact(
                        size,
                        is_buy,
                        impact_params,
                    )?;
                    
                    // Apply price impacts
                    price_impact += permanent;
                    let total_impact = price_impact + temporary;
                    
                    // Execute market order with impact
                    let execution = self.execute_market_order(
                        size,
                        is_buy,
                        total_impact,
                    )?;
                    executions.push(execution);
                    
                    // Decay temporary impact
                    price_impact *= (-impact_params.decay_rate * t).exp();
                }
                OrderType::Limit(is_buy) => {
                    let size = self.generate_order_size(is_buy);
                    let price = self.compute_optimal_limit_price(
                        size,
                        is_buy,
                        price_impact,
                    )?;
                    
                    self.submit_limit_order(
                        Order {
                            id: self.next_order_id(),
                            price: FixedPoint::from_float(price),
                            size,
                            timestamp: t as u64,
                            is_buy,
                        },
                    )?;
                }
            }
        }
        
        Ok(executions)
    }

    /// Computes optimal limit price
    fn compute_optimal_limit_price(
        &self,
        size: f64,
        is_buy: bool,
        price_impact: f64,
    ) -> Result<f64, LOBError> {
        let mid_price = self.compute_mid_price()?;
        let spread = self.compute_spread()?;
        let volatility = self.compute_volatility()?;
        let imbalance = self.compute_order_imbalance()?;
        
        // Base price adjustment
        let sign = if is_buy { -1.0 } else { 1.0 };
        let base_adjustment = sign * spread * 0.5;
        
        // Volatility adjustment
        let vol_adjustment = sign * volatility * 
            (size / self.compute_volume()?).sqrt();
            
        // Imbalance adjustment
        let imb_adjustment = -sign * spread * imbalance;
        
        // Impact adjustment
        let impact_adjustment = price_impact * 0.5;
        
        let price = mid_price + base_adjustment + 
            vol_adjustment + imb_adjustment + impact_adjustment;
            
        Ok(price)
    }

    /// Computes liquidity metrics
    pub fn compute_liquidity_metrics(&self) -> Result<LiquidityMetrics, LOBError> {
        let spread = self.compute_spread()?;
        let depth = self.compute_depth()?;
        let volume = self.compute_volume()?;
        let volatility = self.compute_volatility()?;
        let imbalance = self.compute_order_imbalance()?;
        
        // Compute realized spread
        let trades = self.get_recent_trades(100)?;
        let realized_spread = if trades.len() >= 2 {
            trades.windows(2)
                .map(|w| (w[1].price - w[0].price).abs())
                .sum::<f64>() / (trades.len() - 1) as f64
        } else {
            spread
        };
        
        // Compute price impact
        let price_impact = trades.windows(2)
            .map(|w| {
                let mid_change = (w[1].price - w[0].price) / w[0].price;
                let sign = if w[0].is_buy { 1.0 } else { -1.0 };
                sign * mid_change
            })
            .sum::<f64>() / trades.len() as f64;
            
        // Compute market resiliency
        let resiliency = self.compute_market_resiliency()?;
        
        Ok(LiquidityMetrics {
            spread,
            realized_spread,
            depth,
            volume,
            price_impact,
            resiliency,
            volatility,
            imbalance,
        })
    }

    /// Computes market resiliency
    fn compute_market_resiliency(&self) -> Result<f64, LOBError> {
        let trades = self.get_recent_trades(100)?;
        if trades.len() < 2 {
            return Ok(1.0);
        }
        
        // Compute average time for spread to return to normal after large trades
        let mean_spread = self.compute_spread()?;
        let mut recovery_times = Vec::new();
        let mut current_deviation = 0.0;
        let mut deviation_start = 0.0;
        
        for (i, trade) in trades.iter().enumerate() {
            let spread_at_trade = trade.price_impact.abs();
            
            if spread_at_trade > 2.0 * mean_spread {
                if current_deviation == 0.0 {
                    deviation_start = trade.timestamp;
                }
                current_deviation = spread_at_trade;
            } else if current_deviation > 0.0 {
                recovery_times.push(trade.timestamp - deviation_start);
                current_deviation = 0.0;
            }
        }
        
        if recovery_times.is_empty() {
            Ok(1.0)
        } else {
            let mean_recovery = recovery_times.iter().sum::<f64>() / 
                recovery_times.len() as f64;
            Ok((-mean_recovery).exp())
        }
    }

    /// Submit an order with latency modeling
    pub fn submit_order_with_latency(&mut self, order: Order, latency: f64) {
        let submit_time = self.current_time + latency;
        self.latency_queue.push_back(LatencyOrder { order, submit_time });
    }

    /// Process latency queue, moving orders into the book when their time arrives
    pub fn process_latency_queue(&mut self) {
        while let Some(lat_order) = self.latency_queue.front() {
            if lat_order.submit_time <= self.current_time {
                let lat_order = self.latency_queue.pop_front().unwrap();
                self.insert_order(lat_order.order);
            } else {
                break;
            }
        }
    }

    /// Insert order into the book, handling advanced order types
    pub fn insert_order(&mut self, order: Order) {
        match &order.order_type {
            AdvancedOrderType::Standard => {
                self.insert_standard_order(order);
            }
            AdvancedOrderType::Iceberg { peak } => {
                // Only show the peak in the book, keep the rest hidden
                let visible = Order { size: FixedPoint::from_float(*peak).min(order.size), ..order.clone() };
                let hidden_size = order.size - visible.size;
                self.insert_standard_order(visible);
                if hidden_size > FixedPoint::zero() {
                    let mut hidden = order.clone();
                    hidden.size = FixedPoint::from_float(hidden_size);
                    self.hidden_orders.push(hidden);
                }
            }
            AdvancedOrderType::Pegged { reference, offset } => {
                // Store pegged orders for periodic re-pricing
                self.pegged_orders.push(order);
            }
            AdvancedOrderType::PostOnly => {
                // Only add if it does not cross the spread
                let (best_bid, best_ask) = self.get_best_bid_ask();
                if (order.is_buy && (best_ask.is_none() || order.price < best_ask.unwrap())) ||
                   (!order.is_buy && (best_bid.is_none() || order.price > best_bid.unwrap())) {
                    self.insert_standard_order(order);
                }
                // else: cancel silently
            }
            AdvancedOrderType::Hidden => {
                // Not visible in book, but available for matching
                self.hidden_orders.push(order);
            }
        }
    }

    /// Helper: insert standard order into the book
    fn insert_standard_order(&mut self, order: Order) {
        let price_level = (order.price / FixedPoint::from_float(self.params.tick_size)).round() as i64;
        let book = if order.is_buy { &mut self.buy_orders } else { &mut self.sell_orders };
        book.entry(price_level).or_insert_with(VecDeque::new).push_back(order);
    }

    /// Get best bid and ask prices
    fn get_best_bid_ask(&self) -> (Option<f64>, Option<f64>) {
        let best_bid = self.buy_orders.iter().next_back().map(|(&p, _)| p as f64 * self.params.tick_size);
        let best_ask = self.sell_orders.iter().next().map(|(&p, _)| p as f64 * self.params.tick_size);
        (best_bid, best_ask)
    }

    /// Match hidden orders against the book
    fn match_hidden_orders(&mut self) {
        let mut remaining_hidden = Vec::new();
        for mut order in self.hidden_orders.drain(..) {
            let price_level = (order.price / FixedPoint::from_float(self.params.tick_size)).round() as i64;
            let book = if order.is_buy { &mut self.sell_orders } else { &mut self.buy_orders };
            let mut matched = false;
            if let Some(queue) = book.get_mut(&price_level) {
                while let Some(mut counter_order) = queue.pop_front() {
                    let match_size = order.size.min(counter_order.size);
                    order.size -= match_size;
                    counter_order.size -= match_size;
                    if counter_order.size > FixedPoint::zero() {
                        queue.push_front(counter_order);
                        break;
                    }
                    if order.size <= FixedPoint::zero() {
                        matched = true;
                        break;
                    }
                }
                if queue.is_empty() {
                    book.remove(&price_level);
                }
            }
            if !matched && order.size > FixedPoint::zero() {
                remaining_hidden.push(order);
            }
        }
        self.hidden_orders = remaining_hidden;
    }

    /// Process all pegged orders: update their price and insert if marketable
    pub fn process_pegged_orders(&mut self) {
        let mut remaining_pegged = Vec::new();
        for mut order in self.pegged_orders.drain(..) {
            // Determine new price based on reference
            let reference_price = match &order.order_type {
                AdvancedOrderType::Pegged { reference, offset } => {
                    let (best_bid, best_ask) = self.get_best_bid_ask();
                    match reference {
                        PeggedReference::BestBid => best_bid.map(|p| p + *offset),
                        PeggedReference::BestAsk => best_ask.map(|p| p + *offset),
                        PeggedReference::Mid => {
                            match (best_bid, best_ask) {
                                (Some(bid), Some(ask)) => Some(((bid + ask) / 2.0) + *offset),
                                _ => None,
                            }
                        }
                    }
                }
                _ => None,
            };
            if let Some(new_price) = reference_price {
                order.price = FixedPoint::from_float(new_price);
                // Check if order is now marketable (crosses the spread)
                let (best_bid, best_ask) = self.get_best_bid_ask();
                let marketable = if order.is_buy {
                    best_ask.map_or(false, |ask| order.price >= FixedPoint::from_float(ask))
                } else {
                    best_bid.map_or(false, |bid| order.price <= FixedPoint::from_float(bid))
                };
                if marketable {
                    // Insert as a standard order for matching
                    let mut std_order = order.clone();
                    std_order.order_type = AdvancedOrderType::Standard;
                    self.insert_standard_order(std_order);
                } else {
                    // Not marketable, keep pegged for next round
                    remaining_pegged.push(order);
                }
            } else {
                // No reference price available, keep pegged
                remaining_pegged.push(order);
            }
        }
        self.pegged_orders = remaining_pegged;
    }
}

#[derive(Debug)]
pub struct MarketImpactParams {
    pub permanent_impact: f64,
    pub temporary_impact: f64,
    pub decay_rate: f64,
    pub nonlinear_factor: f64,
}

#[derive(Debug)]
pub struct LiquidityMetrics {
    pub spread: f64,
    pub realized_spread: f64,
    pub depth: f64,
    pub volume: f64,
    pub price_impact: f64,
    pub resiliency: f64,
    pub volatility: f64,
    pub imbalance: f64,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market(bool),  // bool indicates if buy
    Limit(bool),
}

#[derive(Debug, Clone)]
pub struct BookState {
    pub buy_levels: BTreeMap<i64, VecDeque<Order>>,
    pub sell_levels: BTreeMap<i64, VecDeque<Order>>,
}

#[derive(Debug)]
pub struct LOBSimulation {
    pub mid_prices: Vec<f64>,
    pub spreads: Vec<f64>,
    pub depths: Vec<f64>,
    pub volumes: Vec<f64>,
    pub final_state: BookState,
}

#[derive(Debug)]
pub struct TradeExecution {
    pub price: f64,
    pub size: f64,
    pub timestamp: f64,
    pub is_buy: bool,
    pub price_impact: f64,
}

// Latency modeling: simple struct for delayed order entry
#[derive(Debug, Clone)]
pub struct LatencyOrder {
    pub order: Order,
    pub submit_time: f64, // When the order should be entered into the book
}
