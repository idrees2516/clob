//! Integration tests for market data generation with the CLOB system

use hf_quoting_liquidity_clob::orderbook::{
    CentralLimitOrderBook, Order, OrderId, Symbol, Side, Trade,
    MarketDataGenerator, MatchingEngine, PRICE_SCALE, VOLUME_SCALE
};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn test_market_data_integration_with_order_book() {
    // Create test components
    let symbol = Symbol::new("ETHUSD").unwrap();
    let mut order_book = CentralLimitOrderBook::new(symbol.clone());
    let mut market_data = MarketDataGenerator::new(symbol.clone());
    let mut matching_engine = MatchingEngine::new();
    
    // Set deterministic timestamp
    let base_time = 1000000000u64;
    matching_engine.set_current_time(base_time);
    
    // Add initial liquidity
    let buy_order = Order::new_limit(
        OrderId::new(1),
        symbol.clone(),
        Side::Buy,
        2000 * PRICE_SCALE, // $2000
        10 * VOLUME_SCALE,   // 10 ETH
        base_time,
    );
    
    let sell_order = Order::new_limit(
        OrderId::new(2),
        symbol.clone(),
        Side::Sell,
        2010 * PRICE_SCALE, // $2010
        8 * VOLUME_SCALE,    // 8 ETH
        base_time + 1000,
    );
    
    // Process orders through matching engine
    let buy_result = matching_engine.process_order(&mut order_book, buy_order).unwrap();
    let sell_result = matching_engine.process_order(&mut order_book, sell_order).unwrap();
    
    // Verify orders were added (no immediate matches)
    assert_eq!(buy_result.filled_size, 0);
    assert_eq!(sell_result.filled_size, 0);
    
    // Update market data from order book state
    market_data.update_from_order_book(&order_book).unwrap();
    
    // Verify best bid/ask tracking
    let (best_bid, best_ask) = market_data.get_best_bid_ask();
    assert_eq!(best_bid, Some(2000 * PRICE_SCALE));
    assert_eq!(best_ask, Some(2010 * PRICE_SCALE));
    assert_eq!(market_data.get_spread(), Some(10 * PRICE_SCALE));
    assert_eq!(market_data.get_mid_price(), Some(2005 * PRICE_SCALE));
    
    // Execute a market order that will generate trades
    let market_buy = Order::new_market(
        OrderId::new(3),
        symbol.clone(),
        Side::Buy,
        5 * VOLUME_SCALE, // 5 ETH
        base_time + 2000,
    );
    
    matching_engine.set_current_time(base_time + 2000);
    let market_result = matching_engine.process_order(&mut order_book, market_buy).unwrap();
    
    // Verify trade was executed
    assert_eq!(market_result.filled_size, 5 * VOLUME_SCALE);
    assert_eq!(market_result.trades.len(), 1);
    
    let trade = &market_result.trades[0];
    assert_eq!(trade.price, 2010 * PRICE_SCALE); // Matched at ask price
    assert_eq!(trade.size, 5 * VOLUME_SCALE);
    
    // Process trade through market data generator
    let tick = market_data.process_trade(trade).unwrap();
    
    // Verify trade tick
    assert_eq!(tick.price, 2010 * PRICE_SCALE);
    assert_eq!(tick.size, 5 * VOLUME_SCALE);
    assert_eq!(tick.vwap, 2010 * PRICE_SCALE); // First trade, VWAP equals price
    assert_eq!(tick.sequence, 1);
    assert_eq!(tick.side, Side::Buy); // Taker side
    
    // Update market data after trade
    market_data.update_from_order_book(&order_book).unwrap();
    
    // Verify updated best bid/ask (ask should have reduced size)
    let (updated_bid, updated_ask) = market_data.get_best_bid_ask();
    assert_eq!(updated_bid, Some(2000 * PRICE_SCALE)); // Bid unchanged
    assert_eq!(updated_ask, Some(2010 * PRICE_SCALE)); // Ask price same, but size reduced
    
    // Verify market statistics
    assert_eq!(market_data.market_stats.total_trades, 1);
    assert_eq!(market_data.market_stats.total_volume, 5 * VOLUME_SCALE);
    assert_eq!(market_data.market_stats.last_price, Some(2010 * PRICE_SCALE));
    
    // Verify OHLCV data
    let ohlcv = market_data.get_current_ohlcv().unwrap();
    assert_eq!(ohlcv.open, 2010 * PRICE_SCALE);
    assert_eq!(ohlcv.high, 2010 * PRICE_SCALE);
    assert_eq!(ohlcv.low, 2010 * PRICE_SCALE);
    assert_eq!(ohlcv.close, 2010 * PRICE_SCALE);
    assert_eq!(ohlcv.volume, 5 * VOLUME_SCALE);
    assert_eq!(ohlcv.trade_count, 1);
    assert_eq!(ohlcv.vwap, 2010 * PRICE_SCALE);
    
    // Verify volume profile
    let profile = market_data.get_volume_profile();
    assert_eq!(profile.total_volume, 5 * VOLUME_SCALE);
    assert_eq!(profile.poc, Some(2010 * PRICE_SCALE));
    assert_eq!(profile.price_volumes.get(&(2010 * PRICE_SCALE)), Some(&(5 * VOLUME_SCALE)));
    
    // Verify daily statistics
    let daily_stats = market_data.get_daily_stats();
    assert_eq!(daily_stats.high_24h, Some(2010 * PRICE_SCALE));
    assert_eq!(daily_stats.low_24h, Some(2010 * PRICE_SCALE));
    assert_eq!(daily_stats.volume_24h, 5 * VOLUME_SCALE);
    assert_eq!(daily_stats.trade_count_24h, 1);
}

#[test]
fn test_multiple_trades_vwap_calculation() {
    let symbol = Symbol::new("BTCUSD").unwrap();
    let mut market_data = MarketDataGenerator::new(symbol.clone());
    
    // Create multiple trades at different prices
    let trades = vec![
        Trade::new(
            1,
            symbol.clone(),
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            1000,
            OrderId::new(1),
            OrderId::new(2),
            false,
            1,
        ),
        Trade::new(
            2,
            symbol.clone(),
            50200 * PRICE_SCALE,
            200 * VOLUME_SCALE,
            1001,
            OrderId::new(3),
            OrderId::new(4),
            true,
            2,
        ),
        Trade::new(
            3,
            symbol.clone(),
            49800 * PRICE_SCALE,
            150 * VOLUME_SCALE,
            1002,
            OrderId::new(5),
            OrderId::new(6),
            false,
            3,
        ),
    ];
    
    let mut expected_vwap_values = Vec::new();
    
    // Process trades and verify VWAP calculation
    for (i, trade) in trades.iter().enumerate() {
        let tick = market_data.process_trade(trade).unwrap();
        
        // Calculate expected VWAP manually
        let total_notional: u64 = trades[..=i].iter()
            .map(|t| t.price * t.size / PRICE_SCALE)
            .sum();
        let total_volume: u64 = trades[..=i].iter()
            .map(|t| t.size)
            .sum();
        let expected_vwap = total_notional * PRICE_SCALE / total_volume;
        
        expected_vwap_values.push(expected_vwap);
        
        // Verify VWAP in tick
        assert_eq!(tick.vwap, expected_vwap);
        assert_eq!(tick.sequence, (i + 1) as u64);
    }
    
    // Verify final VWAP
    let final_vwap = market_data.get_current_vwap().unwrap();
    assert_eq!(final_vwap, *expected_vwap_values.last().unwrap());
}

#[test]
fn test_market_depth_with_multiple_levels() {
    let symbol = Symbol::new("ADAUSD").unwrap();
    let mut order_book = CentralLimitOrderBook::new(symbol.clone());
    let market_data = MarketDataGenerator::new(symbol.clone());
    
    // Add multiple orders at different price levels
    let orders = vec![
        // Bids (descending price order expected)
        Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 100 * PRICE_SCALE, 1000 * VOLUME_SCALE, 1000),
        Order::new_limit(OrderId::new(2), symbol.clone(), Side::Buy, 99 * PRICE_SCALE, 1500 * VOLUME_SCALE, 1001),
        Order::new_limit(OrderId::new(3), symbol.clone(), Side::Buy, 98 * PRICE_SCALE, 2000 * VOLUME_SCALE, 1002),
        
        // Asks (ascending price order expected)
        Order::new_limit(OrderId::new(4), symbol.clone(), Side::Sell, 101 * PRICE_SCALE, 800 * VOLUME_SCALE, 1003),
        Order::new_limit(OrderId::new(5), symbol.clone(), Side::Sell, 102 * PRICE_SCALE, 1200 * VOLUME_SCALE, 1004),
        Order::new_limit(OrderId::new(6), symbol.clone(), Side::Sell, 103 * PRICE_SCALE, 900 * VOLUME_SCALE, 1005),
    ];
    
    // Add all orders
    for order in orders {
        order_book.add_order(order).unwrap();
    }
    
    // Get market depth
    let depth = market_data.get_market_depth(&order_book, 3);
    
    // Verify bid levels (highest price first)
    assert_eq!(depth.bids.len(), 3);
    assert_eq!(depth.bids[0], (100 * PRICE_SCALE, 1000 * VOLUME_SCALE)); // Best bid
    assert_eq!(depth.bids[1], (99 * PRICE_SCALE, 1500 * VOLUME_SCALE));
    assert_eq!(depth.bids[2], (98 * PRICE_SCALE, 2000 * VOLUME_SCALE));
    
    // Verify ask levels (lowest price first)
    assert_eq!(depth.asks.len(), 3);
    assert_eq!(depth.asks[0], (101 * PRICE_SCALE, 800 * VOLUME_SCALE)); // Best ask
    assert_eq!(depth.asks[1], (102 * PRICE_SCALE, 1200 * VOLUME_SCALE));
    assert_eq!(depth.asks[2], (103 * PRICE_SCALE, 900 * VOLUME_SCALE));
    
    // Verify depth calculations
    assert_eq!(depth.best_bid(), Some(100 * PRICE_SCALE));
    assert_eq!(depth.best_ask(), Some(101 * PRICE_SCALE));
    assert_eq!(depth.spread(), Some(1 * PRICE_SCALE));
    assert_eq!(depth.mid_price(), Some(100 * PRICE_SCALE + PRICE_SCALE / 2)); // 100.5
}

#[test]
fn test_ohlcv_period_transitions() {
    let symbol = Symbol::new("SOLUSD").unwrap();
    let mut market_data = MarketDataGenerator::new(symbol.clone());
    
    // Set short OHLCV period for testing
    market_data.market_stats.ohlcv_period_ns = 1000; // 1 microsecond
    
    // Create trades in different periods
    let trades = vec![
        // Period 1 (timestamp 1000-1999)
        Trade::new(1, symbol.clone(), 100 * PRICE_SCALE, 50 * VOLUME_SCALE, 1000, OrderId::new(1), OrderId::new(2), false, 1),
        Trade::new(2, symbol.clone(), 105 * PRICE_SCALE, 75 * VOLUME_SCALE, 1500, OrderId::new(3), OrderId::new(4), true, 2),
        
        // Period 2 (timestamp 2000-2999)
        Trade::new(3, symbol.clone(), 102 * PRICE_SCALE, 100 * VOLUME_SCALE, 2000, OrderId::new(5), OrderId::new(6), false, 3),
        Trade::new(4, symbol.clone(), 98 * PRICE_SCALE, 60 * VOLUME_SCALE, 2500, OrderId::new(7), OrderId::new(8), true, 4),
    ];
    
    // Process first two trades (same period)
    market_data.process_trade(&trades[0]).unwrap();
    market_data.process_trade(&trades[1]).unwrap();
    
    // Verify current OHLCV for period 1
    let ohlcv1 = market_data.get_current_ohlcv().unwrap();
    assert_eq!(ohlcv1.open, 100 * PRICE_SCALE);
    assert_eq!(ohlcv1.high, 105 * PRICE_SCALE);
    assert_eq!(ohlcv1.low, 100 * PRICE_SCALE);
    assert_eq!(ohlcv1.close, 105 * PRICE_SCALE);
    assert_eq!(ohlcv1.volume, 125 * VOLUME_SCALE);
    assert_eq!(ohlcv1.trade_count, 2);
    
    // Process third trade (new period)
    market_data.process_trade(&trades[2]).unwrap();
    
    // Verify period 1 moved to history
    assert_eq!(market_data.market_stats.ohlcv_history.len(), 1);
    let historical_ohlcv = &market_data.market_stats.ohlcv_history[0];
    assert_eq!(historical_ohlcv.open, 100 * PRICE_SCALE);
    assert_eq!(historical_ohlcv.close, 105 * PRICE_SCALE);
    assert_eq!(historical_ohlcv.volume, 125 * VOLUME_SCALE);
    
    // Verify new current OHLCV for period 2
    let ohlcv2 = market_data.get_current_ohlcv().unwrap();
    assert_eq!(ohlcv2.open, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2.high, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2.low, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2.close, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2.volume, 100 * VOLUME_SCALE);
    assert_eq!(ohlcv2.trade_count, 1);
    
    // Process fourth trade (same period)
    market_data.process_trade(&trades[3]).unwrap();
    
    // Verify updated OHLCV for period 2
    let ohlcv2_updated = market_data.get_current_ohlcv().unwrap();
    assert_eq!(ohlcv2_updated.open, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2_updated.high, 102 * PRICE_SCALE);
    assert_eq!(ohlcv2_updated.low, 98 * PRICE_SCALE);
    assert_eq!(ohlcv2_updated.close, 98 * PRICE_SCALE);
    assert_eq!(ohlcv2_updated.volume, 160 * VOLUME_SCALE);
    assert_eq!(ohlcv2_updated.trade_count, 2);
}