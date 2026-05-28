//! Tests for market data generation functionality

use super::market_data::*;
use super::types::{CentralLimitOrderBook, Order, OrderId, Symbol, Side, Trade};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_symbol() -> Symbol {
        Symbol::new("BTCUSD").unwrap()
    }

    fn create_test_order_book() -> CentralLimitOrderBook {
        CentralLimitOrderBook::new(create_test_symbol())
    }

    fn create_test_trade(id: u64, price: u64, size: u64, timestamp: u64) -> Trade {
        Trade::new(
            id,
            create_test_symbol(),
            price,
            size,
            timestamp,
            OrderId::new(1),
            OrderId::new(2),
            false,
            id,
        )
    }

    #[test]
    fn test_market_data_generator_creation() {
        let symbol = create_test_symbol();
        let generator = MarketDataGenerator::new(symbol.clone());
        
        assert_eq!(generator.symbol, symbol);
        assert_eq!(generator.best_bid_ask.best_bid, None);
        assert_eq!(generator.best_bid_ask.best_ask, None);
        assert_eq!(generator.trade_tick_generator.current_vwap, None);
    }

    #[test]
    fn test_best_bid_ask_tracker() {
        let mut order_book = create_test_order_book();
        let mut tracker = BestBidAskTracker::new();
        
        // Initially no best bid/ask
        tracker.update_from_order_book(&order_book).unwrap();
        assert_eq!(tracker.best_bid, None);
        assert_eq!(tracker.best_ask, None);
        assert_eq!(tracker.spread, None);
        assert_eq!(tracker.mid_price, None);
        
        // Add a buy order
        let buy_order = Order::new_limit(
            OrderId::new(1),
            create_test_symbol(),
            Side::Buy,
            50000 * super::super::types::PRICE_SCALE, // $50,000
            100 * super::super::types::VOLUME_SCALE,   // 100 units
            1000,
        );
        
        order_book.add_order(buy_order).unwrap();
        tracker.update_from_order_book(&order_book).unwrap();
        
        assert_eq!(tracker.best_bid, Some(50000 * super::super::types::PRICE_SCALE));
        assert_eq!(tracker.best_ask, None);
        assert_eq!(tracker.spread, None);
        assert_eq!(tracker.mid_price, None);
        
        // Add a sell order
        let sell_order = Order::new_limit(
            OrderId::new(2),
            create_test_symbol(),
            Side::Sell,
            50100 * super::super::types::PRICE_SCALE, // $50,100
            50 * super::super::types::VOLUME_SCALE,    // 50 units
            1001,
        );
        
        order_book.add_order(sell_order).unwrap();
        tracker.update_from_order_book(&order_book).unwrap();
        
        let expected_bid = 50000 * super::super::types::PRICE_SCALE;
        let expected_ask = 50100 * super::super::types::PRICE_SCALE;
        let expected_spread = expected_ask - expected_bid;
        let expected_mid = (expected_ask + expected_bid) / 2;
        
        assert_eq!(tracker.best_bid, Some(expected_bid));
        assert_eq!(tracker.best_ask, Some(expected_ask));
        assert_eq!(tracker.spread, Some(expected_spread));
        assert_eq!(tracker.mid_price, Some(expected_mid));
    }

    #[test]
    fn test_trade_tick_generator() {
        let mut generator = TradeTickGenerator::new();
        let symbol = create_test_symbol();
        
        // Create a test trade
        let trade = create_test_trade(1, 50000, 100, 1000);
        
        // Generate tick
        let tick = generator.generate_tick(&trade, &symbol).unwrap();
        
        assert_eq!(tick.trade_id, 1);
        assert_eq!(tick.symbol, symbol);
        assert_eq!(tick.price, 50000);
        assert_eq!(tick.size, 100);
        assert_eq!(tick.timestamp, 1000);
        assert_eq!(tick.sequence, 1);
        assert_eq!(tick.vwap, 50000); // First trade, VWAP equals trade price
        
        // Add another trade
        let trade2 = create_test_trade(2, 50200, 200, 1001);
        let tick2 = generator.generate_tick(&trade2, &symbol).unwrap();
        
        // VWAP should be weighted average: (50000*100 + 50200*200) / (100+200) = 50133.33
        let expected_vwap = (50000 * 100 + 50200 * 200) / (100 + 200);
        assert_eq!(tick2.vwap, expected_vwap);
        assert_eq!(tick2.sequence, 2);
    }

    #[test]
    fn test_vwap_window_cleanup() {
        let mut generator = TradeTickGenerator::new();
        generator.vwap_window_ns = 1000; // 1 microsecond window for testing
        let symbol = create_test_symbol();
        
        // Add old trade
        let old_trade = create_test_trade(1, 50000, 100, 1000);
        generator.generate_tick(&old_trade, &symbol).unwrap();
        
        // Add new trade outside window
        let new_trade = create_test_trade(2, 50200, 200, 3000); // 2 microseconds later
        let tick = generator.generate_tick(&new_trade, &symbol).unwrap();
        
        // VWAP should only include new trade since old one is outside window
        assert_eq!(tick.vwap, 50200);
        assert_eq!(generator.recent_trades.len(), 1);
    }

    #[test]
    fn test_market_statistics() {
        let mut stats = MarketStatistics::new();
        
        // Process first trade
        let trade1 = create_test_trade(1, 50000, 100, 1000);
        stats.process_trade(&trade1).unwrap();
        
        assert_eq!(stats.total_trades, 1);
        assert_eq!(stats.total_volume, 100);
        assert_eq!(stats.last_price, Some(50000));
        
        // Check OHLCV
        let ohlcv = stats.current_ohlcv.as_ref().unwrap();
        assert_eq!(ohlcv.open, 50000);
        assert_eq!(ohlcv.high, 50000);
        assert_eq!(ohlcv.low, 50000);
        assert_eq!(ohlcv.close, 50000);
        assert_eq!(ohlcv.volume, 100);
        assert_eq!(ohlcv.trade_count, 1);
        assert_eq!(ohlcv.vwap, 50000);
        
        // Process second trade in same period
        let trade2 = create_test_trade(2, 50200, 200, 1500);
        stats.process_trade(&trade2).unwrap();
        
        let ohlcv = stats.current_ohlcv.as_ref().unwrap();
        assert_eq!(ohlcv.open, 50000);
        assert_eq!(ohlcv.high, 50200);
        assert_eq!(ohlcv.low, 50000);
        assert_eq!(ohlcv.close, 50200);
        assert_eq!(ohlcv.volume, 300);
        assert_eq!(ohlcv.trade_count, 2);
        
        // VWAP should be (50000*100 + 50200*200) / 300
        let expected_vwap = (50000 * 100 + 50200 * 200) / 300;
        assert_eq!(ohlcv.vwap, expected_vwap);
    }

    #[test]
    fn test_ohlcv_period_transition() {
        let mut stats = MarketStatistics::new();
        stats.ohlcv_period_ns = 1000; // 1 microsecond periods for testing
        
        // First trade in period 1
        let trade1 = create_test_trade(1, 50000, 100, 1000);
        stats.process_trade(&trade1).unwrap();
        
        // Second trade in period 2 (should create new OHLCV)
        let trade2 = create_test_trade(2, 50200, 200, 2000);
        stats.process_trade(&trade2).unwrap();
        
        // Should have moved first OHLCV to history
        assert_eq!(stats.ohlcv_history.len(), 1);
        
        let historical_ohlcv = &stats.ohlcv_history[0];
        assert_eq!(historical_ohlcv.open, 50000);
        assert_eq!(historical_ohlcv.close, 50000);
        assert_eq!(historical_ohlcv.volume, 100);
        
        // Current OHLCV should be for second trade
        let current_ohlcv = stats.current_ohlcv.as_ref().unwrap();
        assert_eq!(current_ohlcv.open, 50200);
        assert_eq!(current_ohlcv.close, 50200);
        assert_eq!(current_ohlcv.volume, 200);
    }

    #[test]
    fn test_volume_profile() {
        let mut profile = VolumeProfile::new();
        
        // Add trades at different prices
        let trade1 = create_test_trade(1, 50000, 100, 1000);
        let trade2 = create_test_trade(2, 50000, 200, 1001); // Same price
        let trade3 = create_test_trade(3, 50100, 150, 1002); // Different price
        
        profile.add_trade(&trade1);
        profile.add_trade(&trade2);
        profile.add_trade(&trade3);
        
        assert_eq!(profile.total_volume, 450);
        assert_eq!(profile.price_volumes.get(&50000), Some(&300)); // 100 + 200
        assert_eq!(profile.price_volumes.get(&50100), Some(&150));
        
        // POC should be 50000 (highest volume)
        assert_eq!(profile.poc, Some(50000));
        
        // Check time period
        assert_eq!(profile.period_start, 1000);
        assert_eq!(profile.period_end, 1002);
    }

    #[test]
    fn test_daily_statistics() {
        let mut daily_stats = DailyStatistics::new();
        
        // Process trades
        let trade1 = create_test_trade(1, 50000, 100, 1000);
        let trade2 = create_test_trade(2, 50200, 200, 1001);
        let trade3 = create_test_trade(3, 49800, 150, 1002);
        
        daily_stats.update_with_trade(&trade1);
        daily_stats.update_with_trade(&trade2);
        daily_stats.update_with_trade(&trade3);
        
        assert_eq!(daily_stats.high_24h, Some(50200));
        assert_eq!(daily_stats.low_24h, Some(49800));
        assert_eq!(daily_stats.volume_24h, 450);
        assert_eq!(daily_stats.trade_count_24h, 3);
        assert_eq!(daily_stats.last_update, 1002);
    }

    #[test]
    fn test_market_data_integration() {
        let mut generator = MarketDataGenerator::new(create_test_symbol());
        let mut order_book = create_test_order_book();
        
        // Add orders to create market depth
        let buy_order = Order::new_limit(
            OrderId::new(1),
            create_test_symbol(),
            Side::Buy,
            50000,
            100,
            1000,
        );
        let sell_order = Order::new_limit(
            OrderId::new(2),
            create_test_symbol(),
            Side::Sell,
            50100,
            150,
            1001,
        );
        
        order_book.add_order(buy_order).unwrap();
        order_book.add_order(sell_order).unwrap();
        
        // Update market data from order book
        generator.update_from_order_book(&order_book).unwrap();
        
        // Check best bid/ask
        let (best_bid, best_ask) = generator.get_best_bid_ask();
        assert_eq!(best_bid, Some(50000));
        assert_eq!(best_ask, Some(50100));
        assert_eq!(generator.get_spread(), Some(100));
        assert_eq!(generator.get_mid_price(), Some(50050));
        
        // Process a trade
        let trade = create_test_trade(1, 50050, 75, 1002);
        let tick = generator.process_trade(&trade).unwrap();
        
        assert_eq!(tick.price, 50050);
        assert_eq!(tick.size, 75);
        assert_eq!(tick.vwap, 50050);
        
        // Check market statistics were updated
        assert_eq!(generator.market_stats.total_trades, 1);
        assert_eq!(generator.market_stats.total_volume, 75);
        
        // Check volume profile
        assert_eq!(generator.volume_profile.total_volume, 75);
        assert_eq!(generator.volume_profile.poc, Some(50050));
    }

    #[test]
    fn test_market_depth_calculation() {
        let generator = MarketDataGenerator::new(create_test_symbol());
        let mut order_book = create_test_order_book();
        
        // Add multiple orders at different price levels
        let orders = vec![
            Order::new_limit(OrderId::new(1), create_test_symbol(), Side::Buy, 50000, 100, 1000),
            Order::new_limit(OrderId::new(2), create_test_symbol(), Side::Buy, 49900, 200, 1001),
            Order::new_limit(OrderId::new(3), create_test_symbol(), Side::Sell, 50100, 150, 1002),
            Order::new_limit(OrderId::new(4), create_test_symbol(), Side::Sell, 50200, 250, 1003),
        ];
        
        for order in orders {
            order_book.add_order(order).unwrap();
        }
        
        // Get market depth
        let depth = generator.get_market_depth(&order_book, 2);
        
        // Check bid levels (highest price first)
        assert_eq!(depth.bids.len(), 2);
        assert_eq!(depth.bids[0], (50000, 100)); // Best bid
        assert_eq!(depth.bids[1], (49900, 200)); // Second best bid
        
        // Check ask levels (lowest price first)
        assert_eq!(depth.asks.len(), 2);
        assert_eq!(depth.asks[0], (50100, 150)); // Best ask
        assert_eq!(depth.asks[1], (50200, 250)); // Second best ask
        
        // Check depth metadata
        assert_eq!(depth.best_bid(), Some(50000));
        assert_eq!(depth.best_ask(), Some(50100));
        assert_eq!(depth.spread(), Some(100));
        assert_eq!(depth.mid_price(), Some(50050));
    }

    #[test]
    fn test_utils_vwap_calculation() {
        let trades = vec![
            create_test_trade(1, 50000, 100, 1000),
            create_test_trade(2, 50200, 200, 1001),
            create_test_trade(3, 49800, 150, 1002),
        ];
        
        let vwap = utils::calculate_vwap(&trades).unwrap();
        
        // Expected: (50000*100 + 50200*200 + 49800*150) / (100+200+150)
        let expected = (50000 * 100 + 50200 * 200 + 49800 * 150) / 450;
        assert_eq!(vwap, expected);
    }

    #[test]
    fn test_utils_twap_calculation() {
        let trades = vec![
            create_test_trade(1, 50000, 100, 1000),
            create_test_trade(2, 50200, 200, 1001),
            create_test_trade(3, 49800, 150, 1002),
        ];
        
        let twap = utils::calculate_twap(&trades).unwrap();
        
        // Expected: (50000 + 50200 + 49800) / 3
        let expected = (50000 + 50200 + 49800) / 3;
        assert_eq!(twap, expected);
    }

    #[test]
    fn test_utils_volatility_calculation() {
        let prices = vec![50000, 50200, 49800, 50100, 49900];
        let volatility = utils::calculate_volatility(&prices);
        
        // Should return some volatility value
        assert!(volatility.is_some());
        assert!(volatility.unwrap() > 0.0);
    }

    #[test]
    fn test_empty_collections() {
        // Test empty trade collections
        assert_eq!(utils::calculate_vwap(&[]), None);
        assert_eq!(utils::calculate_twap(&[]), None);
        
        // Test single price volatility
        assert_eq!(utils::calculate_volatility(&[50000]), None);
        
        // Test empty volume profile
        let profile = VolumeProfile::new();
        assert_eq!(profile.total_volume, 0);
        assert_eq!(profile.poc, None);
    }
}