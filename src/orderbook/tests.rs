use super::types::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// Helper function to create a test timestamp
fn test_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Helper function to create a test symbol
fn test_symbol() -> Symbol {
    Symbol::new("BTCUSD").unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_id_creation_and_operations() {
        let id = OrderId::new(12345);
        assert_eq!(id.as_u64(), 12345);
        
        let next_id = id.next();
        assert_eq!(next_id.as_u64(), 12346);
        
        // Test ordering
        assert!(id < next_id);
        
        // Test display
        assert_eq!(format!("{}", id), "#12345");
        
        // Test conversions
        let from_u64: OrderId = 67890u64.into();
        assert_eq!(from_u64.as_u64(), 67890);
        
        let to_u64: u64 = from_u64.into();
        assert_eq!(to_u64, 67890);
    }

    #[test]
    fn test_symbol_creation_and_validation() {
        // Valid symbols
        assert!(Symbol::new("BTCUSD").is_ok());
        assert!(Symbol::new("ETHUSD").is_ok());
        assert!(Symbol::new("btcusd").is_ok()); // Should normalize to uppercase
        assert!(Symbol::new(" BTCUSD ").is_ok()); // Should trim whitespace
        
        // Invalid symbols
        assert!(Symbol::new("").is_err()); // Empty
        assert!(Symbol::new("BTC-USD").is_err()); // Invalid characters
        assert!(Symbol::new("A".repeat(25).as_str()).is_err()); // Too long
        
        // Test normalization
        let symbol = Symbol::new("btcusd").unwrap();
        assert_eq!(symbol.as_str(), "BTCUSD");
        
        // Test display
        let symbol = Symbol::new("ETHUSD").unwrap();
        assert_eq!(format!("{}", symbol), "ETHUSD");
    }

    #[test]
    fn test_order_creation_and_validation() {
        let timestamp = test_timestamp();
        let symbol = test_symbol();
        
        // Valid limit order
        let limit_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE, // $50,000
            100 * VOLUME_SCALE,   // 0.1 BTC
            timestamp,
        );
        
        assert!(limit_order.validate().is_ok());
        assert!(limit_order.is_buy());
        assert!(!limit_order.is_sell());
        assert!(limit_order.is_limit_order());
        assert!(!limit_order.is_market_order());
        
        // Valid market order
        let market_order = Order::new_market(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            50 * VOLUME_SCALE, // 0.05 BTC
            timestamp,
        );
        
        assert!(market_order.validate().is_ok());
        assert!(market_order.is_sell());
        assert!(!market_order.is_buy());
        assert!(market_order.is_market_order());
        assert!(!market_order.is_limit_order());
        
        // Invalid orders
        let mut invalid_order = limit_order.clone();
        invalid_order.size = 0; // Zero size
        assert!(invalid_order.validate().is_err());
        
        let mut invalid_limit = limit_order.clone();
        invalid_limit.price = 0; // Zero price for limit order
        assert!(invalid_limit.validate().is_err());
        
        let mut invalid_timestamp = limit_order.clone();
        invalid_timestamp.timestamp = 0; // Zero timestamp
        assert!(invalid_timestamp.validate().is_err());
    }

    #[test]
    fn test_price_level_operations() {
        let price = 50000 * PRICE_SCALE;
        let mut level = PriceLevel::new(price);
        let timestamp = test_timestamp();
        
        // Test empty level
        assert!(level.is_empty());
        assert_eq!(level.len(), 0);
        assert_eq!(level.total_size, 0);
        assert_eq!(level.order_count, 0);
        assert!(level.front_order_id().is_none());
        
        // Add orders
        let order_id_1 = OrderId::new(1);
        let order_id_2 = OrderId::new(2);
        let size_1 = 100 * VOLUME_SCALE;
        let size_2 = 200 * VOLUME_SCALE;
        
        assert!(level.add_order_id(order_id_1, size_1, timestamp).is_ok());
        assert!(level.add_order_id(order_id_2, size_2, timestamp + 1000).is_ok());
        
        // Test level state after additions
        assert!(!level.is_empty());
        assert_eq!(level.len(), 2);
        assert_eq!(level.total_size, size_1 + size_2);
        assert_eq!(level.order_count, 2);
        assert_eq!(level.front_order_id(), Some(order_id_1)); // FIFO order
        assert_eq!(level.timestamp, timestamp); // First order timestamp
        
        // Test validation
        assert!(level.validate().is_ok());
        
        // Remove first order
        assert!(level.remove_order_id(order_id_1, size_1, timestamp + 2000).is_ok());
        assert_eq!(level.len(), 1);
        assert_eq!(level.total_size, size_2);
        assert_eq!(level.order_count, 1);
        assert_eq!(level.front_order_id(), Some(order_id_2));
        
        // Pop front order
        let popped = level.pop_front_order_id(size_2, timestamp + 3000);
        assert_eq!(popped, Some(order_id_2));
        assert!(level.is_empty());
        assert_eq!(level.total_size, 0);
        assert_eq!(level.order_count, 0);
        
        // Test max orders limit
        let mut full_level = PriceLevel::new(price);
        for i in 0..MAX_ORDERS_PER_LEVEL {
            assert!(full_level.add_order_id(OrderId::new(i as u64), 100, timestamp).is_ok());
        }
        
        // Should fail to add one more
        assert!(full_level.add_order_id(OrderId::new(MAX_ORDERS_PER_LEVEL as u64), 100, timestamp).is_err());
    }

    #[test]
    fn test_trade_creation_and_methods() {
        let symbol = test_symbol();
        let timestamp = test_timestamp();
        let buyer_id = OrderId::new(1);
        let seller_id = OrderId::new(2);
        
        // Test buyer maker trade
        let trade = Trade::new(
            1,
            symbol.clone(),
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
            buyer_id,
            seller_id,
            true, // Buyer is maker
            1,
        );
        
        assert_eq!(trade.maker_order_id(), buyer_id);
        assert_eq!(trade.taker_order_id(), seller_id);
        
        // Test seller maker trade
        let trade2 = Trade::new(
            2,
            symbol.clone(),
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
            buyer_id,
            seller_id,
            false, // Seller is maker
            2,
        );
        
        assert_eq!(trade2.maker_order_id(), seller_id);
        assert_eq!(trade2.taker_order_id(), buyer_id);
        
        // Test notional value calculation
        let expected_notional = (50000 * PRICE_SCALE * 100 * VOLUME_SCALE) / PRICE_SCALE;
        assert_eq!(trade.notional_value(), expected_notional);
    }

    #[test]
    fn test_market_depth_operations() {
        let timestamp = test_timestamp();
        let sequence = 1;
        
        let mut depth = MarketDepth::new(timestamp, sequence);
        assert!(depth.best_bid().is_none());
        assert!(depth.best_ask().is_none());
        assert!(depth.spread().is_none());
        assert!(depth.mid_price().is_none());
        
        // Add some levels
        depth.bids.push((50000, 100));
        depth.bids.push((49900, 200));
        depth.asks.push((50100, 150));
        depth.asks.push((50200, 250));
        
        assert_eq!(depth.best_bid(), Some(50000));
        assert_eq!(depth.best_ask(), Some(50100));
        assert_eq!(depth.spread(), Some(100));
        assert_eq!(depth.mid_price(), Some(50050));
    }

    #[test]
    fn test_central_limit_order_book_creation() {
        let symbol = test_symbol();
        let book = CentralLimitOrderBook::new(symbol.clone());
        
        assert_eq!(book.symbol, symbol);
        assert!(book.bids.is_empty());
        assert!(book.asks.is_empty());
        assert!(book.orders.is_empty());
        assert_eq!(book.total_orders, 0);
        assert_eq!(book.total_bid_volume, 0);
        assert_eq!(book.total_ask_volume, 0);
        assert!(book.get_best_bid().is_none());
        assert!(book.get_best_ask().is_none());
        assert!(book.get_spread().is_none());
        assert!(book.get_mid_price().is_none());
    }

    #[test]
    fn test_order_book_add_limit_orders() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add a buy order
        let buy_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
        );
        
        let trades = book.add_order(buy_order.clone()).unwrap();
        assert!(trades.is_empty()); // No matching orders
        assert_eq!(book.total_orders, 1);
        assert_eq!(book.total_bid_volume, 100 * VOLUME_SCALE);
        assert_eq!(book.get_best_bid(), Some(50000 * PRICE_SCALE));
        assert!(book.orders.contains_key(&OrderId::new(1)));
        
        // Add a sell order at higher price (no match)
        let sell_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            51000 * PRICE_SCALE,
            150 * VOLUME_SCALE,
            timestamp + 1000,
        );
        
        let trades = book.add_order(sell_order.clone()).unwrap();
        assert!(trades.is_empty()); // No matching orders
        assert_eq!(book.total_orders, 2);
        assert_eq!(book.total_ask_volume, 150 * VOLUME_SCALE);
        assert_eq!(book.get_best_ask(), Some(51000 * PRICE_SCALE));
        assert_eq!(book.get_spread(), Some(1000 * PRICE_SCALE));
        assert_eq!(book.get_mid_price(), Some(50500 * PRICE_SCALE));
        
        // Test market depth
        let depth = book.get_market_depth(5);
        assert_eq!(depth.bids.len(), 1);
        assert_eq!(depth.asks.len(), 1);
        assert_eq!(depth.bids[0], (50000 * PRICE_SCALE, 100 * VOLUME_SCALE));
        assert_eq!(depth.asks[0], (51000 * PRICE_SCALE, 150 * VOLUME_SCALE));
    }

    #[test]
    fn test_order_book_matching() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add a sell order first
        let sell_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Sell,
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
        );
        
        let trades = book.add_order(sell_order).unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.total_orders, 1);
        
        // Add a matching buy order
        let buy_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,
            80 * VOLUME_SCALE, // Partial fill
            timestamp + 1000,
        );
        
        let trades = book.add_order(buy_order).unwrap();
        assert_eq!(trades.len(), 1);
        
        let trade = &trades[0];
        assert_eq!(trade.price, 50000 * PRICE_SCALE);
        assert_eq!(trade.size, 80 * VOLUME_SCALE);
        assert_eq!(trade.buyer_order_id, OrderId::new(2));
        assert_eq!(trade.seller_order_id, OrderId::new(1));
        assert!(!trade.is_buyer_maker); // Seller was the maker
        
        // Check remaining order in book
        assert_eq!(book.total_orders, 1); // Only the partially filled sell order remains
        assert_eq!(book.total_ask_volume, 20 * VOLUME_SCALE); // 100 - 80 = 20
        assert_eq!(book.total_bid_volume, 0); // Buy order was fully filled
        
        // Verify the remaining sell order size
        let remaining_order = book.orders.get(&OrderId::new(1)).unwrap();
        assert_eq!(remaining_order.size, 20 * VOLUME_SCALE);
    }

    #[test]
    fn test_order_book_market_orders() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add some limit orders to provide liquidity
        let sell_order_1 = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Sell,
            50000 * PRICE_SCALE,
            50 * VOLUME_SCALE,
            timestamp,
        );
        
        let sell_order_2 = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            50100 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp + 1000,
        );
        
        book.add_order(sell_order_1).unwrap();
        book.add_order(sell_order_2).unwrap();
        
        // Add a market buy order that crosses multiple levels
        let market_buy = Order::new_market(
            OrderId::new(3),
            symbol.clone(),
            Side::Buy,
            120 * VOLUME_SCALE, // More than first level
            timestamp + 2000,
        );
        
        let trades = book.add_order(market_buy).unwrap();
        assert_eq!(trades.len(), 2); // Should match against both levels
        
        // First trade at better price
        assert_eq!(trades[0].price, 50000 * PRICE_SCALE);
        assert_eq!(trades[0].size, 50 * VOLUME_SCALE);
        
        // Second trade at worse price
        assert_eq!(trades[1].price, 50100 * PRICE_SCALE);
        assert_eq!(trades[1].size, 70 * VOLUME_SCALE); // 120 - 50 = 70
        
        // Check remaining state
        assert_eq!(book.total_orders, 1); // Only partially filled second sell order
        assert_eq!(book.total_ask_volume, 30 * VOLUME_SCALE); // 100 - 70 = 30
    }

    #[test]
    fn test_order_cancellation() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add some orders
        let buy_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
        );
        
        let sell_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            51000 * PRICE_SCALE,
            150 * VOLUME_SCALE,
            timestamp + 1000,
        );
        
        book.add_order(buy_order.clone()).unwrap();
        book.add_order(sell_order.clone()).unwrap();
        
        assert_eq!(book.total_orders, 2);
        assert_eq!(book.total_bid_volume, 100 * VOLUME_SCALE);
        assert_eq!(book.total_ask_volume, 150 * VOLUME_SCALE);
        
        // Cancel the buy order
        let cancelled = book.cancel_order(OrderId::new(1)).unwrap();
        assert_eq!(cancelled.id, OrderId::new(1));
        assert_eq!(book.total_orders, 1);
        assert_eq!(book.total_bid_volume, 0);
        assert_eq!(book.total_ask_volume, 150 * VOLUME_SCALE);
        assert!(book.get_best_bid().is_none());
        assert!(!book.orders.contains_key(&OrderId::new(1)));
        
        // Try to cancel non-existent order
        assert!(book.cancel_order(OrderId::new(999)).is_err());
    }

    #[test]
    fn test_order_book_validation() {
        let symbol = test_symbol();
        let book = CentralLimitOrderBook::new(symbol.clone());
        
        // Empty book should validate
        assert!(book.validate().is_ok());
        
        // Test with orders would require more complex setup
        // This is a basic validation test
    }

    #[test]
    fn test_volume_at_or_better() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add multiple ask levels
        let asks = vec![
            (50000 * PRICE_SCALE, 100 * VOLUME_SCALE),
            (50100 * PRICE_SCALE, 200 * VOLUME_SCALE),
            (50200 * PRICE_SCALE, 300 * VOLUME_SCALE),
        ];
        
        for (i, (price, size)) in asks.iter().enumerate() {
            let order = Order::new_limit(
                OrderId::new(i as u64 + 1),
                symbol.clone(),
                Side::Sell,
                *price,
                *size,
                timestamp + (i as u64 * 1000),
            );
            book.add_order(order).unwrap();
        }
        
        // Test volume at or better for buy orders
        assert_eq!(book.get_volume_at_or_better(Side::Buy, 49900 * PRICE_SCALE), 0);
        assert_eq!(book.get_volume_at_or_better(Side::Buy, 50000 * PRICE_SCALE), 100 * VOLUME_SCALE);
        assert_eq!(book.get_volume_at_or_better(Side::Buy, 50100 * PRICE_SCALE), 300 * VOLUME_SCALE);
        assert_eq!(book.get_volume_at_or_better(Side::Buy, 50200 * PRICE_SCALE), 600 * VOLUME_SCALE);
        assert_eq!(book.get_volume_at_or_better(Side::Buy, 60000 * PRICE_SCALE), 600 * VOLUME_SCALE);
    }

    #[test]
    fn test_order_book_statistics() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add some orders
        let buy_order = Order::new_limit(
            OrderId::new(1),
            symbol.clone(),
            Side::Buy,
            50000 * PRICE_SCALE,
            100 * VOLUME_SCALE,
            timestamp,
        );
        
        let sell_order = Order::new_limit(
            OrderId::new(2),
            symbol.clone(),
            Side::Sell,
            51000 * PRICE_SCALE,
            150 * VOLUME_SCALE,
            timestamp + 1000,
        );
        
        book.add_order(buy_order).unwrap();
        book.add_order(sell_order).unwrap();
        
        let stats = book.get_statistics();
        assert_eq!(stats.symbol, symbol);
        assert_eq!(stats.total_orders, 2);
        assert_eq!(stats.bid_levels, 1);
        assert_eq!(stats.ask_levels, 1);
        assert_eq!(stats.total_bid_volume, 100 * VOLUME_SCALE);
        assert_eq!(stats.total_ask_volume, 150 * VOLUME_SCALE);
        assert_eq!(stats.best_bid, Some(50000 * PRICE_SCALE));
        assert_eq!(stats.best_ask, Some(51000 * PRICE_SCALE));
        assert_eq!(stats.spread, Some(1000 * PRICE_SCALE));
        assert_eq!(stats.mid_price, Some(50500 * PRICE_SCALE));
    }

    #[test]
    fn test_price_time_priority() {
        let symbol = test_symbol();
        let mut book = CentralLimitOrderBook::new(symbol.clone());
        let timestamp = test_timestamp();
        
        // Add multiple orders at the same price level
        let orders = vec![
            Order::new_limit(OrderId::new(1), symbol.clone(), Side::Buy, 50000 * PRICE_SCALE, 100 * VOLUME_SCALE, timestamp),
            Order::new_limit(OrderId::new(2), symbol.clone(), Side::Buy, 50000 * PRICE_SCALE, 200 * VOLUME_SCALE, timestamp + 1000),
            Order::new_limit(OrderId::new(3), symbol.clone(), Side::Buy, 50000 * PRICE_SCALE, 300 * VOLUME_SCALE, timestamp + 2000),
        ];
        
        for order in orders {
            book.add_order(order).unwrap();
        }
        
        // Add a sell order that partially matches
        let sell_order = Order::new_limit(
            OrderId::new(4),
            symbol.clone(),
            Side::Sell,
            50000 * PRICE_SCALE,
            150 * VOLUME_SCALE, // Should match first order (100) and part of second (50)
            timestamp + 3000,
        );
        
        let trades = book.add_order(sell_order).unwrap();
        assert_eq!(trades.len(), 2);
        
        // First trade should be with the earliest order (time priority)
        assert_eq!(trades[0].buyer_order_id, OrderId::new(1));
        assert_eq!(trades[0].size, 100 * VOLUME_SCALE);
        
        // Second trade should be with the second order
        assert_eq!(trades[1].buyer_order_id, OrderId::new(2));
        assert_eq!(trades[1].size, 50 * VOLUME_SCALE);
        
        // Check remaining orders
        assert_eq!(book.total_orders, 2); // Orders 2 (partial) and 3 (untouched)
        
        // Order 1 should be gone, order 2 should have reduced size
        assert!(!book.orders.contains_key(&OrderId::new(1)));
        let remaining_order_2 = book.orders.get(&OrderId::new(2)).unwrap();
        assert_eq!(remaining_order_2.size, 150 * VOLUME_SCALE); // 200 - 50 = 150
    }
}