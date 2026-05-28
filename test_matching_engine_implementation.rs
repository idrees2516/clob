// Test file to verify the matching engine implementation
use std::collections::{BTreeMap, VecDeque, HashMap};

// Import the types we need
use hf_quoting_liquidity_clob::orderbook::{
    Order, OrderBook, MatchingEngine, OrderId, Symbol, Side, OrderType, OrderStatus
};

fn main() {
    println!("Testing deterministic matching engine implementation...");
    
    // Test 1: Basic limit order matching
    test_limit_order_matching();
    
    // Test 2: Market order execution
    test_market_order_execution();
    
    // Test 3: Partial fill handling
    test_partial_fill_handling();
    
    // Test 4: Price-time priority
    test_price_time_priority();
    
    println!("All tests completed successfully!");
}

fn test_limit_order_matching() {
    println!("Test 1: Limit order matching");
    
    let mut engine = MatchingEngine::new();
    engine.set_current_time(1000);
    
    let mut order_book = OrderBook::new("BTCUSD".to_string());

    // Add a sell order to the book
    let sell_order = Order::new_limit(
        OrderId::new(1),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50000,
        100,
        1000,
    );

    let result = engine.process_order(&mut order_book, sell_order).unwrap();
    assert_eq!(result.status, OrderStatus::Added);
    assert_eq!(result.filled_size, 0);

    // Add a buy order that should match
    let buy_order = Order::new_limit(
        OrderId::new(2),
        Symbol::new("BTCUSD").unwrap(),
        Side::Buy,
        50000,
        50,
        1001,
    );

    let result = engine.process_order(&mut order_book, buy_order).unwrap();
    assert_eq!(result.status, OrderStatus::FullyFilled);
    assert_eq!(result.filled_size, 50);
    assert_eq!(result.trades.len(), 1);
    assert_eq!(result.trades[0].price, 50000);
    assert_eq!(result.trades[0].size, 50);
    
    println!("✓ Limit order matching test passed");
}

fn test_market_order_execution() {
    println!("Test 2: Market order execution");
    
    let mut engine = MatchingEngine::new();
    engine.set_current_time(1000);
    
    let mut order_book = OrderBook::new("BTCUSD".to_string());

    // Add multiple sell orders at different prices
    let sell_order1 = Order::new_limit(
        OrderId::new(1),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50000,
        50,
        1000,
    );
    let sell_order2 = Order::new_limit(
        OrderId::new(2),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50100,
        75,
        1001,
    );

    engine.process_order(&mut order_book, sell_order1).unwrap();
    engine.process_order(&mut order_book, sell_order2).unwrap();

    // Market buy order should match against best prices
    let market_buy = Order::new_market(
        OrderId::new(3),
        Symbol::new("BTCUSD").unwrap(),
        Side::Buy,
        100,
        1002,
    );

    let result = engine.process_order(&mut order_book, market_buy).unwrap();
    assert_eq!(result.status, OrderStatus::FullyFilled);
    assert_eq!(result.filled_size, 100);
    assert_eq!(result.trades.len(), 2);
    
    // Should match at 50000 first (better price)
    assert_eq!(result.trades[0].price, 50000);
    assert_eq!(result.trades[0].size, 50);
    
    // Then at 50100
    assert_eq!(result.trades[1].price, 50100);
    assert_eq!(result.trades[1].size, 50);
    
    println!("✓ Market order execution test passed");
}

fn test_partial_fill_handling() {
    println!("Test 3: Partial fill handling");
    
    let mut engine = MatchingEngine::new();
    engine.set_current_time(1000);
    
    let mut order_book = OrderBook::new("BTCUSD".to_string());

    // Add a small sell order
    let sell_order = Order::new_limit(
        OrderId::new(1),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50000,
        30,
        1000,
    );

    engine.process_order(&mut order_book, sell_order).unwrap();

    // Large buy order should be partially filled
    let buy_order = Order::new_limit(
        OrderId::new(2),
        Symbol::new("BTCUSD").unwrap(),
        Side::Buy,
        50000,
        100,
        1001,
    );

    let result = engine.process_order(&mut order_book, buy_order).unwrap();
    assert_eq!(result.status, OrderStatus::PartiallyFilled);
    assert_eq!(result.filled_size, 30);
    assert_eq!(result.remaining_size, 70);
    assert_eq!(result.trades.len(), 1);
    
    // Remaining 70 should be added to book
    assert_eq!(order_book.get_best_bid(), Some(50000));
    
    println!("✓ Partial fill handling test passed");
}

fn test_price_time_priority() {
    println!("Test 4: Price-time priority");
    
    let mut engine = MatchingEngine::new();
    
    let mut order_book = OrderBook::new("BTCUSD".to_string());

    // Add two sell orders at same price, different times
    engine.set_current_time(1000);
    let sell_order1 = Order::new_limit(
        OrderId::new(1),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50000,
        50,
        1000,
    );
    engine.process_order(&mut order_book, sell_order1).unwrap();

    engine.set_current_time(1001);
    let sell_order2 = Order::new_limit(
        OrderId::new(2),
        Symbol::new("BTCUSD").unwrap(),
        Side::Sell,
        50000,
        50,
        1001,
    );
    engine.process_order(&mut order_book, sell_order2).unwrap();

    // Buy order should match first order first (time priority)
    engine.set_current_time(1002);
    let buy_order = Order::new_limit(
        OrderId::new(3),
        Symbol::new("BTCUSD").unwrap(),
        Side::Buy,
        50000,
        50,
        1002,
    );

    let result = engine.process_order(&mut order_book, buy_order).unwrap();
    assert_eq!(result.trades[0].seller_order_id, 1); // First order matched first
    
    println!("✓ Price-time priority test passed");
}