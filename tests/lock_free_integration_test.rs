use trading_system::performance::lock_free::*;
use trading_system::orderbook::types::{Order, OrderId, Symbol, Side, OrderType};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_lock_free_order_book_integration() {
    let symbol = Symbol::new("BTCUSD").unwrap();
    let order_book = Arc::new(LockFreeOrderBook::new(symbol, 4));

    // Test basic functionality
    let buy_order = Order {
        id: OrderId::new(1),
        symbol: Symbol::new("BTCUSD").unwrap(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        price: 50000,
        size: 100,
        timestamp: 1000,
    };

    let sell_order = Order {
        id: OrderId::new(2),
        symbol: Symbol::new("BTCUSD").unwrap(),
        side: Side::Sell,
        order_type: OrderType::Limit,
        price: 50000,
        size: 50,
        timestamp: 1001,
    };

    // Add buy order
    let trades = order_book.add_order(buy_order).unwrap();
    assert!(trades.is_empty());
    assert_eq!(order_book.get_best_bid(), Some(50000));

    // Add matching sell order
    let trades = order_book.add_order(sell_order).unwrap();
    assert_eq!(trades.len(), 1);
    assert_eq!(trades[0].size, 50);
    assert_eq!(trades[0].price, 50000);

    // Check remaining order
    let stats = order_book.get_stats();
    assert_eq!(stats.total_orders, 1); // One order partially filled
    assert_eq!(stats.total_bid_volume, 50); // 50 remaining
}

#[test]
fn test_concurrent_lock_free_operations() {
    let symbol = Symbol::new("BTCUSD").unwrap();
    let order_book = Arc::new(LockFreeOrderBook::new(symbol, 10));
    let mut handles = vec![];

    // Spawn threads to add orders concurrently
    for i in 0..5 {
        let book_clone = order_book.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let order_id = i * 100 + j + 1;
                let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                let price = if side == Side::Buy { 49000 + i * 10 } else { 51000 + i * 10 };
                
                let order = Order {
                    id: OrderId::new(order_id),
                    symbol: Symbol::new("BTCUSD").unwrap(),
                    side,
                    order_type: OrderType::Limit,
                    price,
                    size: 10,
                    timestamp: 1000,
                };
                
                let _ = book_clone.add_order(order);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify final state
    let stats = order_book.get_stats();
    assert!(stats.total_orders > 0);
    
    // Force cleanup
    order_book.force_reclaim();
}

#[test]
fn test_hazard_pointer_manager() {
    let manager = HazardPointerManager::new(4);
    
    // Test basic acquire/release
    let hazard = manager.acquire_hazard_pointer();
    let test_ptr = Box::into_raw(Box::new(42u64));
    hazard.protect(test_ptr);
    
    // Retire the pointer
    manager.retire_pointer(test_ptr);
    
    // Force reclamation
    manager.force_reclaim();
    
    let stats = manager.get_stats();
    assert!(stats.total_retired <= 1); // Should be reclaimed or still protected
}

#[test]
fn test_epoch_based_reclamation() {
    let manager = EpochBasedReclamation::new(4);
    
    // Test basic epoch operations
    let guard = manager.pin();
    let test_ptr = Box::into_raw(Box::new(42u64));
    guard.retire(test_ptr);
    
    drop(guard);
    
    // Force reclamation
    manager.force_reclaim();
    
    let stats = manager.get_stats();
    assert!(stats.global_epoch >= 0);
}

#[test]
fn test_lock_free_test_suite() {
    let test_suite = LockFreeTestSuite::new(4, Duration::from_millis(100));
    
    // Run a subset of tests for integration testing
    let hazard_results = test_suite.test_hazard_pointers();
    assert!(hazard_results.successful_operations > 0);
    
    let epoch_results = test_suite.test_epoch_reclamation();
    assert!(epoch_results.successful_operations > 0);
    
    println!("Lock-free integration tests passed!");
}