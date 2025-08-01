use super::*;
use crate::orderbook::types::{Order, OrderId, Symbol, Side, OrderType, Trade};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::collections::HashMap;

/// Comprehensive test suite for lock-free data structures
pub struct LockFreeTestSuite {
    max_threads: usize,
    test_duration: Duration,
}

impl LockFreeTestSuite {
    pub fn new(max_threads: usize, test_duration: Duration) -> Self {
        Self {
            max_threads,
            test_duration,
        }
    }

    /// Run all lock-free tests
    pub fn run_all_tests(&self) -> TestResults {
        let mut results = TestResults::new();

        println!("Running lock-free data structure tests...");

        // Test hazard pointers
        results.hazard_pointer_results = self.test_hazard_pointers();
        println!("✓ Hazard pointer tests completed");

        // Test epoch-based reclamation
        results.epoch_reclamation_results = self.test_epoch_reclamation();
        println!("✓ Epoch-based reclamation tests completed");

        // Test price levels
        results.price_level_results = self.test_price_levels();
        println!("✓ Price level tests completed");

        // Test order nodes
        results.order_node_results = self.test_order_nodes();
        println!("✓ Order node tests completed");

        // Test order book
        results.order_book_results = self.test_order_book();
        println!("✓ Order book tests completed");

        // Performance benchmarks
        results.performance_results = self.run_performance_benchmarks();
        println!("✓ Performance benchmarks completed");

        // Stress tests
        results.stress_test_results = self.run_stress_tests();
        println!("✓ Stress tests completed");

        results
    }

    /// Test hazard pointer functionality
    fn test_hazard_pointers(&self) -> HazardPointerTestResults {
        let manager = HazardPointerManager::new(self.max_threads);
        let mut results = HazardPointerTestResults::new();

        // Test basic acquire/release
        let start = Instant::now();
        for _ in 0..1000 {
            let hazard = manager.acquire_hazard_pointer();
            let test_ptr = Box::into_raw(Box::new(42u64));
            hazard.protect(test_ptr);
            unsafe { Box::from_raw(test_ptr); }
        }
        results.basic_operations_time = start.elapsed();

        // Test concurrent access
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let manager = Arc::new(manager);
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for _ in 0..self.max_threads {
            let manager_clone = manager.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for _ in 0..100 {
                    let hazard = manager_clone.acquire_hazard_pointer();
                    let test_ptr = Box::into_raw(Box::new(thread::current().id()));
                    hazard.protect(test_ptr);
                    
                    // Simulate some work
                    thread::sleep(Duration::from_micros(1));
                    
                    manager_clone.retire_pointer(test_ptr);
                    success_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        results.concurrent_operations_time = start.elapsed();
        results.successful_operations = success_count.load(Ordering::Relaxed);

        // Test memory reclamation
        let start = Instant::now();
        manager.force_reclaim();
        results.reclamation_time = start.elapsed();

        let stats = manager.get_stats();
        results.final_stats = stats;

        results
    }

    /// Test epoch-based reclamation functionality
    fn test_epoch_reclamation(&self) -> EpochReclamationTestResults {
        let manager = EpochBasedReclamation::new(self.max_threads);
        let mut results = EpochReclamationTestResults::new();

        // Test basic epoch operations
        let start = Instant::now();
        for _ in 0..1000 {
            let guard = manager.pin();
            let test_ptr = Box::into_raw(Box::new(42u64));
            guard.retire(test_ptr);
        }
        results.basic_operations_time = start.elapsed();

        // Test concurrent epochs
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let manager = Arc::new(manager);
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for _ in 0..self.max_threads {
            let manager_clone = manager.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for _ in 0..100 {
                    let guard = manager_clone.pin();
                    let test_ptr = Box::into_raw(Box::new(thread::current().id()));
                    guard.retire(test_ptr);
                    
                    // Try to advance epoch
                    manager_clone.try_advance_epoch();
                    success_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        results.concurrent_operations_time = start.elapsed();
        results.successful_operations = success_count.load(Ordering::Relaxed);

        // Test forced reclamation
        let start = Instant::now();
        manager.force_reclaim();
        results.reclamation_time = start.elapsed();

        let stats = manager.get_stats();
        results.final_stats = stats;

        results
    }

    /// Test price level functionality
    fn test_price_levels(&self) -> PriceLevelTestResults {
        let hazard_manager = HazardPointerManager::new(self.max_threads);
        let price_level = LockFreePriceLevel::new(50000);
        let mut results = PriceLevelTestResults::new();

        // Test basic operations
        let start = Instant::now();
        for i in 0..1000 {
            let order = create_test_order(i, 100);
            price_level.add_order(order, &hazard_manager).unwrap();
        }
        results.add_operations_time = start.elapsed();

        let start = Instant::now();
        for i in 0..500 {
            price_level.remove_order(OrderId::new(i), &hazard_manager).unwrap();
        }
        results.remove_operations_time = start.elapsed();

        // Test concurrent operations
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let price_level = Arc::new(price_level);
        let hazard_manager = Arc::new(hazard_manager);
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let price_level_clone = price_level.clone();
            let hazard_manager_clone = hazard_manager.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for i in 0..100 {
                    let order_id = thread_id * 100 + i + 1000;
                    let order = create_test_order(order_id as u64, 10);
                    
                    if price_level_clone.add_order(order, &hazard_manager_clone).is_ok() {
                        success_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        results.concurrent_operations_time = start.elapsed();
        results.successful_operations = success_count.load(Ordering::Relaxed);

        results.final_order_count = price_level.get_order_count() as usize;
        results.final_volume = price_level.get_total_volume();

        results
    }

    /// Test order node functionality
    fn test_order_nodes(&self) -> OrderNodeTestResults {
        let hazard_manager = Arc::new(HazardPointerManager::new(self.max_threads));
        let order_list = LockFreeOrderList::new(hazard_manager.clone());
        let head = AlignedAtomicPtr::new(ptr::null_mut());
        let tail = AlignedAtomicPtr::new(ptr::null_mut());
        let mut results = OrderNodeTestResults::new();

        // Test basic list operations
        let start = Instant::now();
        let mut nodes = vec![];
        for i in 0..1000 {
            let order = create_test_order(i, 100);
            let node = Box::into_raw(Box::new(LockFreeOrderNode::new(order)));
            nodes.push(node);
            order_list.insert_tail(&head, &tail, node).unwrap();
        }
        results.insert_operations_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..500 {
            order_list.remove_head(&head, &tail).unwrap();
        }
        results.remove_operations_time = start.elapsed();

        // Test concurrent operations
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let order_list = Arc::new(order_list);
        let head = Arc::new(head);
        let tail = Arc::new(tail);
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_list_clone = order_list.clone();
            let head_clone = head.clone();
            let tail_clone = tail.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for i in 0..50 {
                    let order_id = thread_id * 50 + i + 2000;
                    let order = create_test_order(order_id as u64, 10);
                    let node = Box::into_raw(Box::new(LockFreeOrderNode::new(order)));
                    
                    if order_list_clone.insert_tail(&head_clone, &tail_clone, node).is_ok() {
                        success_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        results.concurrent_operations_time = start.elapsed();
        results.successful_operations = success_count.load(Ordering::Relaxed);

        results.final_node_count = LockFreeOrderList::count_nodes(&head);

        // Cleanup remaining nodes
        while let Ok(node) = order_list.remove_head(&head, &tail) {
            unsafe { Box::from_raw(node); }
        }

        results
    }

    /// Test order book functionality
    fn test_order_book(&self) -> OrderBookTestResults {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = LockFreeOrderBook::new(symbol, self.max_threads);
        let mut results = OrderBookTestResults::new();

        // Test basic order book operations
        let start = Instant::now();
        for i in 0..1000 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            let price = if side == Side::Buy { 49000 + (i % 100) } else { 51000 + (i % 100) };
            let order = create_test_order_with_side(i, side, price, 100);
            order_book.add_order(order).unwrap();
        }
        results.add_operations_time = start.elapsed();

        let start = Instant::now();
        for i in 0..500 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            order_book.cancel_order(OrderId::new(i), side).unwrap();
        }
        results.cancel_operations_time = start.elapsed();

        // Test concurrent order book operations
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let order_book = Arc::new(order_book);
        let success_count = Arc::new(AtomicUsize::new(0));
        let trade_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_book_clone = order_book.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();
            let trade_clone = trade_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for i in 0..100 {
                    let order_id = thread_id * 100 + i + 10000;
                    let side = if thread_id % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = if side == Side::Buy { 50000 - (i % 50) } else { 50000 + (i % 50) };
                    let order = create_test_order_with_side(order_id as u64, side, price, 10);
                    
                    match order_book_clone.add_order(order) {
                        Ok(trades) => {
                            success_clone.fetch_add(1, Ordering::Relaxed);
                            trade_clone.fetch_add(trades.len(), Ordering::Relaxed);
                        }
                        Err(_) => {}
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        results.concurrent_operations_time = start.elapsed();
        results.successful_operations = success_count.load(Ordering::Relaxed);
        results.total_trades = trade_count.load(Ordering::Relaxed);

        let stats = order_book.get_stats();
        results.final_stats = stats;

        results
    }

    /// Run performance benchmarks
    fn run_performance_benchmarks(&self) -> PerformanceBenchmarkResults {
        let mut results = PerformanceBenchmarkResults::new();

        // Latency benchmark
        results.latency_benchmark = self.benchmark_latency();
        
        // Throughput benchmark
        results.throughput_benchmark = self.benchmark_throughput();
        
        // Memory usage benchmark
        results.memory_benchmark = self.benchmark_memory_usage();

        results
    }

    /// Benchmark latency
    fn benchmark_latency(&self) -> LatencyBenchmarkResults {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = LockFreeOrderBook::new(symbol, self.max_threads);
        let mut results = LatencyBenchmarkResults::new();

        let num_operations = 10000;
        let mut latencies = Vec::with_capacity(num_operations);

        // Measure order addition latency
        for i in 0..num_operations {
            let order = create_test_order(i as u64, 100);
            let start = Instant::now();
            order_book.add_order(order).unwrap();
            let latency = start.elapsed();
            latencies.push(latency);
        }

        latencies.sort();
        results.min_latency = latencies[0];
        results.max_latency = latencies[num_operations - 1];
        results.median_latency = latencies[num_operations / 2];
        results.p99_latency = latencies[(num_operations as f64 * 0.99) as usize];
        results.p999_latency = latencies[(num_operations as f64 * 0.999) as usize];

        let total: Duration = latencies.iter().sum();
        results.average_latency = total / num_operations as u32;

        results
    }

    /// Benchmark throughput
    fn benchmark_throughput(&self) -> ThroughputBenchmarkResults {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = Arc::new(LockFreeOrderBook::new(symbol, self.max_threads));
        let mut results = ThroughputBenchmarkResults::new();

        let operations_per_thread = 10000;
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let total_operations = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_book_clone = order_book.clone();
            let barrier_clone = barrier.clone();
            let total_clone = total_operations.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                for i in 0..operations_per_thread {
                    let order_id = thread_id * operations_per_thread + i;
                    let side = if thread_id % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = if side == Side::Buy { 49000 + (i % 1000) } else { 51000 + (i % 1000) };
                    let order = create_test_order_with_side(order_id as u64, side, price, 10);
                    
                    if order_book_clone.add_order(order).is_ok() {
                        total_clone.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed();

        let total_ops = total_operations.load(Ordering::Relaxed);
        results.total_operations = total_ops;
        results.duration = duration;
        results.operations_per_second = (total_ops as f64 / duration.as_secs_f64()) as u64;

        results
    }

    /// Benchmark memory usage
    fn benchmark_memory_usage(&self) -> MemoryBenchmarkResults {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = LockFreeOrderBook::new(symbol, self.max_threads);
        let mut results = MemoryBenchmarkResults::new();

        // Add orders and measure memory usage
        let num_orders = 100000;
        for i in 0..num_orders {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            let price = if side == Side::Buy { 49000 + (i % 1000) } else { 51000 + (i % 1000) };
            let order = create_test_order_with_side(i as u64, side, price, 100);
            order_book.add_order(order).unwrap();
        }

        // Force reclamation and measure
        let start = Instant::now();
        order_book.force_reclaim();
        results.reclamation_time = start.elapsed();

        let stats = order_book.get_stats();
        results.total_orders = stats.total_orders;
        results.total_volume = stats.total_bid_volume + stats.total_ask_volume;

        results
    }

    /// Run stress tests
    fn run_stress_tests(&self) -> StressTestResults {
        let mut results = StressTestResults::new();

        // High contention test
        results.high_contention_result = self.stress_test_high_contention();
        
        // Memory pressure test
        results.memory_pressure_result = self.stress_test_memory_pressure();
        
        // Long running test
        results.long_running_result = self.stress_test_long_running();

        results
    }

    /// Stress test with high contention
    fn stress_test_high_contention(&self) -> StressTestResult {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = Arc::new(LockFreeOrderBook::new(symbol, self.max_threads * 2));
        
        let barrier = Arc::new(Barrier::new(self.max_threads));
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_book_clone = order_book.clone();
            let barrier_clone = barrier.clone();
            let success_clone = success_count.clone();
            let error_clone = error_count.clone();

            let handle = thread::spawn(move || {
                barrier_clone.wait();
                
                // All threads operate on the same price range for maximum contention
                for i in 0..1000 {
                    let order_id = thread_id * 1000 + i;
                    let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = 50000; // Same price for all orders = maximum contention
                    let order = create_test_order_with_side(order_id as u64, side, price, 1);
                    
                    match order_book_clone.add_order(order) {
                        Ok(_) => success_clone.fetch_add(1, Ordering::Relaxed),
                        Err(_) => error_clone.fetch_add(1, Ordering::Relaxed),
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed();

        StressTestResult {
            duration,
            successful_operations: success_count.load(Ordering::Relaxed),
            failed_operations: error_count.load(Ordering::Relaxed),
            operations_per_second: (success_count.load(Ordering::Relaxed) as f64 / duration.as_secs_f64()) as u64,
        }
    }

    /// Stress test with memory pressure
    fn stress_test_memory_pressure(&self) -> StressTestResult {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = Arc::new(LockFreeOrderBook::new(symbol, self.max_threads));
        
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_book_clone = order_book.clone();
            let success_clone = success_count.clone();
            let error_clone = error_count.clone();

            let handle = thread::spawn(move || {
                // Create many orders to pressure memory system
                for i in 0..10000 {
                    let order_id = thread_id * 10000 + i;
                    let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = if side == Side::Buy { 
                        49000 + (i % 2000) 
                    } else { 
                        51000 + (i % 2000) 
                    };
                    let order = create_test_order_with_side(order_id as u64, side, price, 1);
                    
                    match order_book_clone.add_order(order) {
                        Ok(_) => success_clone.fetch_add(1, Ordering::Relaxed),
                        Err(_) => error_clone.fetch_add(1, Ordering::Relaxed),
                    }
                    
                    // Occasionally force reclamation
                    if i % 1000 == 0 {
                        order_book_clone.force_reclaim();
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed();

        StressTestResult {
            duration,
            successful_operations: success_count.load(Ordering::Relaxed),
            failed_operations: error_count.load(Ordering::Relaxed),
            operations_per_second: (success_count.load(Ordering::Relaxed) as f64 / duration.as_secs_f64()) as u64,
        }
    }

    /// Long running stress test
    fn stress_test_long_running(&self) -> StressTestResult {
        let symbol = Symbol::new("BTCUSD").unwrap();
        let order_book = Arc::new(LockFreeOrderBook::new(symbol, self.max_threads));
        
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        let start = Instant::now();
        for thread_id in 0..self.max_threads {
            let order_book_clone = order_book.clone();
            let success_clone = success_count.clone();
            let error_clone = error_count.clone();
            let test_duration = self.test_duration;

            let handle = thread::spawn(move || {
                let thread_start = Instant::now();
                let mut operation_count = 0;
                
                while thread_start.elapsed() < test_duration {
                    let order_id = thread_id * 1000000 + operation_count;
                    let side = if operation_count % 2 == 0 { Side::Buy } else { Side::Sell };
                    let price = if side == Side::Buy { 
                        49000 + (operation_count % 1000) 
                    } else { 
                        51000 + (operation_count % 1000) 
                    };
                    let order = create_test_order_with_side(order_id as u64, side, price, 10);
                    
                    match order_book_clone.add_order(order) {
                        Ok(_) => success_clone.fetch_add(1, Ordering::Relaxed),
                        Err(_) => error_clone.fetch_add(1, Ordering::Relaxed),
                    }
                    
                    operation_count += 1;
                    
                    // Periodic cleanup
                    if operation_count % 10000 == 0 {
                        order_book_clone.force_reclaim();
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        let duration = start.elapsed();

        StressTestResult {
            duration,
            successful_operations: success_count.load(Ordering::Relaxed),
            failed_operations: error_count.load(Ordering::Relaxed),
            operations_per_second: (success_count.load(Ordering::Relaxed) as f64 / duration.as_secs_f64()) as u64,
        }
    }
}

// Helper functions
fn create_test_order(id: u64, size: u64) -> Order {
    Order {
        id: OrderId::new(id),
        symbol: Symbol::new("BTCUSD").unwrap(),
        side: Side::Buy,
        order_type: OrderType::Limit,
        price: 50000,
        size,
        timestamp: 1000,
    }
}

fn create_test_order_with_side(id: u64, side: Side, price: u64, size: u64) -> Order {
    Order {
        id: OrderId::new(id),
        symbol: Symbol::new("BTCUSD").unwrap(),
        side,
        order_type: OrderType::Limit,
        price,
        size,
        timestamp: 1000,
    }
}

// Test result structures
#[derive(Debug)]
pub struct TestResults {
    pub hazard_pointer_results: HazardPointerTestResults,
    pub epoch_reclamation_results: EpochReclamationTestResults,
    pub price_level_results: PriceLevelTestResults,
    pub order_node_results: OrderNodeTestResults,
    pub order_book_results: OrderBookTestResults,
    pub performance_results: PerformanceBenchmarkResults,
    pub stress_test_results: StressTestResults,
}

impl TestResults {
    fn new() -> Self {
        Self {
            hazard_pointer_results: HazardPointerTestResults::new(),
            epoch_reclamation_results: EpochReclamationTestResults::new(),
            price_level_results: PriceLevelTestResults::new(),
            order_node_results: OrderNodeTestResults::new(),
            order_book_results: OrderBookTestResults::new(),
            performance_results: PerformanceBenchmarkResults::new(),
            stress_test_results: StressTestResults::new(),
        }
    }

    pub fn print_summary(&self) {
        println!("\n=== Lock-Free Test Results Summary ===");
        
        println!("\nHazard Pointers:");
        println!("  Basic ops: {:?}", self.hazard_pointer_results.basic_operations_time);
        println!("  Concurrent ops: {:?}", self.hazard_pointer_results.concurrent_operations_time);
        println!("  Success rate: {}/{}", 
                 self.hazard_pointer_results.successful_operations,
                 self.hazard_pointer_results.successful_operations);

        println!("\nEpoch Reclamation:");
        println!("  Basic ops: {:?}", self.epoch_reclamation_results.basic_operations_time);
        println!("  Concurrent ops: {:?}", self.epoch_reclamation_results.concurrent_operations_time);
        
        println!("\nOrder Book:");
        println!("  Add ops: {:?}", self.order_book_results.add_operations_time);
        println!("  Concurrent ops: {:?}", self.order_book_results.concurrent_operations_time);
        println!("  Total trades: {}", self.order_book_results.total_trades);

        println!("\nPerformance:");
        println!("  Median latency: {:?}", self.performance_results.latency_benchmark.median_latency);
        println!("  P99 latency: {:?}", self.performance_results.latency_benchmark.p99_latency);
        println!("  Throughput: {} ops/sec", self.performance_results.throughput_benchmark.operations_per_second);

        println!("\nStress Tests:");
        println!("  High contention: {} ops/sec", self.stress_test_results.high_contention_result.operations_per_second);
        println!("  Memory pressure: {} ops/sec", self.stress_test_results.memory_pressure_result.operations_per_second);
        println!("  Long running: {} ops/sec", self.stress_test_results.long_running_result.operations_per_second);
    }
}

#[derive(Debug, Default)]
pub struct HazardPointerTestResults {
    pub basic_operations_time: Duration,
    pub concurrent_operations_time: Duration,
    pub reclamation_time: Duration,
    pub successful_operations: usize,
    pub final_stats: crate::performance::lock_free::hazard_pointers::HazardPointerStats,
}

impl HazardPointerTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct EpochReclamationTestResults {
    pub basic_operations_time: Duration,
    pub concurrent_operations_time: Duration,
    pub reclamation_time: Duration,
    pub successful_operations: usize,
    pub final_stats: crate::performance::lock_free::memory_reclamation::EpochReclamationStats,
}

impl EpochReclamationTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct PriceLevelTestResults {
    pub add_operations_time: Duration,
    pub remove_operations_time: Duration,
    pub concurrent_operations_time: Duration,
    pub successful_operations: usize,
    pub final_order_count: usize,
    pub final_volume: u64,
}

impl PriceLevelTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct OrderNodeTestResults {
    pub insert_operations_time: Duration,
    pub remove_operations_time: Duration,
    pub concurrent_operations_time: Duration,
    pub successful_operations: usize,
    pub final_node_count: usize,
}

impl OrderNodeTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct OrderBookTestResults {
    pub add_operations_time: Duration,
    pub cancel_operations_time: Duration,
    pub concurrent_operations_time: Duration,
    pub successful_operations: usize,
    pub total_trades: usize,
    pub final_stats: crate::performance::lock_free::order_book::OrderBookStats,
}

impl OrderBookTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct PerformanceBenchmarkResults {
    pub latency_benchmark: LatencyBenchmarkResults,
    pub throughput_benchmark: ThroughputBenchmarkResults,
    pub memory_benchmark: MemoryBenchmarkResults,
}

impl PerformanceBenchmarkResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct LatencyBenchmarkResults {
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub average_latency: Duration,
    pub median_latency: Duration,
    pub p99_latency: Duration,
    pub p999_latency: Duration,
}

impl LatencyBenchmarkResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct ThroughputBenchmarkResults {
    pub total_operations: usize,
    pub duration: Duration,
    pub operations_per_second: u64,
}

impl ThroughputBenchmarkResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct MemoryBenchmarkResults {
    pub total_orders: usize,
    pub total_volume: u64,
    pub reclamation_time: Duration,
}

impl MemoryBenchmarkResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct StressTestResults {
    pub high_contention_result: StressTestResult,
    pub memory_pressure_result: StressTestResult,
    pub long_running_result: StressTestResult,
}

impl StressTestResults {
    fn new() -> Self {
        Default::default()
    }
}

#[derive(Debug, Default)]
pub struct StressTestResult {
    pub duration: Duration,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub operations_per_second: u64,
}