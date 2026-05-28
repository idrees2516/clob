use super::lock_free_pool::{LockFreePool, PooledObject, PoolError, PoolStats};
use super::numa_allocator::{NumaAllocator, AllocationPolicy, NumaError};
use crate::orderbook::types::{Order, Trade, OrderId, Symbol, Side, OrderType};
use crate::performance::lock_free::{LockFreeOrderNode, LockFreePriceLevel};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem;

/// Specialized object pools for trading system components
pub struct TradingObjectPools {
    /// Pool for Order objects
    order_pool: Arc<LockFreePool<Order>>,
    
    /// Pool for Trade objects
    trade_pool: Arc<LockFreePool<Trade>>,
    
    /// Pool for OrderNode objects
    order_node_pool: Arc<LockFreePool<LockFreeOrderNode>>,
    
    /// Pool for PriceLevel objects
    price_level_pool: Arc<LockFreePool<LockFreePriceLevel>>,
    
    /// Pool for generic byte buffers
    buffer_pools: HashMap<usize, Arc<LockFreePool<Vec<u8>>>>,
    
    /// NUMA-aware allocator for large objects
    numa_allocator: Arc<NumaAllocator>,
    
    /// Pool statistics
    stats: PoolManagerStats,
}

/// Statistics for the pool manager
#[derive(Debug)]
pub struct PoolManagerStats {
    /// Total objects allocated across all pools
    pub total_allocations: AtomicUsize,
    
    /// Total objects deallocated across all pools
    pub total_deallocations: AtomicUsize,
    
    /// Pool expansion events
    pub pool_expansions: AtomicUsize,
    
    /// Allocation failures
    pub allocation_failures: AtomicUsize,
    
    /// Cross-NUMA allocations
    pub cross_numa_allocations: AtomicUsize,
}

impl TradingObjectPools {
    /// Create a new trading object pool manager
    pub fn new(numa_policy: AllocationPolicy) -> Result<Self, ObjectPoolError> {
        let numa_allocator = Arc::new(NumaAllocator::new(numa_policy)?);
        
        // Create pools with appropriate initial capacities
        let order_pool = Arc::new(LockFreePool::new(10000, 0)?); // High-frequency orders
        let trade_pool = Arc::new(LockFreePool::new(5000, 0)?);  // Trades generated from orders
        let order_node_pool = Arc::new(LockFreePool::new(10000, 0)?); // Order book nodes
        let price_level_pool = Arc::new(LockFreePool::new(1000, 0)?); // Price levels
        
        // Create buffer pools for different sizes
        let mut buffer_pools = HashMap::new();
        let buffer_sizes = vec![64, 128, 256, 512, 1024, 2048, 4096];
        
        for &size in &buffer_sizes {
            let capacity = Self::calculate_buffer_pool_capacity(size);
            let pool = Arc::new(LockFreePool::new(capacity, 0)?);
            buffer_pools.insert(size, pool);
        }

        Ok(Self {
            order_pool,
            trade_pool,
            order_node_pool,
            price_level_pool,
            buffer_pools,
            numa_allocator,
            stats: PoolManagerStats::new(),
        })
    }

    /// Allocate a new Order object
    pub fn allocate_order(&self) -> Result<PooledOrder, ObjectPoolError> {
        let pooled_obj = self.order_pool.allocate()
            .map_err(ObjectPoolError::PoolError)?;
        
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(PooledOrder {
            inner: pooled_obj,
            pools: self,
        })
    }

    /// Allocate a new Trade object
    pub fn allocate_trade(&self) -> Result<PooledTrade, ObjectPoolError> {
        let pooled_obj = self.trade_pool.allocate()
            .map_err(ObjectPoolError::PoolError)?;
        
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(PooledTrade {
            inner: pooled_obj,
            pools: self,
        })
    }

    /// Allocate a new OrderNode object
    pub fn allocate_order_node(&self, order: Order) -> Result<PooledOrderNode, ObjectPoolError> {
        let mut pooled_obj = self.order_node_pool.allocate()
            .map_err(ObjectPoolError::PoolError)?;
        
        // Initialize the node with the order
        *pooled_obj = LockFreeOrderNode::new(order);
        
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(PooledOrderNode {
            inner: pooled_obj,
            pools: self,
        })
    }

    /// Allocate a new PriceLevel object
    pub fn allocate_price_level(&self, price: u64) -> Result<PooledPriceLevel, ObjectPoolError> {
        let mut pooled_obj = self.price_level_pool.allocate()
            .map_err(ObjectPoolError::PoolError)?;
        
        // Initialize the price level
        *pooled_obj = LockFreePriceLevel::new(price);
        
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
        
        Ok(PooledPriceLevel {
            inner: pooled_obj,
            pools: self,
        })
    }

    /// Allocate a buffer of specified size
    pub fn allocate_buffer(&self, size: usize) -> Result<PooledBuffer, ObjectPoolError> {
        let pool_size = self.get_buffer_pool_size(size);
        
        if let Some(pool) = self.buffer_pools.get(&pool_size) {
            let mut pooled_obj = pool.allocate()
                .map_err(ObjectPoolError::PoolError)?;
            
            // Resize the buffer to the requested size
            pooled_obj.clear();
            pooled_obj.resize(size, 0);
            
            self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
            
            Ok(PooledBuffer {
                inner: pooled_obj,
                pools: self,
            })
        } else {
            // Use NUMA allocator for large buffers
            let allocation = self.numa_allocator.allocate(size)
                .map_err(ObjectPoolError::NumaError)?;
            
            self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
            self.stats.cross_numa_allocations.fetch_add(1, Ordering::Relaxed);
            
            Ok(PooledBuffer::from_numa_allocation(allocation, self))
        }
    }

    /// Create a pre-initialized Order
    pub fn create_order(
        &self,
        id: OrderId,
        symbol: Symbol,
        side: Side,
        order_type: OrderType,
        price: u64,
        size: u64,
        timestamp: u64,
    ) -> Result<PooledOrder, ObjectPoolError> {
        let mut order = self.allocate_order()?;
        
        order.id = id;
        order.symbol = symbol;
        order.side = side;
        order.order_type = order_type;
        order.price = price;
        order.size = size;
        order.timestamp = timestamp;
        
        Ok(order)
    }

    /// Create a pre-initialized Trade
    pub fn create_trade(
        &self,
        id: u64,
        buyer_order_id: u64,
        seller_order_id: u64,
        symbol: Symbol,
        price: u64,
        size: u64,
        timestamp: u64,
    ) -> Result<PooledTrade, ObjectPoolError> {
        let mut trade = self.allocate_trade()?;
        
        trade.id = id;
        trade.buyer_order_id = buyer_order_id;
        trade.seller_order_id = seller_order_id;
        trade.symbol = symbol;
        trade.price = price;
        trade.size = size;
        trade.timestamp = timestamp;
        
        Ok(trade)
    }

    /// Get appropriate buffer pool size for requested size
    fn get_buffer_pool_size(&self, size: usize) -> usize {
        // Find the smallest pool size that can accommodate the request
        for &pool_size in &[64, 128, 256, 512, 1024, 2048, 4096] {
            if size <= pool_size {
                return pool_size;
            }
        }
        
        // For larger sizes, round up to next 4KB boundary
        ((size + 4095) / 4096) * 4096
    }

    /// Calculate initial capacity for buffer pools
    fn calculate_buffer_pool_capacity(size: usize) -> usize {
        match size {
            64..=128 => 2000,   // Small buffers - high frequency
            129..=512 => 1000,  // Medium buffers - moderate frequency
            513..=2048 => 500,  // Large buffers - lower frequency
            _ => 100,           // Very large buffers - rare
        }
    }

    /// Get comprehensive statistics for all pools
    pub fn get_comprehensive_stats(&self) -> TradingPoolStats {
        TradingPoolStats {
            order_pool: self.order_pool.get_stats(),
            trade_pool: self.trade_pool.get_stats(),
            order_node_pool: self.order_node_pool.get_stats(),
            price_level_pool: self.price_level_pool.get_stats(),
            buffer_pool_stats: self.buffer_pools
                .iter()
                .map(|(&size, pool)| (size, pool.get_stats()))
                .collect(),
            numa_stats: self.numa_allocator.get_all_stats(),
            manager_stats: PoolManagerStatsSnapshot {
                total_allocations: self.stats.total_allocations.load(Ordering::Acquire),
                total_deallocations: self.stats.total_deallocations.load(Ordering::Acquire),
                pool_expansions: self.stats.pool_expansions.load(Ordering::Acquire),
                allocation_failures: self.stats.allocation_failures.load(Ordering::Acquire),
                cross_numa_allocations: self.stats.cross_numa_allocations.load(Ordering::Acquire),
            },
        }
    }

    /// Warm up pools by pre-allocating objects
    pub fn warm_up(&self) -> Result<(), ObjectPoolError> {
        // Pre-allocate and immediately deallocate objects to warm up pools
        let mut temp_objects = Vec::new();

        // Warm up order pool
        for _ in 0..100 {
            temp_objects.push(self.allocate_order()?);
        }
        temp_objects.clear();

        // Warm up trade pool
        for _ in 0..50 {
            temp_objects.push(self.allocate_trade()?);
        }
        temp_objects.clear();

        // Warm up buffer pools
        for &size in &[64, 128, 256, 512, 1024] {
            for _ in 0..20 {
                let _buffer = self.allocate_buffer(size)?;
            }
        }

        Ok(())
    }

    /// Force garbage collection of unused objects
    pub fn force_gc(&self) {
        // In a real implementation, this would trigger cleanup of unused objects
        // For now, we'll just update statistics
        self.stats.pool_expansions.fetch_add(0, Ordering::Relaxed);
    }

    /// Get memory usage statistics
    pub fn get_memory_usage(&self) -> MemoryUsageStats {
        let order_memory = self.order_pool.get_stats().capacity * mem::size_of::<Order>();
        let trade_memory = self.trade_pool.get_stats().capacity * mem::size_of::<Trade>();
        let node_memory = self.order_node_pool.get_stats().capacity * mem::size_of::<LockFreeOrderNode>();
        let level_memory = self.price_level_pool.get_stats().capacity * mem::size_of::<LockFreePriceLevel>();
        
        let buffer_memory: usize = self.buffer_pools
            .iter()
            .map(|(&size, pool)| pool.get_stats().capacity * size)
            .sum();

        MemoryUsageStats {
            order_pool_bytes: order_memory,
            trade_pool_bytes: trade_memory,
            order_node_pool_bytes: node_memory,
            price_level_pool_bytes: level_memory,
            buffer_pool_bytes: buffer_memory,
            total_bytes: order_memory + trade_memory + node_memory + level_memory + buffer_memory,
        }
    }
}

/// RAII wrapper for pooled Order objects
pub struct PooledOrder<'a> {
    inner: PooledObject<Order>,
    pools: &'a TradingObjectPools,
}

impl<'a> PooledOrder<'a> {
    /// Clone the order (creates a new pooled order)
    pub fn clone_order(&self) -> Result<PooledOrder<'a>, ObjectPoolError> {
        let mut new_order = self.pools.allocate_order()?;
        *new_order = self.inner.clone();
        Ok(new_order)
    }
}

impl<'a> std::ops::Deref for PooledOrder<'a> {
    type Target = Order;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for PooledOrder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PooledOrder<'a> {
    fn drop(&mut self) {
        self.pools.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII wrapper for pooled Trade objects
pub struct PooledTrade<'a> {
    inner: PooledObject<Trade>,
    pools: &'a TradingObjectPools,
}

impl<'a> std::ops::Deref for PooledTrade<'a> {
    type Target = Trade;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for PooledTrade<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PooledTrade<'a> {
    fn drop(&mut self) {
        self.pools.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII wrapper for pooled OrderNode objects
pub struct PooledOrderNode<'a> {
    inner: PooledObject<LockFreeOrderNode>,
    pools: &'a TradingObjectPools,
}

impl<'a> std::ops::Deref for PooledOrderNode<'a> {
    type Target = LockFreeOrderNode;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for PooledOrderNode<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PooledOrderNode<'a> {
    fn drop(&mut self) {
        self.pools.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII wrapper for pooled PriceLevel objects
pub struct PooledPriceLevel<'a> {
    inner: PooledObject<LockFreePriceLevel>,
    pools: &'a TradingObjectPools,
}

impl<'a> std::ops::Deref for PooledPriceLevel<'a> {
    type Target = LockFreePriceLevel;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for PooledPriceLevel<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PooledPriceLevel<'a> {
    fn drop(&mut self) {
        self.pools.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAII wrapper for pooled buffer objects
pub struct PooledBuffer<'a> {
    inner: PooledObject<Vec<u8>>,
    pools: &'a TradingObjectPools,
}

impl<'a> PooledBuffer<'a> {
    fn from_numa_allocation(
        _allocation: super::numa_allocator::NumaAllocation,
        pools: &'a TradingObjectPools,
    ) -> Self {
        // In a real implementation, this would wrap the NUMA allocation
        // For now, we'll create a regular pooled buffer
        let inner = pools.buffer_pools[&4096].allocate().unwrap();
        Self { inner, pools }
    }
}

impl<'a> std::ops::Deref for PooledBuffer<'a> {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> std::ops::DerefMut for PooledBuffer<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PooledBuffer<'a> {
    fn drop(&mut self) {
        self.pools.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Comprehensive statistics for all trading pools
#[derive(Debug, Clone)]
pub struct TradingPoolStats {
    pub order_pool: PoolStats,
    pub trade_pool: PoolStats,
    pub order_node_pool: PoolStats,
    pub price_level_pool: PoolStats,
    pub buffer_pool_stats: HashMap<usize, PoolStats>,
    pub numa_stats: HashMap<usize, super::numa_allocator::NumaNodeStatsSnapshot>,
    pub manager_stats: PoolManagerStatsSnapshot,
}

/// Snapshot of pool manager statistics
#[derive(Debug, Clone)]
pub struct PoolManagerStatsSnapshot {
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub pool_expansions: usize,
    pub allocation_failures: usize,
    pub cross_numa_allocations: usize,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub order_pool_bytes: usize,
    pub trade_pool_bytes: usize,
    pub order_node_pool_bytes: usize,
    pub price_level_pool_bytes: usize,
    pub buffer_pool_bytes: usize,
    pub total_bytes: usize,
}

impl PoolManagerStats {
    fn new() -> Self {
        Self {
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
            pool_expansions: AtomicUsize::new(0),
            allocation_failures: AtomicUsize::new(0),
            cross_numa_allocations: AtomicUsize::new(0),
        }
    }
}

/// Errors related to object pool operations
#[derive(Debug)]
pub enum ObjectPoolError {
    PoolError(PoolError),
    NumaError(NumaError),
    InitializationFailed,
    InvalidSize(usize),
}

impl std::fmt::Display for ObjectPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectPoolError::PoolError(e) => write!(f, "Pool error: {}", e),
            ObjectPoolError::NumaError(e) => write!(f, "NUMA error: {}", e),
            ObjectPoolError::InitializationFailed => write!(f, "Pool initialization failed"),
            ObjectPoolError::InvalidSize(size) => write!(f, "Invalid size: {}", size),
        }
    }
}

impl std::error::Error for ObjectPoolError {}

impl From<PoolError> for ObjectPoolError {
    fn from(error: PoolError) -> Self {
        ObjectPoolError::PoolError(error)
    }
}

impl From<NumaError> for ObjectPoolError {
    fn from(error: NumaError) -> Self {
        ObjectPoolError::NumaError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{OrderId, Symbol, Side, OrderType};

    #[test]
    fn test_trading_pools_creation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        let stats = pools.get_comprehensive_stats();
        
        assert!(stats.order_pool.capacity > 0);
        assert!(stats.trade_pool.capacity > 0);
        assert!(stats.order_node_pool.capacity > 0);
        assert!(stats.price_level_pool.capacity > 0);
    }

    #[test]
    fn test_order_allocation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let order = pools.create_order(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            OrderType::Limit,
            50000,
            100,
            1000,
        ).unwrap();
        
        assert_eq!(order.id, OrderId::new(1));
        assert_eq!(order.price, 50000);
        assert_eq!(order.size, 100);
    }

    #[test]
    fn test_trade_allocation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let trade = pools.create_trade(
            1,
            100,
            200,
            Symbol::new("BTCUSD").unwrap(),
            50000,
            50,
            1001,
        ).unwrap();
        
        assert_eq!(trade.id, 1);
        assert_eq!(trade.buyer_order_id, 100);
        assert_eq!(trade.seller_order_id, 200);
        assert_eq!(trade.price, 50000);
        assert_eq!(trade.size, 50);
    }

    #[test]
    fn test_buffer_allocation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let mut buffer = pools.allocate_buffer(256).unwrap();
        assert_eq!(buffer.len(), 256);
        
        // Test writing to buffer
        buffer[0] = 42;
        buffer[255] = 24;
        
        assert_eq!(buffer[0], 42);
        assert_eq!(buffer[255], 24);
    }

    #[test]
    fn test_order_node_allocation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let order = Order {
            id: OrderId::new(1),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: 50000,
            size: 100,
            timestamp: 1000,
        };
        
        let node = pools.allocate_order_node(order.clone()).unwrap();
        assert_eq!(node.order.id, order.id);
        assert_eq!(node.order.price, order.price);
    }

    #[test]
    fn test_price_level_allocation() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let level = pools.allocate_price_level(50000).unwrap();
        assert_eq!(level.price, 50000);
        assert_eq!(level.get_total_volume(), 0);
        assert_eq!(level.get_order_count(), 0);
    }

    #[test]
    fn test_concurrent_allocation() {
        use std::sync::Arc;
        use std::thread;
        
        let pools = Arc::new(TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap());
        let mut handles = vec![];

        for i in 0..4 {
            let pools_clone = pools.clone();
            let handle = thread::spawn(move || {
                let mut orders = vec![];
                
                for j in 0..100 {
                    let order = pools_clone.create_order(
                        OrderId::new((i * 100 + j) as u64),
                        Symbol::new("BTCUSD").unwrap(),
                        Side::Buy,
                        OrderType::Limit,
                        50000,
                        100,
                        1000,
                    ).unwrap();
                    orders.push(order);
                }
                
                orders.len()
            });
            handles.push(handle);
        }

        let mut total_orders = 0;
        for handle in handles {
            total_orders += handle.join().unwrap();
        }

        assert_eq!(total_orders, 400);
        
        let stats = pools.get_comprehensive_stats();
        assert_eq!(stats.manager_stats.total_allocations, 400);
    }

    #[test]
    fn test_pool_warm_up() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let stats_before = pools.get_comprehensive_stats();
        pools.warm_up().unwrap();
        let stats_after = pools.get_comprehensive_stats();
        
        // Should have allocated and deallocated objects during warm-up
        assert!(stats_after.manager_stats.total_allocations > stats_before.manager_stats.total_allocations);
    }

    #[test]
    fn test_memory_usage_stats() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        // Allocate some objects
        let _order = pools.allocate_order().unwrap();
        let _trade = pools.allocate_trade().unwrap();
        let _buffer = pools.allocate_buffer(1024).unwrap();
        
        let memory_stats = pools.get_memory_usage();
        assert!(memory_stats.total_bytes > 0);
        assert!(memory_stats.order_pool_bytes > 0);
        assert!(memory_stats.trade_pool_bytes > 0);
        assert!(memory_stats.buffer_pool_bytes > 0);
    }

    #[test]
    fn test_buffer_size_selection() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        assert_eq!(pools.get_buffer_pool_size(32), 64);
        assert_eq!(pools.get_buffer_pool_size(64), 64);
        assert_eq!(pools.get_buffer_pool_size(65), 128);
        assert_eq!(pools.get_buffer_pool_size(1000), 1024);
        assert_eq!(pools.get_buffer_pool_size(5000), 8192); // Rounded up to 4KB boundary
    }

    #[test]
    fn test_order_cloning() {
        let pools = TradingObjectPools::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let original_order = pools.create_order(
            OrderId::new(1),
            Symbol::new("BTCUSD").unwrap(),
            Side::Buy,
            OrderType::Limit,
            50000,
            100,
            1000,
        ).unwrap();
        
        let cloned_order = original_order.clone_order().unwrap();
        
        assert_eq!(original_order.id, cloned_order.id);
        assert_eq!(original_order.price, cloned_order.price);
        assert_eq!(original_order.size, cloned_order.size);
    }
}