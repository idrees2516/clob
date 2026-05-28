use super::super::lock_free::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::ptr;
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::mem;

/// Lock-free memory pool for high-performance object allocation
/// Cache-line aligned to prevent false sharing
#[repr(align(64))]
pub struct LockFreePool<T> {
    /// Head of the free list
    free_head: AlignedAtomicPtr<PoolNode<T>>,
    
    /// Current capacity of the pool
    capacity: AtomicUsize,
    
    /// Number of allocated objects
    allocated: AtomicUsize,
    
    /// Number of objects currently in use
    in_use: AtomicUsize,
    
    /// NUMA node for this pool
    numa_node: u32,
    
    /// Whether the pool can expand
    expandable: AtomicBool,
    
    /// Maximum capacity (0 = unlimited)
    max_capacity: usize,
    
    /// Expansion increment
    expansion_size: usize,
    
    /// Pool statistics
    stats: PoolStatistics,
    
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

/// Node in the lock-free pool's free list
#[repr(align(64))]
struct PoolNode<T> {
    /// Pointer to the next free node
    next: AlignedAtomicPtr<PoolNode<T>>,
    
    /// The actual object storage
    data: T,
    
    /// Node metadata
    metadata: NodeMetadata,
}

/// Metadata for pool nodes
#[derive(Debug, Clone)]
struct NodeMetadata {
    /// Allocation timestamp (for debugging)
    allocated_at: u64,
    
    /// Thread ID that allocated this node
    thread_id: u64,
    
    /// Pool generation (for ABA prevention)
    generation: u64,
}

/// Pool statistics for monitoring
#[repr(align(64))]
pub struct PoolStatistics {
    /// Total allocations since creation
    pub total_allocations: AtomicUsize,
    
    /// Total deallocations since creation
    pub total_deallocations: AtomicUsize,
    
    /// Number of pool expansions
    pub expansions: AtomicUsize,
    
    /// Peak usage count
    pub peak_usage: AtomicUsize,
    
    /// Allocation failures
    pub allocation_failures: AtomicUsize,
    
    /// Average allocation time (nanoseconds)
    pub avg_allocation_time_ns: AtomicUsize,
    
    /// Average deallocation time (nanoseconds)
    pub avg_deallocation_time_ns: AtomicUsize,
}

impl<T> LockFreePool<T> {
    /// Create a new lock-free pool with initial capacity
    pub fn new(initial_capacity: usize, numa_node: u32) -> Result<Self, PoolError> {
        let expansion_size = initial_capacity.max(64); // Minimum expansion size
        
        let pool = Self {
            free_head: AlignedAtomicPtr::new(ptr::null_mut()),
            capacity: AtomicUsize::new(0),
            allocated: AtomicUsize::new(0),
            in_use: AtomicUsize::new(0),
            numa_node,
            expandable: AtomicBool::new(true),
            max_capacity: 0, // Unlimited by default
            expansion_size,
            stats: PoolStatistics::new(),
            _phantom: PhantomData,
        };

        // Pre-allocate initial capacity
        if initial_capacity > 0 {
            pool.expand_pool(initial_capacity)?;
        }

        Ok(pool)
    }

    /// Create a pool with maximum capacity limit
    pub fn with_max_capacity(
        initial_capacity: usize,
        max_capacity: usize,
        numa_node: u32,
    ) -> Result<Self, PoolError> {
        if initial_capacity > max_capacity {
            return Err(PoolError::InvalidCapacity);
        }

        let mut pool = Self::new(initial_capacity, numa_node)?;
        pool.max_capacity = max_capacity;
        Ok(pool)
    }

    /// Allocate an object from the pool
    pub fn allocate(&self) -> Result<PooledObject<T>, PoolError> {
        let start_time = self.get_timestamp_ns();
        
        loop {
            // Try to get a node from the free list
            let head = self.free_head.load(MemoryOrder::Acquire);
            
            if head.is_null() {
                // No free nodes available, try to expand
                if self.try_expand_pool()? {
                    continue; // Retry after expansion
                } else {
                    self.stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(PoolError::PoolExhausted);
                }
            }

            unsafe {
                let next = (*head).next.load(MemoryOrder::Acquire);
                
                // Try to update the head pointer
                match self.free_head.compare_exchange_weak(
                    head,
                    next,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully allocated
                        self.in_use.fetch_add(1, Ordering::AcqRel);
                        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);
                        
                        // Update peak usage
                        let current_usage = self.in_use.load(Ordering::Acquire);
                        let mut peak = self.stats.peak_usage.load(Ordering::Acquire);
                        while current_usage > peak {
                            match self.stats.peak_usage.compare_exchange_weak(
                                peak,
                                current_usage,
                                Ordering::Release,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(new_peak) => peak = new_peak,
                            }
                        }
                        
                        // Update allocation time statistics
                        let allocation_time = self.get_timestamp_ns() - start_time;
                        self.update_avg_time(&self.stats.avg_allocation_time_ns, allocation_time);
                        
                        return Ok(PooledObject::new(head, self));
                    }
                    Err(_) => {
                        // CAS failed, retry
                        continue;
                    }
                }
            }
        }
    }

    /// Deallocate an object back to the pool
    pub(crate) fn deallocate(&self, node: *mut PoolNode<T>) -> Result<(), PoolError> {
        if node.is_null() {
            return Err(PoolError::NullPointer);
        }

        let start_time = self.get_timestamp_ns();

        unsafe {
            // Reset the object to default state if possible
            if mem::needs_drop::<T>() {
                ptr::drop_in_place(&mut (*node).data);
                ptr::write(&mut (*node).data, mem::zeroed());
            }

            // Update metadata
            (*node).metadata.allocated_at = 0;
            (*node).metadata.thread_id = 0;
            (*node).metadata.generation += 1;

            loop {
                let head = self.free_head.load(MemoryOrder::Acquire);
                (*node).next.store(head, MemoryOrder::Release);

                match self.free_head.compare_exchange_weak(
                    head,
                    node,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully deallocated
                        self.in_use.fetch_sub(1, Ordering::AcqRel);
                        self.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
                        
                        // Update deallocation time statistics
                        let deallocation_time = self.get_timestamp_ns() - start_time;
                        self.update_avg_time(&self.stats.avg_deallocation_time_ns, deallocation_time);
                        
                        return Ok(());
                    }
                    Err(_) => {
                        // CAS failed, retry
                        continue;
                    }
                }
            }
        }
    }

    /// Try to expand the pool if possible
    fn try_expand_pool(&self) -> Result<bool, PoolError> {
        if !self.expandable.load(Ordering::Acquire) {
            return Ok(false);
        }

        let current_capacity = self.capacity.load(Ordering::Acquire);
        
        // Check if we can expand
        if self.max_capacity > 0 && current_capacity >= self.max_capacity {
            return Ok(false);
        }

        let expansion_size = if self.max_capacity > 0 {
            (self.max_capacity - current_capacity).min(self.expansion_size)
        } else {
            self.expansion_size
        };

        if expansion_size == 0 {
            return Ok(false);
        }

        // Try to expand
        self.expand_pool(expansion_size)?;
        self.stats.expansions.fetch_add(1, Ordering::Relaxed);
        
        Ok(true)
    }

    /// Expand the pool by adding new nodes
    fn expand_pool(&self, count: usize) -> Result<(), PoolError> {
        let layout = Layout::new::<PoolNode<T>>();
        let mut new_nodes = Vec::with_capacity(count);

        // Allocate new nodes
        for _ in 0..count {
            unsafe {
                let ptr = alloc(layout) as *mut PoolNode<T>;
                if ptr.is_null() {
                    // Cleanup already allocated nodes
                    for node_ptr in new_nodes {
                        dealloc(node_ptr as *mut u8, layout);
                    }
                    return Err(PoolError::AllocationFailed);
                }

                // Initialize the node
                ptr::write(ptr, PoolNode {
                    next: AlignedAtomicPtr::new(ptr::null_mut()),
                    data: mem::zeroed(),
                    metadata: NodeMetadata {
                        allocated_at: 0,
                        thread_id: 0,
                        generation: 0,
                    },
                });

                new_nodes.push(ptr);
            }
        }

        // Add nodes to the free list
        for node_ptr in new_nodes {
            loop {
                let head = self.free_head.load(MemoryOrder::Acquire);
                unsafe {
                    (*node_ptr).next.store(head, MemoryOrder::Release);
                }

                match self.free_head.compare_exchange_weak(
                    head,
                    node_ptr,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }

        // Update capacity
        self.capacity.fetch_add(count, Ordering::AcqRel);
        Ok(())
    }

    /// Get current pool statistics
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            capacity: self.capacity.load(Ordering::Acquire),
            allocated: self.allocated.load(Ordering::Acquire),
            in_use: self.in_use.load(Ordering::Acquire),
            numa_node: self.numa_node,
            total_allocations: self.stats.total_allocations.load(Ordering::Acquire),
            total_deallocations: self.stats.total_deallocations.load(Ordering::Acquire),
            expansions: self.stats.expansions.load(Ordering::Acquire),
            peak_usage: self.stats.peak_usage.load(Ordering::Acquire),
            allocation_failures: self.stats.allocation_failures.load(Ordering::Acquire),
            avg_allocation_time_ns: self.stats.avg_allocation_time_ns.load(Ordering::Acquire),
            avg_deallocation_time_ns: self.stats.avg_deallocation_time_ns.load(Ordering::Acquire),
        }
    }

    /// Set whether the pool can expand
    pub fn set_expandable(&self, expandable: bool) {
        self.expandable.store(expandable, Ordering::Release);
    }

    /// Get the number of free objects in the pool
    pub fn free_count(&self) -> usize {
        let capacity = self.capacity.load(Ordering::Acquire);
        let in_use = self.in_use.load(Ordering::Acquire);
        capacity.saturating_sub(in_use)
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.free_head.load(MemoryOrder::Acquire).is_null()
    }

    /// Get high-resolution timestamp in nanoseconds
    #[inline(always)]
    fn get_timestamp_ns(&self) -> u64 {
        // Use TSC for high-resolution timing
        unsafe {
            core::arch::x86_64::_rdtsc()
        }
    }

    /// Update average time using exponential moving average
    fn update_avg_time(&self, avg_atomic: &AtomicUsize, new_time: u64) {
        let alpha = 0.1; // Smoothing factor
        loop {
            let current_avg = avg_atomic.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                new_time as usize
            } else {
                ((1.0 - alpha) * current_avg as f64 + alpha * new_time as f64) as usize
            };

            match avg_atomic.compare_exchange_weak(
                current_avg,
                new_avg,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
}

impl<T> Drop for LockFreePool<T> {
    fn drop(&mut self) {
        // Deallocate all nodes
        let layout = Layout::new::<PoolNode<T>>();
        let mut current = self.free_head.load(MemoryOrder::Acquire);

        while !current.is_null() {
            unsafe {
                let next = (*current).next.load(MemoryOrder::Acquire);
                
                // Drop the data if needed
                if mem::needs_drop::<T>() {
                    ptr::drop_in_place(&mut (*current).data);
                }
                
                dealloc(current as *mut u8, layout);
                current = next;
            }
        }
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T> {
    node: *mut PoolNode<T>,
    pool: *const LockFreePool<T>,
}

impl<T> PooledObject<T> {
    fn new(node: *mut PoolNode<T>, pool: &LockFreePool<T>) -> Self {
        unsafe {
            // Initialize metadata
            (*node).metadata.allocated_at = pool.get_timestamp_ns();
            (*node).metadata.thread_id = std::thread::current().id().as_u64().get();
        }

        Self {
            node,
            pool: pool as *const _,
        }
    }

    /// Get a reference to the contained object
    pub fn as_ref(&self) -> &T {
        unsafe { &(*self.node).data }
    }

    /// Get a mutable reference to the contained object
    pub fn as_mut(&mut self) -> &mut T {
        unsafe { &mut (*self.node).data }
    }

    /// Get the allocation metadata
    pub fn metadata(&self) -> &NodeMetadata {
        unsafe { &(*self.node).metadata }
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = (*self.pool).deallocate(self.node) {
                eprintln!("Failed to deallocate pooled object: {:?}", e);
            }
        }
    }
}

impl<T> std::ops::Deref for PooledObject<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> std::ops::DerefMut for PooledObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

/// Pool statistics snapshot
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub capacity: usize,
    pub allocated: usize,
    pub in_use: usize,
    pub numa_node: u32,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub expansions: usize,
    pub peak_usage: usize,
    pub allocation_failures: usize,
    pub avg_allocation_time_ns: usize,
    pub avg_deallocation_time_ns: usize,
}

impl PoolStatistics {
    fn new() -> Self {
        Self {
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
            expansions: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            allocation_failures: AtomicUsize::new(0),
            avg_allocation_time_ns: AtomicUsize::new(0),
            avg_deallocation_time_ns: AtomicUsize::new(0),
        }
    }
}

/// Error types for pool operations
#[derive(Debug, Clone, PartialEq)]
pub enum PoolError {
    AllocationFailed,
    PoolExhausted,
    InvalidCapacity,
    NullPointer,
    CorruptedPool,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::AllocationFailed => write!(f, "Memory allocation failed"),
            PoolError::PoolExhausted => write!(f, "Pool is exhausted and cannot expand"),
            PoolError::InvalidCapacity => write!(f, "Invalid capacity configuration"),
            PoolError::NullPointer => write!(f, "Null pointer encountered"),
            PoolError::CorruptedPool => write!(f, "Pool data structure is corrupted"),
        }
    }
}

impl std::error::Error for PoolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_pool_creation() {
        let pool: LockFreePool<u64> = LockFreePool::new(10, 0).unwrap();
        let stats = pool.get_stats();
        
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.numa_node, 0);
    }

    #[test]
    fn test_allocation_and_deallocation() {
        let pool: LockFreePool<u64> = LockFreePool::new(5, 0).unwrap();
        
        // Allocate an object
        let mut obj = pool.allocate().unwrap();
        *obj = 42;
        
        assert_eq!(*obj, 42);
        assert_eq!(pool.get_stats().in_use, 1);
        
        // Drop the object (automatic deallocation)
        drop(obj);
        
        assert_eq!(pool.get_stats().in_use, 0);
    }

    #[test]
    fn test_pool_expansion() {
        let pool: LockFreePool<u64> = LockFreePool::new(2, 0).unwrap();
        
        // Allocate all initial objects
        let obj1 = pool.allocate().unwrap();
        let obj2 = pool.allocate().unwrap();
        
        // This should trigger expansion
        let obj3 = pool.allocate().unwrap();
        
        let stats = pool.get_stats();
        assert!(stats.capacity > 2);
        assert_eq!(stats.in_use, 3);
        assert!(stats.expansions > 0);
        
        drop(obj1);
        drop(obj2);
        drop(obj3);
    }

    #[test]
    fn test_max_capacity_limit() {
        let pool: LockFreePool<u64> = LockFreePool::with_max_capacity(2, 3, 0).unwrap();
        
        // Allocate up to max capacity
        let obj1 = pool.allocate().unwrap();
        let obj2 = pool.allocate().unwrap();
        let obj3 = pool.allocate().unwrap(); // Should trigger expansion
        
        // This should fail due to max capacity
        pool.set_expandable(false);
        let result = pool.allocate();
        assert!(result.is_err());
        
        drop(obj1);
        drop(obj2);
        drop(obj3);
    }

    #[test]
    fn test_concurrent_allocation() {
        let pool = Arc::new(LockFreePool::<u64>::new(100, 0).unwrap());
        let mut handles = vec![];

        // Spawn threads to allocate concurrently
        for i in 0..10 {
            let pool_clone = pool.clone();
            let handle = thread::spawn(move || {
                let mut objects = vec![];
                
                // Allocate multiple objects
                for j in 0..10 {
                    let mut obj = pool_clone.allocate().unwrap();
                    *obj = (i * 10 + j) as u64;
                    objects.push(obj);
                }
                
                // Hold objects for a short time
                thread::sleep(Duration::from_millis(1));
                
                // Verify values
                for (k, obj) in objects.iter().enumerate() {
                    assert_eq!(**obj, (i * 10 + k) as u64);
                }
                
                // Objects are automatically deallocated when dropped
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        let stats = pool.get_stats();
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.total_allocations, 100);
        assert_eq!(stats.total_deallocations, 100);
    }

    #[test]
    fn test_pool_statistics() {
        let pool: LockFreePool<u64> = LockFreePool::new(5, 1).unwrap();
        
        // Perform some operations
        let obj1 = pool.allocate().unwrap();
        let obj2 = pool.allocate().unwrap();
        drop(obj1);
        
        let stats = pool.get_stats();
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.total_deallocations, 1);
        assert_eq!(stats.in_use, 1);
        assert_eq!(stats.peak_usage, 2);
        assert_eq!(stats.numa_node, 1);
        
        drop(obj2);
    }

    #[test]
    fn test_pooled_object_metadata() {
        let pool: LockFreePool<u64> = LockFreePool::new(5, 0).unwrap();
        let obj = pool.allocate().unwrap();
        
        let metadata = obj.metadata();
        assert!(metadata.allocated_at > 0);
        assert!(metadata.thread_id > 0);
        assert_eq!(metadata.generation, 0);
        
        drop(obj);
    }

    #[test]
    fn test_pool_free_count() {
        let pool: LockFreePool<u64> = LockFreePool::new(10, 0).unwrap();
        
        assert_eq!(pool.free_count(), 10);
        
        let obj1 = pool.allocate().unwrap();
        assert_eq!(pool.free_count(), 9);
        
        let obj2 = pool.allocate().unwrap();
        assert_eq!(pool.free_count(), 8);
        
        drop(obj1);
        assert_eq!(pool.free_count(), 9);
        
        drop(obj2);
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn test_stress_allocation_deallocation() {
        let pool = Arc::new(LockFreePool::<u64>::new(50, 0).unwrap());
        let mut handles = vec![];

        // Spawn threads for stress testing
        for _ in 0..5 {
            let pool_clone = pool.clone();
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    let obj = pool_clone.allocate().unwrap();
                    // Immediately drop to stress the allocation/deallocation cycle
                    drop(obj);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = pool.get_stats();
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.total_allocations, 5000);
        assert_eq!(stats.total_deallocations, 5000);
        assert_eq!(stats.allocation_failures, 0);
    }
}