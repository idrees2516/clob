use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use super::hazard_pointers::{HazardPointerManager, HazardPointer};
use super::memory_reclamation::EpochBasedReclamation;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;
use std::marker::PhantomData;

/// Default initial pool size
const DEFAULT_POOL_SIZE: usize = 1024;

/// Maximum pool size to prevent unbounded growth
const MAX_POOL_SIZE: usize = 65536;

/// Pool growth factor when expanding
const GROWTH_FACTOR: usize = 2;

/// Lock-free object pool node
#[repr(align(64))]
pub struct PoolNode<T> {
    /// The pooled object (None when node is in free list)
    pub object: Option<T>,
    
    /// Next node in the free list
    pub next: AlignedAtomicPtr<PoolNode<T>>,
    
    /// Epoch for memory reclamation
    pub epoch: AtomicU64,
    
    /// Reference count for safe access
    pub ref_count: AtomicU64,
    
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl<T> PoolNode<T> {
    /// Create a new pool node
    pub fn new() -> Self {
        Self {
            object: None,
            next: AlignedAtomicPtr::new(ptr::null_mut()),
            epoch: AtomicU64::new(0),
            ref_count: AtomicU64::new(0),
            _padding: [],
        }
    }

    /// Create a new pool node with an object
    pub fn with_object(object: T) -> Self {
        Self {
            object: Some(object),
            next: AlignedAtomicPtr::new(ptr::null_mut()),
            epoch: AtomicU64::new(0),
            ref_count: AtomicU64::new(0),
            _padding: [],
        }
    }

    /// Get the next node in the free list
    #[inline(always)]
    pub fn get_next(&self) -> *mut PoolNode<T> {
        self.next.load(MemoryOrder::Acquire)
    }

    /// Set the next node using compare-and-swap
    #[inline(always)]
    pub fn set_next(&self, expected: *mut PoolNode<T>, new: *mut PoolNode<T>) -> Result<*mut PoolNode<T>, *mut PoolNode<T>> {
        self.next.compare_exchange_weak(expected, new, MemoryOrder::Release, MemoryOrder::Relaxed)
    }

    /// Store the next node
    #[inline(always)]
    pub fn store_next(&self, next: *mut PoolNode<T>) {
        self.next.store(next, MemoryOrder::Release);
    }

    /// Increment reference count
    #[inline(always)]
    pub fn acquire(&self) -> u64 {
        self.ref_count.fetch_add(1, Ordering::AcqRel)
    }

    /// Decrement reference count
    #[inline(always)]
    pub fn release(&self) -> u64 {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) - 1
    }

    /// Get current reference count
    #[inline(always)]
    pub fn get_ref_count(&self) -> u64 {
        self.ref_count.load(Ordering::Acquire)
    }
}

/// Lock-free object pool for high-performance memory management
pub struct LockFreeObjectPool<T> {
    /// Head of the free list
    free_head: AlignedAtomicPtr<PoolNode<T>>,
    
    /// Number of objects currently in the pool
    pool_size: AtomicUsize,
    
    /// Total number of objects allocated (including those in use)
    total_allocated: AtomicU64,
    
    /// Number of objects currently in use
    objects_in_use: AtomicU64,
    
    /// Maximum pool size
    max_pool_size: usize,
    
    /// Hazard pointer manager for memory safety
    hazard_manager: Arc<HazardPointerManager>,
    
    /// Epoch-based reclamation for better performance
    epoch_manager: Arc<EpochBasedReclamation>,
    
    /// Factory function for creating new objects
    factory: Option<Box<dyn Fn() -> T + Send + Sync>>,
    
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<T> LockFreeObjectPool<T>
where
    T: Send + Sync,
{
    /// Create a new lock-free object pool
    pub fn new(max_threads: usize) -> Self {
        Self {
            free_head: AlignedAtomicPtr::new(ptr::null_mut()),
            pool_size: AtomicUsize::new(0),
            total_allocated: AtomicU64::new(0),
            objects_in_use: AtomicU64::new(0),
            max_pool_size: MAX_POOL_SIZE,
            hazard_manager: Arc::new(HazardPointerManager::new(max_threads)),
            epoch_manager: Arc::new(EpochBasedReclamation::new(max_threads)),
            factory: None,
            _phantom: PhantomData,
        }
    }

    /// Create a new lock-free object pool with a factory function
    pub fn with_factory<F>(factory: F, max_threads: usize) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let mut pool = Self::new(max_threads);
        pool.factory = Some(Box::new(factory));
        pool
    }

    /// Create a new lock-free object pool with initial capacity
    pub fn with_capacity(initial_capacity: usize, max_threads: usize) -> Self
    where
        T: Default,
    {
        let mut pool = Self::new(max_threads);
        pool.factory = Some(Box::new(|| T::default()));
        
        // Pre-populate the pool
        for _ in 0..initial_capacity {
            let node = Box::into_raw(Box::new(PoolNode::with_object(T::default())));
            pool.return_node_to_pool(node);
        }
        
        pool
    }

    /// Acquire an object from the pool
    pub fn acquire(&self) -> Result<PooledObject<T>, PoolError> {
        let _guard = self.epoch_manager.pin();
        
        // Try to get an object from the free list
        if let Some(node) = self.pop_from_free_list() {
            unsafe {
                if let Some(object) = (*node).object.take() {
                    self.objects_in_use.fetch_add(1, Ordering::AcqRel);
                    return Ok(PooledObject::new(object, node, self));
                } else {
                    // Node has no object, return it to pool and try again
                    self.return_node_to_pool(node);
                }
            }
        }

        // No objects available in pool, create a new one
        if let Some(ref factory) = self.factory {
            let object = factory();
            let node = Box::into_raw(Box::new(PoolNode::new()));
            self.total_allocated.fetch_add(1, Ordering::AcqRel);
            self.objects_in_use.fetch_add(1, Ordering::AcqRel);
            Ok(PooledObject::new(object, node, self))
        } else {
            Err(PoolError::NoFactory)
        }
    }

    /// Try to acquire an object from the pool without blocking
    pub fn try_acquire(&self) -> Result<Option<PooledObject<T>>, PoolError> {
        let _guard = self.epoch_manager.pin();
        
        // Try to get an object from the free list
        if let Some(node) = self.pop_from_free_list() {
            unsafe {
                if let Some(object) = (*node).object.take() {
                    self.objects_in_use.fetch_add(1, Ordering::AcqRel);
                    return Ok(Some(PooledObject::new(object, node, self)));
                } else {
                    // Node has no object, return it to pool
                    self.return_node_to_pool(node);
                }
            }
        }

        Ok(None)
    }

    /// Pop a node from the free list
    fn pop_from_free_list(&self) -> Option<*mut PoolNode<T>> {
        let hazard = self.hazard_manager.acquire_hazard_pointer();

        loop {
            let head = self.free_head.load(MemoryOrder::Acquire);
            
            if head.is_null() {
                return None;
            }

            hazard.protect(head);
            
            // Verify head hasn't changed after protection
            if self.free_head.load(MemoryOrder::Acquire) != head {
                continue;
            }

            unsafe {
                let next = (*head).get_next();
                
                match self.free_head.compare_exchange_weak(
                    head,
                    next,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ) {
                    Ok(_) => {
                        self.pool_size.fetch_sub(1, Ordering::AcqRel);
                        return Some(head);
                    }
                    Err(_) => {
                        // Head changed, retry
                        continue;
                    }
                }
            }
        }
    }

    /// Return a node to the pool
    fn return_node_to_pool(&self, node: *mut PoolNode<T>) {
        if node.is_null() {
            return;
        }

        // Check if pool is at capacity
        if self.pool_size.load(Ordering::Acquire) >= self.max_pool_size {
            // Pool is full, just deallocate the node
            unsafe { Box::from_raw(node); }
            self.total_allocated.fetch_sub(1, Ordering::AcqRel);
            return;
        }

        loop {
            let current_head = self.free_head.load(MemoryOrder::Acquire);
            
            unsafe {
                (*node).store_next(current_head);
            }

            match self.free_head.compare_exchange_weak(
                current_head,
                node,
                MemoryOrder::Release,
                MemoryOrder::Relaxed,
            ) {
                Ok(_) => {
                    self.pool_size.fetch_add(1, Ordering::AcqRel);
                    break;
                }
                Err(_) => {
                    // Head changed, retry
                    continue;
                }
            }
        }
    }

    /// Return an object to the pool (called by PooledObject::drop)
    fn return_object(&self, object: T, node: *mut PoolNode<T>) {
        if !node.is_null() {
            unsafe {
                (*node).object = Some(object);
            }
            self.return_node_to_pool(node);
        } else {
            // No node provided, create a new one
            let new_node = Box::into_raw(Box::new(PoolNode::with_object(object)));
            self.return_node_to_pool(new_node);
        }
        
        self.objects_in_use.fetch_sub(1, Ordering::AcqRel);
    }

    /// Get the number of objects currently in the pool
    pub fn pool_size(&self) -> usize {
        self.pool_size.load(Ordering::Acquire)
    }

    /// Get the total number of objects allocated
    pub fn total_allocated(&self) -> u64 {
        self.total_allocated.load(Ordering::Acquire)
    }

    /// Get the number of objects currently in use
    pub fn objects_in_use(&self) -> u64 {
        self.objects_in_use.load(Ordering::Acquire)
    }

    /// Get the maximum pool size
    pub fn max_pool_size(&self) -> usize {
        self.max_pool_size
    }

    /// Set the maximum pool size
    pub fn set_max_pool_size(&mut self, max_size: usize) {
        self.max_pool_size = max_size;
    }

    /// Force memory reclamation
    pub fn force_reclaim(&self) {
        self.epoch_manager.force_reclaim();
        self.hazard_manager.force_reclaim();
    }

    /// Get statistics about the object pool
    pub fn get_stats(&self) -> ObjectPoolStats {
        ObjectPoolStats {
            pool_size: self.pool_size(),
            total_allocated: self.total_allocated(),
            objects_in_use: self.objects_in_use(),
            max_pool_size: self.max_pool_size(),
            utilization: if self.total_allocated() > 0 {
                self.objects_in_use() as f64 / self.total_allocated() as f64
            } else {
                0.0
            },
            hazard_stats: self.hazard_manager.get_stats(),
            epoch_stats: self.epoch_manager.get_stats(),
        }
    }

    /// Clear the pool and deallocate all objects
    pub fn clear(&self) {
        while let Some(node) = self.pop_from_free_list() {
            unsafe { Box::from_raw(node); }
            self.total_allocated.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl<T> Drop for LockFreeObjectPool<T> {
    fn drop(&mut self) {
        // Clear all objects from the pool
        self.clear();
        
        // Force reclamation
        self.force_reclaim();
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<T> {
    object: Option<T>,
    node: *mut PoolNode<T>,
    pool: *const LockFreeObjectPool<T>,
}

impl<T> PooledObject<T> {
    /// Create a new pooled object
    fn new(object: T, node: *mut PoolNode<T>, pool: &LockFreeObjectPool<T>) -> Self {
        Self {
            object: Some(object),
            node,
            pool: pool as *const _,
        }
    }

    /// Get a reference to the pooled object
    pub fn as_ref(&self) -> Option<&T> {
        self.object.as_ref()
    }

    /// Get a mutable reference to the pooled object
    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.object.as_mut()
    }

    /// Take ownership of the object (prevents return to pool)
    pub fn take(mut self) -> Option<T> {
        self.object.take()
    }
}

impl<T> std::ops::Deref for PooledObject<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.object.as_ref().expect("PooledObject should contain an object")
    }
}

impl<T> std::ops::DerefMut for PooledObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.object.as_mut().expect("PooledObject should contain an object")
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(object) = self.object.take() {
            unsafe {
                (*self.pool).return_object(object, self.node);
            }
        }
    }
}

unsafe impl<T: Send> Send for PooledObject<T> {}
unsafe impl<T: Sync> Sync for PooledObject<T> {}

/// Error types for object pool operations
#[derive(Debug, Clone)]
pub enum PoolError {
    NoFactory,
    PoolExhausted,
    InvalidOperation,
}

/// Statistics about object pool performance
#[derive(Debug, Clone)]
pub struct ObjectPoolStats {
    pub pool_size: usize,
    pub total_allocated: u64,
    pub objects_in_use: u64,
    pub max_pool_size: usize,
    pub utilization: f64,
    pub hazard_stats: super::hazard_pointers::HazardPointerStats,
    pub epoch_stats: super::memory_reclamation::EpochReclamationStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[derive(Debug, Clone, PartialEq)]
    struct TestObject {
        id: u64,
        data: String,
    }

    impl Default for TestObject {
        fn default() -> Self {
            Self {
                id: 0,
                data: "default".to_string(),
            }
        }
    }

    #[test]
    fn test_object_pool_creation() {
        let pool: LockFreeObjectPool<TestObject> = LockFreeObjectPool::new(4);
        assert_eq!(pool.pool_size(), 0);
        assert_eq!(pool.total_allocated(), 0);
        assert_eq!(pool.objects_in_use(), 0);
    }

    #[test]
    fn test_pool_with_factory() {
        let pool = LockFreeObjectPool::with_factory(
            || TestObject { id: 42, data: "test".to_string() },
            4
        );

        let obj = pool.acquire().unwrap();
        assert_eq!(obj.id, 42);
        assert_eq!(obj.data, "test");
        assert_eq!(pool.objects_in_use(), 1);
    }

    #[test]
    fn test_pool_with_capacity() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(10, 4);
        assert_eq!(pool.pool_size(), 10);
        assert_eq!(pool.total_allocated(), 10);
        assert_eq!(pool.objects_in_use(), 0);
    }

    #[test]
    fn test_acquire_and_return() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(5, 4);
        
        // Acquire an object
        let obj = pool.acquire().unwrap();
        assert_eq!(pool.objects_in_use(), 1);
        assert_eq!(pool.pool_size(), 4); // One less in pool
        
        // Return object by dropping it
        drop(obj);
        assert_eq!(pool.objects_in_use(), 0);
        assert_eq!(pool.pool_size(), 5); // Back in pool
    }

    #[test]
    fn test_try_acquire() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(2, 4);
        
        // Should succeed when objects are available
        let obj1 = pool.try_acquire().unwrap();
        assert!(obj1.is_some());
        
        let obj2 = pool.try_acquire().unwrap();
        assert!(obj2.is_some());
        
        // Should return None when pool is empty
        let obj3 = pool.try_acquire().unwrap();
        assert!(obj3.is_none());
    }

    #[test]
    fn test_object_modification() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(1, 4);
        
        {
            let mut obj = pool.acquire().unwrap();
            obj.id = 123;
            obj.data = "modified".to_string();
        } // Object returned to pool
        
        // Acquire again and verify modifications are preserved
        let obj = pool.acquire().unwrap();
        assert_eq!(obj.id, 123);
        assert_eq!(obj.data, "modified");
    }

    #[test]
    fn test_take_ownership() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(1, 4);
        
        let obj = pool.acquire().unwrap();
        let owned = obj.take().unwrap();
        
        assert_eq!(owned.id, 0);
        assert_eq!(pool.objects_in_use(), 0); // Object not returned to pool
    }

    #[test]
    fn test_concurrent_operations() {
        let pool = Arc::new(LockFreeObjectPool::<TestObject>::with_capacity(100, 20));
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 100;

        for thread_id in 0..num_threads {
            let pool_clone = pool.clone();
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    // Acquire object
                    let mut obj = pool_clone.acquire().unwrap();
                    
                    // Modify object
                    obj.id = (thread_id * ops_per_thread + i) as u64;
                    obj.data = format!("thread_{}_op_{}", thread_id, i);
                    
                    // Hold for a short time
                    thread::sleep(Duration::from_micros(1));
                    
                    // Object automatically returned when dropped
                }
            });
            
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(pool.objects_in_use(), 0);
        assert!(pool.pool_size() > 0);
    }

    #[test]
    fn test_pool_size_limit() {
        let mut pool = LockFreeObjectPool::<TestObject>::with_capacity(5, 4);
        pool.set_max_pool_size(3);

        // Acquire all objects
        let obj1 = pool.acquire().unwrap();
        let obj2 = pool.acquire().unwrap();
        let obj3 = pool.acquire().unwrap();
        let obj4 = pool.acquire().unwrap();
        let obj5 = pool.acquire().unwrap();

        assert_eq!(pool.objects_in_use(), 5);
        assert_eq!(pool.pool_size(), 0);

        // Return objects - only 3 should be kept in pool due to size limit
        drop(obj1);
        drop(obj2);
        drop(obj3);
        drop(obj4);
        drop(obj5);

        assert_eq!(pool.objects_in_use(), 0);
        assert!(pool.pool_size() <= 3);
    }

    #[test]
    fn test_stats() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(10, 4);
        
        // Acquire some objects
        let _obj1 = pool.acquire().unwrap();
        let _obj2 = pool.acquire().unwrap();

        let stats = pool.get_stats();
        assert_eq!(stats.total_allocated, 10);
        assert_eq!(stats.objects_in_use, 2);
        assert_eq!(stats.pool_size, 8);
        assert_eq!(stats.utilization, 0.2); // 2/10
    }

    #[test]
    fn test_clear_pool() {
        let pool = LockFreeObjectPool::<TestObject>::with_capacity(10, 4);
        
        assert_eq!(pool.pool_size(), 10);
        assert_eq!(pool.total_allocated(), 10);
        
        pool.clear();
        
        assert_eq!(pool.pool_size(), 0);
        assert_eq!(pool.total_allocated(), 0);
    }

    #[test]
    fn test_no_factory_error() {
        let pool: LockFreeObjectPool<TestObject> = LockFreeObjectPool::new(4);
        
        let result = pool.acquire();
        assert!(matches!(result, Err(PoolError::NoFactory)));
    }

    #[test]
    fn test_stress_test() {
        let pool = Arc::new(LockFreeObjectPool::<TestObject>::with_capacity(50, 50));
        let mut handles = vec![];
        let num_threads = 20;
        let ops_per_thread = 1000;

        for thread_id in 0..num_threads {
            let pool_clone = pool.clone();
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    if let Ok(mut obj) = pool_clone.acquire() {
                        obj.id = (thread_id * ops_per_thread + i) as u64;
                        
                        // Randomly hold the object for different durations
                        if i % 10 == 0 {
                            thread::sleep(Duration::from_micros(10));
                        }
                    }
                }
            });
            
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify no objects are leaked
        assert_eq!(pool.objects_in_use(), 0);
        
        let stats = pool.get_stats();
        assert!(stats.total_allocated > 0);
    }
}