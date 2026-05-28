use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicU32, AtomicUsize, Ordering};
use std::ptr;
use crate::math::fixed_point::FixedPoint;

/// Memory ordering for high-performance atomic operations
#[derive(Debug, Clone, Copy)]
pub enum MemoryOrder {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

impl From<MemoryOrder> for Ordering {
    fn from(order: MemoryOrder) -> Self {
        match order {
            MemoryOrder::Relaxed => Ordering::Relaxed,
            MemoryOrder::Acquire => Ordering::Acquire,
            MemoryOrder::Release => Ordering::Release,
            MemoryOrder::AcqRel => Ordering::AcqRel,
            MemoryOrder::SeqCst => Ordering::SeqCst,
        }
    }
}

/// High-performance atomic operations with optimized memory ordering
pub struct AtomicOperations;

impl AtomicOperations {
    /// Compare-and-swap with explicit memory ordering
    #[inline(always)]
    pub fn compare_exchange_weak<T>(
        atomic: &AtomicPtr<T>,
        current: *mut T,
        new: *mut T,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<*mut T, *mut T> {
        atomic.compare_exchange_weak(current, new, success.into(), failure.into())
    }

    /// Compare-and-swap with strong ordering guarantees
    #[inline(always)]
    pub fn compare_exchange_strong<T>(
        atomic: &AtomicPtr<T>,
        current: *mut T,
        new: *mut T,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<*mut T, *mut T> {
        atomic.compare_exchange(current, new, success.into(), failure.into())
    }

    /// Atomic load with specified memory ordering
    #[inline(always)]
    pub fn load<T>(atomic: &AtomicPtr<T>, order: MemoryOrder) -> *mut T {
        atomic.load(order.into())
    }

    /// Atomic store with specified memory ordering
    #[inline(always)]
    pub fn store<T>(atomic: &AtomicPtr<T>, value: *mut T, order: MemoryOrder) {
        atomic.store(value, order.into());
    }

    /// Atomic swap operation
    #[inline(always)]
    pub fn swap<T>(atomic: &AtomicPtr<T>, value: *mut T, order: MemoryOrder) -> *mut T {
        atomic.swap(value, order.into())
    }

    /// Atomic fetch-and-add for counters
    #[inline(always)]
    pub fn fetch_add_u64(atomic: &AtomicU64, value: u64, order: MemoryOrder) -> u64 {
        atomic.fetch_add(value, order.into())
    }

    /// Atomic fetch-and-sub for counters
    #[inline(always)]
    pub fn fetch_sub_u64(atomic: &AtomicU64, value: u64, order: MemoryOrder) -> u64 {
        atomic.fetch_sub(value, order.into())
    }

    /// Atomic compare-and-swap for u64 values
    #[inline(always)]
    pub fn compare_exchange_u64(
        atomic: &AtomicU64,
        current: u64,
        new: u64,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<u64, u64> {
        atomic.compare_exchange_weak(current, new, success.into(), failure.into())
    }

    /// Memory fence with specified ordering
    #[inline(always)]
    pub fn fence(order: MemoryOrder) {
        std::sync::atomic::fence(order.into());
    }

    /// Compiler fence to prevent reordering
    #[inline(always)]
    pub fn compiler_fence(order: MemoryOrder) {
        std::sync::atomic::compiler_fence(order.into());
    }
}

/// Cache-line aligned atomic pointer for preventing false sharing
#[repr(align(64))]
pub struct AlignedAtomicPtr<T> {
    ptr: AtomicPtr<T>,
    _padding: [u8; 64 - std::mem::size_of::<AtomicPtr<T>>()],
}

impl<T> AlignedAtomicPtr<T> {
    pub fn new(value: *mut T) -> Self {
        Self {
            ptr: AtomicPtr::new(value),
            _padding: [0; 64 - std::mem::size_of::<AtomicPtr<T>>()],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: MemoryOrder) -> *mut T {
        self.ptr.load(order.into())
    }

    #[inline(always)]
    pub fn store(&self, value: *mut T, order: MemoryOrder) {
        self.ptr.store(value, order.into());
    }

    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        current: *mut T,
        new: *mut T,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<*mut T, *mut T> {
        self.ptr.compare_exchange_weak(current, new, success.into(), failure.into())
    }

    #[inline(always)]
    pub fn swap(&self, value: *mut T, order: MemoryOrder) -> *mut T {
        self.ptr.swap(value, order.into())
    }
}

/// Cache-line aligned atomic u64 for high-performance counters
#[repr(align(64))]
pub struct AlignedAtomicU64 {
    value: AtomicU64,
    _padding: [u8; 64 - std::mem::size_of::<AtomicU64>()],
}

impl AlignedAtomicU64 {
    pub fn new(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
            _padding: [0; 64 - std::mem::size_of::<AtomicU64>()],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: MemoryOrder) -> u64 {
        self.value.load(order.into())
    }

    #[inline(always)]
    pub fn store(&self, value: u64, order: MemoryOrder) {
        self.value.store(value, order.into());
    }

    #[inline(always)]
    pub fn fetch_add(&self, value: u64, order: MemoryOrder) -> u64 {
        self.value.fetch_add(value, order.into())
    }

    #[inline(always)]
    pub fn fetch_sub(&self, value: u64, order: MemoryOrder) -> u64 {
        self.value.fetch_sub(value, order.into())
    }

    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: MemoryOrder,
        failure: MemoryOrder,
    ) -> Result<u64, u64> {
        self.value.compare_exchange_weak(current, new, success.into(), failure.into())
    }
}

/// Atomic operations with retry logic for high contention scenarios
pub struct RetryableAtomicOps;

impl RetryableAtomicOps {
    /// Retry compare-and-swap with exponential backoff
    pub fn retry_cas<T, F>(
        atomic: &AtomicPtr<T>,
        mut update_fn: F,
        max_retries: usize,
    ) -> Result<*mut T, ()>
    where
        F: FnMut(*mut T) -> *mut T,
    {
        let mut retries = 0;
        let mut backoff = 1;

        loop {
            let current = atomic.load(Ordering::Acquire);
            let new = update_fn(current);

            match atomic.compare_exchange_weak(
                current,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(old) => return Ok(old),
                Err(_) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(());
                    }

                    // Exponential backoff with jitter
                    for _ in 0..backoff {
                        std::hint::spin_loop();
                    }
                    backoff = (backoff * 2).min(64);
                }
            }
        }
    }

    /// Retry atomic increment with overflow protection
    pub fn retry_increment(
        atomic: &AtomicU64,
        increment: u64,
        max_value: u64,
        max_retries: usize,
    ) -> Result<u64, ()> {
        let mut retries = 0;

        loop {
            let current = atomic.load(Ordering::Acquire);
            
            if current.saturating_add(increment) > max_value {
                return Err(());
            }

            let new = current + increment;

            match atomic.compare_exchange_weak(
                current,
                new,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(old) => return Ok(old),
                Err(_) => {
                    retries += 1;
                    if retries >= max_retries {
                        return Err(());
                    }
                    std::hint::spin_loop();
                }
            }
        }
    }
}

/// Atomic FixedPoint for ultra-low latency financial calculations
#[repr(align(64))]
pub struct AtomicFixedPoint {
    value: AtomicU64,
    _padding: [u8; 64 - std::mem::size_of::<AtomicU64>()],
}

impl AtomicFixedPoint {
    pub fn new(value: FixedPoint) -> Self {
        Self {
            value: AtomicU64::new(value.to_raw()),
            _padding: [0; 64 - std::mem::size_of::<AtomicU64>()],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: Ordering) -> FixedPoint {
        FixedPoint::from_raw(self.value.load(order))
    }

    #[inline(always)]
    pub fn store(&self, value: FixedPoint, order: Ordering) {
        self.value.store(value.to_raw(), order);
    }

    #[inline(always)]
    pub fn compare_exchange_weak(
        &self,
        current: FixedPoint,
        new: FixedPoint,
        success: Ordering,
        failure: Ordering,
    ) -> Result<FixedPoint, FixedPoint> {
        match self.value.compare_exchange_weak(
            current.to_raw(),
            new.to_raw(),
            success,
            failure,
        ) {
            Ok(old) => Ok(FixedPoint::from_raw(old)),
            Err(actual) => Err(FixedPoint::from_raw(actual)),
        }
    }

    #[inline(always)]
    pub fn fetch_add(&self, value: FixedPoint, order: Ordering) -> FixedPoint {
        // Note: This is a simplified implementation for demonstration
        // In practice, you'd need proper fixed-point arithmetic
        let old = self.value.fetch_add(value.to_raw(), order);
        FixedPoint::from_raw(old)
    }

    #[inline(always)]
    pub fn fetch_sub(&self, value: FixedPoint, order: Ordering) -> FixedPoint {
        let old = self.value.fetch_sub(value.to_raw(), order);
        FixedPoint::from_raw(old)
    }

    #[inline(always)]
    pub fn swap(&self, value: FixedPoint, order: Ordering) -> FixedPoint {
        FixedPoint::from_raw(self.value.swap(value.to_raw(), order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_aligned_atomic_ptr() {
        let aligned = AlignedAtomicPtr::new(ptr::null_mut::<u64>());
        let test_value = Box::into_raw(Box::new(42u64));
        
        aligned.store(test_value, MemoryOrder::Release);
        let loaded = aligned.load(MemoryOrder::Acquire);
        
        assert_eq!(loaded, test_value);
        
        // Cleanup
        unsafe { Box::from_raw(test_value); }
    }

    #[test]
    fn test_aligned_atomic_u64() {
        let aligned = AlignedAtomicU64::new(0);
        
        let old = aligned.fetch_add(10, MemoryOrder::AcqRel);
        assert_eq!(old, 0);
        assert_eq!(aligned.load(MemoryOrder::Acquire), 10);
    }

    #[test]
    fn test_retryable_cas_concurrent() {
        let atomic = Arc::new(AtomicPtr::new(ptr::null_mut::<u64>()));
        let mut handles = vec![];

        for i in 0..10 {
            let atomic_clone = atomic.clone();
            let handle = thread::spawn(move || {
                let value = Box::into_raw(Box::new(i));
                RetryableAtomicOps::retry_cas(
                    &atomic_clone,
                    |_current| value,
                    100,
                ).is_ok()
            });
            handles.push(handle);
        }

        let mut success_count = 0;
        for handle in handles {
            if handle.join().unwrap() {
                success_count += 1;
            }
        }

        // At least one thread should succeed
        assert!(success_count >= 1);

        // Cleanup
        let final_ptr = atomic.load(Ordering::Acquire);
        if !final_ptr.is_null() {
            unsafe { Box::from_raw(final_ptr); }
        }
    }

    #[test]
    fn test_memory_ordering() {
        let atomic = AtomicPtr::new(ptr::null_mut::<u64>());
        let test_value = Box::into_raw(Box::new(42u64));

        AtomicOperations::store(&atomic, test_value, MemoryOrder::Release);
        let loaded = AtomicOperations::load(&atomic, MemoryOrder::Acquire);

        assert_eq!(loaded, test_value);

        // Cleanup
        unsafe { Box::from_raw(test_value); }
    }
}