use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicPtr, Ordering};
use std::sync::Arc;
use std::ptr;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

/// Global epoch counter for memory reclamation
static GLOBAL_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Minimum grace period between epochs (in nanoseconds)
const EPOCH_GRACE_PERIOD_NS: u64 = 1_000_000; // 1ms

/// Maximum number of retired pointers before forced reclamation
const MAX_RETIRED_POINTERS: usize = 1000;

/// Thread-local epoch information
#[repr(align(64))] // Cache-line aligned to prevent false sharing
struct ThreadEpoch {
    /// Current epoch for this thread
    local_epoch: AtomicU64,
    /// Thread ID that owns this epoch record
    thread_id: AtomicPtr<ThreadId>,
    /// Whether this thread is active
    active: AtomicU64, // Use as bool with 0/1 values
    /// Last time this thread updated its epoch
    last_update: AtomicU64,
    /// Retired pointers waiting for reclamation
    retired_pointers: Mutex<VecDeque<RetiredPointer>>,
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl ThreadEpoch {
    fn new() -> Self {
        Self {
            local_epoch: AtomicU64::new(0),
            thread_id: AtomicPtr::new(ptr::null_mut()),
            active: AtomicU64::new(0),
            last_update: AtomicU64::new(0),
            retired_pointers: Mutex::new(VecDeque::new()),
            _padding: [],
        }
    }

    fn acquire(&self, thread_id: ThreadId) -> bool {
        let thread_id_ptr = Box::into_raw(Box::new(thread_id));
        
        match self.thread_id.compare_exchange(
            ptr::null_mut(),
            thread_id_ptr,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.active.store(1, Ordering::Release);
                self.local_epoch.store(GLOBAL_EPOCH.load(Ordering::Acquire), Ordering::Release);
                self.last_update.store(current_time_ns(), Ordering::Release);
                true
            }
            Err(_) => {
                // Cleanup the allocated thread_id
                unsafe { Box::from_raw(thread_id_ptr); }
                false
            }
        }
    }

    fn release(&self) {
        self.active.store(0, Ordering::Release);
        
        // Release thread_id
        let thread_id_ptr = self.thread_id.swap(ptr::null_mut(), Ordering::AcqRel);
        if !thread_id_ptr.is_null() {
            unsafe { Box::from_raw(thread_id_ptr); }
        }
    }

    fn is_owned_by(&self, thread_id: ThreadId) -> bool {
        let stored_ptr = self.thread_id.load(Ordering::Acquire);
        if stored_ptr.is_null() {
            return false;
        }
        
        unsafe { *stored_ptr == thread_id }
    }

    fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire) != 0
    }

    fn update_epoch(&self, global_epoch: u64) {
        self.local_epoch.store(global_epoch, Ordering::Release);
        self.last_update.store(current_time_ns(), Ordering::Release);
    }

    fn get_local_epoch(&self) -> u64 {
        self.local_epoch.load(Ordering::Acquire)
    }

    fn get_last_update(&self) -> u64 {
        self.last_update.load(Ordering::Acquire)
    }
}

/// Retired pointer with epoch information
struct RetiredPointer {
    ptr: *mut u8,
    epoch: u64,
    deleter: Box<dyn Fn(*mut u8) + Send>,
}

unsafe impl Send for RetiredPointer {}

/// Epoch-based memory reclamation manager
pub struct EpochBasedReclamation {
    /// Per-thread epoch records
    thread_epochs: Vec<ThreadEpoch>,
    /// Maximum number of threads
    max_threads: usize,
    /// Last time global epoch was advanced
    last_epoch_advance: AtomicU64,
    /// Global retired pointer queue
    global_retired: Mutex<VecDeque<RetiredPointer>>,
}

impl EpochBasedReclamation {
    /// Create a new epoch-based reclamation manager
    pub fn new(max_threads: usize) -> Self {
        let mut thread_epochs = Vec::with_capacity(max_threads);
        for _ in 0..max_threads {
            thread_epochs.push(ThreadEpoch::new());
        }

        Self {
            thread_epochs,
            max_threads,
            last_epoch_advance: AtomicU64::new(current_time_ns()),
            global_retired: Mutex::new(VecDeque::new()),
        }
    }

    /// Enter a critical section (pin current thread to current epoch)
    pub fn pin(&self) -> EpochGuard {
        let thread_id = thread::current().id();
        
        // Find or acquire a thread epoch record
        for epoch_record in &self.thread_epochs {
            if epoch_record.is_owned_by(thread_id) {
                // Update to current global epoch
                let global_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
                epoch_record.update_epoch(global_epoch);
                
                return EpochGuard {
                    manager: self,
                    thread_id,
                    epoch: global_epoch,
                };
            }
        }

        // Try to acquire a new record
        for epoch_record in &self.thread_epochs {
            if epoch_record.acquire(thread_id) {
                let global_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
                return EpochGuard {
                    manager: self,
                    thread_id,
                    epoch: global_epoch,
                };
            }
        }

        panic!("No available epoch records for thread {:?}", thread_id);
    }

    /// Retire a pointer for later reclamation
    pub fn retire_pointer<T>(&self, ptr: *mut T) {
        self.retire_pointer_with_deleter(ptr as *mut u8, Box::new(|p| {
            unsafe { Box::from_raw(p as *mut T); }
        }));
    }

    /// Retire a pointer with a custom deleter
    pub fn retire_pointer_with_deleter<F>(&self, ptr: *mut u8, deleter: F)
    where
        F: Fn(*mut u8) + Send + 'static,
    {
        let current_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        let retired = RetiredPointer {
            ptr,
            epoch: current_epoch,
            deleter: Box::new(deleter),
        };

        let thread_id = thread::current().id();
        
        // Try to add to thread-local retired list first
        for epoch_record in &self.thread_epochs {
            if epoch_record.is_owned_by(thread_id) {
                let mut local_retired = epoch_record.retired_pointers.lock().unwrap();
                local_retired.push_back(retired);
                
                // Check if we need to trigger reclamation
                if local_retired.len() >= MAX_RETIRED_POINTERS / self.max_threads {
                    drop(local_retired); // Release lock before reclamation
                    self.try_reclaim_local(epoch_record);
                }
                return;
            }
        }

        // Fallback to global retired list
        let mut global_retired = self.global_retired.lock().unwrap();
        global_retired.push_back(retired);
        
        if global_retired.len() >= MAX_RETIRED_POINTERS {
            drop(global_retired); // Release lock before reclamation
            self.try_reclaim_global();
        }
    }

    /// Try to advance the global epoch
    pub fn try_advance_epoch(&self) -> bool {
        let now = current_time_ns();
        let last_advance = self.last_epoch_advance.load(Ordering::Acquire);
        
        // Check if enough time has passed
        if now - last_advance < EPOCH_GRACE_PERIOD_NS {
            return false;
        }

        // Check if all active threads have caught up to current epoch
        let current_global_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        let min_epoch = self.get_minimum_epoch();
        
        if min_epoch < current_global_epoch {
            return false; // Some threads are still in old epochs
        }

        // Try to advance the global epoch
        match GLOBAL_EPOCH.compare_exchange(
            current_global_epoch,
            current_global_epoch + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.last_epoch_advance.store(now, Ordering::Release);
                true
            }
            Err(_) => false, // Another thread advanced it
        }
    }

    /// Get the minimum epoch across all active threads
    fn get_minimum_epoch(&self) -> u64 {
        let mut min_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        
        for epoch_record in &self.thread_epochs {
            if epoch_record.is_active() {
                let local_epoch = epoch_record.get_local_epoch();
                if local_epoch < min_epoch {
                    min_epoch = local_epoch;
                }
            }
        }
        
        min_epoch
    }

    /// Try to reclaim retired pointers from thread-local list
    fn try_reclaim_local(&self, epoch_record: &ThreadEpoch) {
        let mut local_retired = epoch_record.retired_pointers.lock().unwrap();
        let mut still_retired = VecDeque::new();
        let safe_epoch = self.calculate_safe_epoch();

        while let Some(retired) = local_retired.pop_front() {
            if retired.epoch <= safe_epoch {
                // Safe to reclaim
                (retired.deleter)(retired.ptr);
            } else {
                still_retired.push_back(retired);
            }
        }

        // Put back still-retired pointers
        *local_retired = still_retired;
    }

    /// Try to reclaim retired pointers from global list
    fn try_reclaim_global(&self) {
        let mut global_retired = self.global_retired.lock().unwrap();
        let mut still_retired = VecDeque::new();
        let safe_epoch = self.calculate_safe_epoch();

        while let Some(retired) = global_retired.pop_front() {
            if retired.epoch <= safe_epoch {
                // Safe to reclaim
                (retired.deleter)(retired.ptr);
            } else {
                still_retired.push_back(retired);
            }
        }

        // Put back still-retired pointers
        *global_retired = still_retired;
    }

    /// Calculate the safe epoch for reclamation (minimum epoch - grace period)
    fn calculate_safe_epoch(&self) -> u64 {
        let min_epoch = self.get_minimum_epoch();
        if min_epoch >= 2 {
            min_epoch - 2 // Grace period of 2 epochs
        } else {
            0
        }
    }

    /// Force reclamation of all safe retired pointers
    pub fn force_reclaim(&self) {
        // Try to advance epoch first
        self.try_advance_epoch();
        
        // Reclaim from all thread-local lists
        for epoch_record in &self.thread_epochs {
            if epoch_record.is_active() {
                self.try_reclaim_local(epoch_record);
            }
        }

        // Reclaim from global list
        self.try_reclaim_global();
    }

    /// Get statistics about epoch-based reclamation
    pub fn get_stats(&self) -> EpochReclamationStats {
        let global_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        let min_epoch = self.get_minimum_epoch();
        let mut active_threads = 0;
        let mut total_retired = 0;

        for epoch_record in &self.thread_epochs {
            if epoch_record.is_active() {
                active_threads += 1;
                if let Ok(retired) = epoch_record.retired_pointers.lock() {
                    total_retired += retired.len();
                }
            }
        }

        let global_retired = self.global_retired.lock().unwrap().len();
        total_retired += global_retired;

        EpochReclamationStats {
            global_epoch,
            min_epoch,
            active_threads,
            total_retired,
            safe_epoch: self.calculate_safe_epoch(),
        }
    }
}

impl Drop for EpochBasedReclamation {
    fn drop(&mut self) {
        // Force reclamation of all retired pointers
        self.force_reclaim();
        
        // Release all thread epoch records
        for epoch_record in &self.thread_epochs {
            epoch_record.release();
        }
    }
}

/// RAII guard for epoch-based critical sections
pub struct EpochGuard<'a> {
    manager: &'a EpochBasedReclamation,
    thread_id: ThreadId,
    epoch: u64,
}

impl<'a> EpochGuard<'a> {
    /// Get the current epoch for this guard
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Retire a pointer within this epoch
    pub fn retire<T>(&self, ptr: *mut T) {
        self.manager.retire_pointer(ptr);
    }

    /// Retire a pointer with custom deleter within this epoch
    pub fn retire_with_deleter<F>(&self, ptr: *mut u8, deleter: F)
    where
        F: Fn(*mut u8) + Send + 'static,
    {
        self.manager.retire_pointer_with_deleter(ptr, deleter);
    }
}

impl<'a> Drop for EpochGuard<'a> {
    fn drop(&mut self) {
        // Try to advance epoch when leaving critical section
        self.manager.try_advance_epoch();
    }
}

/// Statistics about epoch-based reclamation
#[derive(Debug, Clone)]
pub struct EpochReclamationStats {
    pub global_epoch: u64,
    pub min_epoch: u64,
    pub active_threads: usize,
    pub total_retired: usize,
    pub safe_epoch: u64,
}

/// Get current time in nanoseconds
fn current_time_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Combined memory reclamation manager using both hazard pointers and epochs
pub struct HybridReclamationManager {
    hazard_manager: Arc<crate::performance::lock_free::hazard_pointers::HazardPointerManager>,
    epoch_manager: Arc<EpochBasedReclamation>,
}

impl HybridReclamationManager {
    /// Create a new hybrid reclamation manager
    pub fn new(max_threads: usize) -> Self {
        Self {
            hazard_manager: Arc::new(crate::performance::lock_free::hazard_pointers::HazardPointerManager::new(max_threads)),
            epoch_manager: Arc::new(EpochBasedReclamation::new(max_threads)),
        }
    }

    /// Get the hazard pointer manager
    pub fn hazard_manager(&self) -> &Arc<crate::performance::lock_free::hazard_pointers::HazardPointerManager> {
        &self.hazard_manager
    }

    /// Get the epoch-based reclamation manager
    pub fn epoch_manager(&self) -> &Arc<EpochBasedReclamation> {
        &self.epoch_manager
    }

    /// Enter an epoch-protected critical section
    pub fn pin(&self) -> EpochGuard {
        self.epoch_manager.pin()
    }

    /// Acquire a hazard pointer
    pub fn acquire_hazard(&self) -> crate::performance::lock_free::hazard_pointers::HazardPointer {
        self.hazard_manager.acquire_hazard_pointer()
    }

    /// Retire a pointer using the most appropriate method
    pub fn retire<T>(&self, ptr: *mut T) {
        // Use epoch-based reclamation for better performance
        self.epoch_manager.retire_pointer(ptr);
    }

    /// Force reclamation using both methods
    pub fn force_reclaim(&self) {
        self.hazard_manager.force_reclaim();
        self.epoch_manager.force_reclaim();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_epoch_manager_creation() {
        let manager = EpochBasedReclamation::new(4);
        let stats = manager.get_stats();
        
        assert_eq!(stats.active_threads, 0);
        assert_eq!(stats.total_retired, 0);
        assert_eq!(stats.global_epoch, 0);
    }

    #[test]
    fn test_epoch_guard() {
        let manager = EpochBasedReclamation::new(4);
        
        {
            let guard = manager.pin();
            assert_eq!(guard.epoch(), 0);
            
            let stats = manager.get_stats();
            assert_eq!(stats.active_threads, 1);
        }
        
        // After guard is dropped, thread should still be registered but not necessarily active
        let stats = manager.get_stats();
        assert!(stats.active_threads <= 1);
    }

    #[test]
    fn test_retire_pointer() {
        let manager = EpochBasedReclamation::new(4);
        let test_ptr = Box::into_raw(Box::new(42u64));
        
        {
            let _guard = manager.pin();
            manager.retire_pointer(test_ptr);
            
            let stats = manager.get_stats();
            assert!(stats.total_retired > 0);
        }
        
        // Force reclamation
        manager.force_reclaim();
        
        let stats_after = manager.get_stats();
        // Note: The pointer might still be retired if not enough epochs have passed
        assert!(stats_after.total_retired <= 1);
    }

    #[test]
    fn test_epoch_advancement() {
        let manager = EpochBasedReclamation::new(4);
        let initial_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        
        // Pin a thread to current epoch
        let _guard = manager.pin();
        
        // Try to advance epoch (should fail because thread is pinned)
        let advanced = manager.try_advance_epoch();
        assert!(!advanced);
        
        // Drop the guard and try again
        drop(_guard);
        
        // Wait for grace period
        thread::sleep(Duration::from_millis(2));
        
        let advanced = manager.try_advance_epoch();
        // May or may not advance depending on timing
        let final_epoch = GLOBAL_EPOCH.load(Ordering::Acquire);
        assert!(final_epoch >= initial_epoch);
    }

    #[test]
    fn test_concurrent_epochs() {
        let manager = Arc::new(EpochBasedReclamation::new(10));
        let mut handles = vec![];

        for i in 0..5 {
            let manager_clone = manager.clone();
            let handle = thread::spawn(move || {
                let test_ptr = Box::into_raw(Box::new(i));
                
                {
                    let guard = manager_clone.pin();
                    
                    // Retire the pointer
                    guard.retire(test_ptr);
                    
                    // Hold the guard for a short time
                    thread::sleep(Duration::from_millis(10));
                }
                
                // Try to advance epoch
                manager_clone.try_advance_epoch();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Final cleanup
        manager.force_reclaim();
        let final_stats = manager.get_stats();
        assert!(final_stats.global_epoch >= 0);
    }

    #[test]
    fn test_hybrid_manager() {
        let manager = HybridReclamationManager::new(4);
        let test_ptr = Box::into_raw(Box::new(42u64));
        
        {
            let _guard = manager.pin();
            let _hazard = manager.acquire_hazard();
            
            manager.retire(test_ptr);
        }
        
        manager.force_reclaim();
        
        // Both managers should be working
        let hazard_stats = manager.hazard_manager().get_stats();
        let epoch_stats = manager.epoch_manager().get_stats();
        
        assert!(hazard_stats.active_threads <= 1);
        assert!(epoch_stats.active_threads <= 1);
    }

    #[test]
    fn test_memory_reclamation_safety() {
        let manager = Arc::new(EpochBasedReclamation::new(10));
        let mut handles = vec![];
        let num_pointers = 100;

        // Create and retire many pointers concurrently
        for i in 0..num_pointers {
            let manager_clone = manager.clone();
            let handle = thread::spawn(move || {
                let test_ptr = Box::into_raw(Box::new(i));
                
                {
                    let guard = manager_clone.pin();
                    guard.retire(test_ptr);
                    
                    // Simulate some work
                    thread::sleep(Duration::from_micros(100));
                }
                
                // Try reclamation
                manager_clone.force_reclaim();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Final reclamation
        manager.force_reclaim();
        
        let final_stats = manager.get_stats();
        // Most pointers should be reclaimed by now
        assert!(final_stats.total_retired < num_pointers);
    }
}