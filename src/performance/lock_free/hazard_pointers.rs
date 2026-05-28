use std::sync::atomic::{AtomicPtr, AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::ptr;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::thread::{self, ThreadId};
use std::collections::HashMap;

/// Thread-local hazard pointer for protecting memory from reclamation
pub struct HazardPointer<'a> {
    slot: &'a AtomicPtr<u8>,
    manager: &'a HazardPointerManager,
    thread_id: ThreadId,
}

impl<'a> HazardPointer<'a> {
    /// Protect a pointer from being reclaimed
    #[inline(always)]
    pub fn protect<T>(&self, ptr: *mut T) {
        self.slot.store(ptr as *mut u8, Ordering::Release);
    }

    /// Clear the protection
    #[inline(always)]
    pub fn clear(&self) {
        self.slot.store(ptr::null_mut(), Ordering::Release);
    }

    /// Get the currently protected pointer
    #[inline(always)]
    pub fn get_protected<T>(&self) -> *mut T {
        self.slot.load(Ordering::Acquire) as *mut T
    }
}

impl<'a> Drop for HazardPointer<'a> {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Per-thread hazard pointer record
#[repr(align(64))] // Cache-line aligned to prevent false sharing
struct HazardRecord {
    /// Hazard pointer slots for this thread
    hazards: Vec<AtomicPtr<u8>>,
    /// Thread ID that owns this record
    thread_id: AtomicPtr<ThreadId>,
    /// Whether this record is active
    active: AtomicBool,
    /// Retired pointers waiting for reclamation
    retired: Mutex<VecDeque<RetiredPointer>>,
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl HazardRecord {
    fn new(num_hazards: usize) -> Self {
        let mut hazards = Vec::with_capacity(num_hazards);
        for _ in 0..num_hazards {
            hazards.push(AtomicPtr::new(ptr::null_mut()));
        }

        Self {
            hazards,
            thread_id: AtomicPtr::new(ptr::null_mut()),
            active: AtomicBool::new(false),
            retired: Mutex::new(VecDeque::new()),
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
                self.active.store(true, Ordering::Release);
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
        // Clear all hazard pointers
        for hazard in &self.hazards {
            hazard.store(ptr::null_mut(), Ordering::Release);
        }

        self.active.store(false, Ordering::Release);
        
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
}

/// Retired pointer waiting for safe reclamation
struct RetiredPointer {
    ptr: *mut u8,
    deleter: Box<dyn Fn(*mut u8) + Send>,
}

unsafe impl Send for RetiredPointer {}

/// Hazard pointer manager for lock-free memory reclamation
pub struct HazardPointerManager {
    /// Per-thread hazard records
    records: Vec<HazardRecord>,
    /// Number of hazard pointers per thread
    hazards_per_thread: usize,
    /// Maximum number of retired pointers before forced reclamation
    retire_threshold: usize,
    /// Global retired pointer list
    global_retired: Mutex<VecDeque<RetiredPointer>>,
}

impl HazardPointerManager {
    /// Create a new hazard pointer manager
    pub fn new(max_threads: usize) -> Self {
        let hazards_per_thread = 8; // Default number of hazard pointers per thread
        let retire_threshold = 2 * max_threads * hazards_per_thread;
        
        let mut records = Vec::with_capacity(max_threads);
        for _ in 0..max_threads {
            records.push(HazardRecord::new(hazards_per_thread));
        }

        Self {
            records,
            hazards_per_thread,
            retire_threshold,
            global_retired: Mutex::new(VecDeque::new()),
        }
    }

    /// Acquire a hazard pointer for the current thread
    pub fn acquire_hazard_pointer(&self) -> HazardPointer {
        let thread_id = thread::current().id();
        
        // Try to find an existing record for this thread
        for record in &self.records {
            if record.is_owned_by(thread_id) {
                // Find an available hazard slot
                for (i, hazard) in record.hazards.iter().enumerate() {
                    if hazard.load(Ordering::Acquire).is_null() {
                        return HazardPointer {
                            slot: hazard,
                            manager: self,
                            thread_id,
                        };
                    }
                }
                // All slots are in use, panic or return error
                panic!("All hazard pointer slots are in use for thread {:?}", thread_id);
            }
        }

        // Try to acquire a new record
        for record in &self.records {
            if record.acquire(thread_id) {
                return HazardPointer {
                    slot: &record.hazards[0],
                    manager: self,
                    thread_id,
                };
            }
        }

        panic!("No available hazard pointer records");
    }

    /// Release a hazard pointer
    pub fn release_hazard_pointer(&self, hazard: HazardPointer) {
        hazard.clear();
        // The hazard pointer will be automatically cleared when dropped
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
        let retired = RetiredPointer {
            ptr,
            deleter: Box::new(deleter),
        };

        let thread_id = thread::current().id();
        
        // Try to add to thread-local retired list first
        for record in &self.records {
            if record.is_owned_by(thread_id) {
                let mut local_retired = record.retired.lock().unwrap();
                local_retired.push_back(retired);
                
                // Check if we need to trigger reclamation
                if local_retired.len() >= self.retire_threshold / self.records.len() {
                    drop(local_retired); // Release lock before reclamation
                    self.try_reclaim_local(record);
                }
                return;
            }
        }

        // Fallback to global retired list
        let mut global_retired = self.global_retired.lock().unwrap();
        global_retired.push_back(retired);
        
        if global_retired.len() >= self.retire_threshold {
            drop(global_retired); // Release lock before reclamation
            self.try_reclaim_global();
        }
    }

    /// Try to reclaim retired pointers from thread-local list
    fn try_reclaim_local(&self, record: &HazardRecord) {
        let mut local_retired = record.retired.lock().unwrap();
        let mut still_protected = VecDeque::new();

        while let Some(retired) = local_retired.pop_front() {
            if self.is_pointer_protected(retired.ptr) {
                still_protected.push_back(retired);
            } else {
                // Safe to reclaim
                (retired.deleter)(retired.ptr);
            }
        }

        // Put back still-protected pointers
        *local_retired = still_protected;
    }

    /// Try to reclaim retired pointers from global list
    fn try_reclaim_global(&self) {
        let mut global_retired = self.global_retired.lock().unwrap();
        let mut still_protected = VecDeque::new();

        while let Some(retired) = global_retired.pop_front() {
            if self.is_pointer_protected(retired.ptr) {
                still_protected.push_back(retired);
            } else {
                // Safe to reclaim
                (retired.deleter)(retired.ptr);
            }
        }

        // Put back still-protected pointers
        *global_retired = still_protected;
    }

    /// Check if a pointer is currently protected by any hazard pointer
    fn is_pointer_protected(&self, ptr: *mut u8) -> bool {
        for record in &self.records {
            if record.active.load(Ordering::Acquire) {
                for hazard in &record.hazards {
                    if hazard.load(Ordering::Acquire) == ptr {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Force reclamation of all unprotected retired pointers
    pub fn force_reclaim(&self) {
        // Reclaim from all thread-local lists
        for record in &self.records {
            self.try_reclaim_local(record);
        }

        // Reclaim from global list
        self.try_reclaim_global();
    }

    /// Get statistics about hazard pointer usage
    pub fn get_stats(&self) -> HazardPointerStats {
        let mut active_threads = 0;
        let mut total_hazards = 0;
        let mut used_hazards = 0;
        let mut total_retired = 0;

        for record in &self.records {
            if record.active.load(Ordering::Acquire) {
                active_threads += 1;
                total_hazards += record.hazards.len();
                
                for hazard in &record.hazards {
                    if !hazard.load(Ordering::Acquire).is_null() {
                        used_hazards += 1;
                    }
                }

                if let Ok(retired) = record.retired.lock() {
                    total_retired += retired.len();
                }
            }
        }

        let global_retired = self.global_retired.lock().unwrap().len();
        total_retired += global_retired;

        HazardPointerStats {
            active_threads,
            total_hazards,
            used_hazards,
            total_retired,
            retire_threshold: self.retire_threshold,
        }
    }
}

impl Drop for HazardPointerManager {
    fn drop(&mut self) {
        // Force reclamation of all retired pointers
        self.force_reclaim();
        
        // Release all records
        for record in &self.records {
            record.release();
        }
    }
}

/// Statistics about hazard pointer usage
#[derive(Debug, Clone)]
pub struct HazardPointerStats {
    pub active_threads: usize,
    pub total_hazards: usize,
    pub used_hazards: usize,
    pub total_retired: usize,
    pub retire_threshold: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_hazard_pointer_manager_creation() {
        let manager = HazardPointerManager::new(4);
        let stats = manager.get_stats();
        
        assert_eq!(stats.active_threads, 0);
        assert_eq!(stats.used_hazards, 0);
        assert_eq!(stats.total_retired, 0);
    }

    #[test]
    fn test_acquire_and_release_hazard_pointer() {
        let manager = HazardPointerManager::new(4);
        let test_ptr = Box::into_raw(Box::new(42u64));
        
        {
            let hazard = manager.acquire_hazard_pointer();
            hazard.protect(test_ptr);
            
            let protected = hazard.get_protected::<u64>();
            assert_eq!(protected, test_ptr);
            
            let stats = manager.get_stats();
            assert_eq!(stats.active_threads, 1);
            assert_eq!(stats.used_hazards, 1);
        }
        
        // Cleanup
        unsafe { Box::from_raw(test_ptr); }
    }

    #[test]
    fn test_retire_pointer() {
        let manager = HazardPointerManager::new(4);
        let test_ptr = Box::into_raw(Box::new(42u64));
        
        // Protect the pointer
        let hazard = manager.acquire_hazard_pointer();
        hazard.protect(test_ptr);
        
        // Retire the pointer (should not be reclaimed yet)
        manager.retire_pointer(test_ptr);
        
        let stats = manager.get_stats();
        assert!(stats.total_retired > 0);
        
        // Clear protection and force reclamation
        hazard.clear();
        manager.force_reclaim();
        
        let stats_after = manager.get_stats();
        assert!(stats_after.total_retired < stats.total_retired);
    }

    #[test]
    fn test_concurrent_hazard_pointers() {
        let manager = Arc::new(HazardPointerManager::new(10));
        let mut handles = vec![];

        for i in 0..5 {
            let manager_clone = manager.clone();
            let handle = thread::spawn(move || {
                let test_ptr = Box::into_raw(Box::new(i));
                
                {
                    let hazard = manager_clone.acquire_hazard_pointer();
                    hazard.protect(test_ptr);
                    
                    // Hold the hazard pointer for a short time
                    thread::sleep(Duration::from_millis(10));
                    
                    // Retire the pointer
                    manager_clone.retire_pointer(test_ptr);
                }
                
                // Force reclamation attempt
                manager_clone.force_reclaim();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Final cleanup
        manager.force_reclaim();
        let final_stats = manager.get_stats();
        assert_eq!(final_stats.used_hazards, 0);
    }

    #[test]
    fn test_protection_prevents_reclamation() {
        let manager = HazardPointerManager::new(4);
        let test_ptr = Box::into_raw(Box::new(42u64));
        
        // Protect the pointer
        let hazard = manager.acquire_hazard_pointer();
        hazard.protect(test_ptr);
        
        // Retire the pointer
        manager.retire_pointer(test_ptr);
        
        // Force reclamation - should not reclaim protected pointer
        manager.force_reclaim();
        
        let stats = manager.get_stats();
        assert!(stats.total_retired > 0); // Should still be retired, not reclaimed
        
        // Clear protection and reclaim again
        hazard.clear();
        manager.force_reclaim();
        
        let stats_after = manager.get_stats();
        assert_eq!(stats_after.total_retired, 0); // Should now be reclaimed
    }

    #[test]
    fn test_multiple_hazards_per_thread() {
        let manager = HazardPointerManager::new(4);
        let ptr1 = Box::into_raw(Box::new(1u64));
        let ptr2 = Box::into_raw(Box::new(2u64));
        
        let hazard1 = manager.acquire_hazard_pointer();
        let hazard2 = manager.acquire_hazard_pointer();
        
        hazard1.protect(ptr1);
        hazard2.protect(ptr2);
        
        let stats = manager.get_stats();
        assert_eq!(stats.active_threads, 1);
        assert_eq!(stats.used_hazards, 2);
        
        // Cleanup
        hazard1.clear();
        hazard2.clear();
        unsafe { 
            Box::from_raw(ptr1);
            Box::from_raw(ptr2);
        }
    }
}