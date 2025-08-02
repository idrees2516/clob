use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use super::hazard_pointers::{HazardPointerManager, HazardPointer};
use super::memory_reclamation::EpochBasedReclamation;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering};
use std::sync::Arc;
use std::ptr;
use std::cmp::Ordering as CmpOrdering;
use std::marker::PhantomData;

/// Maximum number of levels in the skip list
const MAX_LEVEL: u8 = 32;

/// Probability factor for level generation (1/4 chance of going to next level)
const LEVEL_PROBABILITY: f64 = 0.25;

/// Lock-free skip list node with cache-line alignment
#[repr(align(64))]
pub struct SkipListNode<K, V> {
    /// Key for this node (immutable after creation)
    pub key: K,
    
    /// Value for this node
    pub value: V,
    
    /// Array of forward pointers for each level
    pub forward: Vec<AlignedAtomicPtr<SkipListNode<K, V>>>,
    
    /// Level of this node
    pub level: u8,
    
    /// Epoch for memory reclamation
    pub epoch: AtomicU64,
    
    /// Marked for deletion flag
    pub marked: AtomicU64, // Use as bool with 0/1 values
    
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl<K, V> SkipListNode<K, V> {
    /// Create a new skip list node
    pub fn new(key: K, value: V, level: u8) -> Self {
        let mut forward = Vec::with_capacity(level as usize + 1);
        for _ in 0..=level {
            forward.push(AlignedAtomicPtr::new(ptr::null_mut()));
        }

        Self {
            key,
            value,
            forward,
            level,
            epoch: AtomicU64::new(0),
            marked: AtomicU64::new(0),
            _padding: [],
        }
    }

    /// Check if this node is marked for deletion
    #[inline(always)]
    pub fn is_marked(&self) -> bool {
        self.marked.load(Ordering::Acquire) != 0
    }

    /// Mark this node for deletion
    #[inline(always)]
    pub fn mark_for_deletion(&self) -> bool {
        self.marked.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok()
    }

    /// Get forward pointer at specified level
    #[inline(always)]
    pub fn get_forward(&self, level: usize) -> *mut SkipListNode<K, V> {
        if level <= self.level as usize {
            self.forward[level].load(MemoryOrder::Acquire)
        } else {
            ptr::null_mut()
        }
    }

    /// Set forward pointer at specified level
    #[inline(always)]
    pub fn set_forward(&self, level: usize, node: *mut SkipListNode<K, V>) -> Result<*mut SkipListNode<K, V>, *mut SkipListNode<K, V>> {
        if level <= self.level as usize {
            self.forward[level].compare_exchange_weak(
                self.get_forward(level),
                node,
                MemoryOrder::Release,
                MemoryOrder::Relaxed,
            )
        } else {
            Err(ptr::null_mut())
        }
    }

    /// Store forward pointer at specified level
    #[inline(always)]
    pub fn store_forward(&self, level: usize, node: *mut SkipListNode<K, V>) {
        if level <= self.level as usize {
            self.forward[level].store(node, MemoryOrder::Release);
        }
    }
}

/// Lock-free skip list implementation with hazard pointer protection
pub struct LockFreeSkipList<K, V> {
    /// Head sentinel node
    head: AlignedAtomicPtr<SkipListNode<K, V>>,
    
    /// Current maximum level in the skip list
    max_level: AtomicU8,
    
    /// Number of elements in the skip list
    size: AtomicU64,
    
    /// Hazard pointer manager for memory safety
    hazard_manager: Arc<HazardPointerManager>,
    
    /// Epoch-based reclamation for better performance
    epoch_manager: Arc<EpochBasedReclamation>,
    
    /// Random number generator state (thread-local)
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> LockFreeSkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Create a new lock-free skip list
    pub fn new(max_threads: usize) -> Self {
        // Create head sentinel with maximum level
        let head_node = Box::into_raw(Box::new(SkipListNode::new(
            unsafe { std::mem::zeroed() }, // Sentinel key (never used for comparison)
            unsafe { std::mem::zeroed() }, // Sentinel value (never used)
            MAX_LEVEL - 1,
        )));

        Self {
            head: AlignedAtomicPtr::new(head_node),
            max_level: AtomicU8::new(0),
            size: AtomicU64::new(0),
            hazard_manager: Arc::new(HazardPointerManager::new(max_threads)),
            epoch_manager: Arc::new(EpochBasedReclamation::new(max_threads)),
            _phantom: PhantomData,
        }
    }

    /// Generate a random level for a new node
    fn random_level() -> u8 {
        let mut level = 0u8;
        let mut rng = fastrand::Rng::new();
        
        while level < MAX_LEVEL - 1 && rng.f64() < LEVEL_PROBABILITY {
            level += 1;
        }
        
        level
    }

    /// Find the position where a key should be inserted or exists
    fn find_position(
        &self,
        key: &K,
        hazard_manager: &HazardPointerManager,
    ) -> (Vec<*mut SkipListNode<K, V>>, *mut SkipListNode<K, V>>) {
        let mut predecessors = vec![ptr::null_mut(); MAX_LEVEL as usize];
        let mut current_level = self.max_level.load(Ordering::Acquire) as usize;
        
        let hazard_pred = hazard_manager.acquire_hazard_pointer();
        let hazard_curr = hazard_manager.acquire_hazard_pointer();
        let hazard_next = hazard_manager.acquire_hazard_pointer();

        loop {
            let mut pred = self.head.load(MemoryOrder::Acquire);
            hazard_pred.protect(pred);
            
            for level in (0..=current_level).rev() {
                let mut curr = unsafe { (*pred).get_forward(level) };
                hazard_curr.protect(curr);
                
                while !curr.is_null() {
                    // Verify current node hasn't been reclaimed
                    if unsafe { (*pred).get_forward(level) } != curr {
                        break; // Restart from current level
                    }
                    
                    let next = unsafe { (*curr).get_forward(level) };
                    hazard_next.protect(next);
                    
                    unsafe {
                        // Skip marked nodes
                        if (*curr).is_marked() {
                            // Try to help remove the marked node
                            if (*pred).set_forward(level, next).is_ok() {
                                self.epoch_manager.retire_pointer(curr);
                            }
                            curr = next;
                            hazard_curr.protect(curr);
                            continue;
                        }
                        
                        // Compare keys
                        match (*curr).key.cmp(key) {
                            CmpOrdering::Less => {
                                pred = curr;
                                hazard_pred.protect(pred);
                                curr = next;
                                hazard_curr.protect(curr);
                            }
                            CmpOrdering::Equal => {
                                // Found exact match
                                return (predecessors, curr);
                            }
                            CmpOrdering::Greater => {
                                break; // Found insertion point
                            }
                        }
                    }
                }
                
                predecessors[level] = pred;
            }
            
            // Return the position where key should be inserted
            let curr = if pred.is_null() {
                ptr::null_mut()
            } else {
                unsafe { (*pred).get_forward(0) }
            };
            
            return (predecessors, curr);
        }
    }

    /// Insert a key-value pair into the skip list
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, SkipListError> {
        let _guard = self.epoch_manager.pin();
        let level = Self::random_level();
        let new_node = Box::into_raw(Box::new(SkipListNode::new(key.clone(), value, level)));

        loop {
            let (predecessors, found) = self.find_position(&key, &self.hazard_manager);
            
            if !found.is_null() {
                unsafe {
                    if !(*found).is_marked() && (*found).key == key {
                        // Key already exists, update value
                        let old_value = (*found).value.clone();
                        (*found).value = (*new_node).value.clone();
                        
                        // Clean up the new node we created
                        Box::from_raw(new_node);
                        return Ok(Some(old_value));
                    }
                }
            }

            // Link the new node at all levels
            let mut linked_levels = 0;
            for i in 0..=level as usize {
                if i < predecessors.len() && !predecessors[i].is_null() {
                    unsafe {
                        let next = (*predecessors[i]).get_forward(i);
                        (*new_node).store_forward(i, next);
                        
                        if (*predecessors[i]).set_forward(i, new_node).is_ok() {
                            linked_levels += 1;
                        } else {
                            // Failed to link at this level, retry
                            break;
                        }
                    }
                } else {
                    break;
                }
            }

            if linked_levels == (level as usize + 1) {
                // Successfully linked at all levels
                self.size.fetch_add(1, Ordering::AcqRel);
                
                // Update max level if necessary
                let current_max = self.max_level.load(Ordering::Acquire);
                if level > current_max {
                    let _ = self.max_level.compare_exchange(
                        current_max,
                        level,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    );
                }
                
                return Ok(None);
            } else {
                // Failed to link completely, clean up and retry
                for i in 0..linked_levels {
                    unsafe {
                        if !predecessors[i].is_null() {
                            let _ = (*predecessors[i]).set_forward(i, (*new_node).get_forward(i));
                        }
                    }
                }
                // Continue with retry loop
            }
        }
    }

    /// Remove a key from the skip list
    pub fn remove(&self, key: &K) -> Result<Option<V>, SkipListError> {
        let _guard = self.epoch_manager.pin();

        loop {
            let (predecessors, found) = self.find_position(key, &self.hazard_manager);
            
            if found.is_null() {
                return Ok(None); // Key not found
            }

            unsafe {
                if (*found).is_marked() || (*found).key != *key {
                    continue; // Node was already removed or key doesn't match
                }

                // Mark the node for deletion
                if !(*found).mark_for_deletion() {
                    continue; // Another thread marked it first
                }

                let removed_value = (*found).value.clone();
                let node_level = (*found).level;

                // Unlink the node at all levels
                let mut unlinked_levels = 0;
                for i in 0..=node_level as usize {
                    if i < predecessors.len() && !predecessors[i].is_null() {
                        let next = (*found).get_forward(i);
                        if (*predecessors[i]).set_forward(i, next).is_ok() {
                            unlinked_levels += 1;
                        }
                    }
                }

                if unlinked_levels > 0 {
                    // Successfully unlinked, schedule for reclamation
                    self.epoch_manager.retire_pointer(found);
                    self.size.fetch_sub(1, Ordering::AcqRel);
                    return Ok(Some(removed_value));
                }
            }
        }
    }

    /// Search for a key in the skip list
    pub fn get(&self, key: &K) -> Option<V> {
        let _guard = self.epoch_manager.pin();
        let (_, found) = self.find_position(key, &self.hazard_manager);
        
        if found.is_null() {
            return None;
        }

        unsafe {
            if !(*found).is_marked() && (*found).key == *key {
                Some((*found).value.clone())
            } else {
                None
            }
        }
    }

    /// Check if the skip list contains a key
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get the number of elements in the skip list
    pub fn len(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    /// Check if the skip list is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current maximum level
    pub fn max_level(&self) -> u8 {
        self.max_level.load(Ordering::Acquire)
    }

    /// Force memory reclamation
    pub fn force_reclaim(&self) {
        self.epoch_manager.force_reclaim();
        self.hazard_manager.force_reclaim();
    }

    /// Get statistics about the skip list
    pub fn get_stats(&self) -> SkipListStats {
        SkipListStats {
            size: self.len(),
            max_level: self.max_level(),
            hazard_stats: self.hazard_manager.get_stats(),
            epoch_stats: self.epoch_manager.get_stats(),
        }
    }
}

impl<K, V> Drop for LockFreeSkipList<K, V> {
    fn drop(&mut self) {
        // Force reclamation of all nodes
        self.force_reclaim();
        
        // Clean up head node
        let head = self.head.load(MemoryOrder::Acquire);
        if !head.is_null() {
            unsafe { Box::from_raw(head); }
        }
    }
}

/// Error types for skip list operations
#[derive(Debug, Clone)]
pub enum SkipListError {
    MemoryAllocationFailed,
    ConcurrentModification,
    InvalidOperation,
}

/// Statistics about skip list performance
#[derive(Debug, Clone)]
pub struct SkipListStats {
    pub size: u64,
    pub max_level: u8,
    pub hazard_stats: super::hazard_pointers::HazardPointerStats,
    pub epoch_stats: super::memory_reclamation::EpochReclamationStats,
}

/// Iterator for skip list (level 0 traversal)
pub struct SkipListIterator<'a, K, V> {
    current: *mut SkipListNode<K, V>,
    hazard: HazardPointer<'a>,
    _phantom: PhantomData<(&'a K, &'a V)>,
}

impl<'a, K, V> SkipListIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    fn new(skip_list: &'a LockFreeSkipList<K, V>) -> Self {
        let hazard = skip_list.hazard_manager.acquire_hazard_pointer();
        let head = skip_list.head.load(MemoryOrder::Acquire);
        let current = if head.is_null() {
            ptr::null_mut()
        } else {
            unsafe { (*head).get_forward(0) }
        };

        Self {
            current,
            hazard,
            _phantom: PhantomData,
        }
    }
}

impl<'a, K, V> Iterator for SkipListIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.current.is_null() {
            self.hazard.protect(self.current);
            
            unsafe {
                if !(*self.current).is_marked() {
                    let key = (*self.current).key.clone();
                    let value = (*self.current).value.clone();
                    self.current = (*self.current).get_forward(0);
                    return Some((key, value));
                } else {
                    // Skip marked nodes
                    self.current = (*self.current).get_forward(0);
                }
            }
        }
        
        None
    }
}

impl<K, V> LockFreeSkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// Create an iterator over the skip list
    pub fn iter(&self) -> SkipListIterator<K, V> {
        SkipListIterator::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::collections::HashSet;

    #[test]
    fn test_skip_list_creation() {
        let skip_list: LockFreeSkipList<i32, String> = LockFreeSkipList::new(4);
        assert_eq!(skip_list.len(), 0);
        assert!(skip_list.is_empty());
        assert_eq!(skip_list.max_level(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let skip_list = LockFreeSkipList::new(4);
        
        // Insert some key-value pairs
        assert!(skip_list.insert(1, "one".to_string()).unwrap().is_none());
        assert!(skip_list.insert(2, "two".to_string()).unwrap().is_none());
        assert!(skip_list.insert(3, "three".to_string()).unwrap().is_none());
        
        assert_eq!(skip_list.len(), 3);
        assert!(!skip_list.is_empty());
        
        // Test retrieval
        assert_eq!(skip_list.get(&1), Some("one".to_string()));
        assert_eq!(skip_list.get(&2), Some("two".to_string()));
        assert_eq!(skip_list.get(&3), Some("three".to_string()));
        assert_eq!(skip_list.get(&4), None);
    }

    #[test]
    fn test_update_existing_key() {
        let skip_list = LockFreeSkipList::new(4);
        
        // Insert initial value
        assert!(skip_list.insert(1, "one".to_string()).unwrap().is_none());
        
        // Update existing key
        let old_value = skip_list.insert(1, "ONE".to_string()).unwrap();
        assert_eq!(old_value, Some("one".to_string()));
        
        // Verify update
        assert_eq!(skip_list.get(&1), Some("ONE".to_string()));
        assert_eq!(skip_list.len(), 1);
    }

    #[test]
    fn test_remove() {
        let skip_list = LockFreeSkipList::new(4);
        
        // Insert some values
        skip_list.insert(1, "one".to_string()).unwrap();
        skip_list.insert(2, "two".to_string()).unwrap();
        skip_list.insert(3, "three".to_string()).unwrap();
        
        // Remove middle value
        let removed = skip_list.remove(&2).unwrap();
        assert_eq!(removed, Some("two".to_string()));
        assert_eq!(skip_list.len(), 2);
        
        // Verify removal
        assert_eq!(skip_list.get(&2), None);
        assert_eq!(skip_list.get(&1), Some("one".to_string()));
        assert_eq!(skip_list.get(&3), Some("three".to_string()));
        
        // Remove non-existent key
        let removed = skip_list.remove(&4).unwrap();
        assert_eq!(removed, None);
    }

    #[test]
    fn test_contains_key() {
        let skip_list = LockFreeSkipList::new(4);
        
        skip_list.insert(1, "one".to_string()).unwrap();
        skip_list.insert(3, "three".to_string()).unwrap();
        
        assert!(skip_list.contains_key(&1));
        assert!(!skip_list.contains_key(&2));
        assert!(skip_list.contains_key(&3));
    }

    #[test]
    fn test_iterator() {
        let skip_list = LockFreeSkipList::new(4);
        
        // Insert values in random order
        skip_list.insert(3, "three".to_string()).unwrap();
        skip_list.insert(1, "one".to_string()).unwrap();
        skip_list.insert(4, "four".to_string()).unwrap();
        skip_list.insert(2, "two".to_string()).unwrap();
        
        // Collect items from iterator
        let items: Vec<(i32, String)> = skip_list.iter().collect();
        
        // Should be sorted by key
        assert_eq!(items.len(), 4);
        assert_eq!(items[0], (1, "one".to_string()));
        assert_eq!(items[1], (2, "two".to_string()));
        assert_eq!(items[2], (3, "three".to_string()));
        assert_eq!(items[3], (4, "four".to_string()));
    }

    #[test]
    fn test_concurrent_operations() {
        let skip_list = Arc::new(LockFreeSkipList::new(20));
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 100;

        // Spawn threads to perform concurrent operations
        for thread_id in 0..num_threads {
            let skip_list_clone = skip_list.clone();
            
            let handle = thread::spawn(move || {
                let mut inserted_keys = HashSet::new();
                
                // Insert keys
                for i in 0..ops_per_thread {
                    let key = thread_id * ops_per_thread + i;
                    let value = format!("value_{}", key);
                    skip_list_clone.insert(key, value).unwrap();
                    inserted_keys.insert(key);
                }
                
                // Verify insertions
                for key in &inserted_keys {
                    assert!(skip_list_clone.contains_key(key));
                }
                
                // Remove half of the keys
                let keys_to_remove: Vec<_> = inserted_keys.iter().take(ops_per_thread / 2).cloned().collect();
                for key in &keys_to_remove {
                    skip_list_clone.remove(key).unwrap();
                    inserted_keys.remove(key);
                }
                
                // Verify remaining keys
                for key in &inserted_keys {
                    assert!(skip_list_clone.contains_key(key));
                }
                
                inserted_keys.len()
            });
            
            handles.push(handle);
        }

        // Wait for all threads and collect results
        let mut total_remaining = 0;
        for handle in handles {
            total_remaining += handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(skip_list.len() as usize, total_remaining);
        
        // Force cleanup
        skip_list.force_reclaim();
    }

    #[test]
    fn test_large_dataset() {
        let skip_list = LockFreeSkipList::new(4);
        let num_items = 10000;

        // Insert many items
        for i in 0..num_items {
            skip_list.insert(i, format!("value_{}", i)).unwrap();
        }

        assert_eq!(skip_list.len(), num_items as u64);

        // Verify all items can be retrieved
        for i in 0..num_items {
            assert_eq!(skip_list.get(&i), Some(format!("value_{}", i)));
        }

        // Remove every other item
        for i in (0..num_items).step_by(2) {
            skip_list.remove(&i).unwrap();
        }

        assert_eq!(skip_list.len(), (num_items / 2) as u64);

        // Verify remaining items
        for i in 0..num_items {
            if i % 2 == 0 {
                assert_eq!(skip_list.get(&i), None);
            } else {
                assert_eq!(skip_list.get(&i), Some(format!("value_{}", i)));
            }
        }
    }

    #[test]
    fn test_stats() {
        let skip_list = LockFreeSkipList::new(4);
        
        // Insert some items
        for i in 0..100 {
            skip_list.insert(i, format!("value_{}", i)).unwrap();
        }

        let stats = skip_list.get_stats();
        assert_eq!(stats.size, 100);
        assert!(stats.max_level > 0);
        assert!(stats.hazard_stats.total_hazards > 0);
    }
}