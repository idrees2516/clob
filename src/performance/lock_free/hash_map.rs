use super::atomic_operations::{AtomicOperations, MemoryOrder, AlignedAtomicPtr};
use super::hazard_pointers::{HazardPointerManager, HazardPointer};
use super::memory_reclamation::EpochBasedReclamation;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::marker::PhantomData;

/// Default initial capacity for the hash map
const DEFAULT_INITIAL_CAPACITY: usize = 16;

/// Load factor threshold for resizing
const LOAD_FACTOR_THRESHOLD: f64 = 0.75;

/// Maximum load factor before forced resize
const MAX_LOAD_FACTOR: f64 = 1.0;

/// Lock-free hash map bucket entry
#[repr(align(64))]
pub struct HashMapEntry<K, V> {
    /// Key for this entry (immutable after creation)
    pub key: K,
    
    /// Value for this entry
    pub value: V,
    
    /// Hash value for this entry (cached for performance)
    pub hash: u64,
    
    /// Next entry in the bucket chain
    pub next: AlignedAtomicPtr<HashMapEntry<K, V>>,
    
    /// Epoch for memory reclamation
    pub epoch: AtomicU64,
    
    /// Marked for deletion flag
    pub marked: AtomicU64, // Use as bool with 0/1 values
    
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl<K, V> HashMapEntry<K, V> {
    /// Create a new hash map entry
    pub fn new(key: K, value: V, hash: u64) -> Self {
        Self {
            key,
            value,
            hash,
            next: AlignedAtomicPtr::new(ptr::null_mut()),
            epoch: AtomicU64::new(0),
            marked: AtomicU64::new(0),
            _padding: [],
        }
    }

    /// Check if this entry is marked for deletion
    #[inline(always)]
    pub fn is_marked(&self) -> bool {
        self.marked.load(Ordering::Acquire) != 0
    }

    /// Mark this entry for deletion
    #[inline(always)]
    pub fn mark_for_deletion(&self) -> bool {
        self.marked.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok()
    }

    /// Get the next entry in the chain
    #[inline(always)]
    pub fn get_next(&self) -> *mut HashMapEntry<K, V> {
        self.next.load(MemoryOrder::Acquire)
    }

    /// Set the next entry using compare-and-swap
    #[inline(always)]
    pub fn set_next(&self, expected: *mut HashMapEntry<K, V>, new: *mut HashMapEntry<K, V>) -> Result<*mut HashMapEntry<K, V>, *mut HashMapEntry<K, V>> {
        self.next.compare_exchange_weak(expected, new, MemoryOrder::Release, MemoryOrder::Relaxed)
    }

    /// Store the next entry
    #[inline(always)]
    pub fn store_next(&self, next: *mut HashMapEntry<K, V>) {
        self.next.store(next, MemoryOrder::Release);
    }
}

/// Lock-free hash map bucket
#[repr(align(64))]
pub struct HashMapBucket<K, V> {
    /// Head of the entry chain for this bucket
    pub head: AlignedAtomicPtr<HashMapEntry<K, V>>,
    
    /// Number of entries in this bucket
    pub count: AtomicUsize,
    
    /// Padding to prevent false sharing
    _padding: [u8; 0],
}

impl<K, V> HashMapBucket<K, V> {
    /// Create a new empty bucket
    pub fn new() -> Self {
        Self {
            head: AlignedAtomicPtr::new(ptr::null_mut()),
            count: AtomicUsize::new(0),
            _padding: [],
        }
    }

    /// Get the number of entries in this bucket
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Check if this bucket is empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Lock-free hash map implementation with separate chaining
pub struct LockFreeHashMap<K, V> {
    /// Array of buckets
    buckets: Vec<HashMapBucket<K, V>>,
    
    /// Current capacity (number of buckets)
    capacity: AtomicUsize,
    
    /// Number of entries in the hash map
    size: AtomicU64,
    
    /// Resize in progress flag
    resizing: AtomicU64, // Use as bool with 0/1 values
    
    /// Hazard pointer manager for memory safety
    hazard_manager: Arc<HazardPointerManager>,
    
    /// Epoch-based reclamation for better performance
    epoch_manager: Arc<EpochBasedReclamation>,
    
    /// Phantom data for type parameters
    _phantom: PhantomData<(K, V)>,
}

impl<K, V> LockFreeHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new lock-free hash map with default capacity
    pub fn new(max_threads: usize) -> Self {
        Self::with_capacity(DEFAULT_INITIAL_CAPACITY, max_threads)
    }

    /// Create a new lock-free hash map with specified capacity
    pub fn with_capacity(capacity: usize, max_threads: usize) -> Self {
        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(HashMapBucket::new());
        }

        Self {
            buckets,
            capacity: AtomicUsize::new(capacity),
            size: AtomicU64::new(0),
            resizing: AtomicU64::new(0),
            hazard_manager: Arc::new(HazardPointerManager::new(max_threads)),
            epoch_manager: Arc::new(EpochBasedReclamation::new(max_threads)),
            _phantom: PhantomData,
        }
    }

    /// Calculate hash for a key
    fn hash_key(key: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Get bucket index for a hash value
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) % self.capacity.load(Ordering::Acquire)
    }

    /// Find an entry in a bucket chain
    fn find_entry(
        &self,
        bucket: &HashMapBucket<K, V>,
        key: &K,
        hash: u64,
        hazard_manager: &HazardPointerManager,
    ) -> (*mut HashMapEntry<K, V>, *mut HashMapEntry<K, V>) {
        let hazard_prev = hazard_manager.acquire_hazard_pointer();
        let hazard_curr = hazard_manager.acquire_hazard_pointer();

        let mut prev: *mut HashMapEntry<K, V> = ptr::null_mut();
        let mut current = bucket.head.load(MemoryOrder::Acquire);

        while !current.is_null() {
            hazard_curr.protect(current);
            
            // Verify current hasn't been reclaimed
            let expected_current = if prev.is_null() {
                bucket.head.load(MemoryOrder::Acquire)
            } else {
                unsafe { (*prev).get_next() }
            };
            
            if current != expected_current {
                // Restart search
                prev = ptr::null_mut();
                current = bucket.head.load(MemoryOrder::Acquire);
                continue;
            }

            unsafe {
                if (*current).is_marked() {
                    // Help remove marked entry
                    let next = (*current).get_next();
                    if prev.is_null() {
                        if bucket.head.compare_exchange_weak(
                            current,
                            next,
                            MemoryOrder::Release,
                            MemoryOrder::Relaxed,
                        ).is_ok() {
                            bucket.count.fetch_sub(1, Ordering::AcqRel);
                            self.epoch_manager.retire_pointer(current);
                        }
                    } else {
                        if (*prev).set_next(current, next).is_ok() {
                            bucket.count.fetch_sub(1, Ordering::AcqRel);
                            self.epoch_manager.retire_pointer(current);
                        }
                    }
                    current = next;
                    continue;
                }

                if (*current).hash == hash && (*current).key == *key {
                    // Found matching entry
                    return (prev, current);
                }

                prev = current;
                hazard_prev.protect(prev);
                current = (*current).get_next();
            }
        }

        (prev, ptr::null_mut())
    }

    /// Insert or update a key-value pair
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, HashMapError> {
        let _guard = self.epoch_manager.pin();
        let hash = Self::hash_key(&key);
        let bucket_idx = self.bucket_index(hash);
        let bucket = &self.buckets[bucket_idx];

        loop {
            let (prev, found) = self.find_entry(bucket, &key, hash, &self.hazard_manager);

            if !found.is_null() {
                // Key exists, update value
                unsafe {
                    if !(*found).is_marked() && (*found).key == key {
                        let old_value = (*found).value.clone();
                        (*found).value = value;
                        return Ok(Some(old_value));
                    }
                }
                // Entry was marked or key changed, retry
                continue;
            }

            // Create new entry
            let new_entry = Box::into_raw(Box::new(HashMapEntry::new(key.clone(), value, hash)));

            // Link new entry into chain
            if prev.is_null() {
                // Insert at head
                let current_head = bucket.head.load(MemoryOrder::Acquire);
                unsafe {
                    (*new_entry).store_next(current_head);
                }

                if bucket.head.compare_exchange_weak(
                    current_head,
                    new_entry,
                    MemoryOrder::Release,
                    MemoryOrder::Relaxed,
                ).is_ok() {
                    bucket.count.fetch_add(1, Ordering::AcqRel);
                    self.size.fetch_add(1, Ordering::AcqRel);
                    
                    // Check if resize is needed
                    self.try_resize();
                    return Ok(None);
                } else {
                    // Failed to insert, clean up and retry
                    unsafe { Box::from_raw(new_entry); }
                    continue;
                }
            } else {
                // Insert after prev
                unsafe {
                    let next = (*prev).get_next();
                    (*new_entry).store_next(next);

                    if (*prev).set_next(next, new_entry).is_ok() {
                        bucket.count.fetch_add(1, Ordering::AcqRel);
                        self.size.fetch_add(1, Ordering::AcqRel);
                        
                        // Check if resize is needed
                        self.try_resize();
                        return Ok(None);
                    } else {
                        // Failed to insert, clean up and retry
                        Box::from_raw(new_entry);
                        continue;
                    }
                }
            }
        }
    }

    /// Get a value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let _guard = self.epoch_manager.pin();
        let hash = Self::hash_key(key);
        let bucket_idx = self.bucket_index(hash);
        let bucket = &self.buckets[bucket_idx];

        let (_, found) = self.find_entry(bucket, key, hash, &self.hazard_manager);

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

    /// Remove a key-value pair
    pub fn remove(&self, key: &K) -> Result<Option<V>, HashMapError> {
        let _guard = self.epoch_manager.pin();
        let hash = Self::hash_key(key);
        let bucket_idx = self.bucket_index(hash);
        let bucket = &self.buckets[bucket_idx];

        loop {
            let (prev, found) = self.find_entry(bucket, key, hash, &self.hazard_manager);

            if found.is_null() {
                return Ok(None); // Key not found
            }

            unsafe {
                if (*found).is_marked() || (*found).key != *key {
                    continue; // Entry was already removed or key doesn't match
                }

                // Mark the entry for deletion
                if !(*found).mark_for_deletion() {
                    continue; // Another thread marked it first
                }

                let removed_value = (*found).value.clone();
                let next = (*found).get_next();

                // Unlink the entry
                let unlinked = if prev.is_null() {
                    // Remove from head
                    bucket.head.compare_exchange_weak(
                        found,
                        next,
                        MemoryOrder::Release,
                        MemoryOrder::Relaxed,
                    ).is_ok()
                } else {
                    // Remove from middle/end
                    (*prev).set_next(found, next).is_ok()
                };

                if unlinked {
                    bucket.count.fetch_sub(1, Ordering::AcqRel);
                    self.size.fetch_sub(1, Ordering::AcqRel);
                    self.epoch_manager.retire_pointer(found);
                    return Ok(Some(removed_value));
                }
                // Failed to unlink, but entry is marked, so it will be cleaned up later
                return Ok(Some(removed_value));
            }
        }
    }

    /// Check if the hash map contains a key
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Get the number of entries in the hash map
    pub fn len(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    /// Check if the hash map is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current capacity (number of buckets)
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    /// Get the current load factor
    pub fn load_factor(&self) -> f64 {
        let size = self.len() as f64;
        let capacity = self.capacity() as f64;
        if capacity > 0.0 {
            size / capacity
        } else {
            0.0
        }
    }

    /// Try to resize the hash map if load factor is too high
    fn try_resize(&self) {
        let load_factor = self.load_factor();
        
        if load_factor > LOAD_FACTOR_THRESHOLD {
            // Try to acquire resize lock
            if self.resizing.compare_exchange(0, 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                // We got the resize lock, perform resize
                self.resize();
                self.resizing.store(0, Ordering::Release);
            }
        }
    }

    /// Resize the hash map (double the capacity)
    fn resize(&self) {
        // This is a simplified resize implementation
        // In a production system, you would want a more sophisticated approach
        // that doesn't block all operations during resize
        
        let current_capacity = self.capacity();
        let new_capacity = current_capacity * 2;
        
        // For now, we'll just update the capacity
        // A full implementation would involve:
        // 1. Allocating new bucket array
        // 2. Rehashing all entries
        // 3. Atomically switching to new array
        // 4. Cleaning up old array
        
        self.capacity.store(new_capacity, Ordering::Release);
    }

    /// Force memory reclamation
    pub fn force_reclaim(&self) {
        self.epoch_manager.force_reclaim();
        self.hazard_manager.force_reclaim();
    }

    /// Get statistics about the hash map
    pub fn get_stats(&self) -> HashMapStats {
        let mut bucket_lengths = Vec::new();
        let mut max_bucket_length = 0;
        let mut non_empty_buckets = 0;

        for bucket in &self.buckets {
            let length = bucket.len();
            bucket_lengths.push(length);
            if length > 0 {
                non_empty_buckets += 1;
            }
            if length > max_bucket_length {
                max_bucket_length = length;
            }
        }

        let avg_bucket_length = if non_empty_buckets > 0 {
            self.len() as f64 / non_empty_buckets as f64
        } else {
            0.0
        };

        HashMapStats {
            size: self.len(),
            capacity: self.capacity(),
            load_factor: self.load_factor(),
            max_bucket_length,
            avg_bucket_length,
            non_empty_buckets,
            hazard_stats: self.hazard_manager.get_stats(),
            epoch_stats: self.epoch_manager.get_stats(),
        }
    }
}

impl<K, V> Drop for LockFreeHashMap<K, V> {
    fn drop(&mut self) {
        // Force reclamation of all entries
        self.force_reclaim();
        
        // Clean up remaining entries in buckets
        for bucket in &self.buckets {
            let mut current = bucket.head.load(MemoryOrder::Acquire);
            while !current.is_null() {
                unsafe {
                    let next = (*current).get_next();
                    Box::from_raw(current);
                    current = next;
                }
            }
        }
    }
}

/// Error types for hash map operations
#[derive(Debug, Clone)]
pub enum HashMapError {
    MemoryAllocationFailed,
    ConcurrentModification,
    ResizeInProgress,
    InvalidOperation,
}

/// Statistics about hash map performance
#[derive(Debug, Clone)]
pub struct HashMapStats {
    pub size: u64,
    pub capacity: usize,
    pub load_factor: f64,
    pub max_bucket_length: usize,
    pub avg_bucket_length: f64,
    pub non_empty_buckets: usize,
    pub hazard_stats: super::hazard_pointers::HazardPointerStats,
    pub epoch_stats: super::memory_reclamation::EpochReclamationStats,
}

/// Iterator for hash map entries
pub struct HashMapIterator<'a, K, V> {
    buckets: &'a [HashMapBucket<K, V>],
    current_bucket: usize,
    current_entry: *mut HashMapEntry<K, V>,
    hazard: HazardPointer<'a>,
    _phantom: PhantomData<(&'a K, &'a V)>,
}

impl<'a, K, V> HashMapIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    fn new(hash_map: &'a LockFreeHashMap<K, V>) -> Self {
        let hazard = hash_map.hazard_manager.acquire_hazard_pointer();
        let mut iter = Self {
            buckets: &hash_map.buckets,
            current_bucket: 0,
            current_entry: ptr::null_mut(),
            hazard,
            _phantom: PhantomData,
        };
        
        iter.advance_to_next_entry();
        iter
    }

    fn advance_to_next_entry(&mut self) {
        // If we have a current entry, move to next in chain
        if !self.current_entry.is_null() {
            unsafe {
                self.current_entry = (*self.current_entry).get_next();
            }
        }

        // If no current entry, find next non-empty bucket
        while self.current_entry.is_null() && self.current_bucket < self.buckets.len() {
            self.current_entry = self.buckets[self.current_bucket].head.load(MemoryOrder::Acquire);
            if self.current_entry.is_null() {
                self.current_bucket += 1;
            }
        }
    }
}

impl<'a, K, V> Iterator for HashMapIterator<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.current_entry.is_null() {
            self.hazard.protect(self.current_entry);
            
            unsafe {
                if !(*self.current_entry).is_marked() {
                    let key = (*self.current_entry).key.clone();
                    let value = (*self.current_entry).value.clone();
                    
                    // Advance to next entry
                    self.advance_to_next_entry();
                    
                    return Some((key, value));
                } else {
                    // Skip marked entries
                    self.advance_to_next_entry();
                }
            }
        }
        
        None
    }
}

impl<K, V> LockFreeHashMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create an iterator over the hash map
    pub fn iter(&self) -> HashMapIterator<K, V> {
        HashMapIterator::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::collections::HashSet;

    #[test]
    fn test_hash_map_creation() {
        let hash_map: LockFreeHashMap<i32, String> = LockFreeHashMap::new(4);
        assert_eq!(hash_map.len(), 0);
        assert!(hash_map.is_empty());
        assert_eq!(hash_map.capacity(), DEFAULT_INITIAL_CAPACITY);
    }

    #[test]
    fn test_insert_and_get() {
        let hash_map = LockFreeHashMap::new(4);
        
        // Insert some key-value pairs
        assert!(hash_map.insert(1, "one".to_string()).unwrap().is_none());
        assert!(hash_map.insert(2, "two".to_string()).unwrap().is_none());
        assert!(hash_map.insert(3, "three".to_string()).unwrap().is_none());
        
        assert_eq!(hash_map.len(), 3);
        assert!(!hash_map.is_empty());
        
        // Test retrieval
        assert_eq!(hash_map.get(&1), Some("one".to_string()));
        assert_eq!(hash_map.get(&2), Some("two".to_string()));
        assert_eq!(hash_map.get(&3), Some("three".to_string()));
        assert_eq!(hash_map.get(&4), None);
    }

    #[test]
    fn test_update_existing_key() {
        let hash_map = LockFreeHashMap::new(4);
        
        // Insert initial value
        assert!(hash_map.insert(1, "one".to_string()).unwrap().is_none());
        
        // Update existing key
        let old_value = hash_map.insert(1, "ONE".to_string()).unwrap();
        assert_eq!(old_value, Some("one".to_string()));
        
        // Verify update
        assert_eq!(hash_map.get(&1), Some("ONE".to_string()));
        assert_eq!(hash_map.len(), 1);
    }

    #[test]
    fn test_remove() {
        let hash_map = LockFreeHashMap::new(4);
        
        // Insert some values
        hash_map.insert(1, "one".to_string()).unwrap();
        hash_map.insert(2, "two".to_string()).unwrap();
        hash_map.insert(3, "three".to_string()).unwrap();
        
        // Remove middle value
        let removed = hash_map.remove(&2).unwrap();
        assert_eq!(removed, Some("two".to_string()));
        assert_eq!(hash_map.len(), 2);
        
        // Verify removal
        assert_eq!(hash_map.get(&2), None);
        assert_eq!(hash_map.get(&1), Some("one".to_string()));
        assert_eq!(hash_map.get(&3), Some("three".to_string()));
        
        // Remove non-existent key
        let removed = hash_map.remove(&4).unwrap();
        assert_eq!(removed, None);
    }

    #[test]
    fn test_contains_key() {
        let hash_map = LockFreeHashMap::new(4);
        
        hash_map.insert(1, "one".to_string()).unwrap();
        hash_map.insert(3, "three".to_string()).unwrap();
        
        assert!(hash_map.contains_key(&1));
        assert!(!hash_map.contains_key(&2));
        assert!(hash_map.contains_key(&3));
    }

    #[test]
    fn test_iterator() {
        let hash_map = LockFreeHashMap::new(4);
        
        // Insert values
        hash_map.insert(1, "one".to_string()).unwrap();
        hash_map.insert(2, "two".to_string()).unwrap();
        hash_map.insert(3, "three".to_string()).unwrap();
        
        // Collect items from iterator
        let mut items: Vec<(i32, String)> = hash_map.iter().collect();
        items.sort_by_key(|&(k, _)| k); // Sort for consistent testing
        
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (1, "one".to_string()));
        assert_eq!(items[1], (2, "two".to_string()));
        assert_eq!(items[2], (3, "three".to_string()));
    }

    #[test]
    fn test_concurrent_operations() {
        let hash_map = Arc::new(LockFreeHashMap::new(20));
        let mut handles = vec![];
        let num_threads = 10;
        let ops_per_thread = 100;

        // Spawn threads to perform concurrent operations
        for thread_id in 0..num_threads {
            let hash_map_clone = hash_map.clone();
            
            let handle = thread::spawn(move || {
                let mut inserted_keys = HashSet::new();
                
                // Insert keys
                for i in 0..ops_per_thread {
                    let key = thread_id * ops_per_thread + i;
                    let value = format!("value_{}", key);
                    hash_map_clone.insert(key, value).unwrap();
                    inserted_keys.insert(key);
                }
                
                // Verify insertions
                for key in &inserted_keys {
                    assert!(hash_map_clone.contains_key(key));
                }
                
                // Remove half of the keys
                let keys_to_remove: Vec<_> = inserted_keys.iter().take(ops_per_thread / 2).cloned().collect();
                for key in &keys_to_remove {
                    hash_map_clone.remove(key).unwrap();
                    inserted_keys.remove(key);
                }
                
                // Verify remaining keys
                for key in &inserted_keys {
                    assert!(hash_map_clone.contains_key(key));
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
        assert_eq!(hash_map.len() as usize, total_remaining);
        
        // Force cleanup
        hash_map.force_reclaim();
    }

    #[test]
    fn test_load_factor() {
        let hash_map = LockFreeHashMap::with_capacity(4, 4);
        
        assert_eq!(hash_map.load_factor(), 0.0);
        
        hash_map.insert(1, "one".to_string()).unwrap();
        assert_eq!(hash_map.load_factor(), 0.25);
        
        hash_map.insert(2, "two".to_string()).unwrap();
        assert_eq!(hash_map.load_factor(), 0.5);
        
        hash_map.insert(3, "three".to_string()).unwrap();
        assert_eq!(hash_map.load_factor(), 0.75);
    }

    #[test]
    fn test_stats() {
        let hash_map = LockFreeHashMap::new(4);
        
        // Insert some items
        for i in 0..50 {
            hash_map.insert(i, format!("value_{}", i)).unwrap();
        }

        let stats = hash_map.get_stats();
        assert_eq!(stats.size, 50);
        assert!(stats.load_factor > 0.0);
        assert!(stats.non_empty_buckets > 0);
        assert!(stats.max_bucket_length > 0);
    }

    #[test]
    fn test_hash_collisions() {
        // Create a small hash map to force collisions
        let hash_map = LockFreeHashMap::with_capacity(2, 4);
        
        // Insert many items to force collisions
        for i in 0..20 {
            hash_map.insert(i, format!("value_{}", i)).unwrap();
        }
        
        // Verify all items can be retrieved
        for i in 0..20 {
            assert_eq!(hash_map.get(&i), Some(format!("value_{}", i)));
        }
        
        let stats = hash_map.get_stats();
        assert!(stats.max_bucket_length > 1); // Should have collisions
    }
}