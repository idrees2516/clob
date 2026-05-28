use super::alignment::{CACHE_LINE_SIZE, CacheFriendlyOrder, CacheFriendlyTrade};
use super::prefetch::{DataPrefetcher, PrefetchHint};
use std::cmp::Ordering;
use std::ptr;

/// Cache-aware sorting algorithms optimized for trading data
pub struct CacheAwareSorting;

impl CacheAwareSorting {
    /// Cache-optimized quicksort with prefetching
    pub fn quicksort_orders(orders: &mut [CacheFriendlyOrder]) {
        if orders.len() <= 1 {
            return;
        }
        
        Self::quicksort_orders_impl(orders, 0, orders.len() - 1);
    }

    fn quicksort_orders_impl(orders: &mut [CacheFriendlyOrder], low: usize, high: usize) {
        if low < high {
            // Prefetch data we'll need
            unsafe {
                if high - low > 8 {
                    DataPrefetcher::prefetch_sequential(
                        orders.as_ptr().add(low),
                        high - low + 1,
                        PrefetchHint::T0,
                    );
                }
            }
            
            let pivot = Self::partition_orders(orders, low, high);
            
            if pivot > 0 {
                Self::quicksort_orders_impl(orders, low, pivot - 1);
            }
            if pivot < high {
                Self::quicksort_orders_impl(orders, pivot + 1, high);
            }
        }
    }

    fn partition_orders(orders: &mut [CacheFriendlyOrder], low: usize, high: usize) -> usize {
        let pivot_price = orders[high].price;
        let mut i = low;
        
        for j in low..high {
            if orders[j].price <= pivot_price {
                orders.swap(i, j);
                i += 1;
            }
        }
        
        orders.swap(i, high);
        i
    }

    /// Cache-friendly merge sort for large datasets
    pub fn merge_sort_orders(orders: &mut [CacheFriendlyOrder]) {
        if orders.len() <= 1 {
            return;
        }
        
        let mut temp = vec![unsafe { std::mem::zeroed() }; orders.len()];
        Self::merge_sort_orders_impl(orders, &mut temp, 0, orders.len() - 1);
    }

    fn merge_sort_orders_impl(
        orders: &mut [CacheFriendlyOrder],
        temp: &mut [CacheFriendlyOrder],
        left: usize,
        right: usize,
    ) {
        if left < right {
            let mid = left + (right - left) / 2;
            
            Self::merge_sort_orders_impl(orders, temp, left, mid);
            Self::merge_sort_orders_impl(orders, temp, mid + 1, right);
            Self::merge_orders(orders, temp, left, mid, right);
        }
    }

    fn merge_orders(
        orders: &mut [CacheFriendlyOrder],
        temp: &mut [CacheFriendlyOrder],
        left: usize,
        mid: usize,
        right: usize,
    ) {
        // Copy data to temp array
        for i in left..=right {
            temp[i] = orders[i];
        }
        
        let mut i = left;
        let mut j = mid + 1;
        let mut k = left;
        
        // Prefetch both sides
        unsafe {
            if j <= right {
                DataPrefetcher::prefetch_sequential(
                    temp.as_ptr().add(i),
                    mid - i + 1,
                    PrefetchHint::T0,
                );
                DataPrefetcher::prefetch_sequential(
                    temp.as_ptr().add(j),
                    right - j + 1,
                    PrefetchHint::T0,
                );
            }
        }
        
        while i <= mid && j <= right {
            if temp[i].price <= temp[j].price {
                orders[k] = temp[i];
                i += 1;
            } else {
                orders[k] = temp[j];
                j += 1;
            }
            k += 1;
        }
        
        while i <= mid {
            orders[k] = temp[i];
            i += 1;
            k += 1;
        }
        
        while j <= right {
            orders[k] = temp[j];
            j += 1;
            k += 1;
        }
    }
}

/// Cache-aware binary search optimized for price levels
pub struct CacheAwareBinarySearch;

impl CacheAwareBinarySearch {
    /// Binary search with prefetching for order book price levels
    pub fn search_price_level(
        orders: &[CacheFriendlyOrder],
        target_price: u64,
    ) -> Result<usize, usize> {
        if orders.is_empty() {
            return Err(0);
        }
        
        let mut left = 0;
        let mut right = orders.len();
        
        while left < right {
            let mid = left + (right - left) / 2;
            
            // Prefetch likely next access locations
            unsafe {
                if right - left > 4 {
                    let next_left = left + (mid - left) / 2;
                    let next_right = mid + (right - mid) / 2;
                    
                    DataPrefetcher::prefetch_line(
                        orders.as_ptr().add(next_left),
                        PrefetchHint::T0,
                    );
                    DataPrefetcher::prefetch_line(
                        orders.as_ptr().add(next_right),
                        PrefetchHint::T0,
                    );
                }
            }
            
            match orders[mid].price.cmp(&target_price) {
                Ordering::Less => left = mid + 1,
                Ordering::Greater => right = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        
        Err(left)
    }

    /// Interpolation search for uniformly distributed prices
    pub fn interpolation_search_price(
        orders: &[CacheFriendlyOrder],
        target_price: u64,
    ) -> Result<usize, usize> {
        if orders.is_empty() {
            return Err(0);
        }
        
        let mut low = 0;
        let mut high = orders.len() - 1;
        
        while low <= high && target_price >= orders[low].price && target_price <= orders[high].price {
            if low == high {
                return if orders[low].price == target_price {
                    Ok(low)
                } else {
                    Err(low)
                };
            }
            
            // Interpolation formula
            let pos = low + (((target_price - orders[low].price) as usize * (high - low)) /
                            (orders[high].price - orders[low].price) as usize);
            
            // Prefetch around the interpolated position
            unsafe {
                let prefetch_start = pos.saturating_sub(2);
                let prefetch_end = (pos + 3).min(orders.len());
                DataPrefetcher::prefetch_sequential(
                    orders.as_ptr().add(prefetch_start),
                    prefetch_end - prefetch_start,
                    PrefetchHint::T0,
                );
            }
            
            match orders[pos].price.cmp(&target_price) {
                Ordering::Equal => return Ok(pos),
                Ordering::Less => low = pos + 1,
                Ordering::Greater => {
                    if pos == 0 {
                        break;
                    }
                    high = pos - 1;
                }
            }
        }
        
        Err(low)
    }
}

/// Cache-optimized data structure traversal
pub struct CacheAwareTraversal;

impl CacheAwareTraversal {
    /// Traverse linked list with prefetching
    pub unsafe fn traverse_with_prefetch<T, F, P>(
        start: *const T,
        get_next: F,
        process: P,
        prefetch_distance: usize,
    ) where
        F: Fn(*const T) -> *const T,
        P: Fn(*const T),
    {
        let mut current = start;
        let mut prefetch_queue = Vec::with_capacity(prefetch_distance);
        
        // Initialize prefetch queue
        let mut temp = current;
        for _ in 0..prefetch_distance {
            if temp.is_null() {
                break;
            }
            prefetch_queue.push(temp);
            temp = get_next(temp);
        }
        
        while !current.is_null() {
            // Process current node
            process(current);
            
            // Prefetch future nodes
            if let Some(&prefetch_node) = prefetch_queue.first() {
                DataPrefetcher::prefetch_line(prefetch_node, PrefetchHint::T0);
                prefetch_queue.remove(0);
                
                // Add new node to prefetch queue
                if !temp.is_null() {
                    prefetch_queue.push(temp);
                    temp = get_next(temp);
                }
            }
            
            current = get_next(current);
        }
    }

    /// Cache-friendly array processing with blocking
    pub fn process_array_blocked<T, F>(
        array: &[T],
        process_fn: F,
        block_size: usize,
    ) where
        F: Fn(&T),
    {
        let effective_block_size = block_size.min(CACHE_LINE_SIZE / std::mem::size_of::<T>());
        
        for chunk in array.chunks(effective_block_size) {
            // Prefetch next chunk
            unsafe {
                if let Some(next_chunk_start) = array.get(chunk.len()) {
                    DataPrefetcher::prefetch_sequential(
                        next_chunk_start as *const T,
                        effective_block_size.min(array.len() - chunk.len()),
                        PrefetchHint::T0,
                    );
                }
            }
            
            // Process current chunk
            for item in chunk {
                process_fn(item);
            }
        }
    }
}

/// Cache-aware hash table implementation
pub struct CacheAwareHashMap<K, V> {
    buckets: Vec<Vec<(K, V)>>,
    bucket_count: usize,
    load_factor: f64,
}

impl<K, V> CacheAwareHashMap<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(initial_capacity: usize) -> Self {
        let bucket_count = initial_capacity.next_power_of_two();
        let mut buckets = Vec::with_capacity(bucket_count);
        
        for _ in 0..bucket_count {
            buckets.push(Vec::new());
        }
        
        Self {
            buckets,
            bucket_count,
            load_factor: 0.0,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_key(&key);
        let bucket_index = hash % self.bucket_count;
        
        // Prefetch the bucket we're about to access
        unsafe {
            DataPrefetcher::prefetch_line(
                self.buckets[bucket_index].as_ptr(),
                PrefetchHint::T0,
            );
        }
        
        let bucket = &mut self.buckets[bucket_index];
        
        // Look for existing key
        for (existing_key, existing_value) in bucket.iter_mut() {
            if *existing_key == key {
                let old_value = existing_value.clone();
                *existing_value = value;
                return Some(old_value);
            }
        }
        
        // Insert new key-value pair
        bucket.push((key, value));
        self.load_factor = self.calculate_load_factor();
        
        None
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let hash = self.hash_key(key);
        let bucket_index = hash % self.bucket_count;
        
        // Prefetch the bucket
        unsafe {
            DataPrefetcher::prefetch_line(
                self.buckets[bucket_index].as_ptr(),
                PrefetchHint::T0,
            );
        }
        
        let bucket = &self.buckets[bucket_index];
        
        for (existing_key, value) in bucket {
            if existing_key == key {
                return Some(value);
            }
        }
        
        None
    }

    fn hash_key(&self, key: &K) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize
    }

    fn calculate_load_factor(&self) -> f64 {
        let total_entries: usize = self.buckets.iter().map(|b| b.len()).sum();
        total_entries as f64 / self.bucket_count as f64
    }
}

/// Cache-optimized priority queue for order matching
pub struct CacheAwarePriorityQueue<T> {
    heap: Vec<T>,
    compare: fn(&T, &T) -> Ordering,
}

impl<T> CacheAwarePriorityQueue<T>
where
    T: Clone,
{
    pub fn new(compare: fn(&T, &T) -> Ordering) -> Self {
        Self {
            heap: Vec::new(),
            compare,
        }
    }

    pub fn push(&mut self, item: T) {
        self.heap.push(item);
        self.bubble_up(self.heap.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.heap.is_empty() {
            return None;
        }
        
        let result = self.heap[0].clone();
        let last = self.heap.pop().unwrap();
        
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.bubble_down(0);
        }
        
        Some(result)
    }

    pub fn peek(&self) -> Option<&T> {
        self.heap.first()
    }

    fn bubble_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent_index = (index - 1) / 2;
            
            // Prefetch parent for comparison
            unsafe {
                DataPrefetcher::prefetch_line(
                    self.heap.as_ptr().add(parent_index),
                    PrefetchHint::T0,
                );
            }
            
            if (self.compare)(&self.heap[index], &self.heap[parent_index]) != Ordering::Greater {
                break;
            }
            
            self.heap.swap(index, parent_index);
            index = parent_index;
        }
    }

    fn bubble_down(&mut self, mut index: usize) {
        loop {
            let left_child = 2 * index + 1;
            let right_child = 2 * index + 2;
            let mut largest = index;
            
            // Prefetch children
            unsafe {
                if left_child < self.heap.len() {
                    DataPrefetcher::prefetch_line(
                        self.heap.as_ptr().add(left_child),
                        PrefetchHint::T0,
                    );
                }
                if right_child < self.heap.len() {
                    DataPrefetcher::prefetch_line(
                        self.heap.as_ptr().add(right_child),
                        PrefetchHint::T0,
                    );
                }
            }
            
            if left_child < self.heap.len() &&
               (self.compare)(&self.heap[left_child], &self.heap[largest]) == Ordering::Greater {
                largest = left_child;
            }
            
            if right_child < self.heap.len() &&
               (self.compare)(&self.heap[right_child], &self.heap[largest]) == Ordering::Greater {
                largest = right_child;
            }
            
            if largest == index {
                break;
            }
            
            self.heap.swap(index, largest);
            index = largest;
        }
    }
}

/// Cache-oblivious algorithms that work efficiently across different cache hierarchies
pub struct CacheObliviousAlgorithms;

impl CacheObliviousAlgorithms {
    /// Cache-oblivious matrix transpose
    pub fn transpose_matrix<T: Copy>(
        input: &[T],
        output: &mut [T],
        rows: usize,
        cols: usize,
    ) {
        if rows <= 16 || cols <= 16 {
            // Base case: direct transpose
            for i in 0..rows {
                for j in 0..cols {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
        } else {
            // Recursive case: divide and conquer
            let mid_row = rows / 2;
            let mid_col = cols / 2;
            
            // Transpose four quadrants recursively
            Self::transpose_matrix(
                &input[0..mid_row * cols],
                &mut output[0..mid_col * rows],
                mid_row,
                mid_col,
            );
            
            Self::transpose_matrix(
                &input[mid_row * cols..(mid_row + mid_row) * cols],
                &mut output[mid_col * rows..(mid_col + mid_col) * rows],
                mid_row,
                cols - mid_col,
            );
        }
    }

    /// Cache-oblivious sorting (Funnelsort variant)
    pub fn cache_oblivious_sort<T: Ord + Copy>(arr: &mut [T]) {
        if arr.len() <= 32 {
            // Use insertion sort for small arrays
            Self::insertion_sort(arr);
        } else {
            let mid = arr.len() / 2;
            Self::cache_oblivious_sort(&mut arr[..mid]);
            Self::cache_oblivious_sort(&mut arr[mid..]);
            Self::cache_oblivious_merge(arr, mid);
        }
    }

    fn insertion_sort<T: Ord + Copy>(arr: &mut [T]) {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
    }

    fn cache_oblivious_merge<T: Ord + Copy>(arr: &mut [T], mid: usize) {
        let mut temp = vec![unsafe { std::mem::zeroed() }; arr.len()];
        temp[..arr.len()].copy_from_slice(arr);
        
        let mut i = 0;
        let mut j = mid;
        let mut k = 0;
        
        while i < mid && j < arr.len() {
            if temp[i] <= temp[j] {
                arr[k] = temp[i];
                i += 1;
            } else {
                arr[k] = temp[j];
                j += 1;
            }
            k += 1;
        }
        
        while i < mid {
            arr[k] = temp[i];
            i += 1;
            k += 1;
        }
        
        while j < arr.len() {
            arr[k] = temp[j];
            j += 1;
            k += 1;
        }
    }
}

/// Data structure layout optimizer for cache efficiency
pub struct DataLayoutOptimizer;

impl DataLayoutOptimizer {
    /// Optimize struct field ordering for cache efficiency
    pub fn optimize_field_layout(field_sizes: &[usize]) -> Vec<usize> {
        let mut indexed_sizes: Vec<(usize, usize)> = field_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();
        
        // Sort by size descending to minimize padding
        indexed_sizes.sort_by(|a, b| b.1.cmp(&a.1));
        
        indexed_sizes.into_iter().map(|(i, _)| i).collect()
    }

    /// Calculate optimal array-of-structures vs structure-of-arrays layout
    pub fn analyze_aos_vs_soa<T>(
        access_patterns: &[AccessPattern],
        element_count: usize,
    ) -> LayoutRecommendation {
        let element_size = std::mem::size_of::<T>();
        let cache_line_elements = CACHE_LINE_SIZE / element_size;
        
        // Analyze access patterns
        let sequential_accesses = access_patterns
            .iter()
            .filter(|p| matches!(p.pattern_type, PatternType::Sequential))
            .count();
        
        let field_specific_accesses = access_patterns
            .iter()
            .filter(|p| matches!(p.pattern_type, PatternType::FieldSpecific))
            .count();
        
        if field_specific_accesses > sequential_accesses * 2 {
            LayoutRecommendation::StructureOfArrays
        } else if element_count < cache_line_elements * 4 {
            LayoutRecommendation::ArrayOfStructures
        } else {
            LayoutRecommendation::Hybrid
        }
    }

    /// Generate cache-friendly data structure transformations
    pub fn generate_cache_friendly_layout<T>(
        data: &[T],
        hot_fields: &[usize],
        cold_fields: &[usize],
    ) -> CacheFriendlyLayout<T> {
        CacheFriendlyLayout {
            hot_data: data.iter().cloned().collect(),
            cold_data: Vec::new(), // Would separate cold fields in practice
            hot_field_indices: hot_fields.to_vec(),
            cold_field_indices: cold_fields.to_vec(),
        }
    }
}

/// Memory access pattern analysis
#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub pattern_type: PatternType,
    pub frequency: f64,
    pub cache_efficiency: f64,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Sequential,
    Random,
    Strided(usize),
    FieldSpecific,
}

#[derive(Debug)]
pub enum LayoutRecommendation {
    ArrayOfStructures,
    StructureOfArrays,
    Hybrid,
}

#[derive(Debug)]
pub struct CacheFriendlyLayout<T> {
    pub hot_data: Vec<T>,
    pub cold_data: Vec<T>,
    pub hot_field_indices: Vec<usize>,
    pub cold_field_indices: Vec<usize>,
}

/// Cache-aware memory pool with intelligent prefetching
pub struct CacheAwareMemoryPool<T> {
    /// Hot pool for frequently accessed objects
    hot_pool: Vec<T>,
    /// Cold pool for infrequently accessed objects
    cold_pool: Vec<T>,
    /// Free list for hot pool
    hot_free_list: Vec<usize>,
    /// Free list for cold pool
    cold_free_list: Vec<usize>,
    /// Access frequency tracking
    access_counts: Vec<u32>,
    /// Pool statistics
    stats: PoolStatistics,
}

impl<T: Default + Clone> CacheAwareMemoryPool<T> {
    pub fn new(hot_capacity: usize, cold_capacity: usize) -> Self {
        let mut hot_pool = Vec::with_capacity(hot_capacity);
        let mut cold_pool = Vec::with_capacity(cold_capacity);
        let mut hot_free_list = Vec::with_capacity(hot_capacity);
        let mut cold_free_list = Vec::with_capacity(cold_capacity);
        
        // Initialize pools
        for i in 0..hot_capacity {
            hot_pool.push(T::default());
            hot_free_list.push(i);
        }
        
        for i in 0..cold_capacity {
            cold_pool.push(T::default());
            cold_free_list.push(i);
        }
        
        Self {
            hot_pool,
            cold_pool,
            hot_free_list,
            cold_free_list,
            access_counts: vec![0; hot_capacity + cold_capacity],
            stats: PoolStatistics::new(),
        }
    }

    /// Allocate from the appropriate pool based on predicted access pattern
    pub fn allocate(&mut self, predicted_hot: bool) -> Option<PoolHandle> {
        if predicted_hot && !self.hot_free_list.is_empty() {
            let index = self.hot_free_list.pop().unwrap();
            self.stats.hot_allocations += 1;
            
            // Prefetch nearby objects that might be accessed soon
            unsafe {
                if index + 1 < self.hot_pool.len() {
                    DataPrefetcher::prefetch_line(
                        self.hot_pool.as_ptr().add(index + 1),
                        PrefetchHint::T0,
                    );
                }
            }
            
            Some(PoolHandle::Hot(index))
        } else if !self.cold_free_list.is_empty() {
            let index = self.cold_free_list.pop().unwrap();
            self.stats.cold_allocations += 1;
            Some(PoolHandle::Cold(index))
        } else {
            self.stats.allocation_failures += 1;
            None
        }
    }

    /// Deallocate and update access patterns
    pub fn deallocate(&mut self, handle: PoolHandle) {
        match handle {
            PoolHandle::Hot(index) => {
                self.hot_free_list.push(index);
                self.access_counts[index] += 1;
                
                // Promote to hot pool if access count is high
                if self.access_counts[index] > 10 {
                    self.stats.promotions += 1;
                }
            }
            PoolHandle::Cold(index) => {
                self.cold_free_list.push(index);
                let cold_index = self.hot_pool.len() + index;
                self.access_counts[cold_index] += 1;
                
                // Consider promoting to hot pool
                if self.access_counts[cold_index] > 5 && !self.hot_free_list.is_empty() {
                    self.promote_to_hot(index);
                }
            }
        }
    }

    fn promote_to_hot(&mut self, cold_index: usize) {
        if let Some(hot_index) = self.hot_free_list.pop() {
            // Move object from cold to hot pool
            self.hot_pool[hot_index] = self.cold_pool[cold_index].clone();
            self.cold_free_list.push(cold_index);
            self.stats.promotions += 1;
        }
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> &PoolStatistics {
        &self.stats
    }
}

#[derive(Debug)]
pub enum PoolHandle {
    Hot(usize),
    Cold(usize),
}

#[derive(Debug)]
pub struct PoolStatistics {
    pub hot_allocations: u64,
    pub cold_allocations: u64,
    pub allocation_failures: u64,
    pub promotions: u64,
    pub demotions: u64,
}

impl PoolStatistics {
    fn new() -> Self {
        Self {
            hot_allocations: 0,
            cold_allocations: 0,
            allocation_failures: 0,
            promotions: 0,
            demotions: 0,
        }
    }
}

/// Utility functions for cache-aware algorithm design
pub struct CacheAwareUtils;

impl CacheAwareUtils {
    /// Calculate optimal block size for cache-friendly processing
    pub fn optimal_block_size<T>() -> usize {
        let element_size = std::mem::size_of::<T>();
        let cache_size = CACHE_LINE_SIZE;
        
        (cache_size / element_size).max(1)
    }

    /// Calculate optimal block size for specific cache level
    pub fn optimal_block_size_for_cache<T>(cache_size: usize) -> usize {
        let element_size = std::mem::size_of::<T>();
        let elements_per_cache = cache_size / element_size;
        
        // Use square root for 2D blocking, or linear for 1D
        (elements_per_cache as f64).sqrt() as usize
    }

    /// Check if two pointers are in the same cache line
    pub fn same_cache_line<T>(ptr1: *const T, ptr2: *const T) -> bool {
        let addr1 = ptr1 as usize;
        let addr2 = ptr2 as usize;
        
        (addr1 / CACHE_LINE_SIZE) == (addr2 / CACHE_LINE_SIZE)
    }

    /// Calculate cache line offset for a pointer
    pub fn cache_line_offset<T>(ptr: *const T) -> usize {
        (ptr as usize) % CACHE_LINE_SIZE
    }

    /// Align pointer to next cache line boundary
    pub fn align_to_cache_line<T>(ptr: *const T) -> *const T {
        let addr = ptr as usize;
        let aligned_addr = (addr + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1);
        aligned_addr as *const T
    }

    /// Calculate false sharing potential between data structures
    pub fn false_sharing_risk<T>(ptrs: &[*const T]) -> f64 {
        let mut same_line_pairs = 0;
        let mut total_pairs = 0;
        
        for i in 0..ptrs.len() {
            for j in i + 1..ptrs.len() {
                total_pairs += 1;
                if Self::same_cache_line(ptrs[i], ptrs[j]) {
                    same_line_pairs += 1;
                }
            }
        }
        
        if total_pairs > 0 {
            same_line_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Generate cache-friendly iteration order for multi-dimensional data
    pub fn cache_friendly_iteration_order(
        dimensions: &[usize],
        element_size: usize,
    ) -> Vec<usize> {
        let cache_line_elements = CACHE_LINE_SIZE / element_size;
        
        // Sort dimensions by size, process smallest first for better locality
        let mut indexed_dims: Vec<(usize, usize)> = dimensions
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();
        
        indexed_dims.sort_by_key(|&(_, size)| size);
        indexed_dims.into_iter().map(|(i, _)| i).collect()
    }

    /// Estimate cache miss rate for a given access pattern
    pub fn estimate_cache_miss_rate(
        data_size: usize,
        cache_size: usize,
        access_pattern: &AccessPattern,
    ) -> f64 {
        match access_pattern.pattern_type {
            PatternType::Sequential => {
                if data_size <= cache_size {
                    0.01 // Very low miss rate for sequential access in cache
                } else {
                    1.0 - (cache_size as f64 / data_size as f64)
                }
            }
            PatternType::Random => {
                if data_size <= cache_size {
                    0.1 // Some misses due to randomness
                } else {
                    0.9 // High miss rate for random access
                }
            }
            PatternType::Strided(stride) => {
                let cache_line_size = CACHE_LINE_SIZE;
                if stride <= cache_line_size {
                    0.05 // Good locality
                } else {
                    0.7 // Poor locality with large strides
                }
            }
            PatternType::FieldSpecific => 0.3, // Moderate miss rate
        }
    }
}

/// Cache-friendly B+ tree implementation optimized for trading data
pub struct CacheFriendlyBPlusTree<K, V> {
    root: Option<Box<BPlusNode<K, V>>>,
    order: usize, // Number of keys per node (optimized for cache line)
}

impl<K: Ord + Clone, V: Clone> CacheFriendlyBPlusTree<K, V> {
    /// Create new B+ tree with cache-optimized node size
    pub fn new() -> Self {
        let optimal_order = Self::calculate_optimal_order();
        Self {
            root: None,
            order: optimal_order,
        }
    }

    /// Calculate optimal node order based on cache line size
    fn calculate_optimal_order() -> usize {
        let key_size = std::mem::size_of::<K>();
        let value_size = std::mem::size_of::<V>();
        let pointer_size = std::mem::size_of::<*const ()>();
        
        // Aim for node size close to cache line size
        let node_overhead = 32; // Approximate overhead for node metadata
        let available_space = CACHE_LINE_SIZE - node_overhead;
        let entry_size = key_size + value_size + pointer_size;
        
        (available_space / entry_size).max(4) // Minimum order of 4
    }

    /// Insert with cache-friendly node splitting
    pub fn insert(&mut self, key: K, value: V) {
        if self.root.is_none() {
            self.root = Some(Box::new(BPlusNode::new_leaf(self.order)));
        }
        
        if let Some(ref mut root) = self.root {
            if let Some(new_root) = root.insert(key, value, self.order) {
                self.root = Some(new_root);
            }
        }
    }

    /// Search with prefetching for likely next accesses
    pub fn search(&self, key: &K) -> Option<&V> {
        self.root.as_ref()?.search(key)
    }

    /// Range query with cache-friendly traversal
    pub fn range_query(&self, start: &K, end: &K) -> Vec<(K, V)> {
        self.root.as_ref()
            .map(|root| root.range_query(start, end))
            .unwrap_or_default()
    }
}

/// Cache-optimized B+ tree node
struct BPlusNode<K, V> {
    keys: Vec<K>,
    values: Vec<V>, // Only used in leaf nodes
    children: Vec<Box<BPlusNode<K, V>>>, // Only used in internal nodes
    is_leaf: bool,
    next_leaf: Option<Box<BPlusNode<K, V>>>, // For leaf node linking
}

impl<K: Ord + Clone, V: Clone> BPlusNode<K, V> {
    fn new_leaf(order: usize) -> Self {
        Self {
            keys: Vec::with_capacity(order),
            values: Vec::with_capacity(order),
            children: Vec::new(),
            is_leaf: true,
            next_leaf: None,
        }
    }

    fn new_internal(order: usize) -> Self {
        Self {
            keys: Vec::with_capacity(order - 1),
            values: Vec::new(),
            children: Vec::with_capacity(order),
            is_leaf: false,
            next_leaf: None,
        }
    }

    fn insert(&mut self, key: K, value: V, order: usize) -> Option<Box<BPlusNode<K, V>>> {
        if self.is_leaf {
            self.insert_into_leaf(key, value, order)
        } else {
            self.insert_into_internal(key, value, order)
        }
    }

    fn insert_into_leaf(&mut self, key: K, value: V, order: usize) -> Option<Box<BPlusNode<K, V>>> {
        // Find insertion position
        let pos = self.keys.binary_search(&key).unwrap_or_else(|e| e);
        
        self.keys.insert(pos, key);
        self.values.insert(pos, value);
        
        // Check if split is needed
        if self.keys.len() >= order {
            self.split_leaf(order)
        } else {
            None
        }
    }

    fn insert_into_internal(&mut self, key: K, value: V, order: usize) -> Option<Box<BPlusNode<K, V>>> {
        // Find child to insert into
        let child_index = self.keys.binary_search(&key).unwrap_or_else(|e| e);
        
        if let Some(new_child) = self.children[child_index].insert(key, value, order) {
            // Child split, need to insert new key
            let split_key = new_child.keys[0].clone();
            self.keys.insert(child_index, split_key);
            self.children.insert(child_index + 1, new_child);
            
            // Check if this node needs to split
            if self.keys.len() >= order {
                self.split_internal(order)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn split_leaf(&mut self, order: usize) -> Option<Box<BPlusNode<K, V>>> {
        let mid = order / 2;
        let mut new_node = Box::new(BPlusNode::new_leaf(order));
        
        // Move half the keys and values to new node
        new_node.keys = self.keys.split_off(mid);
        new_node.values = self.values.split_off(mid);
        
        // Link leaf nodes
        new_node.next_leaf = self.next_leaf.take();
        self.next_leaf = Some(new_node.clone());
        
        Some(new_node)
    }

    fn split_internal(&mut self, order: usize) -> Option<Box<BPlusNode<K, V>>> {
        let mid = order / 2;
        let mut new_node = Box::new(BPlusNode::new_internal(order));
        
        // Move keys and children
        new_node.keys = self.keys.split_off(mid + 1);
        new_node.children = self.children.split_off(mid + 1);
        
        Some(new_node)
    }

    fn search(&self, key: &K) -> Option<&V> {
        if self.is_leaf {
            self.keys.binary_search(key)
                .ok()
                .map(|index| &self.values[index])
        } else {
            let child_index = self.keys.binary_search(key).unwrap_or_else(|e| e);
            
            // Prefetch the child we're about to access
            unsafe {
                DataPrefetcher::prefetch_line(
                    self.children[child_index].as_ref(),
                    PrefetchHint::T0,
                );
            }
            
            self.children[child_index].search(key)
        }
    }

    fn range_query(&self, start: &K, end: &K) -> Vec<(K, V)> {
        let mut results = Vec::new();
        self.collect_range(start, end, &mut results);
        results
    }

    fn collect_range(&self, start: &K, end: &K, results: &mut Vec<(K, V)>) {
        if self.is_leaf {
            for (i, key) in self.keys.iter().enumerate() {
                if key >= start && key <= end {
                    results.push((key.clone(), self.values[i].clone()));
                }
            }
        } else {
            for (i, child) in self.children.iter().enumerate() {
                // Prefetch child before accessing
                unsafe {
                    DataPrefetcher::prefetch_line(child.as_ref(), PrefetchHint::T0);
                }
                child.collect_range(start, end, results);
            }
        }
    }
}

/// Cache-friendly skip list for ordered data with probabilistic balancing
pub struct CacheFriendlySkipList<K, V> {
    head: Box<SkipNode<K, V>>,
    max_level: usize,
    current_level: usize,
    rng: fastrand::Rng,
}

impl<K: Ord + Clone, V: Clone> CacheFriendlySkipList<K, V> {
    pub fn new() -> Self {
        let max_level = 16; // Reasonable maximum for most use cases
        Self {
            head: Box::new(SkipNode::new_head(max_level)),
            max_level,
            current_level: 0,
            rng: fastrand::Rng::new(),
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let level = self.random_level();
        let new_node = Box::new(SkipNode::new(key.clone(), value, level));
        
        let mut update = vec![&mut *self.head as *mut SkipNode<K, V>; self.max_level];
        let mut current = &mut *self.head;
        
        // Find insertion point at each level
        for i in (0..=self.current_level).rev() {
            while let Some(ref mut next) = current.forward[i] {
                if next.key < key {
                    current = next;
                } else {
                    break;
                }
            }
            update[i] = current;
        }
        
        // Insert new node
        for i in 0..=level {
            unsafe {
                let update_node = &mut *update[i];
                let new_node_ptr = Box::into_raw(new_node.clone());
                (*new_node_ptr).forward[i] = update_node.forward[i].take();
                update_node.forward[i] = Some(Box::from_raw(new_node_ptr));
            }
        }
        
        if level > self.current_level {
            self.current_level = level;
        }
    }

    pub fn search(&self, key: &K) -> Option<&V> {
        let mut current = &*self.head;
        
        for i in (0..=self.current_level).rev() {
            while let Some(ref next) = current.forward[i] {
                if next.key < *key {
                    // Prefetch the next node we might visit
                    unsafe {
                        if let Some(ref next_next) = next.forward[i] {
                            DataPrefetcher::prefetch_line(
                                next_next.as_ref(),
                                PrefetchHint::T0,
                            );
                        }
                    }
                    current = next;
                } else if next.key == *key {
                    return Some(&next.value);
                } else {
                    break;
                }
            }
        }
        
        None
    }

    fn random_level(&mut self) -> usize {
        let mut level = 0;
        while self.rng.bool() && level < self.max_level - 1 {
            level += 1;
        }
        level
    }
}

/// Cache-aligned skip list node
struct SkipNode<K, V> {
    key: K,
    value: V,
    forward: Vec<Option<Box<SkipNode<K, V>>>>,
}

impl<K: Default, V: Default> SkipNode<K, V> {
    fn new(key: K, value: V, level: usize) -> Self {
        Self {
            key,
            value,
            forward: vec![None; level + 1],
        }
    }

    fn new_head(max_level: usize) -> Self {
        Self {
            key: K::default(),
            value: V::default(),
            forward: vec![None; max_level],
        }
    }
}

/// Cache-friendly circular buffer with prefetching
pub struct CacheFriendlyCircularBuffer<T> {
    buffer: Vec<T>,
    head: usize,
    tail: usize,
    capacity: usize,
    prefetch_distance: usize,
}

impl<T: Default + Clone> CacheFriendlyCircularBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize(capacity, T::default());
        
        // Calculate optimal prefetch distance
        let prefetch_distance = (CACHE_LINE_SIZE / std::mem::size_of::<T>()).max(1);
        
        Self {
            buffer,
            head: 0,
            tail: 0,
            capacity,
            prefetch_distance,
        }
    }

    pub fn push(&mut self, item: T) -> Result<(), T> {
        let next_tail = (self.tail + 1) % self.capacity;
        
        if next_tail == self.head {
            return Err(item); // Buffer full
        }
        
        // Prefetch future write locations
        unsafe {
            let prefetch_index = (self.tail + self.prefetch_distance) % self.capacity;
            DataPrefetcher::prefetch_line(
                self.buffer.as_ptr().add(prefetch_index),
                PrefetchHint::T0,
            );
        }
        
        self.buffer[self.tail] = item;
        self.tail = next_tail;
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.head == self.tail {
            return None; // Buffer empty
        }
        
        // Prefetch future read locations
        unsafe {
            let prefetch_index = (self.head + self.prefetch_distance) % self.capacity;
            DataPrefetcher::prefetch_line(
                self.buffer.as_ptr().add(prefetch_index),
                PrefetchHint::T0,
            );
        }
        
        let item = self.buffer[self.head].clone();
        self.head = (self.head + 1) % self.capacity;
        Some(item)
    }

    pub fn len(&self) -> usize {
        if self.tail >= self.head {
            self.tail - self.head
        } else {
            self.capacity - self.head + self.tail
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    pub fn is_full(&self) -> bool {
        (self.tail + 1) % self.capacity == self.head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_oblivious_sort() {
        let mut data = vec![5, 2, 8, 1, 9, 3, 7, 4, 6];
        CacheObliviousAlgorithms::cache_oblivious_sort(&mut data);
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_cache_aware_memory_pool() {
        let mut pool = CacheAwareMemoryPool::<u64>::new(10, 10);
        
        let handle1 = pool.allocate(true).unwrap();
        let handle2 = pool.allocate(false).unwrap();
        
        pool.deallocate(handle1);
        pool.deallocate(handle2);
        
        let stats = pool.get_statistics();
        assert_eq!(stats.hot_allocations, 1);
        assert_eq!(stats.cold_allocations, 1);
    }

    #[test]
    fn test_cache_friendly_circular_buffer() {
        let mut buffer = CacheFriendlyCircularBuffer::new(5);
        
        assert!(buffer.push(1).is_ok());
        assert!(buffer.push(2).is_ok());
        assert!(buffer.push(3).is_ok());
        
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_data_layout_optimizer() {
        let field_sizes = vec![8, 4, 2, 1, 8, 4];
        let optimized = DataLayoutOptimizer::optimize_field_layout(&field_sizes);
        
        // Should order by size descending: 8, 8, 4, 4, 2, 1
        // Which corresponds to indices: 0, 4, 1, 5, 2, 3
        assert_eq!(optimized[0], 0); // First 8-byte field
        assert_eq!(optimized[1], 4); // Second 8-byte field
    }
}