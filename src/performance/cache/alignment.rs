use std::alloc::{alloc, dealloc, Layout};
use std::mem;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicUsize, Ordering};

/// Cache line size constants for different architectures
pub const CACHE_LINE_SIZE: usize = 64; // Most common cache line size
pub const L1_CACHE_SIZE: usize = 32 * 1024; // 32KB typical L1 cache
pub const L2_CACHE_SIZE: usize = 256 * 1024; // 256KB typical L2 cache
pub const L3_CACHE_SIZE: usize = 8 * 1024 * 1024; // 8MB typical L3 cache

/// Macro to align structures to cache line boundaries
#[macro_export]
macro_rules! cache_aligned {
    ($name:ident { $($field:ident: $type:ty),* $(,)? }) => {
        #[repr(C, align(64))]
        pub struct $name {
            $(pub $field: $type,)*
        }
    };
}

/// Cache-line aligned atomic counter to prevent false sharing
#[repr(C, align(64))]
pub struct CacheAlignedAtomicU64 {
    pub value: AtomicU64,
    _padding: [u8; 64 - 8], // Pad to full cache line
}

impl CacheAlignedAtomicU64 {
    pub fn new(value: u64) -> Self {
        Self {
            value: AtomicU64::new(value),
            _padding: [0; 56],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u64 {
        self.value.load(order)
    }

    #[inline(always)]
    pub fn store(&self, val: u64, order: Ordering) {
        self.value.store(val, order)
    }

    #[inline(always)]
    pub fn fetch_add(&self, val: u64, order: Ordering) -> u64 {
        self.value.fetch_add(val, order)
    }
}

/// Cache-line aligned atomic counter for 32-bit values
#[repr(C, align(64))]
pub struct CacheAlignedAtomicU32 {
    pub value: AtomicU32,
    _padding: [u8; 64 - 4],
}

impl CacheAlignedAtomicU32 {
    pub fn new(value: u32) -> Self {
        Self {
            value: AtomicU32::new(value),
            _padding: [0; 60],
        }
    }

    #[inline(always)]
    pub fn load(&self, order: Ordering) -> u32 {
        self.value.load(order)
    }

    #[inline(always)]
    pub fn store(&self, val: u32, order: Ordering) {
        self.value.store(val, order)
    }

    #[inline(always)]
    pub fn fetch_add(&self, val: u32, order: Ordering) -> u32 {
        self.value.fetch_add(val, order)
    }
}

/// Cache-friendly order structure with optimal field layout
#[repr(C, align(64))]
pub struct CacheFriendlyOrder {
    // Hot fields accessed together (first cache line)
    pub order_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub timestamp: u64,
    
    // Moderately accessed fields (second cache line if needed)
    pub side: u8, // Buy/Sell
    pub order_type: u8, // Market/Limit
    pub status: u8, // Active/Filled/Cancelled
    pub priority: u8,
    pub trader_id: u32,
    
    // Cold fields (accessed less frequently)
    pub symbol: [u8; 16], // Fixed-size symbol
    pub metadata: u64,
    
    // Padding to ensure proper alignment
    _padding: [u8; 8],
}

impl CacheFriendlyOrder {
    pub fn new(
        order_id: u64,
        price: u64,
        quantity: u64,
        side: u8,
        order_type: u8,
        trader_id: u32,
        symbol: &str,
    ) -> Self {
        let mut symbol_bytes = [0u8; 16];
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(16);
        symbol_bytes[..len].copy_from_slice(&bytes[..len]);

        Self {
            order_id,
            price,
            quantity,
            timestamp: 0, // Will be set when processed
            side,
            order_type,
            status: 0, // Active
            priority: 0,
            trader_id,
            symbol: symbol_bytes,
            metadata: 0,
            _padding: [0; 8],
        }
    }
}

/// Cache-friendly trade structure
#[repr(C, align(64))]
pub struct CacheFriendlyTrade {
    // Most frequently accessed fields
    pub trade_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub timestamp: u64,
    
    // Order references
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    
    // Less frequently accessed
    pub symbol: [u8; 16],
    pub fees: u64,
    
    _padding: [u8; 16],
}

/// Cache-aligned memory allocator
pub struct CacheAlignedAllocator;

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory
    pub fn allocate<T>(count: usize) -> Result<NonNull<T>, std::alloc::AllocError> {
        let layout = Layout::from_size_align(
            mem::size_of::<T>() * count,
            CACHE_LINE_SIZE,
        ).map_err(|_| std::alloc::AllocError)?;

        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr as *mut T).ok_or(std::alloc::AllocError)
    }

    /// Deallocate cache-aligned memory
    pub unsafe fn deallocate<T>(ptr: NonNull<T>, count: usize) {
        let layout = Layout::from_size_align_unchecked(
            mem::size_of::<T>() * count,
            CACHE_LINE_SIZE,
        );
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }
}

/// Utility to check if a pointer is cache-aligned
pub fn is_cache_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % CACHE_LINE_SIZE == 0
}

/// Calculate padding needed for cache alignment
pub fn cache_padding_for_size(size: usize) -> usize {
    let remainder = size % CACHE_LINE_SIZE;
    if remainder == 0 {
        0
    } else {
        CACHE_LINE_SIZE - remainder
    }
}

/// Advanced cache-line aligned data structures for trading
#[repr(C, align(64))]
pub struct CacheOptimizedOrderBook {
    // Hot data - accessed on every operation (first cache line)
    pub best_bid: u64,
    pub best_ask: u64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub last_update_time: u64,
    pub sequence_number: u64,
    pub spread: u32,
    pub depth_levels: u32,
    
    // Moderately hot data (second cache line)
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub order_count: u32,
    pub trade_count: u32,
    pub volatility: u32,
    pub _padding1: [u8; 20],
    
    // Cold data - accessed less frequently
    pub symbol: [u8; 16],
    pub market_id: u32,
    pub session_id: u32,
    pub flags: u32,
    pub _padding2: [u8; 12],
}

impl CacheOptimizedOrderBook {
    pub fn new(symbol: &str) -> Self {
        let mut symbol_bytes = [0u8; 16];
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(16);
        symbol_bytes[..len].copy_from_slice(&bytes[..len]);

        Self {
            best_bid: 0,
            best_ask: 0,
            bid_volume: 0,
            ask_volume: 0,
            last_update_time: 0,
            sequence_number: 0,
            spread: 0,
            depth_levels: 0,
            total_bid_volume: 0,
            total_ask_volume: 0,
            order_count: 0,
            trade_count: 0,
            volatility: 0,
            _padding1: [0; 20],
            symbol: symbol_bytes,
            market_id: 0,
            session_id: 0,
            flags: 0,
            _padding2: [0; 12],
        }
    }

    #[inline(always)]
    pub fn update_best_prices(&mut self, bid: u64, ask: u64, timestamp: u64) {
        // All hot data in same cache line - single cache miss
        self.best_bid = bid;
        self.best_ask = ask;
        self.spread = ((ask - bid) / 100) as u32; // Basis points
        self.last_update_time = timestamp;
        self.sequence_number += 1;
    }
}

/// False sharing elimination through strategic padding
#[repr(C, align(64))]
pub struct FalseSharingFreeCounters {
    // Each counter gets its own cache line to prevent false sharing
    pub counter1: AtomicU64,
    _pad1: [u8; 56], // 64 - 8 = 56 bytes padding
    
    pub counter2: AtomicU64,
    _pad2: [u8; 56],
    
    pub counter3: AtomicU64,
    _pad3: [u8; 56],
    
    pub counter4: AtomicU64,
    _pad4: [u8; 56],
}

impl FalseSharingFreeCounters {
    pub fn new() -> Self {
        Self {
            counter1: AtomicU64::new(0),
            _pad1: [0; 56],
            counter2: AtomicU64::new(0),
            _pad2: [0; 56],
            counter3: AtomicU64::new(0),
            _pad3: [0; 56],
            counter4: AtomicU64::new(0),
            _pad4: [0; 56],
        }
    }
}

/// Temporal and spatial locality optimizer
pub struct LocalityOptimizer;

impl LocalityOptimizer {
    /// Reorganize data for better temporal locality
    pub fn optimize_temporal_locality<T: Clone>(
        data: &mut Vec<T>,
        access_pattern: &[usize],
    ) {
        // Create a new ordering based on access frequency
        let mut frequency_map = std::collections::HashMap::new();
        for &index in access_pattern {
            *frequency_map.entry(index).or_insert(0) += 1;
        }
        
        // Sort indices by access frequency (most frequent first)
        let mut sorted_indices: Vec<_> = frequency_map.into_iter().collect();
        sorted_indices.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Reorganize data to place frequently accessed items together
        let mut optimized_data = Vec::with_capacity(data.len());
        let mut remaining_indices: std::collections::HashSet<_> = (0..data.len()).collect();
        
        // Add frequently accessed items first
        for (index, _) in sorted_indices {
            if remaining_indices.remove(&index) {
                optimized_data.push(data[index].clone());
            }
        }
        
        // Add remaining items
        for index in remaining_indices {
            optimized_data.push(data[index].clone());
        }
        
        *data = optimized_data;
    }

    /// Optimize spatial locality through data structure layout
    pub fn optimize_spatial_locality<T>(
        hot_fields: &[T],
        warm_fields: &[T],
        cold_fields: &[T],
    ) -> SpatiallyOptimizedData<T>
    where
        T: Clone,
    {
        SpatiallyOptimizedData {
            hot_data: hot_fields.to_vec(),
            warm_data: warm_fields.to_vec(),
            cold_data: cold_fields.to_vec(),
        }
    }

    /// Calculate optimal data structure packing
    pub fn calculate_optimal_packing(field_sizes: &[usize]) -> Vec<usize> {
        let mut fields: Vec<(usize, usize)> = field_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();
        
        // Sort by size descending to minimize padding
        fields.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Apply additional optimizations for cache line alignment
        let mut optimized_order = Vec::new();
        let mut current_line_usage = 0;
        
        for (index, size) in fields {
            if current_line_usage + size <= CACHE_LINE_SIZE {
                optimized_order.push(index);
                current_line_usage += size;
            } else {
                // Start new cache line
                optimized_order.push(index);
                current_line_usage = size;
            }
        }
        
        optimized_order
    }
}

#[derive(Debug, Clone)]
pub struct SpatiallyOptimizedData<T> {
    pub hot_data: Vec<T>,
    pub warm_data: Vec<T>,
    pub cold_data: Vec<T>,
}

/// Memory layout analyzer for cache optimization
pub struct MemoryLayoutAnalyzer;

impl MemoryLayoutAnalyzer {
    /// Analyze memory layout efficiency
    pub fn analyze_layout<T>(data: &[T]) -> LayoutAnalysis {
        let element_size = std::mem::size_of::<T>();
        let total_size = data.len() * element_size;
        let cache_lines_used = (total_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        let cache_utilization = total_size as f64 / (cache_lines_used * CACHE_LINE_SIZE) as f64;
        
        LayoutAnalysis {
            element_size,
            total_elements: data.len(),
            total_size,
            cache_lines_used,
            cache_utilization,
            alignment_waste: (cache_lines_used * CACHE_LINE_SIZE) - total_size,
            recommendations: Self::generate_layout_recommendations(cache_utilization),
        }
    }

    fn generate_layout_recommendations(utilization: f64) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if utilization < 0.5 {
            recommendations.push("Consider struct packing to reduce memory waste".to_string());
            recommendations.push("Evaluate field ordering for better alignment".to_string());
        }
        
        if utilization < 0.75 {
            recommendations.push("Consider separating hot and cold data".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug)]
pub struct LayoutAnalysis {
    pub element_size: usize,
    pub total_elements: usize,
    pub total_size: usize,
    pub cache_lines_used: usize,
    pub cache_utilization: f64,
    pub alignment_waste: usize,
    pub recommendations: Vec<String>,
}