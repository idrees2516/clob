use std::arch::x86_64::*;
use std::ptr;

/// Prefetch strategies for different access patterns
pub enum PrefetchStrategy {
    /// Prefetch for temporal locality (data will be used again soon)
    Temporal,
    /// Prefetch for non-temporal access (data used once)
    NonTemporal,
    /// Prefetch to L1 cache
    L1,
    /// Prefetch to L2 cache
    L2,
    /// Prefetch to L3 cache
    L3,
}

/// Prefetch hint levels
#[derive(Clone, Copy)]
pub enum PrefetchHint {
    /// Prefetch to all cache levels
    T0 = 3,
    /// Prefetch to L2 and L3, not L1
    T1 = 2,
    /// Prefetch to L3 only
    T2 = 1,
    /// Non-temporal prefetch (bypass cache)
    NTA = 0,
}

/// Data prefetcher for optimizing memory access patterns
pub struct DataPrefetcher;

impl DataPrefetcher {
    /// Prefetch a single cache line
    #[inline(always)]
    pub unsafe fn prefetch_line<T>(ptr: *const T, hint: PrefetchHint) {
        #[cfg(target_arch = "x86_64")]
        {
            match hint {
                PrefetchHint::T0 => _mm_prefetch(ptr as *const i8, _MM_HINT_T0),
                PrefetchHint::T1 => _mm_prefetch(ptr as *const i8, _MM_HINT_T1),
                PrefetchHint::T2 => _mm_prefetch(ptr as *const i8, _MM_HINT_T2),
                PrefetchHint::NTA => _mm_prefetch(ptr as *const i8, _MM_HINT_NTA),
            }
        }
    }

    /// Prefetch multiple cache lines for sequential access
    #[inline(always)]
    pub unsafe fn prefetch_sequential<T>(
        start_ptr: *const T,
        count: usize,
        hint: PrefetchHint,
    ) {
        let cache_line_size = 64;
        let element_size = std::mem::size_of::<T>();
        let elements_per_line = cache_line_size / element_size.max(1);
        
        let mut current_ptr = start_ptr;
        let mut remaining = count;
        
        while remaining > 0 {
            Self::prefetch_line(current_ptr, hint);
            let advance = elements_per_line.min(remaining);
            current_ptr = current_ptr.add(advance);
            remaining = remaining.saturating_sub(advance);
        }
    }

    /// Prefetch with stride pattern (for accessing every Nth element)
    #[inline(always)]
    pub unsafe fn prefetch_strided<T>(
        start_ptr: *const T,
        count: usize,
        stride: usize,
        hint: PrefetchHint,
    ) {
        let mut current_ptr = start_ptr;
        for _ in 0..count {
            Self::prefetch_line(current_ptr, hint);
            current_ptr = current_ptr.add(stride);
        }
    }

    /// Prefetch for linked list traversal
    #[inline(always)]
    pub unsafe fn prefetch_linked_list<T, F>(
        start_ptr: *const T,
        max_depth: usize,
        get_next: F,
        hint: PrefetchHint,
    ) where
        F: Fn(*const T) -> *const T,
    {
        let mut current_ptr = start_ptr;
        let mut depth = 0;
        
        while !current_ptr.is_null() && depth < max_depth {
            Self::prefetch_line(current_ptr, hint);
            current_ptr = get_next(current_ptr);
            depth += 1;
        }
    }
}

/// Prefetch patterns for common trading data structures
pub struct TradingDataPrefetcher;

impl TradingDataPrefetcher {
    /// Prefetch order book levels for price traversal
    pub unsafe fn prefetch_price_levels<T>(
        levels: &[*const T],
        lookahead: usize,
    ) {
        for i in 0..lookahead.min(levels.len()) {
            if !levels[i].is_null() {
                DataPrefetcher::prefetch_line(levels[i], PrefetchHint::T0);
            }
        }
    }

    /// Prefetch orders at a price level
    pub unsafe fn prefetch_order_chain<T, F>(
        first_order: *const T,
        get_next: F,
        prefetch_count: usize,
    ) where
        F: Fn(*const T) -> *const T,
    {
        DataPrefetcher::prefetch_linked_list(
            first_order,
            prefetch_count,
            get_next,
            PrefetchHint::T0,
        );
    }

    /// Prefetch memory pool free list
    pub unsafe fn prefetch_free_list<T>(
        free_list_head: *const T,
        prefetch_depth: usize,
    ) {
        // Assume next pointer is at offset 0
        let get_next = |ptr: *const T| -> *const T {
            if ptr.is_null() {
                ptr::null()
            } else {
                *(ptr as *const *const T)
            }
        };
        
        DataPrefetcher::prefetch_linked_list(
            free_list_head,
            prefetch_depth,
            get_next,
            PrefetchHint::T1, // L2/L3 cache for pool management
        );
    }
}

/// Automatic prefetching for predictable access patterns
pub struct AutoPrefetcher {
    /// History of accessed addresses for pattern detection
    access_history: Vec<usize>,
    /// Maximum history size
    max_history: usize,
    /// Detected stride pattern
    detected_stride: Option<isize>,
    /// Confidence in stride pattern
    stride_confidence: f32,
}

impl AutoPrefetcher {
    pub fn new(max_history: usize) -> Self {
        Self {
            access_history: Vec::with_capacity(max_history),
            max_history,
            detected_stride: None,
            stride_confidence: 0.0,
        }
    }

    /// Record a memory access and potentially trigger prefetch
    pub unsafe fn record_access<T>(&mut self, ptr: *const T) {
        let addr = ptr as usize;
        self.access_history.push(addr);
        
        if self.access_history.len() > self.max_history {
            self.access_history.remove(0);
        }
        
        // Detect stride pattern
        if self.access_history.len() >= 3 {
            self.detect_stride_pattern();
            
            // If we have a confident stride pattern, prefetch ahead
            if let Some(stride) = self.detected_stride {
                if self.stride_confidence > 0.7 {
                    let next_addr = (addr as isize + stride * 2) as usize;
                    DataPrefetcher::prefetch_line(
                        next_addr as *const T,
                        PrefetchHint::T1,
                    );
                }
            }
        }
    }

    fn detect_stride_pattern(&mut self) {
        if self.access_history.len() < 3 {
            return;
        }
        
        let len = self.access_history.len();
        let recent_strides: Vec<isize> = self.access_history
            .windows(2)
            .map(|w| w[1] as isize - w[0] as isize)
            .collect();
        
        // Look for consistent stride
        if recent_strides.len() >= 2 {
            let last_stride = recent_strides[recent_strides.len() - 1];
            let consistent_count = recent_strides
                .iter()
                .rev()
                .take(4) // Look at last 4 strides
                .filter(|&&s| s == last_stride)
                .count();
            
            self.stride_confidence = consistent_count as f32 / 4.0;
            
            if self.stride_confidence > 0.5 {
                self.detected_stride = Some(last_stride);
            } else {
                self.detected_stride = None;
            }
        }
    }
}

/// Software prefetching for specific data structures
pub trait PrefetchOptimized {
    /// Prefetch data that will be needed soon
    unsafe fn prefetch_ahead(&self, lookahead: usize);
    
    /// Prefetch related data structures
    unsafe fn prefetch_related(&self);
}

/// Prefetch configuration for different workload patterns
#[derive(Clone, Copy)]
pub struct PrefetchConfig {
    /// How many elements ahead to prefetch
    pub lookahead_distance: usize,
    /// Prefetch hint to use
    pub hint: PrefetchHint,
    /// Whether to use automatic stride detection
    pub auto_detect_stride: bool,
    /// Minimum confidence for auto prefetch
    pub min_confidence: f32,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            lookahead_distance: 2,
            hint: PrefetchHint::T0,
            auto_detect_stride: true,
            min_confidence: 0.7,
        }
    }
}

impl PrefetchConfig {
    /// Configuration optimized for order book traversal
    pub fn order_book() -> Self {
        Self {
            lookahead_distance: 3,
            hint: PrefetchHint::T0,
            auto_detect_stride: false,
            min_confidence: 0.8,
        }
    }

    /// Configuration optimized for memory pool access
    pub fn memory_pool() -> Self {
        Self {
            lookahead_distance: 4,
            hint: PrefetchHint::T1,
            auto_detect_stride: true,
            min_confidence: 0.6,
        }
    }

    /// Configuration for sequential data processing
    pub fn sequential() -> Self {
        Self {
            lookahead_distance: 8,
            hint: PrefetchHint::T0,
            auto_detect_stride: true,
            min_confidence: 0.9,
        }
    }
}