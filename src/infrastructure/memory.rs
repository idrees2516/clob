use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr::NonNull;
use std::collections::HashMap;
use parking_lot::{RwLock, Mutex};
use crate::error::InfrastructureError;

/// NUMA-aware memory allocator for high-performance trading systems
pub struct NumaAwareAllocator {
    system_allocator: System,
    numa_nodes: Vec<NumaNode>,
    current_node: AtomicUsize,
    allocation_stats: Arc<RwLock<AllocationStats>>,
    memory_pools: Arc<RwLock<HashMap<usize, MemoryPool>>>,
}

#[derive(Debug)]
pub struct NumaNode {
    pub id: usize,
    pub cpu_mask: u64,
    pub memory_size: usize,
    pub free_memory: AtomicUsize,
    pub allocations: AtomicUsize,
}

#[derive(Debug, Default)]
pub struct AllocationStats {
    pub total_allocated: usize,
    pub total_freed: usize,
    pub peak_usage: usize,
    pub current_usage: usize,
    pub allocation_count: usize,
    pub free_count: usize,
    pub numa_node_usage: HashMap<usize, usize>,
}

/// High-performance memory pool for frequent allocations
pub struct MemoryPool {
    pub size_class: usize,
    pub blocks: Mutex<Vec<NonNull<u8>>>,
    pub block_size: usize,
    pub total_blocks: AtomicUsize,
    pub free_blocks: AtomicUsize,
    pub numa_node: usize,
}

/// Lock-free memory arena for order book operations
pub struct LockFreeArena {
    memory: NonNull<u8>,
    size: usize,
    offset: AtomicUsize,
    numa_node: usize,
}

/// Memory-mapped region for large data structures
pub struct MemoryMappedRegion {
    ptr: NonNull<u8>,
    size: usize,
    file_descriptor: Option<i32>,
    is_shared: bool,
}

unsafe impl GlobalAlloc for NumaAwareAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let start_time = std::time::Instant::now();
        
        // Try to allocate from appropriate memory pool first
        if let Some(ptr) = self.try_pool_allocation(layout) {
            self.update_allocation_stats(layout.size(), start_time);
            return ptr.as_ptr();
        }

        // Fall back to system allocator with NUMA awareness
        let ptr = self.numa_aware_alloc(layout);
        self.update_allocation_stats(layout.size(), start_time);
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Try to return to memory pool first
        if self.try_pool_deallocation(ptr, layout) {
            return;
        }

        // Fall back to system deallocation
        self.system_allocator.dealloc(ptr, layout);
        self.update_deallocation_stats(layout.size());
    }
}

impl NumaAwareAllocator {
    pub fn new() -> Result<Self, InfrastructureError> {
        let numa_nodes = Self::detect_numa_topology()?;
        
        Ok(Self {
            system_allocator: System,
            numa_nodes,
            current_node: AtomicUsize::new(0),
            allocation_stats: Arc::new(RwLock::new(AllocationStats::default())),
            memory_pools: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Detect NUMA topology of the system
    fn detect_numa_topology() -> Result<Vec<NumaNode>, InfrastructureError> {
        let mut nodes = Vec::new();
        
        #[cfg(target_os = "linux")]
        {
            // Read NUMA information from /sys/devices/system/node/
            use std::fs;
            
            let node_dirs = fs::read_dir("/sys/devices/system/node/")
                .map_err(|e| InfrastructureError::NumaError(format!("Failed to read NUMA nodes: {}", e)))?;
            
            for entry in node_dirs {
                let entry = entry.map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
                let path = entry.path();
                
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("node") {
                        if let Ok(node_id) = name[4..].parse::<usize>() {
                            let cpu_mask = Self::read_cpu_mask(&path)?;
                            let memory_size = Self::read_memory_size(&path)?;
                            
                            nodes.push(NumaNode {
                                id: node_id,
                                cpu_mask,
                                memory_size,
                                free_memory: AtomicUsize::new(memory_size),
                                allocations: AtomicUsize::new(0),
                            });
                        }
                    }
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback: create single node representing the entire system
            nodes.push(NumaNode {
                id: 0,
                cpu_mask: u64::MAX,
                memory_size: 16 * 1024 * 1024 * 1024, // 16GB default
                free_memory: AtomicUsize::new(16 * 1024 * 1024 * 1024),
                allocations: AtomicUsize::new(0),
            });
        }
        
        if nodes.is_empty() {
            return Err(InfrastructureError::NumaError("No NUMA nodes detected".to_string()));
        }
        
        Ok(nodes)
    }

    #[cfg(target_os = "linux")]
    fn read_cpu_mask(node_path: &std::path::Path) -> Result<u64, InfrastructureError> {
        use std::fs;
        
        let cpulist_path = node_path.join("cpulist");
        let cpulist = fs::read_to_string(cpulist_path)
            .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
        
        let mut mask = 0u64;
        for range in cpulist.trim().split(',') {
            if let Some((start, end)) = range.split_once('-') {
                let start: usize = start.parse()
                    .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
                let end: usize = end.parse()
                    .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
                
                for cpu in start..=end {
                    if cpu < 64 {
                        mask |= 1u64 << cpu;
                    }
                }
            } else {
                let cpu: usize = range.parse()
                    .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
                if cpu < 64 {
                    mask |= 1u64 << cpu;
                }
            }
        }
        
        Ok(mask)
    }

    #[cfg(target_os = "linux")]
    fn read_memory_size(node_path: &std::path::Path) -> Result<usize, InfrastructureError> {
        use std::fs;
        
        let meminfo_path = node_path.join("meminfo");
        let meminfo = fs::read_to_string(meminfo_path)
            .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
        
        for line in meminfo.lines() {
            if line.starts_with("Node") && line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let size_kb: usize = parts[3].parse()
                        .map_err(|e| InfrastructureError::NumaError(e.to_string()))?;
                    return Ok(size_kb * 1024); // Convert KB to bytes
                }
            }
        }
        
        Ok(4 * 1024 * 1024 * 1024) // 4GB default
    }

    /// NUMA-aware allocation strategy
    unsafe fn numa_aware_alloc(&self, layout: Layout) -> *mut u8 {
        // Get current thread's preferred NUMA node
        let preferred_node = self.get_preferred_numa_node();
        
        // Try to allocate on preferred node first
        if let Some(ptr) = self.try_allocate_on_node(layout, preferred_node) {
            return ptr;
        }
        
        // Fall back to round-robin allocation
        let node_count = self.numa_nodes.len();
        for i in 0..node_count {
            let node_id = (preferred_node + i) % node_count;
            if let Some(ptr) = self.try_allocate_on_node(layout, node_id) {
                return ptr;
            }
        }
        
        // Final fallback to system allocator
        self.system_allocator.alloc(layout)
    }

    fn get_preferred_numa_node(&self) -> usize {
        // Get current CPU and map to NUMA node
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpu) = Self::get_current_cpu() {
                for node in &self.numa_nodes {
                    if node.cpu_mask & (1u64 << cpu) != 0 {
                        return node.id;
                    }
                }
            }
        }
        
        // Round-robin fallback
        self.current_node.fetch_add(1, Ordering::Relaxed) % self.numa_nodes.len()
    }

    #[cfg(target_os = "linux")]
    fn get_current_cpu() -> Result<usize, InfrastructureError> {
        unsafe {
            let cpu = libc::sched_getcpu();
            if cpu >= 0 {
                Ok(cpu as usize)
            } else {
                Err(InfrastructureError::NumaError("Failed to get current CPU".to_string()))
            }
        }
    }

    unsafe fn try_allocate_on_node(&self, layout: Layout, node_id: usize) -> Option<*mut u8> {
        if node_id >= self.numa_nodes.len() {
            return None;
        }

        let node = &self.numa_nodes[node_id];
        
        // Check if node has enough free memory
        let current_free = node.free_memory.load(Ordering::Relaxed);
        if current_free < layout.size() {
            return None;
        }

        // Try to allocate using mbind system call on Linux
        #[cfg(target_os = "linux")]
        {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                layout.size(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr != libc::MAP_FAILED {
                // Bind memory to specific NUMA node
                let node_mask = 1u64 << node_id;
                let result = libc::mbind(
                    ptr,
                    layout.size(),
                    libc::MPOL_BIND,
                    &node_mask as *const u64 as *const libc::c_ulong,
                    64,
                    0,
                );

                if result == 0 {
                    node.free_memory.fetch_sub(layout.size(), Ordering::Relaxed);
                    node.allocations.fetch_add(1, Ordering::Relaxed);
                    return Some(ptr as *mut u8);
                } else {
                    libc::munmap(ptr, layout.size());
                }
            }
        }

        None
    }

    fn try_pool_allocation(&self, layout: Layout) -> Option<NonNull<u8>> {
        let size_class = Self::get_size_class(layout.size());
        let pools = self.memory_pools.read();
        
        if let Some(pool) = pools.get(&size_class) {
            return pool.allocate();
        }
        
        None
    }

    fn try_pool_deallocation(&self, ptr: *mut u8, layout: Layout) -> bool {
        let size_class = Self::get_size_class(layout.size());
        let pools = self.memory_pools.read();
        
        if let Some(pool) = pools.get(&size_class) {
            unsafe {
                if let Some(non_null_ptr) = NonNull::new(ptr) {
                    pool.deallocate(non_null_ptr);
                    return true;
                }
            }
        }
        
        false
    }

    fn get_size_class(size: usize) -> usize {
        // Power-of-2 size classes
        if size <= 64 { 64 }
        else if size <= 128 { 128 }
        else if size <= 256 { 256 }
        else if size <= 512 { 512 }
        else if size <= 1024 { 1024 }
        else if size <= 2048 { 2048 }
        else if size <= 4096 { 4096 }
        else if size <= 8192 { 8192 }
        else { size.next_power_of_two() }
    }

    fn update_allocation_stats(&self, size: usize, start_time: std::time::Instant) {
        let mut stats = self.allocation_stats.write();
        stats.total_allocated += size;
        stats.current_usage += size;
        stats.allocation_count += 1;
        
        if stats.current_usage > stats.peak_usage {
            stats.peak_usage = stats.current_usage;
        }
    }

    fn update_deallocation_stats(&self, size: usize) {
        let mut stats = self.allocation_stats.write();
        stats.total_freed += size;
        stats.current_usage = stats.current_usage.saturating_sub(size);
        stats.free_count += 1;
    }

    /// Initialize memory pools for common allocation sizes
    pub fn initialize_pools(&self) -> Result<(), InfrastructureError> {
        let mut pools = self.memory_pools.write();
        
        // Common size classes for trading systems
        let size_classes = vec![64, 128, 256, 512, 1024, 2048, 4096, 8192];
        
        for &size_class in &size_classes {
            let pool = MemoryPool::new(size_class, 1000, 0)?; // 1000 blocks per pool
            pools.insert(size_class, pool);
        }
        
        Ok(())
    }

    pub fn get_stats(&self) -> AllocationStats {
        self.allocation_stats.read().clone()
    }
}

impl MemoryPool {
    pub fn new(size_class: usize, initial_blocks: usize, numa_node: usize) -> Result<Self, InfrastructureError> {
        let mut blocks = Vec::with_capacity(initial_blocks);
        
        // Pre-allocate blocks
        for _ in 0..initial_blocks {
            let layout = Layout::from_size_align(size_class, std::mem::align_of::<u8>())
                .map_err(|e| InfrastructureError::MemoryError(e.to_string()))?;
            
            unsafe {
                let ptr = System.alloc(layout);
                if ptr.is_null() {
                    return Err(InfrastructureError::MemoryError("Failed to allocate memory block".to_string()));
                }
                
                if let Some(non_null_ptr) = NonNull::new(ptr) {
                    blocks.push(non_null_ptr);
                }
            }
        }
        
        Ok(Self {
            size_class,
            blocks: Mutex::new(blocks),
            block_size: size_class,
            total_blocks: AtomicUsize::new(initial_blocks),
            free_blocks: AtomicUsize::new(initial_blocks),
            numa_node,
        })
    }

    pub fn allocate(&self) -> Option<NonNull<u8>> {
        let mut blocks = self.blocks.lock();
        if let Some(ptr) = blocks.pop() {
            self.free_blocks.fetch_sub(1, Ordering::Relaxed);
            Some(ptr)
        } else {
            None
        }
    }

    pub fn deallocate(&self, ptr: NonNull<u8>) {
        let mut blocks = self.blocks.lock();
        blocks.push(ptr);
        self.free_blocks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_utilization(&self) -> f64 {
        let total = self.total_blocks.load(Ordering::Relaxed);
        let free = self.free_blocks.load(Ordering::Relaxed);
        
        if total == 0 {
            0.0
        } else {
            (total - free) as f64 / total as f64
        }
    }
}

impl LockFreeArena {
    pub fn new(size: usize, numa_node: usize) -> Result<Self, InfrastructureError> {
        let layout = Layout::from_size_align(size, 64) // 64-byte alignment for cache lines
            .map_err(|e| InfrastructureError::MemoryError(e.to_string()))?;
        
        unsafe {
            let ptr = System.alloc(layout);
            if ptr.is_null() {
                return Err(InfrastructureError::MemoryError("Failed to allocate arena".to_string()));
            }
            
            let non_null_ptr = NonNull::new(ptr)
                .ok_or_else(|| InfrastructureError::MemoryError("Null pointer in arena".to_string()))?;
            
            Ok(Self {
                memory: non_null_ptr,
                size,
                offset: AtomicUsize::new(0),
                numa_node,
            })
        }
    }

    pub fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let current_offset = self.offset.load(Ordering::Relaxed);
        let aligned_offset = (current_offset + align - 1) & !(align - 1);
        let new_offset = aligned_offset + size;
        
        if new_offset > self.size {
            return None;
        }
        
        // Try to update offset atomically
        match self.offset.compare_exchange_weak(
            current_offset,
            new_offset,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                unsafe {
                    let ptr = self.memory.as_ptr().add(aligned_offset);
                    NonNull::new(ptr)
                }
            }
            Err(_) => {
                // Retry allocation
                self.allocate(size, align)
            }
        }
    }

    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    pub fn usage(&self) -> f64 {
        let used = self.offset.load(Ordering::Relaxed);
        used as f64 / self.size as f64
    }
}

impl Drop for LockFreeArena {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, 64);
            System.dealloc(self.memory.as_ptr(), layout);
        }
    }
}

impl MemoryMappedRegion {
    pub fn new(size: usize, shared: bool) -> Result<Self, InfrastructureError> {
        #[cfg(unix)]
        {
            use std::os::unix::io::RawFd;
            
            let flags = if shared {
                libc::MAP_SHARED | libc::MAP_ANONYMOUS
            } else {
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS
            };
            
            unsafe {
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    flags,
                    -1,
                    0,
                );
                
                if ptr == libc::MAP_FAILED {
                    return Err(InfrastructureError::MemoryError("mmap failed".to_string()));
                }
                
                let non_null_ptr = NonNull::new(ptr as *mut u8)
                    .ok_or_else(|| InfrastructureError::MemoryError("Null pointer from mmap".to_string()))?;
                
                Ok(Self {
                    ptr: non_null_ptr,
                    size,
                    file_descriptor: None,
                    is_shared: shared,
                })
            }
        }
        
        #[cfg(not(unix))]
        {
            // Fallback to regular allocation on non-Unix systems
            let layout = Layout::from_size_align(size, 4096)
                .map_err(|e| InfrastructureError::MemoryError(e.to_string()))?;
            
            unsafe {
                let ptr = System.alloc(layout);
                if ptr.is_null() {
                    return Err(InfrastructureError::MemoryError("Failed to allocate memory".to_string()));
                }
                
                let non_null_ptr = NonNull::new(ptr)
                    .ok_or_else(|| InfrastructureError::MemoryError("Null pointer".to_string()))?;
                
                Ok(Self {
                    ptr: non_null_ptr,
                    size,
                    file_descriptor: None,
                    is_shared: shared,
                })
            }
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
        }
    }
}

impl Drop for MemoryMappedRegion {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size);
        }
        
        #[cfg(not(unix))]
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, 4096);
            System.dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// Global NUMA-aware allocator instance
static NUMA_ALLOCATOR: NumaAwareAllocator = NumaAwareAllocator {
    system_allocator: System,
    numa_nodes: Vec::new(),
    current_node: AtomicUsize::new(0),
    allocation_stats: Arc::new(RwLock::new(AllocationStats {
        total_allocated: 0,
        total_freed: 0,
        peak_usage: 0,
        current_usage: 0,
        allocation_count: 0,
        free_count: 0,
        numa_node_usage: HashMap::new(),
    })),
    memory_pools: Arc::new(RwLock::new(HashMap::new())),
};

/// Initialize the global NUMA allocator
pub fn initialize_numa_allocator() -> Result<(), InfrastructureError> {
    // This would be called at startup to initialize the allocator
    // For now, we'll use a placeholder implementation
    Ok(())
}