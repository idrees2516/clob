use super::lock_free_pool::{LockFreePool, PoolError, PoolStats};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::mem;

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub node_count: usize,
    
    /// CPU to NUMA node mapping
    pub cpu_to_node: HashMap<usize, usize>,
    
    /// NUMA node to CPU list mapping
    pub node_to_cpus: HashMap<usize, Vec<usize>>,
    
    /// Memory size per NUMA node (in bytes)
    pub node_memory_sizes: HashMap<usize, usize>,
    
    /// Distance matrix between NUMA nodes
    pub distance_matrix: Vec<Vec<u32>>,
}

impl NumaTopology {
    /// Detect NUMA topology from the system
    pub fn detect() -> Result<Self, NumaError> {
        // In a real implementation, this would use libnuma or similar
        // For now, we'll create a mock topology
        Self::create_mock_topology()
    }

    /// Create a mock NUMA topology for testing
    fn create_mock_topology() -> Result<Self, NumaError> {
        let node_count = Self::get_numa_node_count();
        let cpu_count = Self::get_cpu_count();
        
        let mut cpu_to_node = HashMap::new();
        let mut node_to_cpus: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut node_memory_sizes = HashMap::new();
        
        // Distribute CPUs evenly across NUMA nodes
        let cpus_per_node = cpu_count / node_count;
        
        for cpu in 0..cpu_count {
            let node = cpu / cpus_per_node.max(1);
            let node = node.min(node_count - 1);
            
            cpu_to_node.insert(cpu, node);
            node_to_cpus.entry(node).or_insert_with(Vec::new).push(cpu);
        }
        
        // Assume 16GB per NUMA node
        for node in 0..node_count {
            node_memory_sizes.insert(node, 16 * 1024 * 1024 * 1024);
        }
        
        // Create distance matrix (simplified)
        let mut distance_matrix = vec![vec![0u32; node_count]; node_count];
        for i in 0..node_count {
            for j in 0..node_count {
                distance_matrix[i][j] = if i == j { 10 } else { 20 };
            }
        }
        
        Ok(Self {
            node_count,
            cpu_to_node,
            node_to_cpus,
            node_memory_sizes,
            distance_matrix,
        })
    }

    /// Get the NUMA node for the current thread
    pub fn get_current_numa_node(&self) -> usize {
        // In a real implementation, this would use getcpu() or similar
        // For now, we'll use thread ID modulo node count
        let thread_id = thread::current().id().as_u64().get() as usize;
        thread_id % self.node_count
    }

    /// Get the NUMA node for a specific CPU
    pub fn get_numa_node_for_cpu(&self, cpu: usize) -> Option<usize> {
        self.cpu_to_node.get(&cpu).copied()
    }

    /// Get CPUs for a specific NUMA node
    pub fn get_cpus_for_node(&self, node: usize) -> Option<&Vec<usize>> {
        self.node_to_cpus.get(&node)
    }

    /// Get memory size for a NUMA node
    pub fn get_node_memory_size(&self, node: usize) -> Option<usize> {
        self.node_memory_sizes.get(&node).copied()
    }

    /// Get distance between two NUMA nodes
    pub fn get_distance(&self, from_node: usize, to_node: usize) -> u32 {
        if from_node < self.node_count && to_node < self.node_count {
            self.distance_matrix[from_node][to_node]
        } else {
            u32::MAX // Invalid nodes
        }
    }

    /// Find the closest NUMA node to a given node
    pub fn find_closest_node(&self, from_node: usize) -> Option<usize> {
        if from_node >= self.node_count {
            return None;
        }

        let mut closest_node = from_node;
        let mut min_distance = u32::MAX;

        for node in 0..self.node_count {
            if node != from_node {
                let distance = self.get_distance(from_node, node);
                if distance < min_distance {
                    min_distance = distance;
                    closest_node = node;
                }
            }
        }

        Some(closest_node)
    }

    /// Get system CPU count
    fn get_cpu_count() -> usize {
        num_cpus::get()
    }

    /// Get NUMA node count
    fn get_numa_node_count() -> usize {
        // In a real implementation, this would query the system
        // For now, assume 2 NUMA nodes for most systems
        (Self::get_cpu_count() / 8).max(1).min(4)
    }
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    /// NUMA topology information
    topology: NumaTopology,
    
    /// Per-NUMA-node memory pools
    node_pools: HashMap<usize, HashMap<usize, Arc<LockFreePool<u8>>>>,
    
    /// Allocation statistics per node
    node_stats: HashMap<usize, NumaNodeStats>,
    
    /// Global allocation policy
    allocation_policy: AllocationPolicy,
    
    /// Fallback enabled for cross-NUMA allocation
    enable_fallback: bool,
}

/// Statistics for a NUMA node
#[derive(Debug, Clone)]
pub struct NumaNodeStats {
    /// Total allocations on this node
    pub total_allocations: AtomicUsize,
    
    /// Total deallocations on this node
    pub total_deallocations: AtomicUsize,
    
    /// Cross-NUMA allocations (fallback)
    pub cross_numa_allocations: AtomicUsize,
    
    /// Failed allocations
    pub failed_allocations: AtomicUsize,
    
    /// Average allocation latency (nanoseconds)
    pub avg_allocation_latency_ns: AtomicUsize,
}

/// NUMA allocation policies
#[derive(Debug, Clone, Copy)]
pub enum AllocationPolicy {
    /// Prefer local NUMA node, fallback to closest
    LocalPreferred,
    
    /// Strict local allocation only
    LocalOnly,
    
    /// Round-robin across all nodes
    RoundRobin,
    
    /// Interleaved allocation across nodes
    Interleaved,
}

impl NumaAllocator {
    /// Create a new NUMA-aware allocator
    pub fn new(policy: AllocationPolicy) -> Result<Self, NumaError> {
        let topology = NumaTopology::detect()?;
        let mut node_pools = HashMap::new();
        let mut node_stats = HashMap::new();

        // Create pools for each NUMA node and size class
        for node in 0..topology.node_count {
            let mut size_pools = HashMap::new();
            
            // Create pools for common allocation sizes
            let size_classes = vec![8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
            
            for &size in &size_classes {
                let initial_capacity = Self::calculate_initial_capacity(size);
                let pool = Arc::new(LockFreePool::new(initial_capacity, node as u32)?);
                size_pools.insert(size, pool);
            }
            
            node_pools.insert(node, size_pools);
            node_stats.insert(node, NumaNodeStats::new());
        }

        Ok(Self {
            topology,
            node_pools,
            node_stats,
            allocation_policy: policy,
            enable_fallback: true,
        })
    }

    /// Allocate memory with NUMA awareness
    pub fn allocate(&self, size: usize) -> Result<NumaAllocation, NumaError> {
        let start_time = self.get_timestamp_ns();
        let preferred_node = self.get_preferred_node(size);
        let size_class = self.get_size_class(size);

        // Try allocation on preferred node first
        if let Some(allocation) = self.try_allocate_on_node(preferred_node, size_class)? {
            self.update_stats(preferred_node, start_time, false);
            return Ok(allocation);
        }

        // Fallback to other nodes if enabled
        if self.enable_fallback {
            for node in 0..self.topology.node_count {
                if node != preferred_node {
                    if let Some(allocation) = self.try_allocate_on_node(node, size_class)? {
                        self.update_stats(node, start_time, true);
                        return Ok(allocation);
                    }
                }
            }
        }

        // All allocations failed
        if let Some(stats) = self.node_stats.get(&preferred_node) {
            stats.failed_allocations.fetch_add(1, Ordering::Relaxed);
        }

        Err(NumaError::AllocationFailed)
    }

    /// Allocate typed object with NUMA awareness
    pub fn allocate_typed<T>(&self) -> Result<NumaTypedAllocation<T>, NumaError> {
        let size = mem::size_of::<T>();
        let allocation = self.allocate(size)?;
        
        Ok(NumaTypedAllocation {
            allocation,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Try to allocate on a specific NUMA node
    fn try_allocate_on_node(
        &self,
        node: usize,
        size_class: usize,
    ) -> Result<Option<NumaAllocation>, NumaError> {
        if let Some(node_pools) = self.node_pools.get(&node) {
            if let Some(pool) = node_pools.get(&size_class) {
                match pool.allocate() {
                    Ok(pooled_obj) => {
                        return Ok(Some(NumaAllocation {
                            data: pooled_obj,
                            numa_node: node,
                            size: size_class,
                        }));
                    }
                    Err(PoolError::PoolExhausted) => {
                        // Pool exhausted, try next node
                        return Ok(None);
                    }
                    Err(e) => {
                        return Err(NumaError::PoolError(e));
                    }
                }
            }
        }
        
        Ok(None)
    }

    /// Get preferred NUMA node based on allocation policy
    fn get_preferred_node(&self, _size: usize) -> usize {
        match self.allocation_policy {
            AllocationPolicy::LocalPreferred | AllocationPolicy::LocalOnly => {
                self.topology.get_current_numa_node()
            }
            AllocationPolicy::RoundRobin => {
                // Simple round-robin based on thread ID
                let thread_id = thread::current().id().as_u64().get() as usize;
                thread_id % self.topology.node_count
            }
            AllocationPolicy::Interleaved => {
                // Interleave based on allocation count
                let total_allocations: usize = self.node_stats
                    .values()
                    .map(|stats| stats.total_allocations.load(Ordering::Relaxed))
                    .sum();
                total_allocations % self.topology.node_count
            }
        }
    }

    /// Get size class for allocation
    fn get_size_class(&self, size: usize) -> usize {
        // Round up to next power of 2, minimum 8 bytes
        let size = size.max(8);
        let size_class = size.next_power_of_two();
        size_class.min(4096) // Maximum size class
    }

    /// Calculate initial capacity for a pool
    fn calculate_initial_capacity(size: usize) -> usize {
        // Smaller objects get larger pools
        match size {
            8..=32 => 1000,
            33..=128 => 500,
            129..=512 => 200,
            513..=2048 => 100,
            _ => 50,
        }
    }

    /// Update allocation statistics
    fn update_stats(&self, node: usize, start_time: u64, cross_numa: bool) {
        if let Some(stats) = self.node_stats.get(&node) {
            stats.total_allocations.fetch_add(1, Ordering::Relaxed);
            
            if cross_numa {
                stats.cross_numa_allocations.fetch_add(1, Ordering::Relaxed);
            }
            
            let latency = self.get_timestamp_ns() - start_time;
            self.update_avg_latency(&stats.avg_allocation_latency_ns, latency);
        }
    }

    /// Update average latency using exponential moving average
    fn update_avg_latency(&self, avg_atomic: &AtomicUsize, new_latency: u64) {
        let alpha = 0.1; // Smoothing factor
        loop {
            let current_avg = avg_atomic.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                new_latency as usize
            } else {
                ((1.0 - alpha) * current_avg as f64 + alpha * new_latency as f64) as usize
            };

            match avg_atomic.compare_exchange_weak(
                current_avg,
                new_avg,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns(&self) -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    /// Get NUMA topology information
    pub fn get_topology(&self) -> &NumaTopology {
        &self.topology
    }

    /// Get statistics for all NUMA nodes
    pub fn get_all_stats(&self) -> HashMap<usize, NumaNodeStatsSnapshot> {
        let mut stats = HashMap::new();
        
        for (&node, node_stats) in &self.node_stats {
            stats.insert(node, NumaNodeStatsSnapshot {
                node,
                total_allocations: node_stats.total_allocations.load(Ordering::Acquire),
                total_deallocations: node_stats.total_deallocations.load(Ordering::Acquire),
                cross_numa_allocations: node_stats.cross_numa_allocations.load(Ordering::Acquire),
                failed_allocations: node_stats.failed_allocations.load(Ordering::Acquire),
                avg_allocation_latency_ns: node_stats.avg_allocation_latency_ns.load(Ordering::Acquire),
            });
        }
        
        stats
    }

    /// Get pool statistics for a specific node and size class
    pub fn get_pool_stats(&self, node: usize, size_class: usize) -> Option<PoolStats> {
        self.node_pools
            .get(&node)?
            .get(&size_class)?
            .get_stats()
            .into()
    }

    /// Set fallback allocation policy
    pub fn set_fallback_enabled(&mut self, enabled: bool) {
        self.enable_fallback = enabled;
    }

    /// Get current allocation policy
    pub fn get_allocation_policy(&self) -> AllocationPolicy {
        self.allocation_policy
    }

    /// Set allocation policy
    pub fn set_allocation_policy(&mut self, policy: AllocationPolicy) {
        self.allocation_policy = policy;
    }
}

/// NUMA allocation wrapper
pub struct NumaAllocation {
    data: super::lock_free_pool::PooledObject<u8>,
    numa_node: usize,
    size: usize,
}

impl NumaAllocation {
    /// Get the NUMA node this allocation is on
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Get the size of this allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get a raw pointer to the data
    pub fn as_ptr(&self) -> *mut u8 {
        self.data.as_ref() as *const u8 as *mut u8
    }

    /// Get a slice view of the data
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size) }
    }

    /// Get a mutable slice view of the data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr(), self.size) }
    }
}

/// Typed NUMA allocation wrapper
pub struct NumaTypedAllocation<T> {
    allocation: NumaAllocation,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> NumaTypedAllocation<T> {
    /// Get the NUMA node this allocation is on
    pub fn numa_node(&self) -> usize {
        self.allocation.numa_node()
    }

    /// Get a reference to the typed data
    pub fn as_ref(&self) -> &T {
        unsafe { &*(self.allocation.as_ptr() as *const T) }
    }

    /// Get a mutable reference to the typed data
    pub fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *(self.allocation.as_ptr() as *mut T) }
    }

    /// Convert to owned value
    pub fn into_inner(self) -> T {
        unsafe { std::ptr::read(self.allocation.as_ptr() as *const T) }
    }
}

impl<T> std::ops::Deref for NumaTypedAllocation<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> std::ops::DerefMut for NumaTypedAllocation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

/// Statistics snapshot for a NUMA node
#[derive(Debug, Clone)]
pub struct NumaNodeStatsSnapshot {
    pub node: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub cross_numa_allocations: usize,
    pub failed_allocations: usize,
    pub avg_allocation_latency_ns: usize,
}

impl NumaNodeStats {
    fn new() -> Self {
        Self {
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
            cross_numa_allocations: AtomicUsize::new(0),
            failed_allocations: AtomicUsize::new(0),
            avg_allocation_latency_ns: AtomicUsize::new(0),
        }
    }
}

/// NUMA-related errors
#[derive(Debug, Clone)]
pub enum NumaError {
    TopologyDetectionFailed,
    AllocationFailed,
    InvalidNode(usize),
    PoolError(PoolError),
}

impl std::fmt::Display for NumaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumaError::TopologyDetectionFailed => write!(f, "Failed to detect NUMA topology"),
            NumaError::AllocationFailed => write!(f, "NUMA allocation failed"),
            NumaError::InvalidNode(node) => write!(f, "Invalid NUMA node: {}", node),
            NumaError::PoolError(e) => write!(f, "Pool error: {}", e),
        }
    }
}

impl std::error::Error for NumaError {}

impl From<PoolError> for NumaError {
    fn from(error: PoolError) -> Self {
        NumaError::PoolError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect().unwrap();
        
        assert!(topology.node_count > 0);
        assert!(!topology.cpu_to_node.is_empty());
        assert!(!topology.node_to_cpus.is_empty());
        assert_eq!(topology.distance_matrix.len(), topology.node_count);
    }

    #[test]
    fn test_numa_allocator_creation() {
        let allocator = NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap();
        
        assert!(allocator.topology.node_count > 0);
        assert!(!allocator.node_pools.is_empty());
        assert_eq!(allocator.allocation_policy as u8, AllocationPolicy::LocalPreferred as u8);
    }

    #[test]
    fn test_basic_allocation() {
        let allocator = NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let allocation = allocator.allocate(64).unwrap();
        assert_eq!(allocation.size(), 64);
        assert!(allocation.numa_node() < allocator.topology.node_count);
    }

    #[test]
    fn test_typed_allocation() {
        let allocator = NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap();
        
        let mut allocation: NumaTypedAllocation<u64> = allocator.allocate_typed().unwrap();
        *allocation = 42;
        
        assert_eq!(*allocation, 42);
        assert!(allocation.numa_node() < allocator.topology.node_count);
    }

    #[test]
    fn test_allocation_policies() {
        let policies = vec![
            AllocationPolicy::LocalPreferred,
            AllocationPolicy::LocalOnly,
            AllocationPolicy::RoundRobin,
            AllocationPolicy::Interleaved,
        ];

        for policy in policies {
            let allocator = NumaAllocator::new(policy).unwrap();
            let allocation = allocator.allocate(32).unwrap();
            assert!(allocation.numa_node() < allocator.topology.node_count);
        }
    }

    #[test]
    fn test_concurrent_allocation() {
        let allocator = Arc::new(NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap());
        let mut handles = vec![];

        for _ in 0..4 {
            let allocator_clone = allocator.clone();
            let handle = thread::spawn(move || {
                let mut allocations = vec![];
                
                for _ in 0..100 {
                    let allocation = allocator_clone.allocate(128).unwrap();
                    allocations.push(allocation);
                }
                
                allocations.len()
            });
            handles.push(handle);
        }

        let mut total_allocations = 0;
        for handle in handles {
            total_allocations += handle.join().unwrap();
        }

        assert_eq!(total_allocations, 400);
    }

    #[test]
    fn test_statistics_collection() {
        let allocator = NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap();
        
        // Perform some allocations
        for _ in 0..10 {
            let _allocation = allocator.allocate(64).unwrap();
        }

        let stats = allocator.get_all_stats();
        assert!(!stats.is_empty());
        
        let total_allocations: usize = stats.values()
            .map(|s| s.total_allocations)
            .sum();
        assert_eq!(total_allocations, 10);
    }

    #[test]
    fn test_size_class_calculation() {
        let allocator = NumaAllocator::new(AllocationPolicy::LocalPreferred).unwrap();
        
        assert_eq!(allocator.get_size_class(1), 8);
        assert_eq!(allocator.get_size_class(8), 8);
        assert_eq!(allocator.get_size_class(9), 16);
        assert_eq!(allocator.get_size_class(16), 16);
        assert_eq!(allocator.get_size_class(17), 32);
        assert_eq!(allocator.get_size_class(1000), 1024);
        assert_eq!(allocator.get_size_class(5000), 4096); // Capped at max
    }

    #[test]
    fn test_numa_node_distance() {
        let topology = NumaTopology::detect().unwrap();
        
        for i in 0..topology.node_count {
            // Distance to self should be minimum
            assert_eq!(topology.get_distance(i, i), 10);
            
            // Distance to other nodes should be higher
            for j in 0..topology.node_count {
                if i != j {
                    assert!(topology.get_distance(i, j) > 10);
                }
            }
        }
    }

    #[test]
    fn test_fallback_allocation() {
        let mut allocator = NumaAllocator::new(AllocationPolicy::LocalOnly).unwrap();
        
        // Disable fallback
        allocator.set_fallback_enabled(false);
        
        // Enable fallback
        allocator.set_fallback_enabled(true);
        
        // Should still be able to allocate
        let allocation = allocator.allocate(64).unwrap();
        assert!(allocation.numa_node() < allocator.topology.node_count);
    }
}