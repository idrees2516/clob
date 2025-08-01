use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::topology::{NumaTopologyDetector, SystemTopology};

/// NUMA memory policy types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumaPolicy {
    /// Allocate on local NUMA node
    Local,
    /// Allocate on specific NUMA node
    Bind(u32),
    /// Interleave allocation across specified nodes
    Interleave(Vec<u32>),
    /// Prefer specific node but allow fallback
    Preferred(u32),
}

/// Memory allocation statistics per NUMA node
#[derive(Debug, Clone)]
pub struct NumaAllocationStats {
    pub node_id: u32,
    pub total_allocated: usize,
    pub total_freed: usize,
    pub current_usage: usize,
    pub allocation_count: usize,
    pub free_count: usize,
    pub cross_numa_accesses: usize,
}

/// Hot data migration candidate
#[derive(Debug)]
pub struct MigrationCandidate {
    pub address: *mut u8,
    pub size: usize,
    pub current_node: u32,
    pub target_node: u32,
    pub access_frequency: f64,
    pub migration_benefit: f64,
}

/// NUMA-aware data placement manager
pub struct NumaDataPlacement {
    topology: Arc<SystemTopology>,
    allocation_stats: Arc<Mutex<HashMap<u32, NumaAllocationStats>>>,
    hot_data_tracker: Arc<Mutex<HashMap<*mut u8, HotDataInfo>>>,
    migration_candidates: Arc<Mutex<Vec<MigrationCandidate>>>,
    default_policy: NumaPolicy,
}

/// Hot data tracking information
#[derive(Debug, Clone)]
struct HotDataInfo {
    address: *mut u8,
    size: usize,
    numa_node: u32,
    access_count: usize,
    last_access_cpu: u32,
    access_pattern: Vec<u32>, // Recent accessing CPUs
}

impl NumaDataPlacement {
    /// Create a new NUMA data placement manager
    pub fn new(topology_detector: &NumaTopologyDetector) -> Self {
        let topology = topology_detector.get_topology();
        let mut allocation_stats = HashMap::new();
        
        // Initialize stats for each NUMA node
        for &node_id in topology.numa_nodes.keys() {
            allocation_stats.insert(node_id, NumaAllocationStats {
                node_id,
                total_allocated: 0,
                total_freed: 0,
                current_usage: 0,
                allocation_count: 0,
                free_count: 0,
                cross_numa_accesses: 0,
            });
        }
        
        Self {
            topology,
            allocation_stats: Arc::new(Mutex::new(allocation_stats)),
            hot_data_tracker: Arc::new(Mutex::new(HashMap::new())),
            migration_candidates: Arc::new(Mutex::new(Vec::new())),
            default_policy: NumaPolicy::Local,
        }
    }

    /// Set default NUMA policy
    pub fn set_default_policy(&mut self, policy: NumaPolicy) {
        self.default_policy = policy;
    }

    /// Allocate memory with specific NUMA policy
    pub fn allocate_with_policy(
        &self,
        size: usize,
        alignment: usize,
        policy: NumaPolicy,
    ) -> Result<NonNull<u8>, std::alloc::AllocError> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| std::alloc::AllocError)?;
        
        let numa_node = self.determine_target_node(&policy)?;
        let ptr = self.allocate_on_node(layout, numa_node)?;
        
        // Update allocation statistics
        self.update_allocation_stats(numa_node, size, true);
        
        // Track hot data if size is significant
        if size >= 4096 { // Track allocations >= 4KB
            self.track_hot_data(ptr.as_ptr(), size, numa_node);
        }
        
        Ok(ptr)
    }

    /// Allocate memory on local NUMA node
    pub fn allocate_local(&self, size: usize, alignment: usize) -> Result<NonNull<u8>, std::alloc::AllocError> {
        self.allocate_with_policy(size, alignment, NumaPolicy::Local)
    }

    /// Allocate memory on specific NUMA node
    pub fn allocate_on_node_id(&self, size: usize, alignment: usize, node_id: u32) -> Result<NonNull<u8>, std::alloc::AllocError> {
        self.allocate_with_policy(size, alignment, NumaPolicy::Bind(node_id))
    }

    /// Free NUMA-allocated memory
    pub fn deallocate(&self, ptr: NonNull<u8>, size: usize, alignment: usize) {
        let layout = Layout::from_size_align(size, alignment).unwrap();
        
        // Determine which NUMA node this was allocated on
        let numa_node = self.get_allocation_node(ptr.as_ptr()).unwrap_or(0);
        
        // Update allocation statistics
        self.update_allocation_stats(numa_node, size, false);
        
        // Remove from hot data tracking
        self.untrack_hot_data(ptr.as_ptr());
        
        // Deallocate memory
        unsafe {
            std::alloc::System.dealloc(ptr.as_ptr(), layout);
        }
    }

    /// Determine target NUMA node based on policy
    fn determine_target_node(&self, policy: &NumaPolicy) -> Result<u32, std::alloc::AllocError> {
        match policy {
            NumaPolicy::Local => {
                // Get current CPU and its NUMA node
                let current_cpu = self.get_current_cpu();
                self.topology.cpu_to_node.get(&current_cpu)
                    .copied()
                    .ok_or(std::alloc::AllocError)
            }
            NumaPolicy::Bind(node_id) => {
                if self.topology.numa_nodes.contains_key(node_id) {
                    Ok(*node_id)
                } else {
                    Err(std::alloc::AllocError)
                }
            }
            NumaPolicy::Interleave(nodes) => {
                if nodes.is_empty() {
                    Err(std::alloc::AllocError)
                } else {
                    // Simple round-robin for interleaving
                    static INTERLEAVE_COUNTER: AtomicUsize = AtomicUsize::new(0);
                    let index = INTERLEAVE_COUNTER.fetch_add(1, Ordering::Relaxed) % nodes.len();
                    Ok(nodes[index])
                }
            }
            NumaPolicy::Preferred(node_id) => {
                if self.topology.numa_nodes.contains_key(node_id) {
                    Ok(*node_id)
                } else {
                    // Fallback to local node
                    let current_cpu = self.get_current_cpu();
                    self.topology.cpu_to_node.get(&current_cpu)
                        .copied()
                        .unwrap_or(0)
                        .into()
                }
            }
        }
    }

    /// Allocate memory on specific NUMA node
    fn allocate_on_node(&self, layout: Layout, numa_node: u32) -> Result<NonNull<u8>, std::alloc::AllocError> {
        #[cfg(target_os = "linux")]
        {
            use std::ffi::c_void;
            
            // Use numa_alloc_onnode if available
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    layout.size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            
            if ptr == libc::MAP_FAILED {
                return Err(std::alloc::AllocError);
            }
            
            // Set NUMA policy for this memory region
            self.set_memory_policy(ptr, layout.size(), numa_node)?;
            
            NonNull::new(ptr as *mut u8).ok_or(std::alloc::AllocError)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback to standard allocation
            let ptr = unsafe { std::alloc::System.alloc(layout) };
            NonNull::new(ptr).ok_or(std::alloc::AllocError)
        }
    }

    /// Set memory policy for a memory region
    #[cfg(target_os = "linux")]
    fn set_memory_policy(&self, ptr: *mut libc::c_void, size: usize, numa_node: u32) -> Result<(), std::alloc::AllocError> {
        // This would use mbind() system call to bind memory to specific NUMA node
        // For now, we'll use a placeholder implementation
        Ok(())
    }

    /// Get current CPU ID
    fn get_current_cpu(&self) -> u32 {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                libc::sched_getcpu() as u32
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            0 // Fallback
        }
    }

    /// Update allocation statistics
    fn update_allocation_stats(&self, numa_node: u32, size: usize, is_allocation: bool) {
        let mut stats = self.allocation_stats.lock().unwrap();
        if let Some(node_stats) = stats.get_mut(&numa_node) {
            if is_allocation {
                node_stats.total_allocated += size;
                node_stats.current_usage += size;
                node_stats.allocation_count += 1;
            } else {
                node_stats.total_freed += size;
                node_stats.current_usage = node_stats.current_usage.saturating_sub(size);
                node_stats.free_count += 1;
            }
        }
    }

    /// Track hot data allocation
    fn track_hot_data(&self, ptr: *mut u8, size: usize, numa_node: u32) {
        let hot_data_info = HotDataInfo {
            address: ptr,
            size,
            numa_node,
            access_count: 0,
            last_access_cpu: self.get_current_cpu(),
            access_pattern: Vec::with_capacity(16),
        };
        
        self.hot_data_tracker.lock().unwrap().insert(ptr, hot_data_info);
    }

    /// Remove hot data tracking
    fn untrack_hot_data(&self, ptr: *mut u8) {
        self.hot_data_tracker.lock().unwrap().remove(&ptr);
    }

    /// Record memory access for hot data tracking
    pub fn record_access(&self, ptr: *mut u8) {
        let current_cpu = self.get_current_cpu();
        let mut tracker = self.hot_data_tracker.lock().unwrap();
        
        if let Some(hot_data) = tracker.get_mut(&ptr) {
            hot_data.access_count += 1;
            hot_data.last_access_cpu = current_cpu;
            
            // Update access pattern (keep last 16 accesses)
            hot_data.access_pattern.push(current_cpu);
            if hot_data.access_pattern.len() > 16 {
                hot_data.access_pattern.remove(0);
            }
            
            // Check if this is a cross-NUMA access
            if let Some(&cpu_numa_node) = self.topology.cpu_to_node.get(&current_cpu) {
                if cpu_numa_node != hot_data.numa_node {
                    // Record cross-NUMA access
                    let mut stats = self.allocation_stats.lock().unwrap();
                    if let Some(node_stats) = stats.get_mut(&hot_data.numa_node) {
                        node_stats.cross_numa_accesses += 1;
                    }
                }
            }
        }
    }

    /// Analyze hot data and identify migration candidates
    pub fn analyze_migration_candidates(&self) -> Vec<MigrationCandidate> {
        let tracker = self.hot_data_tracker.lock().unwrap();
        let mut candidates = Vec::new();
        
        for (_, hot_data) in tracker.iter() {
            if hot_data.access_count < 10 {
                continue; // Not enough access data
            }
            
            // Analyze access pattern to determine optimal NUMA node
            let optimal_node = self.find_optimal_numa_node(&hot_data.access_pattern);
            
            if optimal_node != hot_data.numa_node {
                let migration_benefit = self.calculate_migration_benefit(hot_data, optimal_node);
                
                if migration_benefit > 0.2 { // 20% improvement threshold
                    candidates.push(MigrationCandidate {
                        address: hot_data.address,
                        size: hot_data.size,
                        current_node: hot_data.numa_node,
                        target_node: optimal_node,
                        access_frequency: hot_data.access_count as f64,
                        migration_benefit,
                    });
                }
            }
        }
        
        // Sort by migration benefit (highest first)
        candidates.sort_by(|a, b| b.migration_benefit.partial_cmp(&a.migration_benefit).unwrap());
        
        // Update migration candidates
        *self.migration_candidates.lock().unwrap() = candidates.clone();
        
        candidates
    }

    /// Find optimal NUMA node based on access pattern
    fn find_optimal_numa_node(&self, access_pattern: &[u32]) -> u32 {
        let mut node_scores = HashMap::new();
        
        for &cpu in access_pattern {
            if let Some(&numa_node) = self.topology.cpu_to_node.get(&cpu) {
                *node_scores.entry(numa_node).or_insert(0) += 1;
            }
        }
        
        node_scores
            .into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(node, _)| node)
            .unwrap_or(0)
    }

    /// Calculate migration benefit score
    fn calculate_migration_benefit(&self, hot_data: &HotDataInfo, target_node: u32) -> f64 {
        // Calculate current cross-NUMA access ratio
        let cross_numa_accesses = hot_data.access_pattern
            .iter()
            .filter(|&&cpu| {
                self.topology.cpu_to_node.get(&cpu) != Some(&hot_data.numa_node)
            })
            .count();
        
        let current_cross_ratio = cross_numa_accesses as f64 / hot_data.access_pattern.len() as f64;
        
        // Calculate expected cross-NUMA access ratio after migration
        let future_cross_accesses = hot_data.access_pattern
            .iter()
            .filter(|&&cpu| {
                self.topology.cpu_to_node.get(&cpu) != Some(&target_node)
            })
            .count();
        
        let future_cross_ratio = future_cross_accesses as f64 / hot_data.access_pattern.len() as f64;
        
        // Migration benefit is the reduction in cross-NUMA access ratio
        current_cross_ratio - future_cross_ratio
    }

    /// Migrate hot data to optimal NUMA node
    pub fn migrate_data(&self, candidate: &MigrationCandidate) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            // Use move_pages system call to migrate memory
            let pages = candidate.size / self.topology.page_size;
            if pages == 0 {
                return Ok(());
            }
            
            // This would use the move_pages() system call
            // For now, we'll simulate the migration
            println!(
                "Migrating {} bytes from node {} to node {} (benefit: {:.2})",
                candidate.size,
                candidate.current_node,
                candidate.target_node,
                candidate.migration_benefit
            );
            
            // Update hot data tracking
            let mut tracker = self.hot_data_tracker.lock().unwrap();
            if let Some(hot_data) = tracker.get_mut(&candidate.address) {
                hot_data.numa_node = candidate.target_node;
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            println!("Memory migration not supported on this platform");
        }
        
        Ok(())
    }

    /// Get allocation statistics for a NUMA node
    pub fn get_allocation_stats(&self, numa_node: u32) -> Option<NumaAllocationStats> {
        self.allocation_stats.lock().unwrap().get(&numa_node).cloned()
    }

    /// Get allocation statistics for all NUMA nodes
    pub fn get_all_allocation_stats(&self) -> HashMap<u32, NumaAllocationStats> {
        self.allocation_stats.lock().unwrap().clone()
    }

    /// Get current migration candidates
    pub fn get_migration_candidates(&self) -> Vec<MigrationCandidate> {
        self.migration_candidates.lock().unwrap().clone()
    }

    /// Get NUMA node for an allocation
    fn get_allocation_node(&self, ptr: *mut u8) -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            // Use get_mempolicy to determine NUMA node
            // For now, we'll check our hot data tracker
            self.hot_data_tracker.lock().unwrap()
                .get(&ptr)
                .map(|info| info.numa_node)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    /// Optimize data structure placement for specific access patterns
    pub fn optimize_data_structure_placement<T>(&self, data: &mut [T], access_pattern: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
        let optimal_node = self.find_optimal_numa_node(access_pattern);
        
        // Calculate data size
        let data_size = std::mem::size_of_val(data);
        let data_ptr = data.as_mut_ptr() as *mut u8;
        
        // Check if migration would be beneficial
        if let Some(current_node) = self.get_allocation_node(data_ptr) {
            if current_node != optimal_node {
                let hot_data = HotDataInfo {
                    address: data_ptr,
                    size: data_size,
                    numa_node: current_node,
                    access_count: access_pattern.len(),
                    last_access_cpu: *access_pattern.last().unwrap_or(&0),
                    access_pattern: access_pattern.to_vec(),
                };
                
                let migration_benefit = self.calculate_migration_benefit(&hot_data, optimal_node);
                
                if migration_benefit > 0.1 { // 10% improvement threshold
                    let candidate = MigrationCandidate {
                        address: data_ptr,
                        size: data_size,
                        current_node,
                        target_node: optimal_node,
                        access_frequency: access_pattern.len() as f64,
                        migration_benefit,
                    };
                    
                    self.migrate_data(&candidate)?;
                }
            }
        }
        
        Ok(())
    }

    /// Minimize cross-NUMA access for a data structure
    pub fn minimize_cross_numa_access<T>(&self, data: &[T]) -> Vec<u32> {
        let data_ptr = data.as_ptr() as *mut u8;
        let data_size = std::mem::size_of_val(data);
        
        // Record this access
        self.record_access(data_ptr);
        
        // Get optimal CPUs for accessing this data
        if let Some(hot_data) = self.hot_data_tracker.lock().unwrap().get(&data_ptr) {
            // Return CPUs on the same NUMA node as the data
            self.topology.numa_nodes
                .get(&hot_data.numa_node)
                .map(|node| node.cpus.clone())
                .unwrap_or_default()
        } else {
            // Return CPUs on the current NUMA node
            let current_cpu = self.get_current_cpu();
            if let Some(&numa_node) = self.topology.cpu_to_node.get(&current_cpu) {
                self.topology.numa_nodes
                    .get(&numa_node)
                    .map(|node| node.cpus.clone())
                    .unwrap_or_default()
            } else {
                Vec::new()
            }
        }
    }
}

/// NUMA-aware allocator that can be used as a global allocator
pub struct NumaAllocator {
    placement_manager: NumaDataPlacement,
}

impl NumaAllocator {
    pub fn new(topology_detector: &NumaTopologyDetector) -> Self {
        Self {
            placement_manager: NumaDataPlacement::new(topology_detector),
        }
    }
}

unsafe impl GlobalAlloc for NumaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.placement_manager
            .allocate_local(layout.size(), layout.align())
            .map(|ptr| ptr.as_ptr())
            .unwrap_or(std::ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(non_null_ptr) = NonNull::new(ptr) {
            self.placement_manager.deallocate(non_null_ptr, layout.size(), layout.align());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::numa::topology::NumaTopologyDetector;

    #[test]
    fn test_numa_data_placement_creation() {
        let detector = NumaTopologyDetector::new().unwrap();
        let placement = NumaDataPlacement::new(&detector);
        
        let stats = placement.get_all_allocation_stats();
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_migration_benefit_calculation() {
        let detector = NumaTopologyDetector::new().unwrap();
        let placement = NumaDataPlacement::new(&detector);
        
        let hot_data = HotDataInfo {
            address: std::ptr::null_mut(),
            size: 4096,
            numa_node: 0,
            access_count: 100,
            last_access_cpu: 0,
            access_pattern: vec![0, 1, 2, 3, 4, 5, 6, 7], // All different CPUs
        };
        
        let benefit = placement.calculate_migration_benefit(&hot_data, 1);
        assert!(benefit >= 0.0);
    }

    #[test]
    fn test_optimal_numa_node_finding() {
        let detector = NumaTopologyDetector::new().unwrap();
        let placement = NumaDataPlacement::new(&detector);
        
        let access_pattern = vec![0, 0, 1, 1, 2, 2]; // CPUs 0, 1, 2
        let optimal_node = placement.find_optimal_numa_node(&access_pattern);
        
        // Should return a valid NUMA node
        assert!(placement.topology.numa_nodes.contains_key(&optimal_node));
    }
}