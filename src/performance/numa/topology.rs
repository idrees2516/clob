use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Represents a NUMA node in the system
#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpus: Vec<u32>,
    pub memory_size: usize,
    pub memory_free: usize,
    pub distance_map: HashMap<u32, u32>, // Distance to other NUMA nodes
}

/// CPU information including NUMA affinity
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub cpu_id: u32,
    pub numa_node: u32,
    pub core_id: u32,
    pub thread_id: u32,
    pub frequency: u64,
    pub cache_size_l1: usize,
    pub cache_size_l2: usize,
    pub cache_size_l3: usize,
}

/// Memory bandwidth and latency measurements
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub local_bandwidth_gbps: f64,
    pub local_latency_ns: f64,
    pub remote_bandwidth_gbps: HashMap<u32, f64>, // Per remote NUMA node
    pub remote_latency_ns: HashMap<u32, f64>,     // Per remote NUMA node
}

/// System topology information
#[derive(Debug, Clone)]
pub struct SystemTopology {
    pub numa_nodes: HashMap<u32, NumaNode>,
    pub cpus: HashMap<u32, CpuInfo>,
    pub cpu_to_node: HashMap<u32, u32>,
    pub memory_metrics: HashMap<u32, MemoryMetrics>,
    pub total_memory: usize,
    pub page_size: usize,
    pub huge_page_size: usize,
}

/// NUMA topology detector and manager
pub struct NumaTopologyDetector {
    topology: Arc<SystemTopology>,
    monitoring_active: AtomicBool,
    last_update: AtomicU64,
}

impl NumaTopologyDetector {
    /// Create a new NUMA topology detector
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let topology = Self::detect_topology()?;
        
        Ok(Self {
            topology: Arc::new(topology),
            monitoring_active: AtomicBool::new(false),
            last_update: AtomicU64::new(0),
        })
    }

    /// Get the current system topology
    pub fn get_topology(&self) -> Arc<SystemTopology> {
        self.topology.clone()
    }

    /// Detect the complete system topology
    fn detect_topology() -> Result<SystemTopology, Box<dyn std::error::Error>> {
        let numa_nodes = Self::detect_numa_nodes()?;
        let cpus = Self::detect_cpu_info()?;
        let cpu_to_node = Self::build_cpu_to_node_mapping(&numa_nodes, &cpus);
        let memory_metrics = Self::measure_memory_performance(&numa_nodes)?;
        
        let page_size = Self::get_page_size()?;
        let huge_page_size = Self::get_huge_page_size()?;
        let total_memory = numa_nodes.values().map(|n| n.memory_size).sum();

        Ok(SystemTopology {
            numa_nodes,
            cpus,
            cpu_to_node,
            memory_metrics,
            total_memory,
            page_size,
            huge_page_size,
        })
    }

    /// Detect NUMA nodes from /sys/devices/system/node/
    fn detect_numa_nodes() -> Result<HashMap<u32, NumaNode>, Box<dyn std::error::Error>> {
        let mut numa_nodes = HashMap::new();
        let node_path = Path::new("/sys/devices/system/node");
        
        if !node_path.exists() {
            // Fallback for systems without NUMA support
            let cpus = Self::get_online_cpus()?;
            numa_nodes.insert(0, NumaNode {
                node_id: 0,
                cpus,
                memory_size: Self::get_total_memory()?,
                memory_free: Self::get_free_memory()?,
                distance_map: HashMap::new(),
            });
            return Ok(numa_nodes);
        }

        for entry in fs::read_dir(node_path)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            
            if name_str.starts_with("node") {
                if let Ok(node_id) = name_str[4..].parse::<u32>() {
                    let node_cpus = Self::get_node_cpus(node_id)?;
                    let memory_size = Self::get_node_memory_size(node_id)?;
                    let memory_free = Self::get_node_memory_free(node_id)?;
                    let distance_map = Self::get_node_distances(node_id)?;
                    
                    numa_nodes.insert(node_id, NumaNode {
                        node_id,
                        cpus: node_cpus,
                        memory_size,
                        memory_free,
                        distance_map,
                    });
                }
            }
        }

        Ok(numa_nodes)
    }

    /// Get CPUs belonging to a specific NUMA node
    fn get_node_cpus(node_id: u32) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let cpulist_path = format!("/sys/devices/system/node/node{}/cpulist", node_id);
        let cpulist = fs::read_to_string(cpulist_path)?;
        Self::parse_cpu_list(&cpulist.trim())
    }

    /// Parse CPU list format (e.g., "0-3,8-11")
    fn parse_cpu_list(cpulist: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut cpus = Vec::new();
        
        for range in cpulist.split(',') {
            if range.contains('-') {
                let parts: Vec<&str> = range.split('-').collect();
                if parts.len() == 2 {
                    let start: u32 = parts[0].parse()?;
                    let end: u32 = parts[1].parse()?;
                    for cpu in start..=end {
                        cpus.push(cpu);
                    }
                }
            } else {
                cpus.push(range.parse()?);
            }
        }
        
        Ok(cpus)
    }

    /// Get memory size for a NUMA node
    fn get_node_memory_size(node_id: u32) -> Result<usize, Box<dyn std::error::Error>> {
        let meminfo_path = format!("/sys/devices/system/node/node{}/meminfo", node_id);
        let meminfo = fs::read_to_string(meminfo_path)?;
        
        for line in meminfo.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let size_kb: usize = parts[3].parse()?;
                    return Ok(size_kb * 1024); // Convert KB to bytes
                }
            }
        }
        
        Ok(0)
    }

    /// Get free memory for a NUMA node
    fn get_node_memory_free(node_id: u32) -> Result<usize, Box<dyn std::error::Error>> {
        let meminfo_path = format!("/sys/devices/system/node/node{}/meminfo", node_id);
        let meminfo = fs::read_to_string(meminfo_path)?;
        
        for line in meminfo.lines() {
            if line.contains("MemFree:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let size_kb: usize = parts[3].parse()?;
                    return Ok(size_kb * 1024); // Convert KB to bytes
                }
            }
        }
        
        Ok(0)
    }

    /// Get NUMA node distances
    fn get_node_distances(node_id: u32) -> Result<HashMap<u32, u32>, Box<dyn std::error::Error>> {
        let distance_path = format!("/sys/devices/system/node/node{}/distance", node_id);
        let mut distance_map = HashMap::new();
        
        if let Ok(distances) = fs::read_to_string(distance_path) {
            let distances: Vec<u32> = distances
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            
            for (target_node, &distance) in distances.iter().enumerate() {
                distance_map.insert(target_node as u32, distance);
            }
        }
        
        Ok(distance_map)
    }

    /// Detect detailed CPU information
    fn detect_cpu_info() -> Result<HashMap<u32, CpuInfo>, Box<dyn std::error::Error>> {
        let mut cpus = HashMap::new();
        let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
        
        let mut current_cpu = None;
        let mut processor_id = 0;
        let mut physical_id = 0;
        let mut core_id = 0;
        
        for line in cpuinfo.lines() {
            if line.starts_with("processor") {
                if let Some(cpu_id) = current_cpu {
                    // Save previous CPU info
                    let numa_node = Self::get_cpu_numa_node(cpu_id).unwrap_or(0);
                    let frequency = Self::get_cpu_frequency(cpu_id).unwrap_or(0);
                    let (l1, l2, l3) = Self::get_cpu_cache_sizes(cpu_id);
                    
                    cpus.insert(cpu_id, CpuInfo {
                        cpu_id,
                        numa_node,
                        core_id: core_id as u32,
                        thread_id: (cpu_id - core_id as u32),
                        frequency,
                        cache_size_l1: l1,
                        cache_size_l2: l2,
                        cache_size_l3: l3,
                    });
                }
                
                // Parse new processor
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() == 2 {
                    processor_id = parts[1].trim().parse().unwrap_or(0);
                    current_cpu = Some(processor_id);
                }
            } else if line.starts_with("physical id") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() == 2 {
                    physical_id = parts[1].trim().parse().unwrap_or(0);
                }
            } else if line.starts_with("core id") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() == 2 {
                    core_id = parts[1].trim().parse().unwrap_or(0);
                }
            }
        }
        
        // Handle last CPU
        if let Some(cpu_id) = current_cpu {
            let numa_node = Self::get_cpu_numa_node(cpu_id).unwrap_or(0);
            let frequency = Self::get_cpu_frequency(cpu_id).unwrap_or(0);
            let (l1, l2, l3) = Self::get_cpu_cache_sizes(cpu_id);
            
            cpus.insert(cpu_id, CpuInfo {
                cpu_id,
                numa_node,
                core_id: core_id as u32,
                thread_id: (cpu_id - core_id as u32),
                frequency,
                cache_size_l1: l1,
                cache_size_l2: l2,
                cache_size_l3: l3,
            });
        }
        
        Ok(cpus)
    }

    /// Get NUMA node for a specific CPU
    fn get_cpu_numa_node(cpu_id: u32) -> Result<u32, Box<dyn std::error::Error>> {
        let numa_path = format!("/sys/devices/system/cpu/cpu{}/node", cpu_id);
        if let Ok(node_str) = fs::read_to_string(numa_path) {
            Ok(node_str.trim().parse()?)
        } else {
            Ok(0) // Default to node 0 if not found
        }
    }

    /// Get CPU frequency
    fn get_cpu_frequency(cpu_id: u32) -> Result<u64, Box<dyn std::error::Error>> {
        let freq_path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", cpu_id);
        if let Ok(freq_str) = fs::read_to_string(freq_path) {
            Ok(freq_str.trim().parse::<u64>()? * 1000) // Convert kHz to Hz
        } else {
            // Fallback to base frequency
            let base_freq_path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/base_frequency", cpu_id);
            if let Ok(freq_str) = fs::read_to_string(base_freq_path) {
                Ok(freq_str.trim().parse::<u64>()? * 1000)
            } else {
                Ok(0)
            }
        }
    }

    /// Get CPU cache sizes (L1, L2, L3)
    fn get_cpu_cache_sizes(cpu_id: u32) -> (usize, usize, usize) {
        let mut l1 = 0;
        let mut l2 = 0;
        let mut l3 = 0;
        
        // Try to read cache information
        for level in 0..4 {
            let size_path = format!("/sys/devices/system/cpu/cpu{}/cache/index{}/size", cpu_id, level);
            let level_path = format!("/sys/devices/system/cpu/cpu{}/cache/index{}/level", cpu_id, level);
            
            if let (Ok(size_str), Ok(level_str)) = (fs::read_to_string(size_path), fs::read_to_string(level_path)) {
                if let (Ok(cache_level), Ok(size)) = (level_str.trim().parse::<u32>(), Self::parse_cache_size(&size_str.trim())) {
                    match cache_level {
                        1 => l1 = size,
                        2 => l2 = size,
                        3 => l3 = size,
                        _ => {}
                    }
                }
            }
        }
        
        (l1, l2, l3)
    }

    /// Parse cache size string (e.g., "32K", "256K", "8192K")
    fn parse_cache_size(size_str: &str) -> Result<usize, Box<dyn std::error::Error>> {
        let size_str = size_str.to_uppercase();
        if size_str.ends_with('K') {
            let num: usize = size_str[..size_str.len()-1].parse()?;
            Ok(num * 1024)
        } else if size_str.ends_with('M') {
            let num: usize = size_str[..size_str.len()-1].parse()?;
            Ok(num * 1024 * 1024)
        } else {
            Ok(size_str.parse()?)
        }
    }

    /// Build CPU to NUMA node mapping
    fn build_cpu_to_node_mapping(
        numa_nodes: &HashMap<u32, NumaNode>,
        cpus: &HashMap<u32, CpuInfo>
    ) -> HashMap<u32, u32> {
        let mut cpu_to_node = HashMap::new();
        
        for (_, cpu_info) in cpus {
            cpu_to_node.insert(cpu_info.cpu_id, cpu_info.numa_node);
        }
        
        // Fallback: use NUMA node CPU lists
        for (node_id, node) in numa_nodes {
            for &cpu_id in &node.cpus {
                cpu_to_node.entry(cpu_id).or_insert(*node_id);
            }
        }
        
        cpu_to_node
    }

    /// Measure memory performance for each NUMA node
    fn measure_memory_performance(
        numa_nodes: &HashMap<u32, NumaNode>
    ) -> Result<HashMap<u32, MemoryMetrics>, Box<dyn std::error::Error>> {
        let mut memory_metrics = HashMap::new();
        
        for (&node_id, _node) in numa_nodes {
            let local_bandwidth = Self::measure_local_bandwidth(node_id)?;
            let local_latency = Self::measure_local_latency(node_id)?;
            
            let mut remote_bandwidth = HashMap::new();
            let mut remote_latency = HashMap::new();
            
            // Measure remote access performance
            for (&remote_node_id, _) in numa_nodes {
                if remote_node_id != node_id {
                    let bandwidth = Self::measure_remote_bandwidth(node_id, remote_node_id)?;
                    let latency = Self::measure_remote_latency(node_id, remote_node_id)?;
                    remote_bandwidth.insert(remote_node_id, bandwidth);
                    remote_latency.insert(remote_node_id, latency);
                }
            }
            
            memory_metrics.insert(node_id, MemoryMetrics {
                local_bandwidth_gbps: local_bandwidth,
                local_latency_ns: local_latency,
                remote_bandwidth_gbps: remote_bandwidth,
                remote_latency_ns: remote_latency,
            });
        }
        
        Ok(memory_metrics)
    }

    /// Measure local memory bandwidth for a NUMA node
    fn measure_local_bandwidth(node_id: u32) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified bandwidth measurement
        // In a real implementation, this would use specialized benchmarking
        const BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64MB
        const ITERATIONS: usize = 100;
        
        let buffer = vec![0u8; BUFFER_SIZE];
        let start = Instant::now();
        
        for _ in 0..ITERATIONS {
            // Simulate memory access pattern
            let _sum: u64 = buffer.iter().map(|&x| x as u64).sum();
        }
        
        let elapsed = start.elapsed();
        let bytes_processed = (BUFFER_SIZE * ITERATIONS) as f64;
        let bandwidth_bps = bytes_processed / elapsed.as_secs_f64();
        let bandwidth_gbps = bandwidth_bps / (1024.0 * 1024.0 * 1024.0);
        
        Ok(bandwidth_gbps)
    }

    /// Measure local memory latency for a NUMA node
    fn measure_local_latency(node_id: u32) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified latency measurement
        const ITERATIONS: usize = 10000;
        
        let buffer = vec![0u64; 1024];
        let start = Instant::now();
        
        for i in 0..ITERATIONS {
            // Random memory access to measure latency
            let index = (i * 7) % buffer.len();
            let _value = unsafe { std::ptr::read_volatile(&buffer[index]) };
        }
        
        let elapsed = start.elapsed();
        let avg_latency_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
        
        Ok(avg_latency_ns)
    }

    /// Measure remote memory bandwidth between NUMA nodes
    fn measure_remote_bandwidth(local_node: u32, remote_node: u32) -> Result<f64, Box<dyn std::error::Error>> {
        // Placeholder - would require actual cross-NUMA memory allocation
        // For now, estimate based on typical NUMA penalties
        let local_bandwidth = Self::measure_local_bandwidth(local_node)?;
        Ok(local_bandwidth * 0.6) // Typical 40% penalty for remote access
    }

    /// Measure remote memory latency between NUMA nodes
    fn measure_remote_latency(local_node: u32, remote_node: u32) -> Result<f64, Box<dyn std::error::Error>> {
        // Placeholder - would require actual cross-NUMA memory allocation
        let local_latency = Self::measure_local_latency(local_node)?;
        Ok(local_latency * 2.0) // Typical 2x penalty for remote access
    }

    /// Get list of online CPUs
    fn get_online_cpus() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let online_cpus = fs::read_to_string("/sys/devices/system/cpu/online")?;
        Self::parse_cpu_list(online_cpus.trim())
    }

    /// Get total system memory
    fn get_total_memory() -> Result<usize, Box<dyn std::error::Error>> {
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let size_kb: usize = parts[1].parse()?;
                    return Ok(size_kb * 1024);
                }
            }
        }
        Ok(0)
    }

    /// Get free system memory
    fn get_free_memory() -> Result<usize, Box<dyn std::error::Error>> {
        let meminfo = fs::read_to_string("/proc/meminfo")?;
        for line in meminfo.lines() {
            if line.starts_with("MemFree:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let size_kb: usize = parts[1].parse()?;
                    return Ok(size_kb * 1024);
                }
            }
        }
        Ok(0)
    }

    /// Get system page size
    fn get_page_size() -> Result<usize, Box<dyn std::error::Error>> {
        unsafe {
            let page_size = libc::sysconf(libc::_SC_PAGESIZE);
            if page_size > 0 {
                Ok(page_size as usize)
            } else {
                Ok(4096) // Default 4KB pages
            }
        }
    }

    /// Get huge page size
    fn get_huge_page_size() -> Result<usize, Box<dyn std::error::Error>> {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("Hugepagesize:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let size_kb: usize = parts[1].parse()?;
                        return Ok(size_kb * 1024);
                    }
                }
            }
        }
        Ok(2 * 1024 * 1024) // Default 2MB huge pages
    }

    /// Start monitoring topology changes
    pub fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.monitoring_active.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_err() {
            return Ok((); // Already monitoring
        }

        let topology_clone = self.topology.clone();
        let monitoring_active = self.monitoring_active.clone();
        let last_update = self.last_update.clone();

        thread::spawn(move || {
            while monitoring_active.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(30)); // Check every 30 seconds
                
                // Check for topology changes
                if let Ok(new_topology) = Self::detect_topology() {
                    // Compare with current topology
                    if Self::topology_changed(&topology_clone, &new_topology) {
                        println!("NUMA topology change detected!");
                        last_update.store(
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            Ordering::Relaxed
                        );
                        // In a real implementation, we would update the topology
                        // and notify subscribers
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop monitoring topology changes
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::Relaxed);
    }

    /// Check if topology has changed
    fn topology_changed(old: &SystemTopology, new: &SystemTopology) -> bool {
        // Simple comparison - in practice, this would be more sophisticated
        old.numa_nodes.len() != new.numa_nodes.len() ||
        old.cpus.len() != new.cpus.len() ||
        old.total_memory != new.total_memory
    }

    /// Get the best NUMA node for a given CPU
    pub fn get_best_numa_node_for_cpu(&self, cpu_id: u32) -> Option<u32> {
        self.topology.cpu_to_node.get(&cpu_id).copied()
    }

    /// Get CPUs on a specific NUMA node
    pub fn get_cpus_on_node(&self, node_id: u32) -> Vec<u32> {
        self.topology.numa_nodes
            .get(&node_id)
            .map(|node| node.cpus.clone())
            .unwrap_or_default()
    }

    /// Get memory metrics for a NUMA node
    pub fn get_memory_metrics(&self, node_id: u32) -> Option<&MemoryMetrics> {
        self.topology.memory_metrics.get(&node_id)
    }

    /// Find the closest NUMA node to a given node
    pub fn find_closest_node(&self, node_id: u32) -> Option<u32> {
        self.topology.numa_nodes.get(&node_id)?
            .distance_map
            .iter()
            .filter(|(&target, _)| target != node_id)
            .min_by_key(|(_, &distance)| distance)
            .map(|(&target, _)| target)
    }
}

impl Default for NumaTopologyDetector {
    fn default() -> Self {
        Self::new().expect("Failed to detect NUMA topology")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_list_parsing() {
        assert_eq!(
            NumaTopologyDetector::parse_cpu_list("0-3").unwrap(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            NumaTopologyDetector::parse_cpu_list("0,2,4").unwrap(),
            vec![0, 2, 4]
        );
        assert_eq!(
            NumaTopologyDetector::parse_cpu_list("0-1,4-5").unwrap(),
            vec![0, 1, 4, 5]
        );
    }

    #[test]
    fn test_cache_size_parsing() {
        assert_eq!(NumaTopologyDetector::parse_cache_size("32K").unwrap(), 32 * 1024);
        assert_eq!(NumaTopologyDetector::parse_cache_size("256K").unwrap(), 256 * 1024);
        assert_eq!(NumaTopologyDetector::parse_cache_size("8M").unwrap(), 8 * 1024 * 1024);
    }

    #[test]
    fn test_topology_detection() {
        let detector = NumaTopologyDetector::new();
        assert!(detector.is_ok());
        
        if let Ok(detector) = detector {
            let topology = detector.get_topology();
            assert!(!topology.numa_nodes.is_empty());
            assert!(!topology.cpus.is_empty());
        }
    }
}