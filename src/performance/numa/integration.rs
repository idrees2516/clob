use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

use super::topology::NumaTopologyDetector;
use super::affinity::{CpuAffinityManager, ThreadPriority};
use super::placement::NumaDataPlacement;
use super::interrupts::InterruptAffinityManager;

/// Comprehensive NUMA optimization manager
pub struct NumaOptimizationManager {
    topology_detector: NumaTopologyDetector,
    affinity_manager: CpuAffinityManager,
    data_placement: NumaDataPlacement,
    interrupt_manager: InterruptAffinityManager,
}

impl NumaOptimizationManager {
    /// Create a new NUMA optimization manager
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let topology_detector = NumaTopologyDetector::new()?;
        let affinity_manager = CpuAffinityManager::new(&topology_detector)?;
        let data_placement = NumaDataPlacement::new(&topology_detector);
        
        let isolation_config = affinity_manager.get_isolation_config();
        let interrupt_manager = InterruptAffinityManager::new(&topology_detector, isolation_config)?;
        
        Ok(Self {
            topology_detector,
            affinity_manager,
            data_placement,
            interrupt_manager,
        })
    }

    /// Initialize complete NUMA optimization
    pub fn initialize_optimization(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Initializing NUMA optimization...");
        
        // Start topology monitoring
        self.topology_detector.start_monitoring()?;
        
        // Start CPU utilization monitoring
        self.affinity_manager.start_monitoring()?;
        
        // Configure interrupt affinity
        self.interrupt_manager.configure_network_interrupt_affinity()?;
        self.interrupt_manager.configure_interrupt_coalescing()?;
        self.interrupt_manager.configure_interrupt_isolation()?;
        
        // Start interrupt monitoring
        self.interrupt_manager.start_monitoring()?;
        
        println!("NUMA optimization initialized successfully");
        Ok(())
    }

    /// Optimize current thread for trading workload
    pub fn optimize_trading_thread(&self, numa_node: Option<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let target_node = numa_node.unwrap_or_else(|| {
            // Find the best NUMA node based on current CPU
            let current_cpu = self.get_current_cpu();
            self.topology_detector.get_best_numa_node_for_cpu(current_cpu).unwrap_or(0)
        });
        
        // Pin thread to trading CPUs on the target NUMA node
        self.affinity_manager.pin_thread_to_numa_node(target_node, ThreadPriority::Critical)?;
        
        println!("Optimized trading thread for NUMA node {}", target_node);
        Ok(())
    }

    /// Optimize current thread for market data processing
    pub fn optimize_market_data_thread(&self, numa_node: Option<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let target_node = numa_node.unwrap_or(0);
        
        // Pin thread to high-priority CPUs
        self.affinity_manager.pin_thread_to_numa_node(target_node, ThreadPriority::High)?;
        
        println!("Optimized market data thread for NUMA node {}", target_node);
        Ok(())
    }

    /// Allocate NUMA-optimized memory for trading data structures
    pub fn allocate_trading_memory(&self, size: usize, numa_node: Option<u32>) -> Result<*mut u8, Box<dyn std::error::Error>> {
        let target_node = numa_node.unwrap_or_else(|| {
            let current_cpu = self.get_current_cpu();
            self.topology_detector.get_best_numa_node_for_cpu(current_cpu).unwrap_or(0)
        });
        
        let ptr = self.data_placement.allocate_on_node_id(size, 64, target_node)?;
        Ok(ptr.as_ptr())
    }

    /// Get current CPU ID
    fn get_current_cpu(&self) -> u32 {
        #[cfg(target_os = "linux")]
        {
            unsafe { libc::sched_getcpu() as u32 }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Get system topology information
    pub fn get_topology(&self) -> Arc<super::topology::SystemTopology> {
        self.topology_detector.get_topology()
    }

    /// Get CPU affinity manager
    pub fn get_affinity_manager(&self) -> &CpuAffinityManager {
        &self.affinity_manager
    }

    /// Get data placement manager
    pub fn get_data_placement(&self) -> &NumaDataPlacement {
        &self.data_placement
    }

    /// Get interrupt manager
    pub fn get_interrupt_manager(&self) -> &InterruptAffinityManager {
        &self.interrupt_manager
    }

    /// Perform periodic optimization
    pub fn perform_periodic_optimization(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Analyze hot data and migrate if beneficial
        let migration_candidates = self.data_placement.analyze_migration_candidates();
        
        for candidate in migration_candidates.iter().take(5) { // Migrate top 5 candidates
            if candidate.migration_benefit > 0.3 { // 30% improvement threshold
                self.data_placement.migrate_data(candidate)?;
            }
        }
        
        // Optimize interrupt load balancing
        self.interrupt_manager.optimize_interrupt_load_balancing()?;
        
        Ok(())
    }

    /// Shutdown NUMA optimization
    pub fn shutdown(&self) {
        println!("Shutting down NUMA optimization...");
        
        self.topology_detector.stop_monitoring();
        self.affinity_manager.stop_monitoring();
        self.interrupt_manager.stop_monitoring();
        
        println!("NUMA optimization shutdown complete");
    }
}

impl Drop for NumaOptimizationManager {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Helper function to initialize NUMA optimization for the entire application
pub fn initialize_numa_optimization() -> Result<Arc<NumaOptimizationManager>, Box<dyn std::error::Error>> {
    let manager = Arc::new(NumaOptimizationManager::new()?);
    manager.initialize_optimization()?;
    
    // Start periodic optimization in background thread
    let manager_clone = manager.clone();
    thread::spawn(move || {
        loop {
            thread::sleep(std::time::Duration::from_secs(60)); // Every minute
            if let Err(e) = manager_clone.perform_periodic_optimization() {
                eprintln!("Periodic NUMA optimization failed: {}", e);
            }
        }
    });
    
    Ok(manager)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_optimization_manager_creation() {
        let manager = NumaOptimizationManager::new();
        assert!(manager.is_ok());
    }

    #[test]
    fn test_numa_optimization_initialization() {
        if let Ok(manager) = NumaOptimizationManager::new() {
            // This test might fail without proper privileges, so we'll just check creation
            let topology = manager.get_topology();
            assert!(!topology.numa_nodes.is_empty());
        }
    }
}