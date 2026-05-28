use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use super::topology::{NumaTopologyDetector, SystemTopology};

/// CPU governor types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpuGovernor {
    Performance,
    Powersave,
    Userspace,
    Ondemand,
    Conservative,
    Schedutil,
}

impl CpuGovernor {
    fn as_str(&self) -> &'static str {
        match self {
            CpuGovernor::Performance => "performance",
            CpuGovernor::Powersave => "powersave",
            CpuGovernor::Userspace => "userspace",
            CpuGovernor::Ondemand => "ondemand",
            CpuGovernor::Conservative => "conservative",
            CpuGovernor::Schedutil => "schedutil",
        }
    }
}

/// Thread priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThreadPriority {
    Critical = 99,    // Real-time priority for trading threads
    High = 80,        // High priority for market data
    Normal = 50,      // Normal priority for background tasks
    Low = 20,         // Low priority for maintenance
}

/// CPU utilization statistics
#[derive(Debug, Clone)]
pub struct CpuUtilization {
    pub cpu_id: u32,
    pub utilization_percent: f64,
    pub user_time: u64,
    pub system_time: u64,
    pub idle_time: u64,
    pub iowait_time: u64,
    pub irq_time: u64,
    pub softirq_time: u64,
    pub last_update: Instant,
}

/// Thread affinity configuration
#[derive(Debug, Clone)]
pub struct ThreadAffinity {
    pub thread_id: ThreadId,
    pub cpu_set: HashSet<u32>,
    pub numa_node: u32,
    pub priority: ThreadPriority,
    pub isolated: bool,
}

/// CPU isolation configuration
#[derive(Debug, Clone)]
pub struct CpuIsolation {
    pub isolated_cpus: HashSet<u32>,
    pub trading_cpus: HashSet<u32>,
    pub system_cpus: HashSet<u32>,
    pub interrupt_cpus: HashSet<u32>,
}

/// CPU affinity manager
pub struct CpuAffinityManager {
    topology: Arc<SystemTopology>,
    thread_affinities: Arc<Mutex<HashMap<ThreadId, ThreadAffinity>>>,
    cpu_utilization: Arc<Mutex<HashMap<u32, CpuUtilization>>>,
    isolation_config: Arc<Mutex<CpuIsolation>>,
    monitoring_active: AtomicBool,
    update_interval: Duration,
}

impl CpuAffinityManager {
    /// Create a new CPU affinity manager
    pub fn new(topology_detector: &NumaTopologyDetector) -> Result<Self, Box<dyn std::error::Error>> {
        let topology = topology_detector.get_topology();
        let isolation_config = Self::create_default_isolation(&topology)?;
        
        Ok(Self {
            topology,
            thread_affinities: Arc::new(Mutex::new(HashMap::new())),
            cpu_utilization: Arc::new(Mutex::new(HashMap::new())),
            isolation_config: Arc::new(Mutex::new(isolation_config)),
            monitoring_active: AtomicBool::new(false),
            update_interval: Duration::from_millis(100),
        })
    }

    /// Create default CPU isolation configuration
    fn create_default_isolation(topology: &SystemTopology) -> Result<CpuIsolation, Box<dyn std::error::Error>> {
        let all_cpus: HashSet<u32> = topology.cpus.keys().copied().collect();
        let cpu_count = all_cpus.len();
        
        // Reserve CPUs for different purposes
        let system_cpu_count = (cpu_count / 8).max(2); // At least 2 CPUs for system
        let interrupt_cpu_count = (cpu_count / 16).max(1); // At least 1 CPU for interrupts
        let trading_cpu_count = cpu_count - system_cpu_count - interrupt_cpu_count;
        
        let mut system_cpus = HashSet::new();
        let mut interrupt_cpus = HashSet::new();
        let mut trading_cpus = HashSet::new();
        let mut isolated_cpus = HashSet::new();
        
        let mut cpu_list: Vec<u32> = all_cpus.into_iter().collect();
        cpu_list.sort();
        
        // Assign system CPUs (first few CPUs)
        for &cpu in cpu_list.iter().take(system_cpu_count) {
            system_cpus.insert(cpu);
        }
        
        // Assign interrupt CPUs (next few CPUs)
        for &cpu in cpu_list.iter().skip(system_cpu_count).take(interrupt_cpu_count) {
            interrupt_cpus.insert(cpu);
        }
        
        // Assign trading CPUs (remaining CPUs)
        for &cpu in cpu_list.iter().skip(system_cpu_count + interrupt_cpu_count) {
            trading_cpus.insert(cpu);
            isolated_cpus.insert(cpu); // Trading CPUs are isolated
        }
        
        Ok(CpuIsolation {
            isolated_cpus,
            trading_cpus,
            system_cpus,
            interrupt_cpus,
        })
    }

    /// Pin current thread to specific CPUs
    pub fn pin_thread_to_cpus(&self, cpu_set: &HashSet<u32>, priority: ThreadPriority) -> Result<(), Box<dyn std::error::Error>> {
        let thread_id = thread::current().id();
        
        // Set CPU affinity using sched_setaffinity
        self.set_cpu_affinity(cpu_set)?;
        
        // Set thread priority
        self.set_thread_priority(priority)?;
        
        // Get NUMA node for the CPU set
        let numa_node = self.get_numa_node_for_cpus(cpu_set);
        
        // Store thread affinity configuration
        let affinity = ThreadAffinity {
            thread_id,
            cpu_set: cpu_set.clone(),
            numa_node,
            priority,
            isolated: self.is_cpu_set_isolated(cpu_set),
        };
        
        self.thread_affinities.lock().unwrap().insert(thread_id, affinity);
        
        Ok(())
    }

    /// Pin current thread to a specific CPU
    pub fn pin_thread_to_cpu(&self, cpu_id: u32, priority: ThreadPriority) -> Result<(), Box<dyn std::error::Error>> {
        let mut cpu_set = HashSet::new();
        cpu_set.insert(cpu_id);
        self.pin_thread_to_cpus(&cpu_set, priority)
    }

    /// Pin current thread to trading CPUs
    pub fn pin_thread_to_trading_cpus(&self, priority: ThreadPriority) -> Result<(), Box<dyn std::error::Error>> {
        let isolation_config = self.isolation_config.lock().unwrap();
        self.pin_thread_to_cpus(&isolation_config.trading_cpus, priority)
    }

    /// Pin current thread to a specific NUMA node
    pub fn pin_thread_to_numa_node(&self, numa_node: u32, priority: ThreadPriority) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(node) = self.topology.numa_nodes.get(&numa_node) {
            let cpu_set: HashSet<u32> = node.cpus.iter().copied().collect();
            self.pin_thread_to_cpus(&cpu_set, priority)
        } else {
            Err(format!("NUMA node {} not found", numa_node).into())
        }
    }

    /// Set CPU affinity for current thread
    fn set_cpu_affinity(&self, cpu_set: &HashSet<u32>) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            use std::mem;
            
            let mut cpu_set_native: libc::cpu_set_t = unsafe { mem::zeroed() };
            
            unsafe {
                libc::CPU_ZERO(&mut cpu_set_native);
                for &cpu in cpu_set {
                    libc::CPU_SET(cpu as usize, &mut cpu_set_native);
                }
                
                let result = libc::sched_setaffinity(
                    0, // Current thread
                    mem::size_of::<libc::cpu_set_t>(),
                    &cpu_set_native,
                );
                
                if result != 0 {
                    return Err(format!("Failed to set CPU affinity: {}", std::io::Error::last_os_error()).into());
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Placeholder for non-Linux systems
            println!("CPU affinity setting not implemented for this platform");
        }
        
        Ok(())
    }

    /// Set thread priority
    fn set_thread_priority(&self, priority: ThreadPriority) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "linux")]
        {
            let policy = libc::SCHED_FIFO; // Real-time FIFO scheduling
            let param = libc::sched_param {
                sched_priority: priority as i32,
            };
            
            unsafe {
                let result = libc::sched_setscheduler(0, policy, &param);
                if result != 0 {
                    // Fallback to nice value if real-time scheduling fails
                    let nice_value = match priority {
                        ThreadPriority::Critical => -20,
                        ThreadPriority::High => -10,
                        ThreadPriority::Normal => 0,
                        ThreadPriority::Low => 10,
                    };
                    
                    let result = libc::setpriority(libc::PRIO_PROCESS, 0, nice_value);
                    if result != 0 {
                        return Err(format!("Failed to set thread priority: {}", std::io::Error::last_os_error()).into());
                    }
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Placeholder for non-Linux systems
            println!("Thread priority setting not implemented for this platform");
        }
        
        Ok(())
    }

    /// Get NUMA node for a set of CPUs
    fn get_numa_node_for_cpus(&self, cpu_set: &HashSet<u32>) -> u32 {
        // Find the most common NUMA node among the CPUs
        let mut node_counts = HashMap::new();
        
        for &cpu in cpu_set {
            if let Some(&numa_node) = self.topology.cpu_to_node.get(&cpu) {
                *node_counts.entry(numa_node).or_insert(0) += 1;
            }
        }
        
        node_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(node, _)| node)
            .unwrap_or(0)
    }

    /// Check if CPU set is isolated
    fn is_cpu_set_isolated(&self, cpu_set: &HashSet<u32>) -> bool {
        let isolation_config = self.isolation_config.lock().unwrap();
        cpu_set.iter().all(|cpu| isolation_config.isolated_cpus.contains(cpu))
    }

    /// Set CPU governor for specific CPUs
    pub fn set_cpu_governor(&self, cpu_set: &HashSet<u32>, governor: CpuGovernor) -> Result<(), Box<dyn std::error::Error>> {
        for &cpu in cpu_set {
            let governor_path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor", cpu);
            if let Err(e) = fs::write(&governor_path, governor.as_str()) {
                eprintln!("Failed to set governor for CPU {}: {}", cpu, e);
            }
        }
        Ok(())
    }

    /// Set CPU frequency for specific CPUs
    pub fn set_cpu_frequency(&self, cpu_set: &HashSet<u32>, frequency_khz: u64) -> Result<(), Box<dyn std::error::Error>> {
        for &cpu in cpu_set {
            let freq_path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_setspeed", cpu);
            if let Err(e) = fs::write(&freq_path, frequency_khz.to_string()) {
                eprintln!("Failed to set frequency for CPU {}: {}", cpu, e);
            }
        }
        Ok(())
    }

    /// Isolate CPUs from kernel scheduler
    pub fn isolate_cpus(&self, cpu_set: &HashSet<u32>) -> Result<(), Box<dyn std::error::Error>> {
        // Set CPU isolation in the kernel (requires root privileges)
        let cpu_list = self.format_cpu_list(cpu_set);
        
        // Write to isolcpus kernel parameter (requires reboot)
        println!("To isolate CPUs {}, add 'isolcpus={}' to kernel boot parameters", cpu_list, cpu_list);
        
        // Update isolation configuration
        {
            let mut isolation_config = self.isolation_config.lock().unwrap();
            for &cpu in cpu_set {
                isolation_config.isolated_cpus.insert(cpu);
            }
        }
        
        Ok(())
    }

    /// Format CPU set as a string (e.g., "0-3,8-11")
    fn format_cpu_list(&self, cpu_set: &HashSet<u32>) -> String {
        let mut cpus: Vec<u32> = cpu_set.iter().copied().collect();
        cpus.sort();
        
        if cpus.is_empty() {
            return String::new();
        }
        
        let mut result = String::new();
        let mut range_start = cpus[0];
        let mut range_end = cpus[0];
        
        for &cpu in cpus.iter().skip(1) {
            if cpu == range_end + 1 {
                range_end = cpu;
            } else {
                // End current range
                if !result.is_empty() {
                    result.push(',');
                }
                
                if range_start == range_end {
                    result.push_str(&range_start.to_string());
                } else {
                    result.push_str(&format!("{}-{}", range_start, range_end));
                }
                
                range_start = cpu;
                range_end = cpu;
            }
        }
        
        // Add final range
        if !result.is_empty() {
            result.push(',');
        }
        
        if range_start == range_end {
            result.push_str(&range_start.to_string());
        } else {
            result.push_str(&format!("{}-{}", range_start, range_end));
        }
        
        result
    }

    /// Start CPU utilization monitoring
    pub fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.monitoring_active.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_err() {
            return Ok(()); // Already monitoring
        }

        let cpu_utilization = self.cpu_utilization.clone();
        let monitoring_active = self.monitoring_active.clone();
        let update_interval = self.update_interval;
        let topology = self.topology.clone();

        thread::spawn(move || {
            let mut prev_stats = HashMap::new();
            
            while monitoring_active.load(Ordering::Relaxed) {
                if let Ok(current_stats) = Self::read_cpu_stats() {
                    let mut utilization_map = cpu_utilization.lock().unwrap();
                    
                    for (cpu_id, current_stat) in &current_stats {
                        if let Some(prev_stat) = prev_stats.get(cpu_id) {
                            let utilization = Self::calculate_cpu_utilization(prev_stat, current_stat);
                            utilization_map.insert(*cpu_id, utilization);
                        }
                    }
                    
                    prev_stats = current_stats;
                }
                
                thread::sleep(update_interval);
            }
        });

        Ok(())
    }

    /// Stop CPU utilization monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::Relaxed);
    }

    /// Read CPU statistics from /proc/stat
    fn read_cpu_stats() -> Result<HashMap<u32, CpuStats>, Box<dyn std::error::Error>> {
        let stat_content = fs::read_to_string("/proc/stat")?;
        let mut cpu_stats = HashMap::new();
        
        for line in stat_content.lines() {
            if line.starts_with("cpu") && line.len() > 3 {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 8 {
                    let cpu_name = parts[0];
                    if cpu_name != "cpu" { // Skip the aggregate line
                        if let Ok(cpu_id) = cpu_name[3..].parse::<u32>() {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);
                            let iowait: u64 = parts[5].parse().unwrap_or(0);
                            let irq: u64 = parts[6].parse().unwrap_or(0);
                            let softirq: u64 = parts[7].parse().unwrap_or(0);
                            
                            cpu_stats.insert(cpu_id, CpuStats {
                                user: user + nice,
                                system,
                                idle,
                                iowait,
                                irq,
                                softirq,
                            });
                        }
                    }
                }
            }
        }
        
        Ok(cpu_stats)
    }

    /// Calculate CPU utilization between two measurements
    fn calculate_cpu_utilization(prev: &CpuStats, current: &CpuStats) -> CpuUtilization {
        let total_prev = prev.user + prev.system + prev.idle + prev.iowait + prev.irq + prev.softirq;
        let total_current = current.user + current.system + current.idle + current.iowait + current.irq + current.softirq;
        
        let total_diff = total_current.saturating_sub(total_prev);
        let idle_diff = current.idle.saturating_sub(prev.idle);
        
        let utilization_percent = if total_diff > 0 {
            ((total_diff - idle_diff) as f64 / total_diff as f64) * 100.0
        } else {
            0.0
        };
        
        CpuUtilization {
            cpu_id: 0, // Will be set by caller
            utilization_percent,
            user_time: current.user.saturating_sub(prev.user),
            system_time: current.system.saturating_sub(prev.system),
            idle_time: current.idle.saturating_sub(prev.idle),
            iowait_time: current.iowait.saturating_sub(prev.iowait),
            irq_time: current.irq.saturating_sub(prev.irq),
            softirq_time: current.softirq.saturating_sub(prev.softirq),
            last_update: Instant::now(),
        }
    }

    /// Get CPU utilization for a specific CPU
    pub fn get_cpu_utilization(&self, cpu_id: u32) -> Option<CpuUtilization> {
        self.cpu_utilization.lock().unwrap().get(&cpu_id).cloned()
    }

    /// Get CPU utilization for all CPUs
    pub fn get_all_cpu_utilization(&self) -> HashMap<u32, CpuUtilization> {
        self.cpu_utilization.lock().unwrap().clone()
    }

    /// Get thread affinity information
    pub fn get_thread_affinity(&self, thread_id: ThreadId) -> Option<ThreadAffinity> {
        self.thread_affinities.lock().unwrap().get(&thread_id).cloned()
    }

    /// Get all thread affinities
    pub fn get_all_thread_affinities(&self) -> HashMap<ThreadId, ThreadAffinity> {
        self.thread_affinities.lock().unwrap().clone()
    }

    /// Get isolation configuration
    pub fn get_isolation_config(&self) -> CpuIsolation {
        self.isolation_config.lock().unwrap().clone()
    }

    /// Update isolation configuration
    pub fn update_isolation_config(&self, config: CpuIsolation) {
        *self.isolation_config.lock().unwrap() = config;
    }

    /// Get optimal CPU for trading thread on specific NUMA node
    pub fn get_optimal_trading_cpu(&self, numa_node: u32) -> Option<u32> {
        let isolation_config = self.isolation_config.lock().unwrap();
        let utilization_map = self.cpu_utilization.lock().unwrap();
        
        // Find trading CPUs on the specified NUMA node with lowest utilization
        isolation_config.trading_cpus
            .iter()
            .filter(|&&cpu| {
                self.topology.cpu_to_node.get(&cpu) == Some(&numa_node)
            })
            .min_by(|&&a, &&b| {
                let util_a = utilization_map.get(&a).map(|u| u.utilization_percent).unwrap_or(0.0);
                let util_b = utilization_map.get(&b).map(|u| u.utilization_percent).unwrap_or(0.0);
                util_a.partial_cmp(&util_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
}

/// CPU statistics from /proc/stat
#[derive(Debug, Clone)]
struct CpuStats {
    user: u64,
    system: u64,
    idle: u64,
    iowait: u64,
    irq: u64,
    softirq: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::numa::topology::NumaTopologyDetector;

    #[test]
    fn test_cpu_list_formatting() {
        let manager = create_test_manager();
        
        let mut cpu_set = HashSet::new();
        cpu_set.insert(0);
        cpu_set.insert(1);
        cpu_set.insert(2);
        cpu_set.insert(4);
        cpu_set.insert(5);
        
        let formatted = manager.format_cpu_list(&cpu_set);
        assert_eq!(formatted, "0-2,4-5");
    }

    #[test]
    fn test_affinity_manager_creation() {
        let detector = NumaTopologyDetector::new().unwrap();
        let manager = CpuAffinityManager::new(&detector);
        assert!(manager.is_ok());
    }

    fn create_test_manager() -> CpuAffinityManager {
        let detector = NumaTopologyDetector::new().unwrap();
        CpuAffinityManager::new(&detector).unwrap()
    }
}