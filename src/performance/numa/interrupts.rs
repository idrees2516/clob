use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use super::topology::{NumaTopologyDetector, SystemTopology};
use super::affinity::CpuIsolation;

/// Interrupt types for classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterruptType {
    Network,
    Storage,
    Timer,
    IPI,      // Inter-processor interrupt
    Thermal,
    Other,
}

/// Interrupt statistics
#[derive(Debug, Clone)]
pub struct InterruptStats {
    pub irq_number: u32,
    pub interrupt_type: InterruptType,
    pub device_name: String,
    pub cpu_counts: HashMap<u32, u64>, // Per-CPU interrupt counts
    pub total_count: u64,
    pub rate_per_second: f64,
    pub last_update: Instant,
}

/// Interrupt coalescing configuration
#[derive(Debug, Clone)]
pub struct CoalescingConfig {
    pub rx_usecs: u32,        // Receive interrupt coalescing delay (microseconds)
    pub tx_usecs: u32,        // Transmit interrupt coalescing delay (microseconds)
    pub rx_max_frames: u32,   // Maximum frames before interrupt
    pub tx_max_frames: u32,   // Maximum frames before interrupt
    pub adaptive: bool,       // Enable adaptive interrupt coalescing
}

/// Interrupt load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadBalancingStrategy {
    /// Distribute interrupts evenly across all CPUs
    RoundRobin,
    /// Bind interrupts to specific CPUs
    Static,
    /// Balance based on CPU utilization
    Dynamic,
    /// Isolate interrupts to dedicated CPUs
    Isolated,
}

/// Interrupt affinity optimization manager
pub struct InterruptAffinityManager {
    topology: Arc<SystemTopology>,
    interrupt_stats: Arc<Mutex<HashMap<u32, InterruptStats>>>,
    coalescing_configs: Arc<Mutex<HashMap<String, CoalescingConfig>>>,
    isolation_config: Arc<Mutex<CpuIsolation>>,
    monitoring_active: AtomicBool,
    load_balancing_strategy: LoadBalancingStrategy,
    update_interval: Duration,
}

impl InterruptAffinityManager {
    /// Create a new interrupt affinity manager
    pub fn new(
        topology_detector: &NumaTopologyDetector,
        isolation_config: CpuIsolation,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let topology = topology_detector.get_topology();
        
        Ok(Self {
            topology,
            interrupt_stats: Arc::new(Mutex::new(HashMap::new())),
            coalescing_configs: Arc::new(Mutex::new(HashMap::new())),
            isolation_config: Arc::new(Mutex::new(isolation_config)),
            monitoring_active: AtomicBool::new(false),
            load_balancing_strategy: LoadBalancingStrategy::Isolated,
            update_interval: Duration::from_millis(1000),
        })
    }

    /// Set load balancing strategy
    pub fn set_load_balancing_strategy(&mut self, strategy: LoadBalancingStrategy) {
        self.load_balancing_strategy = strategy;
    }

    /// Configure network interrupt CPU affinity
    pub fn configure_network_interrupt_affinity(&self) -> Result<(), Box<dyn std::error::Error>> {
        let network_interfaces = self.discover_network_interfaces()?;
        let isolation_config = self.isolation_config.lock().unwrap();
        
        for interface in &network_interfaces {
            let irq_numbers = self.get_interface_irq_numbers(interface)?;
            
            for irq in irq_numbers {
                match self.load_balancing_strategy {
                    LoadBalancingStrategy::Isolated => {
                        // Bind to dedicated interrupt CPUs
                        self.set_irq_affinity(irq, &isolation_config.interrupt_cpus)?;
                    }
                    LoadBalancingStrategy::RoundRobin => {
                        // Distribute across all available CPUs
                        let all_cpus: HashSet<u32> = self.topology.cpus.keys().copied().collect();
                        self.set_irq_affinity(irq, &all_cpus)?;
                    }
                    LoadBalancingStrategy::Static => {
                        // Use predefined CPU assignment
                        let cpu_id = self.get_static_cpu_for_irq(irq);
                        let mut cpu_set = HashSet::new();
                        cpu_set.insert(cpu_id);
                        self.set_irq_affinity(irq, &cpu_set)?;
                    }
                    LoadBalancingStrategy::Dynamic => {
                        // Balance based on current CPU utilization
                        let optimal_cpus = self.get_optimal_cpus_for_interrupts(4)?;
                        self.set_irq_affinity(irq, &optimal_cpus)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Discover network interfaces
    fn discover_network_interfaces(&self) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut interfaces = Vec::new();
        let net_path = Path::new("/sys/class/net");
        
        if net_path.exists() {
            for entry in fs::read_dir(net_path)? {
                let entry = entry?;
                let interface_name = entry.file_name().to_string_lossy().to_string();
                
                // Skip loopback and virtual interfaces
                if !interface_name.starts_with("lo") && 
                   !interface_name.starts_with("veth") &&
                   !interface_name.starts_with("docker") {
                    interfaces.push(interface_name);
                }
            }
        }
        
        Ok(interfaces)
    }

    /// Get IRQ numbers for a network interface
    fn get_interface_irq_numbers(&self, interface: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut irq_numbers = Vec::new();
        
        // Check /proc/interrupts for interface-related IRQs
        let interrupts_content = fs::read_to_string("/proc/interrupts")?;
        
        for line in interrupts_content.lines() {
            if line.contains(interface) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(irq_str) = parts.first() {
                    if let Ok(irq) = irq_str.trim_end_matches(':').parse::<u32>() {
                        irq_numbers.push(irq);
                    }
                }
            }
        }
        
        // Also check MSI-X interrupts for the interface
        let pci_devices = self.get_pci_devices_for_interface(interface)?;
        for pci_device in pci_devices {
            let msi_irqs = self.get_msi_irqs_for_device(&pci_device)?;
            irq_numbers.extend(msi_irqs);
        }
        
        Ok(irq_numbers)
    }

    /// Get PCI devices associated with a network interface
    fn get_pci_devices_for_interface(&self, interface: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut pci_devices = Vec::new();
        let device_path = format!("/sys/class/net/{}/device", interface);
        
        if let Ok(device_link) = fs::read_link(&device_path) {
            if let Some(device_name) = device_link.file_name() {
                pci_devices.push(device_name.to_string_lossy().to_string());
            }
        }
        
        Ok(pci_devices)
    }

    /// Get MSI-X IRQ numbers for a PCI device
    fn get_msi_irqs_for_device(&self, pci_device: &str) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let mut irq_numbers = Vec::new();
        let msi_path = format!("/sys/bus/pci/devices/{}/msi_irqs", pci_device);
        
        if Path::new(&msi_path).exists() {
            for entry in fs::read_dir(&msi_path)? {
                let entry = entry?;
                if let Ok(irq) = entry.file_name().to_string_lossy().parse::<u32>() {
                    irq_numbers.push(irq);
                }
            }
        }
        
        Ok(irq_numbers)
    }

    /// Set IRQ affinity to specific CPUs
    pub fn set_irq_affinity(&self, irq: u32, cpu_set: &HashSet<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let affinity_path = format!("/proc/irq/{}/smp_affinity", irq);
        
        if !Path::new(&affinity_path).exists() {
            return Err(format!("IRQ {} does not exist", irq).into());
        }
        
        // Convert CPU set to bitmask
        let cpu_mask = self.cpu_set_to_mask(cpu_set);
        let mask_str = format!("{:x}", cpu_mask);
        
        fs::write(&affinity_path, &mask_str)?;
        
        println!("Set IRQ {} affinity to CPUs {:?} (mask: {})", irq, cpu_set, mask_str);
        
        Ok(())
    }

    /// Convert CPU set to bitmask
    fn cpu_set_to_mask(&self, cpu_set: &HashSet<u32>) -> u64 {
        let mut mask = 0u64;
        for &cpu in cpu_set {
            if cpu < 64 {
                mask |= 1u64 << cpu;
            }
        }
        mask
    }

    /// Get static CPU assignment for an IRQ (round-robin)
    fn get_static_cpu_for_irq(&self, irq: u32) -> u32 {
        let isolation_config = self.isolation_config.lock().unwrap();
        let interrupt_cpus: Vec<u32> = isolation_config.interrupt_cpus.iter().copied().collect();
        
        if interrupt_cpus.is_empty() {
            0
        } else {
            interrupt_cpus[(irq as usize) % interrupt_cpus.len()]
        }
    }

    /// Get optimal CPUs for interrupt handling based on utilization
    fn get_optimal_cpus_for_interrupts(&self, count: usize) -> Result<HashSet<u32>, Box<dyn std::error::Error>> {
        let isolation_config = self.isolation_config.lock().unwrap();
        let mut available_cpus: Vec<u32> = isolation_config.interrupt_cpus.iter().copied().collect();
        
        // Sort by utilization (would need CPU utilization data)
        // For now, just return the first 'count' CPUs
        available_cpus.truncate(count);
        
        Ok(available_cpus.into_iter().collect())
    }

    /// Configure interrupt coalescing for network interfaces
    pub fn configure_interrupt_coalescing(&self) -> Result<(), Box<dyn std::error::Error>> {
        let network_interfaces = self.discover_network_interfaces()?;
        
        for interface in &network_interfaces {
            let config = CoalescingConfig {
                rx_usecs: 50,        // 50 microseconds delay
                tx_usecs: 50,        // 50 microseconds delay
                rx_max_frames: 32,   // Max 32 frames before interrupt
                tx_max_frames: 32,   // Max 32 frames before interrupt
                adaptive: true,      // Enable adaptive coalescing
            };
            
            self.apply_coalescing_config(interface, &config)?;
            
            // Store configuration
            self.coalescing_configs.lock().unwrap().insert(interface.clone(), config);
        }
        
        Ok(())
    }

    /// Apply interrupt coalescing configuration to an interface
    fn apply_coalescing_config(&self, interface: &str, config: &CoalescingConfig) -> Result<(), Box<dyn std::error::Error>> {
        // Use ethtool to configure interrupt coalescing
        let ethtool_cmd = format!(
            "ethtool -C {} rx-usecs {} tx-usecs {} rx-frames {} tx-frames {} adaptive-rx {} adaptive-tx {}",
            interface,
            config.rx_usecs,
            config.tx_usecs,
            config.rx_max_frames,
            config.tx_max_frames,
            if config.adaptive { "on" } else { "off" },
            if config.adaptive { "on" } else { "off" }
        );
        
        // Execute ethtool command (would need proper command execution)
        println!("Would execute: {}", ethtool_cmd);
        
        Ok(())
    }

    /// Start interrupt monitoring
    pub fn start_monitoring(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.monitoring_active.compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed).is_err() {
            return Ok(()); // Already monitoring
        }

        let interrupt_stats = self.interrupt_stats.clone();
        let monitoring_active = self.monitoring_active.clone();
        let update_interval = self.update_interval;

        thread::spawn(move || {
            let mut prev_stats = HashMap::new();
            
            while monitoring_active.load(Ordering::Relaxed) {
                if let Ok(current_stats) = Self::read_interrupt_stats() {
                    let mut stats_map = interrupt_stats.lock().unwrap();
                    
                    for (irq, current_stat) in &current_stats {
                        if let Some(prev_stat) = prev_stats.get(irq) {
                            let rate = Self::calculate_interrupt_rate(prev_stat, current_stat, update_interval);
                            let mut updated_stat = current_stat.clone();
                            updated_stat.rate_per_second = rate;
                            updated_stat.last_update = Instant::now();
                            stats_map.insert(*irq, updated_stat);
                        } else {
                            stats_map.insert(*irq, current_stat.clone());
                        }
                    }
                    
                    prev_stats = current_stats;
                }
                
                thread::sleep(update_interval);
            }
        });

        Ok(())
    }

    /// Stop interrupt monitoring
    pub fn stop_monitoring(&self) {
        self.monitoring_active.store(false, Ordering::Relaxed);
    }

    /// Read interrupt statistics from /proc/interrupts
    fn read_interrupt_stats() -> Result<HashMap<u32, InterruptStats>, Box<dyn std::error::Error>> {
        let interrupts_content = fs::read_to_string("/proc/interrupts")?;
        let mut interrupt_stats = HashMap::new();
        let lines: Vec<&str> = interrupts_content.lines().collect();
        
        // Parse header to get CPU columns
        let header = lines.first().ok_or("Empty /proc/interrupts")?;
        let cpu_count = header.split_whitespace().count();
        
        for line in lines.iter().skip(1) {
            if let Some(stats) = Self::parse_interrupt_line(line, cpu_count) {
                interrupt_stats.insert(stats.irq_number, stats);
            }
        }
        
        Ok(interrupt_stats)
    }

    /// Parse a single line from /proc/interrupts
    fn parse_interrupt_line(line: &str, cpu_count: usize) -> Option<InterruptStats> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < cpu_count + 2 {
            return None;
        }
        
        // Parse IRQ number
        let irq_str = parts[0].trim_end_matches(':');
        let irq_number = irq_str.parse::<u32>().ok()?;
        
        // Parse per-CPU counts
        let mut cpu_counts = HashMap::new();
        let mut total_count = 0u64;
        
        for (cpu_id, count_str) in parts.iter().skip(1).take(cpu_count).enumerate() {
            if let Ok(count) = count_str.parse::<u64>() {
                cpu_counts.insert(cpu_id as u32, count);
                total_count += count;
            }
        }
        
        // Parse device name and determine interrupt type
        let device_info = parts.iter().skip(cpu_count + 1).collect::<Vec<_>>().join(" ");
        let interrupt_type = Self::classify_interrupt(&device_info);
        
        Some(InterruptStats {
            irq_number,
            interrupt_type,
            device_name: device_info,
            cpu_counts,
            total_count,
            rate_per_second: 0.0,
            last_update: Instant::now(),
        })
    }

    /// Classify interrupt type based on device information
    fn classify_interrupt(device_info: &str) -> InterruptType {
        let device_lower = device_info.to_lowercase();
        
        if device_lower.contains("eth") || device_lower.contains("net") || 
           device_lower.contains("ixgbe") || device_lower.contains("i40e") ||
           device_lower.contains("mlx") || device_lower.contains("bnx") {
            InterruptType::Network
        } else if device_lower.contains("nvme") || device_lower.contains("sata") ||
                  device_lower.contains("scsi") || device_lower.contains("ahci") {
            InterruptType::Storage
        } else if device_lower.contains("timer") || device_lower.contains("hpet") {
            InterruptType::Timer
        } else if device_lower.contains("ipi") || device_lower.contains("reschedule") {
            InterruptType::IPI
        } else if device_lower.contains("thermal") || device_lower.contains("temp") {
            InterruptType::Thermal
        } else {
            InterruptType::Other
        }
    }

    /// Calculate interrupt rate between two measurements
    fn calculate_interrupt_rate(
        prev: &InterruptStats,
        current: &InterruptStats,
        interval: Duration,
    ) -> f64 {
        let count_diff = current.total_count.saturating_sub(prev.total_count);
        count_diff as f64 / interval.as_secs_f64()
    }

    /// Get interrupt statistics for a specific IRQ
    pub fn get_interrupt_stats(&self, irq: u32) -> Option<InterruptStats> {
        self.interrupt_stats.lock().unwrap().get(&irq).cloned()
    }

    /// Get all interrupt statistics
    pub fn get_all_interrupt_stats(&self) -> HashMap<u32, InterruptStats> {
        self.interrupt_stats.lock().unwrap().clone()
    }

    /// Get network interrupt statistics
    pub fn get_network_interrupt_stats(&self) -> HashMap<u32, InterruptStats> {
        self.interrupt_stats
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, stats)| stats.interrupt_type == InterruptType::Network)
            .map(|(&irq, stats)| (irq, stats.clone()))
            .collect()
    }

    /// Optimize interrupt load balancing based on current statistics
    pub fn optimize_interrupt_load_balancing(&self) -> Result<(), Box<dyn std::error::Error>> {
        let network_stats = self.get_network_interrupt_stats();
        let isolation_config = self.isolation_config.lock().unwrap();
        
        // Find heavily loaded interrupts
        let mut heavy_interrupts: Vec<_> = network_stats
            .iter()
            .filter(|(_, stats)| stats.rate_per_second > 10000.0) // > 10K interrupts/sec
            .collect();
        
        // Sort by interrupt rate (highest first)
        heavy_interrupts.sort_by(|a, b| b.1.rate_per_second.partial_cmp(&a.1.rate_per_second).unwrap());
        
        // Redistribute heavy interrupts across interrupt CPUs
        let interrupt_cpus: Vec<u32> = isolation_config.interrupt_cpus.iter().copied().collect();
        
        for (i, (&irq, stats)) in heavy_interrupts.iter().enumerate() {
            if !interrupt_cpus.is_empty() {
                let target_cpu = interrupt_cpus[i % interrupt_cpus.len()];
                let mut cpu_set = HashSet::new();
                cpu_set.insert(target_cpu);
                
                self.set_irq_affinity(irq, &cpu_set)?;
                
                println!(
                    "Moved high-rate IRQ {} ({:.0} int/sec) to CPU {}",
                    irq, stats.rate_per_second, target_cpu
                );
            }
        }
        
        Ok(())
    }

    /// Get interrupt latency measurements
    pub fn measure_interrupt_latency(&self) -> Result<HashMap<u32, f64>, Box<dyn std::error::Error>> {
        let mut latencies = HashMap::new();
        
        // This would require specialized hardware or kernel support to measure
        // interrupt latency accurately. For now, we'll provide a placeholder
        // implementation that estimates latency based on interrupt rate and CPU utilization.
        
        let network_stats = self.get_network_interrupt_stats();
        
        for (&irq, stats) in &network_stats {
            // Estimate latency based on interrupt rate (higher rate = higher latency)
            let base_latency = 1.0; // 1 microsecond base latency
            let rate_penalty = (stats.rate_per_second / 100000.0).min(10.0); // Max 10x penalty
            let estimated_latency = base_latency * (1.0 + rate_penalty);
            
            latencies.insert(irq, estimated_latency);
        }
        
        Ok(latencies)
    }

    /// Configure interrupt CPU isolation
    pub fn configure_interrupt_isolation(&self) -> Result<(), Box<dyn std::error::Error>> {
        let isolation_config = self.isolation_config.lock().unwrap();
        
        // Disable irqbalance daemon to prevent automatic interrupt balancing
        println!("Disabling irqbalance daemon for manual interrupt control");
        
        // Move all interrupts away from trading CPUs
        let all_interrupts = self.get_all_interrupt_stats();
        
        for (&irq, _) in &all_interrupts {
            // Set affinity to interrupt CPUs only
            self.set_irq_affinity(irq, &isolation_config.interrupt_cpus)?;
        }
        
        // Configure RPS (Receive Packet Steering) to use interrupt CPUs
        let network_interfaces = self.discover_network_interfaces()?;
        for interface in &network_interfaces {
            self.configure_rps(&interface, &isolation_config.interrupt_cpus)?;
        }
        
        Ok(())
    }

    /// Configure Receive Packet Steering (RPS)
    fn configure_rps(&self, interface: &str, cpu_set: &HashSet<u32>) -> Result<(), Box<dyn std::error::Error>> {
        let cpu_mask = self.cpu_set_to_mask(cpu_set);
        let mask_str = format!("{:x}", cpu_mask);
        
        // Configure RPS for each RX queue
        let rx_queues_path = format!("/sys/class/net/{}/queues", interface);
        
        if Path::new(&rx_queues_path).exists() {
            for entry in fs::read_dir(&rx_queues_path)? {
                let entry = entry?;
                let queue_name = entry.file_name().to_string_lossy();
                
                if queue_name.starts_with("rx-") {
                    let rps_cpus_path = format!("{}/{}/rps_cpus", rx_queues_path, queue_name);
                    if Path::new(&rps_cpus_path).exists() {
                        fs::write(&rps_cpus_path, &mask_str)?;
                        println!("Set RPS for {}/{} to CPUs {:?}", interface, queue_name, cpu_set);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Get coalescing configuration for an interface
    pub fn get_coalescing_config(&self, interface: &str) -> Option<CoalescingConfig> {
        self.coalescing_configs.lock().unwrap().get(interface).cloned()
    }

    /// Update coalescing configuration
    pub fn update_coalescing_config(&self, interface: &str, config: CoalescingConfig) -> Result<(), Box<dyn std::error::Error>> {
        self.apply_coalescing_config(interface, &config)?;
        self.coalescing_configs.lock().unwrap().insert(interface.to_string(), config);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::numa::topology::NumaTopologyDetector;
    use crate::performance::numa::affinity::CpuAffinityManager;

    #[test]
    fn test_interrupt_manager_creation() {
        let detector = NumaTopologyDetector::new().unwrap();
        let affinity_manager = CpuAffinityManager::new(&detector).unwrap();
        let isolation_config = affinity_manager.get_isolation_config();
        
        let interrupt_manager = InterruptAffinityManager::new(&detector, isolation_config);
        assert!(interrupt_manager.is_ok());
    }

    #[test]
    fn test_cpu_set_to_mask() {
        let detector = NumaTopologyDetector::new().unwrap();
        let affinity_manager = CpuAffinityManager::new(&detector).unwrap();
        let isolation_config = affinity_manager.get_isolation_config();
        let manager = InterruptAffinityManager::new(&detector, isolation_config).unwrap();
        
        let mut cpu_set = HashSet::new();
        cpu_set.insert(0);
        cpu_set.insert(2);
        cpu_set.insert(4);
        
        let mask = manager.cpu_set_to_mask(&cpu_set);
        assert_eq!(mask, 0b10101); // Binary: 10101 = CPUs 0, 2, 4
    }

    #[test]
    fn test_interrupt_classification() {
        assert_eq!(
            InterruptAffinityManager::classify_interrupt("eth0-TxRx-0"),
            InterruptType::Network
        );
        assert_eq!(
            InterruptAffinityManager::classify_interrupt("nvme0q1"),
            InterruptType::Storage
        );
        assert_eq!(
            InterruptAffinityManager::classify_interrupt("timer"),
            InterruptType::Timer
        );
    }
}