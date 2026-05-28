use super::ring_buffers::{RingBuffer, RingBufferError};
use super::zero_copy::{ZeroCopyPacket, PacketPool};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::net::{SocketAddr, IpAddr};
use std::time::{Duration, Instant};
use std::thread;

/// Kernel bypass networking interface supporting both DPDK and io_uring
pub struct KernelBypassNetwork {
    /// Network interface configuration
    config: NetworkConfig,
    
    /// Active network interfaces
    interfaces: HashMap<String, Arc<NetworkInterface>>,
    
    /// Packet processing rings
    rx_rings: Vec<Arc<RingBuffer<ZeroCopyPacket>>>,
    tx_rings: Vec<Arc<RingBuffer<ZeroCopyPacket>>>,
    
    /// Packet memory pools
    packet_pools: Vec<Arc<PacketPool>>,
    
    /// Network statistics
    stats: NetworkStatistics,
    
    /// Running state
    running: AtomicBool,
    
    /// Worker threads
    worker_handles: Vec<thread::JoinHandle<()>>,
}

/// Network interface configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Interface name (e.g., "eth0")
    pub interface_name: String,
    
    /// Number of RX queues
    pub rx_queues: usize,
    
    /// Number of TX queues  
    pub tx_queues: usize,
    
    /// Ring buffer size (must be power of 2)
    pub ring_size: usize,
    
    /// Packet buffer size
    pub packet_buffer_size: usize,
    
    /// Number of packet buffers per pool
    pub packets_per_pool: usize,
    
    /// CPU cores for packet processing
    pub cpu_cores: Vec<usize>,
    
    /// Enable hardware timestamping
    pub hardware_timestamps: bool,
    
    /// Enable RSS (Receive Side Scaling)
    pub enable_rss: bool,
    
    /// MTU size
    pub mtu: usize,
    
    /// Polling interval (microseconds)
    pub poll_interval_us: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            interface_name: "eth0".to_string(),
            rx_queues: 4,
            tx_queues: 4,
            ring_size: 2048,
            packet_buffer_size: 2048,
            packets_per_pool: 4096,
            cpu_cores: vec![0, 1, 2, 3],
            hardware_timestamps: true,
            enable_rss: true,
            mtu: 1500,
            poll_interval_us: 1, // 1 microsecond polling
        }
    }
}

/// Network interface abstraction
pub struct NetworkInterface {
    /// Interface name
    pub name: String,
    
    /// Interface index
    pub index: u32,
    
    /// MAC address
    pub mac_address: [u8; 6],
    
    /// IP addresses
    pub ip_addresses: Vec<IpAddr>,
    
    /// Interface statistics
    pub stats: InterfaceStatistics,
    
    /// Hardware capabilities
    pub capabilities: HardwareCapabilities,
}

/// Hardware capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Supports hardware timestamping
    pub hardware_timestamps: bool,
    
    /// Supports RSS
    pub rss_support: bool,
    
    /// Number of RSS queues
    pub rss_queues: usize,
    
    /// Supports checksum offload
    pub checksum_offload: bool,
    
    /// Supports segmentation offload
    pub segmentation_offload: bool,
    
    /// Maximum supported MTU
    pub max_mtu: usize,
}

/// Network statistics
#[repr(align(64))]
pub struct NetworkStatistics {
    /// Packets received
    pub rx_packets: AtomicU64,
    
    /// Packets transmitted
    pub tx_packets: AtomicU64,
    
    /// Bytes received
    pub rx_bytes: AtomicU64,
    
    /// Bytes transmitted
    pub tx_bytes: AtomicU64,
    
    /// Packets dropped (RX)
    pub rx_dropped: AtomicU64,
    
    /// Packets dropped (TX)
    pub tx_dropped: AtomicU64,
    
    /// RX errors
    pub rx_errors: AtomicU64,
    
    /// TX errors
    pub tx_errors: AtomicU64,
    
    /// Average packet processing latency (nanoseconds)
    pub avg_processing_latency_ns: AtomicU64,
    
    /// Peak processing latency (nanoseconds)
    pub peak_processing_latency_ns: AtomicU64,
    
    /// Packets per second (RX)
    pub rx_pps: AtomicU64,
    
    /// Packets per second (TX)
    pub tx_pps: AtomicU64,
}

/// Interface-specific statistics
#[repr(align(64))]
pub struct InterfaceStatistics {
    /// Interface up/down events
    pub link_changes: AtomicU64,
    
    /// Last link change timestamp
    pub last_link_change: AtomicU64,
    
    /// Current link speed (Mbps)
    pub link_speed_mbps: AtomicU64,
    
    /// Link utilization percentage
    pub link_utilization_percent: AtomicU64,
}

impl KernelBypassNetwork {
    /// Create a new kernel bypass network instance
    pub fn new(config: NetworkConfig) -> Result<Self, NetworkError> {
        let mut network = Self {
            config: config.clone(),
            interfaces: HashMap::new(),
            rx_rings: Vec::new(),
            tx_rings: Vec::new(),
            packet_pools: Vec::new(),
            stats: NetworkStatistics::new(),
            running: AtomicBool::new(false),
            worker_handles: Vec::new(),
        };

        // Initialize network interface
        network.initialize_interface()?;
        
        // Create ring buffers
        network.create_ring_buffers()?;
        
        // Create packet pools
        network.create_packet_pools()?;

        Ok(network)
    }

    /// Initialize the network interface
    fn initialize_interface(&mut self) -> Result<(), NetworkError> {
        // In a real implementation, this would use DPDK or similar
        // For now, we'll create a mock interface
        let interface = Arc::new(NetworkInterface {
            name: self.config.interface_name.clone(),
            index: 1,
            mac_address: [0x00, 0x11, 0x22, 0x33, 0x44, 0x55],
            ip_addresses: vec!["192.168.1.100".parse().unwrap()],
            stats: InterfaceStatistics::new(),
            capabilities: HardwareCapabilities {
                hardware_timestamps: self.config.hardware_timestamps,
                rss_support: self.config.enable_rss,
                rss_queues: self.config.rx_queues,
                checksum_offload: true,
                segmentation_offload: true,
                max_mtu: 9000,
            },
        });

        self.interfaces.insert(self.config.interface_name.clone(), interface);
        Ok(())
    }

    /// Create ring buffers for packet processing
    fn create_ring_buffers(&mut self) -> Result<(), NetworkError> {
        // Create RX rings
        for i in 0..self.config.rx_queues {
            let ring = Arc::new(RingBuffer::new(
                self.config.ring_size,
                format!("rx_ring_{}", i),
            )?);
            self.rx_rings.push(ring);
        }

        // Create TX rings
        for i in 0..self.config.tx_queues {
            let ring = Arc::new(RingBuffer::new(
                self.config.ring_size,
                format!("tx_ring_{}", i),
            )?);
            self.tx_rings.push(ring);
        }

        Ok(())
    }

    /// Create packet memory pools
    fn create_packet_pools(&mut self) -> Result<(), NetworkError> {
        for i in 0..self.config.rx_queues {
            let pool = Arc::new(PacketPool::new(
                self.config.packets_per_pool,
                self.config.packet_buffer_size,
                i as u32, // NUMA node
            )?);
            self.packet_pools.push(pool);
        }

        Ok(())
    }

    /// Start the network processing
    pub fn start(&mut self) -> Result<(), NetworkError> {
        if self.running.load(Ordering::Acquire) {
            return Err(NetworkError::AlreadyRunning);
        }

        self.running.store(true, Ordering::Release);

        // Start worker threads for each CPU core
        for (i, &cpu_core) in self.config.cpu_cores.iter().enumerate() {
            let rx_ring = self.rx_rings[i % self.rx_rings.len()].clone();
            let tx_ring = self.tx_rings[i % self.tx_rings.len()].clone();
            let packet_pool = self.packet_pools[i % self.packet_pools.len()].clone();
            let stats = Arc::new(&self.stats as *const NetworkStatistics);
            let running = Arc::new(&self.running as *const AtomicBool);
            let poll_interval = Duration::from_micros(self.config.poll_interval_us);

            let handle = thread::Builder::new()
                .name(format!("net_worker_{}", i))
                .spawn(move || {
                    // Set CPU affinity
                    Self::set_cpu_affinity(cpu_core);
                    
                    // Main packet processing loop
                    Self::packet_processing_loop(
                        rx_ring,
                        tx_ring,
                        packet_pool,
                        stats,
                        running,
                        poll_interval,
                    );
                })
                .map_err(|e| NetworkError::ThreadCreationFailed(e.to_string()))?;

            self.worker_handles.push(handle);
        }

        Ok(())
    }

    /// Stop the network processing
    pub fn stop(&mut self) -> Result<(), NetworkError> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(false, Ordering::Release);

        // Wait for all worker threads to finish
        while let Some(handle) = self.worker_handles.pop() {
            handle.join().map_err(|_| NetworkError::ThreadJoinFailed)?;
        }

        Ok(())
    }

    /// Send a packet
    pub fn send_packet(
        &self,
        packet: ZeroCopyPacket,
        queue_id: usize,
    ) -> Result<(), NetworkError> {
        if queue_id >= self.tx_rings.len() {
            return Err(NetworkError::InvalidQueue(queue_id));
        }

        let tx_ring = &self.tx_rings[queue_id];
        
        match tx_ring.enqueue(packet) {
            Ok(_) => {
                self.stats.tx_packets.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(RingBufferError::Full) => {
                self.stats.tx_dropped.fetch_add(1, Ordering::Relaxed);
                Err(NetworkError::TxQueueFull)
            }
            Err(e) => Err(NetworkError::RingBufferError(e)),
        }
    }

    /// Receive packets from a specific queue
    pub fn receive_packets(
        &self,
        queue_id: usize,
        max_packets: usize,
    ) -> Result<Vec<ZeroCopyPacket>, NetworkError> {
        if queue_id >= self.rx_rings.len() {
            return Err(NetworkError::InvalidQueue(queue_id));
        }

        let rx_ring = &self.rx_rings[queue_id];
        let mut packets = Vec::with_capacity(max_packets);

        for _ in 0..max_packets {
            match rx_ring.dequeue() {
                Ok(packet) => {
                    packets.push(packet);
                    self.stats.rx_packets.fetch_add(1, Ordering::Relaxed);
                }
                Err(RingBufferError::Empty) => break,
                Err(e) => return Err(NetworkError::RingBufferError(e)),
            }
        }

        Ok(packets)
    }

    /// Get network statistics
    pub fn get_statistics(&self) -> NetworkStatsSnapshot {
        NetworkStatsSnapshot {
            rx_packets: self.stats.rx_packets.load(Ordering::Acquire),
            tx_packets: self.stats.tx_packets.load(Ordering::Acquire),
            rx_bytes: self.stats.rx_bytes.load(Ordering::Acquire),
            tx_bytes: self.stats.tx_bytes.load(Ordering::Acquire),
            rx_dropped: self.stats.rx_dropped.load(Ordering::Acquire),
            tx_dropped: self.stats.tx_dropped.load(Ordering::Acquire),
            rx_errors: self.stats.rx_errors.load(Ordering::Acquire),
            tx_errors: self.stats.tx_errors.load(Ordering::Acquire),
            avg_processing_latency_ns: self.stats.avg_processing_latency_ns.load(Ordering::Acquire),
            peak_processing_latency_ns: self.stats.peak_processing_latency_ns.load(Ordering::Acquire),
            rx_pps: self.stats.rx_pps.load(Ordering::Acquire),
            tx_pps: self.stats.tx_pps.load(Ordering::Acquire),
        }
    }

    /// Get interface information
    pub fn get_interface_info(&self, interface_name: &str) -> Option<&NetworkInterface> {
        self.interfaces.get(interface_name).map(|arc| arc.as_ref())
    }

    /// Set CPU affinity for the current thread
    fn set_cpu_affinity(cpu_core: usize) {
        // In a real implementation, this would use sched_setaffinity or similar
        // For now, we'll just log the intended affinity
        println!("Setting CPU affinity to core {}", cpu_core);
    }

    /// Main packet processing loop for worker threads
    fn packet_processing_loop(
        rx_ring: Arc<RingBuffer<ZeroCopyPacket>>,
        tx_ring: Arc<RingBuffer<ZeroCopyPacket>>,
        packet_pool: Arc<PacketPool>,
        stats: Arc<*const NetworkStatistics>,
        running: Arc<*const AtomicBool>,
        poll_interval: Duration,
    ) {
        let mut last_stats_update = Instant::now();
        let mut rx_count_last = 0u64;
        let mut tx_count_last = 0u64;

        while unsafe { (*running.as_ref()).load(Ordering::Acquire) } {
            let loop_start = Self::get_timestamp_ns();

            // Process RX packets
            let rx_processed = Self::process_rx_packets(&rx_ring, &packet_pool, &stats);
            
            // Process TX packets
            let tx_processed = Self::process_tx_packets(&tx_ring, &stats);

            // Update statistics periodically
            if last_stats_update.elapsed() >= Duration::from_secs(1) {
                unsafe {
                    let stats_ref = stats.as_ref();
                    let rx_count = stats_ref.rx_packets.load(Ordering::Acquire);
                    let tx_count = stats_ref.tx_packets.load(Ordering::Acquire);
                    
                    let rx_pps = rx_count - rx_count_last;
                    let tx_pps = tx_count - tx_count_last;
                    
                    stats_ref.rx_pps.store(rx_pps, Ordering::Release);
                    stats_ref.tx_pps.store(tx_pps, Ordering::Release);
                    
                    rx_count_last = rx_count;
                    tx_count_last = tx_count;
                }
                last_stats_update = Instant::now();
            }

            // Calculate processing latency
            let loop_end = Self::get_timestamp_ns();
            let processing_latency = loop_end - loop_start;
            
            unsafe {
                let stats_ref = stats.as_ref();
                Self::update_latency_stats(stats_ref, processing_latency);
            }

            // Sleep if no packets were processed
            if rx_processed == 0 && tx_processed == 0 {
                thread::sleep(poll_interval);
            }
        }
    }

    /// Process RX packets
    fn process_rx_packets(
        rx_ring: &RingBuffer<ZeroCopyPacket>,
        packet_pool: &PacketPool,
        stats: &Arc<*const NetworkStatistics>,
    ) -> usize {
        let mut processed = 0;
        
        // In a real implementation, this would receive packets from hardware
        // For now, we'll simulate packet reception
        
        // Try to allocate new packets from the pool
        for _ in 0..32 { // Batch size
            if let Ok(packet) = packet_pool.allocate_packet() {
                // Simulate packet data
                // In real implementation, this would be filled by hardware
                
                match rx_ring.enqueue(packet) {
                    Ok(_) => {
                        processed += 1;
                        unsafe {
                            (*stats.as_ref()).rx_packets.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(RingBufferError::Full) => {
                        unsafe {
                            (*stats.as_ref()).rx_dropped.fetch_add(1, Ordering::Relaxed);
                        }
                        break;
                    }
                    Err(_) => {
                        unsafe {
                            (*stats.as_ref()).rx_errors.fetch_add(1, Ordering::Relaxed);
                        }
                        break;
                    }
                }
            } else {
                break; // No more packets available
            }
        }
        
        processed
    }

    /// Process TX packets
    fn process_tx_packets(
        tx_ring: &RingBuffer<ZeroCopyPacket>,
        stats: &Arc<*const NetworkStatistics>,
    ) -> usize {
        let mut processed = 0;
        
        // Dequeue packets and transmit them
        for _ in 0..32 { // Batch size
            match tx_ring.dequeue() {
                Ok(_packet) => {
                    // In a real implementation, this would transmit the packet
                    processed += 1;
                    unsafe {
                        (*stats.as_ref()).tx_packets.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(RingBufferError::Empty) => break,
                Err(_) => {
                    unsafe {
                        (*stats.as_ref()).tx_errors.fetch_add(1, Ordering::Relaxed);
                    }
                    break;
                }
            }
        }
        
        processed
    }

    /// Update latency statistics
    fn update_latency_stats(stats: &NetworkStatistics, latency_ns: u64) {
        // Update average latency using exponential moving average
        let alpha = 0.1;
        loop {
            let current_avg = stats.avg_processing_latency_ns.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                latency_ns
            } else {
                ((1.0 - alpha) * current_avg as f64 + alpha * latency_ns as f64) as u64
            };

            match stats.avg_processing_latency_ns.compare_exchange_weak(
                current_avg,
                new_avg,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }

        // Update peak latency
        let mut peak = stats.peak_processing_latency_ns.load(Ordering::Acquire);
        while latency_ns > peak {
            match stats.peak_processing_latency_ns.compare_exchange_weak(
                peak,
                latency_ns,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}

impl Drop for KernelBypassNetwork {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Network statistics snapshot
#[derive(Debug, Clone)]
pub struct NetworkStatsSnapshot {
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_dropped: u64,
    pub tx_dropped: u64,
    pub rx_errors: u64,
    pub tx_errors: u64,
    pub avg_processing_latency_ns: u64,
    pub peak_processing_latency_ns: u64,
    pub rx_pps: u64,
    pub tx_pps: u64,
}

impl NetworkStatistics {
    fn new() -> Self {
        Self {
            rx_packets: AtomicU64::new(0),
            tx_packets: AtomicU64::new(0),
            rx_bytes: AtomicU64::new(0),
            tx_bytes: AtomicU64::new(0),
            rx_dropped: AtomicU64::new(0),
            tx_dropped: AtomicU64::new(0),
            rx_errors: AtomicU64::new(0),
            tx_errors: AtomicU64::new(0),
            avg_processing_latency_ns: AtomicU64::new(0),
            peak_processing_latency_ns: AtomicU64::new(0),
            rx_pps: AtomicU64::new(0),
            tx_pps: AtomicU64::new(0),
        }
    }
}

impl InterfaceStatistics {
    fn new() -> Self {
        Self {
            link_changes: AtomicU64::new(0),
            last_link_change: AtomicU64::new(0),
            link_speed_mbps: AtomicU64::new(1000), // Default to 1Gbps
            link_utilization_percent: AtomicU64::new(0),
        }
    }
}

/// Network-related errors
#[derive(Debug, Clone)]
pub enum NetworkError {
    InterfaceNotFound(String),
    InvalidQueue(usize),
    TxQueueFull,
    RxQueueEmpty,
    AlreadyRunning,
    NotRunning,
    ThreadCreationFailed(String),
    ThreadJoinFailed,
    RingBufferError(RingBufferError),
    PacketAllocationFailed,
    HardwareError(String),
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::InterfaceNotFound(name) => write!(f, "Interface not found: {}", name),
            NetworkError::InvalidQueue(id) => write!(f, "Invalid queue ID: {}", id),
            NetworkError::TxQueueFull => write!(f, "TX queue is full"),
            NetworkError::RxQueueEmpty => write!(f, "RX queue is empty"),
            NetworkError::AlreadyRunning => write!(f, "Network is already running"),
            NetworkError::NotRunning => write!(f, "Network is not running"),
            NetworkError::ThreadCreationFailed(e) => write!(f, "Thread creation failed: {}", e),
            NetworkError::ThreadJoinFailed => write!(f, "Thread join failed"),
            NetworkError::RingBufferError(e) => write!(f, "Ring buffer error: {:?}", e),
            NetworkError::PacketAllocationFailed => write!(f, "Packet allocation failed"),
            NetworkError::HardwareError(e) => write!(f, "Hardware error: {}", e),
        }
    }
}

impl std::error::Error for NetworkError {}

impl From<RingBufferError> for NetworkError {
    fn from(error: RingBufferError) -> Self {
        NetworkError::RingBufferError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::default();
        let network = KernelBypassNetwork::new(config).unwrap();
        
        assert_eq!(network.rx_rings.len(), 4);
        assert_eq!(network.tx_rings.len(), 4);
        assert_eq!(network.packet_pools.len(), 4);
    }

    #[test]
    fn test_network_start_stop() {
        let config = NetworkConfig {
            cpu_cores: vec![0], // Use only one core for testing
            ..Default::default()
        };
        let mut network = KernelBypassNetwork::new(config).unwrap();
        
        assert!(network.start().is_ok());
        assert!(network.running.load(Ordering::Acquire));
        
        assert!(network.stop().is_ok());
        assert!(!network.running.load(Ordering::Acquire));
    }

    #[test]
    fn test_packet_send_receive() {
        let config = NetworkConfig::default();
        let network = KernelBypassNetwork::new(config).unwrap();
        
        // Create a test packet
        let packet_pool = &network.packet_pools[0];
        let packet = packet_pool.allocate_packet().unwrap();
        
        // Send packet
        assert!(network.send_packet(packet, 0).is_ok());
        
        // Check statistics
        let stats = network.get_statistics();
        assert_eq!(stats.tx_packets, 1);
    }

    #[test]
    fn test_statistics_collection() {
        let config = NetworkConfig::default();
        let network = KernelBypassNetwork::new(config).unwrap();
        
        let stats = network.get_statistics();
        assert_eq!(stats.rx_packets, 0);
        assert_eq!(stats.tx_packets, 0);
        assert_eq!(stats.rx_errors, 0);
        assert_eq!(stats.tx_errors, 0);
    }

    #[test]
    fn test_interface_info() {
        let config = NetworkConfig::default();
        let network = KernelBypassNetwork::new(config).unwrap();
        
        let interface = network.get_interface_info("eth0").unwrap();
        assert_eq!(interface.name, "eth0");
        assert_eq!(interface.index, 1);
        assert!(interface.capabilities.hardware_timestamps);
    }

    #[test]
    fn test_invalid_queue() {
        let config = NetworkConfig::default();
        let network = KernelBypassNetwork::new(config).unwrap();
        
        let packet_pool = &network.packet_pools[0];
        let packet = packet_pool.allocate_packet().unwrap();
        
        // Try to send to invalid queue
        let result = network.send_packet(packet, 999);
        assert!(matches!(result, Err(NetworkError::InvalidQueue(999))));
    }

    #[test]
    fn test_network_config_validation() {
        let config = NetworkConfig {
            ring_size: 1023, // Not power of 2
            ..Default::default()
        };
        
        // Should handle non-power-of-2 ring sizes gracefully
        let result = KernelBypassNetwork::new(config);
        assert!(result.is_ok()); // Our implementation handles this
    }
}