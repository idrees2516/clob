use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;

/// Network bandwidth management and traffic shaping
pub struct NetworkBandwidthManager {
    interfaces: HashMap<String, Arc<NetworkInterface>>,
    traffic_shapers: HashMap<String, Arc<TrafficShaper>>,
    bandwidth_monitor: Arc<BandwidthMonitor>,
    congestion_controller: Arc<CongestionController>,
    qos_manager: Arc<QosManager>,
    enabled: AtomicBool,
}

/// Network interface representation
pub struct NetworkInterface {
    name: String,
    max_bandwidth_mbps: AtomicU64,
    current_rx_mbps: AtomicU64,
    current_tx_mbps: AtomicU64,
    utilization: AtomicU64, // Percentage * 100
    packet_loss_rate: AtomicU64, // Per million packets
    latency_us: AtomicU64,
    last_updated: Arc<RwLock<Instant>>,
}

/// Traffic shaping configuration
#[derive(Debug, Clone)]
pub struct TrafficShapingConfig {
    pub max_bandwidth_mbps: u64,
    pub burst_size_kb: u64,
    pub priority_queues: Vec<QueueConfig>,
    pub rate_limiting_enabled: bool,
    pub congestion_control_enabled: bool,
}

/// Queue configuration for QoS
#[derive(Debug, Clone)]
pub struct QueueConfig {
    pub name: String,
    pub priority: u8, // 0 = highest priority
    pub guaranteed_bandwidth_mbps: u64,
    pub max_bandwidth_mbps: u64,
    pub max_queue_size: usize,
    pub drop_policy: DropPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropPolicy {
    TailDrop,
    RandomEarlyDetection,
    WeightedRandomEarlyDetection,
}

/// Traffic shaper implementation
pub struct TrafficShaper {
    interface_name: String,
    config: Arc<RwLock<TrafficShapingConfig>>,
    token_buckets: HashMap<String, Arc<TokenBucket>>,
    queue_stats: Arc<RwLock<HashMap<String, QueueStats>>>,
    shaping_enabled: AtomicBool,
}

/// Token bucket for rate limiting
pub struct TokenBucket {
    capacity: AtomicU64,
    tokens: AtomicU64,
    refill_rate: AtomicU64, // Tokens per second
    last_refill: Arc<RwLock<Instant>>,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub packets_enqueued: u64,
    pub packets_dequeued: u64,
    pub packets_dropped: u64,
    pub bytes_enqueued: u64,
    pub bytes_dequeued: u64,
    pub current_queue_size: usize,
    pub max_queue_size_reached: usize,
    pub average_latency_us: u64,
}

/// Bandwidth monitoring
pub struct BandwidthMonitor {
    measurement_history: Arc<RwLock<HashMap<String, VecDeque<BandwidthMeasurement>>>>,
    monitoring_interval: Duration,
    history_retention: Duration,
    enabled: AtomicBool,
}

/// Bandwidth measurement
#[derive(Debug, Clone)]
pub struct BandwidthMeasurement {
    pub interface_name: String,
    pub rx_mbps: f64,
    pub tx_mbps: f64,
    pub utilization: f64,
    pub packet_loss_rate: f64,
    pub latency_us: u64,
    pub timestamp: Instant,
}

/// Congestion control
pub struct CongestionController {
    congestion_state: Arc<RwLock<HashMap<String, CongestionState>>>,
    control_algorithms: HashMap<String, Box<dyn CongestionControlAlgorithm>>,
    enabled: AtomicBool,
}

/// Congestion state for an interface
#[derive(Debug, Clone)]
pub struct CongestionState {
    pub level: CongestionLevel,
    pub window_size: u64,
    pub slow_start_threshold: u64,
    pub rtt_estimate_us: u64,
    pub packet_loss_rate: f64,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionLevel {
    None,
    Light,
    Moderate,
    Heavy,
    Severe,
}

/// Congestion control algorithm trait
pub trait CongestionControlAlgorithm: Send + Sync {
    fn on_ack_received(&self, state: &mut CongestionState, rtt_us: u64);
    fn on_packet_loss(&self, state: &mut CongestionState);
    fn on_timeout(&self, state: &mut CongestionState);
    fn calculate_send_rate(&self, state: &CongestionState) -> u64;
}

/// Quality of Service manager
pub struct QosManager {
    traffic_classes: HashMap<String, TrafficClass>,
    classification_rules: Vec<ClassificationRule>,
    enabled: AtomicBool,
}

/// Traffic class definition
#[derive(Debug, Clone)]
pub struct TrafficClass {
    pub name: String,
    pub priority: u8,
    pub guaranteed_bandwidth_mbps: u64,
    pub max_bandwidth_mbps: u64,
    pub latency_target_us: u64,
    pub jitter_target_us: u64,
}

/// Packet classification rule
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub name: String,
    pub source_ip_range: Option<String>,
    pub dest_ip_range: Option<String>,
    pub source_port_range: Option<(u16, u16)>,
    pub dest_port_range: Option<(u16, u16)>,
    pub protocol: Option<u8>,
    pub dscp_marking: Option<u8>,
    pub traffic_class: String,
}

impl NetworkBandwidthManager {
    pub fn new() -> Self {
        Self {
            interfaces: HashMap::new(),
            traffic_shapers: HashMap::new(),
            bandwidth_monitor: Arc::new(BandwidthMonitor::new()),
            congestion_controller: Arc::new(CongestionController::new()),
            qos_manager: Arc::new(QosManager::new()),
            enabled: AtomicBool::new(true),
        }
    }

    /// Register a network interface
    pub fn register_interface(&mut self, name: String, max_bandwidth_mbps: u64) {
        let interface = Arc::new(NetworkInterface {
            name: name.clone(),
            max_bandwidth_mbps: AtomicU64::new(max_bandwidth_mbps),
            current_rx_mbps: AtomicU64::new(0),
            current_tx_mbps: AtomicU64::new(0),
            utilization: AtomicU64::new(0),
            packet_loss_rate: AtomicU64::new(0),
            latency_us: AtomicU64::new(0),
            last_updated: Arc::new(RwLock::new(Instant::now())),
        });

        self.interfaces.insert(name, interface);
    }

    /// Configure traffic shaping for an interface
    pub async fn configure_traffic_shaping(
        &mut self,
        interface_name: String,
        config: TrafficShapingConfig,
    ) -> Result<(), String> {
        if !self.interfaces.contains_key(&interface_name) {
            return Err(format!("Interface {} not found", interface_name));
        }

        let shaper = Arc::new(TrafficShaper::new(interface_name.clone(), config));
        self.traffic_shapers.insert(interface_name, shaper);
        Ok(())
    }

    /// Start bandwidth management
    pub async fn start_management(&self) {
        let monitoring_task = self.start_monitoring_loop();
        let shaping_task = self.start_traffic_shaping_loop();
        let congestion_task = self.start_congestion_control_loop();

        tokio::join!(monitoring_task, shaping_task, congestion_task);
    }

    /// Monitor bandwidth utilization
    async fn start_monitoring_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        
        while self.enabled.load(Ordering::Relaxed) {
            interval.tick().await;
            self.update_bandwidth_metrics().await;
        }
    }

    /// Traffic shaping control loop
    async fn start_traffic_shaping_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        
        while self.enabled.load(Ordering::Relaxed) {
            interval.tick().await;
            self.apply_traffic_shaping().await;
        }
    }

    /// Congestion control loop
    async fn start_congestion_control_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_millis(10));
        
        while self.enabled.load(Ordering::Relaxed) {
            interval.tick().await;
            self.update_congestion_control().await;
        }
    }

    /// Update bandwidth metrics for all interfaces
    async fn update_bandwidth_metrics(&self) {
        for (name, interface) in &self.interfaces {
            let (rx_mbps, tx_mbps) = self.measure_interface_bandwidth(name).await;
            let max_bandwidth = interface.max_bandwidth_mbps.load(Ordering::Relaxed) as f64;
            let utilization = ((rx_mbps + tx_mbps) / max_bandwidth * 100.0) as u64;
            
            interface.current_rx_mbps.store((rx_mbps * 100.0) as u64, Ordering::Relaxed);
            interface.current_tx_mbps.store((tx_mbps * 100.0) as u64, Ordering::Relaxed);
            interface.utilization.store(utilization, Ordering::Relaxed);
            
            // Update packet loss and latency
            let (loss_rate, latency) = self.measure_interface_quality(name).await;
            interface.packet_loss_rate.store((loss_rate * 1_000_000.0) as u64, Ordering::Relaxed);
            interface.latency_us.store(latency, Ordering::Relaxed);
            
            let mut last_updated = interface.last_updated.write().await;
            *last_updated = Instant::now();

            // Record measurement
            let measurement = BandwidthMeasurement {
                interface_name: name.clone(),
                rx_mbps,
                tx_mbps,
                utilization: utilization as f64 / 100.0,
                packet_loss_rate: loss_rate,
                latency_us: latency,
                timestamp: Instant::now(),
            };

            self.bandwidth_monitor.record_measurement(measurement).await;
        }
    }

    /// Apply traffic shaping rules
    async fn apply_traffic_shaping(&self) {
        for (interface_name, shaper) in &self.traffic_shapers {
            if shaper.shaping_enabled.load(Ordering::Relaxed) {
                shaper.apply_shaping().await;
            }
        }
    }

    /// Update congestion control
    async fn update_congestion_control(&self) {
        if !self.congestion_controller.enabled.load(Ordering::Relaxed) {
            return;
        }

        for interface_name in self.interfaces.keys() {
            self.congestion_controller.update_congestion_state(interface_name).await;
        }
    }

    /// Get bandwidth statistics
    pub fn get_bandwidth_stats(&self) -> HashMap<String, InterfaceStats> {
        let mut stats = HashMap::new();
        
        for (name, interface) in &self.interfaces {
            let interface_stats = InterfaceStats {
                name: name.clone(),
                max_bandwidth_mbps: interface.max_bandwidth_mbps.load(Ordering::Relaxed),
                current_rx_mbps: interface.current_rx_mbps.load(Ordering::Relaxed) as f64 / 100.0,
                current_tx_mbps: interface.current_tx_mbps.load(Ordering::Relaxed) as f64 / 100.0,
                utilization: interface.utilization.load(Ordering::Relaxed) as f64 / 100.0,
                packet_loss_rate: interface.packet_loss_rate.load(Ordering::Relaxed) as f64 / 1_000_000.0,
                latency_us: interface.latency_us.load(Ordering::Relaxed),
            };
            
            stats.insert(name.clone(), interface_stats);
        }
        
        stats
    }

    /// Get traffic shaping statistics
    pub async fn get_shaping_stats(&self) -> HashMap<String, ShapingStats> {
        let mut stats = HashMap::new();
        
        for (interface_name, shaper) in &self.traffic_shapers {
            let shaping_stats = shaper.get_stats().await;
            stats.insert(interface_name.clone(), shaping_stats);
        }
        
        stats
    }

    /// Detect network congestion
    pub async fn detect_congestion(&self) -> HashMap<String, CongestionLevel> {
        let mut congestion_levels = HashMap::new();
        
        let congestion_state = self.congestion_controller.congestion_state.read().await;
        for (interface_name, state) in congestion_state.iter() {
            congestion_levels.insert(interface_name.clone(), state.level);
        }
        
        congestion_levels
    }

    /// Simulate bandwidth measurement (replace with actual network monitoring)
    async fn measure_interface_bandwidth(&self, _interface_name: &str) -> (f64, f64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), _interface_name).hash(&mut hasher);
        let hash = hasher.finish();
        
        // Simulate varying bandwidth usage
        let rx_mbps = 10.0 + (hash % 500) as f64; // 10-510 Mbps
        let tx_mbps = 5.0 + (hash % 300) as f64;  // 5-305 Mbps
        
        (rx_mbps, tx_mbps)
    }

    /// Simulate interface quality measurement
    async fn measure_interface_quality(&self, _interface_name: &str) -> (f64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), _interface_name, "quality").hash(&mut hasher);
        let hash = hasher.finish();
        
        // Simulate packet loss rate (0-1%) and latency (1-100ms)
        let loss_rate = (hash % 100) as f64 / 10000.0; // 0-0.01 (1%)
        let latency_us = 1000 + (hash % 99000); // 1-100ms in microseconds
        
        (loss_rate, latency_us)
    }

    /// Enable/disable bandwidth management
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Stop bandwidth management
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.bandwidth_monitor.stop();
        self.congestion_controller.stop();
    }
}

impl TrafficShaper {
    pub fn new(interface_name: String, config: TrafficShapingConfig) -> Self {
        let mut token_buckets = HashMap::new();
        
        // Create token buckets for each queue
        for queue in &config.priority_queues {
            let bucket = Arc::new(TokenBucket::new(
                queue.max_bandwidth_mbps * 1024 * 1024 / 8, // Convert to bytes per second
                queue.burst_size_kb * 1024,
            ));
            token_buckets.insert(queue.name.clone(), bucket);
        }

        Self {
            interface_name,
            config: Arc::new(RwLock::new(config)),
            token_buckets,
            queue_stats: Arc::new(RwLock::new(HashMap::new())),
            shaping_enabled: AtomicBool::new(true),
        }
    }

    /// Apply traffic shaping
    pub async fn apply_shaping(&self) {
        // Refill token buckets
        for bucket in self.token_buckets.values() {
            bucket.refill().await;
        }

        // Apply rate limiting and queue management
        // In a real implementation, this would interact with the network stack
        println!("Applying traffic shaping for interface {}", self.interface_name);
    }

    /// Get shaping statistics
    pub async fn get_stats(&self) -> ShapingStats {
        let queue_stats = self.queue_stats.read().await.clone();
        
        ShapingStats {
            interface_name: self.interface_name.clone(),
            shaping_enabled: self.shaping_enabled.load(Ordering::Relaxed),
            queue_stats,
            total_packets_shaped: queue_stats.values().map(|s| s.packets_enqueued).sum(),
            total_packets_dropped: queue_stats.values().map(|s| s.packets_dropped).sum(),
        }
    }

    /// Enable/disable traffic shaping
    pub fn set_enabled(&self, enabled: bool) {
        self.shaping_enabled.store(enabled, Ordering::Relaxed);
    }
}

impl TokenBucket {
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        Self {
            capacity: AtomicU64::new(capacity),
            tokens: AtomicU64::new(capacity),
            refill_rate: AtomicU64::new(refill_rate),
            last_refill: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Refill tokens based on elapsed time
    pub async fn refill(&self) {
        let now = Instant::now();
        let mut last_refill = self.last_refill.write().await;
        let elapsed = now.duration_since(*last_refill);
        
        let tokens_to_add = (elapsed.as_secs_f64() * self.refill_rate.load(Ordering::Relaxed) as f64) as u64;
        
        if tokens_to_add > 0 {
            let current_tokens = self.tokens.load(Ordering::Relaxed);
            let capacity = self.capacity.load(Ordering::Relaxed);
            let new_tokens = (current_tokens + tokens_to_add).min(capacity);
            
            self.tokens.store(new_tokens, Ordering::Relaxed);
            *last_refill = now;
        }
    }

    /// Try to consume tokens
    pub fn try_consume(&self, tokens: u64) -> bool {
        let current_tokens = self.tokens.load(Ordering::Relaxed);
        
        if current_tokens >= tokens {
            let new_tokens = current_tokens - tokens;
            // Use compare_exchange for atomic update
            self.tokens.compare_exchange(current_tokens, new_tokens, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        } else {
            false
        }
    }
}

impl BandwidthMonitor {
    pub fn new() -> Self {
        Self {
            measurement_history: Arc::new(RwLock::new(HashMap::new())),
            monitoring_interval: Duration::from_secs(1),
            history_retention: Duration::from_hours(24),
            enabled: AtomicBool::new(true),
        }
    }

    /// Record a bandwidth measurement
    pub async fn record_measurement(&self, measurement: BandwidthMeasurement) {
        let mut history = self.measurement_history.write().await;
        let interface_history = history.entry(measurement.interface_name.clone()).or_insert_with(VecDeque::new);
        
        interface_history.push_back(measurement);
        
        // Limit history size
        if interface_history.len() > 86400 { // 24 hours of seconds
            interface_history.pop_front();
        }
    }

    /// Get historical measurements
    pub async fn get_history(&self, interface_name: &str, duration: Duration) -> Vec<BandwidthMeasurement> {
        let history = self.measurement_history.read().await;
        let cutoff = Instant::now() - duration;
        
        if let Some(interface_history) = history.get(interface_name) {
            interface_history
                .iter()
                .filter(|m| m.timestamp >= cutoff)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }
}

impl CongestionController {
    pub fn new() -> Self {
        Self {
            congestion_state: Arc::new(RwLock::new(HashMap::new())),
            control_algorithms: HashMap::new(),
            enabled: AtomicBool::new(true),
        }
    }

    /// Update congestion state for an interface
    pub async fn update_congestion_state(&self, interface_name: &str) {
        // In a real implementation, this would analyze network metrics
        // and update congestion state accordingly
        
        let mut state_map = self.congestion_state.write().await;
        let state = state_map.entry(interface_name.to_string()).or_insert_with(|| {
            CongestionState {
                level: CongestionLevel::None,
                window_size: 65536,
                slow_start_threshold: 32768,
                rtt_estimate_us: 10000,
                packet_loss_rate: 0.0,
                last_updated: Instant::now(),
            }
        });

        // Simulate congestion detection
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        (std::time::SystemTime::now(), interface_name).hash(&mut hasher);
        let hash = hasher.finish();
        
        let congestion_indicator = (hash % 100) as f64 / 100.0;
        
        state.level = match congestion_indicator {
            x if x < 0.1 => CongestionLevel::None,
            x if x < 0.3 => CongestionLevel::Light,
            x if x < 0.6 => CongestionLevel::Moderate,
            x if x < 0.9 => CongestionLevel::Heavy,
            _ => CongestionLevel::Severe,
        };
        
        state.last_updated = Instant::now();
    }

    /// Stop congestion control
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }
}

impl QosManager {
    pub fn new() -> Self {
        Self {
            traffic_classes: HashMap::new(),
            classification_rules: Vec::new(),
            enabled: AtomicBool::new(true),
        }
    }

    /// Add a traffic class
    pub fn add_traffic_class(&mut self, traffic_class: TrafficClass) {
        self.traffic_classes.insert(traffic_class.name.clone(), traffic_class);
    }

    /// Add a classification rule
    pub fn add_classification_rule(&mut self, rule: ClassificationRule) {
        self.classification_rules.push(rule);
    }

    /// Classify a packet (simplified)
    pub fn classify_packet(&self, _packet_info: &PacketInfo) -> Option<String> {
        // In a real implementation, this would match packet against rules
        // For now, return a default class
        Some("default".to_string())
    }
}

/// Simplified packet information for classification
#[derive(Debug, Clone)]
pub struct PacketInfo {
    pub source_ip: String,
    pub dest_ip: String,
    pub source_port: u16,
    pub dest_port: u16,
    pub protocol: u8,
    pub dscp: u8,
    pub size: usize,
}

/// Interface statistics
#[derive(Debug, Clone)]
pub struct InterfaceStats {
    pub name: String,
    pub max_bandwidth_mbps: u64,
    pub current_rx_mbps: f64,
    pub current_tx_mbps: f64,
    pub utilization: f64,
    pub packet_loss_rate: f64,
    pub latency_us: u64,
}

/// Traffic shaping statistics
#[derive(Debug, Clone)]
pub struct ShapingStats {
    pub interface_name: String,
    pub shaping_enabled: bool,
    pub queue_stats: HashMap<String, QueueStats>,
    pub total_packets_shaped: u64,
    pub total_packets_dropped: u64,
}

impl Default for NetworkBandwidthManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bandwidth_manager_initialization() {
        let manager = NetworkBandwidthManager::new();
        assert!(manager.interfaces.is_empty());
        assert!(manager.traffic_shapers.is_empty());
    }

    #[tokio::test]
    async fn test_interface_registration() {
        let mut manager = NetworkBandwidthManager::new();
        manager.register_interface("eth0".to_string(), 1000);
        
        assert_eq!(manager.interfaces.len(), 1);
        assert!(manager.interfaces.contains_key("eth0"));
        
        let interface = manager.interfaces.get("eth0").unwrap();
        assert_eq!(interface.max_bandwidth_mbps.load(Ordering::Relaxed), 1000);
    }

    #[tokio::test]
    async fn test_traffic_shaping_configuration() {
        let mut manager = NetworkBandwidthManager::new();
        manager.register_interface("eth0".to_string(), 1000);
        
        let config = TrafficShapingConfig {
            max_bandwidth_mbps: 800,
            burst_size_kb: 64,
            priority_queues: vec![
                QueueConfig {
                    name: "high_priority".to_string(),
                    priority: 0,
                    guaranteed_bandwidth_mbps: 200,
                    max_bandwidth_mbps: 400,
                    max_queue_size: 1000,
                    drop_policy: DropPolicy::TailDrop,
                },
            ],
            rate_limiting_enabled: true,
            congestion_control_enabled: true,
        };
        
        let result = manager.configure_traffic_shaping("eth0".to_string(), config).await;
        assert!(result.is_ok());
        assert!(manager.traffic_shapers.contains_key("eth0"));
    }

    #[tokio::test]
    async fn test_token_bucket() {
        let bucket = TokenBucket::new(1000, 100); // 1000 capacity, 100 tokens/sec
        
        // Should be able to consume initial tokens
        assert!(bucket.try_consume(500));
        assert!(bucket.try_consume(500));
        assert!(!bucket.try_consume(1)); // Should fail, no tokens left
        
        // Wait and refill
        tokio::time::sleep(Duration::from_millis(100)).await;
        bucket.refill().await;
        
        // Should have some tokens now
        assert!(bucket.try_consume(10));
    }

    #[tokio::test]
    async fn test_bandwidth_monitoring() {
        let monitor = BandwidthMonitor::new();
        
        let measurement = BandwidthMeasurement {
            interface_name: "eth0".to_string(),
            rx_mbps: 100.0,
            tx_mbps: 50.0,
            utilization: 0.15,
            packet_loss_rate: 0.001,
            latency_us: 5000,
            timestamp: Instant::now(),
        };
        
        monitor.record_measurement(measurement).await;
        
        let history = monitor.get_history("eth0", Duration::from_secs(60)).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].rx_mbps, 100.0);
    }

    #[tokio::test]
    async fn test_congestion_detection() {
        let controller = CongestionController::new();
        
        controller.update_congestion_state("eth0").await;
        
        let state_map = controller.congestion_state.read().await;
        assert!(state_map.contains_key("eth0"));
        
        let state = state_map.get("eth0").unwrap();
        assert!(matches!(state.level, CongestionLevel::None | CongestionLevel::Light | CongestionLevel::Moderate | CongestionLevel::Heavy | CongestionLevel::Severe));
    }

    #[test]
    fn test_qos_manager() {
        let mut qos_manager = QosManager::new();
        
        let traffic_class = TrafficClass {
            name: "high_priority".to_string(),
            priority: 0,
            guaranteed_bandwidth_mbps: 100,
            max_bandwidth_mbps: 200,
            latency_target_us: 1000,
            jitter_target_us: 100,
        };
        
        qos_manager.add_traffic_class(traffic_class);
        assert!(qos_manager.traffic_classes.contains_key("high_priority"));
        
        let rule = ClassificationRule {
            name: "trading_traffic".to_string(),
            source_ip_range: Some("192.168.1.0/24".to_string()),
            dest_ip_range: None,
            source_port_range: Some((8000, 9000)),
            dest_port_range: None,
            protocol: Some(6), // TCP
            dscp_marking: Some(46), // EF
            traffic_class: "high_priority".to_string(),
        };
        
        qos_manager.add_classification_rule(rule);
        assert_eq!(qos_manager.classification_rules.len(), 1);
    }
}