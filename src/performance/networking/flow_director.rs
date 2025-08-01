use super::zero_copy::{ZeroCopyPacket, PacketType, PacketMetadata, PacketParseError};
use super::ring_buffers::{RingBuffer, RingBufferError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::HashMap;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::hash::{Hash, Hasher};
use std::time::Duration;

/// Hardware flow director for high-performance packet classification and routing
/// Provides RSS-style packet distribution and QoS management
pub struct FlowDirector {
    /// Flow classification rules
    rules: Vec<FlowRule>,
    
    /// Output queues for different flow types
    queues: HashMap<QueueId, Arc<RingBuffer<ZeroCopyPacket>>>,
    
    /// RSS hash configuration
    rss_config: RssConfig,
    
    /// QoS configuration
    qos_config: QosConfig,
    
    /// Flow statistics
    stats: FlowDirectorStats,
    
    /// Default queue for unmatched flows
    default_queue: QueueId,
    
    /// Flow table for connection tracking
    flow_table: FlowTable,
}

/// Flow classification rule
#[derive(Debug, Clone)]
pub struct FlowRule {
    /// Rule priority (higher number = higher priority)
    pub priority: u32,
    
    /// Rule ID for management
    pub rule_id: u32,
    
    /// Flow matcher
    pub matcher: FlowMatcher,
    
    /// Action to take when rule matches
    pub action: FlowAction,
    
    /// Rule statistics
    pub stats: FlowRuleStats,
}

/// Flow matching criteria
#[derive(Debug, Clone)]
pub struct FlowMatcher {
    /// Source IP address range
    pub src_ip: Option<IpRange>,
    
    /// Destination IP address range
    pub dst_ip: Option<IpRange>,
    
    /// Source port range
    pub src_port: Option<PortRange>,
    
    /// Destination port range
    pub dst_port: Option<PortRange>,
    
    /// Protocol filter
    pub protocol: Option<u8>,
    
    /// VLAN ID filter
    pub vlan_id: Option<u16>,
    
    /// Packet type filter
    pub packet_type: Option<PacketType>,
    
    /// Custom payload pattern matching
    pub payload_pattern: Option<PayloadPattern>,
    
    /// Flow direction
    pub direction: FlowDirection,
}

/// IP address range for matching
#[derive(Debug, Clone)]
pub enum IpRange {
    V4 { addr: Ipv4Addr, mask: u32 },
    V6 { addr: Ipv6Addr, mask: u32 },
    V4Range { start: Ipv4Addr, end: Ipv4Addr },
    V6Range { start: Ipv6Addr, end: Ipv6Addr },
}

/// Port range for matching
#[derive(Debug, Clone)]
pub struct PortRange {
    pub start: u16,
    pub end: u16,
}

/// Payload pattern matching
#[derive(Debug, Clone)]
pub struct PayloadPattern {
    /// Pattern bytes to match
    pub pattern: Vec<u8>,
    
    /// Offset in payload to start matching
    pub offset: usize,
    
    /// Mask for pattern matching (optional)
    pub mask: Option<Vec<u8>>,
}

/// Flow direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowDirection {
    Ingress,
    Egress,
    Bidirectional,
}

/// Action to take when flow rule matches
#[derive(Debug, Clone)]
pub enum FlowAction {
    /// Route to specific queue
    RouteToQueue(QueueId),
    
    /// Drop the packet
    Drop,
    
    /// Mark packet with priority
    SetPriority(u8),
    
    /// Rate limit the flow
    RateLimit { rate_mbps: u32, burst_size: u32 },
    
    /// Mirror packet to another queue
    Mirror(QueueId),
    
    /// Redirect to different destination
    Redirect { queue: QueueId, modify_headers: bool },
    
    /// Apply multiple actions
    Multiple(Vec<FlowAction>),
}

/// Queue identifier
pub type QueueId = u16;

/// RSS (Receive Side Scaling) configuration
#[derive(Debug, Clone)]
pub struct RssConfig {
    /// RSS hash key
    pub hash_key: [u8; 40],
    
    /// RSS hash function
    pub hash_function: RssHashFunction,
    
    /// RSS indirection table
    pub indirection_table: Vec<QueueId>,
    
    /// Hash fields to include
    pub hash_fields: RssHashFields,
}

/// RSS hash function types
#[derive(Debug, Clone, Copy)]
pub enum RssHashFunction {
    Toeplitz,
    Crc32,
    Xxhash,
    Custom,
}

/// RSS hash fields configuration
#[derive(Debug, Clone)]
pub struct RssHashFields {
    pub ipv4_src: bool,
    pub ipv4_dst: bool,
    pub ipv6_src: bool,
    pub ipv6_dst: bool,
    pub tcp_src_port: bool,
    pub tcp_dst_port: bool,
    pub udp_src_port: bool,
    pub udp_dst_port: bool,
}

/// Quality of Service configuration
#[derive(Debug, Clone)]
pub struct QosConfig {
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
    
    /// Scheduling algorithm
    pub scheduler: QosScheduler,
    
    /// Rate limiting configuration
    pub rate_limits: HashMap<QueueId, RateLimit>,
}

/// Traffic class definition
#[derive(Debug, Clone)]
pub struct TrafficClass {
    pub class_id: u8,
    pub name: String,
    pub priority: u8,
    pub bandwidth_percent: u8,
    pub max_latency_us: u32,
    pub queue_ids: Vec<QueueId>,
}

/// QoS scheduling algorithms
#[derive(Debug, Clone, Copy)]
pub enum QosScheduler {
    StrictPriority,
    WeightedRoundRobin,
    DeficitRoundRobin,
    WeightedFairQueuing,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub rate_mbps: u32,
    pub burst_size: u32,
    pub current_tokens: AtomicU64,
    pub last_refill: AtomicU64,
}

/// Flow table for connection tracking
pub struct FlowTable {
    /// Active flows
    flows: HashMap<FlowKey, FlowEntry>,
    
    /// Flow timeout
    timeout_seconds: u64,
    
    /// Maximum number of flows
    max_flows: usize,
    
    /// Flow table statistics
    stats: FlowTableStats,
}

/// Flow key for connection tracking
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FlowKey {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
}

/// IP address enum for flow keys
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum IpAddr {
    V4(Ipv4Addr),
    V6(Ipv6Addr),
}

/// Flow entry in the flow table
#[derive(Debug, Clone)]
pub struct FlowEntry {
    /// Flow key
    pub key: FlowKey,
    
    /// Assigned queue
    pub queue_id: QueueId,
    
    /// Flow statistics
    pub packet_count: u64,
    pub byte_count: u64,
    pub first_seen: u64,
    pub last_seen: u64,
    
    /// Flow state
    pub state: FlowState,
}

/// Flow state tracking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowState {
    New,
    Established,
    Closing,
    Closed,
}

/// Flow director statistics
#[repr(align(64))]
pub struct FlowDirectorStats {
    /// Total packets processed
    pub packets_processed: AtomicU64,
    
    /// Packets matched by rules
    pub packets_matched: AtomicU64,
    
    /// Packets using default queue
    pub packets_default: AtomicU64,
    
    /// Packets dropped
    pub packets_dropped: AtomicU64,
    
    /// Classification errors
    pub classification_errors: AtomicU64,
    
    /// Average classification latency (nanoseconds)
    pub avg_classification_latency_ns: AtomicU64,
    
    /// RSS hash collisions
    pub rss_hash_collisions: AtomicU64,
    
    /// Flow table hits
    pub flow_table_hits: AtomicU64,
    
    /// Flow table misses
    pub flow_table_misses: AtomicU64,
}

/// Flow rule statistics
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct FlowRuleStats {
    /// Packets matched by this rule
    pub packets_matched: AtomicU64,
    
    /// Bytes matched by this rule
    pub bytes_matched: AtomicU64,
    
    /// Last match timestamp
    pub last_match: AtomicU64,
}

/// Flow table statistics
#[repr(align(64))]
pub struct FlowTableStats {
    /// Active flows
    pub active_flows: AtomicUsize,
    
    /// Flow insertions
    pub flow_insertions: AtomicU64,
    
    /// Flow deletions
    pub flow_deletions: AtomicU64,
    
    /// Flow timeouts
    pub flow_timeouts: AtomicU64,
    
    /// Table full events
    pub table_full_events: AtomicU64,
}

impl FlowDirector {
    /// Create a new flow director
    pub fn new(
        queues: HashMap<QueueId, Arc<RingBuffer<ZeroCopyPacket>>>,
        default_queue: QueueId,
        rss_config: RssConfig,
        qos_config: QosConfig,
    ) -> Self {
        Self {
            rules: Vec::new(),
            queues,
            rss_config,
            qos_config,
            stats: FlowDirectorStats::new(),
            default_queue,
            flow_table: FlowTable::new(10000, 300), // 10k flows, 5min timeout
        }
    }

    /// Add a flow classification rule
    pub fn add_rule(&mut self, rule: FlowRule) -> Result<(), FlowDirectorError> {
        // Validate rule
        if self.rules.iter().any(|r| r.rule_id == rule.rule_id) {
            return Err(FlowDirectorError::DuplicateRuleId(rule.rule_id));
        }

        // Insert rule in priority order (higher priority first)
        let insert_pos = self.rules
            .binary_search_by(|r| rule.priority.cmp(&r.priority))
            .unwrap_or_else(|pos| pos);
        
        self.rules.insert(insert_pos, rule);
        Ok(())
    }

    /// Remove a flow classification rule
    pub fn remove_rule(&mut self, rule_id: u32) -> Result<FlowRule, FlowDirectorError> {
        let pos = self.rules
            .iter()
            .position(|r| r.rule_id == rule_id)
            .ok_or(FlowDirectorError::RuleNotFound(rule_id))?;
        
        Ok(self.rules.remove(pos))
    }

    /// Classify and route a packet
    pub fn classify_packet(&self, mut packet: ZeroCopyPacket) -> Result<QueueId, FlowDirectorError> {
        let start_time = Self::get_timestamp_ns();
        
        // Parse packet headers if not already done
        if packet.metadata().packet_type == PacketType::Unknown {
            packet.parse_headers()
                .map_err(|e| FlowDirectorError::PacketParseError(e))?;
        }

        // Extract flow key for connection tracking
        let flow_key = self.extract_flow_key(&packet)?;
        
        // Check flow table first for existing flows
        if let Some(queue_id) = self.flow_table.lookup(&flow_key) {
            self.stats.flow_table_hits.fetch_add(1, Ordering::Relaxed);
            self.update_classification_latency(start_time);
            return Ok(queue_id);
        }
        
        self.stats.flow_table_misses.fetch_add(1, Ordering::Relaxed);

        // Apply flow rules in priority order
        for rule in &self.rules {
            if self.match_rule(rule, &packet)? {
                let queue_id = self.apply_action(&rule.action, &packet)?;
                
                // Update rule statistics
                rule.stats.packets_matched.fetch_add(1, Ordering::Relaxed);
                rule.stats.bytes_matched.fetch_add(packet.len() as u64, Ordering::Relaxed);
                rule.stats.last_match.store(start_time, Ordering::Relaxed);
                
                // Add to flow table
                self.flow_table.insert(flow_key, queue_id);
                
                self.stats.packets_matched.fetch_add(1, Ordering::Relaxed);
                self.update_classification_latency(start_time);
                return Ok(queue_id);
            }
        }

        // No rule matched, use RSS or default queue
        let queue_id = self.apply_rss_classification(&packet)?;
        
        // Add to flow table
        self.flow_table.insert(flow_key, queue_id);
        
        self.stats.packets_default.fetch_add(1, Ordering::Relaxed);
        self.update_classification_latency(start_time);
        Ok(queue_id)
    }

    /// Process and route a packet to the appropriate queue
    pub fn process_packet(&self, packet: ZeroCopyPacket) -> Result<(), FlowDirectorError> {
        self.stats.packets_processed.fetch_add(1, Ordering::Relaxed);
        
        let queue_id = self.classify_packet(packet.clone())?;
        
        // Get the target queue
        let queue = self.queues.get(&queue_id)
            .ok_or(FlowDirectorError::QueueNotFound(queue_id))?;
        
        // Enqueue the packet
        queue.enqueue(packet)
            .map_err(|e| FlowDirectorError::QueueError(e))?;
        
        Ok(())
    }

    /// Extract flow key from packet for connection tracking
    fn extract_flow_key(&self, packet: &ZeroCopyPacket) -> Result<FlowKey, FlowDirectorError> {
        let metadata = packet.metadata();
        
        // Extract IP addresses
        let (src_ip, dst_ip) = match metadata.packet_type {
            PacketType::IPv4Tcp | PacketType::IPv4Udp | PacketType::IPv4Icmp => {
                let ip_header = packet.get_header(super::zero_copy::HeaderLayer::L3)
                    .map_err(|e| FlowDirectorError::PacketParseError(e))?;
                
                if ip_header.len() < 20 {
                    return Err(FlowDirectorError::InvalidPacket);
                }
                
                let src = Ipv4Addr::new(ip_header[12], ip_header[13], ip_header[14], ip_header[15]);
                let dst = Ipv4Addr::new(ip_header[16], ip_header[17], ip_header[18], ip_header[19]);
                
                (IpAddr::V4(src), IpAddr::V4(dst))
            }
            PacketType::IPv6Tcp | PacketType::IPv6Udp | PacketType::IPv6Icmp => {
                let ip_header = packet.get_header(super::zero_copy::HeaderLayer::L3)
                    .map_err(|e| FlowDirectorError::PacketParseError(e))?;
                
                if ip_header.len() < 40 {
                    return Err(FlowDirectorError::InvalidPacket);
                }
                
                let src_bytes: [u8; 16] = ip_header[8..24].try_into()
                    .map_err(|_| FlowDirectorError::InvalidPacket)?;
                let dst_bytes: [u8; 16] = ip_header[24..40].try_into()
                    .map_err(|_| FlowDirectorError::InvalidPacket)?;
                
                let src = Ipv6Addr::from(src_bytes);
                let dst = Ipv6Addr::from(dst_bytes);
                
                (IpAddr::V6(src), IpAddr::V6(dst))
            }
            _ => return Err(FlowDirectorError::UnsupportedPacketType),
        };

        // Extract ports for TCP/UDP
        let (src_port, dst_port) = match metadata.packet_type {
            PacketType::IPv4Tcp | PacketType::IPv6Tcp |
            PacketType::IPv4Udp | PacketType::IPv6Udp => {
                let l4_header = packet.get_header(super::zero_copy::HeaderLayer::L4)
                    .map_err(|e| FlowDirectorError::PacketParseError(e))?;
                
                if l4_header.len() < 4 {
                    return Err(FlowDirectorError::InvalidPacket);
                }
                
                let src_port = u16::from_be_bytes([l4_header[0], l4_header[1]]);
                let dst_port = u16::from_be_bytes([l4_header[2], l4_header[3]]);
                
                (src_port, dst_port)
            }
            _ => (0, 0), // No ports for ICMP, etc.
        };

        // Extract protocol
        let protocol = match metadata.packet_type {
            PacketType::IPv4Tcp | PacketType::IPv6Tcp => 6,
            PacketType::IPv4Udp | PacketType::IPv6Udp => 17,
            PacketType::IPv4Icmp => 1,
            PacketType::IPv6Icmp => 58,
            _ => 0,
        };

        Ok(FlowKey {
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
        })
    }

    /// Check if a packet matches a flow rule
    fn match_rule(&self, rule: &FlowRule, packet: &ZeroCopyPacket) -> Result<bool, FlowDirectorError> {
        let metadata = packet.metadata();
        let matcher = &rule.matcher;

        // Check packet type
        if let Some(expected_type) = matcher.packet_type {
            if metadata.packet_type != expected_type {
                return Ok(false);
            }
        }

        // Check protocol
        if let Some(expected_protocol) = matcher.protocol {
            let packet_protocol = match metadata.packet_type {
                PacketType::IPv4Tcp | PacketType::IPv6Tcp => 6,
                PacketType::IPv4Udp | PacketType::IPv6Udp => 17,
                PacketType::IPv4Icmp => 1,
                PacketType::IPv6Icmp => 58,
                _ => return Ok(false),
            };
            
            if packet_protocol != expected_protocol {
                return Ok(false);
            }
        }

        // Check VLAN ID
        if let Some(expected_vlan) = matcher.vlan_id {
            match metadata.vlan_tag {
                Some(vlan) if vlan == expected_vlan => {},
                _ => return Ok(false),
            }
        }

        // Extract flow key for IP/port matching
        let flow_key = self.extract_flow_key(packet)?;

        // Check source IP
        if let Some(ref src_range) = matcher.src_ip {
            if !self.ip_matches_range(&flow_key.src_ip, src_range) {
                return Ok(false);
            }
        }

        // Check destination IP
        if let Some(ref dst_range) = matcher.dst_ip {
            if !self.ip_matches_range(&flow_key.dst_ip, dst_range) {
                return Ok(false);
            }
        }

        // Check source port
        if let Some(ref src_port_range) = matcher.src_port {
            if !self.port_in_range(flow_key.src_port, src_port_range) {
                return Ok(false);
            }
        }

        // Check destination port
        if let Some(ref dst_port_range) = matcher.dst_port {
            if !self.port_in_range(flow_key.dst_port, dst_port_range) {
                return Ok(false);
            }
        }

        // Check payload pattern
        if let Some(ref pattern) = matcher.payload_pattern {
            if !self.payload_matches_pattern(packet, pattern)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if IP address matches range
    fn ip_matches_range(&self, ip: &IpAddr, range: &IpRange) -> bool {
        match (ip, range) {
            (IpAddr::V4(addr), IpRange::V4 { addr: range_addr, mask }) => {
                let addr_u32 = u32::from(*addr);
                let range_u32 = u32::from(*range_addr);
                let mask_bits = 32 - mask;
                (addr_u32 >> mask_bits) == (range_u32 >> mask_bits)
            }
            (IpAddr::V4(addr), IpRange::V4Range { start, end }) => {
                addr >= start && addr <= end
            }
            (IpAddr::V6(addr), IpRange::V6 { addr: range_addr, mask }) => {
                let addr_bytes = addr.octets();
                let range_bytes = range_addr.octets();
                let mask_bytes = *mask / 8;
                let mask_bits = *mask % 8;
                
                // Compare full bytes
                if addr_bytes[..mask_bytes as usize] != range_bytes[..mask_bytes as usize] {
                    return false;
                }
                
                // Compare partial byte if needed
                if mask_bits > 0 && mask_bytes < 16 {
                    let shift = 8 - mask_bits;
                    let addr_partial = addr_bytes[mask_bytes as usize] >> shift;
                    let range_partial = range_bytes[mask_bytes as usize] >> shift;
                    if addr_partial != range_partial {
                        return false;
                    }
                }
                
                true
            }
            (IpAddr::V6(addr), IpRange::V6Range { start, end }) => {
                addr >= start && addr <= end
            }
            _ => false, // IPv4 vs IPv6 mismatch
        }
    }

    /// Check if port is in range
    fn port_in_range(&self, port: u16, range: &PortRange) -> bool {
        port >= range.start && port <= range.end
    }

    /// Check if payload matches pattern
    fn payload_matches_pattern(&self, packet: &ZeroCopyPacket, pattern: &PayloadPattern) -> Result<bool, FlowDirectorError> {
        let payload = packet.payload()
            .map_err(|e| FlowDirectorError::PacketParseError(e))?;
        
        if payload.len() < pattern.offset + pattern.pattern.len() {
            return Ok(false);
        }
        
        let payload_slice = &payload[pattern.offset..pattern.offset + pattern.pattern.len()];
        
        if let Some(ref mask) = pattern.mask {
            if mask.len() != pattern.pattern.len() {
                return Err(FlowDirectorError::InvalidPattern);
            }
            
            for i in 0..pattern.pattern.len() {
                if (payload_slice[i] & mask[i]) != (pattern.pattern[i] & mask[i]) {
                    return Ok(false);
                }
            }
        } else {
            if payload_slice != pattern.pattern.as_slice() {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    /// Apply flow action and return target queue
    fn apply_action(&self, action: &FlowAction, packet: &ZeroCopyPacket) -> Result<QueueId, FlowDirectorError> {
        match action {
            FlowAction::RouteToQueue(queue_id) => Ok(*queue_id),
            FlowAction::Drop => {
                self.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                Err(FlowDirectorError::PacketDropped)
            }
            FlowAction::SetPriority(_priority) => {
                // Priority would be set in packet metadata
                // For now, route to default queue
                Ok(self.default_queue)
            }
            FlowAction::RateLimit { rate_mbps: _, burst_size: _ } => {
                // Rate limiting would be implemented here
                // For now, route to default queue
                Ok(self.default_queue)
            }
            FlowAction::Mirror(queue_id) => {
                // Mirroring would duplicate the packet
                // For now, route to mirror queue
                Ok(*queue_id)
            }
            FlowAction::Redirect { queue, modify_headers: _ } => {
                // Header modification would be done here
                Ok(*queue)
            }
            FlowAction::Multiple(actions) => {
                // Apply first routing action found
                for action in actions {
                    match action {
                        FlowAction::RouteToQueue(queue_id) => return Ok(*queue_id),
                        FlowAction::Redirect { queue, .. } => return Ok(*queue),
                        FlowAction::Mirror(queue_id) => return Ok(*queue_id),
                        _ => continue,
                    }
                }
                Ok(self.default_queue)
            }
        }
    }

    /// Apply RSS classification for unmatched packets
    fn apply_rss_classification(&self, packet: &ZeroCopyPacket) -> Result<QueueId, FlowDirectorError> {
        // Calculate RSS hash
        let hash = self.calculate_rss_hash(packet)?;
        
        // Use indirection table to map hash to queue
        let table_index = (hash as usize) % self.rss_config.indirection_table.len();
        let queue_id = self.rss_config.indirection_table[table_index];
        
        // Verify queue exists
        if !self.queues.contains_key(&queue_id) {
            return Ok(self.default_queue);
        }
        
        Ok(queue_id)
    }

    /// Calculate RSS hash for packet
    fn calculate_rss_hash(&self, packet: &ZeroCopyPacket) -> Result<u32, FlowDirectorError> {
        let flow_key = self.extract_flow_key(packet)?;
        let fields = &self.rss_config.hash_fields;
        
        let mut hash_input = Vec::new();
        
        // Add IP addresses to hash input
        match (&flow_key.src_ip, &flow_key.dst_ip) {
            (IpAddr::V4(src), IpAddr::V4(dst)) => {
                if fields.ipv4_src {
                    hash_input.extend_from_slice(&src.octets());
                }
                if fields.ipv4_dst {
                    hash_input.extend_from_slice(&dst.octets());
                }
            }
            (IpAddr::V6(src), IpAddr::V6(dst)) => {
                if fields.ipv6_src {
                    hash_input.extend_from_slice(&src.octets());
                }
                if fields.ipv6_dst {
                    hash_input.extend_from_slice(&dst.octets());
                }
            }
            _ => {} // Mixed IPv4/IPv6 not supported for RSS
        }
        
        // Add ports to hash input
        match packet.metadata().packet_type {
            PacketType::IPv4Tcp | PacketType::IPv6Tcp => {
                if fields.tcp_src_port {
                    hash_input.extend_from_slice(&flow_key.src_port.to_be_bytes());
                }
                if fields.tcp_dst_port {
                    hash_input.extend_from_slice(&flow_key.dst_port.to_be_bytes());
                }
            }
            PacketType::IPv4Udp | PacketType::IPv6Udp => {
                if fields.udp_src_port {
                    hash_input.extend_from_slice(&flow_key.src_port.to_be_bytes());
                }
                if fields.udp_dst_port {
                    hash_input.extend_from_slice(&flow_key.dst_port.to_be_bytes());
                }
            }
            _ => {}
        }
        
        // Calculate hash based on configured function
        let hash = match self.rss_config.hash_function {
            RssHashFunction::Toeplitz => self.toeplitz_hash(&hash_input),
            RssHashFunction::Crc32 => self.crc32_hash(&hash_input),
            RssHashFunction::Xxhash => self.xxhash(&hash_input),
            RssHashFunction::Custom => self.custom_hash(&hash_input),
        };
        
        Ok(hash)
    }

    /// Toeplitz hash implementation
    fn toeplitz_hash(&self, input: &[u8]) -> u32 {
        let key = &self.rss_config.hash_key;
        let mut hash = 0u32;
        let mut key_index = 0;
        
        for &byte in input {
            for bit in 0..8 {
                if (byte >> (7 - bit)) & 1 == 1 {
                    let key_bit_index = key_index + bit;
                    if key_bit_index < key.len() * 8 {
                        let key_byte = key[key_bit_index / 8];
                        let key_bit = (key_byte >> (7 - (key_bit_index % 8))) & 1;
                        hash ^= (key_bit as u32) << (31 - (hash.leading_zeros() % 32));
                    }
                }
            }
            key_index += 8;
        }
        
        hash
    }

    /// CRC32 hash implementation
    fn crc32_hash(&self, input: &[u8]) -> u32 {
        const CRC32_POLY: u32 = 0xEDB88320;
        let mut crc = 0xFFFFFFFF;
        
        for &byte in input {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 == 1 {
                    crc = (crc >> 1) ^ CRC32_POLY;
                } else {
                    crc >>= 1;
                }
            }
        }
        
        !crc
    }

    /// xxHash implementation (simplified)
    fn xxhash(&self, input: &[u8]) -> u32 {
        const PRIME1: u32 = 0x9E3779B1;
        const PRIME2: u32 = 0x85EBCA77;
        const PRIME3: u32 = 0xC2B2AE3D;
        
        let mut hash = PRIME1;
        
        for &byte in input {
            hash = hash.wrapping_add(byte as u32).wrapping_mul(PRIME2);
            hash = hash.rotate_left(13).wrapping_mul(PRIME3);
        }
        
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(PRIME2);
        hash ^= hash >> 13;
        hash = hash.wrapping_mul(PRIME3);
        hash ^= hash >> 16;
        
        hash
    }

    /// Custom hash implementation
    fn custom_hash(&self, input: &[u8]) -> u32 {
        // Simple FNV-1a hash
        const FNV_OFFSET_BASIS: u32 = 0x811C9DC5;
        const FNV_PRIME: u32 = 0x01000193;
        
        let mut hash = FNV_OFFSET_BASIS;
        for &byte in input {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        
        hash
    }

    /// Update classification latency statistics
    fn update_classification_latency(&self, start_time: u64) {
        let latency = Self::get_timestamp_ns() - start_time;
        
        // Update average using exponential moving average
        loop {
            let current_avg = self.stats.avg_classification_latency_ns.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                latency
            } else {
                // EMA with alpha = 0.1
                ((current_avg as f64 * 0.9) + (latency as f64 * 0.1)) as u64
            };
            
            match self.stats.avg_classification_latency_ns.compare_exchange_weak(
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

    /// Get flow director statistics
    pub fn get_stats(&self) -> FlowDirectorStatsSnapshot {
        FlowDirectorStatsSnapshot {
            packets_processed: self.stats.packets_processed.load(Ordering::Acquire),
            packets_matched: self.stats.packets_matched.load(Ordering::Acquire),
            packets_default: self.stats.packets_default.load(Ordering::Acquire),
            packets_dropped: self.stats.packets_dropped.load(Ordering::Acquire),
            classification_errors: self.stats.classification_errors.load(Ordering::Acquire),
            avg_classification_latency_ns: self.stats.avg_classification_latency_ns.load(Ordering::Acquire),
            rss_hash_collisions: self.stats.rss_hash_collisions.load(Ordering::Acquire),
            flow_table_hits: self.stats.flow_table_hits.load(Ordering::Acquire),
            flow_table_misses: self.stats.flow_table_misses.load(Ordering::Acquire),
            active_rules: self.rules.len(),
            active_queues: self.queues.len(),
            flow_table_stats: self.flow_table.get_stats(),
        }
    }

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}

impl FlowTable {
    /// Create a new flow table
    pub fn new(max_flows: usize, timeout_seconds: u64) -> Self {
        Self {
            flows: HashMap::with_capacity(max_flows),
            timeout_seconds,
            max_flows,
            stats: FlowTableStats::new(),
        }
    }

    /// Lookup a flow in the table
    pub fn lookup(&self, key: &FlowKey) -> Option<QueueId> {
        self.flows.get(key).map(|entry| entry.queue_id)
    }

    /// Insert a new flow into the table
    pub fn insert(&mut self, key: FlowKey, queue_id: QueueId) {
        let now = FlowDirector::get_timestamp_ns();
        
        // Check if table is full
        if self.flows.len() >= self.max_flows {
            self.stats.table_full_events.fetch_add(1, Ordering::Relaxed);
            // Could implement LRU eviction here
            return;
        }
        
        let entry = FlowEntry {
            key: key.clone(),
            queue_id,
            packet_count: 1,
            byte_count: 0,
            first_seen: now,
            last_seen: now,
            state: FlowState::New,
        };
        
        self.flows.insert(key, entry);
        self.stats.flow_insertions.fetch_add(1, Ordering::Relaxed);
        self.stats.active_flows.store(self.flows.len(), Ordering::Release);
    }

    /// Remove expired flows
    pub fn cleanup_expired_flows(&mut self) {
        let now = FlowDirector::get_timestamp_ns();
        let timeout_ns = self.timeout_seconds * 1_000_000_000;
        
        let initial_count = self.flows.len();
        self.flows.retain(|_, entry| {
            if now - entry.last_seen > timeout_ns {
                false
            } else {
                true
            }
        });
        
        let removed = initial_count - self.flows.len();
        if removed > 0 {
            self.stats.flow_timeouts.fetch_add(removed as u64, Ordering::Relaxed);
            self.stats.flow_deletions.fetch_add(removed as u64, Ordering::Relaxed);
            self.stats.active_flows.store(self.flows.len(), Ordering::Release);
        }
    }

    /// Get flow table statistics
    pub fn get_stats(&self) -> FlowTableStatsSnapshot {
        FlowTableStatsSnapshot {
            active_flows: self.stats.active_flows.load(Ordering::Acquire),
            flow_insertions: self.stats.flow_insertions.load(Ordering::Acquire),
            flow_deletions: self.stats.flow_deletions.load(Ordering::Acquire),
            flow_timeouts: self.stats.flow_timeouts.load(Ordering::Acquire),
            table_full_events: self.stats.table_full_events.load(Ordering::Acquire),
            max_flows: self.max_flows,
            timeout_seconds: self.timeout_seconds,
        }
    }
}

impl FlowDirectorStats {
    fn new() -> Self {
        Self {
            packets_processed: AtomicU64::new(0),
            packets_matched: AtomicU64::new(0),
            packets_default: AtomicU64::new(0),
            packets_dropped: AtomicU64::new(0),
            classification_errors: AtomicU64::new(0),
            avg_classification_latency_ns: AtomicU64::new(0),
            rss_hash_collisions: AtomicU64::new(0),
            flow_table_hits: AtomicU64::new(0),
            flow_table_misses: AtomicU64::new(0),
        }
    }
}

impl FlowRuleStats {
    pub fn new() -> Self {
        Self {
            packets_matched: AtomicU64::new(0),
            bytes_matched: AtomicU64::new(0),
            last_match: AtomicU64::new(0),
        }
    }
}

impl FlowTableStats {
    fn new() -> Self {
        Self {
            active_flows: AtomicUsize::new(0),
            flow_insertions: AtomicU64::new(0),
            flow_deletions: AtomicU64::new(0),
            flow_timeouts: AtomicU64::new(0),
            table_full_events: AtomicU64::new(0),
        }
    }
}

/// Flow director statistics snapshot
#[derive(Debug, Clone)]
pub struct FlowDirectorStatsSnapshot {
    pub packets_processed: u64,
    pub packets_matched: u64,
    pub packets_default: u64,
    pub packets_dropped: u64,
    pub classification_errors: u64,
    pub avg_classification_latency_ns: u64,
    pub rss_hash_collisions: u64,
    pub flow_table_hits: u64,
    pub flow_table_misses: u64,
    pub active_rules: usize,
    pub active_queues: usize,
    pub flow_table_stats: FlowTableStatsSnapshot,
}

/// Flow table statistics snapshot
#[derive(Debug, Clone)]
pub struct FlowTableStatsSnapshot {
    pub active_flows: usize,
    pub flow_insertions: u64,
    pub flow_deletions: u64,
    pub flow_timeouts: u64,
    pub table_full_events: u64,
    pub max_flows: usize,
    pub timeout_seconds: u64,
}

/// Flow director error types
#[derive(Debug, Clone)]
pub enum FlowDirectorError {
    DuplicateRuleId(u32),
    RuleNotFound(u32),
    QueueNotFound(QueueId),
    PacketParseError(PacketParseError),
    QueueError(RingBufferError),
    InvalidPacket,
    UnsupportedPacketType,
    InvalidPattern,
    PacketDropped,
    ConfigurationError(String),
}

impl std::fmt::Display for FlowDirectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlowDirectorError::DuplicateRuleId(id) => write!(f, "Duplicate rule ID: {}", id),
            FlowDirectorError::RuleNotFound(id) => write!(f, "Rule not found: {}", id),
            FlowDirectorError::QueueNotFound(id) => write!(f, "Queue not found: {}", id),
            FlowDirectorError::PacketParseError(e) => write!(f, "Packet parse error: {}", e),
            FlowDirectorError::QueueError(e) => write!(f, "Queue error: {}", e),
            FlowDirectorError::InvalidPacket => write!(f, "Invalid packet"),
            FlowDirectorError::UnsupportedPacketType => write!(f, "Unsupported packet type"),
            FlowDirectorError::InvalidPattern => write!(f, "Invalid pattern"),
            FlowDirectorError::PacketDropped => write!(f, "Packet dropped by rule"),
            FlowDirectorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for FlowDirectorError {}

/// Default RSS configuration
impl Default for RssConfig {
    fn default() -> Self {
        Self {
            hash_key: [
                0x6d, 0x5a, 0x56, 0xda, 0x25, 0x5b, 0x0e, 0xc2,
                0x41, 0x67, 0x25, 0x3d, 0x43, 0xa3, 0x8f, 0xb0,
                0xd0, 0xca, 0x2b, 0xcb, 0xae, 0x7b, 0x30, 0xb4,
                0x77, 0xcb, 0x2d, 0xa3, 0x80, 0x30, 0xf2, 0x0c,
                0x6a, 0x42, 0xb7, 0x3b, 0xbe, 0xac, 0x01, 0xfa,
            ],
            hash_function: RssHashFunction::Toeplitz,
            indirection_table: (0..128).map(|i| (i % 4) as QueueId).collect(),
            hash_fields: RssHashFields {
                ipv4_src: true,
                ipv4_dst: true,
                ipv6_src: true,
                ipv6_dst: true,
                tcp_src_port: true,
                tcp_dst_port: true,
                udp_src_port: true,
                udp_dst_port: true,
            },
        }
    }
}

/// Default QoS configuration
impl Default for QosConfig {
    fn default() -> Self {
        Self {
            traffic_classes: vec![
                TrafficClass {
                    class_id: 0,
                    name: "Best Effort".to_string(),
                    priority: 0,
                    bandwidth_percent: 50,
                    max_latency_us: 1000,
                    queue_ids: vec![0, 1],
                },
                TrafficClass {
                    class_id: 1,
                    name: "Low Latency".to_string(),
                    priority: 1,
                    bandwidth_percent: 30,
                    max_latency_us: 100,
                    queue_ids: vec![2, 3],
                },
                TrafficClass {
                    class_id: 2,
                    name: "High Priority".to_string(),
                    priority: 2,
                    bandwidth_percent: 20,
                    max_latency_us: 10,
                    queue_ids: vec![4, 5],
                },
            ],
            scheduler: QosScheduler::WeightedRoundRobin,
            rate_limits: HashMap::new(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::networking::zero_copy::{ZeroCopyPacket, PacketType};
    use std::net::{Ipv4Addr, Ipv6Addr};

    fn create_test_flow_director() -> FlowDirector {
        let mut queues = HashMap::new();
        queues.insert(0, Arc::new(RingBuffer::new(1024, "queue_0".to_string()).unwrap()));
        queues.insert(1, Arc::new(RingBuffer::new(1024, "queue_1".to_string()).unwrap()));
        queues.insert(2, Arc::new(RingBuffer::new(1024, "queue_2".to_string()).unwrap()));
        
        let default_queue = 0;
        let rss_config = RssConfig::default();
        let qos_config = QosConfig::default();

        FlowDirector::new(queues, default_queue, rss_config, qos_config)
    }

    fn create_test_ipv4_tcp_packet() -> ZeroCopyPacket {
        let mut buffer = vec![0u8; 64];
        
        // Ethernet header
        buffer[12..14].copy_from_slice(&[0x08, 0x00]); // IPv4
        
        // IPv4 header
        buffer[14] = 0x45; // Version 4, IHL 5
        buffer[23] = 6;    // Protocol TCP
        buffer[26..30].copy_from_slice(&[192, 168, 1, 1]); // Source IP
        buffer[30..34].copy_from_slice(&[192, 168, 1, 2]); // Dest IP
        
        // TCP header
        buffer[34..36].copy_from_slice(&[0x1F, 0x90]); // Source port 8080
        buffer[36..38].copy_from_slice(&[0x00, 0x50]); // Dest port 80
        buffer[46] = 0x50; // Data offset 5 (20 bytes)
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 64);
        packet.parse_headers().unwrap();
        packet
    }

    fn create_test_ipv4_udp_packet() -> ZeroCopyPacket {
        let mut buffer = vec![0u8; 42];
        
        // Ethernet header
        buffer[12..14].copy_from_slice(&[0x08, 0x00]); // IPv4
        
        // IPv4 header
        buffer[14] = 0x45; // Version 4, IHL 5
        buffer[23] = 17;   // Protocol UDP
        buffer[26..30].copy_from_slice(&[10, 0, 0, 1]); // Source IP
        buffer[30..34].copy_from_slice(&[10, 0, 0, 2]); // Dest IP
        
        // UDP header
        buffer[34..36].copy_from_slice(&[0x27, 0x10]); // Source port 10000
        buffer[36..38].copy_from_slice(&[0x00, 0x35]); // Dest port 53 (DNS)
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 42);
        packet.parse_headers().unwrap();
        packet
    }

    #[test]
    fn test_flow_director_creation() {
        let flow_director = create_test_flow_director();
        assert_eq!(flow_director.default_queue, 0);
        assert_eq!(flow_director.queues.len(), 3);
        assert_eq!(flow_director.rules.len(), 0);
    }

    #[test]
    fn test_add_flow_rule() {
        let mut flow_director = create_test_flow_director();
        
        let rule = FlowRule {
            priority: 100,
            rule_id: 1,
            matcher: FlowMatcher {
                src_ip: Some(IpRange::V4 { 
                    addr: Ipv4Addr::new(192, 168, 1, 0), 
                    mask: 24 
                }),
                dst_ip: None,
                src_port: None,
                dst_port: Some(PortRange { start: 80, end: 80 }),
                protocol: Some(6), // TCP
                vlan_id: None,
                packet_type: Some(PacketType::IPv4Tcp),
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };

        assert!(flow_director.add_rule(rule).is_ok());
        assert_eq!(flow_director.rules.len(), 1);
    }

    #[test]
    fn test_duplicate_rule_id() {
        let mut flow_director = create_test_flow_director();
        
        let rule1 = FlowRule {
            priority: 100,
            rule_id: 1,
            matcher: FlowMatcher {
                src_ip: None,
                dst_ip: None,
                src_port: None,
                dst_port: None,
                protocol: None,
                vlan_id: None,
                packet_type: None,
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };

        let rule2 = FlowRule {
            priority: 200,
            rule_id: 1, // Same ID
            matcher: FlowMatcher {
                src_ip: None,
                dst_ip: None,
                src_port: None,
                dst_port: None,
                protocol: None,
                vlan_id: None,
                packet_type: None,
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(2),
            stats: FlowRuleStats::new(),
        };

        assert!(flow_director.add_rule(rule1).is_ok());
        assert!(matches!(
            flow_director.add_rule(rule2),
            Err(FlowDirectorError::DuplicateRuleId(1))
        ));
    }

    #[test]
    fn test_remove_flow_rule() {
        let mut flow_director = create_test_flow_director();
        
        let rule = FlowRule {
            priority: 100,
            rule_id: 1,
            matcher: FlowMatcher {
                src_ip: None,
                dst_ip: None,
                src_port: None,
                dst_port: None,
                protocol: None,
                vlan_id: None,
                packet_type: None,
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };

        flow_director.add_rule(rule).unwrap();
        assert_eq!(flow_director.rules.len(), 1);

        let removed_rule = flow_director.remove_rule(1).unwrap();
        assert_eq!(removed_rule.rule_id, 1);
        assert_eq!(flow_director.rules.len(), 0);
    }

    #[test]
    fn test_extract_flow_key_ipv4_tcp() {
        let flow_director = create_test_flow_director();
        let packet = create_test_ipv4_tcp_packet();
        
        let flow_key = flow_director.extract_flow_key(&packet).unwrap();
        
        assert_eq!(flow_key.src_ip, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(flow_key.dst_ip, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2)));
        assert_eq!(flow_key.src_port, 8080);
        assert_eq!(flow_key.dst_port, 80);
        assert_eq!(flow_key.protocol, 6);
    }

    #[test]
    fn test_extract_flow_key_ipv4_udp() {
        let flow_director = create_test_flow_director();
        let packet = create_test_ipv4_udp_packet();
        
        let flow_key = flow_director.extract_flow_key(&packet).unwrap();
        
        assert_eq!(flow_key.src_ip, IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        assert_eq!(flow_key.dst_ip, IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)));
        assert_eq!(flow_key.src_port, 10000);
        assert_eq!(flow_key.dst_port, 53);
        assert_eq!(flow_key.protocol, 17);
    }

    #[test]
    fn test_ip_range_matching() {
        let flow_director = create_test_flow_director();
        
        // Test IPv4 CIDR matching
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100));
        let range = IpRange::V4 { 
            addr: Ipv4Addr::new(192, 168, 1, 0), 
            mask: 24 
        };
        assert!(flow_director.ip_matches_range(&ip, &range));
        
        let ip_outside = IpAddr::V4(Ipv4Addr::new(192, 168, 2, 100));
        assert!(!flow_director.ip_matches_range(&ip_outside, &range));
        
        // Test IPv4 range matching
        let range = IpRange::V4Range { 
            start: Ipv4Addr::new(192, 168, 1, 1), 
            end: Ipv4Addr::new(192, 168, 1, 254) 
        };
        assert!(flow_director.ip_matches_range(&ip, &range));
        
        let ip_below = IpAddr::V4(Ipv4Addr::new(192, 168, 0, 255));
        assert!(!flow_director.ip_matches_range(&ip_below, &range));
    }

    #[test]
    fn test_port_range_matching() {
        let flow_director = create_test_flow_director();
        
        let range = PortRange { start: 80, end: 8080 };
        
        assert!(flow_director.port_in_range(80, &range));
        assert!(flow_director.port_in_range(443, &range));
        assert!(flow_director.port_in_range(8080, &range));
        assert!(!flow_director.port_in_range(79, &range));
        assert!(!flow_director.port_in_range(8081, &range));
    }

    #[test]
    fn test_payload_pattern_matching() {
        let flow_director = create_test_flow_director();
        
        // Create packet with specific payload
        let mut buffer = vec![0u8; 100];
        buffer[50..54].copy_from_slice(b"HTTP"); // Add pattern at offset 50
        let packet = ZeroCopyPacket::from_pooled_buffer(buffer, 100);
        
        // Test exact pattern match
        let pattern = PayloadPattern {
            pattern: b"HTTP".to_vec(),
            offset: 50,
            mask: None,
        };
        
        assert!(flow_director.payload_matches_pattern(&packet, &pattern).unwrap());
        
        // attern with mask
        let pattern_with_mask = PayloadPattern {
            pattern: vec![0x48, 0x54, 0x54, 0x50], // "HTTP"
            offset: 50,
            mask: Some(vec![0xFF, 0xFF, 0xFF, 0xFF]), // Exact match
        };
        
        assert!(flow_director.payload_matches_pattern(&packet, &pattern_with_mask).unwrap());
        
        // Test non-matching pattern
        let wrong_pattern = PayloadPattern {
            pattern: b"HTTPS".to_vec(),
            offset: 50,
            mask: None,
        };
        
        assert!(!flow_director.payload_matches_pattern(&packet, &wrong_pattern).unwrap());
    }

    #[test]
    fn test_rule_matching() {
        let flow_director = create_test_flow_director();
        let packet = create_test_ipv4_tcp_packet();
        
        // Rule that should match
        let matching_rule = FlowRule {
            priority: 100,
            rule_id: 1,
            matcher: FlowMatcher {
                src_ip: Some(IpRange::V4 { 
                    addr: Ipv4Addr::new(192, 168, 1, 0), 
                    mask: 24 
                }),
                dst_ip: None,
                src_port: None,
                dst_port: Some(PortRange { start: 80, end: 80 }),
                protocol: Some(6), // TCP
                vlan_id: None,
                packet_type: Some(PacketType::IPv4Tcp),
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };
        
        assert!(flow_director.match_rule(&matching_rule, &packet).unwrap());
        
        // Rule that should not match (wrong protocol)
        let non_matching_rule = FlowRule {
            priority: 100,
            rule_id: 2,
            matcher: FlowMatcher {
                src_ip: Some(IpRange::V4 { 
                    addr: Ipv4Addr::new(192, 168, 1, 0), 
                    mask: 24 
                }),
                dst_ip: None,
                src_port: None,
                dst_port: Some(PortRange { start: 80, end: 80 }),
                protocol: Some(17), // UDP (packet is TCP)
                vlan_id: None,
                packet_type: Some(PacketType::IPv4Tcp),
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };
        
        assert!(!flow_director.match_rule(&non_matching_rule, &packet).unwrap());
    }

    #[test]
    fn test_rss_hash_calculation() {
        let flow_director = create_test_flow_director();
        let packet = create_test_ipv4_tcp_packet();
        
        let hash = flow_director.calculate_rss_hash(&packet).unwrap();
        assert!(hash > 0); // Should produce some hash value
        
        // Same packet should produce same hash
        let hash2 = flow_director.calculate_rss_hash(&packet).unwrap();
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_toeplitz_hash() {
        let flow_director = create_test_flow_director();
        let input = b"test data";
        
        let hash = flow_director.toeplitz_hash(input);
        assert!(hash > 0);
        
        // Same input should produce same hash
        let hash2 = flow_director.toeplitz_hash(input);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_crc32_hash() {
        let flow_director = create_test_flow_director();
        let input = b"test data";
        
        let hash = flow_director.crc32_hash(input);
        assert!(hash > 0);
        
        // Same input should produce same hash
        let hash2 = flow_director.crc32_hash(input);
        assert_eq!(hash, hash2);
        
        // Different input should produce different hash
        let hash3 = flow_director.crc32_hash(b"different data");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_flow_table_operations() {
        let mut flow_table = FlowTable::new(100, 300);
        
        let flow_key = FlowKey {
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2)),
            src_port: 8080,
            dst_port: 80,
            protocol: 6,
        };
        
        // Initially, flow should not exist
        assert!(flow_table.lookup(&flow_key).is_none());
        
        // Insert flow
        flow_table.insert(flow_key.clone(), 1);
        
        // Now flow should exist
        assert_eq!(flow_table.lookup(&flow_key), Some(1));
        
        let stats = flow_table.get_stats();
        assert_eq!(stats.active_flows, 1);
        assert_eq!(stats.flow_insertions, 1);
    }

    #[test]
    fn test_flow_action_application() {
        let flow_director = create_test_flow_director();
        let packet = create_test_ipv4_tcp_packet();
        
        // Test RouteToQueue action
        let route_action = FlowAction::RouteToQueue(2);
        let queue_id = flow_director.apply_action(&route_action, &packet).unwrap();
        assert_eq!(queue_id, 2);
        
        // Test Drop action
        let drop_action = FlowAction::Drop;
        assert!(matches!(
            flow_director.apply_action(&drop_action, &packet),
            Err(FlowDirectorError::PacketDropped)
        ));
        
        // Test Multiple actions
        let multiple_action = FlowAction::Multiple(vec![
            FlowAction::SetPriority(1),
            FlowAction::RouteToQueue(1),
        ]);
        let queue_id = flow_director.apply_action(&multiple_action, &packet).unwrap();
        assert_eq!(queue_id, 1);
    }

    #[test]
    fn test_packet_classification() {
        let mut flow_director = create_test_flow_director();
        let packet = create_test_ipv4_tcp_packet();
        
        // Add a rule that matches the packet
        let rule = FlowRule {
            priority: 100,
            rule_id: 1,
            matcher: FlowMatcher {
                src_ip: Some(IpRange::V4 { 
                    addr: Ipv4Addr::new(192, 168, 1, 0), 
                    mask: 24 
                }),
                dst_ip: None,
                src_port: None,
                dst_port: Some(PortRange { start: 80, end: 80 }),
                protocol: Some(6), // TCP
                vlan_id: None,
                packet_type: Some(PacketType::IPv4Tcp),
                payload_pattern: None,
                direction: FlowDirection::Ingress,
            },
            action: FlowAction::RouteToQueue(1),
            stats: FlowRuleStats::new(),
        };
        
        flow_director.add_rule(rule).unwrap();
        
        // Classify packet
        let queue_id = flow_director.classify_packet(packet).unwrap();
        assert_eq!(queue_id, 1);
        
        // Check statistics
        let stats = flow_director.get_stats();
        assert_eq!(stats.packets_matched, 1);
        assert_eq!(stats.flow_table_misses, 1);
    }

    #[test]
    fn test_flow_director_statistics() {
        let flow_director = create_test_flow_director();
        let stats = flow_director.get_stats();
        
        assert_eq!(stats.packets_processed, 0);
        assert_eq!(stats.packets_matched, 0);
        assert_eq!(stats.packets_default, 0);
        assert_eq!(stats.packets_dropped, 0);
        assert_eq!(stats.active_rules, 0);
        assert_eq!(stats.active_queues, 3);
    }

    #[test]
    fn test_default_configurations() {
        let rss_config = RssConfig::default();
        assert_eq!(rss_config.hash_key.len(), 40);
        assert!(matches!(rss_config.hash_function, RssHashFunction::Toeplitz));
        assert!(!rss_config.indirection_table.is_empty());
        
        let qos_config = QosConfig::default();
        assert_eq!(qos_config.traffic_classes.len(), 3);
        assert!(matches!(qos_config.scheduler, QosScheduler::WeightedRoundRobin));
    }
}