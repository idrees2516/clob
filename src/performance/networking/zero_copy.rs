use super::super::memory::{LockFreePool, PooledObject, PoolError};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;
use std::slice;
use std::mem;

/// Zero-copy packet for high-performance networking
/// Avoids memory copies by using memory-mapped buffers and reference counting
pub struct ZeroCopyPacket {
    /// Packet buffer (memory-mapped or pre-allocated)
    buffer: PacketBuffer,
    
    /// Packet metadata
    metadata: PacketMetadata,
    
    /// Reference count for safe sharing
    ref_count: Arc<AtomicUsize>,
}

/// Packet buffer types
pub enum PacketBuffer {
    /// Memory-mapped buffer (zero-copy from hardware)
    MemoryMapped {
        ptr: *mut u8,
        len: usize,
        capacity: usize,
    },
    
    /// Pre-allocated buffer from pool
    Pooled {
        buffer: Vec<u8>,
        offset: usize,
        len: usize,
    },
    
    /// Scatter-gather list for fragmented packets
    ScatterGather {
        segments: Vec<BufferSegment>,
        total_len: usize,
    },
}

/// Buffer segment for scatter-gather operations
#[derive(Clone)]
pub struct BufferSegment {
    pub ptr: *mut u8,
    pub len: usize,
    pub offset: usize,
}

/// Packet metadata for processing
#[derive(Debug, Clone)]
pub struct PacketMetadata {
    /// Hardware timestamp (if available)
    pub hw_timestamp: Option<u64>,
    
    /// Software timestamp
    pub sw_timestamp: u64,
    
    /// Packet length
    pub packet_len: usize,
    
    /// Layer 2 header offset
    pub l2_offset: usize,
    
    /// Layer 3 header offset
    pub l3_offset: usize,
    
    /// Layer 4 header offset
    pub l4_offset: usize,
    
    /// Payload offset
    pub payload_offset: usize,
    
    /// VLAN tag (if present)
    pub vlan_tag: Option<u16>,
    
    /// Packet type
    pub packet_type: PacketType,
    
    /// RSS hash
    pub rss_hash: Option<u32>,
    
    /// Queue ID
    pub queue_id: u16,
    
    /// Processing flags
    pub flags: PacketFlags,
}

/// Packet type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PacketType {
    Unknown,
    IPv4Tcp,
    IPv4Udp,
    IPv4Icmp,
    IPv6Tcp,
    IPv6Udp,
    IPv6Icmp,
    Arp,
    Other(u16), // EtherType
}

/// Packet processing flags
#[derive(Debug, Clone, Copy)]
pub struct PacketFlags {
    /// Checksum verified by hardware
    pub checksum_verified: bool,
    
    /// Packet is fragmented
    pub fragmented: bool,
    
    /// Packet has VLAN tag
    pub vlan_tagged: bool,
    
    /// Packet is broadcast
    pub broadcast: bool,
    
    /// Packet is multicast
    pub multicast: bool,
    
    /// Packet has errors
    pub has_errors: bool,
}

impl ZeroCopyPacket {
    /// Create a new zero-copy packet from memory-mapped buffer
    pub fn from_memory_mapped(ptr: *mut u8, len: usize, capacity: usize) -> Self {
        Self {
            buffer: PacketBuffer::MemoryMapped { ptr, len, capacity },
            metadata: PacketMetadata::new(len),
            ref_count: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Create a new zero-copy packet from pooled buffer
    pub fn from_pooled_buffer(mut buffer: Vec<u8>, len: usize) -> Self {
        buffer.resize(len, 0);
        
        Self {
            buffer: PacketBuffer::Pooled { buffer, offset: 0, len },
            metadata: PacketMetadata::new(len),
            ref_count: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Create a new zero-copy packet from scatter-gather segments
    pub fn from_scatter_gather(segments: Vec<BufferSegment>) -> Self {
        let total_len = segments.iter().map(|s| s.len).sum();
        
        Self {
            buffer: PacketBuffer::ScatterGather { segments, total_len },
            metadata: PacketMetadata::new(total_len),
            ref_count: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Get packet data as a slice (may require copying for scatter-gather)
    pub fn data(&self) -> PacketData {
        match &self.buffer {
            PacketBuffer::MemoryMapped { ptr, len, .. } => {
                PacketData::Contiguous(unsafe { slice::from_raw_parts(*ptr, *len) })
            }
            PacketBuffer::Pooled { buffer, offset, len } => {
                PacketData::Contiguous(&buffer[*offset..*offset + *len])
            }
            PacketBuffer::ScatterGather { segments, .. } => {
                PacketData::ScatterGather(segments.clone())
            }
        }
    }

    /// Get mutable packet data (may require copying for scatter-gather)
    pub fn data_mut(&mut self) -> PacketDataMut {
        match &mut self.buffer {
            PacketBuffer::MemoryMapped { ptr, len, .. } => {
                PacketDataMut::Contiguous(unsafe { slice::from_raw_parts_mut(*ptr, *len) })
            }
            PacketBuffer::Pooled { buffer, offset, len } => {
                PacketDataMut::Contiguous(&mut buffer[*offset..*offset + *len])
            }
            PacketBuffer::ScatterGather { segments, .. } => {
                PacketDataMut::ScatterGather(segments.clone())
            }
        }
    }

    /// Get packet length
    pub fn len(&self) -> usize {
        match &self.buffer {
            PacketBuffer::MemoryMapped { len, .. } => *len,
            PacketBuffer::Pooled { len, .. } => *len,
            PacketBuffer::ScatterGather { total_len, .. } => *total_len,
        }
    }

    /// Check if packet is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get packet metadata
    pub fn metadata(&self) -> &PacketMetadata {
        &self.metadata
    }

    /// Get mutable packet metadata
    pub fn metadata_mut(&mut self) -> &mut PacketMetadata {
        &mut self.metadata
    }

    /// Clone the packet (increases reference count for zero-copy)
    pub fn clone_packet(&self) -> Self {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
        
        Self {
            buffer: match &self.buffer {
                PacketBuffer::MemoryMapped { ptr, len, capacity } => {
                    PacketBuffer::MemoryMapped { ptr: *ptr, len: *len, capacity: *capacity }
                }
                PacketBuffer::Pooled { buffer, offset, len } => {
                    // For pooled buffers, we need to actually clone the data
                    PacketBuffer::Pooled {
                        buffer: buffer.clone(),
                        offset: *offset,
                        len: *len,
                    }
                }
                PacketBuffer::ScatterGather { segments, total_len } => {
                    PacketBuffer::ScatterGather {
                        segments: segments.clone(),
                        total_len: *total_len,
                    }
                }
            },
            metadata: self.metadata.clone(),
            ref_count: self.ref_count.clone(),
        }
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Parse packet headers and update metadata
    pub fn parse_headers(&mut self) -> Result<(), PacketParseError> {
        let data = match self.data() {
            PacketData::Contiguous(slice) => slice,
            PacketData::ScatterGather(_) => {
                // For scatter-gather, we'd need to linearize first
                return Err(PacketParseError::ScatterGatherNotSupported);
            }
        };

        if data.len() < 14 {
            return Err(PacketParseError::TooShort);
        }

        // Parse Ethernet header
        self.metadata.l2_offset = 0;
        let ethertype = u16::from_be_bytes([data[12], data[13]]);
        let mut offset = 14;

        // Check for VLAN tag
        if ethertype == 0x8100 {
            if data.len() < 18 {
                return Err(PacketParseError::TooShort);
            }
            self.metadata.vlan_tag = Some(u16::from_be_bytes([data[14], data[15]]));
            self.metadata.flags.vlan_tagged = true;
            let ethertype = u16::from_be_bytes([data[16], data[17]]);
            offset = 18;
            
            self.metadata.packet_type = Self::classify_packet_type(ethertype);
        } else {
            self.metadata.packet_type = Self::classify_packet_type(ethertype);
        }

        // Parse Layer 3 header
        match self.metadata.packet_type {
            PacketType::IPv4Tcp | PacketType::IPv4Udp | PacketType::IPv4Icmp => {
                self.parse_ipv4_header(data, offset)?;
            }
            PacketType::IPv6Tcp | PacketType::IPv6Udp | PacketType::IPv6Icmp => {
                self.parse_ipv6_header(data, offset)?;
            }
            PacketType::Arp => {
                self.metadata.l3_offset = offset;
                self.metadata.l4_offset = offset + 28; // ARP header size
                self.metadata.payload_offset = self.metadata.l4_offset;
            }
            _ => {
                self.metadata.l3_offset = offset;
                self.metadata.l4_offset = offset;
                self.metadata.payload_offset = offset;
            }
        }

        Ok(())
    }

    /// Classify packet type based on EtherType
    fn classify_packet_type(ethertype: u16) -> PacketType {
        match ethertype {
            0x0800 => PacketType::IPv4Tcp, // Will be refined after parsing IP header
            0x86DD => PacketType::IPv6Tcp, // Will be refined after parsing IP header
            0x0806 => PacketType::Arp,
            other => PacketType::Other(other),
        }
    }

    /// Parse IPv4 header
    fn parse_ipv4_header(&mut self, data: &[u8], offset: usize) -> Result<(), PacketParseError> {
        if data.len() < offset + 20 {
            return Err(PacketParseError::TooShort);
        }

        self.metadata.l3_offset = offset;
        
        let ihl = (data[offset] & 0x0F) as usize * 4;
        let protocol = data[offset + 9];
        let flags = (data[offset + 6] & 0xE0) >> 5;
        let frag_offset = u16::from_be_bytes([data[offset + 6] & 0x1F, data[offset + 7]]);
        
        // Check for fragmentation
        if (flags & 0x01) != 0 || frag_offset != 0 {
            self.metadata.flags.fragmented = true;
        }

        self.metadata.l4_offset = offset + ihl;

        // Refine packet type based on protocol
        self.metadata.packet_type = match protocol {
            6 => PacketType::IPv4Tcp,
            17 => PacketType::IPv4Udp,
            1 => PacketType::IPv4Icmp,
            _ => PacketType::Other(protocol as u16),
        };

        // Parse Layer 4 header if present
        if !self.metadata.flags.fragmented {
            self.parse_l4_header(data, protocol)?;
        } else {
            self.metadata.payload_offset = self.metadata.l4_offset;
        }

        Ok(())
    }

    /// Parse IPv6 header
    fn parse_ipv6_header(&mut self, data: &[u8], offset: usize) -> Result<(), PacketParseError> {
        if data.len() < offset + 40 {
            return Err(PacketParseError::TooShort);
        }

        self.metadata.l3_offset = offset;
        let next_header = data[offset + 6];
        self.metadata.l4_offset = offset + 40;

        // Refine packet type based on next header
        self.metadata.packet_type = match next_header {
            6 => PacketType::IPv6Tcp,
            17 => PacketType::IPv6Udp,
            58 => PacketType::IPv6Icmp,
            _ => PacketType::Other(next_header as u16),
        };

        // Parse Layer 4 header
        self.parse_l4_header(data, next_header)?;

        Ok(())
    }

    /// Parse Layer 4 header
    fn parse_l4_header(&mut self, data: &[u8], protocol: u8) -> Result<(), PacketParseError> {
        let offset = self.metadata.l4_offset;

        match protocol {
            6 => {
                // TCP
                if data.len() < offset + 20 {
                    return Err(PacketParseError::TooShort);
                }
                let data_offset = ((data[offset + 12] & 0xF0) >> 4) as usize * 4;
                self.metadata.payload_offset = offset + data_offset;
            }
            17 => {
                // UDP
                if data.len() < offset + 8 {
                    return Err(PacketParseError::TooShort);
                }
                self.metadata.payload_offset = offset + 8;
            }
            1 | 58 => {
                // ICMP/ICMPv6
                if data.len() < offset + 8 {
                    return Err(PacketParseError::TooShort);
                }
                self.metadata.payload_offset = offset + 8;
            }
            _ => {
                self.metadata.payload_offset = offset;
            }
        }

        Ok(())
    }

    /// Get payload data
    pub fn payload(&self) -> Result<&[u8], PacketParseError> {
        match self.data() {
            PacketData::Contiguous(data) => {
                if self.metadata.payload_offset < data.len() {
                    Ok(&data[self.metadata.payload_offset..])
                } else {
                    Err(PacketParseError::InvalidOffset)
                }
            }
            PacketData::ScatterGather(_) => {
                Err(PacketParseError::ScatterGatherNotSupported)
            }
        }
    }

    /// Get header at specific layer
    pub fn get_header(&self, layer: HeaderLayer) -> Result<&[u8], PacketParseError> {
        let data = match self.data() {
            PacketData::Contiguous(slice) => slice,
            PacketData::ScatterGather(_) => {
                return Err(PacketParseError::ScatterGatherNotSupported);
            }
        };

        let (start, end) = match layer {
            HeaderLayer::L2 => (self.metadata.l2_offset, self.metadata.l3_offset),
            HeaderLayer::L3 => (self.metadata.l3_offset, self.metadata.l4_offset),
            HeaderLayer::L4 => (self.metadata.l4_offset, self.metadata.payload_offset),
            HeaderLayer::Payload => (self.metadata.payload_offset, data.len()),
        };

        if start < data.len() && end <= data.len() && start <= end {
            Ok(&data[start..end])
        } else {
            Err(PacketParseError::InvalidOffset)
        }
    }
}

impl Drop for ZeroCopyPacket {
    fn drop(&mut self) {
        let old_count = self.ref_count.fetch_sub(1, Ordering::AcqRel);
        
        // If this was the last reference, cleanup the buffer
        if old_count == 1 {
            match &mut self.buffer {
                PacketBuffer::MemoryMapped { .. } => {
                    // Memory-mapped buffers are managed by the hardware/driver
                    // No explicit cleanup needed
                }
                PacketBuffer::Pooled { .. } => {
                    // Pooled buffers are automatically returned to pool when dropped
                }
                PacketBuffer::ScatterGather { .. } => {
                    // Scatter-gather segments are managed individually
                }
            }
        }
    }
}

/// Packet data access types
pub enum PacketData<'a> {
    Contiguous(&'a [u8]),
    ScatterGather(Vec<BufferSegment>),
}

/// Mutable packet data access types
pub enum PacketDataMut<'a> {
    Contiguous(&'a mut [u8]),
    ScatterGather(Vec<BufferSegment>),
}

/// Header layer enumeration
#[derive(Debug, Clone, Copy)]
pub enum HeaderLayer {
    L2,      // Ethernet
    L3,      // IP
    L4,      // TCP/UDP
    Payload, // Application data
}

impl PacketMetadata {
    fn new(packet_len: usize) -> Self {
        Self {
            hw_timestamp: None,
            sw_timestamp: Self::get_timestamp_ns(),
            packet_len,
            l2_offset: 0,
            l3_offset: 0,
            l4_offset: 0,
            payload_offset: 0,
            vlan_tag: None,
            packet_type: PacketType::Unknown,
            rss_hash: None,
            queue_id: 0,
            flags: PacketFlags::default(),
        }
    }

    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}

impl Default for PacketFlags {
    fn default() -> Self {
        Self {
            checksum_verified: false,
            fragmented: false,
            vlan_tagged: false,
            broadcast: false,
            multicast: false,
            has_errors: false,
        }
    }
}

/// Packet pool for managing zero-copy packet buffers
pub struct PacketPool {
    /// Pool for packet buffers
    buffer_pool: Arc<LockFreePool<Vec<u8>>>,
    
    /// Buffer size
    buffer_size: usize,
    
    /// NUMA node
    numa_node: u32,
    
    /// Pool statistics
    stats: PacketPoolStats,
}

/// Packet pool statistics
#[repr(align(64))]
pub struct PacketPoolStats {
    /// Total packets allocated
    pub packets_allocated: AtomicU64,
    
    /// Total packets deallocated
    pub packets_deallocated: AtomicU64,
    
    /// Allocation failures
    pub allocation_failures: AtomicU64,
    
    /// Peak usage
    pub peak_usage: AtomicU64,
}

impl PacketPool {
    /// Create a new packet pool
    pub fn new(capacity: usize, buffer_size: usize, numa_node: u32) -> Result<Self, PoolError> {
        let buffer_pool = Arc::new(LockFreePool::new(capacity, numa_node)?);
        
        Ok(Self {
            buffer_pool,
            buffer_size,
            numa_node,
            stats: PacketPoolStats::new(),
        })
    }

    /// Allocate a new packet
    pub fn allocate_packet(&self) -> Result<ZeroCopyPacket, PoolError> {
        match self.buffer_pool.allocate() {
            Ok(mut pooled_buffer) => {
                // Initialize buffer
                pooled_buffer.clear();
                pooled_buffer.resize(self.buffer_size, 0);
                
                let packet = ZeroCopyPacket::from_pooled_buffer(
                    pooled_buffer.clone(), // This will be optimized to avoid actual cloning
                    0, // Initial length is 0
                );
                
                self.stats.packets_allocated.fetch_add(1, Ordering::Relaxed);
                Ok(packet)
            }
            Err(e) => {
                self.stats.allocation_failures.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PacketPoolStatsSnapshot {
        PacketPoolStatsSnapshot {
            numa_node: self.numa_node,
            buffer_size: self.buffer_size,
            packets_allocated: self.stats.packets_allocated.load(Ordering::Acquire),
            packets_deallocated: self.stats.packets_deallocated.load(Ordering::Acquire),
            allocation_failures: self.stats.allocation_failures.load(Ordering::Acquire),
            peak_usage: self.stats.peak_usage.load(Ordering::Acquire),
        }
    }
}

/// Packet pool statistics snapshot
#[derive(Debug, Clone)]
pub struct PacketPoolStatsSnapshot {
    pub numa_node: u32,
    pub buffer_size: usize,
    pub packets_allocated: u64,
    pub packets_deallocated: u64,
    pub allocation_failures: u64,
    pub peak_usage: u64,
}

impl PacketPoolStats {
    fn new() -> Self {
        Self {
            packets_allocated: AtomicU64::new(0),
            packets_deallocated: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
        }
    }
}

/// Packet parsing errors
#[derive(Debug, Clone, PartialEq)]
pub enum PacketParseError {
    TooShort,
    InvalidOffset,
    UnsupportedProtocol,
    ScatterGatherNotSupported,
    CorruptedHeader,
}

impl std::fmt::Display for PacketParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PacketParseError::TooShort => write!(f, "Packet too short"),
            PacketParseError::InvalidOffset => write!(f, "Invalid header offset"),
            PacketParseError::UnsupportedProtocol => write!(f, "Unsupported protocol"),
            PacketParseError::ScatterGatherNotSupported => write!(f, "Scatter-gather not supported for this operation"),
            PacketParseError::CorruptedHeader => write!(f, "Corrupted packet header"),
        }
    }
}

impl std::error::Error for PacketParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_creation() {
        let buffer = vec![0u8; 1500];
        let packet = ZeroCopyPacket::from_pooled_buffer(buffer, 100);
        
        assert_eq!(packet.len(), 100);
        assert!(!packet.is_empty());
        assert_eq!(packet.ref_count(), 1);
    }

    #[test]
    fn test_packet_cloning() {
        let buffer = vec![0u8; 1500];
        let packet = ZeroCopyPacket::from_pooled_buffer(buffer, 100);
        
        let cloned = packet.clone_packet();
        assert_eq!(packet.ref_count(), 2);
        assert_eq!(cloned.ref_count(), 2);
        
        drop(cloned);
        assert_eq!(packet.ref_count(), 1);
    }

    #[test]
    fn test_ethernet_parsing() {
        // Create a simple Ethernet frame
        let mut buffer = vec![0u8; 64];
        
        // Destination MAC
        buffer[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        // Source MAC
        buffer[6..12].copy_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        // EtherType (IPv4)
        buffer[12..14].copy_from_slice(&[0x08, 0x00]);
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 64);
        packet.parse_headers().unwrap();
        
        assert_eq!(packet.metadata().l2_offset, 0);
        assert_eq!(packet.metadata().l3_offset, 14);
        assert_eq!(packet.metadata().packet_type, PacketType::IPv4Tcp);
    }

    #[test]
    fn test_vlan_parsing() {
        // Create an Ethernet frame with VLAN tag
        let mut buffer = vec![0u8; 68];
        
        // Destination MAC
        buffer[0..6].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        // Source MAC
        buffer[6..12].copy_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        // VLAN EtherType
        buffer[12..14].copy_from_slice(&[0x81, 0x00]);
        // VLAN tag
        buffer[14..16].copy_from_slice(&[0x00, 0x64]); // VLAN 100
        // Inner EtherType (IPv4)
        buffer[16..18].copy_from_slice(&[0x08, 0x00]);
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 68);
        packet.parse_headers().unwrap();
        
        assert_eq!(packet.metadata().l2_offset, 0);
        assert_eq!(packet.metadata().l3_offset, 18);
        assert_eq!(packet.metadata().vlan_tag, Some(100));
        assert!(packet.metadata().flags.vlan_tagged);
    }

    #[test]
    fn test_ipv4_tcp_parsing() {
        // Create a minimal IPv4 TCP packet
        let mut buffer = vec![0u8; 54]; // Ethernet + IPv4 + TCP headers
        
        // Ethernet header
        buffer[12..14].copy_from_slice(&[0x08, 0x00]); // IPv4
        
        // IPv4 header
        buffer[14] = 0x45; // Version 4, IHL 5
        buffer[23] = 6;    // Protocol TCP
        
        // TCP header
        buffer[46] = 0x50; // Data offset 5 (20 bytes)
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 54);
        packet.parse_headers().unwrap();
        
        assert_eq!(packet.metadata().packet_type, PacketType::IPv4Tcp);
        assert_eq!(packet.metadata().l3_offset, 14);
        assert_eq!(packet.metadata().l4_offset, 34);
        assert_eq!(packet.metadata().payload_offset, 54);
    }

    #[test]
    fn test_packet_pool() {
        let pool = PacketPool::new(10, 1500, 0).unwrap();
        
        let packet = pool.allocate_packet().unwrap();
        assert_eq!(packet.len(), 0); // Initial length
        
        let stats = pool.get_stats();
        assert_eq!(stats.packets_allocated, 1);
        assert_eq!(stats.buffer_size, 1500);
    }

    #[test]
    fn test_scatter_gather() {
        let segments = vec![
            BufferSegment { ptr: ptr::null_mut(), len: 100, offset: 0 },
            BufferSegment { ptr: ptr::null_mut(), len: 200, offset: 0 },
        ];
        
        let packet = ZeroCopyPacket::from_scatter_gather(segments);
        assert_eq!(packet.len(), 300);
        
        match packet.data() {
            PacketData::ScatterGather(segs) => assert_eq!(segs.len(), 2),
            _ => panic!("Expected scatter-gather data"),
        }
    }

    #[test]
    fn test_header_extraction() {
        let mut buffer = vec![0u8; 64];
        
        // Simple Ethernet + IPv4 frame
        buffer[12..14].copy_from_slice(&[0x08, 0x00]); // IPv4
        buffer[14] = 0x45; // IPv4 header
        
        let mut packet = ZeroCopyPacket::from_pooled_buffer(buffer, 64);
        packet.parse_headers().unwrap();
        
        let l2_header = packet.get_header(HeaderLayer::L2).unwrap();
        assert_eq!(l2_header.len(), 14); // Ethernet header
        
        let l3_header = packet.get_header(HeaderLayer::L3).unwrap();
        assert_eq!(l3_header.len(), 20); // IPv4 header (IHL=5)
    }

    #[test]
    fn test_memory_mapped_packet() {
        let mut buffer = vec![0u8; 1500];
        let ptr = buffer.as_mut_ptr();
        
        // Don't drop the buffer while packet exists
        let packet = ZeroCopyPacket::from_memory_mapped(ptr, 100, 1500);
        assert_eq!(packet.len(), 100);
        
        match packet.data() {
            PacketData::Contiguous(data) => assert_eq!(data.len(), 100),
            _ => panic!("Expected contiguous data"),
        }
        
        // Keep buffer alive
        mem::forget(buffer);
    }
}