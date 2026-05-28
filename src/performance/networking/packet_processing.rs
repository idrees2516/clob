use super::flow_director::{FlowDirector, FlowDirectorError};
use super::zero_copy::{ZeroCopyPacket, PacketPool, PacketPoolError};
use super::ring_buffers::{RingBuffer, RingBufferError};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// High-performance packet processing pipeline
/// Coordinates packet reception, classification, and distribution
pub struct PacketProcessor {
    /// Flow director for packet classification
    flow_director: Arc<FlowDirector>,
    
    /// Input packet queues from network interfaces
    input_queues: Vec<Arc<RingBuffer<ZeroCopyPacket>>>,
    
    /// Packet pool for buffer management
    packet_pool: Arc<PacketPool>,
    
    /// Processing threads
    worker_threads: Vec<PacketWorkerThread>,
    
    /// Processing statistics
    stats: PacketProcessorStats,
    
    /// Running state
    running: AtomicBool,
    
    /// Configuration
    config: PacketProcessorConfig,
}

/// Packet processing worker thread
pub struct PacketWorkerThread {
    /// Thread handle
    handle: Option<thread::JoinHandle<()>>,
    
    /// Worker ID
    worker_id: usize,
    
    /// CPU affinity
    cpu_affinity: Option<usize>,
    
    /// Worker statistics
    stats: Arc<PacketWorkerStats>,
}

/// Packet processor configuration
#[derive(Debug, Clone)]
pub struct PacketProcessorConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Batch size for packet processing
    pub batch_size: usize,
    
    /// CPU affinity for worker threads
    pub cpu_affinity: Vec<usize>,
    
    /// Processing timeout
    pub processing_timeout_ms: u64,
    
    /// Enable NUMA awareness
    pub numa_aware: bool,
    
    /// Prefetch distance for cache optimization
    pub prefetch_distance: usize,
}

/// Packet processor statistics
#[repr(align(64))]
pub struct PacketProcessorStats {
    /// Total packets processed
    pub packets_processed: AtomicU64,
    
    /// Total bytes processed
    pub bytes_processed: AtomicU64,
    
    /// Processing errors
    pub processing_errors: AtomicU64,
    
    /// Queue full events
    pub queue_full_events: AtomicU64,
    
    /// Average processing latency (nanoseconds)
    pub avg_processing_latency_ns: AtomicU64,
    
    /// Peak processing rate (packets per second)
    pub peak_processing_rate_pps: AtomicU64,
    
    /// Current processing rate (packets per second)
    pub current_processing_rate_pps: AtomicU64,
    
    /// Worker thread utilization
    pub worker_utilization_percent: AtomicU64,
}

/// Worker thread statistics
#[repr(align(64))]
pub struct PacketWorkerStats {
    /// Worker ID
    pub worker_id: usize,
    
    /// Packets processed by this worker
    pub packets_processed: AtomicU64,
    
    /// Bytes processed by this worker
    pub bytes_processed: AtomicU64,
    
    /// Processing errors
    pub processing_errors: AtomicU64,
    
    /// Batches processed
    pub batches_processed: AtomicU64,
    
    /// Average batch size
    pub avg_batch_size: AtomicU64,
    
    /// CPU cycles consumed
    pub cpu_cycles: AtomicU64,
    
    /// Cache misses (estimated)
    pub cache_misses: AtomicU64,
}

impl PacketProcessor {
    /// Create a new packet processor
    pub fn new(
        flow_director: Arc<FlowDirector>,
        input_queues: Vec<Arc<RingBuffer<ZeroCopyPacket>>>,
        packet_pool: Arc<PacketPool>,
        config: PacketProcessorConfig,
    ) -> Self {
        let worker_threads = Vec::with_capacity(config.worker_threads);
        
        Self {
            flow_director,
            input_queues,
            packet_pool,
            worker_threads,
            stats: PacketProcessorStats::new(),
            running: AtomicBool::new(false),
            config,
        }
    }

    /// Start packet processing
    pub fn start(&mut self) -> Result<(), PacketProcessorError> {
        if self.running.load(Ordering::Acquire) {
            return Err(PacketProcessorError::AlreadyRunning);
        }

        self.running.store(true, Ordering::Release);

        // Start worker threads
        for worker_id in 0..self.config.worker_threads {
            let cpu_affinity = self.config.cpu_affinity.get(worker_id).copied();
            let worker = self.start_worker_thread(worker_id, cpu_affinity)?;
            self.worker_threads.push(worker);
        }

        Ok(())
    }

    /// Stop packet processing
    pub fn stop(&mut self) -> Result<(), PacketProcessorError> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(false, Ordering::Release);

        // Wait for all worker threads to finish
        for mut worker in self.worker_threads.drain(..) {
            if let Some(handle) = worker.handle.take() {
                handle.join().map_err(|_| PacketProcessorError::ThreadJoinError)?;
            }
        }

        Ok(())
    }

    /// Start a worker thread
    fn start_worker_thread(
        &self,
        worker_id: usize,
        cpu_affinity: Option<usize>,
    ) -> Result<PacketWorkerThread, PacketProcessorError> {
        let flow_director = self.flow_director.clone();
        let input_queues = self.input_queues.clone();
        let running = Arc::new(&self.running as *const AtomicBool);
        let config = self.config.clone();
        let stats = Arc::new(PacketWorkerStats::new(worker_id));
        let worker_stats = stats.clone();

        let handle = thread::Builder::new()
            .name(format!("packet_worker_{}", worker_id))
            .spawn(move || {
                Self::worker_thread_main(
                    worker_id,
                    flow_director,
                    input_queues,
                    running,
                    config,
                    worker_stats,
                    cpu_affinity,
                );
            })
            .map_err(|_| PacketProcessorError::ThreadCreationError)?;

        Ok(PacketWorkerThread {
            handle: Some(handle),
            worker_id,
            cpu_affinity,
            stats,
        })
    }

    /// Worker thread main loop
    fn worker_thread_main(
        worker_id: usize,
        flow_director: Arc<FlowDirector>,
        input_queues: Vec<Arc<RingBuffer<ZeroCopyPacket>>>,
        running: Arc<*const AtomicBool>,
        config: PacketProcessorConfig,
        stats: Arc<PacketWorkerStats>,
        cpu_affinity: Option<usize>,
    ) {
        // Set CPU affinity if specified
        if let Some(cpu_id) = cpu_affinity {
            Self::set_cpu_affinity(cpu_id);
        }

        let mut packet_buffer = Vec::with_capacity(config.batch_size);
        let mut last_rate_update = Instant::now();
        let mut packets_in_interval = 0u64;

        while unsafe { (*running.as_ref()).load(Ordering::Acquire) } {
            let batch_start = Self::get_timestamp_ns();
            
            // Process packets from all input queues in round-robin fashion
            let mut packets_processed_in_batch = 0;
            
            for queue in &input_queues {
                // Try to dequeue a batch of packets
                packet_buffer.clear();
                match queue.batch_dequeue_into(&mut packet_buffer, config.batch_size) {
                    Ok(count) if count > 0 => {
                        packets_processed_in_batch += count;
                        
                        // Process each packet in the batch
                        for packet in packet_buffer.drain(..) {
                            let packet_start = Self::get_timestamp_ns();
                            
                            // Prefetch next packet data for cache optimization
                            if let Some(next_packet) = packet_buffer.get(0) {
                                Self::prefetch_packet_data(next_packet, config.prefetch_distance);
                            }
                            
                            // Process the packet through flow director
                            match flow_director.process_packet(packet.clone()) {
                                Ok(_) => {
                                    stats.packets_processed.fetch_add(1, Ordering::Relaxed);
                                    stats.bytes_processed.fetch_add(packet.len() as u64, Ordering::Relaxed);
                                    packets_in_interval += 1;
                                }
                                Err(e) => {
                                    stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                                    eprintln!("Worker {}: Packet processing error: {}", worker_id, e);
                                }
                            }
                            
                            // Update processing latency
                            let packet_latency = Self::get_timestamp_ns() - packet_start;
                            Self::update_avg_latency(&stats.cpu_cycles, packet_latency);
                        }
                    }
                    Ok(_) => {
                        // No packets available, yield CPU
                        std::hint::spin_loop();
                    }
                    Err(e) => {
                        stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("Worker {}: Queue error: {}", worker_id, e);
                    }
                }
            }
            
            // Update batch statistics
            if packets_processed_in_batch > 0 {
                stats.batches_processed.fetch_add(1, Ordering::Relaxed);
                Self::update_avg_batch_size(&stats.avg_batch_size, packets_processed_in_batch as u64);
                
                let batch_latency = Self::get_timestamp_ns() - batch_start;
                stats.cpu_cycles.fetch_add(batch_latency, Ordering::Relaxed);
            }
            
            // Update processing rate periodically
            if last_rate_update.elapsed() >= Duration::from_secs(1) {
                // This would be used to update current processing rate
                packets_in_interval = 0;
                last_rate_update = Instant::now();
            }
            
            // Yield CPU if no work was done
            if packets_processed_in_batch == 0 {
                thread::yield_now();
            }
        }
    }

    /// Set CPU affinity for current thread
    fn set_cpu_affinity(cpu_id: usize) {
        // Platform-specific CPU affinity setting would go here
        // For now, just log the intent
        println!("Setting CPU affinity for thread to CPU {}", cpu_id);
    }

    /// Prefetch packet data for cache optimization
    fn prefetch_packet_data(packet: &ZeroCopyPacket, distance: usize) {
        // This would use CPU prefetch instructions
        // For now, just access the data to bring it into cache
        if distance > 0 {
            let _ = packet.len(); // Access packet metadata
            if let Ok(data) = packet.payload() {
                if !data.is_empty() {
                    let _ = data[0]; // Access first byte of payload
                }
            }
        }
    }

    /// Update average latency using exponential moving average
    fn update_avg_latency(avg_atomic: &AtomicU64, new_latency: u64) {
        let alpha = 0.1; // Smoothing factor
        loop {
            let current_avg = avg_atomic.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                new_latency
            } else {
                ((1.0 - alpha) * current_avg as f64 + alpha * new_latency as f64) as u64
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

    /// Update average batch size
    fn update_avg_batch_size(avg_atomic: &AtomicU64, new_batch_size: u64) {
        Self::update_avg_latency(avg_atomic, new_batch_size);
    }

    /// Get processor statistics
    pub fn get_stats(&self) -> PacketProcessorStatsSnapshot {
        let mut worker_stats = Vec::new();
        for worker in &self.worker_threads {
            worker_stats.push(PacketWorkerStatsSnapshot {
                worker_id: worker.stats.worker_id,
                packets_processed: worker.stats.packets_processed.load(Ordering::Acquire),
                bytes_processed: worker.stats.bytes_processed.load(Ordering::Acquire),
                processing_errors: worker.stats.processing_errors.load(Ordering::Acquire),
                batches_processed: worker.stats.batches_processed.load(Ordering::Acquire),
                avg_batch_size: worker.stats.avg_batch_size.load(Ordering::Acquire),
                cpu_cycles: worker.stats.cpu_cycles.load(Ordering::Acquire),
                cache_misses: worker.stats.cache_misses.load(Ordering::Acquire),
                cpu_affinity: worker.cpu_affinity,
            });
        }

        PacketProcessorStatsSnapshot {
            packets_processed: self.stats.packets_processed.load(Ordering::Acquire),
            bytes_processed: self.stats.bytes_processed.load(Ordering::Acquire),
            processing_errors: self.stats.processing_errors.load(Ordering::Acquire),
            queue_full_events: self.stats.queue_full_events.load(Ordering::Acquire),
            avg_processing_latency_ns: self.stats.avg_processing_latency_ns.load(Ordering::Acquire),
            peak_processing_rate_pps: self.stats.peak_processing_rate_pps.load(Ordering::Acquire),
            current_processing_rate_pps: self.stats.current_processing_rate_pps.load(Ordering::Acquire),
            worker_utilization_percent: self.stats.worker_utilization_percent.load(Ordering::Acquire),
            worker_stats,
            active_workers: self.worker_threads.len(),
            input_queues: self.input_queues.len(),
        }
    }

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}

impl PacketProcessorStats {
    fn new() -> Self {
        Self {
            packets_processed: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            processing_errors: AtomicU64::new(0),
            queue_full_events: AtomicU64::new(0),
            avg_processing_latency_ns: AtomicU64::new(0),
            peak_processing_rate_pps: AtomicU64::new(0),
            current_processing_rate_pps: AtomicU64::new(0),
            worker_utilization_percent: AtomicU64::new(0),
        }
    }
}

impl PacketWorkerStats {
    fn new(worker_id: usize) -> Self {
        Self {
            worker_id,
            packets_processed: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            processing_errors: AtomicU64::new(0),
            batches_processed: AtomicU64::new(0),
            avg_batch_size: AtomicU64::new(0),
            cpu_cycles: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }
}

/// Packet processor statistics snapshot
#[derive(Debug, Clone)]
pub struct PacketProcessorStatsSnapshot {
    pub packets_processed: u64,
    pub bytes_processed: u64,
    pub processing_errors: u64,
    pub queue_full_events: u64,
    pub avg_processing_latency_ns: u64,
    pub peak_processing_rate_pps: u64,
    pub current_processing_rate_pps: u64,
    pub worker_utilization_percent: u64,
    pub worker_stats: Vec<PacketWorkerStatsSnapshot>,
    pub active_workers: usize,
    pub input_queues: usize,
}

/// Worker statistics snapshot
#[derive(Debug, Clone)]
pub struct PacketWorkerStatsSnapshot {
    pub worker_id: usize,
    pub packets_processed: u64,
    pub bytes_processed: u64,
    pub processing_errors: u64,
    pub batches_processed: u64,
    pub avg_batch_size: u64,
    pub cpu_cycles: u64,
    pub cache_misses: u64,
    pub cpu_affinity: Option<usize>,
}

/// Default packet processor configuration
impl Default for PacketProcessorConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            batch_size: 32,
            cpu_affinity: Vec::new(),
            processing_timeout_ms: 1000,
            numa_aware: true,
            prefetch_distance: 2,
        }
    }
}

/// Packet processor error types
#[derive(Debug, Clone)]
pub enum PacketProcessorError {
    AlreadyRunning,
    NotRunning,
    ThreadCreationError,
    ThreadJoinError,
    ConfigurationError(String),
    FlowDirectorError(FlowDirectorError),
    QueueError(RingBufferError),
}

impl std::fmt::Display for PacketProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PacketProcessorError::AlreadyRunning => write!(f, "Packet processor is already running"),
            PacketProcessorError::NotRunning => write!(f, "Packet processor is not running"),
            PacketProcessorError::ThreadCreationError => write!(f, "Failed to create worker thread"),
            PacketProcessorError::ThreadJoinError => write!(f, "Failed to join worker thread"),
            PacketProcessorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            PacketProcessorError::FlowDirectorError(e) => write!(f, "Flow director error: {}", e),
            PacketProcessorError::QueueError(e) => write!(f, "Queue error: {}", e),
        }
    }
}

impl std::error::Error for PacketProcessorError {}

impl From<FlowDirectorError> for PacketProcessorError {
    fn from(error: FlowDirectorError) -> Self {
        PacketProcessorError::FlowDirectorError(error)
    }
}

impl From<RingBufferError> for PacketProcessorError {
    fn from(error: RingBufferError) -> Self {
        PacketProcessorError::QueueError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance::networking::zero_copy::PacketType;
    use std::collections::HashMap;

    #[test]
    fn test_packet_processor_creation() {
        let flow_director = Arc::new(create_test_flow_director());
        let input_queues = vec![
            Arc::new(RingBuffer::new(1024, "test_input_0".to_string()).unwrap())
        ];
        let packet_pool = Arc::new(PacketPool::new(100, 1500, 0).unwrap());
        let config = PacketProcessorConfig::default();

        let processor = PacketProcessor::new(
            flow_director,
            input_queues,
            packet_pool,
            config,
        );

        assert_eq!(processor.worker_threads.len(), 0); // Not started yet
        assert!(!processor.running.load(Ordering::Acquire));
    }

    #[test]
    fn test_worker_stats() {
        let stats = PacketWorkerStats::new(0);
        assert_eq!(stats.worker_id, 0);
        assert_eq!(stats.packets_processed.load(Ordering::Acquire), 0);
        assert_eq!(stats.bytes_processed.load(Ordering::Acquire), 0);
    }

    fn create_test_flow_director() -> FlowDirector {
        use crate::performance::networking::flow_director::{RssConfig, QosConfig};
        
        let queues = HashMap::new();
        let default_queue = 0;
        let rss_config = RssConfig::default();
        let qos_config = QosConfig::default();

        FlowDirector::new(queues, default_queue, rss_config, qos_config)
    }
}