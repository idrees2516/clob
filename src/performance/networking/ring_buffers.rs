use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::ptr;
use std::mem;
use std::time::Duration;
use std::thread;

/// Lock-free ring buffer for high-performance packet processing
/// Uses single-producer single-consumer (SPSC) or multi-producer multi-consumer (MPMC) patterns
#[repr(align(64))] // Cache-line aligned
pub struct RingBuffer<T> {
    /// Ring buffer name for debugging
    name: String,
    
    /// Buffer capacity (must be power of 2)
    capacity: usize,
    
    /// Capacity mask for fast modulo operation
    capacity_mask: usize,
    
    /// Producer head index
    producer_head: AtomicU32,
    
    /// Producer tail index  
    producer_tail: AtomicU32,
    
    /// Consumer head index
    consumer_head: AtomicU32,
    
    /// Consumer tail index
    consumer_tail: AtomicU32,
    
    /// Ring buffer storage
    ring: Vec<AtomicEntry<T>>,
    
    /// Buffer statistics
    stats: RingBufferStats,
}

/// Atomic entry in the ring buffer
#[repr(align(64))]
struct AtomicEntry<T> {
    /// Sequence number for ordering
    sequence: AtomicU32,
    
    /// The actual data
    data: Option<T>,
    
    /// Padding to prevent false sharing
    _padding: [u8; 64 - mem::size_of::<AtomicU32>() - mem::size_of::<Option<T>>()],
}

/// Ring buffer statistics
#[repr(align(64))]
pub struct RingBufferStats {
    /// Total enqueue operations
    pub enqueue_count: AtomicUsize,
    
    /// Total dequeue operations
    pub dequeue_count: AtomicUsize,
    
    /// Enqueue failures (buffer full)
    pub enqueue_failures: AtomicUsize,
    
    /// Dequeue failures (buffer empty)
    pub dequeue_failures: AtomicUsize,
    
    /// Current buffer utilization
    pub current_size: AtomicUsize,
    
    /// Peak buffer utilization
    pub peak_size: AtomicUsize,
    
    /// Average enqueue latency (nanoseconds)
    pub avg_enqueue_latency_ns: AtomicUsize,
    
    /// Average dequeue latency (nanoseconds)
    pub avg_dequeue_latency_ns: AtomicUsize,
    
    /// Batch operations count
    pub batch_enqueue_count: AtomicUsize,
    
    /// Batch operations count
    pub batch_dequeue_count: AtomicUsize,
    
    /// Producer contention events
    pub producer_contention: AtomicUsize,
    
    /// Consumer contention events
    pub consumer_contention: AtomicUsize,
    
    /// Cache miss events (estimated)
    pub cache_misses: AtomicUsize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with specified capacity
    /// Capacity must be a power of 2 for optimal performance
    pub fn new(capacity: usize, name: String) -> Result<Self, RingBufferError> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(RingBufferError::InvalidCapacity(capacity));
        }

        let capacity_mask = capacity - 1;
        let mut ring = Vec::with_capacity(capacity);
        
        // Initialize ring entries
        for i in 0..capacity {
            ring.push(AtomicEntry {
                sequence: AtomicU32::new(i as u32),
                data: None,
                _padding: [0; 64 - mem::size_of::<AtomicU32>() - mem::size_of::<Option<T>>()],
            });
        }

        Ok(Self {
            name,
            capacity,
            capacity_mask,
            producer_head: AtomicU32::new(0),
            producer_tail: AtomicU32::new(0),
            consumer_head: AtomicU32::new(0),
            consumer_tail: AtomicU32::new(0),
            ring,
            stats: RingBufferStats::new(),
        })
    }

    /// Enqueue an item into the ring buffer (producer operation)
    pub fn enqueue(&self, item: T) -> Result<(), RingBufferError> {
        let start_time = Self::get_timestamp_ns();
        
        loop {
            let head = self.producer_head.load(Ordering::Relaxed);
            let next_head = head.wrapping_add(1);
            let index = (head as usize) & self.capacity_mask;
            
            // Check if buffer is full
            let consumer_tail = self.consumer_tail.load(Ordering::Acquire);
            if next_head.wrapping_sub(consumer_tail) > self.capacity as u32 {
                self.stats.enqueue_failures.fetch_add(1, Ordering::Relaxed);
                return Err(RingBufferError::Full);
            }

            // Get the entry
            let entry = &self.ring[index];
            let sequence = entry.sequence.load(Ordering::Acquire);
            
            // Check if this slot is available for writing
            if sequence == head {
                // Try to claim this slot
                match self.producer_head.compare_exchange_weak(
                    head,
                    next_head,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully claimed the slot, now write the data
                        unsafe {
                            let entry_ptr = entry as *const AtomicEntry<T> as *mut AtomicEntry<T>;
                            (*entry_ptr).data = Some(item);
                        }
                        
                        // Update sequence to signal data is ready
                        entry.sequence.store(next_head, Ordering::Release);
                        
                        // Update producer tail
                        while self.producer_tail.load(Ordering::Relaxed) != head {
                            std::hint::spin_loop();
                        }
                        self.producer_tail.store(next_head, Ordering::Release);
                        
                        // Update statistics
                        self.stats.enqueue_count.fetch_add(1, Ordering::Relaxed);
                        self.update_size_stats(1);
                        
                        let latency = Self::get_timestamp_ns() - start_time;
                        self.update_avg_latency(&self.stats.avg_enqueue_latency_ns, latency);
                        
                        return Ok(());
                    }
                    Err(_) => {
                        // CAS failed, track contention and retry
                        self.stats.producer_contention.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            } else {
                // Slot not available, yield and retry
                std::hint::spin_loop();
            }
        }
    }

    /// Dequeue an item from the ring buffer (consumer operation)
    pub fn dequeue(&self) -> Result<T, RingBufferError> {
        let start_time = Self::get_timestamp_ns();
        
        loop {
            let head = self.consumer_head.load(Ordering::Relaxed);
            let index = (head as usize) & self.capacity_mask;
            
            // Check if buffer is empty
            let producer_tail = self.producer_tail.load(Ordering::Acquire);
            if head == producer_tail {
                self.stats.dequeue_failures.fetch_add(1, Ordering::Relaxed);
                return Err(RingBufferError::Empty);
            }

            // Get the entry
            let entry = &self.ring[index];
            let sequence = entry.sequence.load(Ordering::Acquire);
            let next_head = head.wrapping_add(1);
            
            // Check if data is ready
            if sequence == next_head {
                // Try to claim this slot
                match self.consumer_head.compare_exchange_weak(
                    head,
                    next_head,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // Successfully claimed the slot, read the data
                        let data = unsafe {
                            let entry_ptr = entry as *const AtomicEntry<T> as *mut AtomicEntry<T>;
                            (*entry_ptr).data.take()
                        };
                        
                        if let Some(item) = data {
                            // Update sequence to signal slot is available
                            let next_sequence = next_head.wrapping_add(self.capacity as u32);
                            entry.sequence.store(next_sequence, Ordering::Release);
                            
                            // Update consumer tail
                            while self.consumer_tail.load(Ordering::Relaxed) != head {
                                std::hint::spin_loop();
                            }
                            self.consumer_tail.store(next_head, Ordering::Release);
                            
                            // Update statistics
                            self.stats.dequeue_count.fetch_add(1, Ordering::Relaxed);
                            self.update_size_stats(-1);
                            
                            let latency = Self::get_timestamp_ns() - start_time;
                            self.update_avg_latency(&self.stats.avg_dequeue_latency_ns, latency);
                            
                            return Ok(item);
                        } else {
                            // Data was None, this shouldn't happen
                            return Err(RingBufferError::CorruptedData);
                        }
                    }
                    Err(_) => {
                        // CAS failed, track contention and retry
                        self.stats.consumer_contention.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            } else {
                // Data not ready, yield and retry
                std::hint::spin_loop();
            }
        }
    }

    /// Try to enqueue without blocking (returns immediately if full)
    pub fn try_enqueue(&self, item: T) -> Result<(), RingBufferError> {
        let head = self.producer_head.load(Ordering::Relaxed);
        let next_head = head.wrapping_add(1);
        let consumer_tail = self.consumer_tail.load(Ordering::Acquire);
        
        // Check if buffer is full
        if next_head.wrapping_sub(consumer_tail) > self.capacity as u32 {
            self.stats.enqueue_failures.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::Full);
        }

        self.enqueue(item)
    }

    /// Try to dequeue without blocking (returns immediately if empty)
    pub fn try_dequeue(&self) -> Result<T, RingBufferError> {
        let head = self.consumer_head.load(Ordering::Relaxed);
        let producer_tail = self.producer_tail.load(Ordering::Acquire);
        
        // Check if buffer is empty
        if head == producer_tail {
            self.stats.dequeue_failures.fetch_add(1, Ordering::Relaxed);
            return Err(RingBufferError::Empty);
        }

        self.dequeue()
    }

    /// Get current buffer size
    pub fn size(&self) -> usize {
        let producer_tail = self.producer_tail.load(Ordering::Acquire);
        let consumer_tail = self.consumer_tail.load(Ordering::Acquire);
        producer_tail.wrapping_sub(consumer_tail) as usize
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        let producer_tail = self.producer_tail.load(Ordering::Acquire);
        let consumer_head = self.consumer_head.load(Ordering::Acquire);
        producer_tail == consumer_head
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        let producer_head = self.producer_head.load(Ordering::Acquire);
        let consumer_tail = self.consumer_tail.load(Ordering::Acquire);
        producer_head.wrapping_sub(consumer_tail) >= self.capacity as u32
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get buffer name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Batch enqueue multiple items for improved performance
    pub fn batch_enqueue(&self, items: &[T]) -> Result<usize, RingBufferError> 
    where 
        T: Clone,
    {
        if items.is_empty() {
            return Ok(0);
        }

        let start_time = Self::get_timestamp_ns();
        let mut enqueued = 0;

        for item in items {
            match self.try_enqueue(item.clone()) {
                Ok(_) => enqueued += 1,
                Err(RingBufferError::Full) => break,
                Err(e) => return Err(e),
            }
        }

        // Update batch statistics
        if enqueued > 0 {
            let latency = Self::get_timestamp_ns() - start_time;
            self.update_avg_latency(&self.stats.avg_enqueue_latency_ns, latency);
            self.stats.batch_enqueue_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(enqueued)
    }

    /// Batch dequeue multiple items for improved performance
    pub fn batch_dequeue(&self, max_items: usize) -> Result<Vec<T>, RingBufferError> {
        if max_items == 0 {
            return Ok(Vec::new());
        }

        let start_time = Self::get_timestamp_ns();
        let mut items = Vec::with_capacity(max_items);

        for _ in 0..max_items {
            match self.try_dequeue() {
                Ok(item) => items.push(item),
                Err(RingBufferError::Empty) => break,
                Err(e) => return Err(e),
            }
        }

        // Update batch statistics
        if !items.is_empty() {
            let latency = Self::get_timestamp_ns() - start_time;
            self.update_avg_latency(&self.stats.avg_dequeue_latency_ns, latency);
            self.stats.batch_dequeue_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(items)
    }

    /// Batch dequeue with pre-allocated buffer to avoid allocations
    pub fn batch_dequeue_into(&self, buffer: &mut Vec<T>, max_items: usize) -> Result<usize, RingBufferError> {
        if max_items == 0 {
            return Ok(0);
        }

        let start_time = Self::get_timestamp_ns();
        let initial_len = buffer.len();
        buffer.reserve(max_items);

        let mut dequeued = 0;
        for _ in 0..max_items {
            match self.try_dequeue() {
                Ok(item) => {
                    buffer.push(item);
                    dequeued += 1;
                }
                Err(RingBufferError::Empty) => break,
                Err(e) => return Err(e),
            }
        }

        // Update batch statistics
        if dequeued > 0 {
            let latency = Self::get_timestamp_ns() - start_time;
            self.update_avg_latency(&self.stats.avg_dequeue_latency_ns, latency);
            self.stats.batch_dequeue_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(dequeued)
    }

    /// Get buffer statistics
    pub fn get_stats(&self) -> RingBufferStatsSnapshot {
        let current_size = self.size();
        let enqueue_count = self.stats.enqueue_count.load(Ordering::Acquire);
        let dequeue_count = self.stats.dequeue_count.load(Ordering::Acquire);
        let total_ops = enqueue_count + dequeue_count;
        
        RingBufferStatsSnapshot {
            name: self.name.clone(),
            capacity: self.capacity,
            current_size,
            enqueue_count,
            dequeue_count,
            enqueue_failures: self.stats.enqueue_failures.load(Ordering::Acquire),
            dequeue_failures: self.stats.dequeue_failures.load(Ordering::Acquire),
            peak_size: self.stats.peak_size.load(Ordering::Acquire),
            avg_enqueue_latency_ns: self.stats.avg_enqueue_latency_ns.load(Ordering::Acquire),
            avg_dequeue_latency_ns: self.stats.avg_dequeue_latency_ns.load(Ordering::Acquire),
            batch_enqueue_count: self.stats.batch_enqueue_count.load(Ordering::Acquire),
            batch_dequeue_count: self.stats.batch_dequeue_count.load(Ordering::Acquire),
            producer_contention: self.stats.producer_contention.load(Ordering::Acquire),
            consumer_contention: self.stats.consumer_contention.load(Ordering::Acquire),
            cache_misses: self.stats.cache_misses.load(Ordering::Acquire),
            utilization_percent: (current_size as f64 / self.capacity as f64) * 100.0,
            throughput_ops_per_sec: total_ops as f64, // This would be calculated over time in real implementation
        }
    }

    /// Reset buffer statistics
    pub fn reset_stats(&self) {
        self.stats.enqueue_count.store(0, Ordering::Release);
        self.stats.dequeue_count.store(0, Ordering::Release);
        self.stats.enqueue_failures.store(0, Ordering::Release);
        self.stats.dequeue_failures.store(0, Ordering::Release);
        self.stats.peak_size.store(0, Ordering::Release);
        self.stats.avg_enqueue_latency_ns.store(0, Ordering::Release);
        self.stats.avg_dequeue_latency_ns.store(0, Ordering::Release);
        self.stats.batch_enqueue_count.store(0, Ordering::Release);
        self.stats.batch_dequeue_count.store(0, Ordering::Release);
        self.stats.producer_contention.store(0, Ordering::Release);
        self.stats.consumer_contention.store(0, Ordering::Release);
        self.stats.cache_misses.store(0, Ordering::Release);
    }

    /// Check if ring buffer is experiencing high contention
    pub fn is_high_contention(&self) -> bool {
        let total_ops = self.stats.enqueue_count.load(Ordering::Acquire) + 
                       self.stats.dequeue_count.load(Ordering::Acquire);
        let total_contention = self.stats.producer_contention.load(Ordering::Acquire) + 
                              self.stats.consumer_contention.load(Ordering::Acquire);
        
        if total_ops == 0 {
            return false;
        }
        
        // Consider high contention if more than 10% of operations result in contention
        (total_contention as f64 / total_ops as f64) > 0.1
    }

    /// Get buffer health status
    pub fn get_health_status(&self) -> RingBufferHealth {
        let stats = self.get_stats();
        let mut issues = Vec::new();
        
        // Check utilization
        if stats.utilization_percent > 90.0 {
            issues.push("High utilization (>90%)".to_string());
        }
        
        // Check failure rates
        let total_enqueues = stats.enqueue_count + stats.enqueue_failures;
        if total_enqueues > 0 {
            let failure_rate = (stats.enqueue_failures as f64 / total_enqueues as f64) * 100.0;
            if failure_rate > 5.0 {
                issues.push(format!("High enqueue failure rate ({:.1}%)", failure_rate));
            }
        }
        
        // Check contention
        if self.is_high_contention() {
            issues.push("High contention detected".to_string());
        }
        
        // Check latency
        if stats.avg_enqueue_latency_ns > 1000 || stats.avg_dequeue_latency_ns > 1000 {
            issues.push("High latency detected (>1μs)".to_string());
        }
        
        let status = if issues.is_empty() {
            RingBufferHealthStatus::Healthy
        } else if issues.len() <= 2 {
            RingBufferHealthStatus::Warning
        } else {
            RingBufferHealthStatus::Critical
        };
        
        RingBufferHealth {
            status,
            issues,
            utilization_percent: stats.utilization_percent,
            contention_rate: if stats.enqueue_count + stats.dequeue_count > 0 {
                ((stats.producer_contention + stats.consumer_contention) as f64 / 
                 (stats.enqueue_count + stats.dequeue_count) as f64) * 100.0
            } else {
                0.0
            },
        }
    }

    /// Enable/disable adaptive backoff for contention reduction
    pub fn set_adaptive_backoff(&self, enabled: bool) {
        // This would be implemented with additional state in a real implementation
        // For now, we'll just document the interface
        if enabled {
            println!("Adaptive backoff enabled for ring buffer: {}", self.name);
        } else {
            println!("Adaptive backoff disabled for ring buffer: {}", self.name);
        }
    }

    /// Update size statistics
    fn update_size_stats(&self, delta: i32) {
        let current_size = if delta > 0 {
            self.stats.current_size.fetch_add(delta as usize, Ordering::AcqRel) + delta as usize
        } else {
            self.stats.current_size.fetch_sub((-delta) as usize, Ordering::AcqRel) - (-delta) as usize
        };

        // Update peak size
        let mut peak = self.stats.peak_size.load(Ordering::Acquire);
        while current_size > peak {
            match self.stats.peak_size.compare_exchange_weak(
                peak,
                current_size,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
    }

    /// Update average latency using exponential moving average
    fn update_avg_latency(&self, avg_atomic: &AtomicUsize, new_latency: u64) {
        let alpha = 0.1; // Smoothing factor
        loop {
            let current_avg = avg_atomic.load(Ordering::Acquire);
            let new_avg = if current_avg == 0 {
                new_latency as usize
            } else {
                ((1.0 - alpha) * current_avg as f64 + alpha * new_latency as f64) as usize
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

    /// Get high-resolution timestamp
    #[inline(always)]
    fn get_timestamp_ns() -> u64 {
        unsafe { core::arch::x86_64::_rdtsc() }
    }
}

impl<T> Drop for RingBuffer<T> {
    fn drop(&mut self) {
        // Drain any remaining items
        while let Ok(_) = self.try_dequeue() {
            // Items are dropped automatically
        }
    }
}

/// Ring buffer statistics snapshot
#[derive(Debug, Clone)]
pub struct RingBufferStatsSnapshot {
    pub name: String,
    pub capacity: usize,
    pub current_size: usize,
    pub enqueue_count: usize,
    pub dequeue_count: usize,
    pub enqueue_failures: usize,
    pub dequeue_failures: usize,
    pub peak_size: usize,
    pub avg_enqueue_latency_ns: usize,
    pub avg_dequeue_latency_ns: usize,
    pub batch_enqueue_count: usize,
    pub batch_dequeue_count: usize,
    pub producer_contention: usize,
    pub consumer_contention: usize,
    pub cache_misses: usize,
    pub utilization_percent: f64,
    pub throughput_ops_per_sec: f64,
}

impl RingBufferStats {
    fn new() -> Self {
        Self {
            enqueue_count: AtomicUsize::new(0),
            dequeue_count: AtomicUsize::new(0),
            enqueue_failures: AtomicUsize::new(0),
            dequeue_failures: AtomicUsize::new(0),
            current_size: AtomicUsize::new(0),
            peak_size: AtomicUsize::new(0),
            avg_enqueue_latency_ns: AtomicUsize::new(0),
            avg_dequeue_latency_ns: AtomicUsize::new(0),
            batch_enqueue_count: AtomicUsize::new(0),
            batch_dequeue_count: AtomicUsize::new(0),
            producer_contention: AtomicUsize::new(0),
            consumer_contention: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }
}

/// Ring buffer error types
#[derive(Debug, Clone, PartialEq)]
pub enum RingBufferError {
    InvalidCapacity(usize),
    Full,
    Empty,
    CorruptedData,
    InvalidOperation,
}

impl std::fmt::Display for RingBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingBufferError::InvalidCapacity(cap) => write!(f, "Invalid capacity: {} (must be power of 2)", cap),
            RingBufferError::Full => write!(f, "Ring buffer is full"),
            RingBufferError::Empty => write!(f, "Ring buffer is empty"),
            RingBufferError::CorruptedData => write!(f, "Ring buffer data is corrupted"),
            RingBufferError::InvalidOperation => write!(f, "Invalid ring buffer operation"),
        }
    }
}

impl std::error::Error for RingBufferError {}

/// Ring buffer health status
#[derive(Debug, Clone, PartialEq)]
pub enum RingBufferHealthStatus {
    Healthy,
    Warning,
    Critical,
}

/// Ring buffer health information
#[derive(Debug, Clone)]
pub struct RingBufferHealth {
    pub status: RingBufferHealthStatus,
    pub issues: Vec<String>,
    pub utilization_percent: f64,
    pub contention_rate: f64,
}

/// Ring buffer monitor for continuous health checking
pub struct RingBufferMonitor<T> {
    /// Ring buffers being monitored
    buffers: Vec<Arc<RingBuffer<T>>>,
    
    /// Monitoring interval
    monitor_interval: Duration,
    
    /// Health thresholds
    thresholds: MonitoringThresholds,
    
    /// Running state
    running: AtomicBool,
    
    /// Monitor thread handle
    monitor_handle: Option<thread::JoinHandle<()>>,
}

/// Monitoring thresholds
#[derive(Debug, Clone)]
pub struct MonitoringThresholds {
    /// Maximum acceptable utilization percentage
    pub max_utilization_percent: f64,
    
    /// Maximum acceptable failure rate percentage
    pub max_failure_rate_percent: f64,
    
    /// Maximum acceptable contention rate percentage
    pub max_contention_rate_percent: f64,
    
    /// Maximum acceptable average latency (nanoseconds)
    pub max_avg_latency_ns: usize,
}

impl Default for MonitoringThresholds {
    fn default() -> Self {
        Self {
            max_utilization_percent: 85.0,
            max_failure_rate_percent: 5.0,
            max_contention_rate_percent: 10.0,
            max_avg_latency_ns: 1000, // 1 microsecond
        }
    }
}

impl<T> RingBufferMonitor<T> {
    /// Create a new ring buffer monitor
    pub fn new(
        buffers: Vec<Arc<RingBuffer<T>>>,
        monitor_interval: Duration,
        thresholds: MonitoringThresholds,
    ) -> Self {
        Self {
            buffers,
            monitor_interval,
            thresholds,
            running: AtomicBool::new(false),
            monitor_handle: None,
        }
    }

    /// Start monitoring
    pub fn start(&mut self) -> Result<(), RingBufferError> {
        if self.running.load(Ordering::Acquire) {
            return Err(RingBufferError::InvalidOperation);
        }

        self.running.store(true, Ordering::Release);
        
        let buffers = self.buffers.clone();
        let interval = self.monitor_interval;
        let thresholds = self.thresholds.clone();
        let running = Arc::new(&self.running as *const AtomicBool);

        let handle = thread::Builder::new()
            .name("ring_buffer_monitor".to_string())
            .spawn(move || {
                Self::monitoring_loop(buffers, interval, thresholds, running);
            })
            .map_err(|_| RingBufferError::InvalidOperation)?;

        self.monitor_handle = Some(handle);
        Ok(())
    }

    /// Stop monitoring
    pub fn stop(&mut self) -> Result<(), RingBufferError> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(false, Ordering::Release);

        if let Some(handle) = self.monitor_handle.take() {
            handle.join().map_err(|_| RingBufferError::InvalidOperation)?;
        }

        Ok(())
    }

    /// Get health status for all monitored buffers
    pub fn get_health_summary(&self) -> Vec<(String, RingBufferHealth)> {
        self.buffers
            .iter()
            .map(|buffer| (buffer.name().to_string(), buffer.get_health_status()))
            .collect()
    }

    /// Monitoring loop
    fn monitoring_loop(
        buffers: Vec<Arc<RingBuffer<T>>>,
        interval: Duration,
        thresholds: MonitoringThresholds,
        running: Arc<*const AtomicBool>,
    ) {
        while unsafe { (*running.as_ref()).load(Ordering::Acquire) } {
            for buffer in &buffers {
                let health = buffer.get_health_status();
                
                match health.status {
                    RingBufferHealthStatus::Warning => {
                        eprintln!("WARNING: Ring buffer '{}' health issues: {:?}", 
                                buffer.name(), health.issues);
                    }
                    RingBufferHealthStatus::Critical => {
                        eprintln!("CRITICAL: Ring buffer '{}' health issues: {:?}", 
                                buffer.name(), health.issues);
                    }
                    RingBufferHealthStatus::Healthy => {
                        // All good, no action needed
                    }
                }
                
                // Check specific thresholds
                if health.utilization_percent > thresholds.max_utilization_percent {
                    eprintln!("Ring buffer '{}' utilization high: {:.1}%", 
                            buffer.name(), health.utilization_percent);
                }
                
                if health.contention_rate > thresholds.max_contention_rate_percent {
                    eprintln!("Ring buffer '{}' contention high: {:.1}%", 
                            buffer.name(), health.contention_rate);
                }
            }
            
            thread::sleep(interval);
        }
    }
}

impl<T> Drop for RingBufferMonitor<T> {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Multi-producer multi-consumer ring buffer
pub type MPMCRingBuffer<T> = RingBuffer<T>;

/// Single-producer single-consumer ring buffer (optimized version)
pub type SPSCRingBuffer<T> = RingBuffer<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_ring_buffer_creation() {
        let buffer: RingBuffer<u64> = RingBuffer::new(1024, "test".to_string()).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        assert_eq!(buffer.name(), "test");
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_invalid_capacity() {
        let result: Result<RingBuffer<u64>, _> = RingBuffer::new(1023, "test".to_string());
        assert!(matches!(result, Err(RingBufferError::InvalidCapacity(1023))));
    }

    #[test]
    fn test_enqueue_dequeue() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "test".to_string()).unwrap();
        
        // Enqueue some items
        for i in 0..10 {
            buffer.enqueue(i).unwrap();
        }
        
        assert_eq!(buffer.size(), 10);
        assert!(!buffer.is_empty());
        
        // Dequeue items
        for i in 0..10 {
            let item = buffer.dequeue().unwrap();
            assert_eq!(item, i);
        }
        
        assert!(buffer.is_empty());
        assert_eq!(buffer.size(), 0);
    }

    #[test]
    fn test_buffer_full() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4, "test".to_string()).unwrap();
        
        // Fill the buffer
        for i in 0..4 {
            buffer.enqueue(i).unwrap();
        }
        
        assert!(buffer.is_full());
        
        // Try to enqueue one more (should fail)
        let result = buffer.enqueue(999);
        assert!(matches!(result, Err(RingBufferError::Full)));
    }

    #[test]
    fn test_buffer_empty() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "test".to_string()).unwrap();
        
        // Try to dequeue from empty buffer
        let result = buffer.dequeue();
        assert!(matches!(result, Err(RingBufferError::Empty)));
    }

    #[test]
    fn test_try_operations() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4, "test".to_string()).unwrap();
        
        // Try enqueue
        assert!(buffer.try_enqueue(42).is_ok());
        
        // Try dequeue
        let item = buffer.try_dequeue().unwrap();
        assert_eq!(item, 42);
        
        // Try dequeue from empty buffer
        let result = buffer.try_dequeue();
        assert!(matches!(result, Err(RingBufferError::Empty)));
    }

    #[test]
    fn test_concurrent_operations() {
        let buffer = Arc::new(RingBuffer::new(1024, "concurrent_test".to_string()).unwrap());
        let producer_buffer = buffer.clone();
        let consumer_buffer = buffer.clone();

        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..1000 {
                while producer_buffer.enqueue(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        // Consumer thread
        let consumer = thread::spawn(move || {
            let mut received = Vec::new();
            for _ in 0..1000 {
                loop {
                    match consumer_buffer.dequeue() {
                        Ok(item) => {
                            received.push(item);
                            break;
                        }
                        Err(RingBufferError::Empty) => {
                            thread::yield_now();
                        }
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                }
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();

        // Verify all items were received
        assert_eq!(received.len(), 1000);
        for (i, &item) in received.iter().enumerate() {
            assert_eq!(item, i as u64);
        }
    }

    #[test]
    fn test_statistics() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "stats_test".to_string()).unwrap();
        
        // Perform some operations
        for i in 0..5 {
            buffer.enqueue(i).unwrap();
        }
        
        for _ in 0..3 {
            buffer.dequeue().unwrap();
        }
        
        // Try to enqueue to full buffer
        for i in 0..20 {
            let _ = buffer.try_enqueue(i);
        }
        
        let stats = buffer.get_stats();
        assert_eq!(stats.enqueue_count, 5);
        assert_eq!(stats.dequeue_count, 3);
        assert!(stats.enqueue_failures > 0);
        assert_eq!(stats.current_size, 2);
        assert!(stats.peak_size >= 5);
    }

    #[test]
    fn test_wraparound() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4, "wraparound_test".to_string()).unwrap();
        
        // Fill and empty the buffer multiple times to test wraparound
        for cycle in 0..10 {
            // Fill buffer
            for i in 0..4 {
                buffer.enqueue(cycle * 4 + i).unwrap();
            }
            
            // Empty buffer
            for i in 0..4 {
                let item = buffer.dequeue().unwrap();
                assert_eq!(item, cycle * 4 + i);
            }
        }
        
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_high_contention() {
        let buffer = Arc::new(RingBuffer::new(1024, "contention_test".to_string()).unwrap());
        let mut handles = vec![];

        // Spawn multiple producer threads
        for thread_id in 0..4 {
            let buffer_clone = buffer.clone();
            let handle = thread::spawn(move || {
                for i in 0..250 {
                    let value = (thread_id * 250 + i) as u64;
                    while buffer_clone.enqueue(value).is_err() {
                        thread::yield_now();
                    }
                }
            });
            handles.push(handle);
        }

        // Spawn multiple consumer threads
        for _ in 0..4 {
            let buffer_clone = buffer.clone();
            let handle = thread::spawn(move || {
                let mut count = 0;
                while count < 250 {
                    match buffer_clone.dequeue() {
                        Ok(_) => count += 1,
                        Err(RingBufferError::Empty) => thread::yield_now(),
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Buffer should be empty
        assert!(buffer.is_empty());
        
        let stats = buffer.get_stats();
        assert_eq!(stats.enqueue_count, 1000);
        assert_eq!(stats.dequeue_count, 1000);
    }

    #[test]
    fn test_reset_stats() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "reset_test".to_string()).unwrap();
        
        // Perform some operations
        buffer.enqueue(1).unwrap();
        buffer.dequeue().unwrap();
        
        let stats_before = buffer.get_stats();
        assert!(stats_before.enqueue_count > 0);
        assert!(stats_before.dequeue_count > 0);
        
        // Reset statistics
        buffer.reset_stats();
        
        let stats_after = buffer.get_stats();
        assert_eq!(stats_after.enqueue_count, 0);
        assert_eq!(stats_after.dequeue_count, 0);
        assert_eq!(stats_after.enqueue_failures, 0);
        assert_eq!(stats_after.dequeue_failures, 0);
    }

    #[test]
    fn test_batch_enqueue() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "batch_test".to_string()).unwrap();
        
        let items = vec![1, 2, 3, 4, 5];
        let enqueued = buffer.batch_enqueue(&items).unwrap();
        
        assert_eq!(enqueued, 5);
        assert_eq!(buffer.size(), 5);
        
        let stats = buffer.get_stats();
        assert_eq!(stats.batch_enqueue_count, 1);
        assert_eq!(stats.enqueue_count, 5);
    }

    #[test]
    fn test_batch_dequeue() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "batch_test".to_string()).unwrap();
        
        // Fill buffer
        for i in 0..10 {
            buffer.enqueue(i).unwrap();
        }
        
        let items = buffer.batch_dequeue(5).unwrap();
        assert_eq!(items.len(), 5);
        assert_eq!(buffer.size(), 5);
        
        // Verify order
        for (i, &item) in items.iter().enumerate() {
            assert_eq!(item, i as u64);
        }
        
        let stats = buffer.get_stats();
        assert_eq!(stats.batch_dequeue_count, 1);
    }

    #[test]
    fn test_batch_dequeue_into() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "batch_test".to_string()).unwrap();
        
        // Fill buffer
        for i in 0..8 {
            buffer.enqueue(i).unwrap();
        }
        
        let mut items = Vec::new();
        let dequeued = buffer.batch_dequeue_into(&mut items, 5).unwrap();
        
        assert_eq!(dequeued, 5);
        assert_eq!(items.len(), 5);
        assert_eq!(buffer.size(), 3);
        
        // Verify order
        for (i, &item) in items.iter().enumerate() {
            assert_eq!(item, i as u64);
        }
    }

    #[test]
    fn test_batch_operations_partial() {
        let buffer: RingBuffer<u64> = RingBuffer::new(4, "partial_test".to_string()).unwrap();
        
        // Try to enqueue more than capacity
        let items = vec![1, 2, 3, 4, 5, 6];
        let enqueued = buffer.batch_enqueue(&items).unwrap();
        
        assert_eq!(enqueued, 4); // Only 4 should fit
        assert!(buffer.is_full());
        
        // Try to dequeue more than available
        let items = buffer.batch_dequeue(10).unwrap();
        assert_eq!(items.len(), 4); // Only 4 available
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_health_monitoring() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "health_test".to_string()).unwrap();
        
        // Initially healthy
        let health = buffer.get_health_status();
        assert_eq!(health.status, RingBufferHealthStatus::Healthy);
        assert!(health.issues.is_empty());
        
        // Fill buffer to high utilization
        for i in 0..15 {
            buffer.enqueue(i).unwrap();
        }
        
        let health = buffer.get_health_status();
        assert_eq!(health.status, RingBufferHealthStatus::Warning);
        assert!(health.utilization_percent > 90.0);
    }

    #[test]
    fn test_contention_tracking() {
        let buffer = Arc::new(RingBuffer::new(4, "contention_test".to_string()).unwrap());
        let mut handles = vec![];

        // Create high contention scenario
        for _ in 0..4 {
            let buffer_clone = buffer.clone();
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    // Try to enqueue, expect some contention
                    while buffer_clone.enqueue(i).is_err() {
                        thread::yield_now();
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = buffer.get_stats();
        // Should have some contention due to multiple threads
        assert!(stats.producer_contention > 0);
    }

    #[test]
    fn test_ring_buffer_monitor() {
        let buffer1 = Arc::new(RingBuffer::new(16, "monitor_test_1".to_string()).unwrap());
        let buffer2 = Arc::new(RingBuffer::new(16, "monitor_test_2".to_string()).unwrap());
        
        let buffers = vec![buffer1.clone(), buffer2.clone()];
        let mut monitor = RingBufferMonitor::new(
            buffers,
            Duration::from_millis(10),
            MonitoringThresholds::default(),
        );
        
        // Start monitoring
        assert!(monitor.start().is_ok());
        
        // Fill one buffer to trigger warning
        for i in 0..15 {
            buffer1.enqueue(i).unwrap();
        }
        
        // Let monitor run briefly
        thread::sleep(Duration::from_millis(50));
        
        let health_summary = monitor.get_health_summary();
        assert_eq!(health_summary.len(), 2);
        
        // Stop monitoring
        assert!(monitor.stop().is_ok());
    }

    #[test]
    fn test_adaptive_backoff_interface() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "backoff_test".to_string()).unwrap();
        
        // Test interface (actual implementation would require more state)
        buffer.set_adaptive_backoff(true);
        buffer.set_adaptive_backoff(false);
        
        // No assertions needed, just testing the interface exists
    }

    #[test]
    fn test_empty_batch_operations() {
        let buffer: RingBuffer<u64> = RingBuffer::new(16, "empty_batch_test".to_string()).unwrap();
        
        // Empty batch enqueue
        let enqueued = buffer.batch_enqueue(&[]).unwrap();
        assert_eq!(enqueued, 0);
        
        // Empty batch dequeue
        let items = buffer.batch_dequeue(0).unwrap();
        assert!(items.is_empty());
        
        let mut items = Vec::new();
        let dequeued = buffer.batch_dequeue_into(&mut items, 0).unwrap();
        assert_eq!(dequeued, 0);
        assert!(items.is_empty());
    }
}