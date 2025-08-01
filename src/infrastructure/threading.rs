use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::collections::HashMap;
use parking_lot::{RwLock, Mutex};
use crossbeam::channel::{self, Receiver, Sender};
use crossbeam::utils::Backoff;
use crate::error::InfrastructureError;

/// High-performance threading system with CPU affinity and NUMA awareness
pub struct ThreadingSystem {
    thread_pools: Arc<RwLock<HashMap<String, ThreadPool>>>,
    cpu_topology: CpuTopology,
    thread_registry: Arc<RwLock<HashMap<thread::ThreadId, ThreadInfo>>>,
    performance_monitor: Arc<PerformanceMonitor>,
}

#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub numa_nodes: usize,
    pub cpu_to_numa: HashMap<usize, usize>,
    pub numa_to_cpus: HashMap<usize, Vec<usize>>,
    pub cache_topology: CacheTopology,
}

#[derive(Debug, Clone)]
pub struct CacheTopology {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,
}

#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub id: thread::ThreadId,
    pub name: String,
    pub cpu_affinity: Option<usize>,
    pub numa_node: Option<usize>,
    pub priority: ThreadPriority,
    pub created_at: std::time::Instant,
    pub last_activity: std::time::Instant,
}

#[derive(Debug, Clone, Copy)]
pub enum ThreadPriority {
    Low,
    Normal,
    High,
    RealTime,
}

/// Lock-free thread pool for order processing
pub struct ThreadPool {
    name: String,
    workers: Vec<Worker>,
    task_queue: crossbeam::queue::SegQueue<Task>,
    shutdown: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
    completed_tasks: Arc<AtomicUsize>,
    cpu_affinity_mask: Option<u64>,
}

struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
    cpu_affinity: Option<usize>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

/// Performance monitoring for threading system
pub struct PerformanceMonitor {
    thread_stats: Arc<RwLock<HashMap<thread::ThreadId, ThreadStats>>>,
    pool_stats: Arc<RwLock<HashMap<String, PoolStats>>>,
    global_stats: Arc<RwLock<GlobalThreadingStats>>,
}

#[derive(Debug, Default)]
pub struct ThreadStats {
    pub tasks_executed: usize,
    pub total_execution_time: std::time::Duration,
    pub avg_execution_time: std::time::Duration,
    pub max_execution_time: std::time::Duration,
    pub cpu_usage: f64,
    pub context_switches: usize,
}

#[derive(Debug, Default)]
pub struct PoolStats {
    pub active_threads: usize,
    pub queued_tasks: usize,
    pub completed_tasks: usize,
    pub avg_queue_time: std::time::Duration,
    pub throughput: f64, // tasks per second
}

#[derive(Debug, Default)]
pub struct GlobalThreadingStats {
    pub total_threads: usize,
    pub active_threads: usize,
    pub total_tasks_executed: usize,
    pub avg_cpu_usage: f64,
    pub memory_usage: usize,
}

impl ThreadingSystem {
    pub fn new() -> Result<Self, InfrastructureError> {
        let cpu_topology = Self::detect_cpu_topology()?;
        
        Ok(Self {
            thread_pools: Arc::new(RwLock::new(HashMap::new())),
            cpu_topology,
            thread_registry: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
        })
    }

    /// Detect CPU topology and NUMA configuration
    fn detect_cpu_topology() -> Result<CpuTopology, InfrastructureError> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_topology()
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback topology detection
            let logical_cores = num_cpus::get();
            let physical_cores = num_cpus::get_physical();
            
            Ok(CpuTopology {
                logical_cores,
                physical_cores,
                numa_nodes: 1,
                cpu_to_numa: (0..logical_cores).map(|i| (i, 0)).collect(),
                numa_to_cpus: [(0, (0..logical_cores).collect())].into_iter().collect(),
                cache_topology: CacheTopology {
                    l1_cache_size: 32 * 1024,   // 32KB
                    l2_cache_size: 256 * 1024,  // 256KB
                    l3_cache_size: 8 * 1024 * 1024, // 8MB
                    cache_line_size: 64,
                },
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_topology() -> Result<CpuTopology, InfrastructureError> {
        use std::fs;
        
        let logical_cores = num_cpus::get();
        let physical_cores = num_cpus::get_physical();
        
        // Detect NUMA nodes
        let mut numa_nodes = 0;
        let mut cpu_to_numa = HashMap::new();
        let mut numa_to_cpus: HashMap<usize, Vec<usize>> = HashMap::new();
        
        // Read NUMA topology
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node/") {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("node") {
                            if let Ok(node_id) = name[4..].parse::<usize>() {
                                numa_nodes = numa_nodes.max(node_id + 1);
                                
                                // Read CPU list for this node
                                let cpulist_path = path.join("cpulist");
                                if let Ok(cpulist) = fs::read_to_string(cpulist_path) {
                                    let cpus = Self::parse_cpu_list(&cpulist)?;
                                    for cpu in &cpus {
                                        cpu_to_numa.insert(*cpu, node_id);
                                    }
                                    numa_to_cpus.insert(node_id, cpus);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback if NUMA detection fails
        if numa_nodes == 0 {
            numa_nodes = 1;
            cpu_to_numa = (0..logical_cores).map(|i| (i, 0)).collect();
            numa_to_cpus.insert(0, (0..logical_cores).collect());
        }
        
        // Detect cache topology
        let cache_topology = Self::detect_cache_topology()?;
        
        Ok(CpuTopology {
            logical_cores,
            physical_cores,
            numa_nodes,
            cpu_to_numa,
            numa_to_cpus,
            cache_topology,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(cpulist: &str) -> Result<Vec<usize>, InfrastructureError> {
        let mut cpus = Vec::new();
        
        for range in cpulist.trim().split(',') {
            if let Some((start, end)) = range.split_once('-') {
                let start: usize = start.parse()
                    .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
                let end: usize = end.parse()
                    .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
                
                for cpu in start..=end {
                    cpus.push(cpu);
                }
            } else {
                let cpu: usize = range.parse()
                    .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
                cpus.push(cpu);
            }
        }
        
        Ok(cpus)
    }

    #[cfg(target_os = "linux")]
    fn detect_cache_topology() -> Result<CacheTopology, InfrastructureError> {
        use std::fs;
        
        let mut l1_cache_size = 32 * 1024;   // Default 32KB
        let mut l2_cache_size = 256 * 1024;  // Default 256KB
        let mut l3_cache_size = 8 * 1024 * 1024; // Default 8MB
        let cache_line_size = 64; // Standard cache line size
        
        // Try to read actual cache sizes from /sys/devices/system/cpu/cpu0/cache/
        if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu/cpu0/cache/") {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("index") {
                            let level_path = path.join("level");
                            let size_path = path.join("size");
                            
                            if let (Ok(level_str), Ok(size_str)) = (
                                fs::read_to_string(level_path),
                                fs::read_to_string(size_path)
                            ) {
                                if let Ok(level) = level_str.trim().parse::<u32>() {
                                    if let Ok(size) = Self::parse_cache_size(&size_str) {
                                        match level {
                                            1 => l1_cache_size = size,
                                            2 => l2_cache_size = size,
                                            3 => l3_cache_size = size,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(CacheTopology {
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            cache_line_size,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_cache_size(size_str: &str) -> Result<usize, InfrastructureError> {
        let size_str = size_str.trim();
        
        if size_str.ends_with('K') {
            let num: usize = size_str[..size_str.len()-1].parse()
                .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
            Ok(num * 1024)
        } else if size_str.ends_with('M') {
            let num: usize = size_str[..size_str.len()-1].parse()
                .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
            Ok(num * 1024 * 1024)
        } else {
            size_str.parse()
                .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))
        }
    }

    /// Create a new thread pool with CPU affinity
    pub fn create_thread_pool(
        &self,
        name: String,
        size: usize,
        cpu_affinity_mask: Option<u64>,
        priority: ThreadPriority,
    ) -> Result<(), InfrastructureError> {
        let pool = ThreadPool::new(name.clone(), size, cpu_affinity_mask, priority)?;
        
        let mut pools = self.thread_pools.write();
        pools.insert(name, pool);
        
        Ok(())
    }

    /// Submit task to specific thread pool
    pub fn submit_task<F>(&self, pool_name: &str, task: F) -> Result<(), InfrastructureError>
    where
        F: FnOnce() + Send + 'static,
    {
        let pools = self.thread_pools.read();
        let pool = pools.get(pool_name)
            .ok_or_else(|| InfrastructureError::ThreadingError(format!("Pool '{}' not found", pool_name)))?;
        
        pool.submit(Box::new(task));
        Ok(())
    }

    /// Set CPU affinity for current thread
    pub fn set_cpu_affinity(&self, cpu_id: usize) -> Result<(), InfrastructureError> {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpu_id, &mut cpu_set);
                
                let result = libc::sched_setaffinity(
                    0, // Current thread
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpu_set,
                );
                
                if result != 0 {
                    return Err(InfrastructureError::ThreadingError(
                        format!("Failed to set CPU affinity to {}", cpu_id)
                    ));
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // CPU affinity not supported on this platform
            println!("Warning: CPU affinity not supported on this platform");
        }
        
        Ok(())
    }

    /// Set thread priority
    pub fn set_thread_priority(&self, priority: ThreadPriority) -> Result<(), InfrastructureError> {
        #[cfg(unix)]
        {
            let policy = match priority {
                ThreadPriority::Low => libc::SCHED_IDLE,
                ThreadPriority::Normal => libc::SCHED_OTHER,
                ThreadPriority::High => libc::SCHED_FIFO,
                ThreadPriority::RealTime => libc::SCHED_RR,
            };
            
            let param = libc::sched_param {
                sched_priority: match priority {
                    ThreadPriority::Low => 0,
                    ThreadPriority::Normal => 0,
                    ThreadPriority::High => 50,
                    ThreadPriority::RealTime => 99,
                },
            };
            
            unsafe {
                let result = libc::sched_setscheduler(0, policy, &param);
                if result != 0 {
                    return Err(InfrastructureError::ThreadingError(
                        "Failed to set thread priority".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// Get optimal CPU for NUMA-aware thread placement
    pub fn get_optimal_cpu(&self, numa_node: Option<usize>) -> Option<usize> {
        if let Some(node) = numa_node {
            if let Some(cpus) = self.cpu_topology.numa_to_cpus.get(&node) {
                // Return least loaded CPU in the NUMA node
                // For now, just return the first CPU
                cpus.first().copied()
            } else {
                None
            }
        } else {
            // Return any available CPU
            Some(0)
        }
    }

    /// Get threading system statistics
    pub fn get_stats(&self) -> GlobalThreadingStats {
        self.performance_monitor.get_global_stats()
    }

    /// Get CPU topology information
    pub fn get_cpu_topology(&self) -> &CpuTopology {
        &self.cpu_topology
    }
}

impl ThreadPool {
    pub fn new(
        name: String,
        size: usize,
        cpu_affinity_mask: Option<u64>,
        priority: ThreadPriority,
    ) -> Result<Self, InfrastructureError> {
        let task_queue = crossbeam::queue::SegQueue::new();
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let completed_tasks = Arc::new(AtomicUsize::new(0));
        
        let mut workers = Vec::with_capacity(size);
        
        for i in 0..size {
            let cpu_affinity = if let Some(mask) = cpu_affinity_mask {
                // Find the i-th set bit in the mask
                let mut cpu_id = None;
                let mut bit_count = 0;
                for bit in 0..64 {
                    if mask & (1u64 << bit) != 0 {
                        if bit_count == i {
                            cpu_id = Some(bit);
                            break;
                        }
                        bit_count += 1;
                    }
                }
                cpu_id
            } else {
                None
            };
            
            let worker = Worker::new(
                i,
                &task_queue,
                Arc::clone(&shutdown),
                Arc::clone(&active_tasks),
                Arc::clone(&completed_tasks),
                cpu_affinity,
                priority,
            )?;
            
            workers.push(worker);
        }
        
        Ok(Self {
            name,
            workers,
            task_queue,
            shutdown,
            active_tasks,
            completed_tasks,
            cpu_affinity_mask,
        })
    }

    pub fn submit(&self, task: Task) {
        self.task_queue.push(task);
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            active_threads: self.workers.len(),
            queued_tasks: self.task_queue.len(),
            completed_tasks: self.completed_tasks.load(Ordering::Relaxed),
            avg_queue_time: std::time::Duration::from_millis(0), // Placeholder
            throughput: 0.0, // Placeholder
        }
    }
}

impl Worker {
    fn new(
        id: usize,
        task_queue: &crossbeam::queue::SegQueue<Task>,
        shutdown: Arc<AtomicBool>,
        active_tasks: Arc<AtomicUsize>,
        completed_tasks: Arc<AtomicUsize>,
        cpu_affinity: Option<usize>,
        priority: ThreadPriority,
    ) -> Result<Self, InfrastructureError> {
        let task_queue = unsafe { 
            // SAFETY: We're creating a reference with the same lifetime as the worker
            std::mem::transmute::<&crossbeam::queue::SegQueue<Task>, &'static crossbeam::queue::SegQueue<Task>>(task_queue)
        };
        
        let thread = thread::Builder::new()
            .name(format!("worker-{}", id))
            .spawn(move || {
                // Set CPU affinity if specified
                if let Some(cpu_id) = cpu_affinity {
                    #[cfg(target_os = "linux")]
                    unsafe {
                        let mut cpu_set: libc::cpu_set_t = std::mem::zeroed();
                        libc::CPU_SET(cpu_id, &mut cpu_set);
                        libc::sched_setaffinity(
                            0,
                            std::mem::size_of::<libc::cpu_set_t>(),
                            &cpu_set,
                        );
                    }
                }
                
                // Set thread priority
                #[cfg(unix)]
                {
                    let policy = match priority {
                        ThreadPriority::Low => libc::SCHED_IDLE,
                        ThreadPriority::Normal => libc::SCHED_OTHER,
                        ThreadPriority::High => libc::SCHED_FIFO,
                        ThreadPriority::RealTime => libc::SCHED_RR,
                    };
                    
                    let param = libc::sched_param {
                        sched_priority: match priority {
                            ThreadPriority::Low => 0,
                            ThreadPriority::Normal => 0,
                            ThreadPriority::High => 50,
                            ThreadPriority::RealTime => 99,
                        },
                    };
                    
                    unsafe {
                        libc::sched_setscheduler(0, policy, &param);
                    }
                }
                
                let backoff = Backoff::new();
                
                // Main worker loop
                while !shutdown.load(Ordering::Relaxed) {
                    if let Some(task) = task_queue.pop() {
                        active_tasks.fetch_add(1, Ordering::Relaxed);
                        
                        // Execute task
                        task();
                        
                        active_tasks.fetch_sub(1, Ordering::Relaxed);
                        completed_tasks.fetch_add(1, Ordering::Relaxed);
                        
                        backoff.reset();
                    } else {
                        backoff.snooze();
                    }
                }
            })
            .map_err(|e| InfrastructureError::ThreadingError(e.to_string()))?;
        
        Ok(Self {
            id,
            thread: Some(thread),
            cpu_affinity,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            thread_stats: Arc::new(RwLock::new(HashMap::new())),
            pool_stats: Arc::new(RwLock::new(HashMap::new())),
            global_stats: Arc::new(RwLock::new(GlobalThreadingStats::default())),
        }
    }

    pub fn record_task_execution(&self, thread_id: thread::ThreadId, duration: std::time::Duration) {
        let mut stats = self.thread_stats.write();
        let thread_stats = stats.entry(thread_id).or_default();
        
        thread_stats.tasks_executed += 1;
        thread_stats.total_execution_time += duration;
        thread_stats.avg_execution_time = thread_stats.total_execution_time / thread_stats.tasks_executed as u32;
        
        if duration > thread_stats.max_execution_time {
            thread_stats.max_execution_time = duration;
        }
    }

    pub fn get_global_stats(&self) -> GlobalThreadingStats {
        self.global_stats.read().clone()
    }

    pub fn get_thread_stats(&self, thread_id: thread::ThreadId) -> Option<ThreadStats> {
        self.thread_stats.read().get(&thread_id).cloned()
    }
}

/// Lock-free data structures for high-performance order processing
pub mod lockfree {
    use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
    use std::ptr;
    
    /// Lock-free queue for order messages
    pub struct LockFreeQueue<T> {
        head: AtomicPtr<Node<T>>,
        tail: AtomicPtr<Node<T>>,
        size: AtomicUsize,
    }
    
    struct Node<T> {
        data: Option<T>,
        next: AtomicPtr<Node<T>>,
    }
    
    impl<T> LockFreeQueue<T> {
        pub fn new() -> Self {
            let dummy = Box::into_raw(Box::new(Node {
                data: None,
                next: AtomicPtr::new(ptr::null_mut()),
            }));
            
            Self {
                head: AtomicPtr::new(dummy),
                tail: AtomicPtr::new(dummy),
                size: AtomicUsize::new(0),
            }
        }
        
        pub fn enqueue(&self, item: T) {
            let new_node = Box::into_raw(Box::new(Node {
                data: Some(item),
                next: AtomicPtr::new(ptr::null_mut()),
            }));
            
            loop {
                let tail = self.tail.load(Ordering::Acquire);
                let next = unsafe { (*tail).next.load(Ordering::Acquire) };
                
                if tail == self.tail.load(Ordering::Acquire) {
                    if next.is_null() {
                        if unsafe { (*tail).next.compare_exchange_weak(
                            next,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        ).is_ok() } {
                            break;
                        }
                    } else {
                        let _ = self.tail.compare_exchange_weak(
                            tail,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    }
                }
            }
            
            let _ = self.tail.compare_exchange_weak(
                self.tail.load(Ordering::Acquire),
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            );
            
            self.size.fetch_add(1, Ordering::Relaxed);
        }
        
        pub fn dequeue(&self) -> Option<T> {
            loop {
                let head = self.head.load(Ordering::Acquire);
                let tail = self.tail.load(Ordering::Acquire);
                let next = unsafe { (*head).next.load(Ordering::Acquire) };
                
                if head == self.head.load(Ordering::Acquire) {
                    if head == tail {
                        if next.is_null() {
                            return None;
                        }
                        let _ = self.tail.compare_exchange_weak(
                            tail,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                    } else {
                        if next.is_null() {
                            continue;
                        }
                        
                        let data = unsafe { (*next).data.take() };
                        
                        if self.head.compare_exchange_weak(
                            head,
                            next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        ).is_ok() {
                            unsafe { Box::from_raw(head) };
                            self.size.fetch_sub(1, Ordering::Relaxed);
                            return data;
                        }
                    }
                }
            }
        }
        
        pub fn len(&self) -> usize {
            self.size.load(Ordering::Relaxed)
        }
        
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }
    
    impl<T> Drop for LockFreeQueue<T> {
        fn drop(&mut self) {
            while self.dequeue().is_some() {}
            
            let head = self.head.load(Ordering::Relaxed);
            if !head.is_null() {
                unsafe { Box::from_raw(head) };
            }
        }
    }
    
    unsafe impl<T: Send> Send for LockFreeQueue<T> {}
    unsafe impl<T: Send> Sync for LockFreeQueue<T> {}
}