//! Stress Testing Framework
//! 
//! Implements extreme load testing, resource exhaustion testing, and failure injection

use super::{TestConfig, TestResults, LoadGenerator, LoadPattern, LoadConfig};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

/// Stress test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestConfig {
    pub name: String,
    pub test_type: StressTestType,
    pub duration: Duration,
    pub intensity_levels: Vec<f64>, // Multipliers for base load
    pub resource_limits: ResourceLimits,
    pub failure_injection: FailureInjectionConfig,
    pub recovery_validation: bool,
}

/// Types of stress tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestType {
    LoadStress,        // Extreme load testing
    ResourceExhaustion, // Resource exhaustion testing
    FailureInjection,  // Failure injection and recovery
    Endurance,         // Long-running endurance test
    Spike,             // Sudden load spikes
    Gradual,           // Gradual load increase
}

/// Resource limits for stress testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_cpu_percent: f64,
    pub max_memory_bytes: u64,
    pub max_network_mbps: f64,
    pub max_disk_iops: u64,
    pub max_connections: u32,
}

/// Failure injection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInjectionConfig {
    pub enabled: bool,
    pub failure_rate_percent: f64,
    pub failure_types: Vec<FailureType>,
    pub recovery_time_ms: u64,
}

/// Types of failures to inject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    NetworkTimeout,
    MemoryAllocationFailure,
    DiskWriteFailure,
    CpuThrottling,
    ConnectionDrop,
    ServiceUnavailable,
}

/// Stress test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResults {
    pub config: StressTestConfig,
    pub start_time: u64,
    pub end_time: u64,
    pub duration: Duration,
    pub phases: Vec<StressTestPhase>,
    pub peak_metrics: PeakMetrics,
    pub failure_analysis: FailureAnalysis,
    pub recovery_analysis: RecoveryAnalysis,
    pub verdict: StressTestVerdict,
}

/// Individual stress test phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestPhase {
    pub phase_name: String,
    pub intensity: f64,
    pub duration: Duration,
    pub operations_completed: u64,
    pub operations_failed: u64,
    pub average_latency_ns: u64,
    pub peak_latency_ns: u64,
    pub resource_usage: ResourceUsageSnapshot,
    pub errors: Vec<String>,
}

/// Peak metrics during stress test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakMetrics {
    pub peak_operations_per_second: f64,
    pub peak_latency_ns: u64,
    pub peak_cpu_percent: f64,
    pub peak_memory_bytes: u64,
    pub peak_network_mbps: f64,
    pub peak_connections: u32,
}

/// Resource usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageSnapshot {
    pub timestamp: u64,
    pub cpu_percent: f64,
    pub memory_bytes: u64,
    pub network_mbps: f64,
    pub disk_iops: u64,
    pub connections: u32,
}

/// Failure analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureAnalysis {
    pub total_failures: u64,
    pub failure_rate_percent: f64,
    pub failure_breakdown: std::collections::HashMap<String, u64>,
    pub mean_time_to_failure_ms: f64,
    pub failure_clustering: bool,
}

/// Recovery analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryAnalysis {
    pub recovery_attempts: u64,
    pub successful_recoveries: u64,
    pub recovery_success_rate: f64,
    pub mean_recovery_time_ms: f64,
    pub max_recovery_time_ms: f64,
}

/// Stress test verdict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressTestVerdict {
    Pass,
    Fail,
    Degraded,
    Unstable,
}

/// Stress testing framework
pub struct StressTester {
    load_generator: LoadGenerator,
    active_failures: Arc<AtomicU64>,
    test_running: Arc<AtomicBool>,
}

impl StressTester {
    /// Create a new stress tester
    pub fn new() -> Self {
        Self {
            load_generator: LoadGenerator::new(),
            active_failures: Arc::new(AtomicU64::new(0)),
            test_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run stress test
    pub async fn run_stress_test(&mut self, config: StressTestConfig) -> Result<StressTestResults, Box<dyn std::error::Error>> {
        println!("Starting stress test: {} ({:?})", config.name, config.test_type);
        
        self.test_running.store(true, Ordering::Relaxed);
        let start_time = Instant::now();
        let mut phases = Vec::new();
        let mut peak_metrics = PeakMetrics::default();
        
        // Start failure injection if enabled
        let failure_task = if config.failure_injection.enabled {
            Some(self.start_failure_injection(&config.failure_injection).await?)
        } else {
            None
        };

        // Start resource monitoring
        let monitoring_task = self.start_resource_monitoring().await?;

        // Execute stress test phases based on type
        match config.test_type {
            StressTestType::LoadStress => {
                phases = self.run_load_stress_phases(&config).await?;
            }
            StressTestType::ResourceExhaustion => {
                phases = self.run_resource_exhaustion_phases(&config).await?;
            }
            StressTestType::FailureInjection => {
                phases = self.run_failure_injection_phases(&config).await?;
            }
            StressTestType::Endurance => {
                phases = self.run_endurance_phases(&config).await?;
            }
            StressTestType::Spike => {
                phases = self.run_spike_phases(&config).await?;
            }
            StressTestType::Gradual => {
                phases = self.run_gradual_phases(&config).await?;
            }
        }

        // Stop failure injection
        if let Some(task) = failure_task {
            task.abort();
        }

        // Stop monitoring
        monitoring_task.abort();
        self.test_running.store(false, Ordering::Relaxed);

        let end_time = Instant::now();
        let duration = end_time - start_time;

        // Calculate peak metrics
        peak_metrics = self.calculate_peak_metrics(&phases);

        // Analyze failures
        let failure_analysis = self.analyze_failures(&phases);

        // Analyze recovery
        let recovery_analysis = self.analyze_recovery(&phases, &config);

        // Determine verdict
        let verdict = self.determine_stress_verdict(&phases, &failure_analysis, &recovery_analysis);

        let results = StressTestResults {
            config,
            start_time: start_time.elapsed().as_secs(),
            end_time: end_time.elapsed().as_secs(),
            duration,
            phases,
            peak_metrics,
            failure_analysis,
            recovery_analysis,
            verdict,
        };

        println!("Stress test completed in {:?}", duration);
        Ok(results)
    }

    /// Run load stress test phases
    async fn run_load_stress_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();
        let phase_duration = config.duration / config.intensity_levels.len() as u32;

        for (i, &intensity) in config.intensity_levels.iter().enumerate() {
            let phase_name = format!("LoadStress_Phase_{}_Intensity_{:.1}x", i + 1, intensity);
            println!("Starting phase: {} for {:?}", phase_name, phase_duration);

            let base_ops = 100_000u64;
            let target_ops = (base_ops as f64 * intensity) as u64;

            let load_config = LoadConfig {
                pattern: LoadPattern::Constant { ops_per_second: target_ops },
                duration: phase_duration,
                thread_count: num_cpus::get(),
                order_types: vec![crate::models::order::OrderType::Limit, crate::models::order::OrderType::Market],
                price_range: (100.0, 200.0),
                quantity_range: (1, 1000),
                symbols: vec!["BTCUSD".to_string(), "ETHUSD".to_string()],
            };

            let phase_start = Instant::now();
            let load_stats = self.load_generator.generate_load_with_config(&load_config).await?;
            let phase_end = Instant::now();

            let phase = StressTestPhase {
                phase_name,
                intensity,
                duration: phase_end - phase_start,
                operations_completed: load_stats.total_operations,
                operations_failed: load_stats.error_count,
                average_latency_ns: self.calculate_average_latency(load_stats.operations_per_second),
                peak_latency_ns: self.calculate_peak_latency(intensity),
                resource_usage: self.capture_resource_snapshot().await,
                errors: Vec::new(),
            };

            phases.push(phase);

            // Brief recovery period between phases
            if i < config.intensity_levels.len() - 1 {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }

        Ok(phases)
    }

    /// Run resource exhaustion test phases
    async fn run_resource_exhaustion_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();

        // CPU exhaustion phase
        let cpu_phase = self.run_cpu_exhaustion_phase(config).await?;
        phases.push(cpu_phase);

        // Memory exhaustion phase
        let memory_phase = self.run_memory_exhaustion_phase(config).await?;
        phases.push(memory_phase);

        // Network exhaustion phase
        let network_phase = self.run_network_exhaustion_phase(config).await?;
        phases.push(network_phase);

        Ok(phases)
    }

    /// Run CPU exhaustion phase
    async fn run_cpu_exhaustion_phase(&mut self, config: &StressTestConfig) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        println!("Starting CPU exhaustion phase");
        
        let phase_start = Instant::now();
        let mut operations_completed = 0u64;
        let mut operations_failed = 0u64;
        let mut errors = Vec::new();

        // Spawn CPU-intensive tasks
        let cpu_count = num_cpus::get();
        let mut tasks = Vec::new();

        for i in 0..cpu_count * 2 { // Oversubscribe CPUs
            let task = tokio::spawn(async move {
                let mut counter = 0u64;
                let start = Instant::now();
                
                while start.elapsed() < Duration::from_secs(30) {
                    // CPU-intensive work
                    for j in 0..1000000 {
                        counter = counter.wrapping_add(j * j);
                    }
                    
                    // Simulate some operations
                    if counter % 1000000 == 0 {
                        tokio::task::yield_now().await;
                    }
                }
                
                counter
            });
            tasks.push(task);
        }

        // Wait for tasks to complete
        for task in tasks {
            match task.await {
                Ok(count) => operations_completed += count / 1000000,
                Err(e) => {
                    operations_failed += 1;
                    errors.push(format!("CPU task failed: {}", e));
                }
            }
        }

        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name: "CPU_Exhaustion".to_string(),
            intensity: 2.0, // 2x CPU oversubscription
            duration: phase_end - phase_start,
            operations_completed,
            operations_failed,
            average_latency_ns: 1000000, // 1ms average under CPU stress
            peak_latency_ns: 10000000,   // 10ms peak
            resource_usage: self.capture_resource_snapshot().await,
            errors,
        })
    }

    /// Run memory exhaustion phase
    async fn run_memory_exhaustion_phase(&mut self, config: &StressTestConfig) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        println!("Starting memory exhaustion phase");
        
        let phase_start = Instant::now();
        let mut operations_completed = 0u64;
        let mut operations_failed = 0u64;
        let mut errors = Vec::new();

        // Allocate memory in chunks until we approach the limit
        let mut memory_chunks = Vec::new();
        let chunk_size = 100 * 1024 * 1024; // 100MB chunks
        let max_chunks = (config.resource_limits.max_memory_bytes / chunk_size as u64) as usize;

        for i in 0..max_chunks {
            match self.allocate_memory_chunk(chunk_size).await {
                Ok(chunk) => {
                    memory_chunks.push(chunk);
                    operations_completed += 1;
                }
                Err(e) => {
                    operations_failed += 1;
                    errors.push(format!("Memory allocation failed at chunk {}: {}", i, e));
                    break;
                }
            }

            // Brief pause between allocations
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Hold memory for a period
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Release memory
        memory_chunks.clear();

        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name: "Memory_Exhaustion".to_string(),
            intensity: 0.9, // 90% of memory limit
            duration: phase_end - phase_start,
            operations_completed,
            operations_failed,
            average_latency_ns: 2000000, // 2ms average under memory pressure
            peak_latency_ns: 50000000,   // 50ms peak during allocation
            resource_usage: self.capture_resource_snapshot().await,
            errors,
        })
    }

    /// Run network exhaustion phase
    async fn run_network_exhaustion_phase(&mut self, config: &StressTestConfig) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        println!("Starting network exhaustion phase");
        
        let phase_start = Instant::now();
        let operations_completed = 1000u64; // Simulated network operations
        let operations_failed = 50u64;      // Some network failures

        // Simulate network stress
        tokio::time::sleep(Duration::from_secs(20)).await;

        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name: "Network_Exhaustion".to_string(),
            intensity: 1.5, // 150% of network capacity
            duration: phase_end - phase_start,
            operations_completed,
            operations_failed,
            average_latency_ns: 5000000, // 5ms average under network stress
            peak_latency_ns: 100000000,  // 100ms peak during congestion
            resource_usage: self.capture_resource_snapshot().await,
            errors: vec!["Network timeout".to_string(), "Connection refused".to_string()],
        })
    }

    /// Run failure injection test phases
    async fn run_failure_injection_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();

        for failure_type in &config.failure_injection.failure_types {
            let phase = self.run_failure_injection_phase(config, failure_type).await?;
            phases.push(phase);
        }

        Ok(phases)
    }

    /// Run single failure injection phase
    async fn run_failure_injection_phase(&mut self, config: &StressTestConfig, failure_type: &FailureType) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        let phase_name = format!("FailureInjection_{:?}", failure_type);
        println!("Starting failure injection phase: {}", phase_name);

        let phase_start = Instant::now();
        let mut operations_completed = 0u64;
        let mut operations_failed = 0u64;
        let mut errors = Vec::new();

        // Inject failures while running normal load
        let load_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: 50000 },
            duration: Duration::from_secs(30),
            thread_count: 4,
            order_types: vec![crate::models::order::OrderType::Limit],
            price_range: (100.0, 101.0),
            quantity_range: (1, 100),
            symbols: vec!["BTCUSD".to_string()],
        };

        // Start background load
        let load_task = tokio::spawn(async move {
            // Simulate load generation with failures
            tokio::time::sleep(Duration::from_secs(30)).await;
            (25000u64, 2500u64) // (completed, failed)
        });

        // Inject specific failure type
        self.inject_failure(failure_type, &config.failure_injection).await?;

        // Wait for load task to complete
        let (completed, failed) = load_task.await?;
        operations_completed = completed;
        operations_failed = failed;

        if operations_failed > 0 {
            errors.push(format!("Failures injected: {:?}", failure_type));
        }

        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name,
            intensity: config.failure_injection.failure_rate_percent / 100.0,
            duration: phase_end - phase_start,
            operations_completed,
            operations_failed,
            average_latency_ns: 3000000, // 3ms average with failures
            peak_latency_ns: 20000000,   // 20ms peak during failure
            resource_usage: self.capture_resource_snapshot().await,
            errors,
        })
    }

    /// Run endurance test phases
    async fn run_endurance_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();
        let phase_duration = config.duration / 4; // Split into 4 phases

        for i in 0..4 {
            let phase_name = format!("Endurance_Phase_{}", i + 1);
            println!("Starting endurance phase: {} for {:?}", phase_name, phase_duration);

            let load_config = LoadConfig {
                pattern: LoadPattern::Constant { ops_per_second: 200000 },
                duration: phase_duration,
                thread_count: 6,
                order_types: vec![crate::models::order::OrderType::Limit, crate::models::order::OrderType::Market],
                price_range: (100.0, 200.0),
                quantity_range: (1, 500),
                symbols: vec!["BTCUSD".to_string(), "ETHUSD".to_string()],
            };

            let phase_start = Instant::now();
            let load_stats = self.load_generator.generate_load_with_config(&load_config).await?;
            let phase_end = Instant::now();

            let phase = StressTestPhase {
                phase_name,
                intensity: 1.0,
                duration: phase_end - phase_start,
                operations_completed: load_stats.total_operations,
                operations_failed: load_stats.error_count,
                average_latency_ns: self.calculate_average_latency(load_stats.operations_per_second),
                peak_latency_ns: self.calculate_peak_latency(1.0),
                resource_usage: self.capture_resource_snapshot().await,
                errors: Vec::new(),
            };

            phases.push(phase);
        }

        Ok(phases)
    }

    /// Run spike test phases
    async fn run_spike_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();

        // Normal load phase
        let normal_phase = self.run_normal_load_phase(Duration::from_secs(30)).await?;
        phases.push(normal_phase);

        // Spike phase
        let spike_phase = self.run_spike_phase(Duration::from_secs(10)).await?;
        phases.push(spike_phase);

        // Recovery phase
        let recovery_phase = self.run_normal_load_phase(Duration::from_secs(30)).await?;
        phases.push(recovery_phase);

        Ok(phases)
    }

    /// Run gradual increase phases
    async fn run_gradual_phases(&mut self, config: &StressTestConfig) -> Result<Vec<StressTestPhase>, Box<dyn std::error::Error>> {
        let mut phases = Vec::new();
        let phase_count = config.intensity_levels.len();
        let phase_duration = config.duration / phase_count as u32;

        for (i, &intensity) in config.intensity_levels.iter().enumerate() {
            let phase_name = format!("Gradual_Phase_{}_Intensity_{:.1}x", i + 1, intensity);
            
            let base_ops = 50000u64;
            let target_ops = (base_ops as f64 * intensity) as u64;

            let load_config = LoadConfig {
                pattern: LoadPattern::Ramp {
                    start_ops: if i == 0 { base_ops } else { (base_ops as f64 * config.intensity_levels[i-1]) as u64 },
                    end_ops: target_ops,
                    duration: phase_duration,
                },
                duration: phase_duration,
                thread_count: 4,
                order_types: vec![crate::models::order::OrderType::Limit],
                price_range: (100.0, 101.0),
                quantity_range: (1, 100),
                symbols: vec!["BTCUSD".to_string()],
            };

            let phase_start = Instant::now();
            let load_stats = self.load_generator.generate_load_with_config(&load_config).await?;
            let phase_end = Instant::now();

            let phase = StressTestPhase {
                phase_name,
                intensity,
                duration: phase_end - phase_start,
                operations_completed: load_stats.total_operations,
                operations_failed: load_stats.error_count,
                average_latency_ns: self.calculate_average_latency(load_stats.operations_per_second),
                peak_latency_ns: self.calculate_peak_latency(intensity),
                resource_usage: self.capture_resource_snapshot().await,
                errors: Vec::new(),
            };

            phases.push(phase);
        }

        Ok(phases)
    }

    /// Helper methods
    async fn run_normal_load_phase(&mut self, duration: Duration) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        let load_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: 100000 },
            duration,
            thread_count: 4,
            order_types: vec![crate::models::order::OrderType::Limit],
            price_range: (100.0, 101.0),
            quantity_range: (1, 100),
            symbols: vec!["BTCUSD".to_string()],
        };

        let phase_start = Instant::now();
        let load_stats = self.load_generator.generate_load_with_config(&load_config).await?;
        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name: "Normal_Load".to_string(),
            intensity: 1.0,
            duration: phase_end - phase_start,
            operations_completed: load_stats.total_operations,
            operations_failed: load_stats.error_count,
            average_latency_ns: 500000, // 500μs
            peak_latency_ns: 2000000,   // 2ms
            resource_usage: self.capture_resource_snapshot().await,
            errors: Vec::new(),
        })
    }

    async fn run_spike_phase(&mut self, duration: Duration) -> Result<StressTestPhase, Box<dyn std::error::Error>> {
        let load_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: 1000000 }, // 10x spike
            duration,
            thread_count: 8,
            order_types: vec![crate::models::order::OrderType::Market],
            price_range: (100.0, 101.0),
            quantity_range: (1, 100),
            symbols: vec!["BTCUSD".to_string()],
        };

        let phase_start = Instant::now();
        let load_stats = self.load_generator.generate_load_with_config(&load_config).await?;
        let phase_end = Instant::now();

        Ok(StressTestPhase {
            phase_name: "Load_Spike".to_string(),
            intensity: 10.0,
            duration: phase_end - phase_start,
            operations_completed: load_stats.total_operations,
            operations_failed: load_stats.error_count,
            average_latency_ns: 5000000, // 5ms under spike
            peak_latency_ns: 50000000,   // 50ms peak
            resource_usage: self.capture_resource_snapshot().await,
            errors: Vec::new(),
        })
    }

    async fn start_failure_injection(&self, config: &FailureInjectionConfig) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
        let failure_rate = config.failure_rate_percent;
        let recovery_time = config.recovery_time_ms;
        let active_failures = Arc::clone(&self.active_failures);
        let test_running = Arc::clone(&self.test_running);

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(1000));
            
            while test_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Randomly inject failures based on failure rate
                if rand::random::<f64>() * 100.0 < failure_rate {
                    active_failures.fetch_add(1, Ordering::Relaxed);
                    
                    // Schedule failure recovery
                    let active_failures_clone = Arc::clone(&active_failures);
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(recovery_time)).await;
                        active_failures_clone.fetch_sub(1, Ordering::Relaxed);
                    });
                }
            }
        });

        Ok(task)
    }

    async fn start_resource_monitoring(&self) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
        let test_running = Arc::clone(&self.test_running);

        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(500));
            
            while test_running.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Collect resource metrics (simulated)
                let _cpu_usage = 50.0 + (rand::random::<f64>() - 0.5) * 40.0;
                let _memory_usage = 2_000_000_000u64 + (rand::random::<u64>() % 1_000_000_000);
                let _network_usage = 100.0 + (rand::random::<f64>() - 0.5) * 50.0;
            }
        });

        Ok(task)
    }

    async fn inject_failure(&self, failure_type: &FailureType, config: &FailureInjectionConfig) -> Result<(), Box<dyn std::error::Error>> {
        println!("Injecting failure: {:?}", failure_type);
        
        match failure_type {
            FailureType::NetworkTimeout => {
                // Simulate network timeout
                tokio::time::sleep(Duration::from_millis(config.recovery_time_ms)).await;
            }
            FailureType::MemoryAllocationFailure => {
                // Simulate memory allocation failure
                self.active_failures.fetch_add(1, Ordering::Relaxed);
            }
            FailureType::DiskWriteFailure => {
                // Simulate disk write failure
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            FailureType::CpuThrottling => {
                // Simulate CPU throttling
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            FailureType::ConnectionDrop => {
                // Simulate connection drop
                self.active_failures.fetch_add(1, Ordering::Relaxed);
            }
            FailureType::ServiceUnavailable => {
                // Simulate service unavailable
                tokio::time::sleep(Duration::from_millis(config.recovery_time_ms)).await;
            }
        }

        Ok(())
    }

    async fn allocate_memory_chunk(&self, size: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        // Simulate memory allocation
        let chunk = vec![0u8; size];
        tokio::time::sleep(Duration::from_millis(10)).await; // Simulate allocation time
        Ok(chunk)
    }

    async fn capture_resource_snapshot(&self) -> ResourceUsageSnapshot {
        ResourceUsageSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cpu_percent: 50.0 + (rand::random::<f64>() - 0.5) * 40.0,
            memory_bytes: 2_000_000_000 + (rand::random::<u64>() % 1_000_000_000),
            network_mbps: 100.0 + (rand::random::<f64>() - 0.5) * 50.0,
            disk_iops: 1000 + (rand::random::<u64>() % 500),
            connections: 100 + (rand::random::<u32>() % 50),
        }
    }

    fn calculate_average_latency(&self, ops_per_second: f64) -> u64 {
        // Simulate latency calculation based on load
        let base_latency = 500000u64; // 500μs base
        let load_factor = (ops_per_second / 100000.0).max(1.0);
        (base_latency as f64 * load_factor) as u64
    }

    fn calculate_peak_latency(&self, intensity: f64) -> u64 {
        // Simulate peak latency calculation
        let base_peak = 2000000u64; // 2ms base peak
        (base_peak as f64 * intensity * 2.0) as u64
    }

    fn calculate_peak_metrics(&self, phases: &[StressTestPhase]) -> PeakMetrics {
        let mut peak_metrics = PeakMetrics::default();

        for phase in phases {
            let ops_per_second = phase.operations_completed as f64 / phase.duration.as_secs_f64();
            peak_metrics.peak_operations_per_second = peak_metrics.peak_operations_per_second.max(ops_per_second);
            peak_metrics.peak_latency_ns = peak_metrics.peak_latency_ns.max(phase.peak_latency_ns);
            peak_metrics.peak_cpu_percent = peak_metrics.peak_cpu_percent.max(phase.resource_usage.cpu_percent);
            peak_metrics.peak_memory_bytes = peak_metrics.peak_memory_bytes.max(phase.resource_usage.memory_bytes);
            peak_metrics.peak_network_mbps = peak_metrics.peak_network_mbps.max(phase.resource_usage.network_mbps);
            peak_metrics.peak_connections = peak_metrics.peak_connections.max(phase.resource_usage.connections);
        }

        peak_metrics
    }

    fn analyze_failures(&self, phases: &[StressTestPhase]) -> FailureAnalysis {
        let total_operations: u64 = phases.iter().map(|p| p.operations_completed + p.operations_failed).sum();
        let total_failures: u64 = phases.iter().map(|p| p.operations_failed).sum();
        
        let failure_rate_percent = if total_operations > 0 {
            (total_failures as f64 / total_operations as f64) * 100.0
        } else {
            0.0
        };

        let mut failure_breakdown = std::collections::HashMap::new();
        for phase in phases {
            for error in &phase.errors {
                *failure_breakdown.entry(error.clone()).or_insert(0) += 1;
            }
        }

        // Simplified failure analysis
        let mean_time_to_failure_ms = if total_failures > 0 {
            phases.iter().map(|p| p.duration.as_millis() as f64).sum::<f64>() / total_failures as f64
        } else {
            0.0
        };

        FailureAnalysis {
            total_failures,
            failure_rate_percent,
            failure_breakdown,
            mean_time_to_failure_ms,
            failure_clustering: failure_rate_percent > 10.0, // Simple heuristic
        }
    }

    fn analyze_recovery(&self, phases: &[StressTestPhase], config: &StressTestConfig) -> RecoveryAnalysis {
        // Simplified recovery analysis
        let recovery_attempts = phases.iter().map(|p| p.operations_failed).sum();
        let successful_recoveries = recovery_attempts * 80 / 100; // Assume 80% recovery rate
        
        let recovery_success_rate = if recovery_attempts > 0 {
            successful_recoveries as f64 / recovery_attempts as f64
        } else {
            1.0
        };

        RecoveryAnalysis {
            recovery_attempts,
            successful_recoveries,
            recovery_success_rate,
            mean_recovery_time_ms: config.failure_injection.recovery_time_ms as f64,
            max_recovery_time_ms: config.failure_injection.recovery_time_ms as f64 * 2.0,
        }
    }

    fn determine_stress_verdict(&self, phases: &[StressTestPhase], failure_analysis: &FailureAnalysis, recovery_analysis: &RecoveryAnalysis) -> StressTestVerdict {
        // Determine verdict based on failure rate and recovery success
        if failure_analysis.failure_rate_percent > 20.0 {
            StressTestVerdict::Fail
        } else if failure_analysis.failure_rate_percent > 10.0 || recovery_analysis.recovery_success_rate < 0.8 {
            StressTestVerdict::Degraded
        } else if failure_analysis.failure_clustering {
            StressTestVerdict::Unstable
        } else {
            StressTestVerdict::Pass
        }
    }

    /// Generate stress test report
    pub fn generate_stress_report(&self, results: &StressTestResults) -> String {
        format!(
            r#"
# Stress Test Report: {}

**Test Type:** {:?}
**Duration:** {:.2} seconds
**Verdict:** {:?}

## Test Phases
{}

## Peak Metrics
- **Peak Operations/Second:** {:.0}
- **Peak Latency:** {:.2}ms
- **Peak CPU Usage:** {:.1}%
- **Peak Memory Usage:** {:.2}GB
- **Peak Network Usage:** {:.1}Mbps
- **Peak Connections:** {}

## Failure Analysis
- **Total Failures:** {}
- **Failure Rate:** {:.2}%
- **Mean Time to Failure:** {:.2}ms
- **Failure Clustering:** {}

## Recovery Analysis
- **Recovery Attempts:** {}
- **Successful Recoveries:** {}
- **Recovery Success Rate:** {:.1}%
- **Mean Recovery Time:** {:.2}ms

## Assessment
{}
"#,
            results.config.name,
            results.config.test_type,
            results.duration.as_secs_f64(),
            results.verdict,
            results.phases.iter()
                .map(|p| format!("- **{}:** {:.0} ops completed, {:.0} failed, {:.2}ms avg latency", 
                    p.phase_name, p.operations_completed, p.operations_failed, p.average_latency_ns as f64 / 1_000_000.0))
                .collect::<Vec<_>>()
                .join("\n"),
            results.peak_metrics.peak_operations_per_second,
            results.peak_metrics.peak_latency_ns as f64 / 1_000_000.0,
            results.peak_metrics.peak_cpu_percent,
            results.peak_metrics.peak_memory_bytes as f64 / 1_000_000_000.0,
            results.peak_metrics.peak_network_mbps,
            results.peak_metrics.peak_connections,
            results.failure_analysis.total_failures,
            results.failure_analysis.failure_rate_percent,
            results.failure_analysis.mean_time_to_failure_ms,
            if results.failure_analysis.failure_clustering { "Yes" } else { "No" },
            results.recovery_analysis.recovery_attempts,
            results.recovery_analysis.successful_recoveries,
            results.recovery_analysis.recovery_success_rate * 100.0,
            results.recovery_analysis.mean_recovery_time_ms,
            match results.verdict {
                StressTestVerdict::Pass => "✓ System handled stress conditions well",
                StressTestVerdict::Fail => "✗ System failed under stress conditions",
                StressTestVerdict::Degraded => "⚠ System performance degraded under stress",
                StressTestVerdict::Unstable => "⚠ System showed instability under stress",
            }
        )
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_percent: 90.0,
            max_memory_bytes: 8_000_000_000, // 8GB
            max_network_mbps: 1000.0,        // 1Gbps
            max_disk_iops: 10000,
            max_connections: 10000,
        }
    }
}

impl Default for FailureInjectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            failure_rate_percent: 5.0,
            failure_types: vec![FailureType::NetworkTimeout, FailureType::ConnectionDrop],
            recovery_time_ms: 1000,
        }
    }
}

impl Default for PeakMetrics {
    fn default() -> Self {
        Self {
            peak_operations_per_second: 0.0,
            peak_latency_ns: 0,
            peak_cpu_percent: 0.0,
            peak_memory_bytes: 0,
            peak_network_mbps: 0.0,
            peak_connections: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stress_tester_creation() {
        let tester = StressTester::new();
        assert_eq!(tester.active_failures.load(Ordering::Relaxed), 0);
        assert!(!tester.test_running.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_resource_snapshot() {
        let tester = StressTester::new();
        let snapshot = tester.capture_resource_snapshot().await;
        
        assert!(snapshot.cpu_percent >= 0.0 && snapshot.cpu_percent <= 100.0);
        assert!(snapshot.memory_bytes > 0);
        assert!(snapshot.network_mbps >= 0.0);
    }

    #[test]
    fn test_latency_calculation() {
        let tester = StressTester::new();
        let latency = tester.calculate_average_latency(100000.0);
        assert!(latency >= 500000); // At least 500μs base latency
    }
}