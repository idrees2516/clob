//! Load Generation and Simulation
//! 
//! Provides realistic load generation for performance testing

use super::TestConfig;
use crate::models::order::{Order, OrderSide, OrderType};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Load generation pattern
#[derive(Debug, Clone)]
pub enum LoadPattern {
    Constant { ops_per_second: u64 },
    Burst { burst_size: u64, interval: Duration },
    Ramp { start_ops: u64, end_ops: u64, duration: Duration },
    Sine { base_ops: u64, amplitude: u64, period: Duration },
    Random { min_ops: u64, max_ops: u64 },
}

/// Load generation configuration
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub pattern: LoadPattern,
    pub duration: Duration,
    pub thread_count: usize,
    pub order_types: Vec<OrderType>,
    pub price_range: (f64, f64),
    pub quantity_range: (u64, u64),
    pub symbols: Vec<String>,
}

/// Generated load statistics
#[derive(Debug, Clone)]
pub struct LoadStats {
    pub total_operations: u64,
    pub operations_per_second: f64,
    pub duration: Duration,
    pub pattern_adherence: f64, // How well the actual load matched the target pattern
    pub error_count: u64,
}

/// Load generator for performance testing
pub struct LoadGenerator {
    operation_count: AtomicU64,
    error_count: AtomicU64,
    rng: StdRng,
}

impl LoadGenerator {
    /// Create a new load generator
    pub fn new() -> Self {
        Self {
            operation_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            rng: StdRng::from_entropy(),
        }
    }

    /// Generate load according to test configuration
    pub async fn generate_load(&mut self, config: &TestConfig) -> Result<LoadStats, Box<dyn std::error::Error>> {
        let load_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: config.target_throughput },
            duration: config.duration,
            thread_count: config.thread_count,
            order_types: vec![OrderType::Limit, OrderType::Market],
            price_range: (100.0, 200.0),
            quantity_range: (1, 1000),
            symbols: vec!["BTCUSD".to_string(), "ETHUSD".to_string(), "ADAUSD".to_string()],
        };

        self.generate_load_with_config(&load_config).await
    }

    /// Generate load with specific configuration
    pub async fn generate_load_with_config(&mut self, config: &LoadConfig) -> Result<LoadStats, Box<dyn std::error::Error>> {
        println!("Starting load generation with pattern: {:?}", config.pattern);
        
        // Reset counters
        self.operation_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);

        let start_time = Instant::now();

        // Create communication channel for coordination
        let (tx, mut rx) = mpsc::channel(1000);

        // Start load generation tasks
        let generation_tasks = self.start_load_generation_tasks(config, tx).await?;

        // Monitor and coordinate load generation
        let monitoring_task = tokio::spawn(async move {
            while let Some(_) = rx.recv().await {
                // Process coordination messages if needed
            }
        });

        // Run for specified duration
        tokio::time::sleep(config.duration).await;

        // Stop all tasks
        for task in generation_tasks {
            task.abort();
        }
        monitoring_task.abort();

        let end_time = Instant::now();
        let actual_duration = end_time - start_time;

        // Calculate statistics
        let total_operations = self.operation_count.load(Ordering::Relaxed);
        let operations_per_second = total_operations as f64 / actual_duration.as_secs_f64();
        let error_count = self.error_count.load(Ordering::Relaxed);

        // Calculate pattern adherence (simplified)
        let target_ops = self.calculate_target_operations(config);
        let pattern_adherence = if target_ops > 0 {
            (total_operations as f64 / target_ops as f64).min(1.0)
        } else {
            1.0
        };

        let stats = LoadStats {
            total_operations,
            operations_per_second,
            duration: actual_duration,
            pattern_adherence,
            error_count,
        };

        println!("Load generation completed: {} operations in {:?}", total_operations, actual_duration);
        Ok(stats)
    }

    /// Start load generation tasks based on pattern
    async fn start_load_generation_tasks(
        &mut self,
        config: &LoadConfig,
        tx: mpsc::Sender<()>,
    ) -> Result<Vec<tokio::task::JoinHandle<()>>, Box<dyn std::error::Error>> {
        let mut tasks = Vec::new();

        for thread_id in 0..config.thread_count {
            let pattern = config.pattern.clone();
            let duration = config.duration;
            let operation_count = Arc::clone(&self.operation_count);
            let error_count = Arc::clone(&self.error_count);
            let tx_clone = tx.clone();
            let order_types = config.order_types.clone();
            let price_range = config.price_range;
            let quantity_range = config.quantity_range;
            let symbols = config.symbols.clone();

            let task = tokio::spawn(async move {
                let mut rng = StdRng::from_entropy();
                let start_time = Instant::now();

                loop {
                    let elapsed = start_time.elapsed();
                    if elapsed >= duration {
                        break;
                    }

                    // Calculate target operations per second based on pattern and elapsed time
                    let target_ops_per_second = Self::calculate_current_target_ops(&pattern, elapsed, duration);
                    
                    if target_ops_per_second > 0 {
                        // Calculate delay between operations
                        let ops_per_thread = target_ops_per_second / config.thread_count as u64;
                        if ops_per_thread > 0 {
                            let delay_ns = 1_000_000_000 / ops_per_thread;
                            let delay = Duration::from_nanos(delay_ns);

                            // Generate operation
                            match Self::generate_operation(
                                &mut rng,
                                &order_types,
                                price_range,
                                quantity_range,
                                &symbols,
                            ).await {
                                Ok(_) => {
                                    operation_count.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(_) => {
                                    error_count.fetch_add(1, Ordering::Relaxed);
                                }
                            }

                            // Wait before next operation
                            tokio::time::sleep(delay).await;
                        }
                    } else {
                        // No operations needed, wait a bit
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                }

                // Notify completion
                let _ = tx_clone.send(()).await;
            });

            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Calculate current target operations per second based on pattern
    fn calculate_current_target_ops(pattern: &LoadPattern, elapsed: Duration, total_duration: Duration) -> u64 {
        match pattern {
            LoadPattern::Constant { ops_per_second } => *ops_per_second,
            LoadPattern::Burst { burst_size, interval } => {
                let cycle_time = elapsed.as_secs_f64() % interval.as_secs_f64();
                if cycle_time < 0.1 { // Burst for first 100ms of each interval
                    *burst_size * 10 // Concentrate burst in 100ms
                } else {
                    0
                }
            }
            LoadPattern::Ramp { start_ops, end_ops, duration: _ } => {
                let progress = elapsed.as_secs_f64() / total_duration.as_secs_f64();
                let progress = progress.min(1.0);
                (*start_ops as f64 + (*end_ops as f64 - *start_ops as f64) * progress) as u64
            }
            LoadPattern::Sine { base_ops, amplitude, period } => {
                let phase = 2.0 * std::f64::consts::PI * elapsed.as_secs_f64() / period.as_secs_f64();
                (*base_ops as f64 + *amplitude as f64 * phase.sin()) as u64
            }
            LoadPattern::Random { min_ops, max_ops } => {
                let mut rng = StdRng::from_entropy();
                rng.gen_range(*min_ops..=*max_ops)
            }
        }
    }

    /// Generate a single operation (order)
    async fn generate_operation(
        rng: &mut StdRng,
        order_types: &[OrderType],
        price_range: (f64, f64),
        quantity_range: (u64, u64),
        symbols: &[String],
    ) -> Result<Order, Box<dyn std::error::Error>> {
        // Simulate operation generation time
        tokio::time::sleep(Duration::from_nanos(100)).await;

        let order_type = order_types[rng.gen_range(0..order_types.len())].clone();
        let side = if rng.gen_bool(0.5) { OrderSide::Buy } else { OrderSide::Sell };
        let price = rng.gen_range(price_range.0..=price_range.1);
        let quantity = rng.gen_range(quantity_range.0..=quantity_range.1);
        let symbol = symbols[rng.gen_range(0..symbols.len())].clone();

        let order = Order {
            id: rng.gen::<u64>(),
            symbol,
            side,
            order_type,
            quantity,
            price: Some(price),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            user_id: rng.gen::<u64>(),
        };

        Ok(order)
    }

    /// Calculate total target operations for the entire test
    fn calculate_target_operations(&self, config: &LoadConfig) -> u64 {
        match &config.pattern {
            LoadPattern::Constant { ops_per_second } => {
                ops_per_second * config.duration.as_secs()
            }
            LoadPattern::Burst { burst_size, interval } => {
                let cycles = config.duration.as_secs_f64() / interval.as_secs_f64();
                (*burst_size as f64 * cycles) as u64
            }
            LoadPattern::Ramp { start_ops, end_ops, duration: _ } => {
                let average_ops = (*start_ops + *end_ops) / 2;
                average_ops * config.duration.as_secs()
            }
            LoadPattern::Sine { base_ops, amplitude: _, period: _ } => {
                base_ops * config.duration.as_secs()
            }
            LoadPattern::Random { min_ops, max_ops } => {
                let average_ops = (*min_ops + *max_ops) / 2;
                average_ops * config.duration.as_secs()
            }
        }
    }

    /// Generate realistic trading load patterns
    pub async fn generate_trading_patterns(&mut self) -> Result<Vec<LoadStats>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();

        // Market open surge
        let market_open_config = LoadConfig {
            pattern: LoadPattern::Burst { 
                burst_size: 10000, 
                interval: Duration::from_secs(1) 
            },
            duration: Duration::from_secs(30),
            thread_count: 8,
            order_types: vec![OrderType::Market, OrderType::Limit],
            price_range: (100.0, 105.0),
            quantity_range: (10, 500),
            symbols: vec!["BTCUSD".to_string()],
        };
        results.push(self.generate_load_with_config(&market_open_config).await?);

        // Steady trading
        let steady_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: 50000 },
            duration: Duration::from_secs(60),
            thread_count: 4,
            order_types: vec![OrderType::Limit],
            price_range: (99.0, 101.0),
            quantity_range: (1, 100),
            symbols: vec!["BTCUSD".to_string(), "ETHUSD".to_string()],
        };
        results.push(self.generate_load_with_config(&steady_config).await?);

        // Volatility spike
        let volatility_config = LoadConfig {
            pattern: LoadPattern::Sine { 
                base_ops: 30000, 
                amplitude: 20000, 
                period: Duration::from_secs(10) 
            },
            duration: Duration::from_secs(60),
            thread_count: 6,
            order_types: vec![OrderType::Market],
            price_range: (95.0, 110.0),
            quantity_range: (50, 1000),
            symbols: vec!["BTCUSD".to_string()],
        };
        results.push(self.generate_load_with_config(&volatility_config).await?);

        Ok(results)
    }

    /// Generate stress test load
    pub async fn generate_stress_load(&mut self, intensity: f64) -> Result<LoadStats, Box<dyn std::error::Error>> {
        let base_ops = 1_000_000u64;
        let stress_ops = (base_ops as f64 * intensity) as u64;

        let stress_config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: stress_ops },
            duration: Duration::from_secs(30),
            thread_count: num_cpus::get(),
            order_types: vec![OrderType::Market, OrderType::Limit],
            price_range: (50.0, 150.0),
            quantity_range: (1, 10000),
            symbols: vec![
                "BTCUSD".to_string(),
                "ETHUSD".to_string(),
                "ADAUSD".to_string(),
                "SOLUSD".to_string(),
            ],
        };

        self.generate_load_with_config(&stress_config).await
    }

    /// Generate load report
    pub fn generate_load_report(&self, stats: &LoadStats) -> String {
        format!(
            r#"
=== Load Generation Report ===
Total Operations: {}
Duration: {:.2} seconds
Operations per Second: {:.2}
Pattern Adherence: {:.1}%
Error Count: {}
Error Rate: {:.2}%

Load Generation Assessment:
- Target Achievement: {}
- Error Rate Acceptable (<1%): {}
- Pattern Adherence Good (>90%): {}
"#,
            stats.total_operations,
            stats.duration.as_secs_f64(),
            stats.operations_per_second,
            stats.pattern_adherence * 100.0,
            stats.error_count,
            (stats.error_count as f64 / stats.total_operations as f64) * 100.0,
            if stats.pattern_adherence > 0.8 { "✓ PASS" } else { "✗ FAIL" },
            if stats.error_count as f64 / stats.total_operations as f64 < 0.01 { "✓ PASS" } else { "✗ FAIL" },
            if stats.pattern_adherence > 0.9 { "✓ PASS" } else { "✗ FAIL" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_generator_creation() {
        let generator = LoadGenerator::new();
        assert_eq!(generator.operation_count.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_constant_load_generation() {
        let mut generator = LoadGenerator::new();
        let config = LoadConfig {
            pattern: LoadPattern::Constant { ops_per_second: 1000 },
            duration: Duration::from_secs(2),
            thread_count: 2,
            order_types: vec![OrderType::Limit],
            price_range: (100.0, 101.0),
            quantity_range: (1, 10),
            symbols: vec!["TEST".to_string()],
        };

        let stats = generator.generate_load_with_config(&config).await.unwrap();
        assert!(stats.total_operations > 0);
        assert!(stats.operations_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_burst_load_generation() {
        let mut generator = LoadGenerator::new();
        let config = LoadConfig {
            pattern: LoadPattern::Burst { 
                burst_size: 100, 
                interval: Duration::from_secs(1) 
            },
            duration: Duration::from_secs(3),
            thread_count: 1,
            order_types: vec![OrderType::Market],
            price_range: (100.0, 101.0),
            quantity_range: (1, 10),
            symbols: vec!["TEST".to_string()],
        };

        let stats = generator.generate_load_with_config(&config).await.unwrap();
        assert!(stats.total_operations > 0);
    }

    #[test]
    fn test_target_ops_calculation() {
        let pattern = LoadPattern::Constant { ops_per_second: 1000 };
        let target = LoadGenerator::calculate_current_target_ops(
            &pattern, 
            Duration::from_secs(1), 
            Duration::from_secs(10)
        );
        assert_eq!(target, 1000);
    }
}