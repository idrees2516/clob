use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Branch prediction monitoring statistics
#[derive(Debug, Clone)]
pub struct BranchStats {
    pub total_branches: u64,
    pub mispredictions: u64,
    pub prediction_rate: f64,
    pub cycles_per_branch: f64,
    pub last_updated: Instant,
}

impl BranchStats {
    pub fn new() -> Self {
        Self {
            total_branches: 0,
            mispredictions: 0,
            prediction_rate: 0.0,
            cycles_per_branch: 0.0,
            last_updated: Instant::now(),
        }
    }

    pub fn update(&mut self, branches: u64, mispredictions: u64, cycles: u64) {
        self.total_branches = branches;
        self.mispredictions = mispredictions;
        self.prediction_rate = if branches > 0 {
            ((branches - mispredictions) as f64 / branches as f64) * 100.0
        } else {
            0.0
        };
        self.cycles_per_branch = if branches > 0 {
            cycles as f64 / branches as f64
        } else {
            0.0
        };
        self.last_updated = Instant::now();
    }
}

/// Real-time branch prediction monitor
pub struct BranchMonitor {
    counters: HashMap<String, (AtomicU64, AtomicU64, AtomicU64)>, // branches, mispredictions, cycles
    stats: Arc<Mutex<HashMap<String, BranchStats>>>,
    monitoring_enabled: AtomicU64, // Using as bool
    update_interval: Duration,
    last_update: Mutex<Instant>,
}

impl BranchMonitor {
    pub fn new(update_interval: Duration) -> Self {
        Self {
            counters: HashMap::new(),
            stats: Arc::new(Mutex::new(HashMap::new())),
            monitoring_enabled: AtomicU64::new(1),
            update_interval,
            last_update: Mutex::new(Instant::now()),
        }
    }

    /// Register a branch point for monitoring
    pub fn register_branch(&mut self, name: &str) {
        self.counters.insert(
            name.to_string(),
            (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0))
        );
        if let Ok(mut stats) = self.stats.lock() {
            stats.insert(name.to_string(), BranchStats::new());
        }
    }

    /// Enable monitoring
    pub fn enable(&self) {
        self.monitoring_enabled.store(1, Ordering::Relaxed);
    }

    /// Disable monitoring
    pub fn disable(&self) {
        self.monitoring_enabled.store(0, Ordering::Relaxed);
    }

    /// Check if monitoring is enabled
    #[inline(always)]
    pub fn is_enabled(&self) -> bool {
        self.monitoring_enabled.load(Ordering::Relaxed) != 0
    }

    /// Record a branch execution
    #[inline(always)]
    pub fn record_branch(&self, name: &str, cycles: u64) {
        if !self.is_enabled() {
            return;
        }

        if let Some((branches, _, total_cycles)) = self.counters.get(name) {
            branches.fetch_add(1, Ordering::Relaxed);
            total_cycles.fetch_add(cycles, Ordering::Relaxed);
        }
    }

    /// Record a branch misprediction
    #[inline(always)]
    pub fn record_misprediction(&self, name: &str) {
        if !self.is_enabled() {
            return;
        }

        if let Some((_, mispredictions, _)) = self.counters.get(name) {
            mispredictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Update statistics if interval has passed
    pub fn maybe_update_stats(&self) {
        if let Ok(mut last_update) = self.last_update.try_lock() {
            if last_update.elapsed() >= self.update_interval {
                self.update_all_stats();
                *last_update = Instant::now();
            }
        }
    }

    /// Force update of all statistics
    pub fn update_all_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            for (name, (branches, mispredictions, cycles)) in &self.counters {
                let branch_count = branches.load(Ordering::Relaxed);
                let mispredict_count = mispredictions.load(Ordering::Relaxed);
                let cycle_count = cycles.load(Ordering::Relaxed);

                if let Some(stat) = stats.get_mut(name) {
                    stat.update(branch_count, mispredict_count, cycle_count);
                }
            }
        }
    }

    /// Get current statistics for a branch
    pub fn get_stats(&self, name: &str) -> Option<BranchStats> {
        if let Ok(stats) = self.stats.lock() {
            stats.get(name).cloned()
        } else {
            None
        }
    }

    /// Get all current statistics
    pub fn get_all_stats(&self) -> HashMap<String, BranchStats> {
        if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            HashMap::new()
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        for (branches, mispredictions, cycles) in self.counters.values() {
            branches.store(0, Ordering::Relaxed);
            mispredictions.store(0, Ordering::Relaxed);
            cycles.store(0, Ordering::Relaxed);
        }

        if let Ok(mut stats) = self.stats.lock() {
            for stat in stats.values_mut() {
                *stat = BranchStats::new();
            }
        }
    }

    /// Get branches with poor prediction rates
    pub fn get_poorly_predicted_branches(&self, threshold: f64) -> Vec<(String, BranchStats)> {
        let mut poor_branches = Vec::new();
        
        if let Ok(stats) = self.stats.lock() {
            for (name, stat) in stats.iter() {
                if stat.prediction_rate < threshold && stat.total_branches > 1000 {
                    poor_branches.push((name.clone(), stat.clone()));
                }
            }
        }

        poor_branches.sort_by(|a, b| a.1.prediction_rate.partial_cmp(&b.1.prediction_rate).unwrap());
        poor_branches
    }

    /// Get most expensive branches by cycles
    pub fn get_expensive_branches(&self, limit: usize) -> Vec<(String, BranchStats)> {
        let mut expensive_branches = Vec::new();
        
        if let Ok(stats) = self.stats.lock() {
            for (name, stat) in stats.iter() {
                if stat.total_branches > 0 {
                    expensive_branches.push((name.clone(), stat.clone()));
                }
            }
        }

        expensive_branches.sort_by(|a, b| b.1.cycles_per_branch.partial_cmp(&a.1.cycles_per_branch).unwrap());
        expensive_branches.truncate(limit);
        expensive_branches
    }
}

/// Macro for monitored branch execution
#[macro_export]
macro_rules! monitored_branch {
    ($monitor:expr, $name:expr, $condition:expr, $if_block:block, $else_block:block) => {{
        let start_cycles = unsafe { core::arch::x86_64::_rdtsc() };
        let condition_result = $condition;
        let result = if condition_result {
            $if_block
        } else {
            $else_block
        };
        let end_cycles = unsafe { core::arch::x86_64::_rdtsc() };
        $monitor.record_branch($name, end_cycles - start_cycles);
        result
    }};
}

/// Branch prediction performance analyzer
pub struct BranchPerformanceAnalyzer {
    monitor: Arc<BranchMonitor>,
    analysis_history: Mutex<Vec<HashMap<String, BranchStats>>>,
    max_history_size: usize,
}

impl BranchPerformanceAnalyzer {
    pub fn new(monitor: Arc<BranchMonitor>, max_history_size: usize) -> Self {
        Self {
            monitor,
            analysis_history: Mutex::new(Vec::new()),
            max_history_size,
        }
    }

    /// Capture current state for analysis
    pub fn capture_snapshot(&self) {
        let current_stats = self.monitor.get_all_stats();
        
        if let Ok(mut history) = self.analysis_history.lock() {
            history.push(current_stats);
            if history.len() > self.max_history_size {
                history.remove(0);
            }
        }
    }

    /// Analyze prediction trends over time
    pub fn analyze_trends(&self) -> HashMap<String, f64> {
        let mut trends = HashMap::new();
        
        if let Ok(history) = self.analysis_history.lock() {
            if history.len() < 2 {
                return trends;
            }

            for branch_name in history[0].keys() {
                let mut rates = Vec::new();
                for snapshot in history.iter() {
                    if let Some(stats) = snapshot.get(branch_name) {
                        rates.push(stats.prediction_rate);
                    }
                }

                if rates.len() >= 2 {
                    let trend = self.calculate_trend(&rates);
                    trends.insert(branch_name.clone(), trend);
                }
            }
        }

        trends
    }

    /// Calculate trend slope for prediction rates
    fn calculate_trend(&self, rates: &[f64]) -> f64 {
        if rates.len() < 2 {
            return 0.0;
        }

        let n = rates.len() as f64;
        let sum_x: f64 = (0..rates.len()).map(|i| i as f64).sum();
        let sum_y: f64 = rates.iter().sum();
        let sum_xy: f64 = rates.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let sum_x2: f64 = (0..rates.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        slope
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let stats = self.monitor.get_all_stats();
        let trends = self.analyze_trends();
        let poor_branches = self.monitor.get_poorly_predicted_branches(85.0);
        let expensive_branches = self.monitor.get_expensive_branches(10);

        let mut report = String::new();
        report.push_str("Branch Prediction Performance Report\n");
        report.push_str("=====================================\n\n");

        report.push_str("Overall Statistics:\n");
        for (name, stat) in &stats {
            report.push_str(&format!(
                "{}: {:.2}% prediction rate, {:.2} cycles/branch, {} total branches\n",
                name, stat.prediction_rate, stat.cycles_per_branch, stat.total_branches
            ));
        }

        report.push_str("\nPoorly Predicted Branches (< 85%):\n");
        for (name, stat) in &poor_branches {
            report.push_str(&format!(
                "{}: {:.2}% prediction rate\n",
                name, stat.prediction_rate
            ));
        }

        report.push_str("\nMost Expensive Branches:\n");
        for (name, stat) in &expensive_branches {
            report.push_str(&format!(
                "{}: {:.2} cycles/branch\n",
                name, stat.cycles_per_branch
            ));
        }

        report.push_str("\nPrediction Rate Trends:\n");
        for (name, trend) in &trends {
            let trend_desc = if *trend > 0.1 {
                "improving"
            } else if *trend < -0.1 {
                "degrading"
            } else {
                "stable"
            };
            report.push_str(&format!("{}: {} ({:.3})\n", name, trend_desc, trend));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_branch_stats() {
        let mut stats = BranchStats::new();
        stats.update(1000, 50, 5000);
        
        assert_eq!(stats.total_branches, 1000);
        assert_eq!(stats.mispredictions, 50);
        assert_eq!(stats.prediction_rate, 95.0);
        assert_eq!(stats.cycles_per_branch, 5.0);
    }

    #[test]
    fn test_branch_monitor() {
        let mut monitor = BranchMonitor::new(Duration::from_millis(100));
        monitor.register_branch("test_branch");
        
        monitor.record_branch("test_branch", 100);
        monitor.record_branch("test_branch", 200);
        monitor.record_misprediction("test_branch");
        
        monitor.update_all_stats();
        
        let stats = monitor.get_stats("test_branch").unwrap();
        assert_eq!(stats.total_branches, 2);
        assert_eq!(stats.mispredictions, 1);
        assert_eq!(stats.prediction_rate, 50.0);
    }

    #[test]
    fn test_performance_analyzer() {
        let monitor = Arc::new(BranchMonitor::new(Duration::from_millis(10)));
        let analyzer = BranchPerformanceAnalyzer::new(monitor.clone(), 10);
        
        // This test would need more complex setup to properly test trend analysis
        // For now, just verify the analyzer can be created and basic methods work
        analyzer.capture_snapshot();
        let trends = analyzer.analyze_trends();
        assert!(trends.is_empty()); // No trends with single snapshot
        
        let report = analyzer.generate_report();
        assert!(report.contains("Branch Prediction Performance Report"));
    }
}