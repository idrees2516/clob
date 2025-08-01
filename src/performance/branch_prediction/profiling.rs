use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Branch prediction profiling data
#[derive(Debug, Clone)]
pub struct BranchProfile {
    pub taken_count: u64,
    pub not_taken_count: u64,
    pub misprediction_count: u64,
    pub prediction_accuracy: f64,
}

impl BranchProfile {
    pub fn new() -> Self {
        Self {
            taken_count: 0,
            not_taken_count: 0,
            misprediction_count: 0,
            prediction_accuracy: 0.0,
        }
    }

    pub fn total_branches(&self) -> u64 {
        self.taken_count + self.not_taken_count
    }

    pub fn calculate_accuracy(&mut self) {
        let total = self.total_branches();
        if total > 0 {
            self.prediction_accuracy = 
                ((total - self.misprediction_count) as f64 / total as f64) * 100.0;
        }
    }
}

/// Branch prediction profiler
pub struct BranchProfiler {
    profiles: HashMap<String, Arc<BranchProfile>>,
    counters: HashMap<String, (AtomicU64, AtomicU64, AtomicU64)>, // taken, not_taken, mispredicted
}

impl BranchProfiler {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            counters: HashMap::new(),
        }
    }

    /// Register a branch for profiling
    pub fn register_branch(&mut self, name: &str) {
        self.counters.insert(
            name.to_string(),
            (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0))
        );
        self.profiles.insert(name.to_string(), Arc::new(BranchProfile::new()));
    }

    /// Record a branch taken
    #[inline(always)]
    pub fn record_taken(&self, name: &str) {
        if let Some((taken, _, _)) = self.counters.get(name) {
            taken.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a branch not taken
    #[inline(always)]
    pub fn record_not_taken(&self, name: &str) {
        if let Some((_, not_taken, _)) = self.counters.get(name) {
            not_taken.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a branch misprediction
    #[inline(always)]
    pub fn record_misprediction(&self, name: &str) {
        if let Some((_, _, mispredicted)) = self.counters.get(name) {
            mispredicted.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get current profile for a branch
    pub fn get_profile(&self, name: &str) -> Option<BranchProfile> {
        if let Some((taken, not_taken, mispredicted)) = self.counters.get(name) {
            let mut profile = BranchProfile {
                taken_count: taken.load(Ordering::Relaxed),
                not_taken_count: not_taken.load(Ordering::Relaxed),
                misprediction_count: mispredicted.load(Ordering::Relaxed),
                prediction_accuracy: 0.0,
            };
            profile.calculate_accuracy();
            Some(profile)
        } else {
            None
        }
    }

    /// Get all profiles
    pub fn get_all_profiles(&self) -> HashMap<String, BranchProfile> {
        let mut profiles = HashMap::new();
        for name in self.counters.keys() {
            if let Some(profile) = self.get_profile(name) {
                profiles.insert(name.clone(), profile);
            }
        }
        profiles
    }

    /// Reset all counters
    pub fn reset(&self) {
        for (taken, not_taken, mispredicted) in self.counters.values() {
            taken.store(0, Ordering::Relaxed);
            not_taken.store(0, Ordering::Relaxed);
            mispredicted.store(0, Ordering::Relaxed);
        }
    }
}

/// Macro for profiled branch execution
#[macro_export]
macro_rules! profiled_branch {
    ($profiler:expr, $name:expr, $condition:expr, $if_block:block, $else_block:block) => {{
        let condition_result = $condition;
        if condition_result {
            $profiler.record_taken($name);
            $if_block
        } else {
            $profiler.record_not_taken($name);
            $else_block
        }
    }};
}

/// Performance counter for branch prediction analysis
pub struct BranchPerformanceCounter {
    name: String,
    start_cycles: u64,
    branch_count: AtomicU64,
    total_cycles: AtomicU64,
}

impl BranchPerformanceCounter {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_cycles: 0,
            branch_count: AtomicU64::new(0),
            total_cycles: AtomicU64::new(0),
        }
    }

    /// Start timing a branch sequence
    #[inline(always)]
    pub fn start(&mut self) {
        self.start_cycles = unsafe { core::arch::x86_64::_rdtsc() };
    }

    /// End timing and record results
    #[inline(always)]
    pub fn end(&self) {
        let end_cycles = unsafe { core::arch::x86_64::_rdtsc() };
        let elapsed = end_cycles - self.start_cycles;
        self.total_cycles.fetch_add(elapsed, Ordering::Relaxed);
        self.branch_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average cycles per branch
    pub fn average_cycles(&self) -> f64 {
        let total = self.total_cycles.load(Ordering::Relaxed);
        let count = self.branch_count.load(Ordering::Relaxed);
        if count > 0 {
            total as f64 / count as f64
        } else {
            0.0
        }
    }

    /// Get total branch count
    pub fn branch_count(&self) -> u64 {
        self.branch_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_profile() {
        let mut profile = BranchProfile::new();
        profile.taken_count = 80;
        profile.not_taken_count = 20;
        profile.misprediction_count = 5;
        profile.calculate_accuracy();
        
        assert_eq!(profile.total_branches(), 100);
        assert_eq!(profile.prediction_accuracy, 95.0);
    }

    #[test]
    fn test_branch_profiler() {
        let mut profiler = BranchProfiler::new();
        profiler.register_branch("test_branch");
        
        profiler.record_taken("test_branch");
        profiler.record_taken("test_branch");
        profiler.record_not_taken("test_branch");
        profiler.record_misprediction("test_branch");
        
        let profile = profiler.get_profile("test_branch").unwrap();
        assert_eq!(profile.taken_count, 2);
        assert_eq!(profile.not_taken_count, 1);
        assert_eq!(profile.misprediction_count, 1);
    }
}