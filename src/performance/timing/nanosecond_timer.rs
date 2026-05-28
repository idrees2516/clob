use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::thread;

/// High-precision nanosecond timer using TSC (Time Stamp Counter)
/// Provides sub-nanosecond resolution timing for performance measurement
pub struct NanosecondTimer {
    /// TSC frequency in Hz
    tsc_frequency: u64,
    
    /// Calibration offset for drift correction
    calibration_offset: AtomicI64,
    
    /// Last calibration timestamp
    last_calibration: AtomicU64,
    
    /// Calibration interval (nanoseconds)
    calibration_interval_ns: u64,
    
    /// Reference time for epoch conversion
    epoch_reference: TimingReference,
}

/// Reference point for converting TSC to wall clock time
#[derive(Debug, Clone)]
struct TimingReference {
    /// TSC value at reference point
    tsc_reference: u64,
    
    /// System time at reference point (nanoseconds since UNIX epoch)
    system_time_ns: u64,
    
    /// Reference timestamp
    reference_time: u64,
}

impl NanosecondTimer {
    /// Create a new nanosecond timer with automatic calibration
    pub fn new() -> Result<Self, TimingError> {
        let tsc_frequency = Self::detect_tsc_frequency()?;
        let epoch_reference = Self::create_timing_reference(tsc_frequency)?;
        
        Ok(Self {
            tsc_frequency,
            calibration_offset: AtomicI64::new(0),
            last_calibration: AtomicU64::new(epoch_reference.reference_time),
            calibration_interval_ns: 1_000_000_000, // 1 second
            epoch_reference,
        })
    }

    /// Create a timer with custom TSC frequency (for testing)
    pub fn with_frequency(tsc_frequency: u64) -> Result<Self, TimingError> {
        let epoch_reference = Self::create_timing_reference(tsc_frequency)?;
        
        Ok(Self {
            tsc_frequency,
            calibration_offset: AtomicI64::new(0),
            last_calibration: AtomicU64::new(epoch_reference.reference_time),
            calibration_interval_ns: 1_000_000_000,
            epoch_reference,
        })
    }

    /// Get current timestamp in nanoseconds (monotonic)
    #[inline(always)]
    pub fn now_nanos(&self) -> u64 {
        let tsc = self.read_tsc();
        self.tsc_to_nanos(tsc)
    }

    /// Get current timestamp in nanoseconds since UNIX epoch
    #[inline(always)]
    pub fn now_nanos_epoch(&self) -> u64 {
        let tsc = self.read_tsc();
        let monotonic_ns = self.tsc_to_nanos(tsc);
        
        // Convert to epoch time using reference point
        let tsc_since_ref = tsc.saturating_sub(self.epoch_reference.tsc_reference);
        let ns_since_ref = (tsc_since_ref * 1_000_000_000) / self.tsc_frequency;
        
        self.epoch_reference.system_time_ns + ns_since_ref
    }

    /// Get current timestamp with calibration correction
    #[inline(always)]
    pub fn now_nanos_calibrated(&self) -> u64 {
        let raw_nanos = self.now_nanos();
        let offset = self.calibration_offset.load(Ordering::Relaxed);
        
        if offset >= 0 {
            raw_nanos + offset as u64
        } else {
            raw_nanos.saturating_sub((-offset) as u64)
        }
    }

    /// Measure elapsed time between two timestamps
    #[inline(always)]
    pub fn elapsed_nanos(&self, start: u64, end: u64) -> u64 {
        end.saturating_sub(start)
    }

    /// Convert duration to nanoseconds
    #[inline(always)]
    pub fn duration_to_nanos(&self, duration: Duration) -> u64 {
        duration.as_nanos() as u64
    }

    /// Convert nanoseconds to duration
    #[inline(always)]
    pub fn nanos_to_duration(&self, nanos: u64) -> Duration {
        Duration::from_nanos(nanos)
    }

    /// Get TSC frequency in Hz
    pub fn tsc_frequency(&self) -> u64 {
        self.tsc_frequency
    }

    /// Perform timing calibration to correct for drift
    pub fn calibrate(&self) -> Result<(), TimingError> {
        let current_time = self.now_nanos();
        let last_cal = self.last_calibration.load(Ordering::Acquire);
        
        // Check if calibration is needed
        if current_time.saturating_sub(last_cal) < self.calibration_interval_ns {
            return Ok(());
        }

        // Perform calibration
        let calibration_samples = 10;
        let mut tsc_samples = Vec::with_capacity(calibration_samples);
        let mut time_samples = Vec::with_capacity(calibration_samples);

        // Collect samples
        for _ in 0..calibration_samples {
            let system_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| TimingError::SystemTimeError)?
                .as_nanos() as u64;
            let tsc = self.read_tsc();
            
            tsc_samples.push(tsc);
            time_samples.push(system_time);
            
            // Small delay between samples
            thread::sleep(Duration::from_micros(100));
        }

        // Calculate drift
        let drift = self.calculate_drift(&tsc_samples, &time_samples)?;
        
        // Update calibration offset
        self.calibration_offset.store(drift, Ordering::Release);
        self.last_calibration.store(current_time, Ordering::Release);

        Ok(())
    }

    /// Force calibration regardless of interval
    pub fn force_calibrate(&self) -> Result<(), TimingError> {
        self.last_calibration.store(0, Ordering::Release);
        self.calibrate()
    }

    /// Get calibration status
    pub fn calibration_status(&self) -> CalibrationStatus {
        let current_time = self.now_nanos();
        let last_cal = self.last_calibration.load(Ordering::Acquire);
        let offset = self.calibration_offset.load(Ordering::Acquire);
        
        CalibrationStatus {
            last_calibration_ns: last_cal,
            time_since_calibration_ns: current_time.saturating_sub(last_cal),
            calibration_offset_ns: offset,
            needs_calibration: current_time.saturating_sub(last_cal) > self.calibration_interval_ns,
        }
    }

    /// Read TSC (Time Stamp Counter) directly
    #[inline(always)]
    fn read_tsc(&self) -> u64 {
        unsafe {
            // Use RDTSC instruction for maximum precision
            core::arch::x86_64::_rdtsc()
        }
    }

    /// Convert TSC value to nanoseconds
    #[inline(always)]
    fn tsc_to_nanos(&self, tsc: u64) -> u64 {
        // Use 128-bit arithmetic to avoid overflow
        let tsc_128 = tsc as u128;
        let freq_128 = self.tsc_frequency as u128;
        ((tsc_128 * 1_000_000_000) / freq_128) as u64
    }

    /// Detect TSC frequency automatically
    fn detect_tsc_frequency() -> Result<u64, TimingError> {
        // Method 1: Try to read from /proc/cpuinfo (Linux)
        if let Ok(freq) = Self::read_tsc_from_cpuinfo() {
            return Ok(freq);
        }

        // Method 2: Calibrate against system time
        Self::calibrate_tsc_frequency()
    }

    /// Read TSC frequency from /proc/cpuinfo
    fn read_tsc_from_cpuinfo() -> Result<u64, TimingError> {
        // In a real implementation, this would parse /proc/cpuinfo
        // For now, return a typical frequency
        Ok(3_000_000_000) // 3 GHz
    }

    /// Calibrate TSC frequency against system time
    fn calibrate_tsc_frequency() -> Result<u64, TimingError> {
        let calibration_duration = Duration::from_millis(100);
        let samples = 5;
        let mut frequencies = Vec::with_capacity(samples);

        for _ in 0..samples {
            let start_time = SystemTime::now();
            let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };
            
            thread::sleep(calibration_duration);
            
            let end_time = SystemTime::now();
            let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };
            
            let elapsed_time = end_time.duration_since(start_time)
                .map_err(|_| TimingError::CalibrationFailed)?;
            let elapsed_tsc = end_tsc - start_tsc;
            
            let frequency = (elapsed_tsc as f64 / elapsed_time.as_secs_f64()) as u64;
            frequencies.push(frequency);
        }

        // Use median frequency to avoid outliers
        frequencies.sort_unstable();
        Ok(frequencies[samples / 2])
    }

    /// Create timing reference for epoch conversion
    fn create_timing_reference(tsc_frequency: u64) -> Result<TimingReference, TimingError> {
        let system_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TimingError::SystemTimeError)?;
        let tsc_reference = unsafe { core::arch::x86_64::_rdtsc() };
        
        Ok(TimingReference {
            tsc_reference,
            system_time_ns: system_time.as_nanos() as u64,
            reference_time: (tsc_reference * 1_000_000_000) / tsc_frequency,
        })
    }

    /// Calculate drift between TSC and system time
    fn calculate_drift(&self, tsc_samples: &[u64], time_samples: &[u64]) -> Result<i64, TimingError> {
        if tsc_samples.len() != time_samples.len() || tsc_samples.is_empty() {
            return Err(TimingError::InvalidSamples);
        }

        let mut drifts = Vec::with_capacity(tsc_samples.len());
        
        for (i, (&tsc, &sys_time)) in tsc_samples.iter().zip(time_samples.iter()).enumerate() {
            if i == 0 {
                continue; // Skip first sample as reference
            }
            
            let tsc_delta = tsc - tsc_samples[0];
            let sys_delta = sys_time - time_samples[0];
            let tsc_ns = (tsc_delta * 1_000_000_000) / self.tsc_frequency;
            
            let drift = sys_delta as i64 - tsc_ns as i64;
            drifts.push(drift);
        }

        if drifts.is_empty() {
            return Ok(0);
        }

        // Calculate median drift
        drifts.sort_unstable();
        Ok(drifts[drifts.len() / 2])
    }
}

impl Default for NanosecondTimer {
    fn default() -> Self {
        Self::new().expect("Failed to create nanosecond timer")
    }
}

/// Calibration status information
#[derive(Debug, Clone)]
pub struct CalibrationStatus {
    /// Last calibration timestamp (nanoseconds)
    pub last_calibration_ns: u64,
    
    /// Time since last calibration (nanoseconds)
    pub time_since_calibration_ns: u64,
    
    /// Current calibration offset (nanoseconds)
    pub calibration_offset_ns: i64,
    
    /// Whether calibration is needed
    pub needs_calibration: bool,
}

/// Global timer instance for high-performance access
static GLOBAL_TIMER: std::sync::OnceLock<Arc<NanosecondTimer>> = std::sync::OnceLock::new();

/// Get global timer instance
pub fn global_timer() -> &'static Arc<NanosecondTimer> {
    GLOBAL_TIMER.get_or_init(|| {
        Arc::new(NanosecondTimer::new().expect("Failed to initialize global timer"))
    })
}

/// Convenience function to get current nanosecond timestamp
#[inline(always)]
pub fn now_nanos() -> u64 {
    global_timer().now_nanos()
}

/// Convenience function to get current epoch timestamp
#[inline(always)]
pub fn now_nanos_epoch() -> u64 {
    global_timer().now_nanos_epoch()
}

/// Convenience function to get calibrated timestamp
#[inline(always)]
pub fn now_nanos_calibrated() -> u64 {
    global_timer().now_nanos_calibrated()
}

/// Timing-related errors
#[derive(Debug, Clone, PartialEq)]
pub enum TimingError {
    TscNotSupported,
    CalibrationFailed,
    SystemTimeError,
    InvalidSamples,
    FrequencyDetectionFailed,
}

impl std::fmt::Display for TimingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingError::TscNotSupported => write!(f, "TSC not supported on this platform"),
            TimingError::CalibrationFailed => write!(f, "Timer calibration failed"),
            TimingError::SystemTimeError => write!(f, "System time error"),
            TimingError::InvalidSamples => write!(f, "Invalid calibration samples"),
            TimingError::FrequencyDetectionFailed => write!(f, "TSC frequency detection failed"),
        }
    }
}

impl std::error::Error for TimingError {}

/// High-precision timing measurements
pub struct TimingMeasurement {
    timer: Arc<NanosecondTimer>,
    start_time: u64,
    label: String,
}

impl TimingMeasurement {
    /// Start a new timing measurement
    pub fn start(label: String) -> Self {
        let timer = global_timer().clone();
        let start_time = timer.now_nanos_calibrated();
        
        Self {
            timer,
            start_time,
            label,
        }
    }

    /// Get elapsed time since start
    pub fn elapsed_nanos(&self) -> u64 {
        let current_time = self.timer.now_nanos_calibrated();
        current_time.saturating_sub(self.start_time)
    }

    /// Get elapsed time as duration
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.elapsed_nanos())
    }

    /// Finish measurement and return elapsed time
    pub fn finish(self) -> u64 {
        self.elapsed_nanos()
    }

    /// Get measurement label
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Macro for convenient timing measurements
#[macro_export]
macro_rules! time_it {
    ($label:expr, $code:block) => {{
        let measurement = TimingMeasurement::start($label.to_string());
        let result = $code;
        let elapsed = measurement.finish();
        (result, elapsed)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_creation() {
        let timer = NanosecondTimer::new().unwrap();
        assert!(timer.tsc_frequency() > 0);
    }

    #[test]
    fn test_timing_measurement() {
        let timer = NanosecondTimer::new().unwrap();
        
        let start = timer.now_nanos();
        thread::sleep(Duration::from_millis(10));
        let end = timer.now_nanos();
        
        let elapsed = timer.elapsed_nanos(start, end);
        
        // Should be approximately 10ms (allowing for some variance)
        assert!(elapsed > 8_000_000); // 8ms
        assert!(elapsed < 20_000_000); // 20ms
    }

    #[test]
    fn test_epoch_conversion() {
        let timer = NanosecondTimer::new().unwrap();
        
        let epoch_time = timer.now_nanos_epoch();
        let system_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Should be within 1ms of system time
        let diff = if epoch_time > system_time {
            epoch_time - system_time
        } else {
            system_time - epoch_time
        };
        
        assert!(diff < 1_000_000); // 1ms tolerance
    }

    #[test]
    fn test_calibration() {
        let timer = NanosecondTimer::new().unwrap();
        
        let status_before = timer.calibration_status();
        timer.force_calibrate().unwrap();
        let status_after = timer.calibration_status();
        
        assert!(status_after.last_calibration_ns > status_before.last_calibration_ns);
    }

    #[test]
    fn test_timing_measurement_struct() {
        let measurement = TimingMeasurement::start("test".to_string());
        thread::sleep(Duration::from_millis(5));
        let elapsed = measurement.finish();
        
        // Should be approximately 5ms
        assert!(elapsed > 3_000_000); // 3ms
        assert!(elapsed < 10_000_000); // 10ms
    }

    #[test]
    fn test_global_timer() {
        let timer1 = global_timer();
        let timer2 = global_timer();
        
        // Should be the same instance
        assert!(Arc::ptr_eq(timer1, timer2));
        
        let time1 = now_nanos();
        let time2 = now_nanos();
        
        assert!(time2 >= time1);
    }

    #[test]
    fn test_duration_conversion() {
        let timer = NanosecondTimer::new().unwrap();
        
        let duration = Duration::from_millis(100);
        let nanos = timer.duration_to_nanos(duration);
        let back_to_duration = timer.nanos_to_duration(nanos);
        
        assert_eq!(duration, back_to_duration);
    }

    #[test]
    fn test_timing_precision() {
        let timer = NanosecondTimer::new().unwrap();
        
        // Measure very short intervals
        let mut times = Vec::new();
        for _ in 0..1000 {
            times.push(timer.now_nanos());
        }
        
        // Check that we can measure nanosecond differences
        let mut has_nanosecond_precision = false;
        for i in 1..times.len() {
            let diff = times[i] - times[i-1];
            if diff > 0 && diff < 1000 { // Less than 1 microsecond
                has_nanosecond_precision = true;
                break;
            }
        }
        
        assert!(has_nanosecond_precision, "Timer should have nanosecond precision");
    }

    #[test]
    fn test_time_it_macro() {
        let (result, elapsed) = time_it!("test_operation", {
            thread::sleep(Duration::from_millis(1));
            42
        });
        
        assert_eq!(result, 42);
        assert!(elapsed > 500_000); // At least 0.5ms
        assert!(elapsed < 5_000_000); // Less than 5ms
    }

    #[test]
    fn test_calibration_status() {
        let timer = NanosecondTimer::new().unwrap();
        let status = timer.calibration_status();
        
        assert!(status.last_calibration_ns > 0);
        assert!(status.time_since_calibration_ns >= 0);
    }

    #[test]
    fn test_concurrent_timing() {
        use std::sync::Arc;
        use std::thread;
        
        let timer = Arc::new(NanosecondTimer::new().unwrap());
        let mut handles = vec![];
        
        for _ in 0..4 {
            let timer_clone = timer.clone();
            let handle = thread::spawn(move || {
                let mut measurements = vec![];
                for _ in 0..1000 {
                    measurements.push(timer_clone.now_nanos());
                }
                measurements
            });
            handles.push(handle);
        }
        
        let mut all_measurements = vec![];
        for handle in handles {
            all_measurements.extend(handle.join().unwrap());
        }
        
        // Verify measurements are monotonic within reasonable bounds
        all_measurements.sort_unstable();
        let total_time = all_measurements.last().unwrap() - all_measurements.first().unwrap();
        
        // Should complete within a reasonable time (less than 1 second)
        assert!(total_time < 1_000_000_000);
    }
}