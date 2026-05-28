use super::alert_manager::{Alert, AlertSeverity, NotificationChannel, NotificationError};
use super::timing::now_nanos;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::collections::{HashMap, VecDeque};
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// High-performance notification system for real-time alerts
/// Supports multiple notification channels with failover and rate limiting
pub struct NotificationSystem {
    /// Registered notification channels
    channels: std::sync::RwLock<HashMap<String, Box<dyn NotificationChannel>>>,
    
    /// Notification configuration
    config: NotificationConfig,
    
    /// Notification statistics
    stats: NotificationStats,
    
    /// Rate limiter for notifications
    rate_limiter: RateLimiter,
    
    /// Notification queue for async processing
    notification_queue: std::sync::Mutex<VecDeque<PendingNotification>>,
    
    /// Channel health tracker
    health_tracker: ChannelHealthTracker,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Maximum notifications per second per channel
    pub max_notifications_per_second: u32,
    
    /// Maximum queue size for pending notifications
    pub max_queue_size: usize,
    
    /// Notification timeout (nanoseconds)
    pub notification_timeout_ns: u64,
    
    /// Retry attempts for failed notifications
    pub max_retry_attempts: u32,
    
    /// Retry delay (nanoseconds)
    pub retry_delay_ns: u64,
    
    /// Enable notification batching
    pub enable_batching: bool,
    
    /// Batch size for notifications
    pub batch_size: usize,
    
    /// Batch timeout (nanoseconds)
    pub batch_timeout_ns: u64,
    
    /// Minimum severity for notifications
    pub min_severity: AlertSeverity,
}

/// Pending notification in queue
#[derive(Debug, Clone)]
struct PendingNotification {
    /// Alert to notify about
    alert: Alert,
    
    /// Target channel names
    channels: Vec<String>,
    
    /// Retry count
    retry_count: u32,
    
    /// Queued timestamp
    queued_at: u64,
    
    /// Next retry timestamp
    next_retry_at: u64,
}

/// Notification statistics
#[repr(align(64))]
pub struct NotificationStats {
    /// Total notifications sent
    pub total_sent: AtomicU64,
    
    /// Total notifications failed
    pub total_failed: AtomicU64,
    
    /// Notifications queued
    pub queued_count: AtomicUsize,
    
    /// Average notification latency (nanoseconds)
    pub avg_latency_ns: AtomicU64,
    
    /// Rate limit hits
    pub rate_limit_hits: AtomicU64,
    
    /// Last notification timestamp
    pub last_notification: AtomicU64,
}

/// Rate limiter for notifications
struct RateLimiter {
    /// Tokens per channel
    tokens: HashMap<String, TokenBucket>,
    
    /// Global rate limit
    global_bucket: TokenBucket,
}

/// Token bucket for rate limiting
struct TokenBucket {
    /// Current token count
    tokens: AtomicU64,
    
    /// Maximum tokens
    max_tokens: u64,
    
    /// Refill rate (tokens per second)
    refill_rate: u64,
    
    /// Last refill timestamp
    last_refill: AtomicU64,
}

/// Channel health tracker
struct ChannelHealthTracker {
    /// Health status per channel
    health_status: HashMap<String, ChannelHealth>,
    
    /// Health check interval (nanoseconds)
    check_interval_ns: u64,
    
    /// Last health check timestamp
    last_check: AtomicU64,
}

/// Channel health information
#[derive(Debug, Clone)]
struct ChannelHealth {
    /// Whether channel is healthy
    is_healthy: bool,
    
    /// Success rate (0.0 to 1.0)
    success_rate: f64,
    
    /// Average response time (nanoseconds)
    avg_response_time_ns: u64,
    
    /// Last successful notification
    last_success: u64,
    
    /// Consecutive failures
    consecutive_failures: u32,
    
    /// Health check timestamp
    last_health_check: u64,
}

impl NotificationSystem {
    /// Create a new notification system
    pub fn new(config: NotificationConfig) -> Self {
        Self {
            channels: std::sync::RwLock::new(HashMap::new()),
            config,
            stats: NotificationStats::new(),
            rate_limiter: RateLimiter::new(),
            notification_queue: std::sync::Mutex::new(VecDeque::new()),
            health_tracker: ChannelHealthTracker::new(),
        }
    }

    /// Create notification system with default configuration
    pub fn with_defaults() -> Self {
        Self::new(NotificationConfig::default())
    }

    /// Register a notification channel
    pub fn register_channel(&self, name: String, channel: Box<dyn NotificationChannel>) {
        let mut channels = self.channels.write().unwrap();
        channels.insert(name.clone(), channel);
        
        // Initialize rate limiter for channel
        self.rate_limiter.add_channel(&name, self.config.max_notifications_per_second);
        
        // Initialize health tracking
        self.health_tracker.add_channel(&name);
    }

    /// Unregister a notification channel
    pub fn unregister_channel(&self, name: &str) -> bool {
        let mut channels = self.channels.write().unwrap();
        channels.remove(name).is_some()
    }

    /// Send notification to all channels
    pub fn send_notification(&self, alert: &Alert) -> Result<NotificationResult, NotificationError> {
        // Check minimum severity
        if alert.severity < self.config.min_severity {
            return Ok(NotificationResult {
                sent_count: 0,
                failed_count: 0,
                rate_limited_count: 0,
                total_latency_ns: 0,
            });
        }

        let start_time = now_nanos();
        let mut result = NotificationResult {
            sent_count: 0,
            failed_count: 0,
            rate_limited_count: 0,
            total_latency_ns: 0,
        };

        let channels = self.channels.read().unwrap();
        let channel_names: Vec<String> = channels.keys().cloned().collect();
        drop(channels);

        // Send to each channel
        for channel_name in channel_names {
            match self.send_to_channel(alert, &channel_name) {
                Ok(_) => result.sent_count += 1,
                Err(NotificationError::ChannelUnavailable) => result.failed_count += 1,
                Err(NotificationError::SendTimeout) => result.failed_count += 1,
                Err(_) => result.rate_limited_count += 1,
            }
        }

        result.total_latency_ns = now_nanos() - start_time;
        
        // Update statistics
        self.update_stats(&result);

        Ok(result)
    }

    /// Send notification to specific channels
    pub fn send_to_channels(&self, alert: &Alert, channel_names: &[String]) -> Result<NotificationResult, NotificationError> {
        let start_time = now_nanos();
        let mut result = NotificationResult {
            sent_count: 0,
            failed_count: 0,
            rate_limited_count: 0,
            total_latency_ns: 0,
        };

        for channel_name in channel_names {
            match self.send_to_channel(alert, channel_name) {
                Ok(_) => result.sent_count += 1,
                Err(NotificationError::ChannelUnavailable) => result.failed_count += 1,
                Err(NotificationError::SendTimeout) => result.failed_count += 1,
                Err(_) => result.rate_limited_count += 1,
            }
        }

        result.total_latency_ns = now_nanos() - start_time;
        self.update_stats(&result);

        Ok(result)
    }

    /// Queue notification for async processing
    pub fn queue_notification(&self, alert: Alert, channels: Vec<String>) -> Result<(), NotificationError> {
        let mut queue = self.notification_queue.lock().unwrap();
        
        // Check queue capacity
        if queue.len() >= self.config.max_queue_size {
            return Err(NotificationError::ChannelUnavailable);
        }

        let pending = PendingNotification {
            alert,
            channels,
            retry_count: 0,
            queued_at: now_nanos(),
            next_retry_at: now_nanos(),
        };

        queue.push_back(pending);
        self.stats.queued_count.store(queue.len(), Ordering::Release);

        Ok(())
    }

    /// Process queued notifications
    pub fn process_queue(&self) -> QueueProcessResult {
        let mut queue = self.notification_queue.lock().unwrap();
        let mut processed = 0;
        let mut succeeded = 0;
        let mut failed = 0;
        let current_time = now_nanos();

        // Process notifications that are ready for retry
        let mut remaining_notifications = VecDeque::new();
        
        while let Some(mut pending) = queue.pop_front() {
            processed += 1;
            
            if current_time >= pending.next_retry_at {
                // Try to send notification
                let mut success = false;
                
                for channel_name in &pending.channels {
                    if self.send_to_channel(&pending.alert, channel_name).is_ok() {
                        success = true;
                    }
                }
                
                if success {
                    succeeded += 1;
                } else {
                    pending.retry_count += 1;
                    
                    if pending.retry_count < self.config.max_retry_attempts {
                        // Schedule for retry
                        pending.next_retry_at = current_time + self.config.retry_delay_ns;
                        remaining_notifications.push_back(pending);
                    } else {
                        failed += 1;
                    }
                }
            } else {
                // Not ready for retry yet
                remaining_notifications.push_back(pending);
            }
        }
        
        *queue = remaining_notifications;
        self.stats.queued_count.store(queue.len(), Ordering::Release);

        QueueProcessResult {
            processed,
            succeeded,
            failed,
            remaining: queue.len(),
        }
    }

    /// Get notification statistics
    pub fn get_stats(&self) -> NotificationStatsSnapshot {
        NotificationStatsSnapshot {
            total_sent: self.stats.total_sent.load(Ordering::Acquire),
            total_failed: self.stats.total_failed.load(Ordering::Acquire),
            queued_count: self.stats.queued_count.load(Ordering::Acquire),
            avg_latency_ns: self.stats.avg_latency_ns.load(Ordering::Acquire),
            rate_limit_hits: self.stats.rate_limit_hits.load(Ordering::Acquire),
            last_notification: self.stats.last_notification.load(Ordering::Acquire),
        }
    }

    /// Get channel health status
    pub fn get_channel_health(&self) -> HashMap<String, ChannelHealth> {
        self.health_tracker.get_all_health()
    }

    /// Perform health check on all channels
    pub fn health_check(&self) -> HealthCheckResult {
        let current_time = now_nanos();
        let channels = self.channels.read().unwrap();
        let mut healthy_channels = 0;
        let mut unhealthy_channels = 0;
        let mut total_response_time = 0u64;

        for (name, channel) in channels.iter() {
            let check_start = now_nanos();
            let is_healthy = channel.is_healthy();
            let response_time = now_nanos() - check_start;
            
            total_response_time += response_time;
            
            if is_healthy {
                healthy_channels += 1;
            } else {
                unhealthy_channels += 1;
            }
            
            // Update health tracker
            self.health_tracker.update_health(name, is_healthy, response_time);
        }

        let total_channels = channels.len();
        let avg_response_time = if total_channels > 0 {
            total_response_time / total_channels as u64
        } else {
            0
        };

        HealthCheckResult {
            total_channels,
            healthy_channels,
            unhealthy_channels,
            avg_response_time_ns: avg_response_time,
            check_timestamp: current_time,
        }
    }

    /// Send notification to a specific channel
    fn send_to_channel(&self, alert: &Alert, channel_name: &str) -> Result<(), NotificationError> {
        // Check rate limit
        if !self.rate_limiter.check_rate_limit(channel_name) {
            self.stats.rate_limit_hits.fetch_add(1, Ordering::Relaxed);
            return Err(NotificationError::ChannelUnavailable);
        }

        // Get channel
        let channels = self.channels.read().unwrap();
        let channel = channels.get(channel_name)
            .ok_or(NotificationError::ChannelUnavailable)?;

        // Check channel health
        if !self.health_tracker.is_channel_healthy(channel_name) {
            return Err(NotificationError::ChannelUnavailable);
        }

        // Send notification
        let send_start = now_nanos();
        let result = channel.send_alert(alert);
        let send_latency = now_nanos() - send_start;

        // Update health tracker
        self.health_tracker.record_notification_result(channel_name, result.is_ok(), send_latency);

        result
    }

    /// Update notification statistics
    fn update_stats(&self, result: &NotificationResult) {
        self.stats.total_sent.fetch_add(result.sent_count as u64, Ordering::Relaxed);
        self.stats.total_failed.fetch_add(result.failed_count as u64, Ordering::Relaxed);
        self.stats.last_notification.store(now_nanos(), Ordering::Release);
        
        // Update average latency
        let current_avg = self.stats.avg_latency_ns.load(Ordering::Acquire);
        let new_avg = (current_avg + result.total_latency_ns) / 2;
        self.stats.avg_latency_ns.store(new_avg, Ordering::Release);
    }
}

impl NotificationStats {
    fn new() -> Self {
        Self {
            total_sent: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            queued_count: AtomicUsize::new(0),
            avg_latency_ns: AtomicU64::new(0),
            rate_limit_hits: AtomicU64::new(0),
            last_notification: AtomicU64::new(0),
        }
    }
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            tokens: HashMap::new(),
            global_bucket: TokenBucket::new(1000, 1000), // 1000 notifications per second globally
        }
    }

    fn add_channel(&mut self, channel_name: &str, max_per_second: u32) {
        let bucket = TokenBucket::new(max_per_second as u64, max_per_second as u64);
        self.tokens.insert(channel_name.to_string(), bucket);
    }

    fn check_rate_limit(&self, channel_name: &str) -> bool {
        // Check global rate limit
        if !self.global_bucket.consume_token() {
            return false;
        }

        // Check channel-specific rate limit
        if let Some(bucket) = self.tokens.get(channel_name) {
            bucket.consume_token()
        } else {
            true // No specific limit for this channel
        }
    }
}

impl TokenBucket {
    fn new(max_tokens: u64, refill_rate: u64) -> Self {
        Self {
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            refill_rate,
            last_refill: AtomicU64::new(now_nanos()),
        }
    }

    fn consume_token(&self) -> bool {
        self.refill_tokens();
        
        let current_tokens = self.tokens.load(Ordering::Acquire);
        if current_tokens > 0 {
            self.tokens.compare_exchange(
                current_tokens,
                current_tokens - 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok()
        } else {
            false
        }
    }

    fn refill_tokens(&self) {
        let current_time = now_nanos();
        let last_refill = self.last_refill.load(Ordering::Acquire);
        let time_elapsed = current_time - last_refill;
        
        if time_elapsed >= 1_000_000_000 { // 1 second
            let tokens_to_add = (time_elapsed / 1_000_000_000) * self.refill_rate;
            let current_tokens = self.tokens.load(Ordering::Acquire);
            let new_tokens = (current_tokens + tokens_to_add).min(self.max_tokens);
            
            self.tokens.store(new_tokens, Ordering::Release);
            self.last_refill.store(current_time, Ordering::Release);
        }
    }
}

impl ChannelHealthTracker {
    fn new() -> Self {
        Self {
            health_status: HashMap::new(),
            check_interval_ns: 60_000_000_000, // 1 minute
            last_check: AtomicU64::new(0),
        }
    }

    fn add_channel(&mut self, channel_name: &str) {
        let health = ChannelHealth {
            is_healthy: true,
            success_rate: 1.0,
            avg_response_time_ns: 0,
            last_success: now_nanos(),
            consecutive_failures: 0,
            last_health_check: now_nanos(),
        };
        self.health_status.insert(channel_name.to_string(), health);
    }

    fn is_channel_healthy(&self, channel_name: &str) -> bool {
        self.health_status.get(channel_name)
            .map(|health| health.is_healthy)
            .unwrap_or(false)
    }

    fn update_health(&mut self, channel_name: &str, is_healthy: bool, response_time: u64) {
        if let Some(health) = self.health_status.get_mut(channel_name) {
            health.is_healthy = is_healthy;
            health.avg_response_time_ns = (health.avg_response_time_ns + response_time) / 2;
            health.last_health_check = now_nanos();
        }
    }

    fn record_notification_result(&mut self, channel_name: &str, success: bool, response_time: u64) {
        if let Some(health) = self.health_status.get_mut(channel_name) {
            if success {
                health.last_success = now_nanos();
                health.consecutive_failures = 0;
            } else {
                health.consecutive_failures += 1;
            }
            
            health.avg_response_time_ns = (health.avg_response_time_ns + response_time) / 2;
            
            // Update success rate (simple moving average)
            let new_success_rate = if success { 1.0 } else { 0.0 };
            health.success_rate = (health.success_rate * 0.9) + (new_success_rate * 0.1);
            
            // Mark as unhealthy if too many consecutive failures
            if health.consecutive_failures >= 5 {
                health.is_healthy = false;
            } else if success && health.consecutive_failures == 0 {
                health.is_healthy = true;
            }
        }
    }

    fn get_all_health(&self) -> HashMap<String, ChannelHealth> {
        self.health_status.clone()
    }
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            max_notifications_per_second: 100,
            max_queue_size: 10000,
            notification_timeout_ns: 5_000_000_000, // 5 seconds
            max_retry_attempts: 3,
            retry_delay_ns: 1_000_000_000, // 1 second
            enable_batching: false,
            batch_size: 10,
            batch_timeout_ns: 1_000_000_000, // 1 second
            min_severity: AlertSeverity::Warning,
        }
    }
}

/// Notification result
#[derive(Debug, Clone)]
pub struct NotificationResult {
    pub sent_count: u32,
    pub failed_count: u32,
    pub rate_limited_count: u32,
    pub total_latency_ns: u64,
}

/// Queue processing result
#[derive(Debug, Clone)]
pub struct QueueProcessResult {
    pub processed: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub remaining: usize,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub total_channels: usize,
    pub healthy_channels: usize,
    pub unhealthy_channels: usize,
    pub avg_response_time_ns: u64,
    pub check_timestamp: u64,
}

/// Notification statistics snapshot
#[derive(Debug, Clone)]
pub struct NotificationStatsSnapshot {
    pub total_sent: u64,
    pub total_failed: u64,
    pub queued_count: usize,
    pub avg_latency_ns: u64,
    pub rate_limit_hits: u64,
    pub last_notification: u64,
}

/// Console notification channel (for testing/debugging)
pub struct ConsoleNotificationChannel {
    name: String,
}

impl ConsoleNotificationChannel {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

impl NotificationChannel for ConsoleNotificationChannel {
    fn send_alert(&self, alert: &Alert) -> Result<(), NotificationError> {
        println!("[{}] ALERT: {} - {} (severity: {:?})", 
                 self.name, alert.name, alert.message, alert.severity);
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

/// Log file notification channel
pub struct LogFileNotificationChannel {
    name: String,
    file_path: String,
}

impl LogFileNotificationChannel {
    pub fn new(name: String, file_path: String) -> Self {
        Self { name, file_path }
    }
}

impl NotificationChannel for LogFileNotificationChannel {
    fn send_alert(&self, alert: &Alert) -> Result<(), NotificationError> {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|_| NotificationError::ChannelUnavailable)?;
        
        let log_entry = format!(
            "{} [{}] {} - {} (severity: {:?}, value: {:.2}, threshold: {:.2})\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            alert.id,
            alert.name,
            alert.message,
            alert.severity,
            alert.current_value,
            alert.threshold_value
        );
        
        file.write_all(log_entry.as_bytes())
            .map_err(|_| NotificationError::SendTimeout)?;
        
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_healthy(&self) -> bool {
        std::path::Path::new(&self.file_path).parent()
            .map(|dir| dir.exists())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notification_system_creation() {
        let system = NotificationSystem::with_defaults();
        let stats = system.get_stats();
        
        assert_eq!(stats.total_sent, 0);
        assert_eq!(stats.queued_count, 0);
    }

    #[test]
    fn test_channel_registration() {
        let system = NotificationSystem::with_defaults();
        let channel = Box::new(ConsoleNotificationChannel::new("test".to_string()));
        
        system.register_channel("test".to_string(), channel);
        
        let health = system.get_channel_health();
        assert!(health.contains_key("test"));
    }

    #[test]
    fn test_console_notification_channel() {
        let channel = ConsoleNotificationChannel::new("test_console".to_string());
        
        assert_eq!(channel.name(), "test_console");
        assert!(channel.is_healthy());
        
        let alert = Alert {
            id: "test_alert".to_string(),
            name: "Test Alert".to_string(),
            severity: AlertSeverity::Warning,
            status: super::alert_manager::AlertStatus::Active,
            message: "Test message".to_string(),
            metric_name: "test_metric".to_string(),
            current_value: 100.0,
            threshold_value: 50.0,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            resolved_at: 0,
            context: HashMap::new(),
            fire_count: 1,
            duration_ns: 0,
        };
        
        assert!(channel.send_alert(&alert).is_ok());
    }

    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(10, 5);
        
        // Should be able to consume initial tokens
        for _ in 0..10 {
            assert!(bucket.consume_token());
        }
        
        // Should be rate limited now
        assert!(!bucket.consume_token());
    }

    #[test]
    fn test_rate_limiter() {
        let mut rate_limiter = RateLimiter::new();
        rate_limiter.add_channel("test_channel", 5);
        
        // Should allow initial requests
        for _ in 0..5 {
            assert!(rate_limiter.check_rate_limit("test_channel"));
        }
        
        // Should be rate limited
        assert!(!rate_limiter.check_rate_limit("test_channel"));
    }

    #[test]
    fn test_channel_health_tracker() {
        let mut tracker = ChannelHealthTracker::new();
        tracker.add_channel("test_channel");
        
        assert!(tracker.is_channel_healthy("test_channel"));
        
        // Record some failures
        for _ in 0..5 {
            tracker.record_notification_result("test_channel", false, 1000);
        }
        
        assert!(!tracker.is_channel_healthy("test_channel"));
        
        // Record success to recover
        tracker.record_notification_result("test_channel", true, 500);
        assert!(tracker.is_channel_healthy("test_channel"));
    }

    #[test]
    fn test_notification_queue() {
        let system = NotificationSystem::with_defaults();
        
        let alert = Alert {
            id: "queued_alert".to_string(),
            name: "Queued Alert".to_string(),
            severity: AlertSeverity::Critical,
            status: super::alert_manager::AlertStatus::Active,
            message: "Queued message".to_string(),
            metric_name: "test_metric".to_string(),
            current_value: 200.0,
            threshold_value: 100.0,
            created_at: now_nanos(),
            updated_at: now_nanos(),
            resolved_at: 0,
            context: HashMap::new(),
            fire_count: 1,
            duration_ns: 0,
        };
        
        let result = system.queue_notification(alert, vec!["test_channel".to_string()]);
        assert!(result.is_ok());
        
        let stats = system.get_stats();
        assert_eq!(stats.queued_count, 1);
    }
}