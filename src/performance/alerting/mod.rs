pub mod alert_manager;
pub mod threshold_monitor;
pub mod notification_system;
pub mod alert_correlation;
pub mod escalation_engine;

pub use alert_manager::*;
pub use threshold_monitor::*;
pub use notification_system::*;
pub use alert_correlation::*;
pub use escalation_engine::*;

// Re-export key types for convenience
pub use alert_manager::{AlertManager, Alert, AlertSeverity, AlertStatus, global_alert_manager};
pub use threshold_monitor::{ThresholdMonitor, ThresholdConfig, ThresholdViolation};
pub use notification_system::{NotificationSystem, NotificationChannel, NotificationConfig};
pub use alert_correlation::{AlertCorrelator, CorrelationRule, CorrelatedAlert};
pub use escalation_engine::{EscalationEngine, EscalationPolicy, EscalationLevel};