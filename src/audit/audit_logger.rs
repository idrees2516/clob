use crate::audit::{AuditEvent, AuditLogEntry, AuditContext, NanoTimestamp, DecisionReconstruction};
use crate::error::AuditError;
use async_trait::async_trait;
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// High-performance audit logger with nanosecond precision
#[async_trait]
pub trait AuditLogger: Send + Sync {
    async fn log_event(&self, event: AuditEvent, context: AuditContext) -> Result<String, AuditError>;
    async fn log_decision(&self, decision: DecisionReconstruction) -> Result<String, AuditError>;
    async fn get_audit_trail(&self, start_time: NanoTimestamp, end_time: NanoTimestamp) -> Result<Vec<AuditLogEntry>, AuditError>;
    async fn 