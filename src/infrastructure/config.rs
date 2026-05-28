use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use tokio::sync::{mpsc, watch};
use tokio::fs;
use tokio::time::interval;
use crate::error::InfrastructureError;

/// Hot-swappable configuration management system for high-frequency trading
pub struct ConfigurationManager {
    config_store: Arc<RwLock<ConfigStore>>,
    watchers: Arc<RwLock<HashMap<String, ConfigWatcher>>>,
    validators: Arc<RwLock<HashMap<String, Box<dyn ConfigValidator + Send + Sync>>>>,
    change_listeners: Arc<RwLock<Vec<Box<dyn ConfigChangeListener + Send + Sync>>>>,
    audit_log: Arc<RwLock<Vec<ConfigAuditEntry>>>,
    rollback_manager: Arc<RollbackManager>,
    environment: Environment,
    hot_reload_enabled: AtomicBool,
}

/// Configuration store with versioning and rollback support
#[derive(Debug, Clone)]
pub struct ConfigStore {
    configs: HashMap<String, ConfigEntry>,
    version: u64,
    last_updated: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigEntry {
    pub key: String,
    pub value: serde_json::Value,
    pub version: u64,
    pub created_at: u64,
    pub updated_at: u64,
    pub environment: Environment,
    pub tags: Vec<String>,
    pub encrypted: bool,
    pub validation_rules: Vec<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

/// Configuration watcher for file-based hot reloading
pub struct ConfigWatcher {
    pub path: PathBuf,
    pub last_modified: SystemTime,
    pub watch_tx: watch::Sender<ConfigChange>,
    pub watch_rx: watch::Receiver<ConfigChange>,
}

#[derive(Debug, Clone)]
pub struct ConfigChange {
    pub key: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: serde_json::Value,
    pub timestamp: u64,
    pub source: ConfigSource,
}

#[derive(Debug, Clone)]
pub enum ConfigSource {
    File(PathBuf),
    Environment,
    Runtime,
    External(String),
}

/// Configuration validation system
pub trait ConfigValidator: Send + Sync {
    fn validate(&self, key: &str, value: &serde_json::Value) -> Result<(), ValidationError>;
    fn get_schema(&self) -> Option<serde_json::Value>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub error_message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    Range { min: f64, max: f64 },
    Regex { pattern: String },
    Enum { values: Vec<String> },
    Required,
    Type { expected_type: String },
    Custom { validator_name: String },
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub rule: ValidationType,
}

/// Configuration change listener for reactive updates
pub trait ConfigChangeListener: Send + Sync {
    fn on_config_changed(&self, change: &ConfigChange) -> Result<(), InfrastructureError>;
    fn get_watched_keys(&self) -> Vec<String>;
}

/// Configuration audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigAuditEntry {
    pub timestamp: u64,
    pub action: ConfigAction,
    pub key: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
    pub user: String,
    pub source: String,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigAction {
    Create,
    Update,
    Delete,
    Rollback,
    Validate,
}

/// Rollback management for configuration changes
pub struct RollbackManager {
    snapshots: Arc<RwLock<Vec<ConfigSnapshot>>>,
    max_snapshots: usize,
    auto_rollback_enabled: AtomicBool,
    rollback_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub id: String,
    pub timestamp: u64,
    pub config_store: ConfigStore,
    pub description: String,
    pub tags: Vec<String>,
}

/// Environment-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    pub trading_parameters: TradingConfig,
    pub risk_limits: RiskConfig,
    pub connectivity: ConnectivityConfig,
    pub performance: PerformanceConfig,
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub max_order_size: f64,
    pub min_order_size: f64,
    pub tick_size: f64,
    pub max_positions: HashMap<String, f64>,
    pub trading_hours: TradingHours,
    pub order_timeout: Duration,
    pub cancel_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingHours {
    pub start: String, // HH:MM format
    pub end: String,   // HH:MM format
    pub timezone: String,
    pub holidays: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_daily_loss: f64,
    pub max_position_size: f64,
    pub var_limit: f64,
    pub concentration_limit: f64,
    pub leverage_limit: f64,
    pub stop_loss_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityConfig {
    pub venues: HashMap<String, VenueConfig>,
    pub market_data_feeds: HashMap<String, MarketDataConfig>,
    pub fix_sessions: HashMap<String, FixConfig>,
    pub timeouts: TimeoutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConfig {
    pub name: String,
    pub endpoint: String,
    pub port: u16,
    pub protocol: String,
    pub credentials: CredentialConfig,
    pub rate_limits: RateLimitConfig,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub feed_name: String,
    pub endpoint: String,
    pub symbols: Vec<String>,
    pub subscription_type: String,
    pub buffer_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixConfig {
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub host: String,
    pub port: u16,
    pub heartbeat_interval: u32,
    pub logon_timeout: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialConfig {
    pub username: String,
    pub password: String, // Should be encrypted
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub orders_per_second: u32,
    pub requests_per_minute: u32,
    pub burst_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub heartbeat_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub thread_pool_size: usize,
    pub buffer_sizes: HashMap<String, usize>,
    pub batch_sizes: HashMap<String, usize>,
    pub latency_targets: HashMap<String, Duration>,
    pub memory_limits: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_interval: Duration,
    pub alert_thresholds: HashMap<String, f64>,
    pub log_levels: HashMap<String, String>,
    pub dashboard_refresh: Duration,
}

impl ConfigurationManager {
    pub fn new(environment: Environment) -> Result<Self, InfrastructureError> {
        let (rollback_tx, rollback_rx) = mpsc::channel(100);
        
        Ok(Self {
            config_store: Arc::new(RwLock::new(ConfigStore::new())),
            watchers: Arc::new(RwLock::new(HashMap::new())),
            validators: Arc::new(RwLock::new(HashMap::new())),
            change_listeners: Arc::new(RwLock::new(Vec::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            rollback_manager: Arc::new(RollbackManager::new(100, Duration::from_secs(300))),
            environment,
            hot_reload_enabled: AtomicBool::new(true),
        })
    }

    /// Load configuration from file with hot reloading
    pub async fn load_from_file<P: AsRef<Path>>(&self, path: P) -> Result<(), InfrastructureError> {
        let path = path.as_ref().to_path_buf();
        let content = fs::read_to_string(&path).await
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to read config file: {}", e)))?;

        let config_data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to parse config: {}", e)))?;

        // Validate configuration
        self.validate_config(&config_data)?;

        // Update configuration store
        self.update_from_json(config_data).await?;

        // Set up file watcher for hot reloading
        if self.hot_reload_enabled.load(Ordering::Relaxed) {
            self.setup_file_watcher(path).await?;
        }

        Ok(())
    }

    /// Load configuration from environment variables
    pub async fn load_from_environment(&self, prefix: &str) -> Result<(), InfrastructureError> {
        let mut config_data = serde_json::Map::new();

        for (key, value) in std::env::vars() {
            if key.starts_with(prefix) {
                let config_key = key.strip_prefix(prefix)
                    .unwrap_or(&key)
                    .trim_start_matches('_')
                    .to_lowercase();

                // Try to parse as JSON, fallback to string
                let parsed_value = serde_json::from_str(&value)
                    .unwrap_or_else(|_| serde_json::Value::String(value));

                config_data.insert(config_key, parsed_value);
            }
        }

        let config_json = serde_json::Value::Object(config_data);
        self.validate_config(&config_json)?;
        self.update_from_json(config_json).await?;

        Ok(())
    }

    /// Get configuration value with type conversion
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<T, InfrastructureError> {
        let store = self.config_store.read();
        
        if let Some(entry) = store.configs.get(key) {
            serde_json::from_value(entry.value.clone())
                .map_err(|e| InfrastructureError::ConfigError(format!("Failed to deserialize config '{}': {}", key, e)))
        } else {
            Err(InfrastructureError::ConfigError(format!("Configuration key '{}' not found", key)))
        }
    }

    /// Get configuration value with default
    pub fn get_or_default<T: DeserializeOwned + Default>(&self, key: &str) -> T {
        self.get(key).unwrap_or_default()
    }

    /// Set configuration value with validation
    pub async fn set<T: Serialize>(&self, key: &str, value: T) -> Result<(), InfrastructureError> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to serialize value: {}", e)))?;

        // Validate the new value
        self.validate_single_config(key, &json_value)?;

        // Create snapshot before change
        self.create_snapshot("Before setting ".to_string() + key).await?;

        // Get old value for audit
        let old_value = {
            let store = self.config_store.read();
            store.configs.get(key).map(|entry| entry.value.clone())
        };

        // Update configuration
        {
            let mut store = self.config_store.write();
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let entry = ConfigEntry {
                key: key.to_string(),
                value: json_value.clone(),
                version: store.version + 1,
                created_at: old_value.as_ref().map(|_| {
                    store.configs.get(key).unwrap().created_at
                }).unwrap_or(timestamp),
                updated_at: timestamp,
                environment: self.environment.clone(),
                tags: Vec::new(),
                encrypted: false,
                validation_rules: Vec::new(),
            };

            store.configs.insert(key.to_string(), entry);
            store.version += 1;
            store.last_updated = timestamp;
        }

        // Log audit entry
        self.log_audit_entry(ConfigAction::Update, key, old_value, Some(json_value.clone()), true, None).await;

        // Notify listeners
        self.notify_change_listeners(&ConfigChange {
            key: key.to_string(),
            old_value,
            new_value: json_value,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            source: ConfigSource::Runtime,
        }).await?;

        Ok(())
    }

    /// Delete configuration key
    pub async fn delete(&self, key: &str) -> Result<(), InfrastructureError> {
        let old_value = {
            let mut store = self.config_store.write();
            if let Some(entry) = store.configs.remove(key) {
                store.version += 1;
                store.last_updated = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                Some(entry.value)
            } else {
                return Err(InfrastructureError::ConfigError(format!("Configuration key '{}' not found", key)));
            }
        };

        // Log audit entry
        self.log_audit_entry(ConfigAction::Delete, key, old_value.clone(), None, true, None).await;

        // Notify listeners
        self.notify_change_listeners(&ConfigChange {
            key: key.to_string(),
            old_value,
            new_value: serde_json::Value::Null,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            source: ConfigSource::Runtime,
        }).await?;

        Ok(())
    }

    /// Create configuration snapshot for rollback
    pub async fn create_snapshot(&self, description: String) -> Result<String, InfrastructureError> {
        let snapshot_id = format!("snapshot_{}", SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs());

        let snapshot = ConfigSnapshot {
            id: snapshot_id.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            config_store: self.config_store.read().clone(),
            description,
            tags: Vec::new(),
        };

        self.rollback_manager.add_snapshot(snapshot).await;
        Ok(snapshot_id)
    }

    /// Rollback to a previous snapshot
    pub async fn rollback_to_snapshot(&self, snapshot_id: &str) -> Result<(), InfrastructureError> {
        let snapshot = self.rollback_manager.get_snapshot(snapshot_id).await
            .ok_or_else(|| InfrastructureError::ConfigError(format!("Snapshot '{}' not found", snapshot_id)))?;

        // Create current snapshot before rollback
        self.create_snapshot("Before rollback".to_string()).await?;

        // Restore configuration
        {
            let mut store = self.config_store.write();
            *store = snapshot.config_store.clone();
        }

        // Log audit entry
        self.log_audit_entry(ConfigAction::Rollback, "system", None, None, true, Some(format!("Rolled back to snapshot {}", snapshot_id))).await;

        Ok(())
    }

    /// Add configuration validator
    pub fn add_validator(&self, key: String, validator: Box<dyn ConfigValidator + Send + Sync>) {
        let mut validators = self.validators.write();
        validators.insert(key, validator);
    }

    /// Add configuration change listener
    pub fn add_change_listener(&self, listener: Box<dyn ConfigChangeListener + Send + Sync>) {
        let mut listeners = self.change_listeners.write();
        listeners.push(listener);
    }

    /// Get configuration keys matching pattern
    pub fn get_keys_matching(&self, pattern: &str) -> Vec<String> {
        let store = self.config_store.read();
        store.configs.keys()
            .filter(|key| key.contains(pattern))
            .cloned()
            .collect()
    }

    /// Get all configuration as JSON
    pub fn get_all_as_json(&self) -> serde_json::Value {
        let store = self.config_store.read();
        let mut result = serde_json::Map::new();
        
        for (key, entry) in &store.configs {
            result.insert(key.clone(), entry.value.clone());
        }
        
        serde_json::Value::Object(result)
    }

    /// Validate entire configuration
    fn validate_config(&self, config: &serde_json::Value) -> Result<(), InfrastructureError> {
        if let serde_json::Value::Object(obj) = config {
            for (key, value) in obj {
                self.validate_single_config(key, value)?;
            }
        }
        Ok(())
    }

    /// Validate single configuration value
    fn validate_single_config(&self, key: &str, value: &serde_json::Value) -> Result<(), InfrastructureError> {
        let validators = self.validators.read();
        
        if let Some(validator) = validators.get(key) {
            validator.validate(key, value)
                .map_err(|e| InfrastructureError::ConfigError(format!("Validation failed for '{}': {}", key, e.message)))?;
        }

        Ok(())
    }

    /// Update configuration from JSON
    async fn update_from_json(&self, config: serde_json::Value) -> Result<(), InfrastructureError> {
        if let serde_json::Value::Object(obj) = config {
            for (key, value) in obj {
                self.set(&key, value).await?;
            }
        }
        Ok(())
    }

    /// Setup file watcher for hot reloading
    async fn setup_file_watcher(&self, path: PathBuf) -> Result<(), InfrastructureError> {
        let (watch_tx, watch_rx) = watch::channel(ConfigChange {
            key: String::new(),
            old_value: None,
            new_value: serde_json::Value::Null,
            timestamp: 0,
            source: ConfigSource::File(path.clone()),
        });

        let metadata = fs::metadata(&path).await
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to get file metadata: {}", e)))?;

        let watcher = ConfigWatcher {
            path: path.clone(),
            last_modified: metadata.modified()
                .map_err(|e| InfrastructureError::ConfigError(format!("Failed to get modification time: {}", e)))?,
            watch_tx,
            watch_rx,
        };

        {
            let mut watchers = self.watchers.write();
            watchers.insert(path.to_string_lossy().to_string(), watcher);
        }

        // Start file watching task
        let config_manager = Arc::new(self.clone());
        let watch_path = path.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            
            loop {
                interval.tick().await;
                
                if let Ok(metadata) = fs::metadata(&watch_path).await {
                    if let Ok(modified) = metadata.modified() {
                        let watchers = config_manager.watchers.read();
                        if let Some(watcher) = watchers.get(&watch_path.to_string_lossy().to_string()) {
                            if modified > watcher.last_modified {
                                // File has been modified, reload configuration
                                if let Err(e) = config_manager.reload_from_file(&watch_path).await {
                                    eprintln!("Failed to reload configuration: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Reload configuration from file
    async fn reload_from_file(&self, path: &Path) -> Result<(), InfrastructureError> {
        let content = fs::read_to_string(path).await
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to read config file: {}", e)))?;

        let config_data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| InfrastructureError::ConfigError(format!("Failed to parse config: {}", e)))?;

        self.validate_config(&config_data)?;
        self.update_from_json(config_data).await?;

        // Update watcher timestamp
        if let Ok(metadata) = fs::metadata(path).await {
            if let Ok(modified) = metadata.modified() {
                let mut watchers = self.watchers.write();
                if let Some(watcher) = watchers.get_mut(&path.to_string_lossy().to_string()) {
                    watcher.last_modified = modified;
                }
            }
        }

        Ok(())
    }

    /// Notify configuration change listeners
    async fn notify_change_listeners(&self, change: &ConfigChange) -> Result<(), InfrastructureError> {
        let listeners = self.change_listeners.read();
        
        for listener in listeners.iter() {
            if listener.get_watched_keys().contains(&change.key) || listener.get_watched_keys().is_empty() {
                if let Err(e) = listener.on_config_changed(change) {
                    eprintln!("Configuration change listener error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Log audit entry
    async fn log_audit_entry(
        &self,
        action: ConfigAction,
        key: &str,
        old_value: Option<serde_json::Value>,
        new_value: Option<serde_json::Value>,
        success: bool,
        error_message: Option<String>,
    ) {
        let entry = ConfigAuditEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            action,
            key: key.to_string(),
            old_value,
            new_value,
            user: "system".to_string(), // TODO: Get actual user
            source: "configuration_manager".to_string(),
            success,
            error_message,
        };

        let mut audit_log = self.audit_log.write();
        audit_log.push(entry);

        // Trim audit log if it gets too large
        if audit_log.len() > 10000 {
            audit_log.drain(0..1000);
        }
    }

    /// Get audit log entries
    pub fn get_audit_log(&self) -> Vec<ConfigAuditEntry> {
        self.audit_log.read().clone()
    }

    /// Enable/disable hot reloading
    pub fn set_hot_reload(&self, enabled: bool) {
        self.hot_reload_enabled.store(enabled, Ordering::Relaxed);
    }
}

impl Clone for ConfigurationManager {
    fn clone(&self) -> Self {
        Self {
            config_store: Arc::clone(&self.config_store),
            watchers: Arc::clone(&self.watchers),
            validators: Arc::clone(&self.validators),
            change_listeners: Arc::clone(&self.change_listeners),
            audit_log: Arc::clone(&self.audit_log),
            rollback_manager: Arc::clone(&self.rollback_manager),
            environment: self.environment.clone(),
            hot_reload_enabled: AtomicBool::new(self.hot_reload_enabled.load(Ordering::Relaxed)),
        }
    }
}

impl ConfigStore {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            version: 0,
            last_updated: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

impl RollbackManager {
    pub fn new(max_snapshots: usize, rollback_timeout: Duration) -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(Vec::new())),
            max_snapshots,
            auto_rollback_enabled: AtomicBool::new(false),
            rollback_timeout,
        }
    }

    pub async fn add_snapshot(&self, snapshot: ConfigSnapshot) {
        let mut snapshots = self.snapshots.write();
        snapshots.push(snapshot);

        // Trim old snapshots
        if snapshots.len() > self.max_snapshots {
            snapshots.remove(0);
        }
    }

    pub async fn get_snapshot(&self, id: &str) -> Option<ConfigSnapshot> {
        let snapshots = self.snapshots.read();
        snapshots.iter().find(|s| s.id == id).cloned()
    }

    pub async fn list_snapshots(&self) -> Vec<ConfigSnapshot> {
        self.snapshots.read().clone()
    }

    pub fn enable_auto_rollback(&self, enabled: bool) {
        self.auto_rollback_enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Built-in validators
pub struct RangeValidator {
    pub min: f64,
    pub max: f64,
}

impl ConfigValidator for RangeValidator {
    fn validate(&self, key: &str, value: &serde_json::Value) -> Result<(), ValidationError> {
        if let Some(num) = value.as_f64() {
            if num < self.min || num > self.max {
                return Err(ValidationError {
                    field: key.to_string(),
                    message: format!("Value {} is outside range [{}, {}]", num, self.min, self.max),
                    rule: ValidationType::Range { min: self.min, max: self.max },
                });
            }
        } else {
            return Err(ValidationError {
                field: key.to_string(),
                message: "Value is not a number".to_string(),
                rule: ValidationType::Type { expected_type: "number".to_string() },
            });
        }
        Ok(())
    }

    fn get_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "number",
            "minimum": self.min,
            "maximum": self.max
        }))
    }
}

pub struct RegexValidator {
    pub pattern: regex::Regex,
}

impl ConfigValidator for RegexValidator {
    fn validate(&self, key: &str, value: &serde_json::Value) -> Result<(), ValidationError> {
        if let Some(string_val) = value.as_str() {
            if !self.pattern.is_match(string_val) {
                return Err(ValidationError {
                    field: key.to_string(),
                    message: format!("Value '{}' does not match pattern", string_val),
                    rule: ValidationType::Regex { pattern: self.pattern.as_str().to_string() },
                });
            }
        } else {
            return Err(ValidationError {
                field: key.to_string(),
                message: "Value is not a string".to_string(),
                rule: ValidationType::Type { expected_type: "string".to_string() },
            });
        }
        Ok(())
    }

    fn get_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "type": "string",
            "pattern": self.pattern.as_str()
        }))
    }
}

/// Configuration macros for easier usage
#[macro_export]
macro_rules! config_get {
    ($config_manager:expr, $key:expr, $type:ty) => {
        $config_manager.get::<$type>($key)
    };
}

#[macro_export]
macro_rules! config_get_or_default {
    ($config_manager:expr, $key:expr, $type:ty) => {
        $config_manager.get_or_default::<$type>($key)
    };
}

/// Example configuration change listener
pub struct TradingConfigListener {
    pub watched_keys: Vec<String>,
}

impl ConfigChangeListener for TradingConfigListener {
    fn on_config_changed(&self, change: &ConfigChange) -> Result<(), InfrastructureError> {
        println!("Trading configuration changed: {} = {:?}", change.key, change.new_value);
        
        // Implement specific logic for trading parameter changes
        match change.key.as_str() {
            "max_order_size" => {
                // Update order size limits
                println!("Updating maximum order size limits");
            }
            "risk_limits" => {
                // Update risk management parameters
                println!("Updating risk management parameters");
            }
            _ => {}
        }
        
        Ok(())
    }

    fn get_watched_keys(&self) -> Vec<String> {
        self.watched_keys.clone()
    }
}