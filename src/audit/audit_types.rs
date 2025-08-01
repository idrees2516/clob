use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Nanosecond precision timestamp for audit logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NanoTimestamp(pub u64);

impl NanoTimestamp {
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");
        Self(duration.as_nanos() as u64)
    }

    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    pub fn as_nanos(&self) -> u64 {
        self.0
    }
}

/// Comprehensive audit event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    OrderPlaced {
        order_id: String,
        symbol: String,
        side: String,
        price: f64,
        quantity: f64,
        order_type: String,
        client_id: String,
        timestamp: NanoTimestamp,
        market_conditions: MarketConditions,
    },
    OrderCancelled {
        order_id: String,
        reason: String,
        timestamp: NanoTimestamp,
        remaining_quantity: f64,
    },
    OrderModified {
        order_id: String,
        old_price: f64,
        new_price: f64,
        old_quantity: f64,
        new_quantity: f64,
        timestamp: NanoTimestamp,
    },
    TradeExecuted {
        trade_id: String,
        buy_order_id: String,
        sell_order_id: String,
        symbol: String,
        price: f64,
        quantity: f64,
        timestamp: NanoTimestamp,
        execution_venue: String,
        market_impact: f64,
    },
    QuoteGenerated {
        symbol: String,
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        spread: f64,
        inventory_level: f64,
        volatility_estimate: f64,
        timestamp: NanoTimestamp,
        strategy_params: StrategyParameters,
    },
    RiskEvent {
        event_type: RiskEventType,
        severity: RiskSeverity,
        description: String,
        affected_positions: Vec<String>,
        timestamp: NanoTimestamp,
        mitigation_actions: Vec<String>,
    },
    ComplianceViolation {
        violation_type: ComplianceViolationType,
        severity: ComplianceSeverity,
        description: String,
        affected_orders: Vec<String>,
        timestamp: NanoTimestamp,
        regulatory_reference: String,
    },
    SystemEvent {
        event_type: SystemEventType,
        component: String,
        description: String,
        timestamp: NanoTimestamp,
        system_state: SystemState,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub best_bid: f64,
    pub best_ask: f64,
    pub spread: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub last_trade_price: f64,
    pub volatility: f64,
    pub volume_profile: VolumeProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    pub total_volume: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub average_trade_size: f64,
    pub trade_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameters {
    pub inventory_penalty: f64,
    pub adverse_selection_cost: f64,
    pub market_impact_coefficient: f64,
    pub volatility_adjustment: f64,
    pub liquidity_adjustment: f64,
    pub regime_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskEventType {
    PositionLimitExceeded,
    DrawdownLimitReached,
    VolatilitySpike,
    LiquidityDried,
    CorrelationBreakdown,
    ModelDivergence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceViolationType {
    PositionLimit,
    OrderSizeLimit,
    PriceCollar,
    MarketManipulation,
    SuspiciousActivity,
    ReportingViolation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceSeverity {
    Warning,
    Minor,
    Major,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemEventType {
    Startup,
    Shutdown,
    ConfigurationChange,
    ComponentFailure,
    PerformanceDegradation,
    SecurityEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_latency: f64,
    pub active_connections: u32,
    pub order_processing_rate: f64,
    pub error_rate: f64,
}

/// Audit log entry with integrity verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: String,
    pub sequence_number: u64,
    pub timestamp: NanoTimestamp,
    pub event: AuditEvent,
    pub context: AuditContext,
    pub hash: String,
    pub previous_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditContext {
    pub session_id: String,
    pub user_id: Option<String>,
    pub client_ip: Option<String>,
    pub component: String,
    pub thread_id: String,
    pub correlation_id: String,
    pub additional_metadata: HashMap<String, String>,
}

/// Decision reconstruction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionReconstruction {
    pub decision_id: String,
    pub timestamp: NanoTimestamp,
    pub decision_type: DecisionType,
    pub inputs: DecisionInputs,
    pub algorithm_state: AlgorithmState,
    pub outputs: DecisionOutputs,
    pub execution_trace: Vec<ExecutionStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionType {
    QuoteGeneration,
    OrderPlacement,
    OrderCancellation,
    RiskManagement,
    PositionAdjustment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionInputs {
    pub market_data: MarketConditions,
    pub portfolio_state: PortfolioState,
    pub risk_metrics: RiskMetrics,
    pub strategy_parameters: StrategyParameters,
    pub external_signals: Vec<ExternalSignal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub positions: HashMap<String, f64>,
    pub cash_balance: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub total_exposure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall: f64,
    pub maximum_drawdown: f64,
    pub sharpe_ratio: f64,
    pub beta: f64,
    pub correlation_matrix: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalSignal {
    pub source: String,
    pub signal_type: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: NanoTimestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmState {
    pub model_parameters: HashMap<String, f64>,
    pub internal_state: HashMap<String, serde_json::Value>,
    pub calibration_timestamp: NanoTimestamp,
    pub model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutputs {
    pub actions: Vec<TradingAction>,
    pub risk_adjustments: Vec<RiskAdjustment>,
    pub confidence_scores: HashMap<String, f64>,
    pub expected_outcomes: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingAction {
    pub action_type: String,
    pub symbol: String,
    pub quantity: f64,
    pub price: Option<f64>,
    pub urgency: ActionUrgency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionUrgency {
    Low,
    Medium,
    High,
    Immediate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAdjustment {
    pub adjustment_type: String,
    pub parameter: String,
    pub old_value: f64,
    pub new_value: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub step_id: u32,
    pub timestamp: NanoTimestamp,
    pub operation: String,
    pub inputs: serde_json::Value,
    pub outputs: serde_json::Value,
    pub duration_nanos: u64,
    pub memory_usage: u64,
}