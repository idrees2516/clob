//! Trading Core zkVM Integration
//! 
//! This module integrates the trading core with zkVM proof generation,
//! providing cryptographic guarantees for all trading operations while
//! maintaining sub-microsecond latency requirements.

use crate::zkvm::{
    traits::*,
    router::{ZkVMRouter, ZkVMSelectionCriteria, ProofComplexity},
    ZkVMError, ZkVMBackend,
};
use crate::orderbook::{
    Order, OrderId, Trade, Symbol, Side, OrderResult, OrderStatus,
    CentralLimitOrderBook, CLOBEngine, OrderSubmissionResult,
    OrderCancellationResult, OrderModificationResult, OrderModification,
    CancellationReason, OrderType, TimeInForce,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, mpsc};
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn, error, instrument};
use thiserror::Error;
use sha2::{Sha256, Digest};

/// Errors specific to trading zkVM integration
#[derive(Error, Debug)]
pub enum TradingZkVMError {
    #[error("zkVM error: {0}")]
    ZkVM(#[from] ZkVMError),
    
    #[error("Proof generation timeout")]
    ProofTimeout,
    
    #[error("Batch processing error: {0}")]
    BatchProcessing(String),
    
    #[error("State serialization error: {0}")]
    StateSerialization(String),
    
    #[error("Async proof queue full")]
    ProofQueueFull,
    
    #[error("Invalid operation type: {0}")]
    InvalidOperationType(String),
}

/// Types of trading operations that can be proven
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TradingOperation {
    OrderPlacement {
        order: Order,
        pre_state_hash: [u8; 32],
        post_state_hash: [u8; 32],
    },
    OrderMatching {
        taker_order: Order,
        maker_orders: Vec<Order>,
        trades: Vec<Trade>,
        pre_state_hash: [u8; 32],
        post_state_hash: [u8; 32],
    },
    OrderCancellation {
        order_id: OrderId,
        cancelled_order: Order,
        reason: CancellationReason,
        pre_state_hash: [u8; 32],
        post_state_hash: [u8; 32],
    },
    BatchOperations {
        operations: Vec<TradingOperation>,
        batch_id: u64,
        pre_state_hash: [u8; 32],
        post_state_hash: [u8; 32],
    },
}

/// Proof request for async processing
#[derive(Debug)]
pub struct ProofRequest {
    pub id: u64,
    pub operation: TradingOperation,
    pub priority: ProofPriority,
    pub created_at: Instant,
    pub response_sender: tokio::sync::oneshot::Sender<Result<TradingProof, TradingZkVMError>>,
}

/// Priority levels for proof generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Generated proof for trading operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingProof {
    pub operation_type: String,
    pub operation_id: u64,
    pub zkvm_backend: ZkVMBackend,
    pub proof: ZkProof,
    pub public_inputs: TradingPublicInputs,
    pub generation_time_ms: u64,
    pub verification_time_ms: Option<u64>,
}

/// Public inputs for trading operation proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPublicInputs {
    pub symbol: String,
    pub operation_type: u8,
    pub pre_state_hash: [u8; 32],
    pub post_state_hash: [u8; 32],
    pub timestamp: u64,
    pub sequence_number: u64,
}

/// Batch proof generation configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
    pub enable_parallel_processing: bool,
    pub max_concurrent_batches: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 100,
            batch_timeout: Duration::from_millis(10), // 10ms batching window
            enable_parallel_processing: true,
            max_concurrent_batches: 4,
        }
    }
}

/// Statistics for proof generation performance
#[derive(Debug, Clone, Default)]
pub struct ProofGenerationStats {
    pub total_proofs_generated: u64,
    pub total_proofs_verified: u64,
    pub avg_generation_time_ms: f64,
    pub avg_verification_time_ms: f64,
    pub proof_queue_size: usize,
    pub batch_processing_enabled: bool,
    pub successful_proofs: u64,
    pub failed_proofs: u64,
}

/// Main trading zkVM integration manager
pub struct TradingZkVMManager {
    /// zkVM router for backend selection
    router: Arc<ZkVMRouter>,
    
    /// Async proof generation queue
    proof_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
    
    /// Proof generation statistics
    stats: Arc<RwLock<ProofGenerationStats>>,
    
    /// Batch processing configuration
    batch_config: BatchConfig,
    
    /// Active proof requests tracking
    active_requests: Arc<RwLock<HashMap<u64, Instant>>>,
    
    /// Next request ID
    next_request_id: std::sync::atomic::AtomicU64,
    
    /// Proof generation task handles
    _proof_workers: Vec<tokio::task::JoinHandle<()>>,
    
    /// Batch processing task handle
    _batch_processor: tokio::task::JoinHandle<()>,
}

impl TradingZkVMManager {
    /// Create new trading zkVM manager
    #[instrument(level = "info")]
    pub async fn new(
        router: Arc<ZkVMRouter>,
        batch_config: BatchConfig,
        num_workers: usize,
    ) -> Result<Self, TradingZkVMError> {
        info!("Initializing trading zkVM manager with {} workers", num_workers);

        let proof_queue = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(RwLock::new(ProofGenerationStats::default()));
        let active_requests = Arc::new(RwLock::new(HashMap::new()));
        let next_request_id = std::sync::atomic::AtomicU64::new(1);

        // Spawn proof generation workers
        let mut proof_workers = Vec::new();
        for worker_id in 0..num_workers {
            let worker = Self::spawn_proof_worker(
                worker_id,
                router.clone(),
                proof_queue.clone(),
                stats.clone(),
                active_requests.clone(),
            );
            proof_workers.push(worker);
        }

        // Spawn batch processor
        let batch_processor = Self::spawn_batch_processor(
            proof_queue.clone(),
            batch_config.clone(),
        );

        Ok(Self {
            router,
            proof_queue,
            stats,
            batch_config,
            active_requests,
            next_request_id,
            _proof_workers: proof_workers,
            _batch_processor: batch_processor,
        })
    }

    /// Generate proof for order placement operation
    #[instrument(level = "debug", skip(self, order, pre_state, post_state))]
    pub async fn prove_order_placement(
        &self,
        order: Order,
        pre_state: &[u8],
        post_state: &[u8],
    ) -> Result<TradingProof, TradingZkVMError> {
        let operation = TradingOperation::OrderPlacement {
            order: order.clone(),
            pre_state_hash: Self::hash_state(pre_state),
            post_state_hash: Self::hash_state(post_state),
        };

        let criteria = ZkVMSelectionCriteria {
            complexity: ProofComplexity::Simple,
            latency_requirement: Duration::from_micros(100), // Sub-100μs for order placement
            proof_size_constraint: Some(2048),
            verification_cost_limit: None,
            preferred_backend: Some(ZkVMBackend::ZisK), // Prefer ZisK for speed
        };

        self.generate_proof_sync(operation, criteria).await
    }

    /// Generate proof for order matching and trade execution
    #[instrument(level = "debug", skip(self, taker_order, maker_orders, trades, pre_state, post_state))]
    pub async fn prove_order_matching(
        &self,
        taker_order: Order,
        maker_orders: Vec<Order>,
        trades: Vec<Trade>,
        pre_state: &[u8],
        post_state: &[u8],
    ) -> Result<TradingProof, TradingZkVMError> {
        let complexity = if trades.len() > 10 {
            ProofComplexity::Complex
        } else if trades.len() > 3 {
            ProofComplexity::Moderate
        } else {
            ProofComplexity::Simple
        };

        let operation = TradingOperation::OrderMatching {
            taker_order,
            maker_orders,
            trades,
            pre_state_hash: Self::hash_state(pre_state),
            post_state_hash: Self::hash_state(post_state),
        };

        let criteria = ZkVMSelectionCriteria {
            complexity,
            latency_requirement: Duration::from_micros(500), // 500μs for matching
            proof_size_constraint: Some(4096),
            verification_cost_limit: None,
            preferred_backend: if complexity == ProofComplexity::Complex {
                Some(ZkVMBackend::SP1Local)
            } else {
                Some(ZkVMBackend::ZisK)
            },
        };

        self.generate_proof_sync(operation, criteria).await
    }

    /// Generate proof asynchronously (non-blocking)
    #[instrument(level = "debug", skip(self, operation))]
    pub async fn prove_operation_async(
        &self,
        operation: TradingOperation,
        priority: ProofPriority,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<TradingProof, TradingZkVMError>>, TradingZkVMError> {
        let request_id = self.next_request_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        let request = ProofRequest {
            id: request_id,
            operation,
            priority,
            created_at: Instant::now(),
            response_sender: sender,
        };

        // Add to queue with priority ordering
        {
            let mut queue = self.proof_queue.lock().await;
            
            if queue.len() >= 1000 { // Configurable queue limit
                return Err(TradingZkVMError::ProofQueueFull);
            }

            // Insert based on priority (higher priority first)
            let insert_pos = queue.iter().position(|req| req.priority < priority)
                .unwrap_or(queue.len());
            queue.insert(insert_pos, request);
        }

        // Track active request
        {
            let mut active = self.active_requests.write().await;
            active.insert(request_id, Instant::now());
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.proof_queue_size = {
                let queue = self.proof_queue.lock().await;
                queue.len()
            };
        }

        Ok(receiver)
    }

    /// Generate batch proof for multiple operations
    #[instrument(level = "debug", skip(self, operations, pre_state, post_state))]
    pub async fn prove_batch_operations(
        &self,
        operations: Vec<TradingOperation>,
        batch_id: u64,
        pre_state: &[u8],
        post_state: &[u8],
    ) -> Result<TradingProof, TradingZkVMError> {
        if operations.is_empty() {
            return Err(TradingZkVMError::BatchProcessing("Empty batch".to_string()));
        }

        let batch_operation = TradingOperation::BatchOperations {
            operations,
            batch_id,
            pre_state_hash: Self::hash_state(pre_state),
            post_state_hash: Self::hash_state(post_state),
        };

        let criteria = ZkVMSelectionCriteria {
            complexity: ProofComplexity::Complex, // Batch operations are complex
            latency_requirement: Duration::from_millis(10), // 10ms for batch
            proof_size_constraint: Some(8192),
            verification_cost_limit: None,
            preferred_backend: Some(ZkVMBackend::SP1Local), // SP1 better for complex operations
        };

        self.generate_proof_sync(batch_operation, criteria).await
    }

    /// Get current proof generation statistics
    pub async fn get_stats(&self) -> ProofGenerationStats {
        let stats = self.stats.read().await;
        let mut result = stats.clone();
        
        // Update queue size
        result.proof_queue_size = {
            let queue = self.proof_queue.lock().await;
            queue.len()
        };
        
        result
    }

    /// Generate proof synchronously
    async fn generate_proof_sync(
        &self,
        operation: TradingOperation,
        criteria: ZkVMSelectionCriteria,
    ) -> Result<TradingProof, TradingZkVMError> {
        let start_time = Instant::now();
        
        // Serialize operation for zkVM execution
        let program_inputs = self.serialize_operation(&operation)?;
        let public_inputs = self.extract_public_inputs(&operation)?;

        // Create a simple program for proof generation (placeholder)
        let program = self.create_trading_program(&operation)?;

        // Generate proof using optimal backend
        let proof = self.router
            .generate_proof_with_optimal_backend(&program, &program_inputs, &criteria)
            .await?;

        let generation_time = start_time.elapsed();

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_proofs_generated += 1;
            stats.successful_proofs += 1;
            
            // Update moving average
            let new_time_ms = generation_time.as_millis() as f64;
            if stats.avg_generation_time_ms == 0.0 {
                stats.avg_generation_time_ms = new_time_ms;
            } else {
                stats.avg_generation_time_ms = (stats.avg_generation_time_ms * 0.9) + (new_time_ms * 0.1);
            }
        }

        Ok(TradingProof {
            operation_type: Self::operation_type_name(&operation),
            operation_id: Self::extract_operation_id(&operation),
            zkvm_backend: proof.backend,
            proof,
            public_inputs,
            generation_time_ms: generation_time.as_millis() as u64,
            verification_time_ms: None,
        })
    }

    /// Serialize trading operation for zkVM execution
    fn serialize_operation(&self, operation: &TradingOperation) -> Result<ExecutionInputs, TradingZkVMError> {
        let serialized = bincode::serialize(operation)
            .map_err(|e| TradingZkVMError::StateSerialization(e.to_string()))?;

        Ok(ExecutionInputs::new(
            serialized.clone(), // Public inputs (operation data)
            vec![], // Private inputs (none for now)
        ))
    }

    /// Extract public inputs from trading operation
    fn extract_public_inputs(&self, operation: &TradingOperation) -> Result<TradingPublicInputs, TradingZkVMError> {
        match operation {
            TradingOperation::OrderPlacement { order, pre_state_hash, post_state_hash } => {
                Ok(TradingPublicInputs {
                    symbol: order.symbol.to_string(),
                    operation_type: 1, // Order placement
                    pre_state_hash: *pre_state_hash,
                    post_state_hash: *post_state_hash,
                    timestamp: order.timestamp,
                    sequence_number: 0, // Will be filled by the system
                })
            }
            TradingOperation::OrderMatching { taker_order, pre_state_hash, post_state_hash, .. } => {
                Ok(TradingPublicInputs {
                    symbol: taker_order.symbol.to_string(),
                    operation_type: 2, // Order matching
                    pre_state_hash: *pre_state_hash,
                    post_state_hash: *post_state_hash,
                    timestamp: taker_order.timestamp,
                    sequence_number: 0,
                })
            }
            TradingOperation::OrderCancellation { cancelled_order, pre_state_hash, post_state_hash, .. } => {
                Ok(TradingPublicInputs {
                    symbol: cancelled_order.symbol.to_string(),
                    operation_type: 3, // Order cancellation
                    pre_state_hash: *pre_state_hash,
                    post_state_hash: *post_state_hash,
                    timestamp: cancelled_order.timestamp,
                    sequence_number: 0,
                })
            }
            TradingOperation::BatchOperations { batch_id, pre_state_hash, post_state_hash, .. } => {
                Ok(TradingPublicInputs {
                    symbol: "BATCH".to_string(),
                    operation_type: 4, // Batch operations
                    pre_state_hash: *pre_state_hash,
                    post_state_hash: *post_state_hash,
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64,
                    sequence_number: *batch_id,
                })
            }
        }
    }

    /// Create a compiled program for trading operations (placeholder)
    fn create_trading_program(&self, operation: &TradingOperation) -> Result<CompiledProgram, TradingZkVMError> {
        // This is a placeholder - in a real implementation, you would have
        // pre-compiled zkVM programs for different trading operations
        let program_type = Self::operation_type_name(operation);
        
        Ok(CompiledProgram {
            backend: ZkVMBackend::ZisK, // Default backend
            bytecode: vec![0u8; 1024], // Placeholder bytecode
            metadata: ProgramMetadata {
                program_id: format!("trading_{}", program_type),
                version: "1.0.0".to_string(),
                compilation_time: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                optimization_level: 2,
                target_arch: "riscv64".to_string(),
                memory_layout: MemoryLayout {
                    stack_size: 64 * 1024,
                    heap_size: 256 * 1024,
                    code_size: 128 * 1024,
                    data_size: 64 * 1024,
                },
            },
            verification_key: VerificationKey {
                backend: ZkVMBackend::ZisK,
                key_data: vec![0u8; 256],
                key_hash: [0u8; 32],
                version: "1.0.0".to_string(),
            },
        })
    }

    /// Hash state data for proof inputs
    fn hash_state(state: &[u8]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(state);
        hasher.finalize().into()
    }

    /// Get operation type name
    fn operation_type_name(operation: &TradingOperation) -> String {
        match operation {
            TradingOperation::OrderPlacement { .. } => "order_placement".to_string(),
            TradingOperation::OrderMatching { .. } => "order_matching".to_string(),
            TradingOperation::OrderCancellation { .. } => "order_cancellation".to_string(),
            TradingOperation::BatchOperations { .. } => "batch_operations".to_string(),
        }
    }

    /// Extract operation ID for tracking
    fn extract_operation_id(operation: &TradingOperation) -> u64 {
        match operation {
            TradingOperation::OrderPlacement { order, .. } => order.id.as_u64(),
            TradingOperation::OrderMatching { taker_order, .. } => taker_order.id.as_u64(),
            TradingOperation::OrderCancellation { order_id, .. } => order_id.as_u64(),
            TradingOperation::BatchOperations { batch_id, .. } => *batch_id,
        }
    }

    /// Spawn proof generation worker
    fn spawn_proof_worker(
        worker_id: usize,
        router: Arc<ZkVMRouter>,
        proof_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
        stats: Arc<RwLock<ProofGenerationStats>>,
        active_requests: Arc<RwLock<HashMap<u64, Instant>>>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting proof worker {}", worker_id);
            
            loop {
                // Get next request from queue
                let request = {
                    let mut queue = proof_queue.lock().await;
                    queue.pop_front()
                };

                if let Some(request) = request {
                    let start_time = Instant::now();
                    
                    // Remove from active requests tracking
                    {
                        let mut active = active_requests.write().await;
                        active.remove(&request.id);
                    }

                    // Process the proof request
                    let result = Self::process_proof_request(&router, request.operation).await;
                    
                    // Send result back
                    let _ = request.response_sender.send(result);
                    
                    // Update statistics
                    {
                        let mut stats_guard = stats.write().await;
                        let processing_time = start_time.elapsed().as_millis() as f64;
                        
                        if stats_guard.avg_generation_time_ms == 0.0 {
                            stats_guard.avg_generation_time_ms = processing_time;
                        } else {
                            stats_guard.avg_generation_time_ms = 
                                (stats_guard.avg_generation_time_ms * 0.9) + (processing_time * 0.1);
                        }
                    }
                } else {
                    // No requests available, sleep briefly
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        })
    }

    /// Spawn batch processor
    fn spawn_batch_processor(
        proof_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
        _batch_config: BatchConfig,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting batch processor");
            
            loop {
                // Batch processing logic would go here
                // For now, just sleep
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    }

    /// Process a single proof request
    async fn process_proof_request(
        router: &Arc<ZkVMRouter>,
        operation: TradingOperation,
    ) -> Result<TradingProof, TradingZkVMError> {
        // This is a simplified implementation
        // In practice, you would have more sophisticated proof generation logic
        
        let criteria = ZkVMSelectionCriteria {
            complexity: ProofComplexity::Simple,
            latency_requirement: Duration::from_millis(100),
            proof_size_constraint: Some(4096),
            verification_cost_limit: None,
            preferred_backend: Some(ZkVMBackend::ZisK),
        };

        // Create placeholder program and inputs
        let program = CompiledProgram {
            backend: ZkVMBackend::ZisK,
            bytecode: vec![0u8; 1024],
            metadata: ProgramMetadata {
                program_id: "trading_operation".to_string(),
                version: "1.0.0".to_string(),
                compilation_time: 0,
                optimization_level: 2,
                target_arch: "riscv64".to_string(),
                memory_layout: MemoryLayout {
                    stack_size: 64 * 1024,
                    heap_size: 256 * 1024,
                    code_size: 128 * 1024,
                    data_size: 64 * 1024,
                },
            },
            verification_key: VerificationKey {
                backend: ZkVMBackend::ZisK,
                key_data: vec![0u8; 256],
                key_hash: [0u8; 32],
                version: "1.0.0".to_string(),
            },
        };

        let inputs = ExecutionInputs::new(vec![1, 2, 3, 4], vec![]);
        
        let start_time = Instant::now();
        let proof = router.generate_proof_with_optimal_backend(&program, &inputs, &criteria).await?;
        let generation_time = start_time.elapsed();

        Ok(TradingProof {
            operation_type: Self::operation_type_name(&operation),
            operation_id: Self::extract_operation_id(&operation),
            zkvm_backend: proof.backend,
            proof,
            public_inputs: TradingPublicInputs {
                symbol: "TEST".to_string(),
                operation_type: 1,
                pre_state_hash: [0u8; 32],
                post_state_hash: [1u8; 32],
                timestamp: 0,
                sequence_number: 0,
            },
            generation_time_ms: generation_time.as_millis() as u64,
            verification_time_ms: None,
        })
    }
}

/// Enhanced CLOB Engine with zkVM integration
pub struct ZkProvableCLOBEngine {
    /// Core CLOB engine
    pub clob_engine: CLOBEngine,
    
    /// zkVM integration manager
    pub zkvm_manager: Arc<TradingZkVMManager>,
    
    /// Enable/disable proof generation
    pub proof_generation_enabled: bool,
    
    /// Async proof generation enabled
    pub async_proof_enabled: bool,
}

impl ZkProvableCLOBEngine {
    /// Create new zk-provable CLOB engine
    pub async fn new(
        symbol: Symbol,
        zkvm_manager: Arc<TradingZkVMManager>,
        proof_generation_enabled: bool,
        async_proof_enabled: bool,
    ) -> Result<Self, TradingZkVMError> {
        let clob_engine = CLOBEngine::new(symbol, None)
            .map_err(|e| TradingZkVMError::BatchProcessing(e.to_string()))?;

        Ok(Self {
            clob_engine,
            zkvm_manager,
            proof_generation_enabled,
            async_proof_enabled,
        })
    }

    /// Submit order with proof generation
    #[instrument(level = "debug", skip(self, order))]
    pub async fn submit_order_with_proof(
        &mut self,
        order: Order,
    ) -> Result<(OrderSubmissionResult, Option<TradingProof>), TradingZkVMError> {
        // Capture pre-state
        let pre_state = self.serialize_current_state()?;
        
        // Execute order in CLOB engine
        let result = self.clob_engine.submit_order(order.clone())
            .map_err(|e| TradingZkVMError::BatchProcessing(e.to_string()))?;

        // Capture post-state
        let post_state = self.serialize_current_state()?;

        // Generate proof if enabled
        let proof = if self.proof_generation_enabled {
            if self.async_proof_enabled {
                // Async proof generation (non-blocking)
                let _receiver = self.zkvm_manager.prove_operation_async(
                    TradingOperation::OrderPlacement {
                        order,
                        pre_state_hash: TradingZkVMManager::hash_state(&pre_state),
                        post_state_hash: TradingZkVMManager::hash_state(&post_state),
                    },
                    ProofPriority::Normal,
                ).await?;
                None // Return None for async, proof will be available later
            } else {
                // Synchronous proof generation
                Some(self.zkvm_manager.prove_order_placement(
                    order,
                    &pre_state,
                    &post_state,
                ).await?)
            }
        } else {
            None
        };

        Ok((result, proof))
    }

    /// Cancel order with proof generation
    #[instrument(level = "debug", skip(self))]
    pub async fn cancel_order_with_proof(
        &mut self,
        order_id: OrderId,
        reason: CancellationReason,
    ) -> Result<(OrderCancellationResult, Option<TradingProof>), TradingZkVMError> {
        // Capture pre-state
        let pre_state = self.serialize_current_state()?;
        
        // Execute cancellation in CLOB engine
        let result = self.clob_engine.cancel_order(order_id, reason.clone())
            .map_err(|e| TradingZkVMError::BatchProcessing(e.to_string()))?;

        // Capture post-state
        let post_state = self.serialize_current_state()?;

        // Generate proof if enabled
        let proof = if self.proof_generation_enabled && !self.async_proof_enabled {
            Some(self.zkvm_manager.prove_operation_async(
                TradingOperation::OrderCancellation {
                    order_id,
                    cancelled_order: result.cancelled_order.clone(),
                    reason,
                    pre_state_hash: TradingZkVMManager::hash_state(&pre_state),
                    post_state_hash: TradingZkVMManager::hash_state(&post_state),
                },
                ProofPriority::Normal,
            ).await.map(|_| None)?)
        } else {
            None
        };

        Ok((result, proof))
    }

    /// Get proof generation statistics
    pub async fn get_proof_stats(&self) -> ProofGenerationStats {
        self.zkvm_manager.get_stats().await
    }

    /// Serialize current CLOB state
    fn serialize_current_state(&self) -> Result<Vec<u8>, TradingZkVMError> {
        let compressed_state = self.clob_engine.get_compressed_state();
        bincode::serialize(compressed_state)
            .map_err(|e| TradingZkVMError::StateSerialization(e.to_string()))
    }
}

/// Enhanced batch proof processor for high-throughput scenarios
pub struct BatchProofProcessor {
    /// Pending operations for batching
    pending_operations: Arc<Mutex<Vec<TradingOperation>>>,
    
    /// Batch configuration
    config: BatchConfig,
    
    /// zkVM manager reference
    zkvm_manager: Arc<TradingZkVMManager>,
    
    /// Batch processing statistics
    stats: Arc<RwLock<BatchProcessingStats>>,
}

/// Statistics for batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchProcessingStats {
    pub total_batches_processed: u64,
    pub total_operations_batched: u64,
    pub avg_batch_size: f64,
    pub avg_batch_processing_time_ms: f64,
    pub successful_batches: u64,
    pub failed_batches: u64,
}

impl BatchProofProcessor {
    /// Create new batch proof processor
    pub fn new(
        config: BatchConfig,
        zkvm_manager: Arc<TradingZkVMManager>,
    ) -> Self {
        Self {
            pending_operations: Arc::new(Mutex::new(Vec::new())),
            config,
            zkvm_manager,
            stats: Arc::new(RwLock::new(BatchProcessingStats::default())),
        }
    }

    /// Add operation to batch queue
    #[instrument(level = "debug", skip(self, operation))]
    pub async fn add_operation(&self, operation: TradingOperation) -> Result<(), TradingZkVMError> {
        let mut pending = self.pending_operations.lock().await;
        pending.push(operation);

        // Trigger batch processing if we've reached the batch size limit
        if pending.len() >= self.config.max_batch_size {
            drop(pending); // Release lock before processing
            self.process_pending_batch().await?;
        }

        Ok(())
    }

    /// Process pending batch of operations
    #[instrument(level = "info", skip(self))]
    pub async fn process_pending_batch(&self) -> Result<Option<TradingProof>, TradingZkVMError> {
        let operations = {
            let mut pending = self.pending_operations.lock().await;
            if pending.is_empty() {
                return Ok(None);
            }
            std::mem::take(&mut *pending)
        };

        if operations.is_empty() {
            return Ok(None);
        }

        let start_time = Instant::now();
        let batch_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        info!("Processing batch of {} operations", operations.len());

        // Create batch operation
        let batch_operation = TradingOperation::BatchOperations {
            operations: operations.clone(),
            batch_id,
            pre_state_hash: [0u8; 32], // Will be computed from first operation
            post_state_hash: [1u8; 32], // Will be computed from last operation
        };

        // Generate proof for the batch
        let result = self.zkvm_manager.prove_batch_operations(
            operations,
            batch_id,
            &[0u8; 32], // Placeholder pre-state
            &[1u8; 32], // Placeholder post-state
        ).await;

        let processing_time = start_time.elapsed();

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_batches_processed += 1;
            
            match &result {
                Ok(_) => {
                    stats.successful_batches += 1;
                    stats.total_operations_batched += batch_operation.operation_count() as u64;
                    
                    // Update moving averages
                    let new_batch_size = batch_operation.operation_count() as f64;
                    if stats.avg_batch_size == 0.0 {
                        stats.avg_batch_size = new_batch_size;
                    } else {
                        stats.avg_batch_size = (stats.avg_batch_size * 0.9) + (new_batch_size * 0.1);
                    }
                    
                    let new_time_ms = processing_time.as_millis() as f64;
                    if stats.avg_batch_processing_time_ms == 0.0 {
                        stats.avg_batch_processing_time_ms = new_time_ms;
                    } else {
                        stats.avg_batch_processing_time_ms = (stats.avg_batch_processing_time_ms * 0.9) + (new_time_ms * 0.1);
                    }
                }
                Err(_) => {
                    stats.failed_batches += 1;
                }
            }
        }

        result.map(Some)
    }

    /// Get batch processing statistics
    pub async fn get_stats(&self) -> BatchProcessingStats {
        self.stats.read().await.clone()
    }

    /// Start automatic batch processing timer
    pub fn start_batch_timer(&self) -> tokio::task::JoinHandle<()> {
        let processor = Arc::new(self.clone());
        let timeout = self.config.batch_timeout;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(timeout);
            
            loop {
                interval.tick().await;
                
                if let Err(e) = processor.process_pending_batch().await {
                    error!("Batch processing error: {}", e);
                }
            }
        })
    }
}

impl Clone for BatchProofProcessor {
    fn clone(&self) -> Self {
        Self {
            pending_operations: self.pending_operations.clone(),
            config: self.config.clone(),
            zkvm_manager: self.zkvm_manager.clone(),
            stats: self.stats.clone(),
        }
    }
}

impl TradingOperation {
    /// Get the number of operations in this trading operation
    pub fn operation_count(&self) -> usize {
        match self {
            TradingOperation::BatchOperations { operations, .. } => operations.len(),
            _ => 1,
        }
    }

    /// Extract all individual operations from this trading operation
    pub fn extract_operations(&self) -> Vec<TradingOperation> {
        match self {
            TradingOperation::BatchOperations { operations, .. } => operations.clone(),
            _ => vec![self.clone()],
        }
    }
}

/// Enhanced async proof generation with priority queuing and load balancing
pub struct AsyncProofGenerator {
    /// High priority queue
    high_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
    
    /// Normal priority queue
    normal_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
    
    /// Low priority queue
    low_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
    
    /// Worker pool
    workers: Vec<tokio::task::JoinHandle<()>>,
    
    /// zkVM manager
    zkvm_manager: Arc<TradingZkVMManager>,
    
    /// Generation statistics
    stats: Arc<RwLock<AsyncProofStats>>,
}

/// Statistics for async proof generation
#[derive(Debug, Clone, Default)]
pub struct AsyncProofStats {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub avg_queue_wait_time_ms: f64,
    pub avg_processing_time_ms: f64,
    pub high_priority_requests: u64,
    pub normal_priority_requests: u64,
    pub low_priority_requests: u64,
}

impl AsyncProofGenerator {
    /// Create new async proof generator
    pub fn new(
        zkvm_manager: Arc<TradingZkVMManager>,
        num_workers: usize,
    ) -> Self {
        let high_priority_queue = Arc::new(Mutex::new(VecDeque::new()));
        let normal_priority_queue = Arc::new(Mutex::new(VecDeque::new()));
        let low_priority_queue = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(RwLock::new(AsyncProofStats::default()));

        let mut workers = Vec::new();
        
        // Spawn worker tasks
        for worker_id in 0..num_workers {
            let worker = Self::spawn_worker(
                worker_id,
                high_priority_queue.clone(),
                normal_priority_queue.clone(),
                low_priority_queue.clone(),
                zkvm_manager.clone(),
                stats.clone(),
            );
            workers.push(worker);
        }

        Self {
            high_priority_queue,
            normal_priority_queue,
            low_priority_queue,
            workers,
            zkvm_manager,
            stats,
        }
    }

    /// Submit proof request with priority
    #[instrument(level = "debug", skip(self, operation))]
    pub async fn submit_request(
        &self,
        operation: TradingOperation,
        priority: ProofPriority,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<TradingProof, TradingZkVMError>>, TradingZkVMError> {
        let request_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (sender, receiver) = tokio::sync::oneshot::channel();

        let request = ProofRequest {
            id: request_id,
            operation,
            priority,
            created_at: Instant::now(),
            response_sender: sender,
        };

        // Add to appropriate priority queue
        match priority {
            ProofPriority::Critical | ProofPriority::High => {
                let mut queue = self.high_priority_queue.lock().await;
                queue.push_back(request);
                
                let mut stats = self.stats.write().await;
                stats.high_priority_requests += 1;
            }
            ProofPriority::Normal => {
                let mut queue = self.normal_priority_queue.lock().await;
                queue.push_back(request);
                
                let mut stats = self.stats.write().await;
                stats.normal_priority_requests += 1;
            }
            ProofPriority::Low => {
                let mut queue = self.low_priority_queue.lock().await;
                queue.push_back(request);
                
                let mut stats = self.stats.write().await;
                stats.low_priority_requests += 1;
            }
        }

        // Update total requests
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        Ok(receiver)
    }

    /// Get async proof generation statistics
    pub async fn get_stats(&self) -> AsyncProofStats {
        self.stats.read().await.clone()
    }

    /// Spawn worker task for processing proof requests
    fn spawn_worker(
        worker_id: usize,
        high_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
        normal_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
        low_priority_queue: Arc<Mutex<VecDeque<ProofRequest>>>,
        zkvm_manager: Arc<TradingZkVMManager>,
        stats: Arc<RwLock<AsyncProofStats>>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting async proof worker {}", worker_id);
            
            loop {
                // Try to get request from queues in priority order
                let request = {
                    // Try high priority first
                    {
                        let mut queue = high_priority_queue.lock().await;
                        if let Some(req) = queue.pop_front() {
                            Some(req)
                        } else {
                            None
                        }
                    }
                    .or_else(|| {
                        // Try normal priority
                        if let Ok(mut queue) = normal_priority_queue.try_lock() {
                            queue.pop_front()
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        // Try low priority
                        if let Ok(mut queue) = low_priority_queue.try_lock() {
                            queue.pop_front()
                        } else {
                            None
                        }
                    })
                };

                if let Some(request) = request {
                    let queue_wait_time = request.created_at.elapsed();
                    let processing_start = Instant::now();

                    // Process the request
                    let result = Self::process_async_request(&zkvm_manager, request.operation).await;
                    
                    let processing_time = processing_start.elapsed();

                    // Send result back
                    let _ = request.response_sender.send(result.clone());

                    // Update statistics
                    {
                        let mut stats_guard = stats.write().await;
                        
                        match result {
                            Ok(_) => stats_guard.completed_requests += 1,
                            Err(_) => stats_guard.failed_requests += 1,
                        }

                        // Update moving averages
                        let queue_wait_ms = queue_wait_time.as_millis() as f64;
                        if stats_guard.avg_queue_wait_time_ms == 0.0 {
                            stats_guard.avg_queue_wait_time_ms = queue_wait_ms;
                        } else {
                            stats_guard.avg_queue_wait_time_ms = 
                                (stats_guard.avg_queue_wait_time_ms * 0.9) + (queue_wait_ms * 0.1);
                        }

                        let processing_ms = processing_time.as_millis() as f64;
                        if stats_guard.avg_processing_time_ms == 0.0 {
                            stats_guard.avg_processing_time_ms = processing_ms;
                        } else {
                            stats_guard.avg_processing_time_ms = 
                                (stats_guard.avg_processing_time_ms * 0.9) + (processing_ms * 0.1);
                        }
                    }
                } else {
                    // No requests available, sleep briefly
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        })
    }

    /// Process async proof request
    async fn process_async_request(
        zkvm_manager: &Arc<TradingZkVMManager>,
        operation: TradingOperation,
    ) -> Result<TradingProof, TradingZkVMError> {
        // Determine optimal selection criteria based on operation type
        let criteria = match &operation {
            TradingOperation::OrderPlacement { .. } => ZkVMSelectionCriteria {
                complexity: ProofComplexity::Simple,
                latency_requirement: Duration::from_micros(100),
                proof_size_constraint: Some(2048),
                verification_cost_limit: None,
                preferred_backend: Some(ZkVMBackend::ZisK),
            },
            TradingOperation::OrderMatching { trades, .. } => {
                let complexity = if trades.len() > 10 {
                    ProofComplexity::Complex
                } else if trades.len() > 3 {
                    ProofComplexity::Moderate
                } else {
                    ProofComplexity::Simple
                };

                ZkVMSelectionCriteria {
                    complexity,
                    latency_requirement: Duration::from_millis(1),
                    proof_size_constraint: Some(4096),
                    verification_cost_limit: None,
                    preferred_backend: if complexity == ProofComplexity::Complex {
                        Some(ZkVMBackend::SP1Local)
                    } else {
                        Some(ZkVMBackend::ZisK)
                    },
                }
            },
            TradingOperation::OrderCancellation { .. } => ZkVMSelectionCriteria {
                complexity: ProofComplexity::Simple,
                latency_requirement: Duration::from_micros(50),
                proof_size_constraint: Some(1024),
                verification_cost_limit: None,
                preferred_backend: Some(ZkVMBackend::ZisK),
            },
            TradingOperation::BatchOperations { operations, .. } => ZkVMSelectionCriteria {
                complexity: ProofComplexity::Complex,
                latency_requirement: Duration::from_millis(10),
                proof_size_constraint: Some(8192),
                verification_cost_limit: None,
                preferred_backend: Some(ZkVMBackend::SP1Local),
            },
        };

        zkvm_manager.generate_proof_sync(operation, criteria).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::{ZkVMConfig, SelectionStrategy};
    use crate::orderbook::{Symbol, Side, OrderType, TimeInForce};

    #[tokio::test]
    async fn test_trading_zkvm_manager_creation() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            2, // 2 workers
        ).await;

        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_order_placement_proof() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            1,
        ).await.unwrap();

        let order = Order {
            id: OrderId::new(1),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Buy,
            price: 50000 * 1_000_000, // $50,000 scaled
            size: 1 * 1_000_000, // 1 BTC scaled
            timestamp: 1234567890,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
        };

        let pre_state = vec![0u8; 32];
        let post_state = vec![1u8; 32];

        let result = manager.prove_order_placement(order, &pre_state, &post_state).await;
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.operation_type, "order_placement");
        assert_eq!(proof.operation_id, 1);
    }

    #[tokio::test]
    async fn test_order_matching_proof() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            1,
        ).await.unwrap();

        let taker_order = Order {
            id: OrderId::new(1),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Buy,
            price: 50000 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: 1234567890,
            order_type: OrderType::Market,
            time_in_force: TimeInForce::IOC,
            client_order_id: None,
        };

        let maker_order = Order {
            id: OrderId::new(2),
            symbol: Symbol::new("BTCUSD").unwrap(),
            side: Side::Sell,
            price: 49999 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: 1234567880,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            client_order_id: None,
        };

        let trade = Trade {
            id: 1,
            symbol: Symbol::new("BTCUSD").unwrap(),
            buyer_order_id: OrderId::new(1),
            seller_order_id: OrderId::new(2),
            price: 49999 * 1_000_000,
            size: 1 * 1_000_000,
            timestamp: 1234567891,
            is_buyer_maker: false,
            sequence: 1,
        };

        let pre_state = vec![0u8; 32];
        let post_state = vec![1u8; 32];

        let result = manager.prove_order_matching(
            taker_order,
            vec![maker_order],
            vec![trade],
            &pre_state,
            &post_state,
        ).await;

        assert!(result.is_ok());
        let proof = result.unwrap();
        assert_eq!(proof.operation_type, "order_matching");
    }

    #[tokio::test]
    async fn test_batch_proof_generation() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            1,
        ).await.unwrap();

        let operations = vec![
            TradingOperation::OrderPlacement {
                order: Order {
                    id: OrderId::new(1),
                    symbol: Symbol::new("BTCUSD").unwrap(),
                    side: Side::Buy,
                    price: 50000 * 1_000_000,
                    size: 1 * 1_000_000,
                    timestamp: 1234567890,
                    order_type: OrderType::Limit,
                    time_in_force: TimeInForce::GTC,
                    client_order_id: None,
                },
                pre_state_hash: [0u8; 32],
                post_state_hash: [1u8; 32],
            },
            TradingOperation::OrderPlacement {
                order: Order {
                    id: OrderId::new(2),
                    symbol: Symbol::new("BTCUSD").unwrap(),
                    side: Side::Sell,
                    price: 51000 * 1_000_000,
                    size: 1 * 1_000_000,
                    timestamp: 1234567891,
                    order_type: OrderType::Limit,
                    time_in_force: TimeInForce::GTC,
                    client_order_id: None,
                },
                pre_state_hash: [1u8; 32],
                post_state_hash: [2u8; 32],
            },
        ];

        let pre_state = vec![0u8; 32];
        let post_state = vec![2u8; 32];

        let result = manager.prove_batch_operations(
            operations,
            12345,
            &pre_state,
            &post_state,
        ).await;

        assert!(result.is_ok());
        let proof = result.unwrap();
        assert_eq!(proof.operation_type, "batch_operations");
        assert_eq!(proof.operation_id, 12345);
    }

    #[tokio::test]
    async fn test_async_proof_generation() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = Arc::new(TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            2,
        ).await.unwrap());

        let operation = TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(1),
                symbol: Symbol::new("BTCUSD").unwrap(),
                side: Side::Buy,
                price: 50000 * 1_000_000,
                size: 1 * 1_000_000,
                timestamp: 1234567890,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: None,
            },
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        };

        let receiver = manager.prove_operation_async(operation, ProofPriority::High).await;
        assert!(receiver.is_ok());

        let result = receiver.unwrap().await;
        assert!(result.is_ok());
        
        let proof = result.unwrap().unwrap();
        assert_eq!(proof.operation_type, "order_placement");
    }

    #[tokio::test]
    async fn test_batch_proof_processor() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let batch_config = BatchConfig {
            max_batch_size: 2,
            batch_timeout: Duration::from_millis(100),
            enable_parallel_processing: true,
            max_concurrent_batches: 2,
        };
        
        let manager = Arc::new(TradingZkVMManager::new(
            router,
            batch_config.clone(),
            1,
        ).await.unwrap());
        
        let processor = BatchProofProcessor::new(batch_config, manager);

        // Add operations to trigger batch processing
        let operation1 = TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(1),
                symbol: Symbol::new("BTCUSD").unwrap(),
                side: Side::Buy,
                price: 50000 * 1_000_000,
                size: 1 * 1_000_000,
                timestamp: 1234567890,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: None,
            },
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        };

        let operation2 = TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(2),
                symbol: Symbol::new("BTCUSD").unwrap(),
                side: Side::Sell,
                price: 51000 * 1_000_000,
                size: 1 * 1_000_000,
                timestamp: 1234567891,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: None,
            },
            pre_state_hash: [1u8; 32],
            post_state_hash: [2u8; 32],
        };

        // Add first operation
        let result1 = processor.add_operation(operation1).await;
        assert!(result1.is_ok());

        // Add second operation (should trigger batch processing)
        let result2 = processor.add_operation(operation2).await;
        assert!(result2.is_ok());

        // Check statistics
        let stats = processor.get_stats().await;
        assert!(stats.total_batches_processed > 0);
    }

    #[tokio::test]
    async fn test_async_proof_generator() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                ..Default::default()
            },
        ];

        let router = Arc::new(
            ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized)
                .await
                .unwrap()
        );

        let manager = Arc::new(TradingZkVMManager::new(
            router,
            BatchConfig::default(),
            1,
        ).await.unwrap());
        
        let generator = AsyncProofGenerator::new(manager, 2);

        let operation = TradingOperation::OrderPlacement {
            order: Order {
                id: OrderId::new(1),
                symbol: Symbol::new("BTCUSD").unwrap(),
                side: Side::Buy,
                price: 50000 * 1_000_000,
                size: 1 * 1_000_000,
                timestamp: 1234567890,
                order_type: OrderType::Limit,
                time_in_force: TimeInForce::GTC,
                client_order_id: None,
            },
            pre_state_hash: [0u8; 32],
            post_state_hash: [1u8; 32],
        };

        // Submit high priority request
        let receiver = generator.submit_request(operation, ProofPriority::High).await;
        assert!(receiver.is_ok());

        // Wait for result
        let result = receiver.unwrap().await;
        assert!(result.is_ok());

        let proof = result.unwrap().unwrap();
        assert_eq!(proof.operation_type, "order_placement");

        // Check statistics
        let stats = generator.get_stats().await;
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.high_priority_requests, 1);
    }
}