//! ethrex L1 Client Integration
//! 
//! This module implements the ethrex L1 client wrapper for interacting with
//! Ethereum L1 through ethrex, providing proof commitment submission,
//! state root anchoring, and transaction confirmation handling.

use crate::rollup::{types::*, RollupError};
use crate::zkvm::ZkProof;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};

/// ethrex L1 client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthrexClientConfig {
    /// ethrex RPC endpoint URL
    pub rpc_url: String,
    /// Private key for transaction signing (hex encoded)
    pub private_key: Option<String>,
    /// Contract address for proof commitments
    pub proof_commitment_contract: String,
    /// Gas limit for transactions
    pub gas_limit: u64,
    /// Gas price multiplier for priority
    pub gas_price_multiplier: f64,
    /// Maximum retry attempts for failed transactions
    pub max_retry_attempts: u32,
    /// Retry backoff base duration in milliseconds
    pub retry_backoff_base_ms: u64,
    /// Block confirmation requirement
    pub confirmation_blocks: u64,
    /// Transaction timeout in seconds
    pub transaction_timeout_seconds: u64,
}

impl Default for EthrexClientConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://localhost:8545".to_string(),
            private_key: None,
            proof_commitment_contract: "0x0000000000000000000000000000000000000000".to_string(),
            gas_limit: 500_000,
            gas_price_multiplier: 1.2,
            max_retry_attempts: 5,
            retry_backoff_base_ms: 1000,
            confirmation_blocks: 3,
            transaction_timeout_seconds: 300,
        }
    }
}

/// ethrex L1 client for proof commitment and state anchoring
pub struct EthrexL1Client {
    config: EthrexClientConfig,
    rpc_client: Arc<dyn EthereumRpcClient>,
    state_commitment_mapping: Arc<RwLock<HashMap<StateRoot, L1CommitmentRecord>>>,
    pending_transactions: Arc<RwLock<HashMap<TxHash, PendingTransaction>>>,
    finalized_transactions: Arc<RwLock<HashMap<TxHash, FinalizedTransaction>>>,
    reorg_handler: ReorgHandler,
}

/// Ethereum RPC client trait for ethrex interaction
#[async_trait::async_trait]
pub trait EthereumRpcClient: Send + Sync {
    /// Get current block number
    async fn get_block_number(&self) -> Result<L1BlockNumber, RollupError>;
    
    /// Get block by number
    async fn get_block(&self, block_number: L1BlockNumber) -> Result<EthereumBlock, RollupError>;
    
    /// Send raw transaction
    async fn send_raw_transaction(&self, tx_data: &[u8]) -> Result<TxHash, RollupError>;
    
    /// Get transaction receipt
    async fn get_transaction_receipt(&self, tx_hash: TxHash) -> Result<Option<TransactionReceipt>, RollupError>;
    
    /// Get transaction by hash
    async fn get_transaction(&self, tx_hash: TxHash) -> Result<Option<EthereumTransaction>, RollupError>;
    
    /// Estimate gas for transaction
    async fn estimate_gas(&self, tx_request: &TransactionRequest) -> Result<u64, RollupError>;
    
    /// Get current gas price
    async fn get_gas_price(&self) -> Result<u64, RollupError>;
    
    /// Call contract method (read-only)
    async fn call_contract(&self, call_data: &ContractCall) -> Result<Vec<u8>, RollupError>;
}

/// L1 commitment record mapping local state to L1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1CommitmentRecord {
    /// Local state root
    pub state_root: StateRoot,
    /// L1 transaction hash containing the commitment
    pub commitment_tx_hash: TxHash,
    /// L1 block number where commitment was included
    pub l1_block_number: L1BlockNumber,
    /// Merkle root of order book state
    pub order_book_merkle_root: [u8; 32],
    /// Proof commitment hash
    pub proof_commitment_hash: [u8; 32],
    /// Timestamp when commitment was created
    pub commitment_timestamp: u64,
    /// Confirmation status
    pub confirmation_status: ConfirmationStatus,
}

/// Transaction confirmation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfirmationStatus {
    Pending { confirmations: u64 },
    Confirmed { final_block: L1BlockNumber },
    Reorged { new_tx_hash: Option<TxHash> },
    Failed { reason: String },
}

/// Pending transaction awaiting confirmation
#[derive(Debug, Clone)]
pub struct PendingTransaction {
    pub tx_hash: TxHash,
    pub transaction_type: TransactionType,
    pub submission_time: Instant,
    pub retry_count: u32,
    pub gas_used: Option<u64>,
    pub block_number: Option<L1BlockNumber>,
    pub confirmations_needed: u64,
    pub current_confirmations: u64,
}

/// Finalized transaction
#[derive(Debug, Clone)]
pub struct FinalizedTransaction {
    pub tx_hash: TxHash,
    pub transaction_type: TransactionType,
    pub finalization_block: L1BlockNumber,
    pub finalization_time: Instant,
    pub gas_used: u64,
    pub total_confirmations: u64,
}

/// Types of transactions submitted to L1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    ProofCommitment { batch_id: BatchId },
    StateRootAnchor { state_root: StateRoot },
    ReorgRecovery { original_tx: TxHash },
}

/// Ethereum block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumBlock {
    pub number: L1BlockNumber,
    pub hash: [u8; 32],
    pub parent_hash: [u8; 32],
    pub timestamp: u64,
    pub transactions: Vec<TxHash>,
    pub gas_used: u64,
    pub gas_limit: u64,
}

/// Ethereum transaction structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumTransaction {
    pub hash: TxHash,
    pub block_number: Option<L1BlockNumber>,
    pub block_hash: Option<[u8; 32]>,
    pub transaction_index: Option<u32>,
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub value: u64,
    pub gas: u64,
    pub gas_price: u64,
    pub input: Vec<u8>,
    pub nonce: u64,
}

/// Transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    pub transaction_hash: TxHash,
    pub block_number: L1BlockNumber,
    pub block_hash: [u8; 32],
    pub transaction_index: u32,
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub gas_used: u64,
    pub status: TransactionStatus,
    pub logs: Vec<EventLog>,
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Success,
    Failed,
}

/// Event log from transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    pub address: [u8; 20],
    pub topics: Vec<[u8; 32]>,
    pub data: Vec<u8>,
}

/// Transaction request for gas estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRequest {
    pub from: Option<[u8; 20]>,
    pub to: Option<[u8; 20]>,
    pub gas: Option<u64>,
    pub gas_price: Option<u64>,
    pub value: Option<u64>,
    pub data: Option<Vec<u8>>,
}

/// Contract call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCall {
    pub to: [u8; 20],
    pub data: Vec<u8>,
    pub block_number: Option<L1BlockNumber>,
}

/// Proof commitment data structure for L1 submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCommitmentData {
    /// Batch ID being committed
    pub batch_id: BatchId,
    /// State root before batch execution
    pub state_root_before: StateRoot,
    /// State root after batch execution
    pub state_root_after: StateRoot,
    /// Merkle root of order book state
    pub order_book_merkle_root: [u8; 32],
    /// ZK proof hash
    pub proof_hash: [u8; 32],
    /// Timestamp of commitment
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: CommitmentMetadata,
}

/// Additional metadata for proof commitments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentMetadata {
    /// Number of orders in the batch
    pub order_count: u32,
    /// Number of trades executed
    pub trade_count: u32,
    /// Total gas used for batch execution
    pub gas_used: u64,
    /// zkVM backend used for proof generation
    pub zkvm_backend: String,
}

/// Reorganization handler for L1 chain reorgs
pub struct ReorgHandler {
    /// Detected reorganizations
    detected_reorgs: Arc<RwLock<Vec<ReorgEvent>>>,
    /// Reorg recovery strategies
    recovery_strategies: HashMap<TransactionType, Box<dyn ReorgRecoveryStrategy>>,
}

/// Reorganization event
#[derive(Debug, Clone)]
pub struct ReorgEvent {
    /// Block number where reorg was detected
    pub reorg_block: L1BlockNumber,
    /// Original block hash that was reorged
    pub original_block_hash: [u8; 32],
    /// New block hash after reorg
    pub new_block_hash: [u8; 32],
    /// Affected transactions
    pub affected_transactions: Vec<TxHash>,
    /// Detection timestamp
    pub detection_time: Instant,
}

/// Strategy for recovering from reorganizations
#[async_trait::async_trait]
pub trait ReorgRecoveryStrategy: Send + Sync {
    /// Handle a reorganization event for a specific transaction type
    async fn handle_reorg(
        &self,
        reorg_event: &ReorgEvent,
        affected_tx: TxHash,
        client: &EthrexL1Client,
    ) -> Result<TxHash, RollupError>;
}

impl EthrexL1Client {
    /// Create a new ethrex L1 client
    pub fn new(
        config: EthrexClientConfig,
        rpc_client: Arc<dyn EthereumRpcClient>,
    ) -> Self {
        Self {
            config,
            rpc_client,
            state_commitment_mapping: Arc::new(RwLock::new(HashMap::new())),
            pending_transactions: Arc::new(RwLock::new(HashMap::new())),
            finalized_transactions: Arc::new(RwLock::new(HashMap::new())),
            reorg_handler: ReorgHandler::new(),
        }
    }

    /// Submit proof commitment to ethrex L1 (Requirement 2.1)
    pub async fn submit_proof_commitment(
        &self,
        batch_id: BatchId,
        proof: &ZkProof,
        state_root_before: StateRoot,
        state_root_after: StateRoot,
        order_book_merkle_root: [u8; 32],
    ) -> Result<TxHash, RollupError> {
        info!("Submitting proof commitment for batch {} to ethrex L1", batch_id);

        // Create proof commitment data (Requirement 2.2)
        let commitment_data = ProofCommitmentData {
            batch_id,
            state_root_before,
            state_root_after,
            order_book_merkle_root,
            proof_hash: self.compute_proof_hash(proof)?,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: CommitmentMetadata {
                order_count: 0, // This would be populated from batch data
                trade_count: 0, // This would be populated from execution result
                gas_used: 0,    // This would be populated from execution result
                zkvm_backend: format!("{:?}", proof.backend),
            },
        };

        // Encode commitment data for contract call
        let encoded_data = self.encode_commitment_data(&commitment_data)?;

        // Create transaction request
        let tx_request = TransactionRequest {
            from: None, // Will be set by wallet
            to: Some(self.parse_address(&self.config.proof_commitment_contract)?),
            gas: Some(self.config.gas_limit),
            gas_price: None, // Will be estimated
            value: Some(0),
            data: Some(encoded_data),
        };

        // Submit transaction with retry logic
        let tx_hash = self.submit_transaction_with_retry(
            tx_request,
            TransactionType::ProofCommitment { batch_id },
        ).await?;

        // Create L1 commitment record (Requirement 2.6)
        let commitment_record = L1CommitmentRecord {
            state_root: state_root_after,
            commitment_tx_hash: tx_hash,
            l1_block_number: 0, // Will be updated when confirmed
            order_book_merkle_root,
            proof_commitment_hash: self.compute_commitment_hash(&commitment_data)?,
            commitment_timestamp: commitment_data.timestamp,
            confirmation_status: ConfirmationStatus::Pending { confirmations: 0 },
        };

        // Store commitment mapping (Requirement 2.6)
        {
            let mut mapping = self.state_commitment_mapping.write().await;
            mapping.insert(state_root_after, commitment_record);
        }

        info!(
            "Submitted proof commitment for batch {} with tx hash: {:?}",
            batch_id, tx_hash
        );

        Ok(tx_hash)
    }

    /// Submit state root anchor to L1
    pub async fn submit_state_root_anchor(
        &self,
        state_root: StateRoot,
        proof: &ZkProof,
    ) -> Result<TxHash, RollupError> {
        info!("Submitting state root anchor to ethrex L1: {:?}", state_root);

        // Create state anchor data
        let anchor_data = self.encode_state_anchor_data(state_root, proof)?;

        let tx_request = TransactionRequest {
            from: None,
            to: Some(self.parse_address(&self.config.proof_commitment_contract)?),
            gas: Some(self.config.gas_limit),
            gas_price: None,
            value: Some(0),
            data: Some(anchor_data),
        };

        let tx_hash = self.submit_transaction_with_retry(
            tx_request,
            TransactionType::StateRootAnchor { state_root },
        ).await?;

        info!("Submitted state root anchor with tx hash: {:?}", tx_hash);
        Ok(tx_hash)
    }

    /// Update transaction confirmation status (Requirement 2.3)
    pub async fn update_transaction_confirmations(&self) -> Result<(), RollupError> {
        let current_block = self.rpc_client.get_block_number().await?;
        
        let pending_txs: Vec<(TxHash, PendingTransaction)> = {
            let pending = self.pending_transactions.read().await;
            pending.iter().map(|(k, v)| (*k, v.clone())).collect()
        };

        for (tx_hash, mut pending_tx) in pending_txs {
            // Get transaction receipt
            if let Some(receipt) = self.rpc_client.get_transaction_receipt(tx_hash).await? {
                let confirmations = current_block.saturating_sub(receipt.block_number);
                pending_tx.current_confirmations = confirmations;
                pending_tx.block_number = Some(receipt.block_number);
                pending_tx.gas_used = Some(receipt.gas_used);

                // Check if transaction is finalized
                if confirmations >= self.config.confirmation_blocks {
                    // Move to finalized transactions
                    let finalized_tx = FinalizedTransaction {
                        tx_hash,
                        transaction_type: pending_tx.transaction_type.clone(),
                        finalization_block: receipt.block_number,
                        finalization_time: Instant::now(),
                        gas_used: receipt.gas_used,
                        total_confirmations: confirmations,
                    };

                    // Update state commitment mapping with L1 block reference (Requirement 2.3)
                    if let TransactionType::ProofCommitment { .. } = pending_tx.transaction_type {
                        self.update_commitment_confirmation(tx_hash, receipt.block_number).await?;
                    }

                    // Move transaction to finalized
                    {
                        let mut pending = self.pending_transactions.write().await;
                        let mut finalized = self.finalized_transactions.write().await;
                        pending.remove(&tx_hash);
                        finalized.insert(tx_hash, finalized_tx);
                    }

                    info!("Transaction {:?} finalized at block {}", tx_hash, receipt.block_number);
                } else {
                    // Update pending transaction
                    let mut pending = self.pending_transactions.write().await;
                    if let Some(tx) = pending.get_mut(&tx_hash) {
                        *tx = pending_tx;
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle L1 reorganization events (Requirement 2.8)
    pub async fn handle_reorganization(&self, reorg_event: ReorgEvent) -> Result<(), RollupError> {
        warn!(
            "Handling L1 reorganization at block {} - original: {:?}, new: {:?}",
            reorg_event.reorg_block, reorg_event.original_block_hash, reorg_event.new_block_hash
        );

        // Check which of our transactions were affected
        for affected_tx in &reorg_event.affected_transactions {
            if let Some(pending_tx) = {
                let pending = self.pending_transactions.read().await;
                pending.get(affected_tx).cloned()
            } {
                // Use recovery strategy based on transaction type
                let recovery_result = self.reorg_handler
                    .handle_transaction_reorg(&reorg_event, *affected_tx, &pending_tx, self)
                    .await?;

                if let Some(new_tx_hash) = recovery_result {
                    info!(
                        "Resubmitted transaction {:?} as {:?} due to reorganization",
                        affected_tx, new_tx_hash
                    );

                    // Update commitment mapping if this was a proof commitment
                    if let TransactionType::ProofCommitment { .. } = pending_tx.transaction_type {
                        self.update_commitment_after_reorg(*affected_tx, new_tx_hash).await?;
                    }
                }
            }
        }

        // Store reorg event for audit trail
        {
            let mut reorgs = self.reorg_handler.detected_reorgs.write().await;
            reorgs.push(reorg_event);
        }

        Ok(())
    }

    /// Get L1 commitment record for a state root (Requirement 2.6)
    pub async fn get_commitment_record(&self, state_root: StateRoot) -> Option<L1CommitmentRecord> {
        let mapping = self.state_commitment_mapping.read().await;
        mapping.get(&state_root).cloned()
    }

    /// Get L1 transaction hash for historical state commitment (Requirement 2.7)
    pub async fn get_l1_transaction_hash(&self, state_root: StateRoot) -> Option<TxHash> {
        let mapping = self.state_commitment_mapping.read().await;
        mapping.get(&state_root).map(|record| record.commitment_tx_hash)
    }

    /// Get all commitment records for audit purposes
    pub async fn get_all_commitment_records(&self) -> HashMap<StateRoot, L1CommitmentRecord> {
        let mapping = self.state_commitment_mapping.read().await;
        mapping.clone()
    }

    /// Submit transaction with exponential backoff retry (Requirement 2.4)
    async fn submit_transaction_with_retry(
        &self,
        mut tx_request: TransactionRequest,
        tx_type: TransactionType,
    ) -> Result<TxHash, RollupError> {
        let mut retry_count = 0;
        let mut backoff_duration = Duration::from_millis(self.config.retry_backoff_base_ms);

        loop {
            // Estimate gas if not provided
            if tx_request.gas.is_none() {
                tx_request.gas = Some(self.rpc_client.estimate_gas(&tx_request).await?);
            }

            // Get current gas price if not provided
            if tx_request.gas_price.is_none() {
                let base_gas_price = self.rpc_client.get_gas_price().await?;
                tx_request.gas_price = Some(
                    (base_gas_price as f64 * self.config.gas_price_multiplier) as u64
                );
            }

            // Create and sign transaction (simplified - in practice would use proper signing)
            let tx_data = self.create_signed_transaction(&tx_request)?;

            // Submit transaction
            match self.rpc_client.send_raw_transaction(&tx_data).await {
                Ok(tx_hash) => {
                    // Track pending transaction
                    let pending_tx = PendingTransaction {
                        tx_hash,
                        transaction_type: tx_type,
                        submission_time: Instant::now(),
                        retry_count,
                        gas_used: None,
                        block_number: None,
                        confirmations_needed: self.config.confirmation_blocks,
                        current_confirmations: 0,
                    };

                    {
                        let mut pending = self.pending_transactions.write().await;
                        pending.insert(tx_hash, pending_tx);
                    }

                    return Ok(tx_hash);
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= self.config.max_retry_attempts {
                        return Err(RollupError::L1SyncError(format!(
                            "Failed to submit transaction after {} attempts: {}",
                            retry_count, e
                        )));
                    }

                    warn!(
                        "Transaction submission failed (attempt {}): {}. Retrying in {:?}",
                        retry_count, e, backoff_duration
                    );

                    tokio::time::sleep(backoff_duration).await;
                    backoff_duration *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Update commitment record confirmation status
    async fn update_commitment_confirmation(
        &self,
        tx_hash: TxHash,
        block_number: L1BlockNumber,
    ) -> Result<(), RollupError> {
        let mut mapping = self.state_commitment_mapping.write().await;
        
        // Find the commitment record with this transaction hash
        for (_, record) in mapping.iter_mut() {
            if record.commitment_tx_hash == tx_hash {
                record.l1_block_number = block_number;
                record.confirmation_status = ConfirmationStatus::Confirmed {
                    final_block: block_number,
                };
                break;
            }
        }

        Ok(())
    }

    /// Update commitment record after reorganization
    async fn update_commitment_after_reorg(
        &self,
        old_tx_hash: TxHash,
        new_tx_hash: TxHash,
    ) -> Result<(), RollupError> {
        let mut mapping = self.state_commitment_mapping.write().await;
        
        for (_, record) in mapping.iter_mut() {
            if record.commitment_tx_hash == old_tx_hash {
                record.commitment_tx_hash = new_tx_hash;
                record.confirmation_status = ConfirmationStatus::Reorged {
                    new_tx_hash: Some(new_tx_hash),
                };
                break;
            }
        }

        Ok(())
    }

    /// Compute hash of ZK proof
    fn compute_proof_hash(&self, proof: &ZkProof) -> Result<[u8; 32], RollupError> {
        let mut hasher = Sha256::new();
        hasher.update(&proof.proof_data);
        hasher.update(&proof.public_inputs);
        hasher.update(&proof.verification_key_hash);
        Ok(hasher.finalize().into())
    }

    /// Compute hash of commitment data
    fn compute_commitment_hash(&self, commitment: &ProofCommitmentData) -> Result<[u8; 32], RollupError> {
        let mut hasher = Sha256::new();
        hasher.update(&commitment.batch_id.to_be_bytes());
        hasher.update(&commitment.state_root_before);
        hasher.update(&commitment.state_root_after);
        hasher.update(&commitment.order_book_merkle_root);
        hasher.update(&commitment.proof_hash);
        hasher.update(&commitment.timestamp.to_be_bytes());
        Ok(hasher.finalize().into())
    }

    /// Encode commitment data for contract call
    fn encode_commitment_data(&self, commitment: &ProofCommitmentData) -> Result<Vec<u8>, RollupError> {
        // Simplified encoding - in practice would use proper ABI encoding
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&commitment.batch_id.to_be_bytes());
        encoded.extend_from_slice(&commitment.state_root_before);
        encoded.extend_from_slice(&commitment.state_root_after);
        encoded.extend_from_slice(&commitment.order_book_merkle_root);
        encoded.extend_from_slice(&commitment.proof_hash);
        encoded.extend_from_slice(&commitment.timestamp.to_be_bytes());
        Ok(encoded)
    }

    /// Encode state anchor data for contract call
    fn encode_state_anchor_data(&self, state_root: StateRoot, proof: &ZkProof) -> Result<Vec<u8>, RollupError> {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&state_root);
        encoded.extend_from_slice(&self.compute_proof_hash(proof)?);
        Ok(encoded)
    }

    /// Parse address string to bytes
    fn parse_address(&self, address_str: &str) -> Result<[u8; 20], RollupError> {
        let address_str = address_str.strip_prefix("0x").unwrap_or(address_str);
        let bytes = hex::decode(address_str)
            .map_err(|e| RollupError::L1SyncError(format!("Invalid address format: {}", e)))?;
        
        if bytes.len() != 20 {
            return Err(RollupError::L1SyncError("Address must be 20 bytes".to_string()));
        }

        let mut address = [0u8; 20];
        address.copy_from_slice(&bytes);
        Ok(address)
    }

    /// Create signed transaction (simplified implementation)
    fn create_signed_transaction(&self, tx_request: &TransactionRequest) -> Result<Vec<u8>, RollupError> {
        // This is a simplified implementation
        // In practice, you would use proper transaction signing with the private key
        let mut tx_data = Vec::new();
        
        if let Some(to) = tx_request.to {
            tx_data.extend_from_slice(&to);
        }
        
        if let Some(data) = &tx_request.data {
            tx_data.extend_from_slice(data);
        }
        
        if let Some(gas) = tx_request.gas {
            tx_data.extend_from_slice(&gas.to_be_bytes());
        }
        
        if let Some(gas_price) = tx_request.gas_price {
            tx_data.extend_from_slice(&gas_price.to_be_bytes());
        }

        Ok(tx_data)
    }
}

impl ReorgHandler {
    /// Create a new reorganization handler
    pub fn new() -> Self {
        let mut recovery_strategies: HashMap<TransactionType, Box<dyn ReorgRecoveryStrategy>> = HashMap::new();
        
        // Register recovery strategies for different transaction types
        recovery_strategies.insert(
            TransactionType::ProofCommitment { batch_id: 0 }, // Template key
            Box::new(ProofCommitmentRecoveryStrategy),
        );
        recovery_strategies.insert(
            TransactionType::StateRootAnchor { state_root: [0u8; 32] }, // Template key
            Box::new(StateRootRecoveryStrategy),
        );

        Self {
            detected_reorgs: Arc::new(RwLock::new(Vec::new())),
            recovery_strategies,
        }
    }

    /// Handle transaction reorganization
    pub async fn handle_transaction_reorg(
        &self,
        reorg_event: &ReorgEvent,
        affected_tx: TxHash,
        pending_tx: &PendingTransaction,
        client: &EthrexL1Client,
    ) -> Result<Option<TxHash>, RollupError> {
        // Find appropriate recovery strategy
        let strategy_key = match &pending_tx.transaction_type {
            TransactionType::ProofCommitment { .. } => {
                TransactionType::ProofCommitment { batch_id: 0 }
            }
            TransactionType::StateRootAnchor { .. } => {
                TransactionType::StateRootAnchor { state_root: [0u8; 32] }
            }
            TransactionType::ReorgRecovery { .. } => {
                // Already a recovery transaction, don't create another one
                return Ok(None);
            }
        };

        if let Some(strategy) = self.recovery_strategies.get(&strategy_key) {
            let new_tx_hash = strategy.handle_reorg(reorg_event, affected_tx, client).await?;
            Ok(Some(new_tx_hash))
        } else {
            warn!("No recovery strategy found for transaction type: {:?}", pending_tx.transaction_type);
            Ok(None)
        }
    }
}

/// Recovery strategy for proof commitment transactions
struct ProofCommitmentRecoveryStrategy;

#[async_trait::async_trait]
impl ReorgRecoveryStrategy for ProofCommitmentRecoveryStrategy {
    async fn handle_reorg(
        &self,
        _reorg_event: &ReorgEvent,
        original_tx: TxHash,
        client: &EthrexL1Client,
    ) -> Result<TxHash, RollupError> {
        info!("Recovering proof commitment transaction after reorg: {:?}", original_tx);
        
        // In a real implementation, we would:
        // 1. Retrieve the original transaction data
        // 2. Resubmit with updated nonce and gas price
        // 3. Update any dependent state
        
        // For now, create a mock recovery transaction
        let recovery_tx_data = vec![0u8; 100]; // Mock transaction data
        let new_tx_hash = client.rpc_client.send_raw_transaction(&recovery_tx_data).await?;
        
        info!("Resubmitted proof commitment as: {:?}", new_tx_hash);
        Ok(new_tx_hash)
    }
}

/// Recovery strategy for state root anchor transactions
struct StateRootRecoveryStrategy;

#[async_trait::async_trait]
impl ReorgRecoveryStrategy for StateRootRecoveryStrategy {
    async fn handle_reorg(
        &self,
        _reorg_event: &ReorgEvent,
        original_tx: TxHash,
        client: &EthrexL1Client,
    ) -> Result<TxHash, RollupError> {
        info!("Recovering state root anchor transaction after reorg: {:?}", original_tx);
        
        // Similar recovery logic for state root anchoring
        let recovery_tx_data = vec![1u8; 100]; // Mock transaction data
        let new_tx_hash = client.rpc_client.send_raw_transaction(&recovery_tx_data).await?;
        
        info!("Resubmitted state root anchor as: {:?}", new_tx_hash);
        Ok(new_tx_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::{ZkVMBackend, ProofMetadata};

    /// Mock ethrex RPC client for testing
    struct MockEthereumRpcClient {
        current_block: L1BlockNumber,
    }

    #[async_trait::async_trait]
    impl EthereumRpcClient for MockEthereumRpcClient {
        async fn get_block_number(&self) -> Result<L1BlockNumber, RollupError> {
            Ok(self.current_block)
        }

        async fn get_block(&self, block_number: L1BlockNumber) -> Result<EthereumBlock, RollupError> {
            Ok(EthereumBlock {
                number: block_number,
                hash: [42u8; 32],
                parent_hash: [41u8; 32],
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                transactions: vec![],
                gas_used: 21000,
                gas_limit: 8000000,
            })
        }

        async fn send_raw_transaction(&self, _tx_data: &[u8]) -> Result<TxHash, RollupError> {
            Ok([123u8; 32])
        }

        async fn get_transaction_receipt(&self, _tx_hash: TxHash) -> Result<Option<TransactionReceipt>, RollupError> {
            Ok(Some(TransactionReceipt {
                transaction_hash: [123u8; 32],
                block_number: self.current_block,
                block_hash: [42u8; 32],
                transaction_index: 0,
                from: [1u8; 20],
                to: Some([2u8; 20]),
                gas_used: 21000,
                status: TransactionStatus::Success,
                logs: vec![],
            }))
        }

        async fn get_transaction(&self, _tx_hash: TxHash) -> Result<Option<EthereumTransaction>, RollupError> {
            Ok(None)
        }

        async fn estimate_gas(&self, _tx_request: &TransactionRequest) -> Result<u64, RollupError> {
            Ok(21000)
        }

        async fn get_gas_price(&self) -> Result<u64, RollupError> {
            Ok(20_000_000_000) // 20 gwei
        }

        async fn call_contract(&self, _call_data: &ContractCall) -> Result<Vec<u8>, RollupError> {
            Ok(vec![])
        }
    }

    fn create_mock_proof() -> ZkProof {
        ZkProof {
            backend: ZkVMBackend::SP1Local,
            proof_data: vec![1, 2, 3, 4],
            public_inputs: vec![5, 6, 7, 8],
            verification_key_hash: [9u8; 32],
            proof_metadata: ProofMetadata {
                proof_id: "test_proof".to_string(),
                generation_time: 1234567890,
                proof_size: 4,
                security_level: 128,
                circuit_size: 1000000,
            },
        }
    }

    #[tokio::test]
    async fn test_ethrex_client_creation() {
        let config = EthrexClientConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        
        let client = EthrexL1Client::new(config, rpc_client);
        
        // Verify initial state
        let records = client.get_all_commitment_records().await;
        assert!(records.is_empty());
    }

    #[tokio::test]
    async fn test_proof_commitment_submission() {
        let config = EthrexClientConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        let client = EthrexL1Client::new(config, rpc_client);

        let proof = create_mock_proof();
        let state_root_before = [1u8; 32];
        let state_root_after = [2u8; 32];
        let order_book_merkle_root = [3u8; 32];

        let tx_hash = client.submit_proof_commitment(
            1,
            &proof,
            state_root_before,
            state_root_after,
            order_book_merkle_root,
        ).await.unwrap();

        assert_eq!(tx_hash, [123u8; 32]);

        // Verify commitment record was created
        let record = client.get_commitment_record(state_root_after).await;
        assert!(record.is_some());
        
        let record = record.unwrap();
        assert_eq!(record.state_root, state_root_after);
        assert_eq!(record.commitment_tx_hash, tx_hash);
        assert_eq!(record.order_book_merkle_root, order_book_merkle_root);
    }

    #[tokio::test]
    async fn test_transaction_confirmation_update() {
        let config = EthrexClientConfig {
            confirmation_blocks: 3,
            ..Default::default()
        };
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 103 });
        let client = EthrexL1Client::new(config, rpc_client);

        // Submit a proof commitment
        let proof = create_mock_proof();
        let tx_hash = client.submit_proof_commitment(
            1,
            &proof,
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
        ).await.unwrap();

        // Update confirmations
        client.update_transaction_confirmations().await.unwrap();

        // Check that transaction was finalized (current_block=103, tx_block=100, confirmations=3)
        let finalized = client.finalized_transactions.read().await;
        assert!(finalized.contains_key(&tx_hash));
    }

    #[tokio::test]
    async fn test_l1_transaction_hash_retrieval() {
        let config = EthrexClientConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        let client = EthrexL1Client::new(config, rpc_client);

        let proof = create_mock_proof();
        let state_root = [2u8; 32];

        let tx_hash = client.submit_proof_commitment(
            1,
            &proof,
            [1u8; 32],
            state_root,
            [3u8; 32],
        ).await.unwrap();

        // Test requirement 2.7: Get L1 transaction hash for historical state commitment
        let retrieved_tx_hash = client.get_l1_transaction_hash(state_root).await;
        assert_eq!(retrieved_tx_hash, Some(tx_hash));
    }

    #[tokio::test]
    async fn test_reorganization_handling() {
        let config = EthrexClientConfig::default();
        let rpc_client = Arc::new(MockEthereumRpcClient { current_block: 100 });
        let client = EthrexL1Client::new(config, rpc_client);

        // Submit a transaction first
        let proof = create_mock_proof();
        let tx_hash = client.submit_proof_commitment(
            1,
            &proof,
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
        ).await.unwrap();

        // Create a reorg event
        let reorg_event = ReorgEvent {
            reorg_block: 99,
            original_block_hash: [42u8; 32],
            new_block_hash: [43u8; 32],
            affected_transactions: vec![tx_hash],
            detection_time: Instant::now(),
        };

        // Handle the reorganization
        client.handle_reorganization(reorg_event).await.unwrap();

        // Verify reorg was recorded
        let reorgs = client.reorg_handler.detected_reorgs.read().await;
        assert_eq!(reorgs.len(), 1);
        assert_eq!(reorgs[0].reorg_block, 99);
    }
}