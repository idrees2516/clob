//! zkVM abstraction traits
//! 
//! This module defines the core traits that abstract over different zkVM backends,
//! allowing the CLOB system to work with both ZisK and SP1 zkVMs seamlessly.

use crate::zkvm::{ZkVMError, ZkVMConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Core zkVM execution trait
/// 
/// This trait provides a unified interface for executing programs and generating
/// proofs across different zkVM backends (ZisK, SP1, etc.)
#[async_trait::async_trait]
pub trait ZkVMInstance: Send + Sync {
    /// Execute a program with given inputs and generate execution trace
    async fn execute_program(
        &self,
        program: &CompiledProgram,
        inputs: &ExecutionInputs,
    ) -> Result<ExecutionResult, ZkVMError>;

    /// Generate a zero-knowledge proof for the execution
    async fn generate_proof(
        &self,
        execution: &ExecutionResult,
    ) -> Result<ZkProof, ZkVMError>;

    /// Verify a zero-knowledge proof
    async fn verify_proof(
        &self,
        proof: &ZkProof,
        public_inputs: &[u8],
        verification_key: &VerificationKey,
    ) -> Result<bool, ZkVMError>;

    /// Get the backend type
    fn backend_type(&self) -> ZkVMBackend;

    /// Get execution statistics
    fn get_stats(&self) -> ExecutionStats;
}

/// Program compilation trait
/// 
/// Handles compilation of Rust programs to zkVM-specific bytecode
#[async_trait::async_trait]
pub trait ProgramCompiler: Send + Sync {
    /// Compile Rust source code to zkVM bytecode
    async fn compile_program(
        &self,
        source_path: &str,
        optimization_level: u8,
    ) -> Result<CompiledProgram, ZkVMError>;

    /// Compile from pre-built ELF binary
    async fn compile_from_elf(
        &self,
        elf_bytes: &[u8],
    ) -> Result<CompiledProgram, ZkVMError>;

    /// Get supported target architecture
    fn target_architecture(&self) -> TargetArchitecture;
}

/// Witness generation trait
/// 
/// Handles generation of execution witnesses for proof creation
pub trait WitnessGenerator: Send + Sync {
    /// Generate witness data from execution trace
    fn generate_witness(
        &self,
        execution: &ExecutionResult,
    ) -> Result<ExecutionWitness, ZkVMError>;

    /// Validate witness consistency
    fn validate_witness(
        &self,
        witness: &ExecutionWitness,
        public_inputs: &[u8],
    ) -> Result<bool, ZkVMError>;
}

/// zkVM backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZkVMBackend {
    ZisK,
    SP1Local,
    SP1Network,
}

/// Target architecture for compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArchitecture {
    RiscV32,
    RiscV64,
    Custom(u32),
}

/// Compiled program representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledProgram {
    pub backend: ZkVMBackend,
    pub bytecode: Vec<u8>,
    pub metadata: ProgramMetadata,
    pub verification_key: VerificationKey,
}

/// Program metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramMetadata {
    pub program_id: String,
    pub version: String,
    pub compilation_time: u64,
    pub optimization_level: u8,
    pub target_arch: String,
    pub memory_layout: MemoryLayout,
}

/// Memory layout information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLayout {
    pub stack_size: usize,
    pub heap_size: usize,
    pub code_size: usize,
    pub data_size: usize,
}

/// Execution inputs for zkVM programs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionInputs {
    pub public_inputs: Vec<u8>,
    pub private_inputs: Vec<u8>,
    pub initial_state: Option<Vec<u8>>,
    pub execution_config: ExecutionConfig,
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub max_cycles: u64,
    pub memory_limit: usize,
    pub timeout_seconds: u64,
    pub enable_profiling: bool,
    pub custom_params: HashMap<String, String>,
}

/// Execution result containing trace and outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub program_id: String,
    pub execution_id: String,
    pub public_outputs: Vec<u8>,
    pub private_outputs: Vec<u8>,
    pub execution_trace: ExecutionTrace,
    pub final_state: Vec<u8>,
    pub stats: ExecutionStats,
}

/// Execution trace for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub cycles_used: u64,
    pub memory_accesses: Vec<MemoryAccess>,
    pub register_states: Vec<RegisterState>,
    pub instruction_trace: Vec<Instruction>,
    pub syscall_trace: Vec<Syscall>,
}

/// Memory access record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAccess {
    pub cycle: u64,
    pub address: u64,
    pub value: u64,
    pub access_type: MemoryAccessType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAccessType {
    Read,
    Write,
}

/// Register state at specific cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterState {
    pub cycle: u64,
    pub registers: HashMap<u8, u64>,
    pub pc: u64,
}

/// Instruction execution record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instruction {
    pub cycle: u64,
    pub pc: u64,
    pub opcode: u32,
    pub operands: Vec<u64>,
    pub result: Option<u64>,
}

/// System call record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Syscall {
    pub cycle: u64,
    pub syscall_id: u32,
    pub args: Vec<u64>,
    pub return_value: u64,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub total_cycles: u64,
    pub memory_usage: usize,
    pub execution_time_ms: u64,
    pub proof_generation_time_ms: Option<u64>,
    pub verification_time_ms: Option<u64>,
    pub gas_used: u64,
}

/// Zero-knowledge proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkProof {
    pub backend: ZkVMBackend,
    pub proof_data: Vec<u8>,
    pub public_inputs: Vec<u8>,
    pub verification_key_hash: [u8; 32],
    pub proof_metadata: ProofMetadata,
}

/// Proof metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub proof_id: String,
    pub generation_time: u64,
    pub proof_size: usize,
    pub security_level: u8,
    pub circuit_size: u64,
}

/// Verification key for proof verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationKey {
    pub backend: ZkVMBackend,
    pub key_data: Vec<u8>,
    pub key_hash: [u8; 32],
    pub version: String,
}

/// Execution witness for proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionWitness {
    pub execution_id: String,
    pub witness_data: Vec<u8>,
    pub public_inputs: Vec<u8>,
    pub private_inputs: Vec<u8>,
    pub auxiliary_data: HashMap<String, Vec<u8>>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_cycles: 1_000_000,
            memory_limit: 64 * 1024 * 1024, // 64MB
            timeout_seconds: 300,
            enable_profiling: false,
            custom_params: HashMap::new(),
        }
    }
}

impl ExecutionInputs {
    /// Create new execution inputs
    pub fn new(public_inputs: Vec<u8>, private_inputs: Vec<u8>) -> Self {
        Self {
            public_inputs,
            private_inputs,
            initial_state: None,
            execution_config: ExecutionConfig::default(),
        }
    }

    /// Set execution configuration
    pub fn with_config(mut self, config: ExecutionConfig) -> Self {
        self.execution_config = config;
        self
    }

    /// Set initial state
    pub fn with_initial_state(mut self, state: Vec<u8>) -> Self {
        self.initial_state = Some(state);
        self
    }
}

impl ZkProof {
    /// Get proof size in bytes
    pub fn size(&self) -> usize {
        self.proof_data.len()
    }

    /// Verify proof integrity
    pub fn verify_integrity(&self) -> bool {
        // Basic integrity checks
        !self.proof_data.is_empty() && 
        !self.public_inputs.is_empty() &&
        self.verification_key_hash != [0u8; 32]
    }
}