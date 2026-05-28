//! zkVM execution utilities
//! 
//! This module provides common execution utilities and helpers
//! for zkVM program execution across different backends.

use crate::zkvm::{ZkVMError, traits::*};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;

/// Execution context builder
pub struct ExecutionContextBuilder {
    max_cycles: u64,
    memory_limit: usize,
    timeout_seconds: u64,
    enable_profiling: bool,
    custom_params: HashMap<String, String>,
}

impl ExecutionContextBuilder {
    pub fn new() -> Self {
        Self {
            max_cycles: 1_000_000,
            memory_limit: 64 * 1024 * 1024,
            timeout_seconds: 300,
            enable_profiling: false,
            custom_params: HashMap::new(),
        }
    }

    pub fn max_cycles(mut self, cycles: u64) -> Self {
        self.max_cycles = cycles;
        self
    }

    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    pub fn timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    pub fn enable_profiling(mut self) -> Self {
        self.enable_profiling = true;
        self
    }

    pub fn custom_param(mut self, key: String, value: String) -> Self {
        self.custom_params.insert(key, value);
        self
    }

    pub fn build(self) -> ExecutionConfig {
        ExecutionConfig {
            max_cycles: self.max_cycles,
            memory_limit: self.memory_limit,
            timeout_seconds: self.timeout_seconds,
            enable_profiling: self.enable_profiling,
            custom_params: self.custom_params,
        }
    }
}

impl Default for ExecutionContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate unique execution ID
pub fn generate_execution_id() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random = rand::thread_rng().gen::<u32>();
    format!("exec_{}_{:x}", timestamp, random)
}

/// Validate execution inputs
pub fn validate_execution_inputs(inputs: &ExecutionInputs) -> Result<(), ZkVMError> {
    if inputs.public_inputs.is_empty() {
        return Err(ZkVMError::ExecutionFailed(
            "Public inputs cannot be empty".to_string()
        ));
    }

    if inputs.execution_config.max_cycles == 0 {
        return Err(ZkVMError::ExecutionFailed(
            "Max cycles must be greater than 0".to_string()
        ));
    }

    if inputs.execution_config.memory_limit == 0 {
        return Err(ZkVMError::ExecutionFailed(
            "Memory limit must be greater than 0".to_string()
        ));
    }

    Ok(())
}

/// Calculate execution statistics
pub fn calculate_execution_stats(
    start_time: std::time::Instant,
    cycles_used: u64,
    memory_usage: usize,
) -> ExecutionStats {
    ExecutionStats {
        total_cycles: cycles_used,
        memory_usage,
        execution_time_ms: start_time.elapsed().as_millis() as u64,
        proof_generation_time_ms: None,
        verification_time_ms: None,
        gas_used: cycles_used * 2, // Simple gas model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_context_builder() {
        let config = ExecutionContextBuilder::new()
            .max_cycles(2_000_000)
            .memory_limit(128 * 1024 * 1024)
            .timeout(600)
            .enable_profiling()
            .custom_param("test_param".to_string(), "test_value".to_string())
            .build();

        assert_eq!(config.max_cycles, 2_000_000);
        assert_eq!(config.memory_limit, 128 * 1024 * 1024);
        assert_eq!(config.timeout_seconds, 600);
        assert!(config.enable_profiling);
        assert_eq!(config.custom_params.get("test_param"), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_generate_execution_id() {
        let id1 = generate_execution_id();
        let id2 = generate_execution_id();
        
        assert_ne!(id1, id2);
        assert!(id1.starts_with("exec_"));
        assert!(id2.starts_with("exec_"));
    }

    #[test]
    fn test_validate_execution_inputs() {
        let valid_inputs = ExecutionInputs::new(vec![1, 2, 3], vec![4, 5, 6]);
        assert!(validate_execution_inputs(&valid_inputs).is_ok());

        let invalid_inputs = ExecutionInputs::new(vec![], vec![4, 5, 6]);
        assert!(validate_execution_inputs(&invalid_inputs).is_err());
    }
}