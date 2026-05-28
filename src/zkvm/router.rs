//! zkVM Router Implementation
//! 
//! This module provides automatic zkVM selection based on proof complexity,
//! latency requirements, and performance characteristics.

use crate::zkvm::{
    traits::*,
    ZkVMError, ZkVMConfig, ZkVMBackend,
    ZkVMFactory,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error, instrument};

/// zkVM selection criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkVMSelectionCriteria {
    pub complexity: ProofComplexity,
    pub latency_requirement: std::time::Duration,
    pub proof_size_constraint: Option<usize>,
    pub verification_cost_limit: Option<u64>,
    pub preferred_backend: Option<ZkVMBackend>,
}

/// Proof complexity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofComplexity {
    Simple,      // Basic order validation, matching
    Moderate,    // Risk calculations, batch processing
    Complex,     // Advanced analytics, compliance checks
}

/// Performance profile for zkVM backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub backend: ZkVMBackend,
    pub avg_proof_generation_time: std::time::Duration,
    pub avg_verification_time: std::time::Duration,
    pub avg_proof_size: usize,
    pub supported_complexity: Vec<ProofComplexity>,
    pub max_cycles: u64,
    pub memory_limit: usize,
    pub reliability_score: f64, // 0.0 to 1.0
}

/// zkVM Router for automatic backend selection
pub struct ZkVMRouter {
    backends: Arc<RwLock<HashMap<ZkVMBackend, Box<dyn ZkVMInstance>>>>,
    performance_profiles: Arc<RwLock<HashMap<ZkVMBackend, PerformanceProfile>>>,
    selection_strategy: SelectionStrategy,
    load_balancer: Arc<RwLock<LoadBalancer>>,
}

/// Selection strategy for choosing zkVM backends
#[derive(Debug, Clone, Copy)]
pub enum SelectionStrategy {
    LatencyOptimized,    // Minimize proof generation time
    SizeOptimized,       // Minimize proof size
    ReliabilityOptimized, // Maximize success rate
    Balanced,            // Balance all factors
}

/// Load balancer for distributing work across backends
#[derive(Debug)]
pub struct LoadBalancer {
    backend_loads: HashMap<ZkVMBackend, usize>,
    max_concurrent_proofs: usize,
}

impl ZkVMRouter {
    /// Create new zkVM router
    #[instrument(level = "info")]
    pub async fn new(
        configs: Vec<ZkVMConfig>,
        strategy: SelectionStrategy,
    ) -> Result<Self, ZkVMError> {
        info!("Initializing zkVM router with {} backends", configs.len());

        let mut backends = HashMap::new();
        let mut performance_profiles = HashMap::new();

        // Initialize all configured backends
        for config in configs {
            let backend_type = config.backend;
            
            match ZkVMFactory::create(config.clone()) {
                Ok(backend) => {
                    let profile = Self::create_performance_profile(backend_type, &config);
                    backends.insert(backend_type, backend);
                    performance_profiles.insert(backend_type, profile);
                    info!("Initialized {} backend", backend_type_name(backend_type));
                }
                Err(e) => {
                    warn!("Failed to initialize {} backend: {}", backend_type_name(backend_type), e);
                }
            }
        }

        if backends.is_empty() {
            return Err(ZkVMError::ExecutionFailed("No backends available".to_string()));
        }

        let load_balancer = LoadBalancer {
            backend_loads: backends.keys().map(|&k| (k, 0)).collect(),
            max_concurrent_proofs: 10, // Configurable
        };

        Ok(Self {
            backends: Arc::new(RwLock::new(backends)),
            performance_profiles: Arc::new(RwLock::new(performance_profiles)),
            selection_strategy: strategy,
            load_balancer: Arc::new(RwLock::new(load_balancer)),
        })
    }

    /// Select optimal zkVM backend based on criteria
    #[instrument(level = "debug", skip(self))]
    pub async fn select_optimal_backend(
        &self,
        criteria: &ZkVMSelectionCriteria,
    ) -> Result<ZkVMBackend, ZkVMError> {
        debug!("Selecting optimal backend for complexity: {:?}", criteria.complexity);

        let profiles = self.performance_profiles.read().await;
        let load_balancer = self.load_balancer.read().await;

        // Filter backends that support the required complexity
        let mut candidates: Vec<_> = profiles
            .iter()
            .filter(|(_, profile)| {
                profile.supported_complexity.contains(&criteria.complexity)
            })
            .collect();

        if candidates.is_empty() {
            return Err(ZkVMError::ExecutionFailed(
                "No backend supports required complexity".to_string()
            ));
        }

        // Apply preferred backend if specified
        if let Some(preferred) = criteria.preferred_backend {
            if candidates.iter().any(|(&backend, _)| backend == preferred) {
                debug!("Using preferred backend: {}", backend_type_name(preferred));
                return Ok(preferred);
            }
        }

        // Apply constraints
        candidates.retain(|(_, profile)| {
            let meets_latency = profile.avg_proof_generation_time <= criteria.latency_requirement;
            let meets_size = criteria.proof_size_constraint
                .map_or(true, |limit| profile.avg_proof_size <= limit);
            let meets_cost = criteria.verification_cost_limit
                .map_or(true, |_limit| true); // Simplified cost check

            meets_latency && meets_size && meets_cost
        });

        if candidates.is_empty() {
            return Err(ZkVMError::ExecutionFailed(
                "No backend meets the specified constraints".to_string()
            ));
        }

        // Select based on strategy
        let selected_backend = match self.selection_strategy {
            SelectionStrategy::LatencyOptimized => {
                candidates
                    .iter()
                    .min_by_key(|(_, profile)| profile.avg_proof_generation_time)
                    .map(|(&backend, _)| backend)
                    .unwrap()
            }
            SelectionStrategy::SizeOptimized => {
                candidates
                    .iter()
                    .min_by_key(|(_, profile)| profile.avg_proof_size)
                    .map(|(&backend, _)| backend)
                    .unwrap()
            }
            SelectionStrategy::ReliabilityOptimized => {
                candidates
                    .iter()
                    .max_by(|(_, a), (_, b)| a.reliability_score.partial_cmp(&b.reliability_score).unwrap())
                    .map(|(&backend, _)| backend)
                    .unwrap()
            }
            SelectionStrategy::Balanced => {
                // Score each candidate based on multiple factors
                let best = candidates
                    .iter()
                    .max_by(|(backend_a, profile_a), (backend_b, profile_b)| {
                        let score_a = self.calculate_balanced_score(profile_a, &load_balancer, backend_a);
                        let score_b = self.calculate_balanced_score(profile_b, &load_balancer, backend_b);
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .map(|(&backend, _)| backend)
                    .unwrap();
                best
            }
        };

        debug!("Selected backend: {}", backend_type_name(selected_backend));
        Ok(selected_backend)
    }

    /// Generate proof using optimal backend
    #[instrument(level = "info", skip(self, program, inputs))]
    pub async fn generate_proof_with_optimal_backend(
        &self,
        program: &CompiledProgram,
        inputs: &ExecutionInputs,
        criteria: &ZkVMSelectionCriteria,
    ) -> Result<ZkProof, ZkVMError> {
        let backend_type = self.select_optimal_backend(criteria).await?;
        
        // Update load balancer
        {
            let mut load_balancer = self.load_balancer.write().await;
            load_balancer.increment_load(backend_type);
        }

        let result = {
            let backends = self.backends.read().await;
            let backend = backends.get(&backend_type)
                .ok_or_else(|| ZkVMError::ExecutionFailed("Backend not available".to_string()))?;

            // Execute program
            let execution_result = backend.execute_program(program, inputs).await?;
            
            // Generate proof
            backend.generate_proof(&execution_result).await
        };

        // Update load balancer
        {
            let mut load_balancer = self.load_balancer.write().await;
            load_balancer.decrement_load(backend_type);
        }

        // Update performance metrics based on result
        if let Ok(ref proof) = result {
            self.update_performance_metrics(backend_type, proof).await;
        }

        result
    }

    /// Get available backends
    pub async fn get_available_backends(&self) -> Vec<ZkVMBackend> {
        let backends = self.backends.read().await;
        backends.keys().copied().collect()
    }

    /// Get performance profile for a backend
    pub async fn get_performance_profile(&self, backend: ZkVMBackend) -> Option<PerformanceProfile> {
        let profiles = self.performance_profiles.read().await;
        profiles.get(&backend).cloned()
    }

    /// Update performance metrics after proof generation
    async fn update_performance_metrics(&self, backend: ZkVMBackend, proof: &ZkProof) {
        let mut profiles = self.performance_profiles.write().await;
        if let Some(profile) = profiles.get_mut(&backend) {
            // Update average proof size (simple moving average)
            profile.avg_proof_size = (profile.avg_proof_size + proof.size()) / 2;
            
            // Update reliability score based on successful proof generation
            profile.reliability_score = (profile.reliability_score * 0.9) + 0.1;
        }
    }

    /// Calculate balanced score for backend selection
    fn calculate_balanced_score(
        &self,
        profile: &PerformanceProfile,
        load_balancer: &LoadBalancer,
        backend: &ZkVMBackend,
    ) -> f64 {
        let latency_score = 1.0 / (profile.avg_proof_generation_time.as_millis() as f64 + 1.0);
        let size_score = 1.0 / (profile.avg_proof_size as f64 + 1.0);
        let reliability_score = profile.reliability_score;
        let load_score = 1.0 / (load_balancer.backend_loads.get(backend).unwrap_or(&1) + 1) as f64;

        // Weighted combination
        (latency_score * 0.3) + (size_score * 0.2) + (reliability_score * 0.3) + (load_score * 0.2)
    }

    /// Create performance profile for a backend
    fn create_performance_profile(backend: ZkVMBackend, config: &ZkVMConfig) -> PerformanceProfile {
        match backend {
            ZkVMBackend::ZisK => PerformanceProfile {
                backend,
                avg_proof_generation_time: std::time::Duration::from_millis(100), // Fast
                avg_verification_time: std::time::Duration::from_millis(10),
                avg_proof_size: 2048, // Compact
                supported_complexity: vec![ProofComplexity::Simple, ProofComplexity::Moderate],
                max_cycles: config.max_cycles,
                memory_limit: config.memory_limit,
                reliability_score: 0.95,
            },
            ZkVMBackend::SP1Local => PerformanceProfile {
                backend,
                avg_proof_generation_time: std::time::Duration::from_millis(500), // Slower but more capable
                avg_verification_time: std::time::Duration::from_millis(20),
                avg_proof_size: 4096, // Larger
                supported_complexity: vec![ProofComplexity::Moderate, ProofComplexity::Complex],
                max_cycles: config.max_cycles,
                memory_limit: config.memory_limit,
                reliability_score: 0.90,
            },
            ZkVMBackend::SP1Network => PerformanceProfile {
                backend,
                avg_proof_generation_time: std::time::Duration::from_millis(2000), // Network latency
                avg_verification_time: std::time::Duration::from_millis(15),
                avg_proof_size: 3072,
                supported_complexity: vec![ProofComplexity::Complex],
                max_cycles: config.max_cycles * 10, // Higher capacity
                memory_limit: config.memory_limit * 4,
                reliability_score: 0.85, // Network dependency
            },
        }
    }
}

impl LoadBalancer {
    fn increment_load(&mut self, backend: ZkVMBackend) {
        *self.backend_loads.entry(backend).or_insert(0) += 1;
    }

    fn decrement_load(&mut self, backend: ZkVMBackend) {
        if let Some(load) = self.backend_loads.get_mut(&backend) {
            if *load > 0 {
                *load -= 1;
            }
        }
    }

    fn get_load(&self, backend: ZkVMBackend) -> usize {
        self.backend_loads.get(&backend).copied().unwrap_or(0)
    }
}

impl Default for ZkVMSelectionCriteria {
    fn default() -> Self {
        Self {
            complexity: ProofComplexity::Simple,
            latency_requirement: std::time::Duration::from_millis(1000),
            proof_size_constraint: None,
            verification_cost_limit: None,
            preferred_backend: None,
        }
    }
}

/// Helper function to get backend type name
fn backend_type_name(backend: ZkVMBackend) -> &'static str {
    match backend {
        ZkVMBackend::ZisK => "ZisK",
        ZkVMBackend::SP1Local => "SP1Local",
        ZkVMBackend::SP1Network => "SP1Network",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::ZiskConfig;

    #[tokio::test]
    async fn test_router_initialization() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                max_cycles: 1_000_000,
                memory_limit: 64 * 1024 * 1024,
                timeout_seconds: 300,
                network_config: None,
                zisk_config: Some(ZiskConfig {
                    optimization_level: 2,
                    enable_gpu: false,
                    memory_pool_size: 512 * 1024 * 1024,
                }),
                sp1_config: None,
            },
        ];

        let router = ZkVMRouter::new(configs, SelectionStrategy::Balanced).await;
        assert!(router.is_ok());

        let r = router.unwrap();
        let backends = r.get_available_backends().await;
        assert!(!backends.is_empty());
    }

    #[tokio::test]
    async fn test_backend_selection() {
        let configs = vec![
            ZkVMConfig {
                backend: ZkVMBackend::ZisK,
                max_cycles: 1_000_000,
                memory_limit: 64 * 1024 * 1024,
                timeout_seconds: 300,
                network_config: None,
                zisk_config: Some(ZiskConfig {
                    optimization_level: 2,
                    enable_gpu: false,
                    memory_pool_size: 512 * 1024 * 1024,
                }),
                sp1_config: None,
            },
        ];

        let router = ZkVMRouter::new(configs, SelectionStrategy::LatencyOptimized).await.unwrap();

        let criteria = ZkVMSelectionCriteria {
            complexity: ProofComplexity::Simple,
            latency_requirement: std::time::Duration::from_millis(200),
            proof_size_constraint: Some(4096),
            verification_cost_limit: None,
            preferred_backend: None,
        };

        let selected = router.select_optimal_backend(&criteria).await;
        assert!(selected.is_ok());
        assert_eq!(selected.unwrap(), ZkVMBackend::ZisK);
    }
}