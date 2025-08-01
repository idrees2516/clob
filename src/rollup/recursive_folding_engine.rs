use crate::rollup::recursive_folding::{FoldingScheme, FoldableProof, FoldingResult, FoldOperation};
use crate::rollup::lattice_fold_plus::{LatticeParams, SecurityLevel};
use crate::error::LatticeFoldError;
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

pub type Result<T> = std::result::Result<T, LatticeFoldError>;

/// A recursive folding engine that can handle multi-level proof aggregation
#[derive(Clone)]
pub struct RecursiveFoldingEngine {
    /// The base folding scheme
    pub folding_scheme: Arc<RwLock<FoldingScheme>>,
    /// Configuration parameters
    pub config: RecursiveFoldingConfig,
    /// Cache for memoization of folding operations
    pub fold_cache: Arc<Mutex<HashMap<String, FoldingResult>>>,
    /// Performance metrics
    pub metrics: Arc<Mutex<FoldingMetrics>>,
}

/// Configuration for the recursive folding engine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursiveFoldingConfig {
    /// Maximum number of proofs to fold in a single operation
    pub max_fold_arity: usize,
    /// Maximum recursion depth
    pub max_recursion_depth: usize,
    /// Whether to enable memoization
    pub enable_memoization: bool,
    /// Whether to enable parallel folding
    pub enable_parallel: bool,
    /// Minimum compression ratio to continue folding
    pub min_compression_ratio: f64,
    /// Cache size limit (number of entries)
    pub cache_size_limit: usize,
}

impl Default for RecursiveFoldingConfig {
    fn default() -> Self {
        Self {
            max_fold_arity: 8,
            max_recursion_depth: 10,
            enable_memoization: true,
            enable_parallel: true,
            min_compression_ratio: 1.1,
            cache_size_limit: 1000,
        }
    }
}

/// Performance metrics for folding operations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FoldingMetrics {
    /// Total number of folding operations performed
    pub total_folds: u64,
    /// Total number of proofs folded
    pub total_proofs_folded: u64,
    /// Average compression ratio achieved
    pub average_compression_ratio: f64,
    /// Average verification complexity
    pub average_verification_complexity: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Total time spent folding (milliseconds)
    pub total_folding_time_ms: u64,
    /// Number of cache hits
    pub cache_hits: u64,
    /// Number of cache misses
    pub cache_misses: u64,
}

/// A fold tree node representing the structure of recursive folding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FoldTreeNode {
    /// A leaf node containing an original proof
    Leaf {
        proof: FoldableProof,
        index: usize,
    },
    /// An internal node containing a folded proof and its children
    Internal {
        folded_proof: FoldableProof,
        fold_operation: FoldOperation,
        children: Vec<FoldTreeNode>,
        compression_ratio: f64,
    },
}

/// The result of recursive folding with tree structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecursiveFoldingResult {
    /// The final folded proof
    pub final_proof: FoldableProof,
    /// The fold tree showing the structure
    pub fold_tree: FoldTreeNode,
    /// Overall compression ratio
    pub compression_ratio: f64,
    /// Total verification complexity
    pub verification_complexity: usize,
    /// Folding depth achieved
    pub folding_depth: usize,
    /// Performance metrics for this operation
    pub metrics: FoldingMetrics,
}

impl RecursiveFoldingEngine {
    /// Create a new recursive folding engine
    pub async fn new<R: RngCore + CryptoRng>(
        params: LatticeParams,
        config: RecursiveFoldingConfig,
        rng: &mut R,
    ) -> Result<Self> {
        let folding_scheme = FoldingScheme::new(params, rng)?;
        
        Ok(Self {
            folding_scheme: Arc::new(RwLock::new(folding_scheme)),
            config,
            fold_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(FoldingMetrics::default())),
        })
    }
    
    /// Recursively fold a set of proofs with optimal tree structure
    pub async fn fold_recursive<R: RngCore + CryptoRng>(
        &self,
        proofs: Vec<FoldableProof>,
        rng: &mut R,
    ) -> Result<RecursiveFoldingResult> {
        let start_time = std::time::Instant::now();
        
        if proofs.is_empty() {
            return Err(LatticeFoldError::InvalidInput(
                "Cannot fold empty set of proofs".to_string(),
            ));
        }
        
        if proofs.len() == 1 {
            return Ok(RecursiveFoldingResult {
                final_proof: proofs[0].clone(),
                fold_tree: FoldTreeNode::Leaf {
                    proof: proofs[0].clone(),
                    index: 0,
                },
                compression_ratio: 1.0,
                verification_complexity: 1,
                folding_depth: 0,
                metrics: FoldingMetrics::default(),
            });
        }
        
        // Build the optimal fold tree
        let fold_tree = self.build_fold_tree(proofs, 0, rng).await?;
        
        // Extract the final proof and metrics
        let (final_proof, compression_ratio, verification_complexity) = 
            self.extract_tree_metrics(&fold_tree);
        
        let folding_depth = self.compute_tree_depth(&fold_tree);
        
        // Update global metrics
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        self.update_metrics(compression_ratio, verification_complexity as f64, elapsed_ms).await;
        
        let metrics = self.get_current_metrics().await;
        
        Ok(RecursiveFoldingResult {
            final_proof,
            fold_tree,
            compression_ratio,
            verification_complexity,
            folding_depth,
            metrics,
        })
    }
    
    /// Build an optimal fold tree using dynamic programming
    async fn build_fold_tree<R: RngCore + CryptoRng>(
        &self,
        proofs: Vec<FoldableProof>,
        depth: usize,
        rng: &mut R,
    ) -> Result<FoldTreeNode> {
        if depth >= self.config.max_recursion_depth {
            return Err(LatticeFoldError::InvalidInput(
                "Maximum recursion depth exceeded".to_string(),
            ));
        }
        
        if proofs.len() == 1 {
            return Ok(FoldTreeNode::Leaf {
                proof: proofs[0].clone(),
                index: 0,
            });
        }
        
        // Check cache if memoization is enabled
        if self.config.enable_memoization {
            let cache_key = self.compute_cache_key(&proofs);
            if let Some(cached_result) = self.get_from_cache(&cache_key).await {
                self.record_cache_hit().await;
                return Ok(cached_result);
            }
            self.record_cache_miss().await;
        }
        
        // Determine optimal folding strategy
        let fold_strategy = self.determine_fold_strategy(&proofs);
        
        match fold_strategy {
            FoldStrategy::Direct => {
                // Fold all proofs directly
                self.fold_direct(proofs, depth, rng).await
            },
            FoldStrategy::Hierarchical(partitions) => {
                // Fold in hierarchical manner
                self.fold_hierarchical(proofs, partitions, depth, rng).await
            },
            FoldStrategy::Balanced => {
                // Fold in balanced binary tree manner
                self.fold_balanced(proofs, depth, rng).await
            },
        }
    }
    
    /// Fold proofs directly in a single operation
    async fn fold_direct<R: RngCore + CryptoRng>(
        &self,
        proofs: Vec<FoldableProof>,
        depth: usize,
        rng: &mut R,
    ) -> Result<FoldTreeNode> {
        let mut scheme = self.folding_scheme.write().await;
        let folding_result = scheme.fold_proofs(&proofs, rng)?;
        
        let children: Vec<FoldTreeNode> = proofs.into_iter().enumerate().map(|(i, proof)| {
            FoldTreeNode::Leaf { proof, index: i }
        }).collect();
        
        Ok(FoldTreeNode::Internal {
            folded_proof: folding_result.folded_proof,
            fold_operation: folding_result.fold_operation,
            children,
            compression_ratio: folding_result.compression_ratio,
        })
    }
    
    /// Fold proofs in a hierarchical manner
    async fn fold_hierarchical<R: RngCore + CryptoRng>(
        &self,
        proofs: Vec<FoldableProof>,
        partitions: Vec<usize>,
        depth: usize,
        rng: &mut R,
    ) -> Result<FoldTreeNode> {
        let mut children = Vec::new();
        let mut current_idx = 0;
        
        // First level: fold within partitions
        for &partition_size in &partitions {
            let partition_proofs = proofs[current_idx..current_idx + partition_size].to_vec();
            let child_tree = self.build_fold_tree(partition_proofs, depth + 1, rng).await?;
            children.push(child_tree);
            current_idx += partition_size;
        }
        
        // Second level: fold the partition results
        let partition_proofs: Vec<FoldableProof> = children.iter().map(|child| {
            match child {
                FoldTreeNode::Leaf { proof, .. } => proof.clone(),
                FoldTreeNode::Internal { folded_proof, .. } => folded_proof.clone(),
            }
        }).collect();
        
        let mut scheme = self.folding_scheme.write().await;
        let folding_result = scheme.fold_proofs(&partition_proofs, rng)?;
        
        Ok(FoldTreeNode::Internal {
            folded_proof: folding_result.folded_proof,
            fold_operation: folding_result.fold_operation,
            children,
            compression_ratio: folding_result.compression_ratio,
        })
    }
    
    /// Fold proofs in a balanced binary tree manner
    async fn fold_balanced<R: RngCore + CryptoRng>(
        &self,
        proofs: Vec<FoldableProof>,
        depth: usize,
        rng: &mut R,
    ) -> Result<FoldTreeNode> {
        if proofs.len() <= 2 {
            return self.fold_direct(proofs, depth, rng).await;
        }
        
        let mid = proofs.len() / 2;
        let left_proofs = proofs[..mid].to_vec();
        let right_proofs = proofs[mid..].to_vec();
        
        // Recursively fold left and right halves
        let left_tree = self.build_fold_tree(left_proofs, depth + 1, rng).await?;
        let right_tree = self.build_fold_tree(right_proofs, depth + 1, rng).await?;
        
        // Extract proofs from trees
        let left_proof = match &left_tree {
            FoldTreeNode::Leaf { proof, .. } => proof.clone(),
            FoldTreeNode::Internal { folded_proof, .. } => folded_proof.clone(),
        };
        
        let right_proof = match &right_tree {
            FoldTreeNode::Leaf { proof, .. } => proof.clone(),
            FoldTreeNode::Internal { folded_proof, .. } => folded_proof.clone(),
        };
        
        // Fold the two results
        let mut scheme = self.folding_scheme.write().await;
        let folding_result = scheme.fold_proofs(&[left_proof, right_proof], rng)?;
        
        Ok(FoldTreeNode::Internal {
            folded_proof: folding_result.folded_proof,
            fold_operation: folding_result.fold_operation,
            children: vec![left_tree, right_tree],
            compression_ratio: folding_result.compression_ratio,
        })
    }
    
    /// Determine the optimal folding strategy for a set of proofs
    fn determine_fold_strategy(&self, proofs: &[FoldableProof]) -> FoldStrategy {
        let num_proofs = proofs.len();
        
        if num_proofs <= self.config.max_fold_arity {
            FoldStrategy::Direct
        } else if num_proofs <= self.config.max_fold_arity * self.config.max_fold_arity {
            // Hierarchical folding with optimal partitioning
            let partition_size = (num_proofs as f64).sqrt().ceil() as usize;
            let num_partitions = (num_proofs + partition_size - 1) / partition_size;
            
            let mut partitions = vec![partition_size; num_partitions - 1];
            let last_partition_size = num_proofs - (num_partitions - 1) * partition_size;
            if last_partition_size > 0 {
                partitions.push(last_partition_size);
            }
            
            FoldStrategy::Hierarchical(partitions)
        } else {
            FoldStrategy::Balanced
        }
    }
    
    /// Compute a cache key for a set of proofs
    fn compute_cache_key(&self, proofs: &[FoldableProof]) -> String {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        for proof in proofs {
            hasher.update(&proof.commitment.value.to_bytes());
            hasher.update(&proof.proof_point.to_bytes());
        }
        
        hex::encode(hasher.finalize().as_bytes())
    }
    
    /// Get a result from cache
    async fn get_from_cache(&self, key: &str) -> Option<FoldTreeNode> {
        // For simplicity, we'll return None here
        // In a full implementation, this would deserialize from cache
        None
    }
    
    /// Record a cache hit
    async fn record_cache_hit(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.cache_hits += 1;
    }
    
    /// Record a cache miss
    async fn record_cache_miss(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.cache_misses += 1;
    }
    
    /// Extract metrics from a fold tree
    fn extract_tree_metrics(&self, tree: &FoldTreeNode) -> (FoldableProof, f64, usize) {
        match tree {
            FoldTreeNode::Leaf { proof, .. } => {
                (proof.clone(), 1.0, 1)
            },
            FoldTreeNode::Internal { folded_proof, compression_ratio, children, .. } => {
                let verification_complexity = children.iter().map(|child| {
                    let (_, _, complexity) = self.extract_tree_metrics(child);
                    complexity
                }).sum::<usize>() + 1;
                
                (folded_proof.clone(), *compression_ratio, verification_complexity)
            },
        }
    }
    
    /// Compute the depth of a fold tree
    fn compute_tree_depth(&self, tree: &FoldTreeNode) -> usize {
        match tree {
            FoldTreeNode::Leaf { .. } => 0,
            FoldTreeNode::Internal { children, .. } => {
                1 + children.iter().map(|child| self.compute_tree_depth(child)).max().unwrap_or(0)
            },
        }
    }
    
    /// Update global metrics
    async fn update_metrics(&self, compression_ratio: f64, verification_complexity: f64, elapsed_ms: u64) {
        let mut metrics = self.metrics.lock().unwrap();
        
        metrics.total_folds += 1;
        metrics.total_folding_time_ms += elapsed_ms;
        
        // Update running averages
        let n = metrics.total_folds as f64;
        metrics.average_compression_ratio = 
            (metrics.average_compression_ratio * (n - 1.0) + compression_ratio) / n;
        metrics.average_verification_complexity = 
            (metrics.average_verification_complexity * (n - 1.0) + verification_complexity) / n;
        
        // Update cache hit rate
        let total_cache_requests = metrics.cache_hits + metrics.cache_misses;
        if total_cache_requests > 0 {
            metrics.cache_hit_rate = metrics.cache_hits as f64 / total_cache_requests as f64;
        }
    }
    
    /// Get current metrics
    async fn get_current_metrics(&self) -> FoldingMetrics {
        self.metrics.lock().unwrap().clone()
    }
    
    /// Verify a recursive folding result
    pub async fn verify_recursive_folding(
        &self,
        result: &RecursiveFoldingResult,
        original_proofs: &[FoldableProof],
    ) -> Result<bool> {
        // Verify the tree structure is consistent
        if !self.verify_tree_structure(&result.fold_tree, original_proofs) {
            return Ok(false);
        }
        
        // Verify each folding operation in the tree
        self.verify_tree_operations(&result.fold_tree).await
    }
    
    /// Verify the structure of a fold tree
    fn verify_tree_structure(&self, tree: &FoldTreeNode, original_proofs: &[FoldableProof]) -> bool {
        let leaf_proofs = self.collect_leaf_proofs(tree);
        
        // Check that all original proofs are represented
        if leaf_proofs.len() != original_proofs.len() {
            return false;
        }
        
        // Check that leaf proofs match original proofs (order may differ)
        for original_proof in original_proofs {
            if !leaf_proofs.iter().any(|leaf_proof| {
                leaf_proof.commitment.value == original_proof.commitment.value &&
                leaf_proof.proof_point == original_proof.proof_point
            }) {
                return false;
            }
        }
        
        true
    }
    
    /// Collect all leaf proofs from a tree
    fn collect_leaf_proofs(&self, tree: &FoldTreeNode) -> Vec<FoldableProof> {
        match tree {
            FoldTreeNode::Leaf { proof, .. } => vec![proof.clone()],
            FoldTreeNode::Internal { children, .. } => {
                children.iter().flat_map(|child| self.collect_leaf_proofs(child)).collect()
            },
        }
    }
    
    /// Verify all folding operations in a tree
    async fn verify_tree_operations(&self, tree: &FoldTreeNode) -> Result<bool> {
        match tree {
            FoldTreeNode::Leaf { .. } => Ok(true),
            FoldTreeNode::Internal { folded_proof, fold_operation, children, .. } => {
                // Verify child operations first
                for child in children {
                    if !self.verify_tree_operations(child).await? {
                        return Ok(false);
                    }
                }
                
                // Extract child proofs
                let child_proofs: Vec<FoldableProof> = children.iter().map(|child| {
                    match child {
                        FoldTreeNode::Leaf { proof, .. } => proof.clone(),
                        FoldTreeNode::Internal { folded_proof, .. } => folded_proof.clone(),
                    }
                }).collect();
                
                // Verify this folding operation
                let scheme = self.folding_scheme.read().await;
                scheme.verify_folded_proof(folded_proof, &child_proofs, fold_operation)
            },
        }
    }
    
    /// Clear the folding cache
    pub async fn clear_cache(&self) {
        let mut cache = self.fold_cache.lock().unwrap();
        cache.clear();
    }
    
    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.fold_cache.lock().unwrap();
        let metrics = self.metrics.lock().unwrap();
        (cache.len(), self.config.cache_size_limit)
    }
}

/// Strategy for folding proofs
#[derive(Clone, Debug)]
enum FoldStrategy {
    /// Fold all proofs in a single operation
    Direct,
    /// Fold hierarchically with given partition sizes
    Hierarchical(Vec<usize>),
    /// Fold in a balanced binary tree
    Balanced,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[tokio::test]
    async fn test_recursive_folding_engine_creation() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let config = RecursiveFoldingConfig::default();
        
        let engine = RecursiveFoldingEngine::new(params, config, &mut rng).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_single_proof_recursive_folding() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let config = RecursiveFoldingConfig::default();
        let engine = RecursiveFoldingEngine::new(params.clone(), config, &mut rng).await.unwrap();
        
        // Create a single proof
        let scheme = engine.folding_scheme.read().await;
        let proof = FoldableProof::new(
            b"test data",
            "test",
            &scheme.commitment_scheme,
            &params,
            &mut rng,
        ).unwrap();
        drop(scheme);
        
        let result = engine.fold_recursive(vec![proof.clone()], &mut rng).await.unwrap();
        
        assert_eq!(result.compression_ratio, 1.0);
        assert_eq!(result.verification_complexity, 1);
        assert_eq!(result.folding_depth, 0);
        
        // Verify the result
        let is_valid = engine.verify_recursive_folding(&result, &[proof]).await.unwrap();
        assert!(is_valid);
    }
    
    #[tokio::test]
    async fn test_multiple_proof_recursive_folding() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let config = RecursiveFoldingConfig::default();
        let engine = RecursiveFoldingEngine::new(params.clone(), config, &mut rng).await.unwrap();
        
        // Create multiple proofs
        let mut proofs = Vec::new();
        let scheme = engine.folding_scheme.read().await;
        for i in 0..10 {
            let data = format!("test data {}", i);
            let proof = FoldableProof::new(
                data.as_bytes(),
                "test",
                &scheme.commitment_scheme,
                &params,
                &mut rng,
            ).unwrap();
            proofs.push(proof);
        }
        drop(scheme);
        
        let original_proofs = proofs.clone();
        let result = engine.fold_recursive(proofs, &mut rng).await.unwrap();
        
        assert!(result.compression_ratio > 1.0);
        assert!(result.verification_complexity > 1);
        assert!(result.folding_depth > 0);
        
        // Verify the result
        let is_valid = engine.verify_recursive_folding(&result, &original_proofs).await.unwrap();
        assert!(is_valid);
    }
    
    #[tokio::test]
    async fn test_fold_strategy_determination() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let config = RecursiveFoldingConfig {
            max_fold_arity: 4,
            ..Default::default()
        };
        let engine = RecursiveFoldingEngine::new(params.clone(), config, &mut rng).await.unwrap();
        
        // Create proofs for testing different strategies
        let scheme = engine.folding_scheme.read().await;
        
        // Test direct strategy (small number of proofs)
        let small_proofs: Vec<FoldableProof> = (0..3).map(|i| {
            FoldableProof::new(
                format!("data {}", i).as_bytes(),
                "test",
                &scheme.commitment_scheme,
                &params,
                &mut rng,
            ).unwrap()
        }).collect();
        
        let strategy = engine.determine_fold_strategy(&small_proofs);
        matches!(strategy, FoldStrategy::Direct);
        
        // Test hierarchical strategy (medium number of proofs)
        let medium_proofs: Vec<FoldableProof> = (0..10).map(|i| {
            FoldableProof::new(
                format!("data {}", i).as_bytes(),
                "test",
                &scheme.commitment_scheme,
                &params,
                &mut rng,
            ).unwrap()
        }).collect();
        
        let strategy = engine.determine_fold_strategy(&medium_proofs);
        matches!(strategy, FoldStrategy::Hierarchical(_));
    }
    
    #[tokio::test]
    async fn test_metrics_tracking() {
        let mut rng = thread_rng();
        let params = LatticeParams::new(SecurityLevel::Medium);
        let config = RecursiveFoldingConfig::default();
        let engine = RecursiveFoldingEngine::new(params.clone(), config, &mut rng).await.unwrap();
        
        // Perform multiple folding operations
        let scheme = engine.folding_scheme.read().await;
        for _ in 0..3 {
            let proofs: Vec<FoldableProof> = (0..5).map(|i| {
                FoldableProof::new(
                    format!("data {}", i).as_bytes(),
                    "test",
                    &scheme.commitment_scheme,
                    &params,
                    &mut rng,
                ).unwrap()
            }).collect();
            
            drop(scheme);
            let _result = engine.fold_recursive(proofs, &mut rng).await.unwrap();
            let scheme = engine.folding_scheme.read().await;
        }
        drop(scheme);
        
        let metrics = engine.get_current_metrics().await;
        assert_eq!(metrics.total_folds, 3);
        assert!(metrics.average_compression_ratio > 1.0);
        assert!(metrics.total_folding_time_ms > 0);
    }
}