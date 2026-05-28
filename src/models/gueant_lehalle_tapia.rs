//! Guéant-Lehalle-Tapia Multi-Asset Market Making Framework
//!
//! This module implements the sophisticated multi-asset market making model
//! from Guéant, Lehalle, and Tapia, featuring:
//! - Multi-dimensional HJB solver for portfolio optimization
//! - Dynamic correlation estimation with regime detection
//! - Cross-asset inventory management with portfolio constraints
//! - Arbitrage detection and execution

use crate::math::fixed_point::{FixedPoint, DeterministicRng};
use crate::math::optimization::{OptimizationError, StateGrid, BoundaryConditions, NumericalScheme};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MultiAssetError {
    #[error("Optimization error: {0}")]
    OptimizationError(#[from] OptimizationError),
    #[error("Matrix operation failed: {0}")]
    MatrixError(String),
    #[error("Invalid correlation matrix: {0}")]
    InvalidCorrelation(String),
    #[error("Portfolio constraint violation: {0}")]
    ConstraintViolation(String),
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

/// Asset identifier
pub type AssetId = u32;

/// Multi-asset model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAssetModelParameters {
    /// Risk aversion parameter γ
    pub risk_aversion: FixedPoint,
    /// Volatilities for each asset σᵢ
    pub volatilities: HashMap<AssetId, FixedPoint>,
    /// Adverse selection costs for each asset κᵢ
    pub adverse_selection_costs: HashMap<AssetId, FixedPoint>,
    /// Inventory penalties for each asset γᵢ
    pub inventory_penalties: HashMap<AssetId, FixedPoint>,
    /// Cross-asset penalty coefficient
    pub cross_asset_penalty: FixedPoint,
    /// Default correlation for missing pairs
    pub default_correlation: FixedPoint,
    /// Terminal time T
    pub terminal_time: FixedPoint,
    /// Maximum inventory limits
    pub max_inventories: HashMap<AssetId, FixedPoint>,
    /// Market impact parameters
    pub market_impact_params: HashMap<AssetId, MarketImpactParams>,
}

/// Market impact parameters for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactParams {
    /// Temporary impact coefficient η
    pub temporary_impact: FixedPoint,
    /// Permanent impact coefficient λ
    pub permanent_impact: FixedPoint,
    /// Impact exponent α
    pub impact_exponent: FixedPoint,
}

impl Default for MultiAssetModelParameters {
    fn default() -> Self {
        Self {
            risk_aversion: FixedPoint::from_float(1.0),
            volatilities: HashMap::new(),
            adverse_selection_costs: HashMap::new(),
            inventory_penalties: HashMap::new(),
            cross_asset_penalty: FixedPoint::from_float(0.001),
            default_correlation: FixedPoint::from_float(0.1),
            terminal_time: FixedPoint::from_float(1.0),
            max_inventories: HashMap::new(),
            market_impact_params: HashMap::new(),
        }
    }
}

impl MultiAssetModelParameters {
    /// Add asset with default parameters
    pub fn add_asset(&mut self, asset_id: AssetId) -> &mut Self {
        self.volatilities.insert(asset_id, FixedPoint::from_float(0.2));
        self.adverse_selection_costs.insert(asset_id, FixedPoint::from_float(0.001));
        self.inventory_penalties.insert(asset_id, FixedPoint::from_float(0.01));
        self.max_inventories.insert(asset_id, FixedPoint::from_float(1000.0));
        self.market_impact_params.insert(asset_id, MarketImpactParams {
            temporary_impact: FixedPoint::from_float(0.1),
            permanent_impact: FixedPoint::from_float(0.01),
            impact_exponent: FixedPoint::from_float(0.5),
        });
        self
    }
    
    /// Set volatility for an asset
    pub fn set_volatility(&mut self, asset_id: AssetId, volatility: FixedPoint) -> &mut Self {
        self.volatilities.insert(asset_id, volatility);
        self
    }
    
    /// Set adverse selection cost for an asset
    pub fn set_adverse_selection_cost(&mut self, asset_id: AssetId, cost: FixedPoint) -> &mut Self {
        self.adverse_selection_costs.insert(asset_id, cost);
        self
    }
    
    /// Set inventory penalty for an asset
    pub fn set_inventory_penalty(&mut self, asset_id: AssetId, penalty: FixedPoint) -> &mut Self {
        self.inventory_penalties.insert(asset_id, penalty);
        self
    }
}

/// Multi-dimensional state grid for portfolio optimization
#[derive(Debug, Clone)]
pub struct MultiAssetStateGrid {
    /// Inventory grids for each asset: q_i ∈ [-Q_max,i, Q_max,i]
    pub inventory_grids: HashMap<AssetId, Vec<FixedPoint>>,
    /// Time grid: t ∈ [0, T]
    pub time_grid: Vec<FixedPoint>,
    /// Price grids for each asset: S_i ∈ [S_min,i, S_max,i]
    pub price_grids: HashMap<AssetId, Vec<FixedPoint>>,
    /// Asset ordering for consistent indexing
    pub asset_order: Vec<AssetId>,
}

impl MultiAssetStateGrid {
    /// Create a new multi-asset state grid
    pub fn new(
        assets: Vec<AssetId>,
        inventory_ranges: HashMap<AssetId, (FixedPoint, FixedPoint)>,
        inventory_steps: HashMap<AssetId, usize>,
        time_range: (FixedPoint, FixedPoint),
        time_steps: usize,
        price_ranges: HashMap<AssetId, (FixedPoint, FixedPoint)>,
        price_steps: HashMap<AssetId, usize>,
    ) -> Result<Self, MultiAssetError> {
        let mut inventory_grids = HashMap::new();
        let mut price_grids = HashMap::new();
        
        for &asset in &assets {
            let inv_range = inventory_ranges.get(&asset)
                .ok_or_else(|| MultiAssetError::MatrixError(format!("Missing inventory range for asset {}", asset)))?;
            let inv_steps = inventory_steps.get(&asset)
                .ok_or_else(|| MultiAssetError::MatrixError(format!("Missing inventory steps for asset {}", asset)))?;
            
            let price_range = price_ranges.get(&asset)
                .ok_or_else(|| MultiAssetError::MatrixError(format!("Missing price range for asset {}", asset)))?;
            let price_step = price_steps.get(&asset)
                .ok_or_else(|| MultiAssetError::MatrixError(format!("Missing price steps for asset {}", asset)))?;
            
            inventory_grids.insert(asset, Self::linspace(inv_range.0, inv_range.1, *inv_steps));
            price_grids.insert(asset, Self::linspace(price_range.0, price_range.1, *price_step));
        }
        
        let time_grid = Self::linspace(time_range.0, time_range.1, time_steps);
        
        Ok(Self {
            inventory_grids,
            time_grid,
            price_grids,
            asset_order: assets,
        })
    }
    
    /// Create linearly spaced points
    fn linspace(start: FixedPoint, end: FixedPoint, num: usize) -> Vec<FixedPoint> {
        if num == 0 {
            return Vec::new();
        }
        if num == 1 {
            return vec![start];
        }
        
        let step = (end - start) / FixedPoint::from_int((num - 1) as i32);
        (0..num)
            .map(|i| start + step * FixedPoint::from_int(i as i32))
            .collect()
    }
    
    /// Get total number of grid points
    pub fn total_points(&self) -> usize {
        let mut total = self.time_grid.len();
        
        for &asset in &self.asset_order {
            if let (Some(inv_grid), Some(price_grid)) = 
                (self.inventory_grids.get(&asset), self.price_grids.get(&asset)) {
                total *= inv_grid.len() * price_grid.len();
            }
        }
        
        total
    }
    
    /// Convert multi-dimensional indices to flat index
    pub fn to_flat_index(&self, indices: &HashMap<AssetId, (usize, usize)>, time_idx: usize) -> usize {
        let mut flat_index = time_idx;
        let mut multiplier = self.time_grid.len();
        
        for &asset in &self.asset_order {
            if let Some(&(inv_idx, price_idx)) = indices.get(&asset) {
                if let (Some(inv_grid), Some(price_grid)) = 
                    (self.inventory_grids.get(&asset), self.price_grids.get(&asset)) {
                    flat_index += multiplier * (inv_idx * price_grid.len() + price_idx);
                    multiplier *= inv_grid.len() * price_grid.len();
                }
            }
        }
        
        flat_index
    }
}

/// Correlation matrix with validation and regularization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationMatrix {
    /// Correlation coefficients ρ_ij
    pub matrix: Vec<Vec<FixedPoint>>,
    /// Eigenvalues for stability analysis
    pub eigenvalues: Vec<FixedPoint>,
    /// Condition number κ(Σ) = λ_max/λ_min
    pub condition_number: FixedPoint,
    /// Asset ordering
    pub assets: Vec<AssetId>,
    /// Last update timestamp
    pub last_update: u64,
}

impl CorrelationMatrix {
    /// Create a new correlation matrix
    pub fn new(assets: Vec<AssetId>) -> Self {
        let n = assets.len();
        let mut matrix = vec![vec![FixedPoint::zero(); n]; n];
        
        // Initialize as identity matrix
        for i in 0..n {
            matrix[i][i] = FixedPoint::one();
        }
        
        Self {
            matrix,
            eigenvalues: vec![FixedPoint::one(); n],
            condition_number: FixedPoint::one(),
            assets,
            last_update: 0,
        }
    }
    
    /// Validate that the matrix is a valid correlation matrix
    pub fn validate(&self) -> Result<(), MultiAssetError> {
        let n = self.matrix.len();
        
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                if (self.matrix[i][j] - self.matrix[j][i]).abs() > FixedPoint::from_float(1e-10) {
                    return Err(MultiAssetError::InvalidCorrelation(
                        "Matrix is not symmetric".to_string()
                    ));
                }
            }
        }
        
        // Check diagonal elements are 1
        for i in 0..n {
            if (self.matrix[i][i] - FixedPoint::one()).abs() > FixedPoint::from_float(1e-10) {
                return Err(MultiAssetError::InvalidCorrelation(
                    "Diagonal elements must be 1".to_string()
                ));
            }
        }
        
        // Check off-diagonal elements are in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let corr = self.matrix[i][j];
                    if corr < FixedPoint::from_float(-1.0) || corr > FixedPoint::one() {
                        return Err(MultiAssetError::InvalidCorrelation(
                            "Correlation coefficients must be in [-1, 1]".to_string()
                        ));
                    }
                }
            }
        }
        
        // Check positive semi-definiteness (simplified check)
        if self.eigenvalues.iter().any(|&λ| λ < FixedPoint::zero()) {
            return Err(MultiAssetError::InvalidCorrelation(
                "Matrix is not positive semi-definite".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Regularize the correlation matrix if needed
    pub fn regularize(&mut self, epsilon: FixedPoint) -> Result<(), MultiAssetError> {
        let n = self.matrix.len();
        
        // Add small value to diagonal for numerical stability
        for i in 0..n {
            self.matrix[i][i] = self.matrix[i][i] + epsilon;
        }
        
        // Renormalize to ensure diagonal is 1
        for i in 0..n {
            let diag_sqrt = self.matrix[i][i].sqrt();
            for j in 0..n {
                self.matrix[i][j] = self.matrix[i][j] / diag_sqrt;
                self.matrix[j][i] = self.matrix[j][i] / diag_sqrt;
            }
        }
        
        // Recompute eigenvalues (simplified)
        self.compute_eigenvalues()?;
        
        Ok(())
    }
    
    /// Compute eigenvalues (simplified power iteration method)
    fn compute_eigenvalues(&mut self) -> Result<(), MultiAssetError> {
        let n = self.matrix.len();
        self.eigenvalues.clear();
        
        // Simplified eigenvalue computation - in practice would use LAPACK
        for i in 0..n {
            // Use Gershgorin circle theorem for rough eigenvalue bounds
            let mut sum = FixedPoint::zero();
            for j in 0..n {
                if i != j {
                    sum = sum + self.matrix[i][j].abs();
                }
            }
            let eigenvalue_bound = self.matrix[i][i] + sum;
            self.eigenvalues.push(eigenvalue_bound);
        }
        
        // Compute condition number
        if let (Some(&max_eig), Some(&min_eig)) = 
            (self.eigenvalues.iter().max(), self.eigenvalues.iter().min()) {
            if min_eig > FixedPoint::zero() {
                self.condition_number = max_eig / min_eig;
            } else {
                self.condition_number = FixedPoint::from_float(1e10); // Very large condition number
            }
        }
        
        Ok(())
    }
    
    /// Get correlation between two assets
    pub fn get_correlation(&self, asset1: AssetId, asset2: AssetId) -> Result<FixedPoint, MultiAssetError> {
        let idx1 = self.assets.iter().position(|&a| a == asset1)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("Asset {} not found", asset1)))?;
        let idx2 = self.assets.iter().position(|&a| a == asset2)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("Asset {} not found", asset2)))?;
        
        Ok(self.matrix[idx1][idx2])
    }
    
    /// Set correlation between two assets
    pub fn set_correlation(&mut self, asset1: AssetId, asset2: AssetId, correlation: FixedPoint) -> Result<(), MultiAssetError> {
        let idx1 = self.assets.iter().position(|&a| a == asset1)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("Asset {} not found", asset1)))?;
        let idx2 = self.assets.iter().position(|&a| a == asset2)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("Asset {} not found", asset2)))?;
        
        self.matrix[idx1][idx2] = correlation;
        self.matrix[idx2][idx1] = correlation; // Ensure symmetry
        
        Ok(())
    }
    
    /// Apply shrinkage estimation for robust correlation matrix
    pub fn apply_shrinkage(&mut self, shrinkage_intensity: FixedPoint) -> Result<(), MultiAssetError> {
        let n = self.matrix.len();
        
        // Create identity matrix as shrinkage target
        let mut identity = vec![vec![FixedPoint::zero(); n]; n];
        for i in 0..n {
            identity[i][i] = FixedPoint::one();
        }
        
        // Apply shrinkage: Σ_shrunk = (1-λ)*Σ_sample + λ*I
        for i in 0..n {
            for j in 0..n {
                self.matrix[i][j] = (FixedPoint::one() - shrinkage_intensity) * self.matrix[i][j] + 
                                   shrinkage_intensity * identity[i][j];
            }
        }
        
        // Recompute eigenvalues after shrinkage
        self.compute_eigenvalues()?;
        
        Ok(())
    }
}

/// Market regime states for correlation modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    Normal,
    Stress,
    Crisis,
}

/// Dynamic correlation estimator with EWMA and DCC-GARCH models
#[derive(Debug, Clone)]
pub struct DynamicCorrelationEstimator {
    /// Assets being tracked
    pub assets: Vec<AssetId>,
    /// EWMA decay parameter λ
    pub ewma_lambda: FixedPoint,
    /// DCC-GARCH parameters (α, β)
    pub dcc_alpha: FixedPoint,
    pub dcc_beta: FixedPoint,
    /// Current correlation matrix
    pub current_correlation: CorrelationMatrix,
    /// Historical returns for correlation estimation
    pub return_history: HashMap<AssetId, Vec<FixedPoint>>,
    /// Maximum history length
    pub max_history_length: usize,
    /// Current market regime
    pub current_regime: MarketRegime,
    /// Regime-dependent correlation matrices
    pub regime_correlations: HashMap<MarketRegime, CorrelationMatrix>,
    /// Regime detection parameters
    pub regime_detection_window: usize,
    pub stress_threshold: FixedPoint,
    pub crisis_threshold: FixedPoint,
}

impl DynamicCorrelationEstimator {
    /// Create a new dynamic correlation estimator
    pub fn new(
        assets: Vec<AssetId>,
        ewma_lambda: FixedPoint,
        dcc_alpha: FixedPoint,
        dcc_beta: FixedPoint,
    ) -> Self {
        let current_correlation = CorrelationMatrix::new(assets.clone());
        let mut regime_correlations = HashMap::new();
        
        // Initialize regime-specific correlation matrices
        regime_correlations.insert(MarketRegime::Normal, CorrelationMatrix::new(assets.clone()));
        regime_correlations.insert(MarketRegime::Stress, CorrelationMatrix::new(assets.clone()));
        regime_correlations.insert(MarketRegime::Crisis, CorrelationMatrix::new(assets.clone()));
        
        Self {
            assets,
            ewma_lambda,
            dcc_alpha,
            dcc_beta,
            current_correlation,
            return_history: HashMap::new(),
            max_history_length: 1000,
            current_regime: MarketRegime::Normal,
            regime_correlations,
            regime_detection_window: 50,
            stress_threshold: FixedPoint::from_float(2.0),
            crisis_threshold: FixedPoint::from_float(3.0),
        }
    }
    
    /// Update correlation matrix with new returns using EWMA
    pub fn update_ewma_correlation(
        &mut self,
        new_returns: &HashMap<AssetId, FixedPoint>,
        timestamp: u64,
    ) -> Result<(), MultiAssetError> {
        // Add new returns to history
        for (&asset, &return_val) in new_returns {
            let history = self.return_history.entry(asset).or_insert_with(Vec::new);
            history.push(return_val);
            
            // Maintain maximum history length
            if history.len() > self.max_history_length {
                history.remove(0);
            }
        }
        
        // Update correlation matrix using EWMA
        let n = self.assets.len();
        for i in 0..n {
            for j in i..n {
                let asset_i = self.assets[i];
                let asset_j = self.assets[j];
                
                if i == j {
                    // Diagonal elements are 1
                    self.current_correlation.set_correlation(asset_i, asset_j, FixedPoint::one())?;
                } else {
                    // Compute EWMA correlation
                    let correlation = self.compute_ewma_correlation(asset_i, asset_j)?;
                    self.current_correlation.set_correlation(asset_i, asset_j, correlation)?;
                }
            }
        }
        
        self.current_correlation.last_update = timestamp;
        
        // Validate and regularize if needed
        if let Err(_) = self.current_correlation.validate() {
            self.current_correlation.regularize(FixedPoint::from_float(1e-6))?;
        }
        
        Ok(())
    }
    
    /// Compute EWMA correlation between two assets
    fn compute_ewma_correlation(
        &self,
        asset_i: AssetId,
        asset_j: AssetId,
    ) -> Result<FixedPoint, MultiAssetError> {
        let history_i = self.return_history.get(&asset_i)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset_i)))?;
        let history_j = self.return_history.get(&asset_j)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset_j)))?;
        
        let min_length = history_i.len().min(history_j.len());
        if min_length < 2 {
            return Ok(FixedPoint::zero());
        }
        
        // Compute EWMA covariance and variances
        let mut ewma_cov = FixedPoint::zero();
        let mut ewma_var_i = FixedPoint::zero();
        let mut ewma_var_j = FixedPoint::zero();
        let mut weight = FixedPoint::one();
        let mut total_weight = FixedPoint::zero();
        
        for k in (0..min_length).rev() {
            let ret_i = history_i[history_i.len() - 1 - k];
            let ret_j = history_j[history_j.len() - 1 - k];
            
            ewma_cov = ewma_cov + weight * ret_i * ret_j;
            ewma_var_i = ewma_var_i + weight * ret_i * ret_i;
            ewma_var_j = ewma_var_j + weight * ret_j * ret_j;
            total_weight = total_weight + weight;
            
            weight = weight * self.ewma_lambda;
        }
        
        if total_weight > FixedPoint::zero() {
            ewma_cov = ewma_cov / total_weight;
            ewma_var_i = ewma_var_i / total_weight;
            ewma_var_j = ewma_var_j / total_weight;
        }
        
        // Compute correlation
        let std_i = ewma_var_i.sqrt();
        let std_j = ewma_var_j.sqrt();
        
        if std_i > FixedPoint::zero() && std_j > FixedPoint::zero() {
            let correlation = ewma_cov / (std_i * std_j);
            // Clamp to [-1, 1]
            Ok(correlation.max(FixedPoint::from_float(-1.0)).min(FixedPoint::one()))
        } else {
            Ok(FixedPoint::zero())
        }
    }
    
    /// Update correlation using DCC-GARCH model
    pub fn update_dcc_garch_correlation(
        &mut self,
        new_returns: &HashMap<AssetId, FixedPoint>,
        timestamp: u64,
    ) -> Result<(), MultiAssetError> {
        // Add new returns to history
        for (&asset, &return_val) in new_returns {
            let history = self.return_history.entry(asset).or_insert_with(Vec::new);
            history.push(return_val);
            
            if history.len() > self.max_history_length {
                history.remove(0);
            }
        }
        
        // DCC-GARCH model: Q_t = (1-α-β)*Q̄ + α*ε_{t-1}*ε'_{t-1} + β*Q_{t-1}
        // where Q̄ is unconditional correlation matrix
        
        let n = self.assets.len();
        let mut new_correlation_matrix = vec![vec![FixedPoint::zero(); n]; n];
        
        // Compute unconditional correlation matrix Q̄
        let unconditional_corr = self.compute_unconditional_correlation()?;
        
        // Get standardized residuals
        let standardized_returns = self.compute_standardized_returns(new_returns)?;
        
        // Update dynamic correlation matrix
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    new_correlation_matrix[i][j] = FixedPoint::one();
                } else {
                    let asset_i = self.assets[i];
                    let asset_j = self.assets[j];
                    
                    let unconditional_element = unconditional_corr.get_correlation(asset_i, asset_j)
                        .unwrap_or(FixedPoint::zero());
                    let previous_element = self.current_correlation.get_correlation(asset_i, asset_j)
                        .unwrap_or(FixedPoint::zero());
                    
                    let epsilon_i = standardized_returns.get(&asset_i)
                        .copied().unwrap_or(FixedPoint::zero());
                    let epsilon_j = standardized_returns.get(&asset_j)
                        .copied().unwrap_or(FixedPoint::zero());
                    
                    // DCC update equation
                    let new_element = (FixedPoint::one() - self.dcc_alpha - self.dcc_beta) * unconditional_element +
                                     self.dcc_alpha * epsilon_i * epsilon_j +
                                     self.dcc_beta * previous_element;
                    
                    new_correlation_matrix[i][j] = new_element;
                }
            }
        }
        
        // Update correlation matrix
        self.current_correlation.matrix = new_correlation_matrix;
        self.current_correlation.last_update = timestamp;
        
        // Validate and regularize
        if let Err(_) = self.current_correlation.validate() {
            self.current_correlation.regularize(FixedPoint::from_float(1e-6))?;
        }
        
        Ok(())
    }
    
    /// Compute unconditional correlation matrix
    fn compute_unconditional_correlation(&self) -> Result<CorrelationMatrix, MultiAssetError> {
        let mut unconditional = CorrelationMatrix::new(self.assets.clone());
        
        let n = self.assets.len();
        for i in 0..n {
            for j in i..n {
                let asset_i = self.assets[i];
                let asset_j = self.assets[j];
                
                if i == j {
                    unconditional.set_correlation(asset_i, asset_j, FixedPoint::one())?;
                } else {
                    // Use sample correlation as unconditional correlation
                    let correlation = self.compute_sample_correlation(asset_i, asset_j)?;
                    unconditional.set_correlation(asset_i, asset_j, correlation)?;
                }
            }
        }
        
        Ok(unconditional)
    }
    
    /// Compute sample correlation between two assets
    fn compute_sample_correlation(
        &self,
        asset_i: AssetId,
        asset_j: AssetId,
    ) -> Result<FixedPoint, MultiAssetError> {
        let history_i = self.return_history.get(&asset_i)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset_i)))?;
        let history_j = self.return_history.get(&asset_j)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset_j)))?;
        
        let min_length = history_i.len().min(history_j.len());
        if min_length < 2 {
            return Ok(FixedPoint::zero());
        }
        
        // Compute means
        let mean_i = history_i.iter().take(min_length)
            .fold(FixedPoint::zero(), |acc, &x| acc + x) / FixedPoint::from_int(min_length as i32);
        let mean_j = history_j.iter().take(min_length)
            .fold(FixedPoint::zero(), |acc, &x| acc + x) / FixedPoint::from_int(min_length as i32);
        
        // Compute covariance and variances
        let mut covariance = FixedPoint::zero();
        let mut var_i = FixedPoint::zero();
        let mut var_j = FixedPoint::zero();
        
        for k in 0..min_length {
            let dev_i = history_i[k] - mean_i;
            let dev_j = history_j[k] - mean_j;
            
            covariance = covariance + dev_i * dev_j;
            var_i = var_i + dev_i * dev_i;
            var_j = var_j + dev_j * dev_j;
        }
        
        let n_minus_1 = FixedPoint::from_int((min_length - 1) as i32);
        covariance = covariance / n_minus_1;
        var_i = var_i / n_minus_1;
        var_j = var_j / n_minus_1;
        
        // Compute correlation
        let std_i = var_i.sqrt();
        let std_j = var_j.sqrt();
        
        if std_i > FixedPoint::zero() && std_j > FixedPoint::zero() {
            let correlation = covariance / (std_i * std_j);
            Ok(correlation.max(FixedPoint::from_float(-1.0)).min(FixedPoint::one()))
        } else {
            Ok(FixedPoint::zero())
        }
    }
    
    /// Compute standardized returns for DCC-GARCH
    fn compute_standardized_returns(
        &self,
        returns: &HashMap<AssetId, FixedPoint>,
    ) -> Result<HashMap<AssetId, FixedPoint>, MultiAssetError> {
        let mut standardized = HashMap::new();
        
        for (&asset, &return_val) in returns {
            // Compute rolling standard deviation
            if let Some(history) = self.return_history.get(&asset) {
                if history.len() >= 10 {
                    let recent_returns = &history[history.len().saturating_sub(20)..];
                    let mean = recent_returns.iter()
                        .fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                        FixedPoint::from_int(recent_returns.len() as i32);
                    
                    let variance = recent_returns.iter()
                        .map(|&x| (x - mean) * (x - mean))
                        .fold(FixedPoint::zero(), |acc, x| acc + x) / 
                        FixedPoint::from_int((recent_returns.len() - 1) as i32);
                    
                    let std_dev = variance.sqrt();
                    
                    if std_dev > FixedPoint::zero() {
                        standardized.insert(asset, (return_val - mean) / std_dev);
                    } else {
                        standardized.insert(asset, FixedPoint::zero());
                    }
                } else {
                    standardized.insert(asset, return_val);
                }
            } else {
                standardized.insert(asset, return_val);
            }
        }
        
        Ok(standardized)
    }
    
    /// Detect market regime based on correlation breakdown
    pub fn detect_market_regime(
        &mut self,
        new_returns: &HashMap<AssetId, FixedPoint>,
    ) -> Result<MarketRegime, MultiAssetError> {
        // Compute current market stress indicators
        let stress_level = self.compute_market_stress_level(new_returns)?;
        
        // Detect regime based on stress level
        let new_regime = if stress_level > self.crisis_threshold {
            MarketRegime::Crisis
        } else if stress_level > self.stress_threshold {
            MarketRegime::Stress
        } else {
            MarketRegime::Normal
        };
        
        // Update current regime
        if new_regime != self.current_regime {
            println!("Market regime changed from {:?} to {:?} (stress level: {:.3})", 
                    self.current_regime, new_regime, stress_level.to_float());
            self.current_regime = new_regime;
        }
        
        Ok(new_regime)
    }
    
    /// Compute market stress level for regime detection
    fn compute_market_stress_level(
        &self,
        new_returns: &HashMap<AssetId, FixedPoint>,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Compute portfolio volatility as stress indicator
        let mut portfolio_variance = FixedPoint::zero();
        let n_assets = self.assets.len();
        
        if n_assets == 0 {
            return Ok(FixedPoint::zero());
        }
        
        let equal_weight = FixedPoint::one() / FixedPoint::from_int(n_assets as i32);
        
        // Compute weighted portfolio variance
        for i in 0..n_assets {
            for j in 0..n_assets {
                let asset_i = self.assets[i];
                let asset_j = self.assets[j];
                
                let return_i = new_returns.get(&asset_i).copied().unwrap_or(FixedPoint::zero());
                let return_j = new_returns.get(&asset_j).copied().unwrap_or(FixedPoint::zero());
                
                let correlation = if i == j {
                    FixedPoint::one()
                } else {
                    self.current_correlation.get_correlation(asset_i, asset_j)
                        .unwrap_or(FixedPoint::zero())
                };
                
                // Simplified volatility estimation
                let vol_i = return_i.abs();
                let vol_j = return_j.abs();
                
                portfolio_variance = portfolio_variance + 
                    equal_weight * equal_weight * vol_i * vol_j * correlation;
            }
        }
        
        let portfolio_volatility = portfolio_variance.sqrt();
        
        // Normalize by historical average (simplified)
        let historical_avg_vol = FixedPoint::from_float(0.02); // 2% daily volatility
        let stress_level = portfolio_volatility / historical_avg_vol;
        
        Ok(stress_level)
    }
    
    /// Get regime-specific correlation matrix
    pub fn get_regime_correlation(&self, regime: MarketRegime) -> Option<&CorrelationMatrix> {
        self.regime_correlations.get(&regime)
    }
    
    /// Update regime-specific correlation matrix
    pub fn update_regime_correlation(
        &mut self,
        regime: MarketRegime,
        correlation_matrix: CorrelationMatrix,
    ) {
        self.regime_correlations.insert(regime, correlation_matrix);
    }
    
    /// Apply shrinkage estimation to current correlation matrix
    pub fn apply_shrinkage_estimation(
        &mut self,
        shrinkage_intensity: FixedPoint,
    ) -> Result<(), MultiAssetError> {
        self.current_correlation.apply_shrinkage(shrinkage_intensity)
    }
    
    /// Get current correlation matrix
    pub fn get_current_correlation(&self) -> &CorrelationMatrix {
        &self.current_correlation
    }
    
    /// Validate correlation matrix and apply regularization if needed
    pub fn validate_and_regularize(&mut self) -> Result<(), MultiAssetError> {
        if let Err(_) = self.current_correlation.validate() {
            println!("Correlation matrix validation failed, applying regularization");
            self.current_correlation.regularize(FixedPoint::from_float(1e-6))?;
        }
        Ok(())
    }
}

/// Portfolio position for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetPosition {
    pub asset_id: AssetId,
    pub quantity: FixedPoint,
    pub average_price: FixedPoint,
    pub market_value: FixedPoint,
    pub unrealized_pnl: FixedPoint,
    pub last_update: u64,
}

/// Portfolio constraints for risk management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioConstraints {
    /// Maximum position size per asset
    pub max_position_sizes: HashMap<AssetId, FixedPoint>,
    /// Maximum portfolio value
    pub max_portfolio_value: FixedPoint,
    /// Maximum concentration per asset (as fraction of portfolio)
    pub max_concentration: FixedPoint,
    /// Maximum leverage ratio
    pub max_leverage: FixedPoint,
    /// Minimum diversification (minimum number of assets)
    pub min_diversification: usize,
    /// Sector limits
    pub sector_limits: HashMap<String, FixedPoint>,
    /// Geographic limits
    pub geographic_limits: HashMap<String, FixedPoint>,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            max_position_sizes: HashMap::new(),
            max_portfolio_value: FixedPoint::from_float(1_000_000.0),
            max_concentration: FixedPoint::from_float(0.2), // 20% max per asset
            max_leverage: FixedPoint::from_float(2.0),
            min_diversification: 3,
            sector_limits: HashMap::new(),
            geographic_limits: HashMap::new(),
        }
    }
}

/// Cross-asset portfolio optimizer
#[derive(Debug, Clone)]
pub struct CrossAssetPortfolioOptimizer {
    /// Assets in the portfolio
    pub assets: Vec<AssetId>,
    /// Current positions
    pub positions: HashMap<AssetId, AssetPosition>,
    /// Portfolio constraints
    pub constraints: PortfolioConstraints,
    /// Dynamic correlation estimator
    pub correlation_estimator: DynamicCorrelationEstimator,
    /// Transaction cost model
    pub transaction_costs: HashMap<AssetId, FixedPoint>,
    /// Risk aversion parameter
    pub risk_aversion: FixedPoint,
    /// Rebalancing threshold
    pub rebalancing_threshold: FixedPoint,
}

impl CrossAssetPortfolioOptimizer {
    /// Create a new cross-asset portfolio optimizer
    pub fn new(
        assets: Vec<AssetId>,
        constraints: PortfolioConstraints,
        risk_aversion: FixedPoint,
    ) -> Self {
        let correlation_estimator = DynamicCorrelationEstimator::new(
            assets.clone(),
            FixedPoint::from_float(0.94), // EWMA lambda
            FixedPoint::from_float(0.01), // DCC alpha
            FixedPoint::from_float(0.95), // DCC beta
        );
        
        Self {
            assets,
            positions: HashMap::new(),
            constraints,
            correlation_estimator,
            transaction_costs: HashMap::new(),
            risk_aversion,
            rebalancing_threshold: FixedPoint::from_float(0.05), // 5% threshold
        }
    }
    
    /// Calculate portfolio risk: R = q^T Σ q
    pub fn calculate_portfolio_risk(
        &self,
        positions: &HashMap<AssetId, FixedPoint>,
        correlation_matrix: &CorrelationMatrix,
        volatilities: &HashMap<AssetId, FixedPoint>,
    ) -> Result<FixedPoint, MultiAssetError> {
        let mut risk = FixedPoint::zero();
        
        for &asset_i in &self.assets {
            for &asset_j in &self.assets {
                let qi = positions.get(&asset_i).copied().unwrap_or(FixedPoint::zero());
                let qj = positions.get(&asset_j).copied().unwrap_or(FixedPoint::zero());
                
                let vol_i = volatilities.get(&asset_i)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                let vol_j = volatilities.get(&asset_j)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                
                let correlation = if asset_i == asset_j {
                    FixedPoint::one()
                } else {
                    correlation_matrix.get_correlation(asset_i, asset_j)
                        .unwrap_or(FixedPoint::zero())
                };
                
                // Covariance = σᵢ * σⱼ * ρᵢⱼ
                let covariance = vol_i * vol_j * correlation;
                
                risk = risk + qi * qj * covariance;
            }
        }
        
        Ok(risk)
    }
    
    /// Calculate cross-asset reservation prices with correlation effects
    pub fn calculate_cross_asset_reservation_prices(
        &self,
        mid_prices: &HashMap<AssetId, FixedPoint>,
        current_positions: &HashMap<AssetId, FixedPoint>,
        correlation_matrix: &CorrelationMatrix,
        volatilities: &HashMap<AssetId, FixedPoint>,
        time_horizon: FixedPoint,
    ) -> Result<HashMap<AssetId, FixedPoint>, MultiAssetError> {
        let mut reservation_prices = HashMap::new();
        
        for &asset in &self.assets {
            let mid_price = mid_prices.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(100.0));
            let position = current_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let volatility = volatilities.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.2));
            
            // Base reservation price: r₀ = S - q*γ*σ²*(T-t)
            let base_reservation = mid_price - position * self.risk_aversion * 
                volatility * volatility * time_horizon;
            
            // Cross-asset adjustment based on correlations
            let mut cross_asset_adjustment = FixedPoint::zero();
            
            for &other_asset in &self.assets {
                if other_asset == asset {
                    continue;
                }
                
                let other_position = current_positions.get(&other_asset)
                    .copied().unwrap_or(FixedPoint::zero());
                let other_volatility = volatilities.get(&other_asset)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                
                let correlation = correlation_matrix.get_correlation(asset, other_asset)
                    .unwrap_or(FixedPoint::zero());
                
                // Cross-asset risk adjustment: -Σⱼ≠ᵢ qⱼ*ρᵢⱼ*σᵢ*σⱼ*γ*(T-t)
                cross_asset_adjustment = cross_asset_adjustment - 
                    other_position * correlation * volatility * other_volatility * 
                    self.risk_aversion * time_horizon;
            }
            
            let final_reservation = base_reservation + cross_asset_adjustment;
            reservation_prices.insert(asset, final_reservation);
        }
        
        Ok(reservation_prices)
    }
    
    /// Check portfolio constraints
    pub fn check_portfolio_constraints(
        &self,
        positions: &HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<Vec<String>, MultiAssetError> {
        let mut violations = Vec::new();
        
        // Calculate portfolio metrics
        let mut total_value = FixedPoint::zero();
        let mut asset_values = HashMap::new();
        
        for &asset in &self.assets {
            let position = positions.get(&asset).copied().unwrap_or(FixedPoint::zero());
            let price = prices.get(&asset).copied().unwrap_or(FixedPoint::from_float(100.0));
            let value = position.abs() * price;
            
            asset_values.insert(asset, value);
            total_value = total_value + value;
        }
        
        // Check individual position limits
        for &asset in &self.assets {
            let position = positions.get(&asset).copied().unwrap_or(FixedPoint::zero());
            
            if let Some(&max_position) = self.constraints.max_position_sizes.get(&asset) {
                if position.abs() > max_position {
                    violations.push(format!(
                        "Asset {} position {} exceeds limit {}", 
                        asset, position.to_float(), max_position.to_float()
                    ));
                }
            }
        }
        
        // Check portfolio value limit
        if total_value > self.constraints.max_portfolio_value {
            violations.push(format!(
                "Portfolio value {} exceeds limit {}", 
                total_value.to_float(), self.constraints.max_portfolio_value.to_float()
            ));
        }
        
        // Check concentration limits
        for &asset in &self.assets {
            let asset_value = asset_values.get(&asset).copied().unwrap_or(FixedPoint::zero());
            
            if total_value > FixedPoint::zero() {
                let concentration = asset_value / total_value;
                
                if concentration > self.constraints.max_concentration {
                    violations.push(format!(
                        "Asset {} concentration {:.2}% exceeds limit {:.2}%", 
                        asset, 
                        (concentration * FixedPoint::from_float(100.0)).to_float(),
                        (self.constraints.max_concentration * FixedPoint::from_float(100.0)).to_float()
                    ));
                }
            }
        }
        
        // Check diversification
        let active_positions = positions.values()
            .filter(|&&pos| pos.abs() > FixedPoint::from_float(1e-6))
            .count();
        
        if active_positions < self.constraints.min_diversification {
            violations.push(format!(
                "Portfolio has {} active positions, minimum required: {}", 
                active_positions, self.constraints.min_diversification
            ));
        }
        
        Ok(violations)
    }
    
    /// Optimize portfolio allocation using mean-variance optimization
    pub fn optimize_portfolio_allocation(
        &mut self,
        expected_returns: &HashMap<AssetId, FixedPoint>,
        correlation_matrix: &CorrelationMatrix,
        volatilities: &HashMap<AssetId, FixedPoint>,
        current_positions: &HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<HashMap<AssetId, FixedPoint>, MultiAssetError> {
        // Simplified mean-variance optimization
        // In practice, would use quadratic programming solver
        
        let n = self.assets.len();
        if n == 0 {
            return Ok(HashMap::new());
        }
        
        let mut optimal_positions = HashMap::new();
        
        // Calculate expected portfolio return and risk for each asset
        let mut asset_scores = HashMap::new();
        
        for &asset in &self.assets {
            let expected_return = expected_returns.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let volatility = volatilities.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.2));
            
            // Simple risk-adjusted score: (μ - r) / (γ * σ²)
            let risk_free_rate = FixedPoint::from_float(0.02); // 2% risk-free rate
            let excess_return = expected_return - risk_free_rate;
            let risk_penalty = self.risk_aversion * volatility * volatility;
            
            let score = if risk_penalty > FixedPoint::zero() {
                excess_return / risk_penalty
            } else {
                FixedPoint::zero()
            };
            
            asset_scores.insert(asset, score);
        }
        
        // Normalize scores and apply constraints
        let total_score: FixedPoint = asset_scores.values().map(|&s| s.max(FixedPoint::zero())).sum();
        
        if total_score > FixedPoint::zero() {
            let base_allocation = FixedPoint::from_float(10000.0); // Base allocation size
            
            for &asset in &self.assets {
                let score = asset_scores.get(&asset).copied().unwrap_or(FixedPoint::zero());
                let weight = score.max(FixedPoint::zero()) / total_score;
                let target_position = weight * base_allocation;
                
                // Apply position limits
                let max_position = self.constraints.max_position_sizes.get(&asset)
                    .copied().unwrap_or(FixedPoint::from_float(5000.0));
                
                let constrained_position = target_position.min(max_position);
                optimal_positions.insert(asset, constrained_position);
            }
        } else {
            // If no positive scores, maintain current positions
            for &asset in &self.assets {
                let current_pos = current_positions.get(&asset)
                    .copied().unwrap_or(FixedPoint::zero());
                optimal_positions.insert(asset, current_pos);
            }
        }
        
        // Validate constraints
        let violations = self.check_portfolio_constraints(&optimal_positions, prices)?;
        if !violations.is_empty() {
            println!("Portfolio constraint violations detected: {:?}", violations);
            
            // Apply constraint corrections (simplified)
            self.apply_constraint_corrections(&mut optimal_positions, prices)?;
        }
        
        Ok(optimal_positions)
    }
    
    /// Apply constraint corrections to portfolio positions
    fn apply_constraint_corrections(
        &self,
        positions: &mut HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<(), MultiAssetError> {
        // Calculate total portfolio value
        let mut total_value = FixedPoint::zero();
        for &asset in &self.assets {
            let position = positions.get(&asset).copied().unwrap_or(FixedPoint::zero());
            let price = prices.get(&asset).copied().unwrap_or(FixedPoint::from_float(100.0));
            total_value = total_value + position.abs() * price;
        }
        
        // Scale down if portfolio value exceeds limit
        if total_value > self.constraints.max_portfolio_value {
            let scale_factor = self.constraints.max_portfolio_value / total_value;
            
            for &asset in &self.assets {
                if let Some(position) = positions.get_mut(&asset) {
                    *position = *position * scale_factor;
                }
            }
        }
        
        // Apply individual position limits
        for &asset in &self.assets {
            if let Some(position) = positions.get_mut(&asset) {
                if let Some(&max_position) = self.constraints.max_position_sizes.get(&asset) {
                    if position.abs() > max_position {
                        *position = if *position > FixedPoint::zero() {
                            max_position
                        } else {
                            -max_position
                        };
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate transaction costs for rebalancing
    pub fn calculate_transaction_costs(
        &self,
        current_positions: &HashMap<AssetId, FixedPoint>,
        target_positions: &HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<FixedPoint, MultiAssetError> {
        let mut total_cost = FixedPoint::zero();
        
        for &asset in &self.assets {
            let current_pos = current_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let target_pos = target_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let price = prices.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(100.0));
            
            let trade_size = (target_pos - current_pos).abs();
            let trade_value = trade_size * price;
            
            let cost_rate = self.transaction_costs.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.001)); // 0.1% default cost
            
            let trade_cost = trade_value * cost_rate;
            total_cost = total_cost + trade_cost;
        }
        
        Ok(total_cost)
    }
    
    /// Determine if rebalancing is needed
    pub fn should_rebalance(
        &self,
        current_positions: &HashMap<AssetId, FixedPoint>,
        target_positions: &HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<bool, MultiAssetError> {
        let mut max_deviation = FixedPoint::zero();
        
        // Calculate total portfolio values
        let mut current_total = FixedPoint::zero();
        let mut target_total = FixedPoint::zero();
        
        for &asset in &self.assets {
            let current_pos = current_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let target_pos = target_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let price = prices.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(100.0));
            
            current_total = current_total + current_pos.abs() * price;
            target_total = target_total + target_pos.abs() * price;
        }
        
        // Calculate relative deviations
        for &asset in &self.assets {
            let current_pos = current_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let target_pos = target_positions.get(&asset)
                .copied().unwrap_or(FixedPoint::zero());
            let price = prices.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(100.0));
            
            let current_weight = if current_total > FixedPoint::zero() {
                (current_pos.abs() * price) / current_total
            } else {
                FixedPoint::zero()
            };
            
            let target_weight = if target_total > FixedPoint::zero() {
                (target_pos.abs() * price) / target_total
            } else {
                FixedPoint::zero()
            };
            
            let deviation = (current_weight - target_weight).abs();
            if deviation > max_deviation {
                max_deviation = deviation;
            }
        }
        
        Ok(max_deviation > self.rebalancing_threshold)
    }
    
    /// Execute dynamic rebalancing with transaction cost optimization
    pub fn execute_dynamic_rebalancing(
        &mut self,
        expected_returns: &HashMap<AssetId, FixedPoint>,
        prices: &HashMap<AssetId, FixedPoint>,
        timestamp: u64,
    ) -> Result<HashMap<AssetId, FixedPoint>, MultiAssetError> {
        // Get current correlation matrix
        let correlation_matrix = self.correlation_estimator.get_current_correlation();
        
        // Estimate volatilities from recent returns
        let mut volatilities = HashMap::new();
        for &asset in &self.assets {
            // Simplified volatility estimation
            volatilities.insert(asset, FixedPoint::from_float(0.2));
        }
        
        // Get current positions
        let current_positions: HashMap<AssetId, FixedPoint> = self.positions.iter()
            .map(|(&asset, pos)| (asset, pos.quantity))
            .collect();
        
        // Optimize portfolio allocation
        let target_positions = self.optimize_portfolio_allocation(
            expected_returns,
            correlation_matrix,
            &volatilities,
            &current_positions,
            prices,
        )?;
        
        // Check if rebalancing is needed
        let should_rebalance = self.should_rebalance(&current_positions, &target_positions, prices)?;
        
        if should_rebalance {
            // Calculate transaction costs
            let transaction_costs = self.calculate_transaction_costs(
                &current_positions, &target_positions, prices
            )?;
            
            println!("Rebalancing portfolio at timestamp {}, transaction costs: {:.2}", 
                    timestamp, transaction_costs.to_float());
            
            // Update positions
            for &asset in &self.assets {
                let target_pos = target_positions.get(&asset)
                    .copied().unwrap_or(FixedPoint::zero());
                let price = prices.get(&asset)
                    .copied().unwrap_or(FixedPoint::from_float(100.0));
                
                let position = AssetPosition {
                    asset_id: asset,
                    quantity: target_pos,
                    average_price: price,
                    market_value: target_pos * price,
                    unrealized_pnl: FixedPoint::zero(),
                    last_update: timestamp,
                };
                
                self.positions.insert(asset, position);
            }
            
            Ok(target_positions)
        } else {
            Ok(current_positions)
        }
    }
    
    /// Get current portfolio positions
    pub fn get_current_positions(&self) -> HashMap<AssetId, FixedPoint> {
        self.positions.iter()
            .map(|(&asset, pos)| (asset, pos.quantity))
            .collect()
    }
    
    /// Update transaction costs for an asset
    pub fn set_transaction_cost(&mut self, asset: AssetId, cost_rate: FixedPoint) {
        self.transaction_costs.insert(asset, cost_rate);
    }
    
    /// Set rebalancing threshold
    pub fn set_rebalancing_threshold(&mut self, threshold: FixedPoint) {
        self.rebalancing_threshold = threshold;
    }
}

/// Cointegration relationship between assets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CointegrationRelationship {
    /// Assets in the cointegration relationship
    pub assets: Vec<AssetId>,
    /// Cointegration weights (β coefficients)
    pub weights: Vec<FixedPoint>,
    /// Mean reversion speed (θ)
    pub mean_reversion_speed: FixedPoint,
    /// Long-term equilibrium level
    pub equilibrium_level: FixedPoint,
    /// Current spread value
    pub current_spread: FixedPoint,
    /// Spread standard deviation
    pub spread_std: FixedPoint,
    /// Last update timestamp
    pub last_update: u64,
}

/// Arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    /// Type of arbitrage
    pub arbitrage_type: ArbitrageType,
    /// Assets involved
    pub assets: Vec<AssetId>,
    /// Expected profit
    pub expected_profit: FixedPoint,
    /// Confidence level (0-1)
    pub confidence: FixedPoint,
    /// Recommended positions
    pub recommended_positions: HashMap<AssetId, FixedPoint>,
    /// Time horizon for the opportunity
    pub time_horizon: FixedPoint,
    /// Risk level
    pub risk_level: FixedPoint,
}

/// Types of arbitrage opportunities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbitrageType {
    /// Statistical arbitrage based on mean reversion
    StatisticalArbitrage,
    /// Pairs trading
    PairsTrading,
    /// Triangular arbitrage
    TriangularArbitrage,
    /// Cross-asset momentum
    CrossAssetMomentum,
}

/// Arbitrage detection and execution engine
#[derive(Debug, Clone)]
pub struct ArbitrageDetectionEngine {
    /// Assets being monitored
    pub assets: Vec<AssetId>,
    /// Cointegration relationships
    pub cointegration_relationships: Vec<CointegrationRelationship>,
    /// Price history for analysis
    pub price_history: HashMap<AssetId, Vec<FixedPoint>>,
    /// Maximum history length
    pub max_history_length: usize,
    /// Minimum confidence threshold for opportunities
    pub min_confidence_threshold: FixedPoint,
    /// Maximum risk tolerance
    pub max_risk_tolerance: FixedPoint,
    /// Johansen test parameters
    pub johansen_test_window: usize,
    pub johansen_significance_level: FixedPoint,
}

impl ArbitrageDetectionEngine {
    /// Create a new arbitrage detection engine
    pub fn new(assets: Vec<AssetId>) -> Self {
        Self {
            assets,
            cointegration_relationships: Vec::new(),
            price_history: HashMap::new(),
            max_history_length: 1000,
            min_confidence_threshold: FixedPoint::from_float(0.7),
            max_risk_tolerance: FixedPoint::from_float(0.05),
            johansen_test_window: 100,
            johansen_significance_level: FixedPoint::from_float(0.05),
        }
    }
    
    /// Update price history
    pub fn update_price_history(
        &mut self,
        new_prices: &HashMap<AssetId, FixedPoint>,
        timestamp: u64,
    ) -> Result<(), MultiAssetError> {
        for (&asset, &price) in new_prices {
            let history = self.price_history.entry(asset).or_insert_with(Vec::new);
            history.push(price);
            
            // Maintain maximum history length
            if history.len() > self.max_history_length {
                history.remove(0);
            }
        }
        
        // Update cointegration relationships
        self.update_cointegration_relationships(timestamp)?;
        
        Ok(())
    }
    
    /// Implement cointegration testing using Johansen methodology
    pub fn johansen_cointegration_test(
        &self,
        asset_pair: (AssetId, AssetId),
    ) -> Result<Option<CointegrationRelationship>, MultiAssetError> {
        let (asset1, asset2) = asset_pair;
        
        let history1 = self.price_history.get(&asset1)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset1)))?;
        let history2 = self.price_history.get(&asset2)
            .ok_or_else(|| MultiAssetError::MatrixError(format!("No history for asset {}", asset2)))?;
        
        let min_length = history1.len().min(history2.len());
        if min_length < self.johansen_test_window {
            return Ok(None);
        }
        
        // Use recent data for testing
        let start_idx = min_length.saturating_sub(self.johansen_test_window);
        let prices1 = &history1[start_idx..];
        let prices2 = &history2[start_idx..];
        
        // Convert to log prices
        let log_prices1: Vec<FixedPoint> = prices1.iter()
            .map(|&p| p.ln())
            .collect();
        let log_prices2: Vec<FixedPoint> = prices2.iter()
            .map(|&p| p.ln())
            .collect();
        
        // Simplified Johansen test - in practice would use proper eigenvalue decomposition
        // For now, use Engle-Granger two-step method as approximation
        
        // Step 1: Estimate cointegrating relationship using OLS
        let (beta, residuals) = self.estimate_cointegrating_vector(&log_prices1, &log_prices2)?;
        
        // Step 2: Test residuals for unit root (simplified ADF test)
        let is_stationary = self.augmented_dickey_fuller_test(&residuals)?;
        
        if is_stationary {
            // Calculate mean reversion parameters
            let mean_reversion_speed = self.estimate_mean_reversion_speed(&residuals)?;
            let equilibrium_level = residuals.iter()
                .fold(FixedPoint::zero(), |acc, &x| acc + x) / 
                FixedPoint::from_int(residuals.len() as i32);
            
            let spread_variance = residuals.iter()
                .map(|&x| (x - equilibrium_level) * (x - equilibrium_level))
                .fold(FixedPoint::zero(), |acc, x| acc + x) / 
                FixedPoint::from_int((residuals.len() - 1) as i32);
            let spread_std = spread_variance.sqrt();
            
            // Current spread
            let current_spread = log_prices1.last().copied().unwrap_or(FixedPoint::zero()) - 
                               beta * log_prices2.last().copied().unwrap_or(FixedPoint::zero());
            
            let relationship = CointegrationRelationship {
                assets: vec![asset1, asset2],
                weights: vec![FixedPoint::one(), -beta],
                mean_reversion_speed,
                equilibrium_level,
                current_spread,
                spread_std,
                last_update: 0, // Would use actual timestamp
            };
            
            Ok(Some(relationship))
        } else {
            Ok(None)
        }
    }
    
    /// Estimate cointegrating vector using OLS
    fn estimate_cointegrating_vector(
        &self,
        y: &[FixedPoint],
        x: &[FixedPoint],
    ) -> Result<(FixedPoint, Vec<FixedPoint>), MultiAssetError> {
        if y.len() != x.len() || y.is_empty() {
            return Err(MultiAssetError::MatrixError("Invalid input lengths".to_string()));
        }
        
        let n = y.len();
        let n_fp = FixedPoint::from_int(n as i32);
        
        // Calculate means
        let mean_x = x.iter().fold(FixedPoint::zero(), |acc, &val| acc + val) / n_fp;
        let mean_y = y.iter().fold(FixedPoint::zero(), |acc, &val| acc + val) / n_fp;
        
        // Calculate beta = Cov(x,y) / Var(x)
        let mut numerator = FixedPoint::zero();
        let mut denominator = FixedPoint::zero();
        
        for i in 0..n {
            let x_dev = x[i] - mean_x;
            let y_dev = y[i] - mean_y;
            
            numerator = numerator + x_dev * y_dev;
            denominator = denominator + x_dev * x_dev;
        }
        
        let beta = if denominator > FixedPoint::zero() {
            numerator / denominator
        } else {
            FixedPoint::zero()
        };
        
        // Calculate residuals: e_t = y_t - β*x_t
        let residuals: Vec<FixedPoint> = y.iter().zip(x.iter())
            .map(|(&y_val, &x_val)| y_val - beta * x_val)
            .collect();
        
        Ok((beta, residuals))
    }
    
    /// Simplified Augmented Dickey-Fuller test for stationarity
    fn augmented_dickey_fuller_test(
        &self,
        series: &[FixedPoint],
    ) -> Result<bool, MultiAssetError> {
        if series.len() < 10 {
            return Ok(false);
        }
        
        // Create first differences
        let mut differences = Vec::new();
        for i in 1..series.len() {
            differences.push(series[i] - series[i-1]);
        }
        
        // Estimate AR(1) model: Δy_t = α + ρ*y_{t-1} + ε_t
        let lagged_levels: Vec<FixedPoint> = series[..series.len()-1].to_vec();
        
        // Simple OLS estimation for ρ
        let (rho, _) = self.estimate_cointegrating_vector(&differences, &lagged_levels)?;
        
        // Critical values for ADF test (simplified)
        let critical_value_5pct = FixedPoint::from_float(-2.86); // 5% critical value
        
        // Test statistic (simplified)
        let test_statistic = rho; // In practice would compute proper t-statistic
        
        // Reject null hypothesis of unit root if test statistic < critical value
        Ok(test_statistic < critical_value_5pct)
    }
    
    /// Estimate mean reversion speed
    fn estimate_mean_reversion_speed(
        &self,
        residuals: &[FixedPoint],
    ) -> Result<FixedPoint, MultiAssetError> {
        if residuals.len() < 2 {
            return Ok(FixedPoint::zero());
        }
        
        // Estimate AR(1) model: e_t = φ*e_{t-1} + u_t
        let lagged_residuals: Vec<FixedPoint> = residuals[..residuals.len()-1].to_vec();
        let current_residuals: Vec<FixedPoint> = residuals[1..].to_vec();
        
        let (phi, _) = self.estimate_cointegrating_vector(&current_residuals, &lagged_residuals)?;
        
        // Mean reversion speed: θ = -ln(φ)
        let theta = if phi > FixedPoint::zero() && phi < FixedPoint::one() {
            -phi.ln()
        } else {
            FixedPoint::from_float(0.1) // Default value
        };
        
        Ok(theta)
    }
    
    /// Update cointegration relationships
    fn update_cointegration_relationships(
        &mut self,
        timestamp: u64,
    ) -> Result<(), MultiAssetError> {
        // Test all asset pairs for cointegration
        for i in 0..self.assets.len() {
            for j in i+1..self.assets.len() {
                let asset1 = self.assets[i];
                let asset2 = self.assets[j];
                
                if let Some(relationship) = self.johansen_cointegration_test((asset1, asset2))? {
                    // Check if relationship already exists
                    let mut found = false;
                    for existing in &mut self.cointegration_relationships {
                        if existing.assets.contains(&asset1) && existing.assets.contains(&asset2) {
                            // Update existing relationship
                            *existing = relationship;
                            existing.last_update = timestamp;
                            found = true;
                            break;
                        }
                    }
                    
                    if !found {
                        // Add new relationship
                        let mut new_relationship = relationship;
                        new_relationship.last_update = timestamp;
                        self.cointegration_relationships.push(new_relationship);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create spread relationship monitoring and mean reversion detection
    pub fn monitor_spread_relationships(&self) -> Result<Vec<ArbitrageOpportunity>, MultiAssetError> {
        let mut opportunities = Vec::new();
        
        for relationship in &self.cointegration_relationships {
            if relationship.assets.len() != 2 {
                continue; // Only handle pairs for now
            }
            
            let asset1 = relationship.assets[0];
            let asset2 = relationship.assets[1];
            
            // Calculate current spread deviation from equilibrium
            let spread_deviation = relationship.current_spread - relationship.equilibrium_level;
            let normalized_deviation = if relationship.spread_std > FixedPoint::zero() {
                spread_deviation / relationship.spread_std
            } else {
                FixedPoint::zero()
            };
            
            // Check for mean reversion opportunity
            let threshold = FixedPoint::from_float(2.0); // 2 standard deviations
            
            if normalized_deviation.abs() > threshold {
                // Calculate expected profit and time to reversion
                let expected_reversion = -spread_deviation; // Expected move back to equilibrium
                let time_to_reversion = if relationship.mean_reversion_speed > FixedPoint::zero() {
                    FixedPoint::one() / relationship.mean_reversion_speed
                } else {
                    FixedPoint::from_float(30.0) // Default 30 days
                };
                
                // Calculate position sizes
                let mut recommended_positions = HashMap::new();
                
                if spread_deviation > FixedPoint::zero() {
                    // Spread is above equilibrium - short the spread
                    // Short asset1, long asset2
                    recommended_positions.insert(asset1, -FixedPoint::from_float(100.0));
                    recommended_positions.insert(asset2, relationship.weights[1].abs() * FixedPoint::from_float(100.0));
                } else {
                    // Spread is below equilibrium - long the spread
                    // Long asset1, short asset2
                    recommended_positions.insert(asset1, FixedPoint::from_float(100.0));
                    recommended_positions.insert(asset2, -relationship.weights[1].abs() * FixedPoint::from_float(100.0));
                }
                
                // Calculate confidence based on deviation magnitude and historical performance
                let confidence = (normalized_deviation.abs() / FixedPoint::from_float(3.0))
                    .min(FixedPoint::one());
                
                // Calculate risk level
                let risk_level = relationship.spread_std / relationship.equilibrium_level.abs();
                
                if confidence >= self.min_confidence_threshold && risk_level <= self.max_risk_tolerance {
                    let opportunity = ArbitrageOpportunity {
                        arbitrage_type: ArbitrageType::StatisticalArbitrage,
                        assets: relationship.assets.clone(),
                        expected_profit: expected_reversion.abs(),
                        confidence,
                        recommended_positions,
                        time_horizon: time_to_reversion,
                        risk_level,
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        }
        
        Ok(opportunities)
    }
    
    /// Generate cross-asset arbitrage signals
    pub fn generate_arbitrage_signals(
        &self,
        current_prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<Vec<ArbitrageOpportunity>, MultiAssetError> {
        let mut all_opportunities = Vec::new();
        
        // Statistical arbitrage opportunities
        let stat_arb_opportunities = self.monitor_spread_relationships()?;
        all_opportunities.extend(stat_arb_opportunities);
        
        // Pairs trading opportunities
        let pairs_opportunities = self.detect_pairs_trading_opportunities(current_prices)?;
        all_opportunities.extend(pairs_opportunities);
        
        // Cross-asset momentum opportunities
        let momentum_opportunities = self.detect_momentum_opportunities(current_prices)?;
        all_opportunities.extend(momentum_opportunities);
        
        // Sort by expected profit and confidence
        all_opportunities.sort_by(|a, b| {
            let score_a = a.expected_profit * a.confidence;
            let score_b = b.expected_profit * b.confidence;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(all_opportunities)
    }
    
    /// Detect pairs trading opportunities
    fn detect_pairs_trading_opportunities(
        &self,
        current_prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<Vec<ArbitrageOpportunity>, MultiAssetError> {
        let mut opportunities = Vec::new();
        
        // Look for highly correlated pairs with temporary divergence
        for i in 0..self.assets.len() {
            for j in i+1..self.assets.len() {
                let asset1 = self.assets[i];
                let asset2 = self.assets[j];
                
                if let (Some(history1), Some(history2)) = 
                    (self.price_history.get(&asset1), self.price_history.get(&asset2)) {
                    
                    let min_length = history1.len().min(history2.len());
                    if min_length < 20 {
                        continue;
                    }
                    
                    // Calculate correlation
                    let correlation = self.calculate_correlation(
                        &history1[history1.len()-20..],
                        &history2[history2.len()-20..]
                    )?;
                    
                    // Look for high correlation pairs
                    if correlation.abs() > FixedPoint::from_float(0.8) {
                        // Calculate price ratio
                        let current_price1 = current_prices.get(&asset1)
                            .copied().unwrap_or(history1.last().copied().unwrap_or(FixedPoint::one()));
                        let current_price2 = current_prices.get(&asset2)
                            .copied().unwrap_or(history2.last().copied().unwrap_or(FixedPoint::one()));
                        
                        let current_ratio = current_price1 / current_price2;
                        
                        // Calculate historical ratio statistics
                        let ratios: Vec<FixedPoint> = history1.iter().zip(history2.iter())
                            .map(|(&p1, &p2)| p1 / p2)
                            .collect();
                        
                        let mean_ratio = ratios.iter()
                            .fold(FixedPoint::zero(), |acc, &r| acc + r) / 
                            FixedPoint::from_int(ratios.len() as i32);
                        
                        let ratio_variance = ratios.iter()
                            .map(|&r| (r - mean_ratio) * (r - mean_ratio))
                            .fold(FixedPoint::zero(), |acc, v| acc + v) / 
                            FixedPoint::from_int((ratios.len() - 1) as i32);
                        let ratio_std = ratio_variance.sqrt();
                        
                        // Check for significant deviation
                        let z_score = if ratio_std > FixedPoint::zero() {
                            (current_ratio - mean_ratio) / ratio_std
                        } else {
                            FixedPoint::zero()
                        };
                        
                        if z_score.abs() > FixedPoint::from_float(2.0) {
                            let mut recommended_positions = HashMap::new();
                            
                            if z_score > FixedPoint::zero() {
                                // Ratio is high - short asset1, long asset2
                                recommended_positions.insert(asset1, -FixedPoint::from_float(100.0));
                                recommended_positions.insert(asset2, FixedPoint::from_float(100.0));
                            } else {
                                // Ratio is low - long asset1, short asset2
                                recommended_positions.insert(asset1, FixedPoint::from_float(100.0));
                                recommended_positions.insert(asset2, -FixedPoint::from_float(100.0));
                            }
                            
                            let opportunity = ArbitrageOpportunity {
                                arbitrage_type: ArbitrageType::PairsTrading,
                                assets: vec![asset1, asset2],
                                expected_profit: z_score.abs() * ratio_std,
                                confidence: correlation.abs(),
                                recommended_positions,
                                time_horizon: FixedPoint::from_float(5.0), // 5 days
                                risk_level: ratio_std / mean_ratio,
                            };
                            
                            opportunities.push(opportunity);
                        }
                    }
                }
            }
        }
        
        Ok(opportunities)
    }
    
    /// Detect cross-asset momentum opportunities
    fn detect_momentum_opportunities(
        &self,
        current_prices: &HashMap<AssetId, FixedPoint>,
    ) -> Result<Vec<ArbitrageOpportunity>, MultiAssetError> {
        let mut opportunities = Vec::new();
        
        // Look for momentum patterns across assets
        for &asset in &self.assets {
            if let Some(history) = self.price_history.get(&asset) {
                if history.len() < 10 {
                    continue;
                }
                
                // Calculate short-term and long-term momentum
                let short_window = 5;
                let long_window = 20.min(history.len());
                
                let recent_prices = &history[history.len()-short_window..];
                let longer_prices = &history[history.len()-long_window..];
                
                let short_return = (recent_prices.last().unwrap() / recent_prices.first().unwrap()) - FixedPoint::one();
                let long_return = (longer_prices.last().unwrap() / longer_prices.first().unwrap()) - FixedPoint::one();
                
                // Look for strong momentum
                let momentum_threshold = FixedPoint::from_float(0.05); // 5%
                
                if short_return.abs() > momentum_threshold && 
                   short_return * long_return > FixedPoint::zero() { // Same direction
                    
                    let mut recommended_positions = HashMap::new();
                    let position_size = if short_return > FixedPoint::zero() {
                        FixedPoint::from_float(100.0) // Long position
                    } else {
                        -FixedPoint::from_float(100.0) // Short position
                    };
                    
                    recommended_positions.insert(asset, position_size);
                    
                    let opportunity = ArbitrageOpportunity {
                        arbitrage_type: ArbitrageType::CrossAssetMomentum,
                        assets: vec![asset],
                        expected_profit: short_return.abs(),
                        confidence: (short_return.abs() / momentum_threshold).min(FixedPoint::one()),
                        recommended_positions,
                        time_horizon: FixedPoint::from_float(3.0), // 3 days
                        risk_level: short_return.abs(),
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        }
        
        Ok(opportunities)
    }
    
    /// Calculate correlation between two price series
    fn calculate_correlation(
        &self,
        series1: &[FixedPoint],
        series2: &[FixedPoint],
    ) -> Result<FixedPoint, MultiAssetError> {
        if series1.len() != series2.len() || series1.is_empty() {
            return Ok(FixedPoint::zero());
        }
        
        let n = series1.len();
        let n_fp = FixedPoint::from_int(n as i32);
        
        let mean1 = series1.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / n_fp;
        let mean2 = series2.iter().fold(FixedPoint::zero(), |acc, &x| acc + x) / n_fp;
        
        let mut covariance = FixedPoint::zero();
        let mut var1 = FixedPoint::zero();
        let mut var2 = FixedPoint::zero();
        
        for i in 0..n {
            let dev1 = series1[i] - mean1;
            let dev2 = series2[i] - mean2;
            
            covariance = covariance + dev1 * dev2;
            var1 = var1 + dev1 * dev1;
            var2 = var2 + dev2 * dev2;
        }
        
        let std1 = var1.sqrt();
        let std2 = var2.sqrt();
        
        if std1 > FixedPoint::zero() && std2 > FixedPoint::zero() {
            Ok(covariance / (std1 * std2))
        } else {
            Ok(FixedPoint::zero())
        }
    }
    
    /// Execute arbitrage opportunity with optimal timing
    pub fn execute_arbitrage_opportunity(
        &self,
        opportunity: &ArbitrageOpportunity,
        current_prices: &HashMap<AssetId, FixedPoint>,
        max_position_size: FixedPoint,
    ) -> Result<HashMap<AssetId, FixedPoint>, MultiAssetError> {
        let mut execution_positions = HashMap::new();
        
        // Scale positions based on confidence and risk
        let scale_factor = opportunity.confidence * (FixedPoint::one() - opportunity.risk_level);
        
        for (&asset, &recommended_position) in &opportunity.recommended_positions {
            let scaled_position = recommended_position * scale_factor;
            
            // Apply position size limits
            let final_position = if scaled_position.abs() > max_position_size {
                if scaled_position > FixedPoint::zero() {
                    max_position_size
                } else {
                    -max_position_size
                }
            } else {
                scaled_position
            };
            
            execution_positions.insert(asset, final_position);
        }
        
        println!("Executing {} arbitrage opportunity with {} assets, expected profit: {:.4}, confidence: {:.2}%",
                format!("{:?}", opportunity.arbitrage_type),
                opportunity.assets.len(),
                opportunity.expected_profit.to_float(),
                (opportunity.confidence * FixedPoint::from_float(100.0)).to_float());
        
        Ok(execution_positions)
    }
    
    /// Set minimum confidence threshold
    pub fn set_min_confidence_threshold(&mut self, threshold: FixedPoint) {
        self.min_confidence_threshold = threshold;
    }
    
    /// Set maximum risk tolerance
    pub fn set_max_risk_tolerance(&mut self, tolerance: FixedPoint) {
        self.max_risk_tolerance = tolerance;
    }
    
    /// Get current cointegration relationships
    pub fn get_cointegration_relationships(&self) -> &[CointegrationRelationship] {
        &self.cointegration_relationships
    }
}

/// Multi-asset value function
#[derive(Debug, Clone)]
pub struct MultiAssetValueFunction {
    /// State grid
    pub grid: MultiAssetStateGrid,
    /// Value function values u(t, q₁, S₁, q₂, S₂, ...)
    pub values: Vec<FixedPoint>,
    /// Gradient with respect to each variable
    pub gradients: HashMap<String, Vec<FixedPoint>>,
}

impl MultiAssetValueFunction {
    /// Create a new multi-asset value function
    pub fn new(grid: MultiAssetStateGrid) -> Self {
        let total_points = grid.total_points();
        
        Self {
            grid,
            values: vec![FixedPoint::zero(); total_points],
            gradients: HashMap::new(),
        }
    }
    
    /// Get value at specific state
    pub fn get_value(&self, indices: &HashMap<AssetId, (usize, usize)>, time_idx: usize) -> FixedPoint {
        let flat_index = self.grid.to_flat_index(indices, time_idx);
        if flat_index < self.values.len() {
            self.values[flat_index]
        } else {
            FixedPoint::zero()
        }
    }
    
    /// Set value at specific state
    pub fn set_value(&mut self, indices: &HashMap<AssetId, (usize, usize)>, time_idx: usize, value: FixedPoint) {
        let flat_index = self.grid.to_flat_index(indices, time_idx);
        if flat_index < self.values.len() {
            self.values[flat_index] = value;
        }
    }
}

/// Multi-dimensional HJB solver for portfolio optimization
pub struct MultiDimensionalHJBSolver {
    /// Multi-asset state grid
    pub grid: MultiAssetStateGrid,
    /// Boundary conditions for each asset
    pub boundary_conditions: HashMap<AssetId, BoundaryConditions>,
    /// Numerical scheme
    pub numerical_scheme: NumericalScheme,
    /// Time step size
    pub dt: FixedPoint,
    /// Convergence tolerance
    pub tolerance: FixedPoint,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Stability check parameters
    pub stability_threshold: FixedPoint,
    /// CFL condition parameter
    pub cfl_parameter: FixedPoint,
    /// Numerical damping coefficient
    pub damping_coefficient: FixedPoint,
}

impl MultiDimensionalHJBSolver {
    /// Create a new multi-dimensional HJB solver
    pub fn new(
        grid: MultiAssetStateGrid,
        boundary_conditions: HashMap<AssetId, BoundaryConditions>,
        numerical_scheme: NumericalScheme,
        dt: FixedPoint,
    ) -> Self {
        Self {
            grid,
            boundary_conditions,
            numerical_scheme,
            dt,
            tolerance: FixedPoint::from_float(1e-6),
            max_iterations: 10000,
            stability_threshold: FixedPoint::from_float(1e10),
            cfl_parameter: FixedPoint::from_float(0.5),
            damping_coefficient: FixedPoint::from_float(0.01),
        }
    }
    
    /// Create solver with custom stability parameters
    pub fn with_stability_params(
        mut self,
        stability_threshold: FixedPoint,
        cfl_parameter: FixedPoint,
        damping_coefficient: FixedPoint,
    ) -> Self {
        self.stability_threshold = stability_threshold;
        self.cfl_parameter = cfl_parameter;
        self.damping_coefficient = damping_coefficient;
        self
    }
    
    /// Validate CFL condition for numerical stability
    pub fn validate_cfl_condition(
        &self,
        params: &MultiAssetModelParameters,
    ) -> Result<(), MultiAssetError> {
        // Check CFL condition: dt ≤ CFL * min(dx²/(2*σ²))
        let mut min_stability_dt = FixedPoint::from_float(1e10);
        
        for &asset in &self.grid.asset_order {
            if let Some(price_grid) = self.grid.price_grids.get(&asset) {
                if price_grid.len() < 2 {
                    continue;
                }
                
                let dx = price_grid[1] - price_grid[0];
                let volatility = params.volatilities.get(&asset)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                
                let stability_dt = self.cfl_parameter * dx * dx / 
                    (FixedPoint::from_float(2.0) * volatility * volatility);
                
                if stability_dt < min_stability_dt {
                    min_stability_dt = stability_dt;
                }
            }
        }
        
        if self.dt > min_stability_dt {
            return Err(MultiAssetError::NumericalInstability(
                format!("CFL condition violated: dt={} > max_dt={}", 
                       self.dt.to_float(), min_stability_dt.to_float())
            ));
        }
        
        Ok(())
    }
    
    /// Check numerical stability of the solution
    pub fn check_numerical_stability(
        &self,
        value_function: &MultiAssetValueFunction,
    ) -> Result<(), MultiAssetError> {
        // Check for NaN or infinite values
        for &value in &value_function.values {
            if value.abs() > self.stability_threshold {
                return Err(MultiAssetError::NumericalInstability(
                    format!("Value function contains unstable values: {}", value.to_float())
                ));
            }
        }
        
        // Check for oscillations (simplified)
        let mut max_gradient = FixedPoint::zero();
        let state_combinations = self.generate_state_combinations();
        
        for state in state_combinations.iter().take(100) { // Sample check
            for t_idx in 1..self.grid.time_grid.len()-1 {
                let current_value = value_function.get_value(state, t_idx);
                let prev_value = value_function.get_value(state, t_idx - 1);
                let next_value = value_function.get_value(state, t_idx + 1);
                
                let gradient = (next_value - prev_value) / (FixedPoint::from_float(2.0) * self.dt);
                if gradient.abs() > max_gradient {
                    max_gradient = gradient.abs();
                }
            }
        }
        
        if max_gradient > self.stability_threshold / FixedPoint::from_float(100.0) {
            return Err(MultiAssetError::NumericalInstability(
                format!("Large gradients detected: {}", max_gradient.to_float())
            ));
        }
        
        Ok(())
    }
    
    /// Implement adaptive time stepping for better convergence
    pub fn solve_value_function_adaptive(
        &mut self,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<MultiAssetValueFunction, MultiAssetError> {
        // Validate CFL condition
        self.validate_cfl_condition(params)?;
        
        let mut value_function = MultiAssetValueFunction::new(self.grid.clone());
        
        // Initialize terminal condition
        self.apply_terminal_condition(&mut value_function, params)?;
        
        // Adaptive time stepping
        let time_steps = self.grid.time_grid.len();
        let mut adaptive_dt = self.dt;
        let mut convergence_history = Vec::new();
        
        for t_idx in (0..time_steps-1).rev() {
            let mut step_converged = false;
            let mut sub_iterations = 0;
            let max_sub_iterations = 10;
            
            while !step_converged && sub_iterations < max_sub_iterations {
                // Store previous values for convergence check
                let prev_values = value_function.values.clone();
                
                // Solve one time step with current dt
                self.solve_time_step_with_dt(
                    &mut value_function, t_idx, params, correlation_matrix, adaptive_dt
                )?;
                
                // Apply boundary conditions
                self.apply_boundary_conditions(&mut value_function, t_idx)?;
                
                // Check stability
                if let Err(_) = self.check_numerical_stability(&value_function) {
                    // Reduce time step and retry
                    adaptive_dt = adaptive_dt * FixedPoint::from_float(0.5);
                    value_function.values = prev_values;
                    sub_iterations += 1;
                    continue;
                }
                
                // Check local convergence
                let local_error = self.compute_local_error(&value_function.values, &prev_values);
                convergence_history.push(local_error);
                
                if local_error < self.tolerance {
                    step_converged = true;
                    
                    // Increase time step if converging well
                    if local_error < self.tolerance / FixedPoint::from_float(10.0) {
                        adaptive_dt = (adaptive_dt * FixedPoint::from_float(1.1)).min(self.dt);
                    }
                } else if local_error > self.tolerance * FixedPoint::from_float(10.0) {
                    // Reduce time step for better accuracy
                    adaptive_dt = adaptive_dt * FixedPoint::from_float(0.8);
                    value_function.values = prev_values;
                    sub_iterations += 1;
                } else {
                    step_converged = true;
                }
            }
            
            if !step_converged {
                return Err(MultiAssetError::NumericalInstability(
                    format!("Failed to converge at time step {}", t_idx)
                ));
            }
            
            // Progress reporting
            if t_idx % 100 == 0 {
                println!("Time step {}/{}, adaptive_dt={:.6}, local_error={:.2e}", 
                        time_steps - t_idx, time_steps, 
                        adaptive_dt.to_float(), 
                        convergence_history.last().unwrap_or(&FixedPoint::zero()).to_float());
            }
        }
        
        // Final stability check
        self.check_numerical_stability(&value_function)?;
        
        Ok(value_function)
    }
    
    /// Compute local error between iterations
    fn compute_local_error(&self, current: &[FixedPoint], previous: &[FixedPoint]) -> FixedPoint {
        if current.len() != previous.len() {
            return FixedPoint::from_float(1e10);
        }
        
        let mut max_error = FixedPoint::zero();
        let mut sum_squared_error = FixedPoint::zero();
        
        for (curr, prev) in current.iter().zip(previous.iter()) {
            let error = (*curr - *prev).abs();
            if error > max_error {
                max_error = error;
            }
            sum_squared_error = sum_squared_error + error * error;
        }
        
        // Return RMS error
        (sum_squared_error / FixedPoint::from_int(current.len() as i32)).sqrt()
    }
    
    /// Solve time step with specified dt
    fn solve_time_step_with_dt(
        &self,
        value_function: &mut MultiAssetValueFunction,
        t_idx: usize,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
        dt: FixedPoint,
    ) -> Result<(), MultiAssetError> {
        let state_combinations = self.generate_state_combinations();
        
        for state in state_combinations {
            // Skip boundary states
            if self.is_boundary_state(&state) {
                continue;
            }
            
            // Compute optimal controls at this state
            let optimal_controls = self.compute_optimal_controls(
                value_function, &state, t_idx, params, correlation_matrix
            )?;
            
            // Compute HJB operator based on numerical scheme
            let hjb_value = match self.numerical_scheme {
                NumericalScheme::ExplicitEuler => {
                    self.compute_explicit_hjb_operator(
                        value_function, &state, t_idx, &optimal_controls, params, correlation_matrix
                    )?
                }
                NumericalScheme::ImplicitEuler => {
                    self.compute_implicit_hjb_operator(
                        value_function, &state, t_idx, &optimal_controls, params, correlation_matrix
                    )?
                }
                NumericalScheme::CrankNicolson => {
                    self.compute_crank_nicolson_hjb_operator(
                        value_function, &state, t_idx, &optimal_controls, params, correlation_matrix
                    )?
                }
                NumericalScheme::Upwind => {
                    self.compute_upwind_hjb_operator(
                        value_function, &state, t_idx, &optimal_controls, params, correlation_matrix
                    )?
                }
            };
            
            // Update value function with damping for stability
            let current_value = value_function.get_value(&state, t_idx + 1);
            let update = dt * hjb_value;
            let damped_update = update * (FixedPoint::one() - self.damping_coefficient) + 
                               current_value * self.damping_coefficient;
            let new_value = current_value + damped_update;
            
            value_function.set_value(&state, t_idx, new_value);
        }
        
        Ok(())
    }
    
    /// Compute explicit Euler HJB operator
    fn compute_explicit_hjb_operator(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        controls: &HashMap<AssetId, (FixedPoint, FixedPoint)>,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Use values at current time level (explicit)
        self.compute_multi_asset_hjb_operator(
            value_function, state, t_idx + 1, controls, params, correlation_matrix
        )
    }
    
    /// Compute implicit Euler HJB operator
    fn compute_implicit_hjb_operator(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        controls: &HashMap<AssetId, (FixedPoint, FixedPoint)>,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Use values at next time level (implicit) - simplified implementation
        // In practice, would solve linear system
        self.compute_multi_asset_hjb_operator(
            value_function, state, t_idx, controls, params, correlation_matrix
        )
    }
    
    /// Compute Crank-Nicolson HJB operator
    fn compute_crank_nicolson_hjb_operator(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        controls: &HashMap<AssetId, (FixedPoint, FixedPoint)>,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Average of explicit and implicit (simplified)
        let explicit_term = self.compute_multi_asset_hjb_operator(
            value_function, state, t_idx + 1, controls, params, correlation_matrix
        )?;
        
        let implicit_term = self.compute_multi_asset_hjb_operator(
            value_function, state, t_idx, controls, params, correlation_matrix
        )?;
        
        Ok((explicit_term + implicit_term) / FixedPoint::from_float(2.0))
    }
    
    /// Compute upwind HJB operator
    fn compute_upwind_hjb_operator(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        controls: &HashMap<AssetId, (FixedPoint, FixedPoint)>,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Use upwind differences for convection terms
        // For simplicity, fall back to explicit scheme
        self.compute_explicit_hjb_operator(
            value_function, state, t_idx, controls, params, correlation_matrix
        )
    }
    
    /// Solve the multi-dimensional HJB equation
    pub fn solve_value_function(
        &self,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<MultiAssetValueFunction, MultiAssetError> {
        let mut value_function = MultiAssetValueFunction::new(self.grid.clone());
        
        // Initialize terminal condition
        self.apply_terminal_condition(&mut value_function, params)?;
        
        // Time stepping (backward in time)
        let time_steps = self.grid.time_grid.len();
        
        for t_idx in (0..time_steps-1).rev() {
            // Solve one time step
            self.solve_time_step(&mut value_function, t_idx, params, correlation_matrix)?;
            
            // Apply boundary conditions
            self.apply_boundary_conditions(&mut value_function, t_idx)?;
            
            // Check convergence (simplified)
            if t_idx % 10 == 0 {
                let convergence_metric = self.compute_convergence_metric(&value_function, t_idx)?;
                if convergence_metric < self.tolerance {
                    println!("Converged at time step {}", t_idx);
                    break;
                }
            }
        }
        
        Ok(value_function)
    }
    
    /// Apply terminal condition for multi-asset portfolio
    fn apply_terminal_condition(
        &self,
        value_function: &mut MultiAssetValueFunction,
        params: &MultiAssetModelParameters,
    ) -> Result<(), MultiAssetError> {
        let t_final = self.grid.time_grid.len() - 1;
        
        // Generate all possible state combinations
        let state_combinations = self.generate_state_combinations();
        
        for state in state_combinations {
            // Terminal penalty: -γ/2 * q^T Σ q (portfolio risk)
            let portfolio_risk = self.compute_portfolio_risk(&state, params)?;
            let terminal_value = -params.risk_aversion * portfolio_risk / FixedPoint::from_float(2.0);
            
            value_function.set_value(&state, t_final, terminal_value);
        }
        
        Ok(())
    }
    
    /// Generate all possible state combinations for the grid
    fn generate_state_combinations(&self) -> Vec<HashMap<AssetId, (usize, usize)>> {
        let mut combinations = Vec::new();
        
        // This is a simplified version - in practice would use iterative generation
        // for large state spaces to avoid memory issues
        
        if self.grid.asset_order.len() == 2 {
            // Two-asset case for demonstration
            let asset1 = self.grid.asset_order[0];
            let asset2 = self.grid.asset_order[1];
            
            if let (Some(inv1), Some(price1), Some(inv2), Some(price2)) = (
                self.grid.inventory_grids.get(&asset1),
                self.grid.price_grids.get(&asset1),
                self.grid.inventory_grids.get(&asset2),
                self.grid.price_grids.get(&asset2),
            ) {
                for i1 in 0..inv1.len() {
                    for p1 in 0..price1.len() {
                        for i2 in 0..inv2.len() {
                            for p2 in 0..price2.len() {
                                let mut state = HashMap::new();
                                state.insert(asset1, (i1, p1));
                                state.insert(asset2, (i2, p2));
                                combinations.push(state);
                            }
                        }
                    }
                }
            }
        }
        
        combinations
    }
    
    /// Compute portfolio risk q^T Σ q
    fn compute_portfolio_risk(
        &self,
        state: &HashMap<AssetId, (usize, usize)>,
        params: &MultiAssetModelParameters,
    ) -> Result<FixedPoint, MultiAssetError> {
        let mut risk = FixedPoint::zero();
        
        // Get inventory vector
        let mut inventories = Vec::new();
        for &asset in &self.grid.asset_order {
            if let Some(&(inv_idx, _)) = state.get(&asset) {
                if let Some(inv_grid) = self.grid.inventory_grids.get(&asset) {
                    inventories.push(inv_grid[inv_idx]);
                }
            }
        }
        
        // Compute q^T Σ q where Σ is the covariance matrix
        for (i, &qi) in inventories.iter().enumerate() {
            for (j, &qj) in inventories.iter().enumerate() {
                if let (Some(&asset_i), Some(&asset_j)) = 
                    (self.grid.asset_order.get(i), self.grid.asset_order.get(j)) {
                    
                    // Get volatilities
                    let vol_i = params.volatilities.get(&asset_i)
                        .copied().unwrap_or(FixedPoint::from_float(0.2));
                    let vol_j = params.volatilities.get(&asset_j)
                        .copied().unwrap_or(FixedPoint::from_float(0.2));
                    
                    // Get correlation (simplified - would use correlation matrix)
                    let correlation = if i == j { 
                        FixedPoint::one() 
                    } else { 
                        params.default_correlation 
                    };
                    
                    // Covariance = σᵢ * σⱼ * ρᵢⱼ
                    let covariance = vol_i * vol_j * correlation;
                    
                    risk = risk + qi * qj * covariance;
                }
            }
        }
        
        Ok(risk)
    }
    
    /// Solve one time step of the multi-dimensional HJB equation
    fn solve_time_step(
        &self,
        value_function: &mut MultiAssetValueFunction,
        t_idx: usize,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<(), MultiAssetError> {
        let state_combinations = self.generate_state_combinations();
        
        for state in state_combinations {
            // Skip boundary states (simplified)
            if self.is_boundary_state(&state) {
                continue;
            }
            
            // Compute optimal controls at this state
            let optimal_controls = self.compute_optimal_controls(
                value_function, &state, t_idx, params, correlation_matrix
            )?;
            
            // Compute HJB operator
            let hjb_value = self.compute_multi_asset_hjb_operator(
                value_function, &state, t_idx, &optimal_controls, params, correlation_matrix
            )?;
            
            // Update value function
            let current_value = value_function.get_value(&state, t_idx + 1);
            let new_value = current_value + self.dt * hjb_value;
            value_function.set_value(&state, t_idx, new_value);
        }
        
        Ok(())
    }
    
    /// Check if a state is on the boundary
    fn is_boundary_state(&self, state: &HashMap<AssetId, (usize, usize)>) -> bool {
        for &asset in &self.grid.asset_order {
            if let Some(&(inv_idx, price_idx)) = state.get(&asset) {
                if let (Some(inv_grid), Some(price_grid)) = 
                    (self.grid.inventory_grids.get(&asset), self.grid.price_grids.get(&asset)) {
                    
                    if inv_idx == 0 || inv_idx == inv_grid.len() - 1 ||
                       price_idx == 0 || price_idx == price_grid.len() - 1 {
                        return true;
                    }
                }
            }
        }
        false
    }
    
    /// Compute optimal controls for multi-asset case
    fn compute_optimal_controls(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<HashMap<AssetId, (FixedPoint, FixedPoint)>, MultiAssetError> {
        let mut controls = HashMap::new();
        
        for &asset in &self.grid.asset_order {
            // Compute derivatives for this asset
            let dv_dq = self.compute_inventory_derivative(value_function, state, t_idx, asset)?;
            
            // Get current inventory
            let inventory = if let Some(&(inv_idx, _)) = state.get(&asset) {
                if let Some(inv_grid) = self.grid.inventory_grids.get(&asset) {
                    inv_grid[inv_idx]
                } else {
                    FixedPoint::zero()
                }
            } else {
                FixedPoint::zero()
            };
            
            // Compute cross-asset effects
            let cross_asset_adjustment = self.compute_cross_asset_adjustment(
                state, asset, params, correlation_matrix
            )?;
            
            // Optimal spreads with cross-asset effects
            let base_spread = params.adverse_selection_costs.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.001));
            
            let inventory_penalty = params.inventory_penalties.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.01));
            
            let bid_spread = base_spread + inventory_penalty * inventory.abs() + cross_asset_adjustment;
            let ask_spread = base_spread + inventory_penalty * inventory.abs() + cross_asset_adjustment;
            
            controls.insert(asset, (
                bid_spread.max(FixedPoint::from_float(0.0001)),
                ask_spread.max(FixedPoint::from_float(0.0001))
            ));
        }
        
        Ok(controls)
    }
    
    /// Compute cross-asset adjustment based on correlations
    fn compute_cross_asset_adjustment(
        &self,
        state: &HashMap<AssetId, (usize, usize)>,
        target_asset: AssetId,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        let mut adjustment = FixedPoint::zero();
        
        for &other_asset in &self.grid.asset_order {
            if other_asset == target_asset {
                continue;
            }
            
            // Get correlation
            let correlation = correlation_matrix.get_correlation(target_asset, other_asset)
                .unwrap_or(params.default_correlation);
            
            // Get other asset's inventory
            let other_inventory = if let Some(&(inv_idx, _)) = state.get(&other_asset) {
                if let Some(inv_grid) = self.grid.inventory_grids.get(&other_asset) {
                    inv_grid[inv_idx]
                } else {
                    FixedPoint::zero()
                }
            } else {
                FixedPoint::zero()
            };
            
            // Cross-asset adjustment proportional to correlation and other inventory
            let cross_penalty = params.cross_asset_penalty;
            adjustment = adjustment + cross_penalty * correlation.abs() * other_inventory.abs();
        }
        
        Ok(adjustment)
    }
    
    /// Compute multi-asset HJB operator
    fn compute_multi_asset_hjb_operator(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        controls: &HashMap<AssetId, (FixedPoint, FixedPoint)>,
        params: &MultiAssetModelParameters,
        correlation_matrix: &CorrelationMatrix,
    ) -> Result<FixedPoint, MultiAssetError> {
        let mut hjb_operator = FixedPoint::zero();
        
        // Time derivative term
        let dv_dt = self.compute_time_derivative(value_function, state, t_idx)?;
        hjb_operator = hjb_operator + dv_dt;
        
        // Diffusion terms for each asset
        for &asset in &self.grid.asset_order {
            let price = if let Some(&(_, price_idx)) = state.get(&asset) {
                if let Some(price_grid) = self.grid.price_grids.get(&asset) {
                    price_grid[price_idx]
                } else {
                    FixedPoint::from_float(100.0)
                }
            } else {
                FixedPoint::from_float(100.0)
            };
            
            let volatility = params.volatilities.get(&asset)
                .copied().unwrap_or(FixedPoint::from_float(0.2));
            
            let d2v_ds2 = self.compute_second_price_derivative(value_function, state, t_idx, asset)?;
            
            // Diffusion term: (1/2)σᵢ²Sᵢ² ∂²V/∂Sᵢ²
            let diffusion_term = volatility * volatility * price * price * d2v_ds2 / FixedPoint::from_float(2.0);
            hjb_operator = hjb_operator + diffusion_term;
        }
        
        // Cross-diffusion terms: σᵢσⱼρᵢⱼSᵢSⱼ ∂²V/∂Sᵢ∂Sⱼ
        for (i, &asset_i) in self.grid.asset_order.iter().enumerate() {
            for (j, &asset_j) in self.grid.asset_order.iter().enumerate() {
                if i >= j { continue; } // Only compute upper triangle
                
                let correlation = correlation_matrix.get_correlation(asset_i, asset_j)
                    .unwrap_or(params.default_correlation);
                
                let vol_i = params.volatilities.get(&asset_i)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                let vol_j = params.volatilities.get(&asset_j)
                    .copied().unwrap_or(FixedPoint::from_float(0.2));
                
                let price_i = if let Some(&(_, price_idx)) = state.get(&asset_i) {
                    if let Some(price_grid) = self.grid.price_grids.get(&asset_i) {
                        price_grid[price_idx]
                    } else {
                        FixedPoint::from_float(100.0)
                    }
                } else {
                    FixedPoint::from_float(100.0)
                };
                
                let price_j = if let Some(&(_, price_idx)) = state.get(&asset_j) {
                    if let Some(price_grid) = self.grid.price_grids.get(&asset_j) {
                        price_grid[price_idx]
                    } else {
                        FixedPoint::from_float(100.0)
                    }
                } else {
                    FixedPoint::from_float(100.0)
                };
                
                // Cross derivative (simplified - would need proper finite difference)
                let d2v_dsids = self.compute_cross_price_derivative(
                    value_function, state, t_idx, asset_i, asset_j
                )?;
                
                let cross_diffusion = vol_i * vol_j * correlation * price_i * price_j * d2v_dsids;
                hjb_operator = hjb_operator + cross_diffusion;
            }
        }
        
        // Market making utility terms for each asset
        for &asset in &self.grid.asset_order {
            if let Some(&(bid_spread, ask_spread)) = controls.get(&asset) {
                let dv_dq = self.compute_inventory_derivative(value_function, state, t_idx, asset)?;
                
                let bid_intensity = self.compute_arrival_intensity(bid_spread, params, asset);
                let ask_intensity = self.compute_arrival_intensity(ask_spread, params, asset);
                
                let bid_utility = bid_intensity * (bid_spread + dv_dq);
                let ask_utility = ask_intensity * (ask_spread - dv_dq);
                
                hjb_operator = hjb_operator + bid_utility + ask_utility;
            }
        }
        
        // Portfolio risk penalty
        let portfolio_risk = self.compute_portfolio_risk(state, params)?;
        let risk_penalty = -params.risk_aversion * portfolio_risk / FixedPoint::from_float(2.0);
        hjb_operator = hjb_operator + risk_penalty;
        
        Ok(-hjb_operator) // Negative because solving backward in time
    }
    
    /// Compute arrival intensity for an asset
    fn compute_arrival_intensity(&self, spread: FixedPoint, params: &MultiAssetModelParameters, asset: AssetId) -> FixedPoint {
        let base_intensity = params.base_intensities.get(&asset)
            .copied().unwrap_or(FixedPoint::from_float(1.0));
        let decay_rate = params.adverse_selection_costs.get(&asset)
            .copied().unwrap_or(FixedPoint::from_float(0.1));
        
        base_intensity * (-decay_rate * spread).exp()
    }
    
    /// Compute time derivative using finite differences
    fn compute_time_derivative(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
    ) -> Result<FixedPoint, MultiAssetError> {
        if t_idx == 0 || t_idx >= self.grid.time_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let v_plus = value_function.get_value(state, t_idx + 1);
        let v_minus = value_function.get_value(state, t_idx - 1);
        let dt = self.grid.time_grid[t_idx + 1] - self.grid.time_grid[t_idx - 1];
        
        Ok((v_plus - v_minus) / dt)
    }
    
    /// Compute inventory derivative for specific asset
    fn compute_inventory_derivative(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        asset: AssetId,
    ) -> Result<FixedPoint, MultiAssetError> {
        if let Some(&(inv_idx, price_idx)) = state.get(&asset) {
            if let Some(inv_grid) = self.grid.inventory_grids.get(&asset) {
                if inv_idx == 0 || inv_idx >= inv_grid.len() - 1 {
                    return Ok(FixedPoint::zero());
                }
                
                // Create states with perturbed inventory
                let mut state_plus = state.clone();
                let mut state_minus = state.clone();
                state_plus.insert(asset, (inv_idx + 1, price_idx));
                state_minus.insert(asset, (inv_idx - 1, price_idx));
                
                let v_plus = value_function.get_value(&state_plus, t_idx);
                let v_minus = value_function.get_value(&state_minus, t_idx);
                let dq = inv_grid[inv_idx + 1] - inv_grid[inv_idx - 1];
                
                return Ok((v_plus - v_minus) / dq);
            }
        }
        
        Ok(FixedPoint::zero())
    }
    
    /// Compute second price derivative for specific asset
    fn compute_second_price_derivative(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        asset: AssetId,
    ) -> Result<FixedPoint, MultiAssetError> {
        if let Some(&(inv_idx, price_idx)) = state.get(&asset) {
            if let Some(price_grid) = self.grid.price_grids.get(&asset) {
                if price_idx == 0 || price_idx >= price_grid.len() - 1 {
                    return Ok(FixedPoint::zero());
                }
                
                // Create states with perturbed prices
                let mut state_plus = state.clone();
                let mut state_minus = state.clone();
                state_plus.insert(asset, (inv_idx, price_idx + 1));
                state_minus.insert(asset, (inv_idx, price_idx - 1));
                
                let v_plus = value_function.get_value(&state_plus, t_idx);
                let v_center = value_function.get_value(state, t_idx);
                let v_minus = value_function.get_value(&state_minus, t_idx);
                
                let ds = price_grid[price_idx + 1] - price_grid[price_idx];
                
                return Ok((v_plus - FixedPoint::from_float(2.0) * v_center + v_minus) / (ds * ds));
            }
        }
        
        Ok(FixedPoint::zero())
    }
    
    /// Compute cross price derivative (simplified)
    fn compute_cross_price_derivative(
        &self,
        value_function: &MultiAssetValueFunction,
        state: &HashMap<AssetId, (usize, usize)>,
        t_idx: usize,
        asset_i: AssetId,
        asset_j: AssetId,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Simplified cross derivative computation
        // In practice would use more sophisticated finite difference schemes
        
        if let (Some(&(inv_i, price_i)), Some(&(inv_j, price_j))) = 
            (state.get(&asset_i), state.get(&asset_j)) {
            
            if let (Some(price_grid_i), Some(price_grid_j)) = 
                (self.grid.price_grids.get(&asset_i), self.grid.price_grids.get(&asset_j)) {
                
                if price_i == 0 || price_i >= price_grid_i.len() - 1 ||
                   price_j == 0 || price_j >= price_grid_j.len() - 1 {
                    return Ok(FixedPoint::zero());
                }
                
                // Four-point finite difference for cross derivative
                let mut state_pp = state.clone();
                let mut state_pm = state.clone();
                let mut state_mp = state.clone();
                let mut state_mm = state.clone();
                
                state_pp.insert(asset_i, (inv_i, price_i + 1));
                state_pp.insert(asset_j, (inv_j, price_j + 1));
                
                state_pm.insert(asset_i, (inv_i, price_i + 1));
                state_pm.insert(asset_j, (inv_j, price_j - 1));
                
                state_mp.insert(asset_i, (inv_i, price_i - 1));
                state_mp.insert(asset_j, (inv_j, price_j + 1));
                
                state_mm.insert(asset_i, (inv_i, price_i - 1));
                state_mm.insert(asset_j, (inv_j, price_j - 1));
                
                let v_pp = value_function.get_value(&state_pp, t_idx);
                let v_pm = value_function.get_value(&state_pm, t_idx);
                let v_mp = value_function.get_value(&state_mp, t_idx);
                let v_mm = value_function.get_value(&state_mm, t_idx);
                
                let ds_i = price_grid_i[price_i + 1] - price_grid_i[price_i - 1];
                let ds_j = price_grid_j[price_j + 1] - price_grid_j[price_j - 1];
                
                let cross_derivative = (v_pp - v_pm - v_mp + v_mm) / (ds_i * ds_j);
                
                return Ok(cross_derivative);
            }
        }
        
        Ok(FixedPoint::zero())
    }
    
    /// Apply boundary conditions for all assets
    fn apply_boundary_conditions(
        &self,
        value_function: &mut MultiAssetValueFunction,
        t_idx: usize,
    ) -> Result<(), MultiAssetError> {
        // Simplified boundary condition application
        // In practice would handle each boundary face separately
        
        let boundary_states = self.get_boundary_states();
        
        for state in boundary_states {
            // Apply Neumann boundary conditions (zero derivative)
            let interior_state = self.get_nearest_interior_state(&state)?;
            let boundary_value = value_function.get_value(&interior_state, t_idx);
            value_function.set_value(&state, t_idx, boundary_value);
        }
        
        Ok(())
    }
    
    /// Get all boundary states
    fn get_boundary_states(&self) -> Vec<HashMap<AssetId, (usize, usize)>> {
        let mut boundary_states = Vec::new();
        
        // This is simplified - would generate all boundary combinations
        let all_states = self.generate_state_combinations();
        
        for state in all_states {
            if self.is_boundary_state(&state) {
                boundary_states.push(state);
            }
        }
        
        boundary_states
    }
    
    /// Get nearest interior state for boundary condition
    fn get_nearest_interior_state(
        &self,
        boundary_state: &HashMap<AssetId, (usize, usize)>
    ) -> Result<HashMap<AssetId, (usize, usize)>, MultiAssetError> {
        let mut interior_state = boundary_state.clone();
        
        for &asset in &self.grid.asset_order {
            if let Some(&(inv_idx, price_idx)) = boundary_state.get(&asset) {
                if let (Some(inv_grid), Some(price_grid)) = 
                    (self.grid.inventory_grids.get(&asset), self.grid.price_grids.get(&asset)) {
                    
                    let new_inv_idx = if inv_idx == 0 { 1 } 
                                     else if inv_idx == inv_grid.len() - 1 { inv_grid.len() - 2 } 
                                     else { inv_idx };
                    
                    let new_price_idx = if price_idx == 0 { 1 } 
                                       else if price_idx == price_grid.len() - 1 { price_grid.len() - 2 } 
                                       else { price_idx };
                    
                    interior_state.insert(asset, (new_inv_idx, new_price_idx));
                }
            }
        }
        
        Ok(interior_state)
    }
    
    /// Compute convergence metric
    fn compute_convergence_metric(
        &self,
        value_function: &MultiAssetValueFunction,
        t_idx: usize,
    ) -> Result<FixedPoint, MultiAssetError> {
        // Simplified convergence check - would use more sophisticated metrics
        if t_idx >= self.grid.time_grid.len() - 1 {
            return Ok(FixedPoint::zero());
        }
        
        let states = self.generate_state_combinations();
        let mut max_change = FixedPoint::zero();
        
        for state in states.iter().take(100) { // Sample for efficiency
            let current_value = value_function.get_value(state, t_idx);
            let previous_value = value_function.get_value(state, t_idx + 1);
            let change = (current_value - previous_value).abs();
            
            if change > max_change {
                max_change = change;
            }
        }
        
        Ok(max_change)
    }
}

/// Multi-asset model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAssetModelParameters {
    /// Risk aversion parameter γ
    pub risk_aversion: FixedPoint,
    /// Volatilities for each asset σᵢ
    pub volatilities: HashMap<AssetId, FixedPoint>,
    /// Inventory penalties for each asset
    pub inventory_penalties: HashMap<AssetId, FixedPoint>,
    /// Adverse selection costs for each asset
    pub adverse_selection_costs: HashMap<AssetId, FixedPoint>,
    /// Base arrival intensities for each asset
    pub base_intensities: HashMap<AssetId, FixedPoint>,
    /// Default correlation for missing pairs
    pub default_correlation: FixedPoint,
    /// Cross-asset penalty coefficient
    pub cross_asset_penalty: FixedPoint,
    /// Terminal time T
    pub terminal_time: FixedPoint,
    /// Maximum inventory limits
    pub max_inventories: HashMap<AssetId, FixedPoint>,
    /// Market impact parameters
    pub market_impact_params: HashMap<AssetId, MarketImpactParams>,
}

/// Market impact parameters for an asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpactParams {
    /// Temporary impact coefficient η
    pub temporary_impact: FixedPoint,
    /// Permanent impact coefficient λ
    pub permanent_impact: FixedPoint,
    /// Impact exponent α
    pub impact_exponent: FixedPoint,
}

impl Default for MultiAssetModelParameters {
    fn default() -> Self {
        Self {
            risk_aversion: FixedPoint::from_float(1.0),
            volatilities: HashMap::new(),
            inventory_penalties: HashMap::new(),
            adverse_selection_costs: HashMap::new(),
            base_intensities: HashMap::new(),
            default_correlation: FixedPoint::from_float(0.3),
            cross_asset_penalty: FixedPoint::from_float(0.01),
            terminal_time: FixedPoint::from_float(1.0),
            max_inventories: HashMap::new(),
            market_impact_params: HashMap::new(),
        }
    }
}

impl MultiAssetModelParameters {
    /// Add asset with default parameters
    pub fn add_asset(&mut self, asset_id: AssetId) -> &mut Self {
        self.volatilities.insert(asset_id, FixedPoint::from_float(0.2));
        self.adverse_selection_costs.insert(asset_id, FixedPoint::from_float(0.001));
        self.inventory_penalties.insert(asset_id, FixedPoint::from_float(0.01));
        self.base_intensities.insert(asset_id, FixedPoint::from_float(1.0));
        self.max_inventories.insert(asset_id, FixedPoint::from_float(1000.0));
        self.market_impact_params.insert(asset_id, MarketImpactParams {
            temporary_impact: FixedPoint::from_float(0.1),
            permanent_impact: FixedPoint::from_float(0.01),
            impact_exponent: FixedPoint::from_float(0.5),
        });
        self
    }
    
    /// Set volatility for an asset
    pub fn set_volatility(&mut self, asset_id: AssetId, volatility: FixedPoint) -> &mut Self {
        self.volatilities.insert(asset_id, volatility);
        self
    }
    
    /// Set adverse selection cost for an asset
    pub fn set_adverse_selection_cost(&mut self, asset_id: AssetId, cost: FixedPoint) -> &mut Self {
        self.adverse_selection_costs.insert(asset_id, cost);
        self
    }
    
    /// Set inventory penalty for an asset
    pub fn set_inventory_penalty(&mut self, asset_id: AssetId, penalty: FixedPoint) -> &mut Self {
        self.inventory_penalties.insert(asset_id, penalty);
        self
    }
    
    /// Set base intensity for an asset
    pub fn set_base_intensity(&mut self, asset_id: AssetId, intensity: FixedPoint) -> &mut Self {
        self.base_intensities.insert(asset_id, intensity);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_correlation_matrix_validation() {
        let assets = vec![1, 2];
        let mut corr_matrix = CorrelationMatrix::new(assets);
        
        // Set valid correlation
        corr_matrix.set_correlation(1, 2, FixedPoint::from_float(0.5)).unwrap();
        assert!(corr_matrix.validate().is_ok());
        
        // Set invalid correlation
        corr_matrix.set_correlation(1, 2, FixedPoint::from_float(1.5)).unwrap();
        assert!(corr_matrix.validate().is_err());
    }
    
    #[test]
    fn test_multi_asset_state_grid() {
        let assets = vec![1, 2];
        let mut inventory_ranges = HashMap::new();
        let mut inventory_steps = HashMap::new();
        let mut price_ranges = HashMap::new();
        let mut price_steps = HashMap::new();
        
        inventory_ranges.insert(1, (FixedPoint::from_float(-100.0), FixedPoint::from_float(100.0)));
        inventory_ranges.insert(2, (FixedPoint::from_float(-50.0), FixedPoint::from_float(50.0)));
        inventory_steps.insert(1, 21);
        inventory_steps.insert(2, 11);
        
        price_ranges.insert(1, (FixedPoint::from_float(90.0), FixedPoint::from_float(110.0)));
        price_ranges.insert(2, (FixedPoint::from_float(45.0), FixedPoint::from_float(55.0)));
        price_steps.insert(1, 21);
        price_steps.insert(2, 11);
        
        let grid = MultiAssetStateGrid::new(
            assets,
            inventory_ranges,
            inventory_steps,
            (FixedPoint::zero(), FixedPoint::one()),
            11,
            price_ranges,
            price_steps,
        ).unwrap();
        
        assert_eq!(grid.asset_order.len(), 2);
        assert!(grid.total_points() > 0);
    }
    
    #[test]
    fn test_multi_dimensional_hjb_solver_creation() {
        let assets = vec![1, 2];
        let mut inventory_ranges = HashMap::new();
        let mut inventory_steps = HashMap::new();
        let mut price_ranges = HashMap::new();
        let mut price_steps = HashMap::new();
        
        inventory_ranges.insert(1, (FixedPoint::from_float(-10.0), FixedPoint::from_float(10.0)));
        inventory_ranges.insert(2, (FixedPoint::from_float(-10.0), FixedPoint::from_float(10.0)));
        inventory_steps.insert(1, 5);
        inventory_steps.insert(2, 5);
        
        price_ranges.insert(1, (FixedPoint::from_float(95.0), FixedPoint::from_float(105.0)));
        price_ranges.insert(2, (FixedPoint::from_float(95.0), FixedPoint::from_float(105.0)));
        price_steps.insert(1, 5);
        price_steps.insert(2, 5);
        
        let grid = MultiAssetStateGrid::new(
            assets,
            inventory_ranges,
            inventory_steps,
            (FixedPoint::zero(), FixedPoint::from_float(0.1)),
            3,
            price_ranges,
            price_steps,
        ).unwrap();
        
        let boundary_conditions = HashMap::new();
        
        let solver = MultiDimensionalHJBSolver::new(
            grid,
            boundary_conditions,
            NumericalScheme::ImplicitEuler,
            FixedPoint::from_float(0.01),
        );
        
        assert_eq!(solver.numerical_scheme, NumericalScheme::ImplicitEuler);
        assert_eq!(solver.dt, FixedPoint::from_float(0.01));
    }
}