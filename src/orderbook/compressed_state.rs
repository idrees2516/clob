//! Compressed State Representation for zkVM-Optimized Order Book
//! 
//! This module implements efficient state compression and Merkle tree structures
//! for the order book to minimize data availability costs and enable efficient
//! zkVM execution with deterministic state transitions.

use crate::math::fixed_point::{FixedPoint, MerkleNode, MerkleTree};
use crate::orderbook::types::{Order, Trade, OrderBook};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::{HashMap, BTreeMap};
use thiserror::Error;
use bincode;

#[derive(Error, Debug)]
pub enum StateError {
    #[error("Invalid state transition: {0}")]
    InvalidTransition(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),
    #[error("Merkle tree error: {0}")]
    MerkleError(String),
    #[error("Compression error: {0}")]
    CompressionError(String),
    #[error("State verification failed: {0}")]
    VerificationFailed(String),
}

/// Fixed-point scaling constants for deterministic arithmetic
pub const PRICE_SCALE: u64 = 1_000_000_000_000_000_000; // 18 decimal places
pub const VOLUME_SCALE: u64 = 1_000_000_000_000_000_000; // 18 decimal places
pub const TIMESTAMP_SCALE: u32 = 1000; // Millisecond precision

/// Compressed representation of a price level for zkVM efficiency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompressedPriceLevel {
    /// Fixed-point price representation (scaled by PRICE_SCALE)
    pub price: u64,
    /// Fixed-point volume representation (scaled by VOLUME_SCALE)
    pub volume: u64,
    /// Number of orders at this price level
    pub order_count: u16,
    /// Compressed timestamp (seconds since epoch)
    pub timestamp: u32,
    /// Hash of all orders at this level for verification
    pub orders_hash: [u8; 32],
}

impl CompressedPriceLevel {
    /// Create a new compressed price level from order book data
    pub fn from_orders(price: u64, orders: &[Order], timestamp: u64) -> Self {
        let total_volume = orders.iter().map(|o| o.size).sum();
        let order_count = orders.len() as u16;
        
        // Create hash of all orders for verification
        let mut hasher = Sha256::new();
        for order in orders {
            hasher.update(&order.id.to_be_bytes());
            hasher.update(&order.size.to_be_bytes());
            hasher.update(&order.timestamp.to_be_bytes());
        }
        let orders_hash = hasher.finalize().into();
        
        Self {
            price,
            volume: total_volume,
            order_count,
            timestamp: (timestamp / 1000) as u32, // Convert to seconds
            orders_hash,
        }
    }
    
    /// Convert to bytes for hashing
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    /// Check if this level is empty
    pub fn is_empty(&self) -> bool {
        self.volume == 0 || self.order_count == 0
    }
}

/// Compressed representation of bid or ask tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedPriceTree {
    /// Sorted price levels (ascending for asks, descending for bids)
    pub levels: Vec<CompressedPriceLevel>,
    /// Total volume across all levels
    pub total_volume: u64,
    /// Number of active price levels
    pub level_count: u16,
    /// Merkle root of all price levels
    pub merkle_root: [u8; 32],
}

impl CompressedPriceTree {
    /// Create a new compressed price tree
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
            total_volume: 0,
            level_count: 0,
            merkle_root: [0u8; 32],
        }
    }
    
    /// Build compressed tree from order book side
    pub fn from_order_book_side(
        price_levels: &BTreeMap<u64, crate::orderbook::types::PriceLevel>,
        is_bid_side: bool,
    ) -> Result<Self, StateError> {
        let mut levels = Vec::new();
        let mut total_volume = 0u64;
        
        // Convert price levels to compressed format
        let sorted_levels: Vec<_> = if is_bid_side {
            // Bids: highest price first (descending)
            price_levels.iter().rev().collect()
        } else {
            // Asks: lowest price first (ascending)
            price_levels.iter().collect()
        };
        
        for (&price, level) in sorted_levels {
            if !level.is_empty() {
                let orders: Vec<Order> = level.orders.iter().cloned().collect();
                let compressed_level = CompressedPriceLevel::from_orders(
                    price,
                    &orders,
                    level.timestamp,
                );
                
                total_volume += compressed_level.volume;
                levels.push(compressed_level);
            }
        }
        
        let level_count = levels.len() as u16;
        
        // Compute Merkle root
        let merkle_root = Self::compute_merkle_root(&levels)?;
        
        Ok(Self {
            levels,
            total_volume,
            level_count,
            merkle_root,
        })
    }
    
    /// Compute Merkle root of price levels
    fn compute_merkle_root(levels: &[CompressedPriceLevel]) -> Result<[u8; 32], StateError> {
        if levels.is_empty() {
            return Ok([0u8; 32]);
        }
        
        // Create leaf nodes from price levels
        let leaves: Vec<MerkleNode> = levels
            .iter()
            .map(|level| MerkleNode::new_leaf(&level.to_bytes()))
            .collect();
        
        // Build Merkle tree
        let tree = MerkleTree::from_leaves(leaves);
        Ok(tree.root_hash())
    }
    
    /// Add a new price level
    pub fn add_level(&mut self, level: CompressedPriceLevel) -> Result<(), StateError> {
        if level.is_empty() {
            return Ok(());
        }
        
        // Insert in sorted order
        let insert_pos = self.levels
            .binary_search_by_key(&level.price, |l| l.price)
            .unwrap_or_else(|pos| pos);
        
        self.levels.insert(insert_pos, level.clone());
        self.total_volume += level.volume;
        self.level_count += 1;
        
        // Recompute Merkle root
        self.merkle_root = Self::compute_merkle_root(&self.levels)?;
        
        Ok(())
    }
    
    /// Remove a price level
    pub fn remove_level(&mut self, price: u64) -> Result<Option<CompressedPriceLevel>, StateError> {
        if let Some(pos) = self.levels.iter().position(|l| l.price == price) {
            let removed = self.levels.remove(pos);
            self.total_volume = self.total_volume.saturating_sub(removed.volume);
            self.level_count = self.level_count.saturating_sub(1);
            
            // Recompute Merkle root
            self.merkle_root = Self::compute_merkle_root(&self.levels)?;
            
            Ok(Some(removed))
        } else {
            Ok(None)
        }
    }
    
    /// Update an existing price level
    pub fn update_level(&mut self, price: u64, new_level: CompressedPriceLevel) -> Result<(), StateError> {
        if let Some(pos) = self.levels.iter().position(|l| l.price == price) {
            let old_volume = self.levels[pos].volume;
            self.levels[pos] = new_level;
            
            // Update total volume
            self.total_volume = self.total_volume.saturating_sub(old_volume) + new_level.volume;
            
            // Remove level if empty
            if new_level.is_empty() {
                self.levels.remove(pos);
                self.level_count = self.level_count.saturating_sub(1);
            }
            
            // Recompute Merkle root
            self.merkle_root = Self::compute_merkle_root(&self.levels)?;
        }
        
        Ok(())
    }
}

impl Default for CompressedPriceTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Compressed order book state optimized for zkVM execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedOrderBook {
    /// Compressed symbol identifier
    pub symbol_id: u32,
    /// Compressed bid tree (highest price first)
    pub bid_tree: CompressedPriceTree,
    /// Compressed ask tree (lowest price first)
    pub ask_tree: CompressedPriceTree,
    /// Global state root (Merkle root of bid and ask trees)
    pub state_root: [u8; 32],
    /// Monotonic sequence number for ordering
    pub sequence: u64,
    /// Last trade price (fixed-point)
    pub last_trade_price: Option<u64>,
    /// Last trade timestamp
    pub last_trade_time: u64,
    /// Best bid price (cached for efficiency)
    pub best_bid: Option<u64>,
    /// Best ask price (cached for efficiency)
    pub best_ask: Option<u64>,
}

impl CompressedOrderBook {
    /// Create a new compressed order book
    pub fn new(symbol_id: u32) -> Self {
        Self {
            symbol_id,
            bid_tree: CompressedPriceTree::new(),
            ask_tree: CompressedPriceTree::new(),
            state_root: [0u8; 32],
            sequence: 0,
            last_trade_price: None,
            last_trade_time: 0,
            best_bid: None,
            best_ask: None,
        }
    }
    
    /// Create compressed order book from regular order book
    pub fn from_order_book(order_book: &OrderBook, symbol_id: u32) -> Result<Self, StateError> {
        let bid_tree = CompressedPriceTree::from_order_book_side(&order_book.bids, true)?;
        let ask_tree = CompressedPriceTree::from_order_book_side(&order_book.asks, false)?;
        
        let mut compressed = Self {
            symbol_id,
            bid_tree,
            ask_tree,
            state_root: [0u8; 32],
            sequence: order_book.sequence_number,
            last_trade_price: order_book.last_trade_price,
            last_trade_time: order_book.last_trade_time,
            best_bid: order_book.get_best_bid(),
            best_ask: order_book.get_best_ask(),
        };
        
        // Compute global state root
        compressed.state_root = compressed.compute_state_root()?;
        
        Ok(compressed)
    }
    
    /// Compute global state root from bid and ask trees
    pub fn compute_state_root(&self) -> Result<[u8; 32], StateError> {
        let mut hasher = Sha256::new();
        
        // Hash symbol ID
        hasher.update(&self.symbol_id.to_be_bytes());
        
        // Hash bid tree root
        hasher.update(&self.bid_tree.merkle_root);
        
        // Hash ask tree root
        hasher.update(&self.ask_tree.merkle_root);
        
        // Hash sequence number
        hasher.update(&self.sequence.to_be_bytes());
        
        // Hash last trade info
        if let Some(price) = self.last_trade_price {
            hasher.update(&price.to_be_bytes());
        }
        hasher.update(&self.last_trade_time.to_be_bytes());
        
        Ok(hasher.finalize().into())
    }
    
    /// Get current spread
    pub fn get_spread(&self) -> Option<u64> {
        match (self.best_ask, self.best_bid) {
            (Some(ask), Some(bid)) => Some(ask.saturating_sub(bid)),
            _ => None,
        }
    }
    
    /// Get mid price
    pub fn get_mid_price(&self) -> Option<u64> {
        match (self.best_ask, self.best_bid) {
            (Some(ask), Some(bid)) => Some((ask + bid) / 2),
            _ => None,
        }
    }
    
    /// Update best bid/ask cache
    fn update_best_prices(&mut self) {
        self.best_bid = self.bid_tree.levels.first().map(|level| level.price);
        self.best_ask = self.ask_tree.levels.first().map(|level| level.price);
    }
    
    /// Serialize to bytes for storage/transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>, StateError> {
        Ok(bincode::serialize(self)?)
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, StateError> {
        Ok(bincode::deserialize(data)?)
    }
    
    /// Get compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        self.to_bytes().map(|bytes| bytes.len()).unwrap_or(0)
    }
    
    /// Verify state integrity and consistency
    pub fn verify_state(&self) -> Result<bool, StateError> {
        // Verify bid tree integrity
        if !self.verify_price_tree(&self.bid_tree, true)? {
            return Ok(false);
        }
        
        // Verify ask tree integrity
        if !self.verify_price_tree(&self.ask_tree, false)? {
            return Ok(false);
        }
        
        // Verify state root matches computed value
        let computed_root = self.compute_state_root()?;
        if computed_root != self.state_root {
            return Ok(false);
        }
        
        // Verify best prices are consistent
        let expected_best_bid = self.bid_tree.levels.first().map(|l| l.price);
        let expected_best_ask = self.ask_tree.levels.first().map(|l| l.price);
        
        if self.best_bid != expected_best_bid || self.best_ask != expected_best_ask {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Verify price tree internal consistency
    fn verify_price_tree(&self, tree: &CompressedPriceTree, is_bid_side: bool) -> Result<bool, StateError> {
        // Verify level count matches actual levels
        if tree.level_count != tree.levels.len() as u16 {
            return Ok(false);
        }
        
        // Verify total volume calculation
        let calculated_volume: u64 = tree.levels.iter().map(|l| l.volume).sum();
        if calculated_volume != tree.total_volume {
            return Ok(false);
        }
        
        // Verify price ordering
        for i in 1..tree.levels.len() {
            let prev_price = tree.levels[i - 1].price;
            let curr_price = tree.levels[i].price;
            
            if is_bid_side {
                // Bids should be in descending order
                if prev_price < curr_price {
                    return Ok(false);
                }
            } else {
                // Asks should be in ascending order
                if prev_price > curr_price {
                    return Ok(false);
                }
            }
        }
        
        // Verify Merkle root
        let computed_root = CompressedPriceTree::compute_merkle_root(&tree.levels)?;
        if computed_root != tree.merkle_root {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Create a state diff between two compressed order books
    pub fn create_state_diff(&self, other: &CompressedOrderBook) -> Result<StateDiff, StateError> {
        let mut diff = StateDiff::new(self.state_root, other.state_root);
        
        // Compare bid trees
        self.compare_price_trees(&self.bid_tree, &other.bid_tree, true, &mut diff)?;
        
        // Compare ask trees
        self.compare_price_trees(&self.ask_tree, &other.ask_tree, false, &mut diff)?;
        
        // Compare metadata
        if self.sequence != other.sequence {
            diff.sequence_change = Some((self.sequence, other.sequence));
        }
        
        if self.last_trade_price != other.last_trade_price {
            diff.last_trade_price_change = Some((self.last_trade_price, other.last_trade_price));
        }
        
        Ok(diff)
    }
    
    /// Compare two price trees and record differences
    fn compare_price_trees(
        &self,
        tree1: &CompressedPriceTree,
        tree2: &CompressedPriceTree,
        is_bid_side: bool,
        diff: &mut StateDiff,
    ) -> Result<(), StateError> {
        let mut i = 0;
        let mut j = 0;
        
        while i < tree1.levels.len() || j < tree2.levels.len() {
            match (tree1.levels.get(i), tree2.levels.get(j)) {
                (Some(level1), Some(level2)) => {
                    if level1.price == level2.price {
                        // Same price level - check for changes
                        if level1.volume != level2.volume || level1.order_count != level2.order_count {
                            diff.level_changes.push(LevelChange {
                                price: level1.price,
                                is_bid_side,
                                old_volume: Some(level1.volume),
                                new_volume: Some(level2.volume),
                                old_order_count: Some(level1.order_count),
                                new_order_count: Some(level2.order_count),
                            });
                        }
                        i += 1;
                        j += 1;
                    } else if (is_bid_side && level1.price > level2.price) || (!is_bid_side && level1.price < level2.price) {
                        // Level removed from tree1
                        diff.level_changes.push(LevelChange {
                            price: level1.price,
                            is_bid_side,
                            old_volume: Some(level1.volume),
                            new_volume: None,
                            old_order_count: Some(level1.order_count),
                            new_order_count: None,
                        });
                        i += 1;
                    } else {
                        // Level added to tree2
                        diff.level_changes.push(LevelChange {
                            price: level2.price,
                            is_bid_side,
                            old_volume: None,
                            new_volume: Some(level2.volume),
                            old_order_count: None,
                            new_order_count: Some(level2.order_count),
                        });
                        j += 1;
                    }
                }
                (Some(level1), None) => {
                    // Level removed
                    diff.level_changes.push(LevelChange {
                        price: level1.price,
                        is_bid_side,
                        old_volume: Some(level1.volume),
                        new_volume: None,
                        old_order_count: Some(level1.order_count),
                        new_order_count: None,
                    });
                    i += 1;
                }
                (None, Some(level2)) => {
                    // Level added
                    diff.level_changes.push(LevelChange {
                        price: level2.price,
                        is_bid_side,
                        old_volume: None,
                        new_volume: Some(level2.volume),
                        old_order_count: None,
                        new_order_count: Some(level2.order_count),
                    });
                    j += 1;
                }
                (None, None) => break,
            }
        }
        
        Ok(())
    }
}

/// State transition record for zkVM execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Initial state root before transition
    pub initial_root: [u8; 32],
    /// Final state root after transition
    pub final_root: [u8; 32],
    /// List of individual order transitions
    pub transitions: Vec<OrderTransition>,
    /// Estimated gas cost for this transition
    pub gas_used: u64,
    /// Timestamp of the transition
    pub timestamp: u64,
}

/// Individual order transition within a state transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderTransition {
    /// Type of transition
    pub transition_type: TransitionType,
    /// Order involved in the transition
    pub order_id: u64,
    /// Price level affected
    pub price: u64,
    /// Volume change
    pub volume_delta: i64, // Positive for additions, negative for removals
    /// Trades generated (if any)
    pub trades: Vec<Trade>,
    /// Gas cost for this specific transition
    pub gas_cost: u64,
}

/// Types of state transitions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransitionType {
    /// Add new order to book
    AddOrder,
    /// Cancel existing order
    CancelOrder,
    /// Partial fill of order
    PartialFill,
    /// Complete fill of order
    CompleteFill,
    /// Modify existing order
    ModifyOrder,
}

impl CompressedOrderBook {
    /// Apply a deterministic state transition for zkVM execution
    pub fn apply_transition(&mut self, transition: OrderTransition) -> Result<(), StateError> {
        let initial_root = self.state_root;
        
        match transition.transition_type {
            TransitionType::AddOrder => {
                self.apply_add_order(&transition)?;
            }
            TransitionType::CancelOrder => {
                self.apply_cancel_order(&transition)?;
            }
            TransitionType::PartialFill => {
                self.apply_partial_fill(&transition)?;
            }
            TransitionType::CompleteFill => {
                self.apply_complete_fill(&transition)?;
            }
            TransitionType::ModifyOrder => {
                self.apply_modify_order(&transition)?;
            }
        }
        
        // Update sequence number
        self.sequence += 1;
        
        // Update best prices
        self.update_best_prices();
        
        // Recompute state root
        self.state_root = self.compute_state_root()?;
        
        // Verify state transition is valid
        if self.state_root == initial_root {
            return Err(StateError::InvalidTransition(
                "State root unchanged after transition".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Apply add order transition
    fn apply_add_order(&mut self, transition: &OrderTransition) -> Result<(), StateError> {
        // Determine which side to update based on volume delta sign
        let is_bid = transition.volume_delta > 0;
        
        // Create new price level or update existing one
        let new_level = CompressedPriceLevel {
            price: transition.price,
            volume: transition.volume_delta.abs() as u64,
            order_count: 1,
            timestamp: (self.last_trade_time / 1000) as u32,
            orders_hash: [0u8; 32], // Simplified for zkVM efficiency
        };
        
        if is_bid {
            self.bid_tree.add_level(new_level)?;
        } else {
            self.ask_tree.add_level(new_level)?;
        }
        
        Ok(())
    }
    
    /// Apply cancel order transition
    fn apply_cancel_order(&mut self, transition: &OrderTransition) -> Result<(), StateError> {
        let is_bid = transition.volume_delta < 0;
        let volume_change = transition.volume_delta.abs() as u64;
        
        if is_bid {
            if let Some(pos) = self.bid_tree.levels.iter().position(|l| l.price == transition.price) {
                let level = &mut self.bid_tree.levels[pos];
                level.volume = level.volume.saturating_sub(volume_change);
                level.order_count = level.order_count.saturating_sub(1);
                
                if level.is_empty() {
                    self.bid_tree.remove_level(transition.price)?;
                }
            }
        } else {
            if let Some(pos) = self.ask_tree.levels.iter().position(|l| l.price == transition.price) {
                let level = &mut self.ask_tree.levels[pos];
                level.volume = level.volume.saturating_sub(volume_change);
                level.order_count = level.order_count.saturating_sub(1);
                
                if level.is_empty() {
                    self.ask_tree.remove_level(transition.price)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply partial fill transition
    fn apply_partial_fill(&mut self, transition: &OrderTransition) -> Result<(), StateError> {
        // Update last trade information
        if let Some(trade) = transition.trades.first() {
            self.last_trade_price = Some(trade.price);
            self.last_trade_time = trade.timestamp;
        }
        
        // Reduce volume at the price level
        self.apply_cancel_order(transition)?;
        
        Ok(())
    }
    
    /// Apply complete fill transition
    fn apply_complete_fill(&mut self, transition: &OrderTransition) -> Result<(), StateError> {
        // Same as partial fill but removes entire order
        self.apply_partial_fill(transition)?;
        
        Ok(())
    }
    
    /// Apply modify order transition
    fn apply_modify_order(&mut self, transition: &OrderTransition) -> Result<(), StateError> {
        // Cancel old order and add new one
        let cancel_transition = OrderTransition {
            transition_type: TransitionType::CancelOrder,
            order_id: transition.order_id,
            price: transition.price,
            volume_delta: -transition.volume_delta,
            trades: Vec::new(),
            gas_cost: 0,
        };
        
        self.apply_cancel_order(&cancel_transition)?;
        self.apply_add_order(transition)?;
        
        Ok(())
    }
    
    /// Estimate gas cost for a transition
    pub fn estimate_gas_cost(&self, transitions: &[OrderTransition]) -> u64 {
        let base_cost = 21000u64; // Base transaction cost
        let mut total_cost = base_cost;
        
        for transition in transitions {
            let operation_cost = match transition.transition_type {
                TransitionType::AddOrder => 50000,      // Higher cost for storage
                TransitionType::CancelOrder => 20000,   // Lower cost for deletion
                TransitionType::PartialFill => 30000,   // Medium cost for update
                TransitionType::CompleteFill => 25000,  // Medium cost for removal
                TransitionType::ModifyOrder => 60000,   // Highest cost for modification
            };
            
            total_cost += operation_cost;
            
            // Add cost for each trade generated
            total_cost += transition.trades.len() as u64 * 10000;
        }
        
        total_cost
    }
}

/// State difference between two compressed order books
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff {
    /// Initial state root
    pub initial_root: [u8; 32],
    /// Final state root
    pub final_root: [u8; 32],
    /// Changes to price levels
    pub level_changes: Vec<LevelChange>,
    /// Sequence number change (old, new)
    pub sequence_change: Option<(u64, u64)>,
    /// Last trade price change (old, new)
    pub last_trade_price_change: Option<(Option<u64>, Option<u64>)>,
}

impl StateDiff {
    /// Create a new state diff
    pub fn new(initial_root: [u8; 32], final_root: [u8; 32]) -> Self {
        Self {
            initial_root,
            final_root,
            level_changes: Vec::new(),
            sequence_change: None,
            last_trade_price_change: None,
        }
    }
    
    /// Check if this diff represents any changes
    pub fn has_changes(&self) -> bool {
        !self.level_changes.is_empty() 
            || self.sequence_change.is_some() 
            || self.last_trade_price_change.is_some()
    }
    
    /// Get the number of level changes
    pub fn change_count(&self) -> usize {
        self.level_changes.len()
    }
}

/// Represents a change to a price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelChange {
    /// Price of the level that changed
    pub price: u64,
    /// Whether this is a bid side change
    pub is_bid_side: bool,
    /// Old volume (None if level was added)
    pub old_volume: Option<u64>,
    /// New volume (None if level was removed)
    pub new_volume: Option<u64>,
    /// Old order count (None if level was added)
    pub old_order_count: Option<u16>,
    /// New order count (None if level was removed)
    pub new_order_count: Option<u16>,
}

impl LevelChange {
    /// Check if this represents a level addition
    pub fn is_addition(&self) -> bool {
        self.old_volume.is_none() && self.new_volume.is_some()
    }
    
    /// Check if this represents a level removal
    pub fn is_removal(&self) -> bool {
        self.old_volume.is_some() && self.new_volume.is_none()
    }
    
    /// Check if this represents a level modification
    pub fn is_modification(&self) -> bool {
        self.old_volume.is_some() && self.new_volume.is_some()
    }
}

/// Batch of state transitions for efficient processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateBatch {
    /// Batch identifier
    pub batch_id: u64,
    /// Initial state root
    pub initial_state_root: [u8; 32],
    /// Final state root
    pub final_state_root: [u8; 32],
    /// List of state transitions in the batch
    pub transitions: Vec<StateTransition>,
    /// Total gas used for the batch
    pub total_gas_used: u64,
    /// Batch timestamp
    pub timestamp: u64,
    /// Merkle proof for the batch
    pub merkle_proof: Vec<[u8; 32]>,
}

impl StateBatch {
    /// Create a new state batch
    pub fn new(batch_id: u64, initial_state_root: [u8; 32]) -> Self {
        Self {
            batch_id,
            initial_state_root,
            final_state_root: initial_state_root,
            transitions: Vec::new(),
            total_gas_used: 0,
            timestamp: 0,
            merkle_proof: Vec::new(),
        }
    }
    
    /// Create a new state batch with pre-allocated capacity
    pub fn with_capacity(batch_id: u64, initial_state_root: [u8; 32], capacity: usize) -> Self {
        Self {
            batch_id,
            initial_state_root,
            final_state_root: initial_state_root,
            transitions: Vec::with_capacity(capacity),
            total_gas_used: 0,
            timestamp: 0,
            merkle_proof: Vec::new(),
        }
    }
    
    /// Add a state transition to the batch
    pub fn add_transition(&mut self, transition: StateTransition) {
        self.total_gas_used += transition.gas_used;
        self.final_state_root = transition.final_root;
        self.transitions.push(transition);
    }
    
    /// Finalize the batch and compute Merkle proof
    pub fn finalize(&mut self, timestamp: u64) -> Result<(), StateError> {
        self.timestamp = timestamp;
        
        // Create Merkle tree from transitions
        if !self.transitions.is_empty() {
            let leaves: Vec<MerkleNode> = self.transitions
                .iter()
                .map(|t| MerkleNode::new_leaf(&bincode::serialize(t).unwrap_or_default()))
                .collect();
            
            let tree = MerkleTree::from_leaves(leaves);
            
            // Generate Merkle proof (simplified - in practice would be more sophisticated)
            self.merkle_proof = vec![tree.root_hash()];
        }
        
        Ok(())
    }
    
    /// Verify the batch integrity
    pub fn verify(&self) -> Result<bool, StateError> {
        // Verify that final state root matches the last transition
        if let Some(last_transition) = self.transitions.last() {
            if last_transition.final_root != self.final_state_root {
                return Ok(false);
            }
        }
        
        // Verify gas calculation
        let calculated_gas: u64 = self.transitions.iter().map(|t| t.gas_used).sum();
        if calculated_gas != self.total_gas_used {
            return Ok(false);
        }
        
        // Verify Merkle proof (simplified)
        if self.merkle_proof.is_empty() && !self.transitions.is_empty() {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Serialize batch for storage/transmission
    pub fn to_bytes(&self) -> Result<Vec<u8>, StateError> {
        Ok(bincode::serialize(self)?)
    }
    
    /// Deserialize batch from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, StateError> {
        Ok(bincode::deserialize(data)?)
    }
    
    /// Zero-copy view of batch data for efficient processing
    pub fn as_bytes_view(&self) -> Result<&[u8], StateError> {
        // In a real implementation, this would use zero-copy serialization
        // For now, we simulate with a reference to demonstrate the concept
        unsafe {
            let ptr = self as *const Self as *const u8;
            let size = std::mem::size_of::<Self>();
            Ok(std::slice::from_raw_parts(ptr, size))
        }
    }
    
    /// Compress batch data for efficient storage/transmission
    pub fn compress(&self) -> Result<Vec<u8>, StateError> {
        let serialized = self.to_bytes()?;
        
        // Use a simple compression scheme (in practice, would use zstd or similar)
        let mut compressed = Vec::new();
        
        // Delta encode batch IDs and timestamps for better compression
        compressed.extend_from_slice(&self.batch_id.to_be_bytes());
        compressed.extend_from_slice(&self.initial_state_root);
        compressed.extend_from_slice(&self.final_state_root);
        compressed.extend_from_slice(&(self.transitions.len() as u32).to_be_bytes());
        
        // Delta encode transition data
        let mut prev_timestamp = 0u64;
        for transition in &self.transitions {
            let timestamp_delta = transition.timestamp.wrapping_sub(prev_timestamp);
            compressed.extend_from_slice(&timestamp_delta.to_be_bytes());
            compressed.extend_from_slice(&transition.gas_used.to_be_bytes());
            prev_timestamp = transition.timestamp;
        }
        
        Ok(compressed)
    }
    
    /// Get compression ratio compared to uncompressed size
    pub fn compression_ratio(&self) -> Result<f64, StateError> {
        let original_size = self.to_bytes()?.len();
        let compressed_size = self.compress()?.len();
        
        if original_size == 0 {
            return Ok(1.0);
        }
        
        Ok(compressed_size as f64 / original_size as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::types::{Order, PriceLevel};
    use std::collections::VecDeque;

    #[test]
    fn test_compressed_price_level() {
        let orders = vec![
            Order {
                id: 1,
                price: 50000,
                size: 100,
                timestamp: 1000,
                is_buy: true,
            },
            Order {
                id: 2,
                price: 50000,
                size: 200,
                timestamp: 1001,
                is_buy: true,
            },
        ];
        
        let compressed = CompressedPriceLevel::from_orders(50000, &orders, 1000);
        
        assert_eq!(compressed.price, 50000);
        assert_eq!(compressed.volume, 300);
        assert_eq!(compressed.order_count, 2);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_compressed_order_book_creation() {
        let mut order_book = OrderBook::new("BTCUSD".to_string());
        
        // Add some test data
        let mut bid_level = PriceLevel::new(50000);
        bid_level.add_order(Order {
            id: 1,
            price: 50000,
            size: 100,
            timestamp: 1000,
            is_buy: true,
        });
        order_book.bids.insert(50000, bid_level);
        
        let mut ask_level = PriceLevel::new(50100);
        ask_level.add_order(Order {
            id: 2,
            price: 50100,
            size: 150,
            timestamp: 1001,
            is_buy: false,
        });
        order_book.asks.insert(50100, ask_level);
        
        let compressed = CompressedOrderBook::from_order_book(&order_book, 1).unwrap();
        
        assert_eq!(compressed.symbol_id, 1);
        assert_eq!(compressed.bid_tree.level_count, 1);
        assert_eq!(compressed.ask_tree.level_count, 1);
        assert_ne!(compressed.state_root, [0u8; 32]);
        assert_eq!(compressed.get_spread(), Some(100));
    }

    #[test]
    fn test_state_transition() {
        let mut compressed = CompressedOrderBook::new(1);
        
        let transition = OrderTransition {
            transition_type: TransitionType::AddOrder,
            order_id: 1,
            price: 50000,
            volume_delta: 100,
            trades: Vec::new(),
            gas_cost: 50000,
        };
        
        let initial_root = compressed.state_root;
        compressed.apply_transition(transition).unwrap();
        
        assert_ne!(compressed.state_root, initial_root);
        assert_eq!(compressed.sequence, 1);
        assert_eq!(compressed.bid_tree.level_count, 1);
    }

    #[test]
    fn test_state_batch() {
        let mut batch = StateBatch::new(1, [0u8; 32]);
        
        let transition = StateTransition {
            initial_root: [0u8; 32],
            final_root: [1u8; 32],
            transitions: Vec::new(),
            gas_used: 50000,
            timestamp: 1000,
        };
        
        batch.add_transition(transition);
        batch.finalize(1000).unwrap();
        
        assert_eq!(batch.total_gas_used, 50000);
        assert_eq!(batch.final_state_root, [1u8; 32]);
        assert!(batch.verify().unwrap());
    }

    #[test]
    fn test_serialization() {
        let compressed = CompressedOrderBook::new(1);
        let bytes = compressed.to_bytes().unwrap();
        let deserialized = CompressedOrderBook::from_bytes(&bytes).unwrap();
        
        assert_eq!(compressed.symbol_id, deserialized.symbol_id);
        assert_eq!(compressed.state_root, deserialized.state_root);
        assert_eq!(compressed.sequence, deserialized.sequence);
    }
}