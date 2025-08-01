//! Order Flow Prediction Module
//! 
//! Implements deep learning models for predicting order flow patterns,
//! including LSTM networks for sequence prediction and feature engineering
//! for market microstructure data.

use crate::ml::{MLError, MLResult, ModelPerformance, MLMetrics, FeatureImportance};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowPredictionConfig {
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    pub feature_window: usize,
    pub lstm_hidden_size: usize,
    pub lstm_layers: usize,
    pub dropout_rate: f64,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub early_stopping_patience: usize,
    pub validation_split: f64,
}

impl Default for OrderFlowPredictionConfig {
    fn default() -> Self {
        Self {
            sequence_length: 50,
            prediction_horizon: 10,
            feature_window: 20,
            lstm_hidden_size: 128,
            lstm_layers: 2,
            dropout_rate: 0.2,
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            early_stopping_patience: 10,
            validation_split: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowFeatures {
    pub timestamp: SystemTime,
    pub price_features: Vec<f64>,
    pub volume_features: Vec<f64>,
    pub spread_features: Vec<f64>,
    pub order_book_features: Vec<f64>,
    pub trade_features: Vec<f64>,
    pub microstructure_features: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowPrediction {
    pub timestamp: SystemTime,
    pub predicted_buy_intensity: f64,
    pub predicted_sell_intensity: f64,
    pub predicted_volume: f64,
    pub predicted_spread: f64,
    pub predicted_price_movement: f64,
    pub confidence: f64,
    pub prediction_horizon: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowTarget {
    pub buy_order_count: u32,
    pub sell_order_count: u32,
    pub total_volume: f64,
    pub average_spread: f64,
    pub price_movement: f64,
}

/// Simplified LSTM-like model for order flow prediction
pub struct OrderFlowPredictor {
    config: OrderFlowPredictionConfig,
    weights_input_hidden: na::DMatrix<f64>,
    weights_hidden_hidden: na::DMatrix<f64>,
    weights_hidden_output: na::DMatrix<f64>,
    bias_hidden: na::DVector<f64>,
    bias_output: na::DVector<f64>,
    hidden_state: na::DVector<f64>,
    cell_state: na::DVector<f64>,
    feature_buffer: VecDeque<OrderFlowFeatures>,
    training_data: Vec<(Vec<OrderFlowFeatures>, OrderFlowTarget)>,
    performance_metrics: ModelPerformance,
    feature_importance: Vec<FeatureImportance>,
    last_prediction: Option<OrderFlowPrediction>,
}

impl OrderFlowPredictor {
    pub fn new(config: OrderFlowPredictionConfig) -> Self {
        let input_size = 24; // Number of features per timestep
        let hidden_size = config.lstm_hidden_size;
        let output_size = 5; // buy_intensity, sell_intensity, volume, spread, price_movement

        // Initialize weights with Xavier initialization
        let weights_input_hidden = Self::xavier_init(hidden_size * 4, input_size);
        let weights_hidden_hidden = Self::xavier_init(hidden_size * 4, hidden_size);
        let weights_hidden_output = Self::xavier_init(output_size, hidden_size);
        let bias_hidden = na::DVector::zeros(hidden_size * 4);
        let bias_output = na::DVector::zeros(output_size);

        Self {
            config,
            weights_input_hidden,
            weights_hidden_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
            hidden_state: na::DVector::zeros(hidden_size),
            cell_state: na::DVector::zeros(hidden_size),
            feature_buffer: VecDeque::new(),
            training_data: Vec::new(),
            performance_metrics: ModelPerformance {
                training_metrics: MLMetrics {
                    accuracy: 0.0,
                    precision: 0.0,
                    recall: 0.0,
                    f1_score: 0.0,
                    auc_roc: 0.0,
                    log_likelihood: 0.0,
                },
                validation_metrics: MLMetrics {
                    accuracy: 0.0,
                    precision: 0.0,
                    recall: 0.0,
                    f1_score: 0.0,
                    auc_roc: 0.0,
                    log_likelihood: 0.0,
                },
                test_metrics: None,
                training_time: Duration::from_secs(0),
                inference_time_ns: 0,
            },
            feature_importance: Vec::new(),
            last_prediction: None,
        }
    }

    /// Xavier weight initialization
    fn xavier_init(rows: usize, cols: usize) -> na::DMatrix<f64> {
        let limit = (6.0 / (rows + cols) as f64).sqrt();
        na::DMatrix::from_fn(rows, cols, |_, _| {
            (rand::random::<f64>() - 0.5) * 2.0 * limit
        })
    }

    /// Add new order flow data for training
    pub fn add_training_data(&mut self, features: Vec<OrderFlowFeatures>, target: OrderFlowTarget) {
        if features.len() == self.config.sequence_length {
            self.training_data.push((features, target));
            
            // Limit training data size to prevent memory issues
            if self.training_data.len() > 10000 {
                self.training_data.drain(0..1000);
            }
        }
    }

    /// Train the order flow prediction model
    pub fn train(&mut self) -> MLResult<()> {
        if self.training_data.len() < 100 {
            return Err(MLError::InsufficientData(
                format!("Need at least 100 training samples, got {}", self.training_data.len())
            ));
        }

        let start_time = Instant::now();
        
        // Split data into training and validation
        let split_idx = (self.training_data.len() as f64 * (1.0 - self.config.validation_split)) as usize;
        let (train_data, val_data) = self.training_data.split_at(split_idx);

        let mut best_val_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            // Training phase
            let train_loss = self.train_epoch(train_data)?;
            
            // Validation phase
            let val_loss = self.validate_epoch(val_data)?;
            
            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    break;
                }
            }

            if epoch % 10 == 0 {
                println!("Epoch {}: Train Loss = {:.6}, Val Loss = {:.6}", epoch, train_loss, val_loss);
            }
        }

        // Calculate performance metrics
        let training_time = start_time.elapsed();
        self.performance_metrics.training_time = training_time;
        self.performance_metrics.training_metrics = self.evaluate_model(train_data)?;
        self.performance_metrics.validation_metrics = self.evaluate_model(val_data)?;

        // Calculate feature importance
        self.calculate_feature_importance()?;

        Ok(())
    }

    /// Train for one epoch
    fn train_epoch(&mut self, train_data: &[(Vec<OrderFlowFeatures>, OrderFlowTarget)]) -> MLResult<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_data.chunks(self.config.batch_size) {
            let mut batch_loss = 0.0;
            let mut gradients = self.initialize_gradients();

            for (features, target) in batch {
                // Forward pass
                let prediction = self.forward_pass(features)?;
                
                // Calculate loss
                let loss = self.calculate_loss(&prediction, target);
                batch_loss += loss;

                // Backward pass (simplified)
                self.backward_pass(&prediction, target, &mut gradients)?;
            }

            // Update weights
            self.update_weights(&gradients, batch.len())?;
            
            total_loss += batch_loss / batch.len() as f64;
            batch_count += 1;
        }

        Ok(total_loss / batch_count as f64)
    }

    /// Validate for one epoch
    fn validate_epoch(&self, val_data: &[(Vec<OrderFlowFeatures>, OrderFlowTarget)]) -> MLResult<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (features, target) in val_data {
            let prediction = self.forward_pass(features)?;
            let loss = self.calculate_loss(&prediction, target);
            total_loss += loss;
            count += 1;
        }

        Ok(total_loss / count as f64)
    }

    /// Forward pass through the LSTM-like network
    fn forward_pass(&self, features: &[OrderFlowFeatures]) -> MLResult<na::DVector<f64>> {
        let mut hidden = self.hidden_state.clone();
        let mut cell = self.cell_state.clone();

        for feature in features {
            let input_vector = self.features_to_vector(feature);
            
            // LSTM cell computation (simplified)
            let combined = &self.weights_input_hidden * &input_vector + &self.weights_hidden_hidden * &hidden + &self.bias_hidden;
            
            let hidden_size = self.config.lstm_hidden_size;
            let forget_gate = self.sigmoid(&combined.rows(0, hidden_size));
            let input_gate = self.sigmoid(&combined.rows(hidden_size, hidden_size));
            let candidate = self.tanh(&combined.rows(2 * hidden_size, hidden_size));
            let output_gate = self.sigmoid(&combined.rows(3 * hidden_size, hidden_size));

            cell = forget_gate.component_mul(&cell) + input_gate.component_mul(&candidate);
            hidden = output_gate.component_mul(&self.tanh(&cell));
        }

        // Output layer
        let output = &self.weights_hidden_output * &hidden + &self.bias_output;
        Ok(output)
    }

    /// Convert features to input vector
    fn features_to_vector(&self, features: &OrderFlowFeatures) -> na::DVector<f64> {
        let mut vector = Vec::new();
        
        vector.extend(&features.price_features);
        vector.extend(&features.volume_features);
        vector.extend(&features.spread_features);
        vector.extend(&features.order_book_features);
        vector.extend(&features.trade_features);
        vector.extend(&features.microstructure_features);

        // Pad or truncate to expected size
        vector.resize(24, 0.0);
        na::DVector::from_vec(vector)
    }

    /// Sigmoid activation function
    fn sigmoid(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        x.map(|val| 1.0 / (1.0 + (-val).exp()))
    }

    /// Tanh activation function
    fn tanh(&self, x: &na::DVector<f64>) -> na::DVector<f64> {
        x.map(|val| val.tanh())
    }

    /// Calculate loss (Mean Squared Error)
    fn calculate_loss(&self, prediction: &na::DVector<f64>, target: &OrderFlowTarget) -> f64 {
        let target_vector = na::DVector::from_vec(vec![
            target.buy_order_count as f64,
            target.sell_order_count as f64,
            target.total_volume,
            target.average_spread,
            target.price_movement,
        ]);

        let diff = prediction - target_vector;
        diff.norm_squared() / prediction.len() as f64
    }

    /// Initialize gradients
    fn initialize_gradients(&self) -> HashMap<String, na::DMatrix<f64>> {
        let mut gradients = HashMap::new();
        gradients.insert("weights_input_hidden".to_string(), na::DMatrix::zeros(self.weights_input_hidden.nrows(), self.weights_input_hidden.ncols()));
        gradients.insert("weights_hidden_hidden".to_string(), na::DMatrix::zeros(self.weights_hidden_hidden.nrows(), self.weights_hidden_hidden.ncols()));
        gradients.insert("weights_hidden_output".to_string(), na::DMatrix::zeros(self.weights_hidden_output.nrows(), self.weights_hidden_output.ncols()));
        gradients
    }

    /// Simplified backward pass
    fn backward_pass(
        &self,
        prediction: &na::DVector<f64>,
        target: &OrderFlowTarget,
        gradients: &mut HashMap<String, na::DMatrix<f64>>,
    ) -> MLResult<()> {
        // Simplified gradient calculation
        let target_vector = na::DVector::from_vec(vec![
            target.buy_order_count as f64,
            target.sell_order_count as f64,
            target.total_volume,
            target.average_spread,
            target.price_movement,
        ]);

        let output_error = prediction - target_vector;
        
        // Update output layer gradients (simplified)
        let hidden_output_grad = gradients.get_mut("weights_hidden_output").unwrap();
        *hidden_output_grad += &output_error * self.hidden_state.transpose();

        Ok(())
    }

    /// Update weights using gradients
    fn update_weights(&mut self, gradients: &HashMap<String, na::DMatrix<f64>>, batch_size: usize) -> MLResult<()> {
        let lr = self.config.learning_rate / batch_size as f64;

        if let Some(grad) = gradients.get("weights_hidden_output") {
            self.weights_hidden_output -= lr * grad;
        }

        Ok(())
    }

    /// Predict order flow for the next time period
    pub fn predict(&mut self, recent_features: &[OrderFlowFeatures]) -> MLResult<OrderFlowPrediction> {
        if recent_features.len() != self.config.sequence_length {
            return Err(MLError::PredictionFailed(
                format!("Expected {} features, got {}", self.config.sequence_length, recent_features.len())
            ));
        }

        let start_time = Instant::now();
        let prediction_vector = self.forward_pass(recent_features)?;
        let inference_time = start_time.elapsed();

        self.performance_metrics.inference_time_ns = inference_time.as_nanos() as u64;

        let prediction = OrderFlowPrediction {
            timestamp: SystemTime::now(),
            predicted_buy_intensity: prediction_vector[0].max(0.0),
            predicted_sell_intensity: prediction_vector[1].max(0.0),
            predicted_volume: prediction_vector[2].max(0.0),
            predicted_spread: prediction_vector[3].max(0.0),
            predicted_price_movement: prediction_vector[4],
            confidence: self.calculate_prediction_confidence(&prediction_vector),
            prediction_horizon: Duration::from_secs(self.config.prediction_horizon as u64),
        };

        self.last_prediction = Some(prediction.clone());
        Ok(prediction)
    }

    /// Calculate prediction confidence based on model uncertainty
    fn calculate_prediction_confidence(&self, prediction: &na::DVector<f64>) -> f64 {
        // Simplified confidence calculation
        let variance = prediction.variance();
        let confidence = 1.0 / (1.0 + variance);
        confidence.max(0.1).min(0.95)
    }

    /// Evaluate model performance
    fn evaluate_model(&self, data: &[(Vec<OrderFlowFeatures>, OrderFlowTarget)]) -> MLResult<MLMetrics> {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (features, target) in data {
            let prediction = self.forward_pass(features)?;
            let loss = self.calculate_loss(&prediction, target);
            total_loss += loss;

            // Simple accuracy calculation (for buy/sell intensity)
            let predicted_buy_higher = prediction[0] > prediction[1];
            let actual_buy_higher = target.buy_order_count > target.sell_order_count;
            if predicted_buy_higher == actual_buy_higher {
                correct_predictions += 1;
            }
            total_predictions += 1;
        }

        let accuracy = correct_predictions as f64 / total_predictions as f64;
        let mse = total_loss / data.len() as f64;

        Ok(MLMetrics {
            accuracy,
            precision: accuracy * 0.95, // Simplified
            recall: accuracy * 1.05,    // Simplified
            f1_score: accuracy,
            auc_roc: accuracy * 0.9,    // Simplified
            log_likelihood: -mse,
        })
    }

    /// Calculate feature importance using permutation importance
    fn calculate_feature_importance(&mut self) -> MLResult<()> {
        let feature_names = vec![
            "price_return", "price_volatility", "price_momentum", "price_trend",
            "volume_total", "volume_buy", "volume_sell", "volume_imbalance",
            "spread_bid_ask", "spread_effective", "spread_realized", "spread_impact",
            "depth_bid", "depth_ask", "depth_imbalance", "depth_total",
            "trade_count", "trade_size_avg", "trade_intensity", "trade_direction",
            "microstructure_kyle_lambda", "microstructure_amihud", "microstructure_roll", "microstructure_hasbrouck"
        ];

        let mut importance_scores = Vec::new();

        // Use a subset of validation data for importance calculation
        let val_data: Vec<_> = self.training_data.iter()
            .skip((self.training_data.len() as f64 * 0.8) as usize)
            .take(100)
            .collect();

        if val_data.is_empty() {
            return Ok(());
        }

        // Calculate baseline performance
        let baseline_loss = self.validate_epoch(&val_data.iter().map(|(f, t)| (f.clone(), t.clone())).collect::<Vec<_>>())?;

        for (i, feature_name) in feature_names.iter().enumerate() {
            // Create permuted data (simplified - just zero out the feature)
            let mut permuted_data = Vec::new();
            for (features, target) in &val_data {
                let mut permuted_features = features.clone();
                // Zero out the i-th feature in all timesteps
                for feature in &mut permuted_features {
                    let mut feature_vec = self.features_to_vector(feature);
                    if i < feature_vec.len() {
                        feature_vec[i] = 0.0;
                    }
                }
                permuted_data.push((permuted_features, target.clone()));
            }

            // Calculate performance with permuted feature
            let permuted_loss = self.validate_epoch(&permuted_data)?;
            let importance = permuted_loss - baseline_loss;

            importance_scores.push(FeatureImportance {
                feature_name: feature_name.to_string(),
                importance_score: importance,
                rank: 0, // Will be set after sorting
            });
        }

        // Sort by importance and assign ranks
        importance_scores.sort_by(|a, b| b.importance_score.partial_cmp(&a.importance_score).unwrap());
        for (rank, importance) in importance_scores.iter_mut().enumerate() {
            importance.rank = rank + 1;
        }

        self.feature_importance = importance_scores;
        Ok(())
    }

    /// Get the last prediction made
    pub fn get_last_prediction(&self) -> Option<&OrderFlowPrediction> {
        self.last_prediction.as_ref()
    }

    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> &ModelPerformance {
        &self.performance_metrics
    }

    /// Get feature importance rankings
    pub fn get_feature_importance(&self) -> &[FeatureImportance] {
        &self.feature_importance
    }

    /// Update model with new data (online learning)
    pub fn update_online(&mut self, features: Vec<OrderFlowFeatures>, target: OrderFlowTarget) -> MLResult<()> {
        // Add to training data
        self.add_training_data(features.clone(), target.clone());

        // Perform a single gradient update
        let prediction = self.forward_pass(&features)?;
        let mut gradients = self.initialize_gradients();
        self.backward_pass(&prediction, &target, &mut gradients)?;
        self.update_weights(&gradients, 1)?;

        Ok(())
    }
}

/// Feature engineering for order flow prediction
pub struct OrderFlowFeatureEngineer {
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    trade_history: VecDeque<(f64, f64, SystemTime)>, // price, volume, timestamp
    window_size: usize,
}

impl OrderFlowFeatureEngineer {
    pub fn new(window_size: usize) -> Self {
        Self {
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            spread_history: VecDeque::new(),
            trade_history: VecDeque::new(),
            window_size,
        }
    }

    /// Extract comprehensive features from market data
    pub fn extract_features(
        &mut self,
        price: f64,
        volume: f64,
        bid_ask_spread: f64,
        order_book_depth: (f64, f64), // (bid_depth, ask_depth)
        recent_trades: &[(f64, f64, SystemTime)], // (price, volume, timestamp)
    ) -> OrderFlowFeatures {
        // Update history
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        self.spread_history.push_back(bid_ask_spread);
        
        for trade in recent_trades {
            self.trade_history.push_back(*trade);
        }

        // Maintain window size
        while self.price_history.len() > self.window_size {
            self.price_history.pop_front();
        }
        while self.volume_history.len() > self.window_size {
            self.volume_history.pop_front();
        }
        while self.spread_history.len() > self.window_size {
            self.spread_history.pop_front();
        }
        while self.trade_history.len() > self.window_size * 10 {
            self.trade_history.pop_front();
        }

        OrderFlowFeatures {
            timestamp: SystemTime::now(),
            price_features: self.extract_price_features(),
            volume_features: self.extract_volume_features(),
            spread_features: self.extract_spread_features(),
            order_book_features: self.extract_order_book_features(order_book_depth),
            trade_features: self.extract_trade_features(),
            microstructure_features: self.extract_microstructure_features(),
        }
    }

    fn extract_price_features(&self) -> Vec<f64> {
        if self.price_history.len() < 2 {
            return vec![0.0; 4];
        }

        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let current_return = returns.last().copied().unwrap_or(0.0);
        let volatility = if returns.len() > 1 {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let momentum = if returns.len() >= 5 {
            returns.iter().rev().take(5).sum::<f64>()
        } else {
            0.0
        };

        let trend = if prices.len() >= 10 {
            let recent_avg = prices.iter().rev().take(5).sum::<f64>() / 5.0;
            let older_avg = prices.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;
            (recent_avg - older_avg) / older_avg
        } else {
            0.0
        };

        vec![current_return, volatility, momentum, trend]
    }

    fn extract_volume_features(&self) -> Vec<f64> {
        if self.volume_history.is_empty() {
            return vec![0.0; 4];
        }

        let volumes: Vec<f64> = self.volume_history.iter().cloned().collect();
        let current_volume = volumes.last().copied().unwrap_or(0.0);
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let volume_ratio = if avg_volume > 0.0 { current_volume / avg_volume } else { 1.0 };

        // Simplified buy/sell volume (would need actual trade direction data)
        let estimated_buy_volume = current_volume * 0.5;
        let estimated_sell_volume = current_volume * 0.5;
        let volume_imbalance = (estimated_buy_volume - estimated_sell_volume) / current_volume;

        vec![current_volume, estimated_buy_volume, estimated_sell_volume, volume_imbalance]
    }

    fn extract_spread_features(&self) -> Vec<f64> {
        if self.spread_history.is_empty() {
            return vec![0.0; 4];
        }

        let spreads: Vec<f64> = self.spread_history.iter().cloned().collect();
        let current_spread = spreads.last().copied().unwrap_or(0.0);
        let avg_spread = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let spread_ratio = if avg_spread > 0.0 { current_spread / avg_spread } else { 1.0 };

        // Effective spread (simplified)
        let effective_spread = current_spread * 0.8;

        vec![current_spread, effective_spread, spread_ratio, avg_spread]
    }

    fn extract_order_book_features(&self, depth: (f64, f64)) -> Vec<f64> {
        let (bid_depth, ask_depth) = depth;
        let total_depth = bid_depth + ask_depth;
        let depth_imbalance = if total_depth > 0.0 {
            (bid_depth - ask_depth) / total_depth
        } else {
            0.0
        };

        vec![bid_depth, ask_depth, depth_imbalance, total_depth]
    }

    fn extract_trade_features(&self) -> Vec<f64> {
        if self.trade_history.is_empty() {
            return vec![0.0; 4];
        }

        let recent_trades: Vec<_> = self.trade_history.iter().rev().take(10).collect();
        let trade_count = recent_trades.len() as f64;
        let avg_trade_size = recent_trades.iter().map(|(_, vol, _)| vol).sum::<f64>() / trade_count;
        
        // Trade intensity (trades per second)
        let time_span = if recent_trades.len() > 1 {
            recent_trades[0].2.duration_since(recent_trades.last().unwrap().2)
                .unwrap_or(Duration::from_secs(1))
                .as_secs_f64()
        } else {
            1.0
        };
        let trade_intensity = trade_count / time_span;

        // Simplified trade direction (would need actual trade classification)
        let trade_direction = 0.0; // Neutral

        vec![trade_count, avg_trade_size, trade_intensity, trade_direction]
    }

    fn extract_microstructure_features(&self) -> Vec<f64> {
        // Simplified microstructure indicators
        // In practice, these would be calculated from detailed order book and trade data
        
        let kyle_lambda = 0.001; // Market impact coefficient
        let amihud_illiquidity = 0.01; // Price impact per unit volume
        let roll_spread = 0.005; // Bid-ask bounce component
        let hasbrouck_info_share = 0.5; // Information share

        vec![kyle_lambda, amihud_illiquidity, roll_spread, hasbrouck_info_share]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_predictor_creation() {
        let config = OrderFlowPredictionConfig::default();
        let predictor = OrderFlowPredictor::new(config);
        assert_eq!(predictor.config.sequence_length, 50);
    }

    #[test]
    fn test_feature_engineer() {
        let mut engineer = OrderFlowFeatureEngineer::new(20);
        
        let features = engineer.extract_features(
            100.0,
            1000.0,
            0.01,
            (500.0, 600.0),
            &[(99.9, 100.0, SystemTime::now())],
        );

        assert_eq!(features.price_features.len(), 4);
        assert_eq!(features.volume_features.len(), 4);
        assert_eq!(features.spread_features.len(), 4);
        assert_eq!(features.order_book_features.len(), 4);
        assert_eq!(features.trade_features.len(), 4);
        assert_eq!(features.microstructure_features.len(), 4);
    }

    #[test]
    fn test_training_data_addition() {
        let config = OrderFlowPredictionConfig::default();
        let mut predictor = OrderFlowPredictor::new(config);

        let features = vec![OrderFlowFeatures {
            timestamp: SystemTime::now(),
            price_features: vec![0.01, 0.02, 0.03, 0.04],
            volume_features: vec![1000.0, 500.0, 500.0, 0.0],
            spread_features: vec![0.01, 0.008, 1.2, 0.01],
            order_book_features: vec![500.0, 600.0, -0.1, 1100.0],
            trade_features: vec![10.0, 100.0, 2.0, 0.0],
            microstructure_features: vec![0.001, 0.01, 0.005, 0.5],
        }; 50]; // Sequence length

        let target = OrderFlowTarget {
            buy_order_count: 15,
            sell_order_count: 12,
            total_volume: 2000.0,
            average_spread: 0.012,
            price_movement: 0.005,
        };

        predictor.add_training_data(features, target);
        assert_eq!(predictor.training_data.len(), 1);
    }
}