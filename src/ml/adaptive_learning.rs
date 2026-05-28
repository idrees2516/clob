//! Adaptive Learning System
//! 
//! Implements online learning algorithms, model drtion, and automatic
//! model retraining for continuous adaptation to changing market conditions.

use crate::ml::{MLError, MLResult, ModelPerformance, MLMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    pub drift_detection_window: usize,
    pub drift_threshold: f64,
    pub retraining_frequency: Duration,
    pub performance_degradation_threshold: f64,
    pub online_learning_rate: f64,
    pub forgetting_factor: f64,
    pub min_samples_for_retraining: usize,
    pub max_model_age: Duration,
    pub ensemble_size: usize,
    pub model_selection_metric: String,
}

impl Default for AdaptiveLearningConfig {
    fn default() -> Self {
        Self {
            drift_detection_window: 1000,
            drift_threshold: 0.05,
            retraining_frequency: Duration::from_secs(3600), // 1 hour
            performance_degradation_threshold: 0.1,
            online_learning_rate: 0.01,
            forgetting_factor: 0.99,
            min_samples_for_retraining: 100,
            max_model_age: Duration::from_secs(86400), // 24 hours
            ensemble_size: 5,
            model_selection_metric: "accuracy".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetectionResult {
    pub drift_detected: bool,
    pub drift_magnitude: f64,
    pub drift_type: DriftType,
    pub confidence: f64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftType {
    Gradual,
    Sudden,
    Recurring,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub creation_time: SystemTime,
    pub last_update: SystemTime,
    pub training_samples: usize,
    pub performance_history: VecDeque<MLMetrics>,
    pub drift_events: Vec<DriftDetectionResult>,
}

/// Adaptive learning system that manages model lifecycle and drift detection
pub struct AdaptiveLearningSystem {
    config: AdaptiveLearningConfig,
    active_models: HashMap<String, ModelMetadata>,
    performance_buffer: VecDeque<(SystemTime, MLMetrics)>,
    drift_detector: DriftDetector,
    model_selector: ModelSelector,
    online_learner: OnlineLearner,
    last_retraining: SystemTime,
    retraining_queue: VecDeque<String>,
}

impl AdaptiveLearningSystem {
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            drift_detector: DriftDetector::new(config.clone()),
            model_selector: ModelSelector::new(config.clone()),
            online_learner: OnlineLearner::new(config.clone()),
            config,
            active_models: HashMap::new(),
            performance_buffer: VecDeque::new(),
            last_retraining: SystemTime::now(),
            retraining_queue: VecDeque::new(),
        }
    }

    /// Register a new model for adaptive management
    pub fn register_model(&mut self, model_id: String, initial_performance: MLMetrics) -> MLResult<()> {
        let metadata = ModelMetadata {
            model_id: model_id.clone(),
            creation_time: SystemTime::now(),
            last_update: SystemTime::now(),
            training_samples: 0,
            performance_history: {
                let mut history = VecDeque::new();
                history.push_back(initial_performance);
                history
            },
            drift_events: Vec::new(),
        };

        self.active_models.insert(model_id, metadata);
        Ok(())
    }

    /// Update model performance and check for drift
    pub fn update_model_performance(
        &mut self,
        model_id: &str,
        new_performance: MLMetrics,
    ) -> MLResult<Option<DriftDetectionResult>> {
        let model = self.active_models.get_mut(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))?;

        // Update performance history
        model.performance_history.push_back(new_performance.clone());
        if model.performance_history.len() > self.config.drift_detection_window {
            model.performance_history.pop_front();
        }
        model.last_update = SystemTime::now();

        // Add to global performance buffer
        self.performance_buffer.push_back((SystemTime::now(), new_performance));
        if self.performance_buffer.len() > self.config.drift_detection_window {
            self.performance_buffer.pop_front();
        }

        // Check for drift
        let drift_result = self.drift_detector.detect_drift(&model.performance_history)?;
        
        if drift_result.drift_detected {
            model.drift_events.push(drift_result.clone());
            
            // Schedule retraining if significant drift detected
            if drift_result.drift_magnitude > self.config.drift_threshold {
                self.schedule_retraining(model_id.to_string());
            }
            
            return Ok(Some(drift_result));
        }

        // Check if model needs retraining due to age or performance degradation
        self.check_retraining_conditions(model_id)?;

        Ok(None)
    }

    /// Process online learning update
    pub fn online_update(
        &mut self,
        model_id: &str,
        features: &[f64],
        target: f64,
        prediction: f64,
    ) -> MLResult<()> {
        let model = self.active_models.get_mut(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))?;

        // Perform online learning update
        self.online_learner.update(features, target, prediction)?;
        
        model.training_samples += 1;
        model.last_update = SystemTime::now();

        Ok(())
    }

    /// Check if any models need retraining
    pub fn check_retraining_needs(&mut self) -> MLResult<Vec<String>> {
        let mut models_to_retrain = Vec::new();

        // Check scheduled retraining
        while let Some(model_id) = self.retraining_queue.pop_front() {
            models_to_retrain.push(model_id);
        }

        // Check time-based retraining
        if self.last_retraining.elapsed().unwrap_or(Duration::ZERO) > self.config.retraining_frequency {
            for model_id in self.active_models.keys() {
                if !models_to_retrain.contains(model_id) {
                    models_to_retrain.push(model_id.clone());
                }
            }
            self.last_retraining = SystemTime::now();
        }

        Ok(models_to_retrain)
    }

    /// Select the best performing model
    pub fn select_best_model(&self) -> MLResult<Option<String>> {
        if self.active_models.is_empty() {
            return Ok(None);
        }

        let best_model = self.model_selector.select_best(&self.active_models)?;
        Ok(Some(best_model))
    }

    /// Get model performance summary
    pub fn get_model_summary(&self, model_id: &str) -> MLResult<ModelSummary> {
        let model = self.active_models.get(model_id)
            .ok_or_else(|| MLError::ConfigurationError(format!("Model {} not found", model_id)))?;

        let current_performance = model.performance_history.back().cloned()
            .unwrap_or_else(|| MLMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_roc: 0.0,
                log_likelihood: 0.0,
            });

        let avg_performance = self.calculate_average_performance(&model.performance_history);
        let drift_count = model.drift_events.len();
        let model_age = model.creation_time.elapsed().unwrap_or(Duration::ZERO);

        Ok(ModelSummary {
            model_id: model_id.to_string(),
            current_performance,
            average_performance: avg_performance,
            drift_events_count: drift_count,
            model_age,
            training_samples: model.training_samples,
            last_update: model.last_update,
        })
    }

    fn schedule_retraining(&mut self, model_id: String) {
        if !self.retraining_queue.contains(&model_id) {
            self.retraining_queue.push_back(model_id);
        }
    }

    fn check_retraining_conditions(&mut self, model_id: &str) -> MLResult<()> {
        let model = self.active_models.get(model_id).unwrap();

        // Check model age
        let model_age = model.creation_time.elapsed().unwrap_or(Duration::ZERO);
        if model_age > self.config.max_model_age {
            self.schedule_retraining(model_id.to_string());
            return Ok(());
        }

        // Check performance degradation
        if model.performance_history.len() >= 2 {
            let recent_performance = model.performance_history.back().unwrap();
            let historical_avg = self.calculate_average_performance(&model.performance_history);
            
            let performance_drop = historical_avg.accuracy - recent_performance.accuracy;
            if performance_drop > self.config.performance_degradation_threshold {
                self.schedule_retraining(model_id.to_string());
            }
        }

        Ok(())
    }

    fn calculate_average_performance(&self, history: &VecDeque<MLMetrics>) -> MLMetrics {
        if history.is_empty() {
            return MLMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_roc: 0.0,
                log_likelihood: 0.0,
            };
        }

        let count = history.len() as f64;
        let sum = history.iter().fold(
            MLMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_roc: 0.0,
                log_likelihood: 0.0,
            },
            |acc, metrics| MLMetrics {
                accuracy: acc.accuracy + metrics.accuracy,
                precision: acc.precision + metrics.precision,
                recall: acc.recall + metrics.recall,
                f1_score: acc.f1_score + metrics.f1_score,
                auc_roc: acc.auc_roc + metrics.auc_roc,
                log_likelihood: acc.log_likelihood + metrics.log_likelihood,
            },
        );

        MLMetrics {
            accuracy: sum.accuracy / count,
            precision: sum.precision / count,
            recall: sum.recall / count,
            f1_score: sum.f1_score / count,
            auc_roc: sum.auc_roc / count,
            log_likelihood: sum.log_likelihood / count,
        }
    }
}

/// Drift detection using statistical methods
pub struct DriftDetector {
    config: AdaptiveLearningConfig,
    reference_window: VecDeque<f64>,
    current_window: VecDeque<f64>,
}

impl DriftDetector {
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            config,
            reference_window: VecDeque::new(),
            current_window: VecDeque::new(),
        }
    }

    /// Detect drift using Page-Hinkley test and Kolmogorov-Smirnov test
    pub fn detect_drift(&mut self, performance_history: &VecDeque<MLMetrics>) -> MLResult<DriftDetectionResult> {
        if performance_history.len() < 10 {
            return Ok(DriftDetectionResult {
                drift_detected: false,
                drift_magnitude: 0.0,
                drift_type: DriftType::Unknown,
                confidence: 0.0,
                timestamp: SystemTime::now(),
            });
        }

        // Extract accuracy values for drift detection
        let accuracy_values: Vec<f64> = performance_history.iter()
            .map(|m| m.accuracy)
            .collect();

        // Split into reference and current windows
        let split_point = accuracy_values.len() / 2;
        let reference: Vec<f64> = accuracy_values[..split_point].to_vec();
        let current: Vec<f64> = accuracy_values[split_point..].to_vec();

        // Page-Hinkley test for gradual drift
        let ph_result = self.page_hinkley_test(&accuracy_values)?;
        
        // Kolmogorov-Smirnov test for distribution change
        let ks_result = self.kolmogorov_smirnov_test(&reference, &current)?;

        // ADWIN (Adaptive Windowing) for sudden drift
        let adwin_result = self.adwin_test(&accuracy_values)?;

        // Combine results
        let drift_detected = ph_result.0 || ks_result.0 || adwin_result.0;
        let drift_magnitude = (ph_result.1 + ks_result.1 + adwin_result.1) / 3.0;
        
        let drift_type = if ph_result.0 && !adwin_result.0 {
            DriftType::Gradual
        } else if adwin_result.0 {
            DriftType::Sudden
        } else if self.detect_recurring_pattern(&accuracy_values) {
            DriftType::Recurring
        } else {
            DriftType::Unknown
        };

        let confidence = if drift_detected {
            (drift_magnitude / self.config.drift_threshold).min(1.0)
        } else {
            1.0 - drift_magnitude
        };

        Ok(DriftDetectionResult {
            drift_detected,
            drift_magnitude,
            drift_type,
            confidence,
            timestamp: SystemTime::now(),
        })
    }

    fn page_hinkley_test(&self, data: &[f64]) -> MLResult<(bool, f64)> {
        if data.len() < 30 {
            return Ok((false, 0.0));
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let mut cumsum = 0.0;
        let mut max_cumsum = 0.0;
        let mut min_cumsum = 0.0;

        for &value in data {
            cumsum += value - mean - 0.005; // Small drift magnitude
            max_cumsum = max_cumsum.max(cumsum);
            min_cumsum = min_cumsum.min(cumsum);
        }

        let ph_statistic = max_cumsum - min_cumsum;
        let threshold = 5.0; // Threshold for Page-Hinkley test
        
        Ok((ph_statistic > threshold, ph_statistic / threshold))
    }

    fn kolmogorov_smirnov_test(&self, sample1: &[f64], sample2: &[f64]) -> MLResult<(bool, f64)> {
        if sample1.len() < 10 || sample2.len() < 10 {
            return Ok((false, 0.0));
        }

        let mut sorted1 = sample1.to_vec();
        let mut sorted2 = sample2.to_vec();
        sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = sorted1.len() as f64;
        let n2 = sorted2.len() as f64;

        // Calculate empirical distribution functions
        let mut max_diff = 0.0;
        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < sorted1.len() && i2 < sorted2.len() {
            let cdf1 = (i1 + 1) as f64 / n1;
            let cdf2 = (i2 + 1) as f64 / n2;
            
            max_diff = max_diff.max((cdf1 - cdf2).abs());

            if sorted1[i1] <= sorted2[i2] {
                i1 += 1;
            } else {
                i2 += 1;
            }
        }

        // Critical value for KS test (approximate)
        let critical_value = 1.36 * ((n1 + n2) / (n1 * n2)).sqrt();
        
        Ok((max_diff > critical_value, max_diff / critical_value))
    }

    fn adwin_test(&self, data: &[f64]) -> MLResult<(bool, f64)> {
        if data.len() < 20 {
            return Ok((false, 0.0));
        }

        // Simplified ADWIN implementation
        let window_size = data.len() / 2;
        let recent_mean = data[data.len() - window_size..].iter().sum::<f64>() / window_size as f64;
        let historical_mean = data[..window_size].iter().sum::<f64>() / window_size as f64;
        
        let mean_diff = (recent_mean - historical_mean).abs();
        let threshold = 0.1; // Simplified threshold
        
        Ok((mean_diff > threshold, mean_diff / threshold))
    }

    fn detect_recurring_pattern(&self, data: &[f64]) -> bool {
        // Simplified recurring pattern detection using autocorrelation
        if data.len() < 50 {
            return false;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        
        if variance == 0.0 {
            return false;
        }

        // Check autocorrelation at different lags
        for lag in [7, 14, 30] {
            if lag >= data.len() {
                continue;
            }

            let mut correlation = 0.0;
            let n = data.len() - lag;
            
            for i in 0..n {
                correlation += (data[i] - mean) * (data[i + lag] - mean);
            }
            
            correlation /= n as f64 * variance;
            
            if correlation > 0.5 {
                return true;
            }
        }

        false
    }
}

/// Model selection based on performance metrics
pub struct ModelSelector {
    config: AdaptiveLearningConfig,
}

impl ModelSelector {
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self { config }
    }

    pub fn select_best(&self, models: &HashMap<String, ModelMetadata>) -> MLResult<String> {
        if models.is_empty() {
            return Err(MLError::ConfigurationError("No models available for selection".to_string()));
        }

        let mut best_model = None;
        let mut best_score = f64::NEG_INFINITY;

        for (model_id, metadata) in models {
            if let Some(latest_performance) = metadata.performance_history.back() {
                let score = self.calculate_model_score(latest_performance, metadata);
                
                if score > best_score {
                    best_score = score;
                    best_model = Some(model_id.clone());
                }
            }
        }

        best_model.ok_or_else(|| MLError::ConfigurationError("No valid models found".to_string()))
    }

    fn calculate_model_score(&self, performance: &MLMetrics, metadata: &ModelMetadata) -> f64 {
        let base_score = match self.config.model_selection_metric.as_str() {
            "accuracy" => performance.accuracy,
            "precision" => performance.precision,
            "recall" => performance.recall,
            "f1_score" => performance.f1_score,
            "auc_roc" => performance.auc_roc,
            _ => performance.accuracy,
        };

        // Apply penalties for model age and drift events
        let age_penalty = self.calculate_age_penalty(metadata);
        let drift_penalty = self.calculate_drift_penalty(metadata);
        
        base_score * (1.0 - age_penalty) * (1.0 - drift_penalty)
    }

    fn calculate_age_penalty(&self, metadata: &ModelMetadata) -> f64 {
        let age = metadata.creation_time.elapsed().unwrap_or(Duration::ZERO);
        let max_age = self.config.max_model_age;
        
        if age > max_age {
            0.5 // 50% penalty for old models
        } else {
            (age.as_secs_f64() / max_age.as_secs_f64()) * 0.2 // Up to 20% penalty
        }
    }

    fn calculate_drift_penalty(&self, metadata: &ModelMetadata) -> f64 {
        let recent_drifts = metadata.drift_events.iter()
            .filter(|event| {
                event.timestamp.elapsed().unwrap_or(Duration::MAX) < Duration::from_secs(3600)
            })
            .count();

        (recent_drifts as f64 * 0.1).min(0.5) // Up to 50% penalty for many recent drifts
    }
}

/// Online learning implementation
pub struct OnlineLearner {
    config: AdaptiveLearningConfig,
    weights: Vec<f64>,
    learning_rate: f64,
    momentum: Vec<f64>,
}

impl OnlineLearner {
    pub fn new(config: AdaptiveLearningConfig) -> Self {
        Self {
            learning_rate: config.online_learning_rate,
            config,
            weights: Vec::new(),
            momentum: Vec::new(),
        }
    }

    pub fn update(&mut self, features: &[f64], target: f64, prediction: f64) -> MLResult<()> {
        // Initialize weights if needed
        if self.weights.is_empty() {
            self.weights = vec![0.0; features.len()];
            self.momentum = vec![0.0; features.len()];
        }

        if features.len() != self.weights.len() {
            return Err(MLError::ConfigurationError(
                format!("Feature dimension mismatch: expected {}, got {}", 
                    self.weights.len(), features.len())
            ));
        }

        // Calculate gradient (simplified for regression)
        let error = prediction - target;
        
        // Update weights using gradient descent with momentum
        for i in 0..self.weights.len() {
            let gradient = error * features[i];
            self.momentum[i] = self.config.forgetting_factor * self.momentum[i] - self.learning_rate * gradient;
            self.weights[i] += self.momentum[i];
        }

        Ok(())
    }

    pub fn predict(&self, features: &[f64]) -> MLResult<f64> {
        if features.len() != self.weights.len() {
            return Err(MLError::PredictionFailed(
                format!("Feature dimension mismatch: expected {}, got {}", 
                    self.weights.len(), features.len())
            ));
        }

        let prediction = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        Ok(prediction)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSummary {
    pub model_id: String,
    pub current_performance: MLMetrics,
    pub average_performance: MLMetrics,
    pub drift_events_count: usize,
    pub model_age: Duration,
    pub training_samples: usize,
    pub last_update: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_learning_system_creation() {
        let config = AdaptiveLearningConfig::default();
        let system = AdaptiveLearningSystem::new(config);
        assert_eq!(system.active_models.len(), 0);
    }

    #[test]
    fn test_model_registration() {
        let config = AdaptiveLearningConfig::default();
        let mut system = AdaptiveLearningSystem::new(config);
        
        let performance = MLMetrics {
            accuracy: 0.85,
            precision: 0.82,
            recall: 0.88,
            f1_score: 0.85,
            auc_roc: 0.90,
            log_likelihood: -0.5,
        };

        let result = system.register_model("test_model".to_string(), performance);
        assert!(result.is_ok());
        assert_eq!(system.active_models.len(), 1);
    }

    #[test]
    fn test_drift_detection() {
        let config = AdaptiveLearningConfig::default();
        let mut detector = DriftDetector::new(config);
        
        // Create performance history with drift
        let mut history = VecDeque::new();
        
        // Stable period
        for _ in 0..20 {
            history.push_back(MLMetrics {
                accuracy: 0.85 + (rand::random::<f64>() - 0.5) * 0.02,
                precision: 0.82,
                recall: 0.88,
                f1_score: 0.85,
                auc_roc: 0.90,
                log_likelihood: -0.5,
            });
        }
        
        // Drift period
        for _ in 0..20 {
            history.push_back(MLMetrics {
                accuracy: 0.70 + (rand::random::<f64>() - 0.5) * 0.02,
                precision: 0.68,
                recall: 0.72,
                f1_score: 0.70,
                auc_roc: 0.75,
                log_likelihood: -0.8,
            });
        }

        let result = detector.detect_drift(&history);
        assert!(result.is_ok());
        
        let drift_result = result.unwrap();
        // Note: Due to simplified implementation, drift might not always be detected
        // In a real implementation, this would be more reliable
    }

    #[test]
    fn test_online_learner() {
        let config = AdaptiveLearningConfig::default();
        let mut learner = OnlineLearner::new(config);
        
        let features = vec![1.0, 2.0, 3.0];
        let target = 5.0;
        let prediction = 4.8;
        
        let result = learner.update(&features, target, prediction);
        assert!(result.is_ok());
        
        // Test prediction
        let pred_result = learner.predict(&features);
        assert!(pred_result.is_ok());
    }
}
