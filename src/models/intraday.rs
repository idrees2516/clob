use chrono::{NaiveDateTime, Timelike};
use nalgebra as na;
use rayon::prelude::*;
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum IntradayError {
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

#[derive(Debug, Clone)]
pub struct IntradayObservation {
    pub timestamp: NaiveDateTime,
    pub spread: f64,
    pub volume: f64,
    pub price: f64,
    pub volatility: f64,
}

pub struct IntradayAnalysis {
    interval_minutes: u32,
    min_observations: usize,
    smoothing_window: usize,
}

impl IntradayAnalysis {
    pub fn new(
        interval_minutes: u32,
        min_observations: usize,
        smoothing_window: usize,
    ) -> Result<Self, IntradayError> {
        if interval_minutes == 0 {
            return Err(IntradayError::InvalidData(
                "Interval minutes must be positive".to_string(),
            ));
        }

        Ok(Self {
            interval_minutes,
            min_observations,
            smoothing_window,
        })
    }

    pub fn analyze_patterns(
        &self,
        observations: &[IntradayObservation],
    ) -> Result<IntradayPatterns, IntradayError> {
        let binned_data = self.bin_observations(observations)?;
        let patterns = self.compute_patterns(&binned_data)?;
        let seasonality = self.estimate_seasonality(&binned_data)?;
        let auctions = self.analyze_auctions(observations)?;
        
        Ok(IntradayPatterns {
            patterns,
            seasonality,
            auctions,
        })
    }

    fn bin_observations(
        &self,
        observations: &[IntradayObservation],
    ) -> Result<HashMap<u32, Vec<IntradayObservation>>, IntradayError> {
        let mut binned = HashMap::new();
        
        for obs in observations {
            let minutes_since_open = obs.timestamp.hour() * 60 + obs.timestamp.minute();
            let bin = minutes_since_open / self.interval_minutes;
            
            binned.entry(bin)
                .or_insert_with(Vec::new)
                .push(obs.clone());
        }

        Ok(binned)
    }

    fn compute_patterns(
        &self,
        binned_data: &HashMap<u32, Vec<IntradayObservation>>,
    ) -> Result<IntradayMetrics, IntradayError> {
        let mut spreads = Vec::new();
        let mut volumes = Vec::new();
        let mut volatilities = Vec::new();
        
        for bin in 0..24 * 60 / self.interval_minutes {
            if let Some(observations) = binned_data.get(&bin) {
                if observations.len() >= self.min_observations {
                    let mean_spread = observations.iter()
                        .map(|o| o.spread)
                        .sum::<f64>() / observations.len() as f64;
                    
                    let mean_volume = observations.iter()
                        .map(|o| o.volume)
                        .sum::<f64>() / observations.len() as f64;
                    
                    let mean_volatility = observations.iter()
                        .map(|o| o.volatility)
                        .sum::<f64>() / observations.len() as f64;
                    
                    spreads.push(mean_spread);
                    volumes.push(mean_volume);
                    volatilities.push(mean_volatility);
                }
            }
        }

        // Apply smoothing
        let smoothed_spreads = self.apply_smoothing(&spreads);
        let smoothed_volumes = self.apply_smoothing(&volumes);
        let smoothed_volatilities = self.apply_smoothing(&volatilities);

        Ok(IntradayMetrics {
            spreads: smoothed_spreads,
            volumes: smoothed_volumes,
            volatilities: smoothed_volatilities,
        })
    }

    fn apply_smoothing(&self, series: &[f64]) -> Vec<f64> {
        let n = series.len();
        if n < self.smoothing_window {
            return series.to_vec();
        }

        let mut smoothed = Vec::with_capacity(n);
        let half_window = self.smoothing_window / 2;

        for i in 0..n {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);
            let window = &series[start..end];
            
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            smoothed.push(mean);
        }

        smoothed
    }

    fn estimate_seasonality(
        &self,
        binned_data: &HashMap<u32, Vec<IntradayObservation>>,
    ) -> Result<SeasonalityComponents, IntradayError> {
        let n_components = 4; // Number of harmonic components
        let n_bins = 24 * 60 / self.interval_minutes as usize;
        
        let mut time_series = Vec::with_capacity(n_bins);
        for bin in 0..n_bins {
            if let Some(observations) = binned_data.get(&(bin as u32)) {
                let mean_spread = observations.iter()
                    .map(|o| o.spread)
                    .sum::<f64>() / observations.len() as f64;
                time_series.push(mean_spread);
            } else {
                time_series.push(0.0);
            }
        }

        // Construct design matrix for harmonic regression
        let mut X = na::DMatrix::zeros(n_bins, 2 * n_components + 1);
        for i in 0..n_bins {
            X[(i, 0)] = 1.0; // Intercept
            for j in 0..n_components {
                let freq = 2.0 * std::f64::consts::PI * (j + 1) as f64 / n_bins as f64;
                X[(i, 2*j + 1)] = (freq * i as f64).cos();
                X[(i, 2*j + 2)] = (freq * i as f64).sin();
            }
        }

        let y = na::DVector::from_vec(time_series);
        
        // Solve using OLS
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let coefficients = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(IntradayError::AnalysisError(
                "Singular matrix in seasonality estimation".to_string(),
            )),
        };

        let mut components = Vec::with_capacity(n_components);
        for i in 0..n_components {
            components.push(HarmonicComponent {
                frequency: (i + 1) as f64 * 2.0 * std::f64::consts::PI / n_bins as f64,
                amplitude: (coefficients[2*i + 1].powi(2) + coefficients[2*i + 2].powi(2)).sqrt(),
                phase: coefficients[2*i + 2].atan2(coefficients[2*i + 1]),
            });
        }

        Ok(SeasonalityComponents {
            mean: coefficients[0],
            components,
        })
    }

    fn analyze_auctions(
        &self,
        observations: &[IntradayObservation],
    ) -> Result<AuctionMetrics, IntradayError> {
        let mut opening = Vec::new();
        let mut closing = Vec::new();

        for obs in observations {
            let hour = obs.timestamp.hour();
            let minute = obs.timestamp.minute();

            if hour == 9 && minute <= 30 {
                opening.push(obs);
            } else if hour == 15 && minute >= 30 {
                closing.push(obs);
            }
        }

        Ok(AuctionMetrics {
            opening: self.compute_auction_metrics(&opening)?,
            closing: self.compute_auction_metrics(&closing)?,
        })
    }

    fn compute_auction_metrics(
        &self,
        observations: &[&IntradayObservation],
    ) -> Result<AuctionPeriodMetrics, IntradayError> {
        if observations.is_empty() {
            return Err(IntradayError::InvalidData(
                "No observations for auction period".to_string(),
            ));
        }

        let mean_spread = observations.iter()
            .map(|o| o.spread)
            .sum::<f64>() / observations.len() as f64;

        let mean_volume = observations.iter()
            .map(|o| o.volume)
            .sum::<f64>() / observations.len() as f64;

        let mean_volatility = observations.iter()
            .map(|o| o.volatility)
            .sum::<f64>() / observations.len() as f64;

        let price_impact = self.estimate_price_impact(observations);

        Ok(AuctionPeriodMetrics {
            mean_spread,
            mean_volume,
            mean_volatility,
            price_impact,
        })
    }

    fn estimate_price_impact(&self, observations: &[&IntradayObservation]) -> f64 {
        if observations.len() < 2 {
            return 0.0;
        }

        let initial_price = observations.first().unwrap().price;
        let final_price = observations.last().unwrap().price;
        let total_volume: f64 = observations.iter().map(|o| o.volume).sum();

        (final_price - initial_price).abs() / total_volume
    }
}

#[derive(Debug)]
pub struct IntradayMetrics {
    pub spreads: Vec<f64>,
    pub volumes: Vec<f64>,
    pub volatilities: Vec<f64>,
}

#[derive(Debug)]
pub struct HarmonicComponent {
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
}

#[derive(Debug)]
pub struct SeasonalityComponents {
    pub mean: f64,
    pub components: Vec<HarmonicComponent>,
}

#[derive(Debug)]
pub struct AuctionPeriodMetrics {
    pub mean_spread: f64,
    pub mean_volume: f64,
    pub mean_volatility: f64,
    pub price_impact: f64,
}

#[derive(Debug)]
pub struct AuctionMetrics {
    pub opening: AuctionPeriodMetrics,
    pub closing: AuctionPeriodMetrics,
}

#[derive(Debug)]
pub struct IntradayPatterns {
    pub patterns: IntradayMetrics,
    pub seasonality: SeasonalityComponents,
    pub auctions: AuctionMetrics,
}
