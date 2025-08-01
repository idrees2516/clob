use nalgebra as na;
use rayon::prelude::*;
use statrs::distribution::{Normal, StudentsT, ContinuousCDF};
use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug, Error)]
pub enum CrossSectionalError {
    #[error("Estimation error: {0}")]
    EstimationError(String),
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

#[derive(Debug, Clone)]
pub struct SecurityCharacteristics {
    pub market_cap: f64,
    pub volume: f64,
    pub volatility: f64,
    pub price: f64,
    pub sector: String,
}

pub struct CrossSectionalAnalysis {
    confidence_level: f64,
    min_observations: usize,
    max_outlier_z_score: f64,
}

impl CrossSectionalAnalysis {
    pub fn new(
        confidence_level: f64,
        min_observations: usize,
        max_outlier_z_score: f64,
    ) -> Result<Self, CrossSectionalError> {
        if !(0.0..=1.0).contains(&confidence_level) {
            return Err(CrossSectionalError::InvalidData(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            confidence_level,
            min_observations,
            max_outlier_z_score,
        })
    }

    pub fn panel_regression(
        &self,
        spreads: &HashMap<String, Vec<f64>>,
        characteristics: &HashMap<String, Vec<SecurityCharacteristics>>,
    ) -> Result<PanelRegressionResults, CrossSectionalError> {
        let mut regression_data = Vec::new();
        
        // Prepare panel data
        for (ticker, spread_series) in spreads {
            if let Some(char_series) = characteristics.get(ticker) {
                if spread_series.len() != char_series.len() {
                    return Err(CrossSectionalError::InvalidData(
                        format!("Mismatched data lengths for ticker {}", ticker)
                    ));
                }

                for (i, &spread) in spread_series.iter().enumerate() {
                    let chars = &char_series[i];
                    regression_data.push(RegressionObservation {
                        spread,
                        log_market_cap: chars.market_cap.ln(),
                        log_volume: chars.volume.ln(),
                        volatility: chars.volatility,
                        log_price: chars.price.ln(),
                        sector: chars.sector.clone(),
                    });
                }
            }
        }

        // Remove outliers
        let filtered_data = self.remove_outliers(&regression_data)?;

        // Perform panel regression
        let results = self.estimate_panel_coefficients(&filtered_data)?;
        
        // Calculate standard errors and t-stats
        let (std_errors, t_stats) = self.calculate_statistics(&filtered_data, &results)?;

        // Sector analysis
        let sector_effects = self.analyze_sector_effects(&filtered_data)?;

        Ok(PanelRegressionResults {
            coefficients: results,
            standard_errors: std_errors,
            t_statistics: t_stats,
            sector_effects,
            n_observations: filtered_data.len(),
        })
    }

    fn remove_outliers(
        &self,
        data: &[RegressionObservation],
    ) -> Result<Vec<RegressionObservation>, CrossSectionalError> {
        let spreads: Vec<f64> = data.iter().map(|x| x.spread).collect();
        let mean = spreads.iter().sum::<f64>() / spreads.len() as f64;
        let std_dev = (spreads.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (spreads.len() - 1) as f64)
            .sqrt();

        Ok(data.iter()
            .filter(|obs| {
                let z_score = (obs.spread - mean).abs() / std_dev;
                z_score <= self.max_outlier_z_score
            })
            .cloned()
            .collect())
    }

    fn estimate_panel_coefficients(
        &self,
        data: &[RegressionObservation],
    ) -> Result<RegressionCoefficients, CrossSectionalError> {
        let n = data.len();
        if n < self.min_observations {
            return Err(CrossSectionalError::InvalidData(
                "Insufficient observations for regression".to_string(),
            ));
        }

        // Prepare design matrix X and response vector y
        let mut X = na::DMatrix::zeros(n, 4);
        let mut y = na::DVector::zeros(n);

        for (i, obs) in data.iter().enumerate() {
            X[(i, 0)] = obs.log_market_cap;
            X[(i, 1)] = obs.log_volume;
            X[(i, 2)] = obs.volatility;
            X[(i, 3)] = obs.log_price;
            y[i] = obs.spread;
        }

        // Add constant term
        let ones = na::DVector::from_element(n, 1.0);
        let X = na::DMatrix::from_columns(&[ones.as_slice(), X.as_slice()]);

        // Compute coefficients using OLS
        let XtX = X.transpose() * &X;
        let Xty = X.transpose() * y;
        
        let coefficients = match XtX.try_inverse() {
            Some(XtX_inv) => XtX_inv * Xty,
            None => return Err(CrossSectionalError::EstimationError(
                "Singular matrix in regression".to_string(),
            )),
        };

        Ok(RegressionCoefficients {
            intercept: coefficients[0],
            market_cap: coefficients[1],
            volume: coefficients[2],
            volatility: coefficients[3],
            price: coefficients[4],
        })
    }

    fn calculate_statistics(
        &self,
        data: &[RegressionObservation],
        coef: &RegressionCoefficients,
    ) -> Result<(RegressionCoefficients, RegressionCoefficients), CrossSectionalError> {
        let n = data.len();
        let k = 5; // number of parameters

        // Calculate residuals and residual variance
        let mut residual_sum_squares = 0.0;
        for obs in data {
            let predicted = coef.intercept
                + coef.market_cap * obs.log_market_cap
                + coef.volume * obs.log_volume
                + coef.volatility * obs.volatility
                + coef.price * obs.log_price;
            residual_sum_squares += (obs.spread - predicted).powi(2);
        }
        let residual_variance = residual_sum_squares / (n - k) as f64;

        // Prepare design matrix for variance calculation
        let mut X = na::DMatrix::zeros(n, k);
        for (i, obs) in data.iter().enumerate() {
            X[(i, 0)] = 1.0;
            X[(i, 1)] = obs.log_market_cap;
            X[(i, 2)] = obs.log_volume;
            X[(i, 3)] = obs.volatility;
            X[(i, 4)] = obs.log_price;
        }

        // Calculate variance-covariance matrix
        let XtX_inv = match (X.transpose() * &X).try_inverse() {
            Some(inv) => inv,
            None => return Err(CrossSectionalError::EstimationError(
                "Singular matrix in standard error calculation".to_string(),
            )),
        };

        let var_covar = XtX_inv * residual_variance;

        // Extract standard errors and calculate t-statistics
        let std_errors = RegressionCoefficients {
            intercept: var_covar[(0, 0)].sqrt(),
            market_cap: var_covar[(1, 1)].sqrt(),
            volume: var_covar[(2, 2)].sqrt(),
            volatility: var_covar[(3, 3)].sqrt(),
            price: var_covar[(4, 4)].sqrt(),
        };

        let t_stats = RegressionCoefficients {
            intercept: coef.intercept / std_errors.intercept,
            market_cap: coef.market_cap / std_errors.market_cap,
            volume: coef.volume / std_errors.volume,
            volatility: coef.volatility / std_errors.volatility,
            price: coef.price / std_errors.price,
        };

        Ok((std_errors, t_stats))
    }

    fn analyze_sector_effects(
        &self,
        data: &[RegressionObservation],
    ) -> Result<HashMap<String, SectorEffect>, CrossSectionalError> {
        let mut sector_data: HashMap<String, Vec<f64>> = HashMap::new();
        
        // Group spreads by sector
        for obs in data {
            sector_data.entry(obs.sector.clone())
                .or_default()
                .push(obs.spread);
        }

        let mut sector_effects = HashMap::new();
        let overall_mean: f64 = data.iter().map(|obs| obs.spread).sum::<f64>() / data.len() as f64;

        for (sector, spreads) in sector_data {
            let n = spreads.len();
            if n < self.min_observations {
                continue;
            }

            let mean = spreads.iter().sum::<f64>() / n as f64;
            let variance = spreads.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / (n - 1) as f64;
            
            let std_error = (variance / n as f64).sqrt();
            let t_stat = (mean - overall_mean) / std_error;
            
            let df = n - 1;
            let t_dist = StudentsT::new(df as f64)
                .map_err(|e| CrossSectionalError::EstimationError(e.to_string()))?;
            let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

            sector_effects.insert(sector, SectorEffect {
                mean_spread: mean,
                difference_from_overall: mean - overall_mean,
                standard_error: std_error,
                t_statistic: t_stat,
                p_value,
                n_observations: n,
            });
        }

        Ok(sector_effects)
    }
}

#[derive(Debug, Clone)]
struct RegressionObservation {
    spread: f64,
    log_market_cap: f64,
    log_volume: f64,
    volatility: f64,
    log_price: f64,
    sector: String,
}

#[derive(Debug, Clone)]
pub struct RegressionCoefficients {
    pub intercept: f64,
    pub market_cap: f64,
    pub volume: f64,
    pub volatility: f64,
    pub price: f64,
}

#[derive(Debug)]
pub struct SectorEffect {
    pub mean_spread: f64,
    pub difference_from_overall: f64,
    pub standard_error: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub n_observations: usize,
}

#[derive(Debug)]
pub struct PanelRegressionResults {
    pub coefficients: RegressionCoefficients,
    pub standard_errors: RegressionCoefficients,
    pub t_statistics: RegressionCoefficients,
    pub sector_effects: HashMap<String, SectorEffect>,
    pub n_observations: usize,
}
