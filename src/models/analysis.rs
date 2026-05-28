use thiserror::Error;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SpreadDynamics {
    pub persistence: f64,
    pub half_life: f64,
    pub volatility: f64,
    pub acf: Vec<f64>,
    pub pacf: Vec<f64>,
    pub seasonal_factors: Option<Vec<f64>>,
    pub trend_component: Option<Vec<f64>>,
}

#[derive(Debug)]
pub struct CommonalityAnalysis {
    pub market_beta: f64,
    pub industry_beta: f64,
    pub idiosyncratic_component: f64,
    pub r_squared: f64,
    pub factor_loadings: HashMap<String, f64>,
    pub cross_correlations: Vec<f64>,
}

#[derive(Debug)]
pub struct StructuralBreaks {
    pub break_points: Vec<usize>,
    pub break_dates: Vec<i64>,
    pub test_statistics: Vec<f64>,
    pub p_values: Vec<f64>,
    pub regime_means: Vec<f64>,
    pub regime_volatilities: Vec<f64>,
}

#[derive(Debug)]
pub struct LiquidityProvision {
    pub market_maker_participation: f64,
    pub tick_size_clustering: f64,
    pub quote_duration: f64,
    pub depth_profile: Vec<(f64, f64)>,
    pub resiliency_measures: ResiliencyMeasures,
}

#[derive(Debug)]
pub struct ResiliencyMeasures {
    pub price_impact_decay: f64,
    pub order_book_recovery: f64,
    pub volume_replenishment: f64,
    pub spread_recovery_time: f64,
}

#[derive(Debug)]
pub struct TradingPatterns {
    pub intraday_seasonality: Vec<f64>,
    pub trade_size_distribution: HashMap<String, f64>,
    pub order_type_distribution: HashMap<String, f64>,
    pub execution_quality: ExecutionQuality,
}

#[derive(Debug)]
pub struct ExecutionQuality {
    pub implementation_shortfall: f64,
    pub price_improvement: f64,
    pub execution_speed: f64,
    pub fill_rates: HashMap<String, f64>,
}

#[derive(Debug)]
pub struct AlternativeSpecifications {
    pub different_lags: HashMap<usize, f64>,
    pub different_estimators: HashMap<String, f64>,
    pub different_weightings: HashMap<String, f64>,
    pub model_diagnostics: ModelDiagnostics,
}

#[derive(Debug)]
pub struct ModelDiagnostics {
    pub residual_normality: f64,
    pub residual_autocorrelation: Vec<f64>,
    pub heteroskedasticity_test: f64,
    pub specification_test: f64,
}

#[derive(Debug)]
pub struct BootstrapResults {
    pub spread_distribution: Vec<f64>,
    pub confidence_intervals: Vec<(f64, f64)>,
    pub standard_errors: Vec<f64>,
    pub bias_estimates: Vec<f64>,
}

#[derive(Debug)]
pub struct SensitivityAnalysis {
    pub parameter_elasticity: HashMap<String, f64>,
    pub outlier_impact: HashMap<String, f64>,
    pub sample_sensitivity: HashMap<String, f64>,
    pub cross_validation: CrossValidationResults,
}

#[derive(Debug)]
pub struct CrossValidationResults {
    pub mean_error: f64,
    pub std_error: f64,
    pub fold_results: Vec<f64>,
    pub stability_metrics: HashMap<String, f64>,
}

impl SpreadDynamics {
    pub fn new(
        persistence: f64,
        half_life: f64,
        volatility: f64,
        acf: Vec<f64>,
        pacf: Vec<f64>,
        seasonal_factors: Option<Vec<f64>>,
        trend_component: Option<Vec<f64>>,
    ) -> Self {
        Self {
            persistence,
            half_life,
            volatility,
            acf,
            pacf,
            seasonal_factors,
            trend_component,
        }
    }
}

impl CommonalityAnalysis {
    pub fn new(
        market_beta: f64,
        industry_beta: f64,
        idiosyncratic_component: f64,
        r_squared: f64,
        factor_loadings: HashMap<String, f64>,
        cross_correlations: Vec<f64>,
    ) -> Self {
        Self {
            market_beta,
            industry_beta,
            idiosyncratic_component,
            r_squared,
            factor_loadings,
            cross_correlations,
        }
    }
}

impl StructuralBreaks {
    pub fn new(
        break_points: Vec<usize>,
        break_dates: Vec<i64>,
        test_statistics: Vec<f64>,
        p_values: Vec<f64>,
        regime_means: Vec<f64>,
        regime_volatilities: Vec<f64>,
    ) -> Self {
        Self {
            break_points,
            break_dates,
            test_statistics,
            p_values,
            regime_means,
            regime_volatilities,
        }
    }
}

impl LiquidityProvision {
    pub fn new(
        market_maker_participation: f64,
        tick_size_clustering: f64,
        quote_duration: f64,
        depth_profile: Vec<(f64, f64)>,
        resiliency_measures: ResiliencyMeasures,
    ) -> Self {
        Self {
            market_maker_participation,
            tick_size_clustering,
            quote_duration,
            depth_profile,
            resiliency_measures,
        }
    }
}

impl ResiliencyMeasures {
    pub fn new(
        price_impact_decay: f64,
        order_book_recovery: f64,
        volume_replenishment: f64,
        spread_recovery_time: f64,
    ) -> Self {
        Self {
            price_impact_decay,
            order_book_recovery,
            volume_replenishment,
            spread_recovery_time,
        }
    }
}

impl TradingPatterns {
    pub fn new(
        intraday_seasonality: Vec<f64>,
        trade_size_distribution: HashMap<String, f64>,
        order_type_distribution: HashMap<String, f64>,
        execution_quality: ExecutionQuality,
    ) -> Self {
        Self {
            intraday_seasonality,
            trade_size_distribution,
            order_type_distribution,
            execution_quality,
        }
    }
}

impl ExecutionQuality {
    pub fn new(
        implementation_shortfall: f64,
        price_improvement: f64,
        execution_speed: f64,
        fill_rates: HashMap<String, f64>,
    ) -> Self {
        Self {
            implementation_shortfall,
            price_improvement,
            execution_speed,
            fill_rates,
        }
    }
}

impl AlternativeSpecifications {
    pub fn new(
        different_lags: HashMap<usize, f64>,
        different_estimators: HashMap<String, f64>,
        different_weightings: HashMap<String, f64>,
        model_diagnostics: ModelDiagnostics,
    ) -> Self {
        Self {
            different_lags,
            different_estimators,
            different_weightings,
            model_diagnostics,
        }
    }
}

impl ModelDiagnostics {
    pub fn new(
        residual_normality: f64,
        residual_autocorrelation: Vec<f64>,
        heteroskedasticity_test: f64,
        specification_test: f64,
    ) -> Self {
        Self {
            residual_normality,
            residual_autocorrelation,
            heteroskedasticity_test,
            specification_test,
        }
    }
}

impl BootstrapResults {
    pub fn new(
        spread_distribution: Vec<f64>,
        confidence_intervals: Vec<(f64, f64)>,
        standard_errors: Vec<f64>,
        bias_estimates: Vec<f64>,
    ) -> Self {
        Self {
            spread_distribution,
            confidence_intervals,
            standard_errors,
            bias_estimates,
        }
    }
}

impl SensitivityAnalysis {
    pub fn new(
        parameter_elasticity: HashMap<String, f64>,
        outlier_impact: HashMap<String, f64>,
        sample_sensitivity: HashMap<String, f64>,
        cross_validation: CrossValidationResults,
    ) -> Self {
        Self {
            parameter_elasticity,
            outlier_impact,
            sample_sensitivity,
            cross_validation,
        }
    }
}

impl CrossValidationResults {
    pub fn new(
        mean_error: f64,
        std_error: f64,
        fold_results: Vec<f64>,
        stability_metrics: HashMap<String, f64>,
    ) -> Self {
        Self {
            mean_error,
            std_error,
            fold_results,
            stability_metrics,
        }
    }
}
