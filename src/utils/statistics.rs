//! Statistical utility functions for risk calculations

/// Calculate the mean of a slice of f64 values
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate the standard deviation of a slice of f64 values
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    
    let mean_val = mean(values);
    let variance = values.iter()
        .map(|v| (v - mean_val).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    
    variance.sqrt()
}

/// Calculate the percentile of a slice of f64 values
pub fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len();
    let index = (p * (n - 1) as f64) as usize;
    
    if index >= n - 1 {
        sorted[n - 1]
    } else {
        let lower = sorted[index];
        let upper = sorted[index + 1];
        let weight = p * (n - 1) as f64 - index as f64;
        lower + weight * (upper - lower)
    }
}

/// Calculate the variance of a slice of f64 values
pub fn variance(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    
    let mean_val = mean(values);
    values.iter()
        .map(|v| (v - mean_val).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64
}

/// Calculate the skewness of a slice of f64 values
pub fn skewness(values: &[f64]) -> f64 {
    if values.len() < 3 {
        return 0.0;
    }
    
    let mean_val = mean(values);
    let std_val = std_dev(values);
    
    if std_val == 0.0 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let sum_cubed = values.iter()
        .map(|v| ((v - mean_val) / std_val).powi(3))
        .sum::<f64>();
    
    (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
}

/// Calculate the kurtosis of a slice of f64 values
pub fn kurtosis(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return 0.0;
    }
    
    let mean_val = mean(values);
    let std_val = std_dev(values);
    
    if std_val == 0.0 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let sum_fourth = values.iter()
        .map(|v| ((v - mean_val) / std_val).powi(4))
        .sum::<f64>();
    
    let numerator = n * (n + 1.0) * sum_fourth - 3.0 * (n - 1.0).powi(2);
    let denominator = (n - 1.0) * (n - 2.0) * (n - 3.0);
    
    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&values), 3.0);
    }

    #[test]
    fn test_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_dev(&values);
        assert!((result - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 0.5), 3.0);
        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
    }

    #[test]
    fn test_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = variance(&values);
        assert!((result - 2.5).abs() < 1e-10);
    }
}