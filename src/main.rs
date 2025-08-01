use bid_ask_spread_estimation::{SpreadEstimator, TradeData};
use chrono::DateTime;
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use tracing::{info, Level};
use tracing_subscriber;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    // Parse command line arguments or use default values
    let window_size = 50;
    let confidence_level = 0.95;

    // Load and process data
    let price_data = load_price_data("data/trades.csv")?;
    info!("Loaded {} price observations", price_data.len());

    // Initialize spread estimator
    let estimator = SpreadEstimator::new(price_data, window_size, confidence_level)?;

    // Calculate spread estimates
    let metrics = estimator.estimate_spread()?;
    
    // Output results
    println!("\nBid-Ask Spread Estimation Results:");
    println!("----------------------------------");
    println!("Estimated Spread: {:.6}", metrics.spread);
    println!("95% Confidence Interval: ({:.6}, {:.6})", 
             metrics.confidence_interval.0, 
             metrics.confidence_interval.1);
    println!("Serial Covariance: {:.6}", metrics.serial_covariance);
    println!("Return Variance: {:.6}", metrics.variance);
    println!("Sample Size: {}", metrics.sample_size);

    Ok(())
}

fn load_price_data<P: AsRef<Path>>(path: P) -> Result<Vec<TradeData>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let timestamp = DateTime::parse_from_rfc3339(&record[0])?.timestamp();
        let price = record[1].parse::<f64>()?;
        let volume = record[2].parse::<f64>()?;
        let direction = record.get(3).and_then(|d| d.parse::<i8>().ok());

        data.push(TradeData {
            timestamp,
            price,
            volume,
            direction,
        });
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_price_data() -> Result<(), Box<dyn Error>> {
        let dir = tempdir()?;
        let file_path = dir.path().join("test_data.csv");
        let mut file = File::create(&file_path)?;

        writeln!(
            file,
            "timestamp,price,volume,direction\n\
             2023-01-01T00:00:00+00:00,100.0,1000,1\n\
             2023-01-01T00:00:01+00:00,100.1,1500,-1"
        )?;

        let data = load_price_data(file_path)?;
        assert_eq!(data.len(), 2);
        assert!(data[0].price == 100.0);
        assert!(data[1].volume == 1500.0);
        assert_eq!(data[1].direction, Some(-1));

        Ok(())
    }
}
