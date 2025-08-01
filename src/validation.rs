use crate::error::{DataError, ValidationError};
use crate::TradeData;
use chrono::{DateTime, TimeZone, Utc};
use std::str::FromStr;

pub trait Validate {
    fn validate(&self) -> Result<(), ValidationError>;
}

impl Validate for TradeData {
    fn validate(&self) -> Result<(), ValidationError> {
        if self.price <= 0.0 {
            return Err(ValidationError::InvalidPrice(
                "Price must be positive".to_string(),
            ));
        }

        if self.volume <= 0.0 {
            return Err(ValidationError::InvalidVolume(
                "Volume must be positive".to_string(),
            ));
        }

        if let Some(direction) = self.direction {
            if direction != 1 && direction != -1 {
                return Err(ValidationError::InvalidDirection(
                    "Direction must be 1 or -1".to_string(),
                ));
            }
        }

        // Validate timestamp is not in the future
        let current_time = Utc::now().timestamp();
        if self.timestamp > current_time {
            return Err(ValidationError::InvalidTimestamp(
                "Timestamp cannot be in the future".to_string(),
            ));
        }

        Ok(())
    }
}

pub fn validate_csv_record(record: &csv::StringRecord) -> Result<TradeData, DataError> {
    if record.len() < 3 {
        return Err(DataError::InvalidFormat(
            "Record must have at least 3 fields".to_string(),
        ));
    }

    let timestamp = parse_timestamp(&record[0])?;
    let price = parse_price(&record[1])?;
    let volume = parse_volume(&record[2])?;
    let direction = if record.len() > 3 {
        Some(parse_direction(&record[3])?)
    } else {
        None
    };

    let trade = TradeData {
        timestamp,
        price,
        volume,
        direction,
    };

    trade.validate().map_err(|e| DataError::InvalidValue(e.to_string()))?;
    Ok(trade)
}

fn parse_timestamp(s: &str) -> Result<i64, DataError> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.timestamp())
        .map_err(|_| {
            DataError::InvalidFormat(format!(
                "Invalid timestamp format. Expected RFC3339, got: {}",
                s
            ))
        })
}

fn parse_price(s: &str) -> Result<f64, DataError> {
    f64::from_str(s).map_err(|_| {
        DataError::InvalidValue(format!("Invalid price value: {}", s))
    })
}

fn parse_volume(s: &str) -> Result<f64, DataError> {
    f64::from_str(s).map_err(|_| {
        DataError::InvalidValue(format!("Invalid volume value: {}", s))
    })
}

fn parse_direction(s: &str) -> Result<i8, DataError> {
    match s.trim() {
        "1" => Ok(1),
        "-1" => Ok(-1),
        _ => Err(DataError::InvalidValue(format!(
            "Invalid direction value: {}. Expected 1 or -1",
            s
        ))),
    }
}

pub fn validate_data_series(data: &[TradeData]) -> Result<(), DataError> {
    if data.is_empty() {
        return Err(DataError::InsufficientData(
            "Empty data series".to_string(),
        ));
    }

    // Check for monotonically increasing timestamps
    for window in data.windows(2) {
        if window[0].timestamp >= window[1].timestamp {
            return Err(DataError::InvalidFormat(
                "Timestamps must be strictly increasing".to_string(),
            ));
        }
    }

    // Validate each trade
    for trade in data {
        trade.validate()
            .map_err(|e| DataError::InvalidValue(e.to_string()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_trade_data_validation() {
        let valid_trade = TradeData {
            timestamp: Utc::now().timestamp(),
            price: 100.0,
            volume: 1000.0,
            direction: Some(1),
        };
        assert!(valid_trade.validate().is_ok());

        let invalid_price = TradeData {
            timestamp: Utc::now().timestamp(),
            price: -100.0,
            volume: 1000.0,
            direction: Some(1),
        };
        assert!(invalid_price.validate().is_err());

        let invalid_direction = TradeData {
            timestamp: Utc::now().timestamp(),
            price: 100.0,
            volume: 1000.0,
            direction: Some(2),
        };
        assert!(invalid_direction.validate().is_err());
    }

    #[test]
    fn test_csv_record_validation() {
        let record = csv::StringRecord::from(vec![
            "2023-01-01T00:00:00+00:00",
            "100.0",
            "1000",
            "1",
        ]);
        assert!(validate_csv_record(&record).is_ok());

        let invalid_record = csv::StringRecord::from(vec![
            "invalid_timestamp",
            "100.0",
            "1000",
            "1",
        ]);
        assert!(validate_csv_record(&invalid_record).is_err());
    }

    #[test]
    fn test_data_series_validation() {
        let data = vec![
            TradeData {
                timestamp: Utc.timestamp_opt(1000, 0).unwrap().timestamp(),
                price: 100.0,
                volume: 1000.0,
                direction: Some(1),
            },
            TradeData {
                timestamp: Utc.timestamp_opt(2000, 0).unwrap().timestamp(),
                price: 101.0,
                volume: 1000.0,
                direction: Some(-1),
            },
        ];
        assert!(validate_data_series(&data).is_ok());

        let invalid_data = vec![
            TradeData {
                timestamp: Utc.timestamp_opt(2000, 0).unwrap().timestamp(),
                price: 100.0,
                volume: 1000.0,
                direction: Some(1),
            },
            TradeData {
                timestamp: Utc.timestamp_opt(1000, 0).unwrap().timestamp(),
                price: 101.0,
                volume: 1000.0,
                direction: Some(-1),
            },
        ];
        assert!(validate_data_series(&invalid_data).is_err());
    }
}
