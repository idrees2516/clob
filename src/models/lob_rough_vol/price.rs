use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use rust_decimal::{Decimal, MathematicalOps};
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use super::{LOBError, LOBResult};

/// Represents a price level in the order book
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PriceLevel {
    /// The actual price value
    value: Decimal,
    /// The tick size for this price level
    tick_size: Decimal,
}

/// Error type for price-related operations
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceError {
    #[error("Invalid price: {0}")]
    InvalidPrice(Decimal),
    #[error("Invalid tick size: {0}")]
    InvalidTickSize(Decimal),
    #[error("Price not aligned to tick size: {price} not a multiple of {tick_size}")]
    PriceNotAlignedToTick { price: Decimal, tick_size: Decimal },
    #[error("Price out of valid range: {0}")]
    PriceOutOfRange(Decimal),
    #[error("Invalid price format: {0}")]
    InvalidFormat(String),
}

impl From<PriceError> for LOBError {
    fn from(err: PriceError) -> Self {
        LOBError::InvalidPrice(err.to_string().parse().unwrap_or(0.0))
    }
}

/// Configuration for tick sizes at different price levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickSizeTable {
    /// Sorted vector of (price_level, tick_size) pairs
    levels: Vec<(Decimal, Decimal)>,
}

impl Default for TickSizeTable {
    fn default() -> Self {
        // Default tick size table with common values
        let levels = vec![
            (dec!(0.0), dec!(0.0001)),   // 0.01% for prices < 1.0
            (dec!(1.0), dec!(0.0001)),   // 0.01% for prices >= 1.0
            (dec!(10.0), dec!(0.001)),   // 0.1% for prices >= 10.0
            (dec!(100.0), dec!(0.01)),   // 1% for prices >= 100.0
            (dec!(1000.0), dec!(0.1)),   // 10% for prices >= 1000.0
        ];
        Self { levels }
    }
}

impl TickSizeTable {
    /// Create a new tick size table
    pub fn new(levels: Vec<(Decimal, Decimal)>) -> LOBResult<Self> {
        if levels.is_empty() {
            return Err(PriceError::InvalidTickSize(Decimal::ZERO).into());
        }

        // Sort by price level
        let mut levels = levels;
        levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Validate tick sizes
        for (_, tick_size) in &levels {
            if *tick_size <= Decimal::ZERO {
                return Err(PriceError::InvalidTickSize(*tick_size).into());
            }
        }

        Ok(Self { levels })
    }

    /// Get the tick size for a given price
    pub fn get_tick_size(&self, price: Decimal) -> Decimal {
        self.levels
            .iter()
            .rev()
            .find(|&&(level, _)| price >= level)
            .map(|&(_, tick_size)| tick_size)
            .unwrap_or_else(|| self.levels.last().unwrap().1)
    }
}

impl PriceLevel {
    /// Create a new price level
    pub fn new(price: Decimal, tick_size: Decimal) -> LOBResult<Self> {
        if price <= Decimal::ZERO {
            return Err(PriceError::InvalidPrice(price).into());
        }
        if tick_size <= Decimal::ZERO {
            return Err(PriceError::InvalidTickSize(tick_size).into());
        }

        // Round to nearest tick
        let rounded = (price / tick_size).round() * tick_size;
        
        Ok(Self {
            value: rounded,
            tick_size,
        })
    }

    /// Create a price level using a tick size table
    pub fn with_tick_table(price: Decimal, tick_table: &TickSizeTable) -> LOBResult<Self> {
        let tick_size = tick_table.get_tick_size(price);
        Self::new(price, tick_size)
    }

    /// Get the price value
    pub fn value(&self) -> Decimal {
        self.value
    }

    /// Get the tick size
    pub fn tick_size(&self) -> Decimal {
        self.tick_size
    }

    /// Increment the price by one tick
    pub fn increment(&self) -> Self {
        Self {
            value: self.value + self.tick_size,
            tick_size: self.tick_size,
        }
    }

    /// Decrement the price by one tick
    pub fn decrement(&self) -> LOBResult<Self> {
        if self.value <= self.tick_size {
            return Err(PriceError::PriceOutOfRange(self.value - self.tick_size).into());
        }
        
        Ok(Self {
            value: self.value - self.tick_size,
            tick_size: self.tick_size,
        })
    }

    /// Calculate the number of ticks between two prices
    pub fn ticks_between(&self, other: &Self) -> LOBResult<i64> {
        if self.tick_size != other.tick_size {
            return Err(PriceError::InvalidTickSize(self.tick_size).into());
        }
        
        let diff = (self.value - other.value).abs();
        let ticks = diff / self.tick_size;
        
        if ticks.fract() != Decimal::ZERO {
            return Err(PriceError::PriceNotAlignedToTick {
                price: self.value,
                tick_size: self.tick_size,
            }.into());
        }
        
        Ok(ticks.to_i64().unwrap_or(0))
    }
}

impl fmt::Display for PriceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl FromStr for PriceLevel {
    type Err = PriceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = Decimal::from_str(s)
            .map_err(|_| PriceError::InvalidFormat(s.to_string()))?;
            
        if value <= Decimal::ZERO {
            return Err(PriceError::InvalidPrice(value));
        }
        
        // Use default tick size based on the value
        let tick_table = TickSizeTable::default();
        let tick_size = tick_table.get_tick_size(value);
        
        Ok(Self { value, tick_size })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_price_level_creation() {
        let price = PriceLevel::new(dec!(100.0), dec!(0.01)).unwrap();
        assert_eq!(price.value(), dec!(100.0));
        assert_eq!(price.tick_size(), dec!(0.01));
    }

    #[test]
    fn test_price_level_increment() {
        let price = PriceLevel::new(dec!(100.0), dec!(0.01)).unwrap();
        let next = price.increment();
        assert_eq!(next.value(), dec!(100.01));
    }

    #[test]
    fn test_price_level_decrement() {
        let price = PriceLevel::new(dec!(100.0), dec!(0.01)).unwrap();
        let prev = price.decrement().unwrap();
        assert_eq!(prev.value(), dec!(99.99));
    }

    #[test]
    fn test_price_level_ticks_between() {
        let p1 = PriceLevel::new(dec!(100.0), dec!(0.01)).unwrap();
        let p2 = PriceLevel::new(dec!(100.05), dec!(0.01)).unwrap();
        assert_eq!(p1.ticks_between(&p2).unwrap(), 5);
    }

    #[test]
    fn test_tick_size_table() {
        let table = TickSizeTable::default();
        
        assert_eq!(table.get_tick_size(dec!(0.5)), dec!(0.0001));  // 0.01%
        assert_eq!(table.get_tick_size(dec!(1.0)), dec!(0.0001));  // 0.01%
        assert_eq!(table.get_tick_size(dec!(10.0)), dec!(0.001));  // 0.1%
        assert_eq!(table.get_tick_size(dec!(100.0)), dec!(0.01));  // 1%
        assert_eq!(table.get_tick_size(dec!(1000.0)), dec!(0.1));  // 10%
        assert_eq!(table.get_tick_size(dec!(10000.0)), dec!(0.1)); // 10% (uses last level)
    }

    #[test]
    fn test_price_level_with_tick_table() {
        let table = TickSizeTable::default();
        let price = PriceLevel::with_tick_table(dec!(150.0), &table).unwrap();
        assert_eq!(price.tick_size(), dec!(0.01)); // From the 100.0 level
        assert_eq!(price.value(), dec!(150.00));   // Rounded to tick size
    }
}
