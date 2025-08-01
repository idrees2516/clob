use std::error::Error;
use std::fmt;
use std::io;

#[derive(Debug)]
pub enum DataError {
    InvalidFormat(String),
    InvalidValue(String),
    MissingField(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::InvalidFormat(msg) => write!(f, "Invalid data format: {}", msg),
            DataError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
            DataError::MissingField(msg) => write!(f, "Missing field: {}", msg),
        }
    }
}

impl Error for DataError {}

#[derive(Debug)]
pub enum EstimationError {
    InsufficientData(String),
    ComputationError(String),
    IoError(io::Error),
    DataError(DataError),
}

impl fmt::Display for EstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EstimationError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            EstimationError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            EstimationError::IoError(err) => write!(f, "IO error: {}", err),
            EstimationError::DataError(err) => write!(f, "Data error: {}", err),
        }
    }
}

impl Error for EstimationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EstimationError::IoError(err) => Some(err),
            EstimationError::DataError(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for EstimationError {
    fn from(err: io::Error) -> Self {
        EstimationError::IoError(err)
    }
}

impl From<DataError> for EstimationError {
    fn from(err: DataError) -> Self {
        EstimationError::DataError(err)
    }
}

#[derive(Debug)]
pub enum ValidationError {
    InvalidPrice(String),
    InvalidVolume(String),
    InvalidTimestamp(String),
    InvalidDirection(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidPrice(msg) => write!(f, "Invalid price: {}", msg),
            ValidationError::InvalidVolume(msg) => write!(f, "Invalid volume: {}", msg),
            ValidationError::InvalidTimestamp(msg) => write!(f, "Invalid timestamp: {}", msg),
            ValidationError::InvalidDirection(msg) => write!(f, "Invalid direction: {}", msg),
        }
    }
}

impl Error for ValidationError {}

#[derive(Debug)]
pub enum InfrastructureError {
    NetworkError(String),
    MemoryError(String),
    NumaError(String),
    ThreadingError(String),
    MonitoringError(String),
    ConfigError(String),
    InvalidMessage(String),
}

impl fmt::Display for InfrastructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InfrastructureError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            InfrastructureError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            InfrastructureError::NumaError(msg) => write!(f, "NUMA error: {}", msg),
            InfrastructureError::ThreadingError(msg) => write!(f, "Threading error: {}", msg),
            InfrastructureError::MonitoringError(msg) => write!(f, "Monitoring error: {}", msg),
            InfrastructureError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            InfrastructureError::InvalidMessage(msg) => write!(f, "Invalid message: {}", msg),
        }
    }
}

impl Error for InfrastructureError {}

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;
