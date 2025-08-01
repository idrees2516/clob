//! Configuration module for compute backend and GPU settings

use serde::{Deserialize};
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct ComputeConfig {
    pub backend: ComputeBackend,
    pub gpu_device: Option<String>,
    pub gpu_enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeBackend {
    Cpu,
    Gpu,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            gpu_device: None,
            gpu_enabled: false,
        }
    }
}

impl ComputeConfig {
    pub fn from_env() -> Self {
        let mut config = ComputeConfig::default();
        if let Ok(backend) = env::var("RUST_COMPUTE_BACKEND") {
            config.backend = match backend.to_lowercase().as_str() {
                "gpu" => ComputeBackend::Gpu,
                _ => ComputeBackend::Cpu,
            };
        }
        if let Ok(enabled) = env::var("RUST_GPU") {
            config.gpu_enabled = enabled == "on" || enabled == "1";
        }
        if let Ok(device) = env::var("RUST_GPU_DEVICE") {
            config.gpu_device = Some(device);
        }
        config
    }

    pub fn from_toml<P: AsRef<Path>>(path: P) -> Self {
        let content = fs::read_to_string(path).unwrap_or_default();
        toml::from_str(&content).unwrap_or_else(|_| ComputeConfig::default())
    }
} 