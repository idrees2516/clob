pub mod price_comparison;
pub mod memory_operations;
pub mod arithmetic;
pub mod validation;

pub use price_comparison::*;
pub use memory_operations::*;
pub use arithmetic::*;
pub use validation::*;

use std::arch::x86_64::*;

/// SIMD capability detection and initialization
pub struct SimdCapabilities {
    pub has_sse2: bool,
    pub has_sse4_1: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
}

impl SimdCapabilities {
    pub fn detect() -> Self {
        Self {
            has_sse2: is_x86_feature_detected!("sse2"),
            has_sse4_1: is_x86_feature_detected!("sse4.1"),
            has_avx: is_x86_feature_detected!("avx"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
        }
    }

    pub fn best_vector_width(&self) -> usize {
        if self.has_avx512f {
            64 // 512 bits = 64 bytes
        } else if self.has_avx2 || self.has_avx {
            32 // 256 bits = 32 bytes
        } else if self.has_sse2 {
            16 // 128 bits = 16 bytes
        } else {
            8 // Fallback to scalar operations
        }
    }
}

/// Global SIMD capabilities instance
pub static SIMD_CAPS: std::sync::OnceLock<SimdCapabilities> = std::sync::OnceLock::new();

pub fn get_simd_capabilities() -> &'static SimdCapabilities {
    SIMD_CAPS.get_or_init(SimdCapabilities::detect)
}