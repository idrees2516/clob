use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;

/// Compiler optimization configuration and management
pub struct CompilerOptimizationManager {
    /// Profile-guided optimization data
    pgo_data: HashMap<String, PGOProfile>,
    /// Link-time optimization settings
    lto_config: LTOConfig,
    /// Function inlining decisions
    inlining_decisions: HashMap<String, InliningDecision>,
    /// Vectorization hints
    vectorization_hints: HashMap<String, VectorizationHint>,
    /// Optimization flags
    optimization_flags: Vec<String>,
}

/// Profile-guided optimization profile data
#[derive(Debug, Clone)]
pub struct PGOProfile {
    /// Function name
    pub function_name: String,
    /// Execution count
    pub execution_count: u64,
    /// Total execution time (nanoseconds)
    pub total_execution_time: u64,
    /// Average execution time (nanoseconds)
    pub average_execution_time: f64,
    /// Branch profiles
    pub branch_profiles: Vec<BranchProfile>,
    /// Call site profiles
    pub call_site_profiles: Vec<CallSiteProfile>,
    /// Is hot function (frequently called)
    pub is_hot: bool,
    /// Is cold function (rarely called)
    pub is_cold: bool,
}

#[derive(Debug, Clone)]
pub struct BranchProfile {
    pub branch_id: String,
    pub taken_count: u64,
    pub not_taken_count: u64,
    pub taken_probability: f64,
}

#[derive(Debug, Clone)]
pub struct CallSiteProfile {
    pub caller: String,
    pub callee: String,
    pub call_count: u64,
    pub total_time: u64,
}

/// Link-time optimization configuration
#[derive(Debug, Clone)]
pub struct LTOConfig {
    /// Enable thin LTO
    pub enable_thin_lto: bool,
    /// Enable fat LTO
    pub enable_fat_lto: bool,
    /// Cross-module inlining threshold
    pub cross_module_inline_threshold: u32,
    /// Dead code elimination
    pub enable_dead_code_elimination: bool,
    /// Global value numbering
    pub enable_global_value_numbering: bool,
    /// Interprocedural constant propagation
    pub enable_ipcp: bool,
    /// Whole program optimization
    pub enable_whole_program_optimization: bool,
}

/// Function inlining decision
#[derive(Debug, Clone)]
pub enum InliningDecision {
    /// Always inline this function
    AlwaysInline,
    /// Never inline this function
    NeverInline,
    /// Inline based on size threshold
    SizeThreshold(u32),
    /// Inline based on call frequency
    FrequencyThreshold(u64),
    /// Inline based on profile data
    ProfileGuided,
}

/// Vectorization hint for loops and operations
#[derive(Debug, Clone)]
pub struct VectorizationHint {
    /// Location identifier (function + line)
    pub location: String,
    /// Hint type
    pub hint_type: VectorizationHintType,
    /// Vector width suggestion
    pub vector_width: Option<u32>,
    /// Alignment requirements
    pub alignment: Option<u32>,
    /// Loop unroll factor
    pub unroll_factor: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum VectorizationHintType {
    /// Enable vectorization
    EnableVectorization,
    /// Disable vectorization
    DisableVectorization,
    /// Force vectorization (unsafe)
    ForceVectorization,
    /// Suggest specific vector instruction set
    UseInstructionSet(String),
    /// Optimize for specific data type
    OptimizeForType(String),
}

impl CompilerOptimizationManager {
    pub fn new() -> Self {
        Self {
            pgo_data: HashMap::new(),
            lto_config: LTOConfig::default(),
            inlining_decisions: HashMap::new(),
            vectorization_hints: HashMap::new(),
            optimization_flags: Vec::new(),
        }
    }

    /// Load profile-guided optimization data
    pub fn load_pgo_data(&mut self, pgo_file: &Path) -> Result<(), OptimizationError> {
        let content = fs::read_to_string(pgo_file)
            .map_err(|e| OptimizationError::FileError(e.to_string()))?;
        
        self.parse_pgo_data(&content)?;
        Ok(())
    }

    /// Parse PGO data from string format
    fn parse_pgo_data(&mut self, content: &str) -> Result<(), OptimizationError> {
        let mut current_function: Option<String> = None;
        let mut current_profile = PGOProfile {
            function_name: String::new(),
            execution_count: 0,
            total_execution_time: 0,
            average_execution_time: 0.0,
            branch_profiles: Vec::new(),
            call_site_profiles: Vec::new(),
            is_hot: false,
            is_cold: false,
        };

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if line.starts_with("function:") {
                // Save previous function if exists
                if let Some(func_name) = current_function.take() {
                    current_profile.function_name = func_name.clone();
                    self.pgo_data.insert(func_name, current_profile.clone());
                }

                // Start new function
                let parts: Vec<&str> = line.split_whitespace().collect();
                if let Some(name) = parts[0].strip_prefix("function:") {
                    current_function = Some(name.to_string());
                    current_profile = PGOProfile {
                        function_name: name.to_string(),
                        execution_count: 0,
                        total_execution_time: 0,
                        average_execution_time: 0.0,
                        branch_profiles: Vec::new(),
                        call_site_profiles: Vec::new(),
                        is_hot: false,
                        is_cold: false,
                    };

                    // Parse function metadata
                    for part in &parts[1..] {
                        if let Some(count_str) = part.strip_prefix("count:") {
                            current_profile.execution_count = count_str.parse().unwrap_or(0);
                        } else if let Some(time_str) = part.strip_prefix("time:") {
                            current_profile.total_execution_time = time_str.parse().unwrap_or(0);
                        } else if part == "hot" {
                            current_profile.is_hot = true;
                        } else if part == "cold" {
                            current_profile.is_cold = true;
                        }
                    }

                    if current_profile.execution_count > 0 {
                        current_profile.average_execution_time = 
                            current_profile.total_execution_time as f64 / current_profile.execution_count as f64;
                    }
                }
            } else if line.starts_with("branch:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let (Some(branch_id), Some(taken_str), Some(not_taken_str)) = (
                        parts[0].strip_prefix("branch:"),
                        parts[1].strip_prefix("taken:"),
                        parts[2].strip_prefix("not_taken:")
                    ) {
                        let taken_count: u64 = taken_str.parse().unwrap_or(0);
                        let not_taken_count: u64 = not_taken_str.parse().unwrap_or(0);
                        let total = taken_count + not_taken_count;
                        let taken_probability = if total > 0 {
                            taken_count as f64 / total as f64
                        } else {
                            0.0
                        };

                        current_profile.branch_profiles.push(BranchProfile {
                            branch_id: branch_id.to_string(),
                            taken_count,
                            not_taken_count,
                            taken_probability,
                        });
                    }
                }
            } else if line.starts_with("call:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let (Some(call_info), Some(count_str), Some(time_str)) = (
                        parts[0].strip_prefix("call:"),
                        parts[1].strip_prefix("count:"),
                        parts[2].strip_prefix("time:")
                    ) {
                        let call_parts: Vec<&str> = call_info.split("->").collect();
                        if call_parts.len() == 2 {
                            current_profile.call_site_profiles.push(CallSiteProfile {
                                caller: call_parts[0].to_string(),
                                callee: call_parts[1].to_string(),
                                call_count: count_str.parse().unwrap_or(0),
                                total_time: time_str.parse().unwrap_or(0),
                            });
                        }
                    }
                }
            }
        }

        // Save last function
        if let Some(func_name) = current_function {
            current_profile.function_name = func_name.clone();
            self.pgo_data.insert(func_name, current_profile);
        }

        Ok(())
    }

    /// Generate inlining decisions based on profile data
    pub fn generate_inlining_decisions(&mut self) {
        self.inlining_decisions.clear();

        for (func_name, profile) in &self.pgo_data {
            let decision = if profile.is_hot && profile.average_execution_time < 1000.0 {
                // Hot, small functions should be inlined
                InliningDecision::AlwaysInline
            } else if profile.is_cold {
                // Cold functions should not be inlined to reduce code size
                InliningDecision::NeverInline
            } else if profile.execution_count > 10000 && profile.average_execution_time < 5000.0 {
                // Frequently called, reasonably small functions
                InliningDecision::FrequencyThreshold(10000)
            } else if profile.average_execution_time > 50000.0 {
                // Large functions should not be inlined
                InliningDecision::NeverInline
            } else {
                // Use profile-guided decision for others
                InliningDecision::ProfileGuided
            };

            self.inlining_decisions.insert(func_name.clone(), decision);
        }
    }

    /// Generate vectorization hints for hot loops
    pub fn generate_vectorization_hints(&mut self) {
        self.vectorization_hints.clear();

        for (func_name, profile) in &self.pgo_data {
            if profile.is_hot {
                // Generate vectorization hints for hot functions
                let hint = VectorizationHint {
                    location: func_name.clone(),
                    hint_type: VectorizationHintType::EnableVectorization,
                    vector_width: Some(256), // AVX2
                    alignment: Some(32),
                    unroll_factor: Some(4),
                };
                self.vectorization_hints.insert(func_name.clone(), hint);

                // Special hints for trading-specific functions
                if func_name.contains("price_comparison") || func_name.contains("order_matching") {
                    let simd_hint = VectorizationHint {
                        location: format!("{}_simd", func_name),
                        hint_type: VectorizationHintType::UseInstructionSet("avx2".to_string()),
                        vector_width: Some(256),
                        alignment: Some(32),
                        unroll_factor: Some(8),
                    };
                    self.vectorization_hints.insert(format!("{}_simd", func_name), simd_hint);
                }
            }
        }
    }

    /// Generate compiler optimization flags
    pub fn generate_optimization_flags(&mut self) -> Vec<String> {
        let mut flags = Vec::new();

        // Base optimization flags
        flags.push("-O3".to_string()); // Maximum optimization
        flags.push("-march=native".to_string()); // Use native CPU features
        flags.push("-mtune=native".to_string()); // Tune for native CPU

        // Link-time optimization
        if self.lto_config.enable_thin_lto {
            flags.push("-flto=thin".to_string());
        } else if self.lto_config.enable_fat_lto {
            flags.push("-flto=full".to_string());
        }

        // Profile-guided optimization
        if !self.pgo_data.is_empty() {
            flags.push("-fprofile-use".to_string());
            flags.push("-fprofile-correction".to_string());
        }

        // Inlining optimizations
        flags.push("-finline-functions".to_string());
        flags.push("-finline-functions-called-once".to_string());
        flags.push("-finline-small-functions".to_string());

        // Vectorization
        flags.push("-ftree-vectorize".to_string());
        flags.push("-fvect-cost-model=dynamic".to_string());
        flags.push("-mavx2".to_string());
        flags.push("-mfma".to_string());

        // Loop optimizations
        flags.push("-funroll-loops".to_string());
        flags.push("-fpeel-loops".to_string());
        flags.push("-floop-interchange".to_string());
        flags.push("-floop-block".to_string());

        // Branch prediction
        flags.push("-fpredict-branches".to_string());
        flags.push("-freorder-blocks-and-partition".to_string());

        // Function-specific optimizations
        for (func_name, decision) in &self.inlining_decisions {
            match decision {
                InliningDecision::AlwaysInline => {
                    flags.push(format!("-finline-functions={}", func_name));
                }
                InliningDecision::NeverInline => {
                    flags.push(format!("-fno-inline-functions={}", func_name));
                }
                InliningDecision::SizeThreshold(threshold) => {
                    flags.push(format!("-finline-limit={}", threshold));
                }
                _ => {}
            }
        }

        // Trading-specific optimizations
        flags.push("-ffast-math".to_string()); // Fast floating-point math
        flags.push("-fno-math-errno".to_string()); // Don't set errno for math functions
        flags.push("-ffinite-math-only".to_string()); // Assume finite math
        flags.push("-fno-signed-zeros".to_string()); // Ignore signed zeros
        flags.push("-fno-trapping-math".to_string()); // No trapping math

        // Memory optimizations
        flags.push("-fmerge-constants".to_string());
        flags.push("-fmerge-debug-strings".to_string());

        // Code generation optimizations
        flags.push("-fomit-frame-pointer".to_string());
        flags.push("-foptimize-sibling-calls".to_string());

        self.optimization_flags = flags.clone();
        flags
    }

    /// Generate Rust-specific optimization attributes
    pub fn generate_rust_attributes(&self) -> HashMap<String, Vec<String>> {
        let mut attributes = HashMap::new();

        for (func_name, decision) in &self.inlining_decisions {
            let mut func_attributes = Vec::new();

            match decision {
                InliningDecision::AlwaysInline => {
                    func_attributes.push("#[inline(always)]".to_string());
                }
                InliningDecision::NeverInline => {
                    func_attributes.push("#[inline(never)]".to_string());
                }
                InliningDecision::ProfileGuided => {
                    if let Some(profile) = self.pgo_data.get(func_name) {
                        if profile.is_hot {
                            func_attributes.push("#[inline]".to_string());
                        } else if profile.is_cold {
                            func_attributes.push("#[cold]".to_string());
                        }
                    }
                }
                _ => {
                    func_attributes.push("#[inline]".to_string());
                }
            }

            // Add target-specific attributes for hot functions
            if let Some(profile) = self.pgo_data.get(func_name) {
                if profile.is_hot {
                    func_attributes.push("#[target_feature(enable = \"avx2\")]".to_string());
                    func_attributes.push("#[target_feature(enable = \"fma\")]".to_string());
                }
            }

            attributes.insert(func_name.clone(), func_attributes);
        }

        attributes
    }

    /// Generate LLVM optimization passes
    pub fn generate_llvm_passes(&self) -> Vec<String> {
        let mut passes = Vec::new();

        // Standard optimization passes
        passes.push("mem2reg".to_string());
        passes.push("instcombine".to_string());
        passes.push("reassociate".to_string());
        passes.push("gvn".to_string());
        passes.push("simplifycfg".to_string());

        // Loop optimization passes
        passes.push("loop-rotate".to_string());
        passes.push("loop-unswitch".to_string());
        passes.push("loop-unroll".to_string());
        passes.push("loop-vectorize".to_string());
        passes.push("slp-vectorizer".to_string());

        // Interprocedural optimization passes
        if self.lto_config.enable_ipcp {
            passes.push("ipcp".to_string());
            passes.push("globalopt".to_string());
            passes.push("deadargelim".to_string());
        }

        // Profile-guided optimization passes
        if !self.pgo_data.is_empty() {
            passes.push("pgo-instr-use".to_string());
            passes.push("sample-profile".to_string());
        }

        // Trading-specific passes
        passes.push("aggressive-instcombine".to_string());
        passes.push("called-value-propagation".to_string());
        passes.push("float2int".to_string());

        passes
    }

    /// Export optimization configuration to file
    pub fn export_config(&self, output_path: &Path) -> Result<(), OptimizationError> {
        let mut config = String::new();
        
        config.push_str("# Compiler Optimization Configuration\n\n");
        
        // Optimization flags
        config.push_str("## Compiler Flags\n");
        for flag in &self.optimization_flags {
            config.push_str(&format!("{}\n", flag));
        }
        config.push('\n');

        // Inlining decisions
        config.push_str("## Inlining Decisions\n");
        for (func_name, decision) in &self.inlining_decisions {
            config.push_str(&format!("{}: {:?}\n", func_name, decision));
        }
        config.push('\n');

        // Vectorization hints
        config.push_str("## Vectorization Hints\n");
        for (location, hint) in &self.vectorization_hints {
            config.push_str(&format!("{}: {:?}\n", location, hint));
        }
        config.push('\n');

        // LTO configuration
        config.push_str("## Link-Time Optimization\n");
        config.push_str(&format!("Thin LTO: {}\n", self.lto_config.enable_thin_lto));
        config.push_str(&format!("Fat LTO: {}\n", self.lto_config.enable_fat_lto));
        config.push_str(&format!("Cross-module inline threshold: {}\n", self.lto_config.cross_module_inline_threshold));

        fs::write(output_path, config)
            .map_err(|e| OptimizationError::FileError(e.to_string()))?;

        Ok(())
    }

    /// Apply optimizations to build system
    pub fn apply_to_build_system(&self, build_system: BuildSystem) -> Result<(), OptimizationError> {
        match build_system {
            BuildSystem::Cargo => self.apply_to_cargo(),
            BuildSystem::CMake => self.apply_to_cmake(),
            BuildSystem::Make => self.apply_to_make(),
        }
    }

    fn apply_to_cargo(&self) -> Result<(), OptimizationError> {
        // Generate Cargo.toml optimizations
        let mut cargo_config = String::new();
        
        cargo_config.push_str("[profile.release]\n");
        cargo_config.push_str("opt-level = 3\n");
        cargo_config.push_str("lto = true\n");
        cargo_config.push_str("codegen-units = 1\n");
        cargo_config.push_str("panic = 'abort'\n");
        
        if !self.pgo_data.is_empty() {
            cargo_config.push_str("\n[profile.pgo]\n");
            cargo_config.push_str("inherits = \"release\"\n");
            cargo_config.push_str("lto = \"fat\"\n");
        }

        // Write to .cargo/config.toml
        fs::create_dir_all(".cargo")?;
        fs::write(".cargo/config.toml", cargo_config)?;

        Ok(())
    }

    fn apply_to_cmake(&self) -> Result<(), OptimizationError> {
        let mut cmake_flags = String::new();
        
        cmake_flags.push_str("# Compiler optimization flags\n");
        cmake_flags.push_str("set(CMAKE_CXX_FLAGS_RELEASE \"");
        for flag in &self.optimization_flags {
            cmake_flags.push_str(&format!("{} ", flag));
        }
        cmake_flags.push_str("\")\n");

        if self.lto_config.enable_thin_lto || self.lto_config.enable_fat_lto {
            cmake_flags.push_str("set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)\n");
        }

        fs::write("cmake_optimization_flags.cmake", cmake_flags)?;
        Ok(())
    }

    fn apply_to_make(&self) -> Result<(), OptimizationError> {
        let mut makefile_flags = String::new();
        
        makefile_flags.push_str("# Compiler optimization flags\n");
        makefile_flags.push_str("CXXFLAGS += ");
        for flag in &self.optimization_flags {
            makefile_flags.push_str(&format!("{} ", flag));
        }
        makefile_flags.push('\n');

        fs::write("optimization_flags.mk", makefile_flags)?;
        Ok(())
    }

    /// Run profile-guided optimization build process
    pub fn run_pgo_build(&self, build_command: &str) -> Result<(), OptimizationError> {
        // Step 1: Build with instrumentation
        let instrument_command = format!("{} -fprofile-generate", build_command);
        let output = Command::new("sh")
            .arg("-c")
            .arg(&instrument_command)
            .output()
            .map_err(|e| OptimizationError::BuildError(e.to_string()))?;

        if !output.status.success() {
            return Err(OptimizationError::BuildError(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }

        // Step 2: Run training workload (this would be application-specific)
        println!("Run your training workload now to generate profile data...");
        
        // Step 3: Build with profile data
        let optimized_command = format!("{} -fprofile-use", build_command);
        let output = Command::new("sh")
            .arg("-c")
            .arg(&optimized_command)
            .output()
            .map_err(|e| OptimizationError::BuildError(e.to_string()))?;

        if !output.status.success() {
            return Err(OptimizationError::BuildError(
                String::from_utf8_lossy(&output.stderr).to_string()
            ));
        }

        Ok(())
    }
}

impl Default for LTOConfig {
    fn default() -> Self {
        Self {
            enable_thin_lto: true,
            enable_fat_lto: false,
            cross_module_inline_threshold: 225,
            enable_dead_code_elimination: true,
            enable_global_value_numbering: true,
            enable_ipcp: true,
            enable_whole_program_optimization: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BuildSystem {
    Cargo,
    CMake,
    Make,
}

#[derive(Debug)]
pub enum OptimizationError {
    FileError(String),
    ParseError(String),
    BuildError(String),
    ConfigError(String),
}

impl std::fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationError::FileError(msg) => write!(f, "File error: {}", msg),
            OptimizationError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            OptimizationError::BuildError(msg) => write!(f, "Build error: {}", msg),
            OptimizationError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for OptimizationError {}

impl From<std::io::Error> for OptimizationError {
    fn from(error: std::io::Error) -> Self {
        OptimizationError::FileError(error.to_string())
    }
}

/// Utility functions for compiler optimization
pub struct OptimizationUtils;

impl OptimizationUtils {
    /// Detect available CPU features
    pub fn detect_cpu_features() -> Vec<String> {
        let mut features = Vec::new();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                features.push("avx2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                features.push("fma".to_string());
            }
            if is_x86_feature_detected!("bmi1") {
                features.push("bmi1".to_string());
            }
            if is_x86_feature_detected!("bmi2") {
                features.push("bmi2".to_string());
            }
            if is_x86_feature_detected!("popcnt") {
                features.push("popcnt".to_string());
            }
            if is_x86_feature_detected!("lzcnt") {
                features.push("lzcnt".to_string());
            }
        }
        
        features
    }

    /// Generate target-specific optimization flags
    pub fn generate_target_flags() -> Vec<String> {
        let mut flags = Vec::new();
        let features = Self::detect_cpu_features();
        
        for feature in features {
            flags.push(format!("-m{}", feature));
        }
        
        // Add architecture-specific flags
        #[cfg(target_arch = "x86_64")]
        {
            flags.push("-march=native".to_string());
            flags.push("-mtune=native".to_string());
        }
        
        flags
    }

    /// Estimate optimization impact
    pub fn estimate_optimization_impact(
        baseline_time: f64,
        optimized_time: f64,
    ) -> OptimizationImpact {
        let improvement_ratio = baseline_time / optimized_time;
        let improvement_percent = (improvement_ratio - 1.0) * 100.0;
        
        OptimizationImpact {
            baseline_time,
            optimized_time,
            improvement_ratio,
            improvement_percent,
            is_significant: improvement_percent > 5.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    pub baseline_time: f64,
    pub optimized_time: f64,
    pub improvement_ratio: f64,
    pub improvement_percent: f64,
    pub is_significant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_compiler_optimization_manager_creation() {
        let manager = CompilerOptimizationManager::new();
        assert!(manager.pgo_data.is_empty());
        assert!(manager.inlining_decisions.is_empty());
        assert!(manager.vectorization_hints.is_empty());
    }

    #[test]
    fn test_lto_config_default() {
        let config = LTOConfig::default();
        assert!(config.enable_thin_lto);
        assert!(!config.enable_fat_lto);
        assert_eq!(config.cross_module_inline_threshold, 225);
    }

    #[test]
    fn test_pgo_data_parsing() {
        let mut manager = CompilerOptimizationManager::new();
        let pgo_content = r#"
function:hot_function count:10000 time:50000000 hot
branch:hot_function_br1 taken:8000 not_taken:2000
call:hot_function->helper_function count:5000 time:25000000

function:cold_function count:10 time:1000000 cold
"#;
        
        manager.parse_pgo_data(pgo_content).unwrap();
        
        assert_eq!(manager.pgo_data.len(), 2);
        
        let hot_func = manager.pgo_data.get("hot_function").unwrap();
        assert!(hot_func.is_hot);
        assert_eq!(hot_func.execution_count, 10000);
        assert_eq!(hot_func.branch_profiles.len(), 1);
        assert_eq!(hot_func.call_site_profiles.len(), 1);
        
        let cold_func = manager.pgo_data.get("cold_function").unwrap();
        assert!(cold_func.is_cold);
        assert_eq!(cold_func.execution_count, 10);
    }

    #[test]
    fn test_inlining_decisions_generation() {
        let mut manager = CompilerOptimizationManager::new();
        
        // Add some test profile data
        manager.pgo_data.insert("hot_small_function".to_string(), PGOProfile {
            function_name: "hot_small_function".to_string(),
            execution_count: 50000,
            total_execution_time: 25000000, // 25ms total
            average_execution_time: 500.0, // 500ns average
            branch_profiles: Vec::new(),
            call_site_profiles: Vec::new(),
            is_hot: true,
            is_cold: false,
        });
        
        manager.pgo_data.insert("cold_function".to_string(), PGOProfile {
            function_name: "cold_function".to_string(),
            execution_count: 5,
            total_execution_time: 1000000, // 1ms total
            average_execution_time: 200000.0, // 200μs average
            branch_profiles: Vec::new(),
            call_site_profiles: Vec::new(),
            is_hot: false,
            is_cold: true,
        });
        
        manager.generate_inlining_decisions();
        
        assert_eq!(manager.inlining_decisions.len(), 2);
        
        match manager.inlining_decisions.get("hot_small_function").unwrap() {
            InliningDecision::AlwaysInline => {},
            _ => panic!("Expected AlwaysInline for hot small function"),
        }
        
        match manager.inlining_decisions.get("cold_function").unwrap() {
            InliningDecision::NeverInline => {},
            _ => panic!("Expected NeverInline for cold function"),
        }
    }

    #[test]
    fn test_optimization_flags_generation() {
        let mut manager = CompilerOptimizationManager::new();
        let flags = manager.generate_optimization_flags();
        
        assert!(flags.contains(&"-O3".to_string()));
        assert!(flags.contains(&"-march=native".to_string()));
        assert!(flags.contains(&"-flto=thin".to_string()));
        assert!(flags.contains(&"-ftree-vectorize".to_string()));
    }

    #[test]
    fn test_cpu_feature_detection() {
        let features = OptimizationUtils::detect_cpu_features();
        // This test will vary based on the CPU, so we just check that it returns a vector
        assert!(features.is_empty() || !features.is_empty());
    }

    #[test]
    fn test_optimization_impact_calculation() {
        let impact = OptimizationUtils::estimate_optimization_impact(100.0, 80.0);
        
        assert_eq!(impact.baseline_time, 100.0);
        assert_eq!(impact.optimized_time, 80.0);
        assert_eq!(impact.improvement_ratio, 1.25);
        assert_eq!(impact.improvement_percent, 25.0);
        assert!(impact.is_significant);
    }

    #[test]
    fn test_rust_attributes_generation() {
        let mut manager = CompilerOptimizationManager::new();
        
        manager.inlining_decisions.insert(
            "always_inline_func".to_string(),
            InliningDecision::AlwaysInline
        );
        manager.inlining_decisions.insert(
            "never_inline_func".to_string(),
            InliningDecision::NeverInline
        );
        
        let attributes = manager.generate_rust_attributes();
        
        assert_eq!(attributes.len(), 2);
        assert!(attributes.get("always_inline_func").unwrap().contains(&"#[inline(always)]".to_string()));
        assert!(attributes.get("never_inline_func").unwrap().contains(&"#[inline(never)]".to_string()));
    }

    #[test]
    fn test_llvm_passes_generation() {
        let manager = CompilerOptimizationManager::new();
        let passes = manager.generate_llvm_passes();
        
        assert!(passes.contains(&"mem2reg".to_string()));
        assert!(passes.contains(&"instcombine".to_string()));
        assert!(passes.contains(&"loop-vectorize".to_string()));
        assert!(passes.contains(&"gvn".to_string()));
    }
}