use super::sampling_profiler::ProfileSample;
use super::timing::now_nanos;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// Flame graph representation for visualizing performance profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraph {
    /// Root node of the flame graph
    pub root: FlameGraphNode,
    
    /// Total samples in the graph
    pub total_samples: u64,
    
    /// Generation timestamp
    pub generated_at: u64,
    
    /// Generation duration (nanoseconds)
    pub generation_duration_ns: u64,
    
    /// Metadata about the flame graph
    pub metadata: HashMap<String, String>,
}

/// Individual node in the flame graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphNode {
    /// Function name
    pub name: String,
    
    /// Number of samples for this node
    pub value: u64,
    
    /// Percentage of total samples
    pub percentage: f64,
    
    /// Child nodes
    pub children: Vec<FlameGraphNode>,
    
    /// Node depth in the stack
    pub depth: u32,
    
    /// Cumulative time spent in this function (nanoseconds)
    pub cumulative_time_ns: u64,
    
    /// Self time spent in this function (nanoseconds)
    pub self_time_ns: u64,
    
    /// Additional node metadata
    pub metadata: HashMap<String, String>,
}

/// Flame graph builder for constructing flame graphs from samples
pub struct FlameGraphBuilder {
    /// Configuration for flame graph generation
    config: FlameGraphConfig,
    
    /// Node cache for performance
    node_cache: std::sync::RwLock<HashMap<String, Arc<FlameGraphNode>>>,
}

/// Configuration for flame graph generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameGraphConfig {
    /// Minimum sample count to include a node
    pub min_samples: u64,
    
    /// Maximum depth to include in the graph
    pub max_depth: u32,
    
    /// Whether to merge similar function names
    pub merge_similar_functions: bool,
    
    /// Function name patterns to exclude
    pub exclude_patterns: Vec<String>,
    
    /// Whether to calculate self time
    pub calculate_self_time: bool,
    
    /// Whether to include metadata
    pub include_metadata: bool,
}

impl FlameGraphBuilder {
    /// Create a new flame graph builder
    pub fn new() -> Self {
        Self {
            config: FlameGraphConfig::default(),
            node_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Create builder with custom configuration
    pub fn with_config(config: FlameGraphConfig) -> Self {
        Self {
            config,
            node_cache: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Build flame graph from profile samples
    pub fn build_from_samples(&self, samples: &[ProfileSample]) -> Result<FlameGraph, FlameGraphError> {
        if samples.is_empty() {
            return Err(FlameGraphError::NoSamples);
        }

        let generation_start = now_nanos();
        
        // Build stack frequency map
        let stack_counts = self.build_stack_frequency_map(samples);
        
        // Build flame graph tree
        let root = self.build_flame_graph_tree(&stack_counts, samples.len() as u64)?;
        
        let generation_duration = now_nanos() - generation_start;
        
        let mut metadata = HashMap::new();
        metadata.insert("sample_count".to_string(), samples.len().to_string());
        metadata.insert("unique_stacks".to_string(), stack_counts.len().to_string());
        metadata.insert("generation_time_ns".to_string(), generation_duration.to_string());
        
        Ok(FlameGraph {
            root,
            total_samples: samples.len() as u64,
            generated_at: generation_start,
            generation_duration_ns: generation_duration,
            metadata,
        })
    }

    /// Build flame graph from stack traces
    pub fn build_from_stacks(&self, stacks: &[Vec<String>]) -> Result<FlameGraph, FlameGraphError> {
        if stacks.is_empty() {
            return Err(FlameGraphError::NoSamples);
        }

        let generation_start = now_nanos();
        
        // Convert stacks to frequency map
        let mut stack_counts = HashMap::new();
        for stack in stacks {
            let stack_key = stack.join(";");
            *stack_counts.entry(stack_key).or_insert(0) += 1;
        }
        
        // Build flame graph tree
        let root = self.build_flame_graph_tree(&stack_counts, stacks.len() as u64)?;
        
        let generation_duration = now_nanos() - generation_start;
        
        let mut metadata = HashMap::new();
        metadata.insert("stack_count".to_string(), stacks.len().to_string());
        metadata.insert("unique_stacks".to_string(), stack_counts.len().to_string());
        metadata.insert("generation_time_ns".to_string(), generation_duration.to_string());
        
        Ok(FlameGraph {
            root,
            total_samples: stacks.len() as u64,
            generated_at: generation_start,
            generation_duration_ns: generation_duration,
            metadata,
        })
    }

    /// Export flame graph to SVG format
    pub fn export_to_svg(&self, flame_graph: &FlameGraph) -> Result<String, FlameGraphError> {
        let mut svg = String::new();
        
        // SVG header
        svg.push_str(&format!(
            r#"<svg width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
<style>
.func_g:hover {{ stroke:black; stroke-width:0.5; cursor:pointer; }}
text {{ font-family:Verdana; font-size:12px; fill:black; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="10" y="20" font-size="16">Flame Graph - {} samples</text>
"#,
            flame_graph.total_samples
        ));
        
        // Render flame graph nodes
        self.render_node_svg(&mut svg, &flame_graph.root, 0, 0, 1200, 0)?;
        
        // SVG footer
        svg.push_str("</svg>");
        
        Ok(svg)
    }

    /// Export flame graph to JSON format
    pub fn export_to_json(&self, flame_graph: &FlameGraph) -> Result<String, FlameGraphError> {
        serde_json::to_string_pretty(flame_graph)
            .map_err(|_| FlameGraphError::SerializationError)
    }

    /// Build stack frequency map from samples
    fn build_stack_frequency_map(&self, samples: &[ProfileSample]) -> HashMap<String, u64> {
        let mut stack_counts = HashMap::new();
        
        for sample in samples {
            // Filter stack trace based on configuration
            let filtered_stack = self.filter_stack_trace(&sample.stack_trace);
            
            if !filtered_stack.is_empty() {
                let stack_key = filtered_stack.join(";");
                *stack_counts.entry(stack_key).or_insert(0) += 1;
            }
        }
        
        stack_counts
    }

    /// Build flame graph tree from stack frequency map
    fn build_flame_graph_tree(&self, stack_counts: &HashMap<String, u64>, total_samples: u64) -> Result<FlameGraphNode, FlameGraphError> {
        let mut root = FlameGraphNode {
            name: "root".to_string(),
            value: total_samples,
            percentage: 100.0,
            children: Vec::new(),
            depth: 0,
            cumulative_time_ns: 0,
            self_time_ns: 0,
            metadata: HashMap::new(),
        };
        
        // Build tree structure
        for (stack_str, &count) in stack_counts {
            if count < self.config.min_samples {
                continue;
            }
            
            let stack: Vec<&str> = stack_str.split(';').collect();
            self.insert_stack_into_tree(&mut root, &stack, count, total_samples, 1);
        }
        
        // Calculate self time if enabled
        if self.config.calculate_self_time {
            self.calculate_self_time(&mut root);
        }
        
        // Sort children by value (descending)
        self.sort_children(&mut root);
        
        Ok(root)
    }

    /// Insert a stack trace into the flame graph tree
    fn insert_stack_into_tree(&self, node: &mut FlameGraphNode, stack: &[&str], count: u64, total_samples: u64, depth: u32) {
        if stack.is_empty() || depth > self.config.max_depth {
            return;
        }
        
        let function_name = stack[0].to_string();
        
        // Find or create child node
        let child_index = node.children.iter().position(|child| child.name == function_name);
        
        let child_index = match child_index {
            Some(index) => {
                // Update existing child
                node.children[index].value += count;
                node.children[index].percentage = (node.children[index].value as f64 / total_samples as f64) * 100.0;
                index
            }
            None => {
                // Create new child
                let new_child = FlameGraphNode {
                    name: function_name,
                    value: count,
                    percentage: (count as f64 / total_samples as f64) * 100.0,
                    children: Vec::new(),
                    depth,
                    cumulative_time_ns: 0,
                    self_time_ns: 0,
                    metadata: HashMap::new(),
                };
                
                node.children.push(new_child);
                node.children.len() - 1
            }
        };
        
        // Recursively insert remaining stack
        if stack.len() > 1 {
            self.insert_stack_into_tree(&mut node.children[child_index], &stack[1..], count, total_samples, depth + 1);
        }
    }

    /// Filter stack trace based on configuration
    fn filter_stack_trace(&self, stack_trace: &[String]) -> Vec<String> {
        let mut filtered = Vec::new();
        
        for function in stack_trace {
            // Check exclude patterns
            let should_exclude = self.config.exclude_patterns.iter()
                .any(|pattern| function.contains(pattern));
            
            if !should_exclude {
                // Apply function name merging if enabled
                let function_name = if self.config.merge_similar_functions {
                    self.merge_similar_function_name(function)
                } else {
                    function.clone()
                };
                
                filtered.push(function_name);
            }
        }
        
        filtered
    }

    /// Merge similar function names
    fn merge_similar_function_name(&self, function_name: &str) -> String {
        // Remove template parameters
        if let Some(pos) = function_name.find('<') {
            let base_name = &function_name[..pos];
            format!("{}<...>", base_name)
        } else {
            function_name.to_string()
        }
    }

    /// Calculate self time for all nodes
    fn calculate_self_time(&self, node: &mut FlameGraphNode) {
        let children_cumulative_time: u64 = node.children.iter().map(|child| child.cumulative_time_ns).sum();
        node.self_time_ns = node.cumulative_time_ns.saturating_sub(children_cumulative_time);
        
        // Recursively calculate for children
        for child in &mut node.children {
            self.calculate_self_time(child);
        }
    }

    /// Sort children by value (descending)
    fn sort_children(&self, node: &mut FlameGraphNode) {
        node.children.sort_by(|a, b| b.value.cmp(&a.value));
        
        // Recursively sort children
        for child in &mut node.children {
            self.sort_children(child);
        }
    }

    /// Render a node as SVG
    fn render_node_svg(&self, svg: &mut String, node: &FlameGraphNode, x: i32, y: i32, width: i32, depth: u32) -> Result<(), FlameGraphError> {
        if width < 1 {
            return Ok(());
        }
        
        let height = 20;
        let color = self.get_node_color(node);
        
        // Draw rectangle
        svg.push_str(&format!(
            r#"<g class="func_g">
<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white" stroke-width="1"/>
<text x="{}" y="{}" font-size="12">{}</text>
</g>
"#,
            x, y, width, height, color,
            x + 2, y + 15, self.truncate_text(&node.name, width)
        ));
        
        // Render children
        let mut child_x = x;
        for child in &node.children {
            let child_width = ((child.value as f64 / node.value as f64) * width as f64) as i32;
            if child_width > 0 {
                self.render_node_svg(svg, child, child_x, y + height, child_width, depth + 1)?;
                child_x += child_width;
            }
        }
        
        Ok(())
    }

    /// Get color for a node based on its characteristics
    fn get_node_color(&self, node: &FlameGraphNode) -> String {
        // Generate color based on function name hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        node.name.hash(&mut hasher);
        let hash = hasher.finish();
        
        let r = ((hash & 0xFF) as u8).max(100);
        let g = (((hash >> 8) & 0xFF) as u8).max(100);
        let b = (((hash >> 16) & 0xFF) as u8).max(100);
        
        format!("rgb({},{},{})", r, g, b)
    }

    /// Truncate text to fit in given width
    fn truncate_text(&self, text: &str, width: i32) -> String {
        let max_chars = (width / 8).max(1) as usize; // Approximate character width
        
        if text.len() <= max_chars {
            text.to_string()
        } else if max_chars > 3 {
            format!("{}...", &text[..max_chars - 3])
        } else {
            "...".to_string()
        }
    }
}

impl Default for FlameGraphConfig {
    fn default() -> Self {
        Self {
            min_samples: 1,
            max_depth: 100,
            merge_similar_functions: true,
            exclude_patterns: vec![
                "std::".to_string(),
                "__rust_".to_string(),
                "core::".to_string(),
            ],
            calculate_self_time: true,
            include_metadata: true,
        }
    }
}

impl FlameGraphNode {
    /// Get all descendant nodes
    pub fn get_all_descendants(&self) -> Vec<&FlameGraphNode> {
        let mut descendants = Vec::new();
        self.collect_descendants(&mut descendants);
        descendants
    }

    /// Collect all descendant nodes recursively
    fn collect_descendants(&self, descendants: &mut Vec<&FlameGraphNode>) {
        for child in &self.children {
            descendants.push(child);
            child.collect_descendants(descendants);
        }
    }

    /// Find node by name
    pub fn find_node(&self, name: &str) -> Option<&FlameGraphNode> {
        if self.name == name {
            return Some(self);
        }
        
        for child in &self.children {
            if let Some(found) = child.find_node(name) {
                return Some(found);
            }
        }
        
        None
    }

    /// Get total number of nodes in subtree
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|child| child.node_count()).sum::<usize>()
    }

    /// Get maximum depth of subtree
    pub fn max_depth(&self) -> u32 {
        if self.children.is_empty() {
            self.depth
        } else {
            self.children.iter().map(|child| child.max_depth()).max().unwrap_or(self.depth)
        }
    }
}

/// Flame graph errors
#[derive(Debug, Clone)]
pub enum FlameGraphError {
    NoSamples,
    InvalidConfiguration,
    SerializationError,
    RenderingError,
    ExportError,
}

impl std::fmt::Display for FlameGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlameGraphError::NoSamples => write!(f, "No samples provided for flame graph generation"),
            FlameGraphError::InvalidConfiguration => write!(f, "Invalid flame graph configuration"),
            FlameGraphError::SerializationError => write!(f, "Failed to serialize flame graph"),
            FlameGraphError::RenderingError => write!(f, "Failed to render flame graph"),
            FlameGraphError::ExportError => write!(f, "Failed to export flame graph"),
        }
    }
}

impl std::error::Error for FlameGraphError {}

/// Flame graph analysis utilities
pub struct FlameGraphAnalyzer;

impl FlameGraphAnalyzer {
    /// Find hottest functions in the flame graph
    pub fn find_hottest_functions(flame_graph: &FlameGraph, limit: usize) -> Vec<(&FlameGraphNode, f64)> {
        let mut all_nodes = Vec::new();
        flame_graph.root.collect_descendants(&mut all_nodes);
        
        // Sort by percentage (descending)
        all_nodes.sort_by(|a, b| b.percentage.partial_cmp(&a.percentage).unwrap_or(std::cmp::Ordering::Equal));
        
        all_nodes.into_iter()
            .take(limit)
            .map(|node| (node, node.percentage))
            .collect()
    }

    /// Find deepest call stacks
    pub fn find_deepest_stacks(flame_graph: &FlameGraph, limit: usize) -> Vec<(&FlameGraphNode, u32)> {
        let mut all_nodes = Vec::new();
        flame_graph.root.collect_descendants(&mut all_nodes);
        
        // Sort by depth (descending)
        all_nodes.sort_by(|a, b| b.depth.cmp(&a.depth));
        
        all_nodes.into_iter()
            .take(limit)
            .map(|node| (node, node.depth))
            .collect()
    }

    /// Calculate flame graph statistics
    pub fn calculate_statistics(flame_graph: &FlameGraph) -> FlameGraphStatistics {
        let all_nodes = flame_graph.root.get_all_descendants();
        
        let total_nodes = all_nodes.len() + 1; // +1 for root
        let max_depth = flame_graph.root.max_depth();
        let avg_depth = if !all_nodes.is_empty() {
            all_nodes.iter().map(|node| node.depth as f64).sum::<f64>() / all_nodes.len() as f64
        } else {
            0.0
        };
        
        let unique_functions = all_nodes.iter()
            .map(|node| &node.name)
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        FlameGraphStatistics {
            total_nodes,
            unique_functions,
            max_depth,
            avg_depth,
            total_samples: flame_graph.total_samples,
            generation_time_ns: flame_graph.generation_duration_ns,
        }
    }
}

/// Flame graph statistics
#[derive(Debug, Clone)]
pub struct FlameGraphStatistics {
    pub total_nodes: usize,
    pub unique_functions: usize,
    pub max_depth: u32,
    pub avg_depth: f64,
    pub total_samples: u64,
    pub generation_time_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_samples() -> Vec<ProfileSample> {
        vec![
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 1000000,
                function_name: "main".to_string(),
                stack_trace: vec!["main".to_string(), "process_order".to_string(), "match_order".to_string()],
                cpu_usage: 50.0,
                memory_usage: 1024,
                thread_id: 1,
            },
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 2000000,
                function_name: "main".to_string(),
                stack_trace: vec!["main".to_string(), "process_order".to_string(), "validate_order".to_string()],
                cpu_usage: 60.0,
                memory_usage: 2048,
                thread_id: 1,
            },
            ProfileSample {
                timestamp: now_nanos(),
                duration_ns: 1500000,
                function_name: "main".to_string(),
                stack_trace: vec!["main".to_string(), "send_notification".to_string()],
                cpu_usage: 30.0,
                memory_usage: 512,
                thread_id: 2,
            },
        ]
    }

    #[test]
    fn test_flame_graph_builder_creation() {
        let builder = FlameGraphBuilder::new();
        assert_eq!(builder.config.min_samples, 1);
        assert_eq!(builder.config.max_depth, 100);
        assert!(builder.config.merge_similar_functions);
    }

    #[test]
    fn test_flame_graph_generation() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        
        let flame_graph = builder.build_from_samples(&samples).unwrap();
        
        assert_eq!(flame_graph.total_samples, 3);
        assert_eq!(flame_graph.root.name, "root");
        assert_eq!(flame_graph.root.value, 3);
        assert_eq!(flame_graph.root.percentage, 100.0);
        assert!(!flame_graph.root.children.is_empty());
    }

    #[test]
    fn test_stack_frequency_map() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        
        let stack_counts = builder.build_stack_frequency_map(&samples);
        
        assert!(stack_counts.contains_key("main;process_order;match_order"));
        assert!(stack_counts.contains_key("main;process_order;validate_order"));
        assert!(stack_counts.contains_key("main;send_notification"));
        
        assert_eq!(stack_counts["main;process_order;match_order"], 1);
        assert_eq!(stack_counts["main;process_order;validate_order"], 1);
        assert_eq!(stack_counts["main;send_notification"], 1);
    }

    #[test]
    fn test_flame_graph_from_stacks() {
        let builder = FlameGraphBuilder::new();
        let stacks = vec![
            vec!["main".to_string(), "func_a".to_string(), "func_b".to_string()],
            vec!["main".to_string(), "func_a".to_string(), "func_c".to_string()],
            vec!["main".to_string(), "func_d".to_string()],
        ];
        
        let flame_graph = builder.build_from_stacks(&stacks).unwrap();
        
        assert_eq!(flame_graph.total_samples, 3);
        assert_eq!(flame_graph.root.children.len(), 1); // Only "main"
        
        let main_node = &flame_graph.root.children[0];
        assert_eq!(main_node.name, "main");
        assert_eq!(main_node.value, 3);
        assert_eq!(main_node.children.len(), 2); // "func_a" and "func_d"
    }

    #[test]
    fn test_node_operations() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        let flame_graph = builder.build_from_samples(&samples).unwrap();
        
        // Test node count
        let total_nodes = flame_graph.root.node_count();
        assert!(total_nodes > 1);
        
        // Test max depth
        let max_depth = flame_graph.root.max_depth();
        assert!(max_depth > 0);
        
        // Test find node
        let main_node = flame_graph.root.find_node("main");
        assert!(main_node.is_some());
        assert_eq!(main_node.unwrap().name, "main");
        
        // Test get descendants
        let descendants = flame_graph.root.get_all_descendants();
        assert!(!descendants.is_empty());
    }

    #[test]
    fn test_flame_graph_config() {
        let config = FlameGraphConfig {
            min_samples: 5,
            max_depth: 10,
            merge_similar_functions: false,
            exclude_patterns: vec!["test::".to_string()],
            calculate_self_time: false,
            include_metadata: false,
        };
        
        let builder = FlameGraphBuilder::with_config(config.clone());
        assert_eq!(builder.config.min_samples, 5);
        assert_eq!(builder.config.max_depth, 10);
        assert!(!builder.config.merge_similar_functions);
        assert!(builder.config.exclude_patterns.contains(&"test::".to_string()));
    }

    #[test]
    fn test_function_name_merging() {
        let builder = FlameGraphBuilder::new();
        
        let merged = builder.merge_similar_function_name("std::vector<int>::push_back");
        assert_eq!(merged, "std::vector<...>");
        
        let no_template = builder.merge_similar_function_name("simple_function");
        assert_eq!(no_template, "simple_function");
    }

    #[test]
    fn test_stack_filtering() {
        let config = FlameGraphConfig {
            exclude_patterns: vec!["std::".to_string(), "core::".to_string()],
            ..Default::default()
        };
        
        let builder = FlameGraphBuilder::with_config(config);
        
        let stack = vec![
            "main".to_string(),
            "std::vector::push_back".to_string(),
            "my_function".to_string(),
            "core::mem::drop".to_string(),
        ];
        
        let filtered = builder.filter_stack_trace(&stack);
        
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains(&"main".to_string()));
        assert!(filtered.contains(&"my_function".to_string()));
        assert!(!filtered.iter().any(|f| f.contains("std::")));
        assert!(!filtered.iter().any(|f| f.contains("core::")));
    }

    #[test]
    fn test_flame_graph_analyzer() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        let flame_graph = builder.build_from_samples(&samples).unwrap();
        
        // Test hottest functions
        let hottest = FlameGraphAnalyzer::find_hottest_functions(&flame_graph, 3);
        assert!(!hottest.is_empty());
        
        // Test deepest stacks
        let deepest = FlameGraphAnalyzer::find_deepest_stacks(&flame_graph, 3);
        assert!(!deepest.is_empty());
        
        // Test statistics
        let stats = FlameGraphAnalyzer::calculate_statistics(&flame_graph);
        assert!(stats.total_nodes > 0);
        assert!(stats.unique_functions > 0);
        assert_eq!(stats.total_samples, 3);
    }

    #[test]
    fn test_svg_export() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        let flame_graph = builder.build_from_samples(&samples).unwrap();
        
        let svg = builder.export_to_svg(&flame_graph).unwrap();
        
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Flame Graph"));
        assert!(svg.contains("3 samples"));
    }

    #[test]
    fn test_json_export() {
        let builder = FlameGraphBuilder::new();
        let samples = create_test_samples();
        let flame_graph = builder.build_from_samples(&samples).unwrap();
        
        let json = builder.export_to_json(&flame_graph).unwrap();
        
        assert!(json.contains("\"root\""));
        assert!(json.contains("\"total_samples\""));
        assert!(json.contains("\"generated_at\""));
        
        // Test that we can deserialize it back
        let deserialized: FlameGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_samples, flame_graph.total_samples);
        assert_eq!(deserialized.root.name, flame_graph.root.name);
    }

    #[test]
    fn test_text_truncation() {
        let builder = FlameGraphBuilder::new();
        
        let short_text = builder.truncate_text("short", 100);
        assert_eq!(short_text, "short");
        
        let long_text = builder.truncate_text("this_is_a_very_long_function_name", 40);
        assert!(long_text.len() <= 40 / 8 + 3); // Approximate max chars + "..."
        assert!(long_text.ends_with("..."));
    }

    #[test]
    fn test_empty_samples_error() {
        let builder = FlameGraphBuilder::new();
        let empty_samples: Vec<ProfileSample> = Vec::new();
        
        let result = builder.build_from_samples(&empty_samples);
        assert!(matches!(result, Err(FlameGraphError::NoSamples)));
    }
}