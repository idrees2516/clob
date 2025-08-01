//! ZisK Integration Demo
//! 
//! This example demonstrates how to use the ZisK zkVM backend
//! for compiling, executing, and proving Rust programs.

use hf_quoting_liquidity_clob::zkvm::{
    zisk::*,
    traits::*,
    router::*,
    ZkVMConfig, ZiskConfig, ZkVMBackend,
};
use std::time::Duration;
use tokio;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting ZisK Integration Demo");

    // Demo 1: Basic ZisK VM usage
    demo_basic_zisk_usage().await?;

    // Demo 2: zkVM Router usage
    demo_zkvm_router().await?;

    // Demo 3: Polynomial commitments
    demo_polynomial_commitments().await?;

    // Demo 4: Post-quantum cryptography
    demo_post_quantum_crypto().await?;

    info!("ZisK Integration Demo completed successfully!");
    Ok(())
}

async fn demo_basic_zisk_usage() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demo 1: Basic ZisK VM Usage ===");

    // Create ZisK configuration
    let config = ZkVMConfig {
        backend: ZkVMBackend::ZisK,
        max_cycles: 1_000_000,
        memory_limit: 64 * 1024 * 1024,
        timeout_seconds: 300,
        network_config: None,
        zisk_config: Some(ZiskConfig {
            optimization_level: 2,
            enable_gpu: false,
            memory_pool_size: 512 * 1024 * 1024,
        }),
        sp1_config: None,
    };

    // Initialize ZisK VM
    let zisk_vm = ZiskVM::new(config)?;
    info!("ZisK VM initialized successfully");

    // Create a test program
    let test_program = create_demo_program();
    info!("Created test program: {}", test_program.metadata.program_id);

    // Create execution inputs
    let inputs = ExecutionInputs::new(
        b"demo_public_input".to_vec(),
        b"demo_private_input".to_vec(),
    );

    // Execute the program
    info!("Executing program...");
    let execution_result = zisk_vm.execute_program(&test_program, &inputs).await?;
    info!(
        "Program executed successfully: {} cycles used",
        execution_result.execution_trace.cycles_used
    );

    // Generate proof
    info!("Generating ZK proof...");
    let proof = zisk_vm.generate_proof(&execution_result).await?;
    info!("Proof generated: {} bytes", proof.size());

    // Verify proof
    info!("Verifying proof...");
    let is_valid = zisk_vm.verify_proof(
        &proof,
        &execution_result.public_outputs,
        &test_program.verification_key,
    ).await?;
    info!("Proof verification result: {}", is_valid);

    Ok(())
}

async fn demo_zkvm_router() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demo 2: zkVM Router Usage ===");

    // Create configurations for multiple backends
    let configs = vec![
        ZkVMConfig {
            backend: ZkVMBackend::ZisK,
            max_cycles: 1_000_000,
            memory_limit: 64 * 1024 * 1024,
            timeout_seconds: 300,
            network_config: None,
            zisk_config: Some(ZiskConfig {
                optimization_level: 2,
                enable_gpu: false,
                memory_pool_size: 512 * 1024 * 1024,
            }),
            sp1_config: None,
        },
    ];

    // Initialize router
    let router = ZkVMRouter::new(configs, SelectionStrategy::Balanced).await?;
    info!("zkVM Router initialized");

    // Get available backends
    let backends = router.get_available_backends().await;
    info!("Available backends: {:?}", backends);

    // Define selection criteria
    let criteria = ZkVMSelectionCriteria {
        complexity: ProofComplexity::Simple,
        latency_requirement: Duration::from_millis(500),
        proof_size_constraint: Some(4096),
        verification_cost_limit: None,
        preferred_backend: None,
    };

    // Select optimal backend
    let selected_backend = router.select_optimal_backend(&criteria).await?;
    info!("Selected backend: {:?}", selected_backend);

    // Generate proof using router
    let test_program = create_demo_program();
    let inputs = ExecutionInputs::new(
        b"router_demo_public".to_vec(),
        b"router_demo_private".to_vec(),
    );

    info!("Generating proof using router...");
    let proof = router.generate_proof_with_optimal_backend(
        &test_program,
        &inputs,
        &criteria,
    ).await?;
    info!("Router proof generated: {} bytes", proof.size());

    Ok(())
}

async fn demo_polynomial_commitments() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demo 3: Polynomial Commitments ===");

    // Initialize polynomial committer
    let committer = ZiskPolynomialCommitter::new()?;
    info!("Polynomial committer initialized");

    // Create test polynomial coefficients
    let coefficients = b"test_polynomial_coefficients_for_demo";
    info!("Created polynomial with {} coefficients", coefficients.len());

    // Generate commitment
    let commitment = committer.commit_polynomial(coefficients)?;
    info!("Generated polynomial commitment: {} bytes", commitment.len());

    // Generate opening proof
    let point = 42u64;
    let (value, proof) = committer.open_polynomial(coefficients, point)?;
    info!("Generated opening proof at point {}: value={}, proof={} bytes", 
           point, value, proof.len());

    // Verify opening
    let is_valid = committer.verify_opening(&commitment, point, value, &proof)?;
    info!("Opening verification result: {}", is_valid);

    Ok(())
}

async fn demo_post_quantum_crypto() -> Result<(), Box<dyn std::error::Error>> {
    info!("=== Demo 4: Post-Quantum Cryptography ===");

    // Initialize post-quantum crypto
    let pq_crypto = PostQuantumCrypto::new()?;
    info!("Post-quantum cryptography initialized");

    // Test data
    let test_data = b"sensitive_trading_data_for_encryption";
    info!("Test data: {} bytes", test_data.len());

    // Encrypt data
    let encrypted_data = pq_crypto.encrypt(test_data)?;
    info!("Data encrypted: {} bytes", encrypted_data.len());

    // Decrypt data
    let decrypted_data = pq_crypto.decrypt(&encrypted_data)?;
    info!("Data decrypted: {} bytes", decrypted_data.len());

    // Verify decryption
    let decryption_success = decrypted_data == test_data;
    info!("Decryption verification: {}", decryption_success);

    // Sign data
    let signature = pq_crypto.sign(test_data)?;
    info!("Data signed: {} bytes signature", signature.len());

    // Verify signature
    let signature_valid = pq_crypto.verify_signature(test_data, &signature)?;
    info!("Signature verification: {}", signature_valid);

    Ok(())
}

fn create_demo_program() -> CompiledProgram {
    let program_id = format!("demo_program_{}", uuid::Uuid::new_v4());
    
    // Create demo bytecode
    let mut bytecode = Vec::new();
    bytecode.extend_from_slice(b"ZISK_BYTECODE_V1");
    
    // Add demo hash for verification
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(b"demo_elf_content");
    bytecode.extend_from_slice(&hasher.finalize());
    
    // Add bytecode length and instructions
    let instructions = vec![
        0x01, 0x00, 0x00, 0x00, // ADD instruction
        0x02, 0x01, 0x02, 0x03, // ADD r1, r2 -> r3
        0xFF, 0x00, 0x00, 0x00, // HALT instruction
    ];
    
    bytecode.extend_from_slice(&(instructions.len() as u64).to_le_bytes());
    bytecode.extend_from_slice(b"ZISK_INSTR_V1");
    bytecode.extend_from_slice(&instructions);

    CompiledProgram {
        backend: ZkVMBackend::ZisK,
        bytecode,
        metadata: ProgramMetadata {
            program_id: program_id.clone(),
            version: "1.0.0".to_string(),
            compilation_time: chrono::Utc::now().timestamp() as u64,
            optimization_level: 2,
            target_arch: "ZisK-Custom".to_string(),
            memory_layout: MemoryLayout {
                stack_size: 64 * 1024,
                heap_size: 1024 * 1024,
                code_size: 2048,
                data_size: 4096,
            },
        },
        verification_key: VerificationKey {
            backend: ZkVMBackend::ZisK,
            key_data: b"demo_verification_key_data".to_vec(),
            key_hash: [0u8; 32],
            version: "1.0.0".to_string(),
        },
    }
}