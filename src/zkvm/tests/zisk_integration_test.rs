//! Integration tests for ZisK zkVM implementation
//! 
//! These tests verify that the ZisK zkVM backend works correctly
//! for program compilation, execution, and proof generation.

#[cfg(test)]
mod tests {
    use crate::zkvm::{
        zisk::*,
        traits::*,
        ZkVMConfig, ZiskConfig, ZkVMBackend, ZkVMError,
    };
    use std::path::Path;
    use tokio;
    use tracing_test::traced_test;

    /// Test ZisK VM initialization
    #[tokio::test]
    #[traced_test]
    async fn test_zisk_vm_initialization() {
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

        let zisk_vm = ZiskVM::new(config);
        assert!(zisk_vm.is_ok());

        let vm = zisk_vm.unwrap();
        assert_eq!(vm.backend_type(), ZkVMBackend::ZisK);
    }

    /// Test ZisK compiler initialization
    #[tokio::test]
    #[traced_test]
    async fn test_zisk_compiler_initialization() {
        let config = ZiskConfig {
            optimization_level: 2,
            enable_gpu: false,
            memory_pool_size: 512 * 1024 * 1024,
        };

        let compiler = ZiskCompiler::new(config);
        assert!(compiler.is_ok());

        let comp = compiler.unwrap();
        assert_eq!(comp.target_architecture(), TargetArchitecture::Custom(0x5A15));
    }

    /// Test program compilation from ELF
    #[tokio::test]
    #[traced_test]
    async fn test_program_compilation_from_elf() {
        let config = ZiskConfig {
            optimization_level: 2,
            enable_gpu: false,
            memory_pool_size: 512 * 1024 * 1024,
        };

        let compiler = ZiskCompiler::new(config).unwrap();

        // Create a simple ELF-like binary for testing
        let test_elf = create_test_elf_binary();

        let result = compiler.compile_from_elf(&test_elf).await;
        assert!(result.is_ok());

        let compiled_program = result.unwrap();
        assert_eq!(compiled_program.backend, ZkVMBackend::ZisK);
        assert!(!compiled_program.bytecode.is_empty());
        assert_eq!(compiled_program.metadata.target_arch, "ZisK-Custom");
    }

    /// Test program execution
    #[tokio::test]
    #[traced_test]
    async fn test_program_execution() {
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

        let zisk_vm = ZiskVM::new(config).unwrap();

        // Create a test program
        let test_program = create_test_compiled_program();
        let test_inputs = ExecutionInputs::new(
            b"test_public_input".to_vec(),
            b"test_private_input".to_vec(),
        );

        let result = zisk_vm.execute_program(&test_program, &test_inputs).await;
        assert!(result.is_ok());

        let execution_result = result.unwrap();
        assert_eq!(execution_result.program_id, test_program.metadata.program_id);
        assert!(!execution_result.public_outputs.is_empty());
        assert!(execution_result.execution_trace.cycles_used > 0);
    }

    /// Test proof generation
    #[tokio::test]
    #[traced_test]
    async fn test_proof_generation() {
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

        let zisk_vm = ZiskVM::new(config).unwrap();

        // Create test execution result
        let execution_result = create_test_execution_result();

        let result = zisk_vm.generate_proof(&execution_result).await;
        assert!(result.is_ok());

        let proof = result.unwrap();
        assert_eq!(proof.backend, ZkVMBackend::ZisK);
        assert!(!proof.proof_data.is_empty());
        assert!(proof.verify_integrity());
    }

    /// Test proof verification
    #[tokio::test]
    #[traced_test]
    async fn test_proof_verification() {
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

        let zisk_vm = ZiskVM::new(config).unwrap();

        // Generate a proof first
        let execution_result = create_test_execution_result();
        let proof = zisk_vm.generate_proof(&execution_result).await.unwrap();

        // Create verification key
        let verification_key = VerificationKey {
            backend: ZkVMBackend::ZisK,
            key_data: b"test_verification_key".to_vec(),
            key_hash: [0u8; 32],
            version: "1.0.0".to_string(),
        };

        let result = zisk_vm.verify_proof(
            &proof,
            &execution_result.public_outputs,
            &verification_key,
        ).await;

        assert!(result.is_ok());
        assert!(result.unwrap()); // Proof should be valid
    }

    /// Test witness generation
    #[tokio::test]
    #[traced_test]
    async fn test_witness_generation() {
        let witness_generator = ZiskWitnessGenerator::new();
        let execution_result = create_test_execution_result();

        let result = witness_generator.generate_witness(&execution_result);
        assert!(result.is_ok());

        let witness = result.unwrap();
        assert_eq!(witness.execution_id, execution_result.execution_id);
        assert!(!witness.witness_data.is_empty());
        assert!(witness.witness_data.starts_with(b"ZISK_WITNESS_V1"));
    }

    /// Test witness validation
    #[tokio::test]
    #[traced_test]
    async fn test_witness_validation() {
        let witness_generator = ZiskWitnessGenerator::new();
        let execution_result = create_test_execution_result();
        let witness = witness_generator.generate_witness(&execution_result).unwrap();

        let result = witness_generator.validate_witness(
            &witness,
            &execution_result.public_outputs,
        );

        assert!(result.is_ok());
        assert!(result.unwrap()); // Witness should be valid
    }

    /// Test polynomial commitment system
    #[tokio::test]
    #[traced_test]
    async fn test_polynomial_commitment() {
        let committer = ZiskPolynomialCommitter::new();
        assert!(committer.is_ok());

        let poly_committer = committer.unwrap();
        let test_coefficients = b"test_polynomial_coefficients";

        // Test commitment
        let commitment_result = poly_committer.commit_polynomial(test_coefficients);
        assert!(commitment_result.is_ok());

        let commitment = commitment_result.unwrap();
        assert!(!commitment.is_empty());
        assert!(commitment.starts_with(b"KZG_COMMITMENT_V1"));

        // Test opening
        let point = 42u64;
        let opening_result = poly_committer.open_polynomial(test_coefficients, point);
        assert!(opening_result.is_ok());

        let (value, proof) = opening_result.unwrap();
        assert!(!proof.is_empty());

        // Test verification
        let verification_result = poly_committer.verify_opening(&commitment, point, value, &proof);
        assert!(verification_result.is_ok());
        assert!(verification_result.unwrap());
    }

    /// Test post-quantum cryptography
    #[tokio::test]
    #[traced_test]
    async fn test_post_quantum_crypto() {
        let pq_crypto = PostQuantumCrypto::new();
        assert!(pq_crypto.is_ok());

        let crypto = pq_crypto.unwrap();
        let test_data = b"test_data_for_encryption";

        // Test encryption/decryption
        let encrypted_result = crypto.encrypt(test_data);
        assert!(encrypted_result.is_ok());

        let encrypted_data = encrypted_result.unwrap();
        assert!(!encrypted_data.is_empty());

        let decrypted_result = crypto.decrypt(&encrypted_data);
        assert!(decrypted_result.is_ok());

        let decrypted_data = decrypted_result.unwrap();
        assert_eq!(decrypted_data, test_data);

        // Test signing/verification
        let signature_result = crypto.sign(test_data);
        assert!(signature_result.is_ok());

        let signature = signature_result.unwrap();
        assert!(!signature.is_empty());

        let verification_result = crypto.verify_signature(test_data, &signature);
        assert!(verification_result.is_ok());
        assert!(verification_result.unwrap());
    }

    /// Test end-to-end workflow
    #[tokio::test]
    #[traced_test]
    async fn test_end_to_end_workflow() {
        // Initialize ZisK VM
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

        let zisk_vm = ZiskVM::new(config).unwrap();

        // Compile program
        let test_program = create_test_compiled_program();

        // Execute program
        let test_inputs = ExecutionInputs::new(
            b"end_to_end_public".to_vec(),
            b"end_to_end_private".to_vec(),
        );

        let execution_result = zisk_vm.execute_program(&test_program, &test_inputs).await.unwrap();

        // Generate proof
        let proof = zisk_vm.generate_proof(&execution_result).await.unwrap();

        // Verify proof
        let verification_key = VerificationKey {
            backend: ZkVMBackend::ZisK,
            key_data: b"end_to_end_vk".to_vec(),
            key_hash: [0u8; 32],
            version: "1.0.0".to_string(),
        };

        let is_valid = zisk_vm.verify_proof(
            &proof,
            &execution_result.public_outputs,
            &verification_key,
        ).await.unwrap();

        assert!(is_valid);
    }

    // Helper functions for creating test data

    fn create_test_elf_binary() -> Vec<u8> {
        let mut elf_data = Vec::new();
        elf_data.extend_from_slice(b"\x7fELF"); // ELF magic
        elf_data.extend_from_slice(&[1, 1, 1, 0]); // ELF header continuation
        elf_data.extend_from_slice(&[0u8; 8]); // Padding
        
        // Add some dummy program data
        for i in 0..1000 {
            elf_data.push((i % 256) as u8);
        }
        
        elf_data
    }

    fn create_test_compiled_program() -> CompiledProgram {
        let program_id = "test_program_123".to_string();
        let bytecode = create_test_bytecode();

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
                key_data: b"test_verification_key_data".to_vec(),
                key_hash: [0u8; 32],
                version: "1.0.0".to_string(),
            },
        }
    }

    fn create_test_bytecode() -> Vec<u8> {
        let mut bytecode = Vec::new();
        bytecode.extend_from_slice(b"ZISK_BYTECODE_V1");
        
        // Add some test instructions
        bytecode.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // ADD instruction
        bytecode.extend_from_slice(&[0x02, 0x01, 0x02, 0x03]); // ADD r1, r2 -> r3
        bytecode.extend_from_slice(&[0xFF, 0x00, 0x00, 0x00]); // HALT instruction
        
        bytecode
    }

    fn create_test_execution_result() -> ExecutionResult {
        ExecutionResult {
            program_id: "test_program_123".to_string(),
            execution_id: "test_execution_456".to_string(),
            public_outputs: b"test_public_output".to_vec(),
            private_outputs: b"test_private_output".to_vec(),
            execution_trace: ExecutionTrace {
                cycles_used: 100,
                memory_accesses: vec![
                    MemoryAccess {
                        cycle: 1,
                        address: 0x1000,
                        value: 0x12345678,
                        access_type: MemoryAccessType::Read,
                    },
                    MemoryAccess {
                        cycle: 2,
                        address: 0x1004,
                        value: 0x87654321,
                        access_type: MemoryAccessType::Write,
                    },
                ],
                register_states: vec![
                    RegisterState {
                        cycle: 1,
                        registers: {
                            let mut regs = std::collections::HashMap::new();
                            regs.insert(1, 0x12345678);
                            regs.insert(2, 0x87654321);
                            regs
                        },
                        pc: 0x1000,
                    },
                ],
                instruction_trace: vec![
                    Instruction {
                        cycle: 1,
                        pc: 0x1000,
                        opcode: 0x01,
                        operands: vec![1, 2, 3],
                        result: Some(0x12345678 + 0x87654321),
                    },
                ],
                syscall_trace: vec![],
            },
            final_state: b"test_final_state".to_vec(),
            stats: ExecutionStats {
                total_cycles: 100,
                memory_usage: 1024,
                execution_time_ms: 50,
                proof_generation_time_ms: None,
                verification_time_ms: None,
                gas_used: 200,
            },
        }
    }
}