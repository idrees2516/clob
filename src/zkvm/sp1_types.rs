//! SP1 zkVM specific data types and structures
//! 
//! This module contains all the SP1-specific data structures needed
//! for the SP1 zkVM implementation.

use crate::zkvm::{traits::*, ZkVMError, SP1Config, SP1ProverMode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use rand::{Rng, RngCore};

/// SP1 runtime environment
#[derive(Clone)]
pub struct SP1Runtime {
    pub program_id: String,
    pub execution_context: SP1ExecutionContext,
    pub circuit_info: SP1CircuitInfo,
    pub proving_key: SP1ProvingKey,
    pub verification_key: SP1VerificationKey,
    pub shard_config: SP1ShardConfig,
}

/// SP1 execution context
#[derive(Clone)]
pub struct SP1ExecutionContext {
    pub max_cycles: u64,
    pub memory_limit: usize,
    pub prover_mode: SP1ProverMode,
    pub enable_cuda: bool,
    pub shard_size: u32,
}

/// SP1 circuit information
#[derive(Clone)]
pub struct SP1CircuitInfo {
    pub circuit_id: String,
    pub constraint_count: usize,
    pub witness_size: usize,
    pub public_input_size: usize,
    pub shard_count: usize,
}

/// SP1 proving key
#[derive(Clone)]
pub struct SP1ProvingKey {
    pub key_id: String,
    pub key_data: Vec<u8>,
    pub constraint_count: usize,
    pub shard_size: u32,
}

/// SP1 verification key
#[derive(Clone)]
pub struct SP1VerificationKey {
    pub key_id: String,
    pub key_data: Vec<u8>,
    pub public_input_size: usize,
}

/// SP1 shard configuration
#[derive(Clone)]
pub struct SP1ShardConfig {
    pub shard_size: u32,
    pub max_shards: usize,
    pub parallel_proving: bool,
}

/// SP1 Virtual Machine State
pub struct SP1VMState {
    memory: Vec<u8>,
    registers: [u64; 32],
    pc: u64,
    memory_limit: usize,
    max_cycles: u64,
    cycles_used: u64,
    halted: bool,
    public_inputs: Vec<u8>,
    private_inputs: Vec<u8>,
    public_outputs: Vec<u8>,
    private_outputs: Vec<u8>,
    initial_state: Vec<u8>,
    final_state: Vec<u8>,
    memory_accesses: Vec<MemoryAccess>,
    pending_syscall: Option<Syscall>,
    shard_config: SP1ShardConfig,
    current_shard: usize,
    // SP1-specific features
    heap: Vec<u8>,
    stack: Vec<u8>,
    std_lib_state: SP1StdLibState,
}

/// SP1 standard library state
#[derive(Clone)]
pub struct SP1StdLibState {
    pub heap_pointer: u64,
    pub stack_pointer: u64,
    pub file_descriptors: HashMap<u32, SP1FileDescriptor>,
    pub environment_vars: HashMap<String, String>,
    pub random_state: u64,
}

/// SP1 file descriptor for std library support
#[derive(Clone)]
pub struct SP1FileDescriptor {
    pub fd: u32,
    pub file_type: SP1FileType,
    pub position: u64,
    pub data: Vec<u8>,
}

#[derive(Clone)]
pub enum SP1FileType {
    Stdin,
    Stdout,
    Stderr,
    File(String),
}

impl SP1VMState {
    pub fn new(
        memory_limit: usize,
        max_cycles: u64,
        shard_config: SP1ShardConfig,
    ) -> Result<Self, ZkVMError> {
        let heap_size = memory_limit / 4; // 25% for heap
        let stack_size = memory_limit / 8; // 12.5% for stack
        
        Ok(Self {
            memory: vec![0; memory_limit],
            registers: [0; 32],
            pc: 0,
            memory_limit,
            max_cycles,
            cycles_used: 0,
            halted: false,
            public_inputs: Vec::new(),
            private_inputs: Vec::new(),
            public_outputs: Vec::new(),
            private_outputs: Vec::new(),
            initial_state: Vec::new(),
            final_state: Vec::new(),
            memory_accesses: Vec::new(),
            pending_syscall: None,
            shard_config,
            current_shard: 0,
            heap: vec![0; heap_size],
            stack: vec![0; stack_size],
            std_lib_state: SP1StdLibState::new(),
        })
    }

    pub fn load_program(&mut self, elf_bytes: &[u8]) -> Result<(), ZkVMError> {
        if elf_bytes.len() > self.memory_limit / 2 {
            return Err(ZkVMError::ExecutionFailed("Program too large".to_string()));
        }
        
        // Load ELF into memory starting at address 0x10000
        let load_addr = 0x10000;
        if load_addr + elf_bytes.len() > self.memory.len() {
            return Err(ZkVMError::ExecutionFailed("Not enough memory".to_string()));
        }
        
        self.memory[load_addr..load_addr + elf_bytes.len()].copy_from_slice(elf_bytes);
        self.pc = load_addr as u64;
        
        // Initialize SP1-specific state
        self.std_lib_state.heap_pointer = (load_addr + elf_bytes.len() + 4096) as u64; // Align to 4KB
        self.std_lib_state.stack_pointer = (self.memory_limit - 1024) as u64; // Stack at top
        
        debug!("SP1 program loaded at address 0x{:x}, size: {} bytes", load_addr, elf_bytes.len());
        Ok(())
    }

    pub fn set_public_inputs(&mut self, inputs: &[u8]) -> Result<(), ZkVMError> {
        self.public_inputs = inputs.to_vec();
        // SP1 can pass inputs through memory-mapped region
        if !inputs.is_empty() && inputs.len() <= 4096 {
            let input_addr = 0x8000; // Reserved input region
            self.memory[input_addr..input_addr + inputs.len()].copy_from_slice(inputs);
        }
        Ok(())
    }

    pub fn set_private_inputs(&mut self, inputs: &[u8]) -> Result<(), ZkVMError> {
        self.private_inputs = inputs.to_vec();
        Ok(())
    }

    pub fn set_initial_state(&mut self, state: &[u8]) -> Result<(), ZkVMError> {
        self.initial_state = state.to_vec();
        Ok(())
    }

    pub fn get_public_outputs(&self) -> Result<Vec<u8>, ZkVMError> {
        Ok(self.public_outputs.clone())
    }

    pub fn get_private_outputs(&self) -> Result<Vec<u8>, ZkVMError> {
        Ok(self.private_outputs.clone())
    }

    pub fn get_final_state(&self) -> Result<Vec<u8>, ZkVMError> {
        Ok(self.final_state.clone())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.memory.len() + self.heap.len() + self.stack.len()
    }

    pub fn is_halted(&self) -> bool {
        self.halted
    }

    pub fn get_program_counter(&self) -> u64 {
        self.pc
    }

    pub fn fetch_instruction(&mut self) -> Result<SP1Instruction, ZkVMError> {
        if self.pc as usize + 4 > self.memory.len() {
            return Err(ZkVMError::ExecutionFailed("PC out of bounds".to_string()));
        }
        
        let instr_bytes = [
            self.memory[self.pc as usize],
            self.memory[self.pc as usize + 1],
            self.memory[self.pc as usize + 2],
            self.memory[self.pc as usize + 3],
        ];
        
        let instr_word = u32::from_le_bytes(instr_bytes);
        Ok(SP1Instruction::decode(instr_word))
    }

    pub fn execute_instruction(&mut self, instruction: &SP1Instruction) -> Result<Option<u64>, ZkVMError> {
        match instruction.opcode {
            0x00 => {
                // NOP
                self.pc += 4;
                Ok(None)
            }
            0x01 => {
                // ADD with overflow detection
                let a = self.registers[instruction.rs1 as usize];
                let b = self.registers[instruction.rs2 as usize];
                let (result, overflow) = a.overflowing_add(b);
                self.registers[instruction.rd as usize] = result;
                if overflow {
                    // SP1 can handle overflow conditions
                    self.handle_arithmetic_overflow()?;
                }
                self.pc += 4;
                Ok(Some(result))
            }
            0x02 => {
                // LOAD with bounds checking
                let addr = self.registers[instruction.rs1 as usize].wrapping_add(instruction.imm as u64);
                if addr as usize + 8 <= self.memory.len() {
                    let value = u64::from_le_bytes([
                        self.memory[addr as usize],
                        self.memory[addr as usize + 1],
                        self.memory[addr as usize + 2],
                        self.memory[addr as usize + 3],
                        self.memory[addr as usize + 4],
                        self.memory[addr as usize + 5],
                        self.memory[addr as usize + 6],
                        self.memory[addr as usize + 7],
                    ]);
                    
                    self.registers[instruction.rd as usize] = value;
                    self.memory_accesses.push(MemoryAccess {
                        cycle: self.cycles_used,
                        address: addr,
                        value,
                        access_type: MemoryAccessType::Read,
                    });
                    
                    self.pc += 4;
                    Ok(Some(value))
                } else {
                    Err(ZkVMError::ExecutionFailed("Memory access out of bounds".to_string()))
                }
            }
            0x03 => {
                // STORE with bounds checking
                let addr = self.registers[instruction.rs1 as usize].wrapping_add(instruction.imm as u64);
                let value = self.registers[instruction.rs2 as usize];
                
                if addr as usize + 8 <= self.memory.len() {
                    let bytes = value.to_le_bytes();
                    self.memory[addr as usize..addr as usize + 8].copy_from_slice(&bytes);
                    
                    self.memory_accesses.push(MemoryAccess {
                        cycle: self.cycles_used,
                        address: addr,
                        value,
                        access_type: MemoryAccessType::Write,
                    });
                    
                    self.pc += 4;
                    Ok(Some(value))
                } else {
                    Err(ZkVMError::ExecutionFailed("Memory access out of bounds".to_string()))
                }
            }
            0x10 => {
                // SYSCALL - SP1 supports rich syscall interface
                self.handle_syscall(instruction)?;
                self.pc += 4;
                Ok(None)
            }
            0xFF => {
                // HALT
                self.halted = true;
                self.finalize_execution()?;
                Ok(None)
            }
            _ => {
                // Unknown instruction - SP1 is more permissive
                self.pc += 4;
                Ok(None)
            }
        }
    }

    fn handle_arithmetic_overflow(&mut self) -> Result<(), ZkVMError> {
        // SP1 can handle overflow conditions gracefully
        debug!("Arithmetic overflow detected at cycle {}", self.cycles_used);
        Ok(())
    }

    fn handle_syscall(&mut self, instruction: &SP1Instruction) -> Result<(), ZkVMError> {
        let syscall_id = instruction.imm;
        
        match syscall_id {
            1 => self.syscall_write()?, // write
            2 => self.syscall_read()?,  // read
            3 => self.syscall_open()?,  // open
            4 => self.syscall_close()?, // close
            5 => self.syscall_malloc()?, // malloc
            6 => self.syscall_free()?,  // free
            _ => {
                debug!("Unknown syscall: {}", syscall_id);
            }
        }
        
        Ok(())
    }

    fn syscall_write(&mut self) -> Result<(), ZkVMError> {
        let fd = self.registers[10] as u32; // a0
        let buf_addr = self.registers[11]; // a1
        let count = self.registers[12] as usize; // a2
        
        if buf_addr as usize + count <= self.memory.len() {
            let data = &self.memory[buf_addr as usize..buf_addr as usize + count];
            
            // Handle stdout/stderr specially
            if fd == 1 || fd == 2 {
                self.public_outputs.extend_from_slice(data);
            }
            
            self.registers[10] = count as u64; // Return bytes written
        } else {
            self.registers[10] = u64::MAX; // Error
        }
        
        Ok(())
    }

    fn syscall_read(&mut self) -> Result<(), ZkVMError> {
        let fd = self.registers[10] as u32;
        let buf_addr = self.registers[11];
        let count = self.registers[12] as usize;
        
        // For stdin, read from public inputs
        if fd == 0 && !self.public_inputs.is_empty() {
            let to_read = count.min(self.public_inputs.len());
            if buf_addr as usize + to_read <= self.memory.len() {
                self.memory[buf_addr as usize..buf_addr as usize + to_read]
                    .copy_from_slice(&self.public_inputs[..to_read]);
                self.registers[10] = to_read as u64;
            } else {
                self.registers[10] = u64::MAX; // Error
            }
        } else {
            self.registers[10] = 0; // EOF
        }
        
        Ok(())
    }

    fn syscall_open(&mut self) -> Result<(), ZkVMError> {
        // Simplified file open - return a dummy fd
        self.registers[10] = 3; // Return fd 3
        Ok(())
    }

    fn syscall_close(&mut self) -> Result<(), ZkVMError> {
        // Simplified file close
        self.registers[10] = 0; // Success
        Ok(())
    }

    fn syscall_malloc(&mut self) -> Result<(), ZkVMError> {
        let size = self.registers[10] as usize;
        
        // Simple heap allocation
        if self.std_lib_state.heap_pointer as usize + size < self.heap.len() {
            let addr = self.std_lib_state.heap_pointer;
            self.std_lib_state.heap_pointer += size as u64;
            self.registers[10] = addr; // Return allocated address
        } else {
            self.registers[10] = 0; // NULL - allocation failed
        }
        
        Ok(())
    }

    fn syscall_free(&mut self) -> Result<(), ZkVMError> {
        // Simplified free - in a real implementation, this would manage heap
        self.registers[10] = 0; // Success
        Ok(())
    }

    fn finalize_execution(&mut self) -> Result<(), ZkVMError> {
        // Generate outputs based on execution
        if self.public_outputs.is_empty() {
            self.public_outputs = self.public_inputs.clone();
            self.public_outputs.extend_from_slice(b"_SP1_PROCESSED");
        }
        
        self.final_state = self.initial_state.clone();
        self.final_state.extend_from_slice(b"_SP1_FINAL");
        self.final_state.extend_from_slice(&self.cycles_used.to_le_bytes());
        
        Ok(())
    }

    pub fn get_memory_accesses_since_last_check(&mut self) -> Vec<MemoryAccess> {
        let accesses = self.memory_accesses.clone();
        self.memory_accesses.clear();
        accesses
    }

    pub fn get_register_state(&self) -> HashMap<u8, u64> {
        let mut state = HashMap::new();
        for (i, &value) in self.registers.iter().enumerate() {
            if value != 0 {
                state.insert(i as u8, value);
            }
        }
        state
    }

    pub fn check_pending_syscall(&mut self) -> Option<Syscall> {
        self.pending_syscall.take()
    }
}

/// SP1 instruction representation
#[derive(Debug, Clone)]
pub struct SP1Instruction {
    pub opcode: u32,
    pub rd: u32,
    pub rs1: u32,
    pub rs2: u32,
    pub imm: i32,
}

impl SP1Instruction {
    pub fn decode(instr_word: u32) -> Self {
        // RISC-V instruction decoding
        let opcode = instr_word & 0x7F;
        let rd = (instr_word >> 7) & 0x1F;
        let rs1 = (instr_word >> 15) & 0x1F;
        let rs2 = (instr_word >> 20) & 0x1F;
        let imm = ((instr_word as i32) >> 20); // Sign-extended immediate
        
        Self {
            opcode,
            rd,
            rs1,
            rs2,
            imm,
        }
    }
}

impl SP1StdLibState {
    pub fn new() -> Self {
        let mut file_descriptors = HashMap::new();
        
        // Standard file descriptors
        file_descriptors.insert(0, SP1FileDescriptor {
            fd: 0,
            file_type: SP1FileType::Stdin,
            position: 0,
            data: Vec::new(),
        });
        
        file_descriptors.insert(1, SP1FileDescriptor {
            fd: 1,
            file_type: SP1FileType::Stdout,
            position: 0,
            data: Vec::new(),
        });
        
        file_descriptors.insert(2, SP1FileDescriptor {
            fd: 2,
            file_type: SP1FileType::Stderr,
            position: 0,
            data: Vec::new(),
        });
        
        Self {
            heap_pointer: 0,
            stack_pointer: 0,
            file_descriptors,
            environment_vars: HashMap::new(),
            random_state: rand::thread_rng().gen(),
        }
    }
}