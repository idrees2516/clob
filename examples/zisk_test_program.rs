//! Test program for ZisK zkVM
//! 
//! This is a simple Rust program that can be compiled and executed
//! on the ZisK zkVM to test the implementation.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// Entry point for the ZisK program
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Simple computation that can be proven
    let a = 42u64;
    let b = 24u64;
    let result = a + b;
    
    // Output the result (this would be captured as public output)
    unsafe {
        core::ptr::write_volatile(0x1000 as *mut u64, result);
    }
    
    // Halt the program
    loop {}
}

/// Panic handler required for no_std
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

/// Simple arithmetic function for testing
#[no_mangle]
pub extern "C" fn add_numbers(a: u64, b: u64) -> u64 {
    a + b
}

/// More complex computation for testing
#[no_mangle]
pub extern "C" fn fibonacci(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    
    let mut a = 0u64;
    let mut b = 1u64;
    
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    
    b
}

/// Order validation function for trading
#[no_mangle]
pub extern "C" fn validate_order(price: u64, quantity: u64, max_price: u64, max_quantity: u64) -> bool {
    price > 0 && quantity > 0 && price <= max_price && quantity <= max_quantity
}

/// Simple hash function for testing
#[no_mangle]
pub extern "C" fn simple_hash(input: u64) -> u64 {
    // Simple hash using multiplication and XOR
    let mut hash = input;
    hash = hash.wrapping_mul(0x9e3779b97f4a7c15);
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^= hash >> 31;
    hash
}