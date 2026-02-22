//! FFI Integration tests for pcai-inference
//!
//! These tests verify the FFI boundary between Rust and external callers (PowerShell/C#).
//! They focus on:
//! - Memory safety across the FFI boundary
//! - Concurrent access patterns
//! - Error propagation
//! - Resource cleanup

pub mod concurrent_access;
pub mod error_propagation;
pub mod memory_safety;
