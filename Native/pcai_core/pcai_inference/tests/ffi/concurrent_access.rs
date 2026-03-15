//! Concurrent access tests for FFI boundary
//!
//! These tests verify thread safety of the FFI functions.
//! Note: The global state in the FFI module means only one backend
//! can be active at a time, but the API should be safe from concurrent calls.

#![cfg(feature = "ffi")]

use std::ffi::CString;
use std::sync::Arc;
use std::thread;

use pcai_inference::ffi::{pcai_init, pcai_last_error, pcai_shutdown};

/// Test that concurrent shutdown calls don't crash
#[test]
fn test_concurrent_shutdown() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                pcai_shutdown();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}

/// Test that concurrent init calls are handled safely
#[test]
fn test_concurrent_init_calls() {
    pcai_shutdown();

    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                #[cfg(feature = "llamacpp")]
                {
                    let backend = CString::new("llamacpp").expect("CString should accept valid test input");
                    let _ = pcai_init(backend.as_ptr());
                }

                #[cfg(not(feature = "llamacpp"))]
                {
                    let backend =
                        CString::new(format!("test_{}", i)).expect("CString should accept formatted test input");
                    let _ = pcai_init(backend.as_ptr());
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    pcai_shutdown();
}

/// Test that error access from multiple threads is safe
#[test]
fn test_concurrent_error_access() {
    pcai_shutdown();

    // Cause an error
    let invalid = CString::new("invalid").expect("CString should accept valid test input");
    let _ = pcai_init(invalid.as_ptr());

    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                let _ = pcai_last_error();
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    pcai_shutdown();
}

/// Test rapid init/shutdown cycling from multiple threads
#[test]
fn test_rapid_init_shutdown_cycling() {
    let barrier = Arc::new(std::sync::Barrier::new(5));

    let handles: Vec<_> = (0..5)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for _ in 0..20 {
                    pcai_shutdown();

                    #[cfg(feature = "llamacpp")]
                    {
                        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
                        let _ = pcai_init(backend.as_ptr());
                    }

                    pcai_shutdown();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    pcai_shutdown();
}

#[cfg(feature = "llamacpp")]
mod llamacpp_concurrent_tests {
    use super::*;
    use pcai_inference::ffi::pcai_generate;
    use std::ffi::CString;
    use std::ptr;

    /// Test concurrent generate calls (should be serialized by mutex)
    #[test]
    fn test_concurrent_generate_without_model() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) != 0 {
            return; // Skip if init fails
        }

        let handles: Vec<_> = (0..5)
            .map(|i| {
                thread::spawn(move || {
                    let prompt =
                        CString::new(format!("Test prompt {}", i)).expect("CString should accept formatted test input");
                    let result = pcai_generate(prompt.as_ptr(), 10, 0.7);
                    // Should return null (no model loaded) but not crash
                    assert!(result.is_null());
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }

        pcai_shutdown();
    }
}
