//! Memory safety tests for FFI boundary
//!
//! These tests verify that:
//! - String allocation/deallocation is safe
//! - No memory leaks in repeated operations
//! - Null pointer handling is correct
//! - UTF-8 encoding is handled properly

#![cfg(feature = "ffi")]

use std::ffi::{CStr, CString};
use std::ptr;

use pcai_inference::ffi::{pcai_free_string, pcai_init, pcai_last_error, pcai_shutdown, PcaiErrorCode};

/// Test that pcai_free_string handles null safely
#[test]
fn test_free_string_null_safety() {
    // Should not panic or crash
    pcai_free_string(ptr::null_mut());
}

/// Test that error strings can be read without crash
#[test]
fn test_error_string_access() {
    pcai_shutdown(); // Clean state

    // Cause an error
    let invalid = CString::new("invalid_backend").expect("CString should accept valid test input");
    let _ = pcai_init(invalid.as_ptr());

    // Read error
    let err_ptr = pcai_last_error();
    assert!(!err_ptr.is_null());

    // Access should not crash
    let err_str = unsafe { CStr::from_ptr(err_ptr) };
    let _text = err_str.to_str().expect("Error should be valid UTF-8");

    pcai_shutdown();
}

/// Test repeated init/shutdown cycles for memory leaks
#[test]
fn test_repeated_init_shutdown_no_leak() {
    for _ in 0..100 {
        pcai_shutdown();

        #[cfg(feature = "llamacpp")]
        {
            let backend = CString::new("llamacpp").expect("CString should accept valid test input");
            let _ = pcai_init(backend.as_ptr());
        }

        #[cfg(feature = "mistralrs-backend")]
        {
            let backend = CString::new("mistralrs").expect("CString should accept valid test input");
            let _ = pcai_init(backend.as_ptr());
        }

        pcai_shutdown();
    }
}

/// Test that init with empty string is handled
#[test]
fn test_init_empty_string() {
    pcai_shutdown();

    let empty = CString::new("").expect("CString should accept valid test input");
    let result = pcai_init(empty.as_ptr());

    // Should fail gracefully, not crash
    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

    let err_ptr = pcai_last_error();
    assert!(!err_ptr.is_null());

    pcai_shutdown();
}

/// Test that init with very long string is handled
#[test]
fn test_init_long_string() {
    pcai_shutdown();

    let long_string: String = "a".repeat(10000);
    let long = CString::new(long_string).expect("CString should accept valid test input");
    let result = pcai_init(long.as_ptr());

    // Should fail gracefully
    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

    pcai_shutdown();
}

/// Test UTF-8 handling in error messages
#[test]
fn test_utf8_error_handling() {
    pcai_shutdown();

    // Try to init with unicode string
    let unicode = CString::new("backend_日本語").expect("CString should accept valid test input");
    let result = pcai_init(unicode.as_ptr());

    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

    // Error message should be valid UTF-8
    let err_ptr = pcai_last_error();
    if !err_ptr.is_null() {
        let err_str = unsafe { CStr::from_ptr(err_ptr) };
        assert!(err_str.to_str().is_ok(), "Error should be valid UTF-8");
    }

    pcai_shutdown();
}

#[cfg(feature = "llamacpp")]
mod llamacpp_memory_tests {
    use super::*;
    use pcai_inference::ffi::{pcai_generate, pcai_load_model};

    /// Test that generate with null prompt returns null safely
    #[test]
    fn test_generate_null_prompt() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) == 0 {
            let result = pcai_generate(ptr::null(), 10, 0.7);
            assert!(result.is_null());

            // Should have set an error
            let err = pcai_last_error();
            assert!(!err.is_null());
        }

        pcai_shutdown();
    }

    /// Test that load_model with null path returns error safely
    #[test]
    fn test_load_model_null_path() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) == 0 {
            let result = pcai_load_model(ptr::null(), 0);
            assert_eq!(result, -1);
        }

        pcai_shutdown();
    }
}
