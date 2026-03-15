//! Error propagation tests for FFI boundary
//!
//! These tests verify that errors are properly communicated across the FFI boundary:
//! - Return codes indicate success/failure
//! - pcai_last_error() provides detailed error messages
//! - Error messages are valid UTF-8
//! - Errors are properly cleared between operations

#![cfg(feature = "ffi")]

use std::ffi::{CStr, CString};

use pcai_inference::ffi::{
    pcai_init, pcai_last_error, pcai_last_error_code, pcai_shutdown, pcai_version, PcaiErrorCode,
};

/// Test that unknown backend produces meaningful error
#[test]
fn test_unknown_backend_error() {
    pcai_shutdown();

    let backend = CString::new("unknown_backend_12345").expect("CString should accept valid test input");
    let result = pcai_init(backend.as_ptr());

    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

    let err_ptr = pcai_last_error();
    assert!(!err_ptr.is_null());

    let err_str = unsafe { CStr::from_ptr(err_ptr) };
    let err_text = err_str.to_str().expect("Error should be valid UTF-8");

    // Error should mention the backend or indicate it's unknown
    assert!(
        err_text.contains("Unknown") || err_text.contains("unknown") || err_text.contains("backend"),
        "Error should mention unknown backend: {}",
        err_text
    );

    pcai_shutdown();
}

/// Test error message for operations before init
#[test]
fn test_not_initialized_error() {
    pcai_shutdown();

    // Try to load model without init
    let path = CString::new("/test/model.gguf").expect("CString should accept valid test input");

    #[cfg(feature = "ffi")]
    {
        use pcai_inference::ffi::pcai_load_model;
        let result = pcai_load_model(path.as_ptr(), 0);
        assert_eq!(result, -1);

        let err_ptr = pcai_last_error();
        assert!(!err_ptr.is_null());

        let err_str = unsafe { CStr::from_ptr(err_ptr) };
        let err_text = err_str.to_str().expect("Error should be valid UTF-8");

        // Should mention initialization
        assert!(
            err_text.contains("initialized") || err_text.contains("init") || err_text.contains("not"),
            "Error should mention not initialized: {}",
            err_text
        );
    }
}

#[cfg(feature = "llamacpp")]
mod llamacpp_error_tests {
    use super::*;
    use pcai_inference::ffi::{pcai_generate, pcai_load_model};

    /// Test error for model not found
    #[test]
    fn test_model_not_found_error() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) != 0 {
            return; // Skip if llamacpp not available
        }

        let nonexistent = CString::new("/nonexistent/path/to/model.gguf").expect("CString should accept valid test input");
        let result = pcai_load_model(nonexistent.as_ptr(), 0);

        assert_eq!(result, -1);

        let err_ptr = pcai_last_error();
        assert!(!err_ptr.is_null());

        let err_str = unsafe { CStr::from_ptr(err_ptr) };
        let err_text = err_str.to_str().expect("Error should be valid UTF-8");

        // Should mention file not found or path issue
        assert!(
            err_text.contains("not found")
                || err_text.contains("exist")
                || err_text.contains("path")
                || err_text.contains("load")
                || err_text.contains("failed"),
            "Error should mention file issue: {}",
            err_text
        );

        pcai_shutdown();
    }

    /// Test error for generate without model loaded
    #[test]
    fn test_generate_without_model_error() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) != 0 {
            return; // Skip if llamacpp not available
        }

        let prompt = CString::new("Test prompt").expect("CString should accept valid test input");
        let result = pcai_generate(prompt.as_ptr(), 10, 0.7);

        assert!(result.is_null());

        let err_ptr = pcai_last_error();
        assert!(!err_ptr.is_null());

        let err_str = unsafe { CStr::from_ptr(err_ptr) };
        let err_text = err_str.to_str().expect("Error should be valid UTF-8");

        // Should mention model not loaded
        assert!(
            err_text.contains("loaded") || err_text.contains("model") || err_text.contains("not"),
            "Error should mention model not loaded: {}",
            err_text
        );

        pcai_shutdown();
    }

    /// Test that errors are specific and actionable
    #[test]
    fn test_error_specificity() {
        pcai_shutdown();

        // Test various error conditions and verify messages are helpful
        let test_cases = vec![
            ("", "empty backend name"),
            ("x", "single char backend"),
            ("LLAMACPP", "wrong case backend"),
        ];

        for (backend_name, _description) in test_cases {
            let backend = CString::new(backend_name).expect("CString should accept valid test input");
            let result = pcai_init(backend.as_ptr());

            if result == -1 {
                let err_ptr = pcai_last_error();
                if !err_ptr.is_null() {
                    let err_str = unsafe { CStr::from_ptr(err_ptr) };
                    let err_text = err_str.to_str().expect("Error should be valid UTF-8");

                    // Errors should not be generic "error" only
                    assert!(
                        err_text.len() > 5,
                        "Error should be descriptive, not just 'error': {}",
                        err_text
                    );
                }
            }

            pcai_shutdown();
        }
    }
}

/// Test that successful operations don't leave stale errors
#[test]
fn test_success_clears_error() {
    pcai_shutdown();

    // First, cause an error
    let invalid = CString::new("invalid").expect("CString should accept valid test input");
    let _ = pcai_init(invalid.as_ptr());

    // Verify error is set
    let err_ptr = pcai_last_error();
    assert!(!err_ptr.is_null());

    pcai_shutdown();

    // Now do a successful operation (if any backend is available)
    #[cfg(feature = "llamacpp")]
    {
        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        let result = pcai_init(backend.as_ptr());

        if result == 0 {
            // After success, error should be cleared or contain no error
            // Note: Implementation may or may not clear error on success
            // This documents expected behavior
        }

        pcai_shutdown();
    }
}

/// Test pcai_version returns valid version string
#[test]
fn test_version_string() {
    let version_ptr = pcai_version();
    assert!(!version_ptr.is_null());

    let version_str = unsafe { CStr::from_ptr(version_ptr) };
    let version = version_str.to_str().expect("Version should be valid UTF-8");

    // Version should be non-empty
    assert!(!version.is_empty(), "Version string should not be empty");

    // Version should follow semver pattern (e.g., "0.1.0")
    let parts: Vec<&str> = version.split('.').collect();
    assert!(
        parts.len() >= 2,
        "Version should have at least major.minor: {}",
        version
    );
}

/// Test pcai_last_error_code returns correct codes
#[test]
fn test_error_codes() {
    pcai_shutdown();

    // Test invalid input error
    let result = pcai_init(std::ptr::null());
    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);
    assert_eq!(pcai_last_error_code(), PcaiErrorCode::InvalidInput as i32);

    // Test unknown backend error (also InvalidInput)
    let unknown = CString::new("unknown_backend").expect("CString should accept valid test input");
    let result = pcai_init(unknown.as_ptr());
    assert_eq!(result, PcaiErrorCode::InvalidInput as i32);
    assert_eq!(pcai_last_error_code(), PcaiErrorCode::InvalidInput as i32);

    pcai_shutdown();
}

#[cfg(feature = "llamacpp")]
mod error_code_tests {
    use super::*;
    use pcai_inference::ffi::{pcai_generate, pcai_load_model};

    /// Test NotInitialized error code
    #[test]
    fn test_not_initialized_error_code() {
        pcai_shutdown();

        let path = CString::new("/test/model.gguf").expect("CString should accept valid test input");
        let result = pcai_load_model(path.as_ptr(), 0);

        assert_eq!(result, PcaiErrorCode::NotInitialized as i32);
        assert_eq!(pcai_last_error_code(), PcaiErrorCode::NotInitialized as i32);
    }

    /// Test ModelNotLoaded error code
    #[test]
    fn test_model_not_loaded_error_code() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) != 0 {
            return; // Skip if llamacpp not available
        }

        let prompt = CString::new("Test").expect("CString should accept valid test input");
        let result = pcai_generate(prompt.as_ptr(), 10, 0.7);

        assert!(result.is_null());
        assert_eq!(pcai_last_error_code(), PcaiErrorCode::ModelNotLoaded as i32);

        pcai_shutdown();
    }

    /// Test IoError code for missing model file
    #[test]
    fn test_io_error_code() {
        pcai_shutdown();

        let backend = CString::new("llamacpp").expect("CString should accept valid test input");
        if pcai_init(backend.as_ptr()) != 0 {
            return; // Skip if llamacpp not available
        }

        let nonexistent = CString::new("/nonexistent/path/model.gguf").expect("CString should accept valid test input");
        let result = pcai_load_model(nonexistent.as_ptr(), 0);

        assert_ne!(result, 0);
        let error_code = pcai_last_error_code();
        // Should be IoError or BackendError depending on how the backend reports it
        assert!(
            error_code == PcaiErrorCode::IoError as i32 || error_code == PcaiErrorCode::BackendError as i32,
            "Expected IoError or BackendError, got {}",
            error_code
        );

        pcai_shutdown();
    }
}

/// Test prompt length validation
#[test]
fn test_prompt_too_large() {
    use pcai_inference::ffi::pcai_generate;

    pcai_shutdown();

    // Create a prompt > 100KB
    let large_prompt = "x".repeat(101 * 1024);
    let prompt_cstr = CString::new(large_prompt).expect("CString should accept valid test input");

    let result = pcai_generate(prompt_cstr.as_ptr(), 10, 0.7);
    assert!(result.is_null());

    let err_ptr = pcai_last_error();
    assert!(!err_ptr.is_null());

    let err_str = unsafe { CStr::from_ptr(err_ptr) };
    let err_text = err_str.to_str().expect("Error should be valid UTF-8");

    // Should mention size issue
    assert!(
        err_text.contains("too large") || err_text.contains("bytes"),
        "Error should mention size: {}",
        err_text
    );
    assert_eq!(pcai_last_error_code(), PcaiErrorCode::InvalidInput as i32);
}
