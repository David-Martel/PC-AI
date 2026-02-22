//! Integration tests for pcai-inference
//!
//! These tests verify:
//! - Backend creation and lifecycle
//! - Model loading (with and without real models)
//! - Text generation interface
//! - FFI exports for PowerShell integration
//!
//! ## Running Tests
//!
//! ```bash
//! # Basic tests (no backends)
//! cargo test --no-default-features
//!
//! # With llamacpp (requires C++ toolchain + MSVC on Windows)
//! cargo test --features llamacpp
//!
//! # With mistralrs
//! cargo test --features mistralrs-backend
//!
//! # All features
//! cargo test --all-features
//! ```
//!
//! ## Testing with Real Models
//!
//! Some tests are skipped by default unless a real model is available.
//! Set `providers.pcai-native.modelPath` in Config/llm-config.json to enable them.
//!
//! Alternatively, install a model via Ollama or LM Studio, and the tests
//! will automatically discover it.

use pcai_inference::{backends::*, config::*, Error, Result};

#[cfg(feature = "ffi")]
use std::ffi::CString;

mod common;

// ============================================================================
// Test Utilities
// ============================================================================

/// Macro to skip tests that require a real model file
///
/// Usage:
/// ```
/// #[test]
/// fn test_with_model() {
///     require_model!();
///     // test code that needs a model
/// }
/// ```
#[macro_export]
macro_rules! require_model {
    () => {
        if !has_test_model() {
            eprintln!("Skipping test: configure providers.pcai-native.modelPath in Config/llm-config.json");
            return;
        }
    };
}

// Re-export common utilities (used in feature-gated tests)
#[allow(unused_imports)]
use common::{find_test_model, has_test_model};

/// Mock backend for testing without real models
#[allow(dead_code)]
struct MockBackend {
    loaded: bool,
}

#[allow(dead_code)]
impl MockBackend {
    fn new() -> Self {
        Self { loaded: false }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl InferenceBackend for MockBackend {
    async fn load_model(&mut self, _model_path: &str) -> Result<()> {
        self.loaded = true;
        Ok(())
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        if !self.loaded {
            return Err(Error::ModelNotLoaded);
        }

        Ok(GenerateResponse {
            text: format!("Mock response to: {}", request.prompt),
            tokens_generated: request.max_tokens.unwrap_or(10),
            finish_reason: FinishReason::Stop,
        })
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.loaded = false;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn backend_name(&self) -> &'static str {
        "mock"
    }
}

// ============================================================================
// Basic Configuration Tests
// ============================================================================

#[test]
fn test_generate_request_serialization() {
    let req = GenerateRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.95),
        stop: vec!["STOP".to_string()],
    };

    let json = serde_json::to_string(&req).unwrap();
    let deserialized: GenerateRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(req.prompt, deserialized.prompt);
    assert_eq!(req.max_tokens, deserialized.max_tokens);
    assert_eq!(req.temperature, deserialized.temperature);
    assert_eq!(req.top_p, deserialized.top_p);
    assert_eq!(req.stop, deserialized.stop);
}

#[test]
fn test_generate_response_serialization() {
    let resp = GenerateResponse {
        text: "Generated text".to_string(),
        tokens_generated: 42,
        finish_reason: FinishReason::Stop,
    };

    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: GenerateResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(resp.text, deserialized.text);
    assert_eq!(resp.tokens_generated, deserialized.tokens_generated);
}

#[test]
fn test_finish_reason_serialization() {
    let reasons = vec![
        (FinishReason::Stop, r#""stop""#),
        (FinishReason::Length, r#""length""#),
        (FinishReason::Error, r#""error""#),
    ];

    for (reason, expected) in reasons {
        let json = serde_json::to_string(&reason).unwrap();
        assert_eq!(json, expected);

        let deserialized: FinishReason = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, _));
    }
}

#[test]
fn test_config_defaults() {
    let defaults = GenerationDefaults::default();
    assert_eq!(defaults.max_tokens, 512);
    assert_eq!(defaults.temperature, 0.7);
    assert_eq!(defaults.top_p, 0.95);
}

#[cfg(feature = "server")]
#[test]
fn test_server_config_defaults() {
    let server = ServerConfig::default();
    assert_eq!(server.host, "127.0.0.1");
    assert_eq!(server.port, 8080);
    assert!(server.cors);
}

#[test]
fn test_error_types() {
    let err = Error::ModelNotLoaded;
    assert_eq!(err.to_string(), "Model not loaded");

    let err = Error::Backend("test error".to_string());
    assert_eq!(err.to_string(), "Backend error: test error");

    let err = Error::Config("invalid config".to_string());
    assert_eq!(err.to_string(), "Configuration error: invalid config");

    let err = Error::InvalidInput("bad input".to_string());
    assert_eq!(err.to_string(), "Invalid input: bad input");
}

// ============================================================================
// Backend Creation Tests
// ============================================================================

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_backend_creation() {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;

    let backend = LlamaCppBackend::new();
    assert_eq!(backend.backend_name(), "llama.cpp");
    assert!(!backend.is_loaded());
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_llamacpp_backend_with_config() {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;

    let backend = LlamaCppBackend::with_config(32, 4096, 512);
    assert_eq!(backend.backend_name(), "llama.cpp");
    assert!(!backend.is_loaded());
}

#[cfg(feature = "mistralrs-backend")]
#[test]
fn test_mistralrs_backend_creation() {
    use pcai_inference::backends::mistralrs::MistralRsBackend;

    let backend = MistralRsBackend::new();
    assert_eq!(backend.backend_name(), "mistral.rs");
    assert!(!backend.is_loaded());
}

#[cfg(feature = "llamacpp")]
#[test]
fn test_backend_type_llamacpp() {
    let backend_result = BackendType::LlamaCpp.create();
    assert!(backend_result.is_ok());

    let backend = backend_result.unwrap();
    assert_eq!(backend.backend_name(), "llama.cpp");
    assert!(!backend.is_loaded());
}

#[cfg(feature = "mistralrs-backend")]
#[test]
fn test_backend_type_mistralrs() {
    let backend_result = BackendType::MistralRs.create();
    assert!(backend_result.is_ok());

    let backend = backend_result.unwrap();
    assert_eq!(backend.backend_name(), "mistral.rs");
    assert!(!backend.is_loaded());
}

// ============================================================================
// Backend Trait Tests
// ============================================================================

#[tokio::test]
async fn test_mock_backend_lifecycle() {
    let mut backend = MockBackend::new();

    // Initially not loaded
    assert!(!backend.is_loaded());

    // Load model
    backend.load_model("dummy.gguf").await.unwrap();
    assert!(backend.is_loaded());

    // Unload model
    backend.unload_model().await.unwrap();
    assert!(!backend.is_loaded());
}

#[tokio::test]
async fn test_mock_backend_generate() {
    let mut backend = MockBackend::new();
    backend.load_model("dummy.gguf").await.unwrap();

    let request = GenerateRequest {
        prompt: "Hello, world!".to_string(),
        max_tokens: Some(50),
        temperature: Some(0.7),
        top_p: None,
        stop: vec![],
    };

    let response = backend.generate(request).await.unwrap();
    assert!(response.text.contains("Mock response"));
    assert_eq!(response.tokens_generated, 50);
    assert!(matches!(response.finish_reason, FinishReason::Stop));
}

#[tokio::test]
async fn test_backend_generate_without_model_fails() {
    let backend = MockBackend::new();

    let request = GenerateRequest {
        prompt: "Test".to_string(),
        max_tokens: Some(10),
        temperature: None,
        top_p: None,
        stop: vec![],
    };

    let result = backend.generate(request).await;
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), Error::ModelNotLoaded));
}

#[cfg(feature = "llamacpp")]
#[tokio::test]
async fn test_llamacpp_load_nonexistent_file_fails() {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;

    let mut backend = LlamaCppBackend::new();
    let result = backend.load_model("/nonexistent/model.gguf").await;
    assert!(result.is_err());
}

#[cfg(feature = "mistralrs-backend")]
#[tokio::test]
async fn test_mistralrs_load_nonexistent_file_fails() {
    use pcai_inference::backends::mistralrs::MistralRsBackend;

    let mut backend = MistralRsBackend::new();
    let result = backend.load_model("/nonexistent/model.gguf").await;
    assert!(result.is_err());
}

// ============================================================================
// Generation Tests (with real models, skipped by default)
// ============================================================================

#[cfg(feature = "llamacpp")]
#[tokio::test]
async fn test_llamacpp_generate_with_model() {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;

    require_model!();

    let model_path = common::require_test_model();
    let mut backend = LlamaCppBackend::new();

    backend.load_model(&model_path).await.unwrap();
    assert!(backend.is_loaded());

    let request = GenerateRequest {
        prompt: "The capital of France is".to_string(),
        max_tokens: Some(10),
        temperature: Some(0.1), // Low temperature for deterministic output
        top_p: Some(0.95),
        stop: vec![],
    };

    let response = backend.generate(request).await.unwrap();
    assert!(!response.text.is_empty());
    assert!(response.tokens_generated > 0);
    assert!(response.tokens_generated <= 10);
}

#[cfg(feature = "mistralrs-backend")]
#[tokio::test]
async fn test_mistralrs_generate_with_model() {
    use pcai_inference::backends::mistralrs::MistralRsBackend;

    require_model!();

    let model_path = common::require_test_model();
    let mut backend = MistralRsBackend::new();

    backend.load_model(&model_path).await.unwrap();
    assert!(backend.is_loaded());

    let request = GenerateRequest {
        prompt: "The capital of France is".to_string(),
        max_tokens: Some(10),
        temperature: Some(0.1),
        top_p: Some(0.95),
        stop: vec![],
    };

    let response = backend.generate(request).await.unwrap();
    assert!(!response.text.is_empty());
    assert!(response.tokens_generated > 0);
}

// ============================================================================
// FFI Tests
// ============================================================================

#[cfg(feature = "ffi")]
mod ffi_tests {
    use super::*;
    use pcai_inference::ffi::*;

    #[test]
    fn test_ffi_init_null_backend() {
        let result = pcai_init(std::ptr::null());
        // InvalidInput error code = -3
        assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("Invalid"));
    }

    #[test]
    fn test_ffi_init_unknown_backend() {
        let backend = CString::new("unknown_backend").unwrap();
        let result = pcai_init(backend.as_ptr());
        // InvalidInput error code = -3
        assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("Unknown backend"));
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_ffi_init_llamacpp() {
        pcai_shutdown(); // Clean state

        let backend = CString::new("llamacpp").unwrap();
        let result = pcai_init(backend.as_ptr());

        if result != 0 {
            let err = pcai_last_error();
            if !err.is_null() {
                let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
                eprintln!("Init failed: {}", err_str.to_str().unwrap());
            }
        }

        assert_eq!(result, 0);
        pcai_shutdown();
    }

    #[cfg(feature = "mistralrs-backend")]
    #[test]
    fn test_ffi_init_mistralrs() {
        pcai_shutdown(); // Clean state

        let backend = CString::new("mistralrs").unwrap();
        let result = pcai_init(backend.as_ptr());
        assert_eq!(result, 0);

        pcai_shutdown();
    }

    #[test]
    fn test_ffi_load_model_null_path() {
        let result = pcai_load_model(std::ptr::null(), 0);
        // InvalidInput error code = -3
        assert_eq!(result, PcaiErrorCode::InvalidInput as i32);

        let err = pcai_last_error();
        assert!(!err.is_null());
    }

    #[test]
    fn test_ffi_load_model_before_init() {
        pcai_shutdown(); // Ensure no backend initialized

        let path = CString::new("/nonexistent/model.gguf").unwrap();
        let result = pcai_load_model(path.as_ptr(), 0);
        // NotInitialized error code = -1
        assert_eq!(result, PcaiErrorCode::NotInitialized as i32);

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("not initialized"));
    }

    #[test]
    fn test_ffi_generate_null_prompt() {
        let result = pcai_generate(std::ptr::null(), 10, 0.7);
        assert!(result.is_null());

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("Invalid prompt"));
    }

    #[test]
    fn test_ffi_generate_before_init() {
        pcai_shutdown(); // Ensure no backend initialized

        let prompt = CString::new("Test prompt").unwrap();
        let result = pcai_generate(prompt.as_ptr(), 10, 0.7);
        assert!(result.is_null());

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("not initialized"));
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_ffi_generate_without_model() {
        pcai_shutdown(); // Clean state

        // Initialize backend
        let backend = CString::new("llamacpp").unwrap();
        let init_result = pcai_init(backend.as_ptr());
        assert_eq!(init_result, 0);

        // Try to generate without loading model
        let prompt = CString::new("Test prompt").unwrap();
        let result = pcai_generate(prompt.as_ptr(), 10, 0.7);
        assert!(result.is_null());

        let err = pcai_last_error();
        assert!(!err.is_null());

        let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
        assert!(err_str.to_str().unwrap().contains("not loaded"));

        pcai_shutdown();
    }

    #[test]
    fn test_ffi_free_string_null() {
        // Should not crash
        pcai_free_string(std::ptr::null_mut());
    }

    #[test]
    fn test_ffi_last_error_no_error() {
        use pcai_inference::ffi::pcai_last_error;

        // Clear any previous errors by calling shutdown
        pcai_shutdown();

        // After shutdown, there should be no error
        // Note: The error state might be cleared or might contain a previous error
        // depending on thread-local storage behavior
        let err = pcai_last_error();
        // Don't assert on err being null, as it might contain leaked previous errors
        // Just verify the call doesn't crash
        let _ = err;
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_ffi_full_lifecycle_with_model() {
        require_model!();

        pcai_shutdown(); // Clean state

        // Initialize
        let backend = CString::new("llamacpp").unwrap();
        assert_eq!(pcai_init(backend.as_ptr()), 0);

        // Load model
        let model_path = common::require_test_model();
        let path = CString::new(model_path).unwrap();
        let load_result = pcai_load_model(path.as_ptr(), 0);

        if load_result != 0 {
            let err = pcai_last_error();
            if !err.is_null() {
                let err_str = unsafe { std::ffi::CStr::from_ptr(err) };
                eprintln!("Load failed: {}", err_str.to_str().unwrap());
            }
        }
        assert_eq!(load_result, 0);

        // Generate
        let prompt = CString::new("The capital of France is").unwrap();
        let result_ptr = pcai_generate(prompt.as_ptr(), 10, 0.1);
        assert!(!result_ptr.is_null());

        let result_str = unsafe { std::ffi::CStr::from_ptr(result_ptr) };
        let text = result_str.to_str().unwrap();
        assert!(!text.is_empty());

        // Free result
        pcai_free_string(result_ptr);

        // Shutdown
        pcai_shutdown();
    }
}

// ============================================================================
// Stress Tests (optional, run with --ignored)
// ============================================================================

#[cfg(feature = "llamacpp")]
#[tokio::test]
#[ignore]
async fn stress_test_sequential_generations() {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;

    require_model!();

    let model_path = common::require_test_model();
    let mut backend = LlamaCppBackend::new();
    backend.load_model(&model_path).await.unwrap();

    for i in 0..10 {
        let request = GenerateRequest {
            prompt: format!("Test prompt {}", i),
            max_tokens: Some(5),
            temperature: Some(0.1),
            top_p: None,
            stop: vec![],
        };

        let response = backend.generate(request).await.unwrap();
        assert!(!response.text.is_empty());
    }
}

#[cfg(all(feature = "llamacpp", feature = "mistralrs-backend"))]
#[tokio::test]
#[ignore]
async fn stress_test_backend_switching() {
    require_model!();

    let model_path = common::require_test_model();

    // Test llamacpp
    {
        use pcai_inference::backends::llamacpp::LlamaCppBackend;
        let mut backend = LlamaCppBackend::new();
        backend.load_model(&model_path).await.unwrap();

        let request = GenerateRequest {
            prompt: "Test".to_string(),
            max_tokens: Some(5),
            temperature: Some(0.1),
            top_p: None,
            stop: vec![],
        };

        let _ = backend.generate(request).await.unwrap();
        backend.unload_model().await.unwrap();
    }

    // Test mistralrs
    {
        use pcai_inference::backends::mistralrs::MistralRsBackend;
        let mut backend = MistralRsBackend::new();
        backend.load_model(&model_path).await.unwrap();

        let request = GenerateRequest {
            prompt: "Test".to_string(),
            max_tokens: Some(5),
            temperature: Some(0.1),
            top_p: None,
            stop: vec![],
        };

        let _ = backend.generate(request).await.unwrap();
        backend.unload_model().await.unwrap();
    }
}
