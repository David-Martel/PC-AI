//! Unit tests for backend selection logic

use pcai_inference::backends::{BackendType, FinishReason, GenerateRequest, GenerateResponse};

#[test]
fn test_generate_request_defaults() {
    let request = GenerateRequest {
        prompt: "test".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: vec![],
    };

    assert_eq!(request.prompt, "test");
    assert!(request.max_tokens.is_none());
    assert!(request.temperature.is_none());
}

#[test]
fn test_generate_request_with_params() {
    let request = GenerateRequest {
        prompt: "Hello".to_string(),
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: Some(0.9),
        stop: vec!["END".to_string()],
    };

    assert_eq!(request.max_tokens, Some(100));
    assert_eq!(request.temperature, Some(0.7));
    assert_eq!(request.stop.len(), 1);
}

#[test]
fn test_finish_reason_serialization() {
    let reasons = vec![
        (FinishReason::Stop, "stop"),
        (FinishReason::Length, "length"),
        (FinishReason::Error, "error"),
    ];

    for (reason, expected) in reasons {
        let json = serde_json::to_string(&reason).expect("TODO: Verify unwrap");
        assert!(json.contains(expected), "Should serialize to {}", expected);
    }
}

#[test]
fn test_generate_response_creation() {
    let response = GenerateResponse {
        text: "Hello world".to_string(),
        tokens_generated: 2,
        finish_reason: FinishReason::Stop,
    };

    assert_eq!(response.text, "Hello world");
    assert_eq!(response.tokens_generated, 2);
    matches!(response.finish_reason, FinishReason::Stop);
}

#[cfg(feature = "llamacpp")]
mod llamacpp_tests {
    use pcai_inference::backends::llamacpp::LlamaCppBackend;
    use pcai_inference::backends::InferenceBackend;

    #[test]
    fn test_backend_creation() {
        let backend = LlamaCppBackend::new();
        assert_eq!(backend.backend_name(), "llama.cpp");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_backend_with_config() {
        let backend = LlamaCppBackend::with_config(32, 4096, 512);
        assert_eq!(backend.backend_name(), "llama.cpp");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_backend_type_creation() {
        use pcai_inference::backends::BackendType;

        let backend = BackendType::LlamaCpp.create();
        assert!(backend.is_ok(), "Should create llamacpp backend");
        assert_eq!(backend.unwrap().backend_name(), "llama.cpp");
    }
}

#[cfg(feature = "mistralrs-backend")]
mod mistralrs_tests {
    use pcai_inference::backends::mistralrs::MistralRsBackend;
    use pcai_inference::backends::InferenceBackend;

    #[test]
    fn test_backend_creation() {
        let backend = MistralRsBackend::new();
        assert_eq!(backend.backend_name(), "mistral.rs");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_backend_type_creation() {
        use pcai_inference::backends::BackendType;

        let backend = BackendType::MistralRs.create();
        assert!(backend.is_ok(), "Should create mistralrs backend");
        assert_eq!(backend.unwrap().backend_name(), "mistral.rs");
    }
}

#[test]
fn test_backend_trait_object_safety() {
    // Verify that InferenceBackend can be used as trait object
    use pcai_inference::backends::InferenceBackend;

    fn assert_object_safe(_: &dyn InferenceBackend) {}

    // This test passes if it compiles
}

#[cfg(all(feature = "llamacpp", feature = "mistralrs-backend"))]
#[test]
fn test_backend_switching() {
    use pcai_inference::backends::BackendType;

    let llamacpp = BackendType::LlamaCpp.create().expect("TODO: Verify unwrap");
    let mistralrs = BackendType::MistralRs.create().expect("TODO: Verify unwrap");

    assert_eq!(llamacpp.backend_name(), "llama.cpp");
    assert_eq!(mistralrs.backend_name(), "mistral.rs");
    assert_ne!(llamacpp.backend_name(), mistralrs.backend_name());
}
