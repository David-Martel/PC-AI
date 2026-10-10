use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

fn normalized_config(json: &str) -> RootConfig {
    let mut config: RootConfig = serde_json::from_str(json).expect("valid literal config");
    config.ollama.apply_defaults(&config.providers);
    config
}

fn wire_request(config: &RootConfig) -> Value {
    let request: ChatRequest = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":"inert fixture"}],"numCtx":4096,"temperature":0.25,"maxTokens":7,"numThread":2}"#,
    )
    .expect("valid literal request");
    let messages = convert_messages(request.messages.clone());
    let prompt_chars = messages.iter().map(|message| message.content.len()).sum();
    let chat = ChatMessageRequest::new("inert-model".to_string(), messages)
        .options(build_options(config, &request, prompt_chars));
    serde_json::to_value(apply_keep_alive(chat, config.ollama.keep_alive_seconds))
        .expect("serialize actual ollama-rs request")
}

fn load_literal_fixture(json: &str) -> RootConfig {
    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);
    let directory = std::env::temp_dir().join(format!(
        "pcai-ollama-keepalive-owned-{}-{}",
        std::process::id(),
        NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed),
    ));
    // Create-only ownership; an existing directory is a failure, never reused.
    fs::create_dir(&directory).expect("create owned fixture directory");
    let path = directory.join("config.json");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .expect("create owned literal config");
    file.write_all(json.as_bytes()).expect("write literal config");
    drop(file);
    let loaded = load_config(&path);
    // Retain the failed fixture and its original parser error for diagnosis.
    if loaded.is_ok() {
        fs::remove_file(&path).expect("remove owned successful fixture");
        fs::remove_dir(&directory).expect("remove owned empty fixture directory");
    }
    loaded.expect("load actual maintained config from literal bytes")
}

// Protects: missing keep-alive configuration retains its existing 30-minute policy.
// Detects: deleting the zero override without supplying a real serde default.
// Needs: actual RootConfig deserialization and apply_defaults, without a server.
// Breadcrumb: omitted field and omitted section are distinct default paths.
#[test]
fn missing_keep_alive_field_and_section_default_to_1800() {
    for json in [r#"{"ollama":{}}"#, r#"{}"#] {
        let config = normalized_config(json);
        assert_eq!(config.ollama.keep_alive_seconds, 1800);
        assert_eq!(wire_request(&config)["keep_alive"], serde_json::json!("1800s"));
    }
}

// Protects: explicit zero means unload on completion, including repeated normalization.
// Detects: apply_defaults silently replacing a valid zero with 1800.
// Needs: actual configuration normalization; no unload or model call is performed.
// Breadcrumb: production apply_keep_alive already supports numeric zero.
#[test]
fn explicit_zero_survives_repeated_defaults_and_serializes_as_number() {
    let mut config = normalized_config(r#"{"ollama":{"keep_alive_seconds":0}}"#);
    config.ollama.apply_defaults(&config.providers);
    assert_eq!(config.ollama.keep_alive_seconds, 0);
    assert_eq!(wire_request(&config)["keep_alive"], serde_json::json!(0));
    assert!(wire_request(&config)["keep_alive"].is_number());
}

// Protects: positive configured durations keep their exact seconds value.
// Detects: an unconditional default or unit/type change in the actual request.
// Needs: real ollama-rs serialization, including the existing duration string dialect.
// Breadcrumb: TimeUnit::Seconds serializes as the s suffix in locked ollama-rs 0.3.6.
#[test]
fn positive_keep_alive_preserves_exact_duration_string() {
    for (seconds, expected) in [(1, "1s"), (91, "91s"), (1800, "1800s")] {
        let json = format!(r#"{{"ollama":{{"keep_alive_seconds":{seconds}}}}}"#);
        let config = normalized_config(&json);
        assert_eq!(config.ollama.keep_alive_seconds, seconds);
        assert_eq!(wire_request(&config)["keep_alive"], serde_json::json!(expected));
    }
}

// Protects: existing negative-duration behavior remains indefinite.
// Detects: narrowing the old all-negative branch to only -1 or changing its wire type.
// Needs: actual apply_keep_alive plus locked dependency serialization, with no HTTP.
// Breadcrumb: the production match covers i64::MIN through -1.
#[test]
fn negative_keep_alive_retains_existing_indefinite_behavior() {
    for seconds in [-1, -19, i64::MIN] {
        let json = format!(r#"{{"ollama":{{"keep_alive_seconds":{seconds}}}}}"#);
        let config = normalized_config(&json);
        assert_eq!(config.ollama.keep_alive_seconds, seconds);
        assert_eq!(wire_request(&config)["keep_alive"], serde_json::json!(-1));
    }
}

// Protects: Default and deserialized empty configurations agree without changing other fields.
// Detects: fixing only the field-level serde path and leaving omitted sections or Default at zero.
// Needs: actual Default implementations, deserialization and maintained normalization.
// Breadcrumb: RootConfig derives Default and its omitted ollama section uses OllamaConfig::default.
#[test]
fn default_struct_and_empty_json_have_equivalent_configuration() {
    let default_ollama = OllamaConfig::default();
    let parsed_ollama: OllamaConfig = serde_json::from_str("{}").expect("empty ollama config");
    assert_eq!(default_ollama.keep_alive_seconds, 1800);
    assert_eq!(format!("{default_ollama:?}"), format!("{parsed_ollama:?}"));
    let mut default_root = RootConfig::default();
    default_root.ollama.apply_defaults(&default_root.providers);
    assert_eq!(format!("{default_root:?}"), format!("{:?}", normalized_config("{}")));
}

// Protects: the maintained file-loading path keeps omitted, zero and positive semantics.
// Detects: tests checking only a copied parser or bypassing load_config normalization.
// Needs: tiny literal files in a separately admitted private TEMP and actual load_config.
// Breadcrumb: successful create-only fixtures are removed; failed fixtures are retained.
#[test]
fn load_config_preserves_missing_zero_and_positive_from_real_files() {
    for (json, expected) in [
        (r#"{}"#, 1800),
        (r#"{"ollama":{}}"#, 1800),
        (r#"{"ollama":{"keep_alive_seconds":0}}"#, 0),
        (r#"{"ollama":{"keep_alive_seconds":91}}"#, 91),
    ] {
        let config = load_literal_fixture(json);
        assert_eq!(config.ollama.keep_alive_seconds, expected);
    }
}

// Protects: changing keep-alive does not change options, messages or request model.
// Detects: accidental request reconstruction or loss of existing explicit option values.
// Needs: convert_messages, build_options and actual ChatMessageRequest serialization.
// Breadcrumb: this checks the same construction used by run_chat before any HTTP call.
#[test]
fn keep_alive_change_preserves_actual_request_options_and_messages() {
    let zero = wire_request(&normalized_config(r#"{"ollama":{"keep_alive_seconds":0}}"#));
    let positive = wire_request(&normalized_config(r#"{"ollama":{"keep_alive_seconds":91}}"#));
    assert_eq!(zero["options"], positive["options"]);
    assert_eq!(zero["options"]["num_ctx"], serde_json::json!(4096));
    assert_eq!(zero["options"]["temperature"], serde_json::json!(0.25));
    assert_eq!(zero["options"]["num_predict"], serde_json::json!(7));
    assert_eq!(zero["options"]["num_thread"], serde_json::json!(2));
    assert_eq!(zero["model"], serde_json::json!("inert-model"));
    assert_eq!(zero["messages"], serde_json::json!([{"role":"user","content":"inert fixture","tool_calls":[],"thinking":null}]));
    assert_eq!(zero["stream"], serde_json::json!(false));
    assert_eq!(zero["keep_alive"], serde_json::json!(0));
    assert_eq!(positive["keep_alive"], serde_json::json!("91s"));
}

// Protects: malformed keep-alive values still fail serde parsing instead of receiving a fallback.
// Detects: adding broad coercion or hiding errors as part of the zero repair.
// Needs: the actual typed RootConfig parser; no error text or implementation-mirroring parser.
// Breadcrumb: the field remains i64 and null/string are not missing fields.
#[test]
fn invalid_keep_alive_types_remain_rejected() {
    for json in [
        r#"{"ollama":{"keep_alive_seconds":null}}"#,
        r#"{"ollama":{"keep_alive_seconds":"0"}}"#,
        r#"{"ollama":{"keep_alive_seconds":true}}"#,
    ] {
        assert!(serde_json::from_str::<RootConfig>(json).is_err());
    }
}
