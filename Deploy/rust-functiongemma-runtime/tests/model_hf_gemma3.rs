#![cfg(all(feature = "model", not(feature = "cuda")))]

use serde_json::{json, Value};
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// Protects: an explicitly selected Gemma3 profile emits a real publisher tool call on CPU.
// Detects: checkpoint math drift, fallback confused with model execution, and cache leakage between requests.
// Needs: PC-AI's model feature, loopback, and an installed Google FunctionGemma 270M checkpoint.
// Breadcrumb: inference::hf_gemma3, inference::infer_with_model, and the runtime chat handler.
#[tokio::test]
#[ignore = "requires PCAI_TEST_FUNCTIONGEMMA_MODEL_DIR pointing at the local publisher checkpoint"]
async fn opt_in_publisher_profile_returns_weather_tool_after_overlimit_request() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var_os("PCAI_TEST_FUNCTIONGEMMA_MODEL_DIR").ok_or_else(|| {
        anyhow::anyhow!("PCAI_TEST_FUNCTIONGEMMA_MODEL_DIR must point to a local FunctionGemma checkpoint")
    })?);
    anyhow::ensure!(
        model_dir.is_absolute() && model_dir.join("model.safetensors").is_file(),
        "expected absolute local FunctionGemma checkpoint"
    );

    let config = std::env::temp_dir().join(format!("pcai-gemma3-profile-{}.json", std::process::id()));
    let api_key = format!("local-profile-test-{}", std::process::id());
    let result = async {
        let settings = json!({
            "runtime": {
                "router_engine": "model",
                "router_device": "cpu",
                "router_model_path": model_dir,
                "router_model_profile": "hf_gemma3_4_57_3",
                "router_max_seq_len": 256,
                "router_request_timeout_secs": 180,
                "api_key": api_key
            }
        });
        std::fs::write(&config, serde_json::to_vec(&settings)?)?;
        rust_functiongemma_runtime::init_runtime_config(&config);
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let address = listener.local_addr()?;
        let server = tokio::spawn(async move { axum::serve(listener, rust_functiongemma_runtime::build_router()).await });
        let calls = async {
            let client = reqwest::Client::builder().no_proxy().timeout(Duration::from_secs(180)).build()?;
            let url = format!("http://{address}/v1/chat/completions");
            let mut request = json!({
                "messages": [
                    {"role":"developer","content":"You are a model that can do function calling with the following functions"},
                    {"role":"user","content":"What's the temperature in London?"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_current_temperature",
                        "description": "Gets the current temperature for a given location.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string", "description": "The city name, e.g. San Francisco"}},
                            "required": ["location"]
                        }
                    }
                }],
                "max_tokens": 512
            });

            let rejected = client.post(&url).bearer_auth(&api_key).json(&request).send().await?.error_for_status()?;
            assert_eq!(rejected.headers()["x-pcai-router-engine"], "heuristic-fallback");

            request["max_tokens"] = 24.into();
            for attempt in 1..=2 {
                let started = Instant::now();
                let response = client.post(&url).bearer_auth(&api_key).json(&request).send().await?.error_for_status()?;
                assert_eq!(response.headers()["x-pcai-router-engine"], "model");
                let body: Value = response.json().await?;
                let call = &body["choices"][0]["message"]["tool_calls"][0]["function"];
                assert_eq!(call["name"], "get_current_temperature");
                assert_eq!(call["arguments"], json!({"location": "London"}));
                eprintln!("weather request {attempt}: model tool call in {} ms", started.elapsed().as_millis());
            }
            Ok(())
        }.await;
        server.abort();
        calls
    }.await;
    let _ = std::fs::remove_file(&config);
    result
}
