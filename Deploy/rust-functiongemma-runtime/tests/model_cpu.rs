#![cfg(all(feature = "model", not(feature = "cuda")))]

use candle_core::{Device, Tensor};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

struct ModelFixture(PathBuf);

impl ModelFixture {
    fn new() -> anyhow::Result<Self> {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!("pcai-model-cpu-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)?;
        let fixture = Self(path);
        fixture.write_model()?;
        Ok(fixture)
    }

    fn write_model(&self) -> anyhow::Result<()> {
        write_json(
            &self.0.join("config.json"),
            json!({
                "hidden_size": 2,
                "intermediate_size": 4,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 2,
                "vocab_size": 3,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
            }),
        )?;
        write_json(
            &self.0.join("tokenizer.json"),
            json!({
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "unk_token": "[UNK]",
                    "vocab": {"[UNK]": 0, "question": 1, "CPU_PROOF": 2},
                },
            }),
        )?;
        fs::write(self.0.join("chat_template.jinja"), "{{ messages[0].content }}")?;

        // The tied output projection gives token 2 the highest logit after "question".
        let mut weights = HashMap::from([
            (
                "model.embed_tokens.weight",
                Tensor::new(&[[0f32, 0.], [1., 0.], [2., 0.]], &Device::Cpu)?,
            ),
            ("model.norm.weight", Tensor::new(&[1f32, 1.], &Device::Cpu)?),
        ]);
        let identity = Tensor::new(&[[1f32, 0.], [0., 1.]], &Device::Cpu)?;
        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ] {
            weights.insert(name, identity.clone());
        }
        for name in [
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_feedforward_layernorm.weight",
        ] {
            weights.insert(name, Tensor::new(&[1f32, 1.], &Device::Cpu)?);
        }
        weights.insert(
            "model.layers.0.mlp.gate_proj.weight",
            Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu)?,
        );
        weights.insert(
            "model.layers.0.mlp.up_proj.weight",
            Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu)?,
        );
        weights.insert(
            "model.layers.0.mlp.down_proj.weight",
            Tensor::zeros((2, 4), candle_core::DType::F32, &Device::Cpu)?,
        );
        candle_core::safetensors::save(&weights, self.0.join("model.safetensors"))?;
        Ok(())
    }
}

impl Drop for ModelFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn write_json(path: &Path, data: Value) -> anyhow::Result<()> {
    fs::write(path, serde_json::to_vec(&data)?)?;
    Ok(())
}

// Protects: a CPU model build decodes a token through a loaded attention layer and the chat handler.
// Detects: empty cache attention, silent heuristic fallback, wrong CPU telemetry, or CUDA activation.
// Needs: the model feature without CUDA, synthetic one-layer weights, and loopback TCP.
// Breadcrumb: handlers::chat, inference::infer_with_model, and core Model::generate_with_prealloc_cache.
#[tokio::test]
async fn cpu_model_route_generates_a_token_instead_of_falling_back() -> anyhow::Result<()> {
    let fixture = ModelFixture::new()?;
    let router_config = fixture.0.join("router.json");
    let fixture_authorization = "local-fixture";
    write_json(
        &router_config,
        json!({"runtime": {
            "router_engine": "model",
            "router_model_path": fixture.0,
            "router_device": "cuda:1",
            "router_gpu": 1,
            "router_flash_attn": true,
            "router_kv_cache": true,
            "router_kv_cache_store": "device",
            "router_kv_cache_max_len": 4,
            "api_key": fixture_authorization,
        }}),
    )?;
    rust_functiongemma_runtime::init_runtime_config(&router_config);
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    let addr = listener.local_addr()?;
    let server = tokio::spawn(async move { axum::serve(listener, rust_functiongemma_runtime::build_router()).await });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(10))
        .build()?;

    let health: Value = client
        .get(format!("http://{addr}/health"))
        .bearer_auth(fixture_authorization)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(health["metadata"]["engine"], "model");
    assert_eq!(health["metadata"]["device"], "cpu");

    let response = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .bearer_auth(fixture_authorization)
        .json(&json!({
            "messages": [{"role": "user", "content": "question"}],
            "max_tokens": 1,
        }))
        .send()
        .await?
        .error_for_status()?;
    assert_eq!(response.headers()["x-pcai-router-engine"], "model");
    let response: Value = response.json().await?;
    assert_eq!(response["choices"][0]["message"]["content"], "CPU_PROOF");
    server.abort();
    Ok(())
}
