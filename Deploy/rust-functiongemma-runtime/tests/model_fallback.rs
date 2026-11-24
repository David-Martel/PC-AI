use reqwest::StatusCode;
use serde_json::{json, Value};
use std::fs;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Protects: callers can distinguish successful requests served by the backup router.
// Detects: hidden fallback, lost streaming headers, and double counting in /health.
// Needs: one temporary config and loopback TCP. The model directory has no weights.
// Breadcrumb: handlers::route_request and metrics::router_stats.
#[tokio::test]
async fn failed_model_requests_report_fallback_for_json_and_sse() -> anyhow::Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let fixture = std::env::temp_dir().join(format!("pcai-model-fallback-{}-{nonce}", std::process::id()));
    fs::create_dir(&fixture)?;
    let config_path = fixture.join("router.json");
    fs::write(
        &config_path,
        serde_json::to_vec(&json!({"runtime": {
            "router_engine": "model",
            "router_model_path": fixture,
        }}))?,
    )?;
    rust_functiongemma_runtime::init_runtime_config(&config_path);

    let result = async {
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        let address = listener.local_addr()?;
        let server =
            tokio::spawn(async move { axum::serve(listener, rust_functiongemma_runtime::build_router()).await });
        let base = format!("http://{address}");
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()?;

        for stream in [false, true] {
            let response = client
                .post(format!("{base}/v1/chat/completions"))
                .json(&json!({"messages": [{"role": "user", "content": "hello"}], "stream": stream}))
                .send()
                .await?;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["x-pcai-router-engine"], "heuristic-fallback");
            let body = response.text().await?;
            assert!(body.contains("NO_TOOL"), "{body}");
            if stream {
                assert!(body.contains("[DONE]"), "{body}");
            }
        }

        let health: Value = client
            .get(format!("{base}/health"))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        assert_eq!(health["metadata"]["engine"], "model");
        assert_eq!(health["router_stats"]["model_fallbacks"], 2);
        server.abort();
        Ok::<(), anyhow::Error>(())
    }
    .await;

    fs::remove_dir_all(&fixture)?;
    result
}
