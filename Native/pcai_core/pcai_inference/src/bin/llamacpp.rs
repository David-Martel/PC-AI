//! pcai-llamacpp HTTP server
//! Specialized binary for the llama.cpp backend.

use pcai_inference_lib::{
    backends::BackendType,
    config::{InferenceConfig, ServerConfig},
    http::run_server,
    version,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn extract_config_path(args: &[String]) -> Option<String> {
    let mut i = 0usize;
    while i < args.len() {
        let arg = &args[i];
        if arg == "--config" || arg == "-c" {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = arg.strip_prefix("--config=") {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

fn default_config_path() -> Option<String> {
    let path = std::path::PathBuf::from("Config/pcai-inference.json");
    if path.exists() {
        return Some(path.to_string_lossy().to_string());
    }
    None
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Handle --version flag
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{}", version::build_info());
        return Ok(());
    }
    if args.iter().any(|a| a == "--version-json") {
        println!("{}", version::build_info_json());
        return Ok(());
    }

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pcai_llamacpp=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting pcai-llamacpp server v{}", version::VERSION);

    // Load configuration from file or use defaults
    let config_path = extract_config_path(&args)
        .or_else(default_config_path)
        .ok_or_else(|| anyhow::anyhow!("Configuration required. Pass --config <path>"))?;
    tracing::info!("Loading configuration from {}", config_path);
    let config = InferenceConfig::from_file(config_path)?;

    // Create backend
    let backend_type = match &config.backend {
        #[cfg(feature = "llamacpp")]
        pcai_inference_lib::config::BackendConfig::LlamaCpp { .. } => BackendType::LlamaCpp,

        #[expect(
            unreachable_patterns,
            reason = "fallthrough guard: reachable when llamacpp feature is disabled at compile time"
        )]
        _ => {
            return Err(anyhow::anyhow!(
                "Invalid backend type in configuration. pcai-llamacpp requires llama_cpp configuration."
            ));
        }
    };

    let mut backend = backend_type.create()?;

    // Load model
    tracing::info!("Loading model from {:?}", config.model.path);
    backend
        .load_model(
            config
                .model
                .path
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?,
        )
        .await?;

    // Start server
    let server_config = config.server.clone().unwrap_or_default();
    let router_config = config.router.clone();
    run_server(server_config, router_config, backend).await?;

    Ok(())
}
