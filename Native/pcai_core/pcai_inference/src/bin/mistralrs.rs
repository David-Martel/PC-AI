//! pcai-mistralrs HTTP server
//! Specialized binary for the mistral.rs backend.

use pcai_inference_lib::{
    backends::{mistralrs::MistralRsBackend, InferenceBackend},
    config::{BackendConfig, InferenceConfig},
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

fn create_backend(config: &BackendConfig) -> anyhow::Result<Box<dyn InferenceBackend>> {
    match config {
        BackendConfig::MistralRs { device } => {
            Ok(Box::new(MistralRsBackend::with_device_selection(device.as_deref())?))
        }
        #[cfg(feature = "llamacpp")]
        BackendConfig::LlamaCpp { .. } => Err(anyhow::anyhow!(
            "Invalid backend type in configuration. pcai-mistralrs requires mistral_rs configuration."
        )),
    }
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
                .unwrap_or_else(|_| "pcai_mistralrs=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting pcai-mistralrs server v{}", version::VERSION);

    // Load configuration from file or use defaults
    let config_path = extract_config_path(&args)
        .or_else(default_config_path)
        .ok_or_else(|| anyhow::anyhow!("Configuration required. Pass --config <path>"))?;
    tracing::info!("Loading configuration from {}", config_path);
    let config = InferenceConfig::from_file(config_path)?;

    // Create backend
    let mut backend = create_backend(&config.backend)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    // Protects config propagation; detects discarded selectors before loading any model.
    // Needs no GPU/model. Breadcrumb: bin/mistralrs.rs create_backend.
    #[test]
    fn invalid_config_device_fails_before_model_loading() {
        let config = BackendConfig::MistralRs {
            device: Some("cuda:not-an-index".into()),
        };
        let result = create_backend(&config);
        assert!(matches!(result, Err(error) if error.to_string().contains("Invalid Mistral device selector")));
    }

    // Protects CPU-only startup; detects rejection of valid CPU configuration.
    // Needs no GPU/model. Breadcrumb: bin/mistralrs.rs create_backend.
    #[test]
    fn cpu_config_creates_unloaded_backend() {
        let config = BackendConfig::MistralRs {
            device: Some("cpu".into()),
        };
        let backend = create_backend(&config).expect("CPU config must construct without a model");
        assert!(!backend.is_loaded());
        assert_eq!(backend.backend_name(), "mistral.rs");
    }
}
