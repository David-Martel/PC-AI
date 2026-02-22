#[cfg(feature = "model")]
use anyhow::Result;
#[cfg(feature = "model")]
use hf_hub::{api::sync::Api, Repo, RepoType};
#[cfg(feature = "model")]
use std::path::{Path, PathBuf};

#[cfg(feature = "model")]
#[allow(dead_code)]
pub fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let p = PathBuf::from(model_id);

    // 1. Absolute or existing relative path
    if p.is_absolute() && p.exists() {
        tracing::info!("Model path (absolute): {}", p.display());
        return Ok(p);
    }
    if p.exists() {
        tracing::info!("Model path (relative): {}", p.display());
        return Ok(p);
    }

    // 2. PCAI_MODELS_DIR environment variable
    if let Ok(dir) = std::env::var("PCAI_MODELS_DIR") {
        let candidate = PathBuf::from(&dir).join(model_id);
        if candidate.exists() {
            tracing::info!("Model path ($PCAI_MODELS_DIR): {}", candidate.display());
            return Ok(candidate);
        }
    }

    // 3. Relative to CWD/Models/
    let models_dir = PathBuf::from("Models").join(model_id);
    if models_dir.exists() {
        tracing::info!("Model path (Models/): {}", models_dir.display());
        return Ok(models_dir);
    }

    // 4. Common Windows model locations
    if let Ok(home) = std::env::var("USERPROFILE") {
        let candidate = PathBuf::from(&home)
            .join(".cache")
            .join("pcai")
            .join("models")
            .join(model_id);
        if candidate.exists() {
            tracing::info!(
                "Model path (~/.cache/pcai/models/): {}",
                candidate.display()
            );
            return Ok(candidate);
        }
    }

    // 5. Fallback to HuggingFace hub download
    tracing::info!(
        "Model not found locally, trying HuggingFace hub: {}",
        model_id
    );
    let api = Api::new()?;
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
    let api = api.repo(repo);

    let tokenizer_path = api.get("tokenizer.json")?;
    let template_path = api.get("chat_template.jinja")?;

    let root = tokenizer_path.parent().unwrap_or(Path::new("."));
    if !template_path.exists() {
        return Err(anyhow::anyhow!(
            "chat_template.jinja not found after download"
        ));
    }

    Ok(root.to_path_buf())
}

#[cfg(all(test, feature = "model"))]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_existing_relative_path() {
        let dir = std::env::temp_dir().join("pcai_test_model_resolve");
        std::fs::create_dir_all(&dir).expect("TODO: Verify unwrap");
        std::fs::write(dir.join("config.json"), "{}").expect("TODO: Verify unwrap");

        let result = resolve_model_path(dir.to_str().expect("TODO: Verify unwrap"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dir);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_resolve_with_env_var() {
        let base = std::env::temp_dir().join("pcai_test_models_env");
        let model_dir = base.join("test-model");
        std::fs::create_dir_all(&model_dir).expect("TODO: Verify unwrap");

        std::env::set_var("PCAI_MODELS_DIR", base.to_str().expect("TODO: Verify unwrap"));
        let result = resolve_model_path("test-model");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_dir);

        std::env::remove_var("PCAI_MODELS_DIR");
        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn test_resolve_nonexistent_returns_error_or_fallback() {
        // With no HF token, this should either find nothing locally
        // or fail on HF download - just verify it doesn't panic
        std::env::remove_var("PCAI_MODELS_DIR");
        let _result = resolve_model_path("definitely-not-a-real-model-12345");
        // We don't assert Ok/Err because HF might or might not be available
    }
}
