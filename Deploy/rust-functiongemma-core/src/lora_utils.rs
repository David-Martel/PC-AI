//! Utilities for resolving LoRA adapter paths and reading their configuration.
//!
//! A LoRA adapter can be stored as:
//! - A directory containing `adapter_model.safetensors` and optionally
//!   `adapter_config.json` (standard PEFT layout).
//! - A direct path to a `.safetensors` file, with an optional sibling
//!   `adapter_config.json` in the same directory.
//!
//! When no configuration file is present the rank `r` is inferred by
//! inspecting the shape of the first `lora_a` tensor found in the weights.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// Metadata for a resolved LoRA adapter.
///
/// # Examples
///
/// ```no_run
/// use rust_functiongemma_core::lora_utils::{LoraInfo, resolve_lora_from_path};
///
/// if let Some(info) = resolve_lora_from_path("/path/to/adapter".into()) {
///     println!("r={}, alpha={}", info.r, info.alpha);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct LoraInfo {
    /// Absolute path to the adapter weights file.
    pub path: PathBuf,
    /// LoRA rank.
    pub r: usize,
    /// LoRA alpha scaling factor.
    pub alpha: f64,
    /// Dropout probability applied during training.
    pub dropout: f64,
}

/// Read LoRA hyperparameters from an `adapter_config.json` file.
///
/// Returns `(r, alpha, dropout)` if the file exists and contains at minimum a
/// valid `r` field. `alpha` defaults to `32.0` and `dropout` defaults to `0.0`
/// when those fields are absent.
///
/// Returns [`None`] if the file cannot be read or does not contain a numeric
/// `r` field.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use rust_functiongemma_core::lora_utils::read_lora_config;
///
/// if let Some((r, alpha, dropout)) = read_lora_config(Path::new("adapter_config.json")) {
///     println!("r={r}, alpha={alpha}, dropout={dropout}");
/// }
/// ```
pub fn read_lora_config(config_path: &Path) -> Option<(usize, f64, f64)> {
    let contents = std::fs::read_to_string(config_path).ok()?;
    let val = serde_json::from_str::<Value>(&contents).ok()?;
    let r = val.get("r").and_then(|v| v.as_u64()).map(|v| v as usize)?;
    let alpha = val.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(32.0);
    let dropout = val.get("lora_dropout").and_then(|v| v.as_f64()).unwrap_or(0.0);
    Some((r, alpha, dropout))
}

/// Infer the LoRA rank `r` from the shape of a `lora_a` tensor.
///
/// The function reads the safetensors file at `path`, searches for any tensor
/// whose name contains `"lora_a"`, and returns the size of its first dimension
/// as the rank. Returns [`None`] if the file cannot be read, no matching
/// tensor is found, or the tensor has no dimensions.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use rust_functiongemma_core::lora_utils::infer_lora_r_from_weights;
///
/// if let Some(r) = infer_lora_r_from_weights(Path::new("adapter_model.safetensors")) {
///     println!("inferred r={r}");
/// }
/// ```
pub fn infer_lora_r_from_weights(path: &Path) -> Option<usize> {
    let data = std::fs::read(path).ok()?;
    let tensors = safetensors::SafeTensors::deserialize(&data).ok()?;
    for name in tensors.names() {
        if name.contains("lora_a") {
            if let Ok(info) = tensors.tensor(&name) {
                let shape = info.shape();
                if let Some(&first) = shape.first() {
                    return Some(first);
                }
            }
        }
    }
    None
}

/// Resolve a LoRA adapter from a file or directory path.
///
/// Behaviour:
/// - If `path` is a **directory**, looks for `adapter_model.safetensors` inside
///   it and reads `adapter_config.json` from the same directory.
/// - If `path` is a **file**, uses it directly as the weights path and looks
///   for `adapter_config.json` in the parent directory.
///
/// The rank `r` is read from `adapter_config.json` when available. If no
/// config file exists the rank is inferred from the weight shapes via
/// [`infer_lora_r_from_weights`]. If `r` resolves to `0` the function returns
/// [`None`] because a zero-rank adapter is not useful.
///
/// # Examples
///
/// ```no_run
/// use rust_functiongemma_core::lora_utils::resolve_lora_from_path;
///
/// // Directory layout (standard PEFT)
/// if let Some(info) = resolve_lora_from_path("/checkpoints/my-adapter".into()) {
///     println!("adapter weights: {}", info.path.display());
///     println!("r={}, alpha={}", info.r, info.alpha);
/// }
/// ```
pub fn resolve_lora_from_path(path: PathBuf) -> Option<LoraInfo> {
    let (weights_path, config_path) = if path.is_dir() {
        (path.join("adapter_model.safetensors"), path.join("adapter_config.json"))
    } else {
        let config_path = path.parent().unwrap_or(Path::new(".")).join("adapter_config.json");
        (path, config_path)
    };

    if !weights_path.exists() {
        return None;
    }

    let (r, alpha, dropout) = read_lora_config(&config_path)
        .or_else(|| infer_lora_r_from_weights(&weights_path).map(|r| (r, 32.0, 0.0)))
        .unwrap_or((0, 32.0, 0.0));

    if r == 0 {
        return None;
    }

    Some(LoraInfo {
        path: weights_path,
        r,
        alpha,
        dropout,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_adapter_config(dir: &Path, r: u64, alpha: f64, dropout: f64) {
        let json = serde_json::json!({
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
        });
        fs::write(dir.join("adapter_config.json"), serde_json::to_string(&json).unwrap())
            .expect("failed to write adapter_config.json in test");
    }

    #[test]
    fn read_lora_config_returns_values() {
        let dir = std::env::temp_dir().join("pcai_test_lora_config");
        fs::create_dir_all(&dir).expect("failed to create temp dir for lora config test");
        write_adapter_config(&dir, 16, 32.0, 0.05);

        let result = read_lora_config(&dir.join("adapter_config.json"));
        assert_eq!(result, Some((16, 32.0, 0.05)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_lora_config_defaults_alpha_and_dropout() {
        let dir = std::env::temp_dir().join("pcai_test_lora_defaults");
        fs::create_dir_all(&dir).expect("failed to create temp dir for lora defaults test");
        let json = serde_json::json!({ "r": 8 });
        fs::write(dir.join("adapter_config.json"), serde_json::to_string(&json).unwrap())
            .expect("failed to write adapter_config.json in test");

        let result = read_lora_config(&dir.join("adapter_config.json"));
        assert_eq!(result, Some((8, 32.0, 0.0)));

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_lora_config_missing_r_returns_none() {
        let dir = std::env::temp_dir().join("pcai_test_lora_no_r");
        fs::create_dir_all(&dir).expect("failed to create temp dir for lora no-r test");
        let json = serde_json::json!({ "lora_alpha": 16.0 });
        fs::write(dir.join("adapter_config.json"), serde_json::to_string(&json).unwrap())
            .expect("failed to write adapter_config.json in test");

        let result = read_lora_config(&dir.join("adapter_config.json"));
        assert_eq!(result, None);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_lora_config_nonexistent_file_returns_none() {
        let result = read_lora_config(Path::new("/nonexistent/adapter_config.json"));
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_lora_from_path_nonexistent_returns_none() {
        let result = resolve_lora_from_path(PathBuf::from("/definitely/does/not/exist"));
        assert!(result.is_none());
    }
}
