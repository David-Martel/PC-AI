//! Model download and weight loading for the Janus-Pro pipeline.
//!
//! This module handles:
//! 1. Resolving a model identifier (HuggingFace Hub ID or local path) to a
//!    local directory.
//! 2. Collecting safetensors shard paths within that directory.
//! 3. Memory-mapping the shards into a [`MmapedSafetensors`] archive.
//! 4. Loading tensors from the archive into a [`VarMap`].
//! 5. Loading the HuggingFace tokenizer.
//!
//! # Example
//!
//! ```rust,no_run
//! use pcai_media::hub;
//! use std::path::PathBuf;
//!
//! let model_path = hub::resolve_model_path("deepseek-ai/Janus-Pro-7B")
//!     .expect("failed to download model");
//! let shards = hub::collect_safetensors(&model_path);
//! println!("found {} shard(s)", shards.len());
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use candle_nn::VarMap;

// ---------------------------------------------------------------------------
// resolve_model_path
// ---------------------------------------------------------------------------

/// Resolve a model identifier to a local directory containing model files.
///
/// If `model_id` points to an existing local directory it is returned as-is.
/// Otherwise the function attempts to download the model from HuggingFace Hub
/// using the async tokio API, fetching `config.json` and `tokenizer.json`
/// into the Hub cache directory.
///
/// # Errors
///
/// Returns an error if the Hub download fails or if required files cannot be
/// found after downloading.
pub fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        tracing::info!(model = model_id, "using local model directory");
        return Ok(local);
    }

    tracing::info!(model = model_id, "downloading model from HuggingFace Hub");

    // Use the async tokio API (avoids ureq → rustls → ring build dependency).
    // Block on the async call since resolve_model_path is sync.
    let rt = tokio::runtime::Handle::try_current()
        .map(|h| {
            // We're already inside a tokio runtime — use spawn_blocking to
            // avoid blocking the async executor.
            std::thread::scope(|s| {
                s.spawn(|| {
                    let local_rt = tokio::runtime::Runtime::new().expect("tokio runtime");
                    local_rt.block_on(download_model_async(model_id))
                })
                .join()
                .expect("download thread panicked")
            })
        })
        .unwrap_or_else(|_| {
            // No runtime — create one and block.
            let rt = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
            rt.block_on(download_model_async(model_id))
        })?;

    Ok(rt)
}

/// Async implementation of model download from HuggingFace Hub.
async fn download_model_async(model_id: &str) -> Result<PathBuf> {
    let api = hf_hub::api::tokio::ApiBuilder::new()
        .build()
        .context("failed to initialise HuggingFace Hub API")?;
    let repo = api.model(model_id.to_string());

    // Fetch config.json — the returned path reveals the local cache directory.
    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("failed to download config.json for model '{model_id}'"))?;

    // Fetch tokenizer.json into the same cache directory.
    repo.get("tokenizer.json")
        .await
        .with_context(|| format!("failed to download tokenizer.json for model '{model_id}'"))?;

    // The parent of config.json is the per-revision snapshot directory.
    let model_dir = config_path
        .parent()
        .with_context(|| "config.json path has no parent directory")?
        .to_path_buf();

    tracing::debug!(path = %model_dir.display(), "resolved model path");
    Ok(model_dir)
}

// ---------------------------------------------------------------------------
// collect_safetensors
// ---------------------------------------------------------------------------

/// Collect safetensors weight shard paths from a model directory.
///
/// Checks for a single `model.safetensors` file first. If that does not exist,
/// collects all `model-*.safetensors` shards in lexicographic order (matching
/// the ordering used by HuggingFace sharded checkpoints).
///
/// Returns an empty `Vec` if neither file pattern is found.
pub fn collect_safetensors(model_path: &Path) -> Vec<PathBuf> {
    let direct = model_path.join("model.safetensors");
    if direct.exists() {
        return vec![direct];
    }

    let mut shards: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shards.push(path);
                }
            }
        }
    }
    shards.sort();
    shards
}

// ---------------------------------------------------------------------------
// open_safetensors
// ---------------------------------------------------------------------------

/// Memory-map one or more safetensors files into a [`MmapedSafetensors`].
///
/// Uses [`MmapedSafetensors::new`] for a single file or
/// [`MmapedSafetensors::multi`] for multiple shards.
///
/// # Safety
///
/// This function calls the unsafe [`MmapedSafetensors`] API which memory-maps
/// the files. The caller must ensure no concurrent writer modifies the files
/// while the returned object is alive.
///
/// # Errors
///
/// Returns an error if `paths` is empty or if any I/O operation fails.
pub fn open_safetensors(paths: &[PathBuf]) -> Result<MmapedSafetensors> {
    anyhow::ensure!(!paths.is_empty(), "no safetensors files to open");

    // Safety: no concurrent writer modifies the files while mmap is alive.
    let archive = unsafe {
        if paths.len() == 1 {
            MmapedSafetensors::new(&paths[0]).with_context(|| format!("failed to mmap '{}'", paths[0].display()))?
        } else {
            MmapedSafetensors::multi(paths).context("failed to mmap safetensors shards")?
        }
    };
    Ok(archive)
}

// ---------------------------------------------------------------------------
// load_weights
// ---------------------------------------------------------------------------

/// Load tensors from safetensors files into a [`VarMap`].
///
/// The function memory-maps the provided `paths` and iterates over every
/// variable registered in `varmap`.  For each variable it first attempts a
/// direct key lookup.  If fewer than one-third of variables are found that
/// way it retries with the common `"model."` prefix.
///
/// Tensors whose stored dtype differs from the variable dtype are cast
/// automatically.
///
/// Returns the count of variables successfully loaded.
///
/// # Errors
///
/// Returns an error if `paths` is empty, if any I/O fails, or if a dtype cast
/// fails.
pub fn load_weights(varmap: &VarMap, paths: &[PathBuf], dtype: DType, device: &Device) -> Result<usize> {
    let archive = open_safetensors(paths)?;

    // Build a fast lookup set from archive tensor names.
    use std::collections::HashSet;
    let st_names: HashSet<String> = archive.tensors().iter().map(|(name, _)| name.clone()).collect();

    let data = varmap.data().lock().expect("VarMap lock poisoned");

    // Try to load all variables using an optional dot-separated prefix.
    // Returns the count of successfully updated variables.
    let load_with_prefix = |prefix: Option<&str>| -> Result<usize> {
        let mut count = 0usize;
        for (name, var) in data.iter() {
            let lookup_key = match prefix {
                Some(pfx) if !pfx.is_empty() => format!("{pfx}.{name}"),
                _ => name.clone(),
            };
            if !st_names.contains(&lookup_key) {
                continue;
            }
            // Load onto the target device and cast to the requested dtype.
            let mut tensor = archive
                .load(&lookup_key, device)
                .with_context(|| format!("failed to load tensor '{lookup_key}'"))?;
            if tensor.dtype() != dtype {
                tensor = tensor
                    .to_dtype(dtype)
                    .with_context(|| format!("dtype cast failed for '{lookup_key}'"))?;
            }
            var.set(&tensor)
                .with_context(|| format!("failed to set variable '{name}'"))?;
            count += 1;
        }
        Ok(count)
    };

    // First pass: no prefix.
    let direct_count = load_with_prefix(None)?;
    let total_vars = data.len();

    // Second pass: try common "model." prefix if fewer than 1/3 were found.
    if direct_count < (total_vars / 3).max(1) {
        // Detect prefix from a heuristic scan of the archive names.
        let detected = detect_prefix(&st_names);
        if let Some(ref pfx) = detected {
            let prefixed_count = load_with_prefix(Some(pfx.as_str()))?;
            if prefixed_count > direct_count {
                tracing::debug!(
                    prefix = pfx,
                    loaded = prefixed_count,
                    total = total_vars,
                    "applied safetensors prefix"
                );
                return Ok(prefixed_count);
            }
        }
    }

    tracing::debug!(loaded = direct_count, total = total_vars, "loaded weights (no prefix)");
    Ok(direct_count)
}

/// Heuristic: detect a dot-separated namespace prefix from archive key names.
fn detect_prefix(names: &std::collections::HashSet<String>) -> Option<String> {
    for name in names {
        if name.starts_with("model.") {
            return Some("model".to_string());
        }
        if let Some(pos) = name.find(".layers.") {
            let prefix = name[..pos].trim_end_matches('.');
            if !prefix.is_empty() {
                return Some(prefix.to_string());
            }
        }
        if name.ends_with("embed_tokens.weight") {
            let prefix = name.trim_end_matches("embed_tokens.weight").trim_end_matches('.');
            if !prefix.is_empty() {
                return Some(prefix.to_string());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// load_tokenizer
// ---------------------------------------------------------------------------

/// Load a HuggingFace [`tokenizers::Tokenizer`] from a model directory.
///
/// Looks for `tokenizer.json` inside `model_path`.
///
/// # Errors
///
/// Returns an error if `tokenizer.json` is not found or cannot be parsed.
pub fn load_tokenizer(model_path: &Path) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    anyhow::ensure!(
        tokenizer_path.exists(),
        "tokenizer.json not found at '{}'",
        tokenizer_path.display()
    );
    tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// An empty directory produces an empty shard list.
    #[test]
    fn test_collect_safetensors_empty_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let shards = collect_safetensors(tmp.path());
        assert!(shards.is_empty(), "expected no shards in empty dir");
    }

    /// A single `model.safetensors` is returned as a one-element vec.
    #[test]
    fn test_collect_safetensors_single_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file = tmp.path().join("model.safetensors");
        fs::write(&file, b"").expect("write");

        let shards = collect_safetensors(tmp.path());
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], file);
    }

    /// Sharded files `model-00001-of-00003.safetensors` etc. are returned in
    /// lexicographic (ascending) order.
    #[test]
    fn test_collect_safetensors_sharded_order() {
        let tmp = tempfile::tempdir().expect("tempdir");

        // Write shards out of order.
        let names = [
            "model-00003-of-00003.safetensors",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
        ];
        for name in &names {
            fs::write(tmp.path().join(name), b"").expect("write");
        }

        let shards = collect_safetensors(tmp.path());
        assert_eq!(shards.len(), 3);

        let file_names: Vec<&str> = shards
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(file_names[0], "model-00001-of-00003.safetensors");
        assert_eq!(file_names[1], "model-00002-of-00003.safetensors");
        assert_eq!(file_names[2], "model-00003-of-00003.safetensors");
    }

    /// A directory containing only non-model files produces an empty result.
    #[test]
    fn test_collect_safetensors_ignores_unrelated_files() {
        let tmp = tempfile::tempdir().expect("tempdir");
        fs::write(tmp.path().join("config.json"), b"{}").expect("write");
        fs::write(tmp.path().join("tokenizer.json"), b"{}").expect("write");
        fs::write(tmp.path().join("pytorch_model.bin"), b"").expect("write");

        let shards = collect_safetensors(tmp.path());
        assert!(shards.is_empty());
    }

    /// `model.safetensors` takes priority over any shards in the same dir.
    #[test]
    fn test_collect_safetensors_single_takes_priority() {
        let tmp = tempfile::tempdir().expect("tempdir");
        fs::write(tmp.path().join("model.safetensors"), b"").expect("write");
        fs::write(tmp.path().join("model-00001-of-00002.safetensors"), b"").expect("write");
        fs::write(tmp.path().join("model-00002-of-00002.safetensors"), b"").expect("write");

        let shards = collect_safetensors(tmp.path());
        assert_eq!(shards.len(), 1);
        assert!(shards[0].file_name().unwrap().to_str().unwrap() == "model.safetensors");
    }

    /// `open_safetensors` returns an error on an empty slice.
    #[test]
    fn test_open_safetensors_empty_paths() {
        let result = open_safetensors(&[]);
        assert!(result.is_err());
    }

    /// `load_tokenizer` returns an error when the directory is missing
    /// `tokenizer.json`.
    #[test]
    fn test_load_tokenizer_missing_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let result = load_tokenizer(tmp.path());
        assert!(result.is_err());
    }
}
