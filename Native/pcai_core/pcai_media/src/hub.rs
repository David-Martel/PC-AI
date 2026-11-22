//! Model download and weight loading for the Janus-Pro pipeline.
//!
//! This module handles:
//! 1. Resolving a model identifier (local path, prepared SafeTensors copy of a
//!    HuggingFace Hub ID, or the Hub cache) to a local directory.
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
//!     .expect("failed to resolve model");
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

/// Environment variable naming a directory that holds prepared model
/// directories (`<dir>/<model name>/` with SafeTensors weights).
pub const MODELS_DIR_ENV: &str = "PCAI_MODELS_DIR";

/// Directories searched for a prepared copy of a Hub model id: `$PCAI_MODELS_DIR`
/// first, then `Models` relative to the working directory (the layout written
/// by `Tools/Get-JanusProModel.ps1`).
pub fn prepared_model_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os(MODELS_DIR_ENV).filter(|value| !value.is_empty()) {
        roots.push(PathBuf::from(dir));
    }
    roots.push(PathBuf::from("Models"));
    roots
}

/// Last path component of a model id (`deepseek-ai/Janus-Pro-1B` -> `Janus-Pro-1B`).
fn model_dir_name(model_id: &str) -> Option<&str> {
    let name = model_id.rsplit(['/', '\\']).next()?;
    (!name.is_empty() && name != "." && name != "..").then_some(name)
}

/// Find a prepared copy of `model_id` under one of `roots`.
///
/// A prepared directory is `<root>/<model name>/` holding `config.json`,
/// `tokenizer.json` and SafeTensors weights, which is what the loader needs.
pub fn find_prepared_model(model_id: &str, roots: &[PathBuf]) -> Option<PathBuf> {
    let name = model_dir_name(model_id)?;
    roots.iter().map(|root| root.join(name)).find(|dir| {
        dir.join("config.json").is_file()
            && dir.join("tokenizer.json").is_file()
            && !collect_safetensors(dir).is_empty()
    })
}

/// Resolve a model identifier to a local directory containing model files.
///
/// Order: an existing local directory is returned as-is; otherwise a prepared
/// SafeTensors copy under [`prepared_model_roots`]; otherwise the model is
/// downloaded from HuggingFace Hub (`config.json` and `tokenizer.json` only,
/// into the Hub cache). Janus-Pro repositories publish PyTorch `.bin`
/// weights, so the downloaded directory has no SafeTensors until the weights
/// are prepared; see [`weights_preparation_hint`].
///
/// # Errors
///
/// Returns an error if the Hub download fails or if required files cannot be
/// found after downloading.
pub fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    resolve_model_path_with(model_id, &prepared_model_roots(), download_model_blocking)
}

/// [`resolve_model_path`] with injectable search roots and downloader, so the
/// cold-cache behaviour can be tested without a network.
pub fn resolve_model_path_with(
    model_id: &str,
    roots: &[PathBuf],
    download: impl FnOnce(&str) -> Result<PathBuf>,
) -> Result<PathBuf> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        tracing::info!(model = model_id, "using local model directory");
        return Ok(local);
    }
    if let Some(prepared) = find_prepared_model(model_id, roots) {
        tracing::info!(model = model_id, path = %prepared.display(), "using prepared SafeTensors model directory");
        return Ok(prepared);
    }
    tracing::info!(model = model_id, "downloading model from HuggingFace Hub");
    download(model_id)
}

/// Whether `dir` holds PyTorch checkpoint weights (single, sharded or index).
fn has_pytorch_weights(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("pytorch_model") && (name.ends_with(".bin") || name.ends_with(".bin.index.json"))
            })
        })
        .unwrap_or(false)
}

/// Actionable text for a model directory that has no SafeTensors weights.
///
/// The loader never reads PyTorch `.bin` checkpoints and refuses random
/// weights. Both published Janus-Pro repositories (`deepseek-ai/Janus-Pro-1B`
/// and `-7B`) contain only `.bin` weights, so they must be downloaded and
/// converted once. The commands are the repository's own tools:
/// `Tools/Get-JanusProModel.ps1` (downloads `config.json`, the tokenizer and
/// `pytorch_model*.bin` into `Models/<name>`) and
/// `Tools/convert-sharded-weights.py` (writes `model.safetensors` next to
/// them).
pub fn weights_preparation_hint(model_id: &str, model_path: &Path) -> String {
    let local_dir = PathBuf::from(model_id).is_dir();
    let (prepared_dir, download_step) = if local_dir {
        let dir = model_path.display().to_string();
        let download = (!has_pytorch_weights(model_path)).then(|| {
            format!(
                "1. download the PyTorch weights into it, for example: pwsh Tools/Get-JanusProModel.ps1 -ModelId deepseek-ai/Janus-Pro-1B -OutputDir \"{dir}\"; "
            )
        });
        (dir, download)
    } else {
        let name = model_dir_name(model_id).unwrap_or("Janus-Pro-1B");
        let dir = format!("Models/{name}");
        let download = format!("1. pwsh Tools/Get-JanusProModel.ps1 -ModelId {model_id}; ");
        (dir, Some(download))
    };
    let convert = format!(
        "{}uv run python Tools/convert-sharded-weights.py \"{prepared_dir}\" (writes model.safetensors there)",
        if download_step.is_some() { "2. " } else { "" }
    );
    let finish = if local_dir {
        format!("then load \"{prepared_dir}\" again")
    } else {
        format!(
            "then load \"{prepared_dir}\" (pass it as the model, or keep it under Models/ or ${MODELS_DIR_ENV} so '{model_id}' resolves to it)"
        )
    };
    format!(
        "Janus-Pro is published as PyTorch weights and this loader only reads complete SafeTensors (it refuses random weights). Prepare them once: {}{convert}; {finish}",
        download_step.unwrap_or_default()
    )
}

/// Download `config.json` and `tokenizer.json` for `model_id` into the Hub
/// cache, blocking the caller (safe inside or outside a tokio runtime).
fn download_model_blocking(model_id: &str) -> Result<PathBuf> {
    // Use the async tokio API (avoids ureq -> rustls -> ring build dependency).
    // Block on the async call since the loader is sync.
    let run = || -> Result<PathBuf> {
        let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
        runtime.block_on(download_model_async(model_id))
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        // Already inside a tokio runtime: use a scoped thread so the async
        // executor is not blocked.
        std::thread::scope(|scope| {
            scope
                .spawn(run)
                .join()
                .map_err(|_| anyhow::anyhow!("model download thread panicked"))?
        })
    } else {
        run()
    }
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

/// Missing trained tensors in an otherwise readable checkpoint.
#[derive(Debug, thiserror::Error)]
#[error("incomplete safetensors weights: missing {missing_count} of {total} required tensors ({names})")]
pub(crate) struct IncompleteWeights {
    missing_count: usize,
    total: usize,
    names: String,
}

/// Validate required tensor headers without loading weight payloads.
pub(crate) fn validate_weight_metadata(
    paths: &[PathBuf],
    expected: &std::collections::BTreeMap<String, Vec<usize>>,
) -> Result<()> {
    anyhow::ensure!(!expected.is_empty(), "no required generation tensor metadata");
    let archive = open_safetensors(paths)?;
    let names = archive.tensors().into_iter().map(|(name, _)| name).collect();
    let prefix = detect_prefix(&names);
    let mut missing = Vec::new();
    let mut matched = Vec::with_capacity(expected.len());
    for (name, shape) in expected {
        match matching_tensor_key(name, &names, prefix.as_deref()) {
            Some(key) => matched.push((name, shape, key)),
            None => missing.push(name.as_str()),
        }
    }
    if !missing.is_empty() {
        return Err(IncompleteWeights {
            missing_count: missing.len(),
            total: expected.len(),
            names: missing.iter().take(8).copied().collect::<Vec<_>>().join(", "),
        }
        .into());
    }
    for (name, shape, key) in matched {
        let header = archive
            .get(&key)
            .with_context(|| format!("missing tensor header '{key}'"))?;
        anyhow::ensure!(
            header.shape() == shape.as_slice(),
            "unexpected shape for required tensor '{name}': {:?}, expected {shape:?}",
            header.shape()
        );
    }
    Ok(())
}

fn matching_tensor_key(name: &str, names: &std::collections::HashSet<String>, prefix: Option<&str>) -> Option<String> {
    if names.contains(name) {
        Some(name.to_string())
    } else {
        prefix
            .map(|prefix| format!("{prefix}.{name}"))
            .filter(|key| names.contains(key))
    }
}

/// Load tensors from safetensors files into a [`VarMap`].
///
/// The function memory-maps the provided `paths` and iterates over every
/// variable registered in `varmap`.  For each variable it first attempts a
/// direct key lookup, then a detected namespace prefix for unmatched names.
/// Every registered variable must have a matching trained tensor. Missing
/// names are rejected before any variable is updated.
///
/// Tensors whose stored dtype differs from the variable dtype are cast
/// automatically.
///
/// Returns the count of variables successfully loaded.
///
/// # Errors
///
/// Returns an error if `paths` or `varmap` is empty, any required tensor is
/// missing or misshapen, any I/O fails, or a dtype cast fails.
pub fn load_weights(varmap: &VarMap, paths: &[PathBuf], dtype: DType, device: &Device) -> Result<usize> {
    let archive = open_safetensors(paths)?;

    // Build a fast lookup set from archive tensor names.
    use std::collections::HashSet;
    let st_names: HashSet<String> = archive.tensors().iter().map(|(name, _)| name.clone()).collect();

    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("VarMap lock poisoned"))?;
    anyhow::ensure!(
        !data.is_empty(),
        "cannot load weights before model variables are registered"
    );

    let prefix = detect_prefix(&st_names);
    let mut matched = Vec::with_capacity(data.len());
    let mut missing = Vec::new();
    for (name, var) in data.iter() {
        match matching_tensor_key(name, &st_names, prefix.as_deref()) {
            Some(key) => matched.push((name, var, key)),
            None => missing.push(name.as_str()),
        }
    }
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(IncompleteWeights {
            missing_count: missing.len(),
            total: data.len(),
            names: missing.iter().take(8).copied().collect::<Vec<_>>().join(", "),
        }
        .into());
    }

    for (name, var, key) in matched {
        let mut tensor = archive
            .load(&key, device)
            .with_context(|| format!("failed to load tensor '{key}'"))?;
        if tensor.dtype() != dtype {
            tensor = tensor
                .to_dtype(dtype)
                .with_context(|| format!("dtype cast failed for '{key}'"))?;
        }
        var.set(&tensor)
            .with_context(|| format!("failed to set variable '{name}'"))?;
    }
    tracing::debug!(loaded = data.len(), prefix, "loaded complete model weights");
    Ok(data.len())
}

/// Heuristic: detect a dot-separated namespace prefix from archive key names.
fn detect_prefix(names: &std::collections::HashSet<String>) -> Option<String> {
    // Prefer the supported common namespace before scanning unrelated direct
    // LLM keys; HashSet iteration order must not affect checkpoint loading.
    if names.iter().any(|name| name.starts_with("model.")) {
        return Some("model".to_string());
    }
    let mut ordered: Vec<_> = names.iter().collect();
    ordered.sort_unstable();
    for name in ordered {
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
    use candle_core::Tensor;
    use std::collections::HashMap;
    use std::fs;

    fn weight_fixture(path: &Path, entries: &[(&str, Vec<f32>)]) {
        let tensors: HashMap<String, Tensor> = entries
            .iter()
            .map(|(name, values)| {
                (
                    (*name).to_string(),
                    Tensor::from_vec(values.clone(), values.len(), &Device::Cpu).expect("fixture tensor"),
                )
            })
            .collect();
        candle_core::safetensors::save(&tensors, path).expect("save fixture weights");
    }

    fn required_variables() -> VarMap {
        let vars = VarMap::new();
        for name in ["first.weight", "second.weight"] {
            vars.get(2, name, candle_nn::Init::Const(-1.0), DType::F32, &Device::Cpu)
                .expect("register required tensor");
        }
        vars
    }

    fn required_metadata() -> std::collections::BTreeMap<String, Vec<usize>> {
        [
            ("first.weight".to_string(), vec![2]),
            ("second.weight".to_string(), vec![2]),
        ]
        .into()
    }

    // Protects: header preflight accepts a complete checkpoint that mixes plain and "model."-prefixed names.
    // Detects: prefix detection that rejects valid real-world checkpoints.
    // Needs: temp directory only.
    #[test]
    fn test_weight_metadata_accepts_complete_mixed_prefix_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        weight_fixture(
            &path,
            &[
                ("first.weight", vec![1.0, 2.0]),
                ("model.second.weight", vec![3.0, 4.0]),
            ],
        );
        validate_weight_metadata(&[path], &required_metadata()).unwrap();
    }

    // Protects: header preflight reports missing tensors with a typed IncompleteWeights error.
    // Detects: a preflight that checks only some tensors or loses the typed error.
    // Needs: temp directory only.
    #[test]
    fn test_weight_metadata_rejects_partial_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0, 2.0])]);
        let error = validate_weight_metadata(&[path], &required_metadata()).unwrap_err();
        assert!(error.downcast_ref::<IncompleteWeights>().is_some());
        assert!(error.to_string().contains("second.weight"));
    }

    // Protects: a tensor with the wrong shape is refused from the header alone.
    // Detects: preflight comparing names but not shapes.
    // Needs: temp directory only.
    #[test]
    fn test_weight_metadata_rejects_wrong_shape_headers() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0]), ("second.weight", vec![3.0, 4.0])]);
        let error = validate_weight_metadata(&[path], &required_metadata()).unwrap_err();
        assert!(error.to_string().contains("unexpected shape"));
        assert!(error.to_string().contains("first.weight"));
    }

    // Protects: loading into an empty VarMap is an error, not a silent no-op that leaves a later model random.
    // Detects: a load_weights that returns Ok(0) for an empty registry.
    // Needs: temp directory only.
    #[test]
    fn test_load_weights_rejects_unregistered_variables() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0, 2.0])]);
        let error = load_weights(&VarMap::new(), &[path], DType::F32, &Device::Cpu).expect_err("no variables");
        assert!(error.to_string().contains("before model variables are registered"));
    }

    // Protects: an empty archive cannot satisfy required variables.
    // Detects: the removed one-third-found heuristic returning success.
    // Needs: temp directory only.
    #[test]
    fn test_load_weights_rejects_empty_archive() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[]);
        let error = load_weights(&required_variables(), &[path], DType::F32, &Device::Cpu)
            .expect_err("empty archive must not leave random weights");
        assert!(error.to_string().contains("missing 2 of 2"));
    }

    // Protects: a partial archive fails before any variable is overwritten.
    // Detects: check-and-set interleaving that leaves a half-loaded model after an error.
    // Needs: temp directory only.
    #[test]
    fn test_load_weights_rejects_partial_archive_before_mutation() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0, 2.0])]);
        let vars = required_variables();
        let error = load_weights(&vars, &[path], DType::F32, &Device::Cpu).expect_err("incomplete weights");
        assert!(error.to_string().contains("second.weight"));
        let data = vars.data().lock().expect("varmap lock");
        assert_eq!(
            data["first.weight"].as_tensor().to_vec1::<f32>().unwrap(),
            vec![-1.0, -1.0]
        );
    }

    // Protects: a complete archive with mixed prefixes loads every variable with the right values.
    // Detects: prefix handling regressions and HashSet-order-dependent prefix detection.
    // Needs: temp directory only.
    #[test]
    fn test_load_weights_accepts_complete_mixed_prefix_archive() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(
            &path,
            &[
                ("first.weight", vec![1.0, 2.0]),
                ("model.second.weight", vec![3.0, 4.0]),
                ("language_model.model.layers.0.weight", vec![5.0, 6.0]),
            ],
        );
        let vars = required_variables();
        assert_eq!(load_weights(&vars, &[path], DType::F32, &Device::Cpu).unwrap(), 2);
        let data = vars.data().lock().expect("varmap lock");
        assert_eq!(
            data["first.weight"].as_tensor().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            data["second.weight"].as_tensor().to_vec1::<f32>().unwrap(),
            vec![3.0, 4.0]
        );
    }

    // Protects: a wrong-shaped tensor is an error naming the tensor.
    // Detects: shape mismatches being cast or truncated into the variable.
    // Needs: temp directory only.
    #[test]
    fn test_load_weights_rejects_wrong_shape() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0]), ("second.weight", vec![3.0, 4.0])]);
        let error = load_weights(&required_variables(), &[path], DType::F32, &Device::Cpu)
            .expect_err("mismatched shape must fail");
        assert!(format!("{error:#}").contains("first.weight"));
    }

    // Protects: the cold-cache path for the default model id: a prepared SafeTensors copy
    //   is used without touching the network.
    // Detects: resolution that downloads (or fails) even though Models/<name> is ready.
    // Needs: a temp directory; the downloader is a mock that fails the test if called.
    #[test]
    fn test_prepared_copy_resolves_without_network() {
        let root = tempfile::tempdir().expect("tempdir");
        let prepared = root.path().join("Janus-Pro-1B");
        fs::create_dir(&prepared).unwrap();
        fs::write(prepared.join("config.json"), "{}").unwrap();
        fs::write(prepared.join("tokenizer.json"), "{}").unwrap();
        weight_fixture(&prepared.join("model.safetensors"), &[("first.weight", vec![1.0, 2.0])]);

        let resolved = resolve_model_path_with("deepseek-ai/Janus-Pro-1B", &[root.path().to_path_buf()], |_| {
            anyhow::bail!("the network must not be used when a prepared copy exists")
        })
        .expect("prepared copy resolves");
        assert_eq!(resolved, prepared);
    }

    // Protects: a cold cache (nothing prepared) falls back to the downloader, and the
    //   resulting config-only directory yields an actionable hint naming the repo's tools.
    // Detects: a hint that omits the download or convert command, the target directory,
    //   or the way to point the loader at the result; an incomplete prepared directory
    //   (no tokenizer) being treated as usable.
    // Needs: temp directories; the downloader is a mock returning a config-only cache dir.
    #[test]
    fn test_cold_cache_falls_back_to_download_and_hint_names_the_commands() {
        let roots = tempfile::tempdir().expect("tempdir");
        let incomplete = roots.path().join("Janus-Pro-1B");
        fs::create_dir(&incomplete).unwrap();
        fs::write(incomplete.join("config.json"), "{}").unwrap();
        weight_fixture(&incomplete.join("model.safetensors"), &[("first.weight", vec![1.0])]);

        let cache = tempfile::tempdir().expect("tempdir");
        fs::write(cache.path().join("config.json"), "{}").unwrap();
        fs::write(cache.path().join("tokenizer.json"), "{}").unwrap();
        let cache_path = cache.path().to_path_buf();
        let mut downloaded = false;
        let resolved = resolve_model_path_with("deepseek-ai/Janus-Pro-1B", &[roots.path().to_path_buf()], |id| {
            assert_eq!(id, "deepseek-ai/Janus-Pro-1B");
            downloaded = true;
            Ok(cache_path.clone())
        })
        .expect("mock download resolves");
        assert!(
            downloaded,
            "an incomplete prepared directory must not satisfy resolution"
        );
        assert_eq!(resolved, cache.path());
        assert!(collect_safetensors(&resolved).is_empty());

        let hint = weights_preparation_hint("deepseek-ai/Janus-Pro-1B", &resolved);
        for needle in [
            "Tools/Get-JanusProModel.ps1 -ModelId deepseek-ai/Janus-Pro-1B",
            "Tools/convert-sharded-weights.py \"Models/Janus-Pro-1B\"",
            "PCAI_MODELS_DIR",
            "refuses random weights",
        ] {
            assert!(hint.contains(needle), "missing {needle:?} in: {hint}");
        }
    }

    // Protects: a local directory that already has PyTorch weights gets an in-place
    //   convert command and no download step; an existing local directory is never
    //   replaced by a prepared copy.
    // Detects: telling the user to re-download what is already on disk, or a hint that
    //   names the wrong directory.
    // Needs: a temp directory.
    #[test]
    fn test_local_pytorch_directory_hint_converts_in_place() {
        let local = tempfile::tempdir().expect("tempdir");
        fs::write(local.path().join("pytorch_model-00001-of-00002.bin"), b"x").unwrap();
        let id = local.path().to_string_lossy().into_owned();
        let other = tempfile::tempdir().expect("tempdir");
        let resolved = resolve_model_path_with(&id, &[other.path().to_path_buf()], |_| {
            anyhow::bail!("a local directory needs no download")
        })
        .expect("local directory resolves to itself");
        assert_eq!(resolved, local.path());

        let hint = weights_preparation_hint(&id, &resolved);
        assert!(!hint.contains("Get-JanusProModel"), "{hint}");
        assert!(hint.contains("Tools/convert-sharded-weights.py"), "{hint}");
        assert!(hint.contains(&id), "{hint}");
    }

    // Protects: a corrupt or truncated checkpoint is a typed error at preflight and at
    //   load, never a panic and never a silently random-weight model.
    // Detects: header parsing that unwraps, or a loader that treats an unreadable archive
    //   as "no tensors present, use random weights".
    // Needs: a temp directory only; no model, GPU or network.
    #[test]
    fn test_corrupt_and_truncated_safetensors_are_rejected() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let corrupt = tmp.path().join("corrupt.safetensors");
        fs::write(&corrupt, b"this is not a safetensors archive at all").expect("write corrupt fixture");

        let valid = tmp.path().join("valid.safetensors");
        weight_fixture(
            &valid,
            &[("first.weight", vec![1.0, 2.0]), ("second.weight", vec![3.0, 4.0])],
        );
        let bytes = fs::read(&valid).expect("read valid fixture");
        let truncated = tmp.path().join("truncated.safetensors");
        fs::write(&truncated, &bytes[..bytes.len() - 6]).expect("write truncated fixture");

        for path in [corrupt, truncated] {
            let preflight = validate_weight_metadata(std::slice::from_ref(&path), &required_metadata());
            assert!(preflight.is_err(), "preflight must reject {}", path.display());
            let vars = required_variables();
            let loaded = load_weights(&vars, &[path], DType::F32, &Device::Cpu);
            assert!(loaded.is_err(), "load must reject a damaged archive");
            let data = vars.data().lock().expect("varmap lock");
            assert_eq!(
                data["first.weight"].as_tensor().to_vec1::<f32>().unwrap(),
                vec![-1.0, -1.0],
                "a rejected archive must not touch the model variables"
            );
        }
    }

    // Protects: the metadata preflight itself refuses to run with nothing to check.
    // Detects: an empty expected-tensor map making every checkpoint pass vacuously.
    // Needs: a temp directory only.
    #[test]
    fn test_weight_metadata_rejects_empty_requirements() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("model.safetensors");
        weight_fixture(&path, &[("first.weight", vec![1.0, 2.0])]);
        let error = validate_weight_metadata(&[path], &std::collections::BTreeMap::new()).unwrap_err();
        assert!(error.to_string().contains("no required generation tensor metadata"));
    }

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

    /// `open_safetensors` successfully maps a single valid file.
    #[test]
    fn test_open_safetensors_single_file() {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().expect("tempdir");
        let file_path = tmp.path().join("model.safetensors");

        let data: Vec<u8> = vec![0; 16]; // 4 floats
        let tensor = TensorView::new(Dtype::F32, vec![4], &data).unwrap();
        let mut tensors = HashMap::new();
        tensors.insert("dummy_tensor".to_string(), tensor);

        safetensors::serialize_to_file(&tensors, None, &file_path).expect("serialize");

        let archive = open_safetensors(&[file_path]).expect("open_safetensors");
        assert!(archive.tensors().iter().any(|(name, _)| name == "dummy_tensor"));
    }

    /// `open_safetensors` successfully maps multiple shards.
    #[test]
    fn test_open_safetensors_multiple_files() {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let tmp = tempfile::tempdir().expect("tempdir");
        let path1 = tmp.path().join("model-00001-of-00002.safetensors");
        let path2 = tmp.path().join("model-00002-of-00002.safetensors");

        let data1: Vec<u8> = vec![0; 16];
        let tensor1 = TensorView::new(Dtype::F32, vec![4], &data1).unwrap();
        let mut tensors1 = HashMap::new();
        tensors1.insert("tensor_part_1".to_string(), tensor1);
        safetensors::serialize_to_file(&tensors1, None, &path1).expect("serialize path1");

        let data2: Vec<u8> = vec![1; 16];
        let tensor2 = TensorView::new(Dtype::F32, vec![4], &data2).unwrap();
        let mut tensors2 = HashMap::new();
        tensors2.insert("tensor_part_2".to_string(), tensor2);
        safetensors::serialize_to_file(&tensors2, None, &path2).expect("serialize path2");

        let archive = open_safetensors(&[path1, path2]).expect("open_safetensors multi");

        let mut found_1 = false;
        let mut found_2 = false;
        for (name, _) in archive.tensors() {
            if name == "tensor_part_1" {
                found_1 = true;
            }
            if name == "tensor_part_2" {
                found_2 = true;
            }
        }
        assert!(found_1 && found_2, "Must find both tensors from multi archive");
    }

    /// `open_safetensors` returns an error when the file is invalid (e.g., empty or bad format).
    #[test]
    fn test_open_safetensors_invalid_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file_path = tmp.path().join("invalid.safetensors");

        // Write random invalid data
        std::fs::write(&file_path, b"not a safetensor archive").expect("write invalid data");

        let result = open_safetensors(&[file_path]);
        assert!(result.is_err(), "expected error for invalid file");
    }
}
