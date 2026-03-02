//! Utilities for collecting and loading safetensors model files.
//!
//! Both the runtime and training crates require the same logic for discovering
//! sharded model weights on disk and loading them into a [`candle_nn::VarMap`].
//! This module provides the canonical implementations shared across crates.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::safetensors::MmapedSafetensors;
use candle_nn::VarMap;

/// Open one or more safetensors files as a memory-mapped archive.
///
/// When a single path is provided the simpler [`MmapedSafetensors::new`] path
/// is used.  For multiple shards [`MmapedSafetensors::multi`] is called.
///
/// # Safety
///
/// This function calls the unsafe [`MmapedSafetensors`] API, which
/// memory-maps the files.  The caller must ensure that no concurrent writer
/// modifies the safetensors files while the returned object is alive.
///
/// # Errors
///
/// Returns an error if `paths` is empty or if any I/O operation fails.
pub fn open_mmaped_safetensors(paths: &[PathBuf]) -> Result<MmapedSafetensors> {
    if paths.is_empty() {
        return Err(anyhow::anyhow!("no safetensors files provided"));
    }
    let st = unsafe {
        if paths.len() == 1 {
            MmapedSafetensors::new(&paths[0])?
        } else {
            MmapedSafetensors::multi(paths)?
        }
    };
    Ok(st)
}

/// Detect whether the model ties its embedding and output head weights.
///
/// Returns `true` when the safetensors archive does **not** contain a
/// separate `lm_head.weight` tensor, meaning the model re-uses the
/// embedding matrix for the output projection.
pub fn detect_tie_embeddings(st: &MmapedSafetensors) -> bool {
    !st.tensors().iter().any(|(name, _)| name == "lm_head.weight")
}

/// Collect all model safetensors files from a directory.
///
/// Checks for a single `model.safetensors` first. If that file does not exist
/// the function collects all `model-*.safetensors` shards in lexicographic
/// (sorted) order, which matches the ordering used by HuggingFace sharded
/// checkpoints.
///
/// # Examples
///
/// ```no_run
/// use std::path::PathBuf;
/// use rust_functiongemma_core::safetensors_utils::collect_model_safetensors;
///
/// let files = collect_model_safetensors(PathBuf::from("/path/to/model").as_path());
/// for f in &files {
///     println!("{}", f.display());
/// }
/// ```
pub fn collect_model_safetensors(model_path: &Path) -> Vec<PathBuf> {
    let direct = model_path.join("model.safetensors");
    if direct.exists() {
        return vec![direct];
    }
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|v| v.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

/// Detect the safetensors key prefix (e.g. `"model"`) by examining tensor names.
///
/// Many HuggingFace checkpoints store weights under a top-level namespace such
/// as `model.layers.0.self_attn.q_proj.weight`. This function heuristically
/// identifies that prefix so callers can strip or prepend it when looking up
/// tensor names in a [`candle_nn::VarMap`].
///
/// Returns [`None`] if no recognisable prefix is found.
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use rust_functiongemma_core::safetensors_utils::detect_safetensors_prefix;
///
/// let names: HashSet<String> = [
///     "model.embed_tokens.weight".to_string(),
///     "model.layers.0.self_attn.q_proj.weight".to_string(),
/// ]
/// .into_iter()
/// .collect();
///
/// assert_eq!(detect_safetensors_prefix(&names), Some("model".to_string()));
/// ```
pub fn detect_safetensors_prefix(st_names: &HashSet<String>) -> Option<String> {
    for name in st_names {
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

/// Load safetensors files into a [`VarMap`] with automatic prefix detection.
///
/// The function memory-maps the provided paths (using [`MmapedSafetensors`])
/// and iterates over all variables currently registered in `varmap`. For each
/// variable it first attempts a direct key lookup (no prefix). If fewer than a
/// third of the variables are found that way it falls back to a prefix-based
/// lookup using [`detect_safetensors_prefix`].
///
/// Returns the number of variables successfully loaded from the files.
///
/// # Errors
///
/// Returns an error if `paths` is empty or if any I/O or tensor-cast operation
/// fails.
///
/// # Safety
///
/// Internally this function calls the unsafe
/// [`candle_core::safetensors::MmapedSafetensors`] API, which memory-maps the
/// files. The invariant required is that the files must not be modified while
/// this function is running. Callers must ensure that no concurrent writer
/// modifies the safetensors files.
///
/// # Examples
///
/// ```no_run
/// use candle_nn::VarMap;
/// use std::path::PathBuf;
/// use rust_functiongemma_core::safetensors_utils::custom_load;
///
/// let varmap = VarMap::new();
/// let paths = vec![PathBuf::from("/path/to/model.safetensors")];
/// let count = custom_load(&varmap, &paths).expect("load failed");
/// println!("Loaded {} variables", count);
/// ```
pub fn custom_load(varmap: &VarMap, paths: &[PathBuf]) -> Result<usize> {
    let st = open_mmaped_safetensors(paths)?;

    let st_names: HashSet<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();

    let data = varmap.data().lock().expect("TODO: Verify unwrap");

    // Inner helper: attempt to load all vars using the given optional prefix.
    // Returns the count of successfully updated variables.
    let load_with_prefix = |prefix: Option<&str>| -> Result<usize> {
        let mut count = 0usize;
        for (name, var) in data.iter() {
            let lookup = match prefix {
                Some(pfx) if !pfx.is_empty() => format!("{pfx}.{name}"),
                _ => name.clone(),
            };
            if st_names.contains(&lookup) {
                let mut st_tensor = st.load(&lookup, &var.device())?;
                if st_tensor.dtype() != var.dtype() {
                    st_tensor = st_tensor.to_dtype(var.dtype())?;
                }
                var.set(&st_tensor)?;
                count += 1;
            }
        }
        Ok(count)
    };

    let updated_direct = load_with_prefix(None)?;
    let mut updated = updated_direct;

    // If fewer than a third of variables were found without a prefix, try
    // detecting and applying a prefix. Only adopt the prefixed result if it
    // yields more matches.
    if updated < (data.len() / 3).max(1) {
        if let Some(prefix) = detect_safetensors_prefix(&st_names) {
            let prefixed = load_with_prefix(Some(prefix.as_str()))?;
            if prefixed > updated {
                updated = prefixed;
            }
        }
    }

    Ok(updated)
}

/// Like [`custom_load`] but with optional verbose diagnostic output.
///
/// When `verbose` is `true`, prints sample safetensors keys before loading.
/// Always prints a summary of loaded vs total variables and any applied prefix.
/// When fewer variables are loaded than expected, prints sample missing keys.
///
/// This is the preferred entry point for training and evaluation commands where
/// visibility into the weight-loading process is important.
pub fn custom_load_verbose(varmap: &VarMap, paths: &[PathBuf], verbose: bool) -> Result<usize> {
    let st = open_mmaped_safetensors(paths)?;

    let st_names: HashSet<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();

    if verbose {
        println!(
            "Safetensors sample keys: {:?}",
            st_names.iter().take(5).collect::<Vec<_>>()
        );
    }

    let data = varmap.data().lock().expect("TODO: Verify unwrap");

    let load_with_prefix = |prefix: Option<&str>| -> Result<usize> {
        let mut count = 0usize;
        for (name, var) in data.iter() {
            let lookup = match prefix {
                Some(pfx) if !pfx.is_empty() => format!("{pfx}.{name}"),
                _ => name.clone(),
            };
            if st_names.contains(&lookup) {
                let mut st_tensor = st.load(&lookup, &var.device())?;
                if st_tensor.dtype() != var.dtype() {
                    st_tensor = st_tensor.to_dtype(var.dtype())?;
                }
                var.set(&st_tensor)?;
                count += 1;
            }
        }
        Ok(count)
    };

    let updated_direct = load_with_prefix(None)?;
    let mut updated = updated_direct;
    let mut applied_prefix: Option<String> = None;
    if updated < (data.len() / 3).max(1) {
        if let Some(prefix) = detect_safetensors_prefix(&st_names) {
            let prefixed = load_with_prefix(Some(prefix.as_str()))?;
            if prefixed > updated {
                updated = prefixed;
                applied_prefix = Some(prefix);
            }
        }
    }
    if let Some(prefix) = applied_prefix.as_deref() {
        println!("Applied safetensors prefix '{}'", prefix);
    }
    println!(
        "Loaded {}/{} variables from {:?} files",
        updated,
        data.len(),
        paths.len()
    );
    if updated < data.len() {
        let sample_keys: Vec<_> = st_names.iter().take(5).cloned().collect();
        println!("Safetensors sample keys: {:?}", sample_keys);
        let mut missing = Vec::new();
        for (name, _) in data.iter() {
            let lookup = match applied_prefix.as_deref() {
                Some(pfx) if !pfx.is_empty() => format!("{pfx}.{name}"),
                _ => name.clone(),
            };
            if !st_names.contains(&lookup) {
                missing.push(name.clone());
            }
            if missing.len() >= 5 {
                break;
            }
        }
        if !missing.is_empty() {
            println!("Missing var sample: {:?}", missing);
        }
    }
    Ok(updated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn detect_prefix_model_dot() {
        let names: HashSet<String> = ["model.embed_tokens.weight".to_string()].into_iter().collect();
        assert_eq!(detect_safetensors_prefix(&names), Some("model".to_string()));
    }

    #[test]
    fn detect_prefix_via_layers() {
        let names: HashSet<String> = ["transformer.layers.0.attn.weight".to_string()].into_iter().collect();
        assert_eq!(detect_safetensors_prefix(&names), Some("transformer".to_string()));
    }

    #[test]
    fn detect_prefix_via_embed_tokens() {
        let names: HashSet<String> = ["encoder.embed_tokens.weight".to_string()].into_iter().collect();
        assert_eq!(detect_safetensors_prefix(&names), Some("encoder".to_string()));
    }

    #[test]
    fn detect_prefix_none_when_no_match() {
        let names: HashSet<String> = ["some_random_key".to_string()].into_iter().collect();
        assert_eq!(detect_safetensors_prefix(&names), None);
    }

    #[test]
    fn collect_model_safetensors_empty_dir() {
        let dir = std::env::temp_dir().join("pcai_test_collect_empty");
        std::fs::create_dir_all(&dir).expect("TODO: Verify unwrap");
        let files = collect_model_safetensors(&dir);
        assert!(files.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
