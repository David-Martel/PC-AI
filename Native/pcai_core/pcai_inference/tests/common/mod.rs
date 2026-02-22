//! Shared test utilities for pcai-inference integration tests
//!
//! This module provides common functionality used across multiple test files.

use std::path::PathBuf;

fn load_config_model_path() -> Option<PathBuf> {
    let path = PathBuf::from("Config/llm-config.json");
    let raw = std::fs::read_to_string(path).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let model_path = doc
        .get("providers")
        .and_then(|p| p.get("pcai-native"))
        .and_then(|p| p.get("modelPath"))
        .and_then(|p| p.as_str())?;
    let path_buf = PathBuf::from(model_path);
    if path_buf.exists() {
        Some(path_buf)
    } else {
        None
    }
}

/// Find a test GGUF model in common locations
///
/// Searches in order:
/// 1. Config/llm-config.json providers.pcai-native.modelPath
/// 2. Ollama cache (~/.ollama/models/blobs)
/// 3. LM Studio cache (~/.cache/lm-studio/models)
/// 4. Windows LOCALAPPDATA\lm-studio\models
///
/// Returns the first GGUF file found, or None if no model is available.
pub fn find_test_model() -> Option<PathBuf> {
    // Check llm-config.json first
    if let Some(path) = load_config_model_path() {
        return Some(path);
    }

    // Check Ollama cache (Linux/Mac)
    if let Some(home) = dirs::home_dir() {
        let ollama_path = home.join(".ollama/models/blobs");
        if let Some(model) = find_gguf_in_dir(&ollama_path) {
            return Some(model);
        }

        // Check LM Studio cache (Linux/Mac)
        let lm_studio_path = home.join(".cache/lm-studio/models");
        if let Some(model) = find_gguf_in_dir(&lm_studio_path) {
            return Some(model);
        }
    }

    // Check LM Studio cache (Windows)
    if let Some(localappdata) = dirs::data_local_dir() {
        let lm_studio_path = localappdata.join("lm-studio\\models");
        if let Some(model) = find_gguf_in_dir(&lm_studio_path) {
            return Some(model);
        }
    }

    None
}

/// Find the first .gguf file in a directory (recursive)
fn find_gguf_in_dir(dir: &PathBuf) -> Option<PathBuf> {
    if !dir.exists() || !dir.is_dir() {
        return None;
    }

    // Try direct children first
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            // If it's a GGUF file, return it
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                return Some(path);
            }

            // If it's a directory, search recursively (limit depth to avoid deep scans)
            if path.is_dir() {
                if let Some(model) = find_gguf_in_dir(&path) {
                    return Some(model);
                }
            }
        }
    }

    None
}

/// Get the model path from environment or panic with helpful message
///
/// Use this in tests that require a real model file.
pub fn require_test_model() -> PathBuf {
    find_test_model().unwrap_or_else(|| {
        panic!(
            "No test model found. Set providers.pcai-native.modelPath in Config/llm-config.json, \
             or install a model via Ollama/LM Studio."
        )
    })
}

/// Check if a test model is available
pub fn has_test_model() -> bool {
    find_test_model().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_test_model_respects_env() {
        let found = find_test_model();
        let config_path = load_config_model_path();
        if let Some(expected) = config_path {
            assert!(found.is_some());
            assert_eq!(found.unwrap(), expected);
        }
    }

    #[test]
    fn test_has_test_model() {
        // This should not panic
        let _ = has_test_model();
    }
}
