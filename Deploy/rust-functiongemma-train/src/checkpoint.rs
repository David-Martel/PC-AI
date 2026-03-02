use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Checkpoint metadata for training state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Checkpoint {
    /// Current training epoch
    pub epoch: usize,
    /// Global training step count
    pub global_step: usize,
    /// Best validation loss achieved
    pub best_loss: f64,
    /// Serialized optimizer state (weights, gradients, etc.)
    pub optimizer_state: Vec<f64>,
    /// Random number generator state for reproducibility
    pub rng_state: Option<u64>,
}

/// Configuration for checkpoint management
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory to save checkpoints
    pub output_dir: PathBuf,
    /// Save checkpoint every N steps
    pub save_every_n_steps: usize,
    /// Maximum number of checkpoints to keep (older ones are deleted)
    pub max_checkpoints: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./checkpoints"),
            save_every_n_steps: 500,
            max_checkpoints: 3,
        }
    }
}

impl Checkpoint {
    /// Save checkpoint to disk
    ///
    /// Creates a directory at the specified path and saves metadata.json
    ///
    /// # Arguments
    /// * `path` - Checkpoint directory path (e.g., "./checkpoints/checkpoint-100")
    ///
    /// # Errors
    /// Returns error if directory creation or file write fails
    pub fn save(&self, path: &Path) -> Result<()> {
        fs::create_dir_all(path)
            .with_context(|| format!("Failed to create checkpoint directory: {}", path.display()))?;

        let metadata_path = path.join("metadata.json");
        let json = serde_json::to_string_pretty(self).context("Failed to serialize checkpoint metadata")?;

        fs::write(&metadata_path, json)
            .with_context(|| format!("Failed to write checkpoint metadata to {}", metadata_path.display()))?;

        Ok(())
    }

    /// Load checkpoint from disk
    ///
    /// # Arguments
    /// * `path` - Checkpoint directory path
    ///
    /// # Errors
    /// Returns error if directory doesn't exist or metadata.json is invalid
    pub fn load(path: &Path) -> Result<Self> {
        let metadata_path = path.join("metadata.json");

        if !metadata_path.exists() {
            anyhow::bail!("Checkpoint metadata not found at {}", metadata_path.display());
        }

        let json = fs::read_to_string(&metadata_path)
            .with_context(|| format!("Failed to read checkpoint metadata from {}", metadata_path.display()))?;

        let checkpoint: Checkpoint = serde_json::from_str(&json)
            .with_context(|| format!("Failed to parse checkpoint metadata from {}", metadata_path.display()))?;

        Ok(checkpoint)
    }

    /// Find the latest checkpoint in a directory
    ///
    /// Searches for checkpoint-* directories and returns the one with the highest step number
    ///
    /// # Arguments
    /// * `dir` - Directory containing checkpoints
    ///
    /// # Errors
    /// Returns error if directory is empty or no valid checkpoints found
    pub fn find_latest(dir: &Path) -> Result<Self> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(dir).context("Failed to read checkpoint directory")? {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let dir_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            };

            // Parse checkpoint-{step} format
            if let Some(step_str) = dir_name.strip_prefix("checkpoint-") {
                if let Ok(step) = step_str.parse::<usize>() {
                    if let Ok(checkpoint) = Self::load(&path) {
                        checkpoints.push((step, checkpoint));
                    }
                }
            }
        }

        if checkpoints.is_empty() {
            anyhow::bail!("No valid checkpoints found in {}", dir.display());
        }

        // Sort by step number and return the highest
        checkpoints.sort_by_key(|(step, _)| *step);
        let (_, latest) = checkpoints.into_iter().last().expect("TODO: Verify unwrap");

        Ok(latest)
    }

    /// Clean up old checkpoints, keeping only the most recent max_checkpoints
    ///
    /// # Arguments
    /// * `config` - Checkpoint configuration with max_checkpoints setting
    ///
    /// # Errors
    /// Returns error if directory operations fail
    pub fn cleanup_old(config: &CheckpointConfig) -> Result<()> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&config.output_dir).context("Failed to read checkpoint directory")? {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();

            if !path.is_dir() {
                continue;
            }

            let dir_name = match path.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            };

            // Parse checkpoint-{step} format
            if let Some(step_str) = dir_name.strip_prefix("checkpoint-") {
                if let Ok(step) = step_str.parse::<usize>() {
                    checkpoints.push((step, path));
                }
            }
        }

        // Sort by step number
        checkpoints.sort_by_key(|(step, _)| *step);

        // Remove oldest checkpoints if we exceed max_checkpoints
        if checkpoints.len() > config.max_checkpoints {
            let to_remove = checkpoints.len() - config.max_checkpoints;
            for (_, path) in checkpoints.iter().take(to_remove) {
                fs::remove_dir_all(path)
                    .with_context(|| format!("Failed to remove old checkpoint at {}", path.display()))?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = Checkpoint {
            epoch: 5,
            global_step: 1000,
            best_loss: 0.25,
            optimizer_state: vec![1.0, 2.0, 3.0],
            rng_state: Some(42),
        };

        let json = serde_json::to_string(&checkpoint).expect("TODO: Verify unwrap");
        let deserialized: Checkpoint = serde_json::from_str(&json).expect("TODO: Verify unwrap");

        assert_eq!(checkpoint, deserialized);
    }
}
