use rust_functiongemma_train::checkpoint::{Checkpoint, CheckpointConfig};
use tempfile::TempDir;

#[test]
fn test_checkpoint_save_load() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");
    let checkpoint_path = temp_dir.path().join("checkpoint-100");

    let original = Checkpoint {
        epoch: 2,
        global_step: 100,
        best_loss: 0.5,
        optimizer_state: vec![0.1, 0.2, 0.3],
        rng_state: Some(12345),
    };

    original.save(&checkpoint_path).expect("TODO: Verify unwrap");
    let loaded = Checkpoint::load(&checkpoint_path).expect("TODO: Verify unwrap");

    assert_eq!(loaded.epoch, 2);
    assert_eq!(loaded.global_step, 100);
    assert!((loaded.best_loss - 0.5).abs() < 1e-6);
    assert_eq!(loaded.optimizer_state, vec![0.1, 0.2, 0.3]);
    assert_eq!(loaded.rng_state, Some(12345));
}

#[test]
fn test_checkpoint_find_latest() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");

    for step in [50, 100, 150] {
        let ckpt = Checkpoint {
            epoch: 1,
            global_step: step,
            best_loss: 1.0,
            optimizer_state: vec![],
            rng_state: None,
        };
        ckpt.save(&temp_dir.path().join(format!("checkpoint-{}", step))).expect("TODO: Verify unwrap");
    }

    let latest = Checkpoint::find_latest(temp_dir.path()).expect("TODO: Verify unwrap");
    assert_eq!(latest.global_step, 150);
}

#[test]
fn test_checkpoint_cleanup_old() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");

    // Create 5 checkpoints
    for step in [50, 100, 150, 200, 250] {
        let ckpt = Checkpoint {
            epoch: 1,
            global_step: step,
            best_loss: 1.0,
            optimizer_state: vec![],
            rng_state: None,
        };
        ckpt.save(&temp_dir.path().join(format!("checkpoint-{}", step))).expect("TODO: Verify unwrap");
    }

    let config = CheckpointConfig {
        output_dir: temp_dir.path().to_path_buf(),
        save_every_n_steps: 50,
        max_checkpoints: 3,
    };

    Checkpoint::cleanup_old(&config).expect("TODO: Verify unwrap");

    // Should keep only the latest 3
    let remaining: Vec<_> = std::fs::read_dir(temp_dir.path()).expect("TODO: Verify unwrap")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    assert_eq!(remaining.len(), 3);
    assert!(remaining.contains(&"checkpoint-150".to_string()));
    assert!(remaining.contains(&"checkpoint-200".to_string()));
    assert!(remaining.contains(&"checkpoint-250".to_string()));
}

#[test]
fn test_checkpoint_load_nonexistent() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");
    let result = Checkpoint::load(&temp_dir.path().join("nonexistent"));
    assert!(result.is_err());
}

#[test]
fn test_checkpoint_find_latest_empty_dir() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");
    let result = Checkpoint::find_latest(temp_dir.path());
    assert!(result.is_err());
}
