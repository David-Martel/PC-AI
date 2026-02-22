use rust_functiongemma_train::{
    lora::LoraConfig,
    scheduler::{LRScheduler, SchedulerConfig, SchedulerType},
    checkpoint::{Checkpoint, CheckpointConfig},
    early_stopping::{EarlyStopping, EarlyStoppingConfig},
    trainer::TrainerConfig,
};
use tempfile::TempDir;

#[test]
fn test_full_training_config_integration() {
    // Verify all components can be configured together
    let trainer_cfg = TrainerConfig {
        lr: 1e-4,
        epochs: 3,
        batch_size: 4,
        grad_accum: 4,
        lora_r: 8,
        lora_alpha: 16.0,
        lora_dropout: 0.05,
        pack_sequences: true,
        max_seq_len: Some(512),
        eos_token_id: 2,
        use_lora: true,
        warmup_steps: 100,
        scheduler_type: "cosine".to_string(),
        early_stopping: None,
        use_4bit: false,
    };

    // Verify LoRA config can be derived
    let lora_cfg = LoraConfig {
        r: trainer_cfg.lora_r,
        alpha: trainer_cfg.lora_alpha,
        dropout: trainer_cfg.lora_dropout,
        target_modules: vec!["q_proj".to_string(), "k_proj".to_string()],
    };

    // Verify scheduler config
    let scheduler_cfg = SchedulerConfig {
        scheduler_type: SchedulerType::Cosine,
        warmup_steps: trainer_cfg.warmup_steps,
        total_steps: 1000,
        min_lr: 1e-6,
        max_lr: trainer_cfg.lr,
    };
    let scheduler = LRScheduler::new(scheduler_cfg);

    // Verify checkpoint config
    let _checkpoint_cfg = CheckpointConfig::default();

    // Verify early stopping
    let early_stopping_cfg = EarlyStoppingConfig::default();
    let mut stopper = EarlyStopping::new(early_stopping_cfg);

    // All should be compatible
    assert!(trainer_cfg.use_lora);
    assert_eq!(lora_cfg.r, 8);
    assert!(scheduler.get_lr(0) < trainer_cfg.lr);
    assert!(scheduler.get_lr(100) > scheduler.get_lr(0)); // After warmup
    assert!(!stopper.should_stop(1.0)); // First loss shouldn't stop
}

#[test]
fn test_checkpoint_workflow() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");

    // Create a checkpoint
    let checkpoint = Checkpoint {
        epoch: 2,
        global_step: 500,
        best_loss: 0.25,
        optimizer_state: vec![],
        rng_state: Some(42),
    };

    let ckpt_path = temp_dir.path().join("checkpoint-500");
    checkpoint.save(&ckpt_path).expect("TODO: Verify unwrap");

    // Verify it can be loaded
    let loaded = Checkpoint::load(&ckpt_path).expect("TODO: Verify unwrap");
    assert_eq!(loaded.global_step, 500);
    assert_eq!(loaded.epoch, 2);
    assert_eq!(loaded.best_loss, 0.25);
    assert_eq!(loaded.rng_state, Some(42));
}

#[test]
fn test_training_components_compatibility() {
    // Test that all module types can be used together
    let _trainer_cfg = TrainerConfig::default();
    let _lora_cfg = LoraConfig::default();
    let _scheduler_cfg = SchedulerConfig::default();
    let _checkpoint_cfg = CheckpointConfig::default();
    let _early_stopping_cfg = EarlyStoppingConfig::default();

    // If this compiles, the types are compatible
    assert!(true);
}

#[test]
fn test_scheduler_warmup_and_decay() {
    let config = SchedulerConfig {
        scheduler_type: SchedulerType::Cosine,
        warmup_steps: 100,
        total_steps: 1000,
        min_lr: 1e-6,
        max_lr: 1e-4,
    };
    let scheduler = LRScheduler::new(config);

    // During warmup, LR should increase
    let lr_0 = scheduler.get_lr(0);
    let lr_50 = scheduler.get_lr(50);
    let lr_100 = scheduler.get_lr(100);

    assert!(lr_50 > lr_0, "LR should increase during warmup");
    assert!(lr_100 >= lr_50, "LR should be max at end of warmup");

    // After warmup, LR should decrease
    let lr_500 = scheduler.get_lr(500);
    let lr_1000 = scheduler.get_lr(1000);

    assert!(lr_500 < lr_100, "LR should decrease after warmup");
    assert!(lr_1000 <= lr_500, "LR should continue decreasing");
}

#[test]
fn test_lora_config_derivation() {
    let trainer_cfg = TrainerConfig {
        lr: 2e-4,
        epochs: 5,
        batch_size: 8,
        grad_accum: 2,
        lora_r: 16,
        lora_alpha: 32.0,
        lora_dropout: 0.1,
        pack_sequences: true,
        max_seq_len: Some(1024),
        eos_token_id: 2,
        use_lora: true,
        warmup_steps: 200,
        scheduler_type: "linear".to_string(),
        early_stopping: None,
        use_4bit: false,
    };

    // Derive LoRA config from trainer config
    let lora_cfg = LoraConfig {
        r: trainer_cfg.lora_r,
        alpha: trainer_cfg.lora_alpha,
        dropout: trainer_cfg.lora_dropout,
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
    };

    assert_eq!(lora_cfg.r, 16);
    assert_eq!(lora_cfg.alpha, 32.0);
    assert_eq!(lora_cfg.dropout, 0.1);
    assert_eq!(lora_cfg.target_modules.len(), 4);
}

#[test]
fn test_early_stopping_integration() {
    let config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 0.01,
    };
    let mut stopper = EarlyStopping::new(config);

    // Simulate training losses
    // For counter to increment, improvement must be <= min_delta (0.01)
    // improvement = best_loss - val_loss
    assert!(!stopper.should_stop(1.0)); // First epoch: best=1.0, counter=0
    assert!(!stopper.should_stop(0.8)); // Improvement: 1.0-0.8=0.2 > 0.01, best=0.8, counter=0
    assert!(!stopper.should_stop(0.795)); // Improvement: 0.8-0.795=0.005 <= 0.01, counter=1
    assert!(!stopper.should_stop(0.793)); // Improvement: 0.8-0.793=0.007 <= 0.01, counter=2
    assert!(stopper.should_stop(0.791)); // Improvement: 0.8-0.791=0.009 <= 0.01, counter=3 >= patience

    // Verify best loss tracked correctly (stayed at 0.8 since no significant improvement)
    assert_eq!(stopper.best_loss(), 0.8);
}

#[test]
fn test_checkpoint_cleanup() {
    let temp_dir = TempDir::new().expect("TODO: Verify unwrap");
    let checkpoint_config = CheckpointConfig {
        output_dir: temp_dir.path().to_path_buf(),
        save_every_n_steps: 100,
        max_checkpoints: 2,
    };

    // Create 3 checkpoints
    for step in [100, 200, 300] {
        let checkpoint = Checkpoint {
            epoch: 1,
            global_step: step,
            best_loss: 0.5,
            optimizer_state: vec![],
            rng_state: None,
        };
        let ckpt_path = temp_dir.path().join(format!("checkpoint-{}", step));
        checkpoint.save(&ckpt_path).expect("TODO: Verify unwrap");
    }

    // Cleanup should keep only the 2 most recent
    Checkpoint::cleanup_old(&checkpoint_config).expect("TODO: Verify unwrap");

    // Verify checkpoint-100 was removed
    let ckpt_100 = temp_dir.path().join("checkpoint-100");
    assert!(!ckpt_100.exists(), "Oldest checkpoint should be removed");

    // Verify checkpoint-200 and checkpoint-300 still exist
    let ckpt_200 = temp_dir.path().join("checkpoint-200");
    let ckpt_300 = temp_dir.path().join("checkpoint-300");
    assert!(ckpt_200.exists(), "Second checkpoint should remain");
    assert!(ckpt_300.exists(), "Most recent checkpoint should remain");
}

#[test]
fn test_scheduler_types() {
    let warmup_steps = 50;
    let total_steps = 500;
    let min_lr = 1e-6;
    let max_lr = 1e-4;

    // Test Cosine scheduler
    let cosine_cfg = SchedulerConfig {
        scheduler_type: SchedulerType::Cosine,
        warmup_steps,
        total_steps,
        min_lr,
        max_lr,
    };
    let cosine_scheduler = LRScheduler::new(cosine_cfg);
    let cosine_mid = cosine_scheduler.get_lr(250);

    // Test Linear scheduler
    let linear_cfg = SchedulerConfig {
        scheduler_type: SchedulerType::Linear,
        warmup_steps,
        total_steps,
        min_lr,
        max_lr,
    };
    let linear_scheduler = LRScheduler::new(linear_cfg);
    let linear_mid = linear_scheduler.get_lr(250);

    // Test Constant scheduler
    let constant_cfg = SchedulerConfig {
        scheduler_type: SchedulerType::Constant,
        warmup_steps,
        total_steps,
        min_lr,
        max_lr,
    };
    let constant_scheduler = LRScheduler::new(constant_cfg);
    let constant_mid = constant_scheduler.get_lr(250);

    // All should be different strategies
    assert!(cosine_mid != linear_mid, "Cosine and linear should differ");
    assert_eq!(constant_mid, max_lr, "Constant should stay at max_lr");
}

#[test]
fn test_full_pipeline_simulation() {
    // Simulate a complete training configuration
    let trainer_cfg = TrainerConfig::default();
    let lora_cfg = LoraConfig {
        r: trainer_cfg.lora_r,
        alpha: trainer_cfg.lora_alpha,
        dropout: trainer_cfg.lora_dropout,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
    };

    let scheduler_cfg = SchedulerConfig {
        scheduler_type: SchedulerType::Cosine,
        warmup_steps: trainer_cfg.warmup_steps,
        total_steps: trainer_cfg.epochs * 100, // Assume 100 steps per epoch
        min_lr: trainer_cfg.lr / 10.0,
        max_lr: trainer_cfg.lr,
    };
    let scheduler = LRScheduler::new(scheduler_cfg);

    let checkpoint_cfg = CheckpointConfig::default();
    let early_stopping_cfg = EarlyStoppingConfig::default();
    let mut stopper = EarlyStopping::new(early_stopping_cfg);

    // Simulate training steps
    let mut global_step = 0;
    for epoch in 0..trainer_cfg.epochs {
        for batch in 0..10 {
            let current_lr = scheduler.get_lr(global_step);
            global_step += 1;

            // Verify LR is within bounds
            assert!(current_lr >= scheduler_cfg.min_lr);
            assert!(current_lr <= scheduler_cfg.max_lr);
        }

        // Simulate epoch validation
        let val_loss = 1.0 / (epoch + 1) as f64;
        if stopper.should_stop(val_loss) {
            break;
        }
    }

    // Verify all components worked together
    assert!(global_step > 0);
    assert_eq!(lora_cfg.r, 8);
    assert_eq!(checkpoint_cfg.max_checkpoints, 3);
}
