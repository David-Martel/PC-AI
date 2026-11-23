use rust_functiongemma_train::trainer::TrainerConfig;

#[test]
fn test_trainer_with_lora_config() {
    let config = TrainerConfig::default();
    assert!(config.use_lora);
    assert_eq!(config.lora_r, 8);
    assert_eq!(config.warmup_steps, 100);
}

#[test]
fn test_trainer_config_lora_fields() {
    let config = TrainerConfig::default();

    // Verify LoRA parameters
    assert_eq!(config.lora_r, 8);
    assert_eq!(config.lora_alpha, 16.0);
    assert_eq!(config.lora_dropout, 0.0);
    assert!(config.use_lora);

    // Verify scheduler parameters
    assert_eq!(config.warmup_steps, 100);
    assert_eq!(config.scheduler_type, "cosine");

    // Verify training parameters
    assert_eq!(config.lr, 1e-4);
    assert_eq!(config.epochs, 3);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.grad_accum, 4);

    // Verify sequence parameters
    assert!(config.pack_sequences);
    assert_eq!(config.max_seq_len, Some(512));
    assert_eq!(config.eos_token_id, 2);
}

// Protects: explicit TrainerConfig overrides.
// Detects: the fixture loses rank/dropout/alpha or optimizer schedule settings.
// Needs: an in-memory config. Breadcrumb: trainer::TrainerConfig.
#[test]
fn test_trainer_config_custom_values() {
    let config = TrainerConfig {
        lr: 5e-5,
        epochs: 5,
        batch_size: 8,
        grad_accum: 2,
        lora_r: 16,
        lora_alpha: 32.0,
        lora_dropout: 0.1,
        pack_sequences: false,
        max_seq_len: Some(1024),
        eos_token_id: 1,
        use_lora: false,
        warmup_steps: 200,
        scheduler_type: "linear".to_string(),
        early_stopping: None,
        use_4bit: false,
        ..TrainerConfig::default()
    };

    assert_eq!(config.lora_r, 16);
    assert_eq!(config.lora_alpha, 32.0);
    assert_eq!(config.lora_dropout, 0.1);
    assert!(!config.use_lora);
    assert_eq!(config.warmup_steps, 200);
    assert_eq!(config.scheduler_type, "linear");
}

// Protects storing a scheduler choice in the public trainer configuration.
// Detects incompatible field or initialization changes.
// Needs only in-memory trainer settings. Breadcrumb: trainer::TrainerConfig::scheduler_type.
#[test]
fn test_trainer_config_scheduler_types() {
    let mut config = TrainerConfig {
        scheduler_type: "cosine".to_string(),
        ..TrainerConfig::default()
    };
    assert_eq!(config.scheduler_type, "cosine");

    config.scheduler_type = "linear".to_string();
    assert_eq!(config.scheduler_type, "linear");

    config.scheduler_type = "constant".to_string();
    assert_eq!(config.scheduler_type, "constant");
}
