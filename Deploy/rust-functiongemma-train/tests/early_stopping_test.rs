use rust_functiongemma_train::early_stopping::{EarlyStopping, EarlyStoppingConfig};

#[test]
fn test_early_stopping_patience() {
    let config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 0.001,
    };
    let mut stopper = EarlyStopping::new(config);

    // Improving losses - should not stop
    assert!(!stopper.should_stop(1.0));
    assert!(!stopper.should_stop(0.9));
    assert!(!stopper.should_stop(0.8));

    // Stagnant losses within min_delta
    assert!(!stopper.should_stop(0.8001)); // patience 1
    assert!(!stopper.should_stop(0.8002)); // patience 2
    assert!(stopper.should_stop(0.8003)); // patience 3 -> stop
}

#[test]
fn test_early_stopping_improvement_resets() {
    let config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 0.01,
    };
    let mut stopper = EarlyStopping::new(config);

    stopper.should_stop(1.0);
    stopper.should_stop(1.0); // no improvement #1
    assert!(!stopper.should_stop(1.0)); // no improvement #2, counter=2, don't stop yet

    stopper.should_stop(0.5); // improvement - resets counter to 0
    assert!(!stopper.should_stop(0.5)); // no improvement #1 after reset, counter=1
    assert!(!stopper.should_stop(0.5)); // no improvement #2 after reset, counter=2
    assert!(stopper.should_stop(0.5)); // no improvement #3 after reset, counter=3, should stop
}

#[test]
fn test_early_stopping_zero_patience() {
    let config = EarlyStoppingConfig {
        patience: 0,
        min_delta: 0.001,
    };
    let mut stopper = EarlyStopping::new(config);

    // With zero patience, should stop immediately on first non-improvement
    assert!(!stopper.should_stop(1.0)); // First call sets best
    assert!(stopper.should_stop(1.0)); // No improvement -> stop
}

#[test]
fn test_early_stopping_best_loss() {
    let config = EarlyStoppingConfig {
        patience: 3,
        min_delta: 0.01,
    };
    let mut stopper = EarlyStopping::new(config);

    stopper.should_stop(1.0);
    assert_eq!(stopper.best_loss(), 1.0);

    stopper.should_stop(0.8);
    assert_eq!(stopper.best_loss(), 0.8);

    stopper.should_stop(0.85); // No improvement
    assert_eq!(stopper.best_loss(), 0.8); // Best remains at 0.8
}

#[test]
fn test_early_stopping_min_delta_boundary() {
    let config = EarlyStoppingConfig {
        patience: 2,
        min_delta: 0.1,
    };
    let mut stopper = EarlyStopping::new(config);

    stopper.should_stop(1.0); // best = 1.0
    assert!(!stopper.should_stop(0.91)); // improvement = 0.09 < 0.1 -> no improvement, counter=1
    assert!(stopper.should_stop(0.91)); // no improvement, counter=2 -> stop (patience reached)
}
