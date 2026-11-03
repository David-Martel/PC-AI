use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, PartialEq, Eq)]
enum MediaRequestStatus {
    Running,
    Failed(String),
    Cancelled,
}

#[derive(Debug)]
struct MediaState {
    requests: HashMap<i64, MediaRequestStatus>,
}

static STATE: OnceLock<Mutex<MediaState>> = OnceLock::new();

fn get_state() -> &'static Mutex<MediaState> {
    STATE.get_or_init(|| {
        Mutex::new(MediaState {
            requests: HashMap::from([(7, MediaRequestStatus::Running), (8, MediaRequestStatus::Cancelled)]),
        })
    })
}

include!(concat!(env!("OUT_DIR"), "/worker_acquisition.rs"));

fn healthy(id: i64) {
    let state = get_state();
    run_actual_worker_acquisition(id);
    assert!(!state.is_poisoned(), "healthy acquisition must not poison state");
    let guard = state.lock().expect("healthy fixture state"); // ast-grep-ignore: mandate-rust-panic // SAFETY: Owned std-only child assertion; healthy setup must be unpoisoned.
    assert_eq!(guard.requests.get(&7), Some(&MediaRequestStatus::Running));
    assert_eq!(guard.requests.get(&8), Some(&MediaRequestStatus::Cancelled));
    assert_eq!(guard.requests.len(), 2);
    println!("PCAI_HEALTHY_ACQUISITION_COMPLETE id={id}");
}

fn poisoned() {
    let state = get_state();
    let result = std::panic::catch_unwind(|| {
        let _guard = state.lock().expect("fixture begins unpoisoned"); // ast-grep-ignore: mandate-rust-panic // SAFETY: Owned std-only child setup; the guard is dropped by the deliberate caught panic.
        panic!("owned model-state poisoning control"); // ast-grep-ignore: mandate-rust-panic // SAFETY: Test-only panic is caught to poison this child's mutex; parent bounds execution to 3s and reaps it.
    });
    assert!(result.is_err(), "fixture must actually poison its own mutex");
    let expected = {
        let error = state.lock().expect_err("fixture state must retain poison");
        let message = format!("Lock failed: {error}");
        drop(error);
        message
    };
    eprintln!("PCAI_OWNED_POISON_ACQUISITION_ENTER");
    run_actual_worker_acquisition(7);
    eprintln!("PCAI_OWNED_POISON_ACQUISITION_RETURNED");
    assert!(state.is_poisoned(), "failure must not silently resume poisoned state");
    let guard = state
        .lock()
        .expect_err("repair must preserve poisoned-state refusal")
        .into_inner();
    assert_eq!(guard.requests.get(&7), Some(&MediaRequestStatus::Failed(expected)));
    assert_eq!(guard.requests.get(&8), Some(&MediaRequestStatus::Cancelled));
    assert_eq!(guard.requests.len(), 2);
    println!("PCAI_OWNED_POISON_FAILURE_RECORDED");
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("healthy-running") => healthy(7),
        Some("healthy-cancelled") => healthy(8),
        Some("poisoned-running") => poisoned(),
        _ => {
            eprintln!("expected one owned fixture scenario");
            std::process::exit(2);
        }
    }
}
