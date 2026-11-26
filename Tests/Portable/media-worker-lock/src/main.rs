use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

#[derive(Debug, PartialEq, Eq)]
enum MediaRequestStatus {
    Pending,
    Running,
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
            requests: HashMap::from([(7, MediaRequestStatus::Pending), (8, MediaRequestStatus::Cancelled)]),
        })
    })
}

// `lock_state_recovering` and `begin_async_request`, copied verbatim from the
// maintained media FFI source by build.rs.
include!(concat!(env!("OUT_DIR"), "/worker_acquisition.rs"));

fn healthy(id: i64) {
    let state = get_state();
    let started = begin_async_request(id);
    assert!(!state.is_poisoned(), "healthy acquisition must not poison state");
    // The transition must not return holding the lock: poll and cancel take
    // only this lock, so they stay responsive while generation runs.
    assert!(
        state.try_lock().is_ok(),
        "state lock must be released when the transition returns"
    );
    let guard = state.lock().expect("healthy fixture state"); // ast-grep-ignore: mandate-rust-panic // SAFETY: Owned std-only child assertion; healthy setup must be unpoisoned.
    if id == 7 {
        assert!(started, "a pending request starts");
        assert_eq!(guard.requests.get(&7), Some(&MediaRequestStatus::Running));
    } else {
        assert!(!started, "a cancelled request must not start");
        assert_eq!(guard.requests.get(&7), Some(&MediaRequestStatus::Pending));
    }
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
    assert!(state.lock().is_err(), "fixture state must retain poison");
    eprintln!("PCAI_OWNED_POISON_ACQUISITION_ENTER");
    // Recovery consumes the held guard through into_inner; relocking the
    // poisoned mutex here would deadlock and trip the supervisor's timeout.
    let started = begin_async_request(7);
    eprintln!("PCAI_OWNED_POISON_ACQUISITION_RETURNED");
    assert!(started, "recovery must let the pending request start");
    assert!(state.is_poisoned(), "recovery must not silently clear poison");
    let guard = state
        .lock()
        .expect_err("recovery must preserve poisoned-state refusal")
        .into_inner();
    assert_eq!(guard.requests.get(&7), Some(&MediaRequestStatus::Running));
    assert_eq!(guard.requests.get(&8), Some(&MediaRequestStatus::Cancelled));
    assert_eq!(guard.requests.len(), 2);
    // Marker name kept for the supervisor; it now reports recovered state.
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
