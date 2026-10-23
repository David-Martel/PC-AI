//! Public poll/cancel contracts using the production request registry and owned inert workers.
use super::*;
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Fixture(Arc<Requests<RequestStatus>>);
impl Fixture {
    fn new() -> Self {
        let registry = Arc::new(Requests::new());
        TEST_REQUESTS.with(|slot| {
            assert!(slot.borrow().is_none());
            *slot.borrow_mut() = Some(Arc::clone(&registry));
        });
        Self(registry)
    }
    fn wait_terminal(&self, id: i64) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while self
            .0
            .entries()
            .get(&id)
            .is_some_and(|state| state.is_pending() || state.is_running())
        {
            assert!(Instant::now() < deadline, "owned worker did not finish");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.0.close_and_join();
        TEST_REQUESTS.with(|slot| *slot.borrow_mut() = None);
    }
}

// Fixture-owned readiness does not initialize any backend. Cancellation happens
// before the real worker can acquire the generation lock or execute a provider.
#[test]
fn public_async_admission_does_not_wait_for_generation_model_lock() {
    let fixture = Fixture::new();
    fixture.0.publish(true, true);
    let (held_tx, held_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let holder = std::thread::spawn(move || {
        let _guard = init_global_state().lock().unwrap();
        held_tx.send(()).unwrap();
        let _ = release_rx.recv_timeout(Duration::from_secs(3));
    });
    held_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    let prompt = CString::new("inert admission fixture").unwrap();
    let started = Instant::now();
    let id = unsafe { pcai_generate_async(prompt.as_ptr(), 1, 1.0) };
    let elapsed = started.elapsed();
    assert!(id > 0);
    assert!(
        elapsed < Duration::from_secs(1),
        "submission waited for the model lock: {elapsed:?}"
    );
    assert_eq!(pcai_cancel(id), 0);
    assert_eq!(pcai_poll_result(id).status, 4);
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    fixture.0.close_and_join();
    assert_eq!(pcai_poll_result(id).status, -1);
}

#[test]
fn public_async_refuses_missing_backend_model_and_stale_load_epoch() {
    let fixture = Fixture::new();
    let prompt = CString::new("inert admission fixture").unwrap();
    assert_eq!(unsafe { pcai_generate_async(prompt.as_ptr(), 1, 1.0) }, -1);
    assert_eq!(pcai_last_error_code(), PcaiErrorCode::NotInitialized as i32);
    fixture.0.publish(true, false);
    assert_eq!(unsafe { pcai_generate_async(prompt.as_ptr(), 1, 1.0) }, -1);
    assert_eq!(pcai_last_error_code(), PcaiErrorCode::ModelNotLoaded as i32);
    fixture.0.publish(true, true);
    let ticket = fixture.0.ready_ticket().unwrap();
    fixture.0.invalidate();
    assert_eq!(fixture.0.ready_ticket(), Err(AdmissionError::ModelNotLoaded));
    assert!(fixture
        .0
        .spawn(7, ticket, || panic!("stale load epoch must not execute"))
        .is_err());
}

// Protects complete exported poll/cancel bodies while generation owns its independent model lock.
// Detects blocking public controls and terminal resurrection after cancellation consumption.
// Needs only production registry plus a bounded inert worker, no model/GPU/runtime/provider.
#[test]
fn public_cancel_poll_remain_available_during_owned_generation_and_discard_late_result() {
    let fixture = Fixture::new();
    let model = Arc::new(Mutex::new(()));
    let worker_model = Arc::clone(&model);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    fixture
        .0
        .spawn(7, fixture.0.ticket().unwrap(), move || {
            let _guard = requests::model_lock(&worker_model).unwrap();
            started_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(3)).unwrap();
            RequestStatus::Complete {
                text: "owned-output".to_string(),
                tokens_generated: 2,
                prompt_tokens_estimated: 3,
            }
        })
        .unwrap();
    started_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(pcai_poll_result(7).status, 1);
    assert_eq!(pcai_cancel(7), 0);
    assert_eq!(pcai_poll_result(7).status, 4);
    assert_eq!(pcai_poll_result(7).status, -1);
    assert_eq!(pcai_cancel(7), -1);
    release_tx.send(()).unwrap();
    fixture.0.close_and_join();
    assert_eq!(pcai_poll_result(7).status, -1);
    assert!(model.try_lock().is_ok());
}

#[test]
fn public_pending_cancellation_is_consumed_once() {
    let fixture = Fixture::new();
    fixture.0.entries().insert(7, RequestStatus::Pending);
    assert_eq!(pcai_poll_result(7).status, 0);
    assert_eq!(pcai_cancel(7), 0);
    assert_eq!(pcai_poll_result(7).status, 4);
    assert_eq!(pcai_poll_result(7).status, -1);
}

#[test]
fn public_poll_reads_poisoned_model_failure_and_retains_other_cancellation() {
    let fixture = Fixture::new();
    let model = Arc::new(Mutex::new(()));
    let poisoned = Arc::clone(&model);
    assert!(std::thread::spawn(move || {
        let _guard = poisoned.lock().unwrap();
        panic!("owned synthetic model poison");
    })
    .join()
    .is_err());
    fixture.0.entries().insert(8, RequestStatus::Cancelled);
    fixture
        .0
        .spawn(7, fixture.0.ticket().unwrap(), move || {
            match requests::model_lock(&model) {
                Ok(_) => panic!("fixture must retain real poison"),
                Err(error) => RequestStatus::Failed(error),
            }
        })
        .unwrap();
    fixture.wait_terminal(7);
    let result = pcai_poll_result(7);
    assert_eq!(result.status, 3);
    assert!(!result.text.is_null());
    let message = unsafe { CStr::from_ptr(result.text) }.to_string_lossy().into_owned();
    unsafe { pcai_free_string(result.text) };
    assert!(message.starts_with("Lock failed:"));
    assert_eq!(pcai_poll_result(7).status, -1);
    assert_eq!(pcai_poll_result(8).status, 4);
}

#[test]
fn public_poll_exposes_contained_worker_panic_once() {
    let fixture = Fixture::new();
    fixture
        .0
        .spawn(7, fixture.0.ticket().unwrap(), || {
            panic!("owned synthetic worker panic")
        })
        .unwrap();
    fixture.wait_terminal(7);
    let result = pcai_poll_result(7);
    assert_eq!(result.status, 3);
    assert!(!result.text.is_null());
    unsafe { pcai_free_string(result.text) };
    assert_eq!(pcai_cancel(7), -1);
    assert_eq!(pcai_poll_result(7).status, -1);
}

#[test]
fn public_complete_result_has_owned_text_and_is_consumed_exactly_once() {
    let fixture = Fixture::new();
    fixture
        .0
        .spawn(7, fixture.0.ticket().unwrap(), || RequestStatus::Complete {
            text: "owned-output".to_string(),
            tokens_generated: 2,
            prompt_tokens_estimated: 3,
        })
        .unwrap();
    fixture.wait_terminal(7);
    let result = pcai_poll_result(7);
    assert_eq!(result.status, 2);
    assert!(!result.text.is_null());
    assert_eq!(unsafe { CStr::from_ptr(result.text) }.to_str().unwrap(), "owned-output");
    assert_eq!(result.tokens_generated, 2);
    assert_eq!(result.prompt_tokens_estimated, 3);
    unsafe { pcai_free_string(result.text) };
    assert_eq!(pcai_cancel(7), -1);
    assert_eq!(pcai_poll_result(7).status, -1);
}
