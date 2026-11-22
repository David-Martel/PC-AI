//! Concurrency tests for the async request engine.
//!
//! These run against per-test `AsyncEngine` instances with a controllable mock
//! runner, never against the process-wide global state. The global state is shared
//! by every test in the binary and several existing tests call `pcai_shutdown()`, so
//! a mock installed there would be torn down mid-test and a poison test would poison
//! the lock for everyone. The `extern "C"` functions are one-line delegates to
//! `init_global_state().async_engine`, which is exactly what these tests exercise.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::*;
use crate::backends::FinishReason;

/// Upper bound for any wait in these tests. Only a failing test ever reaches it.
const WAIT: Duration = Duration::from_secs(5);
/// Latency budget for poll and cancel while a generation is running.
const RESPONSIVE: Duration = Duration::from_millis(100);

fn request(prompt: &str) -> GenerateRequest {
    GenerateRequest {
        prompt: prompt.to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: vec![],
    }
}

fn response(text: &str) -> GenerateResponse {
    GenerateResponse {
        text: text.to_string(),
        tokens_generated: 7,
        finish_reason: FinishReason::Stop,
    }
}

/// Spin until `condition` holds or `WAIT` elapses.
fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
    let start = Instant::now();
    while !condition() {
        assert!(start.elapsed() < WAIT, "timed out waiting for: {what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Take ownership of the text in a poll result and free it the way a C caller would.
fn take_text(result: &PcaiAsyncResult) -> Option<String> {
    if result.text.is_null() {
        return None;
    }
    // SAFETY: `text` was produced by `CString::into_raw` in `into_c_text` and is freed
    // exactly once here.
    let owned = unsafe { CString::from_raw(result.text) };
    Some(owned.to_string_lossy().into_owned())
}

/// True once the request has reached a terminal state (without consuming it).
fn is_terminal(engine: &AsyncEngine, id: i64) -> bool {
    lock_recover(&engine.shared.table)
        .get(&id)
        .is_some_and(|entry| entry.status.is_terminal())
}

enum Release {
    Text(&'static str),
}

/// Runner that blocks inside `run` until the test releases it, like a long generation.
struct GatedRunner {
    entered: Mutex<mpsc::Sender<()>>,
    release: Mutex<mpsc::Receiver<Release>>,
    /// Mimics a backend with a per-token cancel check.
    honour_cancel: bool,
    cancel_observed: AtomicBool,
}

impl GenerationRunner for GatedRunner {
    fn run(&self, _request: GenerateRequest, cancel: &AtomicBool) -> Result<GenerateResponse, Error> {
        let _ = lock_recover(&self.entered).send(());
        let release = lock_recover(&self.release);
        loop {
            if self.honour_cancel && cancel.load(Ordering::SeqCst) {
                self.cancel_observed.store(true, Ordering::SeqCst);
                return Ok(response("partial"));
            }
            match release.recv_timeout(Duration::from_millis(2)) {
                Ok(Release::Text(text)) => return Ok(response(text)),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Err(Error::Backend("test: gate closed".to_string())),
            }
        }
    }
}

/// Test-side handle for a [`GatedRunner`].
struct Gate {
    runner: Arc<GatedRunner>,
    entered_rx: mpsc::Receiver<()>,
    release_tx: mpsc::Sender<Release>,
}

impl Gate {
    fn new(honour_cancel: bool) -> Self {
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        Self {
            runner: Arc::new(GatedRunner {
                entered: Mutex::new(entered_tx),
                release: Mutex::new(release_rx),
                honour_cancel,
                cancel_observed: AtomicBool::new(false),
            }),
            entered_rx,
            release_tx,
        }
    }

    /// Block until the worker is inside `run`, so the request is observably Running.
    fn wait_entered(&self) {
        self.entered_rx
            .recv_timeout(WAIT)
            .expect("test: worker never entered the generation");
    }

    fn release(&self, outcome: Release) {
        self.release_tx
            .send(outcome)
            .expect("test: the runner must still be listening");
    }
}

/// Runner that finishes immediately.
enum Mode {
    Ok,
    Fail,
    Panic,
}

struct ImmediateRunner(Mode);

impl GenerationRunner for ImmediateRunner {
    fn run(&self, request: GenerateRequest, _cancel: &AtomicBool) -> Result<GenerateResponse, Error> {
        match self.0 {
            Mode::Ok => Ok(response(&format!("echo:{}", request.prompt))),
            Mode::Fail => Err(Error::Backend("test: backend refused".to_string())),
            Mode::Panic => panic!("test: simulated backend panic"),
        }
    }
}

fn limits(queue_bound: usize, max_entries: usize, ttl: Duration) -> AsyncLimits {
    AsyncLimits {
        queue_bound,
        max_entries,
        terminal_ttl: ttl,
    }
}

fn gated_engine(gate: &Gate, queue_bound: usize, ttl: Duration) -> AsyncEngine {
    AsyncEngine::new(
        Arc::clone(&gate.runner) as Arc<dyn GenerationRunner>,
        limits(queue_bound, 64, ttl),
    )
}

fn immediate_engine(mode: Mode) -> AsyncEngine {
    AsyncEngine::new(Arc::new(ImmediateRunner(mode)), limits(8, 64, Duration::from_secs(600)))
}

// ---------------------------------------------------------------------------
// 1. poll and cancel never wait for a running generation
// ---------------------------------------------------------------------------

/// Protects: the PowerShell timeout (`Invoke-PcaiGenerateAsync -TimeoutSeconds`) and the
/// C# `CancellationToken` path, both of which call `pcai_poll_result` / `pcai_cancel` while
/// a generation is in flight. Before the lock split both took the same mutex the worker
/// held across `block_on`, so they blocked for the entire generation.
///
/// Detects: any change that makes `poll` or `cancel` take a lock a running generation
/// holds (the backend lock, the generation gate, or the table lock held across the runner).
/// The runner here is parked, so a blocking poll would exceed the budget by seconds.
#[test]
fn poll_and_cancel_return_promptly_while_generation_is_running() {
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 4, Duration::from_secs(600));

    let running = engine.submit(request("long job")).expect("test: submit must succeed");
    gate.wait_entered();

    let start = Instant::now();
    let polled = engine.poll(running);
    let poll_latency = start.elapsed();
    assert_eq!(polled.status, 1, "a generation in progress must poll as Running");
    assert!(polled.text.is_null());
    assert!(
        poll_latency < RESPONSIVE,
        "poll took {poll_latency:?} while a generation was running (budget {RESPONSIVE:?})"
    );

    // A queued request is observable too, and also does not wait.
    let queued = engine.submit(request("queued")).expect("test: submit must succeed");
    let start = Instant::now();
    let queued_status = engine.poll(queued).status;
    let queued_latency = start.elapsed();
    assert_eq!(queued_status, 0, "a request behind the running one polls as Pending");
    assert!(queued_latency < RESPONSIVE, "poll of queued id took {queued_latency:?}");

    let start = Instant::now();
    let rc = engine.cancel(running);
    let cancel_latency = start.elapsed();
    assert_eq!(rc, 0, "cancelling a running request must succeed");
    assert!(
        cancel_latency < RESPONSIVE,
        "cancel took {cancel_latency:?} while a generation was running (budget {RESPONSIVE:?})"
    );

    println!(
        "latency while a generation is blocked: poll(running)={poll_latency:?} poll(pending)={queued_latency:?} cancel={cancel_latency:?}"
    );

    gate.release(Release::Text("unused"));
    let _ = engine.cancel(queued);
}

// ---------------------------------------------------------------------------
// 2. cancellation semantics
// ---------------------------------------------------------------------------

/// Protects: the documented rule that a cancelled request stays cancelled. A backend
/// without a cancel hook keeps running after `cancel`, and its result must be discarded.
///
/// Detects: a worker that writes `Complete` over `Cancelled` (the old code `insert`ed
/// unconditionally on completion paths and could resurrect a polled-away entry).
#[test]
fn late_completion_never_overwrites_cancelled() {
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 4, Duration::from_secs(600));

    let first = engine.submit(request("first")).expect("test: submit must succeed");
    gate.wait_entered();
    assert_eq!(engine.cancel(first), 0);

    // Let the backend finish "normally" after the cancel.
    gate.release(Release::Text("too late"));

    // The worker is single-threaded: once it enters the second job, the first job's
    // completion handling has already run. That is the point at which a buggy worker
    // would already have overwritten Cancelled.
    let second = engine.submit(request("second")).expect("test: submit must succeed");
    gate.wait_entered();

    let polled = engine.poll(first);
    assert_eq!(polled.status, 4, "a late completion must not replace Cancelled");
    assert!(polled.text.is_null(), "a cancelled request has no text");

    gate.release(Release::Text("second done"));
    wait_until("second request to finish", || is_terminal(&engine, second));
    let second_result = engine.poll(second);
    assert_eq!(
        second_result.status, 2,
        "the worker keeps serving requests after a cancel"
    );
    assert_eq!(take_text(&second_result).as_deref(), Some("second done"));
}

/// Protects: early stop for backends that honour the per-request cancel flag (llama.cpp
/// checks it once per token). Cancel is useless for long generations otherwise.
///
/// Detects: the flag no longer reaching the runner, or `cancel` failing to raise it.
#[test]
fn cancel_raises_the_flag_the_backend_observes() {
    let gate = Gate::new(true);
    let engine = gated_engine(&gate, 4, Duration::from_secs(600));

    let id = engine.submit(request("long")).expect("test: submit must succeed");
    gate.wait_entered();
    assert!(!gate.runner.cancel_observed.load(Ordering::SeqCst));

    assert_eq!(engine.cancel(id), 0);
    wait_until("backend to observe the cancel flag and stop", || {
        gate.runner.cancel_observed.load(Ordering::SeqCst)
    });

    // The partial output is discarded; the caller sees Cancelled exactly once.
    assert_eq!(engine.poll(id).status, 4);
    assert_eq!(engine.poll(id).status, -1);
}

/// Protects: cancel's return contract (0 once, then -1) that `CancelRequest` in C# turns
/// into a boolean.
///
/// Detects: a double cancel reporting success, or cancel succeeding on a finished request.
#[test]
fn cancel_is_rejected_for_terminal_and_unknown_requests() {
    let engine = immediate_engine(Mode::Ok);
    let id = engine.submit(request("quick")).expect("test: submit must succeed");
    wait_until("request to finish", || is_terminal(&engine, id));

    assert_eq!(engine.cancel(id), -1, "a finished request cannot be cancelled");
    assert_eq!(engine.cancel(987_654), -1, "an unknown id cannot be cancelled");
    assert_eq!(
        engine.poll(id).status,
        2,
        "a rejected cancel must not disturb the result"
    );
}

// ---------------------------------------------------------------------------
// 3. bounded execution
// ---------------------------------------------------------------------------

/// Protects: the "one worker, bounded queue" guarantee. The old design spawned an OS
/// thread per request with no cap, so a flooding caller could exhaust threads and memory.
///
/// Detects: a submit past the bound that blocks, succeeds, or starts another thread, and a
/// rejected submit that leaves a row behind in the table.
#[test]
fn submit_beyond_queue_bound_is_rejected_without_new_threads() {
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 2, Duration::from_secs(600));

    let running = engine.submit(request("running")).expect("test: submit must succeed");
    gate.wait_entered();
    let queued_a = engine.submit(request("a")).expect("test: queue slot 1 must be free");
    let queued_b = engine.submit(request("b")).expect("test: queue slot 2 must be free");

    let rows_before = engine.entry_count();
    for _ in 0..50 {
        let start = Instant::now();
        let rejected = engine.submit(request("overflow"));
        assert!(start.elapsed() < RESPONSIVE, "a rejected submit must not block");
        assert_eq!(rejected, Err(SubmitError::QueueFull { bound: 2 }));
    }

    assert_eq!(
        engine.workers_spawned(),
        1,
        "overflow must not start additional threads"
    );
    assert_eq!(
        engine.entry_count(),
        rows_before,
        "rejected submits must leave no rows behind"
    );

    let message = SubmitError::QueueFull { bound: 2 }.to_string();
    assert!(
        message.contains("queue is full"),
        "error should explain itself: {message}"
    );

    for id in [running, queued_a, queued_b] {
        let _ = engine.cancel(id);
    }
    gate.release(Release::Text("unused"));
}

/// Protects: the worker's survival of a panicking backend. A dead worker would leave the
/// request Running forever and make every later submit queue behind nothing.
///
/// Detects: a missing `catch_unwind` around the runner.
#[test]
fn backend_panic_fails_the_request_and_keeps_the_worker_alive() {
    let panicking = immediate_engine(Mode::Panic);
    let id = panicking.submit(request("boom")).expect("test: submit must succeed");
    wait_until("panicking request to fail", || is_terminal(&panicking, id));
    let failed = panicking.poll(id);
    assert_eq!(failed.status, 3);
    assert!(take_text(&failed).unwrap_or_default().contains("panicked"));

    // The same worker accepts and runs another request afterwards.
    let again = panicking
        .submit(request("again"))
        .expect("test: worker must still accept work");
    wait_until("second request to reach a terminal state", || {
        is_terminal(&panicking, again)
    });
    assert_eq!(panicking.poll(again).status, 3);
    assert_eq!(panicking.workers_spawned(), 1);
}

// ---------------------------------------------------------------------------
// 4. eviction
// ---------------------------------------------------------------------------

/// Protects: the leak where a request cancelled and never polled stayed in the table
/// forever. Both callers do exactly that on timeout or token cancellation.
///
/// Detects: terminal entries surviving past the TTL, and the opposite error of evicting an
/// entry before the TTL or evicting a Running one.
#[test]
fn unpolled_terminal_entries_are_evicted_after_the_ttl() {
    let ttl = Duration::from_secs(60);
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 4, ttl);

    let running = engine
        .submit(request("keeps running"))
        .expect("test: submit must succeed");
    gate.wait_entered();
    let abandoned = engine.submit(request("abandoned")).expect("test: submit must succeed");
    assert_eq!(engine.cancel(abandoned), 0);
    assert_eq!(engine.entry_count(), 2);

    // Advance the clock instead of sleeping, so the test is exact and instant.
    engine.sweep_at(Instant::now() + ttl / 2);
    assert_eq!(engine.entry_count(), 2, "an entry younger than the TTL must be kept");

    engine.sweep_at(Instant::now() + ttl + Duration::from_secs(1));
    assert_eq!(engine.entry_count(), 1, "the abandoned cancelled entry must be evicted");
    assert_eq!(engine.poll(abandoned).status, -1);
    assert_eq!(engine.poll(running).status, 1, "a running entry must never be evicted");

    gate.release(Release::Text("done"));
}

/// Protects: the table cap. A caller that submits and never polls must not grow the map
/// without bound.
///
/// Detects: a missing cap, evicting newer entries before older ones, or a submit failing
/// when the table is full of finished work that could be reclaimed.
#[test]
fn table_cap_reclaims_the_oldest_terminal_entries() {
    // queue_bound 1 gives a normalised cap of 3 rows.
    let engine = AsyncEngine::new(
        Arc::new(ImmediateRunner(Mode::Ok)),
        limits(1, 3, Duration::from_secs(600)),
    );

    let mut ids = Vec::new();
    for n in 0..3 {
        let id = engine
            .submit(request(&format!("job {n}")))
            .expect("test: submit must succeed");
        wait_until("job to finish", || is_terminal(&engine, id));
        ids.push(id);
        // Distinct finish times keep "oldest" unambiguous.
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(engine.entry_count(), 3);

    let newest = engine
        .submit(request("job 3"))
        .expect("test: a full table of finished work must be reclaimable");
    assert!(engine.entry_count() <= 3, "the table must never exceed its cap");
    assert_eq!(
        engine.poll(ids[0]).status,
        -1,
        "the oldest terminal entry is the one evicted"
    );
    wait_until("newest job to finish", || is_terminal(&engine, newest));
    assert_eq!(engine.poll(ids[2]).status, 2, "newer entries survive");
}

/// Protects: eviction racing a still-running cancelled request. Cancel does not stop a
/// backend without a hook, so its entry can be evicted while the generation is running.
///
/// Detects: the old unconditional `insert`, which would resurrect the evicted entry as
/// Complete when the generation eventually finishes.
#[test]
fn late_completion_does_not_resurrect_an_evicted_entry() {
    let ttl = Duration::from_secs(60);
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 4, ttl);

    let id = engine
        .submit(request("evicted while running"))
        .expect("test: submit must succeed");
    gate.wait_entered();
    assert_eq!(engine.cancel(id), 0);
    engine.sweep_at(Instant::now() + ttl + Duration::from_secs(1));
    assert_eq!(engine.entry_count(), 0);

    gate.release(Release::Text("late"));
    let next = engine.submit(request("next")).expect("test: submit must succeed");
    gate.wait_entered(); // the first job's completion handling has run
    assert_eq!(engine.poll(id).status, -1, "an evicted id must stay unknown");
    assert_eq!(engine.entry_count(), 1, "only the next request is in the table");

    gate.release(Release::Text("next done"));
    wait_until("next request to finish", || is_terminal(&engine, next));
}

// ---------------------------------------------------------------------------
// 5. terminal poll frees the entry
// ---------------------------------------------------------------------------

/// Protects: consume-on-poll, which the C# `PollResult` doc and `Get-PcaiAsyncResult`
/// rely on (each polls a terminal result once, then stops).
///
/// Detects: a terminal poll that leaves the entry behind, a second poll that returns the
/// text again (double free hazard for the caller), or wrong status/text/token fields.
#[test]
fn polling_a_terminal_request_frees_it() {
    let engine = immediate_engine(Mode::Ok);
    let id = engine
        .submit(request("hello world"))
        .expect("test: submit must succeed");
    wait_until("request to finish", || is_terminal(&engine, id));

    let first = engine.poll(id);
    assert_eq!(first.status, 2);
    assert_eq!(first.tokens_generated, 7);
    assert_eq!(first.prompt_tokens_estimated, 3, "two words, 11 chars -> max(3, 2)");
    assert_eq!(take_text(&first).as_deref(), Some("echo:hello world"));
    assert_eq!(
        engine.entry_count(),
        0,
        "the entry must be gone after the terminal poll"
    );

    let second = engine.poll(id);
    assert_eq!(second.status, -1, "a second poll of a consumed id is an unknown id");
    assert!(second.text.is_null());
}

/// Protects: the Failed path of consume-on-poll and the error text the callers surface.
///
/// Detects: a failure that is swallowed, loses its message, or is not freed after polling.
#[test]
fn failed_request_reports_its_message_once() {
    let engine = immediate_engine(Mode::Fail);
    let id = engine.submit(request("x")).expect("test: submit must succeed");
    wait_until("request to fail", || is_terminal(&engine, id));

    let polled = engine.poll(id);
    assert_eq!(polled.status, 3);
    assert!(take_text(&polled).unwrap_or_default().contains("backend refused"));
    assert_eq!(engine.poll(id).status, -1);
    assert_eq!(engine.entry_count(), 0);
}

// ---------------------------------------------------------------------------
// 6. poisoned lock
// ---------------------------------------------------------------------------

/// Protects: recovery from a poisoned table mutex. The old worker re-locked while still
/// holding the `PoisonError` guard, a self-deadlock; `pcai_media` already uses
/// `into_inner` for the same situation.
///
/// Detects: poll, cancel or submit returning a bogus status (-1 / error) or hanging after a
/// thread panicked while holding the table lock. The status assertions check the real
/// state comes back, not just that the call returns.
#[test]
fn poisoned_table_lock_does_not_deadlock_or_lose_state() {
    let gate = Gate::new(false);
    let engine = gated_engine(&gate, 4, Duration::from_secs(600));

    let running = engine.submit(request("running")).expect("test: submit must succeed");
    gate.wait_entered();

    engine.poison_table_for_test();
    assert!(
        engine.shared.table.is_poisoned(),
        "test setup: the table mutex must actually be poisoned"
    );

    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let polled = engine.poll(running).status;
            let queued = engine.submit(request("after poison"));
            let cancelled = engine.cancel(running);
            let _ = tx.send((polled, queued, cancelled));
        });
        let (polled, queued, cancelled) = rx
            .recv_timeout(WAIT)
            .expect("test: poll/submit/cancel deadlocked on a poisoned mutex");
        assert_eq!(polled, 1, "the real status must come back, not the unknown-id sentinel");
        let queued_id = queued.expect("test: submit must keep working after a poison");
        assert_eq!(cancelled, 0, "cancel must still find and cancel the running request");
        assert_eq!(engine.poll(running).status, 4);
        let _ = engine.cancel(queued_id);
    });

    gate.release(Release::Text("unused"));
}
