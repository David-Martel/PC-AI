//! Async request engine behind `pcai_generate_async`, `pcai_poll_result` and `pcai_cancel`.
//!
//! The engine owns the request table and a single worker thread. It does not know
//! about the inference backend: generation is delegated to a [`GenerationRunner`],
//! which the FFI layer implements on top of the process-wide backend and tests
//! implement with a controllable mock. That separation is what makes the locking
//! rules below testable without touching global state.
//!
//! # Locking rules
//!
//! ```text
//!   submit / poll / cancel                       worker thread
//!   ----------------------                       -------------
//!   table mutex (short, no I/O)                  table mutex (short)  -> Pending -> Running
//!                                                runner.run(..)       -> NO table lock held
//!                                                table mutex (short)  -> Running -> terminal
//! ```
//!
//! * The table mutex is held only for map operations. It is never held across a
//!   backend call, `block_on`, a channel send that can block, or a subprocess.
//! * Therefore `poll` and `cancel` take milliseconds regardless of how long a
//!   generation runs.
//! * Generations are serialised by construction: one worker thread drains a bounded
//!   queue.
//!
//! # Request life cycle
//!
//! ```text
//!   Pending --worker picks up--> Running --> Complete | Failed
//!      |                            |
//!      +------- cancel -------------+--> Cancelled
//! ```
//!
//! Only `Pending -> Running` and `Running -> {Complete, Failed}` transitions are made
//! by the worker, and only when the entry is still in the expected state. A late
//! completion therefore can never overwrite `Cancelled`, and can never resurrect an
//! entry that was polled or evicted.
//!
//! # Cancellation
//!
//! `cancel` marks the entry `Cancelled` immediately and raises a per-request
//! [`AtomicBool`]. A backend that implements
//! [`InferenceBackend::generate_cancellable`](crate::backends::InferenceBackend::generate_cancellable)
//! checks the flag per token and stops early (llama.cpp does). A backend that does
//! not simply runs to completion and its result is discarded.
//!
//! # Bounds and cleanup
//!
//! * The queue is bounded ([`AsyncLimits::queue_bound`]); a full queue rejects the
//!   submit, it never spawns another thread.
//! * A terminal entry is removed when it is polled. Terminal entries that are never
//!   polled (for example cancel-without-poll) are evicted once they are older than
//!   [`AsyncLimits::terminal_ttl`], and the table never grows past
//!   [`AsyncLimits::max_entries`]. `Pending` and `Running` entries are never evicted.

use std::collections::HashMap;
use std::ffi::{c_char, CString};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use crate::backends::{GenerateRequest, GenerateResponse};
use crate::Error;

use super::PcaiAsyncResult;

/// Status value for an unknown or already-consumed request id.
const STATUS_UNKNOWN: i32 = -1;
/// Status value for a request waiting in the queue.
const STATUS_PENDING: i32 = 0;
/// Status value for a request the worker is running.
const STATUS_RUNNING: i32 = 1;
/// Status value for a finished request; `text` carries the output.
const STATUS_COMPLETE: i32 = 2;
/// Status value for a failed request; `text` carries the error message.
const STATUS_FAILED: i32 = 3;
/// Status value for a cancelled request.
const STATUS_CANCELLED: i32 = 4;

/// Lock a mutex, recovering the guard if a previous holder panicked.
///
/// Every critical section guarded by the engine mutex is a plain map operation that
/// leaves the map consistent at each step, so continuing after a poison is sound.
/// Re-locking while still holding the `PoisonError` guard would self-deadlock.
pub(super) fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Runs one generation for the engine's worker thread.
///
/// Implementations may block for as long as generation takes. The engine calls this
/// without holding any of its own locks.
pub(super) trait GenerationRunner: Send + Sync + 'static {
    /// Generate text for `request`, stopping early if `cancel` becomes true.
    fn run(&self, request: GenerateRequest, cancel: &AtomicBool) -> Result<GenerateResponse, Error>;
}

/// Capacity limits for an [`AsyncEngine`].
#[derive(Debug, Clone, Copy)]
pub(super) struct AsyncLimits {
    /// Requests that may wait in the queue behind the one being generated.
    pub queue_bound: usize,
    /// Maximum number of entries (any state) in the request table.
    pub max_entries: usize,
    /// Age after which an unpolled terminal entry is evicted, measured from the
    /// moment it became terminal.
    pub terminal_ttl: Duration,
}

impl AsyncLimits {
    /// Default queue depth: deep enough for a burst from one script, small enough to
    /// fail fast when a caller is flooding.
    pub const DEFAULT_QUEUE_BOUND: usize = 16;
    /// Default table cap. Must stay well above the queue bound plus one running entry.
    pub const DEFAULT_MAX_ENTRIES: usize = 256;
    /// Default terminal TTL. Generous on purpose: a `-NoWait` caller that polls late
    /// receives "unknown id", which the C# wrapper treats as an error.
    pub const DEFAULT_TERMINAL_TTL: Duration = Duration::from_secs(600);
}

impl Default for AsyncLimits {
    fn default() -> Self {
        Self {
            queue_bound: Self::DEFAULT_QUEUE_BOUND,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
            terminal_ttl: Self::DEFAULT_TERMINAL_TTL,
        }
    }
}

/// Why a submit was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SubmitError {
    /// The bounded queue is full.
    QueueFull {
        /// Configured queue bound.
        bound: usize,
    },
    /// The request table is full of requests that are still pending or running.
    TableFull {
        /// Configured table cap.
        max: usize,
    },
    /// The worker thread could not be started or has stopped.
    Worker(String),
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull { bound } => write!(
                f,
                "Async inference queue is full ({bound} requests waiting). Poll or cancel existing requests and retry."
            ),
            Self::TableFull { max } => write!(
                f,
                "Too many async inference requests in flight (limit {max}). Poll or cancel existing requests and retry."
            ),
            Self::Worker(msg) => write!(f, "Async inference worker unavailable: {msg}"),
        }
    }
}

/// State of one request. Terminal variants carry their payload until polled.
enum RequestStatus {
    Pending,
    Running,
    Complete {
        text: String,
        tokens_generated: u32,
        prompt_tokens_estimated: u32,
    },
    Failed(String),
    Cancelled,
}

impl RequestStatus {
    fn is_terminal(&self) -> bool {
        !matches!(self, Self::Pending | Self::Running)
    }
}

/// One row of the request table.
struct Entry {
    status: RequestStatus,
    /// Raised by `cancel`; read by the backend's token loop.
    cancel: Arc<AtomicBool>,
    /// Set when the entry becomes terminal; drives TTL eviction.
    finished_at: Option<Instant>,
}

/// Unit of work handed to the worker thread.
struct Job {
    id: i64,
    request: GenerateRequest,
    prompt_tokens_estimated: u32,
    cancel: Arc<AtomicBool>,
}

/// State shared between the FFI-facing engine and its worker thread.
struct Shared {
    table: Mutex<HashMap<i64, Entry>>,
    runner: Arc<dyn GenerationRunner>,
    limits: AsyncLimits,
}

impl Shared {
    /// Drop terminal entries older than the TTL. Caller holds the table lock.
    fn sweep_locked(&self, table: &mut HashMap<i64, Entry>, now: Instant) {
        let ttl = self.limits.terminal_ttl;
        table.retain(|_, entry| match entry.finished_at {
            Some(done) => now.saturating_duration_since(done) < ttl,
            None => true,
        });
    }

    /// Evict the oldest terminal entries until there is room for one more.
    /// Returns false when the table is full of non-terminal entries.
    fn make_room_locked(&self, table: &mut HashMap<i64, Entry>) -> bool {
        while table.len() >= self.limits.max_entries {
            let oldest = table
                .iter()
                .filter_map(|(id, entry)| entry.finished_at.map(|done| (*id, done)))
                .min_by_key(|(_, done)| *done)
                .map(|(id, _)| id);
            match oldest {
                Some(id) => {
                    table.remove(&id);
                }
                None => return false,
            }
        }
        true
    }

    /// Transition `Pending -> Running`. Returns false when the job must be skipped
    /// (cancelled while queued, or already evicted).
    fn begin(&self, id: i64) -> bool {
        let mut table = lock_recover(&self.table);
        match table.get_mut(&id) {
            Some(entry) if matches!(entry.status, RequestStatus::Pending) => {
                entry.status = RequestStatus::Running;
                true
            }
            _ => false,
        }
    }

    /// Transition `Running -> terminal`. A cancelled, polled or evicted entry is left
    /// exactly as it is and the result is discarded.
    fn finish(&self, id: i64, terminal: RequestStatus, now: Instant) {
        let mut table = lock_recover(&self.table);
        if let Some(entry) = table.get_mut(&id) {
            if matches!(entry.status, RequestStatus::Running) {
                entry.status = terminal;
                entry.finished_at = Some(now);
            }
        }
        self.sweep_locked(&mut table, now);
    }

    /// Run one job to a terminal state.
    fn execute(&self, job: Job) {
        if !self.begin(job.id) {
            return;
        }

        let Job {
            id,
            request,
            prompt_tokens_estimated,
            cancel,
        } = job;

        // A panicking backend must not kill the only worker: that would leave this
        // entry Running forever and every later submit stuck behind a dead queue.
        let outcome = catch_unwind(AssertUnwindSafe(|| self.runner.run(request, &cancel)));

        let terminal = match outcome {
            Ok(Ok(response)) => RequestStatus::Complete {
                text: response.text,
                tokens_generated: u32::try_from(response.tokens_generated).unwrap_or(u32::MAX),
                prompt_tokens_estimated,
            },
            // The GPU-state enrichment may spawn `nvidia-smi`; it runs here, outside
            // the table lock.
            Ok(Err(error)) => RequestStatus::Failed(super::enrich_with_gpu_state(&error.to_string())),
            Err(_) => RequestStatus::Failed("Inference worker panicked while generating".to_string()),
        };
        self.finish(id, terminal, Instant::now());
    }
}

/// Request table plus the single worker thread that drains the bounded queue.
pub(super) struct AsyncEngine {
    shared: Arc<Shared>,
    next_id: AtomicI64,
    /// Started on first submit so an engine that is never used costs no thread.
    queue: OnceLock<Result<SyncSender<Job>, String>>,
    /// Worker threads ever started. Must stay at most one.
    #[cfg(test)]
    workers_spawned: std::sync::atomic::AtomicUsize,
}

impl AsyncEngine {
    /// Create an engine that generates through `runner`.
    pub(super) fn new(runner: Arc<dyn GenerationRunner>, limits: AsyncLimits) -> Self {
        let limits = AsyncLimits {
            queue_bound: limits.queue_bound.max(1),
            max_entries: limits.max_entries.max(limits.queue_bound.max(1) + 2),
            terminal_ttl: limits.terminal_ttl,
        };
        Self {
            shared: Arc::new(Shared {
                table: Mutex::new(HashMap::new()),
                runner,
                limits,
            }),
            next_id: AtomicI64::new(1),
            queue: OnceLock::new(),
            #[cfg(test)]
            workers_spawned: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn spawn_worker(&self) -> Result<SyncSender<Job>, String> {
        let (tx, rx): (SyncSender<Job>, Receiver<Job>) = sync_channel(self.shared.limits.queue_bound);
        let shared = Arc::clone(&self.shared);
        std::thread::Builder::new()
            .name("pcai-async-inference".to_string())
            .spawn(move || {
                // Ends when the engine (and with it the sender) is dropped.
                while let Ok(job) = rx.recv() {
                    shared.execute(job);
                }
            })
            .map_err(|e| format!("failed to start worker thread: {e}"))?;
        #[cfg(test)]
        self.workers_spawned.fetch_add(1, Ordering::SeqCst);
        Ok(tx)
    }

    /// Queue `request` and return its id.
    ///
    /// Never blocks: a full queue or table is reported as an error.
    pub(super) fn submit(&self, request: GenerateRequest) -> Result<i64, SubmitError> {
        let tx = match self.queue.get_or_init(|| self.spawn_worker()) {
            Ok(tx) => tx,
            Err(msg) => return Err(SubmitError::Worker(msg.clone())),
        };

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let cancel = Arc::new(AtomicBool::new(false));
        let prompt_tokens_estimated = super::estimate_prompt_tokens(&request.prompt);

        {
            let mut table = lock_recover(&self.shared.table);
            self.shared.sweep_locked(&mut table, Instant::now());
            if !self.shared.make_room_locked(&mut table) {
                return Err(SubmitError::TableFull {
                    max: self.shared.limits.max_entries,
                });
            }
            table.insert(
                id,
                Entry {
                    status: RequestStatus::Pending,
                    cancel: Arc::clone(&cancel),
                    finished_at: None,
                },
            );
        }

        let job = Job {
            id,
            request,
            prompt_tokens_estimated,
            cancel,
        };
        match tx.try_send(job) {
            Ok(()) => Ok(id),
            Err(error) => {
                // Roll back the row so a rejected submit leaves no trace.
                lock_recover(&self.shared.table).remove(&id);
                match error {
                    TrySendError::Full(_) => Err(SubmitError::QueueFull {
                        bound: self.shared.limits.queue_bound,
                    }),
                    TrySendError::Disconnected(_) => Err(SubmitError::Worker("worker thread stopped".to_string())),
                }
            }
        }
    }

    /// Report the status of `id`, consuming the entry when it is terminal.
    pub(super) fn poll(&self, id: i64) -> PcaiAsyncResult {
        let mut table = lock_recover(&self.shared.table);

        let non_terminal = match table.get(&id) {
            None => return async_result(STATUS_UNKNOWN),
            Some(entry) => match entry.status {
                RequestStatus::Pending => Some(STATUS_PENDING),
                RequestStatus::Running => Some(STATUS_RUNNING),
                _ => None,
            },
        };
        if let Some(status) = non_terminal {
            return async_result(status);
        }

        let removed = table.remove(&id);
        drop(table);

        match removed.map(|entry| entry.status) {
            Some(RequestStatus::Complete {
                text,
                tokens_generated,
                prompt_tokens_estimated,
            }) => PcaiAsyncResult {
                status: STATUS_COMPLETE,
                text: into_c_text(text),
                tokens_generated,
                prompt_tokens_estimated,
            },
            Some(RequestStatus::Failed(message)) => PcaiAsyncResult {
                status: STATUS_FAILED,
                text: into_c_text(message),
                tokens_generated: 0,
                prompt_tokens_estimated: 0,
            },
            Some(RequestStatus::Cancelled) => async_result(STATUS_CANCELLED),
            _ => async_result(STATUS_UNKNOWN),
        }
    }

    /// Cancel a pending or running request. Returns 0 on success, -1 when the id is
    /// unknown or the request already reached a terminal state.
    pub(super) fn cancel(&self, id: i64) -> i32 {
        let now = Instant::now();
        let mut table = lock_recover(&self.shared.table);
        self.shared.sweep_locked(&mut table, now);
        match table.get_mut(&id) {
            Some(entry) if !entry.status.is_terminal() => {
                entry.cancel.store(true, Ordering::SeqCst);
                entry.status = RequestStatus::Cancelled;
                entry.finished_at = Some(now);
                0
            }
            _ => -1,
        }
    }

    /// Cancel every pending or running request. Used by shutdown so it does not wait
    /// for work nobody will collect.
    pub(super) fn cancel_all(&self) {
        let now = Instant::now();
        let mut table = lock_recover(&self.shared.table);
        for entry in table.values_mut().filter(|entry| !entry.status.is_terminal()) {
            entry.cancel.store(true, Ordering::SeqCst);
            entry.status = RequestStatus::Cancelled;
            entry.finished_at = Some(now);
        }
    }

    /// Evict expired terminal entries as of `now`. Exposed so tests can advance time
    /// without sleeping.
    #[cfg(test)]
    pub(super) fn sweep_at(&self, now: Instant) {
        let mut table = lock_recover(&self.shared.table);
        self.shared.sweep_locked(&mut table, now);
    }

    /// Number of rows currently in the request table.
    #[cfg(test)]
    pub(super) fn entry_count(&self) -> usize {
        lock_recover(&self.shared.table).len()
    }

    /// Worker threads started so far.
    #[cfg(test)]
    pub(super) fn workers_spawned(&self) -> usize {
        self.workers_spawned.load(Ordering::SeqCst)
    }

    /// Poison the table mutex by panicking while holding it.
    #[cfg(test)]
    pub(super) fn poison_table_for_test(&self) {
        let shared = Arc::clone(&self.shared);
        let _ = std::thread::spawn(move || {
            let _guard = shared.table.lock();
            // Intentional: the test needs a holder that unwinds with the guard live.
            std::panic::resume_unwind(Box::new("test: poisoning the request table"));
        })
        .join();
    }
}

/// A result with no text and no token counts.
fn async_result(status: i32) -> PcaiAsyncResult {
    PcaiAsyncResult {
        status,
        text: std::ptr::null_mut(),
        tokens_generated: 0,
        prompt_tokens_estimated: 0,
    }
}

/// Hand `text` to the C caller, replacing interior NULs so the conversion cannot fail.
fn into_c_text(text: String) -> *mut c_char {
    CString::new(text.replace('\0', "\u{FFFD}"))
        .unwrap_or_default()
        .into_raw()
}

#[cfg(test)]
#[path = "async_engine_tests.rs"]
mod tests;
