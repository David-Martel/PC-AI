//! Model-independent async request state and exact worker custody.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

/// Report poisoned model state without reacquiring the guard owned by its error.
pub(super) fn model_lock<T>(state: &Mutex<T>) -> Result<MutexGuard<'_, T>, String> {
    state.lock().map_err(|error| format!("Lock failed: {error}"))
}

pub(super) trait RequestState: Send + 'static {
    fn pending() -> Self;
    fn running() -> Self;
    fn cancelled() -> Self;
    fn failed(message: String) -> Self;
    fn is_pending(&self) -> bool;
    fn is_running(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    #[derive(Debug)]
    enum Status {
        Pending,
        Running,
        Complete,
        Failed(String),
        Cancelled,
    }
    impl RequestState for Status {
        fn pending() -> Self {
            Self::Pending
        }
        fn running() -> Self {
            Self::Running
        }
        fn cancelled() -> Self {
            Self::Cancelled
        }
        fn failed(message: String) -> Self {
            Self::Failed(message)
        }
        fn is_pending(&self) -> bool {
            matches!(self, Self::Pending)
        }
        fn is_running(&self) -> bool {
            matches!(self, Self::Running)
        }
    }

    fn terminal(requests: &Requests<Status>, id: i64) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while requests
            .entries()
            .get(&id)
            .is_some_and(|state| state.is_pending() || state.is_running())
        {
            assert!(Instant::now() < deadline, "owned worker did not publish terminal state");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    // Protects exact transitions; detects consumed cancellation resurrected as Running/Complete.
    // Needs only local request maps, no backend/model/runtime.
    #[test]
    fn consumed_cancel_and_terminal_entries_never_resurrect() {
        let requests = Requests::<Status>::new();
        requests.entries().insert(7, Status::Cancelled);
        assert!(!requests.begin(7));
        requests.entries().remove(&7);
        assert!(!requests.begin(7));
        requests.finish(7, Status::Complete);
        assert!(!requests.entries().contains_key(&7));
        requests.entries().insert(8, Status::Complete);
        requests.finish(8, Status::Failed("late".into()));
        assert!(matches!(requests.entries().get(&8), Some(Status::Complete)));
    }

    #[test]
    fn worker_panic_is_contained_and_failed_without_poisoning_requests() {
        let requests = Arc::new(Requests::<Status>::new());
        requests
            .spawn(7, requests.ticket().unwrap(), || panic!("synthetic worker panic"))
            .unwrap();
        terminal(&requests, 7);
        assert!(
            matches!(requests.entries().get(&7), Some(Status::Failed(message)) if message == "Async generation worker panicked")
        );
        requests.close_and_join();
    }

    #[test]
    fn poisoned_model_reports_failure_without_recursive_lock() {
        let model = Mutex::new(());
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _guard = model.lock().unwrap();
            panic!("synthetic model poison");
        }))
        .is_err());
        let error = model_lock(&model).unwrap_err();
        assert!(error.starts_with("Lock failed:"));
        assert!(model.is_poisoned());
    }

    #[test]
    fn poisoned_registry_fails_only_active_entries() {
        let requests = Requests::<Status>::new();
        requests.entries().insert(7, Status::Running);
        requests.entries().insert(8, Status::Cancelled);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _guard = requests.entries.lock().unwrap();
            panic!("synthetic registry poison");
        }))
        .is_err());
        assert!(matches!(requests.entries().get(&7), Some(Status::Failed(_))));
        assert!(matches!(requests.entries().get(&8), Some(Status::Cancelled)));
    }

    // Protects admission linearization and exact handle custody while shutdown waits.
    // Detects old-generation tickets, submissions during shutdown and detached workers.
    #[test]
    fn shutdown_joins_owned_worker_and_reinitialization_rejects_stale_ticket() {
        let requests = Arc::new(Requests::<Status>::new());
        let ticket = requests.ticket().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        requests
            .spawn(7, ticket, move || {
                started_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                Status::Complete
            })
            .unwrap();
        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(requests.is_running(7));
        let shutdown_registry = Arc::clone(&requests);
        let (closed_tx, closed_rx) = mpsc::channel();
        let shutdown = std::thread::spawn(move || {
            let _lifecycle = shutdown_registry.lifecycle();
            shutdown_registry.close_and_join();
            closed_tx.send(()).unwrap();
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while requests.ticket().is_some() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert!(closed_rx.try_recv().is_err(), "shutdown cannot lose its running worker");
        assert!(requests.spawn(8, ticket, || Status::Complete).is_err());
        release_tx.send(()).unwrap();
        closed_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        shutdown.join().unwrap();
        assert!(matches!(requests.entries().get(&7), Some(Status::Cancelled)));
        requests.reopen();
        assert_ne!(requests.ticket(), Some(ticket));
        assert!(requests.spawn(9, ticket, || Status::Complete).is_err());
    }

    #[test]
    fn authoritative_readiness_invalidates_load_and_shutdown_tickets() {
        let requests = Arc::new(Requests::<Status>::new());
        assert_eq!(requests.ready_ticket(), Err(AdmissionError::NotInitialized));
        requests.publish(true, false);
        assert_eq!(requests.ready_ticket(), Err(AdmissionError::ModelNotLoaded));
        requests.publish(true, true);
        let loaded = requests.ready_ticket().unwrap();
        requests.invalidate();
        assert_eq!(requests.ready_ticket(), Err(AdmissionError::ModelNotLoaded));
        requests.publish(true, true);
        assert!(requests.spawn(7, loaded, || Status::Complete).is_err());
        let reloaded = requests.ready_ticket().unwrap();
        requests.close_and_join();
        assert_eq!(requests.ready_ticket(), Err(AdmissionError::Closed));
        requests.reopen();
        assert!(requests.spawn(8, reloaded, || Status::Complete).is_err());
    }
}

struct Workers {
    accepting: bool,
    initialized: bool,
    ready: bool,
    epoch: u64,
    handles: Vec<JoinHandle<()>>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum AdmissionError {
    Closed,
    NotInitialized,
    ModelNotLoaded,
}

pub(super) struct Requests<S> {
    entries: Mutex<HashMap<i64, S>>,
    workers: Mutex<Workers>,
    lifecycle: Mutex<()>,
}

impl<S: RequestState> Requests<S> {
    pub(super) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            workers: Mutex::new(Workers {
                accepting: true,
                initialized: false,
                ready: false,
                epoch: 0,
                handles: Vec::new(),
            }),
            lifecycle: Mutex::new(()),
        }
    }

    /// Serialize initialization and shutdown without blocking request polling.
    pub(super) fn lifecycle(&self) -> MutexGuard<'_, ()> {
        self.lifecycle.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn entries(&self) -> MutexGuard<'_, HashMap<i64, S>> {
        self.entries.lock().unwrap_or_else(|error| {
            let mut entries = error.into_inner();
            for status in entries.values_mut() {
                if status.is_pending() || status.is_running() {
                    *status = S::failed("Async request registry poisoned".to_string());
                }
            }
            entries
        })
    }

    fn workers(&self) -> MutexGuard<'_, Workers> {
        self.workers.lock().unwrap_or_else(|error| {
            let mut workers = error.into_inner();
            workers.accepting = false;
            workers
        })
    }

    #[cfg(test)]
    pub(super) fn ticket(&self) -> Option<u64> {
        let workers = self.workers();
        workers.accepting.then_some(workers.epoch)
    }

    /// Read authoritative readiness without waiting for model generation.
    pub(super) fn ready_ticket(&self) -> Result<u64, AdmissionError> {
        let workers = self.workers();
        if !workers.accepting {
            Err(AdmissionError::Closed)
        } else if !workers.initialized {
            Err(AdmissionError::NotInitialized)
        } else if !workers.ready {
            Err(AdmissionError::ModelNotLoaded)
        } else {
            Ok(workers.epoch)
        }
    }

    /// Invalidate admission before waiting for model replacement or initialization.
    pub(super) fn invalidate(&self) {
        let mut workers = self.workers();
        workers.ready = false;
        workers.epoch = workers.epoch.wrapping_add(1);
        for status in self.entries().values_mut() {
            if status.is_pending() || status.is_running() {
                *status = S::cancelled();
            }
        }
    }

    /// Publish readiness observed under the model lock by a lifecycle operation.
    pub(super) fn publish(&self, initialized: bool, ready: bool) {
        let mut workers = self.workers();
        workers.epoch = workers.epoch.wrapping_add(1);
        workers.initialized = initialized;
        workers.ready = initialized && ready;
        workers.accepting = true;
    }

    #[cfg(test)]
    pub(super) fn reopen(&self) {
        self.publish(true, true);
    }

    pub(super) fn is_running(&self, id: i64) -> bool {
        self.entries().get(&id).is_some_and(RequestState::is_running)
    }

    fn begin(&self, id: i64) -> bool {
        let mut entries = self.entries();
        match entries.get_mut(&id) {
            Some(status) if status.is_pending() => {
                *status = S::running();
                true
            }
            _ => false,
        }
    }

    fn finish(&self, id: i64, terminal: S) {
        if let Some(status) = self.entries().get_mut(&id) {
            if status.is_running() {
                *status = terminal;
            }
        }
    }

    /// Register the request and its owned worker before shutdown can drain it.
    pub(super) fn spawn(
        self: &Arc<Self>,
        id: i64,
        ticket: u64,
        work: impl FnOnce() -> S + Send + 'static,
    ) -> Result<(), String> {
        let mut workers = self.workers();
        if !workers.accepting || workers.epoch != ticket || id <= 0 {
            return Err("Async admission closed or initialization changed".to_string());
        }
        let mut index = 0;
        while index < workers.handles.len() {
            if workers.handles[index].is_finished() {
                let _ = workers.handles.swap_remove(index).join();
            } else {
                index += 1;
            }
        }
        self.entries().insert(id, S::pending());
        let requests = Arc::clone(self);
        let handle = std::thread::Builder::new()
            .name("pcai-request".to_string())
            .spawn(move || {
                if !requests.begin(id) {
                    return;
                }
                let terminal = catch_unwind(AssertUnwindSafe(work))
                    .unwrap_or_else(|_| S::failed("Async generation worker panicked".to_string()));
                requests.finish(id, terminal);
            });
        match handle {
            Ok(handle) => {
                workers.handles.push(handle);
                Ok(())
            }
            Err(error) => {
                self.entries().remove(&id);
                Err(format!("Async worker spawn failed: {error}"))
            }
        }
    }

    /// Close admission, logically cancel requests, then join without model/map locks.
    ///
    /// Generation kernels are not interrupted; shutdown waits for owned work to exit.
    pub(super) fn close_and_join(&self) {
        let handles = {
            let mut workers = self.workers();
            workers.accepting = false;
            workers.initialized = false;
            workers.ready = false;
            workers.epoch = workers.epoch.wrapping_add(1);
            for status in self.entries().values_mut() {
                if status.is_pending() || status.is_running() {
                    *status = S::cancelled();
                }
            }
            std::mem::take(&mut workers.handles)
        };
        for handle in handles {
            let _ = handle.join();
        }
    }
}
