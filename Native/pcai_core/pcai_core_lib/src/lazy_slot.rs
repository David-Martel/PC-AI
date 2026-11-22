//! Lazily initialised, fallible process-wide state.
//!
//! sysinfo 0.40 made `System::new*` fallible, so the shared samplers can no longer sit
//! in a `OnceLock` (which cannot hold "not yet initialised" after a failure). They live
//! in a `Mutex<Option<T>>` and are filled through [`get_or_try_init`].

use crate::PcaiStatus;

/// Returns the value in `slot`, creating it with `make` on first use.
///
/// A failed `make` leaves `slot` empty, so the next call retries instead of caching the
/// failure. A filled slot is reused and `make` is not called.
///
/// # Errors
///
/// Returns the error produced by `make`.
pub(crate) fn get_or_try_init<T>(
    slot: &mut Option<T>,
    make: impl FnOnce() -> Result<T, PcaiStatus>,
) -> Result<&mut T, PcaiStatus> {
    if slot.is_none() {
        *slot = Some(make()?);
    }
    // Unreachable: the slot was filled above. Reported rather than unwrapped, because
    // library code does not panic.
    slot.as_mut().ok_or(PcaiStatus::InternalError)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Protects: a failed sampler construction is returned and retried, not cached.
    /// Detects: swapping the `Mutex<Option<T>>` slot back to a `OnceLock`, or storing a
    /// placeholder on failure, which would make every later call fail or read stale data.
    /// Needs: no sysinfo; an injected constructor stands in for `System::new*`.
    #[test]
    fn failed_init_leaves_slot_empty_then_success_is_filled_and_reused() {
        let mut slot: Option<u32> = None;

        let failed = get_or_try_init(&mut slot, || Err(PcaiStatus::NotImplemented));
        assert_eq!(failed.err(), Some(PcaiStatus::NotImplemented));
        assert!(slot.is_none(), "a failed init must not fill the slot");

        let built = get_or_try_init(&mut slot, || Ok(7)).map(|value| *value);
        assert_eq!(built, Ok(7));
        assert_eq!(slot, Some(7));

        // A constructor that would fail proves the filled slot is reused, not rebuilt.
        let reused = get_or_try_init(&mut slot, || Err(PcaiStatus::InternalError)).map(|value| *value);
        assert_eq!(reused, Ok(7));
    }
}
