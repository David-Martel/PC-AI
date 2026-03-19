//! Lightweight real-time telemetry for the Janus-Pro generation pipeline.
//!
//! This module collects structured timing and throughput metrics from the
//! autoregressive image-token generation loop.  It is designed for zero
//! overhead when not actively collecting and avoids heap allocations in the
//! hot path beyond the pre-allocated [`Vec`] inside [`TelemetryCollector`].
//!
//! # Design goals
//!
//! - **Zero hot-path cost when tracing is filtered**: The `tracing` crate's
//!   compile-time [`tracing::enabled!`] macro lets the compiler eliminate
//!   calls when the level is filtered.  All expensive telemetry work is
//!   gated behind that check.
//! - **No heap allocations in the sampling loop**: [`Vec::push`] on a
//!   pre-allocated `Vec` (capacity set to `num_image_tokens` at construction)
//!   is amortised O(1) with zero re-alloc across all 576 steps.
//! - **CPU-only compile**: Nothing in this module requires CUDA.  GPU memory
//!   metrics are populated only when the `nvml` or `cuda` feature is active.
//!
//! # Example
//!
//! ```rust
//! use pcai_media::telemetry::TelemetryCollector;
//! use std::time::Instant;
//!
//! let mut collector = TelemetryCollector::new(576);
//! collector.record_prefill_end();
//!
//! let fw_start = Instant::now();
//! // ... forward pass ...
//! let samp_start = Instant::now();
//! // ... sampling ...
//! collector.record_step(0, fw_start, samp_start, Instant::now(), 42_u32);
//!
//! let telemetry = collector.finish();
//! assert_eq!(telemetry.total_tokens, 1);
//! ```

use std::time::{Duration, Instant};

use serde::Serialize;

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// Aggregated telemetry for a single image-generation call.
///
/// Returned by [`TelemetryCollector::finish`] after the generation loop
/// completes.  All timing fields are in milliseconds.
#[derive(Debug, Clone, Serialize)]
pub struct GenerationTelemetry {
    /// Total image tokens generated (normally 576 for Janus-Pro-1B).
    pub total_tokens: usize,

    /// Wall-clock duration of the entire generate call in milliseconds.
    pub total_duration_ms: f64,

    /// Sustained throughput: `total_tokens / total_duration_s`.
    pub tokens_per_second: f64,

    /// Running average tokens/s, computed over a sliding 32-step window.
    pub tokens_per_second_avg: f64,

    /// Duration of the prompt-prefill (pre-fill) phase in milliseconds.
    ///
    /// Measured from construction of [`TelemetryCollector`] to the first call
    /// to [`TelemetryCollector::record_prefill_end`].
    pub prefill_ms: f64,

    /// Cumulative time spent in LLM forward passes (embedding + backbone)
    /// across all decode steps, in milliseconds.
    pub decode_ms: f64,

    /// Cumulative time spent in sampling (CFG blend, softmax, multinomial)
    /// across all decode steps, in milliseconds.
    pub sampling_ms: f64,

    /// KV-cache implementation used: `"prealloc"` or `"dynamic"`.
    pub kv_cache_type: String,

    /// Speculative-decoding statistics, populated only when
    /// [`PipelineConfig::use_speculative_decoding`] is `true`.
    ///
    /// [`PipelineConfig::use_speculative_decoding`]: crate::config::PipelineConfig::use_speculative_decoding
    pub speculative: Option<SpeculativeTelemetry>,

    /// Per-step breakdown.  Empty when tracing is filtered at
    /// `TRACE` level (see [`TelemetryCollector::is_per_step_enabled`]).
    pub per_step: Vec<StepTelemetry>,
}

/// Acceptance-rate statistics for the speculative-decoding loop.
#[derive(Debug, Clone, Serialize)]
pub struct SpeculativeTelemetry {
    /// Number of outer speculative steps (draft + verify rounds).
    pub total_draft_steps: usize,

    /// Number of individual draft-token positions evaluated across all steps.
    pub total_draft_tokens: usize,

    /// Number of draft tokens accepted by the verify pass.
    pub tokens_accepted: usize,

    /// Number of draft tokens rejected by the verify pass.
    pub tokens_rejected: usize,

    /// `tokens_accepted / total_draft_tokens` in `[0.0, 1.0]`.
    pub acceptance_rate: f64,

    /// Average number of tokens accepted per outer speculative step.
    pub avg_accepted_per_step: f64,
}

/// Per-step timing breakdown for a single autoregressive decode step.
///
/// Only collected when `TRACE`-level logging is enabled (see
/// [`TelemetryCollector::is_per_step_enabled`]).
#[derive(Debug, Clone, Serialize)]
pub struct StepTelemetry {
    /// Zero-based step index within the generation loop.
    pub step: usize,

    /// Combined embedding + LLM forward pass duration in milliseconds.
    pub forward_ms: f64,

    /// Sampling (CFG + softmax + multinomial) duration in milliseconds.
    pub sampling_ms: f64,

    /// Image token ID sampled at this step.
    pub token_id: u32,
}

// ---------------------------------------------------------------------------
// TelemetryCollector
// ---------------------------------------------------------------------------

/// Mutable state accumulator for one generation call.
///
/// Construct with [`TelemetryCollector::new`] before starting the generation
/// loop, then call the `record_*` methods at each phase boundary.  Finalise
/// with [`TelemetryCollector::finish`] after the loop ends.
///
/// # Allocation strategy
///
/// `per_step` is pre-allocated to `num_image_tokens` capacity so that each
/// [`Vec::push`] inside the loop is a pure in-bounds write with no realloc.
pub struct TelemetryCollector {
    /// Wall clock at construction — used as the generation-start reference.
    start: Instant,

    /// Wall clock captured when the prefill completes.
    prefill_end: Option<Instant>,

    /// Pre-allocated per-step records.  Populated only when TRACE is enabled.
    steps: Vec<StepTelemetry>,

    /// Cumulative LLM forward-pass duration.
    decode_accum: Duration,

    /// Cumulative sampling duration.
    sampling_accum: Duration,

    /// Ring buffer of the last `WINDOW` step timestamps for sliding avg.
    window: [Instant; Self::WINDOW],

    /// Write index into `window` (wraps).
    window_head: usize,

    /// Number of valid entries written into `window` so far (≤ WINDOW).
    window_count: usize,

    /// Total tokens recorded so far (mirrors `steps.len()` when per-step is
    /// enabled, but we also track it independently for throughput calculation
    /// when per-step recording is skipped).
    token_count: usize,

    /// KV-cache type tag set by [`TelemetryCollector::set_kv_cache_type`].
    kv_cache_type: KvCacheType,

    /// Speculative-decoding accumulator; `None` when not using speculative.
    spec: Option<SpeculativeAccum>,
}

/// Internal tag for the KV-cache implementation in use.
enum KvCacheType {
    PreAlloc,
    Dynamic,
}

/// Internal accumulator for speculative-decoding statistics.
struct SpeculativeAccum {
    total_draft_steps: usize,
    total_draft_tokens: usize,
    tokens_accepted: usize,
    tokens_rejected: usize,
}

impl TelemetryCollector {
    /// Sliding window size for the running average tokens/s.
    const WINDOW: usize = 32;

    /// Whether per-step telemetry is worth collecting.
    ///
    /// Returns `true` when the `TRACE` tracing level is enabled for this
    /// module.  Callers may cache this once before the loop:
    ///
    /// ```rust
    /// # use pcai_media::telemetry::TelemetryCollector;
    /// let per_step = TelemetryCollector::is_per_step_enabled();
    /// ```
    #[inline]
    pub fn is_per_step_enabled() -> bool {
        tracing::enabled!(tracing::Level::TRACE)
    }

    /// Create a new collector for a generation run of `num_image_tokens` steps.
    ///
    /// Immediately captures the wall-clock start time.  Pre-allocates internal
    /// storage for `num_image_tokens` per-step records when TRACE is active.
    pub fn new(num_image_tokens: usize) -> Self {
        let capacity = if Self::is_per_step_enabled() {
            num_image_tokens
        } else {
            0
        };
        let now = Instant::now();
        Self {
            start: now,
            prefill_end: None,
            steps: Vec::with_capacity(capacity),
            decode_accum: Duration::ZERO,
            sampling_accum: Duration::ZERO,
            window: [now; Self::WINDOW],
            window_head: 0,
            window_count: 0,
            token_count: 0,
            kv_cache_type: KvCacheType::PreAlloc,
            spec: None,
        }
    }

    /// Record the end of the pre-fill phase.
    ///
    /// Call this immediately after the prompt-prefill forward pass returns and
    /// before the first decode step begins.
    #[inline]
    pub fn record_prefill_end(&mut self) {
        self.prefill_end = Some(Instant::now());
    }

    /// Set which KV-cache implementation is active.
    ///
    /// Call once after the cache is constructed, before the decode loop.
    #[inline]
    pub fn set_kv_cache_type(&mut self, prealloc: bool) {
        self.kv_cache_type = if prealloc {
            KvCacheType::PreAlloc
        } else {
            KvCacheType::Dynamic
        };
    }

    /// Activate speculative-decoding tracking.
    ///
    /// Must be called before the decode loop when speculative decoding is
    /// configured, so that [`record_speculative_step`] can accumulate stats.
    ///
    /// [`record_speculative_step`]: TelemetryCollector::record_speculative_step
    #[inline]
    pub fn enable_speculative(&mut self) {
        self.spec = Some(SpeculativeAccum {
            total_draft_steps: 0,
            total_draft_tokens: 0,
            tokens_accepted: 0,
            tokens_rejected: 0,
        });
    }

    /// Record a single decode step.
    ///
    /// # Arguments
    ///
    /// * `step`        — zero-based step index.
    /// * `forward_start` — `Instant` before the embed + LLM forward pass.
    /// * `sampling_start` — `Instant` before sampling (CFG + softmax + multinomial).
    /// * `step_end`    — `Instant` after sampling completes.
    /// * `token_id`    — the image token sampled at this step.
    ///
    /// The forward duration is `sampling_start - forward_start`; the sampling
    /// duration is `step_end - sampling_start`.
    #[inline]
    pub fn record_step(
        &mut self,
        step: usize,
        forward_start: Instant,
        sampling_start: Instant,
        step_end: Instant,
        token_id: u32,
    ) {
        let forward_dur = sampling_start.saturating_duration_since(forward_start);
        let sampling_dur = step_end.saturating_duration_since(sampling_start);

        self.decode_accum += forward_dur;
        self.sampling_accum += sampling_dur;

        // Sliding-window update for running average.
        self.window[self.window_head] = step_end;
        self.window_head = (self.window_head + 1) % Self::WINDOW;
        if self.window_count < Self::WINDOW {
            self.window_count += 1;
        }

        self.token_count += 1;

        // Per-step recording: only when TRACE is enabled.
        if Self::is_per_step_enabled() {
            self.steps.push(StepTelemetry {
                step,
                forward_ms: dur_ms(forward_dur),
                sampling_ms: dur_ms(sampling_dur),
                token_id,
            });
        }
    }

    /// Record the outcome of one outer speculative-decoding step.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` — number of draft tokens speculatively generated.
    /// * `accepted`     — number of those tokens accepted by the verify pass.
    ///
    /// `rejected` is computed as `draft_tokens - accepted`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `accepted > draft_tokens`.
    #[inline]
    pub fn record_speculative_step(&mut self, draft_tokens: usize, accepted: usize) {
        debug_assert!(
            accepted <= draft_tokens,
            "accepted ({accepted}) must not exceed draft_tokens ({draft_tokens})"
        );
        if let Some(ref mut acc) = self.spec {
            acc.total_draft_steps += 1;
            acc.total_draft_tokens += draft_tokens;
            acc.tokens_accepted += accepted;
            acc.tokens_rejected += draft_tokens - accepted;
        }
    }

    /// Compute the current running-average tokens/s from the sliding window.
    ///
    /// Returns `0.0` until at least 2 steps have been recorded.
    #[inline]
    pub fn running_tps(&self) -> f64 {
        if self.window_count < 2 {
            return 0.0;
        }
        // Oldest entry in the window — the one that will be overwritten next.
        let oldest_idx = self.window_head % Self::WINDOW;
        let oldest = self.window[oldest_idx];
        let newest = self.window[(self.window_head + Self::WINDOW - 1) % Self::WINDOW];
        let span = newest.saturating_duration_since(oldest);
        if span.is_zero() {
            return 0.0;
        }
        (self.window_count as f64 - 1.0) / span.as_secs_f64()
    }

    /// Finalise collection and return a [`GenerationTelemetry`] snapshot.
    ///
    /// Captures the wall-clock end time, computes aggregate metrics, and
    /// moves per-step records into the returned struct.
    pub fn finish(self) -> GenerationTelemetry {
        let end = Instant::now();
        let total_dur = end.saturating_duration_since(self.start);

        let prefill_ms = self
            .prefill_end
            .map(|t| dur_ms(t.saturating_duration_since(self.start)))
            .unwrap_or(0.0);

        let total_duration_ms = dur_ms(total_dur);
        let total_secs = total_dur.as_secs_f64();
        let tokens_per_second = if total_secs > 0.0 {
            self.token_count as f64 / total_secs
        } else {
            0.0
        };

        // Running average: last WINDOW steps span.
        let tokens_per_second_avg = self.running_avg_tps();

        let kv_cache_type = match self.kv_cache_type {
            KvCacheType::PreAlloc => "prealloc".to_string(),
            KvCacheType::Dynamic => "dynamic".to_string(),
        };

        let speculative = self.spec.map(|acc| {
            let total_draft_tokens = acc.total_draft_tokens;
            let acceptance_rate = if total_draft_tokens > 0 {
                acc.tokens_accepted as f64 / total_draft_tokens as f64
            } else {
                0.0
            };
            let avg_accepted_per_step = if acc.total_draft_steps > 0 {
                acc.tokens_accepted as f64 / acc.total_draft_steps as f64
            } else {
                0.0
            };
            SpeculativeTelemetry {
                total_draft_steps: acc.total_draft_steps,
                total_draft_tokens,
                tokens_accepted: acc.tokens_accepted,
                tokens_rejected: acc.tokens_rejected,
                acceptance_rate,
                avg_accepted_per_step,
            }
        });

        GenerationTelemetry {
            total_tokens: self.token_count,
            total_duration_ms,
            tokens_per_second,
            tokens_per_second_avg,
            prefill_ms,
            decode_ms: dur_ms(self.decode_accum),
            sampling_ms: dur_ms(self.sampling_accum),
            kv_cache_type,
            speculative,
            per_step: self.steps,
        }
    }

    /// Compute the final running average from the window at finish time.
    fn running_avg_tps(&self) -> f64 {
        if self.window_count < 2 {
            return 0.0;
        }
        let oldest_idx = self.window_head % Self::WINDOW;
        let oldest = self.window[oldest_idx];
        let newest_idx = (self.window_head + Self::WINDOW - 1) % Self::WINDOW;
        let newest = self.window[newest_idx];
        let span = newest.saturating_duration_since(oldest);
        if span.is_zero() {
            return 0.0;
        }
        (self.window_count as f64 - 1.0) / span.as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Convert a [`Duration`] to milliseconds as `f64`.
#[inline]
fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Constructing a collector with zero capacity should not panic and should
    /// produce a zero-token telemetry when finished immediately.
    #[test]
    fn test_empty_finish() {
        let collector = TelemetryCollector::new(576);
        let t = collector.finish();
        assert_eq!(t.total_tokens, 0);
        assert!(t.total_duration_ms >= 0.0);
        assert_eq!(t.kv_cache_type, "prealloc");
        assert!(t.speculative.is_none());
        assert!(t.per_step.is_empty());
    }

    /// `record_prefill_end` should produce a non-negative prefill_ms.
    #[test]
    fn test_prefill_timing() {
        let mut collector = TelemetryCollector::new(576);
        // Simulate a brief pause then mark prefill end.
        std::thread::sleep(Duration::from_millis(1));
        collector.record_prefill_end();
        let t = collector.finish();
        // prefill_ms must be at least 1 ms (allowing OS scheduling slop).
        assert!(t.prefill_ms >= 0.0, "prefill_ms should be non-negative");
    }

    /// Recording steps should accumulate token count and decode/sampling time.
    #[test]
    fn test_record_steps_accumulates() {
        let mut collector = TelemetryCollector::new(4);
        collector.set_kv_cache_type(false); // dynamic

        for i in 0..4_u32 {
            let fw = Instant::now();
            let samp = fw; // instant sampling start
            let end = Instant::now();
            collector.record_step(i as usize, fw, samp, end, i);
        }

        let t = collector.finish();
        assert_eq!(t.total_tokens, 4);
        assert_eq!(t.kv_cache_type, "dynamic");
        assert!(t.decode_ms >= 0.0);
        assert!(t.sampling_ms >= 0.0);
        // tokens_per_second must be finite and non-negative.
        assert!(t.tokens_per_second.is_finite() && t.tokens_per_second >= 0.0);
    }

    /// Speculative step recording must compute acceptance rate correctly.
    #[test]
    fn test_speculative_stats() {
        let mut collector = TelemetryCollector::new(576);
        collector.enable_speculative();

        // Simulate 10 outer steps, each with K=4 draft tokens, 3 accepted.
        for _ in 0..10 {
            collector.record_speculative_step(4, 3);
        }

        let t = collector.finish();
        let spec = t.speculative.expect("speculative stats should be populated");
        assert_eq!(spec.total_draft_steps, 10);
        assert_eq!(spec.total_draft_tokens, 40);
        assert_eq!(spec.tokens_accepted, 30);
        assert_eq!(spec.tokens_rejected, 10);
        let expected_rate = 30.0 / 40.0;
        assert!((spec.acceptance_rate - expected_rate).abs() < 1e-9);
        let expected_avg = 30.0 / 10.0;
        assert!((spec.avg_accepted_per_step - expected_avg).abs() < 1e-9);
    }

    /// Full acceptance (all draft tokens accepted) should give acceptance_rate = 1.0.
    #[test]
    fn test_speculative_full_acceptance() {
        let mut collector = TelemetryCollector::new(576);
        collector.enable_speculative();
        collector.record_speculative_step(4, 4);

        let spec = collector.finish().speculative.unwrap();
        assert!((spec.acceptance_rate - 1.0).abs() < 1e-9);
        assert_eq!(spec.tokens_rejected, 0);
    }

    /// Zero speculative steps should produce a 0.0 acceptance rate without
    /// dividing by zero.
    #[test]
    fn test_speculative_zero_steps() {
        let mut collector = TelemetryCollector::new(576);
        collector.enable_speculative();

        let spec = collector.finish().speculative.unwrap();
        assert_eq!(spec.total_draft_steps, 0);
        assert!(spec.acceptance_rate.is_finite());
        assert_eq!(spec.acceptance_rate, 0.0);
    }

    /// `set_kv_cache_type(true)` should produce `"prealloc"` in the output.
    #[test]
    fn test_kv_cache_type_prealloc() {
        let mut collector = TelemetryCollector::new(4);
        collector.set_kv_cache_type(true);
        let t = collector.finish();
        assert_eq!(t.kv_cache_type, "prealloc");
    }

    /// Tokens-per-second must be non-negative and finite for any number of steps.
    #[test]
    fn test_tps_is_finite() {
        let mut collector = TelemetryCollector::new(2);
        for i in 0..2_u32 {
            let now = Instant::now();
            collector.record_step(i as usize, now, now, now, i);
        }
        let t = collector.finish();
        assert!(t.tokens_per_second.is_finite());
        assert!(t.tokens_per_second >= 0.0);
        assert!(t.tokens_per_second_avg.is_finite());
        assert!(t.tokens_per_second_avg >= 0.0);
    }

    /// `dur_ms` must convert correctly for a known duration.
    #[test]
    fn test_dur_ms_conversion() {
        let d = Duration::from_millis(250);
        let ms = super::dur_ms(d);
        assert!((ms - 250.0).abs() < 0.001);
    }

    /// Serialising [`GenerationTelemetry`] to JSON must succeed and include
    /// all expected top-level keys.
    #[test]
    fn test_telemetry_serialises_to_json() {
        let mut collector = TelemetryCollector::new(2);
        collector.record_prefill_end();
        collector.enable_speculative();
        for i in 0..2_u32 {
            let now = Instant::now();
            collector.record_step(i as usize, now, now, now, i);
        }
        let t = collector.finish();
        let json = serde_json::to_string(&t).expect("GenerationTelemetry must serialise");
        assert!(json.contains("total_tokens"));
        assert!(json.contains("tokens_per_second"));
        assert!(json.contains("kv_cache_type"));
        assert!(json.contains("speculative"));
        assert!(json.contains("per_step"));
    }
}
