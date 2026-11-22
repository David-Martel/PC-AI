//! Adversarial-input tests for the GGUF metadata reader.
//!
//! `read_gguf_meta_from` is reachable from PowerShell through the
//! `pcai_gpu_preflight_json` FFI export, and the release profile is
//! `panic = "abort"`, so a panic or a stack overflow in the reader ends the
//! host process. Each test here pins one way a crafted file used to do that, or
//! one limit that now stops it. The file-level comment on each test says which
//! defect it guards and which mutation of the reader it detects.
//!
//! The module under test is compiled only with the `nvml` feature (it lives in
//! `preflight`), which a `--workspace` build enables through `pcai_perf_cli`.

#![cfg(feature = "nvml")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{self, Cursor, Read};

use pcai_core_lib::preflight::gguf::{
    estimate_model_memory_mb, read_gguf_meta_from, GgufFileType, GgufLimitError, GgufModelMeta, MAX_ARRAY_NESTING,
    MAX_KV_COUNT, MAX_STRING_LEN,
};
use proptest::prelude::*;

// -- Allocation tracking ----------------------------------------------------------
//
// Records the largest single allocation request made by the measuring thread
// while tracking is on. Thread-local so the other tests in this binary, which
// run concurrently, cannot disturb a measurement.

struct TrackingAllocator;

thread_local! {
    static TRACKING: Cell<bool> = const { Cell::new(false) };
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

fn note(size: usize) {
    let _ = TRACKING.try_with(|on| {
        if on.get() {
            let _ = LARGEST.try_with(|largest| {
                if size > largest.get() {
                    largest.set(size);
                }
            });
        }
    });
}

// SAFETY: every method forwards to `System` unchanged after recording a size.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

/// Run `f` and return its result with the largest single allocation it made.
fn largest_allocation_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    LARGEST.with(|l| l.set(0));
    TRACKING.with(|t| t.set(true));
    let out = f();
    TRACKING.with(|t| t.set(false));
    (out, LARGEST.with(Cell::get))
}

// -- Fixture builders -------------------------------------------------------------

const TYPE_U8: u32 = 0;
const TYPE_I32: u32 = 5;
const TYPE_F32: u32 = 6;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_U64: u32 = 10;
const TYPE_I64: u32 = 11;
const TYPE_U32: u32 = 4;

fn header(kv_count: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3_u32.to_le_bytes());
    b.extend_from_slice(&0_u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&kv_count.to_le_bytes());
    b
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn kv_header(b: &mut Vec<u8>, key: &str, vtype: u32) {
    put_str(b, key);
    b.extend_from_slice(&vtype.to_le_bytes());
}

fn kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    kv_header(b, key, TYPE_STRING);
    put_str(b, value);
}

fn kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    kv_header(b, key, TYPE_U32);
    b.extend_from_slice(&value.to_le_bytes());
}

fn kv_u64(b: &mut Vec<u8>, key: &str, value: u64) {
    kv_header(b, key, TYPE_U64);
    b.extend_from_slice(&value.to_le_bytes());
}

/// A small but realistic header: scalars, a string array and a float array.
fn valid_fixture() -> Vec<u8> {
    let mut b = header(7);
    kv_string(&mut b, "general.architecture", "llama");
    kv_u32(&mut b, "general.file_type", 15);
    kv_u32(&mut b, "llama.context_length", 4096);
    kv_u32(&mut b, "llama.embedding_length", 2048);
    kv_u32(&mut b, "llama.block_count", 16);
    kv_header(&mut b, "tokenizer.ggml.tokens", TYPE_ARRAY);
    b.extend_from_slice(&TYPE_STRING.to_le_bytes());
    b.extend_from_slice(&3_u64.to_le_bytes());
    for t in ["<s>", "a", "bc"] {
        put_str(&mut b, t);
    }
    kv_header(&mut b, "tokenizer.ggml.scores", TYPE_ARRAY);
    b.extend_from_slice(&TYPE_F32.to_le_bytes());
    b.extend_from_slice(&3_u64.to_le_bytes());
    for s in [0.0_f32, -1.0, -2.0] {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b
}

/// A header with one KV whose value is `levels` arrays nested inside each other.
///
/// Level 1 is the value itself; the innermost array is empty.
fn nested_arrays(key: &str, levels: usize) -> Vec<u8> {
    assert!(levels >= 1);
    let mut b = header(1);
    kv_header(&mut b, key, TYPE_ARRAY);
    for _ in 1..levels {
        b.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
        b.extend_from_slice(&1_u64.to_le_bytes());
    }
    b.extend_from_slice(&TYPE_U8.to_le_bytes());
    b.extend_from_slice(&0_u64.to_le_bytes());
    b
}

fn dims(embedding_length: u64, block_count: u64) -> Vec<u8> {
    let mut b = header(2);
    kv_u64(&mut b, "llama.embedding_length", embedding_length);
    kv_u64(&mut b, "llama.block_count", block_count);
    b
}

fn parse(bytes: &[u8]) -> anyhow::Result<GgufModelMeta> {
    read_gguf_meta_from(Cursor::new(bytes))
}

/// The typed limit error behind `err`, or a panic naming what it was instead.
fn limit(err: &anyhow::Error) -> GgufLimitError {
    err.downcast_ref::<GgufLimitError>()
        .cloned()
        .unwrap_or_else(|| panic!("expected a GgufLimitError, got: {err:#}"))
}

// -- Positive control -------------------------------------------------------------

/// Without this, every "returns an error" test below could pass because the
/// reader rejects everything.
#[test]
fn valid_fixture_parses_to_the_expected_meta() {
    let meta = parse(&valid_fixture()).expect("a well-formed header must parse");
    assert_eq!(meta.architecture, "llama");
    assert!(matches!(meta.file_type, GgufFileType::Q4KM));
    assert_eq!(meta.context_length, 4096);
    assert_eq!(meta.embedding_length, 2048);
    assert_eq!(meta.block_count, 16);
    assert_eq!(meta.parameter_count, 4 * 2048 * 2048 * 16 + 32_000 * 2048);
}

// -- Defect 1: unbounded array recursion ------------------------------------------

/// Detects: removing the nesting limit (the level-limit-plus-one case then parses).
#[test]
fn nesting_at_the_limit_is_accepted() {
    parse(&nested_arrays("x.nested", MAX_ARRAY_NESTING)).expect("nesting at the limit must parse");
}

/// Detects: removing the nesting limit.
#[test]
fn nesting_one_past_the_limit_is_a_typed_error() {
    let err = parse(&nested_arrays("x.nested", MAX_ARRAY_NESTING + 1)).expect_err("one level too deep must fail");
    assert_eq!(limit(&err), GgufLimitError::NestingTooDeep { max: MAX_ARRAY_NESTING });
}

/// The limit also holds on the path taken for a recognised key whose value has
/// the wrong type (the typed readers fall back to the same skipper).
#[test]
fn nesting_limit_applies_under_a_recognised_key() {
    let err = parse(&nested_arrays("llama.block_count", MAX_ARRAY_NESTING + 1)).expect_err("too deep");
    assert!(matches!(limit(&err), GgufLimitError::NestingTooDeep { .. }));
}

/// The original crash: about 200 000 nested arrays overflowed the 2 MiB test
/// thread stack, which aborts the whole process. It needs no small-stack thread
/// because the limit stops recursion at nine levels; on the unfixed reader this
/// test kills the test binary (STATUS_STACK_OVERFLOW) instead of failing.
#[test]
fn very_deep_nesting_is_rejected_without_overflowing_the_stack() {
    let err = parse(&nested_arrays("x.nested", 200_000)).expect_err("must fail");
    assert!(matches!(limit(&err), GgufLimitError::NestingTooDeep { .. }));
}

// -- Defect 2: unchecked size arithmetic ------------------------------------------

/// Detects: reverting to unchecked `4 * emb * emb * blocks` (panics under
/// overflow checks here; wraps silently in release).
#[test]
fn overflowing_dimensions_are_a_typed_error_not_a_panic_or_a_wrapped_estimate() {
    let cases = [
        (u64::MAX / 2, 4),
        (1_u64 << 32, 1), // 4 * emb * emb = 2^66
        (4096, u64::MAX), // blocks alone overflows the product
        (u64::MAX, u64::MAX),
        (u64::MAX / 32_000 + 1, 0), // only the embedding-table term overflows
    ];
    for (emb, blocks) in cases {
        let err = parse(&dims(emb, blocks)).expect_err("overflow must be reported");
        assert_eq!(
            limit(&err),
            GgufLimitError::DimensionsOverflow {
                embedding_length: emb,
                block_count: blocks
            },
            "emb={emb} blocks={blocks}"
        );
    }
}

#[test]
fn large_but_representable_dimensions_still_parse() {
    let meta = parse(&dims(1 << 20, 1 << 10)).expect("2^20 x 2^10 fits");
    assert_eq!(
        meta.parameter_count,
        4 * (1_u64 << 20) * (1_u64 << 20) * (1 << 10) + 32_000 * (1 << 20)
    );
}

#[test]
fn memory_estimate_saturates_instead_of_overflowing() {
    let meta = GgufModelMeta {
        architecture: String::new(),
        file_type: GgufFileType::F32,
        parameter_count: u64::MAX,
        context_length: u64::MAX,
        embedding_length: u64::MAX,
        block_count: u64::MAX,
    };
    let est = estimate_model_memory_mb(&meta, u64::MAX);
    assert!(
        est > 1_000_000_000,
        "an absurd model must look absurdly large, got {est}"
    );
}

#[test]
fn negative_dimensions_are_rejected_rather_than_reinterpreted_as_huge() {
    for (vtype, payload) in [
        (TYPE_I32, (-1_i32).to_le_bytes().to_vec()),
        (TYPE_I64, i64::MIN.to_le_bytes().to_vec()),
    ] {
        let mut b = header(1);
        kv_header(&mut b, "llama.block_count", vtype);
        b.extend_from_slice(&payload);
        let err = parse(&b).expect_err("negative block_count must fail");
        assert!(matches!(limit(&err), GgufLimitError::NegativeValue { .. }), "{err:#}");
    }
}

/// A recognised key carrying the wrong type used to become `Some(0)`, so the
/// parameter estimate collapsed and preflight reported "go". It now falls back
/// to the same defaults as an absent key.
#[test]
fn a_mistyped_dimension_falls_back_to_defaults_not_zero() {
    let mut b = header(2);
    kv_header(&mut b, "llama.embedding_length", TYPE_F32);
    b.extend_from_slice(&1.5_f32.to_le_bytes());
    kv_header(&mut b, "llama.block_count", TYPE_STRING);
    put_str(&mut b, "many");
    let meta = parse(&b).expect("mistyped dimensions are ignored, not fatal");
    assert_eq!(meta.embedding_length, 4096);
    assert_eq!(meta.block_count, 32);
    assert!(meta.parameter_count > 0);
}

// -- Defect 3: counts and lengths taken from the file -----------------------------

/// Detects: removing the string-length cap (the typed error is replaced by an
/// end-of-file error, or on the pre-allocating reader by a huge allocation).
#[test]
fn oversized_string_length_is_a_typed_error() {
    for len in [MAX_STRING_LEN + 1, 1 << 40, u64::MAX] {
        let mut b = header(1);
        put_str_len_only(&mut b, len);
        let err = parse(&b).expect_err("oversized string must fail");
        assert_eq!(
            limit(&err),
            GgufLimitError::StringTooLong {
                len,
                max: MAX_STRING_LEN
            }
        );
    }
}

fn put_str_len_only(b: &mut Vec<u8>, len: u64) {
    b.extend_from_slice(&len.to_le_bytes());
}

/// A declared length exactly at the cap with a short file must fail on the
/// missing bytes without reserving the declared size up front.
///
/// Detects: allocating `vec![0; len]` before reading (largest allocation is the
/// full 1 MB instead of a few bytes).
#[test]
fn declared_string_length_does_not_drive_the_allocation() {
    let mut b = header(1);
    put_str_len_only(&mut b, MAX_STRING_LEN);
    b.extend_from_slice(b"short");
    let (result, largest) = largest_allocation_during(|| parse(&b));
    assert!(result.is_err(), "a truncated string must fail");
    assert!(
        largest < 64 * 1024,
        "largest single allocation was {largest} bytes; it must track bytes actually present"
    );
}

#[test]
fn huge_array_count_with_a_short_file_fails_without_allocating() {
    // Strings, fixed-size elements and nested arrays each take a different skip path.
    for elem in [TYPE_STRING, TYPE_U8, TYPE_U32, TYPE_ARRAY] {
        let mut b = header(1);
        kv_header(&mut b, "x.big", TYPE_ARRAY);
        b.extend_from_slice(&elem.to_le_bytes());
        b.extend_from_slice(&(u64::MAX / 16).to_le_bytes());
        b.extend_from_slice(&[0; 32]);
        let (result, largest) = largest_allocation_during(|| parse(&b));
        assert!(
            result.is_err(),
            "elem type {elem}: a short file cannot hold the declared array"
        );
        assert!(
            largest < 64 * 1024,
            "elem type {elem}: largest allocation {largest} bytes"
        );
    }
}

/// Detects: computing `count * elem_size` unchecked (2^61 eight-byte elements
/// is 2^64 bytes, which wraps to zero in release and panics under overflow checks).
#[test]
fn array_byte_size_overflow_is_a_typed_error() {
    let mut b = header(1);
    kv_header(&mut b, "x.big", TYPE_ARRAY);
    b.extend_from_slice(&TYPE_U64.to_le_bytes());
    b.extend_from_slice(&(1_u64 << 61).to_le_bytes());
    let err = parse(&b).expect_err("2^61 u64 elements cannot exist");
    assert_eq!(
        limit(&err),
        GgufLimitError::ArraySizeOverflow {
            count: 1 << 61,
            elem_size: 8
        }
    );
}

#[test]
fn excessive_entry_count_is_a_typed_error() {
    for count in [MAX_KV_COUNT + 1, u64::MAX] {
        let err = parse(&header(count)).expect_err("too many entries");
        assert_eq!(
            limit(&err),
            GgufLimitError::TooManyEntries {
                count,
                max: MAX_KV_COUNT
            }
        );
    }
}

/// An endless stream must hit the read budget instead of looping for ever.
///
/// Detects: removing the per-stream byte budget. On the unfixed reader this
/// test does not return (it loops 2^62 times), so it is the one test that needs
/// an external timeout to show RED.
#[test]
fn endless_stream_stops_at_the_byte_budget() {
    let mut prefix = header(1);
    kv_header(&mut prefix, "x.endless", TYPE_ARRAY);
    prefix.extend_from_slice(&TYPE_U8.to_le_bytes());
    prefix.extend_from_slice(&(1_u64 << 62).to_le_bytes());
    let stream = Cursor::new(prefix).chain(io::repeat(0));
    assert!(read_gguf_meta_from(stream).is_err());
}

// -- Truncation ------------------------------------------------------------------

/// Every strict prefix of a valid file is an error, never a panic, and the
/// full file is accepted (so the loop is not vacuous).
///
/// This passes on the unfixed reader too (it was already EOF-safe): it is a
/// regression guard, not a red test.
#[test]
fn every_truncation_of_a_valid_header_is_an_error() {
    let full = valid_fixture();
    parse(&full).expect("positive control: the whole fixture parses");
    for cut in 0..full.len() {
        assert!(parse(&full[..cut]).is_err(), "prefix of {cut} bytes must not parse");
    }
}

// -- Never panics ----------------------------------------------------------------

/// Values that sit on arithmetic edges, mixed with arbitrary ones.
fn payload_word() -> impl Strategy<Value = [u8; 8]> {
    prop_oneof![
        Just(0_u64),
        Just(1),
        Just(u64::MAX),
        Just(u64::MAX / 2),
        Just(1 << 61),
        Just(1 << 32),
        Just(1 << 63),
        any::<u64>(),
    ]
    .prop_map(u64::to_le_bytes)
}

fn arbitrary_kv() -> impl Strategy<Value = Vec<u8>> {
    let key = prop_oneof![
        Just("general.architecture"),
        Just("general.file_type"),
        Just("llama.context_length"),
        Just("llama.embedding_length"),
        Just("llama.block_count"),
        Just("tokenizer.ggml.tokens"),
        Just("x.other"),
    ];
    // Raw value-type ids 0..=14 cover every defined type plus two undefined ones.
    (
        key,
        0_u32..=14,
        prop::collection::vec(payload_word(), 0..6),
        prop::collection::vec(any::<u8>(), 0..24),
    )
        .prop_map(|(key, vtype, words, tail)| {
            let mut b = Vec::new();
            kv_header(&mut b, key, vtype);
            for w in words {
                b.extend_from_slice(&w);
            }
            b.extend_from_slice(&tail);
            b
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2048, failure_persistence: None, ..ProptestConfig::default() })]

    /// Valid magic and version followed by structured-but-arbitrary entries.
    /// Raw random bytes would almost never get past the magic, so the header is
    /// generated and only the entries are adversarial.
    ///
    /// Detects: unchecked arithmetic on dimensions (a panic) and any new
    /// unbounded allocation or recursion.
    #[test]
    fn reader_never_panics_on_structured_garbage(
        kv_count in 0_u64..6,
        kvs in prop::collection::vec(arbitrary_kv(), 0..6),
    ) {
        let mut bytes = header(kv_count);
        for kv in kvs {
            bytes.extend_from_slice(&kv);
        }
        if let Ok(meta) = parse(&bytes) {
            let _ = estimate_model_memory_mb(&meta, meta.context_length);
        }
    }

    /// Fully arbitrary bytes behind a valid magic and version.
    #[test]
    fn reader_never_panics_on_arbitrary_tail(tail in prop::collection::vec(any::<u8>(), 0..256)) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&tail);
        let _ = parse(&bytes);
    }

    /// Arbitrary bytes from the first byte: mostly rejected at the magic, kept to
    /// show that rejection itself never panics.
    #[test]
    fn reader_never_panics_on_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..256)) {
        let _ = parse(&bytes);
    }
}
