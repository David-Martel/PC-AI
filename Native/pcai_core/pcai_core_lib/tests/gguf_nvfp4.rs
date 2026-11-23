//! NVFP4 and tensor-size tests for the GGUF layout reader.
//!
//! `read_gguf_layout_from` adds a tensor-info parser to the hardened header
//! reader (see `gguf_hardening.rs`), so a model's weight size comes from each
//! tensor's ggml type and shape instead of a guess from `general.file_type`.
//! These tests build small GGUF headers in memory, with tensor infos and no
//! tensor data, and check three properties: NVFP4 is sized from its real block
//! layout, a tensor type the reader does not know is an error and never a size
//! of zero, and a hostile tensor section fails without panicking or allocating
//! from a declared count.
//!
//! Upstream constants: llama.cpp `ad50351f5092436f312d98d25689207df25f4802`
//! (b12635). NVFP4 is ggml tensor type 40 (`ggml/include/ggml.h`), file type 39
//! (`include/llama.h`), and one block is 64 values in 36 bytes
//! (`block_nvfp4`, `ggml/src/ggml-common.h`).
//!
//! The module under test is compiled only with the `nvml` feature, as in
//! `gguf_hardening.rs`.

#![cfg(feature = "nvml")]

use std::io::{self, Cursor};

use pcai_core_lib::preflight::gguf::{
    estimate_layout_memory_mb, estimate_model_memory_mb, read_gguf_layout, read_gguf_layout_from, read_gguf_meta_from,
    GgmlType, GgufFileType, GgufLayout, GgufLimitError, GgufTensorError, GgufTensorSizeError, MAX_TENSOR_COUNT,
};
use proptest::prelude::*;

// -- Builders ---------------------------------------------------------------------

const TYPE_U32: u32 = 4;
const TYPE_STRING: u32 = 8;

/// GGUF `general.file_type` value of NVFP4 (`LLAMA_FTYPE_MOSTLY_NVFP4`).
const FILE_TYPE_NVFP4: u32 = 39;
/// ggml tensor type id of NVFP4 (`GGML_TYPE_NVFP4`).
const TENSOR_NVFP4: u32 = 40;
const TENSOR_F32: u32 = 0;
const TENSOR_F64: u32 = 28;

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn kv_string(b: &mut Vec<u8>, key: &str, value: &str) {
    put_str(b, key);
    b.extend_from_slice(&TYPE_STRING.to_le_bytes());
    put_str(b, value);
}

fn kv_u32(b: &mut Vec<u8>, key: &str, value: u32) {
    put_str(b, key);
    b.extend_from_slice(&TYPE_U32.to_le_bytes());
    b.extend_from_slice(&value.to_le_bytes());
}

/// A tensor-info record: name, extents, ggml type id and data offset.
fn tensor_info(b: &mut Vec<u8>, name: &str, dims: &[u64], type_id: u32) {
    put_str(b, name);
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&type_id.to_le_bytes());
    b.extend_from_slice(&0_u64.to_le_bytes()); // data offset; never followed
}

/// GGUF v3 header declaring `tensor_count` tensors and the model's metadata.
///
/// The model is a 32-layer, 4096-wide llama with a 4096-token context whose
/// `general.file_type` is `file_type`.
fn header_with(tensor_count: u64, file_type: u32) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3_u32.to_le_bytes());
    b.extend_from_slice(&tensor_count.to_le_bytes());
    b.extend_from_slice(&5_u64.to_le_bytes()); // metadata entries
    kv_string(&mut b, "general.architecture", "llama");
    kv_u32(&mut b, "general.file_type", file_type);
    kv_u32(&mut b, "llama.context_length", 4096);
    kv_u32(&mut b, "llama.embedding_length", 4096);
    kv_u32(&mut b, "llama.block_count", 32);
    b
}

/// The weights of the synthetic model: one NVFP4 matrix and one F32 norm vector.
///
/// - `blk.0.ffn_up.weight`: NVFP4 `[4096, 40960]`, 64 blocks per row, so
///   `64 * 36 * 40960 = 94_371_840` bytes, exactly 90 MiB.
/// - `blk.0.attn_norm.weight`: F32 `[4096]`, `4096 * 4 = 16_384` bytes.
const FIXTURE_TENSOR_BYTES: u64 = 94_371_840 + 16_384;

/// A valid header and two tensor infos, with no tensor data after them.
fn nvfp4_fixture() -> Vec<u8> {
    let mut b = header_with(2, FILE_TYPE_NVFP4);
    tensor_info(&mut b, "blk.0.ffn_up.weight", &[4096, 40_960], TENSOR_NVFP4);
    tensor_info(&mut b, "blk.0.attn_norm.weight", &[4096], TENSOR_F32);
    b
}

fn layout(bytes: &[u8]) -> anyhow::Result<GgufLayout> {
    read_gguf_layout_from(Cursor::new(bytes))
}

fn tensor_err(err: &anyhow::Error) -> GgufTensorError {
    err.downcast_ref::<GgufTensorError>()
        .cloned()
        .unwrap_or_else(|| panic!("expected a GgufTensorError, got: {err:#}"))
}

fn is_truncation(err: &anyhow::Error) -> bool {
    err.chain()
        .filter_map(|cause| cause.downcast_ref::<io::Error>())
        .any(|e| e.kind() == io::ErrorKind::UnexpectedEof)
}

// -- NVFP4 sizing -----------------------------------------------------------------

/// Positive control and the headline property: the synthetic NVFP4 header sums to the
/// hand-computed byte count and the file type is recognised as NVFP4.
///
/// Detects: a wrong NVFP4 block size (64) or bytes per block (36), a wrong type id (40), a
/// wrong file-type id (39), and a reader that skips the tensor-info section.
#[test]
fn synthetic_nvfp4_header_sums_to_the_hand_computed_bytes() {
    let bytes = nvfp4_fixture();
    let l = layout(&bytes).expect("a valid NVFP4 header parses");
    assert!(
        matches!(l.meta.file_type, GgufFileType::Nvfp4),
        "got {:?}",
        l.meta.file_type
    );
    assert_eq!(l.tensor_count, 2);
    assert_eq!(l.tensor_bytes, FIXTURE_TENSOR_BYTES);
    assert_eq!(l.tensor_bytes, 94_388_224, "hand-computed: 90 MiB + 16 KiB");
}

/// The reader consumes the whole header and nothing more: with no tensor data in the fixture,
/// the bytes it read are the fixture's length.
///
/// Detects: a reader that stops before the tensor infos or reads past them.
#[test]
fn header_bytes_cover_exactly_the_metadata_and_tensor_infos() {
    let bytes = nvfp4_fixture();
    let l = layout(&bytes).expect("parses");
    assert_eq!(l.header_bytes, bytes.len() as u64);
}

/// The memory estimate is the tensor total plus KV cache plus 10%, in megabytes.
///
/// Context 0 isolates the weights: 90 MiB (the 16 KiB of norms is under a megabyte) plus
/// 10% (9) is 99. Context 4096 adds `2 * 32 * 4096 * 4096 * 2` bytes of KV cache, 2048 MiB:
/// `(90 + 2048) * 1.1`, with integer division, is 2351.
///
/// Detects: the exact sum not being used (the dimension-based estimate of this same header is
/// about 1.3 GB, not 99 MB), and a changed overhead or KV formula.
#[test]
fn estimate_equals_the_hand_computed_megabytes() {
    let l = layout(&nvfp4_fixture()).expect("parses");
    assert_eq!(estimate_layout_memory_mb(&l, 0), 99);
    assert_eq!(estimate_layout_memory_mb(&l, 4096), 2351);
    assert_ne!(
        estimate_layout_memory_mb(&l, 0),
        estimate_model_memory_mb(&l.meta, 0),
        "the exact path must not collapse into the dimension-based estimate"
    );
}

/// A header with no tensors has nothing to sum, so the estimate falls back to the
/// dimension-based figure rather than reporting zero weights.
///
/// Detects: a fallback that returns zero for an empty tensor list.
#[test]
fn a_header_without_tensors_falls_back_to_the_dimension_estimate() {
    let bytes = header_with(0, FILE_TYPE_NVFP4);
    let l = layout(&bytes).expect("parses");
    assert_eq!(l.tensor_bytes, 0);
    assert_eq!(estimate_layout_memory_mb(&l, 0), estimate_model_memory_mb(&l.meta, 0));
    assert!(
        estimate_layout_memory_mb(&l, 0) > 100,
        "a 32-layer 4096-wide model is not 0 MB"
    );
}

/// The metadata-only reader is unchanged: it does not look at tensor infos at all, so a
/// header whose tensors it could not size still parses.
///
/// Detects: tensor parsing leaking into `read_gguf_meta_from`, which would change the
/// behaviour that the hardening tests and the fuzz seeds pin.
#[test]
fn the_metadata_only_reader_ignores_the_tensor_section() {
    let mut bytes = header_with(1, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "w", &[64], 9_999);
    let meta = read_gguf_meta_from(Cursor::new(bytes.as_slice())).expect("metadata only");
    assert!(matches!(meta.file_type, GgufFileType::Nvfp4));
}

/// The committed fuzz seed is this fixture byte for byte, so the fuzzer starts from an input
/// that reaches the tensor-info parser, and the seed cannot drift from the layout under test.
///
/// Detects: a fixture change that is not mirrored in `fuzz/seeds/gguf_meta/nvfp4_tensors`.
#[test]
fn the_fuzz_seed_is_the_fixture() {
    let seed: &[u8] = include_bytes!("../../fuzz/seeds/gguf_meta/nvfp4_tensors");
    assert_eq!(seed, nvfp4_fixture().as_slice());
}

// -- Unknown and invalid tensors --------------------------------------------------

/// A tensor type id this reader does not know is a typed error that names the tensor and
/// the id, not a size of zero. Upstream defines ids 0..44 (`GGML_TYPE_COUNT` is 44 at the
/// pinned commit), so 44 is the first id a newer llama.cpp could assign; `u32::MAX` is the far end.
///
/// Detects: unknown mapping to a size of zero (the model would then look tiny and "go"),
/// or to a default size.
#[test]
fn an_unknown_tensor_type_is_an_error_not_zero() {
    for id in [44_u32, 45, 1_000, u32::MAX] {
        let mut bytes = header_with(2, FILE_TYPE_NVFP4);
        tensor_info(&mut bytes, "known", &[64], TENSOR_NVFP4);
        tensor_info(&mut bytes, "future", &[64], id);
        let err = layout(&bytes).expect_err("an unknown tensor type must not parse");
        assert_eq!(tensor_err(&err), GgufTensorError::UnknownType { index: 1, id });
        assert!(err.to_string().contains("unknown ggml type"), "message: {err}");
    }
}

/// Positive control for the test above: the same shape with a known type parses, so the
/// error is about the type and not the fixture.
#[test]
fn the_same_tensor_with_a_known_type_parses() {
    let mut bytes = header_with(1, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "w", &[64], TENSOR_NVFP4);
    assert_eq!(layout(&bytes).expect("parses").tensor_bytes, 36);
}

/// Removed upstream ids (4 and 5 were Q4_2 and Q4_3) are also unknown, not zero-sized.
///
/// Detects: a table that maps gaps in the id range to something.
#[test]
fn removed_type_ids_are_unknown_too() {
    for id in [4_u32, 5, 31, 32, 33, 36, 37, 38] {
        let mut bytes = header_with(1, FILE_TYPE_NVFP4);
        tensor_info(&mut bytes, "w", &[64], id);
        let err = layout(&bytes).expect_err("removed type ids are unknown");
        assert_eq!(tensor_err(&err), GgufTensorError::UnknownType { index: 0, id });
    }
}

/// A row that is not a whole number of blocks is rejected even when the element total is a
/// multiple of the block size (`96 * 2 = 192 = 3 * 64`).
///
/// Detects: validating the element total instead of the row, which would accept this shape
/// and size it as three blocks, although upstream's loader refuses the file.
#[test]
fn a_row_that_is_not_a_block_multiple_is_rejected() {
    let mut bytes = header_with(1, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "w", &[96, 2], TENSOR_NVFP4);
    let err = layout(&bytes).expect_err("96-value rows are not NVFP4 blocks");
    assert_eq!(
        tensor_err(&err),
        GgufTensorError::InvalidShape {
            index: 0,
            source: GgufTensorSizeError::RowNotBlockMultiple {
                ggml_type: GgmlType::Nvfp4,
                row_len: 96,
                block_size: 64,
            },
        }
    );
    assert!(err.to_string().starts_with("GGUF tensor 0:"), "message: {err}");
}

/// Five declared dimensions are rejected before any extent is read (`GGML_MAX_DIMS` is 4).
///
/// Detects: sizing a tensor with more extents than upstream allows.
#[test]
fn five_dimensions_are_rejected() {
    let mut bytes = header_with(1, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "w", &[1, 1, 1, 1, 1], TENSOR_F32);
    let err = layout(&bytes).expect_err("five dimensions");
    assert_eq!(
        tensor_err(&err),
        GgufTensorError::InvalidShape {
            index: 0,
            source: GgufTensorSizeError::TooManyDims { n_dims: 5 },
        }
    );
}

/// A shape whose byte size overflows is an error. `[2^60]` of F64 is `2^63` bytes: it
/// fits `u64`, and two of them do not.
///
/// Detects: unchecked or saturating addition of the per-tensor sizes.
#[test]
fn a_total_that_overflows_u64_is_an_error() {
    let mut bytes = header_with(2, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "a", &[1 << 60], TENSOR_F64);
    tensor_info(&mut bytes, "b", &[1 << 60], TENSOR_F64);
    let err = layout(&bytes).expect_err("2^63 + 2^63 overflows u64");
    assert_eq!(tensor_err(&err), GgufTensorError::TotalOverflow { index: 1 });
}

// -- Hostile tensor sections ------------------------------------------------------

/// A declared tensor count above the limit is a typed limit error, before any tensor is read.
#[test]
fn an_excessive_tensor_count_is_a_typed_limit_error() {
    let bytes = header_with(MAX_TENSOR_COUNT + 1, FILE_TYPE_NVFP4);
    let err = layout(&bytes).expect_err("over the tensor limit");
    assert_eq!(
        err.downcast_ref::<GgufLimitError>(),
        Some(&GgufLimitError::TooManyTensors {
            count: MAX_TENSOR_COUNT + 1,
            max: MAX_TENSOR_COUNT,
        })
    );
}

/// A count that is within the limit but larger than the stream is a truncation error, and
/// returns promptly: nothing is allocated or looped over from the declared count alone.
#[test]
fn a_tensor_count_larger_than_the_stream_is_a_truncation_error() {
    let mut bytes = header_with(MAX_TENSOR_COUNT, FILE_TYPE_NVFP4);
    tensor_info(&mut bytes, "only", &[64], TENSOR_NVFP4);
    let err = layout(&bytes).expect_err("one tensor present, a million declared");
    assert!(is_truncation(&err), "expected UnexpectedEof, got: {err:#}");
}

/// A name that declares more bytes than the string limit is rejected, as in the metadata reader.
#[test]
fn an_oversized_tensor_name_is_a_typed_limit_error() {
    let mut bytes = header_with(1, FILE_TYPE_NVFP4);
    bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // name length
    let err = layout(&bytes).expect_err("name too long");
    assert!(matches!(
        err.downcast_ref::<GgufLimitError>(),
        Some(GgufLimitError::StringTooLong { .. })
    ));
}

proptest! {
    /// Whatever bytes follow a valid header, the layout reader returns and never panics,
    /// and any success has a self-consistent header length.
    #[test]
    fn arbitrary_tensor_section_bytes_never_panic(tail in proptest::collection::vec(any::<u8>(), 0..512), count in 0_u64..64) {
        let mut bytes = header_with(count, FILE_TYPE_NVFP4);
        let prefix = bytes.len() as u64;
        bytes.extend_from_slice(&tail);
        if let Ok(l) = layout(&bytes) {
            prop_assert!(l.header_bytes >= prefix);
            prop_assert!(l.header_bytes <= bytes.len() as u64);
        }
    }
}

// -- Hardware-in-the-loop ---------------------------------------------------------

/// Parse a real NVFP4 GGUF and check the header against the file on disk.
///
/// Run with the path of a real file, for example the staged
/// "Spark X2.6 4B Small NVFP4" (sha256 `553c29c63ad34fa95d8dae5fe89e4d4e5ad375bea51cddd435726508a88b9f8f`):
///
/// ```text
/// PCAI_NVFP4_GGUF=<path> cargo test -p pcai_core_lib --features nvml --locked \
///     --test gguf_nvfp4 -- --ignored nvfp4_gguf_on_disk
/// ```
///
/// Reads the header only. Asserts that `general.file_type` is NVFP4 and that the summed
/// tensor bytes are within 2% of the file size. The file also holds the header and
/// per-tensor alignment padding, so header bytes plus tensor bytes must not exceed it.
#[test]
#[ignore = "needs a real NVFP4 GGUF; set PCAI_NVFP4_GGUF to its path"]
fn nvfp4_gguf_on_disk_matches_its_estimate() {
    let path = std::env::var_os("PCAI_NVFP4_GGUF").expect("set PCAI_NVFP4_GGUF to the path of an NVFP4 GGUF");
    let path = std::path::PathBuf::from(path);
    let file_size = std::fs::metadata(&path).expect("model file metadata").len();
    let l = read_gguf_layout(&path).expect("the header of a real NVFP4 model parses");

    assert!(
        matches!(l.meta.file_type, GgufFileType::Nvfp4),
        "got {:?}",
        l.meta.file_type
    );
    assert!(l.tensor_count > 0);

    let diff = file_size.abs_diff(l.tensor_bytes);
    assert!(
        diff * 100 <= file_size * 2,
        "tensor bytes {} differ from file size {file_size} by {diff} ({:.3}%), over 2%",
        l.tensor_bytes,
        diff as f64 * 100.0 / file_size as f64
    );
    assert!(
        l.header_bytes + l.tensor_bytes <= file_size,
        "header {} + tensors {} exceed the file size {file_size}",
        l.header_bytes,
        l.tensor_bytes
    );
    eprintln!(
        "HIL: file_size={file_size} tensor_bytes={} header_bytes={} tensors={} diff={diff} ({:.4}%)",
        l.tensor_bytes,
        l.header_bytes,
        l.tensor_count,
        diff as f64 * 100.0 / file_size as f64
    );
}
