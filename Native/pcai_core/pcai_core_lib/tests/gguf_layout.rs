//! Declared-layout tests for the GGUF validator: tensor offsets, alignment and file extent.
//!
//! `validate_gguf_layout_from` mirrors `gguf_init_from_reader` in llama.cpp
//! `ggml/src/gguf.cpp` at commit `f4f6f2e7db76e98c95cc9b7fc608ea669c806044` (b12646, the
//! build the installer pins). A file whose declared offsets disagree with the sizes the
//! reader computes is a file llama.cpp refuses to load, even when the summed size looks
//! plausible, so these tests pin each rule with a hand-built header and a chosen file length.
//! No tensor data is written anywhere: the validator reads the header and compares against a
//! length, so a header plus a number is a complete fixture.
//!
//! Every test says which mutation of the validator it detects. The four mutations the change
//! was checked against by hand are: dropping the per-tensor padding, ignoring
//! `general.alignment`, skipping the extent check, and an off-by-one in the expected offset.
//!
//! The last test is hardware-in-the-loop and `#[ignore]`d: it runs the validator over real
//! files named by environment variables. Its doc comment gives the command.
//!
//! The module under test is compiled only with the `nvml` feature, as in `gguf_nvfp4.rs`.

#![cfg(feature = "nvml")]

use std::io::{self, Cursor, Read};
use std::path::{Path, PathBuf};

use pcai_core_lib::preflight::gguf::{
    read_gguf_layout_from, validate_gguf_layout, validate_gguf_layout_from, GgufDataLayout, GgufLayoutError,
    GgufLimitError, GgufTensorError, GGUF_DEFAULT_ALIGNMENT, MAX_TENSOR_COUNT,
};
use proptest::prelude::*;

// -- Builders ---------------------------------------------------------------------

const TYPE_U32: u32 = 4;
const TYPE_I32: u32 = 5;
const TYPE_STRING: u32 = 8;
const TYPE_U64: u32 = 10;
const TENSOR_F32: u32 = 0;

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// One synthetic tensor: an F32 vector of `elements` values declared at `offset`.
#[derive(Clone)]
struct T {
    name: String,
    elements: u64,
    offset: u64,
}

impl T {
    fn new(name: &str, elements: u64, offset: u64) -> Self {
        Self {
            name: name.to_owned(),
            elements,
            offset,
        }
    }

    /// Bytes of the tensor data: F32 is 4 bytes a value.
    fn bytes(&self) -> u64 {
        self.elements * 4
    }
}

/// A raw `general.alignment` entry: the value type id and the encoded value bytes.
type AlignmentKv = (u32, Vec<u8>);

fn align_u32(value: u32) -> AlignmentKv {
    (TYPE_U32, value.to_le_bytes().to_vec())
}

/// GGUF v3 header: a model name, an optional `general.alignment` entry (or several), and the tensor infos.
fn header(alignment: &[AlignmentKv], tensors: &[T]) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3_u32.to_le_bytes());
    b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    b.extend_from_slice(&(1 + alignment.len() as u64).to_le_bytes());
    put_str(&mut b, "general.name");
    b.extend_from_slice(&TYPE_STRING.to_le_bytes());
    put_str(&mut b, "layout-fixture");
    for (type_id, value) in alignment {
        put_str(&mut b, "general.alignment");
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(value);
    }
    for t in tensors {
        put_str(&mut b, &t.name);
        b.extend_from_slice(&1_u32.to_le_bytes());
        b.extend_from_slice(&t.elements.to_le_bytes());
        b.extend_from_slice(&TENSOR_F32.to_le_bytes());
        b.extend_from_slice(&t.offset.to_le_bytes());
    }
    b
}

/// Tensors laid out the way a correct writer does: each offset the previous end padded to `align`.
fn laid_out(sizes: &[(&str, u64)], align: u64) -> Vec<T> {
    let mut offset = 0;
    sizes
        .iter()
        .map(|(name, elements)| {
            let t = T::new(name, *elements, offset);
            offset += t.bytes().next_multiple_of(align);
            t
        })
        .collect()
}

/// Three tensors of 32, 40 and 4 bytes: padded to 32, 64 and 32, so offsets 0, 32 and 96.
fn three() -> Vec<T> {
    laid_out(&[("a", 8), ("b", 10), ("c", 1)], 32)
}

/// Padded size of the data section for [`three`] at alignment 32.
const THREE_DATA_BYTES: u64 = 32 + 64 + 32;
/// The end of the last tensor of [`three`], relative to the data section: 96 + 4.
const THREE_LAST_END: u64 = 100;

fn validate(bytes: &[u8], file_len: u64) -> anyhow::Result<GgufDataLayout> {
    validate_gguf_layout_from(Cursor::new(bytes), file_len)
}

fn layout_err(err: &anyhow::Error) -> GgufLayoutError {
    err.downcast_ref::<GgufLayoutError>()
        .cloned()
        .unwrap_or_else(|| panic!("expected a GgufLayoutError, got: {err:#}"))
}

/// The absolute start of the data section for `bytes` (header padded to `align`).
fn data_start(bytes: &[u8], align: u64) -> u64 {
    (bytes.len() as u64).next_multiple_of(align)
}

// -- Valid layouts ----------------------------------------------------------------

/// Positive control and the headline property: a layout written the way upstream writes it,
/// with the data present, validates, and the report says where the data starts and how big it is.
///
/// Detects: dropping the per-tensor padding (offsets 0, 32, 72 would then be expected, so the
/// valid file is refused), an off-by-one in the expected offset, and a wrong data-section start.
#[test]
fn a_correctly_padded_layout_is_valid() {
    let bytes = header(&[], &three());
    let start = data_start(&bytes, 32);
    let got = validate(&bytes, start + THREE_DATA_BYTES).expect("valid layout");
    assert_eq!(
        got,
        GgufDataLayout {
            alignment: 32,
            tensor_count: 3,
            header_bytes: bytes.len() as u64,
            data_offset: start,
            data_bytes: THREE_DATA_BYTES,
            file_len: start + THREE_DATA_BYTES,
        }
    );
    assert_eq!(got.data_offset % 32, 0, "the data section is aligned");
}

/// The extent is the end of the last tensor, with no trailing padding needed.
///
/// A memory-mapped load needs only the tensor. The boundary case, a file exactly as long as the
/// last tensor's end, must pass, and one byte less must not (see the past-EOF test).
///
/// Detects: `>=` where the extent check needs `>`, and demanding the padded section size.
#[test]
fn a_file_without_trailing_padding_is_valid_and_the_exact_end_is_the_boundary() {
    let bytes = header(&[], &three());
    let start = data_start(&bytes, 32);
    validate(&bytes, start + THREE_LAST_END).expect("the file ends exactly at the last tensor");
    validate(&bytes, start + THREE_LAST_END + 1_000).expect("extra bytes after the data are not this check's concern");
}

/// A header with no tensors is not padded, as upstream, and needs no data.
///
/// Detects: padding the data offset when there are no tensors, which would reject a file
/// that ends at its header.
#[test]
fn a_file_with_no_tensors_ends_at_its_header() {
    let bytes = header(&[], &[]);
    let len = bytes.len() as u64;
    assert_ne!(len % 32, 0, "fixture precondition: the header is not 32-aligned");
    let got = validate(&bytes, len).expect("empty tensor list");
    assert_eq!((got.data_offset, got.data_bytes, got.tensor_count), (len, 0, 0));
}

/// A custom `general.alignment` of 64 is honoured: the same tensors, padded to 64, are valid.
///
/// Detects: ignoring `general.alignment` (the offsets 0, 64, 128 would be refused against the
/// default 32), and padding the data section to 32 instead of 64.
#[test]
fn a_custom_alignment_of_64_is_valid() {
    let tensors = laid_out(&[("a", 8), ("b", 10), ("c", 1)], 64);
    assert_eq!(tensors.iter().map(|t| t.offset).collect::<Vec<_>>(), [0, 64, 128]);
    let bytes = header(&[align_u32(64)], &tensors);
    let start = data_start(&bytes, 64);
    let got = validate(&bytes, start + 128 + 4).expect("valid at alignment 64");
    assert_eq!((got.alignment, got.data_offset, got.data_bytes), (64, start, 192));
    assert_eq!(got.data_offset % 64, 0);
}

/// The same bytes without the alignment key fall back to 32 and are then refused.
///
/// This is the pair of the test above: it proves the alignment value is what makes the first
/// layout valid, not a layout that is valid at every alignment.
///
/// Detects: a default other than 32, and an alignment that is always taken from the file.
#[test]
fn the_default_alignment_is_32_when_the_key_is_absent() {
    let tensors = laid_out(&[("a", 8), ("b", 10)], 64);
    let bytes = header(&[], &tensors);
    assert_eq!(GGUF_DEFAULT_ALIGNMENT, 32);
    let err = validate(&bytes, 10_000).expect_err("offset 64 is not the 32-aligned offset");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: 1,
            name: "b".into(),
            declared: 64,
            expected: 32
        }
    );
}

/// An alignment of 1 is a power of two and is accepted: tensors then pack with no padding.
///
/// Upstream tests `alignment & (alignment - 1)`, which is zero for 1. Detects a validator
/// stricter than upstream, for example a minimum alignment of 8.
#[test]
fn an_alignment_of_one_packs_tensors_without_padding() {
    let tensors = laid_out(&[("a", 5), ("b", 3)], 1);
    assert_eq!(tensors[1].offset, 20);
    let bytes = header(&[align_u32(1)], &tensors);
    let got = validate(&bytes, bytes.len() as u64 + 32).expect("alignment 1");
    assert_eq!(
        (got.alignment, got.data_offset, got.data_bytes),
        (1, bytes.len() as u64, 32)
    );
}

// -- Offset defects ---------------------------------------------------------------

/// A skipped gap: the second tensor is declared one alignment unit past where it must start.
///
/// Detects: skipping the offset comparison, and reporting the wrong tensor or numbers.
#[test]
fn a_gap_between_tensors_is_refused_naming_the_tensor_and_both_offsets() {
    let mut tensors = three();
    tensors[1].offset = 64;
    let bytes = header(&[], &tensors);
    let err = validate(&bytes, 1 << 20).expect_err("gap");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: 1,
            name: "b".into(),
            declared: 64,
            expected: 32
        }
    );
    assert_eq!(err.to_string(), "tensor 'b' has offset 64, expected 32");
}

/// Overlapping tensors: the second starts before the first ends.
///
/// Detects: a check that only requires offsets to be non-decreasing or only rejects gaps.
#[test]
fn overlapping_tensors_are_refused() {
    let mut tensors = three();
    tensors[1].offset = 16;
    let err = validate(&header(&[], &tensors), 1 << 20).expect_err("overlap");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: 1,
            name: "b".into(),
            declared: 16,
            expected: 32
        }
    );
}

/// The padding-dropped layout: a writer that packs tensors back to back, ignoring the alignment.
///
/// Tensor `a` is 20 bytes, so the correct next offset is 32 and the packed one is 20.
///
/// Detects: a validator that expects `previous offset + previous size` without padding.
#[test]
fn a_layout_that_omits_the_padding_is_refused() {
    let packed = laid_out(&[("a", 5), ("b", 5)], 1);
    assert_eq!(packed[1].offset, 20);
    let err = validate(&header(&[], &packed), 1 << 20).expect_err("unpadded");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: 1,
            name: "b".into(),
            declared: 20,
            expected: 32
        }
    );
}

/// A non-aligned offset: one byte past the expected, so it is neither the expected one nor a multiple of 32.
///
/// Detects: an off-by-one in the expected offset in either direction (an expected of 31 or 33
/// would accept a declared 33 or 31 respectively, and the valid layouts above would fail).
#[test]
fn an_offset_one_byte_off_is_refused_in_both_directions() {
    for declared in [31_u64, 33] {
        let mut tensors = three();
        tensors[1].offset = declared;
        let err = validate(&header(&[], &tensors), 1 << 20).expect_err("off by one");
        assert_eq!(
            layout_err(&err),
            GgufLayoutError::TensorOffsetMismatch {
                index: 1,
                name: "b".into(),
                declared,
                expected: 32
            }
        );
    }
}

/// The first tensor must be at offset 0, not merely aligned.
///
/// Detects: a loop that starts its expectation from the first tensor's declared offset.
#[test]
fn the_first_tensor_must_start_at_zero() {
    let mut tensors = three();
    tensors[0].offset = 32;
    let err = validate(&header(&[], &tensors), 1 << 20).expect_err("first offset");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: 0,
            name: "a".into(),
            declared: 32,
            expected: 0
        }
    );
}

/// An offset defect is reported before an extent defect: the first failing tensor wins, as upstream.
///
/// Detects: checking the extent first, which would name the truncation instead.
#[test]
fn an_offset_defect_is_reported_before_a_truncation() {
    let mut tensors = three();
    tensors[2].offset = 100;
    let err = validate(&header(&[], &tensors), 0).expect_err("offset");
    assert!(matches!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch { index: 2, .. }
    ));
}

// -- Extent defects ---------------------------------------------------------------

/// A tensor past the end of the file: one byte short fails, naming the last tensor and where it ends.
///
/// Detects: skipping the extent check, and an off-by-one in it.
#[test]
fn a_file_one_byte_short_of_the_last_tensor_is_refused() {
    let bytes = header(&[], &three());
    let start = data_start(&bytes, 32);
    let err = validate(&bytes, start + THREE_LAST_END - 1).expect_err("truncated");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorPastEof {
            index: 2,
            name: "c".into(),
            end: start + THREE_LAST_END,
            file_len: start + THREE_LAST_END - 1
        }
    );
}

/// A file that ends inside the header's padding: the data section starts past the end.
///
/// The stream still carries the whole header, so the header parses; only the length says it is cut.
///
/// Detects: a missing data-section start check, and one that compares against the unpadded header end.
#[test]
fn a_file_that_ends_before_the_data_section_is_refused() {
    let bytes = header(&[], &three());
    assert_ne!(
        bytes.len() % 32,
        0,
        "fixture precondition: the header is not 32-aligned"
    );
    let start = data_start(&bytes, 32);
    let err = validate(&bytes, bytes.len() as u64).expect_err("ends in the padding");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::DataSectionPastEof {
            data_offset: start,
            file_len: bytes.len() as u64
        }
    );
}

/// A zero-length file length is a truncation of every non-empty layout.
#[test]
fn a_zero_file_length_is_refused() {
    validate(&header(&[], &three()), 0).expect_err("zero length");
}

// -- Alignment key ----------------------------------------------------------------

/// Alignment 0 and non-powers of two are refused, as upstream (`gguf.cpp:614`).
///
/// Detects: accepting 0 (which would divide by zero in a padding formula), accepting a
/// non-power-of-two such as 3, 48 or `u32::MAX`, and checking only the low bit.
#[test]
fn an_alignment_that_is_not_a_power_of_two_is_refused() {
    for alignment in [0_u32, 3, 6, 33, 48, 100, (1 << 20) + 1, u32::MAX] {
        let bytes = header(&[align_u32(alignment)], &three());
        let err = validate(&bytes, 1 << 20).expect_err("bad alignment");
        assert_eq!(layout_err(&err), GgufLayoutError::InvalidAlignment { alignment });
        assert_eq!(err.to_string(), format!("alignment {alignment} is not a power of 2"));
    }
}

/// Every power of two up to 2^31 is a valid alignment, including ones larger than the file.
///
/// Detects: an arbitrary upper bound, and an overflow in the padding of a large alignment.
#[test]
fn every_power_of_two_alignment_is_accepted() {
    for shift in 0..32 {
        let alignment = 1_u32 << shift;
        let bytes = header(&[align_u32(alignment)], &[]);
        validate(&bytes, bytes.len() as u64).unwrap_or_else(|e| panic!("alignment {alignment}: {e:#}"));
    }
}

/// `general.alignment` must be a `UINT32`; a 64-bit or signed value is refused (`gguf.cpp:605`).
///
/// Detects: coercing other integer widths, as the metadata reader does for the dimension keys.
#[test]
fn an_alignment_key_of_another_type_is_refused() {
    for (type_id, value) in [
        (TYPE_U64, 64_u64.to_le_bytes().to_vec()),
        (TYPE_I32, 64_i32.to_le_bytes().to_vec()),
    ] {
        let bytes = header(&[(type_id, value)], &three());
        let err = validate(&bytes, 1 << 20).expect_err("wrong type");
        assert_eq!(layout_err(&err), GgufLayoutError::AlignmentKeyType { type_id });
    }
    let mut string = Vec::new();
    put_str(&mut string, "64");
    let err = validate(&header(&[(TYPE_STRING, string)], &three()), 1 << 20).expect_err("string");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::AlignmentKeyType { type_id: TYPE_STRING }
    );
}

/// Two `general.alignment` entries are refused, whichever order they come in.
///
/// Upstream refuses a duplicate of any key. Detects: last-wins or first-wins, either of which
/// would let a file that upstream rejects pass.
#[test]
fn a_duplicated_alignment_key_is_refused() {
    for pair in [
        [align_u32(32), align_u32(64)],
        [align_u32(64), align_u32(32)],
        [align_u32(32), align_u32(32)],
    ] {
        let err = validate(&header(&pair, &three()), 1 << 20).expect_err("duplicate");
        assert_eq!(layout_err(&err), GgufLayoutError::DuplicateAlignment);
    }
}

/// The alignment is checked before the tensors, so a bad alignment wins over an offset defect.
#[test]
fn the_alignment_is_checked_before_the_offsets() {
    let mut tensors = three();
    tensors[1].offset = 1;
    let err = validate(&header(&[align_u32(0)], &tensors), 1 << 20).expect_err("alignment first");
    assert_eq!(layout_err(&err), GgufLayoutError::InvalidAlignment { alignment: 0 });
}

// -- Other failures keep their own types ------------------------------------------

/// An unknown tensor type is still the sizing error, not a layout error.
#[test]
fn an_unknown_tensor_type_is_a_tensor_error() {
    let mut b = header(&[], &[]);
    // Rewrite the tensor count to 1 and append one record of type 9999.
    b[8..16].copy_from_slice(&1_u64.to_le_bytes());
    put_str(&mut b, "x");
    b.extend_from_slice(&1_u32.to_le_bytes());
    b.extend_from_slice(&8_u64.to_le_bytes());
    b.extend_from_slice(&9999_u32.to_le_bytes());
    b.extend_from_slice(&0_u64.to_le_bytes());
    let err = validate(&b, 1 << 20).expect_err("unknown type");
    assert_eq!(
        err.downcast_ref::<GgufTensorError>(),
        Some(&GgufTensorError::UnknownType { index: 0, id: 9999 })
    );
}

/// A running total that overflows `u64` is a typed overflow, not a wrapped offset.
///
/// Two tensors of 2^63 bytes each: the first is placed at 0 and fills half the range, the second
/// is declared at 2^63 (so the offset matches) and its padded size then overflows the total.
///
/// Detects: unchecked or wrapping arithmetic in the padding or the accumulation.
#[test]
fn a_total_that_overflows_u64_is_a_typed_error() {
    let half = 1_u64 << 61;
    let tensors = [T::new("big0", half, 0), T::new("big1", half, 1_u64 << 63)];
    let err = validate(&header(&[], &tensors), u64::MAX).expect_err("overflow");
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorSizeOverflow {
            index: 1,
            name: "big1".into()
        }
    );
}

/// More tensors than the reader's limit is the shared limit error, before any tensor is read.
#[test]
fn too_many_tensors_is_a_limit_error() {
    let mut b = header(&[], &[]);
    b[8..16].copy_from_slice(&(MAX_TENSOR_COUNT + 1).to_le_bytes());
    let err = validate(&b, 1 << 20).expect_err("too many");
    assert!(matches!(
        err.downcast_ref::<GgufLimitError>(),
        Some(GgufLimitError::TooManyTensors { .. })
    ));
}

/// The validator reads the header and nothing after it.
///
/// The stream is the header followed by a source that fails on any further read. A validator
/// that touched tensor data would return that error instead of the layout.
///
/// Detects: any read of the data section, which on a real model is gigabytes.
#[test]
fn the_validator_never_reads_past_the_header() {
    struct HeaderThenPoison {
        header: Cursor<Vec<u8>>,
    }
    impl Read for HeaderThenPoison {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.header.read(buf)?;
            if n == 0 && !buf.is_empty() {
                return Err(io::Error::other("read past the header"));
            }
            Ok(n)
        }
    }
    let bytes = header(&[], &three());
    let start = data_start(&bytes, 32);
    let reader = HeaderThenPoison {
        header: Cursor::new(bytes),
    };
    validate_gguf_layout_from(reader, start + THREE_DATA_BYTES).expect("header only");
}

/// The path wrapper takes the length from the file system: a header-only file is valid for
/// zero tensors and one that is cut short is refused, using a real temporary file.
#[test]
fn the_path_wrapper_measures_the_file_it_opens() {
    let dir = tempfile::tempdir().expect("temp dir");
    let bytes = header(&[], &three());
    let start = data_start(&bytes, 32) as usize;
    let mut file = bytes.clone();
    file.resize(start + THREE_LAST_END as usize, 0);
    let path = dir.path().join("whole.gguf");
    std::fs::write(&path, &file).expect("write");
    assert_eq!(validate_gguf_layout(&path).expect("whole").file_len, file.len() as u64);

    file.pop();
    let cut = dir.path().join("cut.gguf");
    std::fs::write(&cut, &file).expect("write");
    let err = validate_gguf_layout(&cut).expect_err("cut");
    assert!(matches!(layout_err(&err), GgufLayoutError::TensorPastEof { .. }));

    assert!(validate_gguf_layout(&dir.path().join("missing.gguf")).is_err());
}

// -- Properties -------------------------------------------------------------------

proptest! {
    /// For any alignment and tensor sizes, a layout padded the way upstream pads validates at its
    /// exact end; moving any one offset by any non-zero amount is refused naming that tensor; and
    /// any shorter file is refused.
    ///
    /// Detects: all four hand mutations at once (dropped padding, fixed alignment, no extent check,
    /// off-by-one), over many shapes instead of the three fixed above.
    #[test]
    fn padded_layouts_validate_and_any_displacement_or_truncation_is_refused(
        shift in 0_u32..13,
        elements in proptest::collection::vec(1_u64..5_000, 1..8),
        victim in 0_usize..8,
        delta in 1_u64..10_000,
        cut in 1_u64..1_000,
    ) {
        let align = 1_u64 << shift;
        let names: Vec<String> = (0..elements.len()).map(|i| format!("t{i}")).collect();
        let sizes: Vec<(&str, u64)> = names.iter().map(String::as_str).zip(elements.iter().copied()).collect();
        let tensors = laid_out(&sizes, align);
        let key = [align_u32(u32::try_from(align).unwrap())];
        let bytes = header(&key, &tensors);
        let start = data_start(&bytes, align);
        let last = tensors.last().unwrap();
        let end = start + last.offset + last.bytes();

        let got = validate(&bytes, end).expect("a correctly padded layout is valid");
        prop_assert_eq!(got.data_offset, start);
        prop_assert_eq!(got.data_bytes, last.offset + last.bytes().next_multiple_of(align));
        // The validator and the size reader are one parser: they agree on the header length.
        let sized = read_gguf_layout_from(bytes.as_slice()).expect("the size reader accepts it too");
        prop_assert_eq!(sized.header_bytes, got.header_bytes);

        let short = end.saturating_sub(cut);
        let err = validate(&bytes, short).expect_err("shorter file");
        let truncation = matches!(
            layout_err(&err),
            GgufLayoutError::TensorPastEof { .. } | GgufLayoutError::DataSectionPastEof { .. }
        );
        prop_assert!(truncation);

        let k = victim % tensors.len();
        let mut moved = tensors.clone();
        moved[k].offset += delta;
        let err = validate(&header(&key, &moved), end + 1_000_000).expect_err("displaced offset");
        match layout_err(&err) {
            GgufLayoutError::TensorOffsetMismatch { index, declared, expected, .. } => {
                prop_assert_eq!(index, k as u64);
                prop_assert_eq!(declared, tensors[k].offset + delta);
                prop_assert_eq!(expected, tensors[k].offset);
            }
            other => prop_assert!(false, "expected an offset mismatch, got {other:?}"),
        }
    }

    /// Arbitrary bytes never panic, and a success is internally consistent.
    #[test]
    fn arbitrary_bytes_never_panic(data in proptest::collection::vec(any::<u8>(), 0..512), len in any::<u64>()) {
        if let Ok(layout) = validate_gguf_layout_from(data.as_slice(), len) {
            prop_assert!(layout.data_offset <= layout.file_len);
            prop_assert!(layout.data_offset >= layout.header_bytes);
        }
    }
}

// -- Hardware-in-the-loop ---------------------------------------------------------

/// The upstream diagnostic for the known-bad NVFP4 file, from the `llama-b12635` loader log.
const INVALID_TENSOR: &str = "blk.0.attn_k_norm.weight";
const INVALID_DECLARED: u64 = 320_714_240;
const INVALID_EXPECTED: u64 = 320_550_400;

/// Run the validator over real files, read-only.
///
/// Environment (paths; list variables are separated by `;`):
///
/// - `PCAI_GGUF_INVALID`: a file llama.cpp refuses with `tensor blk.0.attn_k_norm.weight has
///   offset 320714240, expected 320550400` (required). The validator must name that tensor and
///   those two numbers.
/// - `PCAI_GGUF_VALID`: a valid NVFP4 file; must validate (optional).
/// - `PCAI_GGUF_POSITIVE`: valid files of other types; each must validate (optional).
/// - `PCAI_GGUF_TRUNCATE_SOURCE`: a small valid file. A temporary copy with the last mebibyte
///   removed is made in the system temp directory, must fail on extent, and is deleted after
///   (optional). The source is only read.
///
/// ```text
/// PCAI_GGUF_INVALID=<path> PCAI_GGUF_VALID=<path> PCAI_GGUF_POSITIVE="<p1>;<p2>" \
/// PCAI_GGUF_TRUNCATE_SOURCE=<path> cargo test -p pcai_core_lib --features nvml --locked \
///     --test gguf_layout -- --ignored --nocapture real_gguf_files
/// ```
///
/// Reads headers only (the truncated copy excepted, which copies the file's bytes once).
#[test]
#[ignore = "needs real GGUF files; set PCAI_GGUF_INVALID and optionally PCAI_GGUF_VALID, PCAI_GGUF_POSITIVE, PCAI_GGUF_TRUNCATE_SOURCE"]
fn real_gguf_files() {
    let invalid = PathBuf::from(std::env::var_os("PCAI_GGUF_INVALID").expect("set PCAI_GGUF_INVALID"));
    let err = validate_gguf_layout(&invalid).expect_err("the known-bad file must be refused");
    eprintln!("INVALID {}: {err}", invalid.display());
    assert_eq!(
        layout_err(&err),
        GgufLayoutError::TensorOffsetMismatch {
            index: layout_index(&err),
            name: INVALID_TENSOR.into(),
            declared: INVALID_DECLARED,
            expected: INVALID_EXPECTED,
        },
        "must report the tensor and both numbers upstream reports"
    );

    let valid = std::env::var_os("PCAI_GGUF_VALID").map(PathBuf::from);
    let positives = std::env::var("PCAI_GGUF_POSITIVE").unwrap_or_default();
    for path in valid
        .iter()
        .cloned()
        .chain(positives.split(';').filter(|p| !p.is_empty()).map(PathBuf::from))
    {
        let layout = validate_gguf_layout(&path).unwrap_or_else(|e| panic!("{} must be valid: {e:#}", path.display()));
        let padded_end = layout.data_offset + layout.data_bytes;
        eprintln!(
            "VALID {}: {layout:?}; file_len - (data_offset + data_bytes) = {}",
            path.display(),
            i128::from(layout.file_len) - i128::from(padded_end)
        );
    }

    if let Some(source) = std::env::var_os("PCAI_GGUF_TRUNCATE_SOURCE").map(PathBuf::from) {
        let copy = TempCopy::truncated(&source, 1 << 20);
        let err = validate_gguf_layout(copy.path()).expect_err("a truncated copy must be refused");
        eprintln!("TRUNCATED {} (-1 MiB): {err}", source.display());
        assert!(
            matches!(layout_err(&err), GgufLayoutError::TensorPastEof { .. }),
            "got {err:#}"
        );
        let source_layout = validate_gguf_layout(&source).expect("the intact source is valid");
        assert_eq!(source_layout.file_len - copy.len(), 1 << 20);
    }
}

fn layout_index(err: &anyhow::Error) -> u64 {
    match layout_err(err) {
        GgufLayoutError::TensorOffsetMismatch { index, .. } => index,
        other => panic!("expected an offset mismatch, got {other:?}"),
    }
}

/// A temporary copy of `source` without its last `cut` bytes, deleted on drop.
struct TempCopy {
    path: PathBuf,
    len: u64,
}

impl TempCopy {
    fn truncated(source: &Path, cut: u64) -> Self {
        let total = std::fs::metadata(source).expect("source metadata").len();
        assert!(total > cut, "source smaller than the cut");
        let path = std::env::temp_dir().join(format!("pcai-gguf-layout-truncated-{}.gguf", std::process::id()));
        let mut input = std::fs::File::open(source).expect("open source").take(total - cut);
        let mut output = std::fs::File::create(&path).expect("create temp copy");
        let len = io::copy(&mut input, &mut output).expect("copy");
        Self { path, len }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn len(&self) -> u64 {
        self.len
    }
}

impl Drop for TempCopy {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
