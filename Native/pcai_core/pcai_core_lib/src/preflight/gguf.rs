//! GGUF header parser for model memory estimation.
//!
//! Reads only the metadata section of a GGUF file (first few KB) to extract
//! the fields needed for VRAM estimation. Does NOT load tensors.

use std::io::{self, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;
use thiserror::Error;

/// Maximum nesting depth of GGUF arrays that the reader follows.
///
/// Real GGUF files hold flat arrays of numbers or strings (vocabularies,
/// scores, token types). Eight levels is far beyond any producer and keeps the
/// recursive skipper to a constant, tiny stack footprint.
pub const MAX_ARRAY_NESTING: usize = 8;

/// Maximum length in bytes of a single GGUF string.
pub const MAX_STRING_LEN: u64 = 1_000_000;

/// Maximum number of metadata key-value pairs a header may declare.
///
/// Real files declare a few dozen to a few thousand.
pub const MAX_KV_COUNT: u64 = 1 << 20;

/// Maximum number of bytes read from one stream while parsing metadata.
///
/// Large tokenizer vocabularies make the metadata section tens of megabytes;
/// this budget bounds the work done on a hostile or endless stream.
pub const MAX_METADATA_BYTES: u64 = 256 * 1024 * 1024;

/// A resource limit or arithmetic bound that GGUF metadata violated.
///
/// Returned (wrapped in [`anyhow::Error`]) by [`read_gguf_meta`] and
/// [`read_gguf_meta_from`]; recover it with
/// `err.downcast_ref::<GgufLimitError>()`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GgufLimitError {
    /// Arrays nest deeper than [`MAX_ARRAY_NESTING`].
    #[error("GGUF array nesting exceeds the limit of {max} levels")]
    NestingTooDeep {
        /// The limit that was exceeded.
        max: usize,
    },
    /// A string is longer than [`MAX_STRING_LEN`].
    #[error("GGUF string length {len} exceeds the limit of {max} bytes")]
    StringTooLong {
        /// The declared length.
        len: u64,
        /// The limit that was exceeded.
        max: u64,
    },
    /// The header declares more than [`MAX_KV_COUNT`] entries.
    #[error("GGUF metadata entry count {count} exceeds the limit of {max}")]
    TooManyEntries {
        /// The declared count.
        count: u64,
        /// The limit that was exceeded.
        max: u64,
    },
    /// An array's total byte size does not fit in `u64`.
    #[error("GGUF array of {count} elements of {elem_size} bytes overflows u64")]
    ArraySizeOverflow {
        /// The declared element count.
        count: u64,
        /// The size of one element in bytes.
        elem_size: u64,
    },
    /// The model dimensions make the parameter estimate overflow `u64`.
    #[error("GGUF embedding_length={embedding_length} and block_count={block_count} overflow the parameter estimate")]
    DimensionsOverflow {
        /// The declared hidden dimension.
        embedding_length: u64,
        /// The declared block count.
        block_count: u64,
    },
    /// A value that must be non-negative is negative.
    #[error("GGUF value {value} is negative but must not be")]
    NegativeValue {
        /// The negative value as read from the file.
        value: i64,
    },
}

/// Metadata extracted from a GGUF file header.
#[derive(Debug, Clone, Serialize)]
pub struct GgufModelMeta {
    /// Model architecture name (e.g. "llama", "phi3", "qwen2").
    pub architecture: String,
    /// Quantization file type.
    pub file_type: GgufFileType,
    /// Estimated total parameter count.
    pub parameter_count: u64,
    /// Maximum context length the model was trained with.
    pub context_length: u64,
    /// Hidden / embedding dimension.
    pub embedding_length: u64,
    /// Number of transformer blocks (layers).
    pub block_count: u64,
}

/// GGUF quantization file types.
///
/// Integer values match the GGUF `general.file_type` metadata field.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum GgufFileType {
    /// 32-bit floating point (unquantized).
    F32,
    /// 16-bit floating point (half precision).
    F16,
    /// 4-bit quantization, variant 0.
    Q4_0,
    /// 4-bit quantization, variant 1.
    Q4_1,
    /// 5-bit quantization, variant 0.
    Q5_0,
    /// 5-bit quantization, variant 1.
    Q5_1,
    /// 8-bit quantization, variant 0.
    Q8_0,
    /// 2-bit K-quant.
    Q2K,
    /// 3-bit K-quant, small.
    Q3KS,
    /// 3-bit K-quant, medium.
    Q3KM,
    /// 3-bit K-quant, large.
    Q3KL,
    /// 4-bit K-quant, small.
    Q4KS,
    /// 4-bit K-quant, medium.
    Q4KM,
    /// 5-bit K-quant, small.
    Q5KS,
    /// 5-bit K-quant, medium.
    Q5KM,
    /// 6-bit K-quant.
    Q6K,
    /// IQ 2-bit, extra-extra-small.
    IQ2XXS,
    /// IQ 2-bit, extra-small.
    IQ2XS,
    /// IQ 3-bit, extra-extra-small.
    IQ3XXS,
    /// Unrecognised file type — stores the raw integer for diagnostics.
    Unknown(u32),
}

impl GgufFileType {
    /// Approximate bits per parameter for VRAM estimation.
    ///
    /// These are estimates -- actual sizes vary by tensor shape and quantization
    /// group size. We err on the high side so the preflight check is conservative.
    #[must_use]
    pub fn bits_per_param(self) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::F16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::Q6K => 6.5,
            Self::Q5KM | Self::Q5KS => 5.5,
            Self::Q5_0 | Self::Q5_1 => 5.5,
            Self::Q4KM | Self::Q4KS => 4.5,
            Self::Q4_0 | Self::Q4_1 => 4.5,
            Self::Q3KL | Self::Q3KM | Self::Q3KS => 3.5,
            Self::Q2K => 2.5,
            Self::IQ2XXS | Self::IQ2XS => 2.5,
            Self::IQ3XXS => 3.5,
            Self::Unknown(_) => 8.0, // conservative default
        }
    }

    /// Convert a raw GGUF `general.file_type` integer to the enum.
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            7 => Self::Q5_0,
            8 => Self::Q5_1,
            9 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3KS,
            12 => Self::Q3KM,
            13 => Self::Q3KL,
            14 => Self::Q4KS,
            15 => Self::Q4KM,
            16 => Self::Q5KS,
            17 => Self::Q5KM,
            18 => Self::Q6K,
            19 => Self::IQ2XXS,
            20 => Self::IQ2XS,
            21 => Self::IQ3XXS,
            other => Self::Unknown(other),
        }
    }
}

/// Estimate total VRAM needed to load and run a model at the given context size.
///
/// Returns estimated megabytes. Components:
/// - Model weights: `parameter_count * bits_per_param / 8`
/// - KV cache: `2 * block_count * embedding_length * context_length * 2 bytes` (fp16)
/// - Overhead: 10% buffer for CUDA kernels, scratch space, allocator fragmentation
#[must_use]
pub fn estimate_model_memory_mb(meta: &GgufModelMeta, context_length: u64) -> u64 {
    let weight_bytes = (meta.parameter_count as f64 * meta.file_type.bits_per_param()) / 8.0;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "weight_bytes is always positive and within u64 range for any realistic model"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "weight_bytes is always non-negative (parameter_count * positive bits / 8)"
    )]
    let weight_mb = (weight_bytes / (1024.0 * 1024.0)) as u64;

    // KV cache at fp16: 2 (K+V) * layers * hidden_dim * ctx_len * 2 bytes
    let kv_bytes = 2_u64
        .saturating_mul(meta.block_count)
        .saturating_mul(meta.embedding_length)
        .saturating_mul(context_length)
        .saturating_mul(2); // fp16
    let kv_mb = kv_bytes / (1024 * 1024);

    // 10% overhead for CUDA scratch, allocator fragmentation
    let subtotal = weight_mb.saturating_add(kv_mb);
    subtotal.saturating_add(subtotal / 10)
}

// -- GGUF binary parser -----------------------------------------------------------

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF metadata value types (subset we need for parsing).
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Read GGUF header metadata from a file path.
///
/// Only reads the metadata section (typically <64KB, tens of MB with a large
/// tokenizer vocabulary). Does not mmap or read tensor data. A thin wrapper
/// over [`read_gguf_meta_from`], which holds the parsing and its limits.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, is not a valid GGUF file,
/// uses an unsupported version, or contains malformed or over-limit metadata
/// (see [`GgufLimitError`]).
///
/// # Panics
///
/// Never. The reader is reachable through the `pcai_gpu_preflight_json` FFI
/// export in a `panic = "abort"` build, so every size taken from the file is
/// bounds-checked or checked-arithmetic and every failure is an `Err`.
pub fn read_gguf_meta(path: &Path) -> Result<GgufModelMeta> {
    let file = std::fs::File::open(path).with_context(|| format!("Cannot open GGUF file: {}", path.display()))?;
    read_gguf_meta_from(io::BufReader::new(file))
}

/// Read GGUF header metadata from any byte stream.
///
/// Reads at most [`MAX_METADATA_BYTES`] from `reader`. Nesting, string length,
/// entry count and every size computed from a value in the stream are limited
/// as documented on [`GgufLimitError`]; nothing is allocated up front from a
/// count or length that the stream declares.
///
/// # Errors
///
/// Returns an error if the stream is not a valid GGUF header, is truncated, or
/// exceeds a limit. Limit violations carry a [`GgufLimitError`], recoverable
/// with `err.downcast_ref::<GgufLimitError>()`.
///
/// # Panics
///
/// Never; see [`read_gguf_meta`].
///
/// # Examples
///
/// ```
/// use pcai_core_lib::preflight::gguf::read_gguf_meta_from;
///
/// let mut bytes = b"GGUF".to_vec();
/// bytes.extend_from_slice(&3_u32.to_le_bytes()); // version
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // tensor count
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // metadata entries
/// let meta = read_gguf_meta_from(bytes.as_slice())?;
/// assert_eq!(meta.embedding_length, 4096); // default when the key is absent
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn read_gguf_meta_from<R: Read>(reader: R) -> Result<GgufModelMeta> {
    let mut reader = reader.take(MAX_METADATA_BYTES);

    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).context("Failed to read GGUF magic")?;
    if &magic != GGUF_MAGIC {
        bail!("Not a GGUF file (magic: {magic:?})");
    }

    // Version
    let version = read_u32_le(&mut reader)?;
    if !(2..=3).contains(&version) {
        bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
    }

    // Tensor count (informational; not directly used in param estimation)
    let _tensor_count = read_u64_le(&mut reader)?;

    // Metadata KV count
    let kv_count = read_u64_le(&mut reader)?;
    if kv_count > MAX_KV_COUNT {
        return Err(GgufLimitError::TooManyEntries {
            count: kv_count,
            max: MAX_KV_COUNT,
        }
        .into());
    }

    // Parse metadata key-value pairs
    let mut architecture = String::new();
    let mut file_type_raw: Option<u32> = None;
    let mut context_length: Option<u64> = None;
    let mut embedding_length: Option<u64> = None;
    let mut block_count: Option<u64> = None;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let vtype_raw = read_u32_le(&mut reader)?;
        let vtype = GgufValueType::from_u32(vtype_raw);

        match key.as_str() {
            "general.architecture" => {
                architecture = read_gguf_typed_string(&mut reader, vtype)?;
            }
            "general.file_type" => {
                file_type_raw = read_gguf_typed_u32(&mut reader, vtype)?;
            }
            k if k.ends_with(".context_length") => {
                context_length = read_gguf_typed_u64(&mut reader, vtype)?;
            }
            k if k.ends_with(".embedding_length") => {
                embedding_length = read_gguf_typed_u64(&mut reader, vtype)?;
            }
            k if k.ends_with(".block_count") => {
                block_count = read_gguf_typed_u64(&mut reader, vtype)?;
            }
            _ => {
                skip_gguf_value_by_type(&mut reader, vtype, 0)?;
            }
        }
    }

    // Estimate parameter count from architecture dimensions.
    // Each transformer block has ~4 * emb^2 parameters (QKV proj + FFN).
    // Plus embedding table: vocab_size * emb (estimate vocab ~32K).
    let emb = embedding_length.unwrap_or(4096);
    let blocks = block_count.unwrap_or(32);
    let estimated_params = estimate_parameter_count(emb, blocks).ok_or(GgufLimitError::DimensionsOverflow {
        embedding_length: emb,
        block_count: blocks,
    })?;

    Ok(GgufModelMeta {
        architecture,
        file_type: GgufFileType::from_u32(file_type_raw.unwrap_or(15)),
        parameter_count: estimated_params,
        context_length: context_length.unwrap_or(2048),
        embedding_length: emb,
        block_count: blocks,
    })
}

/// `4 * emb^2 * blocks + 32_000 * emb`, or `None` if any step overflows `u64`.
///
/// `emb` and `blocks` come straight from the file, so each step is checked:
/// unchecked, this panicked under overflow checks and wrapped to a small,
/// wrong VRAM estimate in release builds.
fn estimate_parameter_count(emb: u64, blocks: u64) -> Option<u64> {
    let per_block = emb.checked_mul(emb)?.checked_mul(4)?;
    let transformer = per_block.checked_mul(blocks)?;
    let embedding_table = emb.checked_mul(32_000)?;
    transformer.checked_add(embedding_table)
}

// -- Binary reader helpers --------------------------------------------------------

fn read_u32_le(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32_le(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_i64_le(r: &mut impl Read) -> Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

/// Read a length-prefixed UTF-8 string.
///
/// The declared length is capped at [`MAX_STRING_LEN`] and compared as `u64`
/// before it is converted to `usize` (a bare `as usize` truncated on 32-bit
/// targets before the cap applied). The buffer grows with the bytes actually
/// read, so a short stream that declares a large string allocates only what it
/// delivered.
fn read_gguf_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64_le(r)?;
    if len > MAX_STRING_LEN {
        return Err(GgufLimitError::StringTooLong {
            len,
            max: MAX_STRING_LEN,
        }
        .into());
    }
    let mut buf = Vec::new();
    let read = r.by_ref().take(len).read_to_end(&mut buf)?;
    if read as u64 != len {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "GGUF string is truncated").into());
    }
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn read_gguf_typed_string(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<String> {
    match vtype {
        Some(GgufValueType::String) => read_gguf_string(r),
        _ => {
            skip_gguf_value_by_type(r, vtype, 0)?;
            Ok(String::new())
        }
    }
}

/// Read a non-negative integer of any width, or `None` if the value has another type.
///
/// A mistyped value is skipped and reported as `None` so the caller falls back
/// to its default; returning `Some(0)` made the parameter estimate collapse to
/// near zero and preflight report "go". A negative signed value is an error:
/// `as u64` turned it into an enormous dimension.
fn read_gguf_typed_u64(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<Option<u64>> {
    let value = match vtype {
        Some(GgufValueType::Uint64) => read_u64_le(r)?,
        Some(GgufValueType::Int64) => non_negative(read_i64_le(r)?)?,
        Some(GgufValueType::Uint32) => u64::from(read_u32_le(r)?),
        Some(GgufValueType::Int32) => non_negative(i64::from(read_i32_le(r)?))?,
        Some(GgufValueType::Uint16) => u64::from(read_u16_le(r)?),
        _ => {
            skip_gguf_value_by_type(r, vtype, 0)?;
            return Ok(None);
        }
    };
    Ok(Some(value))
}

/// As [`read_gguf_typed_u64`], narrowed to the `u32` of `general.file_type`.
fn read_gguf_typed_u32(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<Option<u32>> {
    Ok(read_gguf_typed_u64(r, vtype)?.map(|v| u32::try_from(v).unwrap_or(u32::MAX)))
}

fn read_u16_le(r: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn non_negative(v: i64) -> Result<u64> {
    u64::try_from(v).map_err(|_| GgufLimitError::NegativeValue { value: v }.into())
}

/// Size in bytes of a fixed-width value type, or `None` for strings and arrays.
fn fixed_size(vtype: GgufValueType) -> Option<u64> {
    match vtype {
        GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool => Some(1),
        GgufValueType::Uint16 | GgufValueType::Int16 => Some(2),
        GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32 => Some(4),
        GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64 => Some(8),
        GgufValueType::String | GgufValueType::Array => None,
    }
}

/// Discard exactly `n` bytes, failing if the stream ends first.
fn skip_bytes(r: &mut impl Read, n: u64) -> Result<()> {
    let skipped = io::copy(&mut r.by_ref().take(n), &mut io::sink())?;
    if skipped != n {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "GGUF value is truncated").into());
    }
    Ok(())
}

/// Skip one value. `depth` is the number of arrays already entered.
///
/// Recursion happens only through nested arrays and stops at
/// [`MAX_ARRAY_NESTING`]; before the limit, a crafted file with deeply nested
/// arrays overflowed the stack, which aborts the process whatever the panic
/// strategy. Arrays of fixed-width elements are skipped in one checked
/// byte-count rather than element by element.
fn skip_gguf_value_by_type(r: &mut impl Read, vtype: Option<GgufValueType>, depth: usize) -> Result<()> {
    match vtype {
        Some(GgufValueType::String) => {
            let _ = read_gguf_string(r)?;
        }
        Some(GgufValueType::Array) => {
            if depth >= MAX_ARRAY_NESTING {
                return Err(GgufLimitError::NestingTooDeep { max: MAX_ARRAY_NESTING }.into());
            }
            let elem_type_raw = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            if count == 0 {
                return Ok(());
            }
            let Some(elem_type) = GgufValueType::from_u32(elem_type_raw) else {
                bail!("Unknown GGUF value type");
            };
            if let Some(elem_size) = fixed_size(elem_type) {
                let total = count
                    .checked_mul(elem_size)
                    .ok_or(GgufLimitError::ArraySizeOverflow { count, elem_size })?;
                skip_bytes(r, total)?;
            } else {
                // Every element consumes at least one byte, so this loop is
                // bounded by the stream length (and `MAX_METADATA_BYTES`).
                for _ in 0..count {
                    skip_gguf_value_by_type(r, Some(elem_type), depth + 1)?;
                }
            }
        }
        Some(fixed) => {
            if let Some(size) = fixed_size(fixed) {
                skip_bytes(r, size)?;
            }
        }
        None => bail!("Unknown GGUF value type"),
    }
    Ok(())
}

// -- Tests ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_memory_q4_k_m_7b() {
        let meta = GgufModelMeta {
            architecture: "llama".to_owned(),
            file_type: GgufFileType::Q4KM,
            parameter_count: 7_000_000_000,
            context_length: 8192,
            embedding_length: 4096,
            block_count: 32,
        };
        let est = estimate_model_memory_mb(&meta, 8192);
        // 7B Q4_K_M ~ 4.07 GB model + KV cache
        // Model weights: 7B * 4.5 bits/param / 8 = ~3937 MB
        // KV cache: 2 * 32 * 4096 * 8192 * 2 bytes / 1MB = ~4096 MB at fp16
        // Total should be roughly 4000-10000 MB depending on ctx
        assert!(est > 3500, "7B Q4_K_M should need >3500 MB, got {est}");
        assert!(est < 10000, "7B Q4_K_M at 8K ctx should need <10000 MB, got {est}");
    }

    #[test]
    fn estimate_memory_q8_0_3b() {
        let meta = GgufModelMeta {
            architecture: "llama".to_owned(),
            file_type: GgufFileType::Q8_0,
            parameter_count: 3_000_000_000,
            context_length: 4096,
            embedding_length: 3200,
            block_count: 26,
        };
        let est = estimate_model_memory_mb(&meta, 4096);
        // 3B Q8_0 ~ 3 GB model + smaller KV cache
        assert!(est > 2500, "3B Q8_0 should need >2500 MB, got {est}");
        assert!(est < 7000, "3B Q8_0 at 4K ctx should need <7000 MB, got {est}");
    }

    #[test]
    fn bits_per_param_known_types() {
        assert!(
            (GgufFileType::Q4_0.bits_per_param() - 4.5).abs() < 0.1,
            "Q4_0 should be ~4.5 bits"
        );
        assert!(
            (GgufFileType::Q4KM.bits_per_param() - 4.5).abs() < 0.2,
            "Q4KM should be ~4.5 bits"
        );
        assert!(
            (GgufFileType::Q8_0.bits_per_param() - 8.5).abs() < 0.1,
            "Q8_0 should be ~8.5 bits"
        );
        assert!(
            (GgufFileType::F16.bits_per_param() - 16.0).abs() < 0.1,
            "F16 should be ~16.0 bits"
        );
    }

    #[test]
    fn unknown_file_type_defaults_conservatively() {
        assert!(
            GgufFileType::Unknown(999).bits_per_param() >= 8.0,
            "Unknown file types should default to >= 8.0 bits (conservative)"
        );
    }

    #[test]
    fn file_type_from_u32_roundtrip() {
        // Known types
        assert!(matches!(GgufFileType::from_u32(0), GgufFileType::F32));
        assert!(matches!(GgufFileType::from_u32(1), GgufFileType::F16));
        assert!(matches!(GgufFileType::from_u32(15), GgufFileType::Q4KM));
        assert!(matches!(GgufFileType::from_u32(18), GgufFileType::Q6K));
        // Unknown maps to Unknown variant
        assert!(matches!(GgufFileType::from_u32(255), GgufFileType::Unknown(255)));
    }

    #[test]
    fn value_type_from_u32_known_and_unknown() {
        assert!(GgufValueType::from_u32(0).is_some());
        assert!(GgufValueType::from_u32(8).is_some()); // String
        assert!(GgufValueType::from_u32(12).is_some()); // Float64
        assert!(GgufValueType::from_u32(13).is_none()); // Out of range
        assert!(GgufValueType::from_u32(999).is_none());
    }

    #[test]
    fn estimate_memory_f32_small_model() {
        // Edge case: small unquantized model
        let meta = GgufModelMeta {
            architecture: "gpt2".to_owned(),
            file_type: GgufFileType::F32,
            parameter_count: 125_000_000,
            context_length: 1024,
            embedding_length: 768,
            block_count: 12,
        };
        let est = estimate_model_memory_mb(&meta, 1024);
        // 125M F32 = ~476 MB weights
        // KV cache: 2 * 12 * 768 * 1024 * 2 = ~36 MB
        // Total ~512 MB + 10% = ~563 MB
        assert!(est > 400, "125M F32 should need >400 MB, got {est}");
        assert!(est < 1000, "125M F32 should need <1000 MB, got {est}");
    }

    #[test]
    fn estimate_memory_zero_context() {
        // Zero context should give zero KV cache
        let meta = GgufModelMeta {
            architecture: "llama".to_owned(),
            file_type: GgufFileType::Q4KM,
            parameter_count: 7_000_000_000,
            context_length: 8192,
            embedding_length: 4096,
            block_count: 32,
        };
        let est = estimate_model_memory_mb(&meta, 0);
        // Weights only, no KV cache
        // 7B * 4.5 / 8 = ~3937 MB + 10% = ~4330 MB
        assert!(est > 3500, "Weights-only should still need >3500 MB, got {est}");
        assert!(est < 5000, "Zero ctx should have no KV overhead, got {est}");
    }

    #[test]
    fn all_file_types_have_positive_bits() {
        let types = [
            GgufFileType::F32,
            GgufFileType::F16,
            GgufFileType::Q4_0,
            GgufFileType::Q4_1,
            GgufFileType::Q5_0,
            GgufFileType::Q5_1,
            GgufFileType::Q8_0,
            GgufFileType::Q2K,
            GgufFileType::Q3KS,
            GgufFileType::Q3KM,
            GgufFileType::Q3KL,
            GgufFileType::Q4KS,
            GgufFileType::Q4KM,
            GgufFileType::Q5KS,
            GgufFileType::Q5KM,
            GgufFileType::Q6K,
            GgufFileType::IQ2XXS,
            GgufFileType::IQ2XS,
            GgufFileType::IQ3XXS,
            GgufFileType::Unknown(42),
        ];
        for ft in types {
            assert!(ft.bits_per_param() > 0.0, "{ft:?} should have positive bits_per_param");
        }
    }
}
