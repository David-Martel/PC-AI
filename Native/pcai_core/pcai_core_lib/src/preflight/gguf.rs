//! GGUF header parser for model memory estimation.
//!
//! Reads only the header of a GGUF file to extract the fields needed for VRAM
//! estimation. Never reads tensor data.
//!
//! Two entry points share one bounded reader:
//!
//! - [`read_gguf_meta_from`] stops after the key-value metadata and estimates
//!   the weight size from the architecture dimensions and `general.file_type`.
//! - [`read_gguf_layout_from`] also reads the tensor-info section and sums the
//!   exact byte size of every tensor from its ggml type and shape, the way
//!   upstream llama.cpp does (`ggml/src/gguf.cpp`). A tensor whose type this
//!   module does not know is an error, never a size of zero.

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

/// Maximum number of tensors a header may declare.
///
/// Real models declare a few hundred to a few thousand. The tensor-info reader
/// allocates nothing from the declared count; this bounds only the loop.
pub const MAX_TENSOR_COUNT: u64 = 1 << 20;

/// Maximum number of dimensions of one tensor (`GGML_MAX_DIMS` in `ggml.h`).
pub const MAX_TENSOR_DIMS: usize = 4;

/// Maximum length in bytes of a single GGUF string.
pub const MAX_STRING_LEN: u64 = 1_000_000;

/// Maximum number of metadata key-value pairs a header may declare.
///
/// Real files declare a few dozen to a few thousand.
pub const MAX_KV_COUNT: u64 = 1 << 20;

/// Maximum number of bytes read from one stream while parsing metadata.
///
/// Large tokenizer vocabularies make the metadata section tens of megabytes;
/// this budget bounds the work done on a hostile or endless stream. A stream
/// that still has data when the budget is spent fails with
/// [`GgufLimitError::ByteBudgetExceeded`]; a stream that simply ends early is a
/// truncated file and fails with an ordinary `UnexpectedEof` I/O error.
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
    /// The header declares more than [`MAX_TENSOR_COUNT`] tensors.
    #[error("GGUF tensor count {count} exceeds the limit of {max}")]
    TooManyTensors {
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
    /// The stream held more than [`MAX_METADATA_BYTES`] of metadata.
    ///
    /// Distinct from a truncated file: here the budget ran out while the stream
    /// still had data, not the other way round.
    #[error("GGUF metadata exceeds the read budget of {max} bytes")]
    ByteBudgetExceeded {
        /// The budget that was exceeded.
        max: u64,
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
    /// NVFP4: 4-bit E2M1 values with E4M3 scales per 16 values
    /// (`LLAMA_FTYPE_MOSTLY_NVFP4` = 39).
    Nvfp4,
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
            // 36 bytes per 64 values: 64 * 4 bits of values plus four one-byte scales.
            Self::Nvfp4 => 4.5,
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
            39 => Self::Nvfp4,
            other => Self::Unknown(other),
        }
    }
}

// -- Tensor types and sizes -------------------------------------------------------

/// Elements per super-block of the K-quants and most IQ types (`QK_K` in `ggml-common.h`).
const QK_K: u64 = 256;

/// A ggml tensor element type, as stored in a GGUF tensor-info record.
///
/// The ids and the `(block size, bytes per block)` pairs mirror upstream
/// llama.cpp at commit `ad50351f5092436f312d98d25689207df25f4802` (b12635):
/// `GGMLQuantizationType` and `GGML_QUANT_SIZES` in `gguf-py/gguf/constants.py`,
/// and `ggml/include/ggml.h`. An id that is not listed here is not a variant;
/// [`GgmlType::from_id`] returns `None` for it and the reader reports an error
/// rather than guess a size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum GgmlType {
    /// 32-bit float (id 0).
    F32,
    /// 16-bit float (id 1).
    F16,
    /// 4-bit, variant 0 (id 2).
    Q4_0,
    /// 4-bit, variant 1 (id 3).
    Q4_1,
    /// 5-bit, variant 0 (id 6).
    Q5_0,
    /// 5-bit, variant 1 (id 7).
    Q5_1,
    /// 8-bit, variant 0 (id 8).
    Q8_0,
    /// 8-bit, variant 1 (id 9).
    Q8_1,
    /// 2-bit K-quant (id 10).
    Q2K,
    /// 3-bit K-quant (id 11).
    Q3K,
    /// 4-bit K-quant (id 12).
    Q4K,
    /// 5-bit K-quant (id 13).
    Q5K,
    /// 6-bit K-quant (id 14).
    Q6K,
    /// 8-bit K-quant (id 15).
    Q8K,
    /// IQ 2-bit, extra-extra-small (id 16).
    IQ2XXS,
    /// IQ 2-bit, extra-small (id 17).
    IQ2XS,
    /// IQ 3-bit, extra-extra-small (id 18).
    IQ3XXS,
    /// IQ 1-bit, small (id 19).
    IQ1S,
    /// IQ 4-bit non-linear (id 20).
    IQ4NL,
    /// IQ 3-bit, small (id 21).
    IQ3S,
    /// IQ 2-bit, small (id 22).
    IQ2S,
    /// IQ 4-bit, extra-small (id 23).
    IQ4XS,
    /// 8-bit integer (id 24).
    I8,
    /// 16-bit integer (id 25).
    I16,
    /// 32-bit integer (id 26).
    I32,
    /// 64-bit integer (id 27).
    I64,
    /// 64-bit float (id 28).
    F64,
    /// IQ 1-bit, medium (id 29).
    IQ1M,
    /// bfloat16 (id 30).
    Bf16,
    /// Ternary, 1.6875 bits (id 34).
    TQ1_0,
    /// Ternary, 2.0625 bits (id 35).
    TQ2_0,
    /// MXFP4: 32 E2M1 values with one E8M0 scale (id 39).
    Mxfp4,
    /// NVFP4: 64 E2M1 values with four E4M3 scales, one per 16 values (id 40).
    Nvfp4,
    /// 1-bit, 128 values per block (id 41).
    Q1_0,
    /// 2-bit, 64 values per block (id 42).
    Q2_0,
    /// 8-bit float, E4M3 (id 43).
    F8E4M3,
}

impl GgmlType {
    /// The ggml type for a raw tensor-info type id, or `None` if this reader does not know it.
    ///
    /// # Examples
    ///
    /// ```
    /// use pcai_core_lib::preflight::gguf::GgmlType;
    ///
    /// assert_eq!(GgmlType::from_id(40), Some(GgmlType::Nvfp4));
    /// assert_eq!(GgmlType::from_id(9_999), None);
    /// ```
    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::Bf16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            43 => Self::F8E4M3,
            _ => return None,
        })
    }

    /// `(values per block, bytes per block)`, written as upstream's `GGML_QUANT_SIZES` writes them.
    const fn layout(self) -> (u64, u64) {
        match self {
            Self::F32 | Self::I32 => (1, 4),
            Self::F16 | Self::Bf16 | Self::I16 => (1, 2),
            Self::Q4_0 | Self::IQ4NL => (32, 2 + 16),
            Self::Q4_1 => (32, 2 + 2 + 16),
            Self::Q5_0 => (32, 2 + 4 + 16),
            Self::Q5_1 => (32, 2 + 2 + 4 + 16),
            Self::Q8_0 => (32, 2 + 32),
            Self::Q8_1 => (32, 2 + 2 + 32),
            Self::Q2K => (QK_K, 2 + 2 + QK_K / 16 + QK_K / 4),
            Self::Q3K => (QK_K, 2 + QK_K / 4 + QK_K / 8 + 12),
            Self::Q4K => (QK_K, 2 + 2 + QK_K / 2 + 12),
            Self::Q5K => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 8 + 12),
            Self::Q6K => (QK_K, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16),
            Self::Q8K => (QK_K, 4 + QK_K + QK_K / 8),
            Self::IQ2XXS => (QK_K, 2 + QK_K / 4),
            Self::IQ2XS => (QK_K, 2 + QK_K / 4 + QK_K / 32),
            Self::IQ3XXS => (QK_K, 2 + QK_K / 4 + QK_K / 8),
            Self::IQ1S => (QK_K, 2 + QK_K / 8 + QK_K / 16),
            Self::IQ3S => (QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4),
            Self::IQ2S => (QK_K, 2 + QK_K / 4 + QK_K / 16),
            Self::IQ4XS => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 64),
            Self::I8 | Self::F8E4M3 => (1, 1),
            Self::I64 | Self::F64 => (1, 8),
            Self::IQ1M => (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32),
            Self::TQ1_0 => (QK_K, 2 + 4 * 13),
            Self::TQ2_0 => (QK_K, 2 + 64),
            Self::Mxfp4 => (32, 1 + 16),
            // `block_nvfp4` in ggml-common.h: QK_NVFP4 = 64 values, a 4-byte
            // `d[QK_NVFP4 / QK_NVFP4_SUB]` scale array and 32 bytes of packed nibbles.
            Self::Nvfp4 => (64, 4 + 32),
            Self::Q1_0 => (128, 2 + 16),
            Self::Q2_0 => (64, 2 + 16),
        }
    }

    /// Number of values stored in one block (1 for unquantized types).
    #[must_use]
    pub const fn block_size(self) -> u64 {
        self.layout().0
    }

    /// Number of bytes one block occupies.
    #[must_use]
    pub const fn type_size(self) -> u64 {
        self.layout().1
    }

    /// Size in bytes of a contiguous tensor of this type and shape.
    ///
    /// `dims` lists the extents from the innermost (row) dimension outwards, as
    /// a GGUF tensor-info record does; missing outer dimensions count as 1. The
    /// rule is upstream's (`gguf.cpp`, tensor-info validation): the row length
    /// `dims[0]` must be a multiple of the block size (it is the row, not the
    /// element total, that must divide), and the size is
    /// `type_size * (dims[0] / block_size) * dims[1] * dims[2] * dims[3]`.
    /// Every step is checked.
    ///
    /// # Errors
    ///
    /// Returns [`GgufTensorSizeError`] if more than [`MAX_TENSOR_DIMS`]
    /// dimensions are given, if the row length is not a multiple of the block
    /// size, or if the element count or byte size overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use pcai_core_lib::preflight::gguf::GgmlType;
    ///
    /// // 64 values per 36-byte block: a 4096 x 4096 matrix is 64 blocks per row.
    /// assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[4096, 4096])?, 4096 * 64 * 36);
    /// // A row of 96 is not a whole number of 64-value blocks, even though 96 x 2 = 192 is.
    /// assert!(GgmlType::Nvfp4.tensor_byte_size(&[96, 2]).is_err());
    /// # Ok::<(), pcai_core_lib::preflight::gguf::GgufTensorSizeError>(())
    /// ```
    pub fn tensor_byte_size(self, dims: &[u64]) -> Result<u64, GgufTensorSizeError> {
        if dims.len() > MAX_TENSOR_DIMS {
            return Err(GgufTensorSizeError::TooManyDims {
                n_dims: u64::try_from(dims.len()).unwrap_or(u64::MAX),
            });
        }
        let mut extents = [1_u64; MAX_TENSOR_DIMS];
        for (slot, &dim) in extents.iter_mut().zip(dims) {
            // Upstream reads each extent as `int64_t` and rejects a negative one.
            if i64::try_from(dim).is_err() {
                return Err(GgufTensorSizeError::Overflow);
            }
            *slot = dim;
        }
        // Upstream also requires the element count to fit in `int64_t`.
        let elements = extents
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
            .filter(|&n| i64::try_from(n).is_ok())
            .ok_or(GgufTensorSizeError::Overflow)?;
        let (block, bytes) = self.layout();
        let row_len = extents[0];
        if row_len % block != 0 {
            return Err(GgufTensorSizeError::RowNotBlockMultiple {
                ggml_type: self,
                row_len,
                block_size: block,
            });
        }
        // `elements` is the product of all extents; an empty tensor has no bytes.
        // Otherwise bytes = type_size * (elements / block_size), exact because the row divides.
        (elements / block)
            .checked_mul(bytes)
            .ok_or(GgufTensorSizeError::Overflow)
    }
}

/// Why the byte size of one tensor could not be computed.
///
/// Mirrors the checks upstream's GGUF loader makes before it accepts a
/// tensor-info record.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GgufTensorSizeError {
    /// More than [`MAX_TENSOR_DIMS`] dimensions were declared.
    #[error("tensor declares {n_dims} dimensions, more than the maximum of {MAX_TENSOR_DIMS}")]
    TooManyDims {
        /// The declared dimension count.
        n_dims: u64,
    },
    /// The row length is not a whole number of quantization blocks.
    #[error("{ggml_type:?} row of {row_len} values is not a multiple of its block size {block_size}")]
    RowNotBlockMultiple {
        /// The tensor's type.
        ggml_type: GgmlType,
        /// The innermost extent.
        row_len: u64,
        /// Values per block of `ggml_type`.
        block_size: u64,
    },
    /// An extent, the element count or the byte size does not fit the integer range upstream uses.
    #[error("tensor shape overflows the representable size")]
    Overflow,
}

/// A tensor-info record this reader cannot turn into a size.
///
/// Returned (wrapped in [`anyhow::Error`]) by [`read_gguf_layout_from`];
/// recover it with `err.downcast_ref::<GgufTensorError>()`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum GgufTensorError {
    /// The tensor has a ggml type id this reader does not know.
    ///
    /// A model that uses a newer type than this reader has been taught is not
    /// sized as zero or guessed at; the readiness check fails with this error.
    #[error("GGUF tensor {index} has unknown ggml type id {id}; its size cannot be estimated")]
    UnknownType {
        /// Zero-based position of the tensor in the header.
        index: u64,
        /// The raw type id.
        id: u32,
    },
    /// The tensor's shape is invalid for its type.
    #[error("GGUF tensor {index}: {source}")]
    InvalidShape {
        /// Zero-based position of the tensor in the header.
        index: u64,
        /// What was wrong with the shape.
        #[source]
        source: GgufTensorSizeError,
    },
    /// The summed tensor sizes do not fit in `u64`.
    #[error("GGUF tensor sizes overflow u64 after {index} tensors")]
    TotalOverflow {
        /// Number of tensors summed before the overflow.
        index: u64,
    },
}

/// Header of a GGUF file with the exact weight size read from its tensor infos.
#[derive(Debug, Clone, Serialize)]
pub struct GgufLayout {
    /// The key-value metadata, as [`read_gguf_meta_from`] returns it.
    pub meta: GgufModelMeta,
    /// Number of tensors the header declares.
    pub tensor_count: u64,
    /// Sum of the byte sizes of all tensors, without per-tensor alignment padding.
    pub tensor_bytes: u64,
    /// Number of tensors of type [`GgmlType::Nvfp4`].
    pub nvfp4_tensors: u64,
    /// Bytes read from the stream: everything up to the end of the tensor-info section.
    pub header_bytes: u64,
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
    add_kv_and_overhead_mb(weight_mb, meta, context_length)
}

/// Estimate total VRAM in megabytes from the exact tensor sizes of a [`GgufLayout`].
///
/// Same components as [`estimate_model_memory_mb`], except the weights are the
/// sum of the tensor sizes read from the header rather than a figure derived
/// from the architecture dimensions and `general.file_type`. A header that
/// declares no tensors has nothing to sum, so this falls back to that
/// dimension-based estimate.
///
/// # Examples
///
/// ```
/// use pcai_core_lib::preflight::gguf::{estimate_layout_memory_mb, read_gguf_layout_from};
///
/// let mut bytes = b"GGUF".to_vec();
/// bytes.extend_from_slice(&3_u32.to_le_bytes()); // version
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // tensor count
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // metadata entries
/// let layout = read_gguf_layout_from(bytes.as_slice())?;
/// assert!(estimate_layout_memory_mb(&layout, 0) > 0); // dimension-based fallback
/// # Ok::<(), anyhow::Error>(())
/// ```
#[must_use]
pub fn estimate_layout_memory_mb(layout: &GgufLayout, context_length: u64) -> u64 {
    if layout.tensor_count == 0 {
        return estimate_model_memory_mb(&layout.meta, context_length);
    }
    add_kv_and_overhead_mb(layout.tensor_bytes / (1024 * 1024), &layout.meta, context_length)
}

/// Add the fp16 KV cache and the 10% overhead to a weight size in megabytes.
fn add_kv_and_overhead_mb(weight_mb: u64, meta: &GgufModelMeta, context_length: u64) -> u64 {
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
/// exceeds a limit.
///
/// - Every limit violation, including the read budget, is a
///   [`GgufLimitError`] at the top of the chain, recoverable with
///   `err.downcast_ref::<GgufLimitError>()`.
/// - A stream that ends before the header is complete is *not* a limit
///   violation: it surfaces as a [`std::io::Error`] of kind
///   [`UnexpectedEof`](std::io::ErrorKind::UnexpectedEof) somewhere in
///   `err.chain()`, and `downcast_ref::<GgufLimitError>()` returns `None`.
/// - Bad magic, an unsupported version, invalid UTF-8 and unknown value types
///   are plain errors with a message and no typed cause.
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
    with_budget(reader, |reader| parse_header(reader).map(|(meta, _tensor_count)| meta))
}

/// Read GGUF header metadata and the exact tensor sizes from a file path.
///
/// A thin wrapper over [`read_gguf_layout_from`]. Never reads tensor data.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, or for any reason
/// [`read_gguf_layout_from`] fails.
///
/// # Panics
///
/// Never; see [`read_gguf_meta`].
pub fn read_gguf_layout(path: &Path) -> Result<GgufLayout> {
    let file = std::fs::File::open(path).with_context(|| format!("Cannot open GGUF file: {}", path.display()))?;
    read_gguf_layout_from(io::BufReader::new(file))
}

/// Read GGUF header metadata and sum the byte size of every tensor.
///
/// Parses the key-value metadata as [`read_gguf_meta_from`] does, then the
/// tensor-info section: for each tensor a name, up to [`MAX_TENSOR_DIMS`]
/// extents, a ggml type id and a data offset. The size of each tensor comes
/// from [`GgmlType::tensor_byte_size`]. Tensor data is never read, and the
/// whole read, tensor infos included, stays within [`MAX_METADATA_BYTES`].
///
/// # Errors
///
/// Everything [`read_gguf_meta_from`] rejects, plus:
///
/// - more than [`MAX_TENSOR_COUNT`] tensors is a [`GgufLimitError`];
/// - a tensor of an unknown ggml type, an invalid shape (a row that is not a
///   whole number of blocks, more than [`MAX_TENSOR_DIMS`] dimensions, an
///   overflowing size) or a total that overflows `u64` is a [`GgufTensorError`],
///   recoverable with `err.downcast_ref::<GgufTensorError>()`.
///
/// # Panics
///
/// Never; see [`read_gguf_meta`].
///
/// # Examples
///
/// ```
/// use pcai_core_lib::preflight::gguf::read_gguf_layout_from;
///
/// let mut bytes = b"GGUF".to_vec();
/// bytes.extend_from_slice(&3_u32.to_le_bytes()); // version
/// bytes.extend_from_slice(&1_u64.to_le_bytes()); // tensor count
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // metadata entries
/// bytes.extend_from_slice(&1_u64.to_le_bytes()); // name length
/// bytes.push(b'w'); // name
/// bytes.extend_from_slice(&2_u32.to_le_bytes()); // dimensions
/// bytes.extend_from_slice(&64_u64.to_le_bytes()); // row length
/// bytes.extend_from_slice(&3_u64.to_le_bytes()); // rows
/// bytes.extend_from_slice(&40_u32.to_le_bytes()); // NVFP4
/// bytes.extend_from_slice(&0_u64.to_le_bytes()); // data offset
/// let layout = read_gguf_layout_from(bytes.as_slice())?;
/// assert_eq!(layout.tensor_bytes, 3 * 36); // three 64-value blocks of 36 bytes
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn read_gguf_layout_from<R: Read>(reader: R) -> Result<GgufLayout> {
    with_budget(reader, |reader| {
        let (meta, tensor_count) = parse_header(&mut *reader)?;
        if tensor_count > MAX_TENSOR_COUNT {
            return Err(GgufLimitError::TooManyTensors {
                count: tensor_count,
                max: MAX_TENSOR_COUNT,
            }
            .into());
        }
        let (tensor_bytes, nvfp4_tensors) = read_tensor_bytes(&mut *reader, tensor_count)?;
        Ok(GgufLayout {
            meta,
            tensor_count,
            tensor_bytes,
            nvfp4_tensors,
            header_bytes: reader.consumed(),
        })
    })
}

/// Run `parse` over `reader` limited to [`MAX_METADATA_BYTES`], and report a spent budget as a typed error.
fn with_budget<R: Read, T>(reader: R, parse: impl FnOnce(&mut BudgetReader<R>) -> Result<T>) -> Result<T> {
    let mut reader = BudgetReader::new(reader, MAX_METADATA_BYTES);
    parse(&mut reader).map_err(|err| {
        if reader.exceeded {
            // The budget error crossed `?` boundaries as an `io::Error`; report
            // the typed cause instead so callers can match on it.
            GgufLimitError::ByteBudgetExceeded {
                max: MAX_METADATA_BYTES,
            }
            .into()
        } else {
            err
        }
    })
}

/// A reader that stops after a fixed number of bytes and says why.
///
/// `Read::take` reports an exhausted limit as ordinary end of file, so a file
/// truncated by its writer and a stream cut off by the budget look the same.
/// This reader probes the inner stream once the budget is spent: a stream that
/// is also at its end returns EOF (a genuinely truncated or exactly
/// budget-sized file), while one that still has data sets `exceeded` and fails.
struct BudgetReader<R> {
    inner: R,
    budget: u64,
    remaining: u64,
    exceeded: bool,
}

impl<R: Read> BudgetReader<R> {
    fn new(inner: R, budget: u64) -> Self {
        Self {
            inner,
            budget,
            remaining: budget,
            exceeded: false,
        }
    }

    /// Bytes handed out so far.
    fn consumed(&self) -> u64 {
        self.budget - self.remaining
    }
}

impl<R: Read> Read for BudgetReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut probe = [0u8; 1];
            return if self.inner.read(&mut probe)? == 0 {
                Ok(0)
            } else {
                self.exceeded = true;
                Err(io::Error::other("GGUF metadata read budget exceeded"))
            };
        }
        let allowed = usize::try_from(self.remaining).map_or(buf.len(), |r| r.min(buf.len()));
        let n = self.inner.read(&mut buf[..allowed])?;
        self.remaining -= n as u64;
        Ok(n)
    }
}

/// Parse the magic, counts and key-value metadata from an already budgeted
/// stream. Returns the metadata and the declared tensor count; the reader is
/// left at the start of the tensor-info section.
fn parse_header(mut reader: impl Read) -> Result<(GgufModelMeta, u64)> {
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

    // Tensor count: not used by the dimension-based estimate, read by `read_gguf_layout_from`.
    let tensor_count = read_u64_le(&mut reader)?;

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

    let meta = GgufModelMeta {
        architecture,
        file_type: GgufFileType::from_u32(file_type_raw.unwrap_or(15)),
        parameter_count: estimated_params,
        context_length: context_length.unwrap_or(2048),
        embedding_length: emb,
        block_count: blocks,
    };
    Ok((meta, tensor_count))
}

/// Read `tensor_count` tensor-info records and return the sum of their sizes
/// and the number of NVFP4 tensors among them.
///
/// Nothing is allocated from the declared count or from a name length: names
/// are skipped, and at most [`MAX_TENSOR_DIMS`] extents are held at a time.
fn read_tensor_bytes(r: &mut impl Read, tensor_count: u64) -> Result<(u64, u64)> {
    let mut total = 0_u64;
    let mut nvfp4 = 0_u64;
    for index in 0..tensor_count {
        skip_gguf_string(r)?;
        let n_dims = read_u32_le(r)?;
        // Checked before reading any extent, so a hostile count reads nothing more.
        let n_dims = usize::try_from(n_dims)
            .ok()
            .filter(|&n| n <= MAX_TENSOR_DIMS)
            .ok_or_else(|| GgufTensorError::InvalidShape {
                index,
                source: GgufTensorSizeError::TooManyDims {
                    n_dims: u64::from(n_dims),
                },
            })?;
        let mut dims = [0_u64; MAX_TENSOR_DIMS];
        for dim in dims.iter_mut().take(n_dims) {
            *dim = read_u64_le(r)?;
        }
        let type_id = read_u32_le(r)?;
        let _data_offset = read_u64_le(r)?;

        let ggml_type = GgmlType::from_id(type_id).ok_or(GgufTensorError::UnknownType { index, id: type_id })?;
        let size = ggml_type
            .tensor_byte_size(&dims[..n_dims])
            .map_err(|source| GgufTensorError::InvalidShape { index, source })?;
        total = total
            .checked_add(size)
            .ok_or(GgufTensorError::TotalOverflow { index })?;
        if ggml_type == GgmlType::Nvfp4 {
            // At most `index + 1`, which is bounded by MAX_TENSOR_COUNT.
            nvfp4 += 1;
        }
    }
    Ok((total, nvfp4))
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

/// Skip a length-prefixed string without holding it, applying the [`MAX_STRING_LEN`] cap.
fn skip_gguf_string(r: &mut impl Read) -> Result<()> {
    let len = read_u64_le(r)?;
    if len > MAX_STRING_LEN {
        return Err(GgufLimitError::StringTooLong {
            len,
            max: MAX_STRING_LEN,
        }
        .into());
    }
    skip_bytes(r, len)
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
            GgufFileType::Nvfp4,
            GgufFileType::Unknown(42),
        ];
        for ft in types {
            assert!(ft.bits_per_param() > 0.0, "{ft:?} should have positive bits_per_param");
        }
    }

    // -- NVFP4 and the ggml type table --------------------------------------------
    //
    // The expected numbers below are written out as literals, not computed from
    // the table's own expressions, so a wrong block size or block byte count in
    // `GgmlType::layout` fails here. Sources, all at llama.cpp ad50351f (b12635):
    // `GGML_QUANT_SIZES` in gguf-py/gguf/constants.py and `block_nvfp4` in
    // ggml/src/ggml-common.h.

    /// Why: the NVFP4 constants are the point of this change; id 40 must not drift to MXFP4's 39.
    #[test]
    fn nvfp4_tensor_type_has_the_upstream_id_block_size_and_block_bytes() {
        assert_eq!(GgmlType::from_id(40), Some(GgmlType::Nvfp4));
        assert_eq!(GgmlType::Nvfp4.block_size(), 64, "QK_NVFP4");
        assert_eq!(GgmlType::Nvfp4.type_size(), 36, "4 scale bytes + 32 packed bytes");
    }

    /// Why: tensor id 39 is MXFP4 and file_type 39 is NVFP4; the two id spaces must stay separate.
    #[test]
    fn tensor_type_and_file_type_id_spaces_are_distinct() {
        assert_eq!(GgmlType::from_id(39), Some(GgmlType::Mxfp4));
        assert_eq!(GgmlType::Mxfp4.block_size(), 32);
        assert_eq!(GgmlType::Mxfp4.type_size(), 17);
        assert!(matches!(GgufFileType::from_u32(39), GgufFileType::Nvfp4));
        // file_type 40 is Q1_0 upstream; this reader does not know it, and it must not alias NVFP4.
        assert!(matches!(GgufFileType::from_u32(40), GgufFileType::Unknown(40)));
    }

    /// Why: `bits_per_param` drives the estimate when a header lists no tensors, so it must agree
    /// with the block layout that sizes real tensors.
    #[test]
    fn nvfp4_file_type_bits_match_the_block_layout() {
        let from_layout = (GgmlType::Nvfp4.type_size() * 8) as f64 / GgmlType::Nvfp4.block_size() as f64;
        assert!((GgufFileType::Nvfp4.bits_per_param() - from_layout).abs() < f64::EPSILON);
        assert!((GgufFileType::Nvfp4.bits_per_param() - 4.5).abs() < f64::EPSILON);
    }

    /// Why: the reader must size the 1-d tensors of a "mostly NVFP4" file, which are not NVFP4,
    /// so the whole upstream table is pinned, not just the new row.
    #[test]
    fn every_known_type_matches_the_upstream_quant_sizes_table() {
        // (id, block size, bytes per block), evaluated by hand from GGML_QUANT_SIZES.
        let upstream: [(u32, u64, u64); 36] = [
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (16, 256, 66),
            (17, 256, 74),
            (18, 256, 98),
            (19, 256, 50),
            (20, 32, 18),
            (21, 256, 110),
            (22, 256, 82),
            (23, 256, 136),
            (24, 1, 1),
            (25, 1, 2),
            (26, 1, 4),
            (27, 1, 8),
            (28, 1, 8),
            (29, 256, 56),
            (30, 1, 2),
            (34, 256, 54),
            (35, 256, 66),
            (39, 32, 17),
            (40, 64, 36),
            (41, 128, 18),
            (42, 64, 18),
            (43, 1, 1),
        ];
        for (id, block, bytes) in upstream {
            let ty = GgmlType::from_id(id).unwrap_or_else(|| panic!("type id {id} should be known"));
            assert_eq!(
                (ty.block_size(), ty.type_size()),
                (block, bytes),
                "type id {id} ({ty:?})"
            );
        }
    }

    /// Why: ids outside the table (removed, reserved or newer than this reader) must not map to a type.
    #[test]
    fn unlisted_type_ids_are_not_known() {
        for id in [4, 5, 31, 32, 33, 36, 37, 38, 44, 45, 1_000, u32::MAX] {
            assert_eq!(GgmlType::from_id(id), None, "id {id}");
        }
    }

    /// Why: the hand-computed size of a real-shaped NVFP4 matrix: 4096 rows of 64 blocks of 36 bytes.
    #[test]
    fn nvfp4_tensor_size_is_blocks_times_36() {
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[64]), Ok(36));
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[128, 3]), Ok(2 * 36 * 3));
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[4096, 4096]), Ok(9_437_184));
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[64, 2, 3, 5]), Ok(36 * 2 * 3 * 5));
    }

    /// Why: upstream rejects a row that is not a whole number of blocks, judging the row and not the
    /// total. `[96, 2]` has 192 elements, a multiple of 64, and must still be rejected.
    #[test]
    fn row_not_a_multiple_of_the_block_size_is_rejected_even_when_the_total_is() {
        assert_eq!(
            GgmlType::Nvfp4.tensor_byte_size(&[96, 2]),
            Err(GgufTensorSizeError::RowNotBlockMultiple {
                ggml_type: GgmlType::Nvfp4,
                row_len: 96,
                block_size: 64,
            })
        );
        assert!(GgmlType::Nvfp4.tensor_byte_size(&[63]).is_err());
        assert!(GgmlType::Nvfp4.tensor_byte_size(&[65]).is_err());
        // A scalar has a row of 1, which no multi-value block divides.
        assert!(GgmlType::Nvfp4.tensor_byte_size(&[]).is_err());
    }

    /// Why: unquantized types have a block of 1, so any shape is accepted, a scalar included.
    #[test]
    fn unquantized_types_accept_any_row_length() {
        assert_eq!(GgmlType::F32.tensor_byte_size(&[]), Ok(4));
        assert_eq!(GgmlType::F32.tensor_byte_size(&[7, 3]), Ok(7 * 3 * 4));
        assert_eq!(GgmlType::F16.tensor_byte_size(&[5]), Ok(10));
    }

    /// Why: an empty tensor is representable upstream and has no bytes.
    #[test]
    fn zero_extent_gives_zero_bytes_without_an_error() {
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[0]), Ok(0));
        assert_eq!(GgmlType::Nvfp4.tensor_byte_size(&[64, 0]), Ok(0));
    }

    /// Why: every step of the size arithmetic is checked; a crafted shape must be an error, not a wrap.
    #[test]
    fn overflowing_shapes_are_errors_not_wrapped_sizes() {
        // Element count overflows u64.
        assert_eq!(
            GgmlType::F32.tensor_byte_size(&[1 << 62, 1 << 62]),
            Err(GgufTensorSizeError::Overflow)
        );
        // Element count fits i64 but bytes (8 per element) overflow u64.
        assert_eq!(
            GgmlType::F64.tensor_byte_size(&[i64::MAX as u64]),
            Err(GgufTensorSizeError::Overflow)
        );
        // An extent above i64::MAX is a negative `int64_t` upstream.
        assert_eq!(
            GgmlType::F32.tensor_byte_size(&[u64::MAX]),
            Err(GgufTensorSizeError::Overflow)
        );
        // Element count 2^63 is above i64::MAX but still within u64.
        assert_eq!(
            GgmlType::I8.tensor_byte_size(&[1 << 62, 2]),
            Err(GgufTensorSizeError::Overflow)
        );
    }

    /// Why: GGML_MAX_DIMS is 4.
    #[test]
    fn more_than_four_dimensions_are_rejected() {
        assert_eq!(
            GgmlType::F32.tensor_byte_size(&[1, 1, 1, 1, 1]),
            Err(GgufTensorSizeError::TooManyDims { n_dims: 5 })
        );
        assert!(GgmlType::F32.tensor_byte_size(&[1, 1, 1, 1]).is_ok());
    }
}
