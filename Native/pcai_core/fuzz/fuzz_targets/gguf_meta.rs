//! Fuzz `read_gguf_meta_from`, `read_gguf_layout_from` and `validate_gguf_layout_from` with arbitrary bytes.
//!
//! The readers are reachable from PowerShell through `pcai_gpu_preflight_json`, and the
//! release profile aborts on panic, so any panic, stack overflow or runaway allocation
//! here is a process-killing defect. The property is simply "never panics, never
//! overflows the stack, always returns"; a successful parse must also give a memory
//! estimate without panicking. The layout reader adds the tensor-info section, which a
//! hostile file controls just as much as the metadata. The validator adds the declared
//! offsets and alignment, and takes a file length: the fuzzer's input length, one byte
//! less, zero and `u64::MAX`, so both sides of the extent check are reached.
#![no_main]

use libfuzzer_sys::fuzz_target;

// The reader source itself, so fuzzing exercises exactly the shipped code.
#[path = "../../pcai_core_lib/src/preflight/gguf.rs"]
#[expect(dead_code, reason = "the fuzz target exercises only the two reader entry points and the estimators; the rest of the module is shipped code compiled along with it")]
mod gguf;

use gguf::{
    estimate_layout_memory_mb, estimate_model_memory_mb, read_gguf_header_from, read_gguf_layout_from, read_gguf_meta_from,
    validate_gguf_layout_from,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(meta) = read_gguf_meta_from(data) {
        let _ = estimate_model_memory_mb(&meta, meta.context_length);
        // The readiness scan must never fail where the metadata reader succeeds: the tensor
        // section is best effort there, and main never read it.
        assert!(
            read_gguf_header_from(data).is_ok(),
            "header scan failed where the metadata reader succeeded"
        );
    }
    let layout = read_gguf_layout_from(data);
    if let Ok(layout) = &layout {
        let _ = estimate_layout_memory_mb(layout, layout.meta.context_length);
    }

    let len = data.len() as u64;
    for file_len in [len, len.saturating_sub(1), 0, u64::MAX] {
        if let Ok(valid) = validate_gguf_layout_from(data, file_len) {
            // A layout the validator accepts has been sized by the same reader, so that reader
            // must accept it too and agree on where the header ends.
            let layout = layout
                .as_ref()
                .expect("the layout reader accepts what the validator accepts");
            assert_eq!(valid.header_bytes, layout.header_bytes, "header length disagrees");
            assert_eq!(valid.tensor_count, layout.tensor_count, "tensor count disagrees");
            assert!(valid.data_offset >= valid.header_bytes, "data section starts inside the header");
            assert!(valid.data_offset <= file_len, "data section starts past the file");
            assert!(valid.alignment.is_power_of_two(), "alignment is not a power of two");
        }
    }
});
