//! Fuzz `read_gguf_meta_from` and `read_gguf_layout_from` with arbitrary bytes.
//!
//! The readers are reachable from PowerShell through `pcai_gpu_preflight_json`, and the
//! release profile aborts on panic, so any panic, stack overflow or runaway allocation
//! here is a process-killing defect. The property is simply "never panics, never
//! overflows the stack, always returns"; a successful parse must also give a memory
//! estimate without panicking. The layout reader adds the tensor-info section, which a
//! hostile file controls just as much as the metadata.
#![no_main]

use libfuzzer_sys::fuzz_target;

// The reader source itself, so fuzzing exercises exactly the shipped code.
#[path = "../../pcai_core_lib/src/preflight/gguf.rs"]
#[expect(dead_code, reason = "the fuzz target exercises only the two reader entry points and the estimators; the rest of the module is shipped code compiled along with it")]
mod gguf;

use gguf::{
    estimate_layout_memory_mb, estimate_model_memory_mb, read_gguf_header_from, read_gguf_layout_from, read_gguf_meta_from,
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
    if let Ok(layout) = read_gguf_layout_from(data) {
        let _ = estimate_layout_memory_mb(&layout, layout.meta.context_length);
    }
});
