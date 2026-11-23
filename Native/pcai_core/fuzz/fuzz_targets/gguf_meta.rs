//! Fuzz `read_gguf_meta_from` with arbitrary bytes.
//!
//! The reader is reachable from PowerShell through `pcai_gpu_preflight_json`, and the
//! release profile aborts on panic, so any panic, stack overflow or runaway allocation
//! here is a process-killing defect. The property is simply "never panics, never
//! overflows the stack, always returns"; a successful parse must also give a memory
//! estimate without panicking.
#![no_main]

use libfuzzer_sys::fuzz_target;

// The reader source itself, so fuzzing exercises exactly the shipped code.
#[path = "../../pcai_core_lib/src/preflight/gguf.rs"]
#[expect(dead_code, reason = "the fuzz target exercises only read_gguf_meta_from; the rest of the module is shipped code compiled along with it")]
mod gguf;

use gguf::{estimate_model_memory_mb, read_gguf_meta_from};

fuzz_target!(|data: &[u8]| {
    if let Ok(meta) = read_gguf_meta_from(data) {
        let _ = estimate_model_memory_mb(&meta, meta.context_length);
    }
});
