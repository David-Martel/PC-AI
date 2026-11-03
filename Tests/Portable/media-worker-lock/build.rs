use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=PCAI_MEDIA_FFI_SOURCE");
    let source = PathBuf::from(env::var_os("PCAI_MEDIA_FFI_SOURCE").expect("set an explicit media FFI source path"));
    println!("cargo:rerun-if-changed={}", source.display());
    let text = fs::read_to_string(&source).expect("read selected media FFI source");
    let entry = "pub extern \"C\" fn pcai_media_generate_image_async(";
    assert_eq!(text.matches(entry).count(), 1, "one actual async producer");
    let body = text.split_once(entry).expect("async producer").1;
    let begin = "// Run the synchronous generation under the lock.";
    assert_eq!(body.matches(begin).count(), 1, "one model acquisition boundary");
    let acquisition = body.split_once(begin).expect("acquisition boundary").1;
    let acquisition = acquisition.split_once("let result = {").expect("actual result block").1;
    let end = "// Check for cancellation before expensive work.";
    assert_eq!(acquisition.matches(end).count(), 1, "one cancellation boundary");
    let acquisition = acquisition.split_once(end).expect("cancellation boundary").0;
    assert!(acquisition.contains("let state = get_state();"));
    assert!(acquisition.contains("let guard = match state.lock()"));
    // The body is copied verbatim between maintained producer boundaries. Only
    // its enclosing unit-returning fixture function and final guard drop are new.
    let generated = format!("fn run_actual_worker_acquisition(id: i64) {{\n{acquisition}\n    drop(guard);\n}}\n");
    let target = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    fs::write(target.join("worker_acquisition.rs"), generated).expect("write source-bound probe");
}
