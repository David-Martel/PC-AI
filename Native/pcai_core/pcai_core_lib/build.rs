//! Build script for pcai_core_lib.
#![expect(clippy::expect_used)]
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    // Without PCAI_BUILD_VERSION (set by Build.ps1 from Tools/Get-BuildVersion.ps1) the commit count is unknown, so
    // report the workspace version with an explicit "unknown" build: a "-dev" pre-release would sort before the
    // release it was built after. See docs/VERSIONING.md.
    let version = env::var("PCAI_BUILD_VERSION").unwrap_or_else(|_| format!("{}+unknown", env!("CARGO_PKG_VERSION")));
    let out_dir = env::var_os("OUT_DIR").expect("build script: required operation failed (no OUT_DIR)");
    let dest_path = Path::new(&out_dir).join("version.rs");

    // Create a null-terminated version string for FFI
    let version_cstr = format!("{}\0", version);

    fs::write(
        &dest_path,
        format!(
            "/// Version string\npub const VERSION: &str = \"{}\";\n/// Null-terminated version string for FFI\npub const VERSION_CSTR: &[u8] = b\"{}\";\n",
            version, version_cstr
        ),
    )
    .expect("build script: required operation failed");

    println!("cargo:rerun-if-env-changed=PCAI_BUILD_VERSION");
}
