use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let version = env::var("PCAI_BUILD_VERSION").unwrap_or_else(|_| "0.1.0-dev".to_string());
    let out_dir = env::var_os("OUT_DIR").expect("TODO: Verify unwrap");
    let dest_path = Path::new(&out_dir).join("version.rs");

    // Create a null-terminated version string for FFI
    let version_cstr = format!("{}\0", version);

    fs::write(
        &dest_path,
        format!(
            "pub const VERSION: &str = \"{}\";\npub const VERSION_CSTR: &[u8] = b\"{}\";\n",
            version, version_cstr
        ),
    )
    .expect("TODO: Verify unwrap");

    println!("cargo:rerun-if-env-changed=PCAI_BUILD_VERSION");
}
