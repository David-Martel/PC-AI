use std::env;
use std::fs;
use std::path::Path;

fn env_or_default(name: &str, fallback: &str) -> String {
    env::var(name).unwrap_or_else(|_| fallback.to_string())
}

fn escape_for_rust(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR missing");
    let dest_path = Path::new(&out_dir).join("version.rs");

    let version = env_or_default(
        "NUKENUL_VERSION",
        &env_or_default("BUILD_VERSION", "0.1.0+unknown"),
    );
    let semver = env_or_default("NUKENUL_SEMVER", &env_or_default("BUILD_SEMVER", "0.1.0"));
    let release_tag = env_or_default(
        "NUKENUL_RELEASE_TAG",
        &env_or_default("BUILD_RELEASE_TAG", "v0.1.0"),
    );
    let git_hash = env_or_default(
        "NUKENUL_GIT_HASH",
        &env_or_default("BUILD_GIT_HASH", "unknown"),
    );
    let git_hash_short = env_or_default(
        "NUKENUL_GIT_HASH_SHORT",
        &env_or_default("BUILD_GIT_HASH_SHORT", "unknown"),
    );
    let build_timestamp = env_or_default(
        "NUKENUL_BUILD_TIMESTAMP",
        &env_or_default("BUILD_BUILD_TIMESTAMP", "unknown"),
    );
    let build_type = env_or_default(
        "NUKENUL_BUILD_TYPE",
        &env_or_default("BUILD_BUILD_TYPE", "dev"),
    );

    let version_literal = escape_for_rust(&version);
    let version_cstr_literal = format!("{version_literal}\\0");
    let content = format!(
        "pub const VERSION: &str = \"{version}\";\n\
         pub const SEMVER: &str = \"{semver}\";\n\
         pub const RELEASE_TAG: &str = \"{release_tag}\";\n\
         pub const GIT_HASH: &str = \"{git_hash}\";\n\
         pub const GIT_HASH_SHORT: &str = \"{git_hash_short}\";\n\
         pub const BUILD_TIMESTAMP: &str = \"{build_timestamp}\";\n\
         pub const BUILD_TYPE: &str = \"{build_type}\";\n\
         pub const VERSION_CSTR: &[u8] = b\"{version_cstr}\";\n",
        version = version_literal,
        semver = escape_for_rust(&semver),
        release_tag = escape_for_rust(&release_tag),
        git_hash = escape_for_rust(&git_hash),
        git_hash_short = escape_for_rust(&git_hash_short),
        build_timestamp = escape_for_rust(&build_timestamp),
        build_type = escape_for_rust(&build_type),
        version_cstr = version_cstr_literal,
    );

    fs::write(&dest_path, content).expect("failed to write version.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BUILD_VERSION");
    println!("cargo:rerun-if-env-changed=BUILD_SEMVER");
    println!("cargo:rerun-if-env-changed=BUILD_RELEASE_TAG");
    println!("cargo:rerun-if-env-changed=BUILD_GIT_HASH");
    println!("cargo:rerun-if-env-changed=BUILD_GIT_HASH_SHORT");
    println!("cargo:rerun-if-env-changed=BUILD_BUILD_TIMESTAMP");
    println!("cargo:rerun-if-env-changed=BUILD_BUILD_TYPE");
    println!("cargo:rerun-if-env-changed=NUKENUL_VERSION");
    println!("cargo:rerun-if-env-changed=NUKENUL_SEMVER");
    println!("cargo:rerun-if-env-changed=NUKENUL_RELEASE_TAG");
    println!("cargo:rerun-if-env-changed=NUKENUL_GIT_HASH");
    println!("cargo:rerun-if-env-changed=NUKENUL_GIT_HASH_SHORT");
    println!("cargo:rerun-if-env-changed=NUKENUL_BUILD_TIMESTAMP");
    println!("cargo:rerun-if-env-changed=NUKENUL_BUILD_TYPE");
}
