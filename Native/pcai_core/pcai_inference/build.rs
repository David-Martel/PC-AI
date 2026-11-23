use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Whether this artifact has CUDA is a property of the features it is compiled with, never of the build host.
    // Deriving it from the CUDA toolkit environment made builds without a CUDA feature report `has_cuda: true`
    // (reported by codex on PR #403). Cargo sets CARGO_FEATURE_* for the compiled features, and a feature change
    // already gives this script a new run, so these reads need no trigger of their own.
    let has_cuda = cuda_compiled();
    if has_cuda {
        println!("cargo:rustc-cfg=has_cuda");
        // get_cuda_version runs the nvcc found on PATH to label a CUDA build, so PATH is an input whenever a CUDA
        // feature is compiled. It is not declared otherwise: PATH differs between shells and build wrappers, and
        // nothing else here depends on it.
        println!("cargo:rerun-if-env-changed=PATH");
    }

    // Generate version information
    generate_version_info(has_cuda);

    // Rerun build script if these change
    println!("cargo:rerun-if-changed=build.rs");
    emit_git_rerun_paths();
}

/// True when a CUDA backend feature is compiled in. `cuda` is an umbrella feature that enables both of these, and
/// src/version.rs tests that the result agrees with the compiler's own `cfg!(feature = ...)`.
fn cuda_compiled() -> bool {
    env::var_os("CARGO_FEATURE_CUDA_LLAMACPP").is_some() || env::var_os("CARGO_FEATURE_CUDA_MISTRALRS").is_some()
}

/// Reads an environment variable the generated metadata depends on, after declaring it to Cargo.
///
/// Every external input of this script goes through here, so the declaration cannot be forgotten. Since the Git
/// watch paths were fixed (PR #403) Cargo caches this script, and an input without `rerun-if-env-changed` leaves the
/// previous value embedded after a rebuild: changing only `PCAI_BUILD_VERSION` used to keep the old version.
/// Variables Cargo sets for build scripts (`OUT_DIR`, `TARGET`, `PROFILE`, `CARGO_*`) are read directly: a change to
/// any of them already gives this script a new run. Tests/Portable/test_inference_build_metadata.py checks both the
/// declarations and a warm rebuild per input.
fn tracked_env(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

/// Runs git in the crate directory and returns trimmed stdout, or `None` when git is absent or fails.
fn git_output(args: &[&str]) -> Option<String> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")?;
    let output = Command::new("git")
        .arg("-C")
        .arg(&manifest_dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Tells Cargo to rerun this script when the commit or the tags behind the embedded version change.
///
/// The paths come from `git rev-parse --git-path`, which resolves them for the repository this crate is in: a plain
/// checkout keeps them under `<root>/.git/`, a linked worktree keeps `HEAD` and its reflog in
/// `.git/worktrees/<name>/` and the refs in the common directory. The previous literal `.git/HEAD` was resolved
/// against this crate's directory, where no such file exists, and Cargo reruns a build script on every build when a
/// `rerun-if-changed` path is missing. Only paths that exist are emitted for the same reason, and nothing is emitted
/// when git is unavailable (for example in a source archive), so `build.rs` stays the only trigger there.
/// `.git/index` is no longer watched: nothing this script embeds depends on the index, and it changes on every
/// `git add` and status refresh.
fn emit_git_rerun_paths() {
    let mut relative = vec![
        "HEAD".to_string(),      // branch switches and detached checkouts
        "logs/HEAD".to_string(), // every commit, reset or checkout in this worktree
        "packed-refs".to_string(),
        "refs/tags".to_string(), // a new tag changes `git describe`
    ];
    if let Some(head_ref) = git_output(&["symbolic-ref", "-q", "HEAD"]) {
        relative.push(head_ref); // the loose branch ref, when it is not packed
    }
    for rel in relative {
        let Some(path) = git_output(&["rev-parse", "--path-format=absolute", "--git-path", &rel]) else {
            continue;
        };
        if Path::new(&path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
}

fn generate_version_info(has_cuda: bool) {
    let out_dir = env::var_os("OUT_DIR").expect("build script: required operation failed");
    let dest_path = Path::new(&out_dir).join("build_info.rs");

    // Try to get version from environment (set by Build.ps1). Both names are read eagerly so both stay declared
    // even while PCAI_VERSION takes precedence.
    let pcai_version = tracked_env("PCAI_VERSION");
    let pcai_build_version = tracked_env("PCAI_BUILD_VERSION");
    let version = pcai_version.or(pcai_build_version).unwrap_or_else(get_version_from_git);

    // The crate version is the workspace (product) version, so an untagged build falls back to it rather than to a
    // literal that has to be kept in step by hand. See docs/VERSIONING.md.
    let semver = tracked_env("PCAI_SEMVER").unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());

    let git_hash = tracked_env("PCAI_GIT_HASH").unwrap_or_else(get_git_hash);

    let git_hash_short = tracked_env("PCAI_GIT_HASH_SHORT").unwrap_or_else(|| {
        if git_hash.len() >= 7 {
            git_hash[..7].to_string()
        } else {
            git_hash.clone()
        }
    });

    let git_branch = tracked_env("PCAI_GIT_BRANCH").unwrap_or_else(get_git_branch);

    let build_timestamp = tracked_env("PCAI_BUILD_TIMESTAMP")
        .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string());

    let build_type = tracked_env("PCAI_BUILD_TYPE").unwrap_or_else(|| "dev".to_string());

    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".to_string());

    // Detect features
    let mut features = Vec::new();
    if env::var("CARGO_FEATURE_LLAMACPP").is_ok() {
        features.push("llamacpp");
    }
    if env::var("CARGO_FEATURE_MISTRALRS_BACKEND").is_ok() {
        features.push("mistralrs");
    }
    if env::var("CARGO_FEATURE_CUDA_LLAMACPP").is_ok() {
        features.push("cuda-llamacpp");
    }
    if env::var("CARGO_FEATURE_CUDA_MISTRALRS").is_ok() {
        features.push("cuda-mistralrs");
    }
    if env::var("CARGO_FEATURE_SERVER").is_ok() {
        features.push("server");
    }
    if env::var("CARGO_FEATURE_FFI").is_ok() {
        features.push("ffi");
    }
    let features_str = features.join(",");

    // has_cuda comes from the compiled features (see cuda_compiled); a build without CUDA reports "none".
    let cuda_version = if has_cuda {
        get_cuda_version()
    } else {
        "none".to_string()
    };

    // The release workflow reads this frame without loading the executable or its GPU runtime.
    const IDENTITY_MARKER: &[u8] = b"PCAI_BUILD_IDENTITY_V1\0";
    let identity_json = serde_json::json!({
        "version": version,
        "semver": semver,
        "git_hash": git_hash,
        "git_branch": git_branch,
        "build_timestamp": build_timestamp,
        "build_type": build_type,
        "target": target,
        "profile": profile,
        "features": features_str,
        "cuda_version": cuda_version,
        "has_cuda": has_cuda,
    })
    .to_string();
    let identity = [IDENTITY_MARKER, identity_json.as_bytes(), b"\0"].concat();
    fs::write(Path::new(&out_dir).join("build_identity.bin"), identity).expect("Failed to write build_identity.bin");

    // Generate Rust source file with build info
    let content = format!(
        r##"// Auto-generated build information. Do not edit.
// Generated by build.rs at compile time.

/// Full version string (semver + commits + hash)
pub const VERSION: &str = "{version}";

/// Semantic version only
pub const SEMVER: &str = "{semver}";

/// Full git commit hash
pub const GIT_HASH: &str = "{git_hash}";

/// Short git commit hash (7 chars)
pub const GIT_HASH_SHORT: &str = "{git_hash_short}";

/// Git branch name
pub const GIT_BRANCH: &str = "{git_branch}";

/// Build timestamp (ISO 8601 UTC)
pub const BUILD_TIMESTAMP: &str = "{build_timestamp}";

/// Build type: release, prerelease, or dev
pub const BUILD_TYPE: &str = "{build_type}";

/// Compilation target triple
pub const TARGET: &str = "{target}";

/// Build profile (debug/release)
pub const PROFILE: &str = "{profile}";

/// Enabled features
pub const FEATURES: &str = "{features_str}";

/// CUDA version (if available)
pub const CUDA_VERSION: &str = "{cuda_version}";

/// Whether CUDA is available
pub const HAS_CUDA: bool = {has_cuda};

/// Null-terminated version for FFI
pub const VERSION_CSTR: &[u8] = b"{version}\0";

/// Build info as a formatted string
pub fn build_info() -> String {{
    format!(
        "pcai-inference {{}}\n\
         Build: {{}} ({{}})\n\
         Commit: {{}} ({{}})\n\
         Target: {{}}\n\
         Features: {{}}\n\
         CUDA: {{}}",
        VERSION,
        BUILD_TIMESTAMP,
        BUILD_TYPE,
        GIT_HASH_SHORT,
        GIT_BRANCH,
        TARGET,
        if FEATURES.is_empty() {{ "none" }} else {{ FEATURES }},
        if HAS_CUDA {{ CUDA_VERSION }} else {{ "disabled" }}
    )
}}

// The CLI uses these exact bytes for --version-info. Pass the whole frame through black_box
// before slicing it so the optimizer keeps the marker and terminator in release executables.
static BUILD_IDENTITY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/build_identity.bin"));

/// Version info as JSON
pub fn build_info_json() -> String {{
    let identity = std::hint::black_box(BUILD_IDENTITY);
    let json = &identity[{identity_marker_len}..identity.len() - 1];
    std::str::from_utf8(json).expect("embedded build identity is UTF-8").to_owned()
}}
"##,
        version = version,
        semver = semver,
        git_hash = git_hash,
        git_hash_short = git_hash_short,
        git_branch = git_branch,
        build_timestamp = build_timestamp,
        build_type = build_type,
        target = target,
        profile = profile,
        features_str = features_str,
        cuda_version = cuda_version,
        has_cuda = has_cuda,
        identity_marker_len = IDENTITY_MARKER.len(),
    );

    fs::write(&dest_path, content).expect("Failed to write build_info.rs");

    // Also set as rustc-env for use in code
    println!("cargo:rustc-env=PCAI_VERSION={}", version);
    println!("cargo:rustc-env=PCAI_GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=PCAI_GIT_HASH_SHORT={}", git_hash_short);
    println!("cargo:rustc-env=PCAI_BUILD_TIMESTAMP={}", build_timestamp);
}

fn get_git_hash() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_git_branch() -> String {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn get_version_from_git() -> String {
    // Try to get version from git describe
    let describe = Command::new("git")
        .args(["describe", "--tags", "--long", "--always"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        });

    if let Some(desc) = describe {
        // Parse: v0.2.0-5-gabcdef or just abcdef
        if desc.contains('-') {
            // Has tag info
            let parts: Vec<&str> = desc.rsplitn(3, '-').collect();
            if parts.len() >= 3 {
                let hash = parts[0].trim_start_matches('g');
                let commits = parts[1];
                let tag = parts[2].trim_start_matches('v');
                if commits == "0" {
                    format!("{}+{}", tag, hash)
                } else {
                    format!("{}.{}+{}", tag, commits, hash)
                }
            } else {
                format!("{}+{}", env!("CARGO_PKG_VERSION"), &desc[..7.min(desc.len())])
            }
        } else {
            // Just a hash
            format!("{}+{}", env!("CARGO_PKG_VERSION"), &desc[..7.min(desc.len())])
        }
    } else {
        format!("{}+unknown", env!("CARGO_PKG_VERSION"))
    }
}

/// Labels a CUDA build with the toolkit release that `nvcc --version` reports, or "unknown". Called only when a
/// CUDA feature is compiled; the toolkit environment never decides whether the build has CUDA.
fn get_cuda_version() -> String {
    // Try nvcc --version
    Command::new("nvcc")
        .args(["--version"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse: "Cuda compilation tools, release 13.1, V13.1.80"
                for line in stdout.lines() {
                    if line.contains("release") {
                        if let Some(start) = line.find("release ") {
                            let rest = &line[start + 8..];
                            if let Some(end) = rest.find(',') {
                                return Some(rest[..end].to_string());
                            }
                        }
                    }
                }
                None
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}
