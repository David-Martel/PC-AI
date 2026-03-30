use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Detect the CUDA toolkit major version from nvcc --version output.
/// Returns None if nvcc is not found or version cannot be parsed.
fn detect_cuda_major_version() -> Option<u32> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // nvcc output contains a line like: "Cuda compilation tools, release 13.2, V13.2.68"
    for line in stdout.lines() {
        if let Some(pos) = line.find("release ") {
            let version_str = &line[pos + 8..];
            if let Some(dot) = version_str.find('.') {
                return version_str[..dot].parse::<u32>().ok();
            }
        }
    }
    None
}

/// Add MSVC-specific CUDA 13+ flags to a bindgen_cuda builder.
///
/// CUDA 13.x CCCL headers require the standard conforming preprocessor
/// (`/Zc:preprocessor`) on MSVC. Without this flag, nvcc fails with:
///   "MSVC/cl.exe with traditional preprocessor is used."
fn add_msvc_cuda13_flags(builder: bindgen_cuda::Builder, is_msvc: bool) -> bindgen_cuda::Builder {
    if !is_msvc {
        return builder;
    }
    let cuda_major = detect_cuda_major_version().unwrap_or(0);
    if cuda_major >= 13 {
        builder
            .arg("--compiler-options")
            .arg("/Zc:preprocessor")
    } else {
        builder
    }
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    let is_target_msvc = env::var("TARGET")
        .map(|t| t.contains("msvc"))
        .unwrap_or(false);

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let builder = add_msvc_cuda13_flags(builder, is_target_msvc);
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if is_target_msvc {
        moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
    }

    moe_builder = add_msvc_cuda13_flags(moe_builder, is_target_msvc);

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
