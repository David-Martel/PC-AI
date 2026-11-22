//! Build version and metadata information.
//!
//! This module provides compile-time embedded version information
//! including git hash, build timestamp, and feature flags.

// Include the auto-generated build info
include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_not_empty() {
        assert!(!VERSION.is_empty());
        assert!(!GIT_HASH.is_empty());
        assert!(!BUILD_TIMESTAMP.is_empty());
    }

    #[test]
    fn test_build_info_format() {
        let info = build_info();
        assert!(info.contains("pcai-inference"));
        assert!(info.contains("Commit:"));
    }

    #[test]
    fn test_build_info_json() {
        let json = build_info_json();
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
        assert!(json.contains("\"version\""));
    }

    /// The CUDA features this crate can be compiled with. `cuda` is an umbrella that enables both.
    const CUDA_COMPILED: bool = cfg!(any(feature = "cuda-llamacpp", feature = "cuda-mistralrs"));

    /// Why: installed artifacts reported `has_cuda: true` although they were built without a CUDA feature, because
    /// build.rs derived it from `CUDA_PATH` on the build host (reported by codex on PR #403). The embedded flag must
    /// describe the compiled code, which the compiler's own feature set states exactly.
    #[test]
    fn test_has_cuda_matches_the_compiled_features() {
        assert_eq!(
            HAS_CUDA, CUDA_COMPILED,
            "HAS_CUDA must follow the compiled CUDA features, not the build host"
        );
        let features: Vec<&str> = FEATURES.split(',').filter(|f| !f.is_empty()).collect();
        assert_eq!(
            features.iter().any(|f| f.starts_with("cuda-")),
            CUDA_COMPILED,
            "FEATURES ({FEATURES}) disagrees with the compiled CUDA features"
        );
    }

    #[test]
    fn test_a_build_without_cuda_reports_no_cuda() {
        if CUDA_COMPILED {
            return; // Covered by test_has_cuda_matches_the_compiled_features for CUDA builds.
        }
        assert_eq!(CUDA_VERSION, "none");
        assert!(build_info_json().contains("\"has_cuda\":false"));
        assert!(build_info().contains("CUDA: disabled"));
    }
}
