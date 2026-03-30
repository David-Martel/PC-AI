//! Integration tests for the preflight module.
//!
//! These tests require the `nvml` feature and exercise the full preflight
//! path including NVML queries. They degrade gracefully on machines
//! without NVIDIA hardware.

#[cfg(feature = "nvml")]
mod nvml_tests {
    use pcai_core_lib::preflight;

    #[test]
    fn check_vram_state_returns_valid_result() {
        let result = preflight::check_vram_state(0).expect("check_vram_state should not error");
        assert!(
            matches!(result.verdict, preflight::Verdict::Go | preflight::Verdict::Fail),
            "inventory-only should be Go (with GPU) or Fail (without)"
        );
    }

    #[test]
    fn check_vram_state_with_absurd_requirement_fails() {
        let result = preflight::check_vram_state(999_999).expect("should not error even with huge requirement");
        // No GPU has 999 GB of VRAM
        assert_eq!(result.verdict, preflight::Verdict::Fail);
    }

    #[test]
    fn check_readiness_with_nonexistent_file_errors() {
        let result = preflight::check_readiness("/nonexistent/model.gguf", 0);
        assert!(result.is_err(), "nonexistent model file should return Err");
    }

    #[test]
    fn vram_snapshot_gpu_names_not_empty() {
        let snapshots = preflight::vram_audit::vram_snapshot_all().unwrap_or_default();
        for gpu in &snapshots {
            assert!(!gpu.name.is_empty(), "GPU name should not be empty");
            assert!(gpu.total_mb > 0, "GPU total VRAM should be >0");
        }
    }

    #[test]
    fn preflight_result_json_is_compact() {
        let result = preflight::check_vram_state(0).expect("should not error");
        let json = serde_json::to_string(&result).expect("should serialize");
        assert!(
            json.len() < 4096,
            "preflight JSON should be compact, got {} bytes",
            json.len()
        );
    }
}
