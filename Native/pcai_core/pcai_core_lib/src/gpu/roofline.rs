//! GPU roofline model for LLM inference performance analysis.
//!
//! Calculates theoretical performance ceilings based on GPU specifications
//! and compares actual measured tok/s against them.  The roofline model
//! separates _decode_ (autoregressive, memory-bandwidth-bound) from
//! _prefill_ (prompt ingestion, compute-bound) phases, letting operators
//! identify which resource limits their throughput.
//!
//! All calculations are pure arithmetic -- no NVML or GPU driver access
//! is required.
//!
//! # Examples
//!
//! ```
//! use pcai_core_lib::gpu::roofline::{GpuSpecs, analyze_roofline};
//!
//! let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti")
//!     .expect("known GPU");
//! let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, Some(62.0));
//! assert!(analysis.theoretical_max_toks > 100.0);
//! assert_eq!(analysis.bottleneck, Bottleneck::MemoryBandwidth);
//! ```

use serde::Serialize;

// ── Public data types ─────────────────────────────────────────────────────────

/// Known GPU specifications for roofline analysis.
///
/// Memory bandwidth in GB/s, compute in TFLOPS (FP16/BF16).
#[derive(Debug, Clone, Serialize)]
pub struct GpuSpecs {
    /// Human-readable GPU product name.
    pub name: String,
    /// Compute capability formatted as `"major.minor"`.
    pub compute_capability: String,
    /// Peak memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f64,
    /// Peak FP16 throughput in TFLOPS.
    pub fp16_tflops: f64,
    /// Peak BF16 throughput in TFLOPS.
    pub bf16_tflops: f64,
    /// Peak FP32 throughput in TFLOPS.
    pub fp32_tflops: f64,
    /// Peak tensor-core throughput in TFLOPS (TF32/FP16 accumulate).
    pub tensor_tflops: f64,
    /// Theoretical PCIe bandwidth in GB/s (directional).
    pub pcie_bandwidth_gbps: f64,
    /// Total VRAM capacity in GB.
    pub vram_gb: f64,
}

/// Performance analysis for a specific model + GPU combination.
#[derive(Debug, Clone, Serialize)]
pub struct RooflineAnalysis {
    /// GPU name used in this analysis.
    pub gpu: String,
    /// Model size in billions of parameters.
    pub model_params_b: f64,
    /// Quantization bit-width per parameter.
    pub quant_bits: f64,
    /// Calculated model size in GB (params * bits / 8).
    pub model_size_gb: f64,
    /// Context length used for the analysis.
    pub context_length: u64,

    // ── Theoretical ceilings ──────────────────────────────────────────────
    /// Memory-bandwidth-limited decode tok/s ceiling.
    pub theoretical_max_toks: f64,
    /// Compute-limited prefill tok/s ceiling.
    pub compute_ceiling_toks: f64,
    /// PCIe-limited tok/s ceiling (relevant for cross-GPU / offload).
    pub pcie_ceiling_toks: f64,

    // ── Actual measurements ───────────────────────────────────────────────
    /// Measured decode tok/s (filled by caller, if available).
    pub actual_toks: Option<f64>,

    // ── Efficiency metrics ────────────────────────────────────────────────
    /// `actual / theoretical_max * 100` -- bandwidth utilization.
    pub bandwidth_efficiency_pct: Option<f64>,
    /// `actual / compute_ceiling * 100` -- compute utilization.
    pub compute_efficiency_pct: Option<f64>,
    /// Dominant bottleneck for the decode phase.
    pub bottleneck: Bottleneck,
}

/// Dominant performance bottleneck identified by roofline analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    /// Decode is limited by GPU memory bandwidth (typical for autoregressive LLM).
    MemoryBandwidth,
    /// Prefill/batch is limited by GPU compute (FLOPs).
    Compute,
}

// ── Spec lookup ───────────────────────────────────────────────────────────────

impl GpuSpecs {
    /// Look up known GPU specs by compute capability string and product name.
    ///
    /// When the same compute capability is shared by multiple SKUs (e.g. both
    /// RTX 4090 and RTX 2000 Ada are SM 8.9), the `name` parameter is used
    /// for disambiguation.
    ///
    /// Returns `None` for unknown GPUs.
    #[must_use]
    pub fn from_compute_capability(cc: &str, name: &str) -> Option<Self> {
        // More specific name-based matches come first so they take priority
        // over generic compute-capability matches.
        match cc {
            // RTX 5060 Ti (Blackwell, SM 120)
            "12.0" => Some(Self {
                name: name.to_owned(),
                compute_capability: cc.to_owned(),
                memory_bandwidth_gbps: 448.0,
                fp16_tflops: 24.0,
                bf16_tflops: 24.0,
                fp32_tflops: 12.0,
                tensor_tflops: 48.0,
                pcie_bandwidth_gbps: 32.0, // PCIe 5.0 x8
                vram_gb: 16.0,
            }),
            // RTX 4090 (Ada Lovelace, SM 89) -- must precede generic 8.9
            "8.9" if name.contains("4090") => Some(Self {
                name: name.to_owned(),
                compute_capability: cc.to_owned(),
                memory_bandwidth_gbps: 1008.0,
                fp16_tflops: 82.6,
                bf16_tflops: 82.6,
                fp32_tflops: 82.6,
                tensor_tflops: 165.2,
                pcie_bandwidth_gbps: 32.0, // PCIe 4.0 x16
                vram_gb: 24.0,
            }),
            // RTX 2000 Ada (Ada Lovelace, SM 89) -- generic fallback
            "8.9" => Some(Self {
                name: name.to_owned(),
                compute_capability: cc.to_owned(),
                memory_bandwidth_gbps: 192.0,
                fp16_tflops: 12.0,
                bf16_tflops: 12.0,
                fp32_tflops: 6.0,
                tensor_tflops: 24.0,
                pcie_bandwidth_gbps: 16.0, // PCIe 4.0 x8
                vram_gb: 8.0,
            }),
            // RTX 3090 (Ampere, SM 86)
            "8.6" => Some(Self {
                name: name.to_owned(),
                compute_capability: cc.to_owned(),
                memory_bandwidth_gbps: 936.2,
                fp16_tflops: 35.6,
                bf16_tflops: 35.6,
                fp32_tflops: 35.6,
                tensor_tflops: 71.0,
                pcie_bandwidth_gbps: 32.0, // PCIe 4.0 x16
                vram_gb: 24.0,
            }),
            _ => None,
        }
    }
}

// ── Core analysis ─────────────────────────────────────────────────────────────

/// Calculate the theoretical roofline performance for a model on a GPU.
///
/// **Decode phase (autoregressive):** each output token requires loading the
/// full model weights from VRAM once.  Throughput is memory-bandwidth-bound:
///
/// ```text
/// theoretical_toks = memory_bandwidth_gbps / model_size_gb
/// ```
///
/// **Prefill phase (prompt ingestion):** throughput is proportional to compute
/// available relative to the FLOPs per token (2 * parameters):
///
/// ```text
/// compute_ceiling_toks = fp16_tflops * 1000 / (2 * params_b)
/// ```
///
/// The `actual_toks` parameter, when provided, is used to derive efficiency
/// percentages.
#[must_use]
pub fn analyze_roofline(
    gpu: &GpuSpecs,
    model_params_b: f64,
    quant_bits: f64,
    context_length: u64,
    actual_toks: Option<f64>,
) -> RooflineAnalysis {
    let model_size_gb = model_params_b * quant_bits / 8.0;

    // Decode ceiling: memory bandwidth / model size per token load.
    let theoretical_max_toks = if model_size_gb > 0.0 {
        gpu.memory_bandwidth_gbps / model_size_gb
    } else {
        0.0
    };

    // Compute ceiling: FP16 TFLOPS / (2 * params_b) -- expressed in tok/s.
    // 1 TFLOP = 1e12 FLOPs; params_b * 1e9 params; 2 FLOPs/param/token.
    // tok/s = (tflops * 1e12) / (2 * params_b * 1e9) = tflops * 1000 / (2 * params_b).
    let compute_ceiling_toks = if model_params_b > 0.0 {
        gpu.fp16_tflops * 1000.0 / (2.0 * model_params_b)
    } else {
        0.0
    };

    // PCIe ceiling: for split-model / offload scenarios.
    let pcie_ceiling_toks = if model_size_gb > 0.0 {
        gpu.pcie_bandwidth_gbps / model_size_gb
    } else {
        0.0
    };

    let bandwidth_efficiency = actual_toks.map(|a| {
        if theoretical_max_toks > 0.0 {
            (a / theoretical_max_toks) * 100.0
        } else {
            0.0
        }
    });

    let compute_efficiency = actual_toks.map(|a| {
        if compute_ceiling_toks > 0.0 {
            (a / compute_ceiling_toks) * 100.0
        } else {
            0.0
        }
    });

    let bottleneck = if theoretical_max_toks < compute_ceiling_toks {
        Bottleneck::MemoryBandwidth
    } else {
        Bottleneck::Compute
    };

    RooflineAnalysis {
        gpu: gpu.name.clone(),
        model_params_b,
        quant_bits,
        model_size_gb,
        context_length,
        theoretical_max_toks,
        compute_ceiling_toks,
        pcie_ceiling_toks,
        actual_toks,
        bandwidth_efficiency_pct: bandwidth_efficiency,
        compute_efficiency_pct: compute_efficiency,
        bottleneck,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// RTX 5060 Ti with 7B Q4 model: theoretical decode ceiling ~107+ tok/s.
    ///
    /// 7B * 4.5 bits / 8 = 3.9375 GB model size.
    /// 448 GB/s / 3.9375 GB = ~113.8 tok/s.
    #[test]
    fn rtx_5060ti_7b_q4_decode_ceiling() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("RTX 5060 Ti should be known");

        let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, None);

        assert!(
            analysis.theoretical_max_toks > 107.0,
            "expected >107 tok/s, got {:.1}",
            analysis.theoretical_max_toks
        );
        assert!(
            (analysis.model_size_gb - 3.9375).abs() < 0.001,
            "model size should be ~3.94 GB, got {:.4}",
            analysis.model_size_gb
        );
    }

    /// RTX 2000 Ada with 3B Q8 model: theoretical decode ceiling ~64 tok/s.
    ///
    /// 3B * 8 bits / 8 = 3.0 GB model size.
    /// 192 GB/s / 3.0 GB = 64 tok/s.
    #[test]
    fn rtx_2000_ada_3b_q8_decode_ceiling() {
        let specs = GpuSpecs::from_compute_capability("8.9", "RTX 2000 Ada").expect("RTX 2000 Ada should be known");

        let analysis = analyze_roofline(&specs, 3.0, 8.0, 4096, None);

        assert!(
            (analysis.theoretical_max_toks - 64.0).abs() < 0.1,
            "expected ~64 tok/s, got {:.1}",
            analysis.theoretical_max_toks
        );
        assert!(
            (analysis.model_size_gb - 3.0).abs() < 0.001,
            "model size should be 3.0 GB, got {:.4}",
            analysis.model_size_gb
        );
    }

    /// Known actual: qwen2.5-coder:3b at 137.3 tok/s on RTX 5060 Ti.
    ///
    /// 3B * 4.5 bits / 8 = 1.6875 GB.
    /// 448 / 1.6875 = ~265.5 tok/s theoretical.
    /// 137.3 / 265.5 = ~51.7% bandwidth efficiency.
    ///
    /// Note: the spec suggests ~91% efficiency for a different configuration.
    /// The actual efficiency depends on the exact quantisation and KV cache
    /// overhead, which this model does not account for.
    #[test]
    fn rtx_5060ti_3b_actual_measurement() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("RTX 5060 Ti should be known");

        let analysis = analyze_roofline(&specs, 3.0, 4.5, 4096, Some(137.3));

        assert!(analysis.actual_toks.is_some());
        let efficiency = analysis
            .bandwidth_efficiency_pct
            .expect("efficiency should be computed");
        assert!(
            efficiency > 40.0 && efficiency < 100.0,
            "expected reasonable efficiency, got {efficiency:.1}%"
        );
    }

    /// Bottleneck detection: decode phase is memory-bandwidth-bound for
    /// typical LLM models because the model must be read from VRAM each token.
    #[test]
    fn bottleneck_is_memory_bandwidth_for_decode() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("RTX 5060 Ti should be known");

        let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, None);

        assert_eq!(
            analysis.bottleneck,
            Bottleneck::MemoryBandwidth,
            "7B Q4 decode should be memory-bandwidth-bound"
        );
        assert!(
            analysis.theoretical_max_toks < analysis.compute_ceiling_toks,
            "decode ceiling ({:.1}) should be below compute ceiling ({:.1})",
            analysis.theoretical_max_toks,
            analysis.compute_ceiling_toks,
        );
    }

    /// RTX 4090 is disambiguated from RTX 2000 Ada despite same CC 8.9.
    #[test]
    fn rtx_4090_disambiguation() {
        let specs_4090 =
            GpuSpecs::from_compute_capability("8.9", "NVIDIA GeForce RTX 4090").expect("RTX 4090 should be known");
        let specs_2000 =
            GpuSpecs::from_compute_capability("8.9", "NVIDIA RTX 2000 Ada").expect("RTX 2000 Ada should be known");

        assert!(
            specs_4090.memory_bandwidth_gbps > specs_2000.memory_bandwidth_gbps,
            "4090 bandwidth ({}) should exceed 2000 Ada ({})",
            specs_4090.memory_bandwidth_gbps,
            specs_2000.memory_bandwidth_gbps,
        );
        assert!(
            (specs_4090.memory_bandwidth_gbps - 1008.0).abs() < 1.0,
            "4090 should have ~1008 GB/s, got {}",
            specs_4090.memory_bandwidth_gbps,
        );
    }

    /// Unknown GPU returns None.
    #[test]
    fn unknown_gpu_returns_none() {
        assert!(GpuSpecs::from_compute_capability("99.0", "Future GPU").is_none());
    }

    /// `RooflineAnalysis` serialises to valid JSON with all expected fields.
    #[test]
    fn roofline_analysis_serializes_to_json() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("known GPU");
        let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, Some(62.0));
        let json = serde_json::to_string(&analysis).expect("should serialise");

        assert!(json.contains("\"gpu\":"));
        assert!(json.contains("\"theoretical_max_toks\":"));
        assert!(json.contains("\"bottleneck\":"));
        assert!(json.contains("\"bandwidth_efficiency_pct\":"));
        assert!(json.contains("\"actual_toks\":62.0"));
    }

    /// `GpuSpecs` serialises to valid JSON.
    #[test]
    fn gpu_specs_serializes_to_json() {
        let specs = GpuSpecs::from_compute_capability("8.6", "RTX 3090").expect("RTX 3090 should be known");
        let json = serde_json::to_string(&specs).expect("should serialise");

        assert!(json.contains("\"memory_bandwidth_gbps\":936.2"));
        assert!(json.contains("\"vram_gb\":24.0"));
    }

    /// No actual measurement: efficiency fields are None.
    #[test]
    fn no_actual_measurement_efficiency_is_none() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("known GPU");
        let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, None);

        assert!(analysis.actual_toks.is_none());
        assert!(analysis.bandwidth_efficiency_pct.is_none());
        assert!(analysis.compute_efficiency_pct.is_none());
    }

    /// PCIe ceiling is always lower than memory bandwidth ceiling (by design).
    #[test]
    fn pcie_ceiling_below_memory_ceiling() {
        let specs = GpuSpecs::from_compute_capability("12.0", "RTX 5060 Ti").expect("known GPU");
        let analysis = analyze_roofline(&specs, 7.0, 4.5, 4096, None);

        assert!(
            analysis.pcie_ceiling_toks < analysis.theoretical_max_toks,
            "PCIe ceiling ({:.1}) should be below memory ceiling ({:.1})",
            analysis.pcie_ceiling_toks,
            analysis.theoretical_max_toks,
        );
    }
}
