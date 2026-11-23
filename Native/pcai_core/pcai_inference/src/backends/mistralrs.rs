//! mistral.rs backend implementation
//!
//! This backend uses mistral.rs for high-performance LLM inference with support for:
//! - GGUF quantized models
//! - SafeTensors (HuggingFace) models
//! - CUDA acceleration (when available)
//! - CPU fallback when CUDA is not enabled at build time

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use mistralrs::{Device, GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextModelBuilder};
use mistralrs_core::{ChatCompletionResponse, DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting};

use super::{FinishReason, GenerateRequest, GenerateResponse, InferenceBackend};
use crate::config::{DeviceLayers, DeviceSpec};
use crate::{Error, Result};

/// mistral.rs backend implementation
pub struct MistralRsBackend {
    /// The loaded model instance
    model: Option<Arc<Model>>,
    /// Path to the currently loaded model
    model_path: Option<String>,
    /// Whether the current model is a GGUF model
    is_gguf: bool,
    /// Where the model should run
    device: DeviceSpec,
    /// How layers are placed on that device
    device_layers: DeviceLayers,
}

impl MistralRsBackend {
    #[inline]
    fn cuda_enabled() -> bool {
        // Keep compatibility with legacy umbrella feature (`cuda`) and explicit backend
        // feature (`cuda-mistralrs`) used by Invoke-PcaiBuild for mistralrs.
        cfg!(feature = "cuda") || cfg!(feature = "cuda-mistralrs")
    }

    /// Create a new mistral.rs backend
    pub fn new() -> Self {
        Self {
            model: None,
            model_path: None,
            is_gguf: false,
            device: DeviceSpec::Auto,
            device_layers: DeviceLayers::Auto,
        }
    }

    /// Create a new mistral.rs backend with configuration
    ///
    /// `force_cpu` selects [`DeviceSpec::Cpu`], otherwise [`DeviceSpec::Auto`]. Use
    /// [`MistralRsBackend::with_device`] to request a specific CUDA device.
    pub fn with_config(force_cpu: bool) -> Self {
        Self::with_device(if force_cpu { DeviceSpec::Cpu } else { DeviceSpec::Auto })
    }

    /// Create a new mistral.rs backend for an explicit device policy.
    ///
    /// An explicit [`DeviceSpec::Cuda`] never falls back to the CPU: `load_model` returns an
    /// error when the build has no CUDA support or the device cannot be initialized.
    pub fn with_device(device: DeviceSpec) -> Self {
        Self {
            model: None,
            model_path: None,
            is_gguf: false,
            device,
            device_layers: DeviceLayers::Auto,
        }
    }

    /// Choose how layers are placed; see [`DeviceLayers`]. The default is automatic placement.
    #[must_use]
    pub fn with_device_layers(mut self, device_layers: DeviceLayers) -> Self {
        self.device_layers = device_layers;
        self
    }

    /// Every repeating layer on the primary device, with none on the host.
    ///
    /// mistral.rs clamps the layer count to the model's, so `usize::MAX` means "all" without
    /// reading the layer count first. A model that is too large then fails to load instead of
    /// being split.
    fn all_layers_on_device() -> DeviceMapSetting {
        DeviceMapSetting::Map(DeviceMapMetadata::from_num_device_layers(vec![
            DeviceLayerMapMetadata {
                ordinal: 0,
                layers: usize::MAX,
            },
        ]))
    }

    /// Decide whether the builder must be forced onto the CPU, or fail.
    ///
    /// The pinned builder tries CUDA when it was compiled in, even if a device is missing, so the
    /// device is initialized here before the model is opened.
    fn requires_cpu(&self) -> Result<bool> {
        Self::plan_device(self.device, Self::cuda_enabled(), |ordinal| {
            Device::new_cuda(ordinal).map(|_| ()).map_err(|error| error.to_string())
        })
    }

    /// The `CUDA_VISIBLE_DEVICES` entry to start this process with so that `requested` becomes
    /// logical device 0, or a message saying why none exists.
    ///
    /// `requested` is a LOGICAL index, as the CUDA runtime numbers devices after applying an
    /// inherited `CUDA_VISIBLE_DEVICES`. Putting it into the variable unchanged selects a
    /// different GPU whenever the variable is already set: with `1,0`, logical 1 is physical 0,
    /// and `CUDA_VISIBLE_DEVICES=1` would pick physical 1. So the index is resolved through the
    /// inherited list and the exact entry (an integer, `GPU-<uuid>` or `MIG-<uuid>`) is returned.
    /// With the variable unset the logical and physical indexes agree and `requested` is returned.
    /// Entries are separated by commas; surrounding whitespace is ignored and empty entries are
    /// skipped.
    fn visible_device_entry(requested: usize, inherited: Option<&str>) -> std::result::Result<String, String> {
        let Some(list) = inherited else {
            return Ok(requested.to_string());
        };
        let entries: Vec<&str> = list
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .collect();
        entries.get(requested).map(|entry| (*entry).to_string()).ok_or_else(|| {
            format!(
                "cuda:{requested} is outside the inherited CUDA_VISIBLE_DEVICES={list:?}, which lists {} device(s)",
                entries.len()
            )
        })
    }

    /// Pure device policy: `Ok(true)` means force the CPU, `Ok(false)` means let the builder use
    /// CUDA device 0, `Err` means the request cannot be honoured. `cuda_compiled` and
    /// `probe_cuda` are parameters so every branch is testable without a GPU.
    fn plan_device(
        device: DeviceSpec,
        cuda_compiled: bool,
        probe_cuda: impl FnOnce(usize) -> std::result::Result<(), String>,
    ) -> Result<bool> {
        let inherited = std::env::var("CUDA_VISIBLE_DEVICES").ok();
        Self::plan_device_with_visibility(device, cuda_compiled, probe_cuda, inherited.as_deref())
    }

    /// [`Self::plan_device`] with the inherited `CUDA_VISIBLE_DEVICES` passed in, so the advice for
    /// `cuda:N` (N > 0) is testable.
    fn plan_device_with_visibility(
        device: DeviceSpec,
        cuda_compiled: bool,
        probe_cuda: impl FnOnce(usize) -> std::result::Result<(), String>,
        inherited_visible_devices: Option<&str>,
    ) -> Result<bool> {
        match device {
            DeviceSpec::Cpu => Ok(true),
            DeviceSpec::Auto if cuda_compiled => match probe_cuda(0) {
                Ok(()) => Ok(false),
                Err(error) => {
                    tracing::warn!(%error, "CUDA device unavailable, loading the model on CPU");
                    Ok(true)
                }
            },
            DeviceSpec::Auto => Ok(cfg!(target_os = "windows")),
            DeviceSpec::Cuda(ordinal) => {
                if !cuda_compiled {
                    return Err(Error::Backend(format!(
                        "device cuda:{ordinal} was requested but this build has no CUDA support; \
                         rebuild with the cuda-mistralrs feature or use device \"cpu\" or \"auto\""
                    )));
                }
                probe_cuda(ordinal).map_err(|error| {
                    Error::Backend(format!(
                        "device cuda:{ordinal} was requested but cannot be used: {error}"
                    ))
                })?;
                if ordinal != 0 {
                    // mistral.rs 0.8.1 resolves the device with `resolve_device(force_cpu, None)`
                    // for both the GGUF and the text builder, so `with_device` has no effect and
                    // the model would land on CUDA device 0. Refuse instead of loading on the
                    // wrong GPU.
                    let entry = Self::visible_device_entry(ordinal, inherited_visible_devices).map_err(|reason| {
                        Error::Backend(format!(
                            "device cuda:{ordinal} cannot be selected: mistral.rs 0.8.1 can only load models on CUDA \
                             device 0, and {reason}"
                        ))
                    })?;
                    return Err(Error::Backend(format!(
                        "device cuda:{ordinal} exists, but mistral.rs 0.8.1 can only load models on CUDA \
                         device 0; start this process with CUDA_VISIBLE_DEVICES={entry} and use device \
                         \"cuda:0\""
                    )));
                }
                Ok(false)
            }
        }
    }

    /// Detect if a path is a GGUF model
    fn is_gguf_model(path: &str) -> bool {
        path.to_lowercase().ends_with(".gguf")
    }

    /// Extract model ID and GGUF filename from path
    fn parse_gguf_path(path: &str) -> Result<(String, String)> {
        let path_obj = Path::new(path);

        // If it's a file path, extract directory and filename
        if path_obj.is_file() || path.contains('/') || path.contains('\\') {
            let filename = path_obj
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| Error::Backend("Invalid GGUF filename".to_string()))?
                .to_string();

            // Use parent directory or current directory as model ID
            let model_id = path_obj.parent().and_then(|p| p.to_str()).unwrap_or(".").to_string();

            Ok((model_id, filename))
        } else {
            // Assume it's a HuggingFace repo (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
            // In this case, the user should specify the GGUF filename separately
            Err(Error::Backend(
                "For HuggingFace GGUF models, please provide full path or use load_gguf_hf".to_string(),
            ))
        }
    }

    /// Load a GGUF model from local path or HuggingFace
    async fn load_gguf_model(&mut self, path: &str) -> Result<()> {
        tracing::info!("Loading GGUF model from: {}", path);

        // Try to parse as local path
        let (model_id, gguf_file) = Self::parse_gguf_path(path)?;

        // mistral.rs 0.8.1 resolves the device using its builder's force_cpu setting.
        let mut builder = GgufModelBuilder::new(&model_id, vec![&gguf_file]).with_logging();
        let force_cpu = self.requires_cpu()?;
        if force_cpu {
            builder = builder.with_force_cpu();
        } else if self.device_layers == DeviceLayers::All {
            builder = builder.with_device_mapping(Self::all_layers_on_device());
        }
        tracing::info!(force_cpu, device = %self.device, layers = ?self.device_layers, "Loading GGUF with the selected device policy");
        let model = builder
            .build()
            .await
            .map_err(|e| Error::Backend(format!("Failed to load GGUF model: {}", e)))?;

        self.model = Some(Arc::new(model));
        self.model_path = Some(path.to_string());
        self.is_gguf = true;

        tracing::info!("GGUF model loaded successfully");
        Ok(())
    }

    /// Load a SafeTensors model from HuggingFace or local path
    async fn load_safetensors_model(&mut self, path: &str) -> Result<()> {
        tracing::info!("Loading SafeTensors model from: {}", path);

        let mut builder = TextModelBuilder::new(path.to_string()).with_logging();
        let force_cpu = self.requires_cpu()?;
        if force_cpu {
            builder = builder.with_force_cpu();
        } else if self.device_layers == DeviceLayers::All {
            builder = builder.with_device_mapping(Self::all_layers_on_device());
        }
        tracing::info!(force_cpu, device = %self.device, layers = ?self.device_layers, "Loading text model with the selected device policy");
        let model = builder
            .build()
            .await
            .map_err(|e| Error::Backend(format!("Failed to load SafeTensors model: {}", e)))?;

        self.model = Some(Arc::new(model));
        self.model_path = Some(path.to_string());
        self.is_gguf = false;

        tracing::info!("SafeTensors model loaded successfully");
        Ok(())
    }

    /// Map ChatCompletionResponse to GenerateResponse
    fn map_response(completion: ChatCompletionResponse) -> Result<GenerateResponse> {
        let text = completion.choices[0]
            .message
            .content
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let tokens_generated = completion.usage.completion_tokens;

        let finish_reason = match completion.choices[0].finish_reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        Ok(GenerateResponse {
            text,
            tokens_generated,
            finish_reason,
        })
    }
}

impl Default for MistralRsBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for MistralRsBackend {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Unload any existing model first
        if self.model.is_some() {
            self.unload_model().await?;
        }

        // Detect model type and load appropriately
        if Self::is_gguf_model(model_path) {
            self.load_gguf_model(model_path).await
        } else {
            // Assume SafeTensors/HuggingFace model
            self.load_safetensors_model(model_path).await
        }
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?.clone();

        tracing::debug!("Generating response for prompt (length: {})", request.prompt.len());

        // Build messages - treat prompt as user message
        let mut builder = RequestBuilder::new().add_message(TextMessageRole::User, &request.prompt);

        if let Some(t) = request.temperature {
            builder = builder.set_sampler_temperature(t as f64);
        }
        if let Some(p) = request.top_p {
            builder = builder.set_sampler_topp(p as f64);
        }
        if let Some(max_tokens) = request.max_tokens {
            builder = builder.set_sampler_max_len(max_tokens);
        }
        if !request.stop.is_empty() {
            builder = builder.set_sampler_stop_toks(mistralrs_core::StopTokens::Seqs(request.stop.clone()));
        }

        let response = model
            .send_chat_request(builder)
            .await
            .map_err(|e| Error::Backend(format!("Generation failed: {}", e)))?;

        tracing::debug!("Generated {} tokens", response.usage.completion_tokens);

        Self::map_response(response)
    }

    async fn unload_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            tracing::info!("Unloading model");
            self.model = None;
            self.model_path = None;
            self.is_gguf = false;
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn backend_name(&self) -> &'static str {
        "mistral.rs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_probe(_: usize) -> std::result::Result<(), String> {
        panic!("this device request must not initialize CUDA")
    }

    // Protects explicit CPU mode regardless of the compiled GPU features.
    // Detects changes that allow the automatic device probe in explicit CPU mode.
    // Needs no device or checkpoint. Breadcrumb: MistralRsBackend::plan_device.
    #[test]
    fn explicit_cpu_mode_requires_cpu() {
        assert!(MistralRsBackend::plan_device(DeviceSpec::Cpu, true, no_probe).unwrap());
        assert!(MistralRsBackend::with_config(true).requires_cpu().unwrap());
    }

    // Protects automatic selection for a CUDA artifact with a detachable GPU.
    // Detects retaining CUDA after a failed probe or choosing CPU with a ready device.
    // Needs two synthetic probe outcomes. Breadcrumb: MistralRsBackend::plan_device.
    #[test]
    fn auto_mode_falls_back_only_when_cuda_is_unavailable() {
        assert!(MistralRsBackend::plan_device(DeviceSpec::Auto, true, |_| Err("no device".into())).unwrap());
        assert!(!MistralRsBackend::plan_device(DeviceSpec::Auto, true, |_| Ok(())).unwrap());
    }

    // Protects `cuda:0` as an explicit request, which the binary used to reject.
    // Detects selecting CPU for a usable device. Breadcrumb: MistralRsBackend::plan_device.
    #[test]
    fn explicit_cuda_zero_uses_the_gpu_when_it_initializes() {
        assert!(!MistralRsBackend::plan_device(DeviceSpec::Cuda(0), true, |n| {
            assert_eq!(n, 0);
            Ok(())
        })
        .unwrap());
    }

    // Protects the no-silent-CPU-fallback rule for an explicit device.
    // Detects `cuda:N` degrading to the CPU when the probe fails or CUDA is not built in.
    #[test]
    fn explicit_cuda_never_falls_back_to_cpu() {
        let probe_failed = MistralRsBackend::plan_device(DeviceSpec::Cuda(0), true, |_| Err("driver missing".into()))
            .expect_err("a failed probe must be an error")
            .to_string();
        assert!(
            probe_failed.contains("cuda:0") && probe_failed.contains("driver missing"),
            "{probe_failed}"
        );

        let not_built = MistralRsBackend::plan_device(DeviceSpec::Cuda(0), false, no_probe)
            .expect_err("a build without CUDA must be an error")
            .to_string();
        assert!(not_built.contains("no CUDA support"), "{not_built}");
    }

    // Protects against loading on the wrong GPU. mistral.rs 0.8.1 ignores `with_device` for its
    // GGUF and text builders, so only ordinal 0 can be honoured. Detects `cuda:1` loading on 0.
    #[test]
    fn explicit_cuda_other_than_zero_is_refused_with_the_workaround() {
        let message = MistralRsBackend::plan_device_with_visibility(DeviceSpec::Cuda(1), true, |_| Ok(()), None)
            .expect_err("cuda:1 cannot be honoured")
            .to_string();
        assert!(message.contains("CUDA_VISIBLE_DEVICES=1 "), "{message}");

        // Inherited visibility 1,0: logical cuda:1 is physical GPU 0, so the advice must say 0.
        let remapped =
            MistralRsBackend::plan_device_with_visibility(DeviceSpec::Cuda(1), true, |_| Ok(()), Some("1,0"))
                .expect_err("cuda:1 cannot be honoured")
                .to_string();
        assert!(remapped.contains("CUDA_VISIBLE_DEVICES=0 "), "{remapped}");
        assert!(!remapped.contains("CUDA_VISIBLE_DEVICES=1"), "{remapped}");

        // The inherited list is out of step with what the probe saw: say so instead of guessing.
        let outside = MistralRsBackend::plan_device_with_visibility(DeviceSpec::Cuda(1), true, |_| Ok(()), Some("0"))
            .expect_err("cuda:1 is outside a one-entry list")
            .to_string();
        assert!(outside.contains("outside") && outside.contains("1 device"), "{outside}");

        let missing = MistralRsBackend::plan_device(DeviceSpec::Cuda(1), true, |_| Err("no such device".into()))
            .expect_err("a missing device must be an error")
            .to_string();
        assert!(missing.contains("no such device"), "{missing}");
    }

    // Protects the CUDA_VISIBLE_DEVICES advice for `cuda:N`, N > 0. The index is logical, so with
    // a variable inherited from the environment it must be mapped to the entry at that position,
    // not copied. Detects advising the logical ordinal (picks the wrong physical GPU with `1,0`),
    // and mishandling of UUID, MIG, whitespace and empty entries. Needs no device.
    #[test]
    fn visible_device_entry_resolves_the_logical_index_through_the_inherited_list() {
        let entry = MistralRsBackend::visible_device_entry;
        // Unset: logical and physical indexes agree.
        assert_eq!(entry(1, None).unwrap(), "1");
        assert_eq!(entry(3, None).unwrap(), "3");
        // Identity list.
        assert_eq!(entry(1, Some("0,1")).unwrap(), "1");
        // Reordered list: logical 1 is physical 0, logical 0 is physical 1.
        assert_eq!(entry(1, Some("1,0")).unwrap(), "0");
        assert_eq!(entry(0, Some("1,0")).unwrap(), "1");
        // A subset: logical 1 is the second visible device.
        assert_eq!(entry(1, Some("2,3")).unwrap(), "3");
        // UUID and MIG entries are returned verbatim.
        assert_eq!(entry(1, Some("GPU-aaaa-1111,GPU-bbbb-2222")).unwrap(), "GPU-bbbb-2222");
        assert_eq!(entry(1, Some("MIG-aaaa-1111,MIG-bbbb-2222")).unwrap(), "MIG-bbbb-2222");
        // Whitespace is trimmed and empty entries are skipped.
        assert_eq!(entry(1, Some(" 2 , 3 ")).unwrap(), "3");
        assert_eq!(entry(1, Some("2,,3,")).unwrap(), "3");
        assert_eq!(entry(1, Some(",, 4 ,5")).unwrap(), "5");
    }

    #[test]
    fn visible_device_entry_reports_an_index_outside_the_inherited_list() {
        let entry = MistralRsBackend::visible_device_entry;
        for list in ["0", "0,1", "", " , ", "GPU-aaaa-1111"] {
            let error = entry(2, Some(list)).expect_err(list);
            assert!(
                error.contains("cuda:2") && error.contains("outside"),
                "{list:?}: {error}"
            );
        }
        assert!(entry(1, Some("0")).unwrap_err().contains("1 device(s)"));
        assert!(entry(0, Some("")).unwrap_err().contains("0 device(s)"));
    }

    // Protects `device_layers = "all"`: a single-entry map that covers every layer. Detects a
    // count that does not clamp to the model (mistral.rs bails when device + host layers differ
    // from the model's), and a mapping that leaves layers on the host. Uses the real mapper on
    // the CPU device, so it needs no GPU. Breadcrumb: MistralRsBackend::all_layers_on_device.
    #[test]
    fn all_layers_mapping_clamps_to_any_model_depth() {
        let setting = MistralRsBackend::all_layers_on_device();
        match &setting {
            DeviceMapSetting::Map(metadata) => {
                let layers = metadata.device_layers().expect("test: the map must name device layers");
                assert_eq!(layers.len(), 1);
                assert_eq!(layers[0].ordinal, 0);
                assert_eq!(layers[0].layers, usize::MAX);
                assert_eq!(metadata.host_layers(), None);
            }
            other => panic!("test: expected a manual map, got {other:?}"),
        }
        for depth in [1usize, 22, 28, 80] {
            setting
                .into_mapper(depth, &Device::Cpu, None, &[Device::Cpu])
                .unwrap_or_else(|error| panic!("test: {depth} layers must map: {error}"));
        }
    }

    // Protects the builder API: the knob is stored, and the default stays automatic.
    #[test]
    fn device_layers_defaults_to_auto_and_can_be_set() {
        assert_eq!(MistralRsBackend::new().device_layers, DeviceLayers::Auto);
        let backend = MistralRsBackend::new().with_device_layers(DeviceLayers::All);
        assert_eq!(backend.device_layers, DeviceLayers::All);
    }

    // Protects default loading with a Windows CPU-only artifact.
    // Detects CPU-only deployments entering the CUDA builder policy.
    // Needs Windows without the backend CUDA feature. Breadcrumb: MistralRsBackend::requires_cpu.
    #[cfg(all(target_os = "windows", not(feature = "cuda-mistralrs")))]
    #[test]
    fn windows_cpu_artifact_defaults_to_cpu_mode() {
        assert!(MistralRsBackend::new().requires_cpu().unwrap());
    }

    // Protects the build-without-CUDA path end to end through the real probe closure.
    // Detects `cuda:0` loading anyway on a CPU-only artifact.
    #[cfg(not(any(feature = "cuda", feature = "cuda-mistralrs")))]
    #[test]
    fn explicit_cuda_on_a_cpu_only_build_fails_to_load() {
        let error = MistralRsBackend::with_device(DeviceSpec::Cuda(0))
            .requires_cpu()
            .expect_err("a CPU-only build cannot honour cuda:0");
        assert!(error.to_string().contains("no CUDA support"), "{error}");
    }

    #[test]
    fn test_is_gguf_model() {
        assert!(MistralRsBackend::is_gguf_model("model.gguf"));
        assert!(MistralRsBackend::is_gguf_model("path/to/model.GGUF"));
        assert!(!MistralRsBackend::is_gguf_model("model.safetensors"));
        assert!(!MistralRsBackend::is_gguf_model("microsoft/Phi-3.5-mini"));
    }

    #[test]
    fn test_parse_gguf_path() {
        let result = MistralRsBackend::parse_gguf_path("/path/to/model.gguf");
        assert!(result.is_ok());
        let (model_id, filename) = result.expect("test: parse_gguf_path must succeed for a valid absolute GGUF path");
        assert_eq!(model_id, "/path/to");
        assert_eq!(filename, "model.gguf");
    }

    #[tokio::test]
    async fn test_backend_creation() {
        let backend = MistralRsBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.backend_name(), "mistral.rs");
    }
}
