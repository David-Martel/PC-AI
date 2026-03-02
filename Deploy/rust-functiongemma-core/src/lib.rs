pub mod chat_template;
pub mod config;
pub mod error;
pub mod gpu;
pub mod lora_utils;
pub mod model;
pub mod prompt;
pub mod safetensors_utils;

pub use config::{PcaiConfig, RuntimeConfig};
pub use error::PcaiError;
pub use gpu::{
    auto_cuda_index, default_dtype, normalize_device_label, parse_cuda_index, parse_ggml_dtype,
    resolve_device_with_index, DeviceSelectionParams, GpuInfo,
};
pub use lora_utils::LoraInfo;
pub use model::{Config, KvCacheQuant, LoraSettings, Model};
pub use prompt::{is_degenerate_output, trim_input_ids};
pub use safetensors_utils::{
    collect_model_safetensors, custom_load, custom_load_verbose, detect_safetensors_prefix, detect_tie_embeddings,
    open_mmaped_safetensors,
};
