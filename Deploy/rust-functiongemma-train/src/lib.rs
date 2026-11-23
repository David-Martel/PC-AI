pub use rust_functiongemma_core::model;
pub mod checkpoint;
pub mod data_gen;
pub mod dataset;
pub mod early_stopping;
pub mod eval;
pub mod router_dataset;
pub mod scheduler;
pub mod schema_utils;
pub mod trainer;

pub use rust_functiongemma_core::model::{Config, LoraSettings, Model};
pub use trainer::{TrainingRunMetrics, TrainingStepMetrics};

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    // Protects: the CPU trainer's tensor backend. Detects: CPU tensor kernel
    // failures. Needs: no hardware or model. Breadcrumb: trainer::Trainer.
    #[test]
    fn cpu_tensor_reduction_computes_expected_sum() -> anyhow::Result<()> {
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        assert_eq!(tensor.sum_all()?.to_scalar::<f32>()?, 6.0);
        Ok(())
    }

    // Protects: explicit CUDA trainer qualification. Detects: device/driver
    // initialization or CUDA kernel failure. Needs: CUDA device 0 and build.
    // Breadcrumb: trainer::Trainer, core gpu::resolve_device_with_index.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA-capable device and driver"]
    fn cuda_tensor_reduction_computes_expected_sum() -> anyhow::Result<()> {
        let device = Device::new_cuda(0)?;
        assert!(device.is_cuda());
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
        assert_eq!(tensor.sum_all()?.to_scalar::<f32>()?, 6.0);
        Ok(())
    }
}
