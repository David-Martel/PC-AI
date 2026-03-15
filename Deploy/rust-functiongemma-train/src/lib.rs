pub use rust_functiongemma_core::model;
pub mod checkpoint;
pub mod data_gen;
pub mod dataset;
pub mod early_stopping;
pub mod eval;
pub mod lora;
pub mod router_dataset;
pub mod scheduler;
pub mod schema_utils;
pub mod trainer;

pub use rust_functiongemma_core::model::{Config, LoraSettings, Model};

#[cfg(test)]
mod tests {
    use candle_core::Device;

    #[test]
    fn test_cuda_device_availability() {
        // Try to create a CUDA device
        match Device::new_cuda(0) {
            Ok(device) => {
                println!("CUDA device 0 available: {:?}", device);
                assert!(device.is_cuda(), "Device should be CUDA");
            }
            Err(e) => {
                println!("CUDA not available (falling back to CPU): {}", e);
                // This is not a failure - CUDA may not be available in CI
            }
        }
    }

    #[test]
    fn test_cuda_tensor_operations() {
        use candle_core::Tensor;

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        let is_cuda = device.is_cuda();

        // Create a tensor on the device
        let tensor = Tensor::zeros((2, 3), candle_core::DType::F32, &device)
            .expect("Tensor::zeros on CPU/CUDA should not fail");
        assert_eq!(tensor.dims(), &[2, 3]);

        if is_cuda {
            println!("Tensor created on CUDA device successfully");
        } else {
            println!("Tensor created on CPU device (CUDA not available)");
        }
    }
}
