use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use serde::{Deserialize, Serialize};

/// Configuration for LoRA (Low-Rank Adaptation) layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank decomposition
    pub r: usize,
    /// Scaling factor (typically 2 * r)
    pub alpha: f64,
    /// Dropout probability (not yet implemented)
    pub dropout: f64,
    /// Names of modules to apply LoRA to (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

/// LoRA-adapted linear layer
///
/// Implements the transformation: y = Wx + (BA)x * scaling
/// where W is the frozen base weight, and B, A are trainable low-rank matrices
#[derive(Debug)]
pub struct LoraLinear {
    /// Frozen base linear layer
    base: Linear,
    /// Low-rank matrix A (r x in_features), initialized with Kaiming uniform
    lora_a: Tensor,
    /// Low-rank matrix B (out_features x r), initialized with zeros
    lora_b: Tensor,
    /// Scaling factor: alpha / r
    scaling: f64,
}

impl LoraLinear {
    /// Creates a new LoRA linear layer
    ///
    /// # Arguments
    /// * `in_features` - Input dimension
    /// * `out_features` - Output dimension
    /// * `config` - LoRA configuration
    /// * `device` - Device to place tensors on
    ///
    /// # Initialization
    /// - Base weights: Identity or random (frozen)
    /// - LoRA A: Kaiming uniform (trainable)
    /// - LoRA B: Zeros (trainable)
    pub fn new(in_features: usize, out_features: usize, config: &LoraConfig, device: &Device) -> Result<Self> {
        let r = config.r;

        // Create base linear layer with identity initialization
        // In practice, this would be loaded from a pretrained model
        let base_weight = Tensor::eye(out_features.min(in_features), DType::F32, device)?;
        let base_weight = if out_features > in_features {
            let padding = Tensor::zeros((out_features - in_features, in_features), DType::F32, device)?;
            Tensor::cat(&[&base_weight, &padding], 0)?
        } else if in_features > out_features {
            base_weight.narrow(1, 0, out_features)?
        } else {
            base_weight
        };
        let base = Linear::new(base_weight, None);

        // Initialize LoRA A with Kaiming uniform
        // Kaiming uniform: U(-bound, bound) where bound = sqrt(6 / fan_in)
        let bound = (6.0 / in_features as f64).sqrt();
        let lora_a = Tensor::rand(-bound as f32, bound as f32, (r, in_features), device)?;

        // Initialize LoRA B with zeros
        let lora_b = Tensor::zeros((out_features, r), DType::F32, device)?;

        // Scaling factor
        let scaling = config.alpha / r as f64;

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling,
        })
    }

    /// Returns references to the trainable LoRA parameters
    ///
    /// # Returns
    /// A tuple of (&lora_a, &lora_b)
    pub fn lora_params(&self) -> (&Tensor, &Tensor) {
        (&self.lora_a, &self.lora_b)
    }

    /// Merges the LoRA weights into the base weights
    ///
    /// After merging: W' = W + BA * scaling
    pub fn merge_weights(&mut self) -> Result<()> {
        // Compute delta = B @ A * scaling
        let delta = self.lora_b.matmul(&self.lora_a)?;
        let delta = delta.affine(self.scaling, 0.0)?;

        // Add to base weights
        let new_weight = self.base.weight().add(&delta)?;
        self.base = Linear::new(new_weight, self.base.bias().cloned());

        // Zero out LoRA weights after merge
        let device = self.lora_a.device().clone();
        let shape_a = self.lora_a.shape().clone();
        let shape_b = self.lora_b.shape().clone();
        self.lora_a = Tensor::zeros(shape_a, DType::F32, &device)?;
        self.lora_b = Tensor::zeros(shape_b, DType::F32, &device)?;

        Ok(())
    }
}

impl Module for LoraLinear {
    /// Forward pass: y = Wx + (x @ A^T @ B^T) * scaling
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Base forward pass
        let base_out = self.base.forward(x)?;

        // LoRA forward pass
        // x: (batch, seq_len, in_features)
        // lora_a: (r, in_features) -> need A^T: (in_features, r)
        // lora_b: (out_features, r) -> need B^T: (r, out_features)

        // Flatten batch and sequence dimensions for matmul
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let x_flat = x.reshape((batch_size * seq_len, hidden_dim))?;

        // Compute LoRA path: x @ A^T @ B^T
        let lora_out = x_flat.matmul(&self.lora_a.t()?)?; // (batch*seq, r)
        let lora_out = lora_out.matmul(&self.lora_b.t()?)?; // (batch*seq, out_features)

        // Reshape back to (batch, seq_len, out_features)
        let lora_out = lora_out.reshape((batch_size, seq_len, ()))?;
        let lora_out = lora_out.affine(self.scaling, 0.0)?;

        // Combine base and LoRA outputs
        base_out.add(&lora_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoraConfig::default();
        assert_eq!(config.r, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.dropout, 0.0);
        assert_eq!(config.target_modules.len(), 2);
    }

    #[test]
    fn test_lora_linear_creation() {
        let device = Device::Cpu;
        let config = LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string()],
        };

        let lora = LoraLinear::new(768, 768, &config, &device).expect("LoraLinear::new failed on CPU");
        let (lora_a, lora_b) = lora.lora_params();

        assert_eq!(lora_a.dims(), &[8, 768]);
        assert_eq!(lora_b.dims(), &[768, 8]);
    }

    #[test]
    fn test_lora_linear_params() {
        let device = Device::Cpu;
        let config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec!["test".to_string()],
        };

        let lora = LoraLinear::new(512, 256, &config, &device).expect("LoraLinear::new failed on CPU");
        let (lora_a, lora_b) = lora.lora_params();

        assert_eq!(lora_a.dims(), &[4, 512]);
        assert_eq!(lora_b.dims(), &[256, 4]);

        // Check that lora_b is initialized to zeros
        let sum = lora_b
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .expect("to_scalar::<f32> failed on lora_b sum");
        assert_eq!(sum, 0.0);
    }
}
