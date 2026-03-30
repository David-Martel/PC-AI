🎯 **What:** The testing gap addressed
The `open_safetensors` function in `Native/pcai_core/pcai_media/src/hub.rs` lacked adequate testing for actual file loading scenarios.

📊 **Coverage:** What scenarios are now tested
- Valid single `.safetensors` files containing dummy tensors.
- Valid multiple sharded `.safetensors` files spanning multiple parts containing dummy tensors.
- Invalid or corrupted `.safetensors` files which ensure error propagation.

✨ **Result:** The improvement in test coverage
`open_safetensors` is now comprehensively covered against happy paths (successful parsing) and common edge cases.
