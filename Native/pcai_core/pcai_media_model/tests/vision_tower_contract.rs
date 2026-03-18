use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use pcai_media_model::{
    config::JanusConfig,
    vision::{JanusVisionConfig, JanusVisionTower},
};

fn cpu_tower(cfg: &JanusVisionConfig) -> JanusVisionTower {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    JanusVisionTower::new(vb, cfg).expect("JanusVisionTower should build on CPU with an empty VarMap")
}

#[test]
fn janus_vision_config_matches_patch_grid_contract() {
    let one_b = JanusVisionConfig::from_janus_config(&JanusConfig::janus_pro_1b());
    let seven_b = JanusVisionConfig::from_janus_config(&JanusConfig::janus_pro_7b());

    for cfg in [&one_b, &seven_b] {
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.num_image_tokens(), 576);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 4096);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_hidden_layers, 24);
        assert_eq!(cfg.layer_norm_eps, 1e-6);
    }

    assert_eq!(one_b.num_image_tokens(), seven_b.num_image_tokens());
    assert_eq!(one_b.hidden_size, seven_b.hidden_size);
    assert_eq!(one_b.intermediate_size, seven_b.intermediate_size);
}

#[test]
fn vision_tower_preserves_patch_token_shape() {
    let cfg = JanusVisionConfig::from_janus_config(&JanusConfig::janus_pro_1b());
    let tower = cpu_tower(&cfg);
    let device = Device::Cpu;

    let image = Tensor::zeros((2_usize, 3_usize, cfg.image_size, cfg.image_size), DType::F32, &device)
        .expect("image tensor should allocate");

    let features = tower.forward(&image).expect("vision tower forward should succeed");

    assert_eq!(
        features.dims(),
        &[2, cfg.num_patches(), cfg.hidden_size],
        "vision tower should emit a patch-token grid with the expected width"
    );
    assert_eq!(tower.config().num_patches(), cfg.num_patches());
}
