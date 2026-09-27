//! Shader compilation
//! f32 sgemm
//! f32 sgemm_transb

use super::*;

#[test]
fn all_shaders_compile() {
    let src = larql_compute_metal::shaders::all_shaders();
    assert!(src.len() > 1000, "Shader source too short");

    let device = metal::Device::system_default().expect("No Metal device");
    let opts = metal::CompileOptions::new();
    device
        .new_library_with_source(&src, &opts)
        .expect("Shader compilation failed");
}

#[test]
fn all_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        // f32 matmul
        "sgemm",
        "sgemm_transb",
        // Q4_0 matvec
        "q4_matvec_v4",
        "q4_vecmat",
        "q4_f32_matvec",
        // Q4_K / Q4_KF matvec
        "q4k_matvec",
        "q4k_qkv_proj",
        "q4k_proj",
        "q4kf_qkv_proj",
        "q4kf_proj",
        // Q4_K fused FFN
        "q4k_ffn_gate_up",
        "q4kf_ffn_gate_up",
        "q4k_geglu_silu_down",
        "q4k_geglu_gelu_tanh_down",
        // Activations
        "geglu_silu",
        "geglu_gelu_tanh",
        "silu",
        "gelu_tanh",
        // Quantize / norms / residuals
        "quantize_q8",
        "rms_norm_q8",
        "residual_norm",
        "residual_norm_q8",
        "residual_add",
        "layer_norm",
        "layer_norm_no_bias",
        "v_norm",
        "v_norm_batched",
        "scale_vector",
        // Attention / RoPE
        "causal_attention",
        "kv_attention",
        "kv_cache_append",
        "rope_apply",
        "rope_at_pos",
        "rope_at_pos_batched",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

#[test]
fn sgemm_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(2560, 2560, 43);

    let cpu_result = a.dot(&b);
    let metal_result = metal.matmul(a.view(), b.view());

    let diff = max_diff(
        cpu_result.as_slice().unwrap(),
        metal_result.as_slice().unwrap(),
    );
    assert!(diff < 0.1, "sgemm max diff {diff} exceeds 0.1");
}

#[test]
fn sgemm_transb_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(10240, 2560, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(
        cpu_result.as_slice().unwrap(),
        metal_result.as_slice().unwrap(),
    );
    assert!(diff < 0.1, "sgemm_transb max diff {diff} exceeds 0.1");
}

#[test]
fn sgemm_transb_small_matrix() {
    let metal = get_metal();
    let a = synth(1, 256, 42);
    let b = synth(512, 256, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(
        cpu_result.as_slice().unwrap(),
        metal_result.as_slice().unwrap(),
    );
    assert!(diff < 0.01, "small sgemm_transb max diff {diff}");
}
