//! GEGLU
//! Cross-validation: all kernels listed

use super::*;

#[test]
fn geglu_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("geglu_silu", None).unwrap())
        .unwrap();

    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let n = 256;
    let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 12.8).collect();
    let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);

    // Metal
    let buf_g = bufs.transient_from_f32(&gate);
    let buf_u = bufs.transient_from_f32(&up);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_g), 0);
    enc.set_buffer(1, Some(&buf_u), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(n as u64, 1, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "GEGLU CPU vs Metal diff {diff}");
}

#[test]
fn all_new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();

    let names = [
        "sgemm",
        "sgemm_transb",
        "q4_matvec_v4",
        "q4_vecmat",
        "q4_f32_matvec",
        "q4_sparse_matvec",
        "q8_matvec",
        "geglu_silu",
        "quantize_q8",
        "residual_copy",
        "residual_add",
        "rms_norm",
        "causal_attention",
        "kv_attention",
        "kv_cache_append",
        "rope_apply",
        "fused_attention",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}
