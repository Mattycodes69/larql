//! cooperative SIMD norm (large vector, multi-simdgroup)

use super::*;

#[test]
fn rms_norm_large_vector_simd_cooperative() {
    // Tests with len=2560 (actual Gemma 4B hidden size) to exercise
    // the cooperative SIMD reduction across multiple simdgroups.
    // With TG=256: 8 simdgroups, each sums a 2560/256=10-element stripe.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rms_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let x: Vec<f32> = (0..len).map(|i| (i as f32 * 0.0037).sin() * 2.0).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch — cooperative SIMD reduction needs all threads in one TG.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute_metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(
        diff < 1e-4,
        "rms_norm(len=2560) SIMD cooperative max diff {diff}"
    );
}

#[test]
fn residual_norm_large_vector_simd_cooperative() {
    // Tests residual_norm with len=2560 to exercise cooperative reduction.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("residual_norm", None).unwrap())
        .unwrap();
    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.003).cos() * 1.5).collect();
    let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.007).sin() * 0.5).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.9 + (i as f32 * 0.00005)).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: h = a + b, then rms_norm(h)
    let h: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai + bi).collect();
    let sum_sq: f32 = h.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = h
        .iter()
        .zip(weight.iter())
        .map(|(hi, wi)| hi * (wi + offset) * rms)
        .collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute_metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(
        diff < 1e-4,
        "residual_norm(len=2560) SIMD cooperative max diff {diff}"
    );
}
