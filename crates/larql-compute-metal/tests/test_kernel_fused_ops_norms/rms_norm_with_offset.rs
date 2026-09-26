//! rms_norm with offset

use super::*;

#[test]
fn rms_norm_matches_cpu() {
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

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 2/3 style (Gemma 4 uses 0.0)

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x
        .iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    // Metal
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
    // Single threadgroup dispatch for cooperative SIMD reduction.
    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm max diff {diff}");
}

#[test]
fn rms_norm_zero_offset() {
    // Standard RMS norm (Llama-style, offset=0)
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

    let len = 32usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.2 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0f32; len];
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().map(|xi| xi * rms).collect();

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
    enc.dispatch_thread_groups(
        metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(len as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}
