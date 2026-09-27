//! RoPE shader

use super::*;

#[test]
fn rope_apply_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rope_apply", None).unwrap())
        .unwrap();

    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 10000.0f32;

    // Create test data
    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: apply RoPE manually
    let half = dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half {
            let freq = 1.0 / base.powf(2.0 * d as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half] = re * sin_a + im * cos_a;
        }
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    // Buffer 2 is the frequency table; amplitude at 4. See §4.10.
    let rope_plan =
        larql_compute::attention::rope::RopeFreqPlan::unscaled(dim as usize, 0, base as f64);
    let inv_freq = rope_plan.inv_freq_f32();
    enc.set_bytes(
        2,
        std::mem::size_of_val(&inv_freq[..]) as u64,
        inv_freq.as_ptr() as *const std::ffi::c_void,
    );
    let amplitude = rope_plan.amplitude as f32;
    enc.set_bytes(4, 4, &amplitude as *const f32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full dim rotation
    enc.set_bytes(
        3,
        4,
        &rotary_dim_val as *const u32 as *const std::ffi::c_void,
    );
    enc.dispatch_threads(
        metal::MTLSize::new(half as u64, seq_len as u64, 1),
        metal::MTLSize::new(half as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> =
        unsafe { std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "RoPE max diff {diff} exceeds 1e-4");
}

#[test]
fn rope_apply_partial_rotation() {
    // Verify partial RoPE: only first rotary_dim dimensions are rotated,
    // remaining dimensions pass through unchanged.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute_metal::shaders::all_shaders();
    let lib = device
        .new_library_with_source(&src, &metal::CompileOptions::new())
        .unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&lib.get_function("rope_apply", None).unwrap())
        .unwrap();

    let bufs = larql_compute_metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 1000000.0f32;
    let rotary_dim = 16u32; // 25% of dim (Gemma 4 style)

    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: partial RoPE (rotate first rotary_dim dims, rest unchanged)
    let half_rotary = rotary_dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half_rotary {
            let freq = 1.0 / base.powf(2.0 * d as f32 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half_rotary];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half_rotary] = re * sin_a + im * cos_a;
        }
        // Dimensions [rotary_dim..dim] must remain unchanged
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    // Buffer 2 is the frequency table; amplitude at 4. See §4.10.
    let rope_plan = larql_compute::attention::rope::RopeFreqPlan::unscaled(
        dim as usize,
        rotary_dim as usize,
        base as f64,
    );
    let inv_freq = rope_plan.inv_freq_f32();
    enc.set_bytes(
        2,
        std::mem::size_of_val(&inv_freq[..]) as u64,
        inv_freq.as_ptr() as *const std::ffi::c_void,
    );
    let amplitude = rope_plan.amplitude as f32;
    enc.set_bytes(4, 4, &amplitude as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &rotary_dim as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half_rotary as u64, seq_len as u64, 1),
        metal::MTLSize::new(half_rotary as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> =
        unsafe { std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec() };

    // Rotated dims should match CPU
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "Partial RoPE max diff {diff} exceeds 1e-4");

    // Non-rotated dims (rotary_dim..dim) should be unchanged
    for pos in 0..seq_len as usize {
        for d in rotary_dim as usize..dim as usize {
            let idx = pos * dim as usize + d;
            assert_eq!(
                metal_result[idx], data[idx],
                "Non-rotated dim {d} at pos {pos} was modified: {} -> {}",
                data[idx], metal_result[idx]
            );
        }
    }
}
