//! Isolated raw-kernel timings (34 layers): QKV projection, FFN, O
//! projection, and the element-wise dispatch floor, plus the component
//! breakdown they feed.

use std::time::Instant;

use larql_compute_metal::MetalBackend;

use super::layers::{Layer, HIDDEN, INTER, KV_DIM, Q_DIM};

/// Layers per timed command buffer — Gemma 3 4B's depth.
pub const LAYERS: usize = 34;
/// Untimed iterations before each timed block.
const WARMUP: usize = 5;
/// `residual_add` dispatches per command buffer in the dispatch-floor probe.
const FLOOR_DISPATCHES: usize = 340;
/// Threadgroup width for the element-wise kernels here.
const ELEMWISE_TG: u64 = 256;

fn ms_per_iter(t0: Instant, n: usize) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0 / n as f64
}

/// One fused Q4_K QKV projection per layer, zero surrounding overhead.
pub fn raw_qkv_ms(m: &MetalBackend, layer: &Layer, x: &[f32], n: usize) -> f64 {
    let buf_wq = m.bufs().get_bytes(&layer.wq);
    let buf_wk = m.bufs().get_bytes(&layer.wk);
    let buf_wv = m.bufs().get_bytes(&layer.wv);
    let buf_x = m.bufs().transient_from_f32(x);
    use larql_compute_metal::shaders::q4k_qkv_proj as sh;
    let total = (Q_DIM + KV_DIM + KV_DIM) as u32;
    let num_tgs = (total as u64).div_ceil(sh::ROWS_PER_TG);
    let run = || {
        let cmd = m.queue().new_command_buffer();
        for _ in 0..LAYERS {
            let qo = m.bufs().output((Q_DIM * 4) as u64);
            let ko = m.bufs().output((KV_DIM * 4) as u64);
            let vo = m.bufs().output((KV_DIM * 4) as u64);
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&m.attention.q4k_qkv_proj_pipeline.state);
            enc.set_buffer(0, Some(&buf_wq), 0);
            enc.set_buffer(1, Some(&buf_wk), 0);
            enc.set_buffer(2, Some(&buf_wv), 0);
            enc.set_buffer(3, Some(&buf_x), 0);
            enc.set_buffer(4, Some(&qo), 0);
            enc.set_buffer(5, Some(&ko), 0);
            enc.set_buffer(6, Some(&vo), 0);
            let (q, k, v, h) = (Q_DIM as u32, KV_DIM as u32, KV_DIM as u32, HIDDEN as u32);
            enc.set_bytes(7, 4, &q as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &h as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(num_tgs, 1, 1),
                metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    };
    for _ in 0..WARMUP {
        run();
    }
    let t0 = Instant::now();
    for _ in 0..n {
        run();
    }
    ms_per_iter(t0, n)
}

/// Q4_KF fused gate+up, GEGLU, then down — one set per layer.
pub fn ffn_ms(m: &MetalBackend, layer: &Layer, n: usize) -> f64 {
    use larql_compute_metal::shaders::q4kf_ffn_gate_up as q4kf_gu;
    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let ffn_input = m.bufs().transient_from_f32(&vec![0.1f32; HIDDEN]);
    let n_tgs_gu = (INTER as u64).div_ceil(q4kf_gu::ROWS_PER_TG);
    let n_tgs_down = (HIDDEN as u64).div_ceil(q4kf::ROWS_PER_TG);
    let run = || {
        let cmd = m.queue().new_command_buffer();
        for _ in 0..LAYERS {
            let go = m.bufs().output((INTER * 4) as u64);
            let uo = m.bufs().output((INTER * 4) as u64);
            let ao = m.bufs().output((INTER * 4) as u64);
            let d_out = m.bufs().output((HIDDEN * 4) as u64);
            let enc = cmd.new_compute_command_encoder();
            // fused gate+up
            enc.set_compute_pipeline_state(&m.ffn.q4kf_ffn_gate_up_pipeline.state);
            enc.set_buffer(0, Some(&m.bufs().get_bytes(&layer.g)), 0);
            enc.set_buffer(1, Some(&m.bufs().get_bytes(&layer.u)), 0);
            enc.set_buffer(2, Some(&ffn_input), 0);
            enc.set_buffer(3, Some(&go), 0);
            enc.set_buffer(4, Some(&uo), 0);
            let iv = INTER as u32;
            let hv = HIDDEN as u32;
            enc.set_bytes(5, 4, &iv as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &hv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(n_tgs_gu * 2, 1, 1),
                metal::MTLSize::new(q4kf_gu::THREADS_PER_TG, 1, 1),
            );
            // GEGLU
            enc.set_compute_pipeline_state(&m.ffn.geglu_pipeline);
            enc.set_buffer(0, Some(&go), 0);
            enc.set_buffer(1, Some(&uo), 0);
            enc.set_buffer(2, Some(&ao), 0);
            enc.set_bytes(3, 4, &iv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(
                metal::MTLSize::new(INTER as u64, 1, 1),
                metal::MTLSize::new(ELEMWISE_TG, 1, 1),
            );
            // down
            enc.set_compute_pipeline_state(&m.attention.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(&m.bufs().get_bytes(&layer.d)), 0);
            enc.set_buffer(1, Some(&ao), 0);
            enc.set_buffer(2, Some(&d_out), 0);
            enc.set_bytes(3, 4, &hv as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &iv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(n_tgs_down, 1, 1),
                metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    };
    for _ in 0..WARMUP {
        run();
    }
    let t0 = Instant::now();
    for _ in 0..n {
        run();
    }
    ms_per_iter(t0, n)
}

/// Q4_KF O projection — one per layer.
pub fn o_proj_ms(m: &MetalBackend, layer: &Layer, n: usize) -> f64 {
    use larql_compute_metal::shaders::q4kf_qkv_proj as q4kf;
    let o_input = m.bufs().output((Q_DIM * 4) as u64);
    let o_output = m.bufs().output((HIDDEN * 4) as u64);
    let n_tgs_o = (HIDDEN as u64).div_ceil(q4kf::ROWS_PER_TG);
    let run = || {
        let cmd = m.queue().new_command_buffer();
        for _ in 0..LAYERS {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&m.attention.q4kf_proj_pipeline.state);
            enc.set_buffer(0, Some(&m.bufs().get_bytes(&layer.wo)), 0);
            enc.set_buffer(1, Some(&o_input), 0);
            enc.set_buffer(2, Some(&o_output), 0);
            let nv = HIDDEN as u32;
            let kv = Q_DIM as u32;
            enc.set_bytes(3, 4, &nv as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &kv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(n_tgs_o, 1, 1),
                metal::MTLSize::new(q4kf::THREADS_PER_TG, 1, 1),
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
    };
    for _ in 0..WARMUP {
        run();
    }
    let t0 = Instant::now();
    for _ in 0..n {
        run();
    }
    ms_per_iter(t0, n)
}

/// Raw element-wise dispatch floor: `FLOOR_DISPATCHES` `residual_add`
/// dispatches in one encoder.
pub fn dispatch_floor_ms(m: &MetalBackend, n: usize) -> f64 {
    let a_buf = m.bufs().output((HIDDEN * 4) as u64);
    let b_buf = m.bufs().output((HIDDEN * 4) as u64);
    let c_buf = m.bufs().output((HIDDEN * 4) as u64);
    let hv = HIDDEN as u32;
    let run = || {
        let cmd = m.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        for _ in 0..FLOOR_DISPATCHES {
            enc.set_compute_pipeline_state(&m.norms.residual_add_pipeline);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&c_buf), 0);
            enc.set_bytes(3, 4, &hv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_threads(
                metal::MTLSize::new(HIDDEN as u64, 1, 1),
                metal::MTLSize::new(ELEMWISE_TG, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    };
    for _ in 0..WARMUP {
        run();
    }
    let t0 = Instant::now();
    for _ in 0..n {
        run();
    }
    ms_per_iter(t0, n)
}

/// Time the isolated FFN, O projection and dispatch floor, then print the
/// 34-layer component breakdown against the full Q4_K decode.
pub fn print_component_breakdown(
    m: &MetalBackend,
    layer: &Layer,
    q4k_34_ms: f64,
    raw_34_ms: f64,
    n: usize,
) {
    let ffn_ms = ffn_ms(m, layer, n);
    let o_proj_ms = o_proj_ms(m, layer, n);
    let attn_ms = q4k_34_ms - ffn_ms - raw_34_ms;
    let dispatch_floor_ms = dispatch_floor_ms(m, n);

    let layers = LAYERS as f64;
    let kv_norms_ms = attn_ms - o_proj_ms;
    println!();
    println!("  Component breakdown (34 layers):");
    println!(
        "    FFN (gate+up+GEGLU+down):    {ffn_ms:.1}ms ({:.1}%) = {:.3}ms/layer",
        ffn_ms / q4k_34_ms * 100.0,
        ffn_ms / layers
    );
    println!(
        "    QKV projection:              {raw_34_ms:.1}ms ({:.1}%) = {:.3}ms/layer",
        raw_34_ms / q4k_34_ms * 100.0,
        raw_34_ms / layers
    );
    println!(
        "    O projection:                {o_proj_ms:.1}ms ({:.1}%) = {:.3}ms/layer",
        o_proj_ms / q4k_34_ms * 100.0,
        o_proj_ms / layers
    );
    println!(
        "    KV attend + norms + residual: {kv_norms_ms:.1}ms ({:.1}%) = {:.3}ms/layer",
        kv_norms_ms / q4k_34_ms * 100.0,
        kv_norms_ms / layers
    );
    println!(
        "    Dispatch floor (340×add):     {dispatch_floor_ms:.1}ms = {:.3}ms/dispatch",
        dispatch_floor_ms / FLOOR_DISPATCHES as f64
    );
}
