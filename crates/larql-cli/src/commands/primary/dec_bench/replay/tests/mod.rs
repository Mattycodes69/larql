use super::*;
use larql_compute::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k;
use larql_inference::ffn::moe_remote::multi_layer_wire::{
    MULTI_LAYER_BATCH_PATH, MULTI_LAYER_BATCH_Q8K_PATH,
};
use larql_inference::ffn::moe_remote::{
    MULTI_LAYER_BATCH_CONTENT_TYPE, MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE,
};
use larql_inference::ffn::remote::{WireFormat, WALK_FFN_PATH, WALK_FFN_Q8K_PATH};

fn sample(layer: usize, client_ms: f64, server_ms: Option<f64>) -> RequestSample {
    // serve_us mirrors what send_one does: embedded server_ms × 1000
    // when present; transmit derived through derive_transmit_us.
    let serve_us = server_ms.map(|ms| ms * 1000.0);
    let (transmit_us, transmit_clamped) = derive_transmit_us(client_ms, serve_us);
    RequestSample {
        layer,
        client_ms,
        server_ms,
        serve_us,
        encode_us: 40.0,
        client_decode_us: 25.0,
        transmit_us,
        transmit_clamped,
        bytes_sent: 100,
        bytes_recv: 60,
        served_wire_in: "f32".into(),
        served_wire_out: "f16".into(),
        any_nonzero: true,
    }
}

use super::super::capture_format::{CapturePool as Pool, RoutingCapture};

/// Routed fixture pool: layer 0 routes (prompt p, step s) → experts
/// {(p+s) % 3 @ 0.75, 3 @ 0.25}; layer 1 is non-MoE. Raw rows are the
/// residual plane × 2; normed rows are × 0.5.
fn routed_fixture(dir: &std::path::Path, prompts: usize, steps: usize, hidden: usize) {
    let layers = 2;
    let residuals: Vec<Vec<Vec<Vec<f32>>>> = (0..prompts)
        .map(|p| {
            (0..steps)
                .map(|s| {
                    (0..layers)
                        .map(|l| {
                            (0..hidden)
                                .map(|h| (p * 1000 + s * 100 + l * 10 + h) as f32)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let scale = |k: f32| -> Vec<Vec<Vec<Vec<f32>>>> {
        residuals
            .iter()
            .map(|p| {
                p.iter()
                    .map(|s| {
                        s.iter()
                            .map(|l| l.iter().map(|v| v * k).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect()
    };
    let routing: Vec<Vec<Vec<Vec<(u32, f32)>>>> = (0..prompts)
        .map(|p| {
            (0..steps)
                .map(|s| {
                    (0..layers)
                        .map(|l| {
                            if l == 0 {
                                vec![(((p + s) % 3) as u32, 0.75f32), (3u32, 0.25f32)]
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let rc = RoutingCapture {
        top_k: 2,
        raw: scale(2.0),
        normed: scale(0.5),
        routing,
    };
    let texts: Vec<String> = (0..prompts).map(|i| format!("p{i}")).collect();
    Pool::write_with_routing(dir, "m", hidden, layers, &texts, &residuals, Some(&rc), 0).unwrap();
}

mod endpoint_seam;
mod movement_ratio_weight_bytes;
mod parsing;
mod pool_facing_gather_layer_subset;
mod routed_denominators;
mod routed_frames_round_trip_vs_the_producti;
mod two_scoreboard_timing_decomposition_dec;
