//! The layer's host-side vocabulary — encodings, addressing, the chain
//! timing split — and the refusals and trace readback in `validate.rs`
//! that the end-to-end gates do not reach.

use super::*;

/// Q8_0: 34-byte blocks of 32.
const Q8_0_BLOCK: (usize, usize) = (32, 34);
/// Q6_K / Q4_K: 210- and 144-byte super-blocks of 256.
const Q6_K_SUPERBLOCK: (usize, usize) = (256, 210);
const Q4_K_SUPERBLOCK: (usize, usize) = (256, 144);

#[test]
fn every_expert_encoding_names_its_ggml_type() {
    assert_eq!(ExpertEncoding::Bf16.name(), "BF16");
    assert_eq!(ExpertEncoding::Q80.name(), "Q8_0");
    assert_eq!(ExpertEncoding::Q6K.name(), "Q6_K");
    assert_eq!(ExpertEncoding::Q4K.name(), "Q4_K");
}

#[test]
fn matrix_bytes_follows_each_encodings_block_geometry() {
    let (n, k) = (3, 512);
    assert_eq!(ExpertEncoding::Bf16.matrix_bytes(n, k), Some(n * k * 2));
    let (qb, qbytes) = Q8_0_BLOCK;
    assert_eq!(
        ExpertEncoding::Q80.matrix_bytes(n, k),
        Some(n * k / qb * qbytes)
    );
    let (sb, q6) = Q6_K_SUPERBLOCK;
    assert_eq!(
        ExpertEncoding::Q6K.matrix_bytes(n, k),
        Some(n * k / sb * q6)
    );
    let (_, q4) = Q4_K_SUPERBLOCK;
    assert_eq!(
        ExpertEncoding::Q4K.matrix_bytes(n, k),
        Some(n * k / sb * q4)
    );

    // A K that is not a whole number of blocks cannot be encoded at all —
    // not rounded, refused.
    assert_eq!(ExpertEncoding::Q80.matrix_bytes(n, qb + 1), None);
    assert_eq!(ExpertEncoding::Q6K.matrix_bytes(n, qb), None);
    assert_eq!(ExpertEncoding::Q4K.matrix_bytes(n, sb + qb), None);
    // BF16 has no block, so any K is fine.
    assert_eq!(
        ExpertEncoding::Bf16.matrix_bytes(n, qb + 1),
        Some(n * (qb + 1) * 2)
    );
}

#[test]
fn identity_addressing_multiplies_and_refuses_what_it_does_not_address() {
    let stride = 1000u32;
    let a = ExpertAddressing::Identity { experts: 4, stride };
    assert_eq!(a.experts(), 4);
    assert_eq!(a.identity_stride(), stride);
    assert_eq!(a.offset_of(0), Some(0));
    assert_eq!(a.offset_of(3), Some(3 * u64::from(stride)));
    // Past the bank: not an offset into someone else's weights.
    assert_eq!(a.offset_of(4), None);

    // Wide on purpose: an offset past 4 GiB is answered in 64 bits rather
    // than wrapping into another expert.
    let big = ExpertAddressing::Identity {
        experts: 8,
        stride: u32::MAX,
    };
    assert_eq!(big.offset_of(7), Some(7 * u64::from(u32::MAX)));
    // And a product past 64 bits is refused rather than wrapped.
    let huge = ExpertAddressing::Identity {
        experts: usize::MAX,
        stride: u32::MAX,
    };
    assert_eq!(huge.offset_of(1 << 40), None);
}

#[test]
fn table_addressing_reads_the_table_and_honours_not_resident() {
    let table = [0u32, layer_shader::NOT_RESIDENT, 4096];
    let a = ExpertAddressing::Table(&table);
    assert_eq!(a.experts(), 3);
    // A table bank has no stride: the kernel must consult the table.
    assert_eq!(a.identity_stride(), 0);
    assert_eq!(a.offset_of(0), Some(0));
    assert_eq!(a.offset_of(1), None, "NOT_RESIDENT is not an offset");
    assert_eq!(a.offset_of(2), Some(4096));
    assert_eq!(a.offset_of(3), None, "past the table");
}

#[test]
fn chain_timing_is_returned_once_and_reset() {
    ENCODE_MS.with(|c| c.set(1.5));
    WAIT_MS.with(|c| c.set(2.25));
    assert_eq!(take_chain_timing_ms(), (1.5, 2.25));
    assert_eq!(
        take_chain_timing_ms(),
        (0.0, 0.0),
        "the split resets on read"
    );
}

/// The router width is `experts * hidden`; a matrix of any other length
/// is refused before it is bound.
#[test]
fn a_router_matrix_of_the_wrong_width_is_refused() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let short = &f.router_weight[..f.router_weight.len() - HIDDEN];
    let mut layer = f.layer(&state);
    moe_mut(&mut layer).router_weight = short;
    assert_eq!(
        m.kimi_decoder_layer(layer, &f.x).map(|(o, _)| o),
        Err(GroupedError::SlotCountMismatch {
            expected: EXPERTS * HIDDEN,
            found: short.len(),
        })
    );
}

/// `want_selection_scores` brings back, per layer, the score every expert
/// was ranked by and the combine weights — and they are the ones the
/// route was actually decided by: the chosen ids are the top-k of the
/// scores, and the combine weights are the renormalised, scaled routed
/// weights followed by the shared branch's unit weight.
#[test]
fn selection_scores_and_combine_weights_explain_the_route() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let mut trace = ExecutionTrace {
        want_selection_scores: true,
        ..ExecutionTrace::default()
    };
    m.kimi_decoder_layers(
        &[KimiLayerCall {
            weights: f.layer(&state),
        }],
        &f.x,
        Some(&mut trace),
    )
    .expect("the fixture layer runs");

    assert_eq!(trace.routes.len(), 1);
    assert_eq!(trace.selection_scores.len(), 1);
    assert_eq!(trace.combine_weights.len(), 1);
    let (route, scores, weights) = (
        &trace.routes[0],
        &trace.selection_scores[0],
        &trace.combine_weights[0],
    );
    assert_eq!(route.len(), TOP_K);
    assert_eq!(scores.len(), EXPERTS, "one score per expert");
    assert_eq!(
        weights.len(),
        TOP_K + 1,
        "top-k routed, then the shared branch"
    );

    // The route is the top-k of the scores (ties to the lower index).
    let mut ranked: Vec<usize> = (0..EXPERTS).collect();
    ranked.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut chosen: Vec<usize> = route.iter().map(|&e| e as usize).collect();
    chosen.sort_unstable();
    let mut top: Vec<usize> = ranked[..TOP_K].to_vec();
    top.sort_unstable();
    assert_eq!(chosen, top, "the route is not the top-k of its own scores");

    // Renormalised and scaled: the routed weights sum to the branch
    // scale; the shared branch is unscaled.
    let routed: f32 = weights[..TOP_K].iter().sum();
    assert!(
        (routed - BRANCH_SCALE).abs() < TOLERANCE,
        "routed weights sum to {routed}, want {BRANCH_SCALE}"
    );
    assert_eq!(weights[TOP_K], 1.0, "the shared branch's weight is unit");
}

/// A dense layer routes nothing, so its trace rows are empty rather than
/// a read of the one-element placeholder buffer.
#[test]
fn a_dense_layer_reports_empty_scores_and_weights() {
    let m = backend();
    let f = fixture();
    let state = KdaDeviceState::zeros(&m, shape());
    let mut trace = ExecutionTrace {
        want_selection_scores: true,
        ..ExecutionTrace::default()
    };
    m.kimi_decoder_layers(
        &[KimiLayerCall {
            weights: f.dense_layer(&state),
        }],
        &f.x,
        Some(&mut trace),
    )
    .expect("the dense layer runs");
    assert_eq!(trace.routes, vec![Vec::<u32>::new()]);
    assert_eq!(trace.selection_scores, vec![Vec::<f32>::new()]);
    assert_eq!(trace.combine_weights, vec![Vec::<f32>::new()]);
}
