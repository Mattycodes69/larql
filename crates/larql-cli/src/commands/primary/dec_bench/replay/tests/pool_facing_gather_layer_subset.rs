//! pool-facing gather + layer subset

use super::*;

#[test]
fn routed_layer_subset_excludes_non_moe_layers() {
    let dir = tempfile::tempdir().unwrap();
    routed_fixture(dir.path(), 2, 2, 4);
    let pool = Pool::open(dir.path()).unwrap();
    assert_eq!(routed_layer_subset(&pool, &[0, 1]).unwrap(), vec![0]);
    assert_eq!(
        routed_layer_subset(&pool, &[1]).unwrap(),
        Vec::<usize>::new()
    );
    assert!(routed_layer_subset(&pool, &[0, 5]).is_err(), "oob layer");
}

#[test]
fn routed_layer_subset_errs_on_walk_ffn_only_pool() {
    let dir = tempfile::tempdir().unwrap();
    let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![vec![vec![0.0; 4]; 2]; 1]];
    Pool::write(dir.path(), "m", 4, 2, &["p".into()], &cap, 0).unwrap();
    let pool = Pool::open(dir.path()).unwrap();
    assert!(routed_layer_subset(&pool, &[0]).is_err());
}

#[test]
fn gather_frame_inputs_selects_the_endpoint_plane() {
    let dir = tempfile::tempdir().unwrap();
    routed_fixture(dir.path(), 2, 2, 4);
    let pool = Pool::open(dir.path()).unwrap();

    // Walk-ffn endpoints read the dense-prenormed residual plane.
    let dense = gather_frame_inputs(&pool, Endpoint::WalkFfn, 0, 1, 2).unwrap();
    assert_eq!(dense.rows, pool.rows(2, 1, 0).unwrap());
    assert!(dense.routing.is_none());
    let dense_q8k = gather_frame_inputs(&pool, Endpoint::WalkFfnQ8k, 0, 1, 2).unwrap();
    assert_eq!(dense_q8k.rows, pool.rows(2, 1, 0).unwrap());
    assert!(dense_q8k.routing.is_none());

    // Experts f32 reads raw rows (residual × 2 in this fixture).
    let f32_in = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 0, 1, 2).unwrap();
    assert_eq!(f32_in.rows, pool.raw_rows(2, 1, 0).unwrap());
    for (raw, res) in f32_in.rows.iter().zip(dense.rows.iter()) {
        assert_eq!(*raw, res * 2.0);
    }
    // Experts q8k reads pre-experts-normed rows (× 0.5).
    let q8k_in = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayerQ8k, 0, 1, 2).unwrap();
    assert_eq!(q8k_in.rows, pool.normed_rows(2, 1, 0).unwrap());

    // Per-row routing: prompt 0 step 1 → (1, 0.75), prompt 1 → (2, 0.75).
    let routing = f32_in.routing.as_ref().unwrap();
    assert_eq!(routing.len(), 2);
    assert_eq!(routing[0], vec![(1u32, 0.75f32), (3u32, 0.25f32)]);
    assert_eq!(routing[1], vec![(2u32, 0.75f32), (3u32, 0.25f32)]);
    assert_eq!(q8k_in.routing.as_ref().unwrap(), routing);

    // Non-MoE layer: rows gather fine, routing rows are empty.
    let non_moe = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 1, 0, 2).unwrap();
    assert!(non_moe.routing.as_ref().unwrap().iter().all(Vec::is_empty));

    // Walk-ffn-only pool: experts gather errors loudly.
    let dense_dir = tempfile::tempdir().unwrap();
    let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![vec![vec![0.0; 4]; 2]; 1]];
    Pool::write(dense_dir.path(), "m", 4, 2, &["p".into()], &cap, 0).unwrap();
    let dense_pool = Pool::open(dense_dir.path()).unwrap();
    assert!(gather_frame_inputs(&dense_pool, Endpoint::ExpertsMultiLayer, 0, 0, 1).is_err());
    assert!(gather_frame_inputs(&dense_pool, Endpoint::ExpertsMultiLayerQ8k, 0, 0, 1).is_err());
}

#[test]
fn build_frame_dispatches_per_endpoint_and_pins_walk_ffn_bytes() {
    // hidden must be a Q8K super-block multiple (256): the q8k frame
    // builders quantise rows via `quantize_x_to_q8k`, which
    // debug-asserts 256-alignment — a toy width panics in debug builds.
    const HIDDEN: usize = 256;
    let dir = tempfile::tempdir().unwrap();
    routed_fixture(dir.path(), 2, 2, HIDDEN);
    let pool = Pool::open(dir.path()).unwrap();

    // Walk-ffn: byte-identical to the direct builder (pinned behaviour).
    let f32_arm = WireSpec::Plain(WireArm::F32);
    let dense = gather_frame_inputs(&pool, Endpoint::WalkFfn, 0, 0, 2).unwrap();
    assert_eq!(
        build_frame(Endpoint::WalkFfn, f32_arm, &dense, 2, HIDDEN).unwrap(),
        build_walk_ffn_frame(0, &dense.rows, 2)
    );
    // Plain f16/i8 arms keep the historical f32 request frame — the
    // symmetric arms' bytes are unchanged by the pair axis.
    for arm in [WireArm::F16, WireArm::I8] {
        assert_eq!(
            build_frame(Endpoint::WalkFfn, WireSpec::Plain(arm), &dense, 2, HIDDEN).unwrap(),
            build_walk_ffn_frame(0, &dense.rows, 2)
        );
    }
    assert_eq!(
        build_frame(
            Endpoint::WalkFfnQ8k,
            WireSpec::Plain(WireArm::Q8k),
            &dense,
            2,
            HIDDEN
        )
        .unwrap(),
        build_q8k_frame(0, &dense.rows, 2, HIDDEN)
    );

    // Pair arms encode the request in their inbound format; the frame
    // round-trips through the production request decoder.
    let pair = WireSpec::Pair {
        input: WireFormat::F16,
        output: WireFormat::I8,
    };
    let frame = build_frame(Endpoint::WalkFfn, pair, &dense, 2, HIDDEN).unwrap();
    assert_eq!(
        frame,
        build_walk_ffn_frame_as(WireFormat::F16, 0, &dense.rows, 2)
    );
    let decoded =
        larql_inference::ffn::remote::decode_binary_request_as(WireFormat::F16, &frame).unwrap();
    assert_eq!(decoded.layer, Some(0));
    assert_eq!(decoded.seq_len, 2);
    assert_eq!(decoded.top_k, 0, "replay frames pin top_k=0 in every arm");
    assert_eq!(decoded.residual.len(), 2 * HIDDEN);

    // i8-inbound pair round-trips through the production decoder too.
    let i8_pair = WireSpec::Pair {
        input: WireFormat::I8,
        output: WireFormat::F16,
    };
    let frame = build_frame(Endpoint::WalkFfn, i8_pair, &dense, 2, HIDDEN).unwrap();
    let decoded =
        larql_inference::ffn::remote::decode_binary_request_as(WireFormat::I8, &frame).unwrap();
    assert_eq!(decoded.residual.len(), 2 * HIDDEN);

    // Experts: frame carries per-row routing; missing routing is an error.
    let experts_arm = WireSpec::Plain(WireArm::F32);
    let inputs = gather_frame_inputs(&pool, Endpoint::ExpertsMultiLayer, 0, 0, 2).unwrap();
    let frame = build_frame(Endpoint::ExpertsMultiLayer, experts_arm, &inputs, 2, HIDDEN).unwrap();
    let tasks = larql_inference::ffn::moe_remote::decode_multi_layer_request(&frame).unwrap();
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[1].expert_ids, vec![1, 3]); // prompt 1 step 0 → (0+1)%3
    assert!(build_frame(Endpoint::ExpertsMultiLayer, experts_arm, &dense, 2, HIDDEN).is_err());
}
