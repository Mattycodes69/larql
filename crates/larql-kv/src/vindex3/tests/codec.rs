//! CONTINUATION-CODEC-1 C2: `codec/v1`, the compressed provider.
//!
//! Selection (shipped, KV-only) and configuration (exactly `bits` ∈ {3, 4},
//! named, no default; the two widths are distinct authorities), retention
//! identical to `window/v1`, encode-once storage at exactly the declared
//! bytes, per-head round trips within the codec's pinned cosine floors, the
//! scratch lent only for the layer and range it holds, a head_dim the codec
//! cannot block refused by name before any row, and a handoff resuming at a
//! non-zero base. Fidelity against the frozen yardstick and live-allocation
//! residency are C3's, on the MEM-1 harness.

use larql_vindex::format::vindex3::fixtures::{miniature_glimmer, G_TOKENS, G_WINDOW};
use larql_vindex::format::vindex3::fixtures_kimi::hybrid_kda_mla_f32_model;
use larql_vindex::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::{
    ContinuationAuthority, ContinuationConfig,
};
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, ContinuationRegion, ContinuationRegistryError,
};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::{
    ContinuationError, HistoryRange, KvState, LayerKvGeometry,
};
use larql_vindex::format::vindex3::opplan::exec::prefill_plan;
use larql_vindex::format::vindex3::opplan::exec::reference::ReferenceBackend;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

use super::super::{shipped_continuations, CodecFactory, CodecKvState, WindowKvState};
use super::registry_parity::open;

const RESUME: [u32; 3] = [5, 9, 13];
const DECODE: [u32; 4] = [1, 2, 3, 4];
/// A head width the codec blocks, and enough rows to average over.
const HEAD_DIM: usize = 128;
const HEADS: usize = 2;
const ROWS: usize = 64;
/// The codec's own pinned mean round-trip floors (turbo_quant tests).
const MEAN_COS_FLOOR_4: f32 = 0.995;
const MEAN_COS_FLOOR_3: f32 = 0.982;

fn config(bits: &str) -> ContinuationConfig {
    ContinuationConfig::parse(&[format!("bits={bits}")]).unwrap()
}

fn gaussian_rows(seed: u64, rows: usize, width: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..rows)
        .map(|_| {
            (0..width)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect()
        })
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn wide(history: HistoryRange) -> LayerKvGeometry {
    LayerKvGeometry {
        kv_dim: HEADS * HEAD_DIM,
        head_dim: HEAD_DIM,
        window: match history {
            HistoryRange::Trailing(w) => Some(w),
            _ => None,
        },
        history,
    }
}

#[test]
fn codec_v1_is_shipped_and_holds_only_kv() {
    let registry = shipped_continuations();
    assert!(registry.identities().contains(&CodecKvState::identity()));
    let (_container, plan, _store) = open(hybrid_kda_mla_f32_model, "c2-hybrid");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let refused = registry
        .select(&CodecKvState::identity(), &config("4"), &geometry)
        .expect_err("a hybrid plan needs regions codec/v1 does not hold");
    assert!(
        matches!(refused, ContinuationRegistryError::Unsupported { .. }),
        "{refused}"
    );
    let factory = CodecFactory;
    assert_eq!(factory.identity().to_string(), "codec/v1");
    assert_eq!(factory.regions(), &[ContinuationRegion::Kv]);
}

#[test]
fn bits_is_required_named_and_one_of_three_or_four() {
    let factory = CodecFactory;
    let missing = factory.validate_config(&ContinuationConfig::empty());
    assert!(missing.unwrap_err().contains("no default width"));
    for bad in ["2", "5", "four", ""] {
        let refused = factory.validate_config(&config(bad)).unwrap_err();
        assert!(refused.contains("must be one of"), "{bad}: {refused}");
    }
    let extra = ContinuationConfig::parse(&["bits=4", "window=8"]).unwrap();
    assert!(factory
        .validate_config(&extra)
        .unwrap_err()
        .contains("`window`"));
    for ok in ["3", "4"] {
        factory.validate_config(&config(ok)).unwrap();
        assert_eq!(factory.build(&config(ok)).position(), 0);
    }
    let four = ContinuationAuthority::new(CodecKvState::identity(), &config("4"));
    let three = ContinuationAuthority::new(CodecKvState::identity(), &config("3"));
    assert_ne!(
        four.config_digest, three.config_digest,
        "two widths, two authorities"
    );
}

#[test]
fn a_head_dim_the_codec_cannot_block_is_refused_before_any_row() {
    let mut odd = wide(HistoryRange::Full);
    odd.head_dim = 96;
    odd.kv_dim = 192;
    let mut split = wide(HistoryRange::Full);
    split.kv_dim = HEAD_DIM + HEAD_DIM / 2;
    for (geometry, needle) in [(odd, "not a power of two"), (split, "whole number")] {
        let mut state = CodecKvState::new(4);
        let layers = [
            LayerContinuationGeometry::Kv(wide(HistoryRange::Full)),
            LayerContinuationGeometry::Kv(geometry),
        ];
        let refused = state.prepare_continuation(&layers).unwrap_err();
        assert!(
            matches!(
                &refused,
                ContinuationError::GeometryUnsupported { layer: 1, .. }
            ),
            "{refused}"
        );
        assert!(refused.to_string().contains(needle), "{refused}");
        assert!(refused.to_string().contains("CodecKvState"), "{refused}");
    }
}

#[test]
fn rows_are_encoded_once_at_the_declared_size_and_round_trip_within_the_codecs_floors() {
    for (bits, floor) in [(4u8, MEAN_COS_FLOOR_4), (3, MEAN_COS_FLOOR_3)] {
        let mut state = CodecKvState::new(bits);
        state.prepare(&[wide(HistoryRange::Full)]);
        let keys = gaussian_rows(1, ROWS, HEADS * HEAD_DIM);
        let values = gaussian_rows(2, ROWS, HEADS * HEAD_DIM);
        for (k, v) in keys.iter().zip(&values) {
            state.append(0, k.clone(), v.clone());
        }
        let declared = 2 * HEADS * (4 + HEAD_DIM * bits as usize / 8);
        assert_eq!(state.encoded_row_bytes(0), declared);
        let before: Vec<Vec<u8>> = state.encoded_rows(0).to_vec();
        assert!(before
            .iter()
            .all(|r| r.len() == declared && r.capacity() == declared));

        state.prepare_layer(0);
        assert_eq!(state.encoded_rows(0), before, "reading never re-encodes");
        let view = state.rows(0);
        assert_eq!((view.base(), view.end()), (0, ROWS));
        let mut total = 0.0;
        for p in 0..ROWS {
            for h in 0..HEADS {
                let span = h * HEAD_DIM..(h + 1) * HEAD_DIM;
                total += cos(&keys[p][span.clone()], &view.key(p)[span.clone()]);
                total += cos(&values[p][span.clone()], &view.value(p)[span]);
            }
        }
        let mean = total / (2 * ROWS * HEADS) as f32;
        assert!(
            mean >= floor,
            "{bits}-bit mean head cosine {mean} < {floor}"
        );
    }
}

#[test]
fn the_scratch_is_lent_only_for_the_layer_and_range_it_holds() {
    let mut state = CodecKvState::new(4);
    state.prepare(&[wide(HistoryRange::Trailing(3)), wide(HistoryRange::Full)]);
    let rows = gaussian_rows(3, 5, HEADS * HEAD_DIM);
    for r in &rows {
        state.append(0, r.clone(), r.clone());
        state.append(1, r.clone(), r.clone());
    }
    state.prepare_layer(0);
    let sliding = state.rows(0);
    assert_eq!(
        (sliding.base(), sliding.end()),
        (2, 5),
        "window/v1's retention"
    );
    assert_eq!(
        state.rows(1).end(),
        0,
        "another layer's read is the empty view"
    );
    state.append(0, rows[0].clone(), rows[0].clone());
    assert_eq!(
        state.rows(0).end(),
        5,
        "a read not re-prepared is stale, and the step refuses it"
    );
    state.prepare_layer(1);
    let full = state.rows(1);
    assert_eq!((full.base(), full.end()), (0, 5));
}

#[test]
fn codec_v1_retains_exactly_what_window_v1_retains_through_every_phase() {
    let (_container, plan, store) = open(miniature_glimmer, "c2-retention");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let registry = shipped_continuations();
    let backend = ReferenceBackend::new();
    let mut held = Vec::new();
    for (identity, cfg) in [
        (WindowKvState::identity(), ContinuationConfig::empty()),
        (CodecKvState::identity(), config("4")),
    ] {
        let mut state = registry.select(&identity, &cfg, &geometry).unwrap().build();
        prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *state).unwrap();
        prefill_plan(&plan, &store, &RESUME, &backend, &mut *state).unwrap();
        let mut session =
            DecodeSession::with_kv_state(&plan, &store, &backend, &mut *state).unwrap();
        for &token in &DECODE {
            let logits = session.step(token).unwrap().logits.unwrap();
            assert!(logits.iter().all(|x| x.is_finite()), "{identity}");
        }
        drop(session);
        let ranges: Vec<(usize, usize)> = (0..geometry.len())
            .map(|layer| {
                state.prepare_layer(layer);
                let v = state.rows(layer);
                (v.base(), v.end())
            })
            .collect();
        held.push(ranges);
    }
    let n = G_TOKENS.len() + RESUME.len() + DECODE.len();
    assert_eq!(
        held[0], held[1],
        "compression changes bytes per row, never which rows"
    );
    assert_eq!(held[1][0], (n - G_WINDOW, n));
}

#[test]
fn a_handoff_resumes_at_a_non_zero_base_under_its_own_width_only() {
    let (_container, plan, store) = open(miniature_glimmer, "c2-resume");
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let registry = shipped_continuations();
    let backend = ReferenceBackend::new();
    let decode_after = |state: &mut dyn KvState| -> Vec<Vec<u32>> {
        let mut session = DecodeSession::with_kv_state(&plan, &store, &backend, state).unwrap();
        DECODE
            .iter()
            .map(|&t| {
                session
                    .step(t)
                    .unwrap()
                    .logits
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            })
            .collect()
    };
    let select = |bits: &str| {
        registry
            .select(&CodecKvState::identity(), &config(bits), &geometry)
            .unwrap()
    };

    let mut straight = select("4").build();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, &mut *straight).unwrap();
    prefill_plan(&plan, &store, &RESUME, &backend, &mut *straight).unwrap();
    let expected = decode_after(&mut *straight);

    let four = select("4");
    let authority = four.authority().clone();
    let mut handoff = four.begin();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, handoff.state_mut()).unwrap();
    prefill_plan(&plan, &store, &RESUME, &backend, handoff.state_mut()).unwrap();

    let mut other = select("4").begin();
    prefill_plan(&plan, &store, &G_TOKENS, &backend, other.state_mut()).unwrap();
    assert!(
        other.resume(select("3").authority()).is_err(),
        "state built at 4 bits must refuse the 3-bit authority"
    );

    let mut resumed = handoff.resume(&authority).unwrap();
    resumed.state_mut().prepare_layer(0);
    assert!(
        resumed.state().rows(0).base() > 0,
        "resumed past a drained front"
    );
    assert_eq!(
        decode_after(resumed.state_mut()),
        expected,
        "resume is exact to the uninterrupted codec run"
    );
}

#[test]
fn recurrent_and_latent_state_are_refused_by_name() {
    let mut state = CodecKvState::new(4);
    state.prepare(&[wide(HistoryRange::Full)]);
    assert!(state
        .recurrent_state(0)
        .unwrap_err()
        .to_string()
        .contains("CodecKvState"));
    assert!(state
        .latent_state(0)
        .unwrap_err()
        .to_string()
        .contains("CodecKvState"));
}

#[test]
#[should_panic(expected = "K row at layer 0 is 3 wide; the plan says 256")]
fn a_misfit_row_is_refused() {
    let mut state = CodecKvState::new(4);
    state.prepare(&[wide(HistoryRange::Full)]);
    state.append(0, vec![0.0; 3], vec![0.0; HEADS * HEAD_DIM]);
}
