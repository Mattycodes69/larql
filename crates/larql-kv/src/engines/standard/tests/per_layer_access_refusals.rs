//! `PerLayerKvAccess` on `StandardEngine` refuses — `None` / `false`,
//! with the cache untouched — whenever it cannot act on real per-layer
//! rows: the async slot (no host readback), the coarse path (one
//! whole-model handle), no cache, a layer out of range, or a position
//! map that disagrees with the cache.

use super::*;
use larql_inference::kv_engine::{ExcisedKvRows, PerLayerKvAccess};
use larql_inference::kv_row_positions::KvRowPositions;
use ndarray::Array2;

const PROMPT: [u32; 4] = [0, 1, 2, 3];
const LAYER: usize = 0;
/// Far past the synthetic model's two layers.
const MISSING_LAYER: usize = 99;
const WINDOW: u64 = 2;

fn prefilled(mut engine: StandardEngine) -> StandardEngine {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    engine.prefill(&weights, &ffn, &PROMPT).unwrap();
    engine
}

fn sync_engine() -> StandardEngine {
    prefilled(StandardEngine::new(None))
}

fn async_engine() -> StandardEngine {
    prefilled(StandardEngine::with_async_backend(
        None,
        Box::new(CpuBackend),
    ))
}

fn coarse_engine() -> StandardEngine {
    let mut engine = sync_engine();
    engine.prefill_mode = Some(PrefillDispatchMode::Coarse);
    engine
}

fn handleless_engine() -> StandardEngine {
    let mut engine = sync_engine();
    engine.handles = None;
    engine
}

/// A map that claims one row fewer per layer than the cache holds.
fn skewed_map_engine() -> StandardEngine {
    let mut engine = sync_engine();
    let rows: Vec<usize> = engine
        .layer_handles()
        .unwrap()
        .iter()
        .map(|h| h.cached_len() - 1)
        .collect();
    engine.row_positions = KvRowPositions::tails_ending_at(engine.abs_position as u64, &rows);
    engine
}

fn unmapped_engine() -> StandardEngine {
    let mut engine = sync_engine();
    engine.row_positions = Default::default();
    engine
}

/// One excised row, to splice back in.
fn one_row() -> ExcisedKvRows {
    sync_engine()
        .excise_kv_rows(LAYER, 0..1)
        .expect("a live per-layer cache excises")
}

fn layer_kv(engine: &StandardEngine) -> (Array2<f32>, Array2<f32>) {
    engine.read_layer_kv(LAYER).expect("readable")
}

fn replace_with_own_rows(engine: &mut StandardEngine, layer: usize) -> bool {
    let (k, v) = layer_kv(&sync_engine());
    let positions: Vec<u64> = (0..k.nrows() as u64).collect();
    engine.replace_layer_kv(layer, &k, &v, &positions)
}

#[test]
fn coarse_mode_reports_no_layer_count() {
    assert_eq!(sync_engine().layer_count(), Some(2));
    assert_eq!(coarse_engine().layer_count(), None);
}

#[test]
fn the_async_slot_refuses_every_host_readback_operation() {
    let mut engine = async_engine();
    assert!(engine.read_layer_kv(LAYER).is_none());
    assert!(engine.excise_kv_rows(LAYER, 0..1).is_none());
    assert!(engine.clip_layer_to_logical_window(LAYER, WINDOW).is_none());
    assert!(!engine.splice_kv_rows(LAYER, 0, &one_row()));
    let (k, v) = layer_kv(&sync_engine());
    let positions: Vec<u64> = (0..k.nrows() as u64).collect();
    assert!(!engine.replace_layer_kv(LAYER, &k, &v, &positions));
}

#[test]
fn coarse_mode_refuses_every_row_operation() {
    let mut engine = coarse_engine();
    assert!(engine.excise_kv_rows(LAYER, 0..1).is_none());
    assert!(engine.clip_layer_to_logical_window(LAYER, WINDOW).is_none());
    assert!(!engine.splice_kv_rows(LAYER, 0, &one_row()));
    assert!(!replace_with_own_rows(&mut engine, LAYER));
}

#[test]
fn a_missing_cache_refuses_splice_and_replace() {
    let mut engine = handleless_engine();
    assert!(!engine.splice_kv_rows(LAYER, 0, &one_row()));
    assert!(!replace_with_own_rows(&mut engine, LAYER));
}

#[test]
fn an_out_of_range_layer_refuses_splice_and_replace() {
    let mut engine = sync_engine();
    assert!(!engine.splice_kv_rows(MISSING_LAYER, 0, &one_row()));
    assert!(!replace_with_own_rows(&mut engine, MISSING_LAYER));
}

#[test]
fn excise_refuses_an_empty_or_overlong_range() {
    let mut engine = sync_engine();
    assert!(engine.excise_kv_rows(LAYER, 1..1).is_none());
    let past_end = PROMPT.len()..PROMPT.len() + 1;
    assert!(engine.excise_kv_rows(LAYER, past_end).is_none());
    assert_eq!(engine.resident_rows(LAYER), Some(PROMPT.len()));
}

#[test]
fn a_map_that_disagrees_with_the_cache_refuses_row_surgery() {
    let mut engine = skewed_map_engine();
    assert!(engine.excise_kv_rows(LAYER, 0..1).is_none());
    assert!(engine.clip_layer_to_logical_window(LAYER, WINDOW).is_none());
    assert!(!engine.splice_kv_rows(LAYER, 0, &one_row()));
    assert_eq!(engine.resident_rows(LAYER), Some(PROMPT.len()), "untouched");
}

#[test]
fn an_unmapped_layer_refuses_splice() {
    assert!(!unmapped_engine().splice_kv_rows(LAYER, 0, &one_row()));
}

#[test]
fn splice_refuses_an_offset_past_the_end() {
    let mut engine = sync_engine();
    assert!(!engine.splice_kv_rows(LAYER, PROMPT.len() + 1, &one_row()));
}

#[test]
fn replace_refuses_mismatched_shapes_and_descending_positions() {
    let mut engine = sync_engine();
    let (k, v) = layer_kv(&engine);
    let rows = k.nrows() as u64;
    let ascending: Vec<u64> = (0..rows).collect();
    let too_few = &ascending[..ascending.len() - 1];
    assert!(!engine.replace_layer_kv(LAYER, &k, &v, too_few));
    let descending: Vec<u64> = (0..rows).rev().collect();
    assert!(!engine.replace_layer_kv(LAYER, &k, &v, &descending));
    assert!(engine.replace_layer_kv(LAYER, &k, &v, &ascending));
}

#[test]
fn the_next_position_can_move_ahead_of_every_resident_row() {
    let mut engine = sync_engine();
    let ahead = PROMPT.len() + 1;
    assert!(engine.set_logical_next_position(ahead));
    assert_eq!(engine.logical_next_position(), ahead);
    assert!(
        !engine.set_logical_next_position(0),
        "behind a resident row is refused"
    );
}
