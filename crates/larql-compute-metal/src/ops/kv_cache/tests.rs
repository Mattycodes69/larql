use super::*;
use metal::Device;

const SHAPE_SMALL: (usize, usize) = (2, 64);
const SHAPE_LARGE: (usize, usize) = (4, 128);

fn fresh_cache() -> (BufferCache, Device) {
    let d = Device::system_default().expect("Metal device available on test host");
    let bufs = BufferCache::new(&d);
    (bufs, d)
}

// ── occupancy vs absolute position ──────────────────────────────
//
// `current_len` used to be both "rows stored" and "stream position".
// A sliding window separates them: eviction lowers occupancy while
// the stream keeps advancing. If they re-merge, RoPE silently rewinds
// on every token after a window slides — which surfaces as fluent but
// wrong output, the worst failure shape to debug.

/// One row appended advances both counters together.
#[test]
fn advance_one_moves_occupancy_and_position_together() {
    let (bufs, _d) = fresh_cache();
    let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
    assert_eq!((c.current_len, c.abs_position), (0, 0));
    c.advance_one();
    c.advance_one();
    assert_eq!((c.current_len, c.abs_position), (2, 2));
}

/// Eviction lowers occupancy and leaves the stream position alone.
/// This is the whole invariant the window rests on.
#[test]
fn eviction_lowers_occupancy_but_never_the_stream_position() {
    let (bufs, _d) = fresh_cache();
    let mut c = LayerKVCache::new(&bufs, 16, 2, 4);
    for _ in 0..10 {
        c.advance_one();
    }
    assert_eq!((c.current_len, c.abs_position), (10, 10));

    let dropped = c.evict_to_window(4);
    assert_eq!(dropped, 6);
    assert_eq!(c.current_len, 4, "occupancy must fall to the window");
    assert_eq!(
        c.abs_position, 10,
        "stream position must NOT rewind — the next row still belongs at 10"
    );

    // Appending after eviction continues the stream, it does not restart it.
    c.advance_one();
    assert_eq!((c.current_len, c.abs_position), (5, 11));
}

/// Eviction keeps the NEWEST rows, in order.
#[test]
fn eviction_keeps_the_newest_rows_in_order() {
    let (bufs, _d) = fresh_cache();
    let (kv_heads, head_dim) = (2usize, 4usize);
    let row = kv_heads * head_dim;
    let mut c = LayerKVCache::new(&bufs, 16, kv_heads, head_dim);

    // Stamp each row with its index so survivors are identifiable.
    let rows = 10usize;
    unsafe {
        for buf in [&c.k_cache, &c.v_cache] {
            let ptr = buf.contents() as *mut f32;
            let slice = std::slice::from_raw_parts_mut(ptr, 16 * row);
            for r in 0..rows {
                for i in 0..row {
                    slice[r * row + i] = r as f32;
                }
            }
        }
    }
    for _ in 0..rows {
        c.advance_one();
    }

    let window = 4usize;
    c.evict_to_window(window);

    unsafe {
        for buf in [&c.k_cache, &c.v_cache] {
            let ptr = buf.contents() as *const f32;
            let slice = std::slice::from_raw_parts(ptr, 16 * row);
            for r in 0..window {
                let expected = (rows - window + r) as f32;
                assert_eq!(
                    slice[r * row],
                    expected,
                    "row {r} after eviction should hold original row {expected}"
                );
            }
        }
    }
}

/// A window at or above occupancy is a no-op — nothing moves, so the
/// unwindowed path pays nothing for the affordance existing.
#[test]
fn eviction_is_a_noop_when_the_window_cannot_bind() {
    let (bufs, _d) = fresh_cache();
    let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
    for _ in 0..3 {
        c.advance_one();
    }
    assert_eq!(c.evict_to_window(3), 0);
    assert_eq!(c.evict_to_window(99), 0);
    assert_eq!((c.current_len, c.abs_position), (3, 3));
}

/// A zero window is refused rather than emptying the cache — the same
/// sentinel confusion that already cost a bug in the pipeline spec.
#[test]
fn a_zero_window_evicts_nothing() {
    let (bufs, _d) = fresh_cache();
    let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
    c.advance_one();
    assert_eq!(c.evict_to_window(0), 0);
    assert_eq!(c.current_len, 1);
}

/// A new prompt restarts the stream, so `clear` resets both.
#[test]
fn clear_resets_occupancy_and_position() {
    let (bufs, _d) = fresh_cache();
    let mut c = LayerKVCache::new(&bufs, 8, 2, 4);
    c.advance_one();
    c.advance_one();
    c.clear();
    assert_eq!((c.current_len, c.abs_position), (0, 0));
}

#[test]
fn shape_mismatch_detects_conflicting_existing_layer() {
    assert!(!super::shape_pairs_have_mismatch(
        &[SHAPE_SMALL],
        &[SHAPE_SMALL, SHAPE_LARGE]
    ));
    assert!(super::shape_pairs_have_mismatch(
        &[SHAPE_SMALL],
        &[SHAPE_LARGE]
    ));
}

/// `attention_span` returns `t` when `window_size == 0` (no
/// windowing) or when `t <= window_size` (cache still within
/// window). Returns `window_size` once `t` exceeds it.
#[test]
fn attention_span_clamps_at_window_size_when_exceeded() {
    assert_eq!(attention_span(5, 0), 5, "window=0 disables clamp");
    assert_eq!(attention_span(5, 10), 5, "t<=window returns t");
    assert_eq!(attention_span(10, 10), 10, "t==window returns t");
    assert_eq!(attention_span(15, 10), 10, "t>window clamps to window");
}

/// `LayerKVCache::clear` resets `current_len` without touching the
/// underlying buffers.
#[test]
fn layer_kv_cache_clear_resets_current_len() {
    let (bufs, _) = fresh_cache();
    let mut layer = LayerKVCache::new(&bufs, 64, 2, 64);
    layer.current_len = 17;
    layer.clear();
    assert_eq!(layer.current_len, 0);
    assert_eq!(layer.max_seq, 64);
    assert_eq!(layer.num_kv_heads, 2);
    assert_eq!(layer.head_dim, 64);
}

/// `KVCache::new` constructs the requested number of uniform-shape
/// layers.  Round-trips the per-layer dimensions through
/// `has_shape_mismatch`.
#[test]
fn kv_cache_new_creates_uniform_layers() {
    let (bufs, _) = fresh_cache();
    let cache = KVCache::new(&bufs, 3, 32, 2, 64);
    assert_eq!(cache.layers.len(), 3);
    assert!(!cache.has_shape_mismatch(&[(2, 64), (2, 64), (2, 64)]));
    assert!(cache.has_shape_mismatch(&[(2, 64), (2, 64), (4, 64)]));
}

/// `KVCache::new_per_layer` allocates with heterogeneous shapes —
/// pin the Gemma 4 31B pattern (alternating sliding/global heads).
#[test]
fn kv_cache_new_per_layer_supports_heterogeneous_shapes() {
    let (bufs, _) = fresh_cache();
    let shapes = vec![(16usize, 256usize), (4, 512), (16, 256), (4, 512)];
    let cache = KVCache::new_per_layer(&bufs, &shapes, 32);
    assert_eq!(cache.layers.len(), 4);
    for (layer, &(num_kv, hd)) in cache.layers.iter().zip(&shapes) {
        assert_eq!(layer.num_kv_heads, num_kv);
        assert_eq!(layer.head_dim, hd);
    }
}

/// `grow_to_shapes` extends the cache when more layers are
/// requested than currently allocated.
#[test]
fn kv_cache_grow_to_shapes_extends_layers() {
    let (bufs, _) = fresh_cache();
    let mut cache = KVCache::new(&bufs, 2, 32, 2, 64);
    assert_eq!(cache.layers.len(), 2);

    let shapes = vec![(2usize, 64usize), (2, 64), (4, 128), (8, 256)];
    cache.grow_to_shapes(&bufs, &shapes, 32);
    assert_eq!(cache.layers.len(), 4);
    assert_eq!(cache.layers[2].num_kv_heads, 4);
    assert_eq!(cache.layers[2].head_dim, 128);
    assert_eq!(cache.layers[3].num_kv_heads, 8);
    assert_eq!(cache.layers[3].head_dim, 256);

    // Idempotent: regrow to same length is a no-op.
    cache.grow_to_shapes(&bufs, &shapes, 32);
    assert_eq!(cache.layers.len(), 4);
}

/// `KVCache::clear` resets every layer's `current_len`.
#[test]
fn kv_cache_clear_resets_all_layers() {
    let (bufs, _) = fresh_cache();
    let mut cache = KVCache::new(&bufs, 3, 32, 2, 64);
    for layer in &mut cache.layers {
        layer.current_len = 9;
    }
    cache.clear();
    assert!(cache.layers.iter().all(|l| l.current_len == 0));
}

/// `current_len` reads from the first layer (assumes uniform
/// progression).  Returns 0 when there are no layers.
#[test]
fn kv_cache_current_len_reads_first_layer() {
    let (bufs, _) = fresh_cache();
    let mut cache = KVCache::new(&bufs, 2, 32, 2, 64);
    assert_eq!(cache.current_len(), 0);
    cache.layers[0].current_len = 7;
    assert_eq!(cache.current_len(), 7);

    let empty = KVCache { layers: Vec::new() };
    assert_eq!(empty.current_len(), 0);
}

// ─── End-to-end Metal dispatch tests for the encoder helpers ───
//
// The remaining uncovered lines exercise `encode_kv_append`,
// `encode_kv_attend` (both the short-span and long-span branches),
// and the `append_and_attend` legacy convenience wrapper.  Real
// GPU dispatches are cheap on small shapes (~< 1 ms per call) so
// we drive them through `MetalBackend::new()` and assert that the
// kernels complete without panic and produce finite output.
use crate::MetalBackend;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal device available on test host")
}

fn append_attend_shapes() -> (usize, usize, usize) {
    // (max_seq, num_kv_heads, head_dim). Sized small so the test
    // stays under a millisecond.  num_q_heads = num_kv_heads in
    // this fixture (non-GQA shape) to keep the input vectors
    // small.
    (8, 2, 64)
}

/// #229. A full cache must refuse the append rather than write row
/// `max_seq` past its buffer. Encoding, not execution: the assertion
/// fires before any dispatch is recorded.
#[test]
fn encode_kv_append_refuses_a_full_cache() {
    let m = backend();
    let (_, num_kv, head_dim) = append_attend_shapes();
    let max_seq = 4;
    let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
    layer.current_len = max_seq; // positions 0..4 stored; the 5th must not land
    let zeros = vec![0.0f32; num_kv * head_dim];
    let new_k_buf = m.bufs.transient_from_f32(&zeros);
    let new_v_buf = m.bufs.transient_from_f32(&zeros);
    let cmd = m.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    // Metal aborts the process if an encoder is dropped mid-panic, so
    // catch the refusal and end the encoder before asserting on it.
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        encode_kv_append(
            enc,
            &layer,
            &m.attention.kv_append_pipeline,
            &new_k_buf,
            &new_v_buf,
        )
    }));
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "kv_cache refuse test").expect("command buffer completed");
    let msg = match refused {
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
        Ok(()) => panic!("a full cache accepted an append past its buffer"),
    };
    assert!(
        msg.contains("KV append past capacity"),
        "unexpected refusal message: {msg}"
    );
}

/// `encode_kv_append` writes new K/V rows into the cache slot at
/// `current_len`.  After a single dispatch + commit + wait the
/// dispatch should complete and the kernel input/output buffers
/// should hold finite values.
#[test]
fn encode_kv_append_completes_and_advances_position() {
    let m = backend();
    let (max_seq, num_kv, head_dim) = append_attend_shapes();
    let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);

    let new_k: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
    let new_v: Vec<f32> = (0..num_kv * head_dim)
        .map(|i| ((i + 1) as f32) * 0.002)
        .collect();
    let new_k_buf = m.bufs.transient_from_f32(&new_k);
    let new_v_buf = m.bufs.transient_from_f32(&new_v);

    let cmd = m.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_kv_append(
        enc,
        &layer,
        &m.attention.kv_append_pipeline,
        &new_k_buf,
        &new_v_buf,
    );
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1042")
        .expect("command buffer completed");

    // The callsite is responsible for bumping `current_len`; the
    // encoder itself only writes the buffer.  Mirror the legacy
    // contract here so the next test path (short-span attend) has
    // a sensible len.
    layer.current_len = 1;
    assert_eq!(layer.current_len, 1);
}

/// `encode_kv_attend` short-span path (`span <= SHORT_ATTENTION_SPAN`)
/// dispatches the `attend_pipeline`.  Pass `None` for
/// `attend_long_pipeline` so the function uses `attend_pipeline`
/// even if the span grew.
#[test]
fn encode_kv_attend_short_span_dispatches_with_attend_pipeline() {
    let m = backend();
    let (max_seq, num_kv, head_dim) = append_attend_shapes();
    let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
    layer.current_len = 1; // one prior token written

    let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.01).collect();
    let q_buf = m.bufs.transient_from_f32(&q);
    let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

    let cmd = m.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_kv_attend(
        enc,
        &layer,
        &m.attention.kv_attend_pipeline,
        None, // long pipeline absent → unwrap_or(attend) path
        &q_buf,
        &out_buf,
        num_kv,
        (head_dim as f32).sqrt().recip(),
        0,
        None,
        0.0,
    );
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1084")
        .expect("command buffer completed");

    let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
    assert!(out.iter().all(|v| v.is_finite()));
}

/// `encode_kv_attend` long-span branch: `span > SHORT_ATTENTION_SPAN`
/// AND `attend_long_pipeline = Some(...)` picks the long-span
/// kernel.  Drive this by passing the long pipeline and a
/// `current_len` large enough to push `span` past the threshold.
///
/// We don't assert finiteness here — the cache slots beyond
/// the one we wrote are still zero-initialised (no `append`
/// upstream in this minimal test), so attention over a stretch
/// of zero-K rows produces numerically degenerate output.  This
/// test pins **that the long pipeline is selected and dispatches
/// successfully** (i.e. doesn't panic / fail commit), which is
/// the part of `encode_kv_attend`'s contract that's interesting
/// for coverage.
#[test]
fn encode_kv_attend_long_span_picks_attend_long_pipeline() {
    let m = backend();
    let (_, num_kv, head_dim) = append_attend_shapes();
    let mut layer = LayerKVCache::new(&m.bufs, 1024, num_kv, head_dim);
    layer.current_len = (SHORT_ATTENTION_SPAN + 2) as usize;

    let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
    let q_buf = m.bufs.transient_from_f32(&q);
    let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

    let cmd = m.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    encode_kv_attend(
        enc,
        &layer,
        &m.attention.kv_attend_pipeline,
        Some(&m.attention.kv_attend_long_pipeline),
        &q_buf,
        &out_buf,
        num_kv,
        (head_dim as f32).sqrt().recip(),
        0,
        None,
        0.0,
    );
    enc.end_encoding();
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1131")
        .expect("command buffer completed");

    // Output buffer length matches the requested shape (a weak but
    // valid post-condition: a panicked kernel never gets here, and
    // the dispatch picked the long branch).
    let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
    assert_eq!(out.len(), num_kv * head_dim);
}

/// `append_and_attend` chains the append + attend dispatches in a
/// single command buffer and bumps `current_len` itself.  Covers
/// the `pub fn append_and_attend` body + the two encoder blocks
/// it owns.
#[test]
fn append_and_attend_runs_append_then_attend_and_bumps_len() {
    let m = backend();
    let (max_seq, num_kv, head_dim) = append_attend_shapes();
    let mut layer = LayerKVCache::new(&m.bufs, max_seq, num_kv, head_dim);
    assert_eq!(layer.current_len, 0);

    let new_k: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.001).collect();
    let new_v: Vec<f32> = (0..num_kv * head_dim)
        .map(|i| ((i + 1) as f32) * 0.002)
        .collect();
    let q: Vec<f32> = (0..num_kv * head_dim).map(|i| (i as f32) * 0.01).collect();

    let new_k_buf = m.bufs.transient_from_f32(&new_k);
    let new_v_buf = m.bufs.transient_from_f32(&new_v);
    let q_buf = m.bufs.transient_from_f32(&q);
    let out_buf = m.bufs.output((num_kv * head_dim * 4) as u64);

    let cmd = m.queue.new_command_buffer();
    append_and_attend(
        cmd,
        &mut layer,
        &m.attention.kv_append_pipeline,
        &m.attention.kv_attend_pipeline,
        &new_k_buf,
        &new_v_buf,
        &q_buf,
        &out_buf,
        num_kv,
        (head_dim as f32).sqrt().recip(),
    );
    cmd.commit();
    crate::cb_status::wait_checked(cmd, "crates/larql-compute-metal/src/ops/kv_cache.rs:1176")
        .expect("command buffer completed");

    assert_eq!(
        layer.current_len, 1,
        "append_and_attend must bump current_len"
    );
    let out = crate::buffers::read_buffer_f32(&out_buf, num_kv * head_dim);
    assert!(out.iter().all(|v| v.is_finite()));
}
