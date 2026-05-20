//! RsStore — per-layer residual buffer for MarkovResidualEngine.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

/// Per-layer pre-attention residuals for all stored positions.
///
/// **Hot K/V caching (W2, 2026-05-17 night):** `hot_kv`, when `Some`,
/// caches the K/V projection of `stored` per layer. The engine's
/// contract says K/V is *derivable from residuals* — it does not say
/// "recomputed every step." Caching avoids ~17k K/V row projections
/// per token (W7 measured ~80% of decode time wasted on this) while
/// preserving the residual-stream invariant: drop `hot_kv` and the
/// next step recomputes from `stored`. Bit-equivalent to the
/// non-cached path under fixed RoPE positions.
///
/// Invariants when `hot_kv = Some(kv)`:
///   - `kv.len() == stored.len()` (one entry per layer)
///   - `kv[l].0.shape()[0] == stored[l].shape()[0]` for every `l`
///   - row `i` of `kv[l]` corresponds to row `i` of `stored[l]` at
///     RoPE position `next_position - stored[l].shape()[0] + i`
pub struct RsStore {
    /// Per-layer residual stream. **Possibly over-allocated**: with W8.2,
    /// the dispatch hot path pre-allocates `stored[l]` to a doubling
    /// capacity and only the first `hot_len` rows are logically valid.
    /// Readers that want the row count **must** use [`Self::hot_len`],
    /// not `stored[l].shape()[0]`. Non-dispatch paths (CPU walk,
    /// rs_extend_from_checkpoint_*) still write narrow arrays where
    /// `hot_len == shape()[0]`.
    pub stored: Vec<Array2<f32>>,
    /// Per-layer cold residuals. **Doubling-capacity** as of 2026-05-19
    /// (audit fix): `cold_residuals[l].shape()[0]` is the buffer
    /// capacity, not the logical row count. Use [`Self::cold_len`].
    /// Was reallocated on every overflow step (O(N) per step,
    /// O(N²) total); now geometrically-grown via
    /// [`Self::append_cold_overflow`].
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    /// Same doubling-capacity contract as `cold_residuals`. K and V
    /// arrays in each `(k, v)` pair share the same capacity.
    pub cold_kv: Option<Vec<SharedKV>>,
    /// Per-layer cached K/V for the hot tier. See struct doc for
    /// the invariants. `None` means the decode step must recompute
    /// from `stored` (the legacy path). Same over-allocation rule as
    /// `stored`: `hot_kv[l].0.shape()[0]` is capacity, not logical
    /// length — use `hot_len`.
    pub hot_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
    /// Logical row count of `stored` and `hot_kv`. See field docs above
    /// for the over-allocation contract.
    pub hot_len: usize,
    /// Logical row count of `cold_residuals` and `cold_kv`. Same
    /// doubling-capacity contract as `hot_len`. Default 0 (no cold
    /// rows). Maintained by [`Self::append_cold_overflow`].
    pub cold_len: usize,
}

impl RsStore {
    pub fn memory_bytes(&self) -> usize {
        // W8.2: count only the logically valid rows (hot_len), not the
        // pre-allocated capacity (`stored[l].shape()[0]`). Otherwise
        // `engine.memory_bytes()` would overstate by the doubling slack.
        // 2026-05-19 audit fix: same logic for cold_residuals / cold_kv
        // — use `cold_len`, not `shape()[0]`.
        let rows = self.hot_len;
        let cold_rows = self.cold_len;
        let hot: usize = self.stored.iter().map(|s| rows * s.shape()[1] * 4).sum();
        let cold_res: usize = self
            .cold_residuals
            .as_ref()
            .map(|c| c.iter().map(|s| cold_rows * s.shape()[1] * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| {
                kv.iter()
                    .map(|(k, v)| cold_rows * (k.shape()[1] + v.shape()[1]) * 4)
                    .sum()
            })
            .unwrap_or(0);
        let hot_kv: usize = self
            .hot_kv
            .as_ref()
            .map(|kv| {
                kv.iter()
                    .map(|(k, v)| (k.shape()[1] + v.shape()[1]) * rows * 4)
                    .sum()
            })
            .unwrap_or(0);
        hot + cold_res + cold_kv + hot_kv
    }

    pub fn cold_bytes(&self) -> usize {
        let cold_rows = self.cold_len;
        let cold_res: usize = self
            .cold_residuals
            .as_ref()
            .map(|c| c.iter().map(|s| cold_rows * s.shape()[1] * 4).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| {
                kv.iter()
                    .map(|(k, v)| cold_rows * (k.shape()[1] + v.shape()[1]) * 4)
                    .sum()
            })
            .unwrap_or(0);
        cold_res + cold_kv
    }

    /// Geometric-capacity append into cold_residuals (always) and
    /// cold_kv (if `evicted_kv` is `Some`). 2026-05-19 audit fix:
    /// replaces the prior `Array2::zeros((c_old + c_new, ...))` flow
    /// in `decode_step_via_dispatch` and `compute.rs::rs_decode_step*`
    /// — that path was O(N) per step and O(N²) total across a
    /// long decode. This helper grows the underlying buffer in
    /// doubling steps so total cost is amortised O(1) per row added.
    ///
    /// All overflow vectors must be the same row count `c_new` (the
    /// layer-uniform eviction property; see `clip_layer`).
    pub(crate) fn append_cold_overflow(
        &mut self,
        overflow: Vec<Array2<f32>>,
        evicted_kv: Option<Vec<SharedKV>>,
    ) {
        let c_new = overflow.first().map_or(0, |c| c.shape()[0]);
        if c_new == 0 {
            return;
        }
        let c_old = self.cold_len;
        let new_len = c_old + c_new;

        // Lazily allocate cold buffers on first overflow.
        if self.cold_residuals.is_none() {
            let buffers: Vec<Array2<f32>> = overflow
                .iter()
                .map(|o| {
                    let cols = o.shape()[1];
                    let cap = c_new.next_power_of_two().max(8);
                    let mut buf = Array2::<f32>::zeros((cap, cols));
                    buf.slice_mut(s![..c_new, ..]).assign(o);
                    buf
                })
                .collect();
            self.cold_residuals = Some(buffers);
        } else if let Some(cold) = self.cold_residuals.as_mut() {
            for (layer, src) in overflow.iter().enumerate() {
                let cap = cold[layer].shape()[0];
                if cap < new_len {
                    let cols = cold[layer].shape()[1];
                    let new_cap = (cap * 2).max(new_len).next_power_of_two().max(8);
                    let mut grown = Array2::<f32>::zeros((new_cap, cols));
                    grown
                        .slice_mut(s![..c_old, ..])
                        .assign(&cold[layer].slice(s![..c_old, ..]));
                    cold[layer] = grown;
                }
                cold[layer].slice_mut(s![c_old..new_len, ..]).assign(src);
            }
        }

        // K/V cold tier is optional (lossy codec engines invalidate it
        // on overflow). Mirror the same growth logic when provided.
        if let Some(evicted) = evicted_kv {
            if self.cold_kv.is_none() {
                let buffers: Vec<SharedKV> = evicted
                    .into_iter()
                    .map(|(k_new, v_new)| {
                        let kv_dim = k_new.shape()[1];
                        let cap = c_new.next_power_of_two().max(8);
                        let mut k_buf = Array2::<f32>::zeros((cap, kv_dim));
                        let mut v_buf = Array2::<f32>::zeros((cap, kv_dim));
                        k_buf.slice_mut(s![..c_new, ..]).assign(&k_new);
                        v_buf.slice_mut(s![..c_new, ..]).assign(&v_new);
                        (k_buf, v_buf)
                    })
                    .collect();
                self.cold_kv = Some(buffers);
            } else if let Some(cold_kv) = self.cold_kv.as_mut() {
                for (layer, (k_new, v_new)) in evicted.into_iter().enumerate() {
                    let cap = cold_kv[layer].0.shape()[0];
                    if cap < new_len {
                        let kv_dim = cold_kv[layer].0.shape()[1];
                        let new_cap = (cap * 2).max(new_len).next_power_of_two().max(8);
                        let mut grown_k = Array2::<f32>::zeros((new_cap, kv_dim));
                        let mut grown_v = Array2::<f32>::zeros((new_cap, kv_dim));
                        grown_k
                            .slice_mut(s![..c_old, ..])
                            .assign(&cold_kv[layer].0.slice(s![..c_old, ..]));
                        grown_v
                            .slice_mut(s![..c_old, ..])
                            .assign(&cold_kv[layer].1.slice(s![..c_old, ..]));
                        cold_kv[layer] = (grown_k, grown_v);
                    }
                    cold_kv[layer]
                        .0
                        .slice_mut(s![c_old..new_len, ..])
                        .assign(&k_new);
                    cold_kv[layer]
                        .1
                        .slice_mut(s![c_old..new_len, ..])
                        .assign(&v_new);
                }
            }
        } else {
            // Lossy codec invalidates cold_kv on every overflow.
            self.cold_kv = None;
        }

        self.cold_len = new_len;
    }

    /// Read the logical cold-residual slice for `layer`. Slices to
    /// `cold_len` so callers see the valid rows, not the doubling-
    /// allocated capacity.
    pub fn cold_residual_view(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.cold_residuals
            .as_ref()
            .map(|c| c[layer].slice(s![..self.cold_len, ..]))
    }

    /// Read the logical cold-K/V slice for `layer`.
    pub fn cold_kv_view(
        &self,
        layer: usize,
    ) -> Option<(ndarray::ArrayView2<'_, f32>, ndarray::ArrayView2<'_, f32>)> {
        self.cold_kv.as_ref().map(|kv| {
            (
                kv[layer].0.slice(s![..self.cold_len, ..]),
                kv[layer].1.slice(s![..self.cold_len, ..]),
            )
        })
    }

    pub fn window_tokens(&self) -> usize {
        // W8.2: use the logical-length counter. `stored[l].shape()[0]`
        // may be the doubling-allocated capacity.
        self.hot_len
    }

    pub(crate) fn clip_layer(&mut self, layer: usize, cold: &mut Vec<Array2<f32>>) {
        let window = match self.max_window {
            Some(w) => w,
            None => return,
        };
        // W8.2: use the logical row count, not the pre-allocated
        // capacity. The new layouts are slice-views into the
        // (possibly oversized) underlying Array2.
        let rows = self.hot_len;
        let cols = self.stored[layer].shape()[1];
        if rows <= window {
            cold.push(Array2::zeros((0, cols)));
            return;
        }
        let start = rows - window;
        let s_logical = self.stored[layer].slice(s![..rows, ..]);
        cold.push(s_logical.slice(s![..start, ..]).to_owned());
        self.stored[layer] = s_logical.slice(s![start.., ..]).to_owned();

        // Clip hot_kv consistently — same `start..` slice keeps the K/V
        // cache aligned with the (now smaller) hot residual buffer. The
        // evicted K/V rows are absorbed into the cold tier by the
        // caller via [`take_evicted_hot_kv`].
        if let Some(kv) = self.hot_kv.as_mut() {
            let (k, v) = &kv[layer];
            let k_logical = k.slice(s![..rows, ..]);
            let v_logical = v.slice(s![..rows, ..]);
            kv[layer] = (
                k_logical.slice(s![start.., ..]).to_owned(),
                v_logical.slice(s![start.., ..]).to_owned(),
            );
        }
        // NB: do NOT update `self.hot_len` here — `clip_layer` runs in
        // a per-layer loop and resetting hot_len mid-loop makes
        // subsequent layers see `rows == window` and skip their clip.
        // Callers must reset `hot_len` to `window` AFTER the loop.
    }

    /// Reset the logical row count after a window-clip loop. Call once
    /// after `clip_layer` has been invoked for every layer.
    pub(crate) fn finalise_hot_len_after_clip(&mut self) {
        if let Some(w) = self.max_window {
            self.hot_len = self.hot_len.min(w);
        }
    }

    /// Slice the top `n` rows of every layer's `hot_kv` into a new
    /// `Vec<SharedKV>`. Used during prefill-time overflow to seed
    /// `cold_kv` directly from cached projections instead of calling
    /// `recompute_kv` on the evicted residuals (which was wasteful —
    /// those K/V rows were *just computed* during prefill).
    ///
    /// Returns `None` if `hot_kv` is `None` or every layer's slice
    /// would be empty. The function does **not** mutate `hot_kv`;
    /// the in-place clip in [`clip_layer`] already removes the top
    /// rows from each layer's hot K/V slot.
    pub(crate) fn snapshot_evicted_hot_kv(
        original_hot_kv: &[SharedKV],
        keep_from: &[usize],
    ) -> Option<Vec<SharedKV>> {
        if original_hot_kv.is_empty() || keep_from.iter().all(|&n| n == 0) {
            return None;
        }
        // W8.2 note: `keep_from[layer]` is the per-layer evict-count,
        // which the caller derives from `stored[l].shape()[0]
        // .saturating_sub(window)` pre-clip. With over-allocation that
        // computation is wrong (it'd evict slack). Callers must pass
        // `hot_len.saturating_sub(window)` instead. Slicing `..start`
        // here is safe either way since the slice respects bounds.
        let evicted: Vec<SharedKV> = original_hot_kv
            .iter()
            .zip(keep_from.iter())
            .map(|((k, v), &start)| {
                (
                    k.slice(s![..start, ..]).to_owned(),
                    v.slice(s![..start, ..]).to_owned(),
                )
            })
            .collect();
        Some(evicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(num_layers: usize, seq_len: usize, hidden: usize) -> RsStore {
        let stored = (0..num_layers)
            .map(|_| Array2::from_elem((seq_len, hidden), 1.0f32))
            .collect();
        RsStore {
            stored,
            cold_residuals: None,
            cold_kv: None,
            cold_len: 0,
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: None,
            hot_len: seq_len,
        }
    }

    // ── memory_bytes ──────────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_hot_only() {
        let store = make_store(2, 5, 16);
        // 2 layers × 5 rows × 16 cols × 4 bytes
        assert_eq!(store.memory_bytes(), 2 * 5 * 16 * 4);
    }

    #[test]
    fn memory_bytes_empty_store_is_zero() {
        let store = make_store(0, 0, 16);
        assert_eq!(store.memory_bytes(), 0);
    }

    #[test]
    fn cold_bytes_zero_when_no_cold() {
        let store = make_store(2, 5, 16);
        assert_eq!(store.cold_bytes(), 0);
    }

    // ── window_tokens ─────────────────────────────────────────────────────────

    #[test]
    fn window_tokens_matches_stored_rows() {
        let store = make_store(3, 7, 8);
        assert_eq!(store.window_tokens(), 7);
    }

    #[test]
    fn window_tokens_zero_for_empty_store() {
        let store = make_store(0, 0, 8);
        assert_eq!(store.window_tokens(), 0);
    }

    // ── clip_layer ────────────────────────────────────────────────────────────

    #[test]
    fn clip_layer_no_window_is_noop() {
        let mut store = make_store(1, 10, 4);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        // No window → nothing clipped, cold stays empty
        assert!(cold.is_empty());
        assert_eq!(
            store.stored[0].shape()[0],
            10,
            "hot store should be unchanged"
        );
    }

    #[test]
    fn clip_layer_within_window_pushes_empty_cold() {
        let mut store = make_store(1, 4, 4);
        store.max_window = Some(8); // window larger than rows
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        // rows (4) <= window (8) → empty cold pushed
        assert_eq!(cold.len(), 1);
        assert_eq!(cold[0].shape()[0], 0, "cold should be empty sentinel");
        assert_eq!(store.stored[0].shape()[0], 4, "hot store unchanged");
    }

    #[test]
    fn clip_layer_excess_rows_moved_to_cold() {
        let mut store = make_store(1, 10, 4);
        store.max_window = Some(3);
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        // 10 rows, window=3 → 7 rows clipped to cold, 3 remain hot
        assert_eq!(cold[0].shape()[0], 7);
        assert_eq!(store.stored[0].shape()[0], 3);
    }

    #[test]
    fn clip_layer_exactly_at_window_no_cold() {
        let mut store = make_store(1, 5, 4);
        store.max_window = Some(5); // exactly at limit
        let mut cold = Vec::new();
        store.clip_layer(0, &mut cold);
        assert_eq!(cold[0].shape()[0], 0, "at exactly window size: empty cold");
        assert_eq!(store.stored[0].shape()[0], 5, "hot store intact");
    }
}
