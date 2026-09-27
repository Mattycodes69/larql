//! Path enumeration, per-position diffs and the dual-FFN comparator.

use larql_inference::FfnBackend;
use larql_vindex::{FfnRowAccess, VectorIndex};
use ndarray::Array2;
use std::cell::RefCell;

#[allow(unused_imports)]
use super::*;

#[derive(Clone, Debug)]
pub(super) struct PathSpec {
    /// Display name; matches the dispatch trace label prefix.
    pub(super) name: &'static str,
    /// Mask to apply on top of the live vindex flags.
    pub(super) mask: PathMask,
    /// Sparse-K config (`Some`) or dense ladder (`None`).
    pub(super) sparse_k: Option<usize>,
    /// Assertion bound for this path. Set explicitly per spec — for paths
    /// whose precision is fixed by the path itself (e.g. `interleaved` is
    /// always f32; `interleaved_kquant` is always Q4K), this is hardcoded to
    /// the right bucket. For `sparse`, which dispatches through the
    /// unified `ffn_row_*` chain and walks whatever data the vindex
    /// carries, the bucket is determined by `index.primary_storage_bucket()`.
    pub(super) bound: PathBound,
}

/// Probe the live vindex and return the paths that are actually testable.
/// Q4 metal/CPU and fp4 paths only show up when the corresponding flag is
/// set on the underlying index — skip them silently otherwise.
pub(super) fn enumerate_paths(index: &VectorIndex) -> Vec<PathSpec> {
    let mut out = Vec::new();

    // sparse:* — config-forced walk_ffn_sparse over whatever the unified
    // ffn_row_* dispatch picks. Always available since it doesn't depend
    // on any has_* flag. Bucket is *vindex-dependent*: on an f16 vindex
    // sparse walks f32 features (Exact); on a Q4K vindex sparse walks
    // Q4K via kquant_ffn_row_dot (Quantized). primary_storage_bucket()
    // encapsulates that mapping so future storage formats inherit it.
    out.push(PathSpec {
        name: "sparse",
        mask: PathMask::default(),
        sparse_k: Some(usize::MAX),
        bound: bound_for_bucket(index.primary_storage_bucket()),
    });

    // fp4_storage:sparse — only if the vindex carries FP4 storage.
    if index.has_fp4_storage() {
        out.push(PathSpec {
            name: "fp4_storage",
            mask: PathMask {
                // Don't mask anything: fp4 fires from the dense ladder
                // when has_fp4_storage()=true, which is what we want.
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_FP4,
        });
    }

    // interleaved_q4 — requires a backend with q4 support; skipped in v1
    // since this example doesn't pass a backend. Documented for clarity:
    if index.has_interleaved_q4() {
        eprintln!(
            "[walk_path_audit] interleaved_q4 path skipped (requires Metal/Q4 backend; not wired in v1)"
        );
    }

    // interleaved (f32) — mask fp4 + q4 above it. Always Exact: this
    // path reads f32 interleaved data directly, regardless of what
    // other storage variants the vindex carries.
    if index.has_interleaved() {
        out.push(PathSpec {
            name: "interleaved",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // full_mmap — mask everything above it. Always Exact: walks f32
    // mmap'd gate/up/down.
    if index.has_full_mmap_ffn() {
        out.push(PathSpec {
            name: "full_mmap",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // interleaved_kquant:dequant — mask everything above it. Always
    // Quantized: dequants Q4K bytes per layer.
    if index.has_interleaved_kquant() {
        out.push(PathSpec {
            name: "interleaved_kquant",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                hide_full_mmap: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_QUANTIZED,
        });
    }

    // exact — mask everything above it. Needs has_down_features=true.
    // Always Exact: gate/up from safetensors (f32), down from features
    // (f32).
    if index.has_down_features() {
        out.push(PathSpec {
            name: "exact",
            mask: PathMask {
                hide_fp4: true,
                hide_q4: true,
                hide_interleaved: true,
                hide_full_mmap: true,
                hide_q4k: true,
                ..PathMask::default()
            },
            sparse_k: None,
            bound: BOUND_EXACT,
        });
    }

    // weights_fallback:* is intentionally not in this audit. It's the
    // no-vindex-data corner case (extract_level = Browse without pinned
    // weights), and at any finite K it's measuring approximation quality
    // ("how good is K=N sparse walk vs dense matmul") rather than path
    // equivalence ("do the walk paths agree with dense matmul"). Those
    // are different questions; mixing them muddies the audit headline.
    // The K-sweep belongs in a separate `walk_approximation_quality`
    // example.

    out
}

// ── Diff plumbing ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PositionDiff {
    pub(super) l2: f32,
    pub(super) cos: f32,
    pub(super) max_abs: f32,
    /// ‖primary‖ at this position. Carried so downstream can compute
    /// `rel_L2 = L2 / max(primary_norm, REL_L2_NORM_EPS)` without
    /// re-walking the array. Diagnostic-only; not directly asserted on.
    pub(super) primary_norm: f32,
}

/// Per-(layer, position) diff between primary and secondary. Last-position
/// diff is what walk_correctness reports; we capture every position so we
/// can report worst-case across the whole prompt.
pub(super) fn diff_all_positions(a: &Array2<f32>, b: &Array2<f32>) -> Vec<PositionDiff> {
    let seq_len = a.shape()[0];
    let hidden = a.shape()[1];
    let mut out = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let mut l2_sq = 0.0f32;
        let mut max_abs = 0.0f32;
        let mut dot = 0.0f32;
        let mut a_norm_sq = 0.0f32;
        let mut b_norm_sq = 0.0f32;
        for j in 0..hidden {
            let ai = a[[s, j]];
            let bi = b[[s, j]];
            let d = ai - bi;
            l2_sq += d * d;
            let abs_d = d.abs();
            if abs_d > max_abs {
                max_abs = abs_d;
            }
            dot += ai * bi;
            a_norm_sq += ai * ai;
            b_norm_sq += bi * bi;
        }
        let an = a_norm_sq.sqrt();
        let bn = b_norm_sq.sqrt();
        let cos = if an > 0.0 && bn > 0.0 {
            dot / (an * bn)
        } else {
            0.0
        };
        out.push(PositionDiff {
            l2: l2_sq.sqrt(),
            cos,
            max_abs,
            primary_norm: an,
        });
    }
    out
}

/// DualFfn that records, per layer, the full `[seq_len]` diff vector. The
/// primary drives the residual stream onward (so this measures secondary
/// drift relative to the dense reference at the *same* input residual).
pub(super) struct DualFfn<'a> {
    pub(super) primary: &'a dyn FfnBackend,
    pub(super) secondary: &'a dyn FfnBackend,
    /// Vec<(layer, per-position diffs)> in the order calls arrive.
    pub(super) diffs: RefCell<Vec<(usize, Vec<PositionDiff>)>>,
}

impl<'a> FfnBackend for DualFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let p_out = self.primary.forward(layer, x);
        let s_out = self.secondary.forward(layer, x);
        let positions = diff_all_positions(&p_out, &s_out);
        self.diffs.borrow_mut().push((layer, positions));
        p_out
    }
    fn name(&self) -> &str {
        "dual"
    }
}
