//! A gate index with a per-path feature mask, behind every FFN access trait.

use larql_vindex::{
    FeatureMeta, Fp4FfnAccess, GateIndex, GateLookup, NativeFfnAccess, PatchOverrides,
    QuantizedFfnAccess,
};
use ndarray::{Array1, Array2};

#[allow(unused_imports)]
use super::*;

/// Newtype wrapper that selectively reports availability flags as `false`,
/// forcing the WalkFfn dispatcher down a specific path. Data methods are
/// pure delegations; only the `has_*` booleans are masked.
///
/// Soundness: verified against every walk path in
/// `crates/larql-inference/src/vindex/walk_ffn/*.rs`. Each path gates on a
/// `has_*` flag at the dispatcher *and* early-exits on `Option::None` from
/// data methods, so masking is sufficient — we don't need to override data.
/// The unified `ffn_row_*` default impls also re-check `has_*` on `self`,
/// which is us, so the mask cascades through the row-level dispatch too.
#[derive(Default, Clone, Copy, Debug)]
pub(super) struct PathMask {
    pub(super) hide_fp4: bool,
    pub(super) hide_q4: bool,
    pub(super) hide_interleaved: bool,
    pub(super) hide_full_mmap: bool,
    pub(super) hide_q4k: bool,
    pub(super) hide_down_features: bool,
}

pub(super) struct MaskedGateIndex<'a> {
    pub(super) inner: &'a dyn GateIndex,
    pub(super) mask: PathMask,
}

impl<'a> GateLookup for MaskedGateIndex<'a> {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.inner.gate_knn(layer, residual, top_k)
    }
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.inner.feature_meta(layer, feature)
    }
    fn num_features(&self, layer: usize) -> usize {
        self.inner.num_features(layer)
    }

    fn gate_scores_batch(&self, l: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.inner.gate_scores_batch(l, x)
    }
    fn gate_scores_batch_backend(
        &self,
        l: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        self.inner.gate_scores_batch_backend(l, x, backend)
    }
    fn gate_knn_q4(
        &self,
        l: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        self.inner.gate_knn_q4(l, residual, top_k, backend)
    }
    fn gate_walk(
        &self,
        l: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        self.inner.gate_walk(l, residual, top_k)
    }
}

impl<'a> PatchOverrides for MaskedGateIndex<'a> {
    fn down_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.down_override(l, f)
    }
    fn up_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.up_override(l, f)
    }
    fn gate_override(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.gate_override(l, f)
    }
    fn has_overrides_at(&self, layer: usize) -> bool {
        self.inner.has_overrides_at(layer)
    }
}

impl<'a> NativeFfnAccess for MaskedGateIndex<'a> {
    fn has_down_features(&self) -> bool {
        !self.mask.hide_down_features && self.inner.has_down_features()
    }
    fn down_feature_vector(&self, l: usize, f: usize) -> Option<&[f32]> {
        self.inner.down_feature_vector(l, f)
    }
    fn down_layer_matrix(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.down_layer_matrix(l)
    }
    fn up_layer_matrix(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.up_layer_matrix(l)
    }
    fn has_full_mmap_ffn(&self) -> bool {
        !self.mask.hide_full_mmap && self.inner.has_full_mmap_ffn()
    }
    fn has_interleaved(&self) -> bool {
        !self.mask.hide_interleaved && self.inner.has_interleaved()
    }
    fn interleaved_gate(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_gate(l)
    }
    fn interleaved_up(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_up(l)
    }
    fn interleaved_down(&self, l: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.inner.interleaved_down(l)
    }
    fn prefetch_interleaved_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_layer(l)
    }
}

impl<'a> QuantizedFfnAccess for MaskedGateIndex<'a> {
    fn has_interleaved_q4(&self) -> bool {
        !self.mask.hide_q4 && self.inner.has_interleaved_q4()
    }
    fn interleaved_q4_gate(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_gate(l)
    }
    fn interleaved_q4_up(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_up(l)
    }
    fn interleaved_q4_down(&self, l: usize) -> Option<ndarray::Array2<f32>> {
        self.inner.interleaved_q4_down(l)
    }
    fn prefetch_interleaved_q4_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_q4_layer(l)
    }
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.inner.interleaved_q4_mmap_ref()
    }
    fn has_interleaved_kquant(&self) -> bool {
        !self.mask.hide_q4k && self.inner.has_interleaved_kquant()
    }
    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        self.inner.interleaved_kquant_mmap_ref()
    }
    fn prefetch_interleaved_kquant_layer(&self, l: usize) {
        self.inner.prefetch_interleaved_kquant_layer(l)
    }
    fn interleaved_kquant_layer_data(&self, l: usize) -> Option<[(&[u8], &str); 3]> {
        self.inner.interleaved_kquant_layer_data(l)
    }
    fn has_down_features_kquant(&self) -> bool {
        self.inner.has_down_features_kquant()
    }
    fn kquant_ffn_layer(&self, l: usize, c: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        self.inner.kquant_ffn_layer(l, c)
    }
    fn kquant_ffn_row_into(&self, l: usize, c: usize, f: usize, out: &mut [f32]) -> bool {
        self.inner.kquant_ffn_row_into(l, c, f, out)
    }
    fn kquant_ffn_row_dot(&self, l: usize, c: usize, f: usize, x: &[f32]) -> Option<f32> {
        self.inner.kquant_ffn_row_dot(l, c, f, x)
    }
    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner
            .kquant_ffn_row_scaled_add_via_cache(l, c, f, a, out)
    }
    fn kquant_ffn_row_scaled_add(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner.kquant_ffn_row_scaled_add(l, c, f, a, out)
    }
    fn kquant_down_feature_scaled_add(&self, l: usize, f: usize, a: f32, out: &mut [f32]) -> bool {
        self.inner.kquant_down_feature_scaled_add(l, f, a, out)
    }
    fn kquant_matmul_transb(
        &self,
        l: usize,
        c: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        self.inner.kquant_matmul_transb(l, c, x, x_rows, backend)
    }
}

impl<'a> Fp4FfnAccess for MaskedGateIndex<'a> {
    fn has_fp4_storage(&self) -> bool {
        !self.mask.hide_fp4 && self.inner.has_fp4_storage()
    }
    fn fp4_ffn_row_dot(&self, l: usize, c: usize, f: usize, x: &[f32]) -> Option<f32> {
        self.inner.fp4_ffn_row_dot(l, c, f, x)
    }
    fn fp4_ffn_row_scaled_add(
        &self,
        l: usize,
        c: usize,
        f: usize,
        a: f32,
        out: &mut [f32],
    ) -> bool {
        self.inner.fp4_ffn_row_scaled_add(l, c, f, a, out)
    }
    fn fp4_ffn_row_into(&self, l: usize, c: usize, f: usize, out: &mut [f32]) -> bool {
        self.inner.fp4_ffn_row_into(l, c, f, out)
    }
}
