//! Per-layer KV access and handle rebuilding for the standard engine.

use larql_inference::{EngineBackend, KvHandle};
use ndarray::Array2;

#[allow(unused_imports)]
use super::*;

impl larql_inference::kv_engine::PerLayerKvAccess for StandardEngine {
    fn layer_count(&self) -> Option<usize> {
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => self.handles.as_ref().map(Vec::len),
            PrefillDispatchMode::Coarse => None,
        }
    }

    fn logical_next_position(&self) -> usize {
        self.abs_position
    }

    fn resident_rows(&self, layer: usize) -> Option<usize> {
        self.layer_handles()?.get(layer).map(|h| h.cached_len())
    }

    fn read_layer_kv(&self, layer: usize) -> Option<(Array2<f32>, Array2<f32>)> {
        let BackendSlot::Sync(backend) = &self.backend else {
            return None;
        };
        backend.read_kv_to_host(self.layer_handles()?.get(layer)?)
    }

    fn excise_kv_rows(
        &mut self,
        layer: usize,
        rows: std::ops::Range<usize>,
    ) -> Option<larql_inference::kv_engine::ExcisedKvRows> {
        if rows.start >= rows.end {
            return None;
        }
        let BackendSlot::Sync(backend) = &self.backend else {
            // Async dispatch has no host-readback surface here.
            return None;
        };
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => {}
            PrefillDispatchMode::Coarse => return None,
        }
        let handles = self.handles.as_mut()?;
        let handle = handles.get(layer)?;
        let (k, v) = backend.read_kv_to_host(handle)?;
        if rows.end > k.nrows() {
            return None;
        }
        // The map must already agree with the cache, or the positions
        // this hands back describe rows that were never there.
        let map = self.row_positions.layer_mut(layer)?;
        if map.len() != k.nrows() {
            return None;
        }

        let removed = larql_inference::kv_engine::ExcisedRows {
            k: k.slice(ndarray::s![rows.clone(), ..]).to_owned(),
            v: v.slice(ndarray::s![rows.clone(), ..]).to_owned(),
        };
        let kept: Vec<usize> = (0..k.nrows()).filter(|i| !rows.contains(i)).collect();
        // Rebuild first: a failed rebuild must leave the map describing
        // the cache that is still there.
        let rebuilt = rebuild_handle(backend.as_ref(), layer, &k, &v, &kept)?;
        let positions = map.excise(rows).ok()?;
        handles[layer] = rebuilt;
        larql_inference::kv_engine::ExcisedKvRows::new(removed, positions)
    }

    fn replace_layer_kv(
        &mut self,
        layer: usize,
        k: &Array2<f32>,
        v: &Array2<f32>,
        positions: &[u64],
    ) -> bool {
        if k.nrows() != v.nrows() || k.ncols() != v.ncols() || positions.len() != k.nrows() {
            return false;
        }
        let candidate = larql_inference::kv_row_positions::LayerRowPositions::from_positions(
            positions.to_vec(),
        );
        if candidate.check_ascending().is_err() {
            return false;
        }
        let BackendSlot::Sync(backend) = &self.backend else {
            return false;
        };
        let Some(PrefillDispatchMode::PerLayer) = self.prefill_mode else {
            return false;
        };
        let Some(handles) = self.handles.as_mut() else {
            return false;
        };
        if layer >= handles.len() || self.row_positions.layer(layer).is_none() {
            return false;
        }
        let all: Vec<usize> = (0..k.nrows()).collect();
        match rebuild_handle(backend.as_ref(), layer, k, v, &all) {
            Some(rebuilt) => {
                handles[layer] = rebuilt;
                // Both halves land together, after everything that could
                // have failed already has.
                *self
                    .row_positions
                    .layer_mut(layer)
                    .expect("layer presence checked above") = candidate;
                true
            }
            None => false,
        }
    }

    fn set_logical_next_position(&mut self, position: usize) -> bool {
        // Rows are not re-rotated by this, so moving the next position
        // *behind* a resident row would leave the cache holding a
        // position it claims has not happened yet — and the next append
        // could not ascend from it. Refuse rather than record it.
        if let Some(max) = self.row_positions.max_position() {
            if (position as u64) <= max {
                return false;
            }
        }
        self.abs_position = position;
        true
    }

    fn row_positions(&self) -> Option<&larql_inference::kv_row_positions::KvRowPositions> {
        matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer))
            .then_some(&self.row_positions)
    }

    fn clip_layer_to_logical_window(&mut self, layer: usize, window: u64) -> Option<usize> {
        let BackendSlot::Sync(backend) = &self.backend else {
            return None;
        };
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => {}
            PrefillDispatchMode::Coarse => return None,
        }
        let keep = self
            .row_positions
            .layer(layer)?
            .rows_within_window(self.abs_position as u64, window);
        let handles = self.handles.as_mut()?;
        let handle = handles.get(layer)?;
        let (k, v) = backend.read_kv_to_host(handle)?;
        let map = self.row_positions.layer_mut(layer)?;
        if map.len() != k.nrows() {
            return None;
        }
        if keep.len() == map.len() {
            return Some(0);
        }
        let dropped = map.len() - keep.len();
        let rebuilt = rebuild_handle(backend.as_ref(), layer, &k, &v, &keep)?;
        map.retain_rows(&keep).ok()?;
        handles[layer] = rebuilt;
        Some(dropped)
    }

    fn splice_kv_rows(
        &mut self,
        layer: usize,
        at: usize,
        rows: &larql_inference::kv_engine::ExcisedKvRows,
    ) -> bool {
        let BackendSlot::Sync(backend) = &self.backend else {
            return false;
        };
        let Some(PrefillDispatchMode::PerLayer) = self.prefill_mode else {
            return false;
        };
        let Some(handles) = self.handles.as_mut() else {
            return false;
        };
        let Some(handle) = handles.get(layer) else {
            return false;
        };
        let Some((k, v)) = backend.read_kv_to_host(handle) else {
            return false;
        };
        let kv = rows.kv();
        if at > k.nrows() || kv.k.ncols() != k.ncols() {
            return false;
        }
        let Some(map) = self.row_positions.layer_mut(layer) else {
            return false;
        };
        if map.len() != k.nrows() {
            return false;
        }
        // Splice into a copy first. A splice at the wrong offset leaves
        // positions out of order, and finding that out after the handle
        // has been rebuilt would mean the refusal came too late.
        let mut candidate = map.clone();
        if candidate.splice(at, rows.logical_positions()).is_err() {
            return false;
        }

        let total = k.nrows() + rows.len();
        let kv_dim = k.ncols();
        let mut rebuilt = backend.alloc_kv_buffer(layer, total, kv_dim);
        let push = |handle: &mut KvHandle, kr: &[f32], vr: &[f32], pos: usize| {
            backend.append_kv(handle, kr, vr, pos);
        };
        for i in 0..at {
            push(&mut rebuilt, row_of(&k, i), row_of(&v, i), i);
        }
        for i in 0..rows.len() {
            push(&mut rebuilt, row_of(&kv.k, i), row_of(&kv.v, i), at + i);
        }
        for i in at..k.nrows() {
            push(&mut rebuilt, row_of(&k, i), row_of(&v, i), rows.len() + i);
        }
        handles[layer] = rebuilt;
        *map = candidate;
        true
    }
}

/// Rebuild a layer handle holding only `kept` rows of `(k, v)`.
pub(super) fn rebuild_handle(
    backend: &dyn EngineBackend,
    layer: usize,
    k: &Array2<f32>,
    v: &Array2<f32>,
    kept: &[usize],
) -> Option<KvHandle> {
    let mut rebuilt = backend.alloc_kv_buffer(layer, kept.len().max(1), k.ncols());
    for (physical, &i) in kept.iter().enumerate() {
        backend.append_kv(&mut rebuilt, row_of(k, i), row_of(v, i), physical);
    }
    Some(rebuilt)
}

pub(super) fn row_of(a: &Array2<f32>, i: usize) -> &[f32] {
    a.row(i)
        .to_slice()
        .expect("host-read K/V rows are contiguous")
}
