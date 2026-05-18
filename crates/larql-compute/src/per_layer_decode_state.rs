//! Per-layer pre-attention residual + new K/V state buffer.
//!
//! Used by KvDispatch backends' decode-with-state variants so engines
//! can capture per-layer intermediates without re-running the
//! attention pass. Moved here from `larql-inference/src/kv_dispatch/mod.rs`
//! (ADR-0022 Step 3c) so the moved-down `kquant_forward::cached.rs`
//! can take `Option<&mut crate::PerLayerDecodeState>` parameters.
//! Step 3d will fold this back into `kv_dispatch/mod.rs` when the
//! KvDispatch trait moves.

use ndarray::Array2;

/// Captured per-layer state at a single decode step.
#[derive(Debug, Default, Clone)]
pub struct PerLayerDecodeState {
    /// Pre-attention residual entering each layer's attention block.
    /// Shape: `[num_layers][1, hidden]`.
    pub h_in_per_layer: Vec<Array2<f32>>,
    /// New K row appended this step, per layer.
    /// Shape: `[num_layers][1, kv_dim_for_layer]`.
    pub k_new_per_layer: Vec<Array2<f32>>,
    /// New V row appended this step, per layer.
    /// Shape: `[num_layers][1, kv_dim_for_layer]`.
    pub v_new_per_layer: Vec<Array2<f32>>,
}

impl PerLayerDecodeState {
    /// Pre-allocate vectors sized for `num_layers`. Caller should
    /// invoke this before passing `Some(&mut state)` to a decode
    /// step; backends `.push()` per layer.
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            h_in_per_layer: Vec::with_capacity(num_layers),
            k_new_per_layer: Vec::with_capacity(num_layers),
            v_new_per_layer: Vec::with_capacity(num_layers),
        }
    }

    /// `true` when the state was populated for every layer (one
    /// entry per layer in each vector). Backends MUST guarantee
    /// this on success, but engines may double-check.
    pub fn is_complete_for(&self, num_layers: usize) -> bool {
        self.h_in_per_layer.len() == num_layers
            && self.k_new_per_layer.len() == num_layers
            && self.v_new_per_layer.len() == num_layers
    }

    /// Drop all per-layer entries without freeing capacity. Use
    /// before re-passing to the next decode step.
    pub fn reset(&mut self) {
        self.h_in_per_layer.clear();
        self.k_new_per_layer.clear();
        self.v_new_per_layer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_capacity_returns_empty_vectors() {
        let s = PerLayerDecodeState::with_capacity(4);
        assert!(s.h_in_per_layer.is_empty());
        assert!(s.k_new_per_layer.is_empty());
        assert!(s.v_new_per_layer.is_empty());
        assert!(!s.is_complete_for(4));
    }

    #[test]
    fn is_complete_for_pins_per_layer_count() {
        let mut s = PerLayerDecodeState::with_capacity(2);
        s.h_in_per_layer.push(Array2::zeros((1, 4)));
        s.k_new_per_layer.push(Array2::zeros((1, 2)));
        s.v_new_per_layer.push(Array2::zeros((1, 2)));
        assert!(!s.is_complete_for(2));
        s.h_in_per_layer.push(Array2::zeros((1, 4)));
        s.k_new_per_layer.push(Array2::zeros((1, 2)));
        s.v_new_per_layer.push(Array2::zeros((1, 2)));
        assert!(s.is_complete_for(2));
    }

    #[test]
    fn reset_clears_without_freeing_capacity() {
        let mut s = PerLayerDecodeState::with_capacity(4);
        for _ in 0..4 {
            s.h_in_per_layer.push(Array2::zeros((1, 8)));
        }
        s.reset();
        assert!(s.h_in_per_layer.is_empty());
        assert!(s.h_in_per_layer.capacity() >= 4);
    }
}
