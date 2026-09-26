//! Tests that need a prepared image's private items, and the test-only
//! seams the executor's own tests use to stage an image preparation
//! never produces.
//!
//! Preparation refuses the images below — a whole stack with no final
//! norm, an attention-residual stack without its exit, a hyper-connected
//! stack without its head reduction — so the exits that handle them are
//! reachable only by removing a part AFTER preparation. The seams do
//! exactly that and nothing else.

mod geometry;

use super::*;

impl PreparedOperands {
    /// This image with no final norm, as a component that declares none.
    pub(crate) fn without_final_norm_for_test(mut self) -> Self {
        self.final_norm = None;
        self
    }

    /// This image without its attention-residual exit reduction.
    pub(crate) fn without_attention_residual_exit_for_test(mut self) -> Self {
        self.attention_residual_exit = None;
        self
    }

    /// This image without its output head.
    pub(crate) fn without_output_for_test(mut self) -> Self {
        self.output = None;
        self
    }

    /// This image without its hyper-connection head reduction.
    pub(crate) fn without_hyper_connection_head_for_test(mut self) -> Self {
        self.hyper_connection_head = None;
        self
    }
}

/// Load one layer's attention-residual sites the way preparation does,
/// answering whether the layer carries them — the seam the exec tests use
/// to reach the refusals preparation's own ordering never presents.
pub(crate) fn attention_residual_sites_for_test(
    layer: &LayerPlan,
    declared: bool,
    hidden: usize,
    store: OperandSource<'_>,
) -> Result<bool, VindexError> {
    PreparedAttentionResidual::for_layer(layer, declared, hidden, store).map(|s| s.is_some())
}
