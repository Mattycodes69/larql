//! The [`LatentAttention`] slice of [`super::ModelArchitecture`].

use super::FeedForward;
use crate::config::{LatentNormSpec, MlaQueryForm, RoutedExpertForm};

/// Latent and linear attention variants: MLA, KDA and the DSA indexer.
pub trait LatentAttention: FeedForward {
    /// Whether this model uses MLA instead of standard GQA.
    fn uses_mla(&self) -> bool {
        false
    }

    /// Where this family's ROUTED experts run — at `hidden_size`, or
    /// behind a bottleneck of their own.
    ///
    /// Read from the DECLARATION: `routed_expert_hidden_size`'s
    /// PRESENCE, exactly the reference's `is not None`. Never inferred
    /// from which tensors a checkpoint ships — a `routed_expert_down_proj`
    /// may CONFIRM this form, it must never select it, or a checkpoint
    /// with a stray operand would be executed as a different model.
    ///
    /// The norm is nested inside the latent variant because the reference
    /// nests it: `if self.use_latent_moe:` encloses
    /// `if self.latent_moe_use_norm:`, so a config setting the flag
    /// without the width builds no norm at all. That state is
    /// unrepresentable here rather than merely discouraged.
    fn routed_expert_form(&self) -> RoutedExpertForm {
        match self.config().routed_expert_hidden_size {
            None => RoutedExpertForm::Uniform,
            Some(width) => RoutedExpertForm::Latent {
                width,
                norm: self.latent_moe_uses_norm().then(|| LatentNormSpec {
                    eps: self.routed_expert_norm_eps(),
                }),
            },
        }
    }

    /// Whether the latent routed branch normalises its weighted
    /// aggregate.
    ///
    /// TRUTHINESS, not presence: the reference reads
    /// `getattr(config, "latent_moe_use_norm", False)` and consumes it
    /// in a plain `if`, so absent, `null` and `false` are all "no norm"
    /// and differ only in what the plan reports as declared.
    fn latent_moe_uses_norm(&self) -> bool {
        self.config().latent_moe_use_norm.unwrap_or(false)
    }

    /// The epsilon `routed_expert_norm` runs at.
    ///
    /// The LAYER's eps, because the reference passes it explicitly:
    /// `KimiRMSNorm(self.moe_hidden_size, eps=config.rms_norm_eps)`.
    ///
    /// This is deliberately NOT [`Self::mla_q_a_norm_eps`]'s answer and
    /// not a shared "low-rank norm epsilon" accessor. The two
    /// neighbouring low-rank norms in this same family — `q_a_layernorm`
    /// and `kv_a_layernorm` — are constructed with no override and run
    /// at `KimiRMSNorm`'s class default `1e-6`, a factor of ten away.
    /// Two rungs in a row established that class default; this one
    /// inverts it, and the inversion is transcribed from the
    /// constructor rather than inherited from the neighbours.
    fn routed_expert_norm_eps(&self) -> f64 {
        self.norm_eps() as f64
    }

    /// MLA compressed KV dimension.
    fn kv_lora_rank(&self) -> usize {
        0
    }

    /// MLA Q compression rank.
    fn q_lora_rank(&self) -> usize {
        0
    }

    /// MLA KV down-projection key (compress).
    fn mla_kv_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA KV up-projection key (decompress).
    fn mla_kv_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q down-projection key (compress).
    fn mla_q_a_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// MLA Q up-projection key (decompress).
    fn mla_q_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// DS-V3 MLA: non-RoPE head dim (nope). Combined qk_head_dim = nope + rope.
    fn mla_qk_nope_head_dim(&self) -> Option<usize> {
        None
    }

    /// DS-V3 MLA: RoPE head dim portion.
    fn mla_qk_rope_head_dim(&self) -> Option<usize> {
        None
    }

    /// DS-V3 MLA: V head dim (after absorption may differ from qk dims).
    fn mla_v_head_dim(&self) -> Option<usize> {
        None
    }

    /// The epsilon MLA's QUERY-side norm (`q_a_layernorm`) runs at, when
    /// this family factorises its query and its reference fixes one.
    ///
    /// Its own accessor, deliberately not the KV one and deliberately not
    /// a shared "MLA low-rank norm epsilon". The two are equal today —
    /// `1e-6` — because they share a CAUSE: `KimiRMSNorm(width)` with no
    /// `eps` argument, twice, in the same `__init__`
    /// (`modeling_kimi_linear.py` L368 and L383). They do not share an
    /// AUTHORITY. A family that overrode one and left the other at the
    /// class default would be silently wrong under a shared accessor, and
    /// the coincidence would have been promoted to a contract nobody
    /// checked.
    ///
    /// `None` means **unjudged** — never "use the KV one", never "use the
    /// layer eps". A declared low-rank query with no judged epsilon must
    /// reach a named refusal, not a plausible number.
    fn mla_q_a_norm_eps(&self) -> Option<f64> {
        None
    }

    /// Which query this family's MLA layers build.
    ///
    /// Read from the DECLARATION — `q_lora_rank`'s presence, exactly the
    /// `is not None` the reference branches on (`modeling_kimi_linear.py`
    /// L364, L418) — and never from which tensors a checkpoint happens to
    /// ship. `q_proj` and `q_b_proj` share a row count and differ only in
    /// their column count, so an operand-sniffing default would pick the
    /// form from the very thing the form is supposed to decide.
    ///
    /// A declared `0` selects the factorised form, because `0 is not
    /// None`. Transcription, not endorsement: the geometry it then
    /// describes is degenerate, and closure refuses it by name.
    ///
    /// The epsilon is fetched here so that a family declaring the form
    /// without judging its norm produces a `LowRank` closure can refuse
    /// by name. An architecture may override this whole method; one that
    /// does still gets its own [`Self::mla_q_a_norm_eps`] honoured,
    /// because the declaration chooses the FORM and the override chooses
    /// the semantics WITHIN it.
    fn mla_query_form(&self) -> MlaQueryForm {
        match self.config().q_lora_rank {
            None => MlaQueryForm::Direct,
            Some(rank) => MlaQueryForm::LowRank {
                rank,
                norm_eps: self.mla_q_a_norm_eps(),
            },
        }
    }

    /// The epsilon MLA's latent norm (`kv_a_layernorm`) runs at, when this
    /// family's own reference code fixes one.
    ///
    /// `None` — the default — means **unjudged**, never "use the layer
    /// eps". The two are genuinely different numbers: Kimi Linear
    /// constructs `kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)` with
    /// no override, so it runs at `KimiRMSNorm.__init__`'s class default
    /// `1e-6` while every other norm in the same layer runs at
    /// `config.rms_norm_eps` (`1e-5`) — a factor of ten, on the one norm
    /// standing between the compressed cache and its decompression.
    ///
    /// This is an ARCHITECTURE fact, not a config fact: no checkpoint
    /// declares it, the family's modeling code does. So it is one of the
    /// legitimate overrides — a default that answered `Some(rms_norm_eps)`
    /// here would be the silent-wrong-serving shape the trait-default rule
    /// exists to prevent, and a family whose reference has not been read
    /// must reach an executor's named refusal rather than a plausible
    /// number.
    fn mla_kv_a_norm_eps(&self) -> Option<f64> {
        None
    }

    /// Which decay gate this family's KDA recurrence computes.
    ///
    /// `None` — the default — means **unjudged**, and an executor must
    /// refuse rather than pick a form. It is not a formality: Kimi Linear
    /// and GLM-5.3-Flash both declare
    /// `linear_attn_config.gate_lower_bound: -5.0`, and Kimi's reference
    /// reads it nowhere while GLM's applies it. A default here — either
    /// form — would silently serve one family's arithmetic to the other,
    /// with every shape closing. Measured cost of getting it wrong:
    /// relative 2.50e-2 on a real GLM layer's output, from a 2.8× error in
    /// the mean per-step decay (see [`KdaGateForm`]).
    ///
    /// The same architecture fact as [`Self::mla_kv_a_norm_eps`], for the
    /// same reason: no checkpoint declares which branch its reference
    /// takes, so the declaration lives with the family that owns the
    /// reference.
    fn kda_gate_form(&self) -> Option<crate::config::KdaGateForm> {
        None
    }

    //
    // GLM-5.2 (`glm_moe_dsa`) layers a sparse top-k token-selection
    // indexer on top of MLA: a small side-attention picks which cached
    // K/V entries the real MLA attention is even allowed to look at.
    // These seven methods expose the indexer's config facts and tensor
    // keys **for extraction/inspection only** — no compute path in
    // `larql` implements the indexer's top-k selection or consumes
    // these values. An architecture returning non-default answers here
    // has done nothing but publish inert facts; wiring the sparse
    // attention forward pass is a separate, unimplemented piece of work.

    /// Number of key/value entries the indexer selects per query
    /// (`index_topk` in config.json). `None` = not a DSA architecture.
    fn dsa_index_topk(&self) -> Option<usize> {
        None
    }

    /// Number of indexer attention heads (`index_n_heads`).
    fn dsa_index_n_heads(&self) -> Option<usize> {
        None
    }

    /// Per-head dimension of the indexer's own (small) attention
    /// (`index_head_dim`) — distinct from the main attention's head
    /// dimensions above.
    fn dsa_index_head_dim(&self) -> Option<usize> {
        None
    }

    /// Indexer query down-projection weight key.
    fn dsa_indexer_wq_b_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Indexer key projection weight key.
    fn dsa_indexer_wk_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Indexer key-norm weight key.
    fn dsa_indexer_k_norm_key(&self, _layer: usize) -> Option<String> {
        None
    }

    /// Indexer per-head selection-weight projection key (turns the
    /// indexer's own attention scores into the top-k selection weights).
    fn dsa_indexer_weights_proj_key(&self, _layer: usize) -> Option<String> {
        None
    }
}
