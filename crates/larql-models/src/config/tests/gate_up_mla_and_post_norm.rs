use super::*;

/// **A family override of `activation()` decides the kernel, not the
/// config the family ignored.**
///
/// StarCoder2 declares no `hidden_act` and overrides `activation()` to
/// tanh-GELU. The first version of `gate_up_is_gelu_tanh` answered from
/// the DECLARATION, so it read `Absent`, fell to SiLU, and quietly
/// replaced a family's own judgment — the same shape of defect this rung
/// removes, one level up, and it was caught by the walk-vs-dense parity
/// test rather than by design. The declaration decides whether to REFUSE;
/// `activation()` supplies the answer.
#[test]
fn a_family_override_decides_the_gate_up_kernel() {
    let arch = crate::detect_from_json(&serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "vocab_size": 32,
    }));
    assert_eq!(
        arch.activation_declaration(),
        ActivationDeclaration::Absent,
        "the fixture must declare nothing, or this proves the wrong thing"
    );
    assert_eq!(arch.activation(), Activation::GeluTanh, "the family's own");
    assert!(
        arch.gate_up_is_gelu_tanh(),
        "the kernel family must follow the override, not the absent declaration"
    );
}

/// The declared name survives into the refusal, so a reader is told WHICH
/// declaration was refused rather than that some declaration was.
#[test]
fn a_declaration_can_name_itself_for_a_refusal_message() {
    assert_eq!(
        arch_declaring(Some("relu2"))
            .activation_declaration()
            .declared_name(),
        Some("relu2")
    );
    assert_eq!(
        arch_declaring(Some("situ"))
            .activation_declaration()
            .declared_name(),
        Some("situ")
    );
    assert_eq!(
        arch_declaring(None)
            .activation_declaration()
            .declared_name(),
        None
    );
}

#[test]
fn an_undeclared_q_lora_rank_is_the_direct_query_form() {
    assert_eq!(
        arch_with_q_lora(None).mla_query_form(),
        MlaQueryForm::Direct,
        "absence is the reference's own default (`q_lora_rank: Optional[int] = None`)"
    );
}

#[test]
fn a_declared_q_lora_rank_selects_the_factorised_form() {
    let form = arch_with_q_lora(Some(1536)).mla_query_form();
    assert_eq!(form.rank(), Some(1536));
    assert!(form.is_low_rank());
}

/// **The adversarial control the freeze named.** `q_lora_rank: 0`
/// selects the factorised form, because the reference branches on `is
/// not None` and `0 is not None`.
///
/// Asserted in the SAME test as `activation_situ_beta`'s opposite rule,
/// where the same checkpoint's `beta or 1.0` turns a declared zero into
/// one. Two adjacent fields of one config, two opposite treatments of
/// zero — and the risk is a shared intuition, not a shared identifier,
/// so the two rules are pinned side by side where a reader meets both.
#[test]
fn zero_selects_the_form_here_and_becomes_one_over_in_situ() {
    let form = arch_with_q_lora(Some(0)).mla_query_form();
    assert!(
        form.is_low_rank(),
        "`0 is not None`: a declared zero selects the factorised query"
    );
    assert_eq!(form.rank(), Some(0), "and the rank is carried verbatim");

    let mut config = base_config();
    config.hidden_act = Some("situ".to_string());
    config.activation_situ_beta = Some(0.0);
    match DefaultsArch(config).expert_gate_policy() {
        ExpertGatePolicy::SituGlu { beta, .. } => assert_eq!(
            beta, 1.0,
            "`beta or 1.0`: a declared zero becomes one, the OPPOSITE rule"
        ),
        other => panic!("expected a SiTU policy, got {other:?}"),
    }
}

#[test]
fn an_undeclared_routed_expert_width_is_the_uniform_form() {
    assert_eq!(
        arch_with_latent(None, None).routed_expert_form(),
        RoutedExpertForm::Uniform,
        "absence is the reference's own default (`use_latent_moe = ... is not None`)"
    );
}

#[test]
fn a_declared_routed_expert_width_selects_the_latent_form() {
    let form = arch_with_latent(Some(3584), Some(true)).routed_expert_form();
    assert!(form.is_latent());
    let RoutedExpertForm::Latent { width, norm } = form else {
        panic!("expected the latent form, got {form:?}");
    };
    assert_eq!(width, 3584);
    assert!(norm.is_some(), "the flag is true, so the branch normalises");
}

/// **Four adjacent leaves of one config, three different rules.**
///
/// The subject is the parser architecture itself, not any one leaf:
/// adjacent config leaves do not share truthiness semantics merely
/// because they sit beside one another, and the risk is a shared
/// intuition rather than a shared identifier. So all four are pinned in
/// ONE place, where a reader meets every rule at once:
///
/// ```text
/// activation_situ_beta      = 0     -> 1.0                    `beta or 1.0`
/// q_lora_rank               = 0     -> the form, then refused  `is not None`
/// routed_expert_hidden_size = 0     -> the form, then refused  `is not None`
/// latent_moe_use_norm       = null  -> false, no norm          plain truthiness
/// ```
///
/// The last is the one a reader is most likely to get wrong by analogy
/// with the leaf directly above it: `routed_expert_hidden_size` and
/// `latent_moe_use_norm` are declared side by side in the SAME config and
/// treat `null` differently, because the reference reads one with `is not
/// None` and consumes the other in a plain `if`.
#[test]
fn zero_and_null_are_read_leaf_by_leaf_not_by_neighbourhood() {
    // `0 is not None`: a declared zero SELECTS the latent form, and the
    // degenerate geometry it then describes is refused downstream by
    // name rather than demoted to the uniform form here.
    let zero = arch_with_latent(Some(0), Some(true)).routed_expert_form();
    assert!(
        zero.is_latent(),
        "`0 is not None`: a declared zero selects the latent branch"
    );
    let RoutedExpertForm::Latent { width, .. } = zero else {
        unreachable!("asserted latent immediately above")
    };
    assert_eq!(width, 0, "and the width is carried verbatim, not repaired");

    // `null` is absent, for THIS leaf.
    assert_eq!(
        arch_with_latent(None, Some(true)).routed_expert_form(),
        RoutedExpertForm::Uniform,
        "a null width is the same answer as an absent one"
    );

    // And for the leaf beside it, `null` is falsy — read by a
    // `getattr(..., False)` consumed by a plain `if`, so absent, null and
    // false are one program and differ only in what the plan reports as
    // declared.
    for flag in [None, Some(false)] {
        let form = arch_with_latent(Some(3584), flag).routed_expert_form();
        let RoutedExpertForm::Latent { norm, .. } = form else {
            panic!("the width still selects the form, whatever the flag says");
        };
        assert!(
            norm.is_none(),
            "latent_moe_use_norm {flag:?} must build no norm"
        );
    }

    // The two rules already pinned, restated here so the four sit
    // together: zero selects a form on one leaf and becomes one on
    // another, in the same checkpoint.
    assert!(arch_with_q_lora(Some(0)).mla_query_form().is_low_rank());
    let mut config = base_config();
    config.hidden_act = Some("situ".to_string());
    config.activation_situ_beta = Some(0.0);
    match DefaultsArch(config).expert_gate_policy() {
        ExpertGatePolicy::SituGlu { beta, .. } => assert_eq!(beta, 1.0),
        other => panic!("expected a SiTU policy, got {other:?}"),
    }
}

/// The norm's epsilon is the LAYER's, and this inverts what the two K3
/// rungs before it established: `q_a_layernorm` and `kv_a_layernorm` run
/// at `KimiRMSNorm`'s class default because their constructor passes no
/// override, and `routed_expert_norm`'s passes one.
///
/// Pinned against `mla_q_a_norm_eps` in the same test, because the defect
/// this guards against is generalising the previous result — and a build
/// that did so would still pass every test that looked at this leaf
/// alone.
#[test]
fn the_routed_expert_norm_runs_at_the_layer_epsilon_not_the_class_default() {
    let arch = arch_with_latent(Some(3584), Some(true));
    let RoutedExpertForm::Latent { norm, .. } = arch.routed_expert_form() else {
        panic!("the width selects the form");
    };
    let eps = norm.expect("the flag declares a norm").eps;
    assert_eq!(
        eps,
        arch.norm_eps() as f64,
        "the reference passes `eps=config.rms_norm_eps` explicitly"
    );
    assert_ne!(
        Some(eps),
        arch.mla_q_a_norm_eps(),
        "the neighbouring low-rank norm's epsilon is a DIFFERENT authority"
    );
}

/// The epsilon is the family's, not the config's and not the KV norm's.
///
/// The default is `None` — unjudged — and it reaches the form as a
/// non-executable value so that closure's refusal is what a reader meets.
#[test]
fn an_unjudged_q_a_epsilon_does_not_borrow_a_plausible_one() {
    let arch = arch_with_q_lora(Some(64));
    assert_eq!(arch.mla_q_a_norm_eps(), None, "no family judgment here");
    let form = arch.mla_query_form();
    assert!(
        form.is_low_rank(),
        "the form is declared even when unjudged"
    );
    assert_eq!(
        form.norm_eps(),
        None,
        "an unjudged epsilon must not resolve to the layer eps or the KV one"
    );
}

/// The direct form carries no rank and no epsilon: a norm the layer does
/// not have cannot be described.
#[test]
fn the_direct_form_carries_neither_a_rank_nor_an_epsilon() {
    let form = arch_with_q_lora(None).mla_query_form();
    assert_eq!(form.rank(), None);
    assert_eq!(form.norm_eps(), None);
}

/// An architecture that never mentions the DSA indexer answers `None` on
/// every one of its seven accessors — `None` means "not a DSA
/// architecture" the same way it means "unjudged" for the epsilon
/// accessors above.
#[test]
fn dsa_accessors_default_to_none_for_a_non_dsa_architecture() {
    let arch = DefaultsArch(base_config());
    assert_eq!(arch.dsa_index_topk(), None);
    assert_eq!(arch.dsa_index_n_heads(), None);
    assert_eq!(arch.dsa_index_head_dim(), None);
    assert_eq!(arch.dsa_indexer_wq_b_key(0), None);
    assert_eq!(arch.dsa_indexer_wk_key(0), None);
    assert_eq!(arch.dsa_indexer_k_norm_key(0), None);
    assert_eq!(arch.dsa_indexer_weights_proj_key(0), None);
}

#[test]
fn tie_word_embeddings_is_read_from_the_config() {
    let cfg = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16, "tie_word_embeddings": false,
    }));
    assert_eq!(cfg.config().tie_word_embeddings, Some(false));
}

#[test]
fn tie_word_embeddings_true_is_distinguished_from_absent() {
    let with_true = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16, "tie_word_embeddings": true,
    }));
    assert_eq!(with_true.config().tie_word_embeddings, Some(true));

    let absent = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "intermediate_size": 16,
    }));
    assert_eq!(
        absent.config().tie_word_embeddings,
        None,
        "absent must stay None — it is not a claim that the model is tied"
    );
}

/// GPT-OSS declares `false` at the top level, not inside `text_config`.
#[test]
fn tie_word_embeddings_is_read_from_the_outer_config_too() {
    let cfg = crate::detect::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 16,
        },
    }));
    assert_eq!(cfg.config().tie_word_embeddings, Some(false));
}

/// `Shared` and `Value` are different claims, and `resolve` is the only
/// place either becomes a number.
#[test]
fn post_norm_eps_resolves_shared_and_distinct_differently() {
    use crate::config::PostNormEps;
    const PRE: f64 = 1e-5;
    // Sharing takes the pre-norm epsilon it is handed — never one it
    // read from somewhere else.
    assert_eq!(PostNormEps::Shared.resolve(PRE), PRE);
    assert_eq!(PostNormEps::Shared.resolve(1e-12), 1e-12);
    // A declared value ignores the pre-norm epsilon entirely.
    assert_eq!(PostNormEps::Value(1e-8).resolve(PRE), 1e-8);
    assert_ne!(PostNormEps::Value(1e-8).resolve(PRE), PRE);
}

/// The four-norm Gemma families state the sharing judgment rather than
/// leaving it to be inherited — the state VINDEX3 refuses.
#[test]
fn four_norm_gemma_families_declare_shared_post_norm_eps() {
    use crate::config::PostNormEps;
    for family in ["gemma2", "gemma3"] {
        let arch = crate::detect::detect_from_json(&serde_json::json!({
            "model_type": family,
            "hidden_size": 8, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "intermediate_size": 16,
        }));
        assert!(arch.has_post_norms(), "{family} is a four-norm family");
        assert_eq!(
            arch.post_norm_eps(),
            Some(PostNormEps::Shared),
            "{family} must declare sharing, not leave it unjudged"
        );
    }
}
