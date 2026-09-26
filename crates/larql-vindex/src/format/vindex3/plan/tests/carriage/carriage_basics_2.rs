use super::*;

/// Every span in the vocabulary round-trips through its declared name, so
/// `from_declared` and `declared_name` cannot drift apart and make the
/// probe compare against a spelling no checkpoint uses.
#[test]
fn span_names_round_trip() {
    use crate::format::vindex3::graph::policy::AttentionSpan;
    for span in [
        AttentionSpan::Sliding,
        AttentionSpan::Full,
        AttentionSpan::Windowed,
    ] {
        assert_eq!(
            AttentionSpan::from_declared(span.declared_name()),
            Some(span)
        );
    }
}

/// A rule whose probe never answers (`probe_unrepresented`) reports the
/// declared fact as dropped at the boundary, naming its site — the honest
/// verdict for a leaf the schema has no field for (`mscale_all_dim`), and
/// it blocks.
///
/// The example used to be `norm_topk_prob`, which has since gained a real
/// destination (`ExecutionSurface.ffn.moe.routing_policy`). The idiom is
/// what this test pins, so it moved to a leaf that still has no field
/// rather than being deleted with the rule it happened to name.
#[test]
fn a_no_schema_field_rule_reports_the_fact_unrepresented_and_blocks() {
    let findings = plan_with(|config| {
        config["text_config"]["rope_scaling"] = serde_json::json!({
            "rope_type": "yarn",
            "mscale_all_dim": 1.0,
        });
    });
    let finding = finding_for(&findings, "mscale_all_dim");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert!(
        finding.detail.contains("no schema field"),
        "{}",
        finding.detail
    );
    assert!(finding.blocks());
}

/// Granite's `residual_multiplier` is read by the generic resolution into
/// the surface's residual scale and judged there: a declaration is carried
/// to the lowered op and the probe answers with the stored value.
#[test]
fn a_declared_residual_multiplier_is_carried_to_the_residual_scale() {
    let declared = 0.22;
    let findings = plan_with(|config| {
        config["text_config"]["residual_multiplier"] = serde_json::json!(declared);
    });
    let finding = finding_for(&findings, "residual_multiplier");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.carriage, Some(Carriage::Lowered));
    let resolved = finding.resolved.as_ref().and_then(|v| v.as_f64()).unwrap();
    assert!((resolved - declared).abs() < 1e-6);
}

/// `query_pre_attn_scalar` is judged in the SCORE-SCALE space: the
/// declared scalar is canonicalised to `scalar^-0.5` before comparing with
/// the surface's score scale, so a family that resolves its score scale
/// from head_dim disagrees (and says so) rather than comparing 256 to 0.35.
#[test]
fn query_pre_attn_scalar_is_compared_in_score_scale_space() {
    use crate::format::vindex3::plan::carriage::canonical_declared;
    let scalar: f64 = 256.0;
    let canonical = canonical_declared("query_pre_attn_scalar", &serde_json::json!(scalar));
    let expected = scalar.powf(-0.5);
    assert!((canonical.as_f64().unwrap() - expected).abs() < 1e-12);
    // A non-numeric declaration passes through untouched.
    assert_eq!(
        canonical_declared("query_pre_attn_scalar", &serde_json::json!("x")),
        serde_json::json!("x")
    );
    // Any other leaf is untouched.
    assert_eq!(
        canonical_declared("rope_theta", &serde_json::json!(scalar)),
        serde_json::json!(scalar)
    );
    let findings = plan_with(|config| {
        config["text_config"]["query_pre_attn_scalar"] = serde_json::json!(scalar);
    });
    let finding = finding_for(&findings, "query_pre_attn_scalar");
    // The resolution reads the scalar into the surface's score scale
    // (256 → 0.0625), and the comparison agrees in that space — not by
    // comparing 256 with 0.0625 as raw JSON.
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(expected as f32)));
}

/// A tensor group the graph cannot place — an unknown prefix with no
/// object kind — surfaces as an unrepresented finding that names the
/// prefix and the builder's reason, and it blocks.
#[test]
fn an_unplaceable_tensor_group_is_reported_and_blocks() {
    use crate::format::vindex3::plan::tests_support::custom_artifact;
    let dir = tempfile::tempdir().unwrap();
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0
    });
    let mystery: [usize; 2] = [4, 4];
    let inventory = custom_artifact(
        dir.path(),
        &config,
        &[
            ("model.embed_tokens.weight", &[128usize, 64]),
            ("mystery_block.weight", &mystery),
        ],
    );
    let plan = plan_system(&[("odd-artifact".to_string(), inventory)]);
    let finding = plan
        .artifacts
        .iter()
        .flat_map(|a| a.findings.iter())
        .find(|f| f.subject.starts_with("mystery_block"))
        .expect("the unplaced group is reported");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert_eq!(finding.class, SemanticClass::Unknown);
    assert!(finding.blocks());
    assert!(!plan.admissible);
}

/// A nested component whose execution surface cannot be completed
/// (vision tower missing its head count) surfaces as an
/// `execution_surface` finding naming the missing fact, and it blocks.
#[test]
fn an_incomplete_nested_surface_is_reported_and_blocks() {
    let findings = plan_with(|config| {
        config["vision_config"]
            .as_object_mut()
            .unwrap()
            .remove("num_attention_heads");
    });
    let finding = finding_for(&findings, "vision.execution_surface");
    assert_eq!(finding.category, FindingCategory::Unrepresented);
    assert!(
        finding.detail.contains("num_attention_heads"),
        "{}",
        finding.detail
    );
    assert!(finding.blocks());
}

/// **A declaration no reference implementation reads still has to agree.**
///
/// SmolLM2-135M ships `rope_interleaved: false` under `model_type: llama`,
/// which has no such field — transformers reads it nowhere. This build
/// pairs split-half, so `false` agrees and the key is satisfied.
///
/// The second arm is the whole point of reading it at all: `true` names a
/// different operator, and an unread agreement would have let that
/// through as a rotation performed against different partners in silence.
#[test]
fn a_declared_rotary_pairing_must_agree_with_the_one_that_runs() {
    let findings = plan_with(|config| {
        config["text_config"]["rope_interleaved"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "rope_interleaved");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "split-half agrees with `false`: {}",
        finding.detail
    );
    assert!(!finding.blocks());

    let findings = plan_with(|config| {
        config["text_config"]["rope_interleaved"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "rope_interleaved");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "an interleaved pairing is not what this build performs: {}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(false)));
}

/// The same shape for `use_mrope`, checked against the policy the axis
/// geometry resolves rather than against the flag itself.
#[test]
fn a_declared_mrope_flag_must_agree_with_the_resolved_policy() {
    // Qwen2.5-0.5B's real declaration: `false`, on a config carrying no
    // axis geometry, so no multi-axis policy resolves and the two agree.
    let findings = plan_with(|config| {
        config["text_config"]["use_mrope"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "use_mrope");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );

    // Claiming multi-axis without the geometry to build one is a claim
    // nothing can honour, and must not pass merely because the flag was
    // read.
    let findings = plan_with(|config| {
        config["text_config"]["use_mrope"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "use_mrope");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "no mrope_section resolves no MRope policy: {}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(false)));
}

/// **Falcon's `activation` names the FFN SHAPE, and it is checked against
/// the FFN that runs.** Falcon3-1B-Base declares `activation: "swiglu"`
/// beside `hidden_act: "silu"` under `model_type: llama`; no
/// transformers-5.5.0 loader reads `activation` for that family, so the
/// only behaviour to match is this build's. `swiglu` is "gated, SiLU on
/// the gate" in one word — two facts the execution surface carries as
/// `ffn_type` and `activation` — and the same word for a GELU-gated or an
/// ungated FFN would be one value away from the wrong arithmetic.
#[test]
fn a_declared_ffn_shape_name_must_describe_the_ffn_that_runs() {
    let findings = dense_plan_with(|config| {
        config["activation"] = serde_json::json!("swiglu");
    });
    let finding = finding_for(&findings, "activation");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "a gated SiLU FFN is what `swiglu` names: {}",
        finding.detail
    );
    assert!(!finding.blocks());

    // GELU-gated is a different FFN, and the resolved value says which
    // one actually runs rather than echoing the declaration.
    let findings = dense_plan_with(|config| {
        config["activation"] = serde_json::json!("geglu");
    });
    let finding = finding_for(&findings, "activation");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!("swiglu")));

    // A plain nonlinearity name is the UNGATED shape. On a gated FFN it
    // does not describe what runs, even though the gate is SiLU.
    let findings = dense_plan_with(|config| {
        config["activation"] = serde_json::json!("silu");
    });
    let finding = finding_for(&findings, "activation");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
}

/// **`is_llama_config` is checked against the family the identity
/// resolved to, never echoed.** SmolLM2-135M declares `true` under
/// `model_type: llama`; the key appears nowhere in transformers 5.5.0.
/// The agreeing arm is the recognised dense family; the disagreeing arm
/// is the same flag on a checkpoint whose identity resolves elsewhere,
/// and `false` on a Llama stack — both must refuse, with the resolved
/// value naming what the registry actually answered.
#[test]
fn a_declared_llama_identity_flag_must_agree_with_the_resolved_family() {
    let findings = dense_plan_with(|config| {
        config["is_llama_config"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "is_llama_config");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert!(!finding.blocks());

    let findings = dense_plan_with(|config| {
        config["is_llama_config"] = serde_json::json!(false);
    });
    let finding = finding_for(&findings, "is_llama_config");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(true)));

    // The Glimmer fixture resolves to its own family, not Llama.
    let findings = plan_with(|config| {
        config["text_config"]["is_llama_config"] = serde_json::json!(true);
    });
    let finding = finding_for(&findings, "is_llama_config");
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!(false)));
}

/// The rotary SCHEDULE, answered layer by layer from the graph.
///
/// The fixture already leaves layers 3 and 7 unrotated through
/// `layer_rope_theta`'s zero sentinel, so a `no_rope_layers` mask stating
/// the same schedule must be reported as carried. The second arm is what
/// makes that non-vacuous: a mask that schedules NOTHING changes the
/// answer, so the probe is reading the graph rather than echoing the
/// declaration back.
#[test]
fn a_declared_rope_schedule_is_carried_layer_by_layer() {
    let agreeing = serde_json::json!([1, 1, 1, 0, 1, 1, 1, 0]);
    let findings = plan_with(|config| {
        config["text_config"]["no_rope_layers"] = agreeing.clone();
    });
    let finding = finding_for(&findings, "no_rope_layers");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(
        finding.resolved,
        Some(agreeing),
        "the probe must answer the schedule in the checkpoint's own polarity: {}",
        finding.detail
    );

    // Every layer NoPE: a different declaration must produce a different
    // carried answer, or the first arm proves nothing.
    let all_nope = serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0]);
    let findings = plan_with(|config| {
        config["text_config"]["no_rope_layers"] = all_nope.clone();
    });
    assert_eq!(
        finding_for(&findings, "no_rope_layers").resolved,
        Some(all_nope),
        "a mask that rotates nowhere must be carried as such"
    );
}

/// The positional SCHEME, and the null answer that is an answer.
#[test]
fn a_stack_that_rotates_nowhere_reports_no_scheme() {
    // The fixture rotates on six of its eight layers, so a declared
    // `rope` agrees with what the graph carries.
    let findings = plan_with(|config| {
        config["text_config"]["position_embedding_type"] = serde_json::json!("rope");
    });
    let finding = finding_for(&findings, "position_embedding_type");
    assert_eq!(finding.class, SemanticClass::ExecutionSemantic);
    assert_eq!(
        finding.category,
        FindingCategory::Representable,
        "{}",
        finding.detail
    );
    assert_eq!(finding.resolved, Some(serde_json::json!("rope")));

    // Take the rotation away and the same declaration must stop being
    // satisfied. `null` here is the graph saying "this stack encodes no
    // position" — reporting it as unknown would hide exactly the case
    // the rule exists for.
    let findings = plan_with(|config| {
        config["text_config"]["position_embedding_type"] = serde_json::json!("rope");
        config["text_config"]["no_rope_layers"] = serde_json::json!([0, 0, 0, 0, 0, 0, 0, 0]);
    });
    let finding = finding_for(&findings, "position_embedding_type");
    assert_eq!(
        finding.resolved,
        Some(serde_json::Value::Null),
        "a stack that rotates nowhere answers null: {}",
        finding.detail
    );
    assert_ne!(
        finding.category,
        FindingCategory::Representable,
        "a declared scheme the graph does not carry must not read as satisfied: {}",
        finding.detail
    );
}

#[test]
fn a_declaration_with_no_value_has_nothing_to_carry() {
    // Gemma 4's dense sizes declare `top_k_experts: null` — a dense model
    // stating it has no expert bank. The carriage rule pointed at
    // `ExecutionSurface.ffn.moe.top_k`, which no dense component can
    // answer, so the checkpoint was refused over the absence of a thing
    // it had said it did not have.
    let findings = plan_with(|config| {
        config["text_config"]["top_k_experts"] = serde_json::Value::Null;
    });
    let finding = finding_for(&findings, "top_k_experts");
    assert_eq!(finding.category, FindingCategory::Representable);
    assert!(!finding.blocks());

    // The control, on the same key. A value this build cannot place must
    // still block — otherwise the rule reads as "nulls and everything
    // else are both fine", which is the silent-default shape the whole
    // census exists to catch.
    let findings = plan_with(|config| {
        config["text_config"]["top_k_experts"] = serde_json::json!(8);
    });
    let finding = finding_for(&findings, "top_k_experts");
    assert!(
        finding.blocks(),
        "a declared expert top-k on a fixture with no expert bank must block: {:?}",
        finding.detail
    );
}

#[test]
fn a_window_its_own_switch_turns_off_is_inert() {
    // Qwen2.5 ships `sliding_window: 32768` beside
    // `use_sliding_window: false`. VINDEX3 resolves no window — correctly
    // — and the carriage rule read that agreement as a dropped fact,
    // refusing the checkpoint over a window it had been told not to use.
    let findings = plan_with(|config| {
        config["text_config"]["sliding_window"] = serde_json::json!(4096);
        config["text_config"]["use_sliding_window"] = serde_json::json!(false);
    });
    let finding = carriage_finding_for(&findings, "text_config.sliding_window");
    assert_eq!(finding.category, FindingCategory::Representable);
    assert!(!finding.blocks());
    assert!(
        finding.detail.contains("use_sliding_window"),
        "the agreement must name the switch that made it one: {}",
        finding.detail
    );
}

#[test]
fn an_enabled_window_is_still_asked_of_the_container() {
    // THE CONTROL. Same declared window, switch ON: the gate must not
    // fire, so the finding is whatever probing the graph says — not an
    // assertion that the value is inert. Without this the change reads as
    // "sliding_window stopped being checked".
    let findings = plan_with(|config| {
        config["text_config"]["sliding_window"] = serde_json::json!(4096);
        config["text_config"]["use_sliding_window"] = serde_json::json!(true);
    });
    let finding = carriage_finding_for(&findings, "text_config.sliding_window");
    assert!(
        !finding.detail.contains("switched off by"),
        "the gate must not fire while the switch is on: {}",
        finding.detail
    );
}

#[test]
fn a_switch_in_another_component_does_not_reach_across() {
    // The companion is looked up at the SAME nesting level. A root-level
    // `use_sliding_window: false` must not silence a window declared
    // inside the text component — one tower's flag cannot disable
    // another's attention.
    let findings = plan_with(|config| {
        config["text_config"]["sliding_window"] = serde_json::json!(4096);
        config["text_config"]["use_sliding_window"] = serde_json::json!(true);
        config["use_sliding_window"] = serde_json::json!(false);
    });
    let finding = carriage_finding_for(&findings, "text_config.sliding_window");
    assert!(
        !finding.detail.contains("switched off by"),
        "a root switch must not gate a nested window: {}",
        finding.detail
    );
}
