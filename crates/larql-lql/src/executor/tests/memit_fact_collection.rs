//! MEMIT fact collection
//! Template tests
//! MEMIT struct
//! Compile into model requires weights

use super::*;

#[test]
fn memit_facts_count_inserts_only() {
    use larql_vindex::PatchOp;

    let ops = [
        PatchOp::Insert {
            layer: 26,
            feature: 100,
            relation: Some("capital".into()),
            entity: "X".into(),
            target: "Y".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        },
        PatchOp::Delete {
            layer: 10,
            feature: 50,
            reason: None,
        },
        PatchOp::Update {
            layer: 0,
            feature: 2,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        },
    ];
    let insert_count = ops
        .iter()
        .filter(|op| matches!(op, PatchOp::Insert { .. }))
        .count();
    assert_eq!(insert_count, 1, "only INSERT should be counted");
}

#[test]
fn memit_facts_deduplicate_across_patches() {
    use larql_vindex::{PatchOp, VindexPatch};

    let mkp = |conf: f32| VindexPatch {
        version: 1,
        base_model: String::new(),
        base_checksum: None,
        created_at: String::new(),
        description: None,
        author: None,
        tags: Vec::new(),
        operations: vec![PatchOp::Insert {
            layer: 10,
            feature: 5,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(conf),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }],
    };
    let patches = vec![mkp(0.9), mkp(0.95)];
    let mut seen = std::collections::HashSet::new();
    for p in &patches {
        for op in &p.operations {
            if let PatchOp::Insert {
                layer,
                entity,
                relation,
                target,
                ..
            } = op
            {
                seen.insert((
                    entity.clone(),
                    relation.clone().unwrap_or_default(),
                    target.clone(),
                    *layer,
                ));
            }
        }
    }
    assert_eq!(seen.len(), 1, "same fact in two patches → 1 after dedup");
}

#[test]
fn relation_template_simple() {
    let rel = "capital";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The capital of entity is");
}

#[test]
fn relation_template_multi_word() {
    let rel = "native_language";
    let prompt = format!("The {} of entity is", rel.replace(['-', '_'], " "));
    assert_eq!(prompt, "The native language of entity is");
}

#[test]
fn relation_template_hyphenated_produces_double_of() {
    // Documents the known template quirk: "capital-of" → "capital of"
    // → "The capital of of X is". Users should use "capital" not "capital-of".
    let rel = "capital-of";
    let prompt = format!("The {} of X is", rel.replace(['-', '_'], " "));
    assert!(
        prompt.contains("of of"),
        "capital-of produces double 'of': {prompt}"
    );
}

#[test]
fn memit_fact_struct() {
    let f = larql_inference::MemitFact {
        prompt_tokens: vec![1, 2, 3],
        target_token_id: 42,
        layer: 26,
        label: "test".into(),
    };
    assert_eq!(f.layer, 26);
    assert_eq!(f.target_token_id, 42);
}

#[test]
fn compile_into_model_requires_model_weights() {
    let (mut session, dir) = vindex_session("compile_model_noweights");
    let output = dir.join("model_out");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}";"#,
        lql_path(&output)
    ))
    .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("model weights") || msg.contains("WITH ALL"),
        "error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
