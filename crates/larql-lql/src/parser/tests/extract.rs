//! EXTRACT
//! COMPILE
//! DIFF
//! USE
//! WALK

use super::*;

#[test]
fn parse_extract_minimal() {
    let stmt = parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#).unwrap();
    match stmt {
        Statement::Extract {
            model,
            output,
            components,
            layers,
            extract_level,
            format,
        } => {
            assert_eq!(model, "google/gemma-3-4b-it");
            assert_eq!(output, "gemma3-4b.vindex");
            assert!(components.is_none());
            assert!(layers.is_none());
            assert_eq!(extract_level, ExtractLevel::Browse);
            assert_eq!(format, None, "no FORMAT clause = no preference");
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_with_components_and_layers() {
    let stmt = parse(
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "out.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33;"#,
    )
    .unwrap();
    match stmt {
        Statement::Extract {
            components,
            layers,
            extract_level,
            ..
        } => {
            let c = components.unwrap();
            assert_eq!(c.len(), 4);
            assert_eq!(c[0], Component::FfnGate);
            assert_eq!(c[1], Component::FfnDown);
            assert_eq!(c[2], Component::FfnUp);
            assert_eq!(c[3], Component::Embeddings);
            let l = layers.unwrap();
            assert_eq!(l.start, 0);
            assert_eq!(l.end, 33);
            assert_eq!(extract_level, ExtractLevel::Browse);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_attn_components() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" COMPONENTS ATTN_OV, ATTN_QK;"#).unwrap();
    match stmt {
        Statement::Extract { components, .. } => {
            let c = components.unwrap();
            assert_eq!(c.len(), 2);
            assert_eq!(c[0], Component::AttnOv);
            assert_eq!(c[1], Component::AttnQk);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_with_inference() {
    let stmt =
        parse(r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE;"#)
            .unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::Inference);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_with_all() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" WITH ALL;"#).unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::All);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_with_weights_legacy() {
    // WITH WEIGHTS is legacy syntax, maps to Inference
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" WITH WEIGHTS;"#).unwrap();
    match stmt {
        Statement::Extract { extract_level, .. } => {
            assert_eq!(extract_level, ExtractLevel::Inference);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_extract_with_all_and_components() {
    let stmt = parse(r#"EXTRACT MODEL "m" INTO "o" COMPONENTS FFN_GATE WITH ALL;"#).unwrap();
    match stmt {
        Statement::Extract {
            components,
            extract_level,
            ..
        } => {
            assert_eq!(extract_level, ExtractLevel::All);
            assert_eq!(components.unwrap().len(), 1);
        }
        _ => panic!("expected Extract"),
    }
}

#[test]
fn parse_compile_current_safetensors() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;"#).unwrap();
    match stmt {
        Statement::Compile {
            vindex,
            output,
            format,
            ..
        } => {
            assert!(matches!(vindex, VindexRef::Current));
            assert_eq!(output, "edited/");
            assert_eq!(format, Some(OutputFormat::Safetensors));
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_path_gguf() {
    let stmt = parse(r#"COMPILE "gemma3.vindex" INTO MODEL "out/" FORMAT gguf;"#).unwrap();
    match stmt {
        Statement::Compile {
            vindex,
            output,
            format,
            ..
        } => {
            assert!(matches!(vindex, VindexRef::Path(ref p) if p == "gemma3.vindex"));
            assert_eq!(output, "out/");
            assert_eq!(format, Some(OutputFormat::Gguf));
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_no_format() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    match stmt {
        Statement::Compile { format, .. } => assert!(format.is_none()),
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_diff_two_paths() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex";"#).unwrap();
    match stmt {
        Statement::Diff { a, b, .. } => {
            assert!(matches!(a, VindexRef::Path(ref p) if p == "a.vindex"));
            assert!(matches!(b, VindexRef::Path(ref p) if p == "b.vindex"));
        }
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_current() {
    let stmt = parse(r#"DIFF "gemma3-4b.vindex" CURRENT;"#).unwrap();
    match stmt {
        Statement::Diff {
            a: VindexRef::Path(p),
            b: VindexRef::Current,
            ..
        } => assert_eq!(p, "gemma3-4b.vindex"),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_limit() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LIMIT 20;"#).unwrap();
    match stmt {
        Statement::Diff { limit, .. } => assert_eq!(limit, Some(20)),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_layer() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LAYER 26;"#).unwrap();
    match stmt {
        Statement::Diff { layer, .. } => assert_eq!(layer, Some(26)),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_relation_singular() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" RELATION "lives-in";"#).unwrap();
    match stmt {
        Statement::Diff { relation, .. } => assert_eq!(relation.as_deref(), Some("lives-in")),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_relations_plural() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" RELATIONS "capital-of";"#).unwrap();
    match stmt {
        Statement::Diff { relation, .. } => assert_eq!(relation.as_deref(), Some("capital-of")),
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_with_relation_and_limit() {
    let stmt =
        parse(r#"DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex" RELATION "capital" LIMIT 20;"#)
            .unwrap();
    match stmt {
        Statement::Diff {
            relation, limit, ..
        } => {
            assert_eq!(relation.as_deref(), Some("capital"));
            assert_eq!(limit, Some(20));
        }
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_use_vindex() {
    let stmt = parse(r#"USE "gemma3-4b.vindex";"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Vindex(path),
        } => assert_eq!(path, "gemma3-4b.vindex"),
        _ => panic!("expected Use Vindex"),
    }
}

#[test]
fn parse_use_model() {
    let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it";"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Model { id, auto_extract },
        } => {
            assert_eq!(id, "google/gemma-3-4b-it");
            assert!(!auto_extract);
        }
        _ => panic!("expected Use Model"),
    }
}

#[test]
fn parse_use_model_auto_extract() {
    let stmt = parse(r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#).unwrap();
    match stmt {
        Statement::Use {
            target: UseTarget::Model { auto_extract, .. },
        } => assert!(auto_extract),
        _ => panic!("expected Use Model AUTO_EXTRACT"),
    }
}

#[test]
fn parse_walk_minimal() {
    let stmt = parse(r#"WALK "The capital of France is";"#).unwrap();
    match stmt {
        Statement::Walk {
            prompt,
            top,
            layers,
            mode,
            compare,
        } => {
            assert_eq!(prompt, "The capital of France is");
            assert!(top.is_none());
            assert!(layers.is_none());
            assert!(mode.is_none());
            assert!(!compare);
        }
        _ => panic!("expected Walk"),
    }
}

#[test]
fn parse_walk_with_top() {
    let stmt = parse(r#"WALK "The capital of France is" TOP 5;"#).unwrap();
    match stmt {
        Statement::Walk { top, .. } => assert_eq!(top, Some(5)),
        _ => panic!("expected Walk"),
    }
}

#[test]
fn parse_walk_full_options() {
    let stmt = parse(r#"WALK "prompt" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#).unwrap();
    match stmt {
        Statement::Walk {
            top,
            layers,
            mode,
            compare,
            ..
        } => {
            assert_eq!(top, Some(5));
            let l = layers.unwrap();
            assert_eq!(l.start, 25);
            assert_eq!(l.end, 33);
            assert_eq!(mode, Some(WalkMode::Hybrid));
            assert!(compare);
        }
        _ => panic!("expected Walk"),
    }
}

#[test]
fn parse_walk_mode_pure() {
    let stmt = parse(r#"WALK "x" MODE pure;"#).unwrap();
    match stmt {
        Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Pure)),
        _ => panic!("expected Walk"),
    }
}

#[test]
fn parse_walk_mode_dense() {
    let stmt = parse(r#"WALK "x" MODE dense;"#).unwrap();
    match stmt {
        Statement::Walk { mode, .. } => assert_eq!(mode, Some(WalkMode::Dense)),
        _ => panic!("expected Walk"),
    }
}

#[test]
fn parse_walk_layers_all() {
    let stmt = parse(r#"WALK "x" LAYERS ALL;"#).unwrap();
    match stmt {
        Statement::Walk { layers, .. } => assert!(layers.is_none()),
        _ => panic!("expected Walk"),
    }
}
