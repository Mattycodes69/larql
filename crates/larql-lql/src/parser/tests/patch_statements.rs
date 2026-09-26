//! PATCH STATEMENTS

use super::*;

#[test]
fn parse_begin_patch() {
    let stmt = parse(r#"BEGIN PATCH "medical-knowledge.vlp";"#).unwrap();
    match stmt {
        Statement::BeginPatch { path } => assert_eq!(path, "medical-knowledge.vlp"),
        _ => panic!("expected BeginPatch"),
    }
}

#[test]
fn parse_save_patch() {
    let stmt = parse("SAVE PATCH;").unwrap();
    assert!(matches!(stmt, Statement::SavePatch));
}

#[test]
fn parse_apply_patch() {
    let stmt = parse(r#"APPLY PATCH "medical-knowledge.vlp";"#).unwrap();
    match stmt {
        Statement::ApplyPatch { path } => assert_eq!(path, "medical-knowledge.vlp"),
        _ => panic!("expected ApplyPatch"),
    }
}

#[test]
fn parse_show_patches() {
    let stmt = parse("SHOW PATCHES;").unwrap();
    assert!(matches!(stmt, Statement::ShowPatches));
}

#[test]
fn parse_remove_patch() {
    let stmt = parse(r#"REMOVE PATCH "fix-hallucinations.vlp";"#).unwrap();
    match stmt {
        Statement::RemovePatch { path } => assert_eq!(path, "fix-hallucinations.vlp"),
        _ => panic!("expected RemovePatch"),
    }
}

#[test]
fn parse_patch_workflow() {
    // Full patch workflow from spec
    parse(r#"BEGIN PATCH "medical-knowledge.vlp";"#).unwrap();
    parse(r#"INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "side_effect", "bleeding");"#).unwrap();
    parse("SAVE PATCH;").unwrap();
    parse(r#"APPLY PATCH "medical-knowledge.vlp";"#).unwrap();
    parse("SHOW PATCHES;").unwrap();
    parse(r#"REMOVE PATCH "medical-knowledge.vlp";"#).unwrap();
}

#[test]
fn parse_diff_into_patch() {
    let stmt = parse(
        r#"DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex" INTO PATCH "medical-changes.vlp";"#,
    )
    .unwrap();
    match stmt {
        Statement::Diff {
            a, b, into_patch, ..
        } => {
            assert!(matches!(a, VindexRef::Path(ref p) if p == "gemma3-4b.vindex"));
            assert!(matches!(b, VindexRef::Path(ref p) if p == "gemma3-4b-medical.vindex"));
            assert_eq!(into_patch.as_deref(), Some("medical-changes.vlp"));
        }
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_diff_without_into_patch() {
    let stmt = parse(r#"DIFF "a.vindex" "b.vindex" LIMIT 10;"#).unwrap();
    match stmt {
        Statement::Diff {
            into_patch, limit, ..
        } => {
            assert!(into_patch.is_none());
            assert_eq!(limit, Some(10));
        }
        _ => panic!("expected Diff"),
    }
}

#[test]
fn parse_compile_into_vindex() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "output.vindex";"#).unwrap();
    match stmt {
        Statement::Compile { target, output, .. } => {
            assert_eq!(target, CompileTarget::Vindex);
            assert_eq!(output, "output.vindex");
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_into_vindex_on_conflict_last_wins() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT LAST_WINS;"#).unwrap();
    match stmt {
        Statement::Compile {
            target,
            on_conflict,
            ..
        } => {
            assert_eq!(target, CompileTarget::Vindex);
            assert_eq!(on_conflict, Some(CompileConflict::LastWins));
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_into_vindex_on_conflict_highest_confidence() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#)
        .unwrap();
    match stmt {
        Statement::Compile { on_conflict, .. } => {
            assert_eq!(on_conflict, Some(CompileConflict::HighestConfidence));
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_into_vindex_on_conflict_fail() {
    let stmt = parse(r#"COMPILE CURRENT INTO VINDEX "out.vindex" ON CONFLICT FAIL;"#).unwrap();
    match stmt {
        Statement::Compile { on_conflict, .. } => {
            assert_eq!(on_conflict, Some(CompileConflict::Fail));
        }
        _ => panic!("expected Compile"),
    }
}

#[test]
fn parse_compile_into_model_with_on_conflict_errors() {
    let result = parse(r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors ON CONFLICT FAIL;"#);
    assert!(
        result.is_err(),
        "ON CONFLICT must reject COMPILE INTO MODEL"
    );
}

#[test]
fn parse_compile_into_model_explicit() {
    let stmt = parse(r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#).unwrap();
    match stmt {
        Statement::Compile { target, .. } => {
            assert_eq!(target, CompileTarget::Model);
        }
        _ => panic!("expected Compile"),
    }
}
