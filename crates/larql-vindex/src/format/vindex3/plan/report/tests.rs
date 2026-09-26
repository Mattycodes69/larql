use super::*;

#[test]
fn representable_findings_never_block() {
    let finding = Finding {
        category: FindingCategory::Representable,
        class: SemanticClass::ExecutionSemantic,
        component: "text".into(),
        subject: "x".into(),
        declared: None,
        resolved: None,
        carriage: None,
        detail: String::new(),
    };
    assert!(!finding.blocks());
}

#[test]
fn critical_classes_block_outside_representable() {
    for class in [
        SemanticClass::ExecutionSemantic,
        SemanticClass::TensorSemantic,
        SemanticClass::InterfaceSemantic,
        SemanticClass::Unknown,
    ] {
        let finding = Finding {
            category: FindingCategory::Unrepresented,
            class,
            component: "text".into(),
            subject: "x".into(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: String::new(),
        };
        assert!(finding.blocks(), "{class:?} must block");
    }
}

#[test]
fn benign_classes_do_not_block() {
    for class in [SemanticClass::IgnoredSafe, SemanticClass::MetadataOnly] {
        let finding = Finding {
            category: FindingCategory::Unrepresented,
            class,
            component: "root".into(),
            subject: "x".into(),
            declared: None,
            resolved: None,
            carriage: None,
            detail: String::new(),
        };
        assert!(!finding.blocks(), "{class:?} must not block");
    }
}

#[test]
fn classes_serialise_snake_case() {
    assert_eq!(
        serde_json::to_string(&SemanticClass::ExecutionSemantic).unwrap(),
        "\"execution_semantic\""
    );
    assert_eq!(
        serde_json::to_string(&FindingCategory::Unrepresented).unwrap(),
        "\"unrepresented\""
    );
}
