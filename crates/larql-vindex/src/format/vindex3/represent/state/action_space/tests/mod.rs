//! A map edit of one exception keeps the stored form every record was
//! written in; a group of several round-trips and applies all of them.

use super::*;

fn rule(projection: &str, lo: u32, hi: u32) -> Exception {
    Exception {
        projection: Some(projection.into()),
        layers: Some((lo, hi)),
        encoding: None,
    }
}

#[test]
fn a_single_exception_edit_is_stored_as_before() {
    let edit = MapEdit::new("protect:q_proj@3", rule("q_proj", 3, 3));
    let written = serde_json::to_value(&edit).unwrap();
    assert_eq!(
        written,
        serde_json::json!({
            "name": "protect:q_proj@3",
            "exception": {"projection": "q_proj", "layers": [3, 3]}
        })
    );
    assert_eq!(serde_json::from_value::<MapEdit>(written).unwrap(), edit);
}

#[test]
fn a_group_round_trips_and_applies_every_rule_in_order() {
    let edit = MapEdit::group(
        "attn-qkv-q1",
        vec![
            rule("q_proj", 0, 9),
            rule("k_proj", 0, 9),
            rule("v_proj", 0, 9),
        ],
    )
    .unwrap();
    let written = serde_json::to_value(&edit).unwrap();
    assert!(written.get("exception").is_none());
    assert_eq!(written["exceptions"].as_array().unwrap().len(), 3);
    assert_eq!(serde_json::from_value::<MapEdit>(written).unwrap(), edit);

    let vocabulary = ActionVocabulary::new([edit.clone()]).unwrap();
    let base = PrecisionMap {
        name: "base".into(),
        encoding: "NVFP4".into(),
        roles: vec!["decoder-linear".into()],
        exceptions: vec![rule("o_proj", 0, 0)],
    };
    let map = vocabulary
        .map_for(&base, &["attn-qkv-q1".to_string()].into())
        .unwrap();
    let mut expected = edit.exceptions().to_vec();
    expected.push(rule("o_proj", 0, 0));
    assert_eq!(
        map.exceptions, expected,
        "the edit's rules, then the base map's"
    );
}

#[test]
fn an_edit_that_changes_nothing_or_says_it_twice_is_refused() {
    assert!(MapEdit::group("empty", vec![]).is_err());
    for stored in [
        serde_json::json!({"name": "neither"}),
        serde_json::json!({"name": "empty", "exceptions": []}),
        serde_json::json!({
            "name": "both",
            "exception": {"projection": "q_proj"},
            "exceptions": [{"projection": "k_proj"}]
        }),
    ] {
        let err = serde_json::from_value::<MapEdit>(stored)
            .unwrap_err()
            .to_string();
        assert!(err.contains("declares"), "{err}");
    }
}
