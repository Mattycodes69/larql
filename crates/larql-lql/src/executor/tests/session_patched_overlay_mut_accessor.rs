//! Session::patched_overlay_mut accessor

use super::*;

#[test]
fn patched_overlay_mut_returns_some_for_vindex_backend() {
    let (mut session, dir) = vindex_session("overlay_mut_some");
    assert!(
        session.patched_overlay_mut().is_some(),
        "Vindex backend should yield a mutable overlay"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn patched_overlay_mut_returns_none_for_no_backend() {
    let mut session = Session::new();
    assert!(
        session.patched_overlay_mut().is_none(),
        "fresh session with no backend should yield None"
    );
}

#[test]
fn patched_overlay_mut_round_trip_via_insert_feature() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("overlay_mut_round_trip");
    let gate = vec![0.7_f32, 0.0, 0.0, 0.0];
    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            1,
            gate.clone(),
            FeatureMeta {
                top_token: "z".into(),
                top_token_id: 9,
                c_score: 0.42,
                top_k: vec![TopKEntry {
                    token: "z".into(),
                    token_id: 9,
                    logit: 0.42,
                }],
            },
        );
    }
    // Same accessor, second call: the gate we just wrote must still be there.
    let overlay2 = session.patched_overlay_mut().expect("vindex backend");
    assert_eq!(
        overlay2.overrides_gate_at(0, 1),
        Some(gate.as_slice()),
        "second patched_overlay_mut() call should observe the previous mutation",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_patches_with_no_patches_returns_message() {
    let (mut session, dir) = vindex_session("show_patches_empty");

    let stmt = parser::parse("SHOW PATCHES;").unwrap();
    let out = session.execute(&stmt).expect("SHOW PATCHES");
    let joined = out.join("\n").to_lowercase();
    assert!(
        joined.contains("no") || joined.contains("0") || joined.is_empty(),
        "expected an empty/no-patches message: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refresh_recorded_patch_ops_for_slots_persists_latest_overlay_vectors() {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, PatchOp};

    let (mut session, dir) = vindex_session("refresh_patch_ops");

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0,
            0,
            vec![1.0, 0.0, 0.0, 0.0],
            FeatureMeta {
                top_token: "old".into(),
                top_token_id: 7,
                c_score: 0.5,
                top_k: vec![TopKEntry {
                    token: "old".into(),
                    token_id: 7,
                    logit: 0.5,
                }],
            },
        );
        overlay.set_up_vector(0, 0, vec![0.1, 0.2, 0.3, 0.4]);
        overlay.set_down_vector(0, 0, vec![0.5, 0.6, 0.7, 0.8]);
    }

    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![PatchOp::Insert {
            layer: 0,
            feature: 0,
            relation: Some("capital".into()),
            entity: "Atlantis".into(),
            target: "Poseidon".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            up_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            down_vector_b64: Some(larql_vindex::patch::core::encode_gate_vector(&[
                9.0, 9.0, 9.0, 9.0,
            ])),
            down_meta: None,
        }],
    });

    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.set_up_vector(0, 0, vec![1.1, 1.2, 1.3, 1.4]);
        overlay.set_down_vector(0, 0, vec![2.1, 2.2, 2.3, 2.4]);
    }

    session
        .refresh_recorded_patch_ops_for_slots(&[(0, 0)])
        .expect("refresh patch ops");

    let PatchOp::Insert {
        up_vector_b64,
        down_vector_b64,
        ..
    } = &session.patch_recording.as_ref().unwrap().operations[0]
    else {
        panic!("expected insert op");
    };
    let up = larql_vindex::patch::core::decode_gate_vector(up_vector_b64.as_ref().unwrap())
        .expect("decode refreshed up");
    let down = larql_vindex::patch::core::decode_gate_vector(down_vector_b64.as_ref().unwrap())
        .expect("decode refreshed down");

    assert_eq!(up, vec![1.1, 1.2, 1.3, 1.4]);
    assert_eq!(down, vec![2.1, 2.2, 2.3, 2.4]);
    let _ = std::fs::remove_dir_all(&dir);
}
