//! Admission checks on the VINDEX3 layer-prefix and dense-FFN bindings, and
//! their JSON wire shape (unknown fields refused).

use larql_router_protocol::vindex3::{self, MAX_POSITIONS, SCHEMA};
use larql_router_protocol::vindex3_ffn::{self, OperandIdentity};

const HIDDEN: usize = 4;
const ARTIFACT_HEX_LEN: usize = 64;

fn program() -> vindex3::Binding {
    vindex3::Binding {
        schema: SCHEMA,
        artifact: "a".repeat(ARTIFACT_HEX_LEN),
        execution_identity: "e".repeat(ARTIFACT_HEX_LEN),
        backend: "cpu".into(),
        lowering: "cpu-production/v1".into(),
        start: 0,
        end: 2,
        layers: 4,
        hidden: HIDDEN,
    }
}

fn operand(layer: usize) -> OperandIdentity {
    OperandIdentity {
        layer,
        operand: "ffn.down".into(),
        representation: "f32".into(),
        codec: "raw".into(),
        realization: "dense".into(),
        extent: "full".into(),
        dependencies: String::new(),
    }
}

fn ffn() -> vindex3_ffn::Binding {
    vindex3_ffn::Binding {
        program: program(),
        operands: vec![operand(0), operand(1)],
    }
}

#[test]
fn layer_binding_accepts_a_well_formed_program() {
    assert_eq!(program().validate(), Ok(()));
    assert_eq!(program().validate_rows(&[vec![0.5; HIDDEN]]), Ok(()));
}

/// One way to break a well-formed binding.
type Mutation = Box<dyn Fn(&mut vindex3::Binding)>;

#[test]
fn layer_binding_refuses_each_malformed_field() {
    let cases: Vec<Mutation> = vec![
        Box::new(|b| b.schema = SCHEMA + 1),
        Box::new(|b| b.backend = "metal".into()),
        Box::new(|b| b.lowering.clear()),
        Box::new(|b| b.hidden = 0),
        Box::new(|b| b.start = b.end),
        Box::new(|b| b.end = b.layers + 1),
        Box::new(|b| b.artifact.truncate(ARTIFACT_HEX_LEN - 1)),
        Box::new(|b| b.artifact = "z".repeat(ARTIFACT_HEX_LEN)),
        Box::new(|b| b.execution_identity.truncate(ARTIFACT_HEX_LEN - 1)),
        Box::new(|b| b.execution_identity = "z".repeat(ARTIFACT_HEX_LEN)),
    ];
    for (i, mutate) in cases.iter().enumerate() {
        let mut b = program();
        mutate(&mut b);
        assert!(b.validate().is_err(), "case {i} admitted");
        assert!(b.validate_rows(&[vec![0.0; HIDDEN]]).is_err());
    }
}

/// RESIDUAL-BUS-2 D9: a peer on the schema before the execution identity
/// is refused, and the refusal names both schemas rather than calling the
/// binding merely invalid.
#[test]
fn a_schema_one_peer_is_refused_by_name() {
    let mut b = program();
    b.schema = 1;
    let err = b.validate().unwrap_err();
    assert!(err.contains("schema 1"), "{err}");
    assert!(err.contains(&format!("schema {SCHEMA}")), "{err}");
    assert!(err.contains("execution identity"), "{err}");
}

#[test]
fn layer_rows_must_be_bounded_sized_and_finite() {
    let b = program();
    assert!(b.validate_rows(&[]).is_err(), "no rows");
    assert!(b
        .validate_rows(&vec![vec![0.0; HIDDEN]; MAX_POSITIONS + 1])
        .is_err());
    assert!(b.validate_rows(&[vec![0.0; HIDDEN + 1]]).is_err());
    let err = b.validate_rows(&[vec![f32::NAN; HIDDEN]]).unwrap_err();
    assert!(err.contains("finite"), "got: {err}");
}

#[test]
fn ffn_binding_checks_operand_ownership() {
    assert_eq!(ffn().validate(), Ok(()));

    let mut empty = ffn();
    empty.operands.clear();
    assert!(empty.validate().is_err());

    let mut foreign = ffn();
    foreign.operands.push(operand(3));
    assert!(foreign.validate().is_err(), "operand outside the range");

    let mut bad_program = ffn();
    bad_program.program.hidden = 0;
    assert!(bad_program.validate().is_err());
}

#[test]
fn ffn_rows_must_belong_to_the_worker_and_be_finite() {
    let b = ffn();
    assert_eq!(b.validate_row(1, &[1.0; HIDDEN]), Ok(()));
    let err = b.validate_row(2, &[1.0; HIDDEN]).unwrap_err();
    assert!(err.contains("outside"), "got: {err}");
    assert!(b.validate_row(0, &[1.0; HIDDEN - 1]).is_err());
    assert!(b.validate_row(0, &[f32::INFINITY; HIDDEN]).is_err());

    let mut invalid = ffn();
    invalid.operands.clear();
    assert!(invalid.validate_row(0, &[1.0; HIDDEN]).is_err());
}

#[test]
fn bindings_and_messages_round_trip_json_and_refuse_unknown_fields() {
    let b = ffn();
    let json = serde_json::to_value(&b).unwrap();
    let back: vindex3_ffn::Binding = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(back, b);

    let mut extra = json;
    extra["surprise"] = serde_json::json!(1);
    assert!(serde_json::from_value::<vindex3_ffn::Binding>(extra).is_err());

    let request = vindex3_ffn::Request {
        binding: b.clone(),
        layer: 1,
        row: vec![0.25; HIDDEN],
    };
    let request: vindex3_ffn::Request =
        serde_json::from_str(&serde_json::to_string(&request).unwrap()).unwrap();
    assert_eq!(request.layer, 1);

    let response = vindex3_ffn::Response {
        binding: b,
        layer: 1,
        row: vec![0.5; HIDDEN],
    };
    let response: vindex3_ffn::Response =
        serde_json::from_str(&serde_json::to_string(&response).unwrap()).unwrap();
    assert_eq!(response.row, vec![0.5; HIDDEN]);

    let timing = vindex3_ffn::WorkerTiming {
        ffn_ns: 7,
        ..Default::default()
    };
    let timing: vindex3_ffn::WorkerTiming =
        serde_json::from_str(&serde_json::to_string(&timing).unwrap()).unwrap();
    assert_eq!(timing.ffn_ns, 7);

    let rows = vec![vec![1.0; HIDDEN]];
    let layer_request = vindex3::Request {
        binding: program(),
        rows: rows.clone(),
    };
    let layer_request: vindex3::Request =
        serde_json::from_str(&serde_json::to_string(&layer_request).unwrap()).unwrap();
    assert_eq!(layer_request.rows, rows);
    let layer_response = vindex3::Response {
        binding: program(),
        rows: rows.clone(),
    };
    let layer_response: vindex3::Response =
        serde_json::from_str(&serde_json::to_string(&layer_response).unwrap()).unwrap();
    assert_eq!(layer_response.binding, program());
}
