//! endpoint seam

use super::*;

#[test]
fn endpoint_kind_parses_and_labels() {
    assert_eq!(
        EndpointKind::parse("walk-ffn").unwrap(),
        EndpointKind::WalkFfn
    );
    assert_eq!(
        EndpointKind::parse(" experts ").unwrap(),
        EndpointKind::Experts
    );
    assert!(EndpointKind::parse("expert").is_err());
    assert_eq!(EndpointKind::WalkFfn.label(), "walk-ffn");
    assert_eq!(EndpointKind::Experts.label(), "experts");
}

#[test]
fn endpoint_resolution_maps_wire_axis() {
    use Endpoint::*;
    let plain = WireSpec::Plain;
    assert_eq!(
        Endpoint::resolve(EndpointKind::WalkFfn, plain(WireArm::F32)).unwrap(),
        WalkFfn
    );
    assert_eq!(
        Endpoint::resolve(EndpointKind::WalkFfn, plain(WireArm::F16)).unwrap(),
        WalkFfn,
        "f16/i8 are Accept negotiation on the same endpoint"
    );
    assert_eq!(
        Endpoint::resolve(EndpointKind::WalkFfn, plain(WireArm::I8)).unwrap(),
        WalkFfn
    );
    assert_eq!(
        Endpoint::resolve(EndpointKind::WalkFfn, plain(WireArm::Q8k)).unwrap(),
        WalkFfnQ8k
    );
    assert_eq!(
        Endpoint::resolve(EndpointKind::Experts, plain(WireArm::F32)).unwrap(),
        ExpertsMultiLayer
    );
    assert_eq!(
        Endpoint::resolve(EndpointKind::Experts, plain(WireArm::Q8k)).unwrap(),
        ExpertsMultiLayerQ8k
    );
    assert!(Endpoint::resolve(EndpointKind::Experts, plain(WireArm::F16)).is_err());
    assert!(Endpoint::resolve(EndpointKind::Experts, plain(WireArm::I8)).is_err());
    // Asymmetric pairs: dense walk-ffn only.
    let pair = WireSpec::Pair {
        input: WireFormat::I8,
        output: WireFormat::F16,
    };
    assert_eq!(
        Endpoint::resolve(EndpointKind::WalkFfn, pair).unwrap(),
        WalkFfn
    );
    assert!(Endpoint::resolve(EndpointKind::Experts, pair).is_err());
}

#[test]
fn endpoint_labels_codes_paths_and_behaviour() {
    use Endpoint::*;
    for (e, label, code) in [
        (WalkFfn, "walk-ffn", 0),
        (WalkFfnQ8k, "walk-ffn-q8k", 1),
        (ExpertsMultiLayer, "experts-ml", 2),
        (ExpertsMultiLayerQ8k, "experts-ml-q8k", 3),
    ] {
        assert_eq!(e.label(), label);
        assert_eq!(e.code(), code);
    }
    // Paths are the production constants.
    assert_eq!(WalkFfn.path(), WALK_FFN_PATH);
    assert_eq!(WalkFfnQ8k.path(), WALK_FFN_Q8K_PATH);
    assert_eq!(ExpertsMultiLayer.path(), MULTI_LAYER_BATCH_PATH);
    assert_eq!(ExpertsMultiLayerQ8k.path(), MULTI_LAYER_BATCH_Q8K_PATH);
    // Content types: plain arms keep the historical f32 request CT on
    // walk-ffn; pairs put their inbound arm on the request CT.
    let plain_f16 = WireSpec::Plain(WireArm::F16);
    let pair = WireSpec::Pair {
        input: WireFormat::F16,
        output: WireFormat::I8,
    };
    assert_eq!(
        WalkFfn.request_content_type(plain_f16),
        larql_inference::BINARY_CT
    );
    assert_eq!(WalkFfn.request_content_type(pair), larql_inference::F16_CT);
    assert_eq!(
        WalkFfnQ8k.request_content_type(WireSpec::Plain(WireArm::Q8k)),
        larql_inference::Q8K_BATCH_CT
    );
    assert_eq!(
        ExpertsMultiLayer.request_content_type(WireSpec::Plain(WireArm::F32)),
        MULTI_LAYER_BATCH_CONTENT_TYPE
    );
    assert_eq!(
        ExpertsMultiLayerQ8k.request_content_type(WireSpec::Plain(WireArm::Q8k)),
        MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE
    );
    // Accept: walk-ffn negotiates per wire arm (pairs: strict return
    // arm); q8k walk-ffn none; both multi-layer endpoints ask for the
    // fixed f32 multi-layer CT.
    assert_eq!(WalkFfn.accept(plain_f16), WireArm::F16.accept());
    assert_eq!(WalkFfn.accept(pair), Some(larql_inference::I8_CT));
    assert_eq!(WalkFfnQ8k.accept(WireSpec::Plain(WireArm::Q8k)), None);
    assert_eq!(
        ExpertsMultiLayer.accept(WireSpec::Plain(WireArm::F32)),
        Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
    );
    assert_eq!(
        ExpertsMultiLayerQ8k.accept(WireSpec::Plain(WireArm::Q8k)),
        Some(MULTI_LAYER_BATCH_CONTENT_TYPE)
    );
    // Serve-latency source: only the f32 walk-ffn response embeds
    // latency in its header; the other three use the opt-in trailer.
    assert_eq!(
        WalkFfn.serve_latency_source(),
        ServeLatencySource::EmbeddedMs
    );
    for e in [WalkFfnQ8k, ExpertsMultiLayer, ExpertsMultiLayerQ8k] {
        assert_eq!(e.serve_latency_source(), ServeLatencySource::TimingTrailer);
    }
    // Denominator source + routing requirement.
    assert_eq!(WalkFfn.denominator(), DenominatorSource::Dense);
    assert_eq!(WalkFfnQ8k.denominator(), DenominatorSource::Dense);
    assert_eq!(
        ExpertsMultiLayer.denominator(),
        DenominatorSource::RoutedExperts
    );
    assert_eq!(
        ExpertsMultiLayerQ8k.denominator(),
        DenominatorSource::RoutedExperts
    );
    assert!(!WalkFfn.requires_routing());
    assert!(ExpertsMultiLayer.requires_routing());
    assert!(ExpertsMultiLayerQ8k.requires_routing());
}

#[test]
fn wire_label_maps_multi_layer_cts() {
    assert_eq!(
        wire_label_for_content_type(MULTI_LAYER_BATCH_CONTENT_TYPE),
        "experts-ml"
    );
    assert_eq!(
        wire_label_for_content_type(MULTI_LAYER_BATCH_Q8K_CONTENT_TYPE),
        "experts-ml-q8k"
    );
}
