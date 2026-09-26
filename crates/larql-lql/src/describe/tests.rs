use super::*;

#[test]
fn band_names_parse_case_insensitively() {
    assert_eq!(band_from_name("Syntax"), Some(LayerBand::Syntax));
    assert_eq!(band_from_name("knowledge"), Some(LayerBand::Knowledge));
    assert_eq!(band_from_name("OUTPUT"), Some(LayerBand::Output));
    assert_eq!(band_from_name("all"), Some(LayerBand::All));
    assert_eq!(band_from_name("knowldge"), None);
}

#[test]
fn brief_mode_caps_fewer_edges_than_verbose() {
    assert!(edge_cap(DescribeMode::Brief) < edge_cap(DescribeMode::Verbose));
    assert_eq!(edge_cap(DescribeMode::Raw), edge_cap(DescribeMode::Verbose));
}

#[test]
fn undeclared_bands_on_an_unknown_family_span_the_whole_model() {
    let config = VindexConfig {
        family: "unknown-family".into(),
        num_layers: 6,
        layer_bands: None,
        ..Default::default()
    };
    let bands = layer_bands(&config);
    assert_eq!(bands.knowledge, (0, 5));
    assert_eq!(bands.syntax, (0, 5));
}
