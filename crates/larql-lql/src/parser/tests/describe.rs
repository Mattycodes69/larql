//! DESCRIBE

use super::*;

#[test]
fn parse_describe_minimal() {
    let stmt = parse(r#"DESCRIBE "France";"#).unwrap();
    match stmt {
        Statement::Describe {
            entity,
            band,
            layer,
            relations_only,
            mode,
        } => {
            assert_eq!(entity, "France");
            assert!(band.is_none());
            assert!(layer.is_none());
            assert!(!relations_only);
            assert_eq!(mode, DescribeMode::Brief); // brief is the default (clean output)
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_at_layer() {
    let stmt = parse(r#"DESCRIBE "Mozart" AT LAYER 26;"#).unwrap();
    match stmt {
        Statement::Describe {
            entity,
            band,
            layer,
            ..
        } => {
            assert_eq!(entity, "Mozart");
            assert!(band.is_none());
            assert_eq!(layer, Some(26));
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe { relations_only, .. } => assert!(relations_only),
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_layer_and_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" AT LAYER 26 RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe {
            layer,
            relations_only,
            ..
        } => {
            assert_eq!(layer, Some(26));
            assert!(relations_only);
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_syntax() {
    let stmt = parse(r#"DESCRIBE "def" SYNTAX;"#).unwrap();
    match stmt {
        Statement::Describe { entity, band, .. } => {
            assert_eq!(entity, "def");
            assert_eq!(band, Some(LayerBand::Syntax));
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_knowledge() {
    let stmt = parse(r#"DESCRIBE "France" KNOWLEDGE;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::Knowledge));
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_output() {
    let stmt = parse(r#"DESCRIBE "France" OUTPUT;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::Output));
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_all_layers() {
    let stmt = parse(r#"DESCRIBE "France" ALL LAYERS;"#).unwrap();
    match stmt {
        Statement::Describe { band, .. } => {
            assert_eq!(band, Some(LayerBand::All));
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_band_with_relations_only() {
    let stmt = parse(r#"DESCRIBE "France" KNOWLEDGE RELATIONS ONLY;"#).unwrap();
    match stmt {
        Statement::Describe {
            band,
            relations_only,
            ..
        } => {
            assert_eq!(band, Some(LayerBand::Knowledge));
            assert!(relations_only);
        }
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_verbose() {
    let stmt = parse(r#"DESCRIBE "France" VERBOSE;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Verbose),
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_brief() {
    let stmt = parse(r#"DESCRIBE "France" BRIEF;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Brief),
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_raw() {
    let stmt = parse(r#"DESCRIBE "France" RAW;"#).unwrap();
    match stmt {
        Statement::Describe { mode, .. } => assert_eq!(mode, DescribeMode::Raw),
        _ => panic!("expected Describe"),
    }
}

#[test]
fn parse_describe_band_verbose() {
    let stmt = parse(r#"DESCRIBE "France" ALL LAYERS VERBOSE;"#).unwrap();
    match stmt {
        Statement::Describe { band, mode, .. } => {
            assert_eq!(band, Some(LayerBand::All));
            assert_eq!(mode, DescribeMode::Verbose);
        }
        _ => panic!("expected Describe"),
    }
}
