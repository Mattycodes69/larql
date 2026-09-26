use larql_vindex_spec::{SlicePreset, UNSLICED_PRESET};

#[test]
fn every_spelling_parses_to_its_preset() {
    for &preset in SlicePreset::ALL {
        for spelling in preset.spellings() {
            assert_eq!(spelling.parse::<SlicePreset>().unwrap(), preset);
            assert_eq!(
                spelling.to_uppercase().parse::<SlicePreset>().unwrap(),
                preset
            );
        }
    }
}

#[test]
fn no_spelling_is_claimed_twice() {
    let mut all: Vec<&str> = SlicePreset::ALL
        .iter()
        .flat_map(|p| p.spellings().iter().copied())
        .collect();
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n);
}

#[test]
fn unknown_names_are_refused_with_the_known_list() {
    let err = "xyz".parse::<SlicePreset>().unwrap_err().to_string();
    assert!(
        err.contains("xyz") && err.contains("expert-server"),
        "{err}"
    );
}

#[test]
fn the_unsliced_name_is_not_a_slice_preset() {
    assert!(UNSLICED_PRESET.parse::<SlicePreset>().is_err());
}

#[test]
fn display_is_the_canonical_name() {
    assert_eq!(SlicePreset::Attention.to_string(), "attn");
}
