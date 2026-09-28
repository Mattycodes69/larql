//! The logit sketch at procedure level: written only when asked for, one row
//! per position, and never changing the rest of the record.

use super::*;
use crate::format::vindex3::represent::measure::plan::sketch::{
    SketchSpec, SKETCH_FILE, SKETCH_GENERATOR,
};

const DIM: usize = 64;
const SEED: u64 = 20_260_928;

fn sketched(f: &Fixture, output: PathBuf) -> PlanMeasureRequest {
    PlanMeasureRequest {
        output,
        sketch: Some(SketchSpec::new(DIM, SEED).unwrap()),
        ..request(f)
    }
}

fn report(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(dir.join(REPORT_FILE)).unwrap()).unwrap()
}

#[test]
fn without_a_sketch_the_record_has_none() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    run(&request(&f), &mut r, &mut c).expect("admissible");
    assert!(!f.output.join(SKETCH_FILE).exists());
    assert!(report(&f.output).get("sketch").is_none());
}

/// Control 4 of AUTO-REP-LANDSCAPE-1: asking for the sketch changes no
/// per-position number, and the sketch holds one row per position.
#[test]
fn a_sketch_adds_one_row_per_position_and_changes_no_position() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let plain = run(&request(&f), &mut r, &mut c).expect("admissible");
    let out = f.output.with_file_name("sketched");
    let (mut r, mut c) = reference_and_candidate(&f);
    let with = run(&sketched(&f, out.clone()), &mut r, &mut c).expect("admissible");
    assert_eq!(plain.summary, with.summary);
    assert_eq!(
        std::fs::read(f.output.join(POSITIONS_FILE)).unwrap(),
        std::fs::read(out.join(POSITIONS_FILE)).unwrap()
    );
    let positions = with.facts.positions as usize;
    let bytes = std::fs::read(out.join(SKETCH_FILE)).unwrap();
    assert_eq!(bytes.len(), positions * DIM * std::mem::size_of::<f32>());
    let described = &report(&out)["sketch"];
    assert_eq!(described["generator"], SKETCH_GENERATOR);
    assert_eq!(described["dim"], DIM);
    assert_eq!(described["seed"], SEED);
    assert_eq!(described["positions"], positions);
}

/// The first row is the documented projection of the arms' own logits at
/// sample 0, position 0: the procedure sketches what it measured.
#[test]
fn the_first_row_is_the_projection_of_the_first_position() {
    let f = fixture();
    let (mut r, mut c) = reference_and_candidate(&f);
    let out = f.output.clone();
    run(&sketched(&f, out.clone()), &mut r, &mut c).expect("admissible");
    let bank = crate::format::vindex3::represent::token_bank::TokenBank::open(&f.bank).unwrap();
    let ids = bank.read(0).unwrap();
    let (mut r, mut c) = reference_and_candidate(&f);
    let reference = r.score(&ids).unwrap();
    let candidate = c.score(&ids).unwrap();
    let expected = SketchSpec::new(DIM, SEED)
        .unwrap()
        .project(&reference[0], &candidate[0]);
    let bytes = std::fs::read(out.join(SKETCH_FILE)).unwrap();
    let first: Vec<f32> = bytes[..DIM * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(first, expected);
    assert!(first.iter().any(|&x| x != 0.0), "NVFP4 moves the logits");
}
