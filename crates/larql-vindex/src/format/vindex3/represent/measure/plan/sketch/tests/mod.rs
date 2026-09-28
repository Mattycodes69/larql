//! The sketch's contract: the documented generator, linearity, shift
//! invariance and the file layout.

use super::*;

/// SplitMix64's published sequence for seed 0, and one value for another
/// seed and index, computed independently of this crate.
#[test]
fn splitmix64_matches_its_reference_sequence() {
    assert_eq!(splitmix64(0, 0), 0xe220_a839_7b1d_cdaf);
    assert_eq!(splitmix64(0, 1), 0x6e78_9e6a_a1b9_65f4);
    assert_eq!(splitmix64(0, 2), 0x06c4_5d18_8009_454f);
    assert_eq!(splitmix64(42, 5), 0xde44_31fa_3c80_db06);
}

/// Build `R` entry by entry from the module header's words, independently
/// of `project`'s loop, and apply it to an already-centred vector.
fn explicit(spec: &SketchSpec, centred: &[f64]) -> Vec<f32> {
    let words = spec.dim.div_ceil(64);
    let scale = (spec.dim as f64).sqrt();
    (0..spec.dim)
        .map(|k| {
            let sum: f64 = centred
                .iter()
                .enumerate()
                .map(|(j, &x)| {
                    let word = splitmix64(spec.seed, (j * words + k / 64) as u64);
                    if (word >> (k % 64)) & 1 == 1 {
                        x
                    } else {
                        -x
                    }
                })
                .sum();
            (sum / scale) as f32
        })
        .collect()
}

/// Seventy rows cross a word boundary, so both the per-word bit order and
/// the column's word index are exercised.
#[test]
fn project_is_the_documented_matrix() {
    let spec = SketchSpec::new(70, 7).unwrap();
    let candidate = [1.0f32, -2.0, 0.5, 3.0, 0.0];
    let reference = [0.0f32; 5];
    let mean = candidate.iter().map(|&x| f64::from(x)).sum::<f64>() / 5.0;
    let centred: Vec<f64> = candidate.iter().map(|&x| f64::from(x) - mean).collect();
    assert_eq!(
        spec.project(&reference, &candidate),
        explicit(&spec, &centred)
    );
}

/// The sketch of a sum is the sum of the sketches, which is what lets an
/// additivity test run on sketches instead of logits.
#[test]
fn project_is_linear() {
    let spec = SketchSpec::new(64, 11).unwrap();
    let zero = [0.0f32; 6];
    let a = [0.25f32, -1.0, 2.0, 0.0, 0.5, -0.75];
    let b = [1.0f32, 0.5, -0.5, 3.0, -2.0, 0.25];
    let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let (sa, sb, ss) = (
        spec.project(&zero, &a),
        spec.project(&zero, &b),
        spec.project(&zero, &sum),
    );
    for k in 0..spec.dim {
        assert!((sa[k] + sb[k] - ss[k]).abs() < 1e-5, "row {k}");
    }
}

/// Adding a constant to every candidate logit changes no probability, so it
/// must not change the sketch.
#[test]
fn project_ignores_a_constant_shift() {
    let spec = SketchSpec::new(32, 3).unwrap();
    let reference = [1.0f32, 2.0, 3.0, 4.0];
    let candidate = [2.0f32, 0.0, 5.0, 4.0];
    let shifted: Vec<f32> = candidate.iter().map(|x| x + 8.0).collect();
    assert_eq!(
        spec.project(&reference, &candidate),
        spec.project(&reference, &shifted)
    );
}

/// Identical rows sketch to zero; a real difference does not, and the seed
/// fixes which projection it is.
#[test]
fn project_separates_a_difference_and_depends_on_the_seed() {
    let spec = SketchSpec::new(16, 1).unwrap();
    let row = [0.5f32, -1.0, 2.0];
    assert!(spec.project(&row, &row).iter().all(|&x| x == 0.0));
    let moved = [0.5f32, 1.0, 2.0];
    let sketch = spec.project(&row, &moved);
    assert!(sketch.iter().any(|&x| x != 0.0));
    assert_eq!(sketch, spec.project(&row, &moved), "deterministic");
    let other = SketchSpec::new(16, 2).unwrap();
    assert_ne!(sketch, other.project(&row, &moved));
}

#[test]
fn a_zero_dimension_is_refused() {
    assert!(SketchSpec::new(0, 1).is_err());
}

#[test]
fn an_empty_row_sketches_to_zero() {
    let spec = SketchSpec::new(4, 1).unwrap();
    assert_eq!(spec.project(&[], &[]), vec![0.0; 4]);
}

#[test]
fn the_report_names_the_generator_and_the_file() {
    let d = SketchSpec::new(64, 9).unwrap().describe(12);
    assert_eq!(d["generator"], SKETCH_GENERATOR);
    assert_eq!(d["file"], SKETCH_FILE);
    assert_eq!(d["dim"], 64);
    assert_eq!(d["seed"], 9);
    assert_eq!(d["positions"], 12);
}

#[test]
fn sketch_bytes_are_row_major_little_endian() {
    let bytes = sketch_bytes(&[vec![1.0, -2.0], vec![0.5, 4.0]]);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(values, vec![1.0, -2.0, 0.5, 4.0]);
}
