//! What a packed expert bank, already widened to f32, may become.
//!
//! `from_f32` is the last step of every non-native bank load: the codec
//! has decoded one expert's rows to f32 and the backend's declared format
//! decides what is kept. Formats that promise STORED bytes cannot be
//! honoured from a widened image and must be refused by name; formats
//! that are a function of the values are produced.

use super::super::super::backend::WeightFormat;
use super::super::super::weights::LoadedWeight;
use super::super::from_f32;
use larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;

/// Rows of the fixture expert.
const ROWS: usize = 2;
/// One MXFP4/NVFP4 group wide, so every value-derived format can pack it.
const K: usize = MXFP4_GROUP_ELEMS;
/// The bank name every refusal must carry.
const NAME: &str = "experts.gate_up_proj";

fn values() -> Vec<f32> {
    (0..ROWS * K)
        .map(|i| i as f32 / (ROWS * K) as f32)
        .collect()
}

/// Every format whose promise is a stored form is refused, naming the
/// bank and the reason the widened image cannot keep that promise.
#[test]
fn stored_form_formats_are_refused_for_a_widened_bank() {
    let cases = [
        (WeightFormat::Fp8Block, "FP8-resident"),
        (WeightFormat::Q4, "q4-resident"),
        (WeightFormat::Nvfp4Q8, "stored NVFP4 pack"),
        (WeightFormat::KQuant, "stored K-quant"),
        (WeightFormat::KQuantQ8k, "stored K-quant"),
        (
            WeightFormat::Bf16,
            "compact residency needs the stored bytes",
        ),
        (WeightFormat::Q8, "compact residency needs the stored bytes"),
        (WeightFormat::CodecOwned, "codec-owned bytes"),
    ];
    for (format, reason) in cases {
        let err = from_f32(values(), ROWS, K, format, NAME)
            .expect_err("a stored-form format must be refused for a widened bank")
            .to_string();
        assert!(err.contains(NAME), "{format:?}: {err}");
        assert!(err.contains(reason), "{format:?}: {err}");
    }
}

/// Formats that are a function of the values are produced from them.
#[test]
fn value_derived_formats_are_produced_from_a_widened_bank() {
    assert!(matches!(
        from_f32(values(), ROWS, K, WeightFormat::F32, NAME).unwrap(),
        LoadedWeight::F32(_)
    ));
    assert!(matches!(
        from_f32(values(), ROWS, K, WeightFormat::F16, NAME).unwrap(),
        LoadedWeight::F16(_)
    ));
    assert!(matches!(
        from_f32(values(), ROWS, K, WeightFormat::Mxfp4, NAME).unwrap(),
        LoadedWeight::Mxfp4 { .. }
    ));
    assert!(matches!(
        from_f32(values(), ROWS, K, WeightFormat::Nvfp4, NAME).unwrap(),
        LoadedWeight::Nvfp4 { .. }
    ));
}
