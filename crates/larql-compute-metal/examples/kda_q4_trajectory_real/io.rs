//! Checkpoint location, tensor reads, quantisation and synthetic inputs.

use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::kimi_layer::ExpertEncoding;
use larql_models::config::KdaGateForm;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[allow(unused_imports)]
use super::*;

/// The decay-gate form this checkpoint's FAMILY computes, read from
/// `config.json`'s `architectures` and its declared `gate_lower_bound`.
///
/// **Derived from the family, never from the value.** Kimi and GLM both
/// declare `gate_lower_bound: -5.0` and only GLM applies it, so a
/// checkpoint whose family this build does not judge is REFUSED rather
/// than defaulted — serving one form for the other is a 2.8x per-step
/// decay error that compounds with context.
pub(super) fn gate_form_of(dir: &Path) -> KdaGateForm {
    let cfg: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("parse config");
    let arch = cfg["architectures"][0].as_str().unwrap_or("").to_string();
    let declared = cfg["linear_attn_config"]["gate_lower_bound"].as_f64();
    match arch.as_str() {
        a if a.starts_with("KimiLinear") => {
            println!(
                "family {a}: KdaGateForm::Softplus (declares {declared:?}, applies it nowhere)"
            );
            KdaGateForm::Softplus
        }
        a if a.starts_with("Glm5Next") => {
            let lower_bound = declared.unwrap_or(-5.0) as f32;
            println!("family {a}: KdaGateForm::ClampedSigmoid {{ {lower_bound} }}");
            KdaGateForm::ClampedSigmoid { lower_bound }
        }
        a => panic!(
            "family {a:?} has no judged KDA decay-gate form. Declare it rather than \
             defaulting: gate_lower_bound is present on families that ignore it."
        ),
    }
}

/// One per-projection sensitivity arm: which projection it degraded,
/// its q|k|v bank, its `o_proj`, and its own offset table.
pub(super) type SensArm = (String, Vec<u8>, Vec<u8>, [ExpertOffset; CONV_STREAMS_N]);

/// One small matrix as loaded:
pub(super) type SmallLoaded = (&'static str, (Vec<u8>, Vec<f32>), String);

/// One precision arm: its encoding, its `q|k|v` bank, its `o_proj`, and
/// its own offset table. The table's ADDRESS matters — see where `arms`
/// is built.
pub(super) type Arm = (ExpertEncoding, Vec<u8>, Vec<u8>, [ExpertOffset; 3]);

/// One tensor located inside one shard.
pub(super) struct Located {
    pub(super) shard: PathBuf,
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: String,
}

pub(super) fn locate(dir: &Path, names: &[String]) -> HashMap<String, Located> {
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("model.safetensors.index.json")).expect("index json"),
    )
    .expect("parse index");
    let map = index["weight_map"].as_object().expect("weight_map");
    let mut headers: HashMap<PathBuf, serde_json::Value> = HashMap::new();
    let mut out = HashMap::new();
    for name in names {
        let shard = dir.join(
            map[name]
                .as_str()
                .unwrap_or_else(|| panic!("{name} not in index")),
        );
        let header = headers.entry(shard.clone()).or_insert_with(|| {
            let f = std::fs::File::open(&shard).expect("open shard");
            // SAFETY: read-only mapping of a file we do not mutate.
            let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap shard");
            let n = u64::from_le_bytes(m[..8].try_into().unwrap()) as usize;
            serde_json::from_slice(&m[8..8 + n]).expect("parse header")
        });
        let n = u64::from_le_bytes(
            std::fs::read(&shard).expect("read")[..8]
                .try_into()
                .unwrap(),
        ) as usize;
        let t = &header[name];
        let off = t["data_offsets"].as_array().expect("offsets");
        let base = 8 + n;
        out.insert(
            name.clone(),
            Located {
                shard,
                start: base + off[0].as_u64().unwrap() as usize,
                end: base + off[1].as_u64().unwrap() as usize,
                shape: t["shape"]
                    .as_array()
                    .expect("shape")
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect(),
                dtype: t["dtype"].as_str().expect("dtype").to_string(),
            },
        );
    }
    out
}

/// Raw bytes, and the f32 values those bytes denote.
///
/// **The KDA block is not one dtype.** Kimi ships its wide projections
/// BF16 and some per-channel vectors F32 in the same layer, so the
/// reader dispatches on the declared dtype and refuses anything else
/// rather than reinterpreting bytes it does not recognise. This also
/// makes "small tensors at source precision" a per-tensor fact, not a
/// per-block one.
pub(super) fn read_tensor(l: &Located) -> (Vec<u8>, Vec<f32>) {
    if l.dtype == "F32" {
        let f = std::fs::File::open(&l.shard).expect("open");
        // SAFETY: read-only mapping of a file we do not mutate.
        let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap");
        let bytes = m[l.start..l.end].to_vec();
        let exact = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        return (bytes, exact);
    }
    assert_eq!(l.dtype, "BF16", "unhandled dtype");
    let f = std::fs::File::open(&l.shard).expect("open");
    // SAFETY: read-only mapping of a file we do not mutate.
    let m = unsafe { memmap2::Mmap::map(&f) }.expect("mmap");
    let bytes = m[l.start..l.end].to_vec();
    let exact = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect();
    (bytes, exact)
}

/// The precision ladder, so a state error can be attributed to the
/// REPRESENTATION rather than to the binding: if Q8_0 and Q4_K land on
/// the same number, something other than the codec is moving the state.
pub(super) fn quantise(enc: ExpertEncoding, v: &[f32]) -> Vec<u8> {
    use larql_compute::cpu::ops::q4_common as q;
    match enc {
        ExpertEncoding::Q80 => q::quantize_q8_0(v),
        ExpertEncoding::Q6K => q::quantize_q6_k(v),
        ExpertEncoding::Q4K => q::quantize_q4_k(v),
        ExpertEncoding::Bf16 => panic!("bf16 is the reference, not a candidate"),
    }
}

/// Read a flat little-endian f32 file written by the oracle's
/// `--raw-dir`.
pub(super) fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Unit-RMS input, deterministic, shared by both arms.
pub(super) fn inputs(hidden: usize, steps: usize) -> Vec<Vec<f32>> {
    (0..steps)
        .map(|s| {
            let v: Vec<f32> = (0..hidden)
                .map(|i| ((i as f32) * 0.7351 + s as f32 * 1.9).sin())
                .collect();
            let rms = (v.iter().map(|x| x * x).sum::<f32>() / hidden as f32).sqrt();
            v.into_iter()
                .map(|x| x / rms.max(f32::MIN_POSITIVE))
                .collect()
        })
        .collect()
}
