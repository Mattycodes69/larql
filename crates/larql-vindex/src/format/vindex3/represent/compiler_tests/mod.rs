//! `tests` for [`super`].

use super::*;
use crate::format::vindex3::represent::arena::StoredOperand;
use crate::format::vindex3::represent::compiler::{
    ContainerArtifactDigest, SourceDependency, SourceIdentity,
};
use crate::format::vindex3::represent::experiment::RoleScope;
use crate::format::vindex3::represent::map::Exception;
use crate::format::vindex3::represent::selection::{Refusal, Verdict};
use std::collections::BTreeMap;

const HIDDEN: usize = 512;
const INTER: usize = 256;
const EXPERTS: u32 = 4;

struct Fake {
    bytes: BTreeMap<String, Vec<u8>>,
}

impl Fake {
    fn kimi_layer(layer: u32, seed: f32) -> (Self, Vec<SourceTensor>) {
        let mut bytes = BTreeMap::new();
        let mut tensors = Vec::new();
        for e in 0..EXPERTS {
            for (proj, rows, cols) in [
                ("w1", INTER, HIDDEN),
                ("w2", HIDDEN, INTER),
                ("w3", INTER, HIDDEN),
            ] {
                let name = format!("{layer}.block_sparse_moe.experts.{e}.{proj}.weight");
                let v: Vec<u8> = (0..rows * cols)
                    .flat_map(|i| {
                        let f = ((i as f32) * 0.013 + seed + e as f32).sin();
                        ((f.to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                bytes.insert(name.clone(), v);
                tensors.push(SourceTensor {
                    name,
                    shape: vec![rows, cols],
                });
            }
        }
        (Self { bytes }, tensors)
    }
}

impl SourceOperands for Fake {
    fn load_stored(&self, operand: &OperandRef) -> Result<StoredOperand, VindexError> {
        Ok(StoredOperand {
            dtype: "BF16".into(),
            bytes: self
                .bytes
                .get(&operand.tensor)
                .ok_or_else(|| VindexError::Parse(format!("no `{}`", operand.tensor)))?
                .clone(),
        })
    }
}

fn map_for(exceptions: Vec<Exception>) -> PrecisionMap {
    PrecisionMap {
        name: "q2-layer1-q6".into(),
        encoding: "Q6_K".into(),
        roles: vec!["expert-weight".into()],
        exceptions,
    }
}

fn opts<'a>(object: &'a str, experts: u32, out: &'a std::path::Path) -> CompileOptions<'a> {
    CompileOptions {
        object,
        role: Role::ExpertWeight,
        experts,
        out,
        checkpoint: None,
    }
}

fn index(map: PrecisionMap) -> CandidateIndex {
    CandidateIndex::new(
        "Kimi-Linear-48B-A3B-Instruct",
        SourceDependency {
            identity: SourceIdentity::synthetic(
                "m".repeat(64),
                "g".repeat(64),
                [("target.expert_bank.bin".into(), "a".repeat(64))],
            ),
            locator_hint: "/somewhere/source.vindex3".into(),
        },
        "target.expert_bank",
        map,
    )
}

mod compiler_tests_basics;
mod compiler_tests_basics_2;
