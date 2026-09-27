//! Typed reads of a GGUF file's `{arch}.*` metadata namespace.
//!
//! llama.cpp writes per-architecture keys as `<general.architecture>.<suffix>`.
//! These accessors resolve the prefix once, so the loader and the per-family
//! GGUF hooks (see `detect::registry::GgufTranslation`) read keys the same way.

use super::constants::GGUF_GENERAL_ARCHITECTURE;
use super::types::{GgufFile, GgufValue};

impl GgufFile {
    /// The file's `general.architecture`, or `""` when it declares none.
    pub(crate) fn architecture(&self) -> &str {
        self.metadata
            .get(GGUF_GENERAL_ARCHITECTURE)
            .and_then(GgufValue::as_str)
            .unwrap_or("")
    }

    /// The value at `<arch>.<suffix>`, if present.
    pub(crate) fn arch_value(&self, suffix: &str) -> Option<&GgufValue> {
        self.metadata
            .get(&format!("{}.{suffix}", self.architecture()))
    }

    /// `<arch>.<suffix>` as a scalar `u32`.
    pub(crate) fn arch_u32_opt(&self, suffix: &str) -> Option<u32> {
        self.arch_value(suffix).and_then(GgufValue::as_u32)
    }

    /// `<arch>.<suffix>` as a `u32`: the scalar, or the maximum of a per-layer
    /// array (variable FFN widths); `0` when absent.
    pub(crate) fn arch_u32(&self, suffix: &str) -> u32 {
        match self.arch_value(suffix) {
            Some(GgufValue::Array(arr)) => {
                arr.iter().filter_map(GgufValue::as_u32).max().unwrap_or(0)
            }
            Some(v) => v.as_u32().unwrap_or(0),
            None => 0,
        }
    }

    /// `<arch>.<suffix>` as a per-layer `u32` array, if it is one.
    pub(crate) fn arch_u32_array(&self, suffix: &str) -> Option<Vec<u32>> {
        match self.arch_value(suffix) {
            Some(GgufValue::Array(arr)) => Some(arr.iter().filter_map(GgufValue::as_u32).collect()),
            _ => None,
        }
    }

    /// `<arch>.<suffix>` as a float.
    pub(crate) fn arch_f64(&self, suffix: &str) -> Option<f64> {
        self.arch_value(suffix).and_then(GgufValue::as_f64)
    }

    /// How many tensors' names end with `suffix`.
    pub(crate) fn count_tensors_ending_with(&self, suffix: &str) -> usize {
        self.tensor_infos
            .iter()
            .filter(|t| t.name().ends_with(suffix))
            .count()
    }
}
