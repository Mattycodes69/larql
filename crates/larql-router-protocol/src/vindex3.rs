//! V3 stateless layer-prefix RPC. Positions always start at zero; no remote
//! continuation cache, retries or hidden server affinity are part of this ABI.
use serde::{Deserialize, Serialize};
/// Schema 2 carries the execution identity (RESIDUAL-BUS-2 I3). A
/// schema-1 peer cannot assert which computation it performs, so it is
/// refused by name rather than accepted with its identity assumed.
pub const SCHEMA: u32 = 2;
/// Length of a SHA-256 digest written as lowercase hex.
const DIGEST_HEX_LEN: usize = 64;
pub const PATH: &str = "/v1/vindex3/layers";
pub const MAX_POSITIONS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub schema: u32,
    pub artifact: String,
    pub backend: String,
    /// Numerical provider family and semantic revision.
    pub lowering: String,
    /// Digest of the worker's execution identity: the model authority,
    /// slice, lowering, every pinned realization and the process
    /// arithmetic that change values. Derived from the prepared image,
    /// never from configuration.
    pub execution_identity: String,
    /// Half-open range of plan layer indices.
    pub start: usize,
    pub end: usize,
    pub layers: usize,
    pub hidden: usize,
}
impl Binding {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!(
                "peer speaks V3 binding schema {}, and this build requires schema {SCHEMA}, \
                 which carries the execution identity an exact route needs",
                self.schema
            ));
        }
        if !is_digest(&self.execution_identity) {
            return Err("a V3 shard binding's execution identity is not a digest".into());
        }
        if self.backend != "cpu"
            || self.lowering.is_empty()
            || self.hidden == 0
            || self.start >= self.end
            || self.end > self.layers
            || !is_digest(&self.artifact)
        {
            return Err("invalid or unsupported V3 shard binding".into());
        }
        Ok(())
    }
    pub fn validate_rows(&self, rows: &[Vec<f32>]) -> Result<(), String> {
        self.validate()?;
        if rows.is_empty()
            || rows.len() > MAX_POSITIONS
            || rows
                .iter()
                .any(|r| r.len() != self.hidden || r.iter().any(|x| !x.is_finite()))
        {
            return Err(format!(
                "V3 shard input requires 1..={MAX_POSITIONS} rows of {} finite values",
                self.hidden
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub binding: Binding,
    pub rows: Vec<Vec<f32>>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub binding: Binding,
    pub rows: Vec<Vec<f32>>,
}

fn is_digest(value: &str) -> bool {
    value.len() == DIGEST_HEX_LEN && value.bytes().all(|c| c.is_ascii_hexdigit())
}
