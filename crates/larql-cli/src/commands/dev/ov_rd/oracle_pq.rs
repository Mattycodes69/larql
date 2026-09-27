//! `ov-rd oracle-pq`: oracle product quantisation of selected attention
//! heads' pre-W_O outputs, with Mode D residual tables and a registry of
//! discrete address probes. `run` is the orchestrator; each probe family
//! is one [`probe::AddressProbe`] implementation under `probes/`.

mod args;
mod diagnostics;
mod eval_loop;
mod probe;
mod probes;
mod registry;
mod report;
mod run;
mod setup;
mod specs;
mod validate;

pub(super) use args::OraclePqArgs;
pub(super) use run::run_oracle_pq;

#[cfg(test)]
mod tests;
