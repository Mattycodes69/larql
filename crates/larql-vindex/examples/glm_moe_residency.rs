//! **What does one correctly-executing GLM-5.3-Flash sparse layer cost in
//! physical memory?**
//!
//! The bank is 288 experts, 6.78 GiB of native FP8, and top-8 routing
//! selects a small fraction of it per token. This measures what actually
//! enters RAM — against what the plan predicts — with the routed output
//! held fixed.
//!
//! # The invariant
//!
//! **Same token, same selected experts, same routed output; only the
//! physical access policy changes.** Every arm's output is compared
//! byte-for-byte against the first arm's, and a mismatch aborts: a
//! residency result over a changed computation is not a residency result.
//!
//! # Method
//!
//! `mincore(2)`, not fault counting, for the same reason
//! `vindex3_residency_probe` gives: Darwin's `MADV_DONTNEED` is lazy and
//! re-eviction unreliable, which makes `getrusage` deltas awkward here.
//! Faults are reported too, as a secondary signal, and the two are kept
//! visibly separate rather than blended.
//!
//! `MADV_RANDOM` is set on the mapping: kernel readahead would page in
//! the very sparsity being measured.
//!
//! # Scope
//!
//! This maps the CHECKPOINT's own safetensors shards, because GLM has no
//! VINDEX3 container yet (its plan is not admissible). So the layout
//! measured is the checkpoint's, in which layer 3's experts are 83–98 %
//! dense in their span with other layers' tensors interleaved. A
//! container that carves the expert bank into its own object would have a
//! different — very likely better — layout, and this number is the
//! baseline that claim will have to beat.
//!
//! ```text
//! cargo run --release -p larql-vindex --example glm_moe_residency -- \
//!     <checkpoint-dir> <layer> <input.f32> [--arms demand,advise,warm]
//! ```
// POSIX-only, like `vindex3_residency_probe` beside it: the measurement
// IS the page-fault behaviour, so there is no portable shape for it.
#[cfg(unix)]
#[path = "glm_moe_residency/probe.rs"]
mod probe;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    probe::run()
}

#[cfg(not(unix))]
fn main() {
    // mmap / madvise / msync / mincore / getrusage are POSIX. Windows would
    // need QueryWorkingSetEx and PrefetchVirtualMemory.
    eprintln!("glm_moe_residency is unix-only.");
}
