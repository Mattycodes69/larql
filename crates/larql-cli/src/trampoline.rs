//! argv trampoline, run before clap parses.
//!
//! Research subcommands used to live at the top level. In a `research`
//! build, `larql <legacy-name> …` is rewritten to `larql dev <legacy-name> …`
//! so existing scripts keep working. In a build without the feature the
//! research verbs do not exist, so any research name (legacy or current)
//! is refused with a message that names the missing feature, instead of
//! clap's "unrecognized subcommand" and a "did you mean" for an unrelated
//! verb.

/// Names that used to be top-level verbs and now live under `larql dev`.
/// Every entry must resolve to a real `dev` subcommand (a test enforces it).
pub(crate) const LEGACY_DEV_NAMES: &[&str] = &[
    "weight-extract",
    "attention-extract",
    "vector-extract",
    "residuals",
    "predict",
    "index-gates",
    "walk",
    "attention-capture",
    "qk-templates",
    "qk-rank",
    "qk-modes",
    "ov-gate",
    "circuit-discover",
    "attn-bottleneck",
    "ffn-bottleneck",
    "ffn-overlap",
    "kg-bench",
    "trajectory-trace",
    "projection-test",
    "fingerprint-extract",
    "bottleneck-test",
    "embedding-jump",
    "bfs",
    "ffn-latency",
];

/// Current top-level verbs that exist only in a `research` build. A
/// research-build test checks each one is a real top-level subcommand.
#[cfg(any(test, not(feature = "research")))]
pub(crate) const RESEARCH_TOP_LEVEL: &[&str] = &[
    "dev",
    "k3-ledger",
    "parity",
    "moe-locality",
    "optimizer-mcp",
];

/// `larql shannon <sub>` subcommands that exist only in a `research` build.
#[cfg(any(test, not(feature = "research")))]
pub(crate) const RESEARCH_SHANNON_SUBCOMMANDS: &[&str] = &["verify"];

/// Process exit code for a research command in a non-research build
/// (clap's own usage-error code).
pub(crate) const RESEARCH_UNAVAILABLE_EXIT_CODE: i32 = 2;

/// A research command was invoked on a binary built without `research`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResearchUnavailable {
    /// The command as the user typed it, e.g. `walk` or `shannon verify`.
    pub(crate) command: String,
}

impl std::fmt::Display for ResearchUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "`larql {}` is a research command; this larql binary was built without \
             the `research` cargo feature (rebuild with `--features research`)",
            self.command
        )
    }
}

/// Prepare argv for clap: rewrite legacy research names (research build)
/// or refuse research commands (non-research build).
pub(crate) fn prepare_argv(args: Vec<String>) -> Result<Vec<String>, ResearchUnavailable> {
    #[cfg(feature = "research")]
    {
        Ok(rewrite_legacy_argv(args))
    }
    #[cfg(not(feature = "research"))]
    {
        match research_command(&args) {
            Some(command) => Err(ResearchUnavailable { command }),
            None => Ok(args),
        }
    }
}

/// Rewrite `larql <legacy-name> …` → `larql dev <legacy-name> …`.
#[cfg(feature = "research")]
pub(crate) fn rewrite_legacy_argv(args: Vec<String>) -> Vec<String> {
    if args.len() >= 2 && LEGACY_DEV_NAMES.contains(&args[1].as_str()) {
        let mut rewritten = Vec::with_capacity(args.len() + 1);
        rewritten.push(args[0].clone());
        rewritten.push("dev".to_string());
        rewritten.extend(args.into_iter().skip(1));
        return rewritten;
    }
    args
}

/// The research command named by `args`, if any.
#[cfg(not(feature = "research"))]
pub(crate) fn research_command(args: &[String]) -> Option<String> {
    let verb = args.get(1)?.as_str();
    if LEGACY_DEV_NAMES.contains(&verb) || RESEARCH_TOP_LEVEL.contains(&verb) {
        return Some(verb.to_string());
    }
    match args.get(2).map(String::as_str) {
        Some(sub) if verb == "shannon" && RESEARCH_SHANNON_SUBCOMMANDS.contains(&sub) => {
            Some(format!("{verb} {sub}"))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
