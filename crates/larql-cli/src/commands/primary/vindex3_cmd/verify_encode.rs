//! `vindex3 verify` and `vindex3 encode`.

#[allow(unused_imports)]
use super::*;

pub(super) fn run_verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        // Verification re-reads every payload byte to re-hash it, so an
        // `hf://` artifact would re-transfer the whole checkpoint — the
        // one thing the remote encode exists to avoid. Refuse by name
        // rather than let it look like a path that does not exist.
        if artifact::is_remote_spec(path) {
            return Err(format!(
                "`{}` is a repo, and verification re-reads every payload byte to \
                 re-hash it — pointing it at a repo would re-transfer the whole \
                 checkpoint. Verify against a local checkpoint, or check the \
                 container's own recorded payload_sha256.",
                path.display()
            )
            .into());
        }
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let verification =
        larql_vindex::format::vindex3::verify_system::verify_system(&named, &args.container)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&verification)?);
    } else {
        for defect in &verification.container_defects {
            println!("container defect: {defect}");
        }
        let semantic_pass = verification.semantic.iter().filter(|c| c.pass).count();
        println!(
            "semantic: {semantic_pass}/{} authority checks pass",
            verification.semantic.len()
        );
        for failure in verification.semantic_failures() {
            println!(
                "  FAIL {} ≡ {}  {}: {}",
                failure.left, failure.right, failure.subject, failure.detail
            );
        }
        println!("payloads:");
        for check in &verification.payloads {
            println!(
                "  {} {:40} {:8.3} GB  {}",
                if check.pass { "PASS" } else { "FAIL" },
                check.representation,
                check.payload_bytes as f64 / 1e9,
                if check.pass {
                    format!(
                        "sha256 {}…",
                        &check.recorded_sha256[..12.min(check.recorded_sha256.len())]
                    )
                } else {
                    check.detail.clone()
                },
            );
        }
    }
    if verification.verified {
        eprintln!("verified: Declared ≡ Resolved ≡ Graph ≡ Encoded; payloads byte-equal");
        Ok(())
    } else {
        Err(format!(
            "NOT verified: {} semantic failure(s), {} payload failure(s), {} container defect(s)",
            verification.semantic_failures().count(),
            verification.payload_failures().count(),
            verification.container_defects.len(),
        )
        .into())
    }
}

/// What staging one repo-backed artifact cost, and what it stands in for.
///
/// The library returns these figures rather than printing them, so this is
/// where `larql vindex3` gets its voice back. Headers and metadata are
/// quoted separately and then totalled: the header figure alone
/// understates the transfer, since a tokenizer can outweigh every shard
/// header put together.
pub(super) fn report_staging(entry: &artifact::ResolvedArtifact) {
    let Some(report) = entry.staging() else {
        return;
    };
    if let Some(commit) = entry.commit() {
        eprintln!("artifact `{}` pinned at commit {commit}", entry.name);
    }
    if let Some(revision) = entry.unpinned_revision() {
        eprintln!(
            "warning: the hub named no commit for `{revision}` — provenance records \
             the revision name, which can move"
        );
    }
    eprintln!(
        "staged {} ({} of headers over {} shard(s), {} of metadata)",
        artifact::size(report.staged_bytes()),
        artifact::size(report.header_bytes),
        report.shards,
        artifact::size(report.metadata_bytes),
    );
    match &report.payload_bytes {
        Ok(payload) => {
            eprintln!(
                "  standing in for {} of tensor payload",
                artifact::size(*payload)
            );
            // Only when the index disagrees with its own headers: tied
            // weights are counted once there and serialised twice in the
            // file, and a silent 7% gap reads like a units bug.
            if let Some(declared) = report.declared_total.filter(|d| d != payload) {
                eprintln!(
                    "  note: the shard index declares {} — {} {} its own headers sum to; \
                     the header sum is what transfers",
                    artifact::size(declared),
                    artifact::size(declared.abs_diff(*payload)),
                    if declared < *payload {
                        "less than"
                    } else {
                        "more than"
                    },
                );
            }
        }
        // A census failure is not a reason to abort: the encode reads the
        // same headers and will fail with a better message.
        Err(err) => eprintln!("warning: could not total the staged headers: {err}"),
    }
}

pub(super) fn run_encode(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let resolved = artifact::resolve_all(&args.artifacts)?;
    for entry in &resolved {
        report_staging(entry);
    }
    if let Some(capability) = args.capability {
        eprintln!(
            "admission scoped to {:?}; whole-model completeness is NOT asserted",
            capability
        );
    }

    // One ingest, shared with `vindex encode`. Two orchestrations here
    // would be free to differ on the capability snapshot alone, and a
    // container that binds with token-ids only is not obviously wrong —
    // it just answers differently.
    let outcome =
        artifact::encode_from_specs(resolved, &args.output, args.capability.map(Into::into))?;

    for transfer in &outcome.transfers {
        eprintln!(
            "{}: fetched {} of {} declared ({:.1}%) across {} tensor(s); \
             the checkpoint was never on this disk",
            transfer.name,
            artifact::size(transfer.fetched),
            artifact::size(transfer.declared),
            if transfer.declared == 0 {
                0.0
            } else {
                100.0 * transfer.fetched as f64 / transfer.declared as f64
            },
            transfer.tensors,
        );
    }
    if !outcome.capabilities.is_empty() {
        eprintln!("capabilities: {}", outcome.capabilities.join(", "));
    }
    eprintln!(
        "encoded {} representation(s), {} payload → {}",
        outcome.representations,
        artifact::size(outcome.total_payload_bytes),
        outcome.container.display(),
    );
    Ok(())
}
