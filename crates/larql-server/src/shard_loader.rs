//! Mode B shard downloader — streams a tar from the donor's `/v1/shard`
//! endpoint to a temp file while hashing it, verifies the SHA-256 against
//! the CONTENT hash the router forwarded (`AssignMsg.shard_sha256`), and
//! unpacks it into `store_path/{model_id}/layers-{start}-{end}/`.
//!
//! Verification is mandatory. A missing or placeholder content hash is a
//! hard error unless the operator opted in with `--allow-unverified-shards`
//! ([`UnverifiedShards::Allow`]); a value that is not a SHA-256 at all (an
//! identity hash sent in the wrong field, say) is always an error.
//!
//! The unpack is atomic: the tar is unpacked into a sibling `.tmp` directory
//! that is renamed onto the final path on success. A partial download leaves
//! a `.tmp` directory behind which the next attempt removes; the downloaded
//! tar file itself is always removed.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

use crate::shard_archive::{is_sha256_hex, sha256_hex};

const SHARD_ENDPOINT: &str = "/v1/shard";

/// Whole-request timeout for downloading one shard tar (connect + full body):
/// 10 minutes, sized for multi-GB layer tars over LAN links.
const SHARD_DOWNLOAD_TIMEOUT_SECS: u64 = 600;

/// Upper bound on a `model_id`, so a hostile peer cannot push a path near
/// the filesystem's own limit.
pub const MAX_MODEL_ID_LEN: usize = 128;

/// Whether a shard may be loaded without a content hash to verify against.
/// Maps to the server's `--allow-unverified-shards` flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnverifiedShards {
    /// Missing/placeholder content hash is a hard error (the default).
    #[default]
    Refuse,
    /// Missing/placeholder content hash downloads unverified, with a warning.
    Allow,
}

impl UnverifiedShards {
    pub fn from_flag(allow: bool) -> Self {
        if allow {
            Self::Allow
        } else {
            Self::Refuse
        }
    }
}

/// One shard download request — the fields of an `AssignMsg`.
#[derive(Clone, Copy, Debug)]
pub struct ShardFetch<'a> {
    pub origin_url: &'a str,
    pub model_id: &'a str,
    pub layer_start: u32,
    pub layer_end: u32,
    /// `AssignMsg.shard_sha256`: lowercase-hex SHA-256 of the origin's tar.
    pub expected_sha256: &'a str,
}

/// What a successful load established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShardLoaded {
    /// Downloaded and verified against this content hash.
    Verified(String),
    /// Downloaded without verification (`--allow-unverified-shards`).
    Unverified,
    /// The destination already existed; nothing was downloaded or checked.
    AlreadyPresent,
}

impl ShardLoaded {
    /// The content hash to report in `ReadyMsg.shard_sha256` — only a hash
    /// this load actually checked.
    pub fn verified_sha256(&self) -> &str {
        match self {
            Self::Verified(h) => h,
            Self::Unverified | Self::AlreadyPresent => "",
        }
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Reject a `model_id` that cannot safely become one path segment.
///
/// `model_id` arrives in the router's `AssignMsg` — it is remote input,
/// not a local configuration value — and every use below joins it into a
/// path that is then `create_dir_all`'d and unpacked into. Without this,
/// a router (or anyone who can reach this node's announce socket) can
/// set `model_id` to `../../../../etc/cron.d` and choose where a tar
/// lands on this filesystem.
///
/// A single path component of the conservative character set is all any
/// real model id needs; anything else is refused rather than sanitised,
/// because silently rewriting an id would make the shard land somewhere
/// the router did not ask for and the mismatch would surface later as a
/// confusing cache miss.
fn validated_model_id(model_id: &str) -> Result<&str, String> {
    if model_id.is_empty() {
        return Err("model_id is empty".into());
    }
    if model_id.len() > MAX_MODEL_ID_LEN {
        return Err(format!(
            "model_id is {} bytes, over the {MAX_MODEL_ID_LEN}-byte limit",
            model_id.len()
        ));
    }
    // `..` and separators are the traversal primitives; `.` alone would
    // resolve to the store root. Windows accepts `\\` as a separator, so
    // reject it too even though this path is unix-first.
    if model_id == "." || model_id == ".." {
        return Err(format!("model_id `{model_id}` is a directory reference"));
    }
    if model_id.contains("..") {
        return Err(format!("model_id `{model_id}` contains `..`"));
    }
    if !model_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(format!(
            "model_id `{model_id}` has characters outside [A-Za-z0-9._-]"
        ));
    }
    Ok(model_id)
}

/// Resolve the content hash to verify against, or refuse.
///
/// `Ok(None)` means "download unverified" and is only reachable under
/// [`UnverifiedShards::Allow`].
fn expected_digest(raw: &str, policy: UnverifiedShards) -> Result<Option<String>, String> {
    let placeholder = raw.bytes().all(|b| b == b'0');
    if placeholder {
        return match policy {
            UnverifiedShards::Allow => Ok(None),
            UnverifiedShards::Refuse => Err(format!(
                "no shard content hash (got {raw:?}); refusing an unverified shard — \
                 run with --allow-unverified-shards to accept one"
            )),
        };
    }
    let digest = raw.to_ascii_lowercase();
    if !is_sha256_hex(&digest) {
        return Err(format!(
            "shard content hash {raw:?} is not a SHA-256 hex digest \
             (an identity hash in the content-hash field?)"
        ));
    }
    Ok(Some(digest))
}

/// Download a shard tar from `fetch.origin_url`, verify its content hash,
/// and atomically unpack it to
/// `store_path/{model_id}/layers-{layer_start}-{layer_end}/`.
pub async fn download_and_load_shard(
    fetch: ShardFetch<'_>,
    store_path: &str,
    policy: UnverifiedShards,
) -> Result<ShardLoaded, BoxError> {
    let ShardFetch {
        origin_url,
        model_id,
        layer_start,
        layer_end,
        expected_sha256,
    } = fetch;
    let model_id = validated_model_id(model_id).map_err(|e| {
        warn!(%e, "Mode B: refusing shard download — unsafe model_id");
        e
    })?;
    let expected = expected_digest(expected_sha256, policy).map_err(|e| {
        warn!(%e, "Mode B: refusing shard download");
        e
    })?;

    let model_dir = PathBuf::from(store_path).join(model_id);
    let shard_dir = model_dir.join(format!("layers-{layer_start}-{layer_end}"));
    let tmp_dir = model_dir.join(format!(".tmp-layers-{layer_start}-{layer_end}"));
    let tar_path = model_dir.join(format!(".tmp-layers-{layer_start}-{layer_end}.tar"));

    tokio::fs::create_dir_all(&model_dir).await?;

    // Remove a stale tmp directory from an earlier aborted attempt.
    if tokio::fs::metadata(&tmp_dir).await.is_ok() {
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
    }
    // If the final shard already exists, treat as success (idempotent).
    if tokio::fs::metadata(&shard_dir).await.is_ok() {
        info!(dest = %shard_dir.display(), "Mode B: shard already present — skipping download");
        return Ok(ShardLoaded::AlreadyPresent);
    }

    let url = format!(
        "{}{SHARD_ENDPOINT}/{model_id}/{layer_start}-{layer_end}",
        origin_url.trim_end_matches('/')
    );
    info!(url = %url, dest = %shard_dir.display(), "Mode B: downloading shard tar…");

    let result = fetch_verify_unpack(&url, &tar_path, &tmp_dir, expected.as_deref()).await;
    // The tar is scratch: remove it whether the load succeeded or not.
    let _ = tokio::fs::remove_file(&tar_path).await;
    let loaded = match result {
        Ok(loaded) => loaded,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
            return Err(e);
        }
    };

    // Atomic rename onto the final path.
    if let Err(e) = tokio::fs::rename(&tmp_dir, &shard_dir).await {
        // Best-effort cleanup of the half-unpacked tmp dir.
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
        return Err(format!(
            "atomic rename {} -> {} failed: {e}",
            tmp_dir.display(),
            shard_dir.display()
        )
        .into());
    }

    info!(dest = %shard_dir.display(), "Mode B: shard unpacked — ready");
    Ok(loaded)
}

/// Stream `url` into `tar_path` while hashing, check the digest, then
/// unpack the file into `tmp_dir`.
async fn fetch_verify_unpack(
    url: &str,
    tar_path: &Path,
    tmp_dir: &Path,
    expected: Option<&str>,
) -> Result<ShardLoaded, BoxError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(SHARD_DOWNLOAD_TIMEOUT_SECS))
        .build()?;
    let mut resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        return Err(format!("shard download failed: HTTP {} from {url}", resp.status()).into());
    }

    let mut file = tokio::fs::File::create(tar_path).await?;
    let mut hasher = Sha256::new();
    let mut bytes: u64 = 0;
    while let Some(chunk) = resp.chunk().await? {
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        bytes += chunk.len() as u64;
    }
    file.flush().await?;
    drop(file);
    let got = sha256_hex(hasher);
    info!(bytes, sha256 = %got, "Mode B: download complete");

    let loaded = match expected {
        Some(want) if want == got => {
            info!("Mode B: content hash verified");
            ShardLoaded::Verified(got)
        }
        Some(want) => {
            return Err(format!("shard hash mismatch: expected {want}, got {got}").into());
        }
        None => {
            warn!(sha256 = %got, "Mode B: shard loaded UNVERIFIED (--allow-unverified-shards)");
            ShardLoaded::Unverified
        }
    };

    // Unpack in a blocking task — `tar::Archive` is sync I/O.
    let tar_path = tar_path.to_path_buf();
    let tmp_dir = tmp_dir.to_path_buf();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        std::fs::create_dir_all(&tmp_dir)?;
        let mut archive = tar::Archive::new(std::fs::File::open(&tar_path)?);
        archive.unpack(&tmp_dir)
    })
    .await
    .map_err(|e| format!("unpack task join failed: {e}"))??;
    Ok(loaded)
}

/// Where a shard lands. `None` for a `model_id` that is not a safe single
/// path segment — callers must refuse rather than fall back, since every
/// fallback here is a write outside the store.
pub fn shard_dest_path(store_path: &str, model_id: &str, start: u32, end: u32) -> Option<PathBuf> {
    let model_id = validated_model_id(model_id).ok()?;
    Some(
        Path::new(store_path)
            .join(model_id)
            .join(format!("layers-{start}-{end}")),
    )
}
