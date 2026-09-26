//! `larql serve`: the server flags this CLI forwards, and the exec into
//! the `larql-server` binary.

#[derive(clap::Args)]
pub(crate) struct ServeArgs {
    /// Path to a .vindex directory (or `hf://` path).
    #[arg(value_name = "VINDEX_PATH")]
    pub(crate) vindex_path: Option<String>,

    /// VINDEX3 execution backend (requires a matching larql-server build).
    #[arg(long, value_parser = ["cpu", "metal"])]
    pub(crate) v3_backend: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    pub(crate) dir: Option<std::path::PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    pub(crate) port: u16,

    /// Bind address. Loopback by default; a network address needs
    /// `--api-key` or `--insecure-public`.
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    pub(crate) no_infer: bool,

    /// Run as an FFN-service endpoint for remote clients using
    /// `larql run --ffn URL`. Disables `/v1/infer` and advertises
    /// `mode: ffn-service` in `/v1/stats`. Act 2 of the demo.
    #[arg(long)]
    pub(crate) ffn_only: bool,

    /// Cap decoded f16 gate layers via LRU (bounds server RSS). 0 = unlimited.
    /// On 31B each layer decodes to ~433 MB, so 60 layers = ~26 GB.
    /// Set to N to cap at N layers; evicted layers are re-decoded on access.
    #[arg(long, default_value = "0")]
    pub(crate) max_gate_cache_layers: usize,

    /// madvise(MADV_DONTNEED) on all mmaps after each walk-ffn request.
    /// Enforces a hard RSS bound alongside --max-gate-cache-layers at the
    /// cost of re-fault per request. Prefer --layers sharding for real
    /// deployments (sharding never touches out-of-range pages).
    #[arg(long)]
    pub(crate) release_mmap_after_request: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    pub(crate) cors: bool,

    /// API key for authentication.
    #[arg(long)]
    pub(crate) api_key: Option<String>,

    /// Allow a non-loopback `--host` without `--api-key`.
    #[arg(long)]
    pub(crate) insecure_public: bool,

    /// Directory `POST /v1/patches` may load patch files from.
    #[arg(long, value_name = "DIR")]
    pub(crate) patch_dir: Option<std::path::PathBuf>,

    /// Accept `hf://` patch references on `POST /v1/patches`.
    #[arg(long)]
    pub(crate) allow_hf_patches: bool,

    /// Rate limit per IP (e.g. "100/min", "10/sec").
    #[arg(long)]
    pub(crate) rate_limit: Option<String>,

    /// Max concurrent requests.
    #[arg(long, default_value = "100")]
    pub(crate) max_concurrent: usize,

    /// Cache TTL for DESCRIBE results in seconds (0 = disabled).
    #[arg(long, default_value = "0")]
    pub(crate) cache_ttl: u64,

    /// gRPC port.
    #[arg(long)]
    pub(crate) grpc_port: Option<u16>,

    /// TLS certificate path.
    #[arg(long)]
    pub(crate) tls_cert: Option<std::path::PathBuf>,

    /// TLS private key path.
    #[arg(long)]
    pub(crate) tls_key: Option<std::path::PathBuf>,

    /// Logging level.
    #[arg(long, default_value = "info")]
    pub(crate) log_level: String,

    /// Only load and serve layers in this range (inclusive, e.g. "0-19").
    /// Pages outside the range are never touched; RSS scales with shard size.
    #[arg(long)]
    pub(crate) layers: Option<String>,

    /// Only load and serve experts in this range (inclusive, e.g. "0-63").
    /// Used to shard the expert bank across servers for MoE models.
    /// Mutually exclusive with --units.
    #[arg(long)]
    pub(crate) experts: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) ownership.
    /// Mutually exclusive with --experts.
    #[arg(long, value_name = "PATH")]
    pub(crate) units: Option<std::path::PathBuf>,

    /// Run as an embed-service endpoint (loads only embeddings + lm_head).
    #[arg(long)]
    pub(crate) embed_only: bool,

    /// Eager-build HNSW index for every owned layer at startup. Requires --hnsw.
    #[arg(long)]
    pub(crate) warmup_hnsw: bool,

    /// Pre-load inference weights and prefetch all owned layer mmap pages at boot.
    #[arg(long)]
    pub(crate) warmup_walk_ffn: bool,

    /// Bind a Unix domain socket alongside TCP for same-host MoE shard clients.
    #[arg(long, value_name = "PATH")]
    pub(crate) uds_path: Option<std::path::PathBuf>,

    /// Join one or more router grids (comma-separated gRPC addresses).
    /// Example: "grpc://router-a:50052,grpc://router-b:50052"
    /// Requires --public-url so routers know where to direct clients.
    #[arg(long)]
    pub(crate) join: Option<String>,

    /// Public HTTP URL clients use to reach this server (used with --join).
    #[arg(long)]
    pub(crate) public_url: Option<String>,

    /// Shared secret matching the router's --grid-key (or set LARQL_GRID_KEY env var).
    #[arg(long)]
    pub(crate) grid_key: Option<String>,

    /// Trust X-Forwarded-For when rate limiting (enable only behind a trusted proxy).
    #[arg(long)]
    pub(crate) trust_forwarded_for: bool,

    /// Server-side MoE expert shard map: `"START-END=URL,START-END=URL,..."`
    /// The walk-ffn handler will dispatch MoE expert calls to these remote servers.
    /// Combine with --layers for full 2D (layer × expert) sharding.
    #[arg(long)]
    pub(crate) moe_shards: Option<String>,

    /// Path to a JSON manifest for fine-grained per-(layer, expert) shard ownership.
    /// Mutually exclusive with --moe-shards.
    #[arg(long, value_name = "PATH")]
    pub(crate) moe_units_manifest: Option<std::path::PathBuf>,
}

pub(crate) fn serve_command_args(
    args: &ServeArgs,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut cmd_args = Vec::new();
    if let Some(ref path) = args.vindex_path {
        // Resolve cache shorthands / owner-name / hf:// → actual path so
        // `larql serve gemma3-4b-v2` works the same as `larql run`. A
        // name the VINDEX3 registry has claimed resolves through it
        // exclusively (no fallback on failure); everything else keeps
        // today's cache-shorthand/hf:///local-path behaviour, and a real
        // resolution failure now propagates instead of being silently
        // replaced by the raw, unresolved string. See
        // `commands::primary::serve_resolve` module docs.
        cmd_args.push(super::serve_resolve::resolve_serve_target(path)?);
    }
    if let Some(ref dir) = args.dir {
        cmd_args.push("--dir".into());
        cmd_args.push(dir.display().to_string());
    }
    if let Some(ref backend) = args.v3_backend {
        cmd_args.push("--v3-backend".into());
        cmd_args.push(backend.clone());
    }
    cmd_args.push("--port".into());
    cmd_args.push(args.port.to_string());
    cmd_args.push("--host".into());
    cmd_args.push(args.host.clone());
    cmd_args.push("--log-level".into());
    cmd_args.push(args.log_level.clone());
    cmd_args.push("--max-concurrent".into());
    cmd_args.push(args.max_concurrent.to_string());
    if args.no_infer {
        cmd_args.push("--no-infer".into());
    }
    if args.ffn_only {
        cmd_args.push("--ffn-only".into());
    }
    if args.max_gate_cache_layers > 0 {
        cmd_args.push("--max-gate-cache-layers".into());
        cmd_args.push(args.max_gate_cache_layers.to_string());
    }
    if args.release_mmap_after_request {
        cmd_args.push("--release-mmap-after-request".into());
    }
    if args.cors {
        cmd_args.push("--cors".into());
    }
    if let Some(ref key) = args.api_key {
        cmd_args.push("--api-key".into());
        cmd_args.push(key.clone());
    }
    if args.insecure_public {
        cmd_args.push("--insecure-public".into());
    }
    if let Some(ref dir) = args.patch_dir {
        cmd_args.push("--patch-dir".into());
        cmd_args.push(dir.display().to_string());
    }
    if args.allow_hf_patches {
        cmd_args.push("--allow-hf-patches".into());
    }
    if let Some(ref rl) = args.rate_limit {
        cmd_args.push("--rate-limit".into());
        cmd_args.push(rl.clone());
    }
    if args.cache_ttl > 0 {
        cmd_args.push("--cache-ttl".into());
        cmd_args.push(args.cache_ttl.to_string());
    }
    if let Some(port) = args.grpc_port {
        cmd_args.push("--grpc-port".into());
        cmd_args.push(port.to_string());
    }
    if let Some(ref cert) = args.tls_cert {
        cmd_args.push("--tls-cert".into());
        cmd_args.push(cert.display().to_string());
    }
    if let Some(ref key) = args.tls_key {
        cmd_args.push("--tls-key".into());
        cmd_args.push(key.display().to_string());
    }
    if let Some(ref range) = args.layers {
        cmd_args.push("--layers".into());
        cmd_args.push(range.clone());
    }
    if let Some(ref range) = args.experts {
        cmd_args.push("--experts".into());
        cmd_args.push(range.clone());
    }
    if let Some(ref path) = args.units {
        cmd_args.push("--units".into());
        cmd_args.push(path.display().to_string());
    }
    if args.embed_only {
        cmd_args.push("--embed-only".into());
    }
    if args.warmup_hnsw {
        cmd_args.push("--warmup-hnsw".into());
    }
    if args.warmup_walk_ffn {
        cmd_args.push("--warmup-walk-ffn".into());
    }
    if let Some(ref path) = args.uds_path {
        cmd_args.push("--uds-path".into());
        cmd_args.push(path.display().to_string());
    }
    if let Some(ref addrs) = args.join {
        cmd_args.push("--join".into());
        cmd_args.push(addrs.clone());
    }
    if let Some(ref url) = args.public_url {
        cmd_args.push("--public-url".into());
        cmd_args.push(url.clone());
    }
    if let Some(ref key) = args.grid_key {
        cmd_args.push("--grid-key".into());
        cmd_args.push(key.clone());
    }
    if args.trust_forwarded_for {
        cmd_args.push("--trust-forwarded-for".into());
    }
    if let Some(ref s) = args.moe_shards {
        cmd_args.push("--moe-shards".into());
        cmd_args.push(s.clone());
    }
    if let Some(ref path) = args.moe_units_manifest {
        cmd_args.push("--moe-units-manifest".into());
        cmd_args.push(path.display().to_string());
    }

    Ok(cmd_args)
}

pub(crate) fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cmd_args = serve_command_args(&args)?;
    let exe = std::env::current_exe().ok();
    let server_bin = exe
        .as_ref()
        .and_then(|e| e.parent())
        .map(|d| d.join("larql-server"))
        .filter(|p| p.exists());

    let bin = server_bin
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "larql-server".into());

    let status = std::process::Command::new(&bin).args(&cmd_args).status();

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("larql-server exited with: {s}").into()),
        Err(e) => {
            eprintln!("Failed to exec larql-server: {e}");
            eprintln!(
                "Make sure larql-server is installed (cargo install --path crates/larql-server)"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests;
