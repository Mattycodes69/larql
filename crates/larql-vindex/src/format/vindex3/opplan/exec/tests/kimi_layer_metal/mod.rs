//! Rung 5d: one complete Kimi decoder layer on device, including the
//! router→expert-binding seam.
//!
//! The question this answers is not whether residuals and RMS norms can
//! run on Metal — they obviously can, and they cost nothing. It is
//! whether **Metal can compute the routing decision and consume it in
//! the grouped MoE without the host ever seeing a selected expert id.**
//! If it cannot, every layer pays the ~0.23 ms crossing rung 5a priced,
//! for the sake of eight integers, and the layer is not closed.
//!
//! The parity list below is deliberately long. Each of these semantics
//! was proved independently on the CPU already; what is under test now
//! is the DEVICE DEPENDENCY CHAIN, so every link gets its own
//! comparison rather than being inferred from the final vector.
//!
//! Two controls carried forward from the CPU router's own gates, because
//! they catch the failures that still produce plausible output:
//!   * weights gathered from the BIASED selection scores instead of the
//!     unbiased ones — preserves the selection, changes every routed
//!     contribution;
//!   * the correction bias omitted from selection — changes which
//!     experts run.
//!
//! ```text
//! LARQL_KIMI_KDA_LAYER_FIXTURE=/tmp/kimi_kda_layer_fixture \
//!   cargo test -p larql-vindex --features gpu --release --lib kimi_layer_metal -- --nocapture
//! ```

use larql_models::config::KdaGateForm;
use std::path::{Path, PathBuf};
use std::time::Instant;

use larql_compute::cpu::ops::q4_common::{quantize_q6_k, quantize_q8_0};
use larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT;
use larql_compute_metal::trait_impl::bf16_moe_block::{ExpertBankRef, MoeBlockCall, MoeFfnBanks};
use larql_compute_metal::trait_impl::grouped_experts::ExpertOffset;
use larql_compute_metal::trait_impl::grouped_experts::GroupedError;
use larql_compute_metal::trait_impl::kda::{KdaDeviceState, KdaDeviceWeights, KdaShape};
use larql_compute_metal::trait_impl::kimi_layer::{
    AttentionSpec, EncodedRegion, ExpertAddressing, ExpertEncoding, FfnSpec, KimiLayerWeights,
    KimiMoeWeights, ProjectionBank,
};
use larql_compute_metal::MetalBackend;
use larql_models::config::KdaGeometry;
use serde_json::Value;

use crate::format::vindex3::opplan::exec::cpu::projector::WeightRows;
use crate::format::vindex3::opplan::exec::kda::{zero_state, KdaOutputGateWeights, KdaWeights};
use crate::format::vindex3::opplan::exec::kimi_kda_layer::kda_decoder_layer_forward;
use crate::format::vindex3::opplan::exec::kimi_moe_block::ExpertWeights;

const FIXTURE_ENV: &str = "LARQL_KIMI_KDA_LAYER_FIXTURE";
/// The same ceiling the CPU layer's own oracle gate uses.
const TOLERANCE: f32 = 3e-4;
/// Warmup and repeats for the timing report. Workload-shaped, per the
/// block rung's lesson: one layer moves ~200 MiB.
const WARMUP: usize = 15;
const ITERS: usize = 15;

fn fixture_dir() -> Option<PathBuf> {
    std::env::var_os(FIXTURE_ENV).map(PathBuf::from)
}

fn read_f32(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{name}.f32: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_bf16_bytes(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(format!("{name}.bf16"))).unwrap_or_else(|e| panic!("{name}.bf16: {e}"))
}

fn codes(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// One projection's routed bank, its table, and the shared branch's own
/// region — everything bf16, as the checkpoint stores it.
fn projection<'a>(routed: &'a [u8], table: &'a [u32], shared: &'a [u8]) -> ProjectionBank<'a> {
    ProjectionBank {
        routed: EncodedRegion {
            bytes: routed,
            encoding: ExpertEncoding::Bf16,
        },
        addressing: ExpertAddressing::Table(table),
        shared: Some(EncodedRegion {
            bytes: shared,
            encoding: ExpertEncoding::Bf16,
        }),
    }
}

/// One layer's weights, owned, plus the resident expert bank the device
/// path binds.
///
/// **Only the selected experts plus the shared branch are resident.**
/// The router still scores all 256 from the real `[256, 2304]` matrix —
/// selection is genuine — but the bank holds nine, and every other
/// expert maps to `NOT_RESIDENT`. That is honest about what a fixture
/// can hold and is exactly the condition the device-side refusal counter
/// exists for: if the router ever picked an absent expert, the step
/// would be refused rather than served another expert's weights.
struct Fixture {
    hidden: usize,
    inter: usize,
    experts: usize,
    top_k: usize,
    renormalize: bool,
    branch_scale: f32,
    eps: f32,
    geometry: KdaGeometry,
    ids_order: Vec<usize>,

    x: Vec<f32>,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    router_weight: Vec<f32>,
    router_bias: Vec<f32>,
    oracle_layer_output: Vec<f32>,

    // KDA
    qkv_bank: Vec<u8>,
    qkv_offsets: [ExpertOffset; 3],
    o_proj: Vec<u8>,
    kda_f32: KdaF32,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,

    // MoE: gate/up/down banks over the resident experts. The shared
    // branch lives in its OWN allocations — semantic identity, never
    // co-location.
    bank_gate: Vec<u8>,
    bank_up: Vec<u8>,
    bank_down: Vec<u8>,
    residency: Vec<u32>,
    shared_gate: Vec<u8>,
    shared_up: Vec<u8>,
    shared_down: Vec<u8>,
    /// Widened codes, for the CPU arm.
    cpu_experts: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>,
    cpu_shared: (Vec<u16>, Vec<u16>, Vec<u16>),
}

/// KDA's f32 vectors, grouped so `Fixture` stays readable.
struct KdaF32 {
    qc: Vec<f32>,
    kc: Vec<f32>,
    vc: Vec<f32>,
    fa: Vec<f32>,
    fb: Vec<f32>,
    ga: Vec<f32>,
    gb: Vec<f32>,
    bp: Vec<f32>,
    al: Vec<f32>,
    dt: Vec<f32>,
    on: Vec<f32>,
}

impl Fixture {
    fn kda_cpu(&self) -> KdaWeights<'_> {
        let f = &self.kda_f32;
        KdaWeights {
            gate_form: KdaGateForm::Softplus, // Kimi-derived fixture: the reference reads `gate_lower_bound` nowhere.
            q_proj: WeightRows::Bf16(&self.q),
            k_proj: WeightRows::Bf16(&self.k),
            v_proj: WeightRows::Bf16(&self.v),
            q_conv1d: &f.qc,
            k_conv1d: &f.kc,
            v_conv1d: &f.vc,
            f_a_proj: &f.fa,
            f_b_proj: &f.fb,
            output_gate: KdaOutputGateWeights::LowRank {
                g_a_proj: &f.ga,
                g_b_proj: &f.gb,
            },
            b_proj: &f.bp,
            a_log: &f.al,
            dt_bias: &f.dt,
            o_norm: &f.on,
            o_proj: WeightRows::Bf16(&self.o),
            norm_eps: self.eps,
            // The rank the gate factorisations meet at — this fixture's
            // own `f_a_proj`, not the head dim the executor used to assume.
            gate_rank: f.fa.len() / self.hidden,
        }
    }

    fn kda_device(&self) -> KdaDeviceWeights<'_> {
        let f = &self.kda_f32;
        KdaDeviceWeights {
            qkv_bank: &self.qkv_bank,
            qkv_offsets: &self.qkv_offsets,
            o_proj: &self.o_proj,
            projection_encoding: ExpertEncoding::Bf16,
            gate_form: larql_models::config::KdaGateForm::Softplus,
            q_conv1d: &f.qc,
            k_conv1d: &f.kc,
            v_conv1d: &f.vc,
            f_a_proj: larql_compute_metal::trait_impl::kda::SmallMatrix::F32(&f.fa),
            f_b_proj: larql_compute_metal::trait_impl::kda::SmallMatrix::F32(&f.fb),
            g_a_proj: larql_compute_metal::trait_impl::kda::SmallMatrix::F32(&f.ga),

            g_b_proj: larql_compute_metal::trait_impl::kda::SmallMatrix::F32(&f.gb),
            b_proj: larql_compute_metal::trait_impl::kda::SmallMatrix::F32(&f.bp),
            a_log: &f.al,
            dt_bias: &f.dt,
            o_norm: &f.on,
            norm_eps: self.eps,
        }
    }

    fn layer<'a>(&'a self, state: &'a KdaDeviceState) -> KimiLayerWeights<'a> {
        KimiLayerWeights {
            input_norm: &self.input_norm,
            post_attention_norm: &self.post_norm,
            attention: AttentionSpec::Kda {
                weights: self.kda_device(),
                shape: self.shape(),
                state,
            },
            ffn: FfnSpec::Moe(KimiMoeWeights {
                router_weight: &self.router_weight,
                router_bias: &self.router_bias,
                gate: projection(&self.bank_gate, &self.residency, &self.shared_gate),
                up: projection(&self.bank_up, &self.residency, &self.shared_up),
                down: projection(&self.bank_down, &self.residency, &self.shared_down),
                inter: self.inter,
                top_k: self.top_k,
                renormalize: self.renormalize,
                branch_scale: self.branch_scale,
            }),
            norm_eps: self.eps,
        }
    }

    /// `input_layernorm(x)` on the host — only for the attention-alone
    /// decomposition, which needs the same input the layer's own first
    /// dispatch computes.
    fn input_norm_applied(&self) -> Vec<f32> {
        crate::format::vindex3::opplan::exec::kernels::norm(
            larql_models::config::NormType::RmsNorm,
            &self.x,
            &self.input_norm,
            0.0,
            self.eps as f64,
        )
    }

    fn shape(&self) -> KdaShape {
        KdaShape {
            hidden: self.hidden,
            num_heads: self.geometry.num_heads,
            head_dim: self.geometry.head_dim,
            conv_kernel: self.geometry.conv_kernel,
        }
    }
}

fn load(dir: &Path) -> Fixture {
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    let g = |k: &str| manifest[k].as_u64().unwrap() as usize;
    let (hidden, inter, experts, top_k) = (
        g("hidden"),
        g("moe_intermediate_size"),
        g("experts"),
        g("top_k"),
    );
    let geometry = KdaGeometry {
        num_heads: g("num_heads"),
        head_dim: g("head_dim"),
        conv_kernel: 4,
    };
    let ids_order: Vec<usize> = manifest["selected_ids_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!((experts, top_k), (256, 8), "this gate runs REAL geometry");

    let kda = |n: &str| read_f32(dir, &format!("kda_{n}"));
    let (qb, kb, vb) = (
        read_bf16_bytes(dir, "kda_q_proj"),
        read_bf16_bytes(dir, "kda_k_proj"),
        read_bf16_bytes(dir, "kda_v_proj"),
    );
    let per_qkv = qb.len();
    let mut qkv_bank = Vec::with_capacity(3 * per_qkv);
    for b in [&qb, &kb, &vb] {
        qkv_bank.extend_from_slice(b);
    }
    let o_proj = read_bf16_bytes(dir, "kda_o_proj");

    // The resident bank: the selected experts in `selected_ids_order`,
    // then the shared branch. Identity lives in the residency table, not
    // in this order — the router will pick whatever it picks.
    let mut bank_gate = Vec::new();
    let mut bank_up = Vec::new();
    let mut bank_down = Vec::new();
    let mut residency = vec![NOT_RESIDENT; experts];
    let mut cpu_experts = Vec::with_capacity(ids_order.len());
    for &id in &ids_order {
        residency[id] = bank_gate.len() as u32;
        let (g1, g3, g2) = (
            read_bf16_bytes(dir, &format!("expert{id}_w1")),
            read_bf16_bytes(dir, &format!("expert{id}_w3")),
            read_bf16_bytes(dir, &format!("expert{id}_w2")),
        );
        cpu_experts.push((codes(&g1), codes(&g3), codes(&g2)));
        bank_gate.extend_from_slice(&g1);
        bank_up.extend_from_slice(&g3);
        bank_down.extend_from_slice(&g2);
    }
    let (s1, s3, s2) = (
        read_bf16_bytes(dir, "shared_w1"),
        read_bf16_bytes(dir, "shared_w3"),
        read_bf16_bytes(dir, "shared_w2"),
    );
    let cpu_shared = (codes(&s1), codes(&s3), codes(&s2));

    Fixture {
        hidden,
        inter,
        experts,
        top_k,
        renormalize: manifest["moe_renormalize"].as_bool().unwrap(),
        branch_scale: manifest["routed_scaling_factor"].as_f64().unwrap() as f32,
        eps: manifest["rms_eps"].as_f64().unwrap() as f32,
        geometry,
        ids_order,
        x: read_f32(dir, "input"),
        input_norm: read_f32(dir, "input_norm_weight"),
        post_norm: read_f32(dir, "post_attention_norm_weight"),
        router_weight: read_f32(dir, "router_weight"),
        router_bias: read_f32(dir, "router_bias"),
        oracle_layer_output: read_f32(dir, "out_layer_output"),
        q: codes(&qb),
        k: codes(&kb),
        v: codes(&vb),
        o: codes(&o_proj),
        qkv_offsets: [
            ExpertOffset(0),
            ExpertOffset(per_qkv as u32),
            ExpertOffset((2 * per_qkv) as u32),
        ],
        qkv_bank,
        o_proj,
        kda_f32: KdaF32 {
            qc: kda("q_conv1d"),
            kc: kda("k_conv1d"),
            vc: kda("v_conv1d"),
            fa: kda("f_a_proj"),
            fb: kda("f_b_proj"),
            ga: kda("g_a_proj"),
            gb: kda("g_b_proj"),
            bp: kda("b_proj"),
            al: kda("a_log"),
            dt: kda("dt_bias"),
            on: kda("o_norm"),
        },
        bank_gate,
        bank_up,
        bank_down,
        residency,
        shared_gate: s1,
        shared_up: s3,
        shared_down: s2,
        cpu_experts,
        cpu_shared,
    }
}

fn setup() -> Option<(MetalBackend, Fixture)> {
    let dir = match fixture_dir() {
        Some(d) => d,
        None => {
            eprintln!("skipped: set {FIXTURE_ENV} to the exported fixture directory");
            return None;
        }
    };
    let metal = match MetalBackend::new() {
        Some(m) => m,
        None => {
            // On macOS this means the shader library failed to compile,
            // not that a device is missing — skipping there turns a
            // broken build into a green run.
            #[cfg(target_os = "macos")]
            panic!(
                "MetalBackend::new() returned None on macOS — the shader library \
                 almost certainly failed to compile. Run `cargo test -p \
                 larql-compute-metal --lib`."
            );
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("skipped: no Metal device on this host");
                return None;
            }
        }
    };
    Some((metal, load(&dir)))
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The CPU arm: the proven whole-layer path.
fn cpu_layer(
    fx: &Fixture,
) -> crate::format::vindex3::opplan::exec::kimi_kda_layer::KdaDecoderLayerTrace {
    let by_id = |id: usize| -> ExpertWeights<'_> {
        let slot = fx
            .ids_order
            .iter()
            .position(|&i| i == id)
            .unwrap_or_else(|| panic!("layer asked for un-resident expert {id}"));
        let (gate, up, down) = &fx.cpu_experts[slot];
        ExpertWeights { gate, up, down }
    };
    let shared = ExpertWeights {
        gate: &fx.cpu_shared.0,
        up: &fx.cpu_shared.1,
        down: &fx.cpu_shared.2,
    };
    let mut state = zero_state(fx.geometry);
    kda_decoder_layer_forward(
        &fx.x,
        fx.hidden,
        &fx.input_norm,
        &fx.post_norm,
        fx.eps as f64,
        fx.kda_cpu(),
        fx.geometry,
        &mut state,
        fx.inter,
        &fx.router_weight,
        &fx.router_bias,
        fx.experts,
        fx.top_k,
        fx.renormalize,
        fx.branch_scale as f64,
        by_id,
        Some((shared, fx.inter)),
    )
}

/// Reach the routed weights of a layer, for the controls that corrupt
/// one field. Panics on a dense layer, which no control here builds.
fn moe_mut<'a, 'b>(
    w: &'b mut larql_compute_metal::trait_impl::kimi_layer::KimiLayerWeights<'a>,
) -> &'b mut larql_compute_metal::trait_impl::kimi_layer::KimiMoeWeights<'a> {
    match &mut w.ffn {
        FfnSpec::Moe(m) => m,
        FfnSpec::Dense(_) => panic!("this control corrupts a routed layer"),
    }
}

// **C rung 4 — projection identity, location, backing and encoding are
// independent.**
//
// Runs on the REAL layer fixture because the claim needs superblock-
// aligned shapes: at hidden 2304 / inter 1024 both projections are
// whole multiples of 256, while a toy fixture's 48-element projections
// cannot encode Q6_K at all — a test there would prove nothing about
// mixed representation.
//
// The four properties are varied AT ONCE, which is the point. Mixed
// encoding over identity addresses would leave open that some shared
// physical coordinate still exists; independent permutations under one
// encoding would leave open that representation is still a property of
// "the bank". Together they close both.
/// Three permutations of the bank's blocks that differ at every index.
fn perms(blocks: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // Three PAIRWISE DISCORDANT permutations: no index maps to the same
    // place in any two of them.
    //
    // Affine maps `a*i + b` will not do. Two of them differ everywhere
    // only when their multipliers are equal, which makes them rotations
    // of one another — and a rotation is exactly what one hidden
    // coordinate plus a constant could still reproduce. So: seeded
    // shuffles, searched deterministically until a discordant triple
    // falls out.
    let shuffled = |seed: u64| {
        let mut v: Vec<usize> = (0..blocks).collect();
        let mut st = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        for i in (1..blocks).rev() {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            v.swap(i, (st >> 33) as usize % (i + 1));
        }
        v
    };
    let discordant = |a: &[usize], b: &[usize]| a.iter().zip(b).all(|(x, y)| x != y);
    for s in 0..4096u64 {
        let g = shuffled(s);
        for t in s + 1..s + 64 {
            let u = shuffled(t);
            if !discordant(&g, &u) {
                continue;
            }
            for w in t + 1..t + 64 {
                let d = shuffled(w);
                if discordant(&g, &d) && discordant(&u, &d) {
                    return (g, u, d);
                }
            }
        }
    }
    panic!("no discordant triple found for {blocks} blocks");
}

fn permute(src: &[u8], per: usize, perm: &[usize]) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for (i, &p) in perm.iter().enumerate() {
        out[p * per..(p + 1) * per].copy_from_slice(&src[i * per..(i + 1) * per]);
    }
    out
}

fn widen(bf16: &[u8]) -> Vec<f32> {
    bf16.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Re-encode each block of a bf16 bank to Q6_K, block by block, so the
/// result is a bank of the same block count at a smaller stride.
fn to_q6k_blocks(src: &[u8], per: usize, n: usize, k: usize) -> (Vec<u8>, usize) {
    assert!(k.is_multiple_of(256), "k={k} cannot be Q6_K");
    let q_per = n * k / 256 * 210;
    let mut out = Vec::with_capacity(src.len() / per * q_per);
    for block in src.chunks_exact(per) {
        let q = quantize_q6_k(&widen(block));
        assert_eq!(q.len(), q_per);
        out.extend_from_slice(&q);
    }
    (out, q_per)
}

fn table(residency: &[u32], src_per: usize, dst_per: usize, perm: &[usize]) -> Vec<u32> {
    residency
        .iter()
        .map(|off| {
            if *off == larql_compute_metal::shaders::kimi_layer::NOT_RESIDENT {
                *off
            } else {
                (perm[*off as usize / src_per] * dst_per) as u32
            }
        })
        .collect()
}

/// Re-encode each block of a bf16 bank to Q8_0, block by block — the
/// Q8_0 sibling of `to_q6k_blocks`, at the 34-bytes-per-32 stride.
fn to_q8_0_blocks(src: &[u8], per: usize, n: usize, k: usize) -> (Vec<u8>, usize) {
    assert!(k.is_multiple_of(32), "k={k} cannot be Q8_0");
    let q_per = n * k / 32 * 34;
    let mut out = Vec::with_capacity(src.len() / per * q_per);
    for block in src.chunks_exact(per) {
        let q = quantize_q8_0(&widen(block));
        assert_eq!(q.len(), q_per);
        out.extend_from_slice(&q);
    }
    (out, q_per)
}

mod kimi_layer_metal_basics;
mod kimi_layer_metal_basics_2;
