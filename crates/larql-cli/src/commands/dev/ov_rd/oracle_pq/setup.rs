//! Setup: load the vindex, resolve heads / configs / prompt split, and fit
//! the static means, W_O-visible and PCA bases, PQ codebooks and (with
//! `--mode-d-check`) the Mode D residual tables every probe evaluates
//! against.

use std::collections::HashMap;
use std::time::Instant;

use larql_inference::ModelWeights;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};

use crate::commands::dev::ov_rd::basis::{
    build_roundtrip_bases, fit_z_pca_bases, WoRoundtripBasis, ZPcaBasis,
};
use crate::commands::dev::ov_rd::input::{
    limit_prompts_per_stratum, load_prompts, parse_head_spec, parse_pq_configs,
    split_prompt_records,
};
use crate::commands::dev::ov_rd::oracle_pq_fit::fit_pq_codebooks;
use crate::commands::dev::ov_rd::oracle_pq_mode_d::materialize_mode_d_tables;
use crate::commands::dev::ov_rd::pq::{ModeDTable, PqCodebook};
use crate::commands::dev::ov_rd::static_replace::fit_static_means;
use crate::commands::dev::ov_rd::stats::StaticHeadMeans;
use crate::commands::dev::ov_rd::types::{HeadId, PqConfig, PromptRecord};

use super::args::OraclePqArgs;
use super::probe::{FitContext, HeadConfigMap, ProbeResult};

pub(super) struct LoadedModel {
    pub(super) index: VectorIndex,
    pub(super) weights: ModelWeights,
    pub(super) tokenizer: tokenizers::Tokenizer,
}

pub(super) fn load_model(args: &OraclePqArgs) -> ProbeResult<LoadedModel> {
    eprintln!("Loading vindex: {}", args.index.display());
    let start = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let weights = load_model_weights_kquant(&args.index, &mut cb)?;
    let tokenizer = load_vindex_tokenizer(&args.index)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd oracle-pq currently supports dense FFN vindexes only".into());
    }
    eprintln!(
        "  {} layers, hidden_size={}, q_heads={}, head_dim={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        weights.num_q_heads,
        weights.head_dim,
        start.elapsed().as_secs_f64()
    );
    Ok(LoadedModel {
        index,
        weights,
        tokenizer,
    })
}

/// The heads and PQ configs under study; both must be non-empty.
pub(super) fn parse_targets(args: &OraclePqArgs) -> ProbeResult<(Vec<HeadId>, Vec<PqConfig>)> {
    let heads = parse_head_spec(&args.heads)?;
    if heads.is_empty() {
        return Err("no heads selected for oracle PQ".into());
    }
    let configs = parse_pq_configs(&args.configs)?;
    if configs.is_empty() {
        return Err("no PQ configs selected".into());
    }
    Ok((heads, configs))
}

pub(super) struct PromptSplit {
    pub(super) all: Vec<PromptRecord>,
    pub(super) fit: Vec<PromptRecord>,
    pub(super) eval: Vec<PromptRecord>,
}

pub(super) fn load_prompt_split(
    args: &OraclePqArgs,
    heads: &[HeadId],
    configs: &[PqConfig],
) -> ProbeResult<PromptSplit> {
    let mut prompts = load_prompts(&args.prompts, args.max_prompts)?;
    if let Some(max_per_stratum) = args.max_per_stratum {
        prompts = limit_prompts_per_stratum(prompts, max_per_stratum);
    }
    eprintln!("Selected heads: {:?}", heads);
    eprintln!("PQ configs: {:?}", configs);
    eprintln!("Prompts: {}", prompts.len());
    let (fit, eval) = if let Some(eval_mod) = args.eval_mod {
        split_prompt_records(&prompts, eval_mod, args.eval_offset)?
    } else {
        (prompts.clone(), prompts.clone())
    };
    eprintln!(
        "Oracle PQ split: fit_prompts={}, eval_prompts={}",
        fit.len(),
        eval.len()
    );
    Ok(PromptSplit {
        all: prompts,
        fit,
        eval,
    })
}

/// The fitted compression every probe is measured against.
pub(super) struct BaseFit {
    pub(super) means: HashMap<HeadId, StaticHeadMeans>,
    pub(super) bases: HashMap<HeadId, WoRoundtripBasis>,
    pub(super) pca_bases: HashMap<HeadId, ZPcaBasis>,
    pub(super) codebooks: HeadConfigMap<PqCodebook>,
    pub(super) mode_d_tables: HeadConfigMap<ModeDTable>,
}

pub(super) fn fit_base(
    model: &mut LoadedModel,
    args: &OraclePqArgs,
    heads: &[HeadId],
    configs: &[PqConfig],
    fit_prompts: &[PromptRecord],
    stratum_conditioned_groups: &[usize],
) -> ProbeResult<BaseFit> {
    let LoadedModel {
        index,
        weights,
        tokenizer,
    } = model;
    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(weights, index, tokenizer, fit_prompts, heads)?;

    eprintln!("Building W_O-visible bases");
    let bases = build_roundtrip_bases(weights, index, heads, args.sigma_rel_cutoff)?;

    eprintln!("Fitting empirical z-space PCA bases");
    let pca_bases = fit_z_pca_bases(
        weights,
        index,
        tokenizer,
        fit_prompts,
        heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting product quantizers");
    let codebooks = fit_pq_codebooks(
        weights,
        index,
        tokenizer,
        fit_prompts,
        heads,
        &bases,
        &means,
        &pca_bases,
        configs,
        args.pq_iters,
        stratum_conditioned_groups,
    )?;
    let mode_d_tables = if args.mode_d_check {
        eprintln!("Materializing Mode D residual-space tables");
        materialize_mode_d_tables(
            weights,
            index,
            heads,
            &bases,
            &means,
            &pca_bases,
            &codebooks,
            stratum_conditioned_groups,
        )?
    } else {
        HashMap::new()
    };
    Ok(BaseFit {
        means,
        bases,
        pca_bases,
        codebooks,
        mode_d_tables,
    })
}

/// The fit-time view over the loaded model and base fit.
pub(super) fn fit_context<'a>(
    model: &'a mut LoadedModel,
    base: &'a BaseFit,
    heads: &'a [HeadId],
    fit_prompts: &'a [PromptRecord],
    mode_d_check: bool,
) -> FitContext<'a> {
    FitContext {
        weights: &mut model.weights,
        index: &model.index,
        tokenizer: &model.tokenizer,
        fit_prompts,
        heads,
        bases: &base.bases,
        means: &base.means,
        pca_bases: &base.pca_bases,
        codebooks: &base.codebooks,
        mode_d_check,
    }
}
