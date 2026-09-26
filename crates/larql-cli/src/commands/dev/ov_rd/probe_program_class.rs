use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use clap::Args;
use larql_vindex::{
    load_model_weights_kquant, load_vindex_tokenizer, SilentLoadCallbacks, VectorIndex,
};
use serde::Serialize;

use super::basis::{build_roundtrip_bases, fit_z_pca_bases};
use super::input::{
    limit_prompts_per_stratum, load_prompts, parse_head_spec, split_prompt_records,
};
use super::oracle_pq_fit::fit_pq_codebooks;
use super::oracle_pq_mode_d::materialize_mode_d_tables;
use super::program::Program;
use super::static_replace::fit_static_means;
use super::types::{HeadId, PqConfig};

mod capture;
mod metrics;
use capture::*;
use metrics::*;

#[derive(Args)]
pub struct ProbeProgramClassArgs {
    #[arg(long)]
    pub index: PathBuf,

    #[arg(long)]
    pub program: PathBuf,

    #[arg(long)]
    pub prompts: PathBuf,

    #[arg(long)]
    pub out: PathBuf,

    /// Optional override/guard for the program head, formatted as layer:head.
    #[arg(long)]
    pub head: Option<String>,

    /// Optional override/guard for the program group.
    #[arg(long)]
    pub group: Option<usize>,

    /// Comma-separated source list: residual_input,pre_wo_head_output,symbolic.
    #[arg(long, default_value = "residual_input,pre_wo_head_output,symbolic")]
    pub sources: String,

    /// Maximum prompts per stratum. 0 = unlimited.
    #[arg(long, default_value_t = 0)]
    pub max_per_stratum: usize,

    #[arg(long, default_value_t = 1)]
    pub eval_mod: usize,

    #[arg(long, default_value_t = 0)]
    pub eval_offset: usize,

    #[arg(long, default_value_t = 1e-6)]
    pub sigma_rel_cutoff: f64,

    #[arg(long, default_value_t = 25)]
    pub pq_iters: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
enum ProbeSource {
    ResidualInput,
    PreWoHeadOutput,
    Symbolic,
}

impl ProbeSource {
    fn parse_list(spec: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        for part in spec.split(',') {
            let source = match part.trim() {
                "" => continue,
                "residual_input" => Self::ResidualInput,
                "pre_wo_head_output" => Self::PreWoHeadOutput,
                "symbolic" => Self::Symbolic,
                other => return Err(format!("unknown probe source '{other}'").into()),
            };
            if !out.contains(&source) {
                out.push(source);
            }
        }
        if out.is_empty() {
            return Err("--sources must name at least one source".into());
        }
        Ok(out)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::ResidualInput => "residual_input",
            Self::PreWoHeadOutput => "pre_wo_head_output",
            Self::Symbolic => "symbolic",
        }
    }
}

#[derive(Clone)]
struct ProbePrompt {
    id: String,
    stratum: String,
    token_ids: Vec<u32>,
    oracle_codes: Vec<Vec<usize>>,
    target_classes: Vec<usize>,
    features: BTreeMap<ProbeSource, Vec<Vec<f32>>>,
    baseline_logp: Vec<f64>,
    baseline_top1: u32,
}

#[derive(Default)]
struct FlatDataset {
    features: Vec<Vec<f32>>,
    labels: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct ClassMetrics {
    accuracy: f64,
    macro_f1: f64,
    per_class_f1: BTreeMap<usize, f64>,
    confusion: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize)]
struct ReplacementMetrics {
    mean_kl: f64,
    p95_kl: f64,
    max_kl: f64,
    top1_agreement: f64,
    top5_retention: f64,
}

#[derive(Debug, Serialize)]
struct SourceReport {
    source: ProbeSource,
    classifier: &'static str,
    train_rows: usize,
    eval_rows: usize,
    input_dim: usize,
    class_metrics: ClassMetrics,
    replacement_metrics: ReplacementMetrics,
}

#[derive(Debug, Serialize)]
struct ProbeProgramClassReport {
    program_name: Option<String>,
    head: HeadId,
    group: usize,
    base_config_k: usize,
    base_config_groups: usize,
    base_config_bits_per_group: usize,
    fit_prompts: usize,
    eval_prompts: usize,
    classes: Vec<usize>,
    oracle_program_replacement: ReplacementMetrics,
    sources: Vec<SourceReport>,
}

#[derive(Clone)]
struct CentroidProbe {
    class_codes: Vec<usize>,
    mean: Vec<f64>,
    inv_std: Vec<f64>,
    centroids: Vec<Vec<f64>>,
    centroid_norms: Vec<f64>,
}

impl CentroidProbe {
    fn fit(rows: &[Vec<f32>], labels: &[usize], class_codes: &[usize]) -> Result<Self, String> {
        if rows.is_empty() {
            return Err("cannot fit probe on empty dataset".into());
        }
        let dim = rows[0].len();
        if dim == 0 {
            return Err("cannot fit probe with zero-dimensional rows".into());
        }
        if rows.iter().any(|row| row.len() != dim) {
            return Err("probe feature rows have inconsistent dimensions".into());
        }
        let mut mean = vec![0.0; dim];
        for row in rows {
            for (dst, &value) in mean.iter_mut().zip(row.iter()) {
                *dst += value as f64;
            }
        }
        let inv_n = 1.0 / rows.len() as f64;
        for value in &mut mean {
            *value *= inv_n;
        }
        let mut var = vec![0.0; dim];
        for row in rows {
            for (idx, &value) in row.iter().enumerate() {
                let d = value as f64 - mean[idx];
                var[idx] += d * d;
            }
        }
        let inv_std = var
            .into_iter()
            .map(|v| {
                let std = (v * inv_n).sqrt();
                if std > 1e-12 {
                    1.0 / std
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>();

        let class_to_idx = class_codes
            .iter()
            .enumerate()
            .map(|(idx, &code)| (code, idx))
            .collect::<HashMap<_, _>>();
        let mut centroids = vec![vec![0.0; dim]; class_codes.len()];
        let mut counts = vec![0usize; class_codes.len()];
        for (row, &label) in rows.iter().zip(labels.iter()) {
            let Some(&class_idx) = class_to_idx.get(&label) else {
                continue;
            };
            counts[class_idx] += 1;
            for idx in 0..dim {
                centroids[class_idx][idx] += ((row[idx] as f64) - mean[idx]) * inv_std[idx];
            }
        }
        for (centroid, &count) in centroids.iter_mut().zip(counts.iter()) {
            if count == 0 {
                continue;
            }
            let inv_count = 1.0 / count as f64;
            for value in centroid {
                *value *= inv_count;
            }
        }
        let centroid_norms = centroids
            .iter()
            .map(|c| c.iter().map(|v| v * v).sum::<f64>())
            .collect();

        Ok(Self {
            class_codes: class_codes.to_vec(),
            mean,
            inv_std,
            centroids,
            centroid_norms,
        })
    }

    fn predict(&self, row: &[f32]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (class_idx, centroid) in self.centroids.iter().enumerate() {
            let mut dot = 0.0;
            for idx in 0..row.len() {
                dot += ((row[idx] as f64) - self.mean[idx]) * self.inv_std[idx] * centroid[idx];
            }
            let score = dot - 0.5 * self.centroid_norms[class_idx];
            if score > best_score {
                best_score = score;
                best_idx = class_idx;
            }
        }
        self.class_codes[best_idx]
    }

    fn predict_many(&self, rows: &[Vec<f32>]) -> Vec<usize> {
        rows.iter().map(|row| self.predict(row)).collect()
    }
}

pub(super) fn run_probe_program_class(
    args: ProbeProgramClassArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(&args.out)?;

    let sources = ProbeSource::parse_list(&args.sources)?;
    let program_text = std::fs::read_to_string(&args.program)?;
    let mut program: Program = serde_json::from_str(&program_text)?;
    program.validate()?;
    program.normalize();

    let mut head = program.head;
    if let Some(spec) = args.head.as_deref() {
        let parsed = parse_head_spec(spec)?;
        head = parsed.into_iter().next().ok_or("--head was empty")?;
        if head != program.head {
            eprintln!(
                "WARNING: --head L{}H{} overrides program head L{}H{}",
                head.layer, head.head, program.head.layer, program.head.head
            );
        }
    }
    let group = args.group.unwrap_or(program.group);
    if group != program.group {
        eprintln!(
            "WARNING: --group {group} overrides program group {}",
            program.group
        );
    }

    let config = PqConfig::from(&program.base_config);
    eprintln!(
        "Probe program class: L{}H{} group {} {}:{}:{} sources={}",
        head.layer,
        head.head,
        group,
        config.k,
        config.groups,
        config.bits_per_group,
        sources
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    index.load_attn_kquant(&args.index)?;
    index.load_interleaved_kquant(&args.index)?;
    let mut weights = load_model_weights_kquant(&args.index, &mut cb)?;
    if weights.arch.is_hybrid_moe() {
        return Err("ov-rd probe-program-class currently supports dense FFN vindexes only".into());
    }
    let tokenizer = load_vindex_tokenizer(&args.index)?;

    let mut all_records = load_prompts(&args.prompts, None)?;
    if args.max_per_stratum > 0 {
        all_records = limit_prompts_per_stratum(all_records, args.max_per_stratum);
    }
    let strata = strata_vocab(&all_records);
    let (fit_prompts, eval_prompts) =
        split_prompt_records(&all_records, args.eval_mod, args.eval_offset)?;
    eprintln!(
        "Prompts: {} fit, {} eval",
        fit_prompts.len(),
        eval_prompts.len()
    );

    let selected_heads = vec![head];
    let configs = vec![config];

    eprintln!("Fitting position-mean static bases");
    let means = fit_static_means(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
    )?;

    eprintln!("Building W_O-visible bases");
    let bases =
        build_roundtrip_bases(&mut weights, &index, &selected_heads, args.sigma_rel_cutoff)?;

    eprintln!("Fitting empirical z-space PCA bases");
    let pca_bases = fit_z_pca_bases(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
    )?;

    eprintln!("Fitting product quantizers");
    let codebooks = fit_pq_codebooks(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &configs,
        args.pq_iters,
        &[],
    )?;

    eprintln!("Materializing Mode D residual-space table");
    let mode_d_tables = materialize_mode_d_tables(
        &mut weights,
        &index,
        &selected_heads,
        &bases,
        &means,
        &pca_bases,
        &codebooks,
        &[],
    )?;
    let mode_d_table = mode_d_tables
        .get(&(head, config))
        .ok_or_else(|| format!("Mode D table missing for L{}H{}", head.layer, head.head))?;
    let basis = bases.get(&head).ok_or("W_O basis missing")?;
    let pca_basis = pca_bases.get(&head).ok_or("PCA basis missing")?;
    let head_means = means.get(&head).ok_or("Position means missing")?;
    let codebook = codebooks.get(&(head, config)).ok_or("Codebook missing")?;

    eprintln!("Capturing fit probe rows");
    let fit_captures = capture_probe_prompts(
        &mut weights,
        &index,
        &tokenizer,
        &fit_prompts,
        head,
        group,
        &program,
        basis,
        pca_basis,
        head_means,
        codebook,
        &strata,
    )?;
    eprintln!("Capturing eval probe rows");
    let eval_captures = capture_probe_prompts(
        &mut weights,
        &index,
        &tokenizer,
        &eval_prompts,
        head,
        group,
        &program,
        basis,
        pca_basis,
        head_means,
        codebook,
        &strata,
    )?;

    let classes = class_vocab(&program, &fit_captures, &eval_captures);
    validate_quotient_classes(&program, &classes)?;
    eprintln!("Behavioral classes: {:?}", classes);

    let oracle_program_replacement = evaluate_replacement(
        &mut weights,
        &index,
        head,
        group,
        mode_d_table,
        &eval_captures,
        |prompt, _| prompt.target_classes.clone(),
    )?;

    let mut source_reports = Vec::new();
    for &source in &sources {
        let train = flatten_source(&fit_captures, source)?;
        let eval = flatten_source(&eval_captures, source)?;
        let probe = CentroidProbe::fit(&train.features, &train.labels, &classes)?;
        let predictions = probe.predict_many(&eval.features);
        let class_metrics = class_metrics(&eval.labels, &predictions, &classes);
        let replacement_metrics = evaluate_replacement(
            &mut weights,
            &index,
            head,
            group,
            mode_d_table,
            &eval_captures,
            |_prompt, features| {
                features
                    .get(&source)
                    .expect("source features already validated")
                    .iter()
                    .map(|row| probe.predict(row))
                    .collect::<Vec<_>>()
            },
        )?;

        let input_dim = train.features.first().map(|row| row.len()).unwrap_or(0);
        eprintln!(
            "{}: acc {:.4} macro-F1 {:.4} replacement KL mean {:.6} p95 {:.6}",
            source.as_str(),
            class_metrics.accuracy,
            class_metrics.macro_f1,
            replacement_metrics.mean_kl,
            replacement_metrics.p95_kl
        );
        source_reports.push(SourceReport {
            source,
            classifier: "standardized_nearest_centroid_linear",
            train_rows: train.features.len(),
            eval_rows: eval.features.len(),
            input_dim,
            class_metrics,
            replacement_metrics,
        });
    }

    let report = ProbeProgramClassReport {
        program_name: program.name.clone(),
        head,
        group,
        base_config_k: config.k,
        base_config_groups: config.groups,
        base_config_bits_per_group: config.bits_per_group,
        fit_prompts: fit_captures.len(),
        eval_prompts: eval_captures.len(),
        classes,
        oracle_program_replacement,
        sources: source_reports,
    };

    let out_path = args.out.join("probe_program_class.json");
    serde_json::to_writer_pretty(std::fs::File::create(&out_path)?, &report)?;
    eprintln!("Wrote {}", out_path.display());
    Ok(())
}
