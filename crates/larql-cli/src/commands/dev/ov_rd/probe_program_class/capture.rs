//! Probe prompt and target-feature capture.

use super::super::address::attention_argmax;
use super::super::metrics::{argmax, log_softmax};
use super::super::oracle_pq_forward::{final_logits, forward_q4k_oracle_pq_head};
use super::super::program::{PositionContext, Program};
use super::super::types::{HeadId, PromptRecord};
use larql_inference::attention::{
    run_attention_block_with_pre_o_and_all_attention_weights, SharedKV,
};
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[allow(unused_imports)]
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn capture_probe_prompts(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    records: &[PromptRecord],
    head: HeadId,
    group: usize,
    program: &Program,
    basis: &super::super::basis::WoRoundtripBasis,
    pca_basis: &super::super::basis::ZPcaBasis,
    head_means: &super::super::stats::StaticHeadMeans,
    codebook: &super::super::pq::PqCodebook,
    strata: &[String],
) -> Result<Vec<ProbePrompt>, Box<dyn std::error::Error>> {
    let mut captures = Vec::with_capacity(records.len());
    for (idx, record) in records.iter().enumerate() {
        let label = record
            .id
            .as_deref()
            .or(record.stratum.as_deref())
            .unwrap_or("prompt");
        eprintln!("  [{}/{}] {}", idx + 1, records.len(), label);

        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");

        let baseline_h =
            larql_inference::vindex::predict_kquant_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_h);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);

        let (_, _, oracle_codes) = forward_q4k_oracle_pq_head(
            weights, &token_ids, index, head, basis, pca_basis, head_means, codebook, stratum,
        )?;
        let target_features = capture_target_features(weights, &token_ids, index, head)?;

        let mut target_classes = Vec::with_capacity(token_ids.len());
        let mut symbolic_rows = Vec::with_capacity(token_ids.len());
        for pos in 0..token_ids.len() {
            let original = oracle_codes
                .get(pos)
                .and_then(|codes| codes.get(group))
                .copied()
                .ok_or("oracle code capture missing target group")?;
            let attn_row = target_features
                .attention_rows
                .get(pos)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let attn_argmax = attention_argmax(attn_row, pos);
            let ctx = PositionContext {
                stratum: stratum.to_string(),
                position: pos,
                token_id: token_ids[pos],
                prev_token_id: (pos > 0).then(|| token_ids.get(pos - 1).copied()).flatten(),
                attends_bos: attn_argmax == 0,
                attends_prev: pos > 0 && attn_argmax + 1 == pos,
                original_code: original,
                current_code: original,
            };
            target_classes.push(program.apply_to_code(original, &ctx));
            symbolic_rows.push(symbolic_features(
                tokenizer, strata, stratum, &token_ids, pos, &ctx,
            ));
        }

        let mut features = BTreeMap::new();
        features.insert(
            ProbeSource::ResidualInput,
            array_rows(&target_features.residual_input),
        );
        features.insert(
            ProbeSource::PreWoHeadOutput,
            array_rows(&target_features.pre_wo_head_output),
        );
        features.insert(ProbeSource::Symbolic, symbolic_rows);

        captures.push(ProbePrompt {
            id: label.to_string(),
            stratum: stratum.to_string(),
            token_ids,
            oracle_codes,
            target_classes,
            features,
            baseline_logp,
            baseline_top1,
        });
    }
    Ok(captures)
}

pub(super) struct TargetFeatures {
    pub(super) residual_input: Array2<f32>,
    pub(super) pre_wo_head_output: Array2<f32>,
    pub(super) attention_rows: Vec<Vec<f32>>,
}

pub(super) fn capture_target_features(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
) -> Result<TargetFeatures, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();

    for layer in 0..=head.layer {
        let inserted = super::super::runtime::insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == head.layer {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let (_, pre_o, all_weights) = run_attention_block_with_pre_o_and_all_attention_weights(
                larql_models::WeightsView::dense(weights),
                &h,
                layer,
                shared_kv,
            )
            .ok_or_else(|| {
                format!(
                    "probe feature capture failed at L{}H{}",
                    head.layer, head.head
                )
            })?;
            super::super::runtime::remove_layer_tensors(weights, inserted);

            let head_dim = weights.head_dim;
            let start = head.head * head_dim;
            let end = start + head_dim;
            if end > pre_o.ncols() {
                return Err(format!(
                    "head {} out of range for pre-W_O width {}",
                    head.head,
                    pre_o.ncols()
                )
                .into());
            }
            let pre_wo_head_output = pre_o.slice(s![.., start..end]).to_owned();
            let attention_rows = all_weights.heads.get(head.head).cloned().ok_or_else(|| {
                format!(
                    "attention weights missing for L{}H{}",
                    head.layer, head.head
                )
            })?;
            return Ok(TargetFeatures {
                residual_input: h,
                pre_wo_head_output,
                attention_rows,
            });
        }

        let step = {
            let shared_kv = weights
                .arch
                .kv_shared_source_layer(layer)
                .and_then(|src| kv_cache.get(&src));
            let ffn = WeightFfn { weights };
            run_layer_with_ffn(
                larql_inference::WeightsView::dense(weights),
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                shared_kv,
            )
            .map(|(h_new, _, kv_out)| (h_new, kv_out))
        };
        if let Some((h_new, kv_out)) = step {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        } else {
            super::super::runtime::remove_layer_tensors(weights, inserted);
            return Err(format!("layer {layer} returned no output").into());
        }
        super::super::runtime::remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {} was not reached", head.layer).into())
}

pub(super) fn array_rows(array: &Array2<f32>) -> Vec<Vec<f32>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

pub(super) fn symbolic_features(
    tokenizer: &tokenizers::Tokenizer,
    strata: &[String],
    stratum: &str,
    token_ids: &[u32],
    pos: usize,
    ctx: &PositionContext,
) -> Vec<f32> {
    let mut out = Vec::new();
    for known in strata {
        out.push((known == stratum) as u8 as f32);
    }
    let bucket = position_bucket(pos);
    for idx in 0..8 {
        out.push((idx == bucket) as u8 as f32);
    }
    out.push((pos == 0) as u8 as f32);
    out.push((pos + 1 == token_ids.len()) as u8 as f32);
    out.push(ctx.attends_bos as u8 as f32);
    out.push(ctx.attends_prev as u8 as f32);

    let token_text = tokenizer
        .decode(&[token_ids[pos]], false)
        .unwrap_or_default();
    out.push(token_text.chars().any(|c| c.is_ascii_digit()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_ascii_alphabetic()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_ascii_punctuation()) as u8 as f32);
    out.push(token_text.chars().any(|c| c.is_whitespace()) as u8 as f32);
    out.push((token_text.starts_with(' ') || token_text.starts_with('▁')) as u8 as f32);
    out.push((token_text.len() <= 1) as u8 as f32);
    out.push((token_ids[pos] as f32).ln_1p() / 16.0);
    if pos > 0 {
        out.push((token_ids[pos - 1] as f32).ln_1p() / 16.0);
    } else {
        out.push(0.0);
    }
    out
}

pub(super) fn position_bucket(pos: usize) -> usize {
    match pos {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4..=7 => 4,
        8..=15 => 5,
        16..=31 => 6,
        _ => 7,
    }
}

pub(super) fn flatten_source(
    prompts: &[ProbePrompt],
    source: ProbeSource,
) -> Result<FlatDataset, Box<dyn std::error::Error>> {
    let mut out = FlatDataset::default();
    for prompt in prompts {
        let rows = prompt
            .features
            .get(&source)
            .ok_or_else(|| format!("missing source {}", source.as_str()))?;
        if rows.len() != prompt.target_classes.len() {
            return Err(format!(
                "source {} row count mismatch for {}",
                source.as_str(),
                prompt.id
            )
            .into());
        }
        out.features.extend(rows.iter().cloned());
        out.labels.extend(prompt.target_classes.iter().copied());
    }
    Ok(out)
}

pub(super) fn strata_vocab(records: &[PromptRecord]) -> Vec<String> {
    let mut set = BTreeSet::new();
    for record in records {
        set.insert(
            record
                .stratum
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        );
    }
    set.into_iter().collect()
}

pub(super) fn class_vocab(
    program: &Program,
    fit: &[ProbePrompt],
    eval: &[ProbePrompt],
) -> Vec<usize> {
    let mut set = BTreeSet::new();
    for tc in &program.terminal_classes {
        set.insert(tc.representative_code);
    }
    for prompt in fit.iter().chain(eval.iter()) {
        for &class_code in &prompt.target_classes {
            set.insert(class_code);
        }
    }
    set.into_iter().collect()
}
