//! Exception-catalog fitting and the per-head measurements it needs.

use super::super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::super::metrics::{argmax, kl_logp, log_softmax, token_prob};
use super::super::oracle_pq_forward::{final_logits, forward_q4k_oracle_pq_mode_d_head};
use super::super::pq::{kmeans_centroids, nearest_centroid_index, ModeDTable, PqCodebook};
use super::super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::super::stats::StaticHeadMeans;
use super::super::types::{HeadId, PqConfig, PromptRecord};
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

pub(super) fn fit_exception_catalogs(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    tables: &HashMap<(HeadId, PqConfig), ModeDTable>,
    w_o_heads: &HashMap<HeadId, Vec<Vec<f32>>>,
    base_config: PqConfig,
    exception_edits: &[usize],
    tail_fracs: &[f64],
    tail_selector: TailSelector,
    exception_fit: ExceptionFit,
    prompt_scores: &HashMap<(HeadId, usize), f64>,
    position_scores: &HashMap<(HeadId, usize, usize), f64>,
    iterations: usize,
) -> Result<HashMap<ExceptionKey, ExceptionCatalog>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut samples: HashMap<HeadId, Vec<ErrorSample>> = HashMap::new();
    for head in heads {
        samples.insert(*head, Vec::new());
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  exception-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let mut h = embed_tokens_pub(weights, &token_ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &token_ids);

        for layer in 0..weights.num_layers {
            let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
            if let Some(layer_heads) = heads_by_layer.get(&layer) {
                let (_, pre_o) = run_attention_block_with_pre_o(
                    larql_models::WeightsView::dense(weights),
                    &h,
                    layer,
                )
                .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                for head in layer_heads {
                    let basis = bases.get(head).expect("basis pre-created");
                    let pca_basis = pca_bases.get(head).expect("PCA pre-created");
                    let head_means = means.get(head).expect("means pre-created");
                    let codebook = codebooks
                        .get(&(*head, base_config))
                        .expect("base codebook pre-created");
                    let table = tables
                        .get(&(*head, base_config))
                        .expect("base Mode D table pre-created");
                    let w_o_head = w_o_heads.get(head).expect("W_O head pre-copied");
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during exception fit")?;
                        let base_delta = base_pq_delta(
                            values, basis, pca_basis, head_means, codebook, table, pos, stratum,
                        );
                        let true_delta = project_head_vector_to_hidden(w_o_head, values);
                        let error = true_delta
                            .iter()
                            .zip(base_delta.iter())
                            .map(|(&true_value, &base_value)| true_value as f64 - base_value as f64)
                            .collect::<Vec<_>>();
                        let sq_norm = error.iter().map(|value| value * value).sum::<f64>();
                        let score = match tail_selector {
                            TailSelector::ResidualError => sq_norm,
                            TailSelector::PromptKl => {
                                *prompt_scores.get(&(*head, prompt_idx)).unwrap_or(&0.0)
                            }
                            TailSelector::PositionRestoreKl => *position_scores
                                .get(&(*head, prompt_idx, pos))
                                .unwrap_or(&0.0),
                            TailSelector::PositionRestoreCe => *position_scores
                                .get(&(*head, prompt_idx, pos))
                                .unwrap_or(&0.0),
                        };
                        samples
                            .get_mut(head)
                            .expect("exception samples missing")
                            .push(ErrorSample {
                                score,
                                sq_norm,
                                values: error,
                            });
                    }
                }
            }
            {
                let ffn = WeightFfn { weights };
                if let Some((h_new, _, _)) = run_layer_with_ffn(
                    larql_inference::WeightsView::dense(weights),
                    &h,
                    layer,
                    &ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    h = h_new;
                }
            }
            remove_layer_tensors(weights, inserted);
        }
    }

    let mut catalogs = HashMap::new();
    for head in heads {
        let mut head_samples = samples.remove(head).ok_or_else(|| {
            format!(
                "missing exception samples for L{}H{}",
                head.layer, head.head
            )
        })?;
        head_samples.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.sq_norm
                        .partial_cmp(&a.sq_norm)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let total = head_samples.len();
        for &tail_frac in tail_fracs {
            let used = ((total as f64) * tail_frac).ceil() as usize;
            let used = used.clamp(1, total.max(1));
            let selected = head_samples
                .iter()
                .take(used)
                .map(|sample| sample.values.clone())
                .collect::<Vec<_>>();
            for &edits in exception_edits {
                let centroids = match exception_fit {
                    ExceptionFit::Kmeans => kmeans_centroids(&selected, edits, iterations),
                    ExceptionFit::Exemplar => exemplar_centroids(&selected, edits),
                };
                catalogs.insert(
                    ExceptionKey {
                        head: *head,
                        edits,
                        tail_frac_key: tail_frac_key(tail_frac),
                    },
                    ExceptionCatalog {
                        edits,
                        tail_frac,
                        train_error_samples: total,
                        train_error_samples_used: used,
                        centroids,
                    },
                );
            }
        }
    }

    Ok(catalogs)
}

pub(super) fn exemplar_centroids(selected: &[Vec<f64>], edits: usize) -> Vec<Vec<f64>> {
    if edits == 0 {
        return Vec::new();
    }
    if selected.is_empty() {
        return vec![Vec::new(); edits];
    }
    (0..edits)
        .map(|idx| selected[idx.min(selected.len() - 1)].clone())
        .collect()
}

pub(super) fn measure_fit_prompt_base_pq_kl(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    tables: &HashMap<(HeadId, PqConfig), ModeDTable>,
    base_config: PqConfig,
) -> Result<HashMap<(HeadId, usize), f64>, Box<dyn std::error::Error>> {
    let mut scores = HashMap::new();
    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  selector-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let baseline_hidden =
            larql_inference::vindex::predict_kquant_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);
        for head in heads {
            let basis = bases
                .get(head)
                .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
            let head_means = means
                .get(head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let codebook = codebooks.get(&(*head, base_config)).ok_or_else(|| {
                format!("missing base codebook for L{}H{}", head.layer, head.head)
            })?;
            let table = tables
                .get(&(*head, base_config))
                .ok_or_else(|| format!("missing base table for L{}H{}", head.layer, head.head))?;
            let pq_hidden = forward_q4k_oracle_pq_mode_d_head(
                weights, &token_ids, index, *head, basis, pca_basis, head_means, codebook, table,
                stratum,
            )?;
            let pq_logits = final_logits(weights, &pq_hidden);
            let pq_logp = log_softmax(&pq_logits);
            scores.insert((*head, prompt_idx), kl_logp(&baseline_logp, &pq_logp));
        }
    }
    Ok(scores)
}

pub(super) fn measure_fit_position_restore_gains(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    tables: &HashMap<(HeadId, PqConfig), ModeDTable>,
    w_o_heads: &HashMap<HeadId, Vec<Vec<f32>>>,
    base_config: PqConfig,
    tail_selector: TailSelector,
    candidates_per_prompt: usize,
) -> Result<HashMap<(HeadId, usize, usize), f64>, Box<dyn std::error::Error>> {
    let mut scores = HashMap::new();
    if candidates_per_prompt == 0 {
        return Ok(scores);
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  position-restore-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
        let stratum = record.stratum.as_deref().unwrap_or("unknown");
        let baseline_hidden =
            larql_inference::vindex::predict_kquant_hidden(weights, &token_ids, index, None);
        let baseline_logits = final_logits(weights, &baseline_hidden);
        let baseline_logp = log_softmax(&baseline_logits);
        let baseline_top1 = argmax(&baseline_logits);

        for head in heads {
            let basis = bases
                .get(head)
                .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
            let pca_basis = pca_bases
                .get(head)
                .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
            let head_means = means
                .get(head)
                .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
            let codebook = codebooks.get(&(*head, base_config)).ok_or_else(|| {
                format!("missing base codebook for L{}H{}", head.layer, head.head)
            })?;
            let table = tables
                .get(&(*head, base_config))
                .ok_or_else(|| format!("missing base table for L{}H{}", head.layer, head.head))?;
            let w_o_head = w_o_heads
                .get(head)
                .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;

            let base_hidden = forward_q4k_oracle_pq_mode_d_head(
                weights, &token_ids, index, *head, basis, pca_basis, head_means, codebook, table,
                stratum,
            )?;
            let base_logits = final_logits(weights, &base_hidden);
            let base_logp = log_softmax(&base_logits);
            let base_kl = kl_logp(&baseline_logp, &base_logp);
            let base_ce = -token_prob(&base_logp, baseline_top1).ln();

            let mut candidates = capture_head_position_sq_errors(
                weights, index, &token_ids, *head, basis, pca_basis, head_means, codebook, table,
                w_o_head, stratum,
            )?;
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(candidates_per_prompt.min(candidates.len()));

            for (position, _sq_norm) in candidates {
                let restored_hidden = forward_q4k_oracle_pq_position_restore_head(
                    weights, &token_ids, index, *head, basis, pca_basis, head_means, codebook,
                    table, w_o_head, position, stratum,
                )?;
                let restored_logits = final_logits(weights, &restored_hidden);
                let restored_logp = log_softmax(&restored_logits);
                let gain = match tail_selector {
                    TailSelector::PositionRestoreKl => {
                        let restored_kl = kl_logp(&baseline_logp, &restored_logp);
                        base_kl - restored_kl
                    }
                    TailSelector::PositionRestoreCe => {
                        let restored_ce = -token_prob(&restored_logp, baseline_top1).ln();
                        base_ce - restored_ce
                    }
                    TailSelector::ResidualError | TailSelector::PromptKl => 0.0,
                }
                .max(0.0);
                scores.insert((*head, prompt_idx, position), gain);
            }
        }
    }

    Ok(scores)
}

pub(super) fn capture_head_position_sq_errors(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    w_o_head: &[Vec<f32>],
    stratum: &str,
) -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..weights.num_layers {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        if layer == head.layer {
            let result = (|| -> Result<Vec<(usize, f64)>, Box<dyn std::error::Error>> {
                let (_, pre_o) = run_attention_block_with_pre_o(
                    larql_models::WeightsView::dense(weights),
                    &h,
                    layer,
                )
                .ok_or_else(|| format!("pre-W_O capture failed at layer {layer}"))?;
                let head_dim = weights.arch.head_dim_for_layer(layer);
                let start = head.head * head_dim;
                let end = start + head_dim;
                let mut errors = Vec::with_capacity(pre_o.nrows());
                for pos in 0..pre_o.nrows() {
                    let row = pre_o.slice(s![pos, start..end]);
                    let values = row
                        .as_slice()
                        .ok_or("pre-W_O head row was not contiguous during restore fit")?;
                    let base_delta = base_pq_delta(
                        values, basis, pca_basis, means, codebook, table, pos, stratum,
                    );
                    let true_delta = project_head_vector_to_hidden(w_o_head, values);
                    let sq_norm = true_delta
                        .iter()
                        .zip(base_delta.iter())
                        .map(|(&true_value, &base_value)| {
                            let delta = true_value as f64 - base_value as f64;
                            delta * delta
                        })
                        .sum::<f64>();
                    errors.push((pos, sq_norm));
                }
                Ok(errors)
            })();
            remove_layer_tensors(weights, inserted);
            return result;
        }
        {
            let ffn = WeightFfn { weights };
            if let Some((h_new, _, _)) = run_layer_with_ffn(
                larql_inference::WeightsView::dense(weights),
                &h,
                layer,
                &ffn,
                false,
                ple_inputs.get(layer),
                None,
            ) {
                h = h_new;
            }
        }
        remove_layer_tensors(weights, inserted);
    }

    Err(format!("target layer {} was not reached", head.layer).into())
}

pub(super) fn forward_q4k_oracle_pq_exception_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    w_o_head: &[Vec<f32>],
    catalog: &ExceptionCatalog,
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hidden_size = weights.hidden_size;
    larql_inference::vindex::predict_kquant_hidden_with_mapped_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement_delta = Vec::with_capacity(original_head.nrows() * hidden_size);
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during exception eval")?;
                let base_delta = base_pq_delta(
                    values, basis, pca_basis, means, codebook, table, pos, stratum,
                );
                let true_delta = project_head_vector_to_hidden(w_o_head, values);
                let error = true_delta
                    .iter()
                    .zip(base_delta.iter())
                    .map(|(&true_value, &base_value)| true_value as f64 - base_value as f64)
                    .collect::<Vec<_>>();
                let code = nearest_centroid_index(&error, &catalog.centroids);
                let exception = &catalog.centroids[code];
                for (&base, &extra) in base_delta.iter().zip(exception.iter()) {
                    replacement_delta.push(base + extra as f32);
                }
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

pub(super) fn forward_q4k_oracle_pq_position_restore_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    w_o_head: &[Vec<f32>],
    restore_position: usize,
    stratum: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let hidden_size = weights.hidden_size;
    larql_inference::vindex::predict_kquant_hidden_with_mapped_head_residual_delta(
        weights,
        token_ids,
        index,
        head.layer,
        head.head,
        |original_head| {
            let mut replacement_delta = Vec::with_capacity(original_head.nrows() * hidden_size);
            for pos in 0..original_head.nrows() {
                let row = original_head.row(pos);
                let values = row
                    .as_slice()
                    .ok_or("pre-W_O head row was not contiguous during position restore")?;
                if pos == restore_position {
                    let true_delta = project_head_vector_to_hidden(w_o_head, values);
                    replacement_delta.extend_from_slice(&true_delta);
                } else {
                    let base_delta = base_pq_delta(
                        values, basis, pca_basis, means, codebook, table, pos, stratum,
                    );
                    replacement_delta.extend_from_slice(&base_delta);
                }
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

pub(super) fn base_pq_delta(
    values: &[f32],
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    codebook: &PqCodebook,
    table: &ModeDTable,
    position: usize,
    stratum: &str,
) -> Vec<f32> {
    let base = means.positions.get(position).unwrap_or(&means.global);
    let residual = values
        .iter()
        .zip(base.iter())
        .map(|(&value, &mean)| value - mean)
        .collect::<Vec<_>>();
    let z = basis.residual_to_z(&residual);
    let coords = pca_basis.coordinates_with_rank(&z, codebook.config.k);
    let codes = codebook.quantize_indices_for_stratum(&coords, stratum);
    table.delta_for_position_codes_with_stratum(position, &codes, stratum)
}

pub(super) fn copy_w_o_heads(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
) -> Result<HashMap<HeadId, Vec<Vec<f32>>>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let mut out = HashMap::new();
    for (layer, layer_heads) in heads_by_layer {
        let inserted = insert_q4k_layer_tensors(weights, index, layer)?;
        let w_o = weights
            .tensors
            .get(&weights.arch.attn_o_key(layer))
            .ok_or_else(|| format!("missing W_O tensor at layer {layer}"))?;
        let head_dim = weights.arch.head_dim_for_layer(layer);
        for head in layer_heads {
            let start = head.head * head_dim;
            let end = start + head_dim;
            let w_o_head = w_o.slice(s![.., start..end]);
            let rows = (0..w_o_head.nrows())
                .map(|row| {
                    (0..w_o_head.ncols())
                        .map(|col| w_o_head[[row, col]])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            out.insert(head, rows);
        }
        remove_layer_tensors(weights, inserted);
    }
    Ok(out)
}
