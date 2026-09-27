//! Edit-catalog fitting, per-head forwards and static hidden tables.

use super::super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::super::pq::{kmeans_centroids, nearest_centroid_index};
use super::super::runtime::{insert_q4k_layer_tensors, remove_layer_tensors};
use super::super::stats::StaticHeadMeans;
use super::super::types::{HeadId, PromptRecord};
use larql_inference::attention::run_attention_block_with_pre_o;
use larql_inference::forward::ple::precompute_per_layer_inputs;
use larql_inference::forward::{embed_tokens_pub, run_layer_with_ffn};
use larql_inference::{encode_prompt, WeightFfn};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

pub(super) fn fit_edit_catalogs(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    spaces: &[EditCatalogSpace],
    edit_counts: &[usize],
    pca_rank: usize,
    iterations: usize,
) -> Result<HashMap<EditCatalogKey, EditCatalog>, Box<dyn std::error::Error>> {
    let mut heads_by_layer: HashMap<usize, Vec<HeadId>> = HashMap::new();
    for head in heads {
        heads_by_layer.entry(head.layer).or_default().push(*head);
    }
    let w_o_heads = copy_w_o_heads(weights, index, heads)?;

    let mut samples: HashMap<(HeadId, EditCatalogSpace), Vec<Vec<f64>>> = HashMap::new();
    for head in heads {
        for &space in spaces {
            samples.insert((*head, space), Vec::new());
        }
    }

    for (prompt_idx, record) in prompts.iter().enumerate() {
        let label = prompt_label(record);
        eprintln!(
            "  catalog-fit [{}/{}] {}",
            prompt_idx + 1,
            prompts.len(),
            label
        );
        let token_ids = encode_prompt(tokenizer, &*weights.arch, &record.prompt)?;
        if token_ids.is_empty() {
            continue;
        }
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
                    let basis = bases.get(head).expect("basis pre-created for edit catalog");
                    let head_means = means.get(head).expect("means pre-created for edit catalog");
                    let pca_basis = pca_bases
                        .get(head)
                        .expect("PCA pre-created for edit catalog");
                    if pca_basis.rank() < pca_rank && spaces.contains(&EditCatalogSpace::Pca) {
                        return Err(format!(
                            "PCA rank {} is below requested rank {} for L{}H{}",
                            pca_basis.rank(),
                            pca_rank,
                            head.layer,
                            head.head
                        )
                        .into());
                    }
                    let w_o_head = w_o_heads
                        .get(head)
                        .expect("W_O head pre-copied for edit catalog");
                    let start = head.head * head_dim;
                    let end = start + head_dim;
                    for pos in 0..pre_o.nrows() {
                        let row = pre_o.slice(s![pos, start..end]);
                        let values = row
                            .as_slice()
                            .ok_or("pre-W_O head row was not contiguous during edit catalog fit")?;
                        let residual = head_residual(values, head_means, pos);
                        for &space in spaces {
                            let sample = match space {
                                EditCatalogSpace::Hidden => {
                                    project_head_vector_to_hidden(w_o_head, &residual)
                                        .into_iter()
                                        .map(|value| value as f64)
                                        .collect::<Vec<_>>()
                                }
                                EditCatalogSpace::Pca => {
                                    let z = basis.residual_to_z(&residual);
                                    pca_basis.coordinates_with_rank(&z, pca_rank)
                                }
                            };
                            samples
                                .get_mut(&(*head, space))
                                .expect("edit samples missing")
                                .push(sample);
                        }
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
        let basis = bases
            .get(head)
            .ok_or_else(|| format!("missing basis for L{}H{}", head.layer, head.head))?;
        let pca_basis = pca_bases
            .get(head)
            .ok_or_else(|| format!("missing PCA basis for L{}H{}", head.layer, head.head))?;
        let w_o_head = w_o_heads
            .get(head)
            .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;
        for &space in spaces {
            let head_samples = samples
                .get(&(*head, space))
                .ok_or_else(|| format!("missing edit samples for L{}H{}", head.layer, head.head))?;
            for &edits in edit_counts {
                let feature_centroids = kmeans_centroids(head_samples, edits, iterations);
                let residual_table = match space {
                    EditCatalogSpace::Hidden => feature_centroids
                        .iter()
                        .map(|centroid| centroid.iter().map(|&value| value as f32).collect())
                        .collect(),
                    EditCatalogSpace::Pca => feature_centroids
                        .iter()
                        .map(|centroid| {
                            let z = pca_basis.reconstruct_from_coordinates(centroid);
                            let residual = basis.z_to_residual(&z);
                            project_head_vector_to_hidden(w_o_head, &residual)
                        })
                        .collect(),
                };
                catalogs.insert(
                    EditCatalogKey {
                        head: *head,
                        space,
                        edits,
                    },
                    EditCatalog {
                        space,
                        feature_centroids,
                        residual_table,
                    },
                );
            }
        }
    }

    Ok(catalogs)
}

pub(super) fn forward_q4k_oracle_edit_catalog_head(
    weights: &mut larql_inference::ModelWeights,
    token_ids: &[u32],
    index: &VectorIndex,
    head: HeadId,
    basis: &WoRoundtripBasis,
    pca_basis: &ZPcaBasis,
    means: &StaticHeadMeans,
    static_hidden: &StaticHiddenTable,
    w_o_head: &[Vec<f32>],
    catalog: &EditCatalog,
    pca_rank: usize,
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
                    .ok_or("pre-W_O head row was not contiguous during edit catalog eval")?;
                let residual = head_residual(values, means, pos);
                let feature = match catalog.space {
                    EditCatalogSpace::Hidden => project_head_vector_to_hidden(w_o_head, &residual)
                        .into_iter()
                        .map(|value| value as f64)
                        .collect::<Vec<_>>(),
                    EditCatalogSpace::Pca => {
                        let z = basis.residual_to_z(&residual);
                        pca_basis.coordinates_with_rank(&z, pca_rank)
                    }
                };
                let code = nearest_centroid_index(&feature, &catalog.feature_centroids);
                let static_delta = static_hidden.delta_for_position(pos);
                let edit_delta = &catalog.residual_table[code];
                for (&base, &edit) in static_delta.iter().zip(edit_delta.iter()) {
                    replacement_delta.push(base + edit);
                }
            }
            Array2::from_shape_vec((original_head.nrows(), hidden_size), replacement_delta)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(Into::into)
}

#[derive(Debug, Clone)]
pub(super) struct StaticHiddenTable {
    pub(super) by_position: Vec<Vec<f32>>,
    pub(super) global: Vec<f32>,
}

impl StaticHiddenTable {
    pub(super) fn delta_for_position(&self, position: usize) -> &[f32] {
        self.by_position
            .get(position)
            .map(|delta| delta.as_slice())
            .unwrap_or(&self.global)
    }
}

pub(super) fn build_static_hidden_tables(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    heads: &[HeadId],
    means: &HashMap<HeadId, StaticHeadMeans>,
) -> Result<HashMap<HeadId, StaticHiddenTable>, Box<dyn std::error::Error>> {
    let w_o_heads = copy_w_o_heads(weights, index, heads)?;
    let mut tables = HashMap::new();
    for head in heads {
        let w_o_head = w_o_heads
            .get(head)
            .ok_or_else(|| format!("missing W_O head for L{}H{}", head.layer, head.head))?;
        let head_means = means
            .get(head)
            .ok_or_else(|| format!("missing means for L{}H{}", head.layer, head.head))?;
        let global = project_head_vector_to_hidden(w_o_head, &head_means.global);
        let by_position = head_means
            .positions
            .iter()
            .map(|mean| project_head_vector_to_hidden(w_o_head, mean))
            .collect();
        tables.insert(
            *head,
            StaticHiddenTable {
                by_position,
                global,
            },
        );
    }
    Ok(tables)
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

pub(super) fn head_residual(values: &[f32], means: &StaticHeadMeans, position: usize) -> Vec<f32> {
    let base = means.positions.get(position).unwrap_or(&means.global);
    values
        .iter()
        .zip(base.iter())
        .map(|(&value, &mean)| value - mean)
        .collect()
}

pub(super) fn project_head_vector_to_hidden(w_o_head: &[Vec<f32>], values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w_o_head.len()];
    for (row_idx, row) in w_o_head.iter().enumerate() {
        let mut sum = 0.0f32;
        for (&value, &weight) in values.iter().zip(row.iter()) {
            sum += value * weight;
        }
        out[row_idx] = sum;
    }
    out
}
