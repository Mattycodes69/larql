//! Address group models keyed on attention, reduced-QK and LSH clusters.

use super::super::address::{
    attention_cluster_key, attention_cluster_probe_names, attention_pattern_features, lsh_bucket,
    nearest_attention_cluster, AddressAttentionClusterGroupModel, AddressLshGroupModel,
};
use super::super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::super::metrics::argmax_usize;
use super::super::pq::{kmeans_centroids, PqCodebook};
use super::super::stats::StaticHeadMeans;
use super::super::types::{HeadId, PqConfig, PromptRecord};
use larql_vindex::VectorIndex;
use ndarray::ArrayView1;
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

pub(in super::super) fn fit_address_attention_cluster_group_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    cluster_counts: &[usize],
) -> Result<
    HashMap<(HeadId, PqConfig), Vec<AddressAttentionClusterGroupModel>>,
    Box<dyn std::error::Error>,
> {
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut samples: HashMap<(HeadId, PqConfig), Vec<AttentionClusterFitSample>> = HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "attention-cluster-fit",
        false,
        0,
        0,
        true,
        None,
        |head, config, pos, codes, token_ids, stratum, _, _, _, _, attention_weights| {
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            let attention_weights =
                attention_weights.ok_or("missing attention row during cluster address fit")?;
            samples
                .entry((head, config))
                .or_default()
                .push(AttentionClusterFitSample {
                    features: attention_pattern_features(attention_weights, pos),
                    codes: codes.to_vec(),
                    token_ids: token_ids.to_vec(),
                    stratum: stratum.to_string(),
                    position: pos,
                });
            Ok(())
        },
    )?;

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let train_samples = samples.get(&(*head, *config)).cloned().unwrap_or_default();
        let feature_rows = train_samples
            .iter()
            .map(|sample| sample.features.clone())
            .collect::<Vec<_>>();
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            let majority = majority_counts
                .get(&(*head, *config, group))
                .map(|counts| argmax_usize(counts))
                .unwrap_or(0);
            group_majority.push(majority);
        }

        let mut cluster_models = Vec::new();
        for &cluster_count in cluster_counts {
            let centroids = kmeans_centroids(&feature_rows, cluster_count, 25);
            let assignments = train_samples
                .iter()
                .map(|sample| nearest_attention_cluster(&sample.features, &centroids))
                .collect::<Vec<_>>();
            for name in attention_cluster_probe_names(cluster_count) {
                let mut key_counts: HashMap<(usize, String), Vec<usize>> = HashMap::new();
                for (sample, &cluster) in train_samples.iter().zip(assignments.iter()) {
                    let key = attention_cluster_key(
                        &name,
                        &sample.token_ids,
                        &sample.stratum,
                        sample.position,
                        cluster,
                    );
                    for &group in selected_groups {
                        let levels = 1usize << config.bits_per_group;
                        let counts = key_counts
                            .entry((group, key.clone()))
                            .or_insert_with(|| vec![0; levels]);
                        counts[sample.codes[group]] += 1;
                    }
                }

                let mut group_maps = vec![HashMap::new(); config.groups];
                let mut group_train_accuracy = vec![0.0; config.groups];
                for &group in selected_groups {
                    let mut correct = 0usize;
                    let mut total = 0usize;
                    for ((map_group, key), counts) in key_counts.iter() {
                        if *map_group == group {
                            let best = argmax_usize(counts);
                            correct += counts[best];
                            total += counts.iter().sum::<usize>();
                            group_maps[group].insert(key.clone(), best);
                        }
                    }
                    group_train_accuracy[group] = if total == 0 {
                        0.0
                    } else {
                        correct as f64 / total as f64
                    };
                }
                let selected_group_keys = (0..config.groups)
                    .map(|group| {
                        if selected_groups.contains(&group) {
                            format!("{}_train_acc_{:.3}", name, group_train_accuracy[group])
                        } else {
                            "majority".to_string()
                        }
                    })
                    .collect();
                cluster_models.push(AddressAttentionClusterGroupModel {
                    name,
                    groups: selected_groups.to_vec(),
                    qk_rank: None,
                    centroids: centroids.clone(),
                    group_majority: group_majority.clone(),
                    group_maps,
                    selected_group_keys,
                });
            }
        }
        models.insert((*head, *config), cluster_models);
    }

    Ok(models)
}

pub(in super::super) fn fit_address_reduced_qk_cluster_group_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    qk_ranks: &[usize],
    cluster_counts: &[usize],
) -> Result<
    HashMap<(HeadId, PqConfig), Vec<AddressAttentionClusterGroupModel>>,
    Box<dyn std::error::Error>,
> {
    let mut models: HashMap<(HeadId, PqConfig), Vec<AddressAttentionClusterGroupModel>> =
        HashMap::new();

    for &qk_rank in qk_ranks {
        let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
        let mut samples: HashMap<(HeadId, PqConfig), Vec<AttentionClusterFitSample>> =
            HashMap::new();

        let label = if qk_rank == 0 {
            "full-qk-cluster-fit".to_string()
        } else {
            format!("reduced-qk-r{qk_rank}-cluster-fit")
        };
        visit_code_samples(
            weights,
            index,
            tokenizer,
            prompts,
            heads,
            bases,
            means,
            pca_bases,
            codebooks,
            &label,
            false,
            0,
            0,
            true,
            if qk_rank == 0 { None } else { Some(qk_rank) },
            |head, config, pos, codes, token_ids, stratum, _, _, _, _, attention_weights| {
                for (group, &code) in codes.iter().enumerate() {
                    let levels = 1usize << config.bits_per_group;
                    let counts = majority_counts
                        .entry((head, config, group))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
                let attention_weights =
                    attention_weights.ok_or("missing attention row during reduced-QK fit")?;
                samples
                    .entry((head, config))
                    .or_default()
                    .push(AttentionClusterFitSample {
                        features: attention_pattern_features(attention_weights, pos),
                        codes: codes.to_vec(),
                        token_ids: token_ids.to_vec(),
                        stratum: stratum.to_string(),
                        position: pos,
                    });
                Ok(())
            },
        )?;

        for ((head, config), _) in codebooks {
            let train_samples = samples.get(&(*head, *config)).cloned().unwrap_or_default();
            let feature_rows = train_samples
                .iter()
                .map(|sample| sample.features.clone())
                .collect::<Vec<_>>();
            let mut group_majority = Vec::with_capacity(config.groups);
            for group in 0..config.groups {
                let majority = majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0);
                group_majority.push(majority);
            }

            let rank_prefix = if qk_rank == 0 {
                "qk_full".to_string()
            } else {
                format!("qk_rank{qk_rank}")
            };
            let entry = models.entry((*head, *config)).or_default();
            for &cluster_count in cluster_counts {
                let centroids = kmeans_centroids(&feature_rows, cluster_count, 25);
                let assignments = train_samples
                    .iter()
                    .map(|sample| nearest_attention_cluster(&sample.features, &centroids))
                    .collect::<Vec<_>>();
                for base_name in attention_cluster_probe_names(cluster_count) {
                    let name = format!("{rank_prefix}_{base_name}");
                    let mut key_counts: HashMap<(usize, String), Vec<usize>> = HashMap::new();
                    for (sample, &cluster) in train_samples.iter().zip(assignments.iter()) {
                        let key = attention_cluster_key(
                            &base_name,
                            &sample.token_ids,
                            &sample.stratum,
                            sample.position,
                            cluster,
                        );
                        for &group in selected_groups {
                            let levels = 1usize << config.bits_per_group;
                            let counts = key_counts
                                .entry((group, key.clone()))
                                .or_insert_with(|| vec![0; levels]);
                            counts[sample.codes[group]] += 1;
                        }
                    }

                    let mut group_maps = vec![HashMap::new(); config.groups];
                    let mut group_train_accuracy = vec![0.0; config.groups];
                    for &group in selected_groups {
                        let mut correct = 0usize;
                        let mut total = 0usize;
                        for ((map_group, key), counts) in key_counts.iter() {
                            if *map_group == group {
                                let best = argmax_usize(counts);
                                correct += counts[best];
                                total += counts.iter().sum::<usize>();
                                group_maps[group].insert(key.clone(), best);
                            }
                        }
                        group_train_accuracy[group] = if total == 0 {
                            0.0
                        } else {
                            correct as f64 / total as f64
                        };
                    }
                    let selected_group_keys = (0..config.groups)
                        .map(|group| {
                            if selected_groups.contains(&group) {
                                format!("{name}_train_acc_{:.3}", group_train_accuracy[group])
                            } else {
                                "majority".to_string()
                            }
                        })
                        .collect();
                    entry.push(AddressAttentionClusterGroupModel {
                        name,
                        groups: selected_groups.to_vec(),
                        qk_rank: if qk_rank == 0 { None } else { Some(qk_rank) },
                        centroids: centroids.clone(),
                        group_majority: group_majority.clone(),
                        group_maps,
                        selected_group_keys,
                    });
                }
            }
        }
    }

    Ok(models)
}

pub(in super::super) fn fit_address_lsh_group_models(
    weights: &mut larql_inference::ModelWeights,
    index: &VectorIndex,
    tokenizer: &tokenizers::Tokenizer,
    prompts: &[PromptRecord],
    heads: &[HeadId],
    bases: &HashMap<HeadId, WoRoundtripBasis>,
    means: &HashMap<HeadId, StaticHeadMeans>,
    pca_bases: &HashMap<HeadId, ZPcaBasis>,
    codebooks: &HashMap<(HeadId, PqConfig), PqCodebook>,
    selected_groups: &[usize],
    bits: usize,
    seeds: usize,
) -> Result<HashMap<(HeadId, PqConfig), AddressLshGroupModel>, Box<dyn std::error::Error>> {
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();
    let mut bucket_counts: HashMap<(HeadId, PqConfig, usize, u64, usize), Vec<usize>> =
        HashMap::new();

    visit_code_samples(
        weights,
        index,
        tokenizer,
        prompts,
        heads,
        bases,
        means,
        pca_bases,
        codebooks,
        "lsh-fit",
        true,
        0,
        0,
        false,
        None,
        |head, config, _pos, codes, _token_ids, _stratum, _, input_row, _, _, _| {
            let input_row = input_row.ok_or("missing layer-input row during LSH address fit")?;
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            for &group in selected_groups {
                let code = codes[group];
                for seed in 0..seeds {
                    let bucket = lsh_bucket(ArrayView1::from(input_row), seed as u64, bits);
                    let levels = 1usize << config.bits_per_group;
                    let counts = bucket_counts
                        .entry((head, config, group, seed as u64, bucket))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut group_majority = Vec::with_capacity(config.groups);
        for group in 0..config.groups {
            let majority = majority_counts
                .get(&(*head, *config, group))
                .map(|counts| argmax_usize(counts))
                .unwrap_or(0);
            group_majority.push(majority);
        }

        let mut group_maps = vec![HashMap::new(); config.groups];
        let mut group_seeds = vec![0_u64; config.groups];
        let mut group_train_accuracy = vec![0.0; config.groups];
        for &group in selected_groups {
            let mut best_seed = 0_u64;
            let mut best_accuracy = -1.0_f64;
            let mut best_map = HashMap::new();
            for seed in 0..seeds {
                let seed = seed as u64;
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_group, map_seed, bucket), counts) in
                    bucket_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && *map_group == group
                        && *map_seed == seed
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(*bucket, best);
                    }
                }
                let accuracy = if total == 0 {
                    0.0
                } else {
                    correct as f64 / total as f64
                };
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_seed = seed;
                    best_map = map;
                }
            }
            group_maps[group] = best_map;
            group_seeds[group] = best_seed;
            group_train_accuracy[group] = best_accuracy.max(0.0);
        }

        models.insert(
            (*head, *config),
            AddressLshGroupModel {
                groups: selected_groups.to_vec(),
                bits,
                group_majority,
                group_maps,
                group_seeds,
                group_train_accuracy,
            },
        );
    }

    Ok(models)
}
