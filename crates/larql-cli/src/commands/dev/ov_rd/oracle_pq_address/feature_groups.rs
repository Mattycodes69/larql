//! Address group models keyed on FFN features and attention relations.

use super::super::address::{
    attention_relation_key, attention_relation_probe_names, ffn_first_feature_key,
    ffn_first_feature_probe_names, prev_ffn_feature_key, prev_ffn_feature_probe_names,
    AddressProbeModel,
};
use super::super::basis::{WoRoundtripBasis, ZPcaBasis};
use super::super::metrics::argmax_usize;
use super::super::pq::PqCodebook;
use super::super::stats::StaticHeadMeans;
use super::super::types::{HeadId, PqConfig, PromptRecord};
use larql_vindex::VectorIndex;
use std::collections::HashMap;

#[allow(unused_imports)]
use super::*;

pub(in super::super) fn fit_address_prev_ffn_feature_group_models(
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
    feature_top_k: usize,
) -> Result<HashMap<(HeadId, PqConfig), Vec<AddressProbeModel>>, Box<dyn std::error::Error>> {
    let names = prev_ffn_feature_probe_names();
    let mut key_counts: HashMap<(HeadId, PqConfig, String, usize, String), Vec<usize>> =
        HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

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
        "prev-ffn-feature-fit",
        false,
        feature_top_k,
        0,
        false,
        None,
        |head, config, pos, codes, token_ids, stratum, _, _, prev_features, _, _| {
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            let prev_features = prev_features.unwrap_or(&[]);
            for &group in selected_groups {
                let code = codes[group];
                for name in &names {
                    let key = prev_ffn_feature_key(name, token_ids, stratum, pos, prev_features);
                    let levels = 1usize << config.bits_per_group;
                    let counts = key_counts
                        .entry((head, config, (*name).to_string(), group, key))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut probe_models = Vec::new();
        for name in &names {
            let mut group_majority = Vec::with_capacity(config.groups);
            let mut group_maps = vec![HashMap::new(); config.groups];
            let mut group_train_accuracy = vec![0.0; config.groups];
            for group in 0..config.groups {
                let majority = majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0);
                group_majority.push(majority);
            }
            for &group in selected_groups {
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_name, map_group, key), counts) in key_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && map_name == name
                        && *map_group == group
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(key.clone(), best);
                    }
                }
                group_maps[group] = map;
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
            probe_models.push(AddressProbeModel {
                name: (*name).to_string(),
                group_majority,
                group_maps,
                group_train_accuracy,
                selected_group_keys,
            });
        }
        models.insert((*head, *config), probe_models);
    }

    Ok(models)
}

pub(in super::super) fn fit_address_ffn_first_feature_group_models(
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
    feature_top_k: usize,
) -> Result<HashMap<(HeadId, PqConfig), Vec<AddressProbeModel>>, Box<dyn std::error::Error>> {
    let names = ffn_first_feature_probe_names();
    let mut key_counts: HashMap<(HeadId, PqConfig, String, usize, String), Vec<usize>> =
        HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

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
        "ffn-first-feature-fit",
        false,
        0,
        feature_top_k,
        false,
        None,
        |head, config, pos, codes, token_ids, stratum, _, _, _, ffn_first_features, _| {
            for (group, &code) in codes.iter().enumerate() {
                let levels = 1usize << config.bits_per_group;
                let counts = majority_counts
                    .entry((head, config, group))
                    .or_insert_with(|| vec![0; levels]);
                counts[code] += 1;
            }
            let ffn_first_features = ffn_first_features.unwrap_or(&[]);
            for &group in selected_groups {
                let code = codes[group];
                for name in &names {
                    let key =
                        ffn_first_feature_key(name, token_ids, stratum, pos, ffn_first_features);
                    let levels = 1usize << config.bits_per_group;
                    let counts = key_counts
                        .entry((head, config, (*name).to_string(), group, key))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut probe_models = Vec::new();
        for name in &names {
            let mut group_majority = Vec::with_capacity(config.groups);
            let mut group_maps = vec![HashMap::new(); config.groups];
            let mut group_train_accuracy = vec![0.0; config.groups];
            for group in 0..config.groups {
                let majority = majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0);
                group_majority.push(majority);
            }
            for &group in selected_groups {
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_name, map_group, key), counts) in key_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && map_name == name
                        && *map_group == group
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(key.clone(), best);
                    }
                }
                group_maps[group] = map;
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
            probe_models.push(AddressProbeModel {
                name: (*name).to_string(),
                group_majority,
                group_maps,
                group_train_accuracy,
                selected_group_keys,
            });
        }
        models.insert((*head, *config), probe_models);
    }

    Ok(models)
}

pub(in super::super) fn fit_address_attention_relation_group_models(
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
) -> Result<HashMap<(HeadId, PqConfig), Vec<AddressProbeModel>>, Box<dyn std::error::Error>> {
    let names = attention_relation_probe_names();
    let mut key_counts: HashMap<(HeadId, PqConfig, String, usize, String), Vec<usize>> =
        HashMap::new();
    let mut majority_counts: HashMap<(HeadId, PqConfig, usize), Vec<usize>> = HashMap::new();

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
        "attention-relation-fit",
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
                attention_weights.ok_or("missing attention row during relation address fit")?;
            for &group in selected_groups {
                let code = codes[group];
                for name in &names {
                    let key =
                        attention_relation_key(name, token_ids, stratum, pos, attention_weights);
                    let levels = 1usize << config.bits_per_group;
                    let counts = key_counts
                        .entry((head, config, (*name).to_string(), group, key))
                        .or_insert_with(|| vec![0; levels]);
                    counts[code] += 1;
                }
            }
            Ok(())
        },
    )?;

    let mut models = HashMap::new();
    for ((head, config), _) in codebooks {
        let mut probe_models = Vec::new();
        for name in &names {
            let mut group_majority = Vec::with_capacity(config.groups);
            let mut group_maps = vec![HashMap::new(); config.groups];
            let mut group_train_accuracy = vec![0.0; config.groups];
            for group in 0..config.groups {
                let majority = majority_counts
                    .get(&(*head, *config, group))
                    .map(|counts| argmax_usize(counts))
                    .unwrap_or(0);
                group_majority.push(majority);
            }
            for &group in selected_groups {
                let mut map = HashMap::new();
                let mut correct = 0usize;
                let mut total = 0usize;
                for ((map_head, map_config, map_name, map_group, key), counts) in key_counts.iter()
                {
                    if map_head == head
                        && map_config == config
                        && map_name == name
                        && *map_group == group
                    {
                        let best = argmax_usize(counts);
                        correct += counts[best];
                        total += counts.iter().sum::<usize>();
                        map.insert(key.clone(), best);
                    }
                }
                group_maps[group] = map;
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
            probe_models.push(AddressProbeModel {
                name: (*name).to_string(),
                group_majority,
                group_maps,
                group_train_accuracy,
                selected_group_keys,
            });
        }
        models.insert((*head, *config), probe_models);
    }

    Ok(models)
}
