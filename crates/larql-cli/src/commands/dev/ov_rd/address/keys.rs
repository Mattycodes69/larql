//! Address keys: attention relations/clusters, FFN feature sets and attention buckets.

use ndarray::ArrayView1;

#[allow(unused_imports)]
use super::*;

pub(in super::super) fn attention_relation_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    weights: &[f32],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let argmax = attention_argmax(weights, position);
    let top2 = attention_topk_key(weights, position, 2);
    let top4 = attention_topk_key(weights, position, 4);
    let entropy = attention_entropy_bucket(weights, position);
    let bos = attention_bos_bucket(weights.first().copied().unwrap_or(0.0));
    let distance = attention_distance_bucket(argmax, position);
    let relation = attention_relation_class(argmax, position);
    match name {
        "attn_argmax" => format!("aa:{argmax}"),
        "attn_top2_hash" => format!("at2:{top2}"),
        "attn_top4_hash" => format!("at4:{top4}"),
        "attn_entropy_bucket" => format!("ae:{entropy}"),
        "attn_bos_bucket" => format!("ab:{bos}"),
        "attn_distance_bucket" => format!("ad:{distance}"),
        "attn_relation_class" => format!("ar:{relation}"),
        "stratum_attn_relation_class" => format!("s:{stratum}|ar:{relation}"),
        "token_attn_relation_class" => format!("t:{token}|ar:{relation}"),
        "position_attn_relation_class" => format!("p:{position}|ar:{relation}"),
        _ => format!("ar:{relation}"),
    }
}

pub(in super::super) fn attention_cluster_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    cluster: usize,
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    if name.contains("stratum_attn_cluster_") {
        format!("s:{stratum}|ac:{cluster}")
    } else if name.contains("position_attn_cluster_") {
        format!("p:{position}|ac:{cluster}")
    } else if name.contains("token_attn_cluster_") {
        format!("t:{token}|ac:{cluster}")
    } else {
        format!("ac:{cluster}")
    }
}

pub(in super::super) fn prev_ffn_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    prev_features: &[usize],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let top1 = prev_features
        .first()
        .map(|feature| feature.to_string())
        .unwrap_or_else(|| "none".to_string());
    let top2 = prev_features
        .iter()
        .take(2)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let top2 = if top2.is_empty() {
        "none".to_string()
    } else {
        top2
    };
    let top4 = feature_set_key(prev_features, 4);
    let top8 = feature_set_key(prev_features, 8);
    let top16 = feature_set_key(prev_features, 16);
    match name {
        "prev_ffn_top1" => format!("pf1:{top1}"),
        "prev_ffn_top2_hash" => format!("pf2:{top2}"),
        "prev_ffn_top4_hash" => format!("pf4:{top4}"),
        "prev_ffn_top8_hash" => format!("pf8:{top8}"),
        "prev_ffn_top16_hash" => format!("pf16:{top16}"),
        "stratum_prev_ffn_top1" => format!("s:{stratum}|pf1:{top1}"),
        "stratum_prev_ffn_top8_hash" => format!("s:{stratum}|pf8:{top8}"),
        "token_prev_ffn_top1" => format!("t:{token}|pf1:{top1}"),
        "token_prev_ffn_top8_hash" => format!("t:{token}|pf8:{top8}"),
        "position_prev_ffn_top1" => format!("p:{position}|pf1:{top1}"),
        "position_prev_ffn_top8_hash" => format!("p:{position}|pf8:{top8}"),
        _ => format!("pf1:{top1}"),
    }
}

pub(in super::super) fn ffn_first_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    features: &[usize],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let top1 = features
        .first()
        .map(|feature| feature.to_string())
        .unwrap_or_else(|| "none".to_string());
    let top2 = features
        .iter()
        .take(2)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let top2 = if top2.is_empty() {
        "none".to_string()
    } else {
        top2
    };
    let top4 = feature_set_key(features, 4);
    let top8 = feature_set_key(features, 8);
    let top16 = feature_set_key(features, 16);
    match name {
        "ffn_first_top1" => format!("ff1:{top1}"),
        "ffn_first_top2_hash" => format!("ff2:{top2}"),
        "ffn_first_top4_hash" => format!("ff4:{top4}"),
        "ffn_first_top8_hash" => format!("ff8:{top8}"),
        "ffn_first_top16_hash" => format!("ff16:{top16}"),
        "stratum_ffn_first_top1" => format!("s:{stratum}|ff1:{top1}"),
        "stratum_ffn_first_top8_hash" => format!("s:{stratum}|ff8:{top8}"),
        "token_ffn_first_top1" => format!("t:{token}|ff1:{top1}"),
        "token_ffn_first_top8_hash" => format!("t:{token}|ff8:{top8}"),
        "position_ffn_first_top1" => format!("p:{position}|ff1:{top1}"),
        "position_ffn_first_top8_hash" => format!("p:{position}|ff8:{top8}"),
        _ => format!("ff1:{top1}"),
    }
}

pub(in super::super) fn attention_argmax(weights: &[f32], position: usize) -> usize {
    let causal_len = (position + 1).min(weights.len());
    weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

pub(super) fn attention_topk_key(weights: &[f32], position: usize, k: usize) -> String {
    let causal_len = (position + 1).min(weights.len());
    let mut indexed = weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let key = indexed
        .into_iter()
        .take(k)
        .map(|(source, _)| source.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if key.is_empty() {
        "none".to_string()
    } else {
        key
    }
}

pub(in super::super) fn attention_entropy_bits(weights: &[f32], position: usize) -> f64 {
    let causal_len = (position + 1).min(weights.len());
    weights
        .iter()
        .take(causal_len)
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| {
            let p = p as f64;
            -p * p.log2()
        })
        .sum::<f64>()
}

pub(super) fn attention_entropy_bucket(weights: &[f32], position: usize) -> usize {
    let entropy_bits = attention_entropy_bits(weights, position);
    ((entropy_bits * 2.0).floor() as usize).min(16)
}

pub(super) fn attention_bos_bucket(mass: f32) -> &'static str {
    match mass {
        x if x < 0.01 => "lt001",
        x if x < 0.05 => "lt005",
        x if x < 0.10 => "lt010",
        x if x < 0.25 => "lt025",
        x if x < 0.50 => "lt050",
        _ => "ge050",
    }
}

pub(super) fn attention_distance_bucket(argmax: usize, position: usize) -> &'static str {
    if argmax == 0 {
        "bos"
    } else if argmax == position {
        "self"
    } else if argmax + 1 == position {
        "prev"
    } else if argmax > position {
        "future"
    } else {
        match position - argmax {
            0 => "self",
            1 => "prev",
            2..=4 => "d2_4",
            5..=8 => "d5_8",
            9..=16 => "d9_16",
            _ => "far",
        }
    }
}

pub(super) fn attention_relation_class(argmax: usize, position: usize) -> &'static str {
    if argmax == 0 {
        "bos"
    } else if argmax == position {
        "self"
    } else if argmax + 1 == position {
        "prev"
    } else if argmax > position {
        "future"
    } else {
        match position - argmax {
            0 => "self",
            1 => "prev",
            2..=4 => "local",
            5..=16 => "mid",
            _ => "far",
        }
    }
}

pub(super) fn feature_set_key(prev_features: &[usize], k: usize) -> String {
    let key = prev_features
        .iter()
        .take(k)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if key.is_empty() {
        "none".to_string()
    } else {
        key
    }
}

pub(in super::super) fn top_feature_ids_from_activation_row(
    row: ArrayView1<'_, f32>,
    top_k: usize,
) -> Vec<usize> {
    let mut indexed = row.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(top_k)
        .map(|(feature, _)| feature)
        .collect()
}

pub(in super::super) fn attention_pattern_features(weights: &[f32], position: usize) -> Vec<f64> {
    let causal_len = (position + 1).min(weights.len());
    if causal_len == 0 {
        return vec![0.0; 35];
    }
    let denom = causal_len.max(1) as f64;
    let argmax = attention_argmax(weights, position);
    let max_mass = weights.get(argmax).copied().unwrap_or(0.0) as f64;
    let entropy_bits = weights
        .iter()
        .take(causal_len)
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| {
            let p = p as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    let entropy_norm = if causal_len > 1 {
        entropy_bits / (causal_len as f64).log2()
    } else {
        0.0
    };

    let mut bos_mass = 0.0;
    let mut self_mass = 0.0;
    let mut prev_mass = 0.0;
    let mut local_mass = 0.0;
    let mut mid_mass = 0.0;
    let mut far_mass = 0.0;
    for (source, &mass) in weights.iter().take(causal_len).enumerate() {
        let mass = mass as f64;
        if source == 0 {
            bos_mass += mass;
        }
        if source == position {
            self_mass += mass;
        } else if source + 1 == position {
            prev_mass += mass;
        } else if source < position {
            let distance = position - source;
            if distance <= 4 {
                local_mass += mass;
            } else if distance <= 16 {
                mid_mass += mass;
            } else {
                far_mass += mass;
            }
        }
    }

    let argmax_source_norm = argmax as f64 / denom;
    let argmax_distance_norm = if argmax <= position {
        (position - argmax) as f64 / denom
    } else {
        0.0
    };

    let mut features = vec![
        bos_mass,
        self_mass,
        prev_mass,
        local_mass,
        mid_mass,
        far_mass,
        entropy_bits,
        entropy_norm,
        max_mass,
        argmax_source_norm,
        argmax_distance_norm,
    ];

    let mut indexed = weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for rank in 0..8 {
        if let Some((source, mass)) = indexed.get(rank).copied() {
            let source_norm = source as f64 / denom;
            let rel_distance = if source <= position {
                (position - source) as f64 / denom
            } else {
                0.0
            };
            features.push(mass as f64);
            features.push(source_norm);
            features.push(rel_distance);
        } else {
            features.push(0.0);
            features.push(0.0);
            features.push(0.0);
        }
    }

    features
}
