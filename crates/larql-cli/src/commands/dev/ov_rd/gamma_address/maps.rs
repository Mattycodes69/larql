//! Projected-address map fitting: affine, low-rank and PCA bases.

use super::super::address::{
    predict_code_from_hyperplanes, train_binary_hyperplane, AddressSupervisedGroupModel,
};
use super::super::types::PqConfig;

#[allow(unused_imports)]
use super::*;

pub(super) fn fit_one_projected_model(
    name: &str,
    source: GammaProjectionSource,
    rows: &[Vec<f32>],
    samples: &[&GammaCodeSample],
    config: PqConfig,
    selected_groups: &[usize],
    group_majority: &[usize],
    epochs: usize,
    lr: f32,
    l2: f32,
) -> GammaProjectedAddressModel {
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let row_refs = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut group_hyperplanes = vec![Vec::new(); config.groups];
    let mut group_train_accuracy = vec![0.0; config.groups];
    for &group in selected_groups {
        let mut bit_planes = Vec::with_capacity(config.bits_per_group);
        for bit in 0..config.bits_per_group {
            let labels = samples
                .iter()
                .map(|sample| ((sample.codes[group] >> bit) & 1) != 0)
                .collect::<Vec<_>>();
            bit_planes.push(train_binary_hyperplane(
                &row_refs, &labels, dim, epochs, lr, l2,
            ));
        }

        let mut correct = 0usize;
        for (row, sample) in rows.iter().zip(samples.iter()) {
            let predicted = predict_code_from_hyperplanes(row, &bit_planes);
            if predicted == sample.codes[group] {
                correct += 1;
            }
        }
        group_train_accuracy[group] = if rows.is_empty() {
            0.0
        } else {
            correct as f64 / rows.len() as f64
        };
        group_hyperplanes[group] = bit_planes;
    }

    GammaProjectedAddressModel {
        name: name.to_string(),
        source,
        supervised: AddressSupervisedGroupModel {
            groups: selected_groups.to_vec(),
            bits_per_group: config.bits_per_group,
            epochs,
            lr,
            l2,
            group_majority: group_majority.to_vec(),
            group_hyperplanes,
            group_train_accuracy,
        },
    }
}

pub(super) fn fit_diagonal_affine_map(pairs: &[(&[f32], &[f32])], dim: usize) -> DiagonalAffineMap {
    let n = pairs.len().max(1) as f64;
    let mut sum_x = vec![0.0_f64; dim];
    let mut sum_y = vec![0.0_f64; dim];
    let mut sum_xx = vec![0.0_f64; dim];
    let mut sum_xy = vec![0.0_f64; dim];
    for &(x, y) in pairs {
        for dim_idx in 0..dim {
            let xi = x[dim_idx] as f64;
            let yi = y[dim_idx] as f64;
            sum_x[dim_idx] += xi;
            sum_y[dim_idx] += yi;
            sum_xx[dim_idx] += xi * xi;
            sum_xy[dim_idx] += xi * yi;
        }
    }

    let mut mean_x = vec![0.0_f32; dim];
    let mut mean_y = vec![0.0_f32; dim];
    let mut slope = vec![0.0_f32; dim];
    for dim_idx in 0..dim {
        let mx = sum_x[dim_idx] / n;
        let my = sum_y[dim_idx] / n;
        let var_x = (sum_xx[dim_idx] / n) - mx * mx;
        let cov_xy = (sum_xy[dim_idx] / n) - mx * my;
        mean_x[dim_idx] = mx as f32;
        mean_y[dim_idx] = my as f32;
        slope[dim_idx] = if var_x.abs() > 1e-12 {
            (cov_xy / var_x) as f32
        } else {
            0.0
        };
    }

    DiagonalAffineMap {
        mean_x,
        mean_y,
        slope,
    }
}

pub(super) fn fit_learned_low_rank_map(
    pairs: &[(&[f32], &[f32])],
    dim: usize,
    rank: usize,
    pca_iters: usize,
    epochs: usize,
    lr: f32,
    l2: f32,
    seed: u64,
) -> LearnedLowRankMap {
    let (mean_x, mean_y) = pair_means(pairs, dim);
    let basis_y = fit_target_power_pca_basis(pairs, &mean_y, dim, rank, pca_iters, seed);
    let mut map = LearnedLowRankMap {
        mean_x,
        mean_y,
        basis_y,
        weights: vec![vec![0.0_f32; dim]; rank],
        bias: vec![0.0_f32; rank],
        rank,
    };
    let target_coords = pairs
        .iter()
        .map(|(_, target)| map.target_coordinates(target))
        .collect::<Vec<_>>();
    let input_norms = pairs
        .iter()
        .map(|(input, _)| {
            input
                .iter()
                .zip(map.mean_x.iter())
                .map(|(&x, &mean)| {
                    let centered = x - mean;
                    centered * centered
                })
                .sum::<f32>()
                .max(1.0)
        })
        .collect::<Vec<_>>();

    for _ in 0..epochs {
        for (sample_idx, (input, _)) in pairs.iter().enumerate() {
            let norm = input_norms[sample_idx];
            let step = lr / norm;
            for component in 0..rank {
                let mut pred = map.bias[component];
                for (dim_idx, &x) in input.iter().enumerate() {
                    pred += map.weights[component][dim_idx] * (x - map.mean_x[dim_idx]);
                }
                let err = pred - target_coords[sample_idx][component];
                map.bias[component] -= lr * err * 0.01;
                for (dim_idx, &x) in input.iter().enumerate() {
                    let centered = x - map.mean_x[dim_idx];
                    let grad = err * centered + l2 * map.weights[component][dim_idx];
                    map.weights[component][dim_idx] -= step * grad;
                }
            }
        }
    }
    map
}

pub(super) fn pair_means(pairs: &[(&[f32], &[f32])], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = pairs.len().max(1) as f64;
    let mut mean_x = vec![0.0_f64; dim];
    let mut mean_y = vec![0.0_f64; dim];
    for &(x, y) in pairs {
        for dim_idx in 0..dim {
            mean_x[dim_idx] += x[dim_idx] as f64;
            mean_y[dim_idx] += y[dim_idx] as f64;
        }
    }
    (
        mean_x.into_iter().map(|value| (value / n) as f32).collect(),
        mean_y.into_iter().map(|value| (value / n) as f32).collect(),
    )
}

pub(super) fn fit_target_power_pca_basis(
    pairs: &[(&[f32], &[f32])],
    mean_y: &[f32],
    dim: usize,
    rank: usize,
    pca_iters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut basis = Vec::with_capacity(rank);
    for component in 0..rank {
        let mut v = deterministic_unit_vector(dim, seed ^ component as u64);
        orthonormalize(&mut v, &basis);
        for _ in 0..pca_iters {
            let mut next = vec![0.0_f64; dim];
            for &(_, y) in pairs {
                let dot = y
                    .iter()
                    .zip(mean_y.iter())
                    .zip(v.iter())
                    .map(|((&yi, &mean), &vi)| (yi - mean) as f64 * vi as f64)
                    .sum::<f64>();
                for dim_idx in 0..dim {
                    next[dim_idx] += (y[dim_idx] - mean_y[dim_idx]) as f64 * dot;
                }
            }
            let inv_n = 1.0 / pairs.len().max(1) as f64;
            let mut next_f32 = next
                .into_iter()
                .map(|value| (value * inv_n) as f32)
                .collect::<Vec<_>>();
            orthonormalize(&mut next_f32, &basis);
            v = next_f32;
        }
        basis.push(v);
    }
    basis
}

pub(super) fn deterministic_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut values = (0..dim)
        .map(|idx| {
            let hash = splitmix64(seed ^ (idx as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
            let unit = ((hash >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64));
            (2.0 * unit - 1.0) as f32
        })
        .collect::<Vec<_>>();
    normalize(&mut values);
    values
}

pub(super) fn orthonormalize(v: &mut [f32], basis: &[Vec<f32>]) {
    for prev in basis {
        let dot = v
            .iter()
            .zip(prev.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>() as f32;
        for (value, &prev_value) in v.iter_mut().zip(prev.iter()) {
            *value -= dot * prev_value;
        }
    }
    normalize(v);
}

pub(super) fn normalize(v: &mut [f32]) {
    let norm = v
        .iter()
        .map(|&value| value as f64 * value as f64)
        .sum::<f64>()
        .sqrt();
    if norm > 1e-12 {
        let inv = (1.0 / norm) as f32;
        for value in v {
            *value *= inv;
        }
    }
}
