//! Cross-prompt PCA: singular values, eigenvectors and cumulative variance.

use serde::Serialize;

#[allow(unused_imports)]
use super::*;

#[derive(Serialize)]
pub(super) struct CrossPromptPca {
    pub(super) _type: String,
    pub(super) shared_singular_values: Vec<f32>,
    pub(super) shared_cumulative_variance: Vec<f32>,
    pub(super) shared_dims_for_95: usize,
    // Each prompt's trajectory projected into shared PCA space: [prompt][layer][component]
    pub(super) projections: Vec<PromptProjection>,
    // Pairwise cosine similarity of projected trajectories
    pub(super) pairwise_trajectory_cosine: Vec<PairwiseSim>,
    // Layer-by-layer divergence between prompt pairs
    pub(super) divergence_profiles: Vec<DivergenceProfile>,
}

#[derive(Serialize)]
pub(super) struct PromptProjection {
    pub(super) prompt: String,
    pub(super) prompt_index: usize,
    // coords[layer_idx] = [pc0, pc1, pc2, ...] (top 10 components)
    pub(super) coords: Vec<Vec<f32>>,
}

#[derive(Serialize)]
pub(super) struct PairwiseSim {
    pub(super) prompt_a: String,
    pub(super) prompt_b: String,
    pub(super) cosine: f32,
}

#[derive(Serialize)]
pub(super) struct DivergenceProfile {
    pub(super) prompt_a: String,
    pub(super) prompt_b: String,
    // Per-layer cosine similarity between the two trajectories' residuals
    pub(super) layer_cosines: Vec<LayerCosine>,
    pub(super) diverge_layer: Option<usize>,
}

#[derive(Serialize)]
pub(super) struct LayerCosine {
    pub(super) layer: usize,
    pub(super) cosine: f32,
}

// ── Math helpers ──

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub(super) fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub(super) fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let na = vec_norm(a);
    let nb = vec_norm(b);
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0)
}

/// SVD of an n_rows × n_cols matrix (n_rows << n_cols).
/// Computes via eigendecomposition of the n_rows × n_rows Gram matrix.
/// Returns singular values sorted descending.
pub(super) fn svd_singular_values(rows: &[Vec<f32>], n_rows: usize) -> Vec<f32> {
    // Build Gram matrix G = X × X^T  (n_rows × n_rows)
    let mut gram = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in i..n_rows {
            let d = dot(&rows[i], &rows[j]);
            gram[i * n_rows + j] = d;
            gram[j * n_rows + i] = d;
        }
    }

    // Power iteration with deflation on the Gram matrix
    let mut eigenvalues = Vec::with_capacity(n_rows);
    let iterations = 80;

    for _ in 0..n_rows {
        let mut v = vec![1.0f32; n_rows];
        let n = vec_norm(&v);
        for x in v.iter_mut() {
            *x /= n;
        }

        let mut eigenvalue = 0.0f32;
        for _ in 0..iterations {
            // mv = gram × v
            let mut mv = vec![0.0f32; n_rows];
            for i in 0..n_rows {
                let mut s = 0.0f32;
                for j in 0..n_rows {
                    s += gram[i * n_rows + j] * v[j];
                }
                mv[i] = s;
            }
            eigenvalue = dot(&mv, &v);
            let n = vec_norm(&mv);
            if n < 1e-12 {
                break;
            }
            for (x, m) in v.iter_mut().zip(mv.iter()) {
                *x = m / n;
            }
        }

        if eigenvalue < 1e-8 {
            break;
        }

        eigenvalues.push(eigenvalue.sqrt());

        // Deflate
        for i in 0..n_rows {
            for j in 0..n_rows {
                gram[i * n_rows + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

/// Compute top-k eigenvectors of an n×n Gram matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors[k] has length n.
pub(super) fn svd_eigenvectors(
    rows: &[Vec<f32>],
    n_rows: usize,
    top_k: usize,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut gram = vec![0.0f32; n_rows * n_rows];
    for i in 0..n_rows {
        for j in i..n_rows {
            let d = dot(&rows[i], &rows[j]);
            gram[i * n_rows + j] = d;
            gram[j * n_rows + i] = d;
        }
    }

    let mut eigenvalues = Vec::with_capacity(top_k);
    let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(top_k);
    let iterations = 80;

    for _ in 0..top_k.min(n_rows) {
        let mut v = vec![1.0f32; n_rows];
        let n = vec_norm(&v);
        for x in v.iter_mut() {
            *x /= n;
        }

        let mut eigenvalue = 0.0f32;
        for _ in 0..iterations {
            let mut mv = vec![0.0f32; n_rows];
            for i in 0..n_rows {
                let mut s = 0.0f32;
                for j in 0..n_rows {
                    s += gram[i * n_rows + j] * v[j];
                }
                mv[i] = s;
            }
            eigenvalue = dot(&mv, &v);
            let n = vec_norm(&mv);
            if n < 1e-12 {
                break;
            }
            for (x, m) in v.iter_mut().zip(mv.iter()) {
                *x = m / n;
            }
        }

        if eigenvalue < 1e-8 {
            break;
        }

        eigenvalues.push(eigenvalue.sqrt());
        eigenvectors.push(v.clone());

        for i in 0..n_rows {
            for j in 0..n_rows {
                gram[i * n_rows + j] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

pub(super) fn cumulative_variance(singular_values: &[f32]) -> Vec<f32> {
    let total: f32 = singular_values.iter().map(|s| s * s).sum();
    if total < 1e-12 {
        return vec![0.0; singular_values.len()];
    }
    let mut cum = 0.0f32;
    singular_values
        .iter()
        .map(|s| {
            cum += s * s;
            round4(cum / total)
        })
        .collect()
}

pub(super) fn dims_for_threshold(cumvar: &[f32], threshold: f32) -> usize {
    cumvar
        .iter()
        .position(|&v| v >= threshold)
        .map(|i| i + 1)
        .unwrap_or(cumvar.len())
}

pub(super) fn make_pca_info(singular_values: Vec<f32>) -> PcaInfo {
    let cumvar = cumulative_variance(&singular_values);
    let dims_90 = dims_for_threshold(&cumvar, 0.90);
    let dims_95 = dims_for_threshold(&cumvar, 0.95);
    let dims_99 = dims_for_threshold(&cumvar, 0.99);
    PcaInfo {
        singular_values: singular_values.iter().map(|v| round4(*v)).collect(),
        cumulative_variance: cumvar,
        dims_for_90: dims_90,
        dims_for_95: dims_95,
        dims_for_99: dims_99,
    }
}
