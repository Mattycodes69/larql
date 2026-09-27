//! Self-tests run before a capture.

use larql_compute::ffn::WeightFfn;
use larql_compute::forward::layer::apply_layer_scalar;
use larql_compute::forward::predict::raw::hidden_to_raw_logits;
use larql_compute::forward::{
    apply_per_layer_embedding, embed_tokens_pub, precompute_per_layer_inputs, run_attention,
    run_ffn,
};
use larql_models::ModelWeights;
use ndarray::{s, Array2};
use std::path::Path;

#[allow(unused_imports)]
use super::*;

pub(super) fn run_selftests(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    corpus: &Corpus,
    out_dir: &Path,
    identity: &serde_json::Value,
) {
    let smoke: Vec<&PromptRec> = corpus.prompts.iter().take(SMOKE_N).collect();

    // 1. Determinism: sequential and parallel captures of the smoke set
    //    must serialize byte-identically (licenses parallel bulk capture).
    let mut serialized = Vec::new();
    for parallel in [false, true] {
        let buf = capture_batch(weights, tokenizer, &smoke, parallel);
        println!(
            "selftest determinism {} round: {} rows",
            if parallel { "parallel" } else { "sequential" },
            buf.rows
        );
        serialized.push(buf.serialize_main());
    }
    let determinism = serialized[0] == serialized[1];
    if !determinism {
        refuse("determinism self-test FAILED: sequential and parallel captures differ byte-wise");
    }
    println!("selftest determinism: PASS (sequential == parallel)");

    // 2. Recompute parity: activation rows recomputed via the independent
    //    capture_ffn_activation_matrix path must match the run_ffn capture.
    let mut parity_cells = 0usize;
    for rec in &smoke {
        let ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &rec.text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {}: {e:?}", rec.prompt_id)));
        let sel = selected_positions(ids.len());
        for &layer in PROBE_LAYERS.iter().take(2) {
            let full =
                larql_inference::forward::capture_ffn_activation_matrix(weights, &ids, layer)
                    .unwrap_or_else(|| refuse("capture_ffn_activation_matrix returned None"));
            // Reference from the harness's own loop, via a fresh capture with
            // every cell of this (prompt, layer) checksummed.
            let ffn = WeightFfn { weights };
            let h0 = embed_tokens_pub(weights, &ids);
            let ple_inputs = precompute_per_layer_inputs(weights, &h0, &ids);
            let mut h = h0;
            for l in 0..=layer {
                let h_post_attn =
                    match run_attention(larql_models::WeightsView::dense(weights), &h, l) {
                        Some(pa) => pa,
                        None => h.clone(),
                    };
                let (h_post_ffn, act) = run_ffn(weights, &h_post_attn, l, &ffn, l == layer);
                if l == layer {
                    let act = act.expect("activation");
                    for &p in &sel {
                        let a = act.row(p);
                        let b = full.row(p);
                        if a != b {
                            refuse(&format!(
                                "recompute parity FAILED at {} pos {p} layer {layer}",
                                rec.prompt_id
                            ));
                        }
                        parity_cells += 1;
                    }
                }
                let mut h_new =
                    apply_per_layer_embedding(weights, &h_post_ffn, l, ple_inputs.get(l));
                apply_layer_scalar(weights, &mut h_new, l);
                h = h_new;
            }
        }
    }
    println!("selftest recompute parity: PASS ({parity_cells} cells bit-identical)");

    // 3. Splice floor: resume from captured i_29, logits must be bit-identical.
    let mut floor_prompts = 0usize;
    for rec in &smoke {
        let ids = larql_inference::encode_prompt(tokenizer, &*weights.arch, &rec.text)
            .unwrap_or_else(|e| refuse(&format!("tokenize {}: {e:?}", rec.prompt_id)));
        let cap = capture_prompt(weights, &ids, &rec.prompt_id, true);
        let i29 = cap.i29_full.expect("i29 kept");
        let ffn = WeightFfn { weights };
        let h0 = embed_tokens_pub(weights, &ids);
        let ple_inputs = precompute_per_layer_inputs(weights, &h0, &ids);
        let mut h = i29;
        for layer in 29..weights.num_layers {
            let h_post_attn =
                match run_attention(larql_models::WeightsView::dense(weights), &h, layer) {
                    Some(pa) => pa,
                    None => h.clone(),
                };
            let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
            let mut h_new =
                apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
            apply_layer_scalar(weights, &mut h_new, layer);
            h = h_new;
        }
        let last = ids.len() - 1;
        let resumed = hidden_to_raw_logits(weights, &h.slice(s![last..last + 1, ..]).to_owned());
        let canon_row = {
            // Canonical logits recomputed from the capture's own stored final
            // boundary (i_34) row for the last selected position.
            let sel = selected_positions(ids.len());
            let pi = sel.iter().position(|&p| p == last).expect("last selected");
            let hlen = weights.hidden_size;
            let off = (pi * 35 + 34) * hlen;
            let row = Array2::from_shape_vec((1, hlen), cap.boundaries[off..off + hlen].to_vec())
                .expect("row");
            hidden_to_raw_logits(weights, &row)
        };
        if resumed != canon_row {
            refuse(&format!(
                "splice floor FAILED on {}: resumed logits differ",
                rec.prompt_id
            ));
        }
        floor_prompts += 1;
    }
    println!("selftest splice floor: PASS ({floor_prompts} prompts bit-identical)");

    let report = serde_json::json!({
        "determinism": "pass",
        "recompute_parity": "pass",
        "splice_floor": "pass",
        "smoke_n": SMOKE_N,
        "identity": identity,
    });
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    write_atomic(
        &out_dir.join("selftest-report.json"),
        serde_json::to_string_pretty(&report).unwrap().as_bytes(),
    );
    println!(
        "selftest report written: {}",
        out_dir.join("selftest-report.json").display()
    );
}
