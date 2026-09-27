//! Real headers: Kimi-K3 — the transfer question

use super::*;

/// **Wave 18 did not transfer to K3, and the shapes say why.** The K3
/// programme expected its four `*_res_{norm,proj}` operands to be
/// hyper-connection operands that this wave's generic roles would
/// address. Read from K3's own headers, they are a `[hidden]` norm and a
/// `[1, hidden]` projection per sublayer — and a Sinkhorn site's mix
/// projection is `[(2 + hc)·hc, hc·hidden]`, which equals `[1, hidden]`
/// for NO stream count (the smallest, hc = 1, is `[3, hidden]`). They
/// are a different residual topology (K3 calls it AttnRes: config keys
/// `attn_res_block_size`, `output_attn_res_proj`), not this one's
/// second dialect, and giving them a Sinkhorn role would have been the
/// accommodation the K3 witness warned against.
///
/// Both halves asserted: the names do not classify under any operator
/// this wave touched, and the geometry could not have bound even if a
/// name had matched. The first alone could be a spelling gap; the
/// second says it is not.
#[test]
fn k3s_residual_operands_are_not_sinkhorn_sites_under_any_stream_count() {
    let fixture: serde_json::Value = serde_json::from_str(K3_HEADERS).unwrap();
    let mut shapes: std::collections::BTreeMap<String, Vec<usize>> =
        std::collections::BTreeMap::new();
    for header in fixture["shards"].as_object().unwrap().values() {
        for (name, tensor) in header.as_object().unwrap() {
            if let Some(leaf) = name.strip_prefix("language_model.model.layers.0.") {
                if leaf.contains("_res_") {
                    shapes.insert(
                        leaf.to_string(),
                        tensor["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_u64().unwrap() as usize)
                            .collect(),
                    );
                }
            }
        }
    }
    // The four the K3 witness names, at the shapes the checkpoint writes.
    assert_eq!(
        shapes,
        [
            ("mlp_res_norm.weight".to_string(), vec![K3_HIDDEN]),
            ("mlp_res_proj.weight".to_string(), vec![1, K3_HIDDEN]),
            (
                "self_attention_res_norm.weight".to_string(),
                vec![K3_HIDDEN]
            ),
            (
                "self_attention_res_proj.weight".to_string(),
                vec![1, K3_HIDDEN]
            ),
        ]
        .into_iter()
        .collect()
    );

    // Half one: no spelling classifies, on the operator K3's layer 0 runs.
    for leaf in shapes.keys() {
        assert_eq!(
            classify_stack_tensor_on(&format!("0.{leaf}"), LayerOperator::Kda),
            None,
            "{leaf} acquired a role — wave 18 accommodated K3 instead of transferring"
        );
    }
    // Half two: no Sinkhorn stream count makes a `[1, hidden]` projection
    // a site's mix projection, nor a `[hidden]` norm any site operand.
    for streams in 1..=16 {
        let mix = vec![
            HyperConnectionWeights::mix_rows_for(streams),
            streams * K3_HIDDEN,
        ];
        assert_ne!(
            mix, shapes["self_attention_res_proj.weight"],
            "hc = {streams}"
        );
        assert_ne!(
            vec![HyperConnectionWeights::mix_rows_for(streams)],
            shapes["self_attention_res_norm.weight"],
            "hc = {streams}"
        );
    }
}
