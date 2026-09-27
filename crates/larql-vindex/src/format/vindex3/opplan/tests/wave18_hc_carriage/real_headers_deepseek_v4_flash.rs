//! Real headers: DeepSeek-V4-Flash

use super::*;

/// **The head gains an owner, `mtp.0` does not.** On DeepSeek-V4-Flash's
/// real headers the three bare `hc_head_*` groups — findings 68, 69 and
/// 70 of the cached plan — become one placed object, the layer sites
/// stay inside the decoder stack, and the eighteen hyper-connection
/// tensors under `mtp.0` remain in the external namespace with the same
/// honest refusal they had. The leak the baseline could not yet test
/// for is tested here: no object binding starts with `mtp`.
///
/// This is the whole of what wave 18 does to DeepSeek. Its base dialect
/// (`attn.wq_a`, `attn.wkv`, `attn_norm`, `ffn.experts.N.w1`) is
/// untouched, so its surface still does not build and it stays blocked.
#[test]
fn deepseeks_head_is_owned_under_the_declaration_and_mtp_stays_external() {
    let built = deepseek_graph(|_| {});

    let head = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::HyperConnectionHead)
        .expect("DeepSeek's head is placed");
    let mut prefixes: Vec<&str> = head
        .source_bindings
        .iter()
        .map(|b| b.tensor_prefix.as_str())
        .collect();
    prefixes.sort_unstable();
    assert_eq!(prefixes, ["hc_head_base", "hc_head_fn", "hc_head_scale"]);
    let head_bytes: u64 = head.source_bindings.iter().map(|b| b.bytes).sum();
    // (4 · 16384 + 4 + 1) · 4 bytes of F32, from the real headers.
    assert_eq!(head_bytes, (4 * 16384 + 4 + 1) * 4);

    // The layer sites are stack tensors, owned with the rest of the layer.
    let stack = built
        .graph
        .objects
        .iter()
        .find(|o| o.kind == ObjectKind::DecoderStack)
        .expect("the stack is placed");
    assert!(stack
        .source_bindings
        .iter()
        .any(|b| b.tensor_prefix == "layers"));

    // mtp: unplaced, honestly, and NOT leaked into any object.
    let mtp = built
        .unplaced
        .iter()
        .find(|u| u.prefix == "mtp")
        .expect("mtp stays unplaced");
    assert!(
        mtp.reason.contains("no placement rule owns this group"),
        "{}",
        mtp.reason
    );
    for object in &built.graph.objects {
        for binding in &object.source_bindings {
            assert!(
                !binding.tensor_prefix.starts_with("mtp"),
                "object `{}` absorbed `{}`",
                object.id,
                binding.tensor_prefix
            );
        }
    }
    // Total fate: every recognised hyper-connection tensor on the surface
    // is either in the stack (layer sites), in the head object, or under
    // the external namespace. Nothing is neither.
    let fixture: serde_json::Value = serde_json::from_str(HEADERS).unwrap();
    for name in fixture[DEEPSEEK].as_object().unwrap().keys() {
        if !name.contains("hc_") {
            continue;
        }
        let owned = built
            .graph
            .objects
            .iter()
            .any(|o| o.source_bindings.iter().any(|b| b.covers(name)));
        let external = name.starts_with("mtp.");
        assert!(
            owned != external,
            "{name}: owned {owned}, external {external} — every hyper-connection tensor \
             has exactly one fate"
        );
    }
}

/// **The dialect control on the head.** Withdraw the iteration count and
/// the declaration resolves to NO topology — Hy4-preview's shape — so the
/// same three bare groups lose their owner and say why. A placement rule
/// that matched on the name alone would place them anyway.
#[test]
fn deepseeks_head_loses_its_owner_when_the_topology_is_not_declared() {
    for withdraw in [
        // Hy4's shape: streams and epsilon without an iteration count.
        Box::new(|c: &mut serde_json::Value| {
            c.as_object_mut().unwrap().remove("hc_sinkhorn_iters");
        }) as Box<dyn FnOnce(&mut serde_json::Value)>,
        // No declaration at all: a single-stream checkpoint with the
        // head's names in its estate.
        Box::new(|c: &mut serde_json::Value| {
            let object = c.as_object_mut().unwrap();
            object.remove("hc_mult");
            object.remove("hc_eps");
            object.remove("hc_sinkhorn_iters");
        }),
    ] {
        let built = deepseek_graph(withdraw);
        assert!(
            !built
                .graph
                .objects
                .iter()
                .any(|o| o.kind == ObjectKind::HyperConnectionHead),
            "the head must not be owned without the declaration"
        );
        for group in ["hc_head_fn", "hc_head_base", "hc_head_scale"] {
            let unplaced = built
                .unplaced
                .iter()
                .find(|u| u.prefix == group)
                .unwrap_or_else(|| panic!("{group} was placed: {:?}", built.unplaced));
            assert!(
                unplaced.reason.contains("declares no Sinkhorn-split"),
                "{group}: {}",
                unplaced.reason
            );
        }
    }
}
