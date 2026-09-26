//! Additional op coverage — at least one test per advertised op.
//! registry-level tests

use super::*;

#[test]
fn graph_degrees() {
    if let Some(v) = call(
        "graph",
        "degrees",
        json!({"edges": [["A","B"],["B","C"],["B","D"]]}),
    ) {
        let arr = v.as_array().expect("array");
        let b_degree = arr
            .iter()
            .find(|x| x.get("node").and_then(|n| n.as_str()) == Some("B"))
            .and_then(|x| x.get("degree").and_then(|d| d.as_i64()));
        assert_eq!(b_degree, Some(3));
    }
}

#[test]
fn graph_bipartite_no() {
    // Odd cycle is not bipartite.
    assert_eq_expert(
        "graph",
        "is_bipartite",
        json!({"edges": [["A","B"],["B","C"],["C","A"]]}),
        json!(false),
    );
}

#[test]
fn registry_load_dir_tier_order() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    if reg.len() < 2 {
        return;
    }
    let tiers: Vec<u8> = reg.list().iter().map(|m| m.tier).collect();
    let mut sorted = tiers.clone();
    sorted.sort();
    assert_eq!(tiers, sorted, "experts must be sorted by tier ascending");
}

#[test]
fn registry_dispatches_by_op() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    let result = reg.call("mul", &json!({"a": 6, "b": 7}));
    assert!(result.is_some(), "arithmetic.mul should dispatch");
    assert_eq!(result.unwrap().value, json!(42.0));
}

#[test]
fn registry_unknown_op_returns_none() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    assert!(reg.call("nonexistent_op_abc_xyz", &json!({})).is_none());
}

#[test]
fn registry_all_experts_have_metadata() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    for meta in reg.list() {
        assert!(!meta.id.is_empty(), "id must not be empty");
        assert!(
            !meta.description.is_empty(),
            "description must not be empty"
        );
        assert!(!meta.version.is_empty(), "version must not be empty");
        assert!(meta.tier >= 1, "tier must be >= 1");
        assert!(!meta.ops.is_empty(), "expert {} advertises no ops", meta.id);
    }
}

#[test]
fn registry_memory_stable_across_many_calls() {
    // Without the larql_dealloc pairing in caller.rs, arithmetic's linear
    // memory grew by ~140 bytes per call (op + args + result strings leaked).
    // This test locks that regression down.
    let path = wasm("arithmetic");
    if !path.exists() {
        return;
    }
    let mut reg = ExpertRegistry::default();
    reg.load_file(&path).expect("load arithmetic");

    // Warm up so the expert is instantiated and its allocator has reached
    // steady state.
    for _ in 0..32 {
        let _ = reg.call("gcd", &json!({"a": 144, "b": 60}));
    }
    let pages_before = reg
        .wasm_info_for("arithmetic")
        .expect("present")
        .memory_pages;

    // 2000 calls was empirically enough pre-fix to grow memory by 3+ pages.
    for _ in 0..2000 {
        let _ = reg.call("gcd", &json!({"a": 144, "b": 60}));
    }
    let pages_after = reg
        .wasm_info_for("arithmetic")
        .expect("present")
        .memory_pages;

    assert_eq!(
        pages_before, pages_after,
        "arithmetic linear memory grew from {} to {} pages across 2000 calls — \
         dealloc is probably not paired in caller.rs",
        pages_before, pages_after
    );
}

#[test]
fn module_cache_file_is_written_and_reused() {
    // Exercise the .cwasm precompile cache against a private copy. Other
    // expert tests load the same fixture in parallel, and the cache lives next
    // to the .wasm, so using the shared fixture makes this mtime assertion
    // race-prone.
    let wasm_path = wasm("arithmetic");
    if !wasm_path.exists() {
        return;
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    let private_wasm_path = tmp.path().join("larql_expert_arithmetic.wasm");
    std::fs::copy(&wasm_path, &private_wasm_path).expect("copy wasm fixture");
    let cwasm_path = private_wasm_path.with_extension("cwasm");

    // First load compiles and writes the cache.
    {
        let mut reg = ExpertRegistry::default();
        reg.load_file(&private_wasm_path).expect("first load");
    }
    assert!(
        cwasm_path.exists(),
        "expected cache file {:?} to be created on first load",
        cwasm_path
    );

    // Second load should succeed against the cached artifact. We can't
    // reliably assert a speedup in a unit test, but we can at least confirm
    // the cached file is still present and the registry still functions.
    let cwasm_mtime_before = std::fs::metadata(&cwasm_path).unwrap().modified().unwrap();
    {
        let mut reg = ExpertRegistry::default();
        reg.load_file(&private_wasm_path).expect("second load");
        let result = reg
            .call("gcd", &json!({"a": 12, "b": 8}))
            .expect("gcd dispatches");
        assert_eq!(result.value, json!(4));
    }
    let cwasm_mtime_after = std::fs::metadata(&cwasm_path).unwrap().modified().unwrap();
    assert_eq!(
        cwasm_mtime_before, cwasm_mtime_after,
        "cache file should be reused on second load, not rewritten"
    );
}

#[test]
fn registry_experts_are_lazy_instantiated() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let mut reg = ExpertRegistry::load_dir(&dir).expect("load dir");

    // Freshly loaded: nothing instantiated yet, zero linear memory pages.
    for info in reg.wasm_infos() {
        assert!(
            !info.instantiated,
            "expert {:?} should not be instantiated at load",
            info.path
        );
        assert_eq!(info.memory_pages, 0);
    }

    // One call to arithmetic.gcd instantiates only arithmetic.
    let _ = reg
        .call("gcd", &json!({"a": 12, "b": 8}))
        .expect("gcd dispatches");
    let arith = reg.wasm_info_for("arithmetic").expect("arithmetic present");
    assert!(arith.instantiated);
    assert!(arith.memory_pages > 0);

    // No other expert should be instantiated yet.
    let still_cold: Vec<String> = reg
        .wasm_infos()
        .into_iter()
        .filter(|i| !i.instantiated)
        .map(|i| i.path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    assert!(
        still_cold.len() >= 17,
        "expected ≥17 cold experts, got {}",
        still_cold.len()
    );

    // evict_all drops every live instance.
    reg.evict_all();
    for info in reg.wasm_infos() {
        assert!(!info.instantiated, "evict_all should drop {:?}", info.path);
    }

    // Calls work again after eviction — recompilation is not required.
    let r = reg
        .call("gcd", &json!({"a": 12, "b": 8}))
        .expect("gcd still dispatches");
    assert_eq!(r.value, json!(4));
}

#[test]
fn registry_ops_are_discoverable() {
    let dir = wasm_dir();
    if !dir.exists() {
        return;
    }
    let reg = ExpertRegistry::load_dir(&dir).expect("load dir");
    let ops = reg.ops();
    // A few specific ops we expect to be present somewhere.
    for expected in &[
        "add",
        "gcd",
        "base64_encode",
        "convert",
        "lookup",
        "execute",
    ] {
        assert!(
            ops.contains(expected),
            "op {:?} missing from registry ops",
            expected
        );
    }
}
