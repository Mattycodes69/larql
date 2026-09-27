use super::*;

fn synthetic_capture(
    prompts: usize,
    steps: usize,
    layers: usize,
    hidden: usize,
) -> Vec<Vec<Vec<Vec<f32>>>> {
    (0..prompts)
        .map(|p| {
            (0..steps)
                .map(|s| {
                    (0..layers)
                        .map(|l| {
                            (0..hidden)
                                .map(|h| (p * 1000 + s * 100 + l * 10 + h) as f32)
                                .collect()
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn write_synthetic(dir: &Path, prompts: usize, steps: usize, layers: usize, hidden: usize) {
    let cap = synthetic_capture(prompts, steps, layers, hidden);
    let texts: Vec<String> = (0..prompts).map(|i| format!("prompt {i}")).collect();
    CapturePool::write(dir, "test-model", hidden, layers, &texts, &cap, 12345).unwrap();
}

/// Routing fixture: layer 0 of every (prompt, step) routes to experts
/// (p+s) % 3 and 3 with weights 0.75/0.25; layer 1+ is non-MoE (empty).
fn synthetic_routing(
    prompts: usize,
    steps: usize,
    layers: usize,
    hidden: usize,
    top_k: usize,
) -> RoutingCapture {
    let raw = synthetic_capture(prompts, steps, layers, hidden);
    let normed: Vec<Vec<Vec<Vec<f32>>>> = raw
        .iter()
        .map(|p| {
            p.iter()
                .map(|s| {
                    s.iter()
                        .map(|l| l.iter().map(|v| v * 0.5).collect())
                        .collect()
                })
                .collect()
        })
        .collect();
    let routing: Vec<Vec<Vec<Vec<(u32, f32)>>>> = (0..prompts)
        .map(|p| {
            (0..steps)
                .map(|s| {
                    (0..layers)
                        .map(|l| {
                            if l == 0 {
                                vec![(((p + s) % 3) as u32, 0.75f32), (3u32, 0.25f32)]
                            } else {
                                Vec::new()
                            }
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    RoutingCapture {
        top_k,
        raw,
        normed,
        routing,
    }
}

fn write_synthetic_routed(
    dir: &Path,
    prompts: usize,
    steps: usize,
    layers: usize,
    hidden: usize,
    top_k: usize,
) -> CaptureManifest {
    let cap = synthetic_capture(prompts, steps, layers, hidden);
    let rc = synthetic_routing(prompts, steps, layers, hidden, top_k);
    let texts: Vec<String> = (0..prompts).map(|i| format!("prompt {i}")).collect();
    CapturePool::write_with_routing(
        dir,
        "test-model",
        hidden,
        layers,
        &texts,
        &cap,
        Some(&rc),
        7,
    )
    .unwrap()
}

#[test]
fn manifest_round_trips_through_serde() {
    let m = CaptureManifest {
        version: CAPTURE_VERSION,
        model: "m".into(),
        hidden_size: 8,
        num_layers: 2,
        steps: 3,
        dtype: "f32-le".into(),
        prompts: vec![PromptMeta {
            id: 0,
            text: "p".into(),
            steps_captured: 3,
        }],
        created_unix: 99,
        routing: None,
    };
    let json = serde_json::to_string(&m).unwrap();
    // Walk-ffn-only manifests must not grow a routing key — the shipped
    // 330M pool's manifest stays byte-stable in shape.
    assert!(!json.contains("routing"));
    let back: CaptureManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.version, m.version);
    assert_eq!(back.hidden_size, 8);
    assert_eq!(back.expected_bytes(), 3 * 2 * 8 * 4);
    assert!(back.routing.is_none());
    assert!(back.expected_routing_bytes().is_none());
}

#[test]
fn manifest_routing_block_round_trips_and_sizes() {
    let m = CaptureManifest {
        version: CAPTURE_VERSION,
        model: "m".into(),
        hidden_size: 8,
        num_layers: 2,
        steps: 3,
        dtype: "f32-le".into(),
        prompts: vec![PromptMeta {
            id: 0,
            text: "p".into(),
            steps_captured: 3,
        }],
        created_unix: 99,
        routing: Some(RoutingManifest {
            top_k: 4,
            has_raw: true,
            has_normed: true,
        }),
    };
    let json = serde_json::to_string(&m).unwrap();
    let back: CaptureManifest = serde_json::from_str(&json).unwrap();
    let rb = back.routing.clone().expect("routing block survives serde");
    assert_eq!(rb.top_k, 4);
    assert!(rb.has_raw && rb.has_normed);
    assert_eq!(back.expected_routing_bytes(), Some(3 * 2 * 4 * 8));
}

#[test]
fn write_open_rows_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic(dir.path(), 2, 2, 2, 4);
    let pool = CapturePool::open(dir.path()).unwrap();
    assert_eq!(pool.num_prompts(), 2);

    // Row i of a batch must be prompt i at the same (step, layer).
    let rows = pool.rows(2, 1, 1).unwrap();
    assert_eq!(rows.len(), 2 * 4);
    let expect_p0: Vec<f32> = (0..4).map(|h| (100 + 10 + h) as f32).collect();
    let expect_p1: Vec<f32> = (0..4).map(|h| (1000 + 100 + 10 + h) as f32).collect();
    assert_eq!(&rows[..4], &expect_p0[..]);
    assert_eq!(&rows[4..], &expect_p1[..]);
}

#[test]
fn walk_ffn_only_pool_has_no_routing_and_sidecar_accessors_err() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic(dir.path(), 2, 2, 2, 4);
    // No sidecar files on disk.
    assert!(!dir.path().join("raw.bin").exists());
    assert!(!dir.path().join("normed.bin").exists());
    assert!(!dir.path().join("routing.bin").exists());
    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(!pool.has_routing());
    assert_eq!(pool.routing_top_k(), None);
    assert!(pool.rows(2, 0, 0).is_ok(), "dense replay path unaffected");
    assert!(pool.raw_rows(1, 0, 0).is_err());
    assert!(pool.normed_rows(1, 0, 0).is_err());
    assert!(pool.routing(0, 0, 0).is_err());
}

#[test]
fn routed_write_open_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let manifest = write_synthetic_routed(dir.path(), 2, 2, 2, 4, 2);
    let rb = manifest.routing.as_ref().expect("routing block written");
    assert_eq!(rb.top_k, 2);
    assert!(rb.has_raw && rb.has_normed);

    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.has_routing());
    assert_eq!(pool.routing_top_k(), Some(2));

    // residuals.bin unchanged by the sidecars.
    let rows = pool.rows(2, 1, 1).unwrap();
    assert_eq!(rows.len(), 2 * 4);

    // raw plane = synthetic_capture values; normed plane = raw × 0.5.
    let raw = pool.raw_rows(2, 1, 1).unwrap();
    let expect_p0: Vec<f32> = (0..4).map(|h| (100 + 10 + h) as f32).collect();
    assert_eq!(&raw[..4], &expect_p0[..]);
    let normed = pool.normed_rows(2, 1, 1).unwrap();
    for (n, r) in normed.iter().zip(raw.iter()) {
        assert_eq!(*n, r * 0.5);
    }

    // Layer 0 routed: prompt 1 step 1 → expert (1+1)%3=2 @0.75, 3 @0.25.
    let pairs = pool.routing(1, 1, 0).unwrap().expect("MoE layer routes");
    assert_eq!(pairs, &[(2u32, 0.75f32), (3u32, 0.25f32)]);
    // Layer 1 non-MoE: all-sentinel record → None.
    assert_eq!(pool.routing(1, 1, 1).unwrap(), None);
}

#[test]
fn routing_bounds_checks() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_routed(dir.path(), 2, 2, 2, 4, 2);
    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.routing(2, 0, 0).is_err(), "prompt out of range");
    assert!(pool.routing(0, 2, 0).is_err(), "step out of range");
    assert!(pool.routing(0, 0, 2).is_err(), "layer out of range");
    assert!(pool.raw_rows(3, 0, 0).is_err(), "batch > prompts");
    assert!(pool.normed_rows(1, 0, 2).is_err(), "layer out of range");
}

#[test]
fn zero_weight_pairs_stripped_to_sentinel_at_write() {
    let dir = tempfile::tempdir().unwrap();
    let cap = synthetic_capture(1, 1, 1, 2);
    let mut rc = synthetic_routing(1, 1, 1, 2, 3);
    // One real pair sandwiched between zero-weight pairs.
    rc.routing[0][0][0] = vec![(5u32, 0.0f32), (7u32, 1.0f32), (9u32, 0.0f32)];
    let texts = vec!["a".into()];
    CapturePool::write_with_routing(dir.path(), "m", 2, 1, &texts, &cap, Some(&rc), 0).unwrap();

    // On-disk record: real pair compacted first, then sentinel padding.
    let bytes = std::fs::read(dir.path().join("routing.bin")).unwrap();
    assert_eq!(bytes.len(), 3 * 8, "fixed top_k record layout");
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 7);
    assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1.0);
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        ROUTING_SENTINEL_EXPERT
    );
    assert_eq!(
        u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
        ROUTING_SENTINEL_EXPERT
    );

    let pool = CapturePool::open(dir.path()).unwrap();
    assert_eq!(
        pool.routing(0, 0, 0).unwrap(),
        Some(&[(7u32, 1.0f32)][..]),
        "reader sees only the surviving pair"
    );
}

#[test]
fn routed_write_truncates_ragged_prompts_with_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    let mut cap = synthetic_capture(2, 3, 1, 2);
    cap[1].truncate(1); // prompt 1 stopped early (EOS)
    let mut rc = synthetic_routing(2, 3, 1, 2, 2);
    rc.raw[1].truncate(1);
    rc.normed[1].truncate(1);
    rc.routing[1].truncate(1);
    let texts = vec!["a".into(), "b".into()];
    let m =
        CapturePool::write_with_routing(dir.path(), "m", 2, 1, &texts, &cap, Some(&rc), 0).unwrap();
    assert_eq!(m.steps, 1);
    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.raw_rows(2, 0, 0).is_ok());
    assert!(pool.raw_rows(2, 1, 0).is_err(), "step beyond min must fail");
    assert!(pool.routing(0, 0, 0).unwrap().is_some());
}

#[test]
fn routed_write_rejects_bad_routing_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let cap = synthetic_capture(2, 2, 2, 4);
    let texts = vec!["a".into(), "b".into()];
    let write = |rc: &RoutingCapture| {
        CapturePool::write_with_routing(dir.path(), "m", 4, 2, &texts, &cap, Some(rc), 0)
    };

    // top_k 0.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.top_k = 0;
    assert!(write(&rc).is_err());

    // More pairs than top_k.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.routing[0][0][0] = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
    assert!(write(&rc).is_err());

    // Reserved sentinel expert id in the input.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.routing[0][0][0] = vec![(ROUTING_SENTINEL_EXPERT, 0.5)];
    assert!(write(&rc).is_err());

    // Wrong prompt count on a plane.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.raw.pop();
    assert!(write(&rc).is_err());
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.routing.pop();
    assert!(write(&rc).is_err());

    // Too few steps on a plane.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.normed[0].truncate(1);
    assert!(write(&rc).is_err());
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.routing[0].truncate(1);
    assert!(write(&rc).is_err());

    // Wrong layer count / hidden size on a plane row.
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.raw[0][0].pop();
    assert!(write(&rc).is_err());
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.normed[0][0][0].pop();
    assert!(write(&rc).is_err());
    let mut rc = synthetic_routing(2, 2, 2, 4, 2);
    rc.routing[0][0].pop();
    assert!(write(&rc).is_err());
}

#[test]
fn write_truncates_ragged_prompts_to_min_steps() {
    let dir = tempfile::tempdir().unwrap();
    let mut cap = synthetic_capture(2, 3, 1, 2);
    cap[1].truncate(1); // prompt 1 stopped early (EOS)
    let texts = vec!["a".into(), "b".into()];
    let m = CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).unwrap();
    assert_eq!(m.steps, 1);
    assert_eq!(m.prompts[0].steps_captured, 3);
    assert_eq!(m.prompts[1].steps_captured, 1);
    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.rows(2, 0, 0).is_ok());
    assert!(pool.rows(2, 1, 0).is_err(), "step beyond min must fail");
}

#[test]
fn write_rejects_empty_and_mismatched_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let texts = vec!["a".into()];
    assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &[], 0).is_err());

    // Wrong layer count.
    let cap = synthetic_capture(1, 1, 2, 2);
    assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

    // Wrong hidden size.
    let cap = synthetic_capture(1, 1, 1, 3);
    assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

    // Zero steps.
    let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![]];
    assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());
}

#[test]
fn open_rejects_truncated_bin_and_bad_version() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic(dir.path(), 1, 1, 1, 4);

    // Truncate residuals.bin → open must fail on size mismatch.
    let bin = dir.path().join("residuals.bin");
    let data = std::fs::read(&bin).unwrap();
    std::fs::write(&bin, &data[..data.len() - 4]).unwrap();
    assert!(CapturePool::open(dir.path()).is_err());
    std::fs::write(&bin, &data).unwrap();
    assert!(CapturePool::open(dir.path()).is_ok());

    // Corrupt the version → open must fail.
    let mf = dir.path().join("manifest.json");
    let json = std::fs::read_to_string(&mf).unwrap();
    std::fs::write(&mf, json.replace("\"version\": 1", "\"version\": 999")).unwrap();
    assert!(CapturePool::open(dir.path()).is_err());
}

#[test]
fn open_rejects_truncated_or_missing_sidecars() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_routed(dir.path(), 1, 1, 2, 4, 2);

    for file in ["raw.bin", "normed.bin", "routing.bin"] {
        let path = dir.path().join(file);
        let data = std::fs::read(&path).unwrap();
        // Truncated sidecar → exact-byte-length validation must reject.
        std::fs::write(&path, &data[..data.len() - 4]).unwrap();
        assert!(
            CapturePool::open(dir.path()).is_err(),
            "truncated {file} must be rejected"
        );
        // Missing sidecar (manifest says present) → rejected too.
        std::fs::remove_file(&path).unwrap();
        assert!(
            CapturePool::open(dir.path()).is_err(),
            "missing {file} must be rejected"
        );
        std::fs::write(&path, &data).unwrap();
        assert!(CapturePool::open(dir.path()).is_ok());
    }

    // top_k 0 in the manifest block is impossible from write — corrupt it.
    let mf = dir.path().join("manifest.json");
    let json = std::fs::read_to_string(&mf).unwrap();
    std::fs::write(&mf, json.replace("\"top_k\": 2", "\"top_k\": 0")).unwrap();
    assert!(CapturePool::open(dir.path()).is_err());
}

#[test]
fn open_tolerates_absent_raw_or_normed_when_flagged_off() {
    // has_raw/has_normed false → those planes are optional; routing.bin
    // still required and served.
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_routed(dir.path(), 1, 1, 1, 2, 2);
    std::fs::remove_file(dir.path().join("raw.bin")).unwrap();
    std::fs::remove_file(dir.path().join("normed.bin")).unwrap();
    let mf = dir.path().join("manifest.json");
    let json = std::fs::read_to_string(&mf).unwrap();
    let json = json
        .replace("\"has_raw\": true", "\"has_raw\": false")
        .replace("\"has_normed\": true", "\"has_normed\": false");
    std::fs::write(&mf, json).unwrap();

    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.has_routing());
    assert!(pool.raw_rows(1, 0, 0).is_err());
    assert!(pool.normed_rows(1, 0, 0).is_err());
    assert!(pool.routing(0, 0, 0).unwrap().is_some());
}

#[test]
fn rows_bounds_checks() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic(dir.path(), 2, 2, 2, 2);
    let pool = CapturePool::open(dir.path()).unwrap();
    assert!(pool.rows(0, 0, 0).is_err());
    assert!(pool.rows(3, 0, 0).is_err(), "batch > prompts");
    assert!(pool.rows(1, 2, 0).is_err(), "step out of range");
    assert!(pool.rows(1, 0, 2).is_err(), "layer out of range");
}
