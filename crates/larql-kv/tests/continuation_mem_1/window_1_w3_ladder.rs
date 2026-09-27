//! CONTINUATION-WINDOW-1 W3: identity-witnessed deallocation and the length ladder

use super::*;

/// Positions after the batched prefill at every rung: 16 resumed, 8 decoded.
const W3_RESUME: usize = 16;
const W3_DECODE: usize = 8;

/// Per-layer retained rows and live K/V bytes (adopted row allocations,
/// and the row lists — the declared fixed capacity) at a journey's end.
fn layer_residency(
    subject: &Subject,
    inventory: &[measured::Backing],
    rows: impl Fn(usize) -> (usize, usize),
) -> Vec<Value> {
    use larql_vindex::format::vindex3::opplan::exec::kv::HistoryRange;
    subject
        .geometry
        .iter()
        .enumerate()
        .filter_map(|(layer, g)| {
            let kv = g.kv_side()?;
            let (base, end) = rows(layer);
            let bytes = |kinds: &[&str]| -> usize {
                inventory
                    .iter()
                    .filter(|b| b.layer == layer && kinds.contains(&b.kind))
                    .filter_map(|b| b.bytes)
                    .sum()
            };
            Some(json!({
                "layer": layer,
                "sliding": matches!(kv.history, HistoryRange::Trailing(_)),
                "window": kv.window,
                "base": base,
                "end": end,
                "retained_rows": end - base,
                "row_bytes": bytes(&["k_row", "v_row"]),
                "row_list_bytes": bytes(&["k_header", "v_header"]),
            }))
        })
        .collect()
}

/// One rung: row/v1 and window/v1 over the same journey of `n` positions.
fn window_rung<B: PlanBackend>(
    subject: &Subject,
    ops: &larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands,
    backend: &B,
    n: usize,
) -> Value {
    use larql_vindex::format::vindex3::opplan::exec::kv::ContinuationProvider;
    let journey = Journey {
        prefill: tokens(1000, n - W3_RESUME - W3_DECODE),
        resume: tokens(30_000, W3_RESUME),
        decode: tokens(40_000, W3_DECODE),
    };
    let mut row = Measured::new(RowKvState::default());
    let started = std::time::Instant::now();
    let row_out = subjects::run(subject, ops, backend, &mut row, &journey);
    let row_secs = started.elapsed().as_secs_f64();
    let mut win = Measured::new(WindowKvState::new());
    let started = std::time::Instant::now();
    let win_out = subjects::run(subject, ops, backend, &mut win, &journey);
    let win_secs = started.elapsed().as_secs_f64();

    let digests = |o: &subjects::Outcome| -> Vec<String> {
        [subjects::BATCHED, subjects::RESUMED, subjects::DECODE]
            .iter()
            .map(|phase| subjects::logits_digest(&o.logits, phase))
            .collect()
    };
    let exact = digests(&row_out) == digests(&win_out);
    let row_witness = row.identity_witness();
    let win_witness = win.identity_witness();
    let row_layers = layer_residency(subject, &row_out.inventories.last().unwrap().1, |l| {
        let v = row.inner.rows(l);
        (v.base(), v.end())
    });
    let win_layers = layer_residency(subject, &win_out.inventories.last().unwrap().1, |l| {
        let v = win.inner.rows(l);
        (v.base(), v.end())
    });
    let (row_calls, _) = row.take_records();
    let (win_calls, _) = win.take_records();
    let churn = |calls: &[measured::CallRecord]| -> Value {
        let appends: Vec<_> = calls.iter().filter_map(|c| c.append).collect();
        json!({
            "appends": appends.len(),
            "allocs_in_appends": calls.iter().filter(|c| c.append.is_some()).map(|c| c.delta.allocs).sum::<u64>(),
            "frees_in_appends": calls.iter().filter(|c| c.append.is_some()).map(|c| c.delta.frees).sum::<u64>(),
            "evicted_frees": appends.iter().map(|t| t.evicted_frees).sum::<u64>(),
            "unclassified": appends.iter().map(|t| t.unclassified).sum::<u64>(),
        })
    };
    let sum = |layers: &[Value], sliding: bool, key: &str| -> u64 {
        layers
            .iter()
            .filter(|l| l["sliding"] == sliding)
            .map(|l| l[key].as_u64().unwrap())
            .sum()
    };
    let class = |layers: &[Value]| {
        json!({
            "sliding_retained_rows_per_layer": layers.iter().filter(|l| l["sliding"] == true).map(|l| l["retained_rows"].as_u64().unwrap()).collect::<std::collections::BTreeSet<u64>>().into_iter().collect::<Vec<_>>(),
            "sliding_row_bytes": sum(layers, true, "row_bytes"),
            "sliding_row_list_bytes": sum(layers, true, "row_list_bytes"),
            "full_retained_rows_per_layer": layers.iter().filter(|l| l["sliding"] == false).map(|l| l["retained_rows"].as_u64().unwrap()).collect::<std::collections::BTreeSet<u64>>().into_iter().collect::<Vec<_>>(),
            "full_row_bytes": sum(layers, false, "row_bytes"),
            "full_row_list_bytes": sum(layers, false, "row_list_bytes"),
        })
    };
    json!({
        "subject": subject.name,
        "git_sha": git_sha(),
        "positions": n,
        "journey": {"prefill": journey.prefill.len(), "resume": W3_RESUME, "decode": W3_DECODE},
        "exact_logits_digests_equal": exact,
        "row_v1": {"class": class(&row_layers), "identity_witness": format!("{row_witness:?}"), "churn": churn(&row_calls), "journey_seconds": row_secs},
        "window_v1": {"class": class(&win_layers), "identity_witness": format!("{win_witness:?}"), "witness_holds": win_witness.holds(), "churn": churn(&win_calls), "journey_seconds": win_secs},
        "layers": {"row_v1": row_layers, "window_v1": win_layers},
    })
}

/// W3 on the fixture: the identity witness holds for window/v1 (rows
/// below base freed as that very adoption), and its control — row/v1 has
/// nothing below base and every held row is still the live adoption.
#[test]
fn window_identity_witness_on_the_sliding_fixture() {
    let _serial = serial();
    let subject = subjects::fixture(miniature_glimmer, "mem1-sliding");
    let backend = ReferenceBackend::new();
    let ops = subject.prepare(&backend);
    let journey = Journey {
        prefill: G_TOKENS.to_vec(),
        resume: vec![5, 9],
        decode: vec![1, 2, 3, 4],
    };
    let mut win = Measured::new(WindowKvState::new());
    let win_out = subjects::run(&subject, &ops, &backend, &mut win, &journey);
    let mut row = Measured::new(RowKvState::default());
    let row_out = subjects::run(&subject, &ops, &backend, &mut row, &journey);
    let w = win.identity_witness();
    assert!(w.below_base > 0, "the journey must cross the window");
    assert!(w.holds(), "window/v1 identity witness: {w:?}");
    let r = row.identity_witness();
    assert_eq!(r.below_base, 0, "row/v1 drops nothing");
    assert!(r.holds() && r.held > 0, "row/v1 control: {r:?}");
    for phase in [subjects::BATCHED, subjects::RESUMED, subjects::DECODE] {
        assert_eq!(
            subjects::logits_digest(&win_out.logits, phase),
            subjects::logits_digest(&row_out.logits, phase)
        );
    }
}

/// W3: the frozen length ladder on gemma3-4b-it, both providers per rung,
/// one record per rung. Lengths from LARQL_WINDOW1_LENGTHS (default: the
/// frozen ladder).
#[test]
#[ignore = "real container: LARQL_MEM1_GEMMA (WINDOW-1 W3 ladder; long)"]
fn real_window_ladder_gemma3_4b() {
    let _serial = serial();
    let dir = std::env::var_os("LARQL_MEM1_GEMMA").expect("set LARQL_MEM1_GEMMA");
    let lengths: Vec<usize> = std::env::var("LARQL_WINDOW1_LENGTHS")
        .unwrap_or_else(|_| "512,1024,1124,2048,4096,8192".to_string())
        .split(',')
        .map(|n| n.trim().parse().expect("a length"))
        .collect();
    let subject = subjects::open(std::path::Path::new(&dir), "gemma3-4b-it");
    let backend = ProductionBackend::new();
    let ops = subject.prepare(&backend);
    for n in lengths {
        let record = window_rung(&subject, &ops, &backend, n);
        let path = out_dir().join(format!("window1-gemma3-4b-{n}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&record).unwrap()).unwrap();
        eprintln!(
            "rung {n}: exact {} witness {} sliding rows {} (row/v1 {}) wrote {}",
            record["exact_logits_digests_equal"],
            record["window_v1"]["witness_holds"],
            record["window_v1"]["class"]["sliding_retained_rows_per_layer"],
            record["row_v1"]["class"]["sliding_retained_rows_per_layer"],
            path.display()
        );
    }
}
